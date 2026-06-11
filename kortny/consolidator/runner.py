"""Consolidator worker loop with advisory-lock leader election (HIG-225)."""

from __future__ import annotations

import argparse
import logging
import os
import socket
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from kortny.config import Settings, load_settings
from kortny.consolidator.service import ConsolidationOutcome, ConsolidationService
from kortny.consolidator.trigger import TriggerDecision, evaluate_trigger
from kortny.db.models import Installation
from kortny.db.session import make_session_factory
from kortny.embeddings import EmbeddingIndex, create_embedding_backend
from kortny.logging_config import configure_logging
from kortny.observability import configure_tracing, start_span

logger = logging.getLogger(__name__)

DEFAULT_CONSOLIDATOR_ADVISORY_LOCK_KEY = 759340187
DEFAULT_CONSOLIDATOR_POLL_INTERVAL_SECONDS = 600.0


@dataclass(frozen=True, slots=True)
class ConsolidatorTickResult:
    """Outcome from one consolidator tick."""

    runner_id: str
    status: str
    leader_acquired: bool = True
    decisions: tuple[tuple[uuid.UUID, TriggerDecision], ...] = ()
    outcomes: tuple[ConsolidationOutcome, ...] = ()


class ConsolidatorRunner:
    """Evaluate triggers per installation and run due consolidations."""

    def __init__(
        self,
        session: Session,
        *,
        settings: Settings | None = None,
        service: ConsolidationService | None = None,
        runner_id: str | None = None,
        advisory_lock_key: int | None = None,
    ) -> None:
        self.session = session
        self.settings = settings
        self.service = service or ConsolidationService(
            session,
            settings=settings,
            embedding_index=self._default_embedding_index(session, settings),
        )
        self.runner_id = runner_id or default_consolidator_runner_id()
        self.advisory_lock_key = advisory_lock_key or (
            settings.consolidator_advisory_lock_key
            if settings is not None
            else DEFAULT_CONSOLIDATOR_ADVISORY_LOCK_KEY
        )

    @staticmethod
    def _default_embedding_index(
        session: Session,
        settings: Settings | None,
    ) -> EmbeddingIndex | None:
        if settings is None:
            return None
        backend = create_embedding_backend(settings)
        if backend is None:
            return None
        return EmbeddingIndex(session, backend)

    def run_once(
        self,
        *,
        now: datetime | None = None,
        use_advisory_lock: bool = False,
        force: bool = False,
    ) -> ConsolidatorTickResult:
        run_at = now or datetime.now(UTC)
        if use_advisory_lock and not self._try_advisory_lock():
            return ConsolidatorTickResult(
                runner_id=self.runner_id,
                status="lock_skipped",
                leader_acquired=False,
            )
        try:
            with start_span(
                "consolidator.tick",
                attributes={
                    "openinference.span.kind": "CHAIN",
                    "consolidator.runner.id": self.runner_id,
                },
            ):
                self.service.fail_stale_runs(now=run_at)
                decisions: list[tuple[uuid.UUID, TriggerDecision]] = []
                outcomes: list[ConsolidationOutcome] = []
                for installation_id in self._installation_ids():
                    decision = (
                        TriggerDecision(True, "forced")
                        if force
                        else self._evaluate(installation_id, now=run_at)
                    )
                    decisions.append((installation_id, decision))
                    if not decision.should_run:
                        continue
                    try:
                        outcomes.append(
                            self.service.run_once(
                                installation_id=installation_id,
                                now=run_at,
                            )
                        )
                    except Exception:
                        logger.exception(
                            "consolidation run failed installation_id=%s",
                            installation_id,
                        )
            status = "processed" if outcomes else "idle"
            return ConsolidatorTickResult(
                runner_id=self.runner_id,
                status=status,
                decisions=tuple(decisions),
                outcomes=tuple(outcomes),
            )
        finally:
            if use_advisory_lock:
                self._release_advisory_lock()

    def _evaluate(
        self,
        installation_id: uuid.UUID,
        *,
        now: datetime,
    ) -> TriggerDecision:
        last_success = self.service.last_successful_run_started_at(installation_id)
        new_items = self.service.new_item_count(installation_id, last_success)
        last_activity = self.service.last_activity_at(installation_id)
        if self.settings is not None:
            return evaluate_trigger(
                now=now,
                last_success_at=last_success,
                new_item_count=new_items,
                last_activity_at=last_activity,
                min_new_items=self.settings.consolidator_min_new_items,
                min_interval=timedelta(
                    hours=self.settings.consolidator_min_interval_hours
                ),
                quiet_window=timedelta(
                    minutes=self.settings.consolidator_quiet_minutes
                ),
                nightly_floor=timedelta(
                    hours=self.settings.consolidator_nightly_floor_hours
                ),
            )
        return evaluate_trigger(
            now=now,
            last_success_at=last_success,
            new_item_count=new_items,
            last_activity_at=last_activity,
        )

    def _installation_ids(self) -> list[uuid.UUID]:
        return list(
            self.session.scalars(
                select(Installation.id).order_by(Installation.created_at)
            )
        )

    def _try_advisory_lock(self) -> bool:
        return bool(
            self.session.scalar(
                select(func.pg_try_advisory_lock(self.advisory_lock_key))
            )
        )

    def _release_advisory_lock(self) -> None:
        self.session.execute(select(func.pg_advisory_unlock(self.advisory_lock_key)))


class ConsolidatorWorker:
    """Poll the consolidator runner forever."""

    def __init__(
        self,
        *,
        session_factory: sessionmaker[Session] | None = None,
        settings: Settings | None = None,
        runner_id: str | None = None,
        poll_interval_seconds: float | None = None,
        use_advisory_lock: bool = True,
    ) -> None:
        self.session_factory = session_factory or make_session_factory()
        self.settings = settings
        self.runner_id = runner_id or default_consolidator_runner_id()
        self.poll_interval_seconds = poll_interval_seconds or (
            settings.consolidator_poll_interval_seconds
            if settings is not None
            else DEFAULT_CONSOLIDATOR_POLL_INTERVAL_SECONDS
        )
        self.use_advisory_lock = use_advisory_lock

    def run_once(
        self,
        *,
        now: datetime | None = None,
        force: bool = False,
    ) -> ConsolidatorTickResult:
        # One dedicated connection per tick, NOT a begin() block: the
        # service commits after every pass (crash safety), which would
        # close an enclosing transaction context, and session-level
        # advisory locks are connection-bound — pooled-connection swaps
        # after a commit would leak the lock.
        engine = self.session_factory.kw["bind"]
        with (
            engine.connect() as connection,
            Session(bind=connection, expire_on_commit=False) as session,
        ):
            try:
                result = ConsolidatorRunner(
                    session,
                    settings=self._settings,
                    runner_id=self.runner_id,
                ).run_once(
                    now=now,
                    use_advisory_lock=self.use_advisory_lock,
                    force=force,
                )
                session.commit()
            except Exception:
                session.rollback()
                raise
            logger.info(
                "consolidator tick runner_id=%s status=%s runs=%s",
                result.runner_id,
                result.status,
                len(result.outcomes),
            )
            return result

    def run_forever(self) -> None:
        while True:
            try:
                self.run_once()
            except Exception:
                # A failed tick (transient DB blip, provider outage) must
                # not kill the service into a restart loop.
                logger.exception("consolidator tick failed; continuing")
            time.sleep(self.poll_interval_seconds)

    @property
    def _settings(self) -> Settings:
        if self.settings is None:
            self.settings = load_settings()
        return self.settings


def default_consolidator_runner_id() -> str:
    return f"{socket.gethostname()}-{os.getpid()}"


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entrypoint for the consolidator worker."""

    configure_logging()
    parser = argparse.ArgumentParser(description="Run the Kortny consolidator")
    parser.add_argument("--once", action="store_true", help="Run one tick")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip trigger checks and consolidate every installation now",
    )
    parser.add_argument(
        "--runner-id",
        default=None,
        help="Override runner id used in logs",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=None,
        help="Seconds to sleep between ticks",
    )
    args = parser.parse_args(argv)

    settings = load_settings()
    configure_tracing(settings)
    if not settings.consolidator_enabled:
        logger.info("consolidator disabled by KORTNY_CONSOLIDATOR_ENABLED=false")
        print("consolidator disabled")
        return

    worker = ConsolidatorWorker(
        settings=settings,
        runner_id=args.runner_id,
        poll_interval_seconds=args.poll_interval,
    )
    logger.info(
        "consolidator started runner_id=%s once=%s force=%s",
        worker.runner_id,
        args.once,
        args.force,
    )
    if args.once:
        result = worker.run_once(force=args.force)
        print(
            f"runner_id={result.runner_id} status={result.status} "
            f"runs={len(result.outcomes)}"
        )
        return

    worker.run_forever()
