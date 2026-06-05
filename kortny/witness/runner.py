"""Operational Witness runner.

The runner coordinates existing Witness primitives:

- active channel profiles are re-read by the LLM extractor;
- candidate rows are persisted through the opportunity service;
- optional delivery is restricted to DM-scoped candidates by lifecycle policy.
"""

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

from slack_sdk import WebClient
from sqlalchemy import and_, desc, func, select
from sqlalchemy.orm import Session, sessionmaker

from kortny.config import Settings, load_settings
from kortny.db.models import (
    LLMProvider as DbLLMProvider,
)
from kortny.db.models import (
    ObserveChannelProfile,
    SlackChannelMembership,
    Task,
    TaskEventType,
    TaskStatus,
    WitnessOpportunityCandidate,
)
from kortny.db.session import make_session_factory
from kortny.llm import LLMProvider, LLMService, ModelRoute, ModelRouter, ModelRouteTier
from kortny.llm.runtime_config import (
    create_provider_for_selection,
    select_runtime_model,
)
from kortny.logging_config import configure_logging
from kortny.observability import configure_tracing, start_span
from kortny.tasks import TaskService
from kortny.witness.extractor import WitnessChannelProfileExtractor
from kortny.witness.lifecycle import WitnessSlackClient, send_private_suggestion
from kortny.witness.opportunities import (
    WITNESS_OPPORTUNITY_CANDIDATES_PROJECTED_MESSAGE,
    WitnessOpportunityService,
)

logger = logging.getLogger(__name__)

DEFAULT_WITNESS_PROFILE_SCAN_LIMIT = 10
DEFAULT_WITNESS_DELIVERY_LIMIT = 5
DEFAULT_WITNESS_SCAN_INTERVAL = timedelta(hours=6)
DEFAULT_WITNESS_POLL_INTERVAL_SECONDS = 300.0
DEFAULT_WITNESS_ADVISORY_LOCK_KEY = 759340186

WITNESS_RUNNER_PROFILE_SCAN_STARTED_MESSAGE = "witness_runner_profile_scan_started"
WITNESS_RUNNER_PROFILE_SCAN_COMPLETED_MESSAGE = "witness_runner_profile_scan_completed"
WITNESS_RUNNER_PROFILE_SCAN_FAILED_MESSAGE = "witness_runner_profile_scan_failed"
WITNESS_RUNNER_DELIVERY_SKIPPED_MESSAGE = "witness_runner_delivery_skipped"
WITNESS_RUNNER_DELIVERY_SENT_MESSAGE = "witness_runner_delivery_sent"


@dataclass(frozen=True, slots=True)
class WitnessProjectionOutcome:
    """Projection result for one channel profile."""

    task_id: uuid.UUID
    channel_id: str
    profile_id: uuid.UUID
    created_count: int
    updated_count: int
    skipped_count: int
    candidate_ids: tuple[str, ...]
    raw_candidate_count: int
    skipped_reason: str | None

    @property
    def total_count(self) -> int:
        return self.created_count + self.updated_count


@dataclass(frozen=True, slots=True)
class WitnessDeliveryOutcome:
    """Delivery result for one candidate."""

    candidate_id: uuid.UUID
    status: str
    reason: str | None = None
    channel_id: str | None = None
    message_ts: str | None = None


@dataclass(frozen=True, slots=True)
class WitnessRunResult:
    """Outcome from one Witness runner tick."""

    runner_id: str
    status: str
    projections: tuple[WitnessProjectionOutcome, ...] = ()
    deliveries: tuple[WitnessDeliveryOutcome, ...] = ()
    leader_acquired: bool = True

    @property
    def projected_count(self) -> int:
        return sum(outcome.total_count for outcome in self.projections)

    @property
    def delivered_count(self) -> int:
        return sum(1 for outcome in self.deliveries if outcome.status == "sent")


class WitnessRunner:
    """Project and optionally deliver conservative Witness suggestions."""

    def __init__(
        self,
        session: Session,
        *,
        settings: Settings | None = None,
        llm_provider: LLMProvider | None = None,
        provider_name: DbLLMProvider | str | None = None,
        slack_client: WitnessSlackClient | None = None,
        runner_id: str | None = None,
        advisory_lock_key: int = DEFAULT_WITNESS_ADVISORY_LOCK_KEY,
    ) -> None:
        self.session = session
        self.settings = settings
        self.llm_provider = llm_provider
        self.provider_name = provider_name
        self.slack_client = slack_client
        self.runner_id = runner_id or default_witness_runner_id()
        self.advisory_lock_key = advisory_lock_key

    def run_once(
        self,
        *,
        installation_id: uuid.UUID | None = None,
        now: datetime | None = None,
        profile_limit: int = DEFAULT_WITNESS_PROFILE_SCAN_LIMIT,
        delivery_limit: int = DEFAULT_WITNESS_DELIVERY_LIMIT,
        deliver_private: bool = False,
        min_scan_interval: timedelta = DEFAULT_WITNESS_SCAN_INTERVAL,
        use_advisory_lock: bool = False,
    ) -> WitnessRunResult:
        """Run one Witness tick.

        Channel delivery is intentionally not supported here. Private delivery
        only attempts candidates that already have DM scope.
        """

        if profile_limit < 0:
            raise ValueError("profile_limit must be non-negative")
        if delivery_limit < 0:
            raise ValueError("delivery_limit must be non-negative")
        run_at = _coerce_utc(now)

        lock_acquired = True
        if use_advisory_lock:
            lock_acquired = self._try_advisory_lock()
        if not lock_acquired:
            return WitnessRunResult(
                runner_id=self.runner_id,
                status="lock_skipped",
                leader_acquired=False,
            )

        try:
            with start_span(
                "witness.run",
                attributes={
                    "openinference.span.kind": "CHAIN",
                    "witness.runner.id": self.runner_id,
                    "witness.deliver_private": deliver_private,
                },
            ):
                projections = self._project_due_profiles(
                    installation_id=installation_id,
                    now=run_at,
                    limit=profile_limit,
                    min_scan_interval=min_scan_interval,
                )
                deliveries = (
                    self._deliver_private_suggestions(
                        installation_id=installation_id,
                        now=run_at,
                        limit=delivery_limit,
                    )
                    if deliver_private
                    else ()
                )
            status = "processed" if projections or deliveries else "idle"
            return WitnessRunResult(
                runner_id=self.runner_id,
                status=status,
                projections=projections,
                deliveries=deliveries,
            )
        finally:
            if use_advisory_lock:
                self._release_advisory_lock()

    def _project_due_profiles(
        self,
        *,
        installation_id: uuid.UUID | None,
        now: datetime,
        limit: int,
        min_scan_interval: timedelta,
    ) -> tuple[WitnessProjectionOutcome, ...]:
        if limit == 0:
            return ()
        rows = self._candidate_profiles(
            installation_id=installation_id,
            limit=limit * 3,
        )
        outcomes: list[WitnessProjectionOutcome] = []
        for profile, membership in rows:
            if len(outcomes) >= limit:
                break
            if not _profile_scan_due(
                profile,
                now=now,
                min_scan_interval=min_scan_interval,
            ):
                continue
            try:
                outcomes.append(
                    self._project_profile(
                        profile=profile,
                        membership=membership,
                        now=now,
                    )
                )
            except Exception:
                logger.exception(
                    "witness profile scan failed runner_id=%s profile_id=%s channel_id=%s",
                    self.runner_id,
                    profile.id,
                    membership.channel_id,
                )
        return tuple(outcomes)

    def _project_profile(
        self,
        *,
        profile: ObserveChannelProfile,
        membership: SlackChannelMembership,
        now: datetime,
    ) -> WitnessProjectionOutcome:
        task_service = TaskService(self.session)
        task = self._create_scan_task(
            task_service=task_service,
            profile=profile,
            membership=membership,
            now=now,
        )
        task_service.append_event(
            task,
            TaskEventType.log,
            {
                "message": WITNESS_RUNNER_PROFILE_SCAN_STARTED_MESSAGE,
                "runner_id": self.runner_id,
                "channel_id": membership.channel_id,
                "profile_id": str(profile.id),
                "profile_version": profile.profile_version,
            },
        )
        task_service.transition(task, TaskStatus.running)
        try:
            extraction = self._profile_extractor(
                task=task,
                task_service=task_service,
            ).extract(
                task=task,
                membership=membership,
                profile=profile,
            )
            result = WitnessOpportunityService(
                self.session
            ).project_from_channel_profile(
                task=task,
                membership=membership,
                profile=profile,
                candidates=extraction.candidates,
                extraction_metadata={
                    "runner_id": self.runner_id,
                    "runner_source": "witness_runner",
                    "raw_candidate_count": extraction.raw_candidate_count,
                    "skipped_reason": extraction.skipped_reason,
                },
            )
            _mark_profile_scanned(
                profile,
                now=now,
                task_id=task.id,
                runner_id=self.runner_id,
            )
            task.result_summary = (
                f"Witness projected {result.total_count} candidate(s) from "
                f"{membership.channel_id}."
            )
            task_service.append_event(
                task,
                TaskEventType.log,
                {
                    "message": WITNESS_OPPORTUNITY_CANDIDATES_PROJECTED_MESSAGE,
                    "source_type": "channel_profile",
                    "extractor": "llm",
                    "runner_id": self.runner_id,
                    "channel_id": membership.channel_id,
                    "membership_id": str(membership.id),
                    "profile_id": str(profile.id),
                    "raw_candidate_count": extraction.raw_candidate_count,
                    "skipped_reason": extraction.skipped_reason,
                    "created_count": result.created_count,
                    "updated_count": result.updated_count,
                    "skipped_count": result.skipped_count,
                    "candidate_ids": list(result.candidate_ids),
                },
            )
            task_service.append_event(
                task,
                TaskEventType.log,
                {
                    "message": WITNESS_RUNNER_PROFILE_SCAN_COMPLETED_MESSAGE,
                    "runner_id": self.runner_id,
                    "channel_id": membership.channel_id,
                    "profile_id": str(profile.id),
                    "created_count": result.created_count,
                    "updated_count": result.updated_count,
                    "skipped_count": result.skipped_count,
                },
            )
            task_service.transition(task, TaskStatus.succeeded)
            self.session.flush()
            return WitnessProjectionOutcome(
                task_id=task.id,
                channel_id=membership.channel_id,
                profile_id=profile.id,
                created_count=result.created_count,
                updated_count=result.updated_count,
                skipped_count=result.skipped_count,
                candidate_ids=result.candidate_ids,
                raw_candidate_count=extraction.raw_candidate_count,
                skipped_reason=extraction.skipped_reason,
            )
        except Exception as exc:
            task.error = {
                "type": type(exc).__name__,
                "message": str(exc),
                "runner_id": self.runner_id,
            }
            task_service.append_event(
                task,
                TaskEventType.error,
                {
                    "message": WITNESS_RUNNER_PROFILE_SCAN_FAILED_MESSAGE,
                    "runner_id": self.runner_id,
                    "channel_id": membership.channel_id,
                    "profile_id": str(profile.id),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            task_service.transition(task, TaskStatus.failed)
            self.session.flush()
            raise

    def _deliver_private_suggestions(
        self,
        *,
        installation_id: uuid.UUID | None,
        now: datetime,
        limit: int,
    ) -> tuple[WitnessDeliveryOutcome, ...]:
        if limit == 0:
            return ()
        if self.slack_client is None:
            return ()

        candidates = self._eligible_dm_candidates(
            installation_id=installation_id,
            now=now,
            limit=limit,
        )
        outcomes: list[WitnessDeliveryOutcome] = []
        for candidate in candidates:
            try:
                result = send_private_suggestion(
                    self.session,
                    candidate.id,
                    installation_id=candidate.installation_id,
                    by_user_id="witness_runner",
                    client=self.slack_client,
                    now=now,
                )
            except ValueError as exc:
                _append_candidate_event(
                    self.session,
                    candidate,
                    message=WITNESS_RUNNER_DELIVERY_SKIPPED_MESSAGE,
                    payload={
                        "runner_id": self.runner_id,
                        "reason": str(exc),
                    },
                )
                outcomes.append(
                    WitnessDeliveryOutcome(
                        candidate_id=candidate.id,
                        status="skipped",
                        reason=str(exc),
                    )
                )
                continue
            _append_candidate_event(
                self.session,
                candidate,
                message=WITNESS_RUNNER_DELIVERY_SENT_MESSAGE,
                payload={
                    "runner_id": self.runner_id,
                    "channel_id": result.channel_id,
                    "message_ts": result.message_ts,
                    "side_effect_id": str(result.side_effect_id),
                    "deduped": result.deduped,
                },
            )
            outcomes.append(
                WitnessDeliveryOutcome(
                    candidate_id=candidate.id,
                    status="sent",
                    channel_id=result.channel_id,
                    message_ts=result.message_ts,
                )
            )
        self.session.flush()
        return tuple(outcomes)

    def _candidate_profiles(
        self,
        *,
        installation_id: uuid.UUID | None,
        limit: int,
    ) -> tuple[tuple[ObserveChannelProfile, SlackChannelMembership], ...]:
        filters = [
            ObserveChannelProfile.profile_status == "active",
            SlackChannelMembership.membership_status == "active",
        ]
        if installation_id is not None:
            filters.append(ObserveChannelProfile.installation_id == installation_id)
        return tuple(
            self.session.execute(
                select(ObserveChannelProfile, SlackChannelMembership)
                .join(
                    SlackChannelMembership,
                    and_(
                        SlackChannelMembership.installation_id
                        == ObserveChannelProfile.installation_id,
                        SlackChannelMembership.channel_id
                        == ObserveChannelProfile.channel_id,
                    ),
                )
                .where(*filters)
                .order_by(
                    desc(ObserveChannelProfile.last_profiled_at),
                    desc(ObserveChannelProfile.updated_at),
                )
                .limit(limit)
            ).all()
        )

    def _eligible_dm_candidates(
        self,
        *,
        installation_id: uuid.UUID | None,
        now: datetime,
        limit: int,
    ) -> tuple[WitnessOpportunityCandidate, ...]:
        filters = [
            WitnessOpportunityCandidate.status == "candidate",
            WitnessOpportunityCandidate.visibility_scope_type == "dm",
            WitnessOpportunityCandidate.channel_id.like("D%"),
            WitnessOpportunityCandidate.target_slack_user_id.is_not(None),
            (
                (WitnessOpportunityCandidate.cooldown_until.is_(None))
                | (WitnessOpportunityCandidate.cooldown_until <= now)
            ),
        ]
        if installation_id is not None:
            filters.append(
                WitnessOpportunityCandidate.installation_id == installation_id
            )
        return tuple(
            self.session.scalars(
                select(WitnessOpportunityCandidate)
                .where(*filters)
                .order_by(
                    WitnessOpportunityCandidate.confidence_score.desc(),
                    WitnessOpportunityCandidate.created_at.asc(),
                )
                .with_for_update(skip_locked=True)
                .limit(limit)
            )
        )

    def _create_scan_task(
        self,
        *,
        task_service: TaskService,
        profile: ObserveChannelProfile,
        membership: SlackChannelMembership,
        now: datetime,
    ) -> Task:
        channel_label = (
            f"#{membership.channel_name}"
            if membership.channel_name
            else membership.channel_id
        )
        return task_service.create_task(
            installation_id=profile.installation_id,
            slack_event_id=(
                f"witness:{profile.id}:{profile.profile_version}:{int(now.timestamp())}"
            ),
            slack_channel_id=membership.channel_id,
            slack_thread_ts=membership.channel_id,
            slack_message_ts=None,
            slack_user_id=membership.added_by_user_id or "witness_runner",
            input=f"Run Witness opportunity scan for {channel_label}.",
            source_surface="witness_runner",
        )

    def _profile_extractor(
        self,
        *,
        task: Task,
        task_service: TaskService,
    ) -> WitnessChannelProfileExtractor:
        if self.llm_provider is not None:
            model_route = ModelRoute(
                tier=ModelRouteTier.cheap_fast,
                model=self.llm_provider.model,
                reason="witness_runner_channel_profile_extraction",
            )
            provider = self.llm_provider
            provider_name = self.provider_name or DbLLMProvider.openrouter
        else:
            model_route = ModelRouter(self._settings).route_for_tier(
                ModelRouteTier.cheap_fast,
                reason="witness_runner_channel_profile_extraction",
            )
            selection = select_runtime_model(
                session=self.session,
                settings=self._settings,
                installation_id=task.installation_id,
                model_route=model_route,
            )
            model_route = selection.model_route
            provider = create_provider_for_selection(
                settings=self._settings,
                selection=selection,
            )
            provider_name = selection.provider_name
        return WitnessChannelProfileExtractor(
            LLMService(
                session=self.session,
                provider=provider,
                provider_name=provider_name,
                task_service=task_service,
                model_route=model_route,
            )
        )

    @property
    def _settings(self) -> Settings:
        if self.settings is None:
            self.settings = load_settings()
        return self.settings

    def _try_advisory_lock(self) -> bool:
        return bool(
            self.session.scalar(
                select(func.pg_try_advisory_lock(self.advisory_lock_key))
            )
        )

    def _release_advisory_lock(self) -> None:
        self.session.execute(select(func.pg_advisory_unlock(self.advisory_lock_key)))


class WitnessWorker:
    """Poll the Witness runner forever."""

    def __init__(
        self,
        *,
        session_factory: sessionmaker[Session] | None = None,
        settings: Settings | None = None,
        runner_id: str | None = None,
        poll_interval_seconds: float = DEFAULT_WITNESS_POLL_INTERVAL_SECONDS,
        profile_limit: int = DEFAULT_WITNESS_PROFILE_SCAN_LIMIT,
        delivery_limit: int = DEFAULT_WITNESS_DELIVERY_LIMIT,
        deliver_private: bool = False,
        use_advisory_lock: bool = True,
    ) -> None:
        self.session_factory = session_factory or make_session_factory()
        self.settings = settings
        self.runner_id = runner_id or default_witness_runner_id()
        self.poll_interval_seconds = poll_interval_seconds
        self.profile_limit = profile_limit
        self.delivery_limit = delivery_limit
        self.deliver_private = deliver_private
        self.use_advisory_lock = use_advisory_lock

    def run_once(self, *, now: datetime | None = None) -> WitnessRunResult:
        with self.session_factory.begin() as session:
            slack_client = (
                WebClient(token=self._settings.slack_bot_token)
                if self.deliver_private
                else None
            )
            result = WitnessRunner(
                session,
                settings=self._settings,
                slack_client=slack_client,
                runner_id=self.runner_id,
            ).run_once(
                now=now,
                profile_limit=self.profile_limit,
                delivery_limit=self.delivery_limit,
                deliver_private=self.deliver_private,
                use_advisory_lock=self.use_advisory_lock,
            )
            logger.info(
                "witness runner tick runner_id=%s status=%s projected=%s delivered=%s",
                result.runner_id,
                result.status,
                result.projected_count,
                result.delivered_count,
            )
            return result

    def run_forever(self) -> None:
        while True:
            self.run_once()
            time.sleep(self.poll_interval_seconds)

    @property
    def _settings(self) -> Settings:
        if self.settings is None:
            self.settings = load_settings()
        return self.settings


def default_witness_runner_id() -> str:
    return f"{socket.gethostname()}-{os.getpid()}"


def _profile_scan_due(
    profile: ObserveChannelProfile,
    *,
    now: datetime,
    min_scan_interval: timedelta,
) -> bool:
    metadata = profile.metadata_json if isinstance(profile.metadata_json, dict) else {}
    runner = metadata.get("witness_runner")
    if not isinstance(runner, dict):
        return True
    scanned_version = runner.get("profile_version")
    if scanned_version != profile.profile_version:
        return True
    scanned_at = _parse_datetime(runner.get("last_scanned_at"))
    if scanned_at is None:
        return True
    return scanned_at <= now - min_scan_interval


def _mark_profile_scanned(
    profile: ObserveChannelProfile,
    *,
    now: datetime,
    task_id: uuid.UUID,
    runner_id: str,
) -> None:
    metadata = (
        dict(profile.metadata_json) if isinstance(profile.metadata_json, dict) else {}
    )
    metadata["witness_runner"] = {
        "last_scanned_at": now.isoformat(),
        "profile_version": profile.profile_version,
        "task_id": str(task_id),
        "runner_id": runner_id,
    }
    profile.metadata_json = metadata
    profile.updated_at = now


def _append_candidate_event(
    session: Session,
    candidate: WitnessOpportunityCandidate,
    *,
    message: str,
    payload: dict[str, object],
) -> None:
    if candidate.source_task_id is None:
        return
    task = session.get(Task, candidate.source_task_id)
    if task is None:
        return
    TaskService(session).append_event(
        task,
        TaskEventType.log,
        {
            "message": message,
            "candidate_id": str(candidate.id),
            **payload,
        },
    )


def _parse_datetime(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return _coerce_utc(parsed)


def _coerce_utc(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(UTC)
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entrypoint for the Witness runner."""

    configure_logging()
    parser = argparse.ArgumentParser(description="Run the Kortny Witness runner")
    parser.add_argument("--once", action="store_true", help="Run one Witness tick")
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
    parser.add_argument(
        "--profile-limit",
        type=int,
        default=DEFAULT_WITNESS_PROFILE_SCAN_LIMIT,
        help="Maximum channel profiles to scan per tick",
    )
    parser.add_argument(
        "--delivery-limit",
        type=int,
        default=DEFAULT_WITNESS_DELIVERY_LIMIT,
        help="Maximum private suggestions to deliver per tick",
    )
    parser.add_argument(
        "--deliver-private",
        action="store_true",
        help="Send eligible DM-scoped suggestions",
    )
    args = parser.parse_args(argv)

    settings = load_settings()
    configure_tracing(settings)

    worker = WitnessWorker(
        settings=settings,
        runner_id=args.runner_id,
        poll_interval_seconds=(
            args.poll_interval
            if args.poll_interval is not None
            else DEFAULT_WITNESS_POLL_INTERVAL_SECONDS
        ),
        profile_limit=args.profile_limit,
        delivery_limit=args.delivery_limit,
        deliver_private=args.deliver_private,
    )
    logger.info(
        "witness runner started runner_id=%s once=%s deliver_private=%s",
        worker.runner_id,
        args.once,
        args.deliver_private,
    )
    if args.once:
        result = worker.run_once()
        print(
            f"runner_id={result.runner_id} status={result.status} "
            f"projected_count={result.projected_count} "
            f"delivered_count={result.delivered_count}"
        )
        return

    worker.run_forever()
