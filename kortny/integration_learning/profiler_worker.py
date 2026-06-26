"""Background capability profiler worker (HIG-295).

Runs the capability profiler loop as a supervised ambient thread, completely
decoupled from the catalog sync path.  Each tick:

1. Acquires a Postgres advisory lock (single-leader across replicas).
2. Iterates all installations.
3. For each installation, finds toolkits with unenriched cards (or whose KG
   entity digest is stale).
4. Calls build_capability_profile(..., max_tools=<remaining budget>) per
   pending toolkit until the per-cycle budget is exhausted.
5. Commits per-toolkit so partial progress is never lost.

The lock key (7355608) is distinct from the catalog-sync lock (759340222),
the scheduler lock (759340185), and the consolidator lock (759340187).

Deferred upgrades:
- Lazy-profile-on-first-retrieval: profile a toolkit the first time the
  provider resolves it for a task rather than waiting for the next poll cycle.
- Usage-ranked tool prioritization: within a toolkit, process the most
  frequently retrieved tool slugs first so the highest-impact tools get
  enriched first when the budget is tight.
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from kortny.config import Settings, load_settings
from kortny.db.models import (
    ComposioConnection,
    ComposioToolCard,
    Installation,
    KnowledgeGraphEntity,
)
from kortny.db.session import make_session_factory
from kortny.integration_learning.profiles import (
    _DEFAULT_PROFILE_CARD_BUDGET,
    build_capability_profile,
)
from kortny.llm import LLMService
from kortny.llm.routing import ModelRouter, ModelRouteTier
from kortny.llm.runtime_config import (
    create_provider_for_selection,
    select_runtime_model,
)
from kortny.tasks.identity import TaskIdentity
from kortny.tasks.service import TaskService

logger = logging.getLogger(__name__)

# Advisory lock key for the profiler loop.  Must be unique across all ambient
# loops in the process.
PROFILER_ADVISORY_LOCK_KEY = 7355608

DEFAULT_PROFILER_POLL_INTERVAL_SECONDS = 60.0


class CapabilityProfilerWorker:
    """Time-boxed background loop that enriches Composio tool cards via LLM.

    The loop processes at most ``card_budget`` unenriched tools per tick,
    spread across all pending toolkits across all installations.  Progress is
    committed per-toolkit so partial enrichment is durable.
    """

    def __init__(
        self,
        *,
        session_factory: sessionmaker[Session] | None = None,
        settings: Settings | None = None,
        poll_interval_seconds: float | None = None,
        advisory_lock_key: int = PROFILER_ADVISORY_LOCK_KEY,
        use_advisory_lock: bool = True,
        card_budget: int = _DEFAULT_PROFILE_CARD_BUDGET,
    ) -> None:
        self.session_factory = session_factory or make_session_factory()
        self.settings = settings
        self.poll_interval_seconds = (
            poll_interval_seconds
            if poll_interval_seconds is not None
            else DEFAULT_PROFILER_POLL_INTERVAL_SECONDS
        )
        self.advisory_lock_key = advisory_lock_key
        self.use_advisory_lock = use_advisory_lock
        self.card_budget = max(1, card_budget)

    def run_once(self) -> None:
        """Profile one tick: process pending toolkits up to card_budget tools."""
        engine = self.session_factory.kw["bind"]
        with (
            engine.connect() as connection,
            Session(bind=connection, expire_on_commit=False) as session,
        ):
            if self.use_advisory_lock and not self._try_advisory_lock(session):
                logger.debug("capability_profiler advisory lock held; skipping tick")
                return
            try:
                self._profile_all(session)
            finally:
                if self.use_advisory_lock:
                    self._release_advisory_lock(session)

    def _profile_all(self, outer_session: Session) -> None:
        """Iterate installations and profile pending toolkits within budget."""
        settings = self._settings

        installation_ids: list[uuid.UUID] = list(
            outer_session.scalars(
                select(Installation.id).order_by(Installation.created_at)
            )
        )

        remaining_budget = self.card_budget

        for installation_id in installation_ids:
            if remaining_budget <= 0:
                logger.debug(
                    "capability_profiler budget exhausted installation_id=%s; "
                    "deferring remaining toolkits to next tick",
                    installation_id,
                )
                break

            pending_toolkits = self._pending_toolkits(outer_session, installation_id)
            if not pending_toolkits:
                continue

            for toolkit_slug in pending_toolkits:
                if remaining_budget <= 0:
                    break
                # Each toolkit gets its own session so a failure in one toolkit
                # does not roll back another toolkit's progress.
                engine = self.session_factory.kw["bind"]
                with (
                    engine.connect() as conn,
                    Session(bind=conn, expire_on_commit=False) as session,
                ):
                    try:
                        llm, task_id = self._build_llm(
                            session, installation_id, settings
                        )
                        build_capability_profile(
                            session,
                            installation_id=installation_id,
                            toolkit_slug=toolkit_slug,
                            llm=llm,
                            task_id=task_id,
                            max_tools=remaining_budget,
                        )
                        session.commit()
                        # Decrement budget by the number of cards processed.
                        # We count unenriched cards for this toolkit before the
                        # call; the actual number processed is min(count, budget).
                        unenriched_count = self._unenriched_card_count(
                            outer_session, installation_id, toolkit_slug
                        )
                        consumed = min(unenriched_count, remaining_budget)
                        remaining_budget -= consumed
                    except Exception:
                        session.rollback()
                        logger.exception(
                            "capability_profiler toolkit failed installation_id=%s "
                            "toolkit=%s; continuing",
                            installation_id,
                            toolkit_slug,
                        )

    def _pending_toolkits(
        self, session: Session, installation_id: uuid.UUID
    ) -> list[str]:
        """Return toolkit slugs that have at least one unenriched card or a stale digest."""
        # Connected toolkits for this installation.
        connected_slugs = set(
            session.scalars(
                select(ComposioConnection.toolkit_slug)
                .where(
                    ComposioConnection.installation_id == installation_id,
                    ComposioConnection.status == "active",
                    ComposioConnection.connected_account_id.is_not(None),
                )
                .distinct()
            )
        )
        if not connected_slugs:
            return []

        # Toolkits with at least one unenriched card.
        unenriched_toolkits = set(
            session.scalars(
                select(ComposioToolCard.toolkit_slug)
                .where(
                    ComposioToolCard.installation_id == installation_id,
                    ComposioToolCard.toolkit_slug.in_(list(connected_slugs)),
                    ComposioToolCard.enriched_description.is_(None),
                )
                .distinct()
            )
        )

        # Also include toolkits whose KG entity digest is stale (cards changed
        # since last full enrichment).
        stale_toolkits = self._stale_digest_toolkits(
            session, installation_id, connected_slugs - unenriched_toolkits
        )

        pending = sorted(unenriched_toolkits | stale_toolkits)
        if pending:
            logger.debug(
                "capability_profiler pending toolkits=%s installation_id=%s",
                pending,
                installation_id,
            )
        return pending

    def _stale_digest_toolkits(
        self,
        session: Session,
        installation_id: uuid.UUID,
        fully_enriched_slugs: set[str],
    ) -> set[str]:
        """Toolkits where the stored card_sha_digest no longer matches current cards."""
        stale: set[str] = set()
        for toolkit_slug in fully_enriched_slugs:
            canonical_key = f"composio_app:{toolkit_slug}"
            entity = session.scalars(
                select(KnowledgeGraphEntity).where(
                    KnowledgeGraphEntity.installation_id == installation_id,
                    KnowledgeGraphEntity.canonical_key == canonical_key,
                )
            ).first()
            if entity is None:
                # No entity yet — needs profiling.
                stale.add(toolkit_slug)
                continue
            stored_digest = (
                (entity.attrs_json or {})
                .get("generated_from", {})
                .get("card_sha_digest")
            )
            if stored_digest is None:
                stale.add(toolkit_slug)
                continue
            # Compute current digest.
            card_rows = session.execute(
                select(
                    ComposioToolCard.tool_slug,
                    ComposioToolCard.card_sha,
                ).where(
                    ComposioToolCard.installation_id == installation_id,
                    ComposioToolCard.toolkit_slug == toolkit_slug,
                )
            ).all()
            if not card_rows:
                continue
            sorted_shas = "".join(
                row.card_sha for row in sorted(card_rows, key=lambda r: r.tool_slug)
            )
            current_digest = hashlib.sha256(sorted_shas.encode()).hexdigest()
            if current_digest != stored_digest:
                stale.add(toolkit_slug)
        return stale

    def _unenriched_card_count(
        self, session: Session, installation_id: uuid.UUID, toolkit_slug: str
    ) -> int:
        """Count unenriched cards for a toolkit (used for budget accounting)."""
        result = session.scalar(
            select(func.count(ComposioToolCard.tool_slug)).where(
                ComposioToolCard.installation_id == installation_id,
                ComposioToolCard.toolkit_slug == toolkit_slug,
                ComposioToolCard.enriched_description.is_(None),
            )
        )
        return int(result or 0)

    def _build_llm(
        self,
        session: Session,
        installation_id: uuid.UUID,
        settings: Settings,
    ) -> tuple[LLMService, uuid.UUID]:
        """Build an LLMService on the profiler tier and a stable synthetic task."""
        model_route = ModelRouter(settings).route_for_tier(
            ModelRouteTier.profiler,
            reason="capability_profile",
        )
        selection = select_runtime_model(
            session=session,
            settings=settings,
            installation_id=installation_id,
            model_route=model_route,
        )
        llm = LLMService(
            session=session,
            provider=create_provider_for_selection(
                settings=settings,
                selection=selection,
            ),
            provider_name=selection.provider_name,
            model_route=selection.model_route,
            settings=settings,
        )
        # Stable synthetic task per installation for cost attribution.
        # identity_key = synthetic:capability-profiler:{installation_id}
        task_service = TaskService(session)
        identity = TaskIdentity.synthetic(
            source="capability-profiler",
            source_id=str(installation_id),
            input_text="capability profiler: enrich composio tool descriptions",
        )
        task = task_service.create_task(
            installation_id=installation_id,
            slack_channel_id="SYSTEM",
            slack_user_id="SYSTEM",
            input="capability profiler: enrich composio tool descriptions",
            identity=identity,
        )
        return llm, task.id

    def run_forever(self) -> None:
        """Poll loop: run_once() every poll_interval_seconds."""
        while True:
            try:
                self.run_once()
            except Exception:
                logger.exception("capability_profiler tick failed; continuing")
            time.sleep(self.poll_interval_seconds)

    def _try_advisory_lock(self, session: Session) -> bool:
        return bool(
            session.scalar(select(func.pg_try_advisory_lock(self.advisory_lock_key)))
        )

    def _release_advisory_lock(self, session: Session) -> None:
        session.execute(select(func.pg_advisory_unlock(self.advisory_lock_key)))

    @property
    def _settings(self) -> Settings:
        if self.settings is None:
            self.settings = load_settings()
        return self.settings
