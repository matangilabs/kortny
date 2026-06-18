"""Consolidation run orchestration (HIG-225).

One ``run_once`` call executes all consolidation passes for one installation,
records a ``consolidation_runs`` row with per-pass counters and LLM cost, and
keeps each pass failure-isolated: a broken pass is noted in
``counters_json["pass_errors"]`` and the run still succeeds.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import cast

from slack_sdk import WebClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from kortny.config import Settings
from kortny.consolidator.org_profile import OrgProfilePass
from kortny.consolidator.passes import (
    CONSOLIDATOR_EXTRACTOR,
    adjudicate_candidates,
    age_graph,
    backfill_embeddings,
    merge_duplicate_entities,
    project_confirmed_facts,
    run_hygiene,
)
from kortny.consolidator.project_inference import ProjectInferencePass
from kortny.consolidator.promotion import EpisodePromotionPass
from kortny.consolidator.style_cards import (
    DEFAULT_STYLE_CARD_MIN_MESSAGES,
    StyleCardPass,
)
from kortny.db.models import (
    ConsolidationRun,
    Episode,
    KnowledgeGraphEntity,
    ObservationEvent,
    Task,
    TaskStatus,
)
from kortny.db.models import (
    LLMProvider as DbLLMProvider,
)
from kortny.embeddings import EmbeddingIndex
from kortny.knowledge_graph import GraphService
from kortny.llm import (
    LLMProvider,
    LLMService,
    ModelRoute,
    ModelRouter,
    ModelRouteTier,
)
from kortny.llm.runtime_config import (
    create_provider_for_selection,
    select_runtime_model,
)
from kortny.memory.service import ConfirmationPoster
from kortny.slack import SlackPoster
from kortny.slack.posting import SlackPostingClient
from kortny.tasks import TaskService
from kortny.tasks.identity import TaskIdentity

logger = logging.getLogger(__name__)

CONSOLIDATION_TASK_SOURCE = "consolidator"
DEFAULT_KG_STALE_DAYS = 45
DEFAULT_PROMOTION_EPISODE_CAP = 50


@dataclass(frozen=True, slots=True)
class ConsolidationOutcome:
    """Result of one consolidation run."""

    run_id: uuid.UUID
    installation_id: uuid.UUID
    status: str
    counters: dict[str, object]
    task_id: uuid.UUID | None = None


class ConsolidationService:
    """Runs the full consolidation pass sequence for one installation."""

    def __init__(
        self,
        session: Session,
        *,
        settings: Settings | None = None,
        llm_provider: LLMProvider | None = None,
        provider_name: DbLLMProvider | str | None = None,
        embedding_index: EmbeddingIndex | None = None,
        kg_stale_days: int | None = None,
        promotion_episode_cap: int = DEFAULT_PROMOTION_EPISODE_CAP,
        style_card_min_messages: int | None = None,
    ) -> None:
        self.session = session
        self.settings = settings
        self.llm_provider = llm_provider
        self.provider_name = provider_name
        self.embedding_index = embedding_index
        self.kg_stale_days = kg_stale_days or (
            settings.kg_stale_days if settings is not None else DEFAULT_KG_STALE_DAYS
        )
        self.promotion_episode_cap = promotion_episode_cap
        self.style_card_min_messages = style_card_min_messages or (
            settings.style_card_min_messages
            if settings is not None
            else DEFAULT_STYLE_CARD_MIN_MESSAGES
        )

    # -- trigger inputs -----------------------------------------------------

    def fail_stale_runs(
        self,
        *,
        older_than_hours: int = 2,
        now: datetime | None = None,
    ) -> int:
        """Mark long-`running` runs as failed (dead process, e.g. OOM kill).

        The advisory lock guarantees one live runner, so a `running` row
        older than any plausible run duration belongs to a killed process.
        """

        effective_now = now or datetime.now(UTC)
        cutoff = effective_now - timedelta(hours=older_than_hours)
        stale = list(
            self.session.scalars(
                select(ConsolidationRun).where(
                    ConsolidationRun.status == "running",
                    ConsolidationRun.started_at < cutoff,
                )
            )
        )
        task_service = TaskService(self.session)
        for run in stale:
            run.status = "failed"
            run.error = "interrupted"
            run.finished_at = effective_now
            task = self.session.scalar(
                select(Task).where(
                    Task.slack_event_id == f"consolidator:{run.id}",
                    Task.status == TaskStatus.running,
                )
            )
            if task is not None:
                task_service.transition(task, TaskStatus.failed)
                task.error = {"code": "interrupted", "message": "interrupted"}
        if stale:
            self.session.commit()
            logger.warning(
                "consolidator recovered stale runs count=%s run_ids=%s",
                len(stale),
                ",".join(str(run.id) for run in stale),
            )
        return len(stale)

    def last_successful_run_started_at(
        self, installation_id: uuid.UUID
    ) -> datetime | None:
        return self.session.scalar(
            select(func.max(ConsolidationRun.started_at)).where(
                ConsolidationRun.installation_id == installation_id,
                ConsolidationRun.status == "succeeded",
            )
        )

    def new_item_count(
        self,
        installation_id: uuid.UUID,
        since: datetime | None,
    ) -> int:
        total = 0
        for model, column in (
            (Episode, Episode.created_at),
            (ObservationEvent, ObservationEvent.observed_at),
            (KnowledgeGraphEntity, KnowledgeGraphEntity.created_at),
        ):
            predicates = [model.installation_id == installation_id]
            if model is KnowledgeGraphEntity:
                predicates.append(KnowledgeGraphEntity.lifecycle_state == "candidate")
            if since is not None:
                predicates.append(column > since)
            total += int(
                self.session.scalar(
                    select(func.count()).select_from(model).where(*predicates)
                )
                or 0
            )
        return total

    def promotion_since(self, installation_id: uuid.UUID) -> datetime | None:
        """Window start for episode promotion.

        Uses the anchor (created_at of the last episode the promotion pass
        actually processed) recorded by the previous successful run, so a
        backlog larger than the per-run cap drains across runs instead of
        being skipped. Falls back to the run start time for older runs that
        recorded no anchor.
        """

        last_run = self.session.scalar(
            select(ConsolidationRun)
            .where(
                ConsolidationRun.installation_id == installation_id,
                ConsolidationRun.status == "succeeded",
            )
            .order_by(ConsolidationRun.started_at.desc())
            .limit(1)
        )
        if last_run is None:
            return None
        counters = (
            last_run.counters_json if isinstance(last_run.counters_json, dict) else {}
        )
        promotion = counters.get("promotion")
        if isinstance(promotion, dict) and "anchor" in promotion:
            anchor = promotion.get("anchor")
            if isinstance(anchor, str) and anchor:
                try:
                    return datetime.fromisoformat(anchor)
                except ValueError:
                    return last_run.started_at
            # Anchor recorded as null: nothing was processed yet (e.g. the
            # LLM was unavailable) — keep the window open from the start.
            return None
        return last_run.started_at

    def last_activity_at(self, installation_id: uuid.UUID) -> datetime | None:
        latest_observation = self.session.scalar(
            select(func.max(ObservationEvent.observed_at)).where(
                ObservationEvent.installation_id == installation_id
            )
        )
        # Only user-driven tasks count as activity; synthetic/scheduled tasks
        # (including the consolidator's own) must not block the quiet window.
        latest_task = self.session.scalar(
            select(func.max(Task.created_at)).where(
                Task.installation_id == installation_id,
                (Task.identity_kind.is_(None))
                | (Task.identity_kind.in_(("slack_message", "slack_event", "manual"))),
            )
        )
        candidates = [
            value for value in (latest_observation, latest_task) if value is not None
        ]
        if not candidates:
            return None
        return max(candidates)

    # -- run ----------------------------------------------------------------

    def run_once(
        self,
        *,
        installation_id: uuid.UUID,
        now: datetime | None = None,
    ) -> ConsolidationOutcome:
        effective_now = now or datetime.now(UTC)
        since = self.promotion_since(installation_id)
        run = ConsolidationRun(
            installation_id=installation_id,
            started_at=effective_now,
            status="running",
        )
        self.session.add(run)
        self.session.flush()

        task_service = TaskService(self.session)
        task = self._create_run_task(
            task_service=task_service,
            installation_id=installation_id,
            run_id=run.id,
        )
        task_service.transition(task, TaskStatus.running)
        # Commit before any pass runs: an OOM kill mid-run must leave a
        # visible `running` row (recovered by fail_stale_runs), not nothing.
        self.session.commit()

        counters: dict[str, object] = {}
        pass_errors: dict[str, str] = {}
        graph = GraphService(self.session, embedding_index=self.embedding_index)
        llm = self._llm_service(
            task=task, task_service=task_service, pass_errors=pass_errors
        )

        passes: list[tuple[str, Callable[[], Mapping[str, object]]]] = [
            (
                "promotion",
                lambda: (
                    EpisodePromotionPass(
                        self.session,
                        graph=graph,
                        llm=llm,
                        embedding_index=self.embedding_index,
                    )
                    .run(
                        installation_id=installation_id,
                        task=task,
                        since=since,
                        now=effective_now,
                        episode_cap=self.promotion_episode_cap,
                    )
                    .to_payload()
                ),
            ),
            (
                "adjudication",
                lambda: adjudicate_candidates(
                    self.session,
                    installation_id=installation_id,
                    now=effective_now,
                ).to_payload(),
            ),
            (
                "merge",
                lambda: merge_duplicate_entities(
                    self.session,
                    installation_id=installation_id,
                    graph=graph,
                    embedding_index=self.embedding_index,
                    llm=llm,
                    task=task,
                    now=effective_now,
                ).to_payload(),
            ),
            (
                "aging",
                lambda: age_graph(
                    self.session,
                    installation_id=installation_id,
                    graph=graph,
                    stale_days=self.kg_stale_days,
                    now=effective_now,
                ).to_payload(),
            ),
            (
                "fact_reconciliation",
                lambda: project_confirmed_facts(
                    self.session,
                    installation_id=installation_id,
                    graph=graph,
                    task=task,
                    now=effective_now,
                ).to_payload(),
            ),
            (
                "hygiene",
                lambda: run_hygiene(
                    self.session,
                    installation_id=installation_id,
                    task_service=task_service,
                    now=effective_now,
                ).to_payload(),
            ),
            (
                "style_cards",
                lambda: (
                    StyleCardPass(
                        self.session,
                        llm=llm,
                        min_messages=self.style_card_min_messages,
                    )
                    .run(
                        installation_id=installation_id,
                        task=task,
                        now=effective_now,
                    )
                    .to_payload()
                ),
            ),
            (
                "org_profile",
                lambda: (
                    self._build_org_profile_pass(llm=llm)
                    .run(
                        installation_id=installation_id,
                        task=task,
                        now=effective_now,
                    )
                    .to_payload()
                ),
            ),
            (
                "project_inference",
                lambda: (
                    ProjectInferencePass(self.session, llm=llm)
                    .run(
                        installation_id=installation_id,
                        task_id=task.id,
                        now=effective_now,
                    )
                    .to_payload()
                ),
            ),
            (
                "backfill",
                lambda: backfill_embeddings(
                    self.session,
                    installation_id=installation_id,
                    embedding_index=self.embedding_index,
                ).to_payload(),
            ),
        ]
        for pass_name, pass_fn in passes:
            try:
                counters[pass_name] = pass_fn()
            except Exception as exc:
                logger.exception(
                    "consolidation pass failed installation_id=%s pass=%s",
                    installation_id,
                    pass_name,
                )
                # Roll back the failed pass's partial writes so the session
                # is usable for the remaining passes, then persist the error
                # marker. Prior passes already committed.
                self.session.rollback()
                pass_errors[pass_name] = f"{type(exc).__name__}: {exc}"
                run.counters_json = {**counters, "pass_errors": dict(pass_errors)}
                self.session.commit()
            else:
                # Commit each completed pass: a crash later in the run keeps
                # this pass's work (the first live run lost ~20 minutes of
                # LLM output to a single-transaction OOM kill).
                run.counters_json = dict(counters)
                self.session.commit()

        if "promotion" not in counters:
            # Promotion blew up: keep the episode window open for retry.
            counters["promotion"] = {
                "anchor": since.isoformat() if since is not None else None,
            }
        if pass_errors:
            counters["pass_errors"] = dict(pass_errors)
        counters.update(_rollup_counters(counters))

        run.status = "succeeded"
        run.finished_at = datetime.now(UTC)
        run.counters_json = counters
        self.session.flush()
        self.session.refresh(task)
        run.cost_usd = task.total_cost_usd

        task.result_summary = (
            f"Consolidation run {run.id} finished: "
            f"promoted={counters.get('promoted', 0)} "
            f"merged={counters.get('merged', 0)} "
            f"invalidated={counters.get('invalidated', 0)} "
            f"archived={counters.get('archived', 0)}"
        )
        task_service.transition(task, TaskStatus.succeeded)
        self.session.commit()
        return ConsolidationOutcome(
            run_id=run.id,
            installation_id=installation_id,
            status=run.status,
            counters=counters,
            task_id=task.id,
        )

    def _create_run_task(
        self,
        *,
        task_service: TaskService,
        installation_id: uuid.UUID,
        run_id: uuid.UUID,
    ) -> Task:
        task_input = (
            "Run Kortny's memory consolidation pass (episode promotion, "
            "candidate adjudication, duplicate merge, aging, fact "
            "reconciliation, hygiene, channel style cards, embedding backfill)."
        )
        return task_service.create_task(
            installation_id=installation_id,
            slack_event_id=f"consolidator:{run_id}",
            slack_channel_id="consolidator",
            slack_user_id="consolidator",
            input=task_input,
            identity=TaskIdentity.synthetic(
                source=CONSOLIDATION_TASK_SOURCE,
                source_id=str(run_id),
                input_text=task_input,
            ),
            source_surface=CONSOLIDATION_TASK_SOURCE,
        )

    def _build_org_profile_pass(self, *, llm: LLMService | None) -> OrgProfilePass:
        """Build the org-profile pass with a Slack poster + DM resolver.

        When Slack credentials are absent (some test/headless contexts), the
        poster/resolver are None and the pass skips gracefully — same pattern as
        the LLM-gated passes.
        """

        poster: ConfirmationPoster | None = None
        dm_resolver: Callable[[str], str | None] | None = None
        token = self.settings.slack_bot_token if self.settings is not None else None
        if token:
            client = WebClient(token=token)
            poster = SlackPoster(
                session=self.session,
                client=cast(SlackPostingClient, client),
                task_service=TaskService(self.session),
            )

            def dm_resolver(user_id: str, _client: WebClient = client) -> str | None:
                return _open_dm_channel(_client, user_id)

        return OrgProfilePass(
            self.session,
            llm=llm,
            poster=poster,
            dm_channel_for_user=dm_resolver,
        )

    def _llm_service(
        self,
        *,
        task: Task,
        task_service: TaskService,
        pass_errors: dict[str, str],
    ) -> LLMService | None:
        try:
            if self.llm_provider is not None:
                model_route = ModelRoute(
                    tier=ModelRouteTier.cheap_fast,
                    model=self.llm_provider.model,
                    reason="consolidator_run",
                )
                provider: LLMProvider = self.llm_provider
                provider_name = self.provider_name or DbLLMProvider.openrouter
            else:
                if self.settings is None:
                    raise RuntimeError("Settings are required to build an LLM service")
                model_route = ModelRouter(self.settings).route_for_tier(
                    ModelRouteTier.cheap_fast,
                    reason="consolidator_run",
                )
                selection = select_runtime_model(
                    session=self.session,
                    settings=self.settings,
                    installation_id=task.installation_id,
                    model_route=model_route,
                )
                model_route = selection.model_route
                provider = create_provider_for_selection(
                    settings=self.settings,
                    selection=selection,
                )
                provider_name = selection.provider_name
            return LLMService(
                session=self.session,
                provider=provider,
                provider_name=provider_name,
                task_service=task_service,
                model_route=model_route,
            )
        except Exception as exc:
            logger.warning(
                "consolidator LLM unavailable; LLM passes will be skipped",
                exc_info=True,
            )
            pass_errors["llm"] = f"{type(exc).__name__}: {exc}"
            return None


def _rollup_counters(counters: dict[str, object]) -> dict[str, object]:
    """Flatten the headline counters the dashboard and design doc name."""

    def _get(pass_name: str, key: str) -> int:
        section = counters.get(pass_name)
        if isinstance(section, dict):
            value = section.get(key)
            if isinstance(value, int):
                return value
        return 0

    def _conflicts() -> list[object]:
        section = counters.get("promotion")
        if isinstance(section, dict):
            value = section.get("conflicts")
            if isinstance(value, list):
                return value
        return []

    return {
        "promoted": _get("promotion", "promoted"),
        "updated": _get("promotion", "updated"),
        "invalidated": _get("promotion", "invalidated"),
        "merged": _get("merge", "merged"),
        "archived": _get("adjudication", "archived") + _get("aging", "archived"),
        "purged_observations": _get("hygiene", "purged_observations"),
        "profiles_refreshed": _get("hygiene", "profiles_refreshed"),
        "style_cards_derived": _get("style_cards", "derived"),
        "embedded": _get("backfill", "embedded"),
        "conflicts": _conflicts(),
    }


def _open_dm_channel(client: WebClient, user_id: str) -> str | None:
    """Resolve a user's DM ("D…") channel id via conversations.open, or None."""

    try:
        response = client.conversations_open(users=user_id)
    except Exception as exc:  # noqa: BLE001 — Slack errors must not break the run
        logger.info(
            "org profile dm open failed user=%s error_type=%s error=%s",
            user_id,
            type(exc).__name__,
            exc,
        )
        return None
    channel = response.get("channel") if isinstance(response, Mapping) else None
    channel_id = channel.get("id") if isinstance(channel, Mapping) else None
    return channel_id if isinstance(channel_id, str) and channel_id else None


__all__ = [
    "CONSOLIDATOR_EXTRACTOR",
    "ConsolidationOutcome",
    "ConsolidationService",
]
