"""Manual bootstrap and refresh actions for the workspace knowledge graph."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from kortny.db.models import (
    KnowledgeGraphEntity,
    SlackChannelMembership,
    Task,
    TaskEventType,
    TaskStatus,
)
from kortny.observe.assessment import (
    CHANNEL_ASSESSMENT_REQUESTED_MESSAGE,
    CHANNEL_ASSESSMENT_SUPPRESS_SLACK_POST_KEY,
    build_channel_graph_refresh_input,
)
from kortny.slack.membership import SlackChannelMembershipService
from kortny.tasks import TaskService
from kortny.tasks.identity import TaskIdentity

KG_CHANNEL_REFRESH_REQUESTED_MESSAGE = "kg_channel_refresh_requested"
KG_REFRESH_SOURCE = "dashboard_knowledge_graph_refresh"
RECENT_REFRESH_GUARD = timedelta(minutes=15)
ACTIVE_ASSESSMENT_STATUSES = frozenset(
    {
        TaskStatus.pending,
        TaskStatus.running,
        TaskStatus.waiting_approval,
        TaskStatus.crashed,
    }
)


@dataclass(frozen=True, slots=True)
class KnowledgeGraphRefreshResult:
    """Outcome of queuing graph refresh assessment work."""

    known_channel_count: int
    queued_task_ids: tuple[uuid.UUID, ...]
    deterministic_entity_count: int = 0
    deterministic_edge_count: int = 0
    deterministic_evidence_count: int = 0
    skipped_reasons: dict[str, int] = field(default_factory=dict)

    @property
    def queued_count(self) -> int:
        return len(self.queued_task_ids)

    @property
    def skipped_count(self) -> int:
        return sum(self.skipped_reasons.values())


class KnowledgeGraphRefreshService:
    """Queue bounded background channel assessments for graph bootstrapping."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self.task_service = TaskService(session)
        self.membership_service = SlackChannelMembershipService(session)

    def queue_channel_profile_refresh(
        self,
        *,
        installation_id: uuid.UUID | None,
        requested_by_user_id: str,
        now: datetime | None = None,
    ) -> KnowledgeGraphRefreshResult:
        """Queue refresh assessments for known active channel memberships."""

        queued_task_ids: list[uuid.UUID] = []
        skipped_reasons: dict[str, int] = {}
        requested_at = now or datetime.now(UTC)
        memberships = self._active_memberships(installation_id=installation_id)
        from kortny.knowledge_graph.extraction import KnowledgeGraphExtractionService

        deterministic_projection = (
            KnowledgeGraphExtractionService(
                self.session
            ).project_deterministic_workspace_facts(installation_id=installation_id)
        )

        for membership in memberships:
            skip_reason = self._skip_reason(membership, now=requested_at)
            if skip_reason is not None:
                skipped_reasons[skip_reason] = skipped_reasons.get(skip_reason, 0) + 1
                continue
            task = self._queue_membership_refresh(
                membership,
                requested_by_user_id=requested_by_user_id,
                requested_at=requested_at,
            )
            queued_task_ids.append(task.id)

        return KnowledgeGraphRefreshResult(
            known_channel_count=len(memberships),
            queued_task_ids=tuple(queued_task_ids),
            deterministic_entity_count=deterministic_projection.entity_count,
            deterministic_edge_count=deterministic_projection.edge_count,
            deterministic_evidence_count=deterministic_projection.evidence_count,
            skipped_reasons=skipped_reasons,
        )

    def _active_memberships(
        self,
        *,
        installation_id: uuid.UUID | None,
    ) -> list[SlackChannelMembership]:
        predicates = [SlackChannelMembership.membership_status == "active"]
        if installation_id is not None:
            predicates.append(SlackChannelMembership.installation_id == installation_id)
        return list(
            self.session.scalars(
                select(SlackChannelMembership)
                .where(*predicates)
                .order_by(
                    SlackChannelMembership.last_seen_at.desc(),
                    SlackChannelMembership.channel_id,
                )
            )
        )

    def _skip_reason(
        self,
        membership: SlackChannelMembership,
        *,
        now: datetime,
    ) -> str | None:
        current_task = _metadata_task(self.session, membership.metadata_json)
        if (
            current_task is not None
            and TaskStatus(current_task.status) in ACTIVE_ASSESSMENT_STATUSES
        ):
            return "assessment_already_active"

        has_projection = self._has_channel_profile_projection(membership)
        if has_projection and _metadata_recent(membership.metadata_json, now=now):
            return "recently_refreshed"

        return None

    def _queue_membership_refresh(
        self,
        membership: SlackChannelMembership,
        *,
        requested_by_user_id: str,
        requested_at: datetime,
    ) -> Task:
        run_id = uuid.uuid4().hex[:12]
        task_input = build_channel_graph_refresh_input(channel_id=membership.channel_id)
        task = self.task_service.create_task(
            installation_id=membership.installation_id,
            slack_event_id=f"dashboard:{membership.id}:kg_refresh:{run_id}",
            slack_channel_id=membership.channel_id,
            slack_thread_ts=membership.onboarding_message_ts,
            slack_message_ts=membership.onboarding_message_ts,
            slack_user_id=requested_by_user_id,
            input=task_input,
            identity=TaskIdentity.synthetic(
                source=KG_REFRESH_SOURCE,
                source_id=f"{membership.id}:{run_id}",
                input_text=task_input,
                payload={
                    "channel_id": membership.channel_id,
                    "membership_id": str(membership.id),
                    "requested_by_user_id": requested_by_user_id,
                },
            ),
            source_surface=KG_REFRESH_SOURCE,
        )
        self.task_service.append_event(
            task,
            TaskEventType.log,
            {
                "message": CHANNEL_ASSESSMENT_REQUESTED_MESSAGE,
                "source": KG_REFRESH_SOURCE,
                "channel_id": membership.channel_id,
                "membership_id": str(membership.id),
                "requested_by_user_id": requested_by_user_id,
                "requested_at": requested_at.isoformat(),
                CHANNEL_ASSESSMENT_SUPPRESS_SLACK_POST_KEY: True,
            },
        )
        self.task_service.append_event(
            task,
            TaskEventType.log,
            {
                "message": KG_CHANNEL_REFRESH_REQUESTED_MESSAGE,
                "channel_id": membership.channel_id,
                "membership_id": str(membership.id),
                "requested_by_user_id": requested_by_user_id,
            },
        )
        self.membership_service.mark_assessment_queued(
            membership=membership,
            task_id=task.id,
        )
        metadata = dict(membership.metadata_json or {})
        metadata["assessment_source"] = KG_REFRESH_SOURCE
        metadata["kg_refresh_requested_by"] = requested_by_user_id
        metadata["kg_refresh_requested_at"] = requested_at.isoformat()
        membership.metadata_json = metadata
        self.session.flush()
        return task

    def _has_channel_profile_projection(
        self,
        membership: SlackChannelMembership,
    ) -> bool:
        return bool(
            self.session.scalar(
                select(KnowledgeGraphEntity.id)
                .where(
                    KnowledgeGraphEntity.installation_id == membership.installation_id,
                    KnowledgeGraphEntity.canonical_key
                    == f"channel_profile:{membership.channel_id}",
                    KnowledgeGraphEntity.is_current.is_(True),
                    KnowledgeGraphEntity.expired_at.is_(None),
                )
                .limit(1)
            )
        )


def _metadata_task(session: Session, metadata: dict | None) -> Task | None:
    task_id = (metadata or {}).get("assessment_task_id")
    if not isinstance(task_id, str) or not task_id:
        return None
    try:
        parsed_task_id = uuid.UUID(task_id)
    except ValueError:
        return None
    return session.get(Task, parsed_task_id)


def _metadata_recent(metadata: dict | None, *, now: datetime) -> bool:
    value = (metadata or {}).get("kg_refresh_requested_at")
    if not isinstance(value, str) or not value:
        return False
    try:
        requested_at = datetime.fromisoformat(value)
    except ValueError:
        return False
    if requested_at.tzinfo is None:
        requested_at = requested_at.replace(tzinfo=UTC)
    return now - requested_at < RECENT_REFRESH_GUARD
