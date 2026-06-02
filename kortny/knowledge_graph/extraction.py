"""Low-risk extraction paths into the workspace knowledge graph."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.orm import Session

from kortny.db.models import (
    KnowledgeGraphEdge,
    KnowledgeGraphEntity,
    ObserveChannelProfile,
    SlackChannelMembership,
    Task,
)
from kortny.knowledge_graph.scopes import VisibilityScope
from kortny.knowledge_graph.service import EvidenceInput, GraphService

CHANNEL_ENTITY_TYPE = "channel"
CHANNEL_PROFILE_ENTITY_TYPE = "firm_fact"
CHANNEL_PROFILE_RELATIONSHIP = "relates_to"
KG_CHANNEL_PROFILE_PROJECTED_MESSAGE = "kg_channel_profile_projected"


@dataclass(frozen=True)
class KnowledgeGraphProjectionResult:
    channel_entity_id: str
    profile_entity_id: str | None
    profile_edge_id: str | None
    entity_count: int
    edge_count: int
    evidence_count: int


class KnowledgeGraphExtractionService:
    """Project already-approved system observations into graph rows.

    This service deliberately avoids broad LLM extraction. It only records
    authoritative channel existence and stores channel assessment summaries as
    channel-scoped candidates for later review.
    """

    def __init__(self, session: Session) -> None:
        self.session = session
        self.graph = GraphService(session)

    def project_channel_profile(
        self,
        *,
        task: Task,
        membership: SlackChannelMembership,
        profile: ObserveChannelProfile,
    ) -> KnowledgeGraphProjectionResult:
        """Project a completed Observe channel profile into KG candidates."""

        entity_count = 0
        edge_count = 0
        evidence_count = 0
        channel_entity, created_channel = self._upsert_channel_entity(
            task=task,
            membership=membership,
        )
        entity_count += int(created_channel)
        evidence_count += 1

        profile_entity, profile_created = self._replace_channel_profile_entity(
            task=task,
            membership=membership,
            profile=profile,
        )
        entity_count += int(profile_created)
        evidence_count += 1

        self._supersede_current_profile_edges(channel_entity)
        profile_edge = self.graph.create_edge(
            installation_id=task.installation_id,
            source_entity_id=channel_entity.id,
            target_entity_id=profile_entity.id,
            relationship_type=CHANNEL_PROFILE_RELATIONSHIP,
            visibility_scope=_scope_for_membership(membership),
            source_type="onboarding_scan",
            lifecycle_state="candidate",
            confidence_score=profile.confidence_score or Decimal("0.500"),
            confidence_reason=profile.confidence_reason,
            freshness_window_days=profile.fresh_window_days,
            attrs_json={
                "kind": "channel_profile_projection",
                "profile_id": str(profile.id),
                "profile_version": profile.profile_version,
            },
            evidence=_profile_evidence(task, membership, profile),
        )
        edge_count += 1
        evidence_count += 1

        self.session.flush()
        return KnowledgeGraphProjectionResult(
            channel_entity_id=str(channel_entity.id),
            profile_entity_id=str(profile_entity.id),
            profile_edge_id=str(profile_edge.id),
            entity_count=entity_count,
            edge_count=edge_count,
            evidence_count=evidence_count,
        )

    def _upsert_channel_entity(
        self,
        *,
        task: Task,
        membership: SlackChannelMembership,
    ) -> tuple[KnowledgeGraphEntity, bool]:
        canonical_key = _channel_canonical_key(membership.channel_id)
        existing = self._current_entity_by_key(task.installation_id, canonical_key)
        attrs = {
            "channel_id": membership.channel_id,
            "channel_name": membership.channel_name,
            "channel_type": membership.channel_type,
            "membership_status": membership.membership_status,
            "discovered_via": membership.discovered_via,
            "membership_id": str(membership.id),
        }
        if existing is not None:
            existing.display_name = _channel_display_name(membership)
            existing.attrs_json = attrs
            existing.visibility_scope_type = _scope_for_membership(
                membership
            ).scope_type
            existing.visibility_scope_id = _scope_for_membership(membership).scope_id
            existing.source_type = "slack_authoritative"
            existing.lifecycle_state = "active"
            existing.is_current = True
            self.graph.add_evidence(
                installation_id=task.installation_id,
                target_kind="entity",
                target_id=existing.id,
                evidence=_channel_evidence(task, membership),
            )
            self.session.flush()
            return existing, False

        entity = self.graph.create_entity(
            installation_id=task.installation_id,
            entity_type=CHANNEL_ENTITY_TYPE,
            canonical_key=canonical_key,
            display_name=_channel_display_name(membership),
            external_ref_type="slack_channel",
            external_ref_id=membership.channel_id,
            attrs_json=attrs,
            visibility_scope=_scope_for_membership(membership),
            source_type="slack_authoritative",
            lifecycle_state="active",
            confidence_score=Decimal("1.000"),
            confidence_reason="Slack channel membership is authoritative.",
            evidence=_channel_evidence(task, membership),
        )
        return entity, True

    def _replace_channel_profile_entity(
        self,
        *,
        task: Task,
        membership: SlackChannelMembership,
        profile: ObserveChannelProfile,
    ) -> tuple[KnowledgeGraphEntity, bool]:
        canonical_key = _profile_canonical_key(membership.channel_id)
        existing = self._current_entity_by_key(task.installation_id, canonical_key)
        if existing is not None:
            self.graph.supersede_entity(existing)
            self.session.flush()

        entity = self.graph.create_entity(
            installation_id=task.installation_id,
            entity_type=CHANNEL_PROFILE_ENTITY_TYPE,
            canonical_key=canonical_key,
            display_name=f"Channel profile for {_channel_display_name(membership)}",
            external_ref_type="observe_channel_profile",
            external_ref_id=str(profile.id),
            attrs_json={
                "kind": "observe_channel_profile",
                "profile_id": str(profile.id),
                "profile_version": profile.profile_version,
                "summary": profile.summary,
                "profile": profile.profile_json,
                "assumptions": profile.assumptions_json,
                "evidence_refs": profile.evidence_refs_json,
                "message_count": profile.message_count,
                "file_count": profile.file_count,
                "fresh_window_days": profile.fresh_window_days,
            },
            visibility_scope=_scope_for_membership(membership),
            source_type="onboarding_scan",
            lifecycle_state="candidate",
            confidence_score=profile.confidence_score or Decimal("0.500"),
            confidence_reason=profile.confidence_reason,
            freshness_window_days=profile.fresh_window_days,
            evidence=_profile_evidence(task, membership, profile),
        )
        return entity, True

    def _supersede_current_profile_edges(
        self,
        channel_entity: KnowledgeGraphEntity,
    ) -> None:
        edges = self.session.scalars(
            select(KnowledgeGraphEdge).where(
                KnowledgeGraphEdge.installation_id == channel_entity.installation_id,
                KnowledgeGraphEdge.source_entity_id == channel_entity.id,
                KnowledgeGraphEdge.relationship_type == CHANNEL_PROFILE_RELATIONSHIP,
                KnowledgeGraphEdge.source_type == "onboarding_scan",
                KnowledgeGraphEdge.is_current.is_(True),
                KnowledgeGraphEdge.expired_at.is_(None),
            )
        )
        for edge in edges:
            self.graph.supersede_edge(edge)
        self.session.flush()

    def _current_entity_by_key(
        self,
        installation_id: object,
        canonical_key: str,
    ) -> KnowledgeGraphEntity | None:
        return self.session.scalar(
            select(KnowledgeGraphEntity).where(
                KnowledgeGraphEntity.installation_id == installation_id,
                KnowledgeGraphEntity.canonical_key == canonical_key,
                KnowledgeGraphEntity.is_current.is_(True),
                KnowledgeGraphEntity.expired_at.is_(None),
            )
        )


def _channel_canonical_key(channel_id: str) -> str:
    return f"slack_channel:{channel_id}"


def _profile_canonical_key(channel_id: str) -> str:
    return f"channel_profile:{channel_id}"


def _channel_display_name(membership: SlackChannelMembership) -> str:
    return (
        f"#{membership.channel_name}"
        if membership.channel_name
        else membership.channel_id
    )


def _scope_for_membership(membership: SlackChannelMembership) -> VisibilityScope:
    channel_type = (membership.channel_type or "").lower()
    if channel_type in {"group", "private_channel"} or membership.channel_id.startswith(
        "G"
    ):
        return VisibilityScope.private_channel(membership.channel_id)
    return VisibilityScope.channel(membership.channel_id)


def _channel_evidence(
    task: Task,
    membership: SlackChannelMembership,
) -> EvidenceInput:
    return EvidenceInput(
        source_type="slack_authoritative",
        extracted_by="slack_channel_membership",
        source_task_id=task.id,
        source_slack_channel_id=membership.channel_id,
        source_slack_message_ts=membership.onboarding_message_ts,
        raw_snippet=f"Kortny is active in Slack channel {membership.channel_id}.",
        confidence_score=Decimal("1.000"),
        confidence_reason="Slack channel membership row.",
    )


def _profile_evidence(
    task: Task,
    membership: SlackChannelMembership,
    profile: ObserveChannelProfile,
) -> EvidenceInput:
    return EvidenceInput(
        source_type="onboarding_scan",
        extracted_by="observe_channel_profile",
        source_task_id=task.id,
        source_slack_channel_id=membership.channel_id,
        source_slack_message_ts=task.slack_message_ts,
        raw_snippet=(profile.summary or "")[:1000],
        confidence_score=profile.confidence_score,
        confidence_reason=profile.confidence_reason,
    )
