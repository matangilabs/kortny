"""Low-risk extraction paths into the workspace knowledge graph."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session
from sqlalchemy.sql.elements import ColumnElement

from kortny.db.models import (
    KnowledgeGraphEdge,
    KnowledgeGraphEntity,
    ObservationEvent,
    ObserveChannelProfile,
    SlackChannelMembership,
    SlackIdentity,
    Task,
)
from kortny.knowledge_graph.provenance import with_provenance_attrs
from kortny.knowledge_graph.scopes import VisibilityScope
from kortny.knowledge_graph.service import EvidenceInput, GraphService

CHANNEL_ENTITY_TYPE = "channel"
CHANNEL_PROFILE_ENTITY_TYPE = "firm_fact"
CHANNEL_PROFILE_RELATIONSHIP = "relates_to"
KG_CHANNEL_PROFILE_PROJECTED_MESSAGE = "kg_channel_profile_projected"
DETERMINISTIC_PROJECTION_KIND = "slack_deterministic_projection"
DETERMINISTIC_OBSERVATION_LIMIT = 500
SEMANTIC_PROJECTION_KIND = "channel_semantic_projection"
SEMANTIC_PROJECTION_PREFIXES = (
    "channel_topic",
    "channel_workflow",
    "channel_entity",
    "channel_assumption",
    "channel_help",
)
SEMANTIC_ENTITY_SPECS = (
    ("recurring_topics", "channel_topic", "firm_fact", "topic"),
    ("workflows", "channel_workflow", "commitment", "workflow"),
    ("important_entities", "channel_entity", "external_entity", "important_entity"),
    ("assumptions", "channel_assumption", "firm_fact", "assumption"),
    ("help_opportunities", "channel_help", "firm_fact", "help_opportunity"),
)
AUTO_REVIEW_STATUS = "auto"
NEEDS_REVIEW_STATUS = "needs_review"
LOW_CONFIDENCE_THRESHOLD = Decimal("0.500")
SENSITIVE_REVIEW_RE = re.compile(
    r"\b("
    r"api[-_ ]?key|credential|password|secret|token|"
    r"salary|compensation|payroll|hr|human resources|"
    r"medical|health|diagnosis|legal|lawsuit|attorney|"
    r"fired|termination|underperforming|disciplinary|"
    r"confidential|private"
    r")\b",
    re.I,
)


@dataclass(frozen=True)
class KnowledgeGraphProjectionResult:
    channel_entity_id: str
    profile_entity_id: str | None
    profile_edge_id: str | None
    entity_count: int
    edge_count: int
    evidence_count: int


@dataclass(frozen=True)
class KnowledgeGraphDeterministicProjectionResult:
    channel_count: int
    person_count: int
    artifact_count: int
    membership_edge_count: int
    artifact_edge_count: int
    evidence_count: int

    @property
    def entity_count(self) -> int:
        return self.channel_count + self.person_count + self.artifact_count

    @property
    def edge_count(self) -> int:
        return self.membership_edge_count + self.artifact_edge_count


@dataclass(frozen=True)
class SemanticReviewDecision:
    lifecycle_state: str
    review_status: str
    review_reason: str | None


class KnowledgeGraphExtractionService:
    """Project bounded system observations into graph rows.

    Normal scope-safe observations become auto-active context with evidence and
    confidence. Low-confidence or sensitive claims remain candidates for review.
    """

    def __init__(self, session: Session) -> None:
        self.session = session
        self.graph = GraphService(session)

    def project_deterministic_workspace_facts(
        self,
        *,
        installation_id: uuid.UUID | None = None,
        observation_limit: int = DETERMINISTIC_OBSERVATION_LIMIT,
    ) -> KnowledgeGraphDeterministicProjectionResult:
        """Project trusted Slack DB facts into graph rows without LLM inference."""

        memberships = self._active_memberships(installation_id=installation_id)
        if not memberships:
            return KnowledgeGraphDeterministicProjectionResult(
                channel_count=0,
                person_count=0,
                artifact_count=0,
                membership_edge_count=0,
                artifact_edge_count=0,
                evidence_count=0,
            )

        membership_by_key = {
            (membership.installation_id, membership.channel_id): membership
            for membership in memberships
        }
        observations = self._recent_observations(
            installation_id=installation_id,
            channel_ids={membership.channel_id for membership in memberships},
            limit=observation_limit,
        )
        added_by_user_ids: set[str | None] = {
            user_id
            for membership in memberships
            for user_id in (membership.added_by_user_id,)
            if user_id
        }
        observed_user_ids: set[str | None] = {
            observation.user_id for observation in observations if observation.user_id
        }
        identity_map = self._identity_map(
            installation_ids={membership.installation_id for membership in memberships}
            | {observation.installation_id for observation in observations},
            user_ids=added_by_user_ids | observed_user_ids,
        )

        channel_entities: dict[tuple[uuid.UUID, str], KnowledgeGraphEntity] = {}
        channel_count = 0
        person_count = 0
        artifact_count = 0
        membership_edge_count = 0
        artifact_edge_count = 0
        evidence_count = 0

        for membership in memberships:
            channel_entity, _created = self._upsert_deterministic_channel_entity(
                membership=membership,
            )
            channel_entities[(membership.installation_id, membership.channel_id)] = (
                channel_entity
            )
            channel_count += 1
            evidence_count += 1

            if membership.added_by_user_id:
                identity = identity_map.get(
                    (membership.installation_id, membership.added_by_user_id)
                )
                if not _skip_identity(identity):
                    self._upsert_deterministic_person_entity(
                        membership=membership,
                        user_id=membership.added_by_user_id,
                        identity=identity,
                        observation=None,
                        source_reason="app_added_by_user",
                    )
                    person_count += 1
                    evidence_count += 1

        seen_user_pairs: set[tuple[uuid.UUID, str, str]] = set()
        seen_file_pairs: set[tuple[uuid.UUID, str, str]] = set()
        for observation in observations:
            obs_membership = membership_by_key.get(
                (observation.installation_id, observation.channel_id)
            )
            obs_channel_entity = channel_entities.get(
                (observation.installation_id, observation.channel_id)
            )
            if obs_membership is None or obs_channel_entity is None:
                continue

            if observation.user_id:
                user_pair = (
                    observation.installation_id,
                    observation.channel_id,
                    observation.user_id,
                )
                if user_pair not in seen_user_pairs:
                    seen_user_pairs.add(user_pair)
                    identity = identity_map.get(
                        (observation.installation_id, observation.user_id)
                    )
                    if not _skip_identity(identity):
                        person_entity, _created = (
                            self._upsert_deterministic_person_entity(
                                membership=obs_membership,
                                user_id=observation.user_id,
                                identity=identity,
                                observation=observation,
                                source_reason="observed_channel_participation",
                            )
                        )
                        person_count += 1
                        evidence_count += 1
                        self._upsert_deterministic_membership_edge(
                            membership=obs_membership,
                            person_entity=person_entity,
                            channel_entity=obs_channel_entity,
                            observation=observation,
                        )
                        membership_edge_count += 1
                        evidence_count += 1

            if observation.file_id:
                file_pair = (
                    observation.installation_id,
                    observation.channel_id,
                    observation.file_id,
                )
                if file_pair not in seen_file_pairs:
                    seen_file_pairs.add(file_pair)
                    artifact_entity, _created = (
                        self._upsert_deterministic_file_artifact(
                            membership=obs_membership,
                            observation=observation,
                        )
                    )
                    artifact_count += 1
                    evidence_count += 1
                    self._upsert_deterministic_file_edge(
                        membership=obs_membership,
                        artifact_entity=artifact_entity,
                        channel_entity=obs_channel_entity,
                        observation=observation,
                    )
                    artifact_edge_count += 1
                    evidence_count += 1

        self.session.flush()
        return KnowledgeGraphDeterministicProjectionResult(
            channel_count=channel_count,
            person_count=person_count,
            artifact_count=artifact_count,
            membership_edge_count=membership_edge_count,
            artifact_edge_count=artifact_edge_count,
            evidence_count=evidence_count,
        )

    def project_channel_profile(
        self,
        *,
        task: Task,
        membership: SlackChannelMembership,
        profile: ObserveChannelProfile,
    ) -> KnowledgeGraphProjectionResult:
        """Project a completed Observe channel profile into graph context."""

        entity_count = 0
        edge_count = 0
        evidence_count = 0
        channel_entity, created_channel = self._upsert_channel_entity(
            task=task,
            membership=membership,
        )
        entity_count += int(created_channel)
        evidence_count += 1

        profile_entity, profile_created = self._upsert_channel_profile_entity(
            task=task,
            membership=membership,
            profile=profile,
        )
        entity_count += int(profile_created)
        evidence_count += 1

        profile_edge, profile_edge_created = self._upsert_edge(
            installation_id=task.installation_id,
            source_entity_id=channel_entity.id,
            target_entity_id=profile_entity.id,
            relationship_type=CHANNEL_PROFILE_RELATIONSHIP,
            visibility_scope=_scope_for_membership(membership),
            source_type="onboarding_scan",
            lifecycle_state="active",
            confidence_score=profile.confidence_score or Decimal("0.500"),
            confidence_reason=profile.confidence_reason,
            freshness_window_days=profile.fresh_window_days,
            attrs_json={
                "kind": "channel_profile_projection",
                "review_status": AUTO_REVIEW_STATUS,
                "profile_id": str(profile.id),
                "profile_version": profile.profile_version,
            },
            evidence=_profile_evidence(task, membership, profile),
        )
        edge_count += int(profile_edge_created)
        evidence_count += 1

        semantic_entity_count, semantic_edge_count, semantic_evidence_count = (
            self._project_semantic_extraction(
                task=task,
                membership=membership,
                profile=profile,
                channel_entity=channel_entity,
                profile_entity=profile_entity,
            )
        )
        entity_count += semantic_entity_count
        edge_count += semantic_edge_count
        evidence_count += semantic_evidence_count

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
            existing.attrs_json = with_provenance_attrs(
                attrs,
                source_type="slack_authoritative",
                lifecycle_state="active",
                confidence_score=Decimal("1.000"),
            )
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

    def _recent_observations(
        self,
        *,
        installation_id: uuid.UUID | None,
        channel_ids: set[str],
        limit: int,
    ) -> list[ObservationEvent]:
        if not channel_ids:
            return []
        predicates: list[ColumnElement[bool]] = [
            ObservationEvent.purged_at.is_(None),
            ObservationEvent.channel_id.in_(channel_ids),
        ]
        if installation_id is not None:
            predicates.append(ObservationEvent.installation_id == installation_id)
        return list(
            self.session.scalars(
                select(ObservationEvent)
                .where(*predicates)
                .order_by(ObservationEvent.observed_at.desc(), ObservationEvent.id)
                .limit(limit)
            )
        )

    def _identity_map(
        self,
        *,
        installation_ids: set[uuid.UUID],
        user_ids: set[str | None],
    ) -> dict[tuple[uuid.UUID, str], SlackIdentity]:
        cleaned_user_ids = {user_id for user_id in user_ids if user_id}
        if not installation_ids or not cleaned_user_ids:
            return {}
        rows = self.session.scalars(
            select(SlackIdentity).where(
                SlackIdentity.installation_id.in_(installation_ids),
                SlackIdentity.kind == "user",
                SlackIdentity.slack_id.in_(cleaned_user_ids),
            )
        )
        return {
            (identity.installation_id, identity.slack_id): identity for identity in rows
        }

    def _upsert_deterministic_channel_entity(
        self,
        *,
        membership: SlackChannelMembership,
    ) -> tuple[KnowledgeGraphEntity, bool]:
        attrs = {
            "kind": DETERMINISTIC_PROJECTION_KIND,
            "deterministic_kind": "active_slack_channel",
            "review_status": AUTO_REVIEW_STATUS,
            "channel_id": membership.channel_id,
            "channel_name": membership.channel_name,
            "channel_type": membership.channel_type,
            "membership_status": membership.membership_status,
            "discovered_via": membership.discovered_via,
            "membership_id": str(membership.id),
            "first_seen_at": _isoformat(membership.first_seen_at),
            "last_seen_at": _isoformat(membership.last_seen_at),
            "onboarding_status": membership.onboarding_status,
        }
        return self._upsert_entity(
            installation_id=membership.installation_id,
            entity_type=CHANNEL_ENTITY_TYPE,
            canonical_key=_channel_canonical_key(membership.channel_id),
            display_name=_channel_display_name(membership),
            external_ref_type="slack_channel",
            external_ref_id=membership.channel_id,
            attrs_json=attrs,
            visibility_scope=_scope_for_membership(membership),
            source_type="slack_authoritative",
            lifecycle_state="active",
            confidence_score=Decimal("1.000"),
            confidence_reason="Slack channel membership is authoritative.",
            freshness_window_days=30,
            evidence=_membership_evidence(membership),
        )

    def _upsert_deterministic_person_entity(
        self,
        *,
        membership: SlackChannelMembership,
        user_id: str,
        identity: SlackIdentity | None,
        observation: ObservationEvent | None,
        source_reason: str,
    ) -> tuple[KnowledgeGraphEntity, bool]:
        attrs = {
            "kind": DETERMINISTIC_PROJECTION_KIND,
            "deterministic_kind": "observed_slack_user",
            "review_status": AUTO_REVIEW_STATUS,
            "slack_user_id": user_id,
            "channel_id": membership.channel_id,
            "source_reason": source_reason,
            "identity_display_name": identity.display_name if identity else None,
            "identity_raw_name": identity.raw_name if identity else None,
            "identity_is_bot": identity.is_bot if identity else None,
            "identity_is_deleted": identity.is_deleted if identity else None,
            "last_observed_at": _isoformat(observation.observed_at)
            if observation is not None
            else None,
        }
        confidence = Decimal("0.950") if identity is not None else Decimal("0.850")
        return self._upsert_entity(
            installation_id=membership.installation_id,
            entity_type="person",
            canonical_key=_channel_user_canonical_key(membership.channel_id, user_id),
            display_name=_user_display_name(user_id=user_id, identity=identity),
            external_ref_type="slack_user",
            external_ref_id=user_id,
            attrs_json=attrs,
            visibility_scope=_scope_for_membership(membership),
            source_type="slack_authoritative",
            lifecycle_state="active",
            confidence_score=confidence,
            confidence_reason="Observed Slack user participation in a scoped channel.",
            freshness_window_days=30,
            evidence=_person_evidence(membership, user_id, identity, observation),
        )

    def _upsert_deterministic_file_artifact(
        self,
        *,
        membership: SlackChannelMembership,
        observation: ObservationEvent,
    ) -> tuple[KnowledgeGraphEntity, bool]:
        assert observation.file_id is not None
        attrs = {
            "kind": DETERMINISTIC_PROJECTION_KIND,
            "deterministic_kind": "observed_slack_file",
            "review_status": AUTO_REVIEW_STATUS,
            "slack_file_id": observation.file_id,
            "channel_id": membership.channel_id,
            "message_ts": observation.message_ts,
            "thread_ts": observation.thread_ts,
            "text_preview": observation.text_preview,
            "observed_at": _isoformat(observation.observed_at),
            "visibility_metadata": observation.visibility_metadata,
        }
        return self._upsert_entity(
            installation_id=membership.installation_id,
            entity_type="artifact",
            canonical_key=_channel_file_canonical_key(
                membership.channel_id,
                observation.file_id,
            ),
            display_name=_file_display_name(observation.file_id),
            external_ref_type="slack_file",
            external_ref_id=observation.file_id,
            attrs_json=attrs,
            visibility_scope=_scope_for_membership(membership),
            source_type="slack_authoritative",
            lifecycle_state="active",
            confidence_score=Decimal("0.950"),
            confidence_reason="Observed Slack file share in a scoped channel.",
            freshness_window_days=30,
            evidence=_observation_evidence(
                observation,
                raw_snippet=f"Slack file {observation.file_id} was shared in {membership.channel_id}.",
            ),
        )

    def _upsert_deterministic_membership_edge(
        self,
        *,
        membership: SlackChannelMembership,
        person_entity: KnowledgeGraphEntity,
        channel_entity: KnowledgeGraphEntity,
        observation: ObservationEvent,
    ) -> tuple[KnowledgeGraphEdge, bool]:
        return self._upsert_edge(
            installation_id=membership.installation_id,
            source_entity_id=person_entity.id,
            target_entity_id=channel_entity.id,
            relationship_type="member_of",
            visibility_scope=_scope_for_membership(membership),
            source_type="slack_authoritative",
            lifecycle_state="active",
            confidence_score=Decimal("0.850"),
            confidence_reason="User was observed posting or joining in this channel.",
            freshness_window_days=30,
            attrs_json={
                "kind": DETERMINISTIC_PROJECTION_KIND,
                "deterministic_kind": "observed_channel_participation",
                "review_status": AUTO_REVIEW_STATUS,
                "channel_id": membership.channel_id,
                "slack_user_id": observation.user_id,
                "event_type": observation.event_type,
                "message_ts": observation.message_ts,
            },
            evidence=_observation_evidence(
                observation,
                raw_snippet=(
                    observation.text_preview
                    or f"Slack user {observation.user_id} was observed in {membership.channel_id}."
                ),
            ),
        )

    def _upsert_deterministic_file_edge(
        self,
        *,
        membership: SlackChannelMembership,
        artifact_entity: KnowledgeGraphEntity,
        channel_entity: KnowledgeGraphEntity,
        observation: ObservationEvent,
    ) -> tuple[KnowledgeGraphEdge, bool]:
        return self._upsert_edge(
            installation_id=membership.installation_id,
            source_entity_id=artifact_entity.id,
            target_entity_id=channel_entity.id,
            relationship_type="referenced_in",
            visibility_scope=_scope_for_membership(membership),
            source_type="slack_authoritative",
            lifecycle_state="active",
            confidence_score=Decimal("0.950"),
            confidence_reason="Slack file share was observed in this channel.",
            freshness_window_days=30,
            attrs_json={
                "kind": DETERMINISTIC_PROJECTION_KIND,
                "deterministic_kind": "file_referenced_in_channel",
                "review_status": AUTO_REVIEW_STATUS,
                "channel_id": membership.channel_id,
                "slack_file_id": observation.file_id,
                "message_ts": observation.message_ts,
            },
            evidence=_observation_evidence(
                observation,
                raw_snippet=(
                    observation.text_preview
                    or f"Slack file {observation.file_id} was shared in {membership.channel_id}."
                ),
            ),
        )

    def _upsert_channel_profile_entity(
        self,
        *,
        task: Task,
        membership: SlackChannelMembership,
        profile: ObserveChannelProfile,
    ) -> tuple[KnowledgeGraphEntity, bool]:
        canonical_key = _profile_canonical_key(membership.channel_id)
        existing = self._current_entity_by_key(task.installation_id, canonical_key)
        if existing is not None:
            self._reinforce_entity(
                existing,
                display_name=f"Channel profile for {_channel_display_name(membership)}",
                external_ref_type="observe_channel_profile",
                external_ref_id=str(profile.id),
                attrs_json=_profile_attrs(profile),
                visibility_scope=_scope_for_membership(membership),
                source_type="onboarding_scan",
                lifecycle_state="active",
                confidence_score=profile.confidence_score or Decimal("0.500"),
                confidence_reason=profile.confidence_reason,
                freshness_window_days=profile.fresh_window_days,
                evidence=_profile_evidence(task, membership, profile),
            )
            self.session.flush()
            return existing, False

        entity = self.graph.create_entity(
            installation_id=task.installation_id,
            entity_type=CHANNEL_PROFILE_ENTITY_TYPE,
            canonical_key=canonical_key,
            display_name=f"Channel profile for {_channel_display_name(membership)}",
            external_ref_type="observe_channel_profile",
            external_ref_id=str(profile.id),
            attrs_json=_profile_attrs(profile),
            visibility_scope=_scope_for_membership(membership),
            source_type="onboarding_scan",
            lifecycle_state="active",
            confidence_score=profile.confidence_score or Decimal("0.500"),
            confidence_reason=profile.confidence_reason,
            freshness_window_days=profile.fresh_window_days,
            evidence=_profile_evidence(task, membership, profile),
        )
        return entity, True

    def _project_semantic_extraction(
        self,
        *,
        task: Task,
        membership: SlackChannelMembership,
        profile: ObserveChannelProfile,
        channel_entity: KnowledgeGraphEntity,
        profile_entity: KnowledgeGraphEntity,
    ) -> tuple[int, int, int]:
        extraction = _semantic_extraction(profile)
        if extraction is None:
            return 0, 0, 0

        entity_count = 0
        edge_count = 0
        evidence_count = 0
        scope = _scope_for_membership(membership)
        confidence_score = _semantic_confidence_score(
            extraction.get("confidence"),
            profile.confidence_score,
        )

        for field_name, key_prefix, entity_type, semantic_kind in SEMANTIC_ENTITY_SPECS:
            values = _semantic_values(extraction.get(field_name))
            for value in values:
                review = _semantic_review_decision(
                    value=value,
                    confidence_score=confidence_score,
                )
                canonical_key = _semantic_canonical_key(
                    key_prefix=key_prefix,
                    channel_id=membership.channel_id,
                    label=value,
                )
                semantic_entity, created_entity = self._upsert_entity(
                    installation_id=task.installation_id,
                    entity_type=entity_type,
                    canonical_key=canonical_key,
                    display_name=value,
                    attrs_json={
                        "kind": SEMANTIC_PROJECTION_KIND,
                        "semantic_kind": semantic_kind,
                        "source_field": field_name,
                        "profile_id": str(profile.id),
                        "profile_version": profile.profile_version,
                        "channel_id": membership.channel_id,
                        "confidence": extraction.get("confidence"),
                        "review_status": review.review_status,
                        "review_reason": review.review_reason,
                        "evidence": _semantic_values(extraction.get("evidence")),
                    },
                    visibility_scope=scope,
                    source_type="onboarding_scan",
                    lifecycle_state=review.lifecycle_state,
                    confidence_score=confidence_score,
                    confidence_reason=_semantic_confidence_reason(profile),
                    freshness_window_days=profile.fresh_window_days,
                    evidence=_semantic_evidence(
                        task=task,
                        membership=membership,
                        profile=profile,
                        value=value,
                        semantic_kind=semantic_kind,
                    ),
                )
                entity_count += int(created_entity)
                evidence_count += 1

                channel_edge, channel_edge_created = self._upsert_edge(
                    installation_id=task.installation_id,
                    source_entity_id=channel_entity.id,
                    target_entity_id=semantic_entity.id,
                    relationship_type="relates_to",
                    visibility_scope=scope,
                    source_type="onboarding_scan",
                    lifecycle_state=review.lifecycle_state,
                    confidence_score=confidence_score,
                    confidence_reason=_semantic_confidence_reason(profile),
                    freshness_window_days=profile.fresh_window_days,
                    attrs_json={
                        "kind": SEMANTIC_PROJECTION_KIND,
                        "semantic_kind": semantic_kind,
                        "source_field": field_name,
                        "profile_id": str(profile.id),
                        "review_status": review.review_status,
                        "review_reason": review.review_reason,
                    },
                    evidence=_semantic_evidence(
                        task=task,
                        membership=membership,
                        profile=profile,
                        value=value,
                        semantic_kind=semantic_kind,
                    ),
                )
                profile_edge, profile_edge_created = self._upsert_edge(
                    installation_id=task.installation_id,
                    source_entity_id=profile_entity.id,
                    target_entity_id=semantic_entity.id,
                    relationship_type="maps_to",
                    visibility_scope=scope,
                    source_type="onboarding_scan",
                    lifecycle_state=review.lifecycle_state,
                    confidence_score=confidence_score,
                    confidence_reason=_semantic_confidence_reason(profile),
                    freshness_window_days=profile.fresh_window_days,
                    attrs_json={
                        "kind": SEMANTIC_PROJECTION_KIND,
                        "semantic_kind": semantic_kind,
                        "source_field": field_name,
                        "profile_id": str(profile.id),
                        "review_status": review.review_status,
                        "review_reason": review.review_reason,
                    },
                    evidence=_semantic_evidence(
                        task=task,
                        membership=membership,
                        profile=profile,
                        value=value,
                        semantic_kind=semantic_kind,
                    ),
                )
                edge_count += int(channel_edge_created) + int(profile_edge_created)
                evidence_count += 2

        return entity_count, edge_count, evidence_count

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

    def _upsert_entity(
        self,
        *,
        installation_id: uuid.UUID,
        entity_type: str,
        canonical_key: str,
        display_name: str,
        attrs_json: dict[str, Any],
        visibility_scope: VisibilityScope,
        source_type: str,
        lifecycle_state: str,
        confidence_score: Decimal,
        confidence_reason: str,
        freshness_window_days: int | None,
        evidence: EvidenceInput,
        external_ref_type: str | None = None,
        external_ref_id: str | None = None,
    ) -> tuple[KnowledgeGraphEntity, bool]:
        existing = self._current_entity_by_key(installation_id, canonical_key)
        if existing is not None:
            self._reinforce_entity(
                existing,
                display_name=display_name,
                external_ref_type=external_ref_type or existing.external_ref_type,
                external_ref_id=external_ref_id or existing.external_ref_id,
                attrs_json=attrs_json,
                visibility_scope=visibility_scope,
                source_type=source_type,
                lifecycle_state=lifecycle_state,
                confidence_score=confidence_score,
                confidence_reason=confidence_reason,
                freshness_window_days=freshness_window_days,
                evidence=evidence,
            )
            return existing, False
        entity = self.graph.create_entity(
            installation_id=installation_id,
            entity_type=entity_type,
            canonical_key=canonical_key,
            display_name=display_name,
            external_ref_type=external_ref_type,
            external_ref_id=external_ref_id,
            attrs_json=attrs_json,
            visibility_scope=visibility_scope,
            source_type=source_type,
            lifecycle_state=lifecycle_state,
            confidence_score=confidence_score,
            confidence_reason=confidence_reason,
            freshness_window_days=freshness_window_days,
            evidence=evidence,
        )
        return entity, True

    def _upsert_edge(
        self,
        *,
        installation_id: uuid.UUID,
        source_entity_id: uuid.UUID,
        target_entity_id: uuid.UUID,
        relationship_type: str,
        visibility_scope: VisibilityScope,
        source_type: str,
        attrs_json: dict[str, Any],
        lifecycle_state: str,
        confidence_score: Decimal,
        confidence_reason: str | None,
        freshness_window_days: int | None,
        evidence: EvidenceInput,
    ) -> tuple[KnowledgeGraphEdge, bool]:
        existing = self._current_edge(
            installation_id=installation_id,
            source_entity_id=source_entity_id,
            target_entity_id=target_entity_id,
            relationship_type=relationship_type,
            source_type=source_type,
        )
        if existing is not None:
            self._reinforce_edge(
                existing,
                attrs_json=attrs_json,
                visibility_scope=visibility_scope,
                source_type=source_type,
                lifecycle_state=lifecycle_state,
                confidence_score=confidence_score,
                confidence_reason=confidence_reason,
                freshness_window_days=freshness_window_days,
                evidence=evidence,
            )
            return existing, False
        edge = self.graph.create_edge(
            installation_id=installation_id,
            source_entity_id=source_entity_id,
            target_entity_id=target_entity_id,
            relationship_type=relationship_type,
            visibility_scope=visibility_scope,
            source_type=source_type,
            attrs_json=attrs_json,
            lifecycle_state=lifecycle_state,
            confidence_score=confidence_score,
            confidence_reason=confidence_reason,
            freshness_window_days=freshness_window_days,
            evidence=evidence,
        )
        return edge, True

    def _current_edge(
        self,
        *,
        installation_id: object,
        source_entity_id: uuid.UUID,
        target_entity_id: uuid.UUID,
        relationship_type: str,
        source_type: str,
    ) -> KnowledgeGraphEdge | None:
        return self.session.scalar(
            select(KnowledgeGraphEdge).where(
                KnowledgeGraphEdge.installation_id == installation_id,
                KnowledgeGraphEdge.source_entity_id == source_entity_id,
                KnowledgeGraphEdge.target_entity_id == target_entity_id,
                KnowledgeGraphEdge.relationship_type == relationship_type,
                KnowledgeGraphEdge.source_type == source_type,
                KnowledgeGraphEdge.is_current.is_(True),
                KnowledgeGraphEdge.expired_at.is_(None),
            )
        )

    def _reinforce_entity(
        self,
        entity: KnowledgeGraphEntity,
        *,
        display_name: str,
        external_ref_type: str | None,
        external_ref_id: str | None,
        attrs_json: dict[str, Any],
        visibility_scope: VisibilityScope,
        source_type: str,
        lifecycle_state: str,
        confidence_score: Decimal,
        confidence_reason: str | None,
        freshness_window_days: int | None,
        evidence: EvidenceInput,
    ) -> None:
        now = datetime.now(UTC)
        entity.display_name = display_name
        entity.external_ref_type = external_ref_type
        entity.external_ref_id = external_ref_id
        entity.attrs_json = with_provenance_attrs(
            attrs_json,
            source_type=source_type,
            lifecycle_state=lifecycle_state,
            confidence_score=confidence_score,
        )
        entity.visibility_scope_type = visibility_scope.scope_type
        entity.visibility_scope_id = visibility_scope.scope_id
        entity.source_type = source_type
        entity.lifecycle_state = _reinforced_lifecycle_state(
            current=entity.lifecycle_state,
            proposed=lifecycle_state,
        )
        entity.confidence_score = max(entity.confidence_score, confidence_score)
        entity.confidence_reason = confidence_reason
        entity.freshness_window_days = freshness_window_days
        entity.last_reinforced_at = now
        entity.reinforcement_count = (entity.reinforcement_count or 0) + 1
        entity.is_current = True
        self.graph.add_evidence(
            installation_id=entity.installation_id,
            target_kind="entity",
            target_id=entity.id,
            evidence=evidence,
        )
        self.session.flush()

    def _reinforce_edge(
        self,
        edge: KnowledgeGraphEdge,
        *,
        attrs_json: dict[str, Any],
        visibility_scope: VisibilityScope,
        source_type: str,
        lifecycle_state: str,
        confidence_score: Decimal,
        confidence_reason: str | None,
        freshness_window_days: int | None,
        evidence: EvidenceInput,
    ) -> None:
        now = datetime.now(UTC)
        edge.attrs_json = with_provenance_attrs(
            attrs_json,
            source_type=source_type,
            lifecycle_state=lifecycle_state,
            confidence_score=confidence_score,
        )
        edge.visibility_scope_type = visibility_scope.scope_type
        edge.visibility_scope_id = visibility_scope.scope_id
        edge.source_type = source_type
        edge.lifecycle_state = _reinforced_lifecycle_state(
            current=edge.lifecycle_state,
            proposed=lifecycle_state,
        )
        edge.confidence_score = max(edge.confidence_score, confidence_score)
        edge.confidence_reason = confidence_reason
        edge.freshness_window_days = freshness_window_days
        edge.last_reinforced_at = now
        edge.reinforcement_count = (edge.reinforcement_count or 0) + 1
        edge.is_current = True
        self.graph.add_evidence(
            installation_id=edge.installation_id,
            target_kind="edge",
            target_id=edge.id,
            evidence=evidence,
        )
        self.session.flush()


def _channel_canonical_key(channel_id: str) -> str:
    return f"slack_channel:{channel_id}"


def _profile_canonical_key(channel_id: str) -> str:
    return f"channel_profile:{channel_id}"


def _channel_user_canonical_key(channel_id: str, user_id: str) -> str:
    return f"slack_channel_user:{channel_id}:{user_id}"


def _channel_file_canonical_key(channel_id: str, file_id: str) -> str:
    return f"slack_channel_file:{channel_id}:{file_id}"


def _user_display_name(*, user_id: str, identity: SlackIdentity | None) -> str:
    if identity is not None and identity.display_name:
        return identity.display_name
    return user_id


def _file_display_name(file_id: str) -> str:
    return f"Slack file {file_id}"


def _skip_identity(identity: SlackIdentity | None) -> bool:
    return bool(identity is not None and (identity.is_deleted or identity.is_bot))


def _isoformat(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat()


def _bounded_snippet(value: str | None, fallback: str) -> str:
    snippet = re.sub(r"\s+", " ", value or "").strip()
    if not snippet:
        snippet = fallback
    return snippet[:1000]


def _membership_evidence(membership: SlackChannelMembership) -> EvidenceInput:
    return EvidenceInput(
        source_type="slack_authoritative",
        extracted_by="slack_channel_membership",
        source_slack_channel_id=membership.channel_id,
        source_slack_message_ts=membership.onboarding_message_ts,
        raw_snippet=f"Kortny is active in Slack channel {membership.channel_id}.",
        confidence_score=Decimal("1.000"),
        confidence_reason="Slack channel membership row.",
    )


def _person_evidence(
    membership: SlackChannelMembership,
    user_id: str,
    identity: SlackIdentity | None,
    observation: ObservationEvent | None,
) -> EvidenceInput:
    if observation is not None:
        return _observation_evidence(
            observation,
            raw_snippet=(
                observation.text_preview
                or f"Slack user {user_id} was observed in {membership.channel_id}."
            ),
        )
    display_name = _user_display_name(user_id=user_id, identity=identity)
    return EvidenceInput(
        source_type="slack_authoritative",
        extracted_by="slack_channel_membership",
        source_slack_channel_id=membership.channel_id,
        source_slack_message_ts=membership.onboarding_message_ts,
        raw_snippet=(
            f"{display_name} ({user_id}) is associated with channel "
            f"{membership.channel_id} because they added Kortny."
        ),
        confidence_score=Decimal("0.900"),
        confidence_reason="Slack channel membership added-by user row.",
    )


def _observation_evidence(
    observation: ObservationEvent,
    *,
    raw_snippet: str,
) -> EvidenceInput:
    return EvidenceInput(
        source_type="slack_authoritative",
        extracted_by="slack_observation_event",
        source_observation_id=observation.id,
        source_slack_channel_id=observation.channel_id,
        source_slack_message_ts=observation.message_ts,
        source_slack_file_id=observation.file_id,
        raw_snippet=_bounded_snippet(raw_snippet, "Slack observation event."),
        confidence_score=Decimal("0.900"),
        confidence_reason="Slack observation event recorded by Kortny.",
    )


def _profile_attrs(profile: ObserveChannelProfile) -> dict[str, Any]:
    return {
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
        "review_status": AUTO_REVIEW_STATUS,
    }


def _reinforced_lifecycle_state(*, current: str, proposed: str) -> str:
    if current == "confirmed":
        return "confirmed"
    return proposed


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


def _semantic_extraction(profile: ObserveChannelProfile) -> dict[str, Any] | None:
    candidates = (
        profile.profile_json.get("semantic_extraction")
        if isinstance(profile.profile_json, dict)
        else None
    )
    if not isinstance(candidates, dict):
        candidates = (
            profile.metadata_json.get("semantic_extraction")
            if isinstance(profile.metadata_json, dict)
            else None
        )
    if not isinstance(candidates, dict):
        return None
    if not any(
        _semantic_values(candidates.get(field)) for field, *_ in SEMANTIC_ENTITY_SPECS
    ):
        return None
    return candidates


def _semantic_values(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    output: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            continue
        cleaned = _bounded_semantic_text(item)
        key = cleaned.lower()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        output.append(cleaned)
        if len(output) >= 5:
            break
    return tuple(output)


def _bounded_semantic_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()[:180].strip()


def _semantic_canonical_key(
    *,
    key_prefix: str,
    channel_id: str,
    label: str,
) -> str:
    return f"{key_prefix}:{channel_id}:{_slugify(label)}"


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:72].strip("-") or "unknown"


def _semantic_confidence_score(
    confidence: object,
    fallback: Decimal | None,
) -> Decimal:
    if confidence == "high":
        return Decimal("0.700")
    if confidence == "medium":
        return Decimal("0.600")
    if confidence == "low":
        return Decimal("0.400")
    return fallback or Decimal("0.500")


def _semantic_confidence_reason(profile: ObserveChannelProfile) -> str:
    return (
        "Inferred from bounded channel assessment; usable as scoped background "
        f"context with evidence and confidence. Profile version {profile.profile_version}."
    )


def _semantic_review_decision(
    *,
    value: str,
    confidence_score: Decimal,
) -> SemanticReviewDecision:
    if confidence_score < LOW_CONFIDENCE_THRESHOLD:
        return SemanticReviewDecision(
            lifecycle_state="candidate",
            review_status=NEEDS_REVIEW_STATUS,
            review_reason="low_confidence",
        )
    if SENSITIVE_REVIEW_RE.search(value):
        return SemanticReviewDecision(
            lifecycle_state="candidate",
            review_status=NEEDS_REVIEW_STATUS,
            review_reason="sensitive_or_high_impact_language",
        )
    return SemanticReviewDecision(
        lifecycle_state="active",
        review_status=AUTO_REVIEW_STATUS,
        review_reason=None,
    )


def _semantic_evidence(
    *,
    task: Task,
    membership: SlackChannelMembership,
    profile: ObserveChannelProfile,
    value: str,
    semantic_kind: str,
) -> EvidenceInput:
    return EvidenceInput(
        source_type="onboarding_scan",
        extracted_by="channel_semantic_projection",
        source_task_id=task.id,
        source_slack_channel_id=membership.channel_id,
        source_slack_message_ts=task.slack_message_ts,
        raw_snippet=f"{semantic_kind}: {value}",
        confidence_score=_semantic_confidence_score(
            (profile.profile_json or {})
            .get("semantic_extraction", {})
            .get("confidence")
            if isinstance(profile.profile_json, dict)
            else None,
            profile.confidence_score,
        ),
        confidence_reason=_semantic_confidence_reason(profile),
    )
