"""Deterministic consolidation passes 2-7 (HIG-225).

Pass order and intent (see the HIG-225 design doc):

2. Candidate adjudication — promote multi-evidence candidates, archive
   single-evidence ones that never gathered consensus.
3. Duplicate entity merge — embedding-similarity pairs confirmed by the cheap
   LLM tier; the older row wins, the newer one is superseded.
4. Aging — finally wire ``mark_stale_current``; archive long-stale rows.
5. Fact reconciliation — user-confirmed workspace facts project into the graph
   as ``user_confirmed`` entities (single store, multiple views).
6. Hygiene — observation retention purge, fact TTL purge, profile refresh.
7. Embedding backfill — sha-gated ensure over facts/episodes/entities.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from sqlalchemy import delete, func, select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.orm import Session

from kortny.db.models import (
    Episode,
    KnowledgeGraphEdge,
    KnowledgeGraphEntity,
    KnowledgeGraphEvidence,
    ObservationEvent,
    ObserveChannelProfile,
    ObservePolicy,
    SlackChannelMembership,
    Task,
    TaskEventType,
    TaskStatus,
    WitnessOpportunityCandidate,
    WorkspaceState,
)
from kortny.embeddings import (
    EPISODE_EMBEDDING_KIND,
    FACT_EMBEDDING_KIND,
    KG_ENTITY_EMBEDDING_KIND,
    EmbeddingIndex,
    episode_embedding_text,
    fact_embedding_text,
    kg_entity_embedding_text,
)
from kortny.knowledge_graph import EvidenceInput, GraphService, VisibilityScope
from kortny.llm import ChatMessage, LLMService
from kortny.observe.assessment import (
    CHANNEL_ASSESSMENT_REQUESTED_MESSAGE,
    CHANNEL_ASSESSMENT_SUPPRESS_SLACK_POST_KEY,
    assessment_event_id_for_membership,
    assessment_identity_source_id,
    build_channel_assessment_input,
    build_channel_graph_refresh_input,
    channel_assessment_request_event,
)
from kortny.tasks import TaskService
from kortny.tasks.identity import TaskIdentity
from kortny.tools.types import JsonObject

logger = logging.getLogger(__name__)

CONSOLIDATOR_MERGE_PROMPT_NAME = "kortny.consolidator_merge"
MERGE_RESPONSE_FORMAT: JsonObject = {"type": "json_object"}
CONSOLIDATOR_EXTRACTOR = "kortny.consolidator"
PROFILE_REFRESH_SOURCE = "consolidator_profile_refresh"

CANDIDATE_ADJUDICATION_MIN_AGE = timedelta(days=3)
CANDIDATE_ARCHIVE_MIN_AGE = timedelta(days=7)
CANDIDATE_CONSENSUS_THRESHOLD = 2
STALE_ARCHIVE_AFTER = timedelta(days=90)
PROFILE_REFRESH_AFTER = timedelta(days=7)
PROFILE_STALE_AFTER = timedelta(days=30)
DEFAULT_OBSERVATION_RETENTION_DAYS = 90
MERGE_SIMILARITY_THRESHOLD = 0.92
MERGE_PAIR_CAP = 20
USER_CONFIRMED_SOURCE_TYPE = "user_confirmed"
FACT_PROJECTION_KEY_PREFIX = "workspace_fact"
PROACTIVE_OUTCOMES_BATCH_CAP = 200


@dataclass(frozen=True, slots=True)
class AdjudicationCounters:
    activated: int = 0
    archived: int = 0

    def to_payload(self) -> dict[str, int]:
        return {"activated": self.activated, "archived": self.archived}


def adjudicate_candidates(
    session: Session,
    *,
    installation_id: uuid.UUID,
    now: datetime | None = None,
) -> AdjudicationCounters:
    """Pass 2: deterministic candidate adjudication (no LLM)."""

    effective_now = now or datetime.now(UTC)
    entity_rows: list[KnowledgeGraphEntity | KnowledgeGraphEdge] = list(
        session.scalars(
            select(KnowledgeGraphEntity).where(
                KnowledgeGraphEntity.installation_id == installation_id,
                KnowledgeGraphEntity.lifecycle_state == "candidate",
                KnowledgeGraphEntity.is_current.is_(True),
                KnowledgeGraphEntity.expired_at.is_(None),
                KnowledgeGraphEntity.system_expired_at.is_(None),
                KnowledgeGraphEntity.created_at
                < effective_now - CANDIDATE_ADJUDICATION_MIN_AGE,
            )
        )
    )
    edge_rows: list[KnowledgeGraphEntity | KnowledgeGraphEdge] = list(
        session.scalars(
            select(KnowledgeGraphEdge).where(
                KnowledgeGraphEdge.installation_id == installation_id,
                KnowledgeGraphEdge.lifecycle_state == "candidate",
                KnowledgeGraphEdge.is_current.is_(True),
                KnowledgeGraphEdge.expired_at.is_(None),
                KnowledgeGraphEdge.system_expired_at.is_(None),
                KnowledgeGraphEdge.created_at
                < effective_now - CANDIDATE_ADJUDICATION_MIN_AGE,
            )
        )
    )

    activated = 0
    archived = 0
    for target_kind, rows in (("entity", entity_rows), ("edge", edge_rows)):
        for row in rows:
            consensus = session.scalar(
                select(
                    func.coalesce(func.sum(KnowledgeGraphEvidence.consensus_count), 0)
                ).where(
                    KnowledgeGraphEvidence.installation_id == installation_id,
                    KnowledgeGraphEvidence.target_kind == target_kind,
                    KnowledgeGraphEvidence.target_id == row.id,
                )
            )
            if int(consensus or 0) >= CANDIDATE_CONSENSUS_THRESHOLD:
                row.lifecycle_state = "active"
                if row.valid_at is None:
                    row.valid_at = effective_now
                row.updated_at = effective_now
                activated += 1
            elif row.created_at.replace(tzinfo=row.created_at.tzinfo or UTC) < (
                effective_now - CANDIDATE_ARCHIVE_MIN_AGE
            ):
                row.lifecycle_state = "archived"
                row.system_expired_at = effective_now
                row.is_current = False
                row.updated_at = effective_now
                archived += 1
    session.flush()
    return AdjudicationCounters(activated=activated, archived=archived)


@dataclass(frozen=True, slots=True)
class MergeCounters:
    pairs_considered: int = 0
    merged: int = 0

    def to_payload(self) -> dict[str, int]:
        return {"pairs_considered": self.pairs_considered, "merged": self.merged}


def merge_duplicate_entities(
    session: Session,
    *,
    installation_id: uuid.UUID,
    graph: GraphService,
    embedding_index: EmbeddingIndex | None,
    llm: LLMService | None,
    task: Task,
    now: datetime | None = None,
    similarity_threshold: float = MERGE_SIMILARITY_THRESHOLD,
    pair_cap: int = MERGE_PAIR_CAP,
) -> MergeCounters:
    """Pass 3: merge near-duplicate entities within scope + entity_type."""

    if embedding_index is None or llm is None:
        return MergeCounters()
    effective_now = now or datetime.now(UTC)
    entities = list(
        session.scalars(
            select(KnowledgeGraphEntity).where(
                KnowledgeGraphEntity.installation_id == installation_id,
                KnowledgeGraphEntity.is_current.is_(True),
                KnowledgeGraphEntity.expired_at.is_(None),
                KnowledgeGraphEntity.system_expired_at.is_(None),
                KnowledgeGraphEntity.invalid_at.is_(None),
                KnowledgeGraphEntity.lifecycle_state.in_(
                    ("candidate", "active", "confirmed")
                ),
            )
        )
    )
    groups: dict[tuple[str, str | None, str], list[KnowledgeGraphEntity]] = {}
    for entity in entities:
        key = (
            entity.visibility_scope_type,
            entity.visibility_scope_id,
            entity.entity_type,
        )
        groups.setdefault(key, []).append(entity)

    pairs: list[tuple[KnowledgeGraphEntity, KnowledgeGraphEntity]] = []
    seen_pairs: set[tuple[uuid.UUID, uuid.UUID]] = set()
    for group in groups.values():
        if len(group) < 2:
            continue
        by_id = {str(entity.id): entity for entity in group}
        for entity in group:
            other_ids = [ref for ref in by_id if ref != str(entity.id)]
            ranked = embedding_index.rank(
                KG_ENTITY_EMBEDDING_KIND,
                kg_entity_embedding_text(entity),
                other_ids,
                top_k=len(other_ids),
            )
            if not ranked:
                continue
            for ref_key, similarity in ranked:
                if similarity < similarity_threshold:
                    continue
                other = by_id.get(ref_key)
                if other is None:
                    continue
                older, newer = _ordered_pair(entity, other)
                pair_key = (older.id, newer.id)
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                pairs.append((older, newer))
                if len(pairs) >= pair_cap:
                    break
            if len(pairs) >= pair_cap:
                break
        if len(pairs) >= pair_cap:
            break

    if not pairs:
        return MergeCounters()

    confirmed = _confirm_merges(llm, task, pairs)
    merged = 0
    for older, newer in pairs:
        if (str(older.id), str(newer.id)) not in confirmed:
            continue
        _move_evidence(session, installation_id, source=newer, target=older)
        _move_edges(session, graph, installation_id, source=newer, target=older)
        graph.supersede_entity(newer, older)
        older.last_reinforced_at = effective_now
        older.reinforcement_count = (older.reinforcement_count or 0) + 1
        older.updated_at = effective_now
        merged += 1
    session.flush()
    return MergeCounters(pairs_considered=len(pairs), merged=merged)


def _confirm_merges(
    llm: LLMService,
    task: Task,
    pairs: list[tuple[KnowledgeGraphEntity, KnowledgeGraphEntity]],
) -> set[tuple[str, str]]:
    payload = [
        {
            "keep_id": str(older.id),
            "keep": {
                "canonical_key": older.canonical_key,
                "display_name": older.display_name,
                "summary": _attrs_summary(older),
            },
            "merge_id": str(newer.id),
            "merge": {
                "canonical_key": newer.canonical_key,
                "display_name": newer.display_name,
                "summary": _attrs_summary(newer),
            },
        }
        for older, newer in pairs
    ]
    completion = llm.complete(
        task_id=task.id,
        messages=(
            ChatMessage(
                role="system",
                content=(
                    "You deduplicate entries in Kortny's workspace knowledge "
                    "graph. For each candidate pair decide whether they "
                    "describe the same real-world thing and should be merged. "
                    "Return only the JSON object — no prose, markdown, or "
                    'comments. Schema: {"merges":[{"keep_id":"uuid",'
                    '"merge_id":"uuid","merge":true}]} — include every pair '
                    "with merge true or false. Only merge clear duplicates; "
                    "when in doubt set merge to false. Never invent or change "
                    "entity ids — use only the keep_id and merge_id values "
                    "from the provided pairs. "
                    "Examples: "
                    '{"pairs":[{"keep_id":"aaa","keep":{"canonical_key":"github","display_name":"GitHub"},"merge_id":"bbb","merge":{"canonical_key":"github_com","display_name":"GitHub.com"}}]} '
                    '-> {"merges":[{"keep_id":"aaa","merge_id":"bbb","merge":true}]} '
                    '{"pairs":[{"keep_id":"ccc","keep":{"canonical_key":"stripe","display_name":"Stripe Payments"},"merge_id":"ddd","merge":{"canonical_key":"stripe_integration","display_name":"Stripe API"}}]} '
                    '-> {"merges":[{"keep_id":"ccc","merge_id":"ddd","merge":true}]} '
                    '{"pairs":[{"keep_id":"eee","keep":{"canonical_key":"alice","display_name":"Alice"},"merge_id":"fff","merge":{"canonical_key":"bob","display_name":"Bob"}}]} '
                    '-> {"merges":[{"keep_id":"eee","merge_id":"fff","merge":false}]} '
                    "Ground every field in the input; abstain when unsupported."
                ),
            ),
            ChatMessage(
                role="user",
                content=json.dumps(
                    {"pairs": payload}, separators=(",", ":"), sort_keys=True
                ),
            ),
        ),
        response_format=MERGE_RESPONSE_FORMAT,
        prompt_name=CONSOLIDATOR_MERGE_PROMPT_NAME,
    )
    confirmed: set[tuple[str, str]] = set()
    try:
        parsed = json.loads(completion.content or "{}")
    except json.JSONDecodeError:
        return confirmed
    merges = parsed.get("merges") if isinstance(parsed, dict) else None
    if not isinstance(merges, list):
        return confirmed
    for item in merges:
        if not isinstance(item, dict) or item.get("merge") is not True:
            continue
        keep_id = item.get("keep_id")
        merge_id = item.get("merge_id")
        if isinstance(keep_id, str) and isinstance(merge_id, str):
            confirmed.add((keep_id, merge_id))
    return confirmed


def _move_evidence(
    session: Session,
    installation_id: uuid.UUID,
    *,
    source: KnowledgeGraphEntity,
    target: KnowledgeGraphEntity,
) -> None:
    session.execute(
        update(KnowledgeGraphEvidence)
        .where(
            KnowledgeGraphEvidence.installation_id == installation_id,
            KnowledgeGraphEvidence.target_kind == "entity",
            KnowledgeGraphEvidence.target_id == source.id,
        )
        .values(target_id=target.id)
    )


def _move_edges(
    session: Session,
    graph: GraphService,
    installation_id: uuid.UUID,
    *,
    source: KnowledgeGraphEntity,
    target: KnowledgeGraphEntity,
) -> None:
    edges = list(
        session.scalars(
            select(KnowledgeGraphEdge).where(
                KnowledgeGraphEdge.installation_id == installation_id,
                (KnowledgeGraphEdge.source_entity_id == source.id)
                | (KnowledgeGraphEdge.target_entity_id == source.id),
                KnowledgeGraphEdge.is_current.is_(True),
                KnowledgeGraphEdge.expired_at.is_(None),
            )
        )
    )
    for edge in edges:
        new_source = (
            target.id if edge.source_entity_id == source.id else edge.source_entity_id
        )
        new_target = (
            target.id if edge.target_entity_id == source.id else edge.target_entity_id
        )
        duplicate = session.scalar(
            select(KnowledgeGraphEdge).where(
                KnowledgeGraphEdge.installation_id == installation_id,
                KnowledgeGraphEdge.source_entity_id == new_source,
                KnowledgeGraphEdge.target_entity_id == new_target,
                KnowledgeGraphEdge.relationship_type == edge.relationship_type,
                KnowledgeGraphEdge.visibility_scope_type == edge.visibility_scope_type,
                func.coalesce(KnowledgeGraphEdge.visibility_scope_id, "")
                == (edge.visibility_scope_id or ""),
                KnowledgeGraphEdge.is_current.is_(True),
                KnowledgeGraphEdge.expired_at.is_(None),
                KnowledgeGraphEdge.id != edge.id,
            )
        )
        if duplicate is not None:
            graph.supersede_edge(edge, duplicate)
            continue
        edge.source_entity_id = new_source
        edge.target_entity_id = new_target
    session.flush()


@dataclass(frozen=True, slots=True)
class AgingCounters:
    staled_entities: int = 0
    staled_edges: int = 0
    archived: int = 0

    def to_payload(self) -> dict[str, int]:
        return {
            "staled_entities": self.staled_entities,
            "staled_edges": self.staled_edges,
            "archived": self.archived,
        }


def age_graph(
    session: Session,
    *,
    installation_id: uuid.UUID,
    graph: GraphService,
    stale_days: int,
    now: datetime | None = None,
) -> AgingCounters:
    """Pass 4: stale aging + archive (finally wires mark_stale_current)."""

    effective_now = now or datetime.now(UTC)
    staleness = graph.mark_stale_current(
        installation_id=installation_id,
        now=effective_now,
        default_stale_days=stale_days,
    )
    stale_rows: list[KnowledgeGraphEntity | KnowledgeGraphEdge] = list(
        session.scalars(
            select(KnowledgeGraphEntity).where(
                KnowledgeGraphEntity.installation_id == installation_id,
                KnowledgeGraphEntity.lifecycle_state == "stale",
                KnowledgeGraphEntity.system_expired_at.is_(None),
                KnowledgeGraphEntity.updated_at < effective_now - STALE_ARCHIVE_AFTER,
            )
        )
    )
    stale_rows.extend(
        session.scalars(
            select(KnowledgeGraphEdge).where(
                KnowledgeGraphEdge.installation_id == installation_id,
                KnowledgeGraphEdge.lifecycle_state == "stale",
                KnowledgeGraphEdge.system_expired_at.is_(None),
                KnowledgeGraphEdge.updated_at < effective_now - STALE_ARCHIVE_AFTER,
            )
        )
    )
    archived = 0
    for row in stale_rows:
        row.lifecycle_state = "archived"
        row.system_expired_at = effective_now
        row.is_current = False
        row.updated_at = effective_now
        archived += 1
    session.flush()
    return AgingCounters(
        staled_entities=staleness.entity_count,
        staled_edges=staleness.edge_count,
        archived=archived,
    )


@dataclass(frozen=True, slots=True)
class FactProjectionCounters:
    projected: int = 0
    refreshed: int = 0
    unchanged: int = 0

    def to_payload(self) -> dict[str, int]:
        return {
            "projected": self.projected,
            "refreshed": self.refreshed,
            "unchanged": self.unchanged,
        }


def project_confirmed_facts(
    session: Session,
    *,
    installation_id: uuid.UUID,
    graph: GraphService,
    task: Task,
    now: datetime | None = None,
) -> FactProjectionCounters:
    """Pass 5: project active user-confirmed facts into the graph.

    Idempotent: the entity is keyed by fact scope + key, and the active
    workspace_state row id is stored in the payload — re-running with the same
    fact is a no-op; a superseding fact refreshes the same entity.
    """

    effective_now = now or datetime.now(UTC)
    facts = list(
        session.scalars(
            select(WorkspaceState).where(
                WorkspaceState.installation_id == installation_id,
                WorkspaceState.status == "active",
                (WorkspaceState.expires_at.is_(None))
                | (WorkspaceState.expires_at > effective_now),
            )
        )
    )
    projected = 0
    refreshed = 0
    unchanged = 0
    for fact in facts:
        canonical_key = (
            f"{FACT_PROJECTION_KEY_PREFIX}:{fact.scope_type}:"
            f"{fact.scope_id or 'workspace'}:{fact.key}"
        )
        evidence = EvidenceInput(
            source_type="workspace_state",
            extracted_by=CONSOLIDATOR_EXTRACTOR,
            source_task_id=fact.source_task_id or task.id,
            source_slack_channel_id=fact.source_slack_channel_id,
            source_slack_message_ts=fact.source_slack_message_ts,
            raw_snippet=(fact.value_text or json.dumps(fact.value_json, default=str))[
                :700
            ],
            confidence_score=Decimal("0.900"),
            confidence_reason="User confirmed this fact via the memory flow.",
        )
        existing = session.scalar(
            select(KnowledgeGraphEntity).where(
                KnowledgeGraphEntity.installation_id == installation_id,
                KnowledgeGraphEntity.canonical_key == canonical_key,
                KnowledgeGraphEntity.is_current.is_(True),
                KnowledgeGraphEntity.expired_at.is_(None),
            )
        )
        if existing is not None:
            attrs = existing.attrs_json if isinstance(existing.attrs_json, dict) else {}
            if attrs.get("workspace_state_id") == str(fact.id):
                unchanged += 1
                continue
            new_attrs = dict(attrs)
            new_attrs["workspace_state_id"] = str(fact.id)
            new_attrs["key"] = fact.key
            new_attrs["summary"] = fact.value_text or json.dumps(
                fact.value_json, default=str
            )
            existing.attrs_json = new_attrs
            existing.last_reinforced_at = effective_now
            existing.reinforcement_count = (existing.reinforcement_count or 0) + 1
            existing.updated_at = effective_now
            graph.add_evidence(
                installation_id=installation_id,
                target_kind="entity",
                target_id=existing.id,
                evidence=evidence,
            )
            graph.ensure_entity_embedding(existing)
            refreshed += 1
            continue
        graph.create_entity(
            installation_id=installation_id,
            entity_type="firm_fact",
            canonical_key=canonical_key,
            visibility_scope=_fact_scope(fact),
            source_type=USER_CONFIRMED_SOURCE_TYPE,
            display_name=fact.key.replace("_", " "),
            attrs_json={
                "workspace_state_id": str(fact.id),
                "key": fact.key,
                "summary": fact.value_text or json.dumps(fact.value_json, default=str),
            },
            lifecycle_state="confirmed",
            confidence_score=Decimal("0.900"),
            confidence_reason="Projected from a user-confirmed workspace fact.",
            evidence=evidence,
        )
        projected += 1
    session.flush()
    return FactProjectionCounters(
        projected=projected, refreshed=refreshed, unchanged=unchanged
    )


def _fact_scope(fact: WorkspaceState) -> VisibilityScope:
    if fact.scope_type == "channel" and fact.scope_id:
        if fact.scope_id.startswith("D"):
            return VisibilityScope.dm(fact.scope_id)
        if fact.scope_id.startswith("G"):
            return VisibilityScope.private_channel(fact.scope_id)
        return VisibilityScope.channel(fact.scope_id)
    if fact.scope_type == "user" and fact.scope_id:
        return VisibilityScope.user(fact.scope_id)
    return VisibilityScope.workspace()


@dataclass(frozen=True, slots=True)
class HygieneCounters:
    purged_observations: int = 0
    expired_facts: int = 0
    profiles_refreshed: int = 0
    profiles_marked_stale: int = 0
    assessments_requeued: int = 0
    outcomes_reconciled: int = 0

    def to_payload(self) -> dict[str, int]:
        return {
            "purged_observations": self.purged_observations,
            "expired_facts": self.expired_facts,
            "profiles_refreshed": self.profiles_refreshed,
            "profiles_marked_stale": self.profiles_marked_stale,
            "assessments_requeued": self.assessments_requeued,
            "outcomes_reconciled": self.outcomes_reconciled,
        }


def run_hygiene(
    session: Session,
    *,
    installation_id: uuid.UUID,
    task_service: TaskService | None = None,
    now: datetime | None = None,
) -> HygieneCounters:
    """Pass 6: retention purge, fact TTL purge, stale profile refresh."""

    effective_now = now or datetime.now(UTC)
    effective_task_service = task_service or TaskService(session)
    purged = _purge_observations(
        session, installation_id=installation_id, now=effective_now
    )
    expired = _expire_facts(session, installation_id=installation_id, now=effective_now)
    marked_stale = _mark_stale_profiles(
        session, installation_id=installation_id, now=effective_now
    )
    refreshed = _refresh_stale_profiles(
        session,
        installation_id=installation_id,
        task_service=effective_task_service,
        now=effective_now,
    )
    requeued = _requeue_failed_assessments(
        session,
        installation_id=installation_id,
        task_service=effective_task_service,
        now=effective_now,
    )
    reconciled = _reconcile_proactive_outcomes(
        session,
        installation_id=installation_id,
        now=effective_now,
    )
    session.flush()
    return HygieneCounters(
        purged_observations=purged,
        expired_facts=expired,
        profiles_refreshed=refreshed,
        profiles_marked_stale=marked_stale,
        assessments_requeued=requeued,
        outcomes_reconciled=reconciled,
    )


def _purge_observations(
    session: Session,
    *,
    installation_id: uuid.UUID,
    now: datetime,
) -> int:
    policies = list(
        session.scalars(
            select(ObservePolicy).where(
                ObservePolicy.installation_id == installation_id
            )
        )
    )
    workspace_retention = DEFAULT_OBSERVATION_RETENTION_DAYS
    channel_retention: dict[str, int] = {}
    for policy in policies:
        if policy.retention_days is None:
            continue
        if policy.scope_type == "workspace":
            workspace_retention = policy.retention_days
        elif policy.scope_type == "channel" and policy.scope_id:
            channel_retention[policy.scope_id] = policy.retention_days

    purged = 0
    channels = [
        row[0]
        for row in session.execute(
            select(ObservationEvent.channel_id)
            .where(ObservationEvent.installation_id == installation_id)
            .distinct()
        )
    ]
    for channel_id in channels:
        retention_days = channel_retention.get(channel_id, workspace_retention)
        cutoff = now - timedelta(days=retention_days)
        result = session.execute(
            delete(ObservationEvent).where(
                ObservationEvent.installation_id == installation_id,
                ObservationEvent.channel_id == channel_id,
                ObservationEvent.observed_at < cutoff,
            )
        )
        if isinstance(result, CursorResult):
            purged += int(result.rowcount or 0)
    return purged


def _expire_facts(
    session: Session,
    *,
    installation_id: uuid.UUID,
    now: datetime,
) -> int:
    rows = list(
        session.scalars(
            select(WorkspaceState)
            .where(
                WorkspaceState.installation_id == installation_id,
                WorkspaceState.status == "active",
                WorkspaceState.expires_at.is_not(None),
                WorkspaceState.expires_at <= now,
            )
            .with_for_update()
        )
    )
    for row in rows:
        row.status = "superseded"
        row.superseded_at = now
        row.updated_at = now
    return len(rows)


def _mark_stale_profiles(
    session: Session,
    *,
    installation_id: uuid.UUID,
    now: datetime,
) -> int:
    """Hygiene step: mark active profiles whose last_profiled_at is older than
    PROFILE_STALE_AFTER as 'stale' so the witness runner ignores them while
    the consolidator re-queues a refresh task."""

    stale_cutoff = now - PROFILE_STALE_AFTER
    rows = list(
        session.scalars(
            select(ObserveChannelProfile).where(
                ObserveChannelProfile.installation_id == installation_id,
                ObserveChannelProfile.profile_status == "active",
                ObserveChannelProfile.last_profiled_at.is_not(None),
                ObserveChannelProfile.last_profiled_at < stale_cutoff,
            )
        )
    )
    for profile in rows:
        profile.profile_status = "stale"
        profile.updated_at = now
    return len(rows)


def _refresh_stale_profiles(
    session: Session,
    *,
    installation_id: uuid.UUID,
    task_service: TaskService,
    now: datetime,
) -> int:
    profiles = list(
        session.scalars(
            select(ObserveChannelProfile).where(
                ObserveChannelProfile.installation_id == installation_id,
                ObserveChannelProfile.profile_status.in_(["active", "stale"]),
                (ObserveChannelProfile.last_profiled_at.is_(None))
                | (
                    ObserveChannelProfile.last_profiled_at < now - PROFILE_REFRESH_AFTER
                ),
            )
        )
    )
    refreshed = 0
    for profile in profiles:
        activity_floor = profile.last_profiled_at or profile.created_at
        has_activity = session.scalar(
            select(ObservationEvent.id)
            .where(
                ObservationEvent.installation_id == installation_id,
                ObservationEvent.channel_id == profile.channel_id,
                ObservationEvent.observed_at > activity_floor,
            )
            .limit(1)
        )
        if has_activity is None:
            continue
        membership = session.scalar(
            select(SlackChannelMembership).where(
                SlackChannelMembership.installation_id == installation_id,
                SlackChannelMembership.channel_id == profile.channel_id,
                SlackChannelMembership.membership_status == "active",
            )
        )
        if membership is None:
            continue
        task_input = build_channel_graph_refresh_input(channel_id=profile.channel_id)
        refresh_task = task_service.create_task(
            installation_id=installation_id,
            slack_event_id=(
                f"consolidator:{profile.id}:{profile.profile_version}:"
                f"{now.date().isoformat()}"
            ),
            slack_channel_id=profile.channel_id,
            slack_thread_ts=membership.onboarding_message_ts,
            slack_message_ts=membership.onboarding_message_ts,
            slack_user_id=membership.added_by_user_id or "consolidator",
            input=task_input,
            identity=TaskIdentity.synthetic(
                source=PROFILE_REFRESH_SOURCE,
                source_id=(
                    f"{profile.id}:{profile.profile_version}:{now.date().isoformat()}"
                ),
                input_text=task_input,
                payload={
                    "channel_id": profile.channel_id,
                    "profile_id": str(profile.id),
                },
            ),
            source_surface=PROFILE_REFRESH_SOURCE,
        )
        task_service.append_event(
            refresh_task,
            TaskEventType.log,
            {
                "message": CHANNEL_ASSESSMENT_REQUESTED_MESSAGE,
                "source": PROFILE_REFRESH_SOURCE,
                "channel_id": profile.channel_id,
                "membership_id": str(membership.id),
                "profile_id": str(profile.id),
                "requested_at": now.isoformat(),
                CHANNEL_ASSESSMENT_SUPPRESS_SLACK_POST_KEY: True,
            },
        )
        refreshed += 1
    return refreshed


def _requeue_failed_assessments(
    session: Session,
    *,
    installation_id: uuid.UUID,
    task_service: TaskService,
    now: datetime,
) -> int:
    """Re-queue channel assessment tasks for memberships whose assessment has
    failed (but is not dead-lettered) and whose retry window has passed."""

    _active_statuses = frozenset(
        {
            TaskStatus.pending,
            TaskStatus.running,
            TaskStatus.waiting_approval,
            TaskStatus.crashed,
        }
    )

    memberships = list(
        session.scalars(
            select(SlackChannelMembership).where(
                SlackChannelMembership.installation_id == installation_id,
                SlackChannelMembership.membership_status == "active",
                SlackChannelMembership.onboarding_status == "posted",
                SlackChannelMembership.metadata_json["assessment_status"].as_string()
                == "failed",
            )
        )
    )

    requeued = 0
    for membership in memberships:
        metadata = membership.metadata_json or {}

        # Skip dead-lettered memberships.
        if metadata.get("assessment_dead_lettered_at"):
            continue

        # Skip if still inside the backoff window.
        next_attempt_raw = metadata.get("assessment_next_attempt_at")
        if next_attempt_raw:
            try:
                next_attempt = datetime.fromisoformat(str(next_attempt_raw))
                if next_attempt.tzinfo is None:
                    next_attempt = next_attempt.replace(tzinfo=UTC)
                if next_attempt > now:
                    continue
            except (ValueError, TypeError):
                pass

        # Skip if the previous task is still in-flight.
        task_id_raw = metadata.get("assessment_task_id")
        if task_id_raw:
            try:
                existing_task_id = uuid.UUID(str(task_id_raw))
                existing_task = session.get(Task, existing_task_id)
                if (
                    existing_task is not None
                    and TaskStatus(existing_task.status) in _active_statuses
                ):
                    continue
            except (ValueError, AttributeError):
                pass

        # Attempt number is the current failure count (pre-increment).
        attempt = int(metadata.get("assessment_failure_count") or 0)

        task_input = build_channel_assessment_input(channel_id=membership.channel_id)
        task = task_service.create_task(
            installation_id=installation_id,
            slack_event_id=assessment_event_id_for_membership(membership.id, attempt),
            slack_channel_id=membership.channel_id,
            slack_thread_ts=membership.onboarding_message_ts,
            slack_message_ts=membership.onboarding_message_ts,
            slack_user_id=membership.added_by_user_id or "consolidator",
            input=task_input,
            identity=TaskIdentity.synthetic(
                source="channel_assessment",
                source_id=assessment_identity_source_id(membership.id, attempt),
                input_text=task_input,
                payload={
                    "channel_id": membership.channel_id,
                    "membership_id": str(membership.id),
                },
            ),
            source_surface=PROFILE_REFRESH_SOURCE,
        )
        if channel_assessment_request_event(session, task) is None:
            task_service.append_event(
                task,
                TaskEventType.log,
                {
                    "message": CHANNEL_ASSESSMENT_REQUESTED_MESSAGE,
                    "source": PROFILE_REFRESH_SOURCE,
                    "channel_id": membership.channel_id,
                    "membership_id": str(membership.id),
                },
            )
        requeued += 1
    return requeued


_TERMINAL_TASK_STATUSES: frozenset[TaskStatus] = frozenset(
    {
        TaskStatus.succeeded,
        TaskStatus.failed,
        TaskStatus.crashed,
        TaskStatus.cancelled,
    }
)


def _reconcile_proactive_outcomes(
    session: Session,
    *,
    installation_id: uuid.UUID,
    now: datetime,
) -> int:
    """Hygiene step: back-fill task_status / task_finished_at on reconciled
    proactive candidates whose linked task has reached a terminal state.

    Idempotent: skips rows where task_status is already set.
    """
    from kortny.witness.ledger.service import (  # noqa: PLC0415
        ProactiveActionService,
    )

    ledger = ProactiveActionService()

    candidates = list(
        session.scalars(
            select(WitnessOpportunityCandidate)
            .where(
                WitnessOpportunityCandidate.installation_id == installation_id,
                WitnessOpportunityCandidate.automated_task_id.is_not(None),
                WitnessOpportunityCandidate.task_status.is_(None),
            )
            .limit(PROACTIVE_OUTCOMES_BATCH_CAP)
            .with_for_update(skip_locked=True)
        )
    )

    reconciled = 0
    for candidate in candidates:
        task = session.get(Task, candidate.automated_task_id)
        if task is None:
            continue
        task_status = TaskStatus(task.status)
        if task_status not in _TERMINAL_TASK_STATUSES:
            continue
        candidate.task_status = task.status
        candidate.task_finished_at = task.updated_at
        to_state = (
            "task_succeeded" if task_status is TaskStatus.succeeded else "task_failed"
        )
        ledger.record_transition(
            session,
            candidate,
            to_state=to_state,
            event_type="task_terminal",
            reason_code=task.status,
            task_id=task.id,
            now=now,
        )
        reconciled += 1

    return reconciled


@dataclass(frozen=True, slots=True)
class BackfillCounters:
    embedded: int = 0

    def to_payload(self) -> dict[str, int]:
        return {"embedded": self.embedded}


def backfill_embeddings(
    session: Session,
    *,
    installation_id: uuid.UUID,
    embedding_index: EmbeddingIndex | None,
    max_items: int = 500,
) -> BackfillCounters:
    """Pass 7: sha-gated embedding backfill for facts/episodes/entities.

    Bounded per run (``max_items`` rows scanned across the three stores) so
    a large backlog can't drive an unbounded embed burst in one run — the
    sha gate makes the leftover drain across subsequent runs.
    """

    if embedding_index is None:
        return BackfillCounters()
    embedded = 0
    remaining = max_items
    facts = list(
        session.scalars(
            select(WorkspaceState)
            .where(
                WorkspaceState.installation_id == installation_id,
                WorkspaceState.status == "active",
            )
            .limit(remaining)
        )
    )
    remaining -= len(facts)
    embedded += embedding_index.ensure(
        FACT_EMBEDDING_KIND,
        [(str(fact.id), fact_embedding_text(fact)) for fact in facts],
    )
    episodes = (
        list(
            session.scalars(
                select(Episode)
                .where(Episode.installation_id == installation_id)
                .limit(remaining)
            )
        )
        if remaining > 0
        else []
    )
    remaining -= len(episodes)
    embedded += embedding_index.ensure(
        EPISODE_EMBEDDING_KIND,
        [(str(episode.id), episode_embedding_text(episode)) for episode in episodes],
    )
    entities = (
        list(
            session.scalars(
                select(KnowledgeGraphEntity)
                .where(
                    KnowledgeGraphEntity.installation_id == installation_id,
                    KnowledgeGraphEntity.is_current.is_(True),
                    KnowledgeGraphEntity.expired_at.is_(None),
                    KnowledgeGraphEntity.system_expired_at.is_(None),
                )
                .limit(remaining)
            )
        )
        if remaining > 0
        else []
    )
    embedded += embedding_index.ensure(
        KG_ENTITY_EMBEDDING_KIND,
        [(str(entity.id), kg_entity_embedding_text(entity)) for entity in entities],
    )
    return BackfillCounters(embedded=embedded)


def _ordered_pair(
    a: KnowledgeGraphEntity,
    b: KnowledgeGraphEntity,
) -> tuple[KnowledgeGraphEntity, KnowledgeGraphEntity]:
    if (a.created_at, str(a.id)) <= (b.created_at, str(b.id)):
        return a, b
    return b, a


def _attrs_summary(entity: KnowledgeGraphEntity) -> str | None:
    attrs = entity.attrs_json if isinstance(entity.attrs_json, dict) else {}
    summary = attrs.get("summary")
    if isinstance(summary, str) and summary.strip():
        return summary.strip()[:400]
    return None
