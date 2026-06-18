"""Implicit project inference (HIG-276 increment 2).

A consolidator pass that notices when people keep circling the same
topic/workflow/deadline — a cluster of recurring, co-occurring graph entities —
and PROPOSES it as a project (never auto-creates; a wrong guess colouring every
answer is worse than one confirm step). On confirm it becomes a real `project`
hub via ProjectProposalService.

Deterministic vs LLM split (per the reconciled HIG-276 design): deterministic
code does cluster assembly, scoring, privacy filtering, dedupe, and persistence;
the LLM only NAMES the cluster and writes the public "why" line. The pass runs
headless here (creates the proposal row); Slack delivery + reaction-confirm is a
separate layer.

Privacy: the proposal's public_summary / public_evidence are built ONLY from
workspace/public-channel entities. Private signals are recorded as opaque refs
(private_evidence) and never described in user-facing text — the recipient may
not see those surfaces.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.orm import Session

from kortny.db.models import (
    KnowledgeGraphEntity,
    KnowledgeGraphEvidence,
    ProjectProposal,
)
from kortny.knowledge_graph.project_proposals import ProjectProposalService
from kortny.knowledge_graph.projects import project_slug
from kortny.knowledge_graph.scopes import SCOPE_CHANNEL, SCOPE_WORKSPACE
from kortny.llm import ChatMessage, LLMService

logger = logging.getLogger(__name__)

# Entity types that represent "work" (topics/decisions/deadlines/ideas), the
# things a project clusters — not structural rows like channel/person/project.
_PROJECT_RELEVANT_TYPES = (
    "decision",
    "open_question",
    "commitment",
    "external_entity",
    "artifact",
)
_DEADLINE_TYPES = frozenset({"commitment", "open_question"})
_PUBLIC_SCOPES = frozenset({SCOPE_WORKSPACE, SCOPE_CHANNEL})
_CURRENT_STATES = ("active", "confirmed")

DEFAULT_MIN_CLUSTER_SIZE = 3
DEFAULT_MIN_REINFORCEMENT = 2
DEFAULT_MIN_CHANNEL_SPREAD = 2
DEFAULT_MIN_CONFIDENCE = Decimal("0.6")
DEFAULT_RECENT_WINDOW = timedelta(days=30)
DEFAULT_REPROPOSE_COOLDOWN = timedelta(days=14)
_MAX_CANDIDATES = 400
_NAME_PROMPT = "kortny.project_inference_namer"

# Resolve a user id -> open DM channel id (same seam org_profile uses).
DmChannelResolver = Callable[[str], str | None]


@dataclass(frozen=True)
class ProjectInferenceCounters:
    candidates: int = 0
    clusters: int = 0
    proposed: int = 0
    skipped_reason: str | None = None

    def to_payload(self) -> dict[str, object]:
        return {
            "candidates": self.candidates,
            "clusters": self.clusters,
            "proposed": self.proposed,
            "skipped_reason": self.skipped_reason,
        }


@dataclass
class _Cluster:
    entities: list[KnowledgeGraphEntity] = field(default_factory=list)
    source_tasks: set[uuid.UUID] = field(default_factory=set)

    @property
    def public_entities(self) -> list[KnowledgeGraphEntity]:
        return [e for e in self.entities if e.visibility_scope_type in _PUBLIC_SCOPES]

    @property
    def private_entities(self) -> list[KnowledgeGraphEntity]:
        return [
            e for e in self.entities if e.visibility_scope_type not in _PUBLIC_SCOPES
        ]

    def public_channel_ids(self) -> list[str]:
        ids: list[str] = []
        for entity in self.public_entities:
            if (
                entity.visibility_scope_type == SCOPE_CHANNEL
                and entity.visibility_scope_id
            ):
                ids.append(entity.visibility_scope_id)
        return list(dict.fromkeys(ids))

    def has_deadline(self) -> bool:
        return any(e.entity_type in _DEADLINE_TYPES for e in self.entities)

    def total_reinforcement(self) -> int:
        return sum(e.reinforcement_count for e in self.entities)


class ProjectInferencePass:
    """Infer + propose implicit projects from recurring entity clusters."""

    def __init__(
        self,
        session: Session,
        *,
        llm: LLMService | None = None,
        min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
        min_reinforcement: int = DEFAULT_MIN_REINFORCEMENT,
        min_channel_spread: int = DEFAULT_MIN_CHANNEL_SPREAD,
        min_confidence: Decimal = DEFAULT_MIN_CONFIDENCE,
        recent_window: timedelta = DEFAULT_RECENT_WINDOW,
        repropose_cooldown: timedelta = DEFAULT_REPROPOSE_COOLDOWN,
    ) -> None:
        self.session = session
        self.llm = llm
        self.min_cluster_size = min_cluster_size
        self.min_reinforcement = min_reinforcement
        self.min_channel_spread = min_channel_spread
        self.min_confidence = min_confidence
        self.recent_window = recent_window
        self.repropose_cooldown = repropose_cooldown
        self.proposals = ProjectProposalService(session)

    def run(
        self,
        *,
        installation_id: uuid.UUID,
        task_id: uuid.UUID | None = None,
        now: datetime | None = None,
    ) -> ProjectInferenceCounters:
        effective_now = now or datetime.now(UTC)
        entities = self._candidate_entities(installation_id, effective_now)
        if len(entities) < self.min_cluster_size:
            return ProjectInferenceCounters(
                candidates=len(entities), skipped_reason="insufficient_candidates"
            )

        clusters = self._cluster_by_cooccurrence(installation_id, entities)
        scored = [c for c in clusters if self._qualifies(c)]
        if not scored:
            return ProjectInferenceCounters(
                candidates=len(entities),
                clusters=len(clusters),
                skipped_reason="no_qualifying_cluster",
            )
        scored.sort(key=self._cluster_rank, reverse=True)

        for cluster in scored:
            proposal = self._propose_cluster(
                installation_id=installation_id,
                cluster=cluster,
                task_id=task_id,
                now=effective_now,
            )
            if proposal is not None:
                return ProjectInferenceCounters(
                    candidates=len(entities), clusters=len(clusters), proposed=1
                )
        return ProjectInferenceCounters(
            candidates=len(entities),
            clusters=len(clusters),
            skipped_reason="all_clusters_filtered",
        )

    # -- candidate assembly -------------------------------------------------

    def _candidate_entities(
        self, installation_id: uuid.UUID, now: datetime
    ) -> list[KnowledgeGraphEntity]:
        cutoff = now - self.recent_window
        rows = self.session.scalars(
            select(KnowledgeGraphEntity)
            .where(
                KnowledgeGraphEntity.installation_id == installation_id,
                KnowledgeGraphEntity.entity_type.in_(_PROJECT_RELEVANT_TYPES),
                KnowledgeGraphEntity.lifecycle_state.in_(_CURRENT_STATES),
                KnowledgeGraphEntity.reinforcement_count >= self.min_reinforcement,
                KnowledgeGraphEntity.last_reinforced_at >= cutoff,
            )
            .order_by(KnowledgeGraphEntity.reinforcement_count.desc())
            .limit(_MAX_CANDIDATES)
        ).all()
        return list(rows)

    def _cluster_by_cooccurrence(
        self,
        installation_id: uuid.UUID,
        entities: Sequence[KnowledgeGraphEntity],
    ) -> list[_Cluster]:
        """Union-find clusters of entities that share an evidence source task."""

        by_id = {entity.id: entity for entity in entities}
        tasks_by_entity = self._evidence_tasks(installation_id, list(by_id))

        parent: dict[uuid.UUID, uuid.UUID] = {eid: eid for eid in by_id}

        def find(x: uuid.UUID) -> uuid.UUID:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: uuid.UUID, b: uuid.UUID) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        # Two entities co-occur when they share a source task.
        owners_by_task: dict[uuid.UUID, list[uuid.UUID]] = {}
        for eid, task_ids in tasks_by_entity.items():
            for task_id in task_ids:
                owners_by_task.setdefault(task_id, []).append(eid)
        for owners in owners_by_task.values():
            first = owners[0]
            for other in owners[1:]:
                union(first, other)

        grouped: dict[uuid.UUID, _Cluster] = {}
        for eid, entity in by_id.items():
            root = find(eid)
            cluster = grouped.setdefault(root, _Cluster())
            cluster.entities.append(entity)
            cluster.source_tasks.update(tasks_by_entity.get(eid, set()))
        return list(grouped.values())

    def _evidence_tasks(
        self, installation_id: uuid.UUID, entity_ids: Sequence[uuid.UUID]
    ) -> dict[uuid.UUID, set[uuid.UUID]]:
        if not entity_ids:
            return {}
        rows = self.session.execute(
            select(
                KnowledgeGraphEvidence.target_id,
                KnowledgeGraphEvidence.source_task_id,
            ).where(
                KnowledgeGraphEvidence.installation_id == installation_id,
                KnowledgeGraphEvidence.target_kind == "entity",
                KnowledgeGraphEvidence.target_id.in_(tuple(entity_ids)),
                KnowledgeGraphEvidence.source_task_id.is_not(None),
            )
        ).all()
        out: dict[uuid.UUID, set[uuid.UUID]] = {}
        for target_id, source_task_id in rows:
            if source_task_id is None:
                continue
            out.setdefault(target_id, set()).add(source_task_id)
        return out

    # -- scoring ------------------------------------------------------------

    def _qualifies(self, cluster: _Cluster) -> bool:
        public = cluster.public_entities
        if len(public) < self.min_cluster_size:
            return False
        spread = len(cluster.public_channel_ids())
        return spread >= self.min_channel_spread or cluster.has_deadline()

    def _cluster_rank(self, cluster: _Cluster) -> tuple[int, int, int]:
        return (
            cluster.total_reinforcement(),
            len(cluster.public_channel_ids()),
            len(cluster.public_entities),
        )

    # -- proposal -----------------------------------------------------------

    def _propose_cluster(
        self,
        *,
        installation_id: uuid.UUID,
        cluster: _Cluster,
        task_id: uuid.UUID | None,
        now: datetime,
    ) -> ProjectProposal | None:
        public = cluster.public_entities
        dedupe_key = _dedupe_key(public)
        if self.proposals.has_recent_proposal(
            installation_id=installation_id, dedupe_key=dedupe_key, now=now
        ):
            return None

        named = self._name_cluster(public, task_id=task_id)
        if named is None:
            return None
        title, summary, confidence = named
        if confidence < self.min_confidence:
            return None

        private = cluster.private_entities
        public_evidence = [
            {"entity_id": str(e.id), "label": e.display_name or e.canonical_key}
            for e in public
        ]
        private_evidence = [
            {"entity_id": str(e.id), "scope_type": e.visibility_scope_type}
            for e in private
        ]
        return self.proposals.create_proposal(
            installation_id=installation_id,
            slug=project_slug(title),
            title=title,
            public_summary=summary,
            proposed_channel_ids=cluster.public_channel_ids(),
            proposed_entity_ids=[e.id for e in public],
            public_evidence=public_evidence,
            private_evidence=private_evidence,
            dedupe_key=dedupe_key,
            confidence_score=confidence,
            confidence_reason="Recurring co-occurring work cluster (HIG-276).",
            cooldown_until=now + self.repropose_cooldown,
        )

    def _name_cluster(
        self,
        public: Sequence[KnowledgeGraphEntity],
        *,
        task_id: uuid.UUID | None,
    ) -> tuple[str, str, Decimal] | None:
        labels = [e.display_name or e.canonical_key for e in public]
        if self.llm is None or task_id is None:
            # Deterministic fallback: name from the strongest signal.
            title = labels[0][:60]
            summary = "Recurring work across this workspace: " + ", ".join(labels[:6])
            return (title, summary, self.min_confidence)
        try:
            completion = self.llm.complete(
                task_id=task_id,
                messages=(
                    ChatMessage(role="system", content=_NAMER_SYSTEM_PROMPT),
                    ChatMessage(
                        role="user",
                        content=json.dumps(
                            {"recurring_items": labels[:20]},
                            separators=(",", ":"),
                            sort_keys=True,
                        ),
                    ),
                ),
                response_format=_NAMER_RESPONSE_FORMAT,
                prompt_name=_NAME_PROMPT,
            )
            parsed = json.loads(completion.content or "{}")
        except (json.JSONDecodeError, ValueError) as exc:
            logger.info(
                "project name extraction failed error_type=%s error=%s",
                type(exc).__name__,
                exc,
            )
            return None
        if not isinstance(parsed, dict):
            return None
        title = str(parsed.get("title") or "").strip()
        summary = str(parsed.get("summary") or "").strip()
        confidence = _coerce_confidence(parsed.get("confidence"))
        if not title or not summary or confidence is None:
            return None
        return (title[:80], summary, confidence)


def _dedupe_key(public: Sequence[KnowledgeGraphEntity]) -> str:
    keys = sorted(e.canonical_key for e in public)
    return "project_inference:" + "|".join(keys)


def _coerce_confidence(value: object) -> Decimal | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        candidate = Decimal(str(value))
    elif isinstance(value, str):
        try:
            candidate = Decimal(value)
        except (ValueError, ArithmeticError):
            return None
    else:
        return None
    if candidate < 0 or candidate > 1:
        return None
    return candidate


_NAMER_SYSTEM_PROMPT = (
    "You name an emerging project. Given a list of recurring work items a team "
    'keeps returning to, return strict JSON {"title": str, "summary": str, '
    '"confidence": number 0-1}. The title is a short human project name. The '
    "summary is one sentence describing the project, using ONLY the provided "
    "items (do not invent specifics). Confidence reflects how clearly these "
    "items form one coherent project."
)

_NAMER_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "project_name",
        "schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "summary": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "required": ["title", "summary", "confidence"],
            "additionalProperties": False,
        },
    },
}
