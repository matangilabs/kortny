"""Implicit project inference (HIG-276 increment 2).

A consolidator pass that notices when people keep circling the same
topic/workflow/deadline — a cluster of recurring, co-occurring graph entities —
and learns it as a project ON ITS OWN. No proposal gate: the brain forms the
hypothesis, uses it immediately (an inferred `project` hub at modest confidence),
and self-corrects via reinforcement (re-detection raises confidence/strength)
and the aging pass (a topic that goes quiet is retired). Occasional human
confirmation to upgrade/correct is a separate, non-blocking follow-up.

Why this is safe without a confirm gate: Increment 1's audience barrier is
structural — an inferred project can still only ever surface PUBLIC-channel
facts into a public reply, regardless of confirmation. A wrong grouping degrades
answer quality (graceful, evidence-cited), it does not leak.

Deterministic vs LLM split: deterministic code does cluster assembly, scoring,
privacy filtering, identity/dedupe, and persistence; the LLM only NAMES the
cluster (its display name). Inferred hubs/edges draw ONLY on public entities;
private entities never widen a hub or get named.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.orm import Session

from kortny.db.models import KnowledgeGraphEntity, KnowledgeGraphEvidence
from kortny.knowledge_graph.projects import ProjectGraphService
from kortny.knowledge_graph.scopes import SCOPE_CHANNEL, SCOPE_WORKSPACE
from kortny.knowledge_graph.service import EvidenceInput
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
DEFAULT_RECENT_WINDOW = timedelta(days=30)
# Inferred hubs start modest and earn confidence by re-detection.
_INFERRED_CONFIDENCE = Decimal("0.600")
# Number of strongest entities whose identity defines the project (stable across
# small membership churn, so re-detection reinforces rather than duplicates).
_IDENTITY_ANCHORS = 3
_MAX_CANDIDATES = 400
_NAME_PROMPT = "kortny.project_inference_namer"


@dataclass(frozen=True)
class ProjectInferenceCounters:
    candidates: int = 0
    clusters: int = 0
    learned: int = 0
    reinforced: int = 0
    skipped_reason: str | None = None

    def to_payload(self) -> dict[str, object]:
        return {
            "candidates": self.candidates,
            "clusters": self.clusters,
            "learned": self.learned,
            "reinforced": self.reinforced,
            "skipped_reason": self.skipped_reason,
        }


@dataclass
class _Cluster:
    entities: list[KnowledgeGraphEntity] = field(default_factory=list)
    source_tasks: set[uuid.UUID] = field(default_factory=set)

    @property
    def public_entities(self) -> list[KnowledgeGraphEntity]:
        return [e for e in self.entities if e.visibility_scope_type in _PUBLIC_SCOPES]

    def public_channel_ids(self) -> list[str]:
        ids = [
            e.visibility_scope_id
            for e in self.public_entities
            if e.visibility_scope_type == SCOPE_CHANNEL and e.visibility_scope_id
        ]
        return list(dict.fromkeys(ids))

    def has_deadline(self) -> bool:
        return any(e.entity_type in _DEADLINE_TYPES for e in self.entities)

    def total_reinforcement(self) -> int:
        return sum(e.reinforcement_count for e in self.entities)


class ProjectInferencePass:
    """Infer + autonomously learn implicit projects from recurring clusters."""

    def __init__(
        self,
        session: Session,
        *,
        llm: LLMService | None = None,
        min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
        min_reinforcement: int = DEFAULT_MIN_REINFORCEMENT,
        min_channel_spread: int = DEFAULT_MIN_CHANNEL_SPREAD,
        recent_window: timedelta = DEFAULT_RECENT_WINDOW,
    ) -> None:
        self.session = session
        self.llm = llm
        self.min_cluster_size = min_cluster_size
        self.min_reinforcement = min_reinforcement
        self.min_channel_spread = min_channel_spread
        self.recent_window = recent_window
        self.projects = ProjectGraphService(session)

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
        qualifying = [c for c in clusters if self._qualifies(c)]
        if not qualifying:
            return ProjectInferenceCounters(
                candidates=len(entities),
                clusters=len(clusters),
                skipped_reason="no_qualifying_cluster",
            )

        learned = 0
        reinforced = 0
        for cluster in qualifying:
            created = self._learn_cluster(
                installation_id=installation_id, cluster=cluster, task_id=task_id
            )
            if created:
                learned += 1
            else:
                reinforced += 1
        return ProjectInferenceCounters(
            candidates=len(entities),
            clusters=len(clusters),
            learned=learned,
            reinforced=reinforced,
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

        owners_by_task: dict[uuid.UUID, list[uuid.UUID]] = {}
        for eid, task_ids in tasks_by_entity.items():
            for task_id in task_ids:
                owners_by_task.setdefault(task_id, []).append(eid)
        for owners in owners_by_task.values():
            for other in owners[1:]:
                union(owners[0], other)

        grouped: dict[uuid.UUID, _Cluster] = {}
        for eid, entity in by_id.items():
            cluster = grouped.setdefault(find(eid), _Cluster())
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
            if source_task_id is not None:
                out.setdefault(target_id, set()).add(source_task_id)
        return out

    # -- scoring ------------------------------------------------------------

    def _qualifies(self, cluster: _Cluster) -> bool:
        if len(cluster.public_entities) < self.min_cluster_size:
            return False
        spread = len(cluster.public_channel_ids())
        # Always require at least one public channel anchor: an inferred hub with
        # no channel is unreachable via projects_for_channel (channel retrieval),
        # so a deadline-only, channel-less cluster would create a dead project.
        if spread == 0:
            return False
        return spread >= self.min_channel_spread or cluster.has_deadline()

    # -- learning -----------------------------------------------------------

    def _learn_cluster(
        self,
        *,
        installation_id: uuid.UUID,
        cluster: _Cluster,
        task_id: uuid.UUID | None,
    ) -> bool:
        """Create or reinforce the inferred project hub for a cluster.

        Returns True if a new hub was learned, False if an existing one was
        reinforced.
        """

        public = cluster.public_entities
        slug = _cluster_identity_slug(public)
        title = self._name_cluster(public, task_id=task_id)
        source_task = next(iter(sorted(cluster.source_tasks)), None)
        evidence = EvidenceInput(
            source_type="agent_inferred",
            extracted_by="project_inference",
            source_task_id=source_task,
            raw_snippet=title,
        )
        result = self.projects.declare_project(
            installation_id=installation_id,
            name=title,
            slug=slug,
            channel_ids=cluster.public_channel_ids(),
            evidence=evidence,
            source_type="agent_inferred",
            lifecycle_state="active",
            confidence_score=_INFERRED_CONFIDENCE,
            confidence_reason="Recurring co-occurring work cluster (HIG-276).",
            reinforce=True,
        )
        self.projects.link_project_entities(
            installation_id=installation_id,
            project=result.project,
            entity_ids=[e.id for e in public],
            evidence=evidence,
            # Inferred membership must not masquerade as user-confirmed (HIG-276).
            source_type="agent_inferred",
            lifecycle_state="active",
            confidence_score=_INFERRED_CONFIDENCE,
            confidence_reason="Recurring co-occurring work cluster (HIG-276).",
        )
        return result.created

    def _name_cluster(
        self,
        public: Sequence[KnowledgeGraphEntity],
        *,
        task_id: uuid.UUID | None,
    ) -> str:
        labels = [e.display_name or e.canonical_key for e in public]
        if self.llm is None or task_id is None:
            return labels[0][:60]
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
            return labels[0][:60]
        title = (
            str(parsed.get("title") or "").strip() if isinstance(parsed, dict) else ""
        )
        return title[:80] if title else labels[0][:60]


def _cluster_identity_slug(public: Sequence[KnowledgeGraphEntity]) -> str:
    """Stable slug from the cluster's strongest anchors.

    Keying on the top-N reinforced entities (not the whole set) keeps a project's
    identity stable as members come and go, so re-detection reinforces the same
    hub instead of spawning duplicates.
    """

    anchors = sorted(
        public, key=lambda e: (e.reinforcement_count, e.canonical_key), reverse=True
    )[:_IDENTITY_ANCHORS]
    digest = hashlib.sha1(
        "|".join(sorted(e.canonical_key for e in anchors)).encode()
    ).hexdigest()[:12]
    return f"inferred-{digest}"


_NAMER_SYSTEM_PROMPT = (
    "You name an emerging project. Given a list of recurring work items a team "
    'keeps returning to, return strict JSON {"title": str}. The title is a '
    "short human project name derived ONLY from the items (do not invent "
    "specifics)."
)

_NAMER_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "project_name",
        "schema": {
            "type": "object",
            "properties": {"title": {"type": "string"}},
            "required": ["title"],
            "additionalProperties": False,
        },
    },
}
