"""Pass 1: episode -> knowledge promotion (Mem0-style update loop).

For each new task episode the consolidator retrieves the most similar existing
graph entities and confirmed facts, then asks the cheap LLM tier to arbitrate
one of ADD / UPDATE / INVALIDATE / NOOP per episode. Contradiction is temporal
invalidation — never deletion — and user-confirmed knowledge is never
invalidated by the model (conflicts are flagged instead).
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import TypeVar

from sqlalchemy import select
from sqlalchemy.orm import Session

from kortny.db.models import Episode, KnowledgeGraphEntity, Task, WorkspaceState
from kortny.embeddings import (
    FACT_EMBEDDING_KIND,
    KG_ENTITY_EMBEDDING_KIND,
    EmbeddingIndex,
    episode_embedding_text,
)
from kortny.knowledge_graph import EvidenceInput, GraphService, VisibilityScope
from kortny.llm import ChatMessage, LLMService
from kortny.tools.types import JsonObject

logger = logging.getLogger(__name__)
_RowT = TypeVar("_RowT")

CONSOLIDATOR_PROMOTION_PROMPT_NAME = "kortny.consolidator_promotion"
PROMOTION_RESPONSE_FORMAT: JsonObject = {"type": "json_object"}
PROMOTION_SOURCE_TYPE = "task_summary"
# Knowledge a user confirmed (directly or via the propose->confirm fact flow)
# always outranks generated knowledge; the model may never invalidate it.
PROTECTED_SOURCE_TYPES = frozenset(
    {"user_explicit", "user_confirmed", "workspace_state"}
)
ALLOWED_ENTITY_TYPES = frozenset(
    {
        "person",
        "channel",
        "project",
        "firm_fact",
        "artifact",
        "decision",
        "open_question",
        "commitment",
        "integration",
        "external_entity",
    }
)
ALLOWED_ACTIONS = frozenset({"ADD", "UPDATE", "INVALIDATE", "NOOP"})
DEFAULT_EPISODE_CAP = 50
DEFAULT_BATCH_SIZE = 10
RELATED_TOP_K = 5
MAX_SNIPPET_CHARS = 700
_KEY_RE = re.compile(r"[^a-z0-9_:.\-]+")


@dataclass(slots=True)
class PromotionCounters:
    """Counters from one promotion pass."""

    episodes_reviewed: int = 0
    promoted: int = 0
    updated: int = 0
    invalidated: int = 0
    noop: int = 0
    invalid_decisions: int = 0
    conflicts: list[dict[str, str]] = field(default_factory=list)
    # created_at of the last episode actually processed; the next run resumes
    # from here so a backlog larger than the per-run cap drains over time.
    anchor: str | None = None

    def to_payload(self) -> dict[str, object]:
        return {
            "episodes_reviewed": self.episodes_reviewed,
            "promoted": self.promoted,
            "updated": self.updated,
            "invalidated": self.invalidated,
            "noop": self.noop,
            "invalid_decisions": self.invalid_decisions,
            "conflicts": list(self.conflicts),
            "anchor": self.anchor,
        }


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    """One validated arbitration decision from the model."""

    episode_id: uuid.UUID
    action: str
    entity_id: uuid.UUID | None
    entity_type: str | None
    canonical_key: str | None
    display_name: str | None
    summary: str | None
    replacement_summary: str | None
    confidence: Decimal
    reason: str | None


class EpisodePromotionPass:
    """Promote recent episodes into the workspace knowledge graph."""

    def __init__(
        self,
        session: Session,
        *,
        graph: GraphService,
        llm: LLMService | None,
        embedding_index: EmbeddingIndex | None,
    ) -> None:
        self.session = session
        self.graph = graph
        self.llm = llm
        self.embedding_index = embedding_index

    def run(
        self,
        *,
        installation_id: uuid.UUID,
        task: Task,
        since: datetime | None,
        now: datetime | None = None,
        episode_cap: int = DEFAULT_EPISODE_CAP,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> PromotionCounters:
        counters = PromotionCounters(
            anchor=since.isoformat() if since is not None else None
        )
        if self.llm is None:
            return counters
        effective_now = now or datetime.now(UTC)
        episodes = self._new_episodes(
            installation_id=installation_id,
            since=since,
            cap=episode_cap,
        )
        if not episodes:
            return counters
        counters.anchor = episodes[-1].created_at.isoformat()

        for start in range(0, len(episodes), max(1, batch_size)):
            batch = episodes[start : start + max(1, batch_size)]
            counters.episodes_reviewed += len(batch)
            completion = self.llm.complete(
                task_id=task.id,
                messages=self._batch_messages(installation_id, batch),
                response_format=PROMOTION_RESPONSE_FORMAT,
                prompt_name=CONSOLIDATOR_PROMOTION_PROMPT_NAME,
            )
            episodes_by_id = {episode.id: episode for episode in batch}
            for decision in parse_promotion_decisions(completion.content):
                episode = episodes_by_id.get(decision.episode_id)
                if episode is None:
                    counters.invalid_decisions += 1
                    continue
                self._apply_decision(
                    decision,
                    episode=episode,
                    installation_id=installation_id,
                    now=effective_now,
                    counters=counters,
                )
        self.session.flush()
        return counters

    def _new_episodes(
        self,
        *,
        installation_id: uuid.UUID,
        since: datetime | None,
        cap: int,
    ) -> list[Episode]:
        predicates = [Episode.installation_id == installation_id]
        if since is not None:
            predicates.append(Episode.created_at > since)
        return list(
            self.session.scalars(
                select(Episode)
                .where(*predicates)
                .order_by(Episode.created_at, Episode.id)
                .limit(cap)
            )
        )

    def _batch_messages(
        self,
        installation_id: uuid.UUID,
        batch: list[Episode],
    ) -> tuple[ChatMessage, ...]:
        episode_payloads = []
        for episode in batch:
            episode_payloads.append(
                {
                    "episode_id": str(episode.id),
                    "channel_id": episode.channel_id,
                    "outcome": episode.outcome,
                    "summary": episode_embedding_text(episode),
                    "related_entities": self._related_entities(
                        installation_id, episode
                    ),
                    "related_confirmed_facts": self._related_facts(
                        installation_id, episode
                    ),
                }
            )
        return (
            ChatMessage(
                role="system",
                content=(
                    "You are Kortny's memory consolidator. Kortny is an AI "
                    "coworker in Slack. For each completed task episode decide "
                    "whether it contains durable workspace knowledge worth "
                    "keeping. Compare against the related existing memories "
                    "provided. "
                    "Return only the JSON object — no prose, markdown, or "
                    'comments. Schema: {"decisions":['
                    '{"episode_id":"uuid","action":"ADD|UPDATE|INVALIDATE|NOOP",'
                    '"entity_id":"uuid of an existing related entity for UPDATE '
                    'or INVALIDATE, else null",'
                    '"entity_type":"person|channel|project|firm_fact|artifact|'
                    'decision|open_question|commitment|integration|external_entity",'
                    '"canonical_key":"short_snake_case_key for ADD",'
                    '"display_name":"short human name",'
                    '"summary":"one-sentence durable knowledge statement",'
                    '"replacement_summary":"for INVALIDATE: the corrected '
                    'statement, else null",'
                    '"confidence":0.0,"reason":"why"}]}. '
                    "Rules: ADD only genuinely new durable knowledge (not "
                    "one-off chitchat). UPDATE when an existing entity should "
                    "absorb/refresh this knowledge. INVALIDATE only when the "
                    "episode clearly contradicts an existing entity; never "
                    "invalidate related_confirmed_facts — they are "
                    "user-confirmed and outrank task evidence; use NOOP and "
                    "explain the conflict in reason instead. NOOP for "
                    "everything else. Be conservative: most episodes are NOOP. "
                    "Extract ONLY what the episode text supports; never invent "
                    "entity ids, canonical keys, or facts not present in the "
                    "input. If no episode warrants a durable memory, return all "
                    "actions as NOOP. Confidence must be 0.0..1.0. "
                    "Examples: "
                    '{"episodes":[{"episode_id":"eid-1","outcome":"success","summary":"The team uses Linear for issue tracking.","related_entities":[],"related_confirmed_facts":[]}]} '
                    '-> {"decisions":[{"episode_id":"eid-1","action":"ADD","entity_id":null,"entity_type":"integration","canonical_key":"integration:linear","display_name":"Linear","summary":"Team uses Linear for issue tracking.","replacement_summary":null,"confidence":0.80,"reason":"Durable tool fact from episode."}]} '
                    '{"episodes":[{"episode_id":"eid-2","outcome":"success","summary":"User asked about the weather.","related_entities":[],"related_confirmed_facts":[]}]} '
                    '-> {"decisions":[{"episode_id":"eid-2","action":"NOOP","entity_id":null,"entity_type":null,"canonical_key":null,"display_name":null,"summary":null,"replacement_summary":null,"confidence":0.0,"reason":"One-off question, not durable workspace knowledge."}]} '
                    "Ground every field in the input; abstain when unsupported."
                ),
            ),
            ChatMessage(
                role="user",
                content=json.dumps(
                    {"episodes": episode_payloads},
                    sort_keys=True,
                    default=str,
                    separators=(",", ":"),
                ),
            ),
        )

    def _related_entities(
        self,
        installation_id: uuid.UUID,
        episode: Episode,
    ) -> list[dict[str, object]]:
        pool = list(
            self.session.scalars(
                select(KnowledgeGraphEntity)
                .where(
                    KnowledgeGraphEntity.installation_id == installation_id,
                    KnowledgeGraphEntity.is_current.is_(True),
                    KnowledgeGraphEntity.expired_at.is_(None),
                    KnowledgeGraphEntity.system_expired_at.is_(None),
                    KnowledgeGraphEntity.invalid_at.is_(None),
                    KnowledgeGraphEntity.lifecycle_state.in_(
                        ("candidate", "active", "confirmed")
                    ),
                )
                .order_by(KnowledgeGraphEntity.updated_at.desc())
                .limit(200)
            )
        )
        selected = self._top_similar(
            KG_ENTITY_EMBEDDING_KIND,
            episode_embedding_text(episode),
            {str(entity.id): entity for entity in pool},
        )
        return [
            {
                "entity_id": str(entity.id),
                "entity_type": entity.entity_type,
                "canonical_key": entity.canonical_key,
                "display_name": entity.display_name,
                "source_type": entity.source_type,
                "lifecycle_state": entity.lifecycle_state,
                "summary": _entity_summary(entity),
                "user_confirmed": entity.source_type in PROTECTED_SOURCE_TYPES,
            }
            for entity in selected
        ]

    def _related_facts(
        self,
        installation_id: uuid.UUID,
        episode: Episode,
    ) -> list[dict[str, object]]:
        pool = list(
            self.session.scalars(
                select(WorkspaceState)
                .where(
                    WorkspaceState.installation_id == installation_id,
                    WorkspaceState.status == "active",
                )
                .order_by(WorkspaceState.updated_at.desc())
                .limit(200)
            )
        )
        selected = self._top_similar(
            FACT_EMBEDDING_KIND,
            episode_embedding_text(episode),
            {str(state.id): state for state in pool},
        )
        return [
            {
                "fact_id": str(state.id),
                "key": state.key,
                "scope_type": state.scope_type,
                "scope_id": state.scope_id,
                "value": state.value_text or json.dumps(state.value_json, default=str),
            }
            for state in selected
        ]

    def _top_similar(
        self,
        kind: str,
        query_text: str,
        pool: dict[str, _RowT],
    ) -> list[_RowT]:
        if not pool:
            return []
        if self.embedding_index is None:
            return list(pool.values())[:RELATED_TOP_K]
        ranked = self.embedding_index.rank(
            kind,
            query_text,
            list(pool),
            top_k=RELATED_TOP_K,
        )
        if ranked is None:
            return list(pool.values())[:RELATED_TOP_K]
        return [pool[ref_key] for ref_key, _ in ranked if ref_key in pool]

    def _apply_decision(
        self,
        decision: PromotionDecision,
        *,
        episode: Episode,
        installation_id: uuid.UUID,
        now: datetime,
        counters: PromotionCounters,
    ) -> None:
        if decision.action == "NOOP":
            counters.noop += 1
            return
        if decision.action == "ADD":
            self._apply_add(
                decision,
                episode=episode,
                installation_id=installation_id,
                now=now,
                counters=counters,
            )
            return

        entity = self._target_entity(installation_id, decision.entity_id)
        if entity is None:
            counters.invalid_decisions += 1
            return
        if decision.action == "UPDATE":
            self._apply_update(
                decision, entity=entity, episode=episode, now=now, counters=counters
            )
            return
        # INVALIDATE
        if entity.source_type in PROTECTED_SOURCE_TYPES:
            counters.conflicts.append(
                {
                    "entity_id": str(entity.id),
                    "canonical_key": entity.canonical_key,
                    "episode_id": str(episode.id),
                    "reason": decision.reason
                    or "Episode contradicts a user-confirmed entity.",
                }
            )
            return
        self.graph.invalidate_entity(entity, now=now, reason=decision.reason)
        counters.invalidated += 1
        if decision.replacement_summary:
            successor = self.graph.create_entity(
                installation_id=installation_id,
                entity_type=entity.entity_type,
                canonical_key=entity.canonical_key,
                visibility_scope=VisibilityScope(
                    entity.visibility_scope_type, entity.visibility_scope_id
                ),
                source_type=PROMOTION_SOURCE_TYPE,
                display_name=decision.display_name or entity.display_name,
                attrs_json={"summary": decision.replacement_summary},
                lifecycle_state="active",
                confidence_score=decision.confidence,
                confidence_reason=decision.reason,
                evidence=self._evidence(episode, decision),
            )
            successor.valid_at = now

    def _apply_add(
        self,
        decision: PromotionDecision,
        *,
        episode: Episode,
        installation_id: uuid.UUID,
        now: datetime,
        counters: PromotionCounters,
    ) -> None:
        if decision.entity_type is None or decision.canonical_key is None:
            counters.invalid_decisions += 1
            return
        existing = self.session.scalar(
            select(KnowledgeGraphEntity).where(
                KnowledgeGraphEntity.installation_id == installation_id,
                KnowledgeGraphEntity.canonical_key == decision.canonical_key,
                KnowledgeGraphEntity.is_current.is_(True),
                KnowledgeGraphEntity.expired_at.is_(None),
            )
        )
        if existing is not None:
            self._apply_update(
                decision, entity=existing, episode=episode, now=now, counters=counters
            )
            return
        entity = self.graph.create_entity(
            installation_id=installation_id,
            entity_type=decision.entity_type,
            canonical_key=decision.canonical_key,
            visibility_scope=_episode_scope(episode),
            source_type=PROMOTION_SOURCE_TYPE,
            display_name=decision.display_name,
            attrs_json={"summary": decision.summary or ""},
            lifecycle_state="active",
            confidence_score=decision.confidence,
            confidence_reason=decision.reason,
            evidence=self._evidence(episode, decision),
        )
        entity.valid_at = now
        counters.promoted += 1

    def _apply_update(
        self,
        decision: PromotionDecision,
        *,
        entity: KnowledgeGraphEntity,
        episode: Episode,
        now: datetime,
        counters: PromotionCounters,
    ) -> None:
        if decision.summary:
            attrs = dict(entity.attrs_json or {})
            attrs["summary"] = decision.summary
            entity.attrs_json = attrs
        if decision.display_name and not entity.display_name:
            entity.display_name = decision.display_name
        entity.last_reinforced_at = now
        entity.reinforcement_count = (entity.reinforcement_count or 0) + 1
        if entity.lifecycle_state == "stale":
            entity.lifecycle_state = "active"
        entity.updated_at = now
        self.graph.add_evidence(
            installation_id=entity.installation_id,
            target_kind="entity",
            target_id=entity.id,
            evidence=self._evidence(episode, decision),
        )
        self.graph.ensure_entity_embedding(entity)
        counters.updated += 1

    def _target_entity(
        self,
        installation_id: uuid.UUID,
        entity_id: uuid.UUID | None,
    ) -> KnowledgeGraphEntity | None:
        if entity_id is None:
            return None
        return self.session.scalar(
            select(KnowledgeGraphEntity).where(
                KnowledgeGraphEntity.installation_id == installation_id,
                KnowledgeGraphEntity.id == entity_id,
                KnowledgeGraphEntity.is_current.is_(True),
            )
        )

    def _evidence(
        self,
        episode: Episode,
        decision: PromotionDecision,
    ) -> EvidenceInput:
        return EvidenceInput(
            source_type=PROMOTION_SOURCE_TYPE,
            extracted_by=CONSOLIDATOR_PROMOTION_PROMPT_NAME,
            source_task_id=episode.task_id,
            source_episode_id=episode.id,
            source_slack_channel_id=episode.channel_id,
            raw_snippet=(episode.summary or "")[:MAX_SNIPPET_CHARS],
            confidence_score=decision.confidence,
            confidence_reason=decision.reason,
        )


def parse_promotion_decisions(content: str | None) -> tuple[PromotionDecision, ...]:
    """Parse and validate the JSON arbitration output from the model."""

    if not content:
        return ()
    try:
        payload = json.loads(_extract_json_object(content))
    except (json.JSONDecodeError, ValueError):
        return ()
    if not isinstance(payload, dict):
        return ()
    raw_decisions = payload.get("decisions")
    if not isinstance(raw_decisions, list):
        return ()
    decisions: list[PromotionDecision] = []
    for raw in raw_decisions:
        decision = _decision_from_payload(raw)
        if decision is not None:
            decisions.append(decision)
    return tuple(decisions)


def _decision_from_payload(value: object) -> PromotionDecision | None:
    if not isinstance(value, dict):
        return None
    episode_id = _optional_uuid(value.get("episode_id"))
    action = value.get("action")
    if episode_id is None or action not in ALLOWED_ACTIONS:
        return None
    entity_type = value.get("entity_type")
    if entity_type not in ALLOWED_ENTITY_TYPES:
        entity_type = None
    if action == "ADD" and entity_type is None:
        return None
    canonical_key = _normalized_key(value.get("canonical_key"))
    if action == "ADD" and canonical_key is None:
        return None
    return PromotionDecision(
        episode_id=episode_id,
        action=str(action),
        entity_id=_optional_uuid(value.get("entity_id")),
        entity_type=entity_type,
        canonical_key=canonical_key,
        display_name=_optional_text(value.get("display_name"), max_chars=140),
        summary=_optional_text(value.get("summary"), max_chars=1000),
        replacement_summary=_optional_text(
            value.get("replacement_summary"), max_chars=1000
        ),
        confidence=_confidence(value.get("confidence")),
        reason=_optional_text(value.get("reason"), max_chars=500),
    )


def _entity_summary(entity: KnowledgeGraphEntity) -> str | None:
    attrs = entity.attrs_json if isinstance(entity.attrs_json, dict) else {}
    summary = attrs.get("summary")
    if isinstance(summary, str) and summary.strip():
        return summary.strip()[:500]
    return None


def _episode_scope(episode: Episode) -> VisibilityScope:
    channel_id = episode.channel_id or ""
    if channel_id.startswith("D"):
        return VisibilityScope.dm(channel_id)
    if channel_id.startswith("G"):
        return VisibilityScope.private_channel(channel_id)
    if channel_id:
        return VisibilityScope.channel(channel_id)
    return VisibilityScope.workspace()


def _optional_uuid(value: object) -> uuid.UUID | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return uuid.UUID(value.strip())
    except ValueError:
        return None


def _normalized_key(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    key = _KEY_RE.sub("_", value.strip().lower()).strip("_")
    return key[:200] or None


def _optional_text(value: object, *, max_chars: int) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.split()).strip()
    if not text:
        return None
    return text[:max_chars].strip()


def _confidence(value: object) -> Decimal:
    try:
        score = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return Decimal("0.500")
    if score < 0:
        return Decimal("0.000")
    if score > 1:
        return Decimal("1.000")
    return score.quantize(Decimal("0.001"))


def _extract_json_object(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("No JSON object found")
    return stripped[start : end + 1]
