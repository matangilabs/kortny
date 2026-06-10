"""Extract low-risk graph context from completed task summaries."""

from __future__ import annotations

import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from kortny.db.models import (
    KnowledgeGraphEdge,
    KnowledgeGraphEntity,
    KnowledgeGraphEvidence,
    SlackChannelMembership,
    SlackIdentity,
    Task,
    TaskEvent,
    TaskEventType,
)
from kortny.knowledge_graph.provenance import with_provenance_attrs
from kortny.knowledge_graph.scopes import VisibilityScope
from kortny.knowledge_graph.service import EvidenceInput, GraphService

KG_TASK_SUMMARY_PROJECTED_MESSAGE = "kg_task_summary_projected"
TASK_SUMMARY_SOURCE_TYPE = "task_summary"
TASK_SUMMARY_EXTRACTOR = "kortny.task_summary_graph_extractor"
TASK_SUMMARY_PROJECTION_KIND = "task_summary_projection"
AUTO_REVIEW_STATUS = "auto"
NEEDS_REVIEW_STATUS = "needs_review"
MAX_TASK_SUMMARY_CANDIDATES = 8
MAX_EVIDENCE_SNIPPET_CHARS = 700
MIN_SUMMARY_CHARS = 80

SENSITIVE_TASK_SUMMARY_RE = re.compile(
    r"\b("
    r"api[-_ ]?key|credential|password|secret|token|"
    r"salary|compensation|payroll|hr|human resources|"
    r"medical|health|diagnosis|legal|lawsuit|attorney|"
    r"fired|termination|underperforming|disciplinary|"
    r"confidential|private|destructive|delete|purge|drop table"
    r")\b",
    re.I,
)
PERSON_RESPONSIBILITY_RE = re.compile(
    r"(<@[A-Z0-9]+>|@[a-z0-9_.-]+|\b(owner|assigned to|responsible for|blocked by|waiting on)\b)",
    re.I,
)
GRAPH_SELF_QUERY_RE = re.compile(
    r"\b(what do you know|why do you believe|what have you learned|profile)\b.*\b(channel|workspace|graph|memory)\b",
    re.I,
)
FAILURE_SUMMARY_RE = re.compile(
    r"\b(something went wrong|could not|failed|error|blocked|try again soon)\b",
    re.I,
)
SECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("decision", re.compile(r"\b(decisions?|recommendations?|bottom line)\b", re.I)),
    (
        "open_question",
        re.compile(r"\b(open|unresolved|blockers?|questions?|gaps?|risks?)\b", re.I),
    ),
    (
        "commitment",
        re.compile(
            r"\b(next actions?|follow[- ]?ups?|todos?|workflows?|cadence)\b", re.I
        ),
    ),
)
KIND_SPECS = {
    "topic": ("firm_fact", "task_topic", "relates_to"),
    "decision": ("decision", "task_decision", "made_in"),
    "open_question": ("open_question", "task_open_question", "relates_to"),
    "commitment": ("commitment", "task_commitment", "relates_to"),
}
STOPWORDS = frozenset(
    {
        "about",
        "across",
        "after",
        "again",
        "also",
        "and",
        "any",
        "are",
        "can",
        "channel",
        "check",
        "compare",
        "current",
        "do",
        "few",
        "for",
        "from",
        "give",
        "have",
        "how",
        "into",
        "last",
        "latest",
        "me",
        "my",
        "of",
        "on",
        "open",
        "please",
        "project",
        "recent",
        "research",
        "summarize",
        "summary",
        "tell",
        "the",
        "there",
        "this",
        "to",
        "tools",
        "up",
        "what",
        "where",
        "why",
        "with",
        "you",
    }
)


@dataclass(frozen=True, slots=True)
class TaskSummaryGraphEntry:
    kind: str
    canonical_key: str
    display_name: str
    lifecycle_state: str
    review_status: str
    review_reason: str | None
    entity_id: str
    edge_id: str | None

    def to_payload(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "canonical_key": self.canonical_key,
            "display_name": self.display_name,
            "lifecycle_state": self.lifecycle_state,
            "review_status": self.review_status,
            "review_reason": self.review_reason,
            "entity_id": self.entity_id,
            "edge_id": self.edge_id,
        }


@dataclass(frozen=True, slots=True)
class TaskSummaryGraphProjectionResult:
    entity_count: int = 0
    edge_count: int = 0
    evidence_count: int = 0
    active_count: int = 0
    candidate_count: int = 0
    skipped_count: int = 0
    skipped_reasons: tuple[str, ...] = ()
    message_event_id: int | None = None
    entries: tuple[TaskSummaryGraphEntry, ...] = ()

    @property
    def projected_count(self) -> int:
        return self.active_count + self.candidate_count

    def to_payload(self) -> dict[str, object]:
        return {
            "message": KG_TASK_SUMMARY_PROJECTED_MESSAGE,
            "entity_count": self.entity_count,
            "edge_count": self.edge_count,
            "evidence_count": self.evidence_count,
            "active_count": self.active_count,
            "candidate_count": self.candidate_count,
            "projected_count": self.projected_count,
            "skipped_count": self.skipped_count,
            "skipped_reasons": list(self.skipped_reasons),
            "message_event_id": self.message_event_id,
            "entries": [entry.to_payload() for entry in self.entries],
            "source_type": TASK_SUMMARY_SOURCE_TYPE,
        }


@dataclass(frozen=True, slots=True)
class _Candidate:
    kind: str
    label: str
    source_text: str
    confidence_score: Decimal
    confidence_reason: str


@dataclass(frozen=True, slots=True)
class _ReviewDecision:
    lifecycle_state: str
    review_status: str
    review_reason: str | None


class TaskSummaryGraphExtractionService:
    """Project completed task output into scoped graph rows."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self.graph = GraphService(session)

    def project_task_summary(
        self,
        *,
        task: Task,
        result_summary: str | None,
    ) -> TaskSummaryGraphProjectionResult:
        events = _task_events(self.session, task)
        message_event = _latest_result_message_event(events)
        if message_event is None:
            return TaskSummaryGraphProjectionResult(
                skipped_count=1,
                skipped_reasons=("no_result_message",),
            )
        scope = _visibility_scope_for_task(self.session, task)
        if scope is None:
            return TaskSummaryGraphProjectionResult(
                skipped_count=1,
                skipped_reasons=("dm_or_user_surface_deferred",),
                message_event_id=message_event.id,
            )

        summary = (result_summary or "").strip()
        skip_reason = _summary_skip_reason(task, summary)
        if skip_reason is not None:
            return TaskSummaryGraphProjectionResult(
                skipped_count=1,
                skipped_reasons=(skip_reason,),
                message_event_id=message_event.id,
            )

        candidates = _extract_candidates(task, summary)
        if not candidates:
            return TaskSummaryGraphProjectionResult(
                skipped_count=1,
                skipped_reasons=("no_structured_candidates",),
                message_event_id=message_event.id,
            )

        now = datetime.now(UTC)
        channel_entity, channel_created, channel_evidence = self._upsert_channel_entity(
            task=task,
            scope=scope,
            message_event=message_event,
            now=now,
        )
        entity_count = int(channel_created)
        edge_count = 0
        evidence_count = channel_evidence
        active_count = 0
        candidate_count = 0
        entries: list[TaskSummaryGraphEntry] = []

        for candidate in candidates:
            review = _review_decision(candidate)
            entity_type, key_prefix, relationship = KIND_SPECS[candidate.kind]
            canonical_key = _canonical_key(
                key_prefix=key_prefix,
                channel_id=task.slack_channel_id,
                label=candidate.label,
            )
            entity, created_entity, entity_evidence = self._upsert_entity(
                task=task,
                canonical_key=canonical_key,
                entity_type=entity_type,
                display_name=candidate.label,
                attrs_json={
                    "kind": TASK_SUMMARY_PROJECTION_KIND,
                    "semantic_kind": candidate.kind,
                    "source_text": candidate.source_text,
                    "channel_id": task.slack_channel_id,
                    "review_status": review.review_status,
                    "review_reason": review.review_reason,
                },
                visibility_scope=scope,
                lifecycle_state=review.lifecycle_state,
                confidence_score=candidate.confidence_score,
                confidence_reason=candidate.confidence_reason,
                evidence=_candidate_evidence(
                    task=task,
                    message_event=message_event,
                    candidate=candidate,
                ),
                now=now,
            )
            entity_count += int(created_entity)
            evidence_count += entity_evidence

            edge, created_edge, edge_evidence = self._upsert_edge(
                task=task,
                source_entity_id=channel_entity.id,
                target_entity_id=entity.id,
                relationship_type=relationship,
                attrs_json={
                    "kind": TASK_SUMMARY_PROJECTION_KIND,
                    "semantic_kind": candidate.kind,
                    "source_text": candidate.source_text,
                    "review_status": review.review_status,
                    "review_reason": review.review_reason,
                    "channel_id": task.slack_channel_id,
                },
                visibility_scope=scope,
                lifecycle_state=review.lifecycle_state,
                confidence_score=candidate.confidence_score,
                confidence_reason=candidate.confidence_reason,
                evidence=_candidate_evidence(
                    task=task,
                    message_event=message_event,
                    candidate=candidate,
                ),
                now=now,
            )
            edge_count += int(created_edge)
            evidence_count += edge_evidence
            if review.lifecycle_state == "candidate":
                candidate_count += 1
            else:
                active_count += 1
            entries.append(
                TaskSummaryGraphEntry(
                    kind=candidate.kind,
                    canonical_key=entity.canonical_key,
                    display_name=entity.display_name or candidate.label,
                    lifecycle_state=entity.lifecycle_state,
                    review_status=review.review_status,
                    review_reason=review.review_reason,
                    entity_id=str(entity.id),
                    edge_id=str(edge.id) if edge is not None else None,
                )
            )

        self.session.flush()
        return TaskSummaryGraphProjectionResult(
            entity_count=entity_count,
            edge_count=edge_count,
            evidence_count=evidence_count,
            active_count=active_count,
            candidate_count=candidate_count,
            message_event_id=message_event.id,
            entries=tuple(entries),
        )

    def _upsert_channel_entity(
        self,
        *,
        task: Task,
        scope: VisibilityScope,
        message_event: TaskEvent,
        now: datetime,
    ) -> tuple[KnowledgeGraphEntity, bool, int]:
        canonical_key = f"slack_channel:{task.slack_channel_id}"
        existing = _current_entity_by_key(self.session, task, canonical_key)
        evidence = EvidenceInput(
            source_type=TASK_SUMMARY_SOURCE_TYPE,
            extracted_by=TASK_SUMMARY_EXTRACTOR,
            source_task_id=task.id,
            source_task_event_id=message_event.id,
            source_slack_channel_id=task.slack_channel_id,
            source_slack_message_ts=_string_or_none(
                message_event.payload.get("message_ts")
            ),
            raw_snippet=f"Task completed in Slack channel {task.slack_channel_id}.",
            confidence_score=Decimal("0.800"),
            confidence_reason="Slack task channel from a completed task.",
        )
        attrs = {
            "kind": TASK_SUMMARY_PROJECTION_KIND,
            "semantic_kind": "channel_surface",
            "channel_id": task.slack_channel_id,
            "review_status": AUTO_REVIEW_STATUS,
        }
        if existing is not None:
            evidence_count = _reinforce_entity(
                session=self.session,
                graph=self.graph,
                entity=existing,
                task=task,
                display_name=_channel_display_name(self.session, task),
                attrs_json=attrs,
                visibility_scope=scope,
                lifecycle_state="active",
                confidence_score=Decimal("0.800"),
                confidence_reason="Slack task channel from a completed task.",
                evidence=evidence,
                now=now,
            )
            return existing, False, evidence_count
        entity = self.graph.create_entity(
            installation_id=task.installation_id,
            entity_type="channel",
            canonical_key=canonical_key,
            display_name=_channel_display_name(self.session, task),
            external_ref_type="slack_channel",
            external_ref_id=task.slack_channel_id,
            attrs_json=attrs,
            visibility_scope=scope,
            source_type=TASK_SUMMARY_SOURCE_TYPE,
            lifecycle_state="active",
            confidence_score=Decimal("0.800"),
            confidence_reason="Slack task channel from a completed task.",
            freshness_window_days=30,
            evidence=evidence,
        )
        return entity, True, 1

    def _upsert_entity(
        self,
        *,
        task: Task,
        canonical_key: str,
        entity_type: str,
        display_name: str,
        attrs_json: dict[str, Any],
        visibility_scope: VisibilityScope,
        lifecycle_state: str,
        confidence_score: Decimal,
        confidence_reason: str,
        evidence: EvidenceInput,
        now: datetime,
    ) -> tuple[KnowledgeGraphEntity, bool, int]:
        existing = _current_entity_by_key(self.session, task, canonical_key)
        if existing is not None:
            evidence_count = _reinforce_entity(
                session=self.session,
                graph=self.graph,
                entity=existing,
                task=task,
                display_name=display_name,
                attrs_json=attrs_json,
                visibility_scope=visibility_scope,
                lifecycle_state=lifecycle_state,
                confidence_score=confidence_score,
                confidence_reason=confidence_reason,
                evidence=evidence,
                now=now,
            )
            return existing, False, evidence_count
        entity = self.graph.create_entity(
            installation_id=task.installation_id,
            entity_type=entity_type,
            canonical_key=canonical_key,
            display_name=display_name,
            attrs_json=attrs_json,
            visibility_scope=visibility_scope,
            source_type=TASK_SUMMARY_SOURCE_TYPE,
            lifecycle_state=lifecycle_state,
            confidence_score=confidence_score,
            confidence_reason=confidence_reason,
            freshness_window_days=30,
            evidence=evidence,
        )
        return entity, True, 1

    def _upsert_edge(
        self,
        *,
        task: Task,
        source_entity_id: uuid.UUID,
        target_entity_id: uuid.UUID,
        relationship_type: str,
        attrs_json: dict[str, Any],
        visibility_scope: VisibilityScope,
        lifecycle_state: str,
        confidence_score: Decimal,
        confidence_reason: str,
        evidence: EvidenceInput,
        now: datetime,
    ) -> tuple[KnowledgeGraphEdge, bool, int]:
        existing = _current_edge(
            self.session,
            task=task,
            source_entity_id=source_entity_id,
            target_entity_id=target_entity_id,
            relationship_type=relationship_type,
        )
        if existing is not None:
            evidence_count = _reinforce_edge(
                session=self.session,
                graph=self.graph,
                edge=existing,
                task=task,
                attrs_json=attrs_json,
                visibility_scope=visibility_scope,
                lifecycle_state=lifecycle_state,
                confidence_score=confidence_score,
                confidence_reason=confidence_reason,
                evidence=evidence,
                now=now,
            )
            return existing, False, evidence_count
        edge = self.graph.create_edge(
            installation_id=task.installation_id,
            source_entity_id=source_entity_id,
            target_entity_id=target_entity_id,
            relationship_type=relationship_type,
            visibility_scope=visibility_scope,
            source_type=TASK_SUMMARY_SOURCE_TYPE,
            attrs_json=attrs_json,
            lifecycle_state=lifecycle_state,
            confidence_score=confidence_score,
            confidence_reason=confidence_reason,
            freshness_window_days=30,
            evidence=evidence,
        )
        return edge, True, 1


def _task_events(session: Session, task: Task) -> tuple[TaskEvent, ...]:
    return tuple(
        session.scalars(
            select(TaskEvent)
            .where(TaskEvent.task_id == task.id)
            .order_by(TaskEvent.seq)
        )
    )


def _latest_result_message_event(events: Sequence[TaskEvent]) -> TaskEvent | None:
    for event in reversed(events):
        if (
            event.type is TaskEventType.message_posted
            and event.payload.get("purpose") == "result"
        ):
            return event
    return None


def _visibility_scope_for_task(session: Session, task: Task) -> VisibilityScope | None:
    channel_id = task.slack_channel_id
    if channel_id.startswith("D"):
        return None
    if channel_id.startswith("G") or _is_private_channel(session, task):
        return VisibilityScope.private_channel(channel_id)
    return VisibilityScope.channel(channel_id)


def _is_private_channel(session: Session, task: Task) -> bool:
    membership = session.scalar(
        select(SlackChannelMembership).where(
            SlackChannelMembership.installation_id == task.installation_id,
            SlackChannelMembership.channel_id == task.slack_channel_id,
        )
    )
    channel_type = (membership.channel_type or "").lower() if membership else ""
    if channel_type in {"group", "private_channel", "private"}:
        return True
    identity = session.scalar(
        select(SlackIdentity).where(
            SlackIdentity.installation_id == task.installation_id,
            SlackIdentity.kind == "channel",
            SlackIdentity.slack_id == task.slack_channel_id,
        )
    )
    return bool(identity and identity.is_private)


def _summary_skip_reason(task: Task, summary: str) -> str | None:
    if len(summary) < MIN_SUMMARY_CHARS:
        return "summary_too_short"
    if GRAPH_SELF_QUERY_RE.search(task.input):
        return "graph_self_query"
    if FAILURE_SUMMARY_RE.search(summary):
        return "failure_or_blocker_summary"
    return None


def _extract_candidates(task: Task, summary: str) -> tuple[_Candidate, ...]:
    candidates: list[_Candidate] = []
    topic = _topic_from_task(task.input)
    if topic is not None:
        candidates.append(
            _Candidate(
                kind="topic",
                label=topic,
                source_text=task.input,
                confidence_score=Decimal("0.580"),
                confidence_reason="Topic inferred from the user request for a completed task.",
            )
        )

    current_section: str | None = None
    for raw_line in summary.splitlines():
        line = _clean_line(raw_line)
        if not line:
            continue
        section = _section_kind(line)
        if section is not None and not _looks_like_bullet(raw_line):
            current_section = section
            continue
        kind = _candidate_kind(line, current_section=current_section)
        if kind is None:
            continue
        label = _candidate_label(line, kind=kind)
        if label is None:
            continue
        candidates.append(
            _Candidate(
                kind=kind,
                label=label,
                source_text=line,
                confidence_score=Decimal("0.640"),
                confidence_reason="Extracted from a structured completed task summary.",
            )
        )
        if len(candidates) >= MAX_TASK_SUMMARY_CANDIDATES:
            break

    output: list[_Candidate] = []
    seen: set[tuple[str, str]] = set()
    for candidate in candidates:
        key = (candidate.kind, _slug(candidate.label))
        if key in seen:
            continue
        seen.add(key)
        output.append(candidate)
    return tuple(output[:MAX_TASK_SUMMARY_CANDIDATES])


def _section_kind(line: str) -> str | None:
    stripped = line.strip(" :").lower()
    if len(stripped.split()) > 8:
        return None
    for kind, pattern in SECTION_PATTERNS:
        if pattern.search(stripped):
            return kind
    return None


def _candidate_kind(line: str, *, current_section: str | None) -> str | None:
    lowered = line.lower()
    if current_section is not None and _looks_substantive(line):
        return current_section
    if re.search(
        r"\b(decided|decision|recommend|recommended|default|agreed|accepted|use)\b",
        lowered,
    ):
        return "decision"
    if re.search(
        r"\b(unresolved|open question|open item|needs owner|follow up|blocked|unknown|gap)\b",
        lowered,
    ):
        return "open_question"
    if re.search(
        r"\b(workflow|cadence|schedule|recurring|next action|todo)\b", lowered
    ):
        return "commitment"
    return None


def _candidate_label(line: str, *, kind: str) -> str | None:
    cleaned = re.sub(
        r"^(decision|recommendation|open question|open item|unresolved|next action|todo)\s*[:\-]\s*",
        "",
        line.strip(),
        flags=re.I,
    )
    cleaned = _shorten(cleaned.rstrip("."), 140)
    if len(cleaned) < 8:
        return None
    if kind == "open_question" and cleaned.lower().startswith("no "):
        return None
    return cleaned


def _topic_from_task(value: str) -> str | None:
    words = [
        word
        for word in re.findall(r"[A-Za-z][A-Za-z0-9+\-.]{1,}", value)
        if word.lower() not in STOPWORDS
    ]
    if len(words) < 2:
        return None
    title_words = words[:6]
    topic = " ".join(title_words)
    if len(topic) < 8:
        return None
    return topic[:96]


def _review_decision(candidate: _Candidate) -> _ReviewDecision:
    if candidate.confidence_score < Decimal("0.500"):
        return _ReviewDecision("candidate", NEEDS_REVIEW_STATUS, "low_confidence")
    if SENSITIVE_TASK_SUMMARY_RE.search(candidate.label):
        return _ReviewDecision(
            "candidate",
            NEEDS_REVIEW_STATUS,
            "sensitive_or_high_impact_language",
        )
    if candidate.kind in {
        "open_question",
        "commitment",
    } and PERSON_RESPONSIBILITY_RE.search(candidate.label):
        return _ReviewDecision(
            "candidate",
            NEEDS_REVIEW_STATUS,
            "person_responsibility_requires_review",
        )
    return _ReviewDecision("active", AUTO_REVIEW_STATUS, None)


def _candidate_evidence(
    *,
    task: Task,
    message_event: TaskEvent,
    candidate: _Candidate,
) -> EvidenceInput:
    return EvidenceInput(
        source_type=TASK_SUMMARY_SOURCE_TYPE,
        extracted_by=TASK_SUMMARY_EXTRACTOR,
        source_task_id=task.id,
        source_task_event_id=message_event.id,
        source_slack_channel_id=task.slack_channel_id,
        source_slack_message_ts=_string_or_none(
            message_event.payload.get("message_ts")
        ),
        raw_snippet=_shorten(
            f"Task request: {task.input}. Extracted graph item: {candidate.source_text}",
            MAX_EVIDENCE_SNIPPET_CHARS,
        ),
        confidence_score=candidate.confidence_score,
        confidence_reason=candidate.confidence_reason,
    )


def _current_entity_by_key(
    session: Session,
    task: Task,
    canonical_key: str,
) -> KnowledgeGraphEntity | None:
    return session.scalar(
        select(KnowledgeGraphEntity).where(
            KnowledgeGraphEntity.installation_id == task.installation_id,
            KnowledgeGraphEntity.canonical_key == canonical_key,
            KnowledgeGraphEntity.is_current.is_(True),
            KnowledgeGraphEntity.expired_at.is_(None),
        )
    )


def _current_edge(
    session: Session,
    *,
    task: Task,
    source_entity_id: uuid.UUID,
    target_entity_id: uuid.UUID,
    relationship_type: str,
) -> KnowledgeGraphEdge | None:
    return session.scalar(
        select(KnowledgeGraphEdge).where(
            KnowledgeGraphEdge.installation_id == task.installation_id,
            KnowledgeGraphEdge.source_entity_id == source_entity_id,
            KnowledgeGraphEdge.target_entity_id == target_entity_id,
            KnowledgeGraphEdge.relationship_type == relationship_type,
            KnowledgeGraphEdge.source_type == TASK_SUMMARY_SOURCE_TYPE,
            KnowledgeGraphEdge.is_current.is_(True),
            KnowledgeGraphEdge.expired_at.is_(None),
        )
    )


def _reinforce_entity(
    *,
    session: Session,
    graph: GraphService,
    entity: KnowledgeGraphEntity,
    task: Task,
    display_name: str,
    attrs_json: dict[str, Any],
    visibility_scope: VisibilityScope,
    lifecycle_state: str,
    confidence_score: Decimal,
    confidence_reason: str,
    evidence: EvidenceInput,
    now: datetime,
) -> int:
    entity.display_name = display_name
    entity.attrs_json = with_provenance_attrs(
        attrs_json,
        source_type=TASK_SUMMARY_SOURCE_TYPE,
        lifecycle_state=lifecycle_state,
        confidence_score=confidence_score,
    )
    entity.visibility_scope_type = visibility_scope.scope_type
    entity.visibility_scope_id = visibility_scope.scope_id
    entity.source_type = TASK_SUMMARY_SOURCE_TYPE
    entity.lifecycle_state = _reinforced_lifecycle_state(
        current=entity.lifecycle_state,
        proposed=lifecycle_state,
    )
    entity.confidence_score = max(entity.confidence_score, confidence_score)
    entity.confidence_reason = confidence_reason
    entity.freshness_window_days = 30
    entity.last_reinforced_at = now
    entity.reinforcement_count = (entity.reinforcement_count or 0) + 1
    entity.updated_at = now
    if _task_evidence_exists(
        session, task=task, target_kind="entity", target_id=entity.id
    ):
        return 0
    graph.add_evidence(
        installation_id=task.installation_id,
        target_kind="entity",
        target_id=entity.id,
        evidence=evidence,
    )
    return 1


def _reinforce_edge(
    *,
    session: Session,
    graph: GraphService,
    edge: KnowledgeGraphEdge,
    task: Task,
    attrs_json: dict[str, Any],
    visibility_scope: VisibilityScope,
    lifecycle_state: str,
    confidence_score: Decimal,
    confidence_reason: str,
    evidence: EvidenceInput,
    now: datetime,
) -> int:
    edge.attrs_json = with_provenance_attrs(
        attrs_json,
        source_type=TASK_SUMMARY_SOURCE_TYPE,
        lifecycle_state=lifecycle_state,
        confidence_score=confidence_score,
    )
    edge.visibility_scope_type = visibility_scope.scope_type
    edge.visibility_scope_id = visibility_scope.scope_id
    edge.source_type = TASK_SUMMARY_SOURCE_TYPE
    edge.lifecycle_state = _reinforced_lifecycle_state(
        current=edge.lifecycle_state,
        proposed=lifecycle_state,
    )
    edge.confidence_score = max(edge.confidence_score, confidence_score)
    edge.confidence_reason = confidence_reason
    edge.freshness_window_days = 30
    edge.last_reinforced_at = now
    edge.reinforcement_count = (edge.reinforcement_count or 0) + 1
    edge.updated_at = now
    if _task_evidence_exists(session, task=task, target_kind="edge", target_id=edge.id):
        return 0
    graph.add_evidence(
        installation_id=task.installation_id,
        target_kind="edge",
        target_id=edge.id,
        evidence=evidence,
    )
    return 1


def _task_evidence_exists(
    session: Session,
    *,
    task: Task,
    target_kind: str,
    target_id: uuid.UUID,
) -> bool:
    return (
        session.scalar(
            select(KnowledgeGraphEvidence.id)
            .where(
                KnowledgeGraphEvidence.installation_id == task.installation_id,
                KnowledgeGraphEvidence.target_kind == target_kind,
                KnowledgeGraphEvidence.target_id == target_id,
                KnowledgeGraphEvidence.source_type == TASK_SUMMARY_SOURCE_TYPE,
                KnowledgeGraphEvidence.source_task_id == task.id,
            )
            .limit(1)
        )
        is not None
    )


def _reinforced_lifecycle_state(*, current: str, proposed: str) -> str:
    if current == "confirmed":
        return "confirmed"
    return proposed


def _channel_display_name(session: Session, task: Task) -> str:
    identity = session.scalar(
        select(SlackIdentity).where(
            SlackIdentity.installation_id == task.installation_id,
            SlackIdentity.kind == "channel",
            SlackIdentity.slack_id == task.slack_channel_id,
        )
    )
    if identity and identity.display_name:
        return identity.display_name
    membership = session.scalar(
        select(SlackChannelMembership).where(
            SlackChannelMembership.installation_id == task.installation_id,
            SlackChannelMembership.channel_id == task.slack_channel_id,
        )
    )
    if membership and membership.channel_name:
        return f"#{membership.channel_name}"
    return task.slack_channel_id


def _canonical_key(*, key_prefix: str, channel_id: str, label: str) -> str:
    return f"{key_prefix}:{channel_id}:{_slug(label)}"


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:72].strip("-") or "unknown"


def _clean_line(value: str) -> str:
    line = value.strip()
    line = re.sub(r"^\s*[-*•]\s+", "", line)
    line = re.sub(r"^\s*\d+[\.)]\s+", "", line)
    line = re.sub(r"^\s*[A-Z]{2,8}-\d+\s*[·:-]\s*", "", line)
    line = line.strip("*_` ").strip()
    return " ".join(line.split())


def _looks_like_bullet(value: str) -> bool:
    return bool(re.match(r"^\s*([-*•]|\d+[\.)])\s+", value))


def _looks_substantive(value: str) -> bool:
    return len(value.split()) >= 3 and len(value) >= 18


def _string_or_none(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _shorten(value: str, max_chars: int) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 1].rstrip() + "."
