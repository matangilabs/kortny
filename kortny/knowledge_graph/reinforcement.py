"""Runtime reinforcement for graph rows used in successful task answers."""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.orm import Session

from kortny.db.models import (
    KnowledgeGraphEdge,
    KnowledgeGraphEntity,
    KnowledgeGraphEvidence,
    Task,
    TaskEvent,
    TaskEventType,
)
from kortny.knowledge_graph.service import EvidenceInput, GraphService

KG_RUNTIME_CONTEXT_REINFORCED_MESSAGE = "kg_runtime_context_reinforced"
RUNTIME_REINFORCEMENT_SOURCE_TYPE = "task_summary"
RUNTIME_REINFORCEMENT_EXTRACTOR = "kortny.runtime_graph_reinforcement"
CURRENT_REINFORCEMENT_STATES = ("active", "confirmed")
MAX_REINFORCED_ROWS_PER_TASK = 25
MAX_EVIDENCE_SNIPPET_CHARS = 700


@dataclass(frozen=True, slots=True)
class RuntimeGraphReinforcementResult:
    """Counts from reinforcing graph context used by one completed task."""

    entity_count: int = 0
    edge_count: int = 0
    evidence_count: int = 0
    duplicate_count: int = 0
    source_event_ids: tuple[int, ...] = ()
    message_event_id: int | None = None

    @property
    def reinforced_count(self) -> int:
        return self.entity_count + self.edge_count

    def to_payload(self) -> dict[str, object]:
        return {
            "message": KG_RUNTIME_CONTEXT_REINFORCED_MESSAGE,
            "entity_count": self.entity_count,
            "edge_count": self.edge_count,
            "reinforced_count": self.reinforced_count,
            "evidence_count": self.evidence_count,
            "duplicate_count": self.duplicate_count,
            "source_event_ids": list(self.source_event_ids),
            "message_event_id": self.message_event_id,
            "source_type": RUNTIME_REINFORCEMENT_SOURCE_TYPE,
        }


class RuntimeGraphReinforcementService:
    """Reinforce current graph rows that directly informed a delivered answer."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self.graph = GraphService(session)

    def reinforce_task_context(self, task: Task) -> RuntimeGraphReinforcementResult:
        """Add task-backed evidence to graph rows returned to the agent.

        This is deliberately conservative: it only reinforces current rows that
        a successful `query_workspace_graph` tool result returned by ID. It does
        not create new graph facts from the final answer text.
        """

        events = _task_events(self.session, task)
        message_event = _latest_result_message_event(events)
        if message_event is None:
            return RuntimeGraphReinforcementResult()

        graph_events = [
            event for event in events if _successful_graph_tool_result(event)
        ]
        if not graph_events:
            return RuntimeGraphReinforcementResult(message_event_id=message_event.id)

        entity_ids = _bounded_unique_ids(
            row_id
            for event in graph_events
            for row_id in _graph_result_entity_ids(event.payload)
        )
        edge_ids = _bounded_unique_ids(
            row_id
            for event in graph_events
            for row_id in _graph_result_edge_ids(event.payload)
        )
        if not entity_ids and not edge_ids:
            return RuntimeGraphReinforcementResult(
                source_event_ids=tuple(event.id for event in graph_events),
                message_event_id=message_event.id,
            )

        now = datetime.now(UTC)
        source_event_ids = tuple(event.id for event in graph_events)
        evidence = _runtime_evidence(
            task=task,
            message_event=message_event,
            graph_events=graph_events,
        )
        entity_count, entity_duplicates = self._reinforce_entities(
            task=task,
            entity_ids=entity_ids,
            evidence=evidence,
            now=now,
        )
        edge_count, edge_duplicates = self._reinforce_edges(
            task=task,
            edge_ids=edge_ids,
            evidence=evidence,
            now=now,
        )
        self.session.flush()
        return RuntimeGraphReinforcementResult(
            entity_count=entity_count,
            edge_count=edge_count,
            evidence_count=entity_count + edge_count,
            duplicate_count=entity_duplicates + edge_duplicates,
            source_event_ids=source_event_ids,
            message_event_id=message_event.id,
        )

    def _reinforce_entities(
        self,
        *,
        task: Task,
        entity_ids: Sequence[uuid.UUID],
        evidence: EvidenceInput,
        now: datetime,
    ) -> tuple[int, int]:
        if not entity_ids:
            return 0, 0

        rows = self.session.scalars(
            select(KnowledgeGraphEntity)
            .where(
                KnowledgeGraphEntity.installation_id == task.installation_id,
                KnowledgeGraphEntity.id.in_(entity_ids),
                KnowledgeGraphEntity.is_current.is_(True),
                KnowledgeGraphEntity.expired_at.is_(None),
                KnowledgeGraphEntity.lifecycle_state.in_(CURRENT_REINFORCEMENT_STATES),
            )
            .with_for_update()
        )
        reinforced = 0
        duplicates = 0
        for row in rows:
            if _task_evidence_exists(
                self.session,
                task=task,
                target_kind="entity",
                target_id=row.id,
            ):
                duplicates += 1
                continue
            row.last_reinforced_at = now
            row.reinforcement_count = (row.reinforcement_count or 0) + 1
            row.updated_at = now
            self.graph.add_evidence(
                installation_id=task.installation_id,
                target_kind="entity",
                target_id=row.id,
                evidence=evidence,
            )
            reinforced += 1
        return reinforced, duplicates

    def _reinforce_edges(
        self,
        *,
        task: Task,
        edge_ids: Sequence[uuid.UUID],
        evidence: EvidenceInput,
        now: datetime,
    ) -> tuple[int, int]:
        if not edge_ids:
            return 0, 0

        rows = self.session.scalars(
            select(KnowledgeGraphEdge)
            .where(
                KnowledgeGraphEdge.installation_id == task.installation_id,
                KnowledgeGraphEdge.id.in_(edge_ids),
                KnowledgeGraphEdge.is_current.is_(True),
                KnowledgeGraphEdge.expired_at.is_(None),
                KnowledgeGraphEdge.lifecycle_state.in_(CURRENT_REINFORCEMENT_STATES),
            )
            .with_for_update()
        )
        reinforced = 0
        duplicates = 0
        for row in rows:
            if _task_evidence_exists(
                self.session,
                task=task,
                target_kind="edge",
                target_id=row.id,
            ):
                duplicates += 1
                continue
            row.last_reinforced_at = now
            row.reinforcement_count = (row.reinforcement_count or 0) + 1
            row.updated_at = now
            self.graph.add_evidence(
                installation_id=task.installation_id,
                target_kind="edge",
                target_id=row.id,
                evidence=evidence,
            )
            reinforced += 1
        return reinforced, duplicates


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


def _successful_graph_tool_result(event: TaskEvent) -> bool:
    if event.type is not TaskEventType.tool_result:
        return False
    if event.payload.get("tool") != "query_workspace_graph":
        return False
    output = event.payload.get("output")
    return isinstance(output, dict) and output.get("successful") is True


def _graph_result_entity_ids(payload: dict[str, object]) -> tuple[str, ...]:
    output = payload.get("output")
    if not isinstance(output, dict):
        return ()
    entities = output.get("entities")
    if not isinstance(entities, list):
        return ()
    return tuple(
        row["id"]
        for row in entities
        if isinstance(row, dict) and isinstance(row.get("id"), str)
    )


def _graph_result_edge_ids(payload: dict[str, object]) -> tuple[str, ...]:
    output = payload.get("output")
    if not isinstance(output, dict):
        return ()
    relationships = output.get("relationships")
    if not isinstance(relationships, list):
        return ()
    return tuple(
        row["id"]
        for row in relationships
        if isinstance(row, dict) and isinstance(row.get("id"), str)
    )


def _bounded_unique_ids(values: Iterable[str]) -> tuple[uuid.UUID, ...]:
    output: list[uuid.UUID] = []
    seen: set[uuid.UUID] = set()
    for value in values:
        try:
            row_id = uuid.UUID(value)
        except ValueError:
            continue
        if row_id in seen:
            continue
        seen.add(row_id)
        output.append(row_id)
        if len(output) >= MAX_REINFORCED_ROWS_PER_TASK:
            break
    return tuple(output)


def _runtime_evidence(
    *,
    task: Task,
    message_event: TaskEvent,
    graph_events: Sequence[TaskEvent],
) -> EvidenceInput:
    return EvidenceInput(
        source_type=RUNTIME_REINFORCEMENT_SOURCE_TYPE,
        extracted_by=RUNTIME_REINFORCEMENT_EXTRACTOR,
        source_task_id=task.id,
        source_task_event_id=message_event.id,
        source_slack_channel_id=task.slack_channel_id,
        source_slack_message_ts=_string_or_none(
            message_event.payload.get("message_ts")
        ),
        raw_snippet=_evidence_snippet(task, message_event, graph_events),
        confidence_score=Decimal("0.650"),
        confidence_reason=(
            "Graph row was returned by scope-safe runtime retrieval and used "
            "in a successful Slack answer."
        ),
    )


def _evidence_snippet(
    task: Task,
    message_event: TaskEvent,
    graph_events: Sequence[TaskEvent],
) -> str:
    source_count = len(graph_events)
    answer = _string_or_none(message_event.payload.get("text")) or ""
    text = (
        f"Runtime graph context was used while answering the task: "
        f"{task.input.strip() or 'untitled task'}. "
        f"Graph retrieval event count: {source_count}. "
        f"Delivered answer preview: {answer}"
    )
    return _shorten(text, MAX_EVIDENCE_SNIPPET_CHARS)


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
                KnowledgeGraphEvidence.source_type == RUNTIME_REINFORCEMENT_SOURCE_TYPE,
                KnowledgeGraphEvidence.source_task_id == task.id,
            )
            .limit(1)
        )
        is not None
    )


def _string_or_none(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _shorten(value: str, max_chars: int) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 1].rstrip() + "."
