"""Write actions for dashboard knowledge graph governance."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from kortny.db.models import (
    KnowledgeGraphEdge,
    KnowledgeGraphEntity,
)
from kortny.knowledge_graph.service import EvidenceInput, GraphService


@dataclass(frozen=True)
class GraphArchiveResult:
    archived_entity_id: uuid.UUID | None
    archived_edge_ids: tuple[uuid.UUID, ...]


def confirm_entity(
    session: Session,
    entity_id: uuid.UUID,
    *,
    by_user_id: str,
) -> KnowledgeGraphEntity:
    """Confirm a current graph entity and retain admin evidence."""

    entity = _entity_for_update(session, entity_id)
    _ensure_confirmable(entity.lifecycle_state)
    now = datetime.now(UTC)
    entity.lifecycle_state = "confirmed"
    entity.is_current = True
    entity.expired_at = None
    entity.valid_to = None
    entity.last_reinforced_at = now
    entity.reinforcement_count = (entity.reinforcement_count or 0) + 1
    entity.updated_at = now
    GraphService(session).add_evidence(
        installation_id=entity.installation_id,
        target_kind="entity",
        target_id=entity.id,
        evidence=_admin_evidence(
            by_user_id,
            f"Dashboard operator confirmed graph entity {entity.canonical_key}.",
        ),
    )
    session.flush()
    return entity


def confirm_edge(
    session: Session,
    edge_id: uuid.UUID,
    *,
    by_user_id: str,
) -> KnowledgeGraphEdge:
    """Confirm a current graph edge and retain admin evidence."""

    edge = _edge_for_update(session, edge_id)
    _ensure_confirmable(edge.lifecycle_state)
    now = datetime.now(UTC)
    edge.lifecycle_state = "confirmed"
    edge.is_current = True
    edge.expired_at = None
    edge.valid_to = None
    edge.last_reinforced_at = now
    edge.reinforcement_count = (edge.reinforcement_count or 0) + 1
    edge.updated_at = now
    GraphService(session).add_evidence(
        installation_id=edge.installation_id,
        target_kind="edge",
        target_id=edge.id,
        evidence=_admin_evidence(
            by_user_id,
            f"Dashboard operator confirmed graph edge {edge.relationship_type}.",
        ),
    )
    session.flush()
    return edge


def archive_entity(
    session: Session,
    entity_id: uuid.UUID,
    *,
    by_user_id: str,
) -> GraphArchiveResult:
    """Archive a graph entity and any current edges connected to it."""

    entity = _entity_for_update(session, entity_id)
    now = datetime.now(UTC)
    _archive_entity_row(session, entity, by_user_id=by_user_id, now=now)
    edge_ids: list[uuid.UUID] = []
    edges = session.scalars(
        select(KnowledgeGraphEdge)
        .where(
            KnowledgeGraphEdge.installation_id == entity.installation_id,
            KnowledgeGraphEdge.is_current.is_(True),
            KnowledgeGraphEdge.expired_at.is_(None),
            or_(
                KnowledgeGraphEdge.source_entity_id == entity.id,
                KnowledgeGraphEdge.target_entity_id == entity.id,
            ),
        )
        .with_for_update()
    )
    for edge in edges:
        _archive_edge_row(session, edge, by_user_id=by_user_id, now=now)
        edge_ids.append(edge.id)
    session.flush()
    return GraphArchiveResult(
        archived_entity_id=entity.id,
        archived_edge_ids=tuple(edge_ids),
    )


def archive_edge(
    session: Session,
    edge_id: uuid.UUID,
    *,
    by_user_id: str,
) -> GraphArchiveResult:
    """Archive a graph edge while preserving the source/target entities."""

    edge = _edge_for_update(session, edge_id)
    _archive_edge_row(session, edge, by_user_id=by_user_id, now=datetime.now(UTC))
    session.flush()
    return GraphArchiveResult(archived_entity_id=None, archived_edge_ids=(edge.id,))


def _entity_for_update(session: Session, entity_id: uuid.UUID) -> KnowledgeGraphEntity:
    entity = session.scalar(
        select(KnowledgeGraphEntity)
        .where(KnowledgeGraphEntity.id == entity_id)
        .limit(1)
        .with_for_update()
    )
    if entity is None:
        raise LookupError(f"Graph entity not found: {entity_id}")
    return entity


def _edge_for_update(session: Session, edge_id: uuid.UUID) -> KnowledgeGraphEdge:
    edge = session.scalar(
        select(KnowledgeGraphEdge)
        .where(KnowledgeGraphEdge.id == edge_id)
        .limit(1)
        .with_for_update()
    )
    if edge is None:
        raise LookupError(f"Graph edge not found: {edge_id}")
    return edge


def _ensure_confirmable(lifecycle_state: str) -> None:
    if lifecycle_state not in {"candidate", "active", "stale"}:
        raise ValueError(
            "Only candidate, active, or stale graph rows can be confirmed."
        )


def _archive_entity_row(
    session: Session,
    entity: KnowledgeGraphEntity,
    *,
    by_user_id: str,
    now: datetime,
) -> None:
    if entity.lifecycle_state in {"archived", "forgotten", "superseded"}:
        raise ValueError("This graph entity is already inactive.")
    entity.lifecycle_state = "archived"
    entity.is_current = False
    entity.valid_to = now
    entity.expired_at = now
    entity.updated_at = now
    GraphService(session).add_evidence(
        installation_id=entity.installation_id,
        target_kind="entity",
        target_id=entity.id,
        evidence=_admin_evidence(
            by_user_id,
            f"Dashboard operator archived graph entity {entity.canonical_key}.",
        ),
    )


def _archive_edge_row(
    session: Session,
    edge: KnowledgeGraphEdge,
    *,
    by_user_id: str,
    now: datetime,
) -> None:
    if edge.lifecycle_state in {"archived", "forgotten", "superseded"}:
        return
    edge.lifecycle_state = "archived"
    edge.is_current = False
    edge.valid_to = now
    edge.expired_at = now
    edge.updated_at = now
    GraphService(session).add_evidence(
        installation_id=edge.installation_id,
        target_kind="edge",
        target_id=edge.id,
        evidence=_admin_evidence(
            by_user_id,
            f"Dashboard operator archived graph edge {edge.relationship_type}.",
        ),
    )


def _admin_evidence(by_user_id: str, snippet: str) -> EvidenceInput:
    return EvidenceInput(
        source_type="admin_import",
        extracted_by=by_user_id,
        raw_snippet=snippet,
        confidence_reason="Dashboard graph governance action.",
    )
