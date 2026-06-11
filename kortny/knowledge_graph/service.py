"""Postgres-backed workspace knowledge graph service."""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from sqlalchemy import Text, and_, cast, exists, func, or_, select
from sqlalchemy.orm import Session
from sqlalchemy.sql.elements import ColumnElement

from kortny.db.models import (
    KnowledgeGraphEdge,
    KnowledgeGraphEntity,
    KnowledgeGraphEvidence,
)
from kortny.embeddings import (
    KG_ENTITY_EMBEDDING_KIND,
    EmbeddingIndex,
    kg_entity_embedding_text,
)
from kortny.knowledge_graph.provenance import (
    provenance_kind,
    provenance_label,
    review_status,
    with_provenance_attrs,
)
from kortny.knowledge_graph.scopes import (
    DestinationSurface,
    VisibilityScope,
    compatible_scope_predicate,
    is_scope_compatible,
)

CURRENT_CONTEXT_STATES = ("active", "confirmed")


@dataclass(frozen=True)
class EvidenceInput:
    source_type: str
    extracted_by: str
    source_task_id: uuid.UUID | None = None
    source_episode_id: uuid.UUID | None = None
    source_task_event_id: int | None = None
    source_observation_id: uuid.UUID | None = None
    source_slack_channel_id: str | None = None
    source_slack_message_ts: str | None = None
    source_slack_file_id: str | None = None
    source_url: str | None = None
    raw_snippet: str | None = None
    confidence_score: Decimal | None = None
    confidence_reason: str | None = None
    consensus_count: int = 1


@dataclass(frozen=True)
class RetrievedGraphEntity:
    id: uuid.UUID
    entity_type: str
    canonical_key: str
    display_name: str | None
    source_type: str
    visibility_scope: VisibilityScope
    lifecycle_state: str
    confidence_score: Decimal
    confidence_reason: str | None
    provenance_kind: str
    provenance_label: str
    review_status: str
    evidence_ids: tuple[uuid.UUID, ...]
    last_seen_at: datetime | None = None


@dataclass(frozen=True)
class RetrievedGraphEdge:
    id: uuid.UUID
    source_entity_id: uuid.UUID
    target_entity_id: uuid.UUID
    relationship_type: str
    source_type: str
    visibility_scope: VisibilityScope
    lifecycle_state: str
    confidence_score: Decimal
    confidence_reason: str | None
    provenance_kind: str
    provenance_label: str
    review_status: str
    evidence_ids: tuple[uuid.UUID, ...]


@dataclass(frozen=True)
class GraphContextPack:
    entities: tuple[RetrievedGraphEntity, ...]
    edges: tuple[RetrievedGraphEdge, ...]
    returned_scopes: tuple[VisibilityScope, ...]
    omitted_count: int
    omitted_reasons: tuple[str, ...]


@dataclass(frozen=True)
class GraphStalenessResult:
    entity_ids: tuple[uuid.UUID, ...]
    edge_ids: tuple[uuid.UUID, ...]

    @property
    def entity_count(self) -> int:
        return len(self.entity_ids)

    @property
    def edge_count(self) -> int:
        return len(self.edge_ids)


class GraphService:
    """Service API for the HIG-181 workspace knowledge graph."""

    def __init__(
        self,
        session: Session,
        *,
        embedding_index: EmbeddingIndex | None = None,
    ) -> None:
        self.session = session
        self.embedding_index = embedding_index

    def ensure_entity_embedding(self, entity: KnowledgeGraphEntity) -> None:
        """Embed-on-write hook; no-op without an index, never raises."""

        if self.embedding_index is None:
            return
        self.embedding_index.ensure(
            KG_ENTITY_EMBEDDING_KIND,
            [(str(entity.id), kg_entity_embedding_text(entity))],
        )

    def create_entity(
        self,
        *,
        installation_id: uuid.UUID,
        entity_type: str,
        canonical_key: str,
        visibility_scope: VisibilityScope,
        source_type: str,
        display_name: str | None = None,
        external_ref_type: str | None = None,
        external_ref_id: str | None = None,
        attrs_json: dict | None = None,
        lifecycle_state: str = "candidate",
        confidence_score: Decimal | float = Decimal("0.500"),
        confidence_reason: str | None = None,
        freshness_window_days: int | None = None,
        evidence: EvidenceInput | None = None,
    ) -> KnowledgeGraphEntity:
        entity = KnowledgeGraphEntity(
            installation_id=installation_id,
            entity_type=entity_type,
            canonical_key=canonical_key,
            display_name=display_name,
            external_ref_type=external_ref_type,
            external_ref_id=external_ref_id,
            attrs_json=with_provenance_attrs(
                attrs_json,
                source_type=source_type,
                lifecycle_state=lifecycle_state,
                confidence_score=confidence_score,
            ),
            visibility_scope_type=visibility_scope.scope_type,
            visibility_scope_id=visibility_scope.scope_id,
            source_type=source_type,
            lifecycle_state=lifecycle_state,
            confidence_score=confidence_score,
            confidence_reason=confidence_reason,
            freshness_window_days=freshness_window_days,
        )
        if entity.valid_at is None:
            entity.valid_at = datetime.now(UTC)
        self.session.add(entity)
        self.session.flush()
        if evidence is not None:
            self.add_evidence(
                installation_id=installation_id,
                target_kind="entity",
                target_id=entity.id,
                evidence=evidence,
            )
        self.ensure_entity_embedding(entity)
        return entity

    def create_edge(
        self,
        *,
        installation_id: uuid.UUID,
        source_entity_id: uuid.UUID,
        target_entity_id: uuid.UUID,
        relationship_type: str,
        visibility_scope: VisibilityScope,
        source_type: str,
        attrs_json: dict | None = None,
        lifecycle_state: str = "candidate",
        confidence_score: Decimal | float = Decimal("0.500"),
        confidence_reason: str | None = None,
        freshness_window_days: int | None = None,
        evidence: EvidenceInput | None = None,
    ) -> KnowledgeGraphEdge:
        edge = KnowledgeGraphEdge(
            installation_id=installation_id,
            source_entity_id=source_entity_id,
            target_entity_id=target_entity_id,
            relationship_type=relationship_type,
            attrs_json=with_provenance_attrs(
                attrs_json,
                source_type=source_type,
                lifecycle_state=lifecycle_state,
                confidence_score=confidence_score,
            ),
            visibility_scope_type=visibility_scope.scope_type,
            visibility_scope_id=visibility_scope.scope_id,
            source_type=source_type,
            lifecycle_state=lifecycle_state,
            confidence_score=confidence_score,
            confidence_reason=confidence_reason,
            freshness_window_days=freshness_window_days,
        )
        if edge.valid_at is None:
            edge.valid_at = datetime.now(UTC)
        self.session.add(edge)
        self.session.flush()
        if evidence is not None:
            self.add_evidence(
                installation_id=installation_id,
                target_kind="edge",
                target_id=edge.id,
                evidence=evidence,
            )
        return edge

    def add_evidence(
        self,
        *,
        installation_id: uuid.UUID,
        target_kind: str,
        target_id: uuid.UUID,
        evidence: EvidenceInput,
    ) -> KnowledgeGraphEvidence:
        row = KnowledgeGraphEvidence(
            installation_id=installation_id,
            target_kind=target_kind,
            target_id=target_id,
            source_type=evidence.source_type,
            source_task_id=evidence.source_task_id,
            source_episode_id=evidence.source_episode_id,
            source_task_event_id=evidence.source_task_event_id,
            source_observation_id=evidence.source_observation_id,
            source_slack_channel_id=evidence.source_slack_channel_id,
            source_slack_message_ts=evidence.source_slack_message_ts,
            source_slack_file_id=evidence.source_slack_file_id,
            source_url=evidence.source_url,
            extracted_by=evidence.extracted_by,
            raw_snippet=evidence.raw_snippet,
            confidence_score=evidence.confidence_score,
            confidence_reason=evidence.confidence_reason,
            consensus_count=evidence.consensus_count,
        )
        self.session.add(row)
        self.session.flush()
        return row

    def supersede_entity(
        self,
        current: KnowledgeGraphEntity,
        replacement: KnowledgeGraphEntity | None = None,
    ) -> None:
        now = datetime.now(UTC)
        current.is_current = False
        current.lifecycle_state = "superseded"
        current.valid_to = now
        current.expired_at = now
        current.invalid_at = now
        if replacement is not None:
            replacement.is_current = True

    def supersede_edge(
        self,
        current: KnowledgeGraphEdge,
        replacement: KnowledgeGraphEdge | None = None,
    ) -> None:
        now = datetime.now(UTC)
        current.is_current = False
        current.lifecycle_state = "superseded"
        current.valid_to = now
        current.expired_at = now
        current.invalid_at = now
        if replacement is not None:
            replacement.is_current = True

    def invalidate_entity(
        self,
        current: KnowledgeGraphEntity,
        *,
        now: datetime | None = None,
        reason: str | None = None,
    ) -> None:
        """Temporal contradiction: invalidate without deleting (HIG-225).

        History stays queryable — the row keeps its evidence; ``invalid_at``
        bounds its validity interval and ``is_current`` frees the canonical-key
        slot for a successor.
        """

        effective_now = now or datetime.now(UTC)
        current.invalid_at = effective_now
        current.lifecycle_state = "contradicted"
        current.is_current = False
        current.updated_at = effective_now
        if reason:
            current.confidence_reason = reason

    def invalidate_edge(
        self,
        current: KnowledgeGraphEdge,
        *,
        now: datetime | None = None,
        reason: str | None = None,
    ) -> None:
        """Temporal contradiction for an edge; see ``invalidate_entity``."""

        effective_now = now or datetime.now(UTC)
        current.invalid_at = effective_now
        current.lifecycle_state = "contradicted"
        current.is_current = False
        current.updated_at = effective_now
        if reason:
            current.confidence_reason = reason

    def mark_stale_current(
        self,
        *,
        installation_id: uuid.UUID | None = None,
        now: datetime | None = None,
        default_stale_days: int | None = None,
    ) -> GraphStalenessResult:
        """Mark current graph rows stale when their freshness window has elapsed.

        ``default_stale_days`` additionally ages rows without an explicit
        ``freshness_window_days``: rows not reinforced within that many days
        also go stale (HIG-225 consolidator aging pass).
        """

        effective_now = now or datetime.now(UTC)
        entity_predicates: list[ColumnElement[bool]] = [
            KnowledgeGraphEntity.is_current.is_(True),
            KnowledgeGraphEntity.expired_at.is_(None),
            KnowledgeGraphEntity.lifecycle_state.in_(
                ("candidate", "active", "confirmed")
            ),
        ]
        edge_predicates: list[ColumnElement[bool]] = [
            KnowledgeGraphEdge.is_current.is_(True),
            KnowledgeGraphEdge.expired_at.is_(None),
            KnowledgeGraphEdge.lifecycle_state.in_(
                ("candidate", "active", "confirmed")
            ),
        ]
        if default_stale_days is None:
            entity_predicates.append(
                KnowledgeGraphEntity.freshness_window_days.is_not(None)
            )
            edge_predicates.append(
                KnowledgeGraphEdge.freshness_window_days.is_not(None)
            )
        if installation_id is not None:
            entity_predicates.append(
                KnowledgeGraphEntity.installation_id == installation_id
            )
            edge_predicates.append(
                KnowledgeGraphEdge.installation_id == installation_id
            )

        stale_entities: list[KnowledgeGraphEntity] = []
        for entity in self.session.scalars(
            select(KnowledgeGraphEntity).where(*entity_predicates)
        ):
            if _staleness_elapsed(entity, effective_now, default_stale_days):
                entity.lifecycle_state = "stale"
                entity.updated_at = effective_now
                stale_entities.append(entity)

        stale_edges: list[KnowledgeGraphEdge] = []
        for edge in self.session.scalars(
            select(KnowledgeGraphEdge).where(*edge_predicates)
        ):
            if _staleness_elapsed(edge, effective_now, default_stale_days):
                edge.lifecycle_state = "stale"
                edge.updated_at = effective_now
                stale_edges.append(edge)

        self.session.flush()
        return GraphStalenessResult(
            entity_ids=tuple(row.id for row in stale_entities),
            edge_ids=tuple(row.id for row in stale_edges),
        )

    def retrieve_current_context(
        self,
        *,
        installation_id: uuid.UUID,
        destination: DestinationSurface,
        anchor_keys: Sequence[str] = (),
        max_hops: int = 1,
        max_items: int = 20,
    ) -> GraphContextPack:
        """Return current graph context allowed for the destination surface."""

        if max_hops < 0:
            raise ValueError("max_hops must be non-negative")
        if max_items <= 0:
            return GraphContextPack((), (), (), 0, ("max_items<=0",))

        entity_rows: list[KnowledgeGraphEntity]
        edge_rows: list[KnowledgeGraphEdge]
        omitted_reasons: list[str] = []

        if anchor_keys:
            anchor_rows = self._select_current_entities(
                installation_id=installation_id,
                destination=destination,
                canonical_keys=anchor_keys,
                limit=max_items,
            )
            if not anchor_rows:
                return GraphContextPack((), (), (), 0, ("no_anchors_found",))
            entity_ids = {row.id for row in anchor_rows}
            edge_ids: set[uuid.UUID] = set()
            frontier = set(entity_ids)
            for _hop in range(max_hops):
                if not frontier:
                    break
                hop_edges = self._select_current_edges_for_frontier(
                    installation_id=installation_id,
                    destination=destination,
                    frontier_ids=frontier,
                    limit=max_items,
                )
                next_frontier: set[uuid.UUID] = set()
                for edge in hop_edges:
                    if edge.id in edge_ids:
                        continue
                    edge_ids.add(edge.id)
                    for candidate_id in (edge.source_entity_id, edge.target_entity_id):
                        if candidate_id not in entity_ids:
                            next_frontier.add(candidate_id)
                            entity_ids.add(candidate_id)
                frontier = next_frontier

            entity_rows = self._select_current_entities_by_id(
                installation_id=installation_id,
                destination=destination,
                entity_ids=entity_ids,
                limit=max_items,
            )
            edge_rows = self._select_current_edges_by_id(
                installation_id=installation_id,
                destination=destination,
                edge_ids=edge_ids,
                limit=max_items,
            )
        else:
            entity_rows = self._select_current_entities(
                installation_id=installation_id,
                destination=destination,
                limit=max_items,
            )
            edge_rows = self._select_current_edges(
                installation_id=installation_id,
                destination=destination,
                limit=max_items,
            )

        entity_evidence = self._evidence_ids("entity", [row.id for row in entity_rows])
        edge_evidence = self._evidence_ids("edge", [row.id for row in edge_rows])
        entities = tuple(
            _retrieved_entity(row, entity_evidence.get(row.id, ()))
            for row in entity_rows
        )
        edges = tuple(
            _retrieved_edge(row, edge_evidence.get(row.id, ())) for row in edge_rows
        )
        omitted_count = 0
        if len(entity_rows) >= max_items:
            omitted_count += 1
            omitted_reasons.append("entity_limit_reached")
        if len(edge_rows) >= max_items:
            omitted_count += 1
            omitted_reasons.append("edge_limit_reached")

        return GraphContextPack(
            entities=entities,
            edges=edges,
            returned_scopes=_returned_scopes(entities, edges),
            omitted_count=omitted_count,
            omitted_reasons=tuple(omitted_reasons),
        )

    def query_current_context(
        self,
        *,
        installation_id: uuid.UUID,
        destination: DestinationSurface,
        query: str | None = None,
        anchor_keys: Sequence[str] = (),
        max_hops: int = 1,
        max_items: int = 20,
    ) -> GraphContextPack:
        """Search or traverse current graph context allowed for the destination."""

        normalized_query = " ".join((query or "").split())
        if anchor_keys or not normalized_query:
            return self.retrieve_current_context(
                installation_id=installation_id,
                destination=destination,
                anchor_keys=anchor_keys,
                max_hops=max_hops,
                max_items=max_items,
            )
        if max_items <= 0:
            return GraphContextPack((), (), (), 0, ("max_items<=0",))

        entity_rows = self._search_current_entities(
            installation_id=installation_id,
            destination=destination,
            query=normalized_query,
            limit=max_items,
        )
        edge_rows = self._search_current_edges(
            installation_id=installation_id,
            destination=destination,
            query=normalized_query,
            limit=max_items,
        )
        entity_ids = {row.id for row in entity_rows}
        for edge in edge_rows:
            entity_ids.add(edge.source_entity_id)
            entity_ids.add(edge.target_entity_id)
        entity_rows = self._select_current_entities_by_id(
            installation_id=installation_id,
            destination=destination,
            entity_ids=entity_ids,
            limit=max_items,
        )

        entity_evidence = self._evidence_ids("entity", [row.id for row in entity_rows])
        edge_evidence = self._evidence_ids("edge", [row.id for row in edge_rows])
        entities = tuple(
            _retrieved_entity(row, entity_evidence.get(row.id, ()))
            for row in entity_rows
        )
        edges = tuple(
            _retrieved_edge(row, edge_evidence.get(row.id, ())) for row in edge_rows
        )
        omitted_reasons: list[str] = []
        if len(entity_rows) >= max_items:
            omitted_reasons.append("entity_limit_reached")
        if len(edge_rows) >= max_items:
            omitted_reasons.append("edge_limit_reached")
        return GraphContextPack(
            entities=entities,
            edges=edges,
            returned_scopes=_returned_scopes(entities, edges),
            omitted_count=len(omitted_reasons),
            omitted_reasons=tuple(omitted_reasons),
        )

    @staticmethod
    def scope_guard_violations(
        pack: GraphContextPack,
        destination: DestinationSurface,
    ) -> tuple[VisibilityScope, ...]:
        scopes = [entity.visibility_scope for entity in pack.entities]
        scopes.extend(edge.visibility_scope for edge in pack.edges)
        return tuple(
            scope for scope in scopes if not is_scope_compatible(scope, destination)
        )

    def _select_current_entities(
        self,
        *,
        installation_id: uuid.UUID,
        destination: DestinationSurface,
        canonical_keys: Sequence[str] = (),
        limit: int,
    ) -> list[KnowledgeGraphEntity]:
        predicates = [
            KnowledgeGraphEntity.installation_id == installation_id,
            self._current_entity_predicate(),
            compatible_scope_predicate(KnowledgeGraphEntity, destination),
            self._entity_has_evidence_predicate(),
        ]
        if canonical_keys:
            predicates.append(KnowledgeGraphEntity.canonical_key.in_(canonical_keys))
        return list(
            self.session.scalars(
                select(KnowledgeGraphEntity)
                .where(*predicates)
                .order_by(
                    KnowledgeGraphEntity.confidence_score.desc(),
                    KnowledgeGraphEntity.updated_at.desc(),
                )
                .limit(limit)
            )
        )

    def _select_current_entities_by_id(
        self,
        *,
        installation_id: uuid.UUID,
        destination: DestinationSurface,
        entity_ids: Iterable[uuid.UUID],
        limit: int,
    ) -> list[KnowledgeGraphEntity]:
        ids = tuple(entity_ids)
        if not ids:
            return []
        return list(
            self.session.scalars(
                select(KnowledgeGraphEntity)
                .where(
                    KnowledgeGraphEntity.installation_id == installation_id,
                    KnowledgeGraphEntity.id.in_(ids),
                    self._current_entity_predicate(),
                    compatible_scope_predicate(KnowledgeGraphEntity, destination),
                    self._entity_has_evidence_predicate(),
                )
                .order_by(
                    KnowledgeGraphEntity.confidence_score.desc(),
                    KnowledgeGraphEntity.updated_at.desc(),
                )
                .limit(limit)
            )
        )

    def _select_current_edges(
        self,
        *,
        installation_id: uuid.UUID,
        destination: DestinationSurface,
        limit: int,
    ) -> list[KnowledgeGraphEdge]:
        return list(
            self.session.scalars(
                select(KnowledgeGraphEdge)
                .where(
                    KnowledgeGraphEdge.installation_id == installation_id,
                    self._current_edge_predicate(),
                    compatible_scope_predicate(KnowledgeGraphEdge, destination),
                    self._edge_has_evidence_predicate(),
                )
                .order_by(
                    KnowledgeGraphEdge.confidence_score.desc(),
                    KnowledgeGraphEdge.updated_at.desc(),
                )
                .limit(limit)
            )
        )

    def _select_current_edges_for_frontier(
        self,
        *,
        installation_id: uuid.UUID,
        destination: DestinationSurface,
        frontier_ids: set[uuid.UUID],
        limit: int,
    ) -> list[KnowledgeGraphEdge]:
        if not frontier_ids:
            return []
        return list(
            self.session.scalars(
                select(KnowledgeGraphEdge)
                .where(
                    KnowledgeGraphEdge.installation_id == installation_id,
                    or_(
                        KnowledgeGraphEdge.source_entity_id.in_(frontier_ids),
                        KnowledgeGraphEdge.target_entity_id.in_(frontier_ids),
                    ),
                    self._current_edge_predicate(),
                    compatible_scope_predicate(KnowledgeGraphEdge, destination),
                    self._edge_has_evidence_predicate(),
                )
                .order_by(
                    KnowledgeGraphEdge.confidence_score.desc(),
                    KnowledgeGraphEdge.updated_at.desc(),
                )
                .limit(limit)
            )
        )

    def _select_current_edges_by_id(
        self,
        *,
        installation_id: uuid.UUID,
        destination: DestinationSurface,
        edge_ids: Iterable[uuid.UUID],
        limit: int,
    ) -> list[KnowledgeGraphEdge]:
        ids = tuple(edge_ids)
        if not ids:
            return []
        return list(
            self.session.scalars(
                select(KnowledgeGraphEdge)
                .where(
                    KnowledgeGraphEdge.installation_id == installation_id,
                    KnowledgeGraphEdge.id.in_(ids),
                    self._current_edge_predicate(),
                    compatible_scope_predicate(KnowledgeGraphEdge, destination),
                    self._edge_has_evidence_predicate(),
                )
                .order_by(
                    KnowledgeGraphEdge.confidence_score.desc(),
                    KnowledgeGraphEdge.updated_at.desc(),
                )
                .limit(limit)
            )
        )

    def _search_current_entities(
        self,
        *,
        installation_id: uuid.UUID,
        destination: DestinationSurface,
        query: str,
        limit: int,
    ) -> list[KnowledgeGraphEntity]:
        pattern = f"%{query}%"
        return list(
            self.session.scalars(
                select(KnowledgeGraphEntity)
                .where(
                    KnowledgeGraphEntity.installation_id == installation_id,
                    self._current_entity_predicate(),
                    compatible_scope_predicate(KnowledgeGraphEntity, destination),
                    self._entity_has_evidence_predicate(),
                    or_(
                        KnowledgeGraphEntity.canonical_key.ilike(pattern),
                        KnowledgeGraphEntity.display_name.ilike(pattern),
                        KnowledgeGraphEntity.entity_type.ilike(pattern),
                        KnowledgeGraphEntity.source_type.ilike(pattern),
                        KnowledgeGraphEntity.visibility_scope_id.ilike(pattern),
                        cast(KnowledgeGraphEntity.attrs_json, Text).ilike(pattern),
                    ),
                )
                .order_by(
                    KnowledgeGraphEntity.confidence_score.desc(),
                    KnowledgeGraphEntity.updated_at.desc(),
                )
                .limit(limit)
            )
        )

    def _search_current_edges(
        self,
        *,
        installation_id: uuid.UUID,
        destination: DestinationSurface,
        query: str,
        limit: int,
    ) -> list[KnowledgeGraphEdge]:
        source = KnowledgeGraphEntity
        target = KnowledgeGraphEntity
        pattern = f"%{query}%"
        source_alias = source.__table__.alias("kg_edge_search_source")
        target_alias = target.__table__.alias("kg_edge_search_target")
        return list(
            self.session.scalars(
                select(KnowledgeGraphEdge)
                .join(
                    source_alias,
                    KnowledgeGraphEdge.source_entity_id == source_alias.c.id,
                )
                .join(
                    target_alias,
                    KnowledgeGraphEdge.target_entity_id == target_alias.c.id,
                )
                .where(
                    KnowledgeGraphEdge.installation_id == installation_id,
                    self._current_edge_predicate(),
                    compatible_scope_predicate(KnowledgeGraphEdge, destination),
                    self._edge_has_evidence_predicate(),
                    or_(
                        KnowledgeGraphEdge.relationship_type.ilike(pattern),
                        KnowledgeGraphEdge.source_type.ilike(pattern),
                        KnowledgeGraphEdge.visibility_scope_id.ilike(pattern),
                        cast(KnowledgeGraphEdge.attrs_json, Text).ilike(pattern),
                        source_alias.c.canonical_key.ilike(pattern),
                        source_alias.c.display_name.ilike(pattern),
                        target_alias.c.canonical_key.ilike(pattern),
                        target_alias.c.display_name.ilike(pattern),
                    ),
                )
                .order_by(
                    KnowledgeGraphEdge.confidence_score.desc(),
                    KnowledgeGraphEdge.updated_at.desc(),
                )
                .limit(limit)
            )
        )

    def _evidence_ids(
        self,
        target_kind: str,
        target_ids: Sequence[uuid.UUID],
    ) -> dict[uuid.UUID, list[uuid.UUID]]:
        if not target_ids:
            return {}
        rows = self.session.execute(
            select(KnowledgeGraphEvidence.target_id, KnowledgeGraphEvidence.id).where(
                KnowledgeGraphEvidence.target_kind == target_kind,
                KnowledgeGraphEvidence.target_id.in_(target_ids),
            )
        )
        result: dict[uuid.UUID, list[uuid.UUID]] = {}
        for target_id, evidence_id in rows:
            result.setdefault(target_id, []).append(evidence_id)
        return result

    @staticmethod
    def _current_entity_predicate() -> ColumnElement[bool]:
        return and_(
            KnowledgeGraphEntity.is_current.is_(True),
            KnowledgeGraphEntity.expired_at.is_(None),
            or_(
                KnowledgeGraphEntity.valid_to.is_(None),
                KnowledgeGraphEntity.valid_to > func.now(),
            ),
            or_(
                KnowledgeGraphEntity.invalid_at.is_(None),
                KnowledgeGraphEntity.invalid_at > func.now(),
            ),
            KnowledgeGraphEntity.system_expired_at.is_(None),
            KnowledgeGraphEntity.lifecycle_state.in_(CURRENT_CONTEXT_STATES),
        )

    @staticmethod
    def _current_edge_predicate() -> ColumnElement[bool]:
        return and_(
            KnowledgeGraphEdge.is_current.is_(True),
            KnowledgeGraphEdge.expired_at.is_(None),
            or_(
                KnowledgeGraphEdge.valid_to.is_(None),
                KnowledgeGraphEdge.valid_to > func.now(),
            ),
            or_(
                KnowledgeGraphEdge.invalid_at.is_(None),
                KnowledgeGraphEdge.invalid_at > func.now(),
            ),
            KnowledgeGraphEdge.system_expired_at.is_(None),
            KnowledgeGraphEdge.lifecycle_state.in_(CURRENT_CONTEXT_STATES),
        )

    @staticmethod
    def _entity_has_evidence_predicate() -> ColumnElement[bool]:
        return exists().where(
            KnowledgeGraphEvidence.target_kind == "entity",
            KnowledgeGraphEvidence.target_id == KnowledgeGraphEntity.id,
            KnowledgeGraphEvidence.installation_id
            == KnowledgeGraphEntity.installation_id,
        )

    @staticmethod
    def _edge_has_evidence_predicate() -> ColumnElement[bool]:
        return exists().where(
            KnowledgeGraphEvidence.target_kind == "edge",
            KnowledgeGraphEvidence.target_id == KnowledgeGraphEdge.id,
            KnowledgeGraphEvidence.installation_id
            == KnowledgeGraphEdge.installation_id,
        )


def _returned_scopes(
    entities: Sequence[RetrievedGraphEntity],
    edges: Sequence[RetrievedGraphEdge],
) -> tuple[VisibilityScope, ...]:
    seen: set[tuple[str, str | None]] = set()
    scopes: list[VisibilityScope] = []
    for scope in [entity.visibility_scope for entity in entities] + [
        edge.visibility_scope for edge in edges
    ]:
        key = (scope.scope_type, scope.scope_id)
        if key in seen:
            continue
        seen.add(key)
        scopes.append(scope)
    return tuple(scopes)


def _retrieved_entity(
    row: KnowledgeGraphEntity,
    evidence_ids: Sequence[uuid.UUID],
) -> RetrievedGraphEntity:
    kind = provenance_kind(row.source_type, row.attrs_json)
    return RetrievedGraphEntity(
        id=row.id,
        entity_type=row.entity_type,
        canonical_key=row.canonical_key,
        display_name=row.display_name,
        source_type=row.source_type,
        visibility_scope=VisibilityScope(
            row.visibility_scope_type, row.visibility_scope_id
        ),
        lifecycle_state=row.lifecycle_state,
        confidence_score=row.confidence_score,
        confidence_reason=row.confidence_reason,
        provenance_kind=kind,
        provenance_label=provenance_label(kind),
        review_status=review_status(row.attrs_json, row.lifecycle_state),
        evidence_ids=tuple(evidence_ids),
        last_seen_at=row.last_reinforced_at or row.updated_at,
    )


def _retrieved_edge(
    row: KnowledgeGraphEdge,
    evidence_ids: Sequence[uuid.UUID],
) -> RetrievedGraphEdge:
    kind = provenance_kind(row.source_type, row.attrs_json)
    return RetrievedGraphEdge(
        id=row.id,
        source_entity_id=row.source_entity_id,
        target_entity_id=row.target_entity_id,
        relationship_type=row.relationship_type,
        source_type=row.source_type,
        visibility_scope=VisibilityScope(
            row.visibility_scope_type, row.visibility_scope_id
        ),
        lifecycle_state=row.lifecycle_state,
        confidence_score=row.confidence_score,
        confidence_reason=row.confidence_reason,
        provenance_kind=kind,
        provenance_label=provenance_label(kind),
        review_status=review_status(row.attrs_json, row.lifecycle_state),
        evidence_ids=tuple(evidence_ids),
    )


def _freshness_elapsed(
    row: KnowledgeGraphEntity | KnowledgeGraphEdge,
    now: datetime,
) -> bool:
    if row.freshness_window_days is None:
        return False
    if row.freshness_window_days <= 0:
        return True
    reference = row.last_reinforced_at or row.recorded_at or row.created_at
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=UTC)
    return reference + timedelta(days=row.freshness_window_days) < now


def _staleness_elapsed(
    row: KnowledgeGraphEntity | KnowledgeGraphEdge,
    now: datetime,
    default_stale_days: int | None,
) -> bool:
    if row.freshness_window_days is not None:
        return _freshness_elapsed(row, now)
    if default_stale_days is None:
        return False
    reference = row.last_reinforced_at or row.recorded_at or row.created_at
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=UTC)
    return reference + timedelta(days=default_stale_days) < now
