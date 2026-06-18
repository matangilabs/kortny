"""Project hub layer over the workspace knowledge graph (HIG-276, World Model).

A ``project`` is a first-class graph hub that ties together the channels (and
later integrations/people) that make up a unit of work, so Kortny can synthesize
across them ("how is Apollo going?"). Today graph retrieval anchors only on the
current channel/dm/user and the visibility model is strictly per-channel, so
cross-channel synthesis is impossible. This module adds:

- explicit project declaration (hub entity + ``project_includes_channel`` edges),
- a lookup from a channel to the project(s) it belongs to (for anchoring),
- the project's PUBLIC member-channel scopes (the audience-safe set that widens
  project-aware retrieval, per the reconciled HIG-276 design).

Increment 1 is public-channel-only and declaration-driven. Inference
(infer→propose→confirm) and private/DM cross-channel synthesis are later
increments. The hub is workspace-scoped (its existence/name is workspace
knowledge); edges to PUBLIC channels are workspace-scoped (cross-channel
visible), edges to PRIVATE channels are private-channel-scoped (only visible
when answering in that private channel), so a public reply never even sees a
project's private membership.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.orm import Session

from kortny.db.models import KnowledgeGraphEdge, KnowledgeGraphEntity
from kortny.knowledge_graph.scopes import (
    SCOPE_CHANNEL,
    SCOPE_PRIVATE_CHANNEL,
    VisibilityScope,
)
from kortny.knowledge_graph.service import EvidenceInput, GraphService

PROJECT_ENTITY_TYPE = "project"
PROJECT_INCLUDES_CHANNEL = "project_includes_channel"
PROJECT_INCLUDES_ENTITY = "project_includes_entity"
_CHANNEL_ENTITY_TYPE = "channel"
_CURRENT_STATES = ("active", "confirmed")


def project_slug(name: str) -> str:
    """Slugify a human project name into a stable canonical-key suffix."""

    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().casefold()).strip("-")
    return slug or "project"


def project_canonical_key(slug: str) -> str:
    return f"project:{slug}"


@dataclass(frozen=True)
class DeclareProjectResult:
    project: KnowledgeGraphEntity
    linked_channel_ids: tuple[str, ...]
    skipped_channel_ids: tuple[str, ...] = field(default=())
    created: bool = False


class ProjectGraphService:
    """Manage project hub entities and their channel membership edges."""

    def __init__(self, session: Session, *, graph: GraphService | None = None) -> None:
        self.session = session
        self.graph = graph or GraphService(session)

    # -- declaration ------------------------------------------------------

    def declare_project(
        self,
        *,
        installation_id: uuid.UUID,
        name: str,
        channel_ids: Sequence[str],
        evidence: EvidenceInput | None = None,
        slug: str | None = None,
        source_type: str = "user_explicit",
        lifecycle_state: str = "confirmed",
        confidence_score: Decimal = Decimal("1.000"),
        confidence_reason: str = "Project boundary declared by a user.",
        reinforce: bool = False,
    ) -> DeclareProjectResult:
        """Create/update a project hub and link the given channels (idempotent).

        Defaults create a human-declared (confirmed) hub. Implicit inference
        passes ``source_type='agent_inferred'``, ``lifecycle_state='active'``, a
        cluster-derived ``confidence_score``, a stable ``slug`` (so re-detection
        finds the same hub), and ``reinforce=True`` (so a re-detected hub gets
        its reinforcement_count/last_reinforced bumped — the brain learns
        continuously and self-corrects, no confirmation gate).

        Only channels Kortny already knows (a ``channel`` hub entity exists) are
        linked; unknown channels are skipped and reported. Edge visibility
        follows the channel's privacy: public → workspace (cross-channel
        visible), private → private_channel (gated to that channel).
        """

        resolved_slug = slug or project_slug(name)
        canonical_key = project_canonical_key(resolved_slug)
        project = self._current_entity(
            installation_id, canonical_key, PROJECT_ENTITY_TYPE
        )
        created = project is None
        if project is None:
            project = self.graph.create_entity(
                installation_id=installation_id,
                entity_type=PROJECT_ENTITY_TYPE,
                canonical_key=canonical_key,
                display_name=name.strip() or resolved_slug,
                visibility_scope=VisibilityScope.workspace(),
                source_type=source_type,
                lifecycle_state=lifecycle_state,
                confidence_score=confidence_score,
                confidence_reason=confidence_reason,
                evidence=evidence,
            )
        elif reinforce:
            project.reinforcement_count += 1
            project.last_reinforced_at = datetime.now(UTC)
            self.session.flush()

        linked: list[str] = []
        skipped: list[str] = []
        for channel_id in dict.fromkeys(channel_ids):
            channel_entity = self._current_entity(
                installation_id,
                _channel_canonical_key(channel_id),
                _CHANNEL_ENTITY_TYPE,
            )
            if channel_entity is None:
                skipped.append(channel_id)
                continue
            self._ensure_includes_edge(
                installation_id=installation_id,
                project=project,
                channel_entity=channel_entity,
                evidence=evidence,
            )
            linked.append(channel_id)

        return DeclareProjectResult(
            project=project,
            linked_channel_ids=tuple(linked),
            skipped_channel_ids=tuple(skipped),
            created=created,
        )

    def _ensure_includes_edge(
        self,
        *,
        installation_id: uuid.UUID,
        project: KnowledgeGraphEntity,
        channel_entity: KnowledgeGraphEntity,
        evidence: EvidenceInput | None,
    ) -> None:
        existing = self.session.scalars(
            select(KnowledgeGraphEdge).where(
                KnowledgeGraphEdge.installation_id == installation_id,
                KnowledgeGraphEdge.source_entity_id == project.id,
                KnowledgeGraphEdge.target_entity_id == channel_entity.id,
                KnowledgeGraphEdge.relationship_type == PROJECT_INCLUDES_CHANNEL,
                KnowledgeGraphEdge.lifecycle_state.in_(_CURRENT_STATES),
            )
        ).first()
        if existing is not None:
            return
        # Public-channel membership is workspace-visible (so project answers
        # reach it cross-channel); private-channel membership is gated to that
        # private channel so a public reply never sees it.
        if channel_entity.visibility_scope_type == SCOPE_PRIVATE_CHANNEL:
            edge_scope = VisibilityScope.private_channel(
                channel_entity.visibility_scope_id or ""
            )
        else:
            edge_scope = VisibilityScope.workspace()
        self.graph.create_edge(
            installation_id=installation_id,
            source_entity_id=project.id,
            target_entity_id=channel_entity.id,
            relationship_type=PROJECT_INCLUDES_CHANNEL,
            visibility_scope=edge_scope,
            source_type="user_explicit",
            lifecycle_state="confirmed",
            confidence_score=Decimal("1.000"),
            confidence_reason="Channel declared part of the project by a user.",
            evidence=evidence,
        )

    def link_project_entities(
        self,
        *,
        installation_id: uuid.UUID,
        project: KnowledgeGraphEntity,
        entity_ids: Sequence[uuid.UUID],
        evidence: EvidenceInput | None = None,
    ) -> tuple[uuid.UUID, ...]:
        """Link a project hub to its constituent entities (HIG-276 increment 2).

        Adds ``project_includes_entity`` edges so retrieval reaches the project's
        topics/decisions/commitments directly. The edge inherits each entity's
        own visibility scope, so a public-channel topic widens project answers
        while a private one stays gated — matching the audience-safe model.
        Idempotent; returns the entity ids actually linked.
        """

        wanted = tuple(dict.fromkeys(entity_ids))
        if not wanted:
            return ()
        rows = self.session.scalars(
            select(KnowledgeGraphEntity).where(
                KnowledgeGraphEntity.installation_id == installation_id,
                KnowledgeGraphEntity.id.in_(wanted),
                KnowledgeGraphEntity.lifecycle_state.in_(_CURRENT_STATES),
            )
        ).all()
        linked: list[uuid.UUID] = []
        for entity in rows:
            if entity.id == project.id:
                continue
            existing = self.session.scalars(
                select(KnowledgeGraphEdge).where(
                    KnowledgeGraphEdge.installation_id == installation_id,
                    KnowledgeGraphEdge.source_entity_id == project.id,
                    KnowledgeGraphEdge.target_entity_id == entity.id,
                    KnowledgeGraphEdge.relationship_type == PROJECT_INCLUDES_ENTITY,
                    KnowledgeGraphEdge.lifecycle_state.in_(_CURRENT_STATES),
                )
            ).first()
            if existing is not None:
                linked.append(entity.id)
                continue
            self.graph.create_edge(
                installation_id=installation_id,
                source_entity_id=project.id,
                target_entity_id=entity.id,
                relationship_type=PROJECT_INCLUDES_ENTITY,
                visibility_scope=VisibilityScope(
                    entity.visibility_scope_type, entity.visibility_scope_id
                ),
                source_type="user_explicit",
                lifecycle_state="confirmed",
                confidence_score=Decimal("1.000"),
                confidence_reason="Entity confirmed part of the project.",
                evidence=evidence,
            )
            linked.append(entity.id)
        return tuple(linked)

    # -- lookups (anchoring + authorization) ------------------------------

    def projects_for_channel(
        self, *, installation_id: uuid.UUID, channel_id: str
    ) -> tuple[KnowledgeGraphEntity, ...]:
        """Active project hubs that include the given channel."""

        channel_entity = self._current_entity(
            installation_id, _channel_canonical_key(channel_id), _CHANNEL_ENTITY_TYPE
        )
        if channel_entity is None:
            return ()
        rows = self.session.scalars(
            select(KnowledgeGraphEntity)
            .join(
                KnowledgeGraphEdge,
                KnowledgeGraphEdge.source_entity_id == KnowledgeGraphEntity.id,
            )
            .where(
                KnowledgeGraphEntity.installation_id == installation_id,
                KnowledgeGraphEntity.entity_type == PROJECT_ENTITY_TYPE,
                KnowledgeGraphEntity.lifecycle_state.in_(_CURRENT_STATES),
                KnowledgeGraphEdge.relationship_type == PROJECT_INCLUDES_CHANNEL,
                KnowledgeGraphEdge.target_entity_id == channel_entity.id,
                KnowledgeGraphEdge.lifecycle_state.in_(_CURRENT_STATES),
            )
        ).all()
        return tuple(dict.fromkeys(rows))

    def public_member_channel_scopes(
        self, *, installation_id: uuid.UUID, project_ids: Sequence[uuid.UUID]
    ) -> tuple[VisibilityScope, ...]:
        """PUBLIC member-channel scopes for the given projects (audience-safe).

        These are the extra scopes a project answer may draw from; private member
        channels are deliberately excluded (their inclusion needs per-user
        membership + governance, a later increment).
        """

        if not project_ids:
            return ()
        rows = self.session.scalars(
            select(KnowledgeGraphEntity)
            .join(
                KnowledgeGraphEdge,
                KnowledgeGraphEdge.target_entity_id == KnowledgeGraphEntity.id,
            )
            .where(
                KnowledgeGraphEntity.installation_id == installation_id,
                KnowledgeGraphEntity.entity_type == _CHANNEL_ENTITY_TYPE,
                KnowledgeGraphEntity.visibility_scope_type == SCOPE_CHANNEL,
                KnowledgeGraphEntity.lifecycle_state.in_(_CURRENT_STATES),
                KnowledgeGraphEdge.relationship_type == PROJECT_INCLUDES_CHANNEL,
                KnowledgeGraphEdge.source_entity_id.in_(tuple(project_ids)),
                KnowledgeGraphEdge.lifecycle_state.in_(_CURRENT_STATES),
            )
        ).all()
        scopes: list[VisibilityScope] = []
        seen: set[str] = set()
        for row in rows:
            scope_id = row.visibility_scope_id
            if scope_id and scope_id not in seen:
                seen.add(scope_id)
                scopes.append(VisibilityScope.channel(scope_id))
        return tuple(scopes)

    def _current_entity(
        self, installation_id: uuid.UUID, canonical_key: str, entity_type: str
    ) -> KnowledgeGraphEntity | None:
        return self.session.scalars(
            select(KnowledgeGraphEntity)
            .where(
                KnowledgeGraphEntity.installation_id == installation_id,
                KnowledgeGraphEntity.canonical_key == canonical_key,
                KnowledgeGraphEntity.entity_type == entity_type,
                KnowledgeGraphEntity.lifecycle_state.in_(_CURRENT_STATES),
            )
            .order_by(KnowledgeGraphEntity.updated_at.desc())
        ).first()


def project_anchors_and_scopes(
    session: Session, *, installation_id: uuid.UUID, channel_id: str
) -> tuple[tuple[str, ...], tuple[VisibilityScope, ...]]:
    """Project hub anchor keys + audience-safe extra scopes for a channel.

    Shared by context assembly and the ``query_workspace_graph`` tool so both
    retrieval paths synthesize across a project identically (HIG-276). Returns
    empty tuples when the channel is not part of any project.
    """

    service = ProjectGraphService(session)
    projects = service.projects_for_channel(
        installation_id=installation_id, channel_id=channel_id
    )
    if not projects:
        return ((), ())
    anchor_keys = tuple(project.canonical_key for project in projects)
    scopes = service.public_member_channel_scopes(
        installation_id=installation_id,
        project_ids=[project.id for project in projects],
    )
    return (anchor_keys, scopes)


def _channel_canonical_key(channel_id: str) -> str:
    return f"slack_channel:{channel_id}"
