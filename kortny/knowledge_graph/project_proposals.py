"""Lifecycle for inferred project proposals (HIG-276 increment 2).

Implicit project inference proposes a candidate project to a user; on confirm it
becomes a real ``project`` graph hub (channels + entity links). A project confirm
is a graph mutation, not a memory fact, so it gets its own lifecycle here rather
than reusing the WorkspaceState proposal path.

Privacy: only ``public_summary`` / ``public_evidence_json`` are safe to render in
Slack. ``private_evidence_json`` is persisted for governance and must never be
surfaced in the proposal DM (the recipient may not see those private surfaces).
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.orm import Session

from kortny.db.models import KnowledgeGraphEntity, ProjectProposal
from kortny.knowledge_graph.projects import ProjectGraphService
from kortny.knowledge_graph.service import EvidenceInput, GraphService

PROPOSAL_SOURCE = "project_inference"
_ACTIVE_STATES = ("proposed",)


class ProjectProposalService:
    """Create, gate, and confirm/reject inferred project proposals."""

    def __init__(self, session: Session, *, graph: GraphService | None = None) -> None:
        self.session = session
        self.graph = graph or GraphService(session)
        self.projects = ProjectGraphService(session, graph=self.graph)

    def has_recent_proposal(
        self,
        *,
        installation_id: uuid.UUID,
        dedupe_key: str,
        now: datetime | None = None,
    ) -> bool:
        """True if an open proposal exists or a cooldown has not yet elapsed.

        Prevents re-nagging on the same candidate every consolidator run.
        """

        effective_now = now or datetime.now(UTC)
        rows = self.session.scalars(
            select(ProjectProposal).where(
                ProjectProposal.installation_id == installation_id,
                ProjectProposal.dedupe_key == dedupe_key,
            )
        ).all()
        for row in rows:
            if row.status in _ACTIVE_STATES:
                return True
            if row.cooldown_until is not None and row.cooldown_until > effective_now:
                return True
        return False

    def create_proposal(
        self,
        *,
        installation_id: uuid.UUID,
        slug: str,
        title: str,
        public_summary: str,
        proposed_channel_ids: Sequence[str],
        proposed_entity_ids: Sequence[uuid.UUID],
        public_evidence: Sequence[dict],
        private_evidence: Sequence[dict],
        dedupe_key: str,
        confidence_score: Decimal,
        confidence_reason: str | None = None,
        proposed_to_user_id: str | None = None,
        cooldown_until: datetime | None = None,
    ) -> ProjectProposal:
        proposal = ProjectProposal(
            installation_id=installation_id,
            slug=slug,
            title=title,
            public_summary=public_summary,
            proposed_channel_ids=list(proposed_channel_ids),
            proposed_entity_ids=[str(eid) for eid in proposed_entity_ids],
            public_evidence_json=list(public_evidence),
            private_evidence_json=list(private_evidence),
            has_private_signal=bool(private_evidence),
            status="proposed",
            dedupe_key=dedupe_key,
            confidence_score=confidence_score,
            confidence_reason=confidence_reason,
            proposed_to_user_id=proposed_to_user_id,
            cooldown_until=cooldown_until,
        )
        self.session.add(proposal)
        self.session.flush()
        return proposal

    def record_prompt(
        self, proposal: ProjectProposal, *, channel_id: str, message_ts: str
    ) -> None:
        """Remember where the proposal DM landed, for reaction-based confirm."""

        proposal.prompt_channel_id = channel_id
        proposal.prompt_message_ts = message_ts
        self.session.flush()

    def find_by_prompt(
        self,
        *,
        installation_id: uuid.UUID,
        channel_id: str,
        message_ts: str,
    ) -> ProjectProposal | None:
        return self.session.scalars(
            select(ProjectProposal).where(
                ProjectProposal.installation_id == installation_id,
                ProjectProposal.prompt_channel_id == channel_id,
                ProjectProposal.prompt_message_ts == message_ts,
            )
        ).first()

    def confirm(
        self,
        proposal: ProjectProposal,
        *,
        confirmed_by_user_id: str | None = None,
        now: datetime | None = None,
    ) -> KnowledgeGraphEntity:
        """Materialize a confirmed proposal into a project hub + entity links.

        Idempotent on an already-confirmed proposal (returns its hub). Builds the
        hub from the public summary as evidence so it is retrievable.
        """

        effective_now = now or datetime.now(UTC)
        evidence = EvidenceInput(
            source_type="user_explicit",
            extracted_by=PROPOSAL_SOURCE,
            # Stable provenance ref (the evidence row needs a source reference).
            source_url=f"kortny://project_proposal/{proposal.id}",
            raw_snippet=proposal.public_summary,
        )
        declared = self.projects.declare_project(
            installation_id=proposal.installation_id,
            name=proposal.title,
            channel_ids=[str(cid) for cid in proposal.proposed_channel_ids],
            evidence=evidence,
        )
        entity_ids = [uuid.UUID(str(eid)) for eid in proposal.proposed_entity_ids]
        self.projects.link_project_entities(
            installation_id=proposal.installation_id,
            project=declared.project,
            entity_ids=entity_ids,
            evidence=evidence,
        )
        proposal.status = "confirmed"
        proposal.confirmed_by_user_id = confirmed_by_user_id
        proposal.confirmed_at = effective_now
        proposal.project_entity_id = declared.project.id
        self.session.flush()
        return declared.project

    def reject(
        self,
        proposal: ProjectProposal,
        *,
        rejected_by_user_id: str | None = None,
        cooldown_until: datetime | None = None,
    ) -> None:
        proposal.status = "rejected"
        proposal.confirmed_by_user_id = rejected_by_user_id
        if cooldown_until is not None:
            proposal.cooldown_until = cooldown_until
        self.session.flush()
