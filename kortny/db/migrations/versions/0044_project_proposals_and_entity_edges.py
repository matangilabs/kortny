"""Project inference: project_includes_entity edges + project_proposals (HIG-276).

Increment 2 (implicit project inference) needs: a graph edge type linking a
project hub to its constituent entities (topics/decisions/commitments), and a
proposal lifecycle table so inferred boundaries are confirmed by a human before
they become real project hubs.

Revision ID: 0044
Revises: 0043
Create Date: 2026-06-17
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0044"
down_revision = "0043"
branch_labels = None
depends_on = None

_PREV_RELATIONSHIP_TYPES = (
    "'member_of', 'maps_to', 'works_on', 'owns', 'belongs_to', "
    "'referenced_in', 'made_in', 'affects', 'relates_to', 'available_for', "
    "'project_includes_channel'"
)
_NEW_RELATIONSHIP_TYPES = _PREV_RELATIONSHIP_TYPES + ", 'project_includes_entity'"
_CONSTRAINT = "ck_kg_edges_relationship_type"


def upgrade() -> None:
    op.drop_constraint(_CONSTRAINT, "kg_edges", type_="check")
    op.create_check_constraint(
        _CONSTRAINT,
        "kg_edges",
        f"relationship_type in ({_NEW_RELATIONSHIP_TYPES})",
    )

    op.create_table(
        "project_proposals",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "installation_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("installations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("slug", sa.String(), nullable=False),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("public_summary", sa.Text(), nullable=False),
        sa.Column(
            "proposed_channel_ids",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "proposed_entity_ids",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "public_evidence_json",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "private_evidence_json",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "has_private_signal",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column(
            "status",
            sa.String(),
            nullable=False,
            server_default=sa.text("'proposed'"),
        ),
        sa.Column("proposed_to_user_id", sa.String(), nullable=True),
        sa.Column("prompt_channel_id", sa.String(), nullable=True),
        sa.Column("prompt_message_ts", sa.String(), nullable=True),
        sa.Column("dedupe_key", sa.String(), nullable=False),
        sa.Column(
            "confidence_score",
            sa.Numeric(4, 3),
            nullable=False,
            server_default=sa.text("0.500"),
        ),
        sa.Column("confidence_reason", sa.Text(), nullable=True),
        sa.Column(
            "cooldown_until",
            postgresql.TIMESTAMP(timezone=True),
            nullable=True,
        ),
        sa.Column("confirmed_by_user_id", sa.String(), nullable=True),
        sa.Column(
            "confirmed_at",
            postgresql.TIMESTAMP(timezone=True),
            nullable=True,
        ),
        sa.Column(
            "project_entity_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("kg_entities.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            postgresql.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            postgresql.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint(
            "status in ('proposed', 'confirmed', 'rejected', 'expired', 'superseded')",
            name="ck_project_proposals_status",
        ),
    )
    op.create_index(
        "idx_project_proposals_installation",
        "project_proposals",
        ["installation_id"],
    )
    op.create_index(
        "idx_project_proposals_dedupe",
        "project_proposals",
        ["installation_id", "dedupe_key"],
    )


def downgrade() -> None:
    op.drop_table("project_proposals")
    op.execute(
        "DELETE FROM kg_edges WHERE relationship_type = 'project_includes_entity'"
    )
    op.drop_constraint(_CONSTRAINT, "kg_edges", type_="check")
    op.create_check_constraint(
        _CONSTRAINT,
        "kg_edges",
        f"relationship_type in ({_PREV_RELATIONSHIP_TYPES})",
    )
