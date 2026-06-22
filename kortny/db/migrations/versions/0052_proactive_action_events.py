"""Add proactive_action_events audit log.

Revision ID: 0052
Revises: 0051
Create Date: 2026-06-21
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0052"
down_revision = "0051"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "proactive_action_events",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "candidate_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("witness_opportunity_candidates.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("installation_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("from_state", sa.Text(), nullable=True),
        sa.Column("to_state", sa.Text(), nullable=False),
        sa.Column("event_type", sa.Text(), nullable=False),
        sa.Column("reason_code", sa.Text(), nullable=True),
        sa.Column("actor_id", sa.Text(), nullable=True),
        sa.Column("policy_decision", sa.Text(), nullable=True),
        sa.Column("task_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("delivery_log_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column(
            "detail_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column("created_at", postgresql.TIMESTAMP(timezone=True), nullable=False),
    )
    op.create_index("idx_pae_candidate_id", "proactive_action_events", ["candidate_id"])
    op.create_index(
        "idx_pae_installation_id", "proactive_action_events", ["installation_id"]
    )
    op.create_index(
        "idx_pae_candidate_created_at",
        "proactive_action_events",
        ["candidate_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("idx_pae_candidate_created_at", table_name="proactive_action_events")
    op.drop_index("idx_pae_installation_id", table_name="proactive_action_events")
    op.drop_index("idx_pae_candidate_id", table_name="proactive_action_events")
    op.drop_table("proactive_action_events")
