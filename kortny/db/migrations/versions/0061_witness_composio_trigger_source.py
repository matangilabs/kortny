"""Add composio_trigger to witness_opportunity_candidates source_type constraint.

Revision ID: 0061
Revises: 0060
Create Date: 2026-07-02
"""

from __future__ import annotations

from alembic import op

revision = "0061"
down_revision = "0060"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint(
        "ck_witness_opportunity_candidates_source_type",
        "witness_opportunity_candidates",
        type_="check",
    )
    op.create_check_constraint(
        "ck_witness_opportunity_candidates_source_type",
        "witness_opportunity_candidates",
        "source_type in "
        "('channel_profile', 'knowledge_graph', 'task_summary', "
        "'scheduled_witness', 'manual', 'composio_trigger')",
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_witness_opportunity_candidates_source_type",
        "witness_opportunity_candidates",
        type_="check",
    )
    # Remap any rows ask-mode has already projected: 'composio_trigger' is not
    # in the old allowed set, so recreating the constraint below would fail
    # against existing rows. 'manual' is the closest legacy bucket (no source
    # FKs, no profile); losing the finer-grained provenance label is an
    # accepted downgrade tradeoff.
    op.execute(
        "UPDATE witness_opportunity_candidates "
        "SET source_type = 'manual' "
        "WHERE source_type = 'composio_trigger'"
    )
    op.create_check_constraint(
        "ck_witness_opportunity_candidates_source_type",
        "witness_opportunity_candidates",
        "source_type in "
        "('channel_profile', 'knowledge_graph', 'task_summary', "
        "'scheduled_witness', 'manual')",
    )
