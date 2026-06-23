"""Add outcome columns to witness_opportunity_candidates.

Revision ID: 0053
Revises: 0052
Create Date: 2026-06-21
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0053"
down_revision = "0052"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "witness_opportunity_candidates",
        sa.Column("task_status", sa.Text(), nullable=True),
    )
    op.add_column(
        "witness_opportunity_candidates",
        sa.Column(
            "task_finished_at",
            postgresql.TIMESTAMP(timezone=True),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column("witness_opportunity_candidates", "task_finished_at")
    op.drop_column("witness_opportunity_candidates", "task_status")
