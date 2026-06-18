"""Add tasks.routing_quality + routing_quality_score (HIG-221 learning loop).

A deterministic per-task outcome label computed at completion, so routing
decisions can be scored and fed back (trace->eval, promotion gate, priors).

Revision ID: 0045
Revises: 0044
Create Date: 2026-06-18
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0045"
down_revision = "0044"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("tasks", sa.Column("routing_quality", sa.String(), nullable=True))
    op.add_column(
        "tasks",
        sa.Column("routing_quality_score", sa.Numeric(4, 3), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("tasks", "routing_quality_score")
    op.drop_column("tasks", "routing_quality")
