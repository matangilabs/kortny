"""Add witness activation epoch columns.

Revision ID: 0051
Revises: 0050
Create Date: 2026-06-21
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0051"
down_revision = "0050"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "observe_policies",
        sa.Column(
            "full_enabled_at", postgresql.TIMESTAMP(timezone=True), nullable=True
        ),
    )
    op.add_column(
        "installations",
        sa.Column(
            "digest_enabled_at", postgresql.TIMESTAMP(timezone=True), nullable=True
        ),
    )
    op.add_column(
        "installations",
        sa.Column("autopilot_enabled", sa.Boolean(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("installations", "autopilot_enabled")
    op.drop_column("installations", "digest_enabled_at")
    op.drop_column("observe_policies", "full_enabled_at")
