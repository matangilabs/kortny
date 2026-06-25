"""Add enriched_description to composio_tool_cards (HIG-295 Step A).

Revision ID: 0056
Revises: 0055
Create Date: 2026-06-25
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0056"
down_revision = "0055"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "composio_tool_cards",
        sa.Column("enriched_description", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("composio_tool_cards", "enriched_description")
