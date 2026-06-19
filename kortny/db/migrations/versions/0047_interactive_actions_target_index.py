"""Composite index for interactive_actions sibling lookup (HIG-255 slice 2).

supersede_siblings + candidate lookup scan by (installation_id, target_type,
target_id, status); add a covering index so retiring a target's sibling buttons
doesn't table-scan.

Revision ID: 0047
Revises: 0046
Create Date: 2026-06-19
"""

from __future__ import annotations

from alembic import op

revision = "0047"
down_revision = "0046"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index(
        "idx_interactive_actions_target",
        "interactive_actions",
        ["installation_id", "target_type", "target_id", "status"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("idx_interactive_actions_target", table_name="interactive_actions")
