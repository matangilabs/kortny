"""Add installations.primary_admin_user_id.

The Slack user who installed Kortny, captured at the install-intro DM. Ambient
passes that have no originating user (e.g. the org-profile proposer, HIG-271)
DM this admin to confirm workspace-level facts.

Revision ID: 0042
Revises: 0041
Create Date: 2026-06-16
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0042"
down_revision = "0041"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "installations",
        sa.Column("primary_admin_user_id", sa.String(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("installations", "primary_admin_user_id")
