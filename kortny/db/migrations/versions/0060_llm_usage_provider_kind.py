"""Add provider_kind, provider_account_id, cost_source to llm_usage; make provider nullable.

Revision ID: 0060
Revises: 0059
Create Date: 2026-06-26 00:00:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

# revision identifiers, used by Alembic.
revision = "0060"
down_revision = "0059"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "llm_usage",
        sa.Column("provider_kind", sa.String(), nullable=True),
    )
    op.add_column(
        "llm_usage",
        sa.Column("provider_account_id", UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "llm_usage",
        sa.Column("cost_source", sa.String(), nullable=True),
    )
    op.alter_column("llm_usage", "provider", nullable=True)


def downgrade() -> None:
    op.drop_column("llm_usage", "cost_source")
    op.drop_column("llm_usage", "provider_account_id")
    op.drop_column("llm_usage", "provider_kind")
    op.alter_column("llm_usage", "provider", nullable=False)
