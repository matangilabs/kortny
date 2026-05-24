"""add model tier to llm usage

Revision ID: 0003
Revises: 0002
Create Date: 2026-05-24
"""

import sqlalchemy as sa
from alembic import op

# revision identifiers
revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("llm_usage", sa.Column("model_tier", sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column("llm_usage", "model_tier")
