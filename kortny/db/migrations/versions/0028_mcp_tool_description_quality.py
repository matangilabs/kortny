"""MCP tool description quality: score, enriched description, sha256 gate.

Revision ID: 0028
Revises: 0027
Create Date: 2026-06-10

Backs HIG-215. Adds three nullable columns to ``mcp_server_tools`` so that the
discovery path can score each tool description, enrich poor ones via LLM, and
skip re-scoring when the underlying content has not changed (sha256 gate).

Columns added to ``mcp_server_tools``:
  description_quality_score  -- NUMERIC(4,3), nullable; 0–1 rubric score
  enriched_description       -- TEXT, nullable; LLM-improved description
  description_sha256         -- VARCHAR(64), nullable; hex SHA-256 of raw description
"""

import sqlalchemy as sa
from alembic import op

# revision identifiers
revision = "0028"
down_revision = "0027"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "mcp_server_tools",
        sa.Column(
            "description_quality_score",
            sa.Numeric(precision=4, scale=3),
            nullable=True,
        ),
    )
    op.add_column(
        "mcp_server_tools",
        sa.Column("enriched_description", sa.Text(), nullable=True),
    )
    op.add_column(
        "mcp_server_tools",
        sa.Column("description_sha256", sa.String(length=64), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("mcp_server_tools", "description_sha256")
    op.drop_column("mcp_server_tools", "enriched_description")
    op.drop_column("mcp_server_tools", "description_quality_score")
