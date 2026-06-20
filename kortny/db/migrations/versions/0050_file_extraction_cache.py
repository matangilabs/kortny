"""Add file_extraction_cache table (HIG-279 slice 3b-1).

Revision ID: 0050
Revises: 0049
Create Date: 2026-06-19
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0050"
down_revision = "0049"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "file_extraction_cache",
        sa.Column("content_sha256", sa.String(64), nullable=False),
        sa.Column("backend", sa.String(), nullable=False),
        sa.Column("extraction_supported", sa.Boolean(), nullable=False),
        sa.Column("extracted_text", sa.Text(), nullable=True),
        sa.Column(
            "truncated",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column("page_count", sa.Integer(), nullable=True),
        sa.Column("byte_size", sa.BigInteger(), nullable=False),
        sa.Column(
            "warnings",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "last_accessed_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("content_sha256"),
    )
    op.create_index(
        "idx_file_extraction_cache_last_accessed",
        "file_extraction_cache",
        ["last_accessed_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "idx_file_extraction_cache_last_accessed",
        table_name="file_extraction_cache",
    )
    op.drop_table("file_extraction_cache")
