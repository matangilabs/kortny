"""Composio tool cards: full-catalog sync for single-stage tool RAG.

Revision ID: 0038
Revises: 0037
Create Date: 2026-06-11

Backs HIG-222 (full-catalog Composio embedding sync). Creates the
``composio_tool_cards`` table holding one synced card per connected toolkit
tool, so per-task tool retrieval is a pure semantic rank over
``tool_embeddings`` (kind ``tool_card``) with no hot-path Composio HTTP for
candidate listing. Full input schemas are NOT stored here — they are fetched
lazily only for tools that survive selection.
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = "0038"
down_revision = "0037"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "composio_tool_cards",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "installation_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("installations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("toolkit_slug", sa.String(), nullable=False),
        sa.Column("tool_slug", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("side_effect", sa.String(), nullable=False),
        sa.Column("card_sha", sa.String(length=64), nullable=False),
        sa.Column(
            "synced_at",
            postgresql.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "created_at",
            postgresql.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            postgresql.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint(
            "side_effect in ('read', 'write', 'destructive')",
            name="ck_composio_tool_cards_side_effect",
        ),
        sa.UniqueConstraint(
            "installation_id",
            "tool_slug",
            name="uq_composio_tool_cards_installation_tool",
        ),
    )
    op.create_index(
        "idx_composio_tool_cards_toolkit",
        "composio_tool_cards",
        ["installation_id", "toolkit_slug"],
    )


def downgrade() -> None:
    op.drop_index(
        "idx_composio_tool_cards_toolkit",
        table_name="composio_tool_cards",
    )
    op.drop_table("composio_tool_cards")
