"""Add interactive_actions for Block Kit interactive actions (HIG-255 slice 2).

A server-owned record for each interactive Slack control (approve/reject/retry/
run_again). The Slack button carries only an opaque raw key; this table stores
the key's HMAC hash plus the actor/workspace/container/target needed to
authorize a click and the lifecycle to make it idempotent.

Revision ID: 0046
Revises: 0045
Create Date: 2026-06-19
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0046"
down_revision = "0045"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "interactive_actions",
        sa.Column(
            "id",
            sa.UUID(),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("installation_id", sa.UUID(), nullable=False),
        sa.Column("task_id", sa.UUID(), nullable=True),
        sa.Column("result_task_id", sa.UUID(), nullable=True),
        sa.Column("action_key_hash", sa.String(), nullable=False),
        sa.Column("action_kind", sa.String(), nullable=False),
        sa.Column("route", sa.String(), nullable=False),
        sa.Column(
            "status",
            sa.String(),
            server_default=sa.text("'pending_send'"),
            nullable=False,
        ),
        sa.Column("target_type", sa.String(), nullable=False),
        sa.Column("target_id", sa.String(), nullable=False),
        sa.Column(
            "payload_json",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("slack_team_id", sa.String(), nullable=True),
        sa.Column("slack_channel_id", sa.String(), nullable=True),
        sa.Column("slack_thread_ts", sa.String(), nullable=True),
        sa.Column("slack_message_ts", sa.String(), nullable=True),
        sa.Column("slack_block_id", sa.String(), nullable=True),
        sa.Column("slack_action_id", sa.String(), nullable=True),
        sa.Column("created_for_user_id", sa.String(), nullable=True),
        sa.Column("allowed_user_id", sa.String(), nullable=True),
        sa.Column("required_role", sa.String(), nullable=True),
        sa.Column("allowed_channel_id", sa.String(), nullable=True),
        sa.Column("expires_at", postgresql.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("sent_at", postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("consumed_at", postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("consumed_by_user_id", sa.String(), nullable=True),
        sa.Column("last_denied_at", postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.Column(
            "denied_count", sa.Integer(), server_default=sa.text("0"), nullable=False
        ),
        sa.Column(
            "metadata_json",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["installation_id"], ["installations.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["task_id"], ["tasks.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["result_task_id"], ["tasks.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("action_key_hash"),
    )
    op.create_index(
        "idx_interactive_actions_installation",
        "interactive_actions",
        ["installation_id"],
        unique=False,
    )
    op.create_index(
        "idx_interactive_actions_task",
        "interactive_actions",
        ["task_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("idx_interactive_actions_task", table_name="interactive_actions")
    op.drop_index(
        "idx_interactive_actions_installation", table_name="interactive_actions"
    )
    op.drop_table("interactive_actions")
