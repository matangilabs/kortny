"""Add Composio trigger subscriptions and events tables.

Revision ID: 0058
Revises: 0057
Create Date: 2026-06-26 00:00:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP, UUID

# revision identifiers, used by Alembic.
revision = "0058"
down_revision = "0057"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "composio_trigger_subscriptions",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("installation_id", UUID(as_uuid=True), nullable=False),
        sa.Column("composio_connection_id", UUID(as_uuid=True), nullable=True),
        sa.Column("connected_account_id", sa.String, nullable=False),
        sa.Column("composio_user_id", sa.String, nullable=False),
        sa.Column("owner_slack_user_id", sa.String, nullable=True),
        sa.Column("toolkit_slug", sa.String, nullable=False),
        sa.Column("trigger_slug", sa.String, nullable=False),
        sa.Column("composio_trigger_id", sa.String, nullable=True),
        sa.Column(
            "trigger_config_json",
            JSONB,
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "status",
            sa.String,
            nullable=False,
            server_default=sa.text("'active'"),
        ),
        sa.Column("target_scope_type", sa.String, nullable=True),
        sa.Column("target_scope_id", sa.String, nullable=True),
        sa.Column(
            "min_importance",
            sa.Numeric(4, 3),
            nullable=False,
            server_default=sa.text("0.0"),
        ),
        sa.Column(
            "cooldown_seconds",
            sa.Integer,
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column("daily_cap", sa.Integer, nullable=True),
        sa.Column(
            "digest_enabled",
            sa.Boolean,
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column(
            "created_at",
            TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.ForeignKeyConstraint(
            ["installation_id"],
            ["installations.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["composio_connection_id"],
            ["composio_connections.id"],
            ondelete="SET NULL",
        ),
        sa.CheckConstraint(
            "status in ('active', 'paused', 'disabled')",
            name="ck_composio_trigger_subscriptions_status",
        ),
        sa.UniqueConstraint(
            "installation_id",
            "connected_account_id",
            "trigger_slug",
            name="uq_composio_trigger_subscriptions_account_trigger",
        ),
    )
    op.create_index(
        "idx_composio_trigger_subscriptions_installation",
        "composio_trigger_subscriptions",
        ["installation_id", "status"],
    )
    op.create_index(
        "idx_composio_trigger_subscriptions_account",
        "composio_trigger_subscriptions",
        ["installation_id", "connected_account_id", "trigger_slug"],
    )

    op.create_table(
        "composio_trigger_events",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("installation_id", UUID(as_uuid=True), nullable=False),
        sa.Column("subscription_id", UUID(as_uuid=True), nullable=True),
        sa.Column("composio_trigger_id", sa.String, nullable=True),
        sa.Column("event_id", sa.String, nullable=False),
        sa.Column("trigger_slug", sa.String, nullable=False),
        sa.Column("connected_account_id", sa.String, nullable=True),
        sa.Column("composio_user_id", sa.String, nullable=True),
        sa.Column(
            "raw_payload_json",
            JSONB,
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("importance_score", sa.Numeric(5, 4), nullable=True),
        sa.Column("decision", sa.String, nullable=True),
        sa.Column("decision_reason", sa.Text, nullable=True),
        sa.Column(
            "received_at",
            TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.ForeignKeyConstraint(
            ["installation_id"],
            ["installations.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["subscription_id"],
            ["composio_trigger_subscriptions.id"],
            ondelete="SET NULL",
        ),
        sa.CheckConstraint(
            "decision in ('ask', 'silent', 'digest', 'duplicate', 'unmatched')",
            name="ck_composio_trigger_events_decision",
        ),
        sa.UniqueConstraint(
            "installation_id",
            "trigger_slug",
            "event_id",
            name="uq_composio_trigger_events_dedup",
        ),
    )
    op.create_index(
        "idx_composio_trigger_events_installation",
        "composio_trigger_events",
        ["installation_id", "received_at"],
    )
    op.create_index(
        "idx_composio_trigger_events_subscription",
        "composio_trigger_events",
        ["subscription_id", "received_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "idx_composio_trigger_events_subscription",
        table_name="composio_trigger_events",
    )
    op.drop_index(
        "idx_composio_trigger_events_installation",
        table_name="composio_trigger_events",
    )
    op.drop_table("composio_trigger_events")

    op.drop_index(
        "idx_composio_trigger_subscriptions_account",
        table_name="composio_trigger_subscriptions",
    )
    op.drop_index(
        "idx_composio_trigger_subscriptions_installation",
        table_name="composio_trigger_subscriptions",
    )
    op.drop_table("composio_trigger_subscriptions")
