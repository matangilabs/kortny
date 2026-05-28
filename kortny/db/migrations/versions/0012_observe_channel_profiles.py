"""observe channel profiles

Revision ID: 0012
Revises: 0011
Create Date: 2026-05-27

Stores durable, staleness-aware channel profile summaries derived from observe
assessments. Raw observations and membership state remain separate.
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = "0012"
down_revision = "0011"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "observe_channel_profiles",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("installation_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("channel_id", sa.String(), nullable=False),
        sa.Column(
            "profile_status",
            sa.String(),
            server_default=sa.text("'active'"),
            nullable=False,
        ),
        sa.Column(
            "profile_version",
            sa.Integer(),
            server_default=sa.text("1"),
            nullable=False,
        ),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column(
            "profile_json",
            postgresql.JSONB(),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "assumptions_json",
            postgresql.JSONB(),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "evidence_refs_json",
            postgresql.JSONB(),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column("confidence_score", sa.Numeric(4, 3), nullable=True),
        sa.Column("confidence_reason", sa.Text(), nullable=True),
        sa.Column(
            "fresh_window_days",
            sa.Integer(),
            server_default=sa.text("30"),
            nullable=False,
        ),
        sa.Column(
            "archive_window_days",
            sa.Integer(),
            server_default=sa.text("365"),
            nullable=False,
        ),
        sa.Column("observed_range_start_ts", sa.String(), nullable=True),
        sa.Column("observed_range_end_ts", sa.String(), nullable=True),
        sa.Column(
            "message_count",
            sa.Integer(),
            server_default=sa.text("0"),
            nullable=False,
        ),
        sa.Column(
            "file_count",
            sa.Integer(),
            server_default=sa.text("0"),
            nullable=False,
        ),
        sa.Column("last_scanned_message_ts", sa.String(), nullable=True),
        sa.Column("last_profiled_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("source_task_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column(
            "metadata_json",
            postgresql.JSONB(),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "profile_status in ('active', 'stale', 'disabled', 'failed')",
            name="ck_observe_channel_profiles_status",
        ),
        sa.CheckConstraint(
            "fresh_window_days > 0",
            name="ck_observe_channel_profiles_fresh_window",
        ),
        sa.CheckConstraint(
            "archive_window_days >= fresh_window_days",
            name="ck_observe_channel_profiles_archive_window",
        ),
        sa.ForeignKeyConstraint(
            ["installation_id"], ["installations.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["source_task_id"], ["tasks.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "installation_id",
            "channel_id",
            name="idx_observe_channel_profiles_unique",
        ),
    )
    op.create_index(
        "idx_observe_channel_profiles_lookup",
        "observe_channel_profiles",
        ["installation_id", "channel_id", "profile_status"],
    )
    op.create_index(
        "idx_observe_channel_profiles_last_profiled",
        "observe_channel_profiles",
        ["installation_id", "last_profiled_at"],
    )
    op.create_index(
        "idx_observe_channel_profiles_source_task",
        "observe_channel_profiles",
        ["source_task_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "idx_observe_channel_profiles_source_task",
        table_name="observe_channel_profiles",
    )
    op.drop_index(
        "idx_observe_channel_profiles_last_profiled",
        table_name="observe_channel_profiles",
    )
    op.drop_index(
        "idx_observe_channel_profiles_lookup",
        table_name="observe_channel_profiles",
    )
    op.drop_table("observe_channel_profiles")
