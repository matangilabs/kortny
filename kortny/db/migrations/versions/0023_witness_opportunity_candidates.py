"""witness opportunity candidates

Revision ID: 0023
Revises: 0022
Create Date: 2026-06-05

Adds the backend-only candidate store for HIG-184. Candidates capture
evidence-backed proactive opportunities without delivering Slack suggestions.
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = "0023"
down_revision = "0022"
branch_labels = None
depends_on = None

SCOPE_TYPES = "'workspace', 'channel', 'private_channel', 'dm', 'user'"
CANDIDATE_TYPES = (
    "'workflow_gap', 'artifact_followup', 'unresolved_decision', "
    "'data_quality_issue', 'recurring_check', 'project_status_gap', "
    "'general_help'"
)
SOURCE_TYPES = (
    "'channel_profile', 'knowledge_graph', 'task_summary', "
    "'scheduled_witness', 'manual'"
)
STATUSES = (
    "'candidate', 'sent', 'accepted', 'dismissed', 'cooldown', 'superseded', 'archived'"
)


def upgrade() -> None:
    op.create_table(
        "witness_opportunity_candidates",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("installation_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("channel_id", sa.String(), nullable=True),
        sa.Column("target_slack_user_id", sa.String(), nullable=True),
        sa.Column("visibility_scope_type", sa.String(), nullable=False),
        sa.Column("visibility_scope_id", sa.String(), nullable=True),
        sa.Column("candidate_type", sa.String(), nullable=False),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("suggested_action", sa.Text(), nullable=True),
        sa.Column("suggested_message", sa.Text(), nullable=True),
        sa.Column(
            "evidence_json",
            postgresql.JSONB(),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column("source_type", sa.String(), nullable=False),
        sa.Column("source_id", sa.String(), nullable=True),
        sa.Column("source_task_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("source_profile_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("dedupe_key", sa.String(), nullable=False),
        sa.Column(
            "confidence_score",
            sa.Numeric(4, 3),
            server_default=sa.text("0.500"),
            nullable=False,
        ),
        sa.Column("confidence_reason", sa.Text(), nullable=True),
        sa.Column(
            "status",
            sa.String(),
            server_default=sa.text("'candidate'"),
            nullable=False,
        ),
        sa.Column("cooldown_until", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("last_suggested_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column(
            "feedback_json",
            postgresql.JSONB(),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
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
            f"visibility_scope_type in ({SCOPE_TYPES})",
            name="ck_witness_opportunity_candidates_scope_type",
        ),
        sa.CheckConstraint(
            "(visibility_scope_type = 'workspace' and visibility_scope_id is null) or "
            "(visibility_scope_type in ('channel', 'private_channel', 'dm', 'user') "
            "and visibility_scope_id is not null)",
            name="ck_witness_opportunity_candidates_scope_id",
        ),
        sa.CheckConstraint(
            f"candidate_type in ({CANDIDATE_TYPES})",
            name="ck_witness_opportunity_candidates_type",
        ),
        sa.CheckConstraint(
            f"source_type in ({SOURCE_TYPES})",
            name="ck_witness_opportunity_candidates_source_type",
        ),
        sa.CheckConstraint(
            f"status in ({STATUSES})",
            name="ck_witness_opportunity_candidates_status",
        ),
        sa.CheckConstraint(
            "confidence_score >= 0 and confidence_score <= 1",
            name="ck_witness_opportunity_candidates_confidence",
        ),
        sa.ForeignKeyConstraint(
            ["installation_id"], ["installations.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["source_task_id"], ["tasks.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(
            ["source_profile_id"], ["observe_channel_profiles.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "idx_witness_opportunity_candidates_unique",
        "witness_opportunity_candidates",
        [
            "installation_id",
            "visibility_scope_type",
            sa.text("coalesce(visibility_scope_id, '')"),
            "candidate_type",
            "dedupe_key",
        ],
        unique=True,
    )
    op.create_index(
        "idx_witness_opportunity_candidates_status",
        "witness_opportunity_candidates",
        ["installation_id", "status", "cooldown_until"],
    )
    op.create_index(
        "idx_witness_opportunity_candidates_channel",
        "witness_opportunity_candidates",
        ["installation_id", "channel_id", "status"],
    )
    op.create_index(
        "idx_witness_opportunity_candidates_scope",
        "witness_opportunity_candidates",
        ["installation_id", "visibility_scope_type", "visibility_scope_id"],
    )
    op.create_index(
        "idx_witness_opportunity_candidates_task",
        "witness_opportunity_candidates",
        ["source_task_id"],
    )
    op.create_index(
        "idx_witness_opportunity_candidates_profile",
        "witness_opportunity_candidates",
        ["source_profile_id"],
    )
    op.create_index(
        "idx_witness_opportunity_candidates_evidence",
        "witness_opportunity_candidates",
        ["evidence_json"],
        postgresql_using="gin",
    )


def downgrade() -> None:
    op.drop_index(
        "idx_witness_opportunity_candidates_evidence",
        table_name="witness_opportunity_candidates",
    )
    op.drop_index(
        "idx_witness_opportunity_candidates_profile",
        table_name="witness_opportunity_candidates",
    )
    op.drop_index(
        "idx_witness_opportunity_candidates_task",
        table_name="witness_opportunity_candidates",
    )
    op.drop_index(
        "idx_witness_opportunity_candidates_scope",
        table_name="witness_opportunity_candidates",
    )
    op.drop_index(
        "idx_witness_opportunity_candidates_channel",
        table_name="witness_opportunity_candidates",
    )
    op.drop_index(
        "idx_witness_opportunity_candidates_status",
        table_name="witness_opportunity_candidates",
    )
    op.drop_index(
        "idx_witness_opportunity_candidates_unique",
        table_name="witness_opportunity_candidates",
    )
    op.drop_table("witness_opportunity_candidates")
