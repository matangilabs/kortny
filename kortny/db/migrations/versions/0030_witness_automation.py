"""Witness automation: accepted suggestions become standing automations.

Revision ID: 0030
Revises: 0029
Create Date: 2026-06-11

Backs HIG-224 (Heartbeat parity). Witness candidates gain automation intent
columns (`automation_kind`, `cadence_suggestion`, `deliverable`) and
provenance links to the schedule/task they materialized into
(`automated_schedule_id`, `automated_task_id`). The candidate status set
gains the terminal `automated` value.
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = "0030"
down_revision = "0029"
branch_labels = None
depends_on = None

AUTOMATION_KINDS = "'recurring', 'one_shot', 'watch'"
STATUSES = (
    "'candidate', 'sent', 'accepted', 'automated', 'dismissed', 'cooldown', "
    "'superseded', 'archived'"
)
LEGACY_STATUSES = (
    "'candidate', 'sent', 'accepted', 'dismissed', 'cooldown', 'superseded', 'archived'"
)


def upgrade() -> None:
    op.add_column(
        "witness_opportunity_candidates",
        sa.Column("automation_kind", sa.Text(), nullable=True),
    )
    op.add_column(
        "witness_opportunity_candidates",
        sa.Column("cadence_suggestion", sa.Text(), nullable=True),
    )
    op.add_column(
        "witness_opportunity_candidates",
        sa.Column("deliverable", sa.Text(), nullable=True),
    )
    op.add_column(
        "witness_opportunity_candidates",
        sa.Column(
            "automated_schedule_id", postgresql.UUID(as_uuid=True), nullable=True
        ),
    )
    op.add_column(
        "witness_opportunity_candidates",
        sa.Column("automated_task_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.create_foreign_key(
        "fk_witness_opportunity_candidates_automated_schedule_id",
        "witness_opportunity_candidates",
        "schedules",
        ["automated_schedule_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_foreign_key(
        "fk_witness_opportunity_candidates_automated_task_id",
        "witness_opportunity_candidates",
        "tasks",
        ["automated_task_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_check_constraint(
        "ck_witness_opportunity_candidates_automation_kind",
        "witness_opportunity_candidates",
        f"automation_kind is null or automation_kind in ({AUTOMATION_KINDS})",
    )
    op.drop_constraint(
        "ck_witness_opportunity_candidates_status",
        "witness_opportunity_candidates",
        type_="check",
    )
    op.create_check_constraint(
        "ck_witness_opportunity_candidates_status",
        "witness_opportunity_candidates",
        f"status in ({STATUSES})",
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_witness_opportunity_candidates_status",
        "witness_opportunity_candidates",
        type_="check",
    )
    op.execute(
        "UPDATE witness_opportunity_candidates "
        "SET status = 'accepted' WHERE status = 'automated'"
    )
    op.create_check_constraint(
        "ck_witness_opportunity_candidates_status",
        "witness_opportunity_candidates",
        f"status in ({LEGACY_STATUSES})",
    )
    op.drop_constraint(
        "ck_witness_opportunity_candidates_automation_kind",
        "witness_opportunity_candidates",
        type_="check",
    )
    op.drop_constraint(
        "fk_witness_opportunity_candidates_automated_task_id",
        "witness_opportunity_candidates",
        type_="foreignkey",
    )
    op.drop_constraint(
        "fk_witness_opportunity_candidates_automated_schedule_id",
        "witness_opportunity_candidates",
        type_="foreignkey",
    )
    op.drop_column("witness_opportunity_candidates", "automated_task_id")
    op.drop_column("witness_opportunity_candidates", "automated_schedule_id")
    op.drop_column("witness_opportunity_candidates", "deliverable")
    op.drop_column("witness_opportunity_candidates", "cadence_suggestion")
    op.drop_column("witness_opportunity_candidates", "automation_kind")
