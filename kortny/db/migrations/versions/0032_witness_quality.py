"""Witness proactivity quality loop: reinforcement, receptivity, delivery log.

Revision ID: 0032
Revises: 0031
Create Date: 2026-06-11

Backs HIG-227 (Slice D). Witness candidates gain reinforcement counting
(`reinforcement_count`, `first_observed_at` backfilled from `created_at`),
the last delivery decision (`last_decision`), and the receptivity score
computed at delivery time (`receptivity_score`). A new append-only
`witness_delivery_log` table records every delivery decision — including
silence — and is the queryable budget window + KPI source.
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = "0032"
down_revision = "0031"
branch_labels = None
depends_on = None

DECISIONS = "'notify', 'question', 'draft', 'silent', 'digest'"
LAST_DECISIONS = "'notify', 'question', 'draft', 'silent'"


def upgrade() -> None:
    op.add_column(
        "witness_opportunity_candidates",
        sa.Column(
            "reinforcement_count",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("1"),
        ),
    )
    op.add_column(
        "witness_opportunity_candidates",
        sa.Column("first_observed_at", sa.TIMESTAMP(timezone=True), nullable=True),
    )
    op.add_column(
        "witness_opportunity_candidates",
        sa.Column("last_decision", sa.Text(), nullable=True),
    )
    op.add_column(
        "witness_opportunity_candidates",
        sa.Column("receptivity_score", sa.Numeric(4, 3), nullable=True),
    )
    op.execute(
        "UPDATE witness_opportunity_candidates "
        "SET first_observed_at = created_at WHERE first_observed_at IS NULL"
    )
    op.create_check_constraint(
        "ck_witness_opportunity_candidates_last_decision",
        "witness_opportunity_candidates",
        f"last_decision is null or last_decision in ({LAST_DECISIONS})",
    )
    op.create_check_constraint(
        "ck_witness_opportunity_candidates_receptivity",
        "witness_opportunity_candidates",
        "receptivity_score is null or "
        "(receptivity_score >= 0 and receptivity_score <= 1)",
    )

    op.create_table(
        "witness_delivery_log",
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
        sa.Column("slack_user_id", sa.Text(), nullable=False),
        sa.Column(
            "candidate_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("witness_opportunity_candidates.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("decision", sa.Text(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint(
            f"decision in ({DECISIONS})",
            name="ck_witness_delivery_log_decision",
        ),
    )
    op.create_index(
        "idx_witness_delivery_log_user_window",
        "witness_delivery_log",
        ["installation_id", "slack_user_id", "created_at"],
    )
    op.create_index(
        "idx_witness_delivery_log_candidate",
        "witness_delivery_log",
        ["candidate_id"],
    )
    op.create_index(
        "idx_witness_delivery_log_decision",
        "witness_delivery_log",
        ["installation_id", "decision", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("idx_witness_delivery_log_decision", "witness_delivery_log")
    op.drop_index("idx_witness_delivery_log_candidate", "witness_delivery_log")
    op.drop_index("idx_witness_delivery_log_user_window", "witness_delivery_log")
    op.drop_table("witness_delivery_log")
    op.drop_constraint(
        "ck_witness_opportunity_candidates_receptivity",
        "witness_opportunity_candidates",
        type_="check",
    )
    op.drop_constraint(
        "ck_witness_opportunity_candidates_last_decision",
        "witness_opportunity_candidates",
        type_="check",
    )
    op.drop_column("witness_opportunity_candidates", "receptivity_score")
    op.drop_column("witness_opportunity_candidates", "last_decision")
    op.drop_column("witness_opportunity_candidates", "first_observed_at")
    op.drop_column("witness_opportunity_candidates", "reinforcement_count")
