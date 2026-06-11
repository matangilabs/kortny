"""Memory spine: bi-temporal graph columns, memory embeddings, consolidation runs.

Revision ID: 0031
Revises: 0030
Create Date: 2026-06-11

Backs HIG-225 (memory spine, Slice B):

- ``kg_entities``/``kg_edges`` gain Graphiti-style bi-temporal columns
  (``valid_at`` / ``invalid_at`` plus ``system_expired_at``). Existing rows are
  backfilled with ``valid_at = created_at``. ``lifecycle_state`` stays for
  compatibility; writers set both.
- ``tool_embeddings`` (historical name; it is the general semantic index now)
  accepts the new kinds ``fact`` / ``episode`` / ``kg_entity``.
- ``user_confirmed`` becomes a valid graph ``source_type`` so user-confirmed
  workspace facts can project into the graph and outrank generated knowledge.
- New ``consolidation_runs`` table records every consolidator run with
  counters and cost.

``workspace_state.expires_at`` already exists since 0002 and is reused as the
TTL column for ephemeral facts.
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = "0031"
down_revision = "0030"
branch_labels = None
depends_on = None

_OLD_EMBEDDING_KINDS = "('tool_card', 'skill')"
_NEW_EMBEDDING_KINDS = "('tool_card', 'skill', 'fact', 'episode', 'kg_entity')"
_OLD_GRAPH_SOURCE_TYPES = (
    "('slack_authoritative', 'user_explicit', 'agent_inferred', "
    "'onboarding_scan', 'task_summary', 'integration_result', "
    "'workspace_state', 'admin_import')"
)
_NEW_GRAPH_SOURCE_TYPES = (
    "('slack_authoritative', 'user_explicit', 'user_confirmed', "
    "'agent_inferred', 'onboarding_scan', 'task_summary', "
    "'integration_result', 'workspace_state', 'admin_import')"
)


def upgrade() -> None:
    for table in ("kg_entities", "kg_edges"):
        op.add_column(
            table,
            sa.Column("valid_at", postgresql.TIMESTAMP(timezone=True), nullable=True),
        )
        op.add_column(
            table,
            sa.Column("invalid_at", postgresql.TIMESTAMP(timezone=True), nullable=True),
        )
        op.add_column(
            table,
            sa.Column(
                "system_expired_at",
                postgresql.TIMESTAMP(timezone=True),
                nullable=True,
            ),
        )
        op.execute(f"UPDATE {table} SET valid_at = created_at WHERE valid_at IS NULL")
        op.create_index(f"idx_{table}_invalid_at", table, ["invalid_at"])

        constraint = f"ck_{table}_source_type"
        op.drop_constraint(constraint, table, type_="check")
        op.create_check_constraint(
            constraint,
            table,
            f"source_type in {_NEW_GRAPH_SOURCE_TYPES}",
        )

    op.drop_constraint("ck_tool_embeddings_kind", "tool_embeddings", type_="check")
    op.create_check_constraint(
        "ck_tool_embeddings_kind",
        "tool_embeddings",
        f"kind in {_NEW_EMBEDDING_KINDS}",
    )

    op.create_table(
        "consolidation_runs",
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
        sa.Column(
            "started_at",
            postgresql.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("finished_at", postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.Column(
            "status",
            sa.Text(),
            nullable=False,
            server_default=sa.text("'running'"),
        ),
        sa.Column(
            "counters_json",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("cost_usd", sa.Numeric(12, 6), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.CheckConstraint(
            "status in ('running', 'succeeded', 'failed')",
            name="ck_consolidation_runs_status",
        ),
    )
    op.create_index(
        "idx_consolidation_runs_installation_started",
        "consolidation_runs",
        ["installation_id", "started_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "idx_consolidation_runs_installation_started",
        table_name="consolidation_runs",
    )
    op.drop_table("consolidation_runs")

    op.execute(
        "DELETE FROM tool_embeddings WHERE kind in ('fact', 'episode', 'kg_entity')"
    )
    op.drop_constraint("ck_tool_embeddings_kind", "tool_embeddings", type_="check")
    op.create_check_constraint(
        "ck_tool_embeddings_kind",
        "tool_embeddings",
        f"kind in {_OLD_EMBEDDING_KINDS}",
    )

    for table in ("kg_entities", "kg_edges"):
        op.execute(
            f"UPDATE {table} SET source_type = 'user_explicit' "
            "WHERE source_type = 'user_confirmed'"
        )
        constraint = f"ck_{table}_source_type"
        op.drop_constraint(constraint, table, type_="check")
        op.create_check_constraint(
            constraint,
            table,
            f"source_type in {_OLD_GRAPH_SOURCE_TYPES}",
        )
        op.drop_index(f"idx_{table}_invalid_at", table_name=table)
        op.drop_column(table, "system_expired_at")
        op.drop_column(table, "invalid_at")
        op.drop_column(table, "valid_at")
