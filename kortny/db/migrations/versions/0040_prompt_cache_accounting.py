"""Prompt-cache token accounting (HIG-196).

Revision ID: 0040
Revises: 0039
Create Date: 2026-06-12

Adds the prompt-cache token split to ``llm_usage`` and per-pricing-entry cache
cost multipliers to ``model_pricing`` (HIG-196 D5).

``llm_usage.cache_creation_input_tokens`` / ``cache_read_input_tokens`` are a
partition *within* the existing ``input_tokens`` total (which stays LiteLLM's
``prompt_tokens`` = total prompt count), not additions to it. ``model_pricing``
gains ``cache_write_multiplier`` (default 1.25x for the 5-minute TTL) and
``cache_read_multiplier`` (default 0.10x) so cost math is configurable per
pricing row. All four columns are NOT NULL with server defaults so existing rows
backfill cleanly and the exact-table-set test in tests/test_db_models.py is
untouched (columns only, no new tables).
"""

from alembic import op

# revision identifiers
revision = "0040"
down_revision = "0039"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE llm_usage "
        "ADD COLUMN IF NOT EXISTS cache_creation_input_tokens integer "
        "NOT NULL DEFAULT 0"
    )
    op.execute(
        "ALTER TABLE llm_usage "
        "ADD COLUMN IF NOT EXISTS cache_read_input_tokens integer "
        "NOT NULL DEFAULT 0"
    )
    op.execute(
        "ALTER TABLE model_pricing "
        "ADD COLUMN IF NOT EXISTS cache_write_multiplier numeric(6, 4) "
        "NOT NULL DEFAULT 1.25"
    )
    op.execute(
        "ALTER TABLE model_pricing "
        "ADD COLUMN IF NOT EXISTS cache_read_multiplier numeric(6, 4) "
        "NOT NULL DEFAULT 0.10"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE model_pricing DROP COLUMN IF EXISTS cache_read_multiplier")
    op.execute("ALTER TABLE model_pricing DROP COLUMN IF EXISTS cache_write_multiplier")
    op.execute("ALTER TABLE llm_usage DROP COLUMN IF EXISTS cache_read_input_tokens")
    op.execute(
        "ALTER TABLE llm_usage DROP COLUMN IF EXISTS cache_creation_input_tokens"
    )
