"""Add profiler LLM tier.

Revision ID: 0057
Revises: 0056
Create Date: 2026-06-25 00:00:00.000000
"""

from __future__ import annotations

from alembic import op

# revision identifiers, used by Alembic.
revision = "0057"
down_revision = "0056"
branch_labels = None
depends_on = None

TIERS_WITH_PROFILER = "('cheap_fast', 'standard', 'analysis', 'document', 'high_reasoning', 'humanizer', 'vision', 'profiler')"
TIERS_WITHOUT_PROFILER = "('cheap_fast', 'standard', 'analysis', 'document', 'high_reasoning', 'humanizer', 'vision')"


def upgrade() -> None:
    op.drop_constraint(
        "ck_llm_tier_assignments_tier",
        "llm_tier_assignments",
        type_="check",
    )
    op.create_check_constraint(
        "ck_llm_tier_assignments_tier",
        "llm_tier_assignments",
        f"tier in {TIERS_WITH_PROFILER}",
    )
    op.drop_constraint(
        "ck_llm_budget_policies_tier",
        "llm_budget_policies",
        type_="check",
    )
    op.create_check_constraint(
        "ck_llm_budget_policies_tier",
        "llm_budget_policies",
        f"tier is null or tier in {TIERS_WITH_PROFILER}",
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_llm_tier_assignments_tier",
        "llm_tier_assignments",
        type_="check",
    )
    op.create_check_constraint(
        "ck_llm_tier_assignments_tier",
        "llm_tier_assignments",
        f"tier in {TIERS_WITHOUT_PROFILER}",
    )
    op.drop_constraint(
        "ck_llm_budget_policies_tier",
        "llm_budget_policies",
        type_="check",
    )
    op.create_check_constraint(
        "ck_llm_budget_policies_tier",
        "llm_budget_policies",
        f"tier is null or tier in {TIERS_WITHOUT_PROFILER}",
    )
