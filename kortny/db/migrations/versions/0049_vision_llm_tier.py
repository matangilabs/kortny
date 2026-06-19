"""Add vision LLM tier.

Revision ID: 0049
Revises: 0048
Create Date: 2026-06-19 00:00:00.000000
"""

from __future__ import annotations

from alembic import op

# revision identifiers, used by Alembic.
revision = "0049"
down_revision = "0048"
branch_labels = None
depends_on = None

TIERS_WITH_VISION = "('cheap_fast', 'standard', 'analysis', 'document', 'high_reasoning', 'humanizer', 'vision')"
TIERS_WITHOUT_VISION = (
    "('cheap_fast', 'standard', 'analysis', 'document', 'high_reasoning', 'humanizer')"
)


def upgrade() -> None:
    op.drop_constraint(
        "ck_llm_tier_assignments_tier",
        "llm_tier_assignments",
        type_="check",
    )
    op.create_check_constraint(
        "ck_llm_tier_assignments_tier",
        "llm_tier_assignments",
        f"tier in {TIERS_WITH_VISION}",
    )
    op.drop_constraint(
        "ck_llm_budget_policies_tier",
        "llm_budget_policies",
        type_="check",
    )
    op.create_check_constraint(
        "ck_llm_budget_policies_tier",
        "llm_budget_policies",
        f"tier is null or tier in {TIERS_WITH_VISION}",
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
        f"tier in {TIERS_WITHOUT_VISION}",
    )
    op.drop_constraint(
        "ck_llm_budget_policies_tier",
        "llm_budget_policies",
        type_="check",
    )
    op.create_check_constraint(
        "ck_llm_budget_policies_tier",
        "llm_budget_policies",
        f"tier is null or tier in {TIERS_WITHOUT_VISION}",
    )
