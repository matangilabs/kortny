"""Add humanizer LLM tier.

Revision ID: 0022
Revises: 0021
Create Date: 2026-06-05 16:30:00.000000
"""

from __future__ import annotations

from alembic import op

# revision identifiers, used by Alembic.
revision = "0022"
down_revision = "0021"
branch_labels = None
depends_on = None

TIERS_WITH_HUMANIZER = (
    "('cheap_fast', 'standard', 'analysis', 'document', 'high_reasoning', 'humanizer')"
)
TIERS_LEGACY = "('cheap_fast', 'standard', 'analysis', 'document', 'high_reasoning')"


def upgrade() -> None:
    op.drop_constraint(
        "ck_llm_tier_assignments_tier",
        "llm_tier_assignments",
        type_="check",
    )
    op.create_check_constraint(
        "ck_llm_tier_assignments_tier",
        "llm_tier_assignments",
        f"tier in {TIERS_WITH_HUMANIZER}",
    )
    op.drop_constraint(
        "ck_llm_budget_policies_tier",
        "llm_budget_policies",
        type_="check",
    )
    op.create_check_constraint(
        "ck_llm_budget_policies_tier",
        "llm_budget_policies",
        f"tier is null or tier in {TIERS_WITH_HUMANIZER}",
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
        f"tier in {TIERS_LEGACY}",
    )
    op.drop_constraint(
        "ck_llm_budget_policies_tier",
        "llm_budget_policies",
        type_="check",
    )
    op.create_check_constraint(
        "ck_llm_budget_policies_tier",
        "llm_budget_policies",
        f"tier is null or tier in {TIERS_LEGACY}",
    )
