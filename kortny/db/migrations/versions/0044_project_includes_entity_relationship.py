"""Allow the project_includes_entity graph relationship (HIG-276 increment 2).

Implicit project inference links a `project` hub directly to its constituent
entities (topics/decisions/commitments) so retrieval reaches them. Inference is
autonomous (the brain learns continuously and self-corrects via reinforcement +
aging); there is no proposal-gate table — inferred-vs-confirmed lives in the
graph itself (lifecycle + source_type + confidence).

Revision ID: 0044
Revises: 0043
Create Date: 2026-06-17
"""

from __future__ import annotations

from alembic import op

revision = "0044"
down_revision = "0043"
branch_labels = None
depends_on = None

_PREV_RELATIONSHIP_TYPES = (
    "'member_of', 'maps_to', 'works_on', 'owns', 'belongs_to', "
    "'referenced_in', 'made_in', 'affects', 'relates_to', 'available_for', "
    "'project_includes_channel'"
)
_NEW_RELATIONSHIP_TYPES = _PREV_RELATIONSHIP_TYPES + ", 'project_includes_entity'"
_CONSTRAINT = "ck_kg_edges_relationship_type"


def upgrade() -> None:
    op.drop_constraint(_CONSTRAINT, "kg_edges", type_="check")
    op.create_check_constraint(
        _CONSTRAINT,
        "kg_edges",
        f"relationship_type in ({_NEW_RELATIONSHIP_TYPES})",
    )


def downgrade() -> None:
    op.execute(
        "DELETE FROM kg_edges WHERE relationship_type = 'project_includes_entity'"
    )
    op.drop_constraint(_CONSTRAINT, "kg_edges", type_="check")
    op.create_check_constraint(
        _CONSTRAINT,
        "kg_edges",
        f"relationship_type in ({_PREV_RELATIONSHIP_TYPES})",
    )
