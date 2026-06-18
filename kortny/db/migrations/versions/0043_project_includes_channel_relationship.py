"""Allow the project_includes_channel graph relationship (HIG-276).

The World Model project layer links a ``project`` hub entity to its member
channels with a ``project_includes_channel`` edge so retrieval can synthesize
across a project's channels. Extend the kg_edges relationship_type check
constraint to permit it.

Revision ID: 0043
Revises: 0042
Create Date: 2026-06-17
"""

from __future__ import annotations

from alembic import op

revision = "0043"
down_revision = "0042"
branch_labels = None
depends_on = None

_OLD_RELATIONSHIP_TYPES = (
    "'member_of', 'maps_to', 'works_on', 'owns', 'belongs_to', "
    "'referenced_in', 'made_in', 'affects', 'relates_to', 'available_for'"
)
_NEW_RELATIONSHIP_TYPES = (
    "'member_of', 'maps_to', 'works_on', 'owns', 'belongs_to', "
    "'referenced_in', 'made_in', 'affects', 'relates_to', 'available_for', "
    "'project_includes_channel'"
)
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
        "DELETE FROM kg_edges WHERE relationship_type = 'project_includes_channel'"
    )
    op.drop_constraint(_CONSTRAINT, "kg_edges", type_="check")
    op.create_check_constraint(
        _CONSTRAINT,
        "kg_edges",
        f"relationship_type in ({_OLD_RELATIONSHIP_TYPES})",
    )
