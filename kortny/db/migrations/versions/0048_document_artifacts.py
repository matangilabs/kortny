"""Living-document fields on artifacts (HIG-244 close-out).

Persist the canonical post-critique doc-spec on each rendered artifact so a
document can be re-rendered/edited from Slack buttons. Versions of one document
share doc_group_id; doc_version increments per visible revision.

Revision ID: 0048
Revises: 0047
Create Date: 2026-06-19
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "0048"
down_revision = "0047"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "artifacts", sa.Column("doc_group_id", UUID(as_uuid=True), nullable=True)
    )
    op.add_column("artifacts", sa.Column("doc_version", sa.Integer(), nullable=True))
    op.add_column("artifacts", sa.Column("spec_json", JSONB(), nullable=True))
    op.create_index(
        "idx_artifacts_doc_group",
        "artifacts",
        ["doc_group_id", "doc_version"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("idx_artifacts_doc_group", table_name="artifacts")
    op.drop_column("artifacts", "spec_json")
    op.drop_column("artifacts", "doc_version")
    op.drop_column("artifacts", "doc_group_id")
