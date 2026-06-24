"""Add chat_stopStream to slack side effect operations constraint.

Revision ID: 0054
Revises: 0053
Create Date: 2026-06-23
"""

from __future__ import annotations

from alembic import op

revision = "0054"
down_revision = "0053"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint(
        "ck_slack_side_effects_operation", "slack_side_effects", type_="check"
    )
    op.create_check_constraint(
        "ck_slack_side_effects_operation",
        "slack_side_effects",
        "operation in ('chat_postMessage', 'files_upload_v2', 'reactions_add', "
        "'reactions_remove', 'pins_add', 'bookmarks_add', "
        "'conversations_canvases_create', 'canvases_edit', 'chat_stopStream')",
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_slack_side_effects_operation", "slack_side_effects", type_="check"
    )
    op.create_check_constraint(
        "ck_slack_side_effects_operation",
        "slack_side_effects",
        "operation in ('chat_postMessage', 'files_upload_v2', 'reactions_add', "
        "'reactions_remove', 'pins_add', 'bookmarks_add', "
        "'conversations_canvases_create', 'canvases_edit')",
    )
