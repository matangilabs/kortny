"""slack pin and bookmark side effects

Revision ID: 0024
Revises: 0023
Create Date: 2026-06-07

Allows native Slack pin and bookmark tools to reuse the side-effect outbox.
"""

from alembic import op

# revision identifiers
revision = "0024"
down_revision = "0023"
branch_labels = None
depends_on = None

UPGRADE_OPERATIONS = (
    "'chat_postMessage', 'files_upload_v2', 'reactions_add', "
    "'reactions_remove', 'pins_add', 'bookmarks_add'"
)
DOWNGRADE_OPERATIONS = (
    "'chat_postMessage', 'files_upload_v2', 'reactions_add', 'reactions_remove'"
)


def upgrade() -> None:
    op.drop_constraint(
        "ck_slack_side_effects_operation",
        "slack_side_effects",
        type_="check",
    )
    op.create_check_constraint(
        "ck_slack_side_effects_operation",
        "slack_side_effects",
        f"operation in ({UPGRADE_OPERATIONS})",
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_slack_side_effects_operation",
        "slack_side_effects",
        type_="check",
    )
    op.create_check_constraint(
        "ck_slack_side_effects_operation",
        "slack_side_effects",
        f"operation in ({DOWNGRADE_OPERATIONS})",
    )
