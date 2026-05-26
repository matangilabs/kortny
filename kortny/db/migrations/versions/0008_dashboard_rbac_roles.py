"""dashboard rbac roles

Revision ID: 0008
Revises: 0007
Create Date: 2026-05-26

Normalizes dashboard roles to the simple V1 RBAC model:
admin or member. Existing owner rows become admin.
"""

from alembic import op

# revision identifiers
revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("UPDATE dashboard_users SET role = 'admin' WHERE role = 'owner'")
    op.drop_constraint(
        "ck_dashboard_users_role",
        "dashboard_users",
        type_="check",
    )
    op.create_check_constraint(
        "ck_dashboard_users_role",
        "dashboard_users",
        "role in ('admin', 'member')",
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_dashboard_users_role",
        "dashboard_users",
        type_="check",
    )
    op.create_check_constraint(
        "ck_dashboard_users_role",
        "dashboard_users",
        "role in ('owner', 'admin', 'member')",
    )
