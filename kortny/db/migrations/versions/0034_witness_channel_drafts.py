"""Witness channel delivery + autopilot draft tier decision values.

Revision ID: 0034
Revises: 0033
Create Date: 2026-06-11

Backs HIG-198 + HIG-230. The append-only ``witness_delivery_log`` gains three
decision values: ``channel_sent`` (a proactive post delivered into a channel
whose ObservePolicy proactivity_status is 'full'), ``channel_deferred`` (a
channel delivery deferred for policy/quiet_hours/budget — deferred, never
dropped), and ``draft_executed`` (the autopilot draft tier posted a visible
draft deliverable). Channel-level rows store ``channel:{channel_id}`` in
``slack_user_id`` so sliding budget windows stay queryable per channel.
"""

from alembic import op

# revision identifiers
revision = "0034"
down_revision = "0033"
branch_labels = None
depends_on = None

OLD_DECISIONS = "'notify', 'question', 'draft', 'silent', 'digest'"
NEW_DECISIONS = (
    "'notify', 'question', 'draft', 'silent', 'digest', "
    "'channel_sent', 'channel_deferred', 'draft_executed'"
)


def upgrade() -> None:
    op.drop_constraint(
        "ck_witness_delivery_log_decision",
        "witness_delivery_log",
        type_="check",
    )
    op.create_check_constraint(
        "ck_witness_delivery_log_decision",
        "witness_delivery_log",
        f"decision in ({NEW_DECISIONS})",
    )


def downgrade() -> None:
    op.execute(
        "DELETE FROM witness_delivery_log "
        "WHERE decision in ('channel_sent', 'channel_deferred', 'draft_executed')"
    )
    op.drop_constraint(
        "ck_witness_delivery_log_decision",
        "witness_delivery_log",
        type_="check",
    )
    op.create_check_constraint(
        "ck_witness_delivery_log_decision",
        "witness_delivery_log",
        f"decision in ({OLD_DECISIONS})",
    )
