"""Slack Block Kit affordances for schedule management."""

from __future__ import annotations

import uuid

from kortny.db.models import Schedule

SCHEDULE_ACTION_PREFIX = "kortny_schedule_"
SCHEDULE_ACTION_ACTIVATE = f"{SCHEDULE_ACTION_PREFIX}activate"
SCHEDULE_ACTION_PAUSE = f"{SCHEDULE_ACTION_PREFIX}pause"
SCHEDULE_ACTION_RESUME = f"{SCHEDULE_ACTION_PREFIX}resume"
SCHEDULE_ACTION_CHANGE = f"{SCHEDULE_ACTION_PREFIX}change"
SCHEDULE_ACTION_CANCEL = f"{SCHEDULE_ACTION_PREFIX}cancel"
SCHEDULE_ACTION_IDS = frozenset(
    {
        SCHEDULE_ACTION_ACTIVATE,
        SCHEDULE_ACTION_PAUSE,
        SCHEDULE_ACTION_RESUME,
        SCHEDULE_ACTION_CHANGE,
        SCHEDULE_ACTION_CANCEL,
    }
)


def schedule_action_blocks(schedule: Schedule) -> list[dict]:
    """Return compact schedule management buttons for a Slack confirmation."""

    primary_action: tuple[str, str, str | None]
    if schedule.status == "proposed":
        primary_action = ("Activate", SCHEDULE_ACTION_ACTIVATE, "primary")
    elif schedule.status == "paused":
        primary_action = ("Resume", SCHEDULE_ACTION_RESUME, "primary")
    else:
        primary_action = ("Pause", SCHEDULE_ACTION_PAUSE, None)

    elements = [
        _button(
            text=primary_action[0],
            action_id=primary_action[1],
            schedule_id=schedule.id,
            style=primary_action[2],
        ),
        _button(
            text="Change",
            action_id=SCHEDULE_ACTION_CHANGE,
            schedule_id=schedule.id,
        ),
        _button(
            text="Cancel",
            action_id=SCHEDULE_ACTION_CANCEL,
            schedule_id=schedule.id,
            style="danger",
            confirm={
                "title": {"type": "plain_text", "text": "Cancel this schedule?"},
                "text": {
                    "type": "mrkdwn",
                    "text": "Kortny will stop running this scheduled task.",
                },
                "confirm": {"type": "plain_text", "text": "Cancel"},
                "deny": {"type": "plain_text", "text": "Keep it"},
            },
        ),
    ]
    return [{"type": "actions", "elements": elements}]


def parse_schedule_action_value(value: object) -> uuid.UUID | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return uuid.UUID(value.strip())
    except ValueError:
        return None


def _button(
    *,
    text: str,
    action_id: str,
    schedule_id: uuid.UUID,
    style: str | None = None,
    confirm: dict | None = None,
) -> dict:
    button = {
        "type": "button",
        "text": {"type": "plain_text", "text": text},
        "action_id": action_id,
        "value": str(schedule_id),
    }
    if style is not None:
        button["style"] = style
    if confirm is not None:
        button["confirm"] = confirm
    return button
