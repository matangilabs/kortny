"""Channel assessment task helpers for Kortny Observe."""

from __future__ import annotations

import uuid
from collections.abc import Mapping

from sqlalchemy import select
from sqlalchemy.orm import Session

from kortny.db.models import Task, TaskEvent, TaskEventType

CHANNEL_ASSESSMENT_REQUESTED_MESSAGE = "observe_channel_assessment_requested"
CHANNEL_ASSESSMENT_COMPLETED_MESSAGE = "observe_channel_assessment_completed"
CHANNEL_ASSESSMENT_FAILED_MESSAGE = "observe_channel_assessment_failed"
CHANNEL_ASSESSMENT_SUPPRESS_SLACK_POST_KEY = "suppress_slack_post"


def build_channel_assessment_input(*, channel_id: str) -> str:
    """Return the synthetic worker prompt for onboarding channel assessment."""

    return (
        "Run Kortny's channel onboarding assessment for this Slack channel.\n\n"
        f"Channel ID: {channel_id}\n\n"
        "Use slack_channel_history for the current task channel with a bounded "
        "lookback. Prefer limit 40 and include_threads true. Do not use web "
        "search, Composio, PDF generation, or memory-writing tools for this "
        "assessment.\n\n"
        "Post one concise Slack-native follow-up as Kortny. The follow-up should:\n"
        "- Briefly say you took a quick look at recent channel context.\n"
        "- Identify the likely channel purpose and recurring themes, using careful "
        "language when evidence is thin.\n"
        "- Mention 2-4 practical ways you can help in this channel.\n"
        "- If there is not enough history, say that plainly and offer useful "
        "starting points.\n"
        "- Keep it human, specific, and low-pressure.\n"
        "- Do not claim to have read DMs or anything outside this channel.\n"
        "- Use Slack mrkdwn, not Markdown headings."
    )


def build_channel_graph_refresh_input(*, channel_id: str) -> str:
    """Return the synthetic worker prompt for a silent graph refresh assessment."""

    return (
        "Run Kortny's background channel graph refresh assessment for this Slack "
        "channel.\n\n"
        f"Channel ID: {channel_id}\n\n"
        "Use slack_channel_history for the current task channel with a bounded "
        "lookback. Prefer limit 80 and include_threads true. Do not use web "
        "search, Composio, PDF generation, or memory-writing tools for this "
        "assessment.\n\n"
        "Produce a concise internal assessment summary for the workspace knowledge "
        "graph. The summary should identify the likely channel purpose, recurring "
        "topics, important entities or workflows, and practical ways Kortny may "
        "help in this channel. Use careful language when evidence is thin. Do not "
        "address the channel directly, do not ask a follow-up question, and do not "
        "claim to have read DMs or anything outside this channel."
    )


def assessment_event_id_for_membership(
    membership_id: uuid.UUID, attempt: int = 0
) -> str:
    """Stable synthetic Slack event ID for one channel assessment task.

    Attempt 0 is the initial queue; attempt N (>=1) disambiguates retries so
    that repeated failures each produce a fresh, dedup-safe identity.
    """

    if attempt == 0:
        return f"observe:{membership_id}:channel_assessment"
    return f"observe:{membership_id}:channel_assessment:attempt:{attempt}"


def assessment_identity_source_id(membership_id: uuid.UUID, attempt: int = 0) -> str:
    """Stable synthetic task identity source id for one channel assessment.

    Attempt 0 is the initial queue; attempt N (>=1) disambiguates retries.
    """

    if attempt == 0:
        return str(membership_id)
    return f"{membership_id}:attempt:{attempt}"


def is_channel_assessment_task(session: Session, task: Task) -> bool:
    """Return whether a task is a system channel assessment task."""

    return channel_assessment_request_event(session, task) is not None


def channel_assessment_request_event(
    session: Session,
    task: Task,
) -> TaskEvent | None:
    """Return the channel assessment request metadata event, if any."""

    return session.scalar(
        select(TaskEvent)
        .where(
            TaskEvent.task_id == task.id,
            TaskEvent.type == TaskEventType.log,
            TaskEvent.payload["message"].as_string()
            == CHANNEL_ASSESSMENT_REQUESTED_MESSAGE,
        )
        .order_by(TaskEvent.seq.desc())
        .limit(1)
    )


def request_payload_channel_id(payload: Mapping[str, object]) -> str | None:
    """Read a channel ID from a channel assessment request payload."""

    value = payload.get("channel_id")
    if isinstance(value, str) and value:
        return value
    return None
