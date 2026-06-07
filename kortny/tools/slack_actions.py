"""Slack-native low-risk action tools."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from typing import Any, Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session

from kortny.db.models import Task, TaskEvent, TaskEventType
from kortny.slack.outbox import (
    SlackSideEffectOutbox,
    slack_reaction_key,
)
from kortny.slack_mrkdwn import normalize_slack_mrkdwn
from kortny.tasks import TaskService
from kortny.tools.types import JsonObject, JsonSchema, ToolResult

MAX_TOOL_REPLY_CHARS = 4_000
SLACK_TS_RE = re.compile(r"^\d{10,}\.\d+$")
REACTION_NAME_RE = re.compile(r"^[A-Za-z0-9_+-]+(?:::skin-tone-[2-6])?$")


class SlackActionClient(Protocol):
    """Subset of Slack WebClient used by native Slack action tools."""

    def chat_postMessage(
        self,
        *,
        channel: str,
        text: str,
        thread_ts: str | None = None,
    ) -> Any:
        """Post a Slack message."""

    def reactions_add(
        self,
        *,
        channel: str,
        timestamp: str,
        name: str,
    ) -> Any:
        """Add a Slack reaction."""


class SlackReplyThreadTool:
    """Post a Slack reply in the current task thread."""

    name = "slack_reply_thread"
    description = (
        "Posts a Slack message into the current task's DM or active thread. "
        "Use this only when the user explicitly asks Kortny to post, reply, "
        "send, or leave a visible Slack message before the final answer. Do "
        "not use it for ordinary final answers because Kortny's final response "
        "is posted automatically. The tool is scoped to the current Slack "
        "channel and thread."
    )
    parameters: JsonSchema = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Slack mrkdwn text to post into the current thread or DM.",
            },
            "channel_id": {
                "type": "string",
                "description": (
                    "Optional current Slack channel ID. Omit unless copying the "
                    "current task channel exactly."
                ),
            },
            "thread_ts": {
                "type": "string",
                "description": (
                    "Optional current parent thread timestamp. Omit to use the "
                    "task thread. This cannot target another thread."
                ),
            },
            "purpose": {
                "type": "string",
                "description": (
                    "Optional short purpose for trace labels, such as follow_up "
                    "or status_update."
                ),
            },
        },
        "required": ["text"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        *,
        client: SlackActionClient,
        session: Session,
        task: Task,
        task_service: TaskService | None = None,
    ) -> None:
        self.client = client
        self.session = session
        self.task = task
        self.task_service = task_service or TaskService(session)

    def invoke(self, args: JsonObject) -> ToolResult:
        text = _coerce_reply_text(args.get("text"))
        channel_id = _current_channel_arg(
            value=args.get("channel_id"),
            task=self.task,
            tool_name=self.name,
        )
        thread_ts = _current_thread_arg(args.get("thread_ts"), task=self.task)
        purpose = _coerce_purpose(args.get("purpose"), default="tool_reply_thread")
        post_thread_ts = _post_thread_ts(channel_id=channel_id, thread_ts=thread_ts)
        idempotency_key = _reply_idempotency_key(
            task=self.task,
            purpose=purpose,
            channel_id=channel_id,
            thread_ts=post_thread_ts,
            text=text,
        )
        request: JsonObject = {
            "channel": channel_id,
            "text": text,
            "thread_ts": post_thread_ts,
        }
        result = SlackSideEffectOutbox(self.session).deliver(
            installation_id=self.task.installation_id,
            task_id=self.task.id,
            idempotency_key=idempotency_key,
            operation="chat_postMessage",
            purpose=purpose,
            target_channel_id=channel_id,
            target_thread_ts=post_thread_ts,
            request=request,
            call=lambda: self.client.chat_postMessage(
                channel=channel_id,
                text=text,
                thread_ts=post_thread_ts,
            ),
        )
        response = _require_ok(result.response, "chat.postMessage")
        message_ts = _response_ts(response)
        if message_ts is None:
            raise RuntimeError("Slack chat.postMessage response is missing ts")
        side_effect_id = str(result.side_effect.id)
        if not _task_event_exists(
            self.session,
            task=self.task,
            event_type=TaskEventType.message_posted,
            side_effect_id=side_effect_id,
        ):
            self.task_service.append_event(
                self.task,
                TaskEventType.message_posted,
                {
                    "channel": channel_id,
                    "thread_ts": post_thread_ts,
                    "message_ts": message_ts,
                    "text": text,
                    "purpose": purpose,
                    "slack_side_effect_id": side_effect_id,
                    "idempotency_key": idempotency_key,
                    "tool": self.name,
                    "deduped": result.deduped,
                },
            )
        return ToolResult(
            output={
                "successful": True,
                "channel": channel_id,
                "thread_ts": post_thread_ts,
                "message_ts": message_ts,
                "text_chars": len(text),
                "purpose": purpose,
                "deduped": result.deduped,
                "slack_side_effect_id": side_effect_id,
            }
        )


class SlackAddReactionTool:
    """Add a reaction to the current triggering Slack message."""

    name = "slack_add_reaction"
    description = (
        "Adds an emoji reaction to the current triggering Slack message. Use "
        "this for lightweight acknowledgements, status markers, or explicit "
        "user requests to react. The tool is scoped to the current Slack "
        "channel and triggering message by default."
    )
    parameters: JsonSchema = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": (
                    "Emoji reaction name without surrounding colons, for "
                    "example eyes, white_check_mark, or thumbsup."
                ),
            },
            "channel_id": {
                "type": "string",
                "description": (
                    "Optional current Slack channel ID. Omit unless copying the "
                    "current task channel exactly."
                ),
            },
            "message_ts": {
                "type": "string",
                "description": (
                    "Optional current triggering message timestamp. Omit to use "
                    "the task message timestamp. This cannot target another "
                    "message."
                ),
            },
        },
        "required": ["name"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        *,
        client: SlackActionClient,
        session: Session,
        task: Task,
        task_service: TaskService | None = None,
    ) -> None:
        self.client = client
        self.session = session
        self.task = task
        self.task_service = task_service or TaskService(session)

    def invoke(self, args: JsonObject) -> ToolResult:
        reaction = _coerce_reaction_name(args.get("name"))
        channel_id = _current_channel_arg(
            value=args.get("channel_id"),
            task=self.task,
            tool_name=self.name,
        )
        message_ts = _current_message_ts_arg(args.get("message_ts"), task=self.task)
        idempotency_key = slack_reaction_key(
            task_id=self.task.id,
            operation="reactions_add",
            channel_id=channel_id,
            message_ts=message_ts,
            reaction=reaction,
        )
        request: JsonObject = {
            "channel": channel_id,
            "timestamp": message_ts,
            "name": reaction,
        }
        result = SlackSideEffectOutbox(self.session).deliver(
            installation_id=self.task.installation_id,
            task_id=self.task.id,
            idempotency_key=idempotency_key,
            operation="reactions_add",
            purpose="tool_reaction",
            target_channel_id=channel_id,
            target_message_ts=message_ts,
            request=request,
            call=lambda: self.client.reactions_add(
                channel=channel_id,
                timestamp=message_ts,
                name=reaction,
            ),
        )
        response = _require_ok(result.response, "reactions.add")
        side_effect_id = str(result.side_effect.id)
        if not _task_event_exists(
            self.session,
            task=self.task,
            event_type=TaskEventType.log,
            side_effect_id=side_effect_id,
        ):
            self.task_service.append_event(
                self.task,
                TaskEventType.log,
                {
                    "message": "slack_reaction_added",
                    "channel": channel_id,
                    "message_ts": message_ts,
                    "reaction": reaction,
                    "purpose": "tool_reaction",
                    "slack_side_effect_id": side_effect_id,
                    "idempotency_key": idempotency_key,
                    "tool": self.name,
                    "deduped": result.deduped,
                    "deduped_by_slack": response.get("deduped_by_slack") is True,
                },
            )
        return ToolResult(
            output={
                "successful": True,
                "channel": channel_id,
                "message_ts": message_ts,
                "reaction": reaction,
                "deduped": result.deduped,
                "deduped_by_slack": response.get("deduped_by_slack") is True,
                "slack_side_effect_id": side_effect_id,
            }
        )


def _coerce_reply_text(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("slack_reply_thread 'text' is required")
    text = normalize_slack_mrkdwn(value)
    if not text:
        raise ValueError("slack_reply_thread 'text' cannot be empty")
    if len(text) > MAX_TOOL_REPLY_CHARS:
        raise ValueError(
            f"slack_reply_thread 'text' must be {MAX_TOOL_REPLY_CHARS} characters or fewer"
        )
    return text


def _current_channel_arg(*, value: object, task: Task, tool_name: str) -> str:
    if value is None:
        return task.slack_channel_id
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{tool_name} 'channel_id' must be a Slack channel ID")
    channel_id = value.strip()
    if channel_id != task.slack_channel_id:
        raise ValueError(f"{tool_name} can only target the current Slack channel")
    return channel_id


def _current_thread_arg(value: object, *, task: Task) -> str | None:
    current_thread_ts = task.slack_thread_ts or task.slack_message_ts
    if value is None:
        return current_thread_ts
    if not isinstance(value, str) or not value.strip():
        raise ValueError("slack_reply_thread 'thread_ts' must be a Slack timestamp")
    thread_ts = value.strip()
    allowed = {item for item in (task.slack_thread_ts, task.slack_message_ts) if item}
    if thread_ts not in allowed:
        raise ValueError("slack_reply_thread can only target the current Slack thread")
    return thread_ts


def _current_message_ts_arg(value: object, *, task: Task) -> str:
    current_message_ts = task.slack_message_ts
    if current_message_ts is None and _is_slack_timestamp(task.slack_thread_ts):
        current_message_ts = task.slack_thread_ts
    if value is None:
        if current_message_ts is None:
            raise ValueError("slack_add_reaction requires a current message timestamp")
        return current_message_ts
    if not isinstance(value, str) or not value.strip():
        raise ValueError("slack_add_reaction 'message_ts' must be a Slack timestamp")
    message_ts = value.strip()
    if not _is_slack_timestamp(message_ts):
        raise ValueError("slack_add_reaction 'message_ts' must be a Slack timestamp")
    if message_ts != current_message_ts:
        raise ValueError("slack_add_reaction can only target the current Slack message")
    return message_ts


def _coerce_reaction_name(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("slack_add_reaction 'name' is required")
    reaction = value.strip().strip(":")
    if not reaction or not REACTION_NAME_RE.match(reaction):
        raise ValueError("slack_add_reaction 'name' must be a valid Slack emoji name")
    return reaction


def _coerce_purpose(value: object, *, default: str) -> str:
    if value is None:
        return default
    if not isinstance(value, str):
        raise ValueError("slack_reply_thread 'purpose' must be a string")
    purpose = value.strip().lower().replace(" ", "_")
    if not purpose:
        return default
    if not re.match(r"^[a-z0-9_-]{1,40}$", purpose):
        raise ValueError("slack_reply_thread 'purpose' must be a short label")
    return f"tool_{purpose}" if not purpose.startswith("tool_") else purpose


def _post_thread_ts(*, channel_id: str, thread_ts: str | None) -> str | None:
    if channel_id.startswith("D"):
        return None
    return thread_ts


def _reply_idempotency_key(
    *,
    task: Task,
    purpose: str,
    channel_id: str,
    thread_ts: str | None,
    text: str,
) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"slack:tool_reply:{task.id}:{purpose}:{channel_id}:{thread_ts or 'root'}:{digest}"


def _require_ok(response: Mapping[str, Any], method: str) -> Mapping[str, Any]:
    ok = response.get("ok")
    if ok is False:
        error = response.get("error")
        raise RuntimeError(f"Slack {method} failed: {error or 'unknown_error'}")
    return response


def _response_ts(response: Mapping[str, Any]) -> str | None:
    ts = response.get("ts")
    if isinstance(ts, str) and ts:
        return ts
    message = response.get("message")
    if isinstance(message, Mapping):
        message_ts = message.get("ts")
        if isinstance(message_ts, str) and message_ts:
            return message_ts
    return None


def _is_slack_timestamp(value: str | None) -> bool:
    return isinstance(value, str) and SLACK_TS_RE.match(value) is not None


def _task_event_exists(
    session: Session,
    *,
    task: Task,
    event_type: TaskEventType,
    side_effect_id: str,
) -> bool:
    return (
        session.scalar(
            select(TaskEvent.id)
            .where(
                TaskEvent.task_id == task.id,
                TaskEvent.type == event_type,
                TaskEvent.payload["slack_side_effect_id"].as_string()
                == side_effect_id,
            )
            .limit(1)
        )
        is not None
    )
