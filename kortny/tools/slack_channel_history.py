"""Slack channel history tool."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Protocol
from uuid import UUID

from slack_sdk.errors import SlackApiError
from sqlalchemy import desc, or_, select
from sqlalchemy.orm import Session

from kortny.db.models import ObservationEvent, Task, TaskEvent, TaskEventType
from kortny.tools.channel_access import ChannelAccessGate
from kortny.tools.types import JsonObject, JsonSchema, ToolResult

DEFAULT_CHANNEL_HISTORY_LIMIT = 200
MAX_CHANNEL_HISTORY_LIMIT = 200
DEFAULT_CHANNEL_HISTORY_PAGE_LIMIT = 200
DEFAULT_INCLUDE_THREADS = False
DEFAULT_HISTORY_SOURCE = "auto"
CACHED_HISTORY_EVENT_TYPES = (
    "message",
    "file_share",
    "channel_onboarding_intro",
)
logger = logging.getLogger(__name__)


class SlackChannelHistoryError(RuntimeError):
    """Raised when Slack channel history cannot be fetched."""


class SlackChannelHistoryClient(Protocol):
    """Subset of Slack WebClient used for channel history retrieval."""

    def conversations_history(
        self,
        *,
        channel: str,
        cursor: str | None = None,
        inclusive: bool | None = None,
        limit: int | None = None,
        latest: str | None = None,
        oldest: str | None = None,
    ) -> Any:
        """Fetch a page of channel messages."""

    def conversations_replies(
        self,
        *,
        channel: str,
        ts: str,
        cursor: str | None = None,
        inclusive: bool | None = None,
        limit: int | None = None,
        latest: str | None = None,
        oldest: str | None = None,
    ) -> Any:
        """Fetch a page of thread replies."""


class ChannelHistoryCache(Protocol):
    """Local channel context cache used before polling Slack history APIs."""

    def fetch_messages(
        self,
        *,
        channel_id: str,
        oldest_ts: str | None,
        latest_ts: str | None,
        limit: int,
        include_threads: bool,
    ) -> list[JsonObject]:
        """Return cached messages in ascending Slack timestamp order."""
        ...


class ObservationChannelHistoryCache:
    """Read recent Slack context from Observe observation events."""

    def __init__(self, session: Session, *, installation_id: UUID) -> None:
        self.session = session
        self.installation_id = installation_id

    def fetch_messages(
        self,
        *,
        channel_id: str,
        oldest_ts: str | None,
        latest_ts: str | None,
        limit: int,
        include_threads: bool,
    ) -> list[JsonObject]:
        """Return cached observed messages for a channel."""

        query = select(ObservationEvent).where(
            ObservationEvent.installation_id == self.installation_id,
            ObservationEvent.channel_id == channel_id,
            ObservationEvent.event_type.in_(CACHED_HISTORY_EVENT_TYPES),
            ObservationEvent.message_ts.is_not(None),
            ObservationEvent.purged_at.is_(None),
        )
        if oldest_ts is not None:
            query = query.where(ObservationEvent.message_ts >= oldest_ts)
        if latest_ts is not None:
            query = query.where(ObservationEvent.message_ts <= latest_ts)
        if not include_threads:
            query = query.where(
                or_(
                    ObservationEvent.thread_ts.is_(None),
                    ObservationEvent.thread_ts == ObservationEvent.message_ts,
                )
            )

        events = self.session.scalars(
            query.order_by(
                desc(ObservationEvent.message_ts),
                desc(ObservationEvent.observed_at),
            ).limit(limit)
        ).all()
        messages = [
            message
            for event in events
            if (message := _format_observation_event(event)) is not None
        ]
        messages.extend(
            self._fetch_posted_messages(
                channel_id=channel_id,
                oldest_ts=oldest_ts,
                latest_ts=latest_ts,
                limit=limit,
                include_threads=include_threads,
            )
        )
        return sorted(
            sorted(messages, key=_sort_ts, reverse=True)[:limit],
            key=_sort_ts,
        )

    def _fetch_posted_messages(
        self,
        *,
        channel_id: str,
        oldest_ts: str | None,
        latest_ts: str | None,
        limit: int,
        include_threads: bool,
    ) -> list[JsonObject]:
        """Return Kortny messages posted in the channel from task events."""

        message_ts = TaskEvent.payload["message_ts"].as_string()
        thread_ts = TaskEvent.payload["thread_ts"].as_string()
        query = (
            select(Task, TaskEvent)
            .join(TaskEvent, TaskEvent.task_id == Task.id)
            .where(
                Task.installation_id == self.installation_id,
                Task.slack_channel_id == channel_id,
                TaskEvent.type == TaskEventType.message_posted,
                message_ts.is_not(None),
            )
        )
        if oldest_ts is not None:
            query = query.where(message_ts >= oldest_ts)
        if latest_ts is not None:
            query = query.where(message_ts <= latest_ts)
        if not include_threads:
            query = query.where(or_(thread_ts.is_(None), thread_ts == message_ts))

        rows = self.session.execute(
            query.order_by(desc(message_ts), desc(TaskEvent.created_at)).limit(limit)
        ).all()
        return [
            message
            for task, event in rows
            if (message := _format_posted_message_event(task, event)) is not None
        ]


class SlackChannelHistoryTool:
    """Read recent Slack channel messages for agent context."""

    name = "slack_channel_history"
    description = (
        "Reads recent Slack conversation history. Use it before summarizing the "
        "current channel, answering questions about recent Slack discussion, or "
        "grounding follow-up work in channel messages. Omit channel_id for the "
        "current task channel; only pass channel_id when the user explicitly "
        "provided a different Slack channel ID. Never guess channel IDs. Includes "
        "compact Slack file metadata when messages have file attachments."
    )
    parameters: JsonSchema = {
        "type": "object",
        "properties": {
            "channel_id": {
                "type": "string",
                "description": (
                    "Optional Slack channel ID to read. For phrases like 'this "
                    "channel', 'the channel', 'above', 'recent discussion', or "
                    "'channel history', omit this field so the tool uses the "
                    "current task channel. Only include this when the user "
                    "explicitly provides a Slack channel ID such as C123ABC. Do "
                    "not infer, remember, or guess channel IDs."
                ),
            },
            "oldest_ts": {
                "type": "string",
                "description": (
                    "Optional Slack timestamp lower bound for messages to include."
                ),
            },
            "latest_ts": {
                "type": "string",
                "description": (
                    "Optional Slack timestamp upper bound for messages to include."
                ),
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of messages to return.",
                "minimum": 1,
                "maximum": MAX_CHANNEL_HISTORY_LIMIT,
                "default": DEFAULT_CHANNEL_HISTORY_LIMIT,
            },
            "include_threads": {
                "type": "boolean",
                "description": (
                    "Whether to fan out active thread replies with "
                    "conversations.replies. Enable this when thread detail matters."
                ),
                "default": DEFAULT_INCLUDE_THREADS,
            },
            "source": {
                "type": "string",
                "description": (
                    "Optional source policy. Use 'auto' for the default local "
                    "event cache first, Slack API fallback behavior. Use "
                    "'slack_api' only when the user explicitly needs a live "
                    "Slack backfill."
                ),
                "enum": ["auto", "slack_api"],
                "default": DEFAULT_HISTORY_SOURCE,
            },
        },
        "additionalProperties": False,
    }

    def __init__(
        self,
        client: SlackChannelHistoryClient,
        *,
        default_channel_id: str | None = None,
        page_limit: int = DEFAULT_CHANNEL_HISTORY_PAGE_LIMIT,
        max_limit: int = MAX_CHANNEL_HISTORY_LIMIT,
        cache: ChannelHistoryCache | None = None,
        access_gate: ChannelAccessGate | None = None,
    ) -> None:
        if page_limit < 1:
            raise ValueError("page_limit must be at least 1")
        if max_limit < 1:
            raise ValueError("max_limit must be at least 1")

        self.client = client
        self.default_channel_id = (
            default_channel_id.strip()
            if isinstance(default_channel_id, str) and default_channel_id.strip()
            else None
        )
        self.page_limit = min(page_limit, max_limit)
        self.max_limit = max_limit
        self.cache = cache
        self.access_gate = access_gate

    def invoke(self, args: JsonObject) -> ToolResult:
        """Fetch channel history and return structured message records."""

        channel_id = _channel_id(args, self.default_channel_id)
        oldest_ts = _optional_string(args.get("oldest_ts"), "oldest_ts")
        latest_ts = _optional_string(args.get("latest_ts"), "latest_ts")
        limit = _limit(args, self.max_limit)
        include_threads = _include_threads(args)
        source = _history_source(args)

        # Enforce asker membership before reading any channel data.
        # Raises RecoverableToolError(code="channel_access_denied") for
        # user-initiated tasks targeting a channel the asker is not in.
        # No-op for synthetic/scheduled tasks and same-channel reads.
        if self.access_gate is not None:
            self.access_gate.check(channel_id)

        if self.cache is not None and source == "auto":
            cached_messages = self.cache.fetch_messages(
                channel_id=channel_id,
                oldest_ts=oldest_ts,
                latest_ts=latest_ts,
                limit=limit,
                include_threads=include_threads,
            )
            if cached_messages:
                return ToolResult(
                    output={
                        "channel_id": channel_id,
                        "oldest_ts": oldest_ts,
                        "latest_ts": latest_ts,
                        "limit": limit,
                        "include_threads": include_threads,
                        "context_source": "observation_cache",
                        "cache_hit": True,
                        "slack_api_called": False,
                        "message_count": len(cached_messages),
                        "messages": cached_messages,
                        "provenance": {
                            "source": "observation_events",
                            "status": (
                                "complete_by_limit"
                                if len(cached_messages) >= limit
                                else "partial_recent_cache"
                            ),
                            "hint": (
                                "This response used locally observed Slack "
                                "events to avoid polling Slack history APIs."
                            ),
                        },
                    }
                )

        try:
            root_messages = self._fetch_history_roots(
                channel_id=channel_id,
                oldest_ts=oldest_ts,
                latest_ts=latest_ts,
                limit=limit,
            )
            messages = self._build_output_messages(
                channel_id=channel_id,
                root_messages=root_messages,
                oldest_ts=oldest_ts,
                latest_ts=latest_ts,
                limit=limit,
                include_threads=include_threads,
            )
        except SlackApiError as exc:
            error_code = _slack_api_error_code(exc.response)
            logger.warning(
                "slack_channel_history live api failed channel=%s code=%s retry_after=%s",
                channel_id,
                error_code,
                _slack_retry_after_seconds(exc.response),
            )
            # Auto-fallback: if model supplied a bad channel_id that returned
            # channel_not_found, retry with the task's default channel so the
            # call doesn't hard-fail on hallucinated IDs (HIG-295 Part 3).
            requested_channel_id = _optional_string(
                args.get("channel_id"), "channel_id"
            )
            if (
                error_code in ("channel_not_found", "not_in_channel")
                and self.default_channel_id is not None
                and channel_id != self.default_channel_id
            ):
                logger.info(
                    "slack_channel_history channel_not_found on %s; "
                    "falling back to default channel %s",
                    channel_id,
                    self.default_channel_id,
                )
                try:
                    root_messages = self._fetch_history_roots(
                        channel_id=self.default_channel_id,
                        oldest_ts=oldest_ts,
                        latest_ts=latest_ts,
                        limit=limit,
                    )
                    fallback_messages = self._build_output_messages(
                        channel_id=self.default_channel_id,
                        root_messages=root_messages,
                        oldest_ts=oldest_ts,
                        latest_ts=latest_ts,
                        limit=limit,
                        include_threads=include_threads,
                    )
                    return ToolResult(
                        output={
                            "channel_id": self.default_channel_id,
                            "oldest_ts": oldest_ts,
                            "latest_ts": latest_ts,
                            "limit": limit,
                            "include_threads": include_threads,
                            "context_source": "slack_api",
                            "cache_hit": False,
                            "slack_api_called": True,
                            "message_count": len(fallback_messages),
                            "messages": fallback_messages,
                            "fallback_note": (
                                f"Requested channel {channel_id!r} was not found "
                                f"or not accessible; fell back to the current task "
                                f"channel {self.default_channel_id!r}."
                            ),
                        }
                    )
                except (SlackApiError, SlackChannelHistoryError) as fallback_exc:
                    logger.warning(
                        "slack_channel_history fallback also failed "
                        "default_channel=%s error=%s",
                        self.default_channel_id,
                        fallback_exc,
                    )
            return _recoverable_history_result(
                channel_id=channel_id,
                requested_channel_id=requested_channel_id,
                default_channel_id=self.default_channel_id,
                oldest_ts=oldest_ts,
                latest_ts=latest_ts,
                limit=limit,
                include_threads=include_threads,
                code=error_code,
                message=f"Slack API failed: {error_code}",
                details=_slack_api_error_details(exc.response),
            )
        except SlackChannelHistoryError as exc:
            code = _history_error_code(str(exc))
            return _recoverable_history_result(
                channel_id=channel_id,
                requested_channel_id=_optional_string(
                    args.get("channel_id"), "channel_id"
                ),
                default_channel_id=self.default_channel_id,
                oldest_ts=oldest_ts,
                latest_ts=latest_ts,
                limit=limit,
                include_threads=include_threads,
                code=code,
                message=str(exc),
            )

        return ToolResult(
            output={
                "channel_id": channel_id,
                "oldest_ts": oldest_ts,
                "latest_ts": latest_ts,
                "limit": limit,
                "include_threads": include_threads,
                "context_source": "slack_api",
                "cache_hit": False,
                "slack_api_called": True,
                "message_count": len(messages),
                "messages": messages,
            }
        )

    def _fetch_history_roots(
        self,
        *,
        channel_id: str,
        oldest_ts: str | None,
        latest_ts: str | None,
        limit: int,
    ) -> list[Mapping[str, Any]]:
        cursor: str | None = None
        messages: list[Mapping[str, Any]] = []
        while len(messages) < limit:
            page_size = min(self.page_limit, limit - len(messages))
            response = self.client.conversations_history(
                channel=channel_id,
                cursor=cursor,
                inclusive=True,
                limit=page_size,
                latest=latest_ts,
                oldest=oldest_ts,
            )
            payload = _response_payload(response, "conversations.history")
            for message in _response_messages(payload):
                if _message_ts(message) is None:
                    continue
                messages.append(message)
                if len(messages) >= limit:
                    break

            cursor = _next_cursor(payload)
            if not cursor:
                break

        return sorted(messages, key=_sort_ts)

    def _build_output_messages(
        self,
        *,
        channel_id: str,
        root_messages: list[Mapping[str, Any]],
        oldest_ts: str | None,
        latest_ts: str | None,
        limit: int,
        include_threads: bool,
    ) -> list[JsonObject]:
        output: list[JsonObject] = []
        for message in root_messages:
            formatted = _format_message(message)
            if formatted is None:
                continue
            output.append(formatted)
            if len(output) >= limit:
                break

            if include_threads and _reply_count(message) > 0:
                root_ts = formatted["ts"]
                output.extend(
                    self._fetch_thread_replies(
                        channel_id=channel_id,
                        root_ts=root_ts,
                        oldest_ts=oldest_ts,
                        latest_ts=latest_ts,
                        remaining=limit - len(output),
                    )
                )
                if len(output) >= limit:
                    break

        return output

    def _fetch_thread_replies(
        self,
        *,
        channel_id: str,
        root_ts: str,
        oldest_ts: str | None,
        latest_ts: str | None,
        remaining: int,
    ) -> list[JsonObject]:
        if remaining < 1:
            return []

        cursor: str | None = None
        replies: list[JsonObject] = []
        while len(replies) < remaining:
            page_size = min(self.page_limit, remaining - len(replies) + 1)
            response = self.client.conversations_replies(
                channel=channel_id,
                ts=root_ts,
                cursor=cursor,
                inclusive=True,
                limit=page_size,
                latest=latest_ts,
                oldest=oldest_ts,
            )
            payload = _response_payload(response, "conversations.replies")
            for message in _response_messages(payload):
                ts = _message_ts(message)
                if ts is None or ts == root_ts:
                    continue
                formatted = _format_message(message, fallback_thread_ts=root_ts)
                if formatted is None:
                    continue
                replies.append(formatted)
                if len(replies) >= remaining:
                    break

            cursor = _next_cursor(payload)
            if not cursor:
                break

        return sorted(replies, key=_sort_ts)


def _channel_id(args: Mapping[str, Any], default_channel_id: str | None) -> str:
    value = args.get("channel_id")
    if value is None or (isinstance(value, str) and not value.strip()):
        if default_channel_id is not None:
            return default_channel_id
        raise ValueError(
            "slack_channel_history requires a non-empty 'channel_id' argument "
            "when no default task channel is available"
        )
    if not isinstance(value, str):
        raise ValueError("slack_channel_history 'channel_id' must be a string")
    return value.strip()


def _optional_string(value: object, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"slack_channel_history {name!r} must be a non-empty string")
    stripped = value.strip()
    return stripped or None


def _limit(args: Mapping[str, Any], max_limit: int) -> int:
    value = args.get("limit", DEFAULT_CHANNEL_HISTORY_LIMIT)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("slack_channel_history 'limit' must be an integer")
    if value < 1:
        raise ValueError("slack_channel_history 'limit' must be at least 1")
    return min(value, max_limit)


def _include_threads(args: Mapping[str, Any]) -> bool:
    value = args.get("include_threads", DEFAULT_INCLUDE_THREADS)
    if not isinstance(value, bool):
        raise ValueError("slack_channel_history 'include_threads' must be a boolean")
    return value


def _history_source(args: Mapping[str, Any]) -> str:
    value = args.get("source", DEFAULT_HISTORY_SOURCE)
    if value is None:
        return DEFAULT_HISTORY_SOURCE
    if not isinstance(value, str) or value not in {"auto", "slack_api"}:
        raise ValueError("slack_channel_history 'source' must be 'auto' or 'slack_api'")
    return value


def _response_payload(response: object, method: str) -> Mapping[str, Any]:
    if isinstance(response, Mapping):
        payload = response
    else:
        data = getattr(response, "data", None)
        if not isinstance(data, Mapping):
            raise SlackChannelHistoryError(f"{method} returned a non-object response")
        payload = data

    if payload.get("ok") is False:
        error = payload.get("error")
        if not isinstance(error, str) or not error:
            error = "unknown_error"
        raise SlackChannelHistoryError(f"{method} failed: {error}")

    return payload


def _recoverable_history_result(
    *,
    channel_id: str,
    requested_channel_id: str | None,
    default_channel_id: str | None,
    oldest_ts: str | None,
    latest_ts: str | None,
    limit: int,
    include_threads: bool,
    code: str,
    message: str,
    details: JsonObject | None = None,
) -> ToolResult:
    error: JsonObject = {
        "code": code,
        "message": message,
        "recoverable": True,
        "hint": _history_error_hint(
            requested_channel_id=requested_channel_id,
            default_channel_id=default_channel_id,
        ),
    }
    if details:
        error["details"] = details
    return ToolResult(
        output={
            "channel_id": channel_id,
            "oldest_ts": oldest_ts,
            "latest_ts": latest_ts,
            "limit": limit,
            "include_threads": include_threads,
            "message_count": 0,
            "messages": [],
            "error": error,
        }
    )


def _history_error_hint(
    *,
    requested_channel_id: str | None,
    default_channel_id: str | None,
) -> str:
    if (
        requested_channel_id is not None
        and default_channel_id is not None
        and requested_channel_id != default_channel_id
    ):
        return (
            "If the user means the current Slack channel, retry "
            "slack_channel_history without channel_id."
        )
    return (
        "Use prior thread context if it is sufficient. Otherwise ask the user "
        "to add Kortny to the channel or provide an accessible Slack channel."
    )


def _history_error_code(message: str) -> str:
    marker = " failed: "
    if marker in message:
        code = message.rsplit(marker, maxsplit=1)[-1].strip()
        if code:
            return code
    return "slack_channel_history_failed"


def _slack_api_error_code(response: object) -> str:
    if _slack_response_status_code(response) == 429:
        return "rate_limited"

    if isinstance(response, Mapping):
        error = response.get("error")
        if isinstance(error, str) and error:
            return error

    data = getattr(response, "data", None)
    if isinstance(data, Mapping):
        error = data.get("error")
        if isinstance(error, str) and error:
            return error

    return "slack_api_error"


def _slack_api_error_details(response: object) -> JsonObject | None:
    retry_after = _slack_retry_after_seconds(response)
    if retry_after is None:
        return None
    return {"retry_after_seconds": retry_after}


def _slack_response_status_code(response: object) -> int | None:
    status_code = getattr(response, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    if isinstance(response, Mapping):
        value = response.get("status_code")
        if isinstance(value, int):
            return value
    return None


def _slack_retry_after_seconds(response: object) -> int | None:
    headers = getattr(response, "headers", None)
    if not isinstance(headers, Mapping) and isinstance(response, Mapping):
        headers = response.get("headers")
    if not isinstance(headers, Mapping):
        return None
    value = headers.get("Retry-After") or headers.get("retry-after")
    if isinstance(value, str) and value.isdigit():
        return int(value)
    if isinstance(value, int) and value >= 0:
        return value
    return None


def _response_messages(response: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    raw_messages = response.get("messages")
    if not isinstance(raw_messages, list):
        return ()
    return tuple(message for message in raw_messages if isinstance(message, Mapping))


def _next_cursor(response: Mapping[str, Any]) -> str | None:
    metadata = response.get("response_metadata")
    if not isinstance(metadata, Mapping):
        return None
    cursor = metadata.get("next_cursor")
    if isinstance(cursor, str) and cursor:
        return cursor
    return None


def _format_message(
    message: Mapping[str, Any],
    *,
    fallback_thread_ts: str | None = None,
) -> JsonObject | None:
    ts = _message_ts(message)
    text = message.get("text")
    if ts is None or not isinstance(text, str):
        return None

    formatted: JsonObject = {
        "user": _optional_message_string(message.get("user")),
        "ts": ts,
        "text": text,
        "thread_ts": _optional_message_string(message.get("thread_ts"))
        or fallback_thread_ts,
        "reply_count": _reply_count(message),
    }
    bot_id = _optional_message_string(message.get("bot_id"))
    if bot_id is not None:
        formatted["bot_id"] = bot_id
    subtype = _optional_message_string(message.get("subtype"))
    if subtype is not None:
        formatted["subtype"] = subtype
    files = _format_files(message.get("files"))
    if files:
        formatted["files"] = files
    return formatted


def _format_observation_event(event: ObservationEvent) -> JsonObject | None:
    if event.message_ts is None:
        return None
    formatted: JsonObject = {
        "user": event.user_id,
        "ts": event.message_ts,
        "text": event.text_preview or "",
        "thread_ts": (
            event.thread_ts
            if event.thread_ts is not None and event.thread_ts != event.message_ts
            else None
        ),
        "reply_count": 0,
        "source": "observation_cache",
        "observation_event_id": str(event.id),
        "observed_at": event.observed_at.isoformat(),
    }
    if event.event_type == "channel_onboarding_intro":
        formatted["subtype"] = "channel_onboarding_intro"
    if event.file_id is not None:
        formatted["files"] = [{"id": event.file_id}]
    return formatted


def _format_posted_message_event(task: Task, event: TaskEvent) -> JsonObject | None:
    payload = event.payload
    message_ts = _optional_message_string(payload.get("message_ts"))
    text = payload.get("text")
    if message_ts is None or not isinstance(text, str):
        return None

    thread_ts = _optional_message_string(payload.get("thread_ts"))
    formatted: JsonObject = {
        "user": None,
        "author": "Kortny",
        "is_bot": True,
        "ts": message_ts,
        "text": text,
        "thread_ts": (
            thread_ts if thread_ts is not None and thread_ts != message_ts else None
        ),
        "reply_count": 0,
        "source": "task_events",
        "task_id": str(task.id),
        "task_event_id": event.id,
        "posted_at": event.created_at.isoformat(),
    }
    purpose = _optional_message_string(payload.get("purpose"))
    if purpose is not None:
        formatted["purpose"] = purpose
    return formatted


def _format_files(raw_files: object) -> list[JsonObject]:
    if not isinstance(raw_files, list):
        return []

    files: list[JsonObject] = []
    for raw_file in raw_files:
        if not isinstance(raw_file, Mapping):
            continue
        file_id = _optional_message_string(raw_file.get("id"))
        if file_id is None:
            continue

        file: JsonObject = {"id": file_id}
        for source_key, target_key in (
            ("name", "name"),
            ("title", "title"),
            ("mimetype", "mimetype"),
            ("filetype", "filetype"),
            ("user", "user"),
        ):
            value = _optional_message_string(raw_file.get(source_key))
            if value is not None:
                file[target_key] = value

        size = _optional_positive_int(raw_file.get("size"))
        if size is not None:
            file["size_bytes"] = size
        created = _optional_positive_int(raw_file.get("created"))
        if created is not None:
            file["created"] = created

        files.append(file)

    return files


def _optional_positive_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    return None


def _optional_message_string(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _message_ts(message: Mapping[str, Any]) -> str | None:
    ts = message.get("ts")
    if isinstance(ts, str) and ts:
        return ts
    return None


def _reply_count(message: Mapping[str, Any]) -> int:
    value = message.get("reply_count", 0)
    if isinstance(value, bool):
        return 0
    if isinstance(value, int) and value > 0:
        return value
    return 0


def _sort_ts(message: Mapping[str, Any]) -> str:
    ts = message.get("ts")
    if isinstance(ts, str):
        return ts
    return ""
