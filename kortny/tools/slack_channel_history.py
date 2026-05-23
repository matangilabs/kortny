"""Slack channel history tool."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from kortny.tools.types import JsonObject, JsonSchema, ToolResult

DEFAULT_CHANNEL_HISTORY_LIMIT = 200
MAX_CHANNEL_HISTORY_LIMIT = 200
DEFAULT_CHANNEL_HISTORY_PAGE_LIMIT = 200
DEFAULT_INCLUDE_THREADS = False


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


class SlackChannelHistoryTool:
    """Read recent Slack channel messages for agent context."""

    name = "slack_channel_history"
    description = (
        "Reads recent Slack conversation history from the current task channel or "
        "a provided Slack channel ID. Use it before summarizing channel context, "
        "answering questions about recent Slack discussion, or grounding follow-up "
        "work in channel messages."
    )
    parameters: JsonSchema = {
        "type": "object",
        "properties": {
            "channel_id": {
                "type": "string",
                "description": (
                    "Slack channel ID to read. Omit this to use the current task "
                    "channel."
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

    def invoke(self, args: JsonObject) -> ToolResult:
        """Fetch channel history and return structured message records."""

        channel_id = _channel_id(args, self.default_channel_id)
        oldest_ts = _optional_string(args.get("oldest_ts"), "oldest_ts")
        latest_ts = _optional_string(args.get("latest_ts"), "latest_ts")
        limit = _limit(args, self.max_limit)
        include_threads = _include_threads(args)

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

        return ToolResult(
            output={
                "channel_id": channel_id,
                "oldest_ts": oldest_ts,
                "latest_ts": latest_ts,
                "limit": limit,
                "include_threads": include_threads,
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
    value = args.get("channel_id", default_channel_id)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            "slack_channel_history requires a non-empty 'channel_id' argument "
            "when no default task channel is available"
        )
    return value.strip()


def _optional_string(value: object, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"slack_channel_history {name!r} must be a non-empty string")
    return value.strip()


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
    return formatted


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
