"""Search Kortny's locally observed Slack message cache."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import and_, desc, or_, select
from sqlalchemy.orm import Session
from sqlalchemy.sql.elements import ColumnElement

from kortny.db.models import ObservationEvent, SlackChannelMembership, Task
from kortny.tools.types import JsonObject, JsonSchema, ToolResult

DEFAULT_OBSERVED_SEARCH_LIMIT = 10
MAX_OBSERVED_SEARCH_LIMIT = 50
MAX_SEARCH_QUERY_CHARS = 200
MAX_RESULT_TEXT_CHARS = 700
SEARCHABLE_OBSERVATION_EVENT_TYPES = (
    "message",
    "file_share",
    "channel_onboarding_intro",
)


@dataclass(frozen=True, slots=True)
class ObservedSlackSearchScope:
    """Resolved channel scope for an observed Slack search."""

    scope: str
    channel_ids: tuple[str, ...]
    blocked_reason: str | None = None


class SearchObservedSlackHistoryTool:
    """Search observed Slack messages without calling Slack APIs."""

    name = "search_observed_slack_history"
    description = (
        "Searches Kortny's locally observed Slack message cache. Use this before "
        "live Slack history when the user asks about past channel decisions, "
        "older local context, recurring topics, or whether something was "
        "mentioned before. Defaults to the current Slack channel. Use "
        "include_all_visible_channels only when the user asks for a broader "
        "workspace/channel search."
    )
    parameters: JsonSchema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Search terms or phrase to find in locally observed Slack "
                    "message text."
                ),
            },
            "channel_id": {
                "type": "string",
                "description": (
                    "Optional Slack channel ID. Omit for the current channel. "
                    "A different channel is only searched when "
                    "include_all_visible_channels is true and Kortny is known "
                    "to be active there."
                ),
            },
            "include_all_visible_channels": {
                "type": "boolean",
                "description": (
                    "Search channels where Kortny is known to be active for "
                    "this installation. Use only for explicit cross-channel "
                    "or workspace-history requests."
                ),
                "default": False,
            },
            "user_id": {
                "type": "string",
                "description": "Optional Slack user ID to filter observed messages.",
            },
            "oldest_ts": {
                "type": "string",
                "description": "Optional Slack timestamp lower bound.",
            },
            "latest_ts": {
                "type": "string",
                "description": "Optional Slack timestamp upper bound.",
            },
            "include_threads": {
                "type": "boolean",
                "description": "Whether to include thread replies in search results.",
                "default": True,
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of matches to return.",
                "minimum": 1,
                "maximum": MAX_OBSERVED_SEARCH_LIMIT,
                "default": DEFAULT_OBSERVED_SEARCH_LIMIT,
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def __init__(self, *, session: Session, task: Task) -> None:
        self.session = session
        self.task = task

    def invoke(self, args: JsonObject) -> ToolResult:
        query = _clean_query(args.get("query"))
        limit = _coerce_limit(args.get("limit"))
        include_threads = _coerce_bool(args.get("include_threads"), default=True)
        scope = self._resolve_scope(
            channel_id=_optional_string(args.get("channel_id")),
            include_all_visible_channels=_coerce_bool(
                args.get("include_all_visible_channels"),
                default=False,
            ),
        )
        if scope.blocked_reason is not None:
            return ToolResult(
                output={
                    "query": query,
                    "scope": scope.scope,
                    "channel_ids": list(scope.channel_ids),
                    "blocked": True,
                    "blocked_reason": scope.blocked_reason,
                    "match_count": 0,
                    "results": [],
                }
            )

        events = self._search_events(
            query=query,
            channel_ids=scope.channel_ids,
            user_id=_optional_string(args.get("user_id")),
            oldest_ts=_optional_string(args.get("oldest_ts")),
            latest_ts=_optional_string(args.get("latest_ts")),
            include_threads=include_threads,
            limit=limit,
        )
        results = [_format_search_result(event) for event in events]
        return ToolResult(
            output={
                "query": query,
                "scope": scope.scope,
                "channel_ids": list(scope.channel_ids),
                "searched_channel_count": len(scope.channel_ids),
                "match_count": len(results),
                "results": results,
                "source": "observation_events",
                "source_note": (
                    "Searched Kortny's local observed Slack cache only. No "
                    "Slack API call was made."
                ),
            }
        )

    def _resolve_scope(
        self,
        *,
        channel_id: str | None,
        include_all_visible_channels: bool,
    ) -> ObservedSlackSearchScope:
        current_channel_id = self.task.slack_channel_id
        if include_all_visible_channels:
            visible_channel_ids = self._visible_channel_ids()
            if channel_id is not None:
                if channel_id not in visible_channel_ids:
                    return ObservedSlackSearchScope(
                        scope="blocked_channel",
                        channel_ids=(),
                        blocked_reason=(
                            "Kortny is not known to be active in that channel."
                        ),
                    )
                return ObservedSlackSearchScope(
                    scope="specified_visible_channel",
                    channel_ids=(channel_id,),
                )
            return ObservedSlackSearchScope(
                scope="visible_channels",
                channel_ids=visible_channel_ids,
            )

        resolved_channel_id = channel_id or current_channel_id
        if resolved_channel_id is None:
            return ObservedSlackSearchScope(
                scope="blocked_channel",
                channel_ids=(),
                blocked_reason="No current Slack channel is available for this task.",
            )
        if channel_id is not None and channel_id != current_channel_id:
            return ObservedSlackSearchScope(
                scope="blocked_channel",
                channel_ids=(),
                blocked_reason=(
                    "This search is scoped to the current channel unless a "
                    "broader visible-channel search is explicitly requested."
                ),
            )
        return ObservedSlackSearchScope(
            scope="current_channel",
            channel_ids=(resolved_channel_id,),
        )

    def _visible_channel_ids(self) -> tuple[str, ...]:
        rows = self.session.scalars(
            select(SlackChannelMembership.channel_id).where(
                SlackChannelMembership.installation_id == self.task.installation_id,
                SlackChannelMembership.membership_status == "active",
            )
        ).all()
        channel_ids = tuple(dict.fromkeys(str(row) for row in rows if row))
        if self.task.slack_channel_id and self.task.slack_channel_id not in channel_ids:
            return (self.task.slack_channel_id, *channel_ids)
        return channel_ids

    def _search_events(
        self,
        *,
        query: str,
        channel_ids: Sequence[str],
        user_id: str | None,
        oldest_ts: str | None,
        latest_ts: str | None,
        include_threads: bool,
        limit: int,
    ) -> list[ObservationEvent]:
        conditions = [
            ObservationEvent.installation_id == self.task.installation_id,
            ObservationEvent.channel_id.in_(tuple(channel_ids)),
            ObservationEvent.event_type.in_(SEARCHABLE_OBSERVATION_EVENT_TYPES),
            ObservationEvent.message_ts.is_not(None),
            ObservationEvent.text_preview.is_not(None),
            ObservationEvent.purged_at.is_(None),
            _search_condition(query),
        ]
        if user_id is not None:
            conditions.append(ObservationEvent.user_id == user_id)
        if oldest_ts is not None:
            conditions.append(ObservationEvent.message_ts >= oldest_ts)
        if latest_ts is not None:
            conditions.append(ObservationEvent.message_ts <= latest_ts)
        if not include_threads:
            conditions.append(
                or_(
                    ObservationEvent.thread_ts.is_(None),
                    ObservationEvent.thread_ts == ObservationEvent.message_ts,
                )
            )

        return list(
            self.session.scalars(
                select(ObservationEvent)
                .where(*conditions)
                .order_by(
                    desc(ObservationEvent.message_ts),
                    desc(ObservationEvent.observed_at),
                )
                .limit(limit)
            )
        )


def _search_condition(query: str) -> ColumnElement[bool]:
    tokens = _query_tokens(query)
    if not tokens:
        tokens = (query,)
    return and_(
        *[
            ObservationEvent.text_preview.ilike(
                f"%{_escape_like_token(token)}%",
                escape="\\",
            )
            for token in tokens
        ]
    )


def _format_search_result(event: ObservationEvent) -> JsonObject:
    thread_ts = (
        event.thread_ts
        if event.thread_ts is not None and event.thread_ts != event.message_ts
        else None
    )
    return {
        "observation_event_id": str(event.id),
        "channel_id": event.channel_id,
        "user_id": event.user_id,
        "message_ts": event.message_ts,
        "thread_ts": thread_ts,
        "text": _truncate_text(event.text_preview or "", MAX_RESULT_TEXT_CHARS),
        "event_type": event.event_type,
        "file_id": event.file_id,
        "observed_at": event.observed_at.isoformat(),
        "citation": _citation(
            channel_id=event.channel_id,
            message_ts=event.message_ts,
            thread_ts=thread_ts,
        ),
    }


def _citation(
    *,
    channel_id: str,
    message_ts: str | None,
    thread_ts: str | None,
) -> str:
    if message_ts is None:
        return channel_id
    if thread_ts is not None:
        return f"{channel_id}:{thread_ts}:{message_ts}"
    return f"{channel_id}:{message_ts}"


def _clean_query(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("search_observed_slack_history requires string 'query'")
    cleaned = " ".join(value.strip().split())
    if not cleaned:
        raise ValueError("search_observed_slack_history 'query' cannot be empty")
    return cleaned[:MAX_SEARCH_QUERY_CHARS]


def _query_tokens(query: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in query.casefold().split()
        if len(token) >= 2 and not token.isspace()
    )[:8]


def _escape_like_token(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _optional_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _coerce_bool(value: object, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raise ValueError("search_observed_slack_history boolean arguments must be booleans")


def _coerce_limit(value: object) -> int:
    if value is None:
        return DEFAULT_OBSERVED_SEARCH_LIMIT
    if not isinstance(value, int):
        raise ValueError("search_observed_slack_history 'limit' must be an integer")
    return min(max(value, 1), MAX_OBSERVED_SEARCH_LIMIT)


def _truncate_text(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 1].rstrip() + "..."
