"""Read models for the operator dashboard."""

from __future__ import annotations

import json
import math
import uuid
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from typing import Any

from sqlalchemy import Select, func, select
from sqlalchemy.engine import Row
from sqlalchemy.orm import Session
from sqlalchemy.sql.elements import ColumnElement

from kortny.db.models import Artifact, LLMUsage, SlackIdentity, Task, TaskEvent

DEFAULT_PAGE_SIZE = 25
MAX_PAGE_SIZE = 100


@dataclass(frozen=True)
class TaskListItem:
    task: Task
    channel: IdentityLabel
    user: IdentityLabel
    models: tuple[str, ...]
    turn_count: int


@dataclass(frozen=True)
class TaskListPage:
    items: tuple[TaskListItem, ...]
    page: int
    page_size: int
    total_count: int

    @property
    def total_pages(self) -> int:
        if self.total_count == 0:
            return 1
        return math.ceil(self.total_count / self.page_size)

    @property
    def previous_page(self) -> int | None:
        if self.page <= 1:
            return None
        return self.page - 1

    @property
    def next_page(self) -> int | None:
        if self.page >= self.total_pages:
            return None
        return self.page + 1


@dataclass(frozen=True)
class TaskDetail:
    task: Task
    channel: IdentityLabel
    user: IdentityLabel
    events: tuple[TaskEvent, ...]
    timeline: tuple[TimelineEvent, ...]
    usage: tuple[LLMUsage, ...]
    artifacts: tuple[Artifact, ...]


@dataclass(frozen=True)
class IdentityLabel:
    name: str
    slack_id: str
    found: bool

    @property
    def secondary(self) -> str | None:
        if self.found and self.name != self.slack_id:
            return self.slack_id
        return None


@dataclass(frozen=True)
class TimelineBadge:
    label: str
    tone: str = "neutral"


@dataclass(frozen=True)
class TimelineMetric:
    label: str
    value: str


@dataclass(frozen=True)
class TimelineEvent:
    seq: int
    event_type: str
    tone: str
    title: str
    summary: str
    created_at: datetime
    badges: tuple[TimelineBadge, ...]
    metrics: tuple[TimelineMetric, ...]
    payload_json: str


@dataclass(frozen=True)
class AggregateRow:
    key: str
    calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: Decimal
    label: IdentityLabel | None = None

    @property
    def display_key(self) -> str:
        if self.label is not None:
            return self.label.name
        return self.key

    @property
    def secondary_key(self) -> str | None:
        if self.label is not None:
            return self.label.secondary
        return None


@dataclass(frozen=True)
class DailyUsageRow:
    day: date
    calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: Decimal


@dataclass(frozen=True)
class UsageAggregate:
    start: datetime | None
    end: datetime | None
    by_model: tuple[AggregateRow, ...]
    by_user: tuple[AggregateRow, ...]
    by_channel: tuple[AggregateRow, ...]
    by_day: tuple[DailyUsageRow, ...]


def list_tasks(
    session: Session,
    *,
    page: int = 1,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> TaskListPage:
    """Return a paginated dashboard task list."""

    normalized_page = max(page, 1)
    normalized_size = min(max(page_size, 1), MAX_PAGE_SIZE)
    offset = (normalized_page - 1) * normalized_size

    total_count = session.scalar(select(func.count()).select_from(Task)) or 0
    tasks = tuple(
        session.scalars(
            select(Task)
            .order_by(Task.created_at.desc(), Task.id.desc())
            .offset(offset)
            .limit(normalized_size)
        )
    )
    usage_by_task = _usage_by_task(session, [task.id for task in tasks])
    identities = _identity_map(session, tasks)
    items = tuple(
        TaskListItem(
            task=task,
            channel=_identity_label(
                identities,
                installation_id=task.installation_id,
                kind="channel",
                slack_id=task.slack_channel_id,
            ),
            user=_identity_label(
                identities,
                installation_id=task.installation_id,
                kind="user",
                slack_id=task.slack_user_id,
            ),
            models=tuple(sorted({usage.model for usage in usage_by_task[task.id]})),
            turn_count=len(usage_by_task[task.id]),
        )
        for task in tasks
    )
    return TaskListPage(
        items=items,
        page=normalized_page,
        page_size=normalized_size,
        total_count=total_count,
    )


def get_task_detail(session: Session, task_id: uuid.UUID) -> TaskDetail | None:
    """Return one task and its child rows."""

    task = session.get(Task, task_id)
    if task is None:
        return None
    events = tuple(
        session.scalars(
            select(TaskEvent)
            .where(TaskEvent.task_id == task_id)
            .order_by(TaskEvent.seq.asc())
        )
    )
    usage = tuple(
        session.scalars(
            select(LLMUsage)
            .where(LLMUsage.task_id == task_id)
            .order_by(LLMUsage.created_at.asc(), LLMUsage.id.asc())
        )
    )
    artifacts = tuple(
        session.scalars(
            select(Artifact)
            .where(Artifact.task_id == task_id)
            .order_by(Artifact.created_at.asc(), Artifact.id.asc())
        )
    )
    identities = _identity_map(session, (task,))
    return TaskDetail(
        task=task,
        channel=_identity_label(
            identities,
            installation_id=task.installation_id,
            kind="channel",
            slack_id=task.slack_channel_id,
        ),
        user=_identity_label(
            identities,
            installation_id=task.installation_id,
            kind="user",
            slack_id=task.slack_user_id,
        ),
        events=events,
        timeline=tuple(_timeline_event(event) for event in events),
        usage=usage,
        artifacts=artifacts,
    )


def get_usage_aggregate(
    session: Session,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
) -> UsageAggregate:
    """Return dashboard usage rollups."""

    usage_filter = _usage_filter(start=start, end=end)
    by_model_rows = session.execute(
        _aggregate_query(LLMUsage.model, usage_filter).order_by(
            func.sum(LLMUsage.cost_usd).desc()
        )
    ).all()
    by_user_raw_rows = session.execute(
        select(
            Task.installation_id,
            Task.slack_user_id,
            func.count(LLMUsage.id),
            func.coalesce(func.sum(LLMUsage.input_tokens), 0),
            func.coalesce(func.sum(LLMUsage.output_tokens), 0),
            func.coalesce(func.sum(LLMUsage.cost_usd), 0),
        )
        .join(Task, Task.id == LLMUsage.task_id)
        .where(*usage_filter)
        .group_by(Task.installation_id, Task.slack_user_id)
        .order_by(func.sum(LLMUsage.cost_usd).desc())
    ).all()
    user_identities = _identity_map_from_keys(
        session,
        (
            (row[0], "user", row[1])
            for row in by_user_raw_rows
            if row[0] is not None and row[1] is not None
        ),
    )
    by_user_rows = tuple(
        _aggregate_row(
            (row[1], row[2], row[3], row[4], row[5]),
            label=_identity_label(
                user_identities,
                installation_id=row[0],
                kind="user",
                slack_id=row[1],
            ),
        )
        for row in by_user_raw_rows
    )
    by_channel_raw_rows = session.execute(
        select(
            Task.installation_id,
            Task.slack_channel_id,
            func.count(LLMUsage.id),
            func.coalesce(func.sum(LLMUsage.input_tokens), 0),
            func.coalesce(func.sum(LLMUsage.output_tokens), 0),
            func.coalesce(func.sum(LLMUsage.cost_usd), 0),
        )
        .join(Task, Task.id == LLMUsage.task_id)
        .where(*usage_filter, ~Task.slack_channel_id.startswith("D"))
        .group_by(Task.installation_id, Task.slack_channel_id)
        .order_by(func.sum(LLMUsage.cost_usd).desc())
    ).all()
    channel_identities = _identity_map_from_keys(
        session,
        (
            (row[0], "channel", row[1])
            for row in by_channel_raw_rows
            if row[0] is not None and row[1] is not None
        ),
    )
    by_channel_rows = tuple(
        _aggregate_row(
            (row[1], row[2], row[3], row[4], row[5]),
            label=_identity_label(
                channel_identities,
                installation_id=row[0],
                kind="channel",
                slack_id=row[1],
            ),
        )
        for row in by_channel_raw_rows
    )
    day_bucket = func.date_trunc("day", LLMUsage.created_at).label("day")
    by_day_rows = session.execute(
        select(
            day_bucket,
            func.count(LLMUsage.id),
            func.coalesce(func.sum(LLMUsage.input_tokens), 0),
            func.coalesce(func.sum(LLMUsage.output_tokens), 0),
            func.coalesce(func.sum(LLMUsage.cost_usd), 0),
        )
        .where(*usage_filter)
        .group_by(day_bucket)
        .order_by(day_bucket.desc())
    ).all()
    return UsageAggregate(
        start=start,
        end=end,
        by_model=tuple(_aggregate_row(row) for row in by_model_rows),
        by_user=by_user_rows,
        by_channel=by_channel_rows,
        by_day=tuple(_daily_row(row) for row in by_day_rows),
    )


def parse_date_bound(
    value: str | None, *, inclusive_end: bool = False
) -> datetime | None:
    """Parse dashboard date filters.

    Date-only upper bounds are treated as inclusive by moving to the next day.
    """

    if value is None or value.strip() == "":
        return None
    stripped = value.strip()
    parsed_date = date.fromisoformat(stripped)
    parsed = datetime.combine(parsed_date, time.min, tzinfo=UTC)
    if inclusive_end:
        return parsed + timedelta(days=1)
    return parsed


IdentityKey = tuple[uuid.UUID, str, str]


def _identity_map(
    session: Session,
    tasks: Sequence[Task],
) -> dict[IdentityKey, SlackIdentity]:
    keys: list[IdentityKey] = []
    for task in tasks:
        keys.append((task.installation_id, "channel", task.slack_channel_id))
        keys.append((task.installation_id, "user", task.slack_user_id))
    return _identity_map_from_keys(session, keys)


def _identity_map_from_keys(
    session: Session,
    keys: Iterable[IdentityKey],
) -> dict[IdentityKey, SlackIdentity]:
    normalized = tuple({key for key in keys if key[2]})
    if not normalized:
        return {}
    installation_ids = tuple({key[0] for key in normalized})
    slack_ids = tuple({key[2] for key in normalized})
    rows = session.scalars(
        select(SlackIdentity).where(
            SlackIdentity.installation_id.in_(installation_ids),
            SlackIdentity.slack_id.in_(slack_ids),
        )
    )
    return {
        (row.installation_id, row.kind, row.slack_id): row
        for row in rows
        if (row.installation_id, row.kind, row.slack_id) in normalized
    }


def _identity_label(
    identities: dict[IdentityKey, SlackIdentity],
    *,
    installation_id: uuid.UUID,
    kind: str,
    slack_id: str,
) -> IdentityLabel:
    identity = identities.get((installation_id, kind, slack_id))
    if identity is None:
        return IdentityLabel(name=slack_id, slack_id=slack_id, found=False)
    return IdentityLabel(
        name=identity.display_name,
        slack_id=identity.slack_id,
        found=True,
    )


def _timeline_event(event: TaskEvent) -> TimelineEvent:
    payload = event.payload if isinstance(event.payload, dict) else {}
    message = _payload_string(payload, "message")
    title = _event_title(event.type.value, message)
    summary = _event_summary(event.type.value, payload, message)
    tone = _event_tone(event.type.value, message)
    badges = _event_badges(event.type.value, payload, message, tone)
    metrics = _event_metrics(event.type.value, payload)
    return TimelineEvent(
        seq=event.seq,
        event_type=event.type.value,
        tone=tone,
        title=title,
        summary=summary,
        created_at=event.created_at,
        badges=badges,
        metrics=metrics,
        payload_json=json.dumps(payload, indent=2, sort_keys=True, default=str),
    )


def _event_title(event_type: str, message: str) -> str:
    if event_type == "log" and message:
        return _message_title(message)
    titles = {
        "task_created": "Task created",
        "status_changed": "Status changed",
        "llm_call": "LLM call completed",
        "tool_call": "Tool call started",
        "tool_result": "Tool result recorded",
        "artifact_created": "Artifact created",
        "message_posted": "Slack message posted",
        "error": "Error recorded",
        "log": "Log event",
    }
    return titles.get(event_type, _humanize_slug(event_type))


def _message_title(message: str) -> str:
    titles = {
        "agent_executor_completed": "Agent executor completed",
        "agent_executor_started": "Agent executor started",
        "context_assembled": "Context assembled",
        "episode_recorded": "Episode recorded",
        "episode_retrieval_completed": "Episode retrieval completed",
        "llm_call_failed": "LLM call failed",
        "llm_call_started": "LLM call started",
        "memory_confirmation_posted": "Memory confirmation posted",
        "memory_write_confirmed": "Memory saved",
        "memory_write_skipped": "Memory skipped",
        "task_executor_started": "Worker started task",
        "tool_call_completed": "Tool call completed",
        "tool_call_failed": "Tool call failed",
        "tool_call_started": "Tool call started",
    }
    return titles.get(message, _humanize_slug(message))


def _event_summary(event_type: str, payload: dict[str, Any], message: str) -> str:
    if event_type == "task_created":
        return _summary_from_fields(
            "Created from Slack",
            payload,
            ("slack_channel_id", "channel"),
            ("slack_user_id", "user"),
            ("slack_thread_ts", "thread_ts"),
            ("slack_event_id", "event_id"),
        )
    if event_type == "status_changed":
        from_status = _payload_string(payload, "from")
        to_status = _payload_string(payload, "to") or _payload_string(payload, "status")
        if from_status and to_status:
            return f"Moved from {from_status} to {to_status}."
        if to_status:
            return f"Task status is now {to_status}."
        return "Task status changed."
    if event_type == "llm_call":
        model = _payload_string(payload, "model") or "model"
        total_tokens = _payload_number(payload, "total_tokens")
        cost = _payload_string(payload, "cost_usd")
        pieces = [f"Completed by {model}"]
        if total_tokens:
            pieces.append(f"{total_tokens} tokens")
        if cost:
            pieces.append(f"${cost} recorded cost")
        return ". ".join(pieces) + "."
    if event_type == "tool_call":
        tool = _payload_string(payload, "tool") or "tool"
        argument_keys = payload.get("argument_keys")
        if isinstance(argument_keys, list) and argument_keys:
            return f"Invoked {tool} with {', '.join(map(str, argument_keys))}."
        return f"Invoked {tool}."
    if event_type == "tool_result":
        tool = _payload_string(payload, "tool") or "tool"
        latency = _payload_string(payload, "latency_ms")
        artifacts = _payload_string(payload, "artifact_count")
        pieces = [f"{tool} returned a result"]
        if latency:
            pieces.append(f"{latency} ms")
        if artifacts:
            pieces.append(f"{artifacts} artifacts")
        return ". ".join(pieces) + "."
    if event_type == "artifact_created":
        filename = _payload_string(payload, "filename") or "file"
        return f"Created artifact {filename}."
    if event_type == "message_posted":
        purpose = _payload_string(payload, "purpose") or "Slack update"
        channel = _payload_string(payload, "channel")
        if channel:
            return f"Posted {purpose} to {channel}."
        return f"Posted {purpose}."
    if event_type == "error":
        error_type = _payload_string(payload, "error_type") or "Error"
        error_summary = _payload_string(payload, "error_summary")
        if error_summary:
            return f"{error_type}: {error_summary}"
        return f"{error_type} recorded."
    if message == "context_assembled":
        fact_count = len(_payload_list(payload, "selected_fact_ids"))
        episode_count = len(_payload_list(payload, "selected_episode_ids"))
        artifact_count = len(_payload_list(payload, "selected_artifact_ids"))
        return (
            "Built the prompt context with "
            f"{fact_count} facts, {episode_count} episodes, "
            f"and {artifact_count} artifacts."
        )
    if message == "episode_retrieval_completed":
        selected_count = _payload_string(payload, "selected_count")
        if selected_count:
            return f"Retrieved {selected_count} relevant prior episodes."
    if message == "llm_call_started":
        model = _payload_string(payload, "model") or "model"
        prompt = _payload_string(payload, "prompt_name")
        if prompt:
            return f"Started {model} with prompt {prompt}."
        return f"Started {model}."
    if message:
        return _humanize_slug(message) + "."
    return "Recorded execution metadata."


def _summary_from_fields(
    prefix: str, payload: dict[str, Any], *keys: tuple[str, str]
) -> str:
    fields = [
        f"{label}={value}"
        for key, label in keys
        if (value := _payload_string(payload, key))
    ]
    if not fields:
        return f"{prefix}."
    return f"{prefix}: {', '.join(fields)}."


def _event_tone(event_type: str, message: str) -> str:
    if event_type == "error" or message.endswith("_failed"):
        return "danger"
    if event_type in {"llm_call", "tool_call", "tool_result"}:
        return "accent"
    if event_type in {"artifact_created", "message_posted"}:
        return "success"
    if event_type == "status_changed":
        return "warning"
    return "neutral"


def _event_badges(
    event_type: str, payload: dict[str, Any], message: str, tone: str
) -> tuple[TimelineBadge, ...]:
    badges = [TimelineBadge(label=event_type, tone=tone)]
    if message and message != event_type:
        badges.append(TimelineBadge(label=message, tone="neutral"))
    for key, badge_tone in (
        ("model_tier", "accent"),
        ("provider", "neutral"),
        ("tool", "accent"),
        ("status", "warning"),
        ("to", "warning"),
        ("purpose", "neutral"),
        ("error_type", "danger"),
    ):
        value = _payload_string(payload, key)
        if value:
            badges.append(TimelineBadge(label=value, tone=badge_tone))
    return tuple(_unique_badges(badges)[:5])


def _event_metrics(
    event_type: str, payload: dict[str, Any]
) -> tuple[TimelineMetric, ...]:
    message = _payload_string(payload, "message")
    keys = (
        "model",
        "prompt_name",
        "route_reason",
        "latency_ms",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cost_usd",
        "tool_call_count",
        "artifact_count",
        "selected_count",
        "worker_id",
        "filename",
        "mime_type",
        "size_bytes",
        "channel",
        "thread_ts",
        "message_ts",
        "phase",
    )
    metrics: list[TimelineMetric] = []
    for key in keys:
        value = (
            _payload_number(payload, key)
            if key in _NUMERIC_PAYLOAD_KEYS
            else _payload_string(payload, key)
        )
        if value:
            metrics.append(TimelineMetric(label=_humanize_slug(key), value=value))
    if event_type == "context_assembled" or message == "context_assembled":
        metrics.extend(
            [
                TimelineMetric(
                    label="facts",
                    value=str(len(_payload_list(payload, "selected_fact_ids"))),
                ),
                TimelineMetric(
                    label="episodes",
                    value=str(len(_payload_list(payload, "selected_episode_ids"))),
                ),
            ]
        )
    return tuple(metrics[:8])


def _payload_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if value is None or value == "":
        return ""
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, default=str)
    return str(value)


_NUMERIC_PAYLOAD_KEYS = frozenset(
    {
        "latency_ms",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "tool_call_count",
        "artifact_count",
        "selected_count",
        "size_bytes",
        "turn",
    }
)


def _payload_number(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if value is None or value == "":
        return ""
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return _payload_string(payload, key)


def _payload_list(payload: dict[str, Any], key: str) -> list[Any]:
    value = payload.get(key)
    return value if isinstance(value, list) else []


def _unique_badges(badges: list[TimelineBadge]) -> list[TimelineBadge]:
    seen: set[tuple[str, str]] = set()
    unique: list[TimelineBadge] = []
    for badge in badges:
        marker = (badge.label, badge.tone)
        if marker in seen:
            continue
        seen.add(marker)
        unique.append(badge)
    return unique


def _humanize_slug(value: str) -> str:
    return value.replace("_", " ").replace("-", " ").strip().capitalize()


def _usage_by_task(
    session: Session, task_ids: list[uuid.UUID]
) -> dict[uuid.UUID, list[LLMUsage]]:
    usage_by_task: dict[uuid.UUID, list[LLMUsage]] = defaultdict(list)
    if not task_ids:
        return usage_by_task
    usage_rows = session.scalars(
        select(LLMUsage)
        .where(LLMUsage.task_id.in_(task_ids))
        .order_by(LLMUsage.created_at.asc(), LLMUsage.id.asc())
    )
    for usage in usage_rows:
        usage_by_task[usage.task_id].append(usage)
    return usage_by_task


def _usage_filter(
    *, start: datetime | None, end: datetime | None
) -> list[ColumnElement[bool]]:
    filters: list[ColumnElement[bool]] = []
    if start is not None:
        filters.append(LLMUsage.created_at >= start)
    if end is not None:
        filters.append(LLMUsage.created_at < end)
    return filters


def _aggregate_query(key: Any, filters: list[ColumnElement[bool]]) -> Select[Any]:
    return (
        select(
            key,
            func.count(LLMUsage.id),
            func.coalesce(func.sum(LLMUsage.input_tokens), 0),
            func.coalesce(func.sum(LLMUsage.output_tokens), 0),
            func.coalesce(func.sum(LLMUsage.cost_usd), 0),
        )
        .where(*filters)
        .group_by(key)
    )


def _aggregate_row(
    row: Row[Any] | tuple[Any, ...],
    label: IdentityLabel | None = None,
) -> AggregateRow:
    key, calls, input_tokens, output_tokens, cost_usd = row
    return AggregateRow(
        key=str(key),
        calls=int(calls),
        input_tokens=int(input_tokens),
        output_tokens=int(output_tokens),
        cost_usd=Decimal(cost_usd),
        label=label,
    )


def _daily_row(row: Row[Any]) -> DailyUsageRow:
    day_value, calls, input_tokens, output_tokens, cost_usd = row
    if isinstance(day_value, datetime):
        day = day_value.date()
    elif isinstance(day_value, date):
        day = day_value
    else:
        day = date.fromisoformat(str(day_value)[:10])
    return DailyUsageRow(
        day=day,
        calls=int(calls),
        input_tokens=int(input_tokens),
        output_tokens=int(output_tokens),
        cost_usd=Decimal(cost_usd),
    )
