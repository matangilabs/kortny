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
from urllib.parse import urlencode, urlsplit, urlunsplit

from sqlalchemy import Select, Text, case, cast, func, or_, select
from sqlalchemy.engine import Row
from sqlalchemy.orm import Session
from sqlalchemy.sql.elements import ColumnElement

from kortny.composio import (
    ComposioAuthConfig,
    ComposioCatalogError,
    ComposioClient,
    ComposioConnectionError,
    ComposioToolkit,
)
from kortny.config import Settings
from kortny.dashboard.settings import DashboardSettings
from kortny.db.models import (
    Artifact,
    ComposioConnection,
    Episode,
    LLMUsage,
    SlackIdentity,
    Task,
    TaskEvent,
    TaskStatus,
    WorkspaceState,
)
from kortny.tools.pdf_generator import PdfGeneratorTool
from kortny.tools.slack_channel_history import SlackChannelHistoryTool
from kortny.tools.slack_file_read import SlackFileReadTool
from kortny.tools.web_search import WebSearchTool
from kortny.tools.workspace_memory import (
    ForgetFactTool,
    InspectMemoryTool,
    RecallFactTool,
    RememberFactTool,
)

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
class DailyTaskRow:
    day: date
    task_count: int
    failed_task_count: int


@dataclass(frozen=True)
class ChartBar:
    label: str
    secondary: str | None
    value_label: str
    percent: int


@dataclass(frozen=True)
class ChartPoint:
    label: str
    value_label: str
    percent: int
    tone: str = "accent"
    detail: str | None = None


@dataclass(frozen=True)
class UsageCharts:
    daily_cost: tuple[ChartPoint, ...]
    daily_task_volume: tuple[ChartPoint, ...]
    cost_by_model: tuple[ChartBar, ...]
    cost_by_user: tuple[ChartBar, ...]


@dataclass(frozen=True)
class UsageAggregate:
    start: datetime | None
    end: datetime | None
    by_model: tuple[AggregateRow, ...]
    by_user: tuple[AggregateRow, ...]
    by_channel: tuple[AggregateRow, ...]
    by_day: tuple[DailyUsageRow, ...]
    by_task_day: tuple[DailyTaskRow, ...]

    @property
    def total_calls(self) -> int:
        return sum(row.calls for row in self.by_day)

    @property
    def total_input_tokens(self) -> int:
        return sum(row.input_tokens for row in self.by_day)

    @property
    def total_output_tokens(self) -> int:
        return sum(row.output_tokens for row in self.by_day)

    @property
    def total_cost_usd(self) -> Decimal:
        return sum((row.cost_usd for row in self.by_day), Decimal("0"))

    @property
    def total_tasks(self) -> int:
        return sum(row.task_count for row in self.by_task_day)

    @property
    def failed_tasks(self) -> int:
        return sum(row.failed_task_count for row in self.by_task_day)

    @property
    def task_failure_rate_label(self) -> str:
        if self.total_tasks == 0:
            return "0.0%"
        return f"{(self.failed_tasks / self.total_tasks) * 100:.1f}%"

    @property
    def charts(self) -> UsageCharts:
        return UsageCharts(
            daily_cost=_daily_cost_points(self.by_day),
            daily_task_volume=_daily_task_points(self.by_task_day),
            cost_by_model=_aggregate_bars(self.by_model),
            cost_by_user=_aggregate_bars(self.by_user),
        )


@dataclass(frozen=True)
class UserListItem:
    user: IdentityLabel
    task_count: int
    failed_task_count: int
    artifact_count: int
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: Decimal
    last_activity_at: datetime | None


@dataclass(frozen=True)
class UserDirectory:
    start: datetime | None
    end: datetime | None
    users: tuple[UserListItem, ...]


@dataclass(frozen=True)
class UserTaskRow:
    task: Task
    channel: IdentityLabel
    usage_count: int
    artifact_count: int


@dataclass(frozen=True)
class UserArtifactRow:
    artifact: Artifact
    task: Task


@dataclass(frozen=True)
class UserDetail:
    user: IdentityLabel
    start: datetime | None
    end: datetime | None
    task_count: int
    failed_task_count: int
    artifact_count: int
    usage_call_count: int
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: Decimal
    last_activity_at: datetime | None
    tasks: tuple[UserTaskRow, ...]
    usage: tuple[LLMUsage, ...]
    artifacts: tuple[UserArtifactRow, ...]


@dataclass(frozen=True)
class SystemMetric:
    label: str
    value: str
    detail: str | None = None
    tone: str = "neutral"


@dataclass(frozen=True)
class SystemCheck:
    group: str
    name: str
    status: str
    tone: str
    detail: str
    action: str | None = None


@dataclass(frozen=True)
class SystemConfigRow:
    name: str
    value: str
    detail: str | None = None
    tone: str = "neutral"


@dataclass(frozen=True)
class SystemConfigSection:
    title: str
    rows: tuple[SystemConfigRow, ...]


@dataclass(frozen=True)
class SystemHealth:
    overall_label: str
    overall_tone: str
    metrics: tuple[SystemMetric, ...]
    checks: tuple[SystemCheck, ...]
    config_sections: tuple[SystemConfigSection, ...]


@dataclass(frozen=True)
class OverviewAttentionItem:
    title: str
    detail: str
    tone: str
    badge: str
    href: str


@dataclass(frozen=True)
class DashboardOverview:
    metrics: tuple[SystemMetric, ...]
    attention_items: tuple[OverviewAttentionItem, ...]
    charts: UsageCharts
    top_models: tuple[AggregateRow, ...]
    top_users: tuple[AggregateRow, ...]
    top_channels: tuple[AggregateRow, ...]
    recent_tasks: tuple[TaskListItem, ...]
    system_health: SystemHealth
    window_label: str


@dataclass(frozen=True)
class MemoryFactRow:
    fact: WorkspaceState
    scope: IdentityLabel
    value_summary: str
    confirmed_by: IdentityLabel | None
    proposed_by: IdentityLabel | None
    rejected_by: IdentityLabel | None
    forgotten_by: IdentityLabel | None
    source_task: Task | None
    tone: str


@dataclass(frozen=True)
class MemoryEpisodeRow:
    episode: Episode
    channel: IdentityLabel
    user: IdentityLabel
    task: Task | None
    tools_label: str
    artifacts_label: str
    source_refs_label: str
    tone: str


@dataclass(frozen=True)
class MemoryPageInfo:
    page: int
    page_size: int
    total_count: int
    noun: str

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

    @property
    def first_item(self) -> int:
        if self.total_count == 0:
            return 0
        return ((self.page - 1) * self.page_size) + 1

    @property
    def last_item(self) -> int:
        return min(self.page * self.page_size, self.total_count)


@dataclass(frozen=True)
class MemoryDashboard:
    active_fact_count: int
    proposed_fact_count: int
    episode_count: int
    failed_episode_count: int
    active_view: str
    query: str
    scope_filter: str
    status_filter: str
    outcome_filter: str
    sort: str
    page: MemoryPageInfo
    previous_page_url: str | None
    next_page_url: str | None
    reset_url: str
    facts: tuple[MemoryFactRow, ...]
    episodes: tuple[MemoryEpisodeRow, ...]


@dataclass(frozen=True)
class IntegrationCard:
    name: str
    category: str
    status: str
    tone: str
    description: str
    details: tuple[str, ...]
    env_vars: tuple[str, ...]
    action: str | None = None


@dataclass(frozen=True)
class ComposioToolkitRow:
    slug: str
    name: str
    description: str
    categories: tuple[str, ...]
    auth_schemes: tuple[str, ...]
    managed_auth_schemes: tuple[str, ...]
    tools_count: int
    triggers_count: int
    no_auth: bool
    connection_status: str
    connection_tone: str
    connected: bool


@dataclass(frozen=True)
class ComposioConnectionRow:
    id: uuid.UUID
    toolkit_slug: str
    status: str
    tone: str
    display_name: str
    scope_label: str
    visibility_scope_type: str
    visibility_scope_id: str | None
    owner: IdentityLabel
    connected_account_id: str | None
    auth_config_id: str | None
    updated_at: datetime


@dataclass(frozen=True)
class ComposioAuthConfigRow:
    id: str
    name: str
    toolkit_slug: str
    auth_scheme: str | None
    is_composio_managed: bool
    enabled: bool

    @property
    def managed_label(self) -> str:
        return "Composio managed" if self.is_composio_managed else "Custom"

    @property
    def status_label(self) -> str:
        return "Enabled" if self.enabled else "Disabled"

    @property
    def tone(self) -> str:
        return "success" if self.enabled else "neutral"


@dataclass(frozen=True)
class ComposioCatalogView:
    enabled: bool
    configured: bool
    status: str
    tone: str
    query: str
    total_items: int | None
    visible_count: int
    connection_count: int
    active_connection_count: int
    error: str | None
    toolkits: tuple[ComposioToolkitRow, ...]
    connections: tuple[ComposioConnectionRow, ...]


@dataclass(frozen=True)
class ComposioScopeOption:
    name: str
    key: str
    description: str
    default: bool = False
    risk: str | None = None


@dataclass(frozen=True)
class ComposioToolkitDetail:
    slug: str
    configured: bool
    status: str
    tone: str
    toolkit: ComposioToolkitRow | None
    raw_toolkit: ComposioToolkit | None
    auth_configs: tuple[ComposioAuthConfigRow, ...]
    connections: tuple[ComposioConnectionRow, ...]
    scope_options: tuple[ComposioScopeOption, ...]
    user_options: tuple[IdentityLabel, ...]
    channel_options: tuple[IdentityLabel, ...]
    error: str | None

    @property
    def active_connection(self) -> ComposioConnectionRow | None:
        return next(
            (connection for connection in self.connections if connection.status == "active"),
            None,
        )


@dataclass(frozen=True)
class ToolCapability:
    name: str
    group: str
    status: str
    tone: str
    description: str
    required_args: tuple[str, ...]
    optional_args: tuple[str, ...]
    notes: tuple[str, ...]


@dataclass(frozen=True)
class ToolCapabilityGroup:
    name: str
    description: str
    tools: tuple[ToolCapability, ...]


@dataclass(frozen=True)
class IntegrationDashboard:
    metrics: tuple[SystemMetric, ...]
    integrations: tuple[IntegrationCard, ...]
    composio_catalog: ComposioCatalogView
    tool_groups: tuple[ToolCapabilityGroup, ...]
    runtime_error: str | None


def list_tasks(
    session: Session,
    *,
    page: int = 1,
    page_size: int = DEFAULT_PAGE_SIZE,
    installation_id: uuid.UUID | None = None,
    slack_user_id: str | None = None,
) -> TaskListPage:
    """Return a paginated dashboard task list."""

    normalized_page = max(page, 1)
    normalized_size = min(max(page_size, 1), MAX_PAGE_SIZE)
    offset = (normalized_page - 1) * normalized_size
    task_filter = _task_scope_filter(
        installation_id=installation_id,
        slack_user_id=slack_user_id,
    )

    total_count = (
        session.scalar(select(func.count()).select_from(Task).where(*task_filter)) or 0
    )
    tasks = tuple(
        session.scalars(
            select(Task)
            .where(*task_filter)
            .order_by(Task.created_at.desc(), Task.id.desc())
            .offset(offset)
            .limit(normalized_size)
        )
    )
    return TaskListPage(
        items=_task_items(session, tasks),
        page=normalized_page,
        page_size=normalized_size,
        total_count=total_count,
    )


def get_task_detail(
    session: Session,
    task_id: uuid.UUID,
    *,
    installation_id: uuid.UUID | None = None,
    slack_user_id: str | None = None,
) -> TaskDetail | None:
    """Return one task and its child rows."""

    task = session.get(Task, task_id)
    if task is None:
        return None
    if installation_id is not None and task.installation_id != installation_id:
        return None
    if slack_user_id is not None and task.slack_user_id != slack_user_id:
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
    installation_id: uuid.UUID | None = None,
    slack_user_id: str | None = None,
) -> UsageAggregate:
    """Return dashboard usage rollups."""

    usage_filter = _usage_filter(start=start, end=end)
    scoped_task_filter = _task_scope_filter(
        installation_id=installation_id,
        slack_user_id=slack_user_id,
    )
    by_model_rows = session.execute(
        select(
            LLMUsage.model,
            func.count(LLMUsage.id),
            func.coalesce(func.sum(LLMUsage.input_tokens), 0),
            func.coalesce(func.sum(LLMUsage.output_tokens), 0),
            func.coalesce(func.sum(LLMUsage.cost_usd), 0),
        )
        .join(Task, Task.id == LLMUsage.task_id)
        .where(*usage_filter, *scoped_task_filter)
        .group_by(LLMUsage.model)
        .order_by(func.sum(LLMUsage.cost_usd).desc())
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
        .where(*usage_filter, *scoped_task_filter)
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
        .where(*usage_filter, *scoped_task_filter, ~Task.slack_channel_id.startswith("D"))
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
        .join(Task, Task.id == LLMUsage.task_id)
        .where(*usage_filter, *scoped_task_filter)
        .group_by(day_bucket)
        .order_by(day_bucket.desc())
    ).all()
    task_day_bucket = func.date_trunc("day", Task.created_at).label("day")
    by_task_day_rows = session.execute(
        select(
            task_day_bucket,
            func.count(Task.id),
            func.coalesce(func.sum(_failed_task_case()), 0),
        )
        .where(
            *_task_filter(start=start, end=end),
            *scoped_task_filter,
        )
        .group_by(task_day_bucket)
        .order_by(task_day_bucket.desc())
    ).all()
    return UsageAggregate(
        start=start,
        end=end,
        by_model=tuple(_aggregate_row(row) for row in by_model_rows),
        by_user=by_user_rows,
        by_channel=by_channel_rows,
        by_day=tuple(_daily_row(row) for row in by_day_rows),
        by_task_day=tuple(_daily_task_row(row) for row in by_task_day_rows),
    )


def list_users(
    session: Session,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
) -> UserDirectory:
    """Return user-level task/cost rollups."""

    task_filter = _task_filter(start=start, end=end)
    rows = session.execute(
        select(
            Task.installation_id,
            Task.slack_user_id,
            func.count(Task.id),
            func.coalesce(func.sum(_failed_task_case()), 0),
            func.coalesce(func.sum(Task.total_input_tokens), 0),
            func.coalesce(func.sum(Task.total_output_tokens), 0),
            func.coalesce(func.sum(Task.total_cost_usd), 0),
            func.max(Task.created_at),
        )
        .where(*task_filter)
        .group_by(Task.installation_id, Task.slack_user_id)
        .order_by(
            func.sum(Task.total_cost_usd).desc(), func.max(Task.created_at).desc()
        )
    ).all()
    artifact_counts = _artifact_counts_by_user(session, task_filter)
    identities = _identity_map_from_keys(
        session,
        (
            (row[0], "user", row[1])
            for row in rows
            if row[0] is not None and row[1] is not None
        ),
    )
    users = tuple(
        UserListItem(
            user=_identity_label(
                identities,
                installation_id=row[0],
                kind="user",
                slack_id=row[1],
            ),
            task_count=int(row[2]),
            failed_task_count=int(row[3]),
            total_input_tokens=int(row[4]),
            total_output_tokens=int(row[5]),
            total_cost_usd=Decimal(row[6]),
            last_activity_at=row[7],
            artifact_count=artifact_counts.get((row[0], row[1]), 0),
        )
        for row in rows
    )
    return UserDirectory(start=start, end=end, users=users)


def get_user_detail(
    session: Session,
    slack_user_id: str,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
    installation_id: uuid.UUID | None = None,
) -> UserDetail | None:
    """Return one user's tasks, usage, and artifacts."""

    task_filter = [
        Task.slack_user_id == slack_user_id,
        *_task_scope_filter(installation_id=installation_id),
        *_task_filter(start=start, end=end),
    ]
    stats = session.execute(
        select(
            func.count(Task.id),
            func.coalesce(func.sum(_failed_task_case()), 0),
            func.coalesce(func.sum(Task.total_input_tokens), 0),
            func.coalesce(func.sum(Task.total_output_tokens), 0),
            func.coalesce(func.sum(Task.total_cost_usd), 0),
            func.max(Task.created_at),
        ).where(*task_filter)
    ).one()
    task_count = int(stats[0])
    if task_count == 0:
        return None

    tasks = tuple(
        session.scalars(
            select(Task)
            .where(*task_filter)
            .order_by(Task.created_at.desc(), Task.id.desc())
            .limit(25)
        )
    )
    task_ids = [task.id for task in tasks]
    usage_by_task = _usage_by_task(session, task_ids)
    artifact_counts = _artifact_counts_by_task(session, task_ids)
    identities = _identity_map(session, tasks)
    user_label = _user_label_for_tasks(
        session,
        slack_user_id=slack_user_id,
        tasks=tasks,
    )

    usage_filter = [
        Task.slack_user_id == slack_user_id,
        *_task_scope_filter(installation_id=installation_id),
        *_usage_filter(start=start, end=end),
    ]
    usage = tuple(
        session.scalars(
            select(LLMUsage)
            .join(Task, Task.id == LLMUsage.task_id)
            .where(*usage_filter)
            .order_by(LLMUsage.created_at.desc(), LLMUsage.id.desc())
            .limit(25)
        )
    )
    artifact_rows = tuple(
        session.execute(
            select(Artifact, Task)
            .join(Task, Task.id == Artifact.task_id)
            .where(*task_filter)
            .order_by(Artifact.created_at.desc(), Artifact.id.desc())
            .limit(25)
        )
    )
    artifacts = tuple(
        UserArtifactRow(artifact=row[0], task=row[1]) for row in artifact_rows
    )

    return UserDetail(
        user=user_label,
        start=start,
        end=end,
        task_count=task_count,
        failed_task_count=int(stats[1]),
        total_input_tokens=int(stats[2]),
        total_output_tokens=int(stats[3]),
        total_cost_usd=Decimal(stats[4]),
        last_activity_at=stats[5],
        usage_call_count=len(usage),
        artifact_count=_artifact_count_for_user(session, task_filter),
        tasks=tuple(
            UserTaskRow(
                task=task,
                channel=_identity_label(
                    identities,
                    installation_id=task.installation_id,
                    kind="channel",
                    slack_id=task.slack_channel_id,
                ),
                usage_count=len(usage_by_task[task.id]),
                artifact_count=artifact_counts.get(task.id, 0),
            )
            for task in tasks
        ),
        usage=usage,
        artifacts=artifacts,
    )


def get_dashboard_overview(
    session: Session,
    *,
    system_health: SystemHealth,
    now: datetime | None = None,
) -> DashboardOverview:
    """Return the operator dashboard home read model."""

    current = now or datetime.now(UTC)
    current = current.astimezone(UTC)
    today_start = datetime.combine(current.date(), time.min, tzinfo=UTC)
    week_start = current - timedelta(days=7)
    query_end = current + timedelta(seconds=1)

    usage = get_usage_aggregate(session, start=week_start, end=query_end)
    week_stats = session.execute(
        select(
            func.count(Task.id),
            func.coalesce(func.sum(_failed_task_case()), 0),
            func.coalesce(func.sum(Task.total_cost_usd), 0),
        ).where(*_task_filter(start=week_start, end=query_end))
    ).one()
    active_tasks = (
        session.scalar(
            select(func.count())
            .select_from(Task)
            .where(Task.status.in_((TaskStatus.pending, TaskStatus.running)))
        )
        or 0
    )
    today_cost = session.scalar(
        select(func.coalesce(func.sum(LLMUsage.cost_usd), 0)).where(
            LLMUsage.created_at >= today_start,
            LLMUsage.created_at < query_end,
        )
    ) or Decimal("0")
    last_task_at = session.scalar(select(func.max(Task.created_at)))
    week_task_count = int(week_stats[0])
    week_failed_count = int(week_stats[1])

    metrics = (
        SystemMetric(
            label="Readiness",
            value=system_health.overall_label,
            detail="Worst current system check",
            tone=system_health.overall_tone,
        ),
        SystemMetric(
            label="Active Tasks",
            value=f"{active_tasks:,}",
            detail="Pending or running now",
            tone="warning" if active_tasks else "neutral",
        ),
        SystemMetric(
            label="Today Cost",
            value=_format_money(Decimal(today_cost)),
            detail="LLM usage recorded today",
        ),
        SystemMetric(
            label="7 Day Failures",
            value=_failure_rate_label(week_task_count, week_failed_count),
            detail=f"{week_failed_count:,} of {week_task_count:,} tasks",
            tone="danger" if week_failed_count else "neutral",
        ),
        SystemMetric(
            label="Last Task",
            value=_datetime_label(last_task_at),
            detail="Most recent task creation",
        ),
    )

    attention_items = _overview_attention_items(session, system_health=system_health)
    recent_tasks = list_tasks(session, page=1, page_size=10)

    return DashboardOverview(
        metrics=metrics,
        attention_items=attention_items,
        charts=usage.charts,
        top_models=usage.by_model[:5],
        top_users=usage.by_user[:5],
        top_channels=usage.by_channel[:5],
        recent_tasks=recent_tasks.items,
        system_health=system_health,
        window_label="Last 7 days",
    )


def get_memory_dashboard(
    session: Session,
    *,
    view: str = "facts",
    query: str | None = None,
    scope_filter: str = "all",
    status_filter: str = "active",
    outcome_filter: str = "all",
    sort: str | None = None,
    page: int = 1,
    page_size: int = DEFAULT_PAGE_SIZE,
    installation_id: uuid.UUID | None = None,
    slack_user_id: str | None = None,
    base_path: str = "/memory",
) -> MemoryDashboard:
    """Return read-only memory state for the management console."""

    active_view = "episodes" if view == "episodes" else "facts"
    normalized_query = " ".join((query or "").split())
    normalized_page = max(page, 1)
    normalized_size = min(max(page_size, 1), MAX_PAGE_SIZE)
    normalized_scope = scope_filter if scope_filter in _MEMORY_SCOPES else "all"
    normalized_status = status_filter if status_filter in _MEMORY_STATUSES else "active"
    normalized_outcome = outcome_filter if outcome_filter in _MEMORY_OUTCOMES else "all"
    normalized_sort = _normalize_memory_sort(active_view, sort)
    memory_fact_scope = _workspace_state_scope_filter(
        installation_id=installation_id,
        slack_user_id=slack_user_id,
    )
    memory_episode_scope = _episode_scope_filter(
        installation_id=installation_id,
        slack_user_id=slack_user_id,
    )

    active_fact_count = (
        session.scalar(
            select(func.count())
            .select_from(WorkspaceState)
            .where(WorkspaceState.status == "active", *memory_fact_scope)
        )
        or 0
    )
    proposed_fact_count = (
        session.scalar(
            select(func.count())
            .select_from(WorkspaceState)
            .where(WorkspaceState.status == "proposed", *memory_fact_scope)
        )
        or 0
    )
    episode_count = (
        session.scalar(select(func.count()).select_from(Episode).where(*memory_episode_scope))
        or 0
    )
    failed_episode_count = (
        session.scalar(
            select(func.count())
            .select_from(Episode)
            .where(Episode.outcome == "failed", *memory_episode_scope)
        )
        or 0
    )

    facts: tuple[MemoryFactRow, ...] = ()
    episodes: tuple[MemoryEpisodeRow, ...] = ()
    if active_view == "facts":
        total_count, resolved_page, facts = _memory_fact_rows(
            session,
            query=normalized_query,
            scope_filter=normalized_scope,
            status_filter=normalized_status,
            sort=normalized_sort,
            page=normalized_page,
            page_size=normalized_size,
            installation_id=installation_id,
            slack_user_id=slack_user_id,
        )
        noun = "facts"
    else:
        total_count, resolved_page, episodes = _memory_episode_rows(
            session,
            query=normalized_query,
            outcome_filter=normalized_outcome,
            sort=normalized_sort,
            page=normalized_page,
            page_size=normalized_size,
            installation_id=installation_id,
            slack_user_id=slack_user_id,
        )
        noun = "episodes"

    page_info = MemoryPageInfo(
        page=resolved_page,
        page_size=normalized_size,
        total_count=total_count,
        noun=noun,
    )

    return MemoryDashboard(
        active_fact_count=int(active_fact_count),
        proposed_fact_count=int(proposed_fact_count),
        episode_count=int(episode_count),
        failed_episode_count=int(failed_episode_count),
        active_view=active_view,
        query=normalized_query,
        scope_filter=normalized_scope,
        status_filter=normalized_status,
        outcome_filter=normalized_outcome,
        sort=normalized_sort,
        page=page_info,
        previous_page_url=(
            _memory_page_url(
                view=active_view,
                query=normalized_query,
                scope_filter=normalized_scope,
                status_filter=normalized_status,
                outcome_filter=normalized_outcome,
                sort=normalized_sort,
                page=page_info.previous_page,
                page_size=normalized_size,
                base_path=base_path,
            )
            if page_info.previous_page is not None
            else None
        ),
        next_page_url=(
            _memory_page_url(
                view=active_view,
                query=normalized_query,
                scope_filter=normalized_scope,
                status_filter=normalized_status,
                outcome_filter=normalized_outcome,
                sort=normalized_sort,
                page=page_info.next_page,
                page_size=normalized_size,
                base_path=base_path,
            )
            if page_info.next_page is not None
            else None
        ),
        reset_url=f"{base_path}?view={active_view}",
        facts=facts,
        episodes=episodes,
    )


def get_system_health(
    session: Session,
    *,
    dashboard_settings: DashboardSettings,
    runtime_settings: Settings | None = None,
    runtime_error: str | None = None,
) -> SystemHealth:
    """Return a read-only operator snapshot for setup and system health."""

    total_tasks = session.scalar(select(func.count()).select_from(Task)) or 0
    active_tasks = (
        session.scalar(
            select(func.count())
            .select_from(Task)
            .where(Task.status.in_((TaskStatus.pending, TaskStatus.running)))
        )
        or 0
    )
    failed_tasks = (
        session.scalar(
            select(func.count())
            .select_from(Task)
            .where(Task.status.in_((TaskStatus.failed, TaskStatus.crashed)))
        )
        or 0
    )
    llm_calls = session.scalar(select(func.count()).select_from(LLMUsage)) or 0
    last_task_at = session.scalar(select(func.max(Task.created_at)))

    checks: list[SystemCheck] = [
        SystemCheck(
            group="Core",
            name="Database",
            status="Connected",
            tone="success",
            detail=f"{total_tasks:,} tasks and {llm_calls:,} LLM calls recorded.",
        ),
    ]

    if runtime_settings is None:
        checks.append(
            SystemCheck(
                group="Core",
                name="Runtime settings",
                status="Needs setup",
                tone="danger",
                detail=runtime_error or "Runtime configuration could not be loaded.",
                action="Set the required Slack, LLM, and Postgres environment variables.",
            )
        )
    else:
        checks.extend(_runtime_checks(runtime_settings))

    checks.append(_dashboard_auth_check(dashboard_settings))

    metrics = (
        SystemMetric(
            label="Overall",
            value=_overall_label(checks),
            detail="Worst current setup check.",
            tone=_overall_tone(checks),
        ),
        SystemMetric(
            label="Tasks",
            value=f"{total_tasks:,}",
            detail=f"{active_tasks:,} active",
        ),
        SystemMetric(
            label="Failures",
            value=f"{failed_tasks:,}",
            detail="Failed or crashed tasks",
            tone="danger" if failed_tasks else "neutral",
        ),
        SystemMetric(
            label="Last Task",
            value=_datetime_label(last_task_at),
            detail="Most recent task creation",
        ),
    )

    return SystemHealth(
        overall_label=_overall_label(checks),
        overall_tone=_overall_tone(checks),
        metrics=metrics,
        checks=tuple(checks),
        config_sections=_config_sections(
            dashboard_settings=dashboard_settings,
            runtime_settings=runtime_settings,
            runtime_error=runtime_error,
        ),
    )


def get_integration_dashboard(
    *,
    session: Session | None = None,
    runtime_settings: Settings | None = None,
    runtime_error: str | None = None,
    composio_query: str | None = None,
    composio_client: ComposioClient | None = None,
    installation_id: uuid.UUID | None = None,
    owner_slack_user_id: str | None = None,
) -> IntegrationDashboard:
    """Return configured integration and tool-registry state."""

    integrations = _integration_cards(runtime_settings, runtime_error)
    composio_catalog = _composio_catalog_view(
        session=session,
        settings=runtime_settings,
        query=composio_query,
        client=composio_client,
        installation_id=installation_id,
        owner_slack_user_id=owner_slack_user_id,
    )
    tool_groups = _tool_capability_groups(runtime_settings)
    configured_count = sum(1 for card in integrations if card.tone == "success")
    setup_gap_count = sum(
        1 for card in integrations if card.tone in {"warning", "danger"}
    )
    tool_count = sum(len(group.tools) for group in tool_groups)

    metrics = (
        SystemMetric(
            label="Configured",
            value=f"{configured_count:,}",
            detail="Providers ready for this deployment",
            tone="success" if configured_count else "warning",
        ),
        SystemMetric(
            label="Setup Gaps",
            value=f"{setup_gap_count:,}",
            detail="Missing or planned configuration",
            tone="warning" if setup_gap_count else "success",
        ),
        SystemMetric(
            label="Native Tools",
            value=f"{tool_count:,}",
            detail="Tool contracts exposed to the agent loop",
        ),
        SystemMetric(
            label="External Adapters",
            value=(
                f"{composio_catalog.total_items:,}"
                if composio_catalog.total_items is not None
                else "1 planned"
            ),
            detail="Composio supported toolkits"
            if composio_catalog.total_items is not None
            else "Composio is tracked separately in HIG-35",
            tone=composio_catalog.tone,
        ),
    )
    return IntegrationDashboard(
        metrics=metrics,
        integrations=integrations,
        composio_catalog=composio_catalog,
        tool_groups=tool_groups,
        runtime_error=runtime_error,
    )


def get_composio_catalog_dashboard(
    session: Session,
    *,
    runtime_settings: Settings | None = None,
    query: str | None = None,
    composio_client: ComposioClient | None = None,
    installation_id: uuid.UUID | None = None,
    owner_slack_user_id: str | None = None,
) -> ComposioCatalogView:
    """Return the Composio catalog view for the dedicated management page."""

    return _composio_catalog_view(
        session=session,
        settings=runtime_settings,
        query=query,
        client=composio_client,
        installation_id=installation_id,
        owner_slack_user_id=owner_slack_user_id,
    )


def get_composio_toolkit_detail(
    session: Session,
    *,
    slug: str,
    runtime_settings: Settings | None = None,
    composio_client: ComposioClient | None = None,
    installation_id: uuid.UUID | None = None,
    owner_slack_user_id: str | None = None,
) -> ComposioToolkitDetail:
    """Return one Composio toolkit and local scoped connection metadata."""

    normalized_slug = slug.strip().lower()
    connections = tuple(
        connection
        for connection in _composio_connection_rows(
            session,
            installation_id=installation_id,
            owner_slack_user_id=owner_slack_user_id,
        )
        if connection.toolkit_slug == normalized_slug
    )
    user_options = _slack_identity_options(session, kind="user")
    channel_options = _slack_identity_options(session, kind="channel")
    if runtime_settings is None or not runtime_settings.composio_api_key:
        return ComposioToolkitDetail(
            slug=normalized_slug,
            configured=False,
            status="Not configured",
            tone="neutral",
            toolkit=None,
            raw_toolkit=None,
            auth_configs=(),
            connections=connections,
            scope_options=_composio_scope_options(),
            user_options=user_options,
            channel_options=channel_options,
            error=None,
        )
    if not runtime_settings.composio_catalog_enabled:
        return ComposioToolkitDetail(
            slug=normalized_slug,
            configured=True,
            status="Catalog disabled",
            tone="warning",
            toolkit=None,
            raw_toolkit=None,
            auth_configs=(),
            connections=connections,
            scope_options=_composio_scope_options(),
            user_options=user_options,
            channel_options=channel_options,
            error="COMPOSIO_CATALOG_ENABLED is false.",
        )

    client = composio_client or ComposioClient(
        api_key=runtime_settings.composio_api_key,
        timeout_seconds=runtime_settings.composio_request_timeout_seconds,
    )
    try:
        toolkit = client.get_toolkit(normalized_slug)
    except ComposioCatalogError as exc:
        return ComposioToolkitDetail(
            slug=normalized_slug,
            configured=True,
            status="Unavailable",
            tone="danger",
            toolkit=None,
            raw_toolkit=None,
            auth_configs=(),
            connections=connections,
            scope_options=_composio_scope_options(),
            user_options=user_options,
            channel_options=channel_options,
            error=_short_error(str(exc)),
        )
    auth_configs: tuple[ComposioAuthConfigRow, ...] = ()
    auth_config_error = None
    try:
        auth_configs = tuple(
            _composio_auth_config_row(auth_config)
            for auth_config in client.list_auth_configs(toolkit_slug=normalized_slug)
        )
    except (ComposioCatalogError, ComposioConnectionError) as exc:
        auth_config_error = f"Auth configs unavailable: {_short_error(str(exc))}"

    status_map = _composio_status_by_toolkit(connections)
    return ComposioToolkitDetail(
        slug=normalized_slug,
        configured=True,
        status="Connected" if status_map.get(toolkit.slug) == "active" else "Available",
        tone="success" if status_map.get(toolkit.slug) == "active" else "neutral",
        toolkit=_composio_toolkit_row(toolkit, status_map),
        raw_toolkit=toolkit,
        auth_configs=auth_configs,
        connections=connections,
        scope_options=_composio_scope_options(),
        user_options=user_options,
        channel_options=channel_options,
        error=auth_config_error,
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


def _runtime_checks(settings: Settings) -> tuple[SystemCheck, ...]:
    model_tier_count = len(
        tuple(
            model
            for model in (
                settings.llm_cheap_model,
                settings.llm_standard_model,
                settings.llm_analysis_model,
                settings.llm_document_model,
                settings.llm_high_reasoning_model,
            )
            if model
        )
    )
    return (
        SystemCheck(
            group="Core",
            name="Slack app",
            status="Configured",
            tone="success",
            detail=f"Socket mode credentials are present for app name {settings.slack_app_name!r}.",
        ),
        SystemCheck(
            group="Core",
            name="LLM provider",
            status="Configured",
            tone="success",
            detail=f"{settings.llm_provider.value} using {settings.llm_model}.",
        ),
        SystemCheck(
            group="Models",
            name="Model routing",
            status="Specialized" if model_tier_count else "Fallback only",
            tone="success" if model_tier_count else "warning",
            detail=(
                f"{model_tier_count:,} specialized model tiers configured."
                if model_tier_count
                else "All model tiers fall back to LLM_MODEL."
            ),
            action=(
                None
                if model_tier_count
                else "Set LLM_CHEAP_MODEL, LLM_ANALYSIS_MODEL, or document/high reasoning tiers."
            ),
        ),
        SystemCheck(
            group="Tools",
            name="Web search",
            status="Configured" if settings.brave_search_api_key else "Unavailable",
            tone="success" if settings.brave_search_api_key else "warning",
            detail=(
                "Brave Search API key is present."
                if settings.brave_search_api_key
                else "Web search tool will fail until BRAVE_SEARCH_API_KEY is set."
            ),
        ),
        SystemCheck(
            group="Observability",
            name="Tracing export",
            status=_observability_status(settings),
            tone=_observability_tone(settings),
            detail=_observability_detail(settings),
            action=(
                None
                if settings.otel_exporter_otlp_endpoint
                else "Run the observability profile or configure an OTLP endpoint for external traces."
            ),
        ),
    )


def _dashboard_auth_check(settings: DashboardSettings) -> SystemCheck:
    default_password = settings.password == "change-me"
    default_secret = settings.session_secret == "change-me-dashboard-session-secret"
    if default_password or default_secret:
        return SystemCheck(
            group="Dashboard",
            name="Dashboard auth",
            status="Needs hardening",
            tone="danger",
            detail="Default dashboard credentials or session secret are still in use.",
            action="Set DASHBOARD_PASSWORD and DASHBOARD_SESSION_SECRET before exposing the dashboard.",
        )
    if not settings.secure_cookies:
        return SystemCheck(
            group="Dashboard",
            name="Dashboard auth",
            status="Local mode",
            tone="warning",
            detail="Secure cookies are disabled, which is acceptable for local HTTP only.",
            action="Enable DASHBOARD_SECURE_COOKIES when serving over HTTPS.",
        )
    return SystemCheck(
        group="Dashboard",
        name="Dashboard auth",
        status="Hardened",
        tone="success",
        detail="Custom credentials and secure cookies are configured.",
    )


def _config_sections(
    *,
    dashboard_settings: DashboardSettings,
    runtime_settings: Settings | None,
    runtime_error: str | None,
) -> tuple[SystemConfigSection, ...]:
    dashboard_rows = (
        SystemConfigRow("Dashboard user", dashboard_settings.username),
        SystemConfigRow(
            "Dashboard password",
            "Default" if dashboard_settings.password == "change-me" else "Custom",
            tone=(
                "danger" if dashboard_settings.password == "change-me" else "success"
            ),
        ),
        SystemConfigRow(
            "Session secret",
            (
                "Default"
                if dashboard_settings.session_secret
                == "change-me-dashboard-session-secret"
                else "Custom"
            ),
            tone=(
                "danger"
                if dashboard_settings.session_secret
                == "change-me-dashboard-session-secret"
                else "success"
            ),
        ),
        SystemConfigRow(
            "Secure cookies",
            "Enabled" if dashboard_settings.secure_cookies else "Disabled",
            detail="Use enabled when served over HTTPS.",
            tone="success" if dashboard_settings.secure_cookies else "warning",
        ),
        SystemConfigRow(
            "Postgres URL",
            _redact_url(dashboard_settings.postgres_url),
            detail="Password is always hidden.",
        ),
    )

    sections: list[SystemConfigSection] = [
        SystemConfigSection("Dashboard", dashboard_rows),
    ]

    if runtime_settings is None:
        sections.append(
            SystemConfigSection(
                "Runtime",
                (
                    SystemConfigRow(
                        "Configuration",
                        "Invalid",
                        detail=runtime_error or "Runtime settings could not load.",
                        tone="danger",
                    ),
                ),
            )
        )
        return tuple(sections)

    sections.extend(
        (
            SystemConfigSection(
                "Runtime",
                (
                    SystemConfigRow(
                        "App Postgres URL",
                        _redact_url(runtime_settings.postgres_url),
                        detail="Runtime database target with password hidden.",
                    ),
                    SystemConfigRow(
                        "Release",
                        runtime_settings.kortny_release
                        or runtime_settings.kortny_version
                        or "Not set",
                    ),
                ),
            ),
            SystemConfigSection(
                "Slack",
                (
                    SystemConfigRow("App name", runtime_settings.slack_app_name),
                    SystemConfigRow("Bot token", "Configured", tone="success"),
                    SystemConfigRow("Socket app token", "Configured", tone="success"),
                    SystemConfigRow("Signing secret", "Configured", tone="success"),
                    SystemConfigRow(
                        "File read limit",
                        f"{runtime_settings.slack_file_read_max_bytes:,} bytes",
                    ),
                ),
            ),
            SystemConfigSection(
                "Models",
                (
                    SystemConfigRow("Provider", runtime_settings.llm_provider.value),
                    SystemConfigRow("Default model", runtime_settings.llm_model),
                    _model_row("Cheap model", runtime_settings.llm_cheap_model),
                    _model_row("Standard model", runtime_settings.llm_standard_model),
                    _model_row("Analysis model", runtime_settings.llm_analysis_model),
                    _model_row("Document model", runtime_settings.llm_document_model),
                    _model_row(
                        "High reasoning model",
                        runtime_settings.llm_high_reasoning_model,
                    ),
                ),
            ),
            SystemConfigSection(
                "Tools",
                (
                    SystemConfigRow(
                        "Brave Search",
                        (
                            "Configured"
                            if runtime_settings.brave_search_api_key
                            else "Missing"
                        ),
                        tone=(
                            "success"
                            if runtime_settings.brave_search_api_key
                            else "warning"
                        ),
                    ),
                    SystemConfigRow(
                        "Composio",
                        (
                            "Configured"
                            if runtime_settings.composio_api_key
                            else "Not configured"
                        ),
                        tone=(
                            "success"
                            if runtime_settings.composio_api_key
                            else "neutral"
                        ),
                    ),
                ),
            ),
            SystemConfigSection(
                "Observability",
                (
                    SystemConfigRow(
                        "Enabled",
                        "Yes" if runtime_settings.observability_enabled else "No",
                        tone=(
                            "success"
                            if runtime_settings.observability_enabled
                            else "warning"
                        ),
                    ),
                    SystemConfigRow(
                        "Capture mode",
                        runtime_settings.observability_capture_content,
                    ),
                    SystemConfigRow(
                        "OTLP endpoint",
                        runtime_settings.otel_exporter_otlp_endpoint
                        or "Not configured",
                        tone=(
                            "success"
                            if runtime_settings.otel_exporter_otlp_endpoint
                            else "warning"
                        ),
                    ),
                    SystemConfigRow(
                        "Trace sampling",
                        f"{runtime_settings.otel_trace_sampling_ratio:.2f}",
                    ),
                ),
            ),
        )
    )
    return tuple(sections)


def _integration_cards(
    settings: Settings | None,
    runtime_error: str | None,
) -> tuple[IntegrationCard, ...]:
    if settings is None:
        return (
            IntegrationCard(
                name="Runtime configuration",
                category="Core",
                status="Needs setup",
                tone="danger",
                description="Kortny cannot load runtime settings for integrations.",
                details=(
                    runtime_error or "Required environment variables are missing.",
                    "Set Slack, LLM, and Postgres values before checking tools.",
                ),
                env_vars=("SLACK_BOT_TOKEN", "LLM_API_KEY", "POSTGRES_URL"),
                action="Open System for the redacted configuration error.",
            ),
            IntegrationCard(
                name="Native tool registry",
                category="Tools",
                status="Blocked",
                tone="warning",
                description="Native tools depend on a valid runtime configuration.",
                details=(
                    "Tool metadata is visible, but runtime invocation is blocked.",
                ),
                env_vars=(),
            ),
        )

    model_tiers = tuple(
        model
        for model in (
            settings.llm_cheap_model,
            settings.llm_standard_model,
            settings.llm_analysis_model,
            settings.llm_document_model,
            settings.llm_high_reasoning_model,
        )
        if model
    )
    integrations = [
        IntegrationCard(
            name="Slack workspace",
            category="Transport",
            status="Configured",
            tone="success",
            description="Socket Mode transport for DMs, mentions, reactions, files, and channel context.",
            details=(
                f"App name: {settings.slack_app_name}",
                f"File read limit: {settings.slack_file_read_max_bytes:,} bytes",
                "Bot, app, and signing credentials are present.",
            ),
            env_vars=(
                "SLACK_BOT_TOKEN",
                "SLACK_APP_TOKEN",
                "SLACK_SIGNING_SECRET",
                "SLACK_APP_NAME",
            ),
        ),
        IntegrationCard(
            name="LLM provider",
            category="Inference",
            status="Configured",
            tone="success",
            description="Primary inference backend used by the coordinator, intent classifier, and model router.",
            details=(
                f"Provider: {settings.llm_provider.value}",
                f"Default model: {settings.llm_model}",
                (
                    f"{len(model_tiers):,} specialized routing tiers configured."
                    if model_tiers
                    else "Specialized tiers fall back to LLM_MODEL."
                ),
            ),
            env_vars=(
                "LLM_PROVIDER",
                "LLM_API_KEY",
                "LLM_MODEL",
                "LLM_CHEAP_MODEL",
                "LLM_STANDARD_MODEL",
                "LLM_ANALYSIS_MODEL",
                "LLM_DOCUMENT_MODEL",
                "LLM_HIGH_REASONING_MODEL",
            ),
            action=(
                None
                if model_tiers
                else "Set tier-specific model env vars to make routing explicit."
            ),
        ),
        IntegrationCard(
            name="Brave Search",
            category="Research",
            status="Configured" if settings.brave_search_api_key else "Missing",
            tone="success" if settings.brave_search_api_key else "warning",
            description="Public web search provider used by the native web_search tool.",
            details=(
                (
                    "API key is present. The tool still respects Brave API rate limits."
                    if settings.brave_search_api_key
                    else "The web_search tool needs BRAVE_SEARCH_API_KEY before it can run."
                ),
            ),
            env_vars=("BRAVE_SEARCH_API_KEY",),
            action=(
                None
                if settings.brave_search_api_key
                else "Add BRAVE_SEARCH_API_KEY to enable web research."
            ),
        ),
        IntegrationCard(
            name="PDF generation",
            category="Documents",
            status="Built in",
            tone="success",
            description="ReportLab-backed document generation running inside the worker container.",
            details=(
                "No external account required.",
                "Uses task workspace storage and records generated artifacts.",
            ),
            env_vars=(),
        ),
        IntegrationCard(
            name="Workspace memory",
            category="Memory",
            status="Available",
            tone="success",
            description="Confirm-gated workspace_state memory tools backed by Postgres.",
            details=(
                "Facts, proposals, supersession, and forget events are stored with audit metadata.",
                "Episodic recall is recorded separately from durable facts.",
            ),
            env_vars=("POSTGRES_URL",),
        ),
        IntegrationCard(
            name="Observability",
            category="Operations",
            status=_integration_observability_status(settings),
            tone=_integration_observability_tone(settings),
            description="Structured logs, task events, LLM usage, and optional OTLP export.",
            details=(
                f"Capture mode: {settings.observability_capture_content}",
                (
                    f"OTLP endpoint: {settings.otel_exporter_otlp_endpoint}"
                    if settings.otel_exporter_otlp_endpoint
                    else "No OTLP endpoint configured; dashboard still reads local task and usage rows."
                ),
                f"Trace sampling: {settings.otel_trace_sampling_ratio:.2f}",
            ),
            env_vars=(
                "OBSERVABILITY_ENABLED",
                "OBSERVABILITY_CAPTURE_CONTENT",
                "OTEL_EXPORTER_OTLP_ENDPOINT",
                "OTEL_TRACE_SAMPLING_RATIO",
            ),
            action=(
                None
                if settings.otel_exporter_otlp_endpoint
                else "Run the observability profile or connect an OTLP endpoint for external traces."
            ),
        ),
        IntegrationCard(
            name="Langfuse",
            category="Prompts",
            status="Enabled" if settings.langfuse_enabled else "Optional",
            tone="success" if settings.langfuse_enabled else "neutral",
            description="Optional hosted prompt and trace backend for teams that want cloud prompt management.",
            details=(
                (
                    f"Host: {settings.langfuse_host or 'not set'}"
                    if settings.langfuse_enabled
                    else "Not required for local self-hosting."
                ),
                (
                    "Prompt fetching is enabled."
                    if settings.langfuse_prompts_enabled
                    else "Prompt fetching is disabled."
                ),
            ),
            env_vars=(
                "LANGFUSE_ENABLED",
                "LANGFUSE_HOST",
                "LANGFUSE_PUBLIC_KEY",
                "LANGFUSE_SECRET_KEY",
                "LANGFUSE_PROMPTS_ENABLED",
            ),
        ),
        IntegrationCard(
            name="Composio",
            category="External tools",
            status="Key present, catalog available"
            if settings.composio_api_key
            else "Planned",
            tone="warning" if settings.composio_api_key else "neutral",
            description="Third-party app catalog and scoped connected-account adapter.",
            details=(
                (
                    "COMPOSIO_API_KEY is present; HIG-35 is wiring catalog and scoped-account metadata before runtime tool use."
                    if settings.composio_api_key
                    else "No key configured. HIG-35 tracks the actual integration adapter."
                ),
                "Runtime tool execution stays disabled until per-task visibility gates are in place.",
            ),
            env_vars=(
                "COMPOSIO_API_KEY",
                "COMPOSIO_CATALOG_ENABLED",
                "COMPOSIO_CATALOG_LIMIT",
            ),
            action="Catalog is read-only first; OAuth connect and runtime execution are follow-up slices.",
        ),
    ]
    return tuple(integrations)


def _composio_catalog_view(
    *,
    session: Session | None,
    settings: Settings | None,
    query: str | None,
    client: ComposioClient | None,
    installation_id: uuid.UUID | None = None,
    owner_slack_user_id: str | None = None,
) -> ComposioCatalogView:
    normalized_query = (query or "").strip()
    connections = _composio_connection_rows(
        session,
        installation_id=installation_id,
        owner_slack_user_id=owner_slack_user_id,
    )
    active_count = sum(1 for connection in connections if connection.status == "active")
    if settings is None or not settings.composio_api_key:
        return ComposioCatalogView(
            enabled=False,
            configured=False,
            status="Not configured",
            tone="neutral",
            query=normalized_query,
            total_items=None,
            visible_count=0,
            connection_count=len(connections),
            active_connection_count=active_count,
            error=None,
            toolkits=(),
            connections=connections,
        )
    if not settings.composio_catalog_enabled:
        return ComposioCatalogView(
            enabled=False,
            configured=True,
            status="Catalog disabled",
            tone="warning",
            query=normalized_query,
            total_items=None,
            visible_count=0,
            connection_count=len(connections),
            active_connection_count=active_count,
            error="COMPOSIO_CATALOG_ENABLED is false.",
            toolkits=(),
            connections=connections,
        )

    resolved_client = client or ComposioClient(
        api_key=settings.composio_api_key,
        timeout_seconds=settings.composio_request_timeout_seconds,
    )
    try:
        catalog = resolved_client.list_toolkits(
            search=normalized_query or None,
            limit=settings.composio_catalog_limit,
        )
    except ComposioCatalogError as exc:
        return ComposioCatalogView(
            enabled=True,
            configured=True,
            status="Catalog unavailable",
            tone="danger",
            query=normalized_query,
            total_items=None,
            visible_count=0,
            connection_count=len(connections),
            active_connection_count=active_count,
            error=_short_error(str(exc)),
            toolkits=(),
            connections=connections,
        )

    connection_statuses = _composio_status_by_toolkit(connections)
    toolkits = tuple(
        _composio_toolkit_row(toolkit, connection_statuses)
        for _index, toolkit in sorted(
            enumerate(catalog.items),
            key=lambda item: (
                connection_statuses.get(item[1].slug) != "active",
                item[0],
            ),
        )
    )
    return ComposioCatalogView(
        enabled=True,
        configured=True,
        status="Catalog synced",
        tone="success",
        query=normalized_query,
        total_items=catalog.total_items,
        visible_count=len(toolkits),
        connection_count=len(connections),
        active_connection_count=active_count,
        error=None,
        toolkits=toolkits,
        connections=connections,
    )


def _composio_connection_rows(
    session: Session | None,
    *,
    installation_id: uuid.UUID | None = None,
    owner_slack_user_id: str | None = None,
) -> tuple[ComposioConnectionRow, ...]:
    if session is None:
        return ()
    filters: list[ColumnElement[bool]] = []
    if installation_id is not None:
        filters.append(ComposioConnection.installation_id == installation_id)
    if owner_slack_user_id:
        filters.append(ComposioConnection.owner_slack_user_id == owner_slack_user_id)
    rows = tuple(
        session.scalars(
            select(ComposioConnection)
            .where(*filters)
            .order_by(
                ComposioConnection.updated_at.desc(),
                ComposioConnection.id.desc(),
            )
        )
    )
    identity_keys: list[IdentityKey] = []
    for row in rows:
        identity_keys.append((row.installation_id, "user", row.owner_slack_user_id))
        if row.visibility_scope_type == "channel" and row.visibility_scope_id:
            identity_keys.append((row.installation_id, "channel", row.visibility_scope_id))
        elif row.visibility_scope_type == "user" and row.visibility_scope_id:
            identity_keys.append((row.installation_id, "user", row.visibility_scope_id))
    identities = _identity_map_from_keys(session, identity_keys)
    return tuple(_composio_connection_row(row, identities) for row in rows)


def _composio_connection_row(
    row: ComposioConnection,
    identities: dict[IdentityKey, SlackIdentity],
) -> ComposioConnectionRow:
    owner = _identity_label(
        identities,
        installation_id=row.installation_id,
        kind="user",
        slack_id=row.owner_slack_user_id,
    )
    return ComposioConnectionRow(
        id=row.id,
        toolkit_slug=row.toolkit_slug,
        status=row.status,
        tone=_composio_connection_tone(row.status),
        display_name=row.display_name or row.external_account_label or row.toolkit_slug,
        scope_label=_composio_scope_label(row, identities),
        visibility_scope_type=row.visibility_scope_type,
        visibility_scope_id=row.visibility_scope_id,
        owner=owner,
        connected_account_id=row.connected_account_id,
        auth_config_id=row.auth_config_id,
        updated_at=row.updated_at,
    )


def _composio_auth_config_row(auth_config: ComposioAuthConfig) -> ComposioAuthConfigRow:
    return ComposioAuthConfigRow(
        id=auth_config.id,
        name=auth_config.name,
        toolkit_slug=auth_config.toolkit_slug,
        auth_scheme=auth_config.auth_scheme,
        is_composio_managed=auth_config.is_composio_managed,
        enabled=auth_config.enabled,
    )


def _slack_identity_options(
    session: Session,
    *,
    kind: str,
) -> tuple[IdentityLabel, ...]:
    rows = tuple(
        session.scalars(
            select(SlackIdentity)
            .where(SlackIdentity.kind == kind)
            .order_by(
                SlackIdentity.display_name.asc(),
                SlackIdentity.slack_id.asc(),
                SlackIdentity.updated_at.desc(),
            )
        )
    )
    options: list[IdentityLabel] = []
    seen: set[str] = set()
    for row in rows:
        if row.slack_id in seen:
            continue
        seen.add(row.slack_id)
        options.append(
            IdentityLabel(
                name=row.display_name or row.raw_name or row.slack_id,
                slack_id=row.slack_id,
                found=True,
            )
        )
    return tuple(options)


def _composio_scope_label(
    row: ComposioConnection,
    identities: dict[IdentityKey, SlackIdentity],
) -> str:
    if row.visibility_scope_type == "workspace":
        return "Workspace"
    if row.visibility_scope_id is None:
        return row.visibility_scope_type.title()
    kind = "channel" if row.visibility_scope_type == "channel" else "user"
    label = _identity_label(
        identities,
        installation_id=row.installation_id,
        kind=kind,
        slack_id=row.visibility_scope_id,
    )
    return label.name


def _composio_status_by_toolkit(
    connections: tuple[ComposioConnectionRow, ...],
) -> dict[str, str]:
    statuses: dict[str, str] = {}
    priority = {"active": 4, "pending": 3, "expired": 2, "failed": 1, "disabled": 0}
    for connection in connections:
        current = statuses.get(connection.toolkit_slug)
        if current is None or priority.get(connection.status, -1) > priority.get(current, -1):
            statuses[connection.toolkit_slug] = connection.status
    return statuses


def _composio_toolkit_row(
    toolkit: ComposioToolkit,
    connection_statuses: dict[str, str],
) -> ComposioToolkitRow:
    status = connection_statuses.get(toolkit.slug)
    if status is None:
        connection_status = "Available"
        connection_tone = "neutral"
    else:
        connection_status = status.title()
        connection_tone = _composio_connection_tone(status)
    return ComposioToolkitRow(
        slug=toolkit.slug,
        name=toolkit.name,
        description=toolkit.description,
        categories=toolkit.categories,
        auth_schemes=toolkit.auth_schemes,
        managed_auth_schemes=toolkit.managed_auth_schemes,
        tools_count=toolkit.tools_count,
        triggers_count=toolkit.triggers_count,
        no_auth=toolkit.no_auth,
        connection_status=connection_status,
        connection_tone=connection_tone,
        connected=status == "active",
    )


def _composio_scope_options() -> tuple[ComposioScopeOption, ...]:
    return (
        ComposioScopeOption(
            name="Personal",
            key="user",
            description="Only the Slack user who connected the account can use it.",
            default=True,
        ),
        ComposioScopeOption(
            name="Channel",
            key="channel",
            description="Tasks in one Slack channel can use the connected account.",
            risk="Good for shared project apps; risky for personal inboxes or calendars.",
        ),
        ComposioScopeOption(
            name="Workspace",
            key="workspace",
            description="Any task in the Slack workspace can use the connected account.",
            risk="Requires explicit admin-level intent before enabling.",
        ),
    )


def _composio_connection_tone(status: str) -> str:
    return {
        "active": "success",
        "pending": "warning",
        "expired": "warning",
        "failed": "danger",
        "disabled": "neutral",
    }.get(status, "neutral")


def _short_error(value: str, *, max_chars: int = 220) -> str:
    cleaned = " ".join(value.split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 1].rstrip() + "..."


def _tool_capability_groups(
    settings: Settings | None,
) -> tuple[ToolCapabilityGroup, ...]:
    runtime_available = settings is not None
    slack_available = settings is not None
    web_available = bool(settings and settings.brave_search_api_key)
    return (
        ToolCapabilityGroup(
            name="Research",
            description="Tools that gather external or Slack-grounded context.",
            tools=(
                _tool_capability(
                    WebSearchTool,
                    group="Research",
                    available=web_available,
                    unavailable_note="Requires BRAVE_SEARCH_API_KEY.",
                ),
                _tool_capability(
                    SlackChannelHistoryTool,
                    group="Research",
                    available=slack_available,
                    unavailable_note="Requires valid Slack runtime settings.",
                ),
                _tool_capability(
                    SlackFileReadTool,
                    group="Research",
                    available=slack_available,
                    unavailable_note="Requires valid Slack runtime settings.",
                ),
            ),
        ),
        ToolCapabilityGroup(
            name="Documents",
            description="Tools that create task artifacts.",
            tools=(
                _tool_capability(
                    PdfGeneratorTool,
                    group="Documents",
                    available=runtime_available,
                    unavailable_note="Requires valid runtime settings and worker task storage.",
                ),
            ),
        ),
        ToolCapabilityGroup(
            name="Memory",
            description="Confirm-gated fact memory and operator-visible recall tools.",
            tools=(
                _tool_capability(
                    RememberFactTool,
                    group="Memory",
                    available=runtime_available,
                    unavailable_note="Requires Postgres and Slack confirmation flow.",
                ),
                _tool_capability(
                    RecallFactTool,
                    group="Memory",
                    available=runtime_available,
                    unavailable_note="Requires Postgres-backed workspace_state.",
                ),
                _tool_capability(
                    InspectMemoryTool,
                    group="Memory",
                    available=runtime_available,
                    unavailable_note="Requires Postgres-backed workspace_state.",
                ),
                _tool_capability(
                    ForgetFactTool,
                    group="Memory",
                    available=runtime_available,
                    unavailable_note="Requires Postgres-backed workspace_state.",
                ),
            ),
        ),
    )


def _tool_capability(
    tool: type[Any],
    *,
    group: str,
    available: bool,
    unavailable_note: str,
) -> ToolCapability:
    required_args, optional_args = _tool_argument_names(tool.parameters)
    return ToolCapability(
        name=tool.name,
        group=group,
        status="Available" if available else "Needs setup",
        tone="success" if available else "warning",
        description=tool.description,
        required_args=required_args,
        optional_args=optional_args,
        notes=(
            "Provider-neutral JSON tool contract.",
            unavailable_note
            if not available
            else "Registered by the worker agent executor.",
        ),
    )


def _tool_argument_names(
    schema: dict[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return (), ()
    required_values = schema.get("required", ())
    required = tuple(
        name for name in required_values if isinstance(name, str) and name in properties
    )
    optional = tuple(
        name for name in properties if isinstance(name, str) and name not in required
    )
    return required, optional


def _integration_observability_status(settings: Settings) -> str:
    if not settings.observability_enabled:
        return "Disabled"
    if settings.otel_exporter_otlp_endpoint:
        return "OTLP export"
    return "Local metadata"


def _integration_observability_tone(settings: Settings) -> str:
    if not settings.observability_enabled:
        return "warning"
    if settings.otel_exporter_otlp_endpoint:
        return "success"
    return "neutral"


def _model_row(label: str, value: str | None) -> SystemConfigRow:
    if value:
        return SystemConfigRow(label, value, tone="success")
    return SystemConfigRow(
        label,
        "Fallback to default",
        detail="Set the tier-specific env var to override.",
        tone="warning",
    )


def _overall_tone(checks: Sequence[SystemCheck]) -> str:
    tones = {check.tone for check in checks}
    if "danger" in tones:
        return "danger"
    if "warning" in tones:
        return "warning"
    return "success"


def _overall_label(checks: Sequence[SystemCheck]) -> str:
    tone = _overall_tone(checks)
    if tone == "danger":
        return "Needs setup"
    if tone == "warning":
        return "Needs attention"
    return "Ready"


def _observability_status(settings: Settings) -> str:
    if not settings.observability_enabled:
        return "Disabled"
    if settings.otel_exporter_otlp_endpoint:
        return "Exporting"
    return "Local only"


def _observability_tone(settings: Settings) -> str:
    if not settings.observability_enabled:
        return "warning"
    if settings.otel_exporter_otlp_endpoint:
        return "success"
    return "warning"


def _observability_detail(settings: Settings) -> str:
    if not settings.observability_enabled:
        return "Task events still exist, but OTEL instrumentation is disabled."
    if settings.otel_exporter_otlp_endpoint:
        return f"Traces export to {_redact_url(settings.otel_exporter_otlp_endpoint)}."
    return "Structured logs and task events are available; no external trace sink is configured."


def _datetime_label(value: datetime | None) -> str:
    if value is None:
        return "-"
    return value.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")


def _redact_url(value: str) -> str:
    parsed = urlsplit(value)
    if not parsed.scheme or not parsed.netloc:
        return value

    host = parsed.hostname or ""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"

    if parsed.username:
        user = parsed.username
        auth = f"{user}:***@"
    else:
        auth = ""

    return urlunsplit((parsed.scheme, f"{auth}{host}", parsed.path, "", ""))


def _failure_rate_label(total: int, failed: int) -> str:
    if total <= 0:
        return "0.0%"
    return f"{(failed / total) * 100:.1f}%"


def _overview_attention_items(
    session: Session,
    *,
    system_health: SystemHealth,
) -> tuple[OverviewAttentionItem, ...]:
    items: list[OverviewAttentionItem] = []
    for check in system_health.checks:
        if check.tone not in {"danger", "warning"}:
            continue
        items.append(
            OverviewAttentionItem(
                title=f"{check.group}: {check.name}",
                detail=check.action or check.detail,
                tone=check.tone,
                badge=check.status,
                href="/system",
            )
        )

    tasks = tuple(
        session.scalars(
            select(Task)
            .where(
                Task.status.in_(
                    (
                        TaskStatus.failed,
                        TaskStatus.crashed,
                        TaskStatus.pending,
                        TaskStatus.running,
                    )
                )
            )
            .order_by(
                case(
                    (Task.status.in_((TaskStatus.failed, TaskStatus.crashed)), 0),
                    else_=1,
                ),
                Task.created_at.desc(),
                Task.id.desc(),
            )
            .limit(5)
        )
    )
    for item in _task_items(session, tasks):
        tone = (
            "danger"
            if item.task.status in (TaskStatus.failed, TaskStatus.crashed)
            else "warning"
        )
        items.append(
            OverviewAttentionItem(
                title=f"{item.task.status.value.capitalize()}: {_truncate(item.task.input, 86)}",
                detail=(
                    f"{item.user.name} in {item.channel.name} - "
                    f"{_datetime_label(item.task.created_at)}"
                ),
                tone=tone,
                badge=item.task.status.value,
                href=f"/tasks/{item.task.id}",
            )
        )

    return tuple(items[:8])


def _truncate(value: str, max_length: int) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= max_length:
        return normalized
    return normalized[: max_length - 3].rstrip() + "..."


def _memory_fact_identity_keys(
    facts: Sequence[WorkspaceState],
) -> tuple[IdentityKey, ...]:
    keys: list[IdentityKey] = []
    for fact in facts:
        if fact.scope_type in {"channel", "user"} and fact.scope_id:
            keys.append((fact.installation_id, fact.scope_type, fact.scope_id))
        for user_id in (
            fact.proposed_by,
            fact.confirmed_by_user_id,
            fact.rejected_by_user_id,
            fact.forgotten_by_user_id,
        ):
            if user_id:
                keys.append((fact.installation_id, "user", user_id))
    return tuple(keys)


_MEMORY_SCOPES = frozenset({"all", "workspace", "channel", "user"})
_MEMORY_STATUSES = frozenset(
    {"all", "active", "proposed", "rejected", "superseded", "forgotten"}
)
_MEMORY_OUTCOMES = frozenset({"all", "succeeded", "failed", "cancelled"})
_MEMORY_FACT_SORTS = frozenset({"updated_desc", "created_desc", "key_asc", "scope_asc"})
_MEMORY_EPISODE_SORTS = frozenset({"created_desc", "created_asc", "outcome_asc"})


def _normalize_memory_sort(view: str, sort: str | None) -> str:
    if view == "episodes":
        return sort if sort in _MEMORY_EPISODE_SORTS else "created_desc"
    return sort if sort in _MEMORY_FACT_SORTS else "updated_desc"


def _memory_fact_rows(
    session: Session,
    *,
    query: str,
    scope_filter: str,
    status_filter: str,
    sort: str,
    page: int,
    page_size: int,
    installation_id: uuid.UUID | None = None,
    slack_user_id: str | None = None,
) -> tuple[int, int, tuple[MemoryFactRow, ...]]:
    filters = _memory_fact_filters(
        query=query,
        scope_filter=scope_filter,
        status_filter=status_filter,
        installation_id=installation_id,
        slack_user_id=slack_user_id,
    )
    total_count = (
        session.scalar(select(func.count()).select_from(WorkspaceState).where(*filters))
        or 0
    )
    resolved_page = _resolved_page(
        page=page, page_size=page_size, total_count=total_count
    )
    facts = tuple(
        session.scalars(
            select(WorkspaceState)
            .where(*filters)
            .order_by(*_memory_fact_order(sort))
            .offset((resolved_page - 1) * page_size)
            .limit(page_size)
        )
    )
    source_tasks = _tasks_by_id(
        session,
        [fact.source_task_id for fact in facts if fact.source_task_id],
    )
    fact_identities = _identity_map_from_keys(
        session,
        _memory_fact_identity_keys(facts),
    )
    rows = tuple(
        MemoryFactRow(
            fact=fact,
            scope=_memory_scope_label(fact_identities, fact),
            value_summary=_memory_value_summary(fact),
            confirmed_by=_optional_user_label(
                fact_identities,
                installation_id=fact.installation_id,
                slack_id=fact.confirmed_by_user_id,
            ),
            proposed_by=_optional_user_label(
                fact_identities,
                installation_id=fact.installation_id,
                slack_id=fact.proposed_by,
            ),
            rejected_by=_optional_user_label(
                fact_identities,
                installation_id=fact.installation_id,
                slack_id=fact.rejected_by_user_id,
            ),
            forgotten_by=_optional_user_label(
                fact_identities,
                installation_id=fact.installation_id,
                slack_id=fact.forgotten_by_user_id,
            ),
            source_task=(
                source_tasks.get(fact.source_task_id)
                if fact.source_task_id is not None
                else None
            ),
            tone=_memory_status_tone(fact.status),
        )
        for fact in facts
    )
    return int(total_count), resolved_page, rows


def _memory_fact_filters(
    *,
    query: str,
    scope_filter: str,
    status_filter: str,
    installation_id: uuid.UUID | None = None,
    slack_user_id: str | None = None,
) -> list[ColumnElement[bool]]:
    filters = _workspace_state_scope_filter(
        installation_id=installation_id,
        slack_user_id=slack_user_id,
    )
    if status_filter != "all":
        filters.append(WorkspaceState.status == status_filter)
    if scope_filter != "all":
        filters.append(WorkspaceState.scope_type == scope_filter)
    if query:
        pattern = f"%{query}%"
        scope_identity_match = (
            select(SlackIdentity.id)
            .where(
                SlackIdentity.installation_id == WorkspaceState.installation_id,
                SlackIdentity.kind == WorkspaceState.scope_type,
                SlackIdentity.slack_id == WorkspaceState.scope_id,
                or_(
                    SlackIdentity.display_name.ilike(pattern),
                    SlackIdentity.raw_name.ilike(pattern),
                ),
            )
            .exists()
        )
        filters.append(
            or_(
                WorkspaceState.key.ilike(pattern),
                WorkspaceState.scope_id.ilike(pattern),
                WorkspaceState.value_text.ilike(pattern),
                cast(WorkspaceState.value_json, Text).ilike(pattern),
                scope_identity_match,
            )
        )
    return filters


def _memory_fact_order(sort: str) -> tuple[Any, ...]:
    if sort == "created_desc":
        return (WorkspaceState.created_at.desc(), WorkspaceState.id.desc())
    if sort == "key_asc":
        return (WorkspaceState.key.asc(), WorkspaceState.updated_at.desc())
    if sort == "scope_asc":
        return (
            WorkspaceState.scope_type.asc(),
            WorkspaceState.scope_id.asc().nullsfirst(),
            WorkspaceState.key.asc(),
        )
    return (WorkspaceState.updated_at.desc(), WorkspaceState.created_at.desc())


def _memory_episode_rows(
    session: Session,
    *,
    query: str,
    outcome_filter: str,
    sort: str,
    page: int,
    page_size: int,
    installation_id: uuid.UUID | None = None,
    slack_user_id: str | None = None,
) -> tuple[int, int, tuple[MemoryEpisodeRow, ...]]:
    filters = _memory_episode_filters(
        query=query,
        outcome_filter=outcome_filter,
        installation_id=installation_id,
        slack_user_id=slack_user_id,
    )
    base = select(Episode)
    count_base = select(func.count()).select_from(Episode)
    if query:
        base = base.join(Task, Task.id == Episode.task_id)
        count_base = count_base.join(Task, Task.id == Episode.task_id)
    total_count = session.scalar(count_base.where(*filters)) or 0
    resolved_page = _resolved_page(
        page=page, page_size=page_size, total_count=total_count
    )
    episodes = tuple(
        session.scalars(
            base.where(*filters)
            .order_by(*_memory_episode_order(sort))
            .offset((resolved_page - 1) * page_size)
            .limit(page_size)
        )
    )
    episode_tasks = _tasks_by_id(session, [episode.task_id for episode in episodes])
    episode_identities = _identity_map_from_keys(
        session,
        (
            key
            for episode in episodes
            for key in (
                (episode.installation_id, "channel", episode.channel_id),
                (episode.installation_id, "user", episode.user_id),
            )
        ),
    )
    rows = tuple(
        MemoryEpisodeRow(
            episode=episode,
            channel=_identity_label(
                episode_identities,
                installation_id=episode.installation_id,
                kind="channel",
                slack_id=episode.channel_id,
            ),
            user=_identity_label(
                episode_identities,
                installation_id=episode.installation_id,
                kind="user",
                slack_id=episode.user_id,
            ),
            task=episode_tasks.get(episode.task_id),
            tools_label=_list_count_label(episode.tools_used, "tool"),
            artifacts_label=_list_count_label(episode.artifacts_created, "artifact"),
            source_refs_label=_list_count_label(episode.source_refs, "source"),
            tone=_episode_tone(episode.outcome),
        )
        for episode in episodes
    )
    return int(total_count), resolved_page, rows


def _memory_episode_filters(
    *,
    query: str,
    outcome_filter: str,
    installation_id: uuid.UUID | None = None,
    slack_user_id: str | None = None,
) -> list[ColumnElement[bool]]:
    filters = _episode_scope_filter(
        installation_id=installation_id,
        slack_user_id=slack_user_id,
    )
    if outcome_filter != "all":
        filters.append(Episode.outcome == outcome_filter)
    if query:
        pattern = f"%{query}%"
        channel_identity_match = (
            select(SlackIdentity.id)
            .where(
                SlackIdentity.installation_id == Episode.installation_id,
                SlackIdentity.kind == "channel",
                SlackIdentity.slack_id == Episode.channel_id,
                or_(
                    SlackIdentity.display_name.ilike(pattern),
                    SlackIdentity.raw_name.ilike(pattern),
                ),
            )
            .exists()
        )
        user_identity_match = (
            select(SlackIdentity.id)
            .where(
                SlackIdentity.installation_id == Episode.installation_id,
                SlackIdentity.kind == "user",
                SlackIdentity.slack_id == Episode.user_id,
                or_(
                    SlackIdentity.display_name.ilike(pattern),
                    SlackIdentity.raw_name.ilike(pattern),
                ),
            )
            .exists()
        )
        filters.append(
            or_(
                Episode.summary.ilike(pattern),
                Episode.channel_id.ilike(pattern),
                Episode.user_id.ilike(pattern),
                Task.input.ilike(pattern),
                channel_identity_match,
                user_identity_match,
            )
        )
    return filters


def _memory_episode_order(sort: str) -> tuple[Any, ...]:
    if sort == "created_asc":
        return (Episode.created_at.asc(), Episode.id.asc())
    if sort == "outcome_asc":
        return (Episode.outcome.asc(), Episode.created_at.desc())
    return (Episode.created_at.desc(), Episode.id.desc())


def _memory_page_url(
    *,
    view: str,
    query: str,
    scope_filter: str,
    status_filter: str,
    outcome_filter: str,
    sort: str,
    page: int | None,
    page_size: int,
    base_path: str = "/memory",
) -> str:
    params: dict[str, str | int] = {
        "view": view,
        "sort": sort,
        "page": page or 1,
        "page_size": page_size,
    }
    if query:
        params["q"] = query
    if view == "facts":
        if scope_filter != "all":
            params["scope"] = scope_filter
        if status_filter != "active":
            params["status"] = status_filter
    else:
        if outcome_filter != "all":
            params["outcome"] = outcome_filter
    return f"{base_path}?{urlencode(params)}"


def _resolved_page(*, page: int, page_size: int, total_count: int) -> int:
    total_pages = max(1, math.ceil(total_count / page_size)) if total_count else 1
    return min(max(page, 1), total_pages)


def _memory_scope_label(
    identities: dict[IdentityKey, SlackIdentity],
    fact: WorkspaceState,
) -> IdentityLabel:
    if fact.scope_type == "workspace":
        return IdentityLabel(name="Workspace", slack_id="workspace", found=True)
    if fact.scope_id is None:
        return IdentityLabel(name="-", slack_id="-", found=False)
    return _identity_label(
        identities,
        installation_id=fact.installation_id,
        kind=fact.scope_type,
        slack_id=fact.scope_id,
    )


def _optional_user_label(
    identities: dict[IdentityKey, SlackIdentity],
    *,
    installation_id: uuid.UUID,
    slack_id: str | None,
) -> IdentityLabel | None:
    if not slack_id:
        return None
    return _identity_label(
        identities,
        installation_id=installation_id,
        kind="user",
        slack_id=slack_id,
    )


def _memory_value_summary(fact: WorkspaceState) -> str:
    if fact.value_text:
        return _truncate(fact.value_text, 180)
    return _truncate(json.dumps(fact.value_json, sort_keys=True, default=str), 180)


def _memory_status_tone(status: str) -> str:
    if status == "active":
        return "success"
    if status == "proposed":
        return "warning"
    if status in {"rejected", "forgotten"}:
        return "danger"
    return "neutral"


def _episode_tone(outcome: str) -> str:
    if outcome == "succeeded":
        return "success"
    if outcome == "failed":
        return "danger"
    return "warning"


def _list_count_label(items: object, singular: str) -> str:
    if not isinstance(items, list):
        return f"0 {singular}s"
    count = len(items)
    suffix = "" if count == 1 else "s"
    return f"{count:,} {singular}{suffix}"


def _tasks_by_id(
    session: Session,
    task_ids: Sequence[uuid.UUID | None],
) -> dict[uuid.UUID, Task]:
    normalized = tuple({task_id for task_id in task_ids if task_id is not None})
    if not normalized:
        return {}
    return {
        task.id: task
        for task in session.scalars(select(Task).where(Task.id.in_(normalized)))
    }


IdentityKey = tuple[uuid.UUID, str, str]


def _task_items(session: Session, tasks: Sequence[Task]) -> tuple[TaskListItem, ...]:
    usage_by_task = _usage_by_task(session, [task.id for task in tasks])
    identities = _identity_map(session, tasks)
    return tuple(
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


def _user_label_for_tasks(
    session: Session,
    *,
    slack_user_id: str,
    tasks: Sequence[Task],
) -> IdentityLabel:
    keys = [
        (task.installation_id, "user", slack_user_id)
        for task in tasks
        if task.slack_user_id == slack_user_id
    ]
    identities = _identity_map_from_keys(session, keys)
    for key in keys:
        identity = identities.get(key)
        if identity is not None:
            return IdentityLabel(
                name=identity.display_name,
                slack_id=identity.slack_id,
                found=True,
            )
    return IdentityLabel(name=slack_user_id, slack_id=slack_user_id, found=False)


def _failed_task_case() -> Any:
    return case(
        (Task.status.in_((TaskStatus.failed, TaskStatus.crashed)), 1),
        else_=0,
    )


def _artifact_counts_by_user(
    session: Session,
    task_filter: Sequence[ColumnElement[bool]],
) -> dict[tuple[uuid.UUID, str], int]:
    rows = session.execute(
        select(Task.installation_id, Task.slack_user_id, func.count(Artifact.id))
        .join(Artifact, Artifact.task_id == Task.id)
        .where(*task_filter)
        .group_by(Task.installation_id, Task.slack_user_id)
    )
    return {(row[0], row[1]): int(row[2]) for row in rows}


def _artifact_counts_by_task(
    session: Session,
    task_ids: Sequence[uuid.UUID],
) -> dict[uuid.UUID, int]:
    if not task_ids:
        return {}
    rows = session.execute(
        select(Artifact.task_id, func.count(Artifact.id))
        .where(Artifact.task_id.in_(task_ids))
        .group_by(Artifact.task_id)
    )
    return {row[0]: int(row[1]) for row in rows}


def _artifact_count_for_user(
    session: Session,
    task_filter: Sequence[ColumnElement[bool]],
) -> int:
    return int(
        session.scalar(
            select(func.count(Artifact.id))
            .join(Task, Task.id == Artifact.task_id)
            .where(*task_filter)
        )
        or 0
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


def _task_filter(
    *, start: datetime | None, end: datetime | None
) -> list[ColumnElement[bool]]:
    filters: list[ColumnElement[bool]] = []
    if start is not None:
        filters.append(Task.created_at >= start)
    if end is not None:
        filters.append(Task.created_at < end)
    return filters


def _task_scope_filter(
    *,
    installation_id: uuid.UUID | None = None,
    slack_user_id: str | None = None,
) -> list[ColumnElement[bool]]:
    filters: list[ColumnElement[bool]] = []
    if installation_id is not None:
        filters.append(Task.installation_id == installation_id)
    if slack_user_id:
        filters.append(Task.slack_user_id == slack_user_id)
    return filters


def _workspace_state_scope_filter(
    *,
    installation_id: uuid.UUID | None = None,
    slack_user_id: str | None = None,
) -> list[ColumnElement[bool]]:
    filters: list[ColumnElement[bool]] = []
    if installation_id is not None:
        filters.append(WorkspaceState.installation_id == installation_id)
    if slack_user_id:
        filters.append(WorkspaceState.scope_type == "user")
        filters.append(WorkspaceState.scope_id == slack_user_id)
    return filters


def _episode_scope_filter(
    *,
    installation_id: uuid.UUID | None = None,
    slack_user_id: str | None = None,
) -> list[ColumnElement[bool]]:
    filters: list[ColumnElement[bool]] = []
    if installation_id is not None:
        filters.append(Episode.installation_id == installation_id)
    if slack_user_id:
        filters.append(Episode.user_id == slack_user_id)
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


def _daily_task_row(row: Row[Any]) -> DailyTaskRow:
    day_value, task_count, failed_task_count = row
    if isinstance(day_value, datetime):
        day = day_value.date()
    elif isinstance(day_value, date):
        day = day_value
    else:
        day = date.fromisoformat(str(day_value)[:10])
    return DailyTaskRow(
        day=day,
        task_count=int(task_count),
        failed_task_count=int(failed_task_count),
    )


def _daily_cost_points(rows: Sequence[DailyUsageRow]) -> tuple[ChartPoint, ...]:
    ordered = tuple(reversed(rows))
    max_value = max((row.cost_usd for row in ordered), default=Decimal("0"))
    return tuple(
        ChartPoint(
            label=row.day.isoformat(),
            value_label=_format_money(row.cost_usd),
            percent=_percent_of(row.cost_usd, max_value),
        )
        for row in ordered
    )


def _daily_task_points(rows: Sequence[DailyTaskRow]) -> tuple[ChartPoint, ...]:
    ordered = tuple(reversed(rows))
    max_value = max((row.task_count for row in ordered), default=0)
    return tuple(
        ChartPoint(
            label=row.day.isoformat(),
            value_label=_format_number(row.task_count),
            percent=_percent_of(row.task_count, max_value),
            tone="danger" if row.failed_task_count else "accent",
            detail=(
                f"{_format_number(row.failed_task_count)} failed"
                if row.failed_task_count
                else "0 failed"
            ),
        )
        for row in ordered
    )


def _aggregate_bars(rows: Sequence[AggregateRow]) -> tuple[ChartBar, ...]:
    limited = tuple(rows[:8])
    max_value = max((row.cost_usd for row in limited), default=Decimal("0"))
    return tuple(
        ChartBar(
            label=row.display_key,
            secondary=row.secondary_key,
            value_label=_format_money(row.cost_usd),
            percent=_percent_of(row.cost_usd, max_value),
        )
        for row in limited
    )


def _percent_of(value: Decimal | int, max_value: Decimal | int) -> int:
    if value <= 0 or max_value <= 0:
        return 0
    percent = int((Decimal(value) / Decimal(max_value)) * 100)
    return max(4, min(100, percent))


def _format_money(value: Decimal) -> str:
    return f"${value:,.6f}"


def _format_number(value: int) -> str:
    return f"{value:,}"
