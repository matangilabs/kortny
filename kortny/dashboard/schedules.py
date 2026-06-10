"""Dashboard read and action helpers for scheduled tasks."""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from kortny.db.models import Schedule, SlackIdentity, Task
from kortny.scheduler.creation import parse_schedule_request

SCHEDULE_PAGE_SIZE = 25
SCHEDULE_RUN_LIMIT = 10
SCHEDULE_STATUSES = {"all", "proposed", "active", "paused", "completed", "cancelled"}
SCHEDULE_VIEWS = {"all", "my", "system"}
SCHEDULE_ACTIONS = {"activate", "pause", "resume", "cancel"}
MANAGEABLE_SCHEDULE_STATUSES = {"proposed", "active", "paused"}
SCHEDULE_DELIVERY_OPTIONS = (
    ("slack_dm", "DM"),
    ("slack_thread", "Thread"),
    ("slack_channel", "Channel"),
    ("dashboard_only", "Dashboard"),
)
SCHEDULE_ARTIFACT_OPTIONS = (
    ("message_only", "Message only"),
    ("link_artifacts", "Link artifacts"),
    ("attach_files", "Attach files"),
)


@dataclass(frozen=True)
class ScheduleMetric:
    label: str
    value: str
    detail: str
    tone: str = "neutral"


@dataclass(frozen=True)
class ScheduleRow:
    schedule: Schedule
    cadence: str
    owner: str
    delivery: str
    next_run: str
    last_run: str
    budget: str
    tone: str
    can_activate: bool
    can_pause: bool
    can_resume: bool
    can_cancel: bool


@dataclass(frozen=True)
class ScheduleRunRow:
    task_id: uuid.UUID
    task_path: str
    task_input: str
    status: str
    tone: str
    created: str
    finished: str
    delivery: str
    cost: str
    tokens: str


@dataclass(frozen=True)
class ScheduleDetail:
    schedule: Schedule
    row: ScheduleRow
    runs: tuple[ScheduleRunRow, ...]
    health_notice: str | None
    schedule_text: str
    task_input: str
    budget_value: str
    delivery_options: tuple[tuple[str, str], ...]
    artifact_options: tuple[tuple[str, str], ...]
    can_edit: bool


@dataclass(frozen=True)
class SchedulePage:
    rows: tuple[ScheduleRow, ...]
    metrics: tuple[ScheduleMetric, ...]
    active_view: str
    status_filter: str
    page: int
    page_size: int
    total_count: int
    base_path: str
    previous_page_url: str | None
    next_page_url: str | None

    @property
    def total_pages(self) -> int:
        if self.total_count == 0:
            return 1
        return math.ceil(self.total_count / self.page_size)

    @property
    def first_item(self) -> int:
        if self.total_count == 0:
            return 0
        return ((self.page - 1) * self.page_size) + 1

    @property
    def last_item(self) -> int:
        return min(self.page * self.page_size, self.total_count)


def get_schedule_dashboard(
    session: Session,
    *,
    installation_id: uuid.UUID | None,
    slack_user_id: str | None,
    is_admin: bool,
    view: str = "all",
    status: str = "all",
    page: int = 1,
    base_path: str = "/schedules",
) -> SchedulePage:
    """Return a paginated schedule dashboard scoped to the current principal."""

    active_view = _normalize_view(view=view, is_admin=is_admin)
    status_filter = status if status in SCHEDULE_STATUSES else "all"
    normalized_page = max(page, 1)
    filters = _schedule_filters(
        installation_id=installation_id,
        slack_user_id=slack_user_id,
        is_admin=is_admin,
        view=active_view,
        status=status_filter,
    )
    total_count = (
        session.scalar(select(func.count()).select_from(Schedule).where(*filters)) or 0
    )
    schedules = tuple(
        session.scalars(
            select(Schedule)
            .where(*filters)
            .order_by(
                Schedule.next_run_at.asc().nulls_last(),
                Schedule.updated_at.desc(),
                Schedule.id.desc(),
            )
            .offset((normalized_page - 1) * SCHEDULE_PAGE_SIZE)
            .limit(SCHEDULE_PAGE_SIZE)
        )
    )
    identities = _owner_identity_map(session, schedules)
    page_model = SchedulePage(
        rows=tuple(
            _schedule_row(schedule, identities=identities) for schedule in schedules
        ),
        metrics=_schedule_metrics(
            session,
            installation_id=installation_id,
            slack_user_id=slack_user_id,
            is_admin=is_admin,
            view=active_view,
        ),
        active_view=active_view,
        status_filter=status_filter,
        page=normalized_page,
        page_size=SCHEDULE_PAGE_SIZE,
        total_count=total_count,
        base_path=base_path,
        previous_page_url=None,
        next_page_url=None,
    )
    return SchedulePage(
        **{
            **page_model.__dict__,
            "previous_page_url": _schedule_page_url(
                base_path=base_path,
                view=active_view,
                status=status_filter,
                page=normalized_page - 1,
            )
            if normalized_page > 1
            else None,
            "next_page_url": _schedule_page_url(
                base_path=base_path,
                view=active_view,
                status=status_filter,
                page=normalized_page + 1,
            )
            if normalized_page < page_model.total_pages
            else None,
        }
    )


def apply_schedule_action(
    session: Session,
    *,
    schedule_id: uuid.UUID,
    action: str,
    installation_id: uuid.UUID | None,
    slack_user_id: str | None,
    is_admin: bool,
    now: datetime | None = None,
) -> str:
    """Apply a dashboard schedule action and return a human notice."""

    if action not in SCHEDULE_ACTIONS:
        raise ValueError("Unsupported schedule action.")
    schedule = session.get(Schedule, schedule_id)
    if schedule is None:
        raise ValueError("Scheduled task was not found.")
    if not _can_access_schedule(
        schedule,
        installation_id=installation_id,
        slack_user_id=slack_user_id,
        is_admin=is_admin,
    ):
        raise ValueError("You do not have access to this scheduled task.")

    if action == "activate":
        if schedule.status != "proposed":
            raise ValueError("Only proposed schedules can be activated.")
        schedule.status = "active"
        notice = "Scheduled task activated."
    elif action == "pause":
        if schedule.status != "active":
            raise ValueError("Only active schedules can be paused.")
        schedule.status = "paused"
        notice = "Scheduled task paused."
    elif action == "resume":
        if schedule.status != "paused":
            raise ValueError("Only paused schedules can be resumed.")
        schedule.status = "active"
        notice = "Scheduled task resumed."
    else:
        if schedule.status not in {"proposed", "active", "paused"}:
            raise ValueError(
                "Only proposed, active, or paused schedules can be cancelled."
            )
        schedule.status = "cancelled"
        schedule.next_run_at = None
        notice = "Scheduled task cancelled."

    metadata = dict(schedule.metadata_json or {})
    metadata["dashboard_last_action"] = action
    metadata["dashboard_last_action_at"] = (now or datetime.now(UTC)).isoformat()
    schedule.metadata_json = metadata
    schedule.updated_at = now or datetime.now(UTC)
    session.add(schedule)
    session.commit()
    return notice


def get_schedule_detail(
    session: Session,
    *,
    schedule_id: uuid.UUID,
    installation_id: uuid.UUID | None,
    slack_user_id: str | None,
    is_admin: bool,
) -> ScheduleDetail:
    """Return a single schedule detail model scoped to the current principal."""

    schedule = _get_accessible_schedule(
        session,
        schedule_id=schedule_id,
        installation_id=installation_id,
        slack_user_id=slack_user_id,
        is_admin=is_admin,
    )
    identities = _owner_identity_map(session, (schedule,))
    return ScheduleDetail(
        schedule=schedule,
        row=_schedule_row(schedule, identities=identities),
        runs=_schedule_runs(session, schedule=schedule, is_admin=is_admin),
        health_notice=_schedule_health_notice(schedule),
        schedule_text=_cadence_label(schedule),
        task_input=_schedule_task_input(schedule),
        budget_value=_budget_form_value(schedule.planned_cost_ceiling_usd),
        delivery_options=SCHEDULE_DELIVERY_OPTIONS,
        artifact_options=SCHEDULE_ARTIFACT_OPTIONS,
        can_edit=schedule.status in MANAGEABLE_SCHEDULE_STATUSES,
    )


def update_schedule_from_dashboard(
    session: Session,
    *,
    schedule_id: uuid.UUID,
    installation_id: uuid.UUID | None,
    slack_user_id: str | None,
    is_admin: bool,
    actor: str,
    title: str,
    schedule_text: str,
    task_input: str,
    planned_cost_ceiling_usd: str,
    delivery_kind: str,
    delivery_slack_user_id: str,
    delivery_slack_channel_id: str,
    delivery_slack_thread_ts: str,
    artifact_delivery_policy: str,
    now: datetime | None = None,
) -> str:
    """Update a schedule from dashboard form values."""

    schedule = _get_accessible_schedule(
        session,
        schedule_id=schedule_id,
        installation_id=installation_id,
        slack_user_id=slack_user_id,
        is_admin=is_admin,
    )
    if schedule.status not in MANAGEABLE_SCHEDULE_STATUSES:
        raise ValueError("Only proposed, active, or paused schedules can be edited.")

    task_text = task_input.strip()
    if not task_text:
        raise ValueError("Task instructions are required.")
    cadence_text = schedule_text.strip()
    if not cadence_text:
        raise ValueError("Schedule timing is required.")

    edit_time = _coerce_utc(now or datetime.now(UTC))
    draft = parse_schedule_request(
        f"{cadence_text} {task_text}",
        now=edit_time,
        timezone=schedule.timezone or "UTC",
    )
    if draft is None:
        if cadence_text != _cadence_label(schedule):
            raise ValueError(
                "I could not parse that schedule timing. Try wording like "
                "'Every morning at 8 AM central time' or 'Every Monday morning'."
            )
        cadence_label = cadence_text
    else:
        cadence_label = draft.cadence_label

    cost_ceiling = _parse_budget(planned_cost_ceiling_usd)
    normalized_delivery_kind = delivery_kind.strip()
    if normalized_delivery_kind not in {
        key for key, _label in SCHEDULE_DELIVERY_OPTIONS
    }:
        raise ValueError("Delivery must be DM, thread, channel, or dashboard.")
    normalized_artifact_policy = artifact_delivery_policy.strip() or "message_only"
    if normalized_artifact_policy not in {
        key for key, _label in SCHEDULE_ARTIFACT_OPTIONS
    }:
        raise ValueError("Artifact policy is not supported.")

    target_user_id = (
        delivery_slack_user_id.strip() or schedule.owner_slack_user_id or ""
    )
    target_channel_id = delivery_slack_channel_id.strip()
    target_thread_ts = delivery_slack_thread_ts.strip() or None
    if not target_channel_id:
        raise ValueError("Slack channel or DM id is required for delivery.")
    if not target_user_id:
        raise ValueError("Slack user id is required for delivery.")
    if normalized_delivery_kind == "slack_channel":
        target_thread_ts = None
    elif normalized_delivery_kind == "slack_dm" and target_thread_ts is None:
        target_thread_ts = target_channel_id
    elif normalized_delivery_kind == "slack_thread" and target_thread_ts is None:
        raise ValueError("Thread delivery requires a Slack thread timestamp.")

    previous = _schedule_snapshot(schedule)
    schedule.title = title.strip() or (draft.title if draft is not None else task_text)
    if draft is not None:
        schedule.spec_kind = draft.spec_kind
        schedule.cron_expr = draft.cron_expr
        schedule.interval_seconds = draft.interval_seconds
        schedule.run_at = draft.run_at
        schedule.timezone = draft.timezone
        schedule.next_run_at = draft.next_run_at
    schedule.planned_cost_ceiling_usd = cost_ceiling
    schedule.delivery_kind = normalized_delivery_kind
    schedule.delivery_slack_user_id = target_user_id
    schedule.delivery_slack_channel_id = target_channel_id
    schedule.delivery_slack_thread_ts = target_thread_ts
    schedule.artifact_delivery_policy = normalized_artifact_policy
    schedule.task_template = {
        **dict(schedule.task_template or {}),
        "input": task_text,
        "slack_channel_id": target_channel_id,
        "slack_user_id": target_user_id,
        "slack_thread_ts": target_thread_ts,
        "delivery_surface": _legacy_delivery_surface(normalized_delivery_kind),
        "artifact_delivery_policy": normalized_artifact_policy,
    }
    schedule.metadata_json = _dashboard_edit_metadata(
        schedule,
        actor=actor,
        now=edit_time,
        cadence_label=cadence_label,
        previous=previous,
    )
    schedule.updated_at = edit_time
    session.add(schedule)
    session.commit()
    return "Scheduled task updated."


def _schedule_filters(
    *,
    installation_id: uuid.UUID | None,
    slack_user_id: str | None,
    is_admin: bool,
    view: str,
    status: str,
) -> list[Any]:
    filters: list[Any] = []
    if installation_id is not None:
        filters.append(Schedule.installation_id == installation_id)
    if not is_admin or view == "my":
        filters.extend(
            [
                Schedule.owner_type == "user",
                Schedule.owner_slack_user_id == slack_user_id,
            ]
        )
    elif view == "system":
        filters.append(Schedule.owner_type == "system")
    if status != "all":
        filters.append(Schedule.status == status)
    return filters


def _schedule_metrics(
    session: Session,
    *,
    installation_id: uuid.UUID | None,
    slack_user_id: str | None,
    is_admin: bool,
    view: str,
) -> tuple[ScheduleMetric, ...]:
    filters = _schedule_filters(
        installation_id=installation_id,
        slack_user_id=slack_user_id,
        is_admin=is_admin,
        view=view,
        status="all",
    )

    def count(status: str | None = None) -> int:
        status_filters = [*filters]
        if status is not None:
            status_filters.append(Schedule.status == status)
        return int(
            session.scalar(
                select(func.count()).select_from(Schedule).where(*status_filters)
            )
            or 0
        )

    return (
        ScheduleMetric(
            "Active", f"{count('active'):,}", "Running on schedule", "success"
        ),
        ScheduleMetric(
            "Paused", f"{count('paused'):,}", "Waiting to resume", "warning"
        ),
        ScheduleMetric(
            "Proposed", f"{count('proposed'):,}", "Drafts not yet active", "neutral"
        ),
        ScheduleMetric("Total", f"{count():,}", "All visible schedules", "neutral"),
    )


def _schedule_row(
    schedule: Schedule,
    *,
    identities: dict[tuple[uuid.UUID, str], str],
) -> ScheduleRow:
    return ScheduleRow(
        schedule=schedule,
        cadence=_cadence_label(schedule),
        owner=_owner_label(schedule, identities=identities),
        delivery=_delivery_label(schedule),
        next_run=_datetime_label(schedule.next_run_at),
        last_run=_datetime_label(schedule.last_run_at),
        budget=_budget_label(schedule.planned_cost_ceiling_usd),
        tone=_status_tone(schedule.status),
        can_activate=schedule.status == "proposed",
        can_pause=schedule.status == "active",
        can_resume=schedule.status == "paused",
        can_cancel=schedule.status in {"proposed", "active", "paused"},
    )


def _schedule_runs(
    session: Session,
    *,
    schedule: Schedule,
    is_admin: bool,
) -> tuple[ScheduleRunRow, ...]:
    tasks = tuple(
        session.scalars(
            select(Task)
            .where(
                Task.installation_id == schedule.installation_id,
                Task.identity_kind == "scheduled",
                Task.identity_payload["schedule_id"].as_string() == str(schedule.id),
            )
            .order_by(Task.created_at.desc(), Task.id.desc())
            .limit(SCHEDULE_RUN_LIMIT)
        )
    )
    if not tasks:
        return ()

    identities = _task_identity_map(session, tasks)
    task_base_path = "/tasks" if is_admin else "/me/tasks"
    return tuple(
        _schedule_run_row(
            task,
            identities=identities,
            task_base_path=task_base_path,
        )
        for task in tasks
    )


def _schedule_run_row(
    task: Task,
    *,
    identities: dict[tuple[str, uuid.UUID, str], str],
    task_base_path: str,
) -> ScheduleRunRow:
    status = _task_status_value(task)
    return ScheduleRunRow(
        task_id=task.id,
        task_path=f"{task_base_path}/{task.id}",
        task_input=task.input,
        status=status.replace("_", " "),
        tone=_task_status_tone(status),
        created=_datetime_label(task.created_at),
        finished=_task_finished_label(task),
        delivery=_task_delivery_label(task, identities=identities),
        cost=_budget_label(task.total_cost_usd),
        tokens=f"{task.total_input_tokens + task.total_output_tokens:,}",
    )


def _task_identity_map(
    session: Session,
    tasks: tuple[Task, ...],
) -> dict[tuple[str, uuid.UUID, str], str]:
    pairs: set[tuple[str, uuid.UUID, str]] = set()
    for task in tasks:
        payload = (
            task.identity_payload if isinstance(task.identity_payload, dict) else {}
        )
        user_id = payload.get("delivery_slack_user_id") or task.slack_user_id
        channel_id = payload.get("delivery_slack_channel_id") or task.slack_channel_id
        if isinstance(user_id, str) and user_id:
            pairs.add(("user", task.installation_id, user_id))
        if isinstance(channel_id, str) and channel_id:
            pairs.add(("channel", task.installation_id, channel_id))

    if not pairs:
        return {}
    installation_ids = {installation_id for _kind, installation_id, _slack_id in pairs}
    slack_ids = {slack_id for _kind, _installation_id, slack_id in pairs}
    kinds = {kind for kind, _installation_id, _slack_id in pairs}
    identities = session.scalars(
        select(SlackIdentity).where(
            SlackIdentity.kind.in_(kinds),
            SlackIdentity.installation_id.in_(installation_ids),
            SlackIdentity.slack_id.in_(slack_ids),
        )
    )
    return {
        (
            identity.kind,
            identity.installation_id,
            identity.slack_id,
        ): identity.display_name
        for identity in identities
        if (identity.kind, identity.installation_id, identity.slack_id) in pairs
    }


def _task_status_value(task: Task) -> str:
    value = task.status
    return str(getattr(value, "value", value))


def _task_status_tone(status: str) -> str:
    return {
        "succeeded": "success",
        "pending": "warning",
        "running": "warning",
        "waiting_approval": "warning",
        "failed": "danger",
        "crashed": "danger",
        "cancelled": "neutral",
    }.get(status, "neutral")


def _task_finished_label(task: Task) -> str:
    if task.finished_at is not None:
        return _datetime_label(task.finished_at)
    status = _task_status_value(task)
    if status == "pending":
        return "Queued"
    if status == "running":
        return "Running"
    if status == "waiting_approval":
        return "Waiting approval"
    return "Not finished"


def _task_delivery_label(
    task: Task,
    *,
    identities: dict[tuple[str, uuid.UUID, str], str],
) -> str:
    payload = task.identity_payload if isinstance(task.identity_payload, dict) else {}
    delivery_kind = payload.get("delivery_kind")
    user_id = (
        _optional_string(payload.get("delivery_slack_user_id")) or task.slack_user_id
    )
    channel_id = (
        _optional_string(payload.get("delivery_slack_channel_id"))
        or task.slack_channel_id
    )
    user_label = (
        identities.get(("user", task.installation_id, user_id)) if user_id else None
    ) or user_id
    channel_label = (
        identities.get(("channel", task.installation_id, channel_id))
        if channel_id
        else None
    ) or channel_id

    if delivery_kind == "slack_dm":
        return f"DM to {user_label}"
    if delivery_kind == "slack_channel":
        return f"Channel {channel_label}"
    if delivery_kind == "slack_thread":
        return f"Thread in {channel_label}"
    if delivery_kind == "dashboard_only":
        return "Dashboard only"
    return channel_label or "Unknown"


def _schedule_health_notice(schedule: Schedule) -> str | None:
    metadata = (
        schedule.metadata_json if isinstance(schedule.metadata_json, dict) else {}
    )
    scheduler_error = _optional_string(metadata.get("last_scheduler_error"))
    budget_status = _optional_string(metadata.get("last_budget_status"))
    if scheduler_error == "missing_planned_cost_ceiling":
        return (
            "This schedule is paused because it needs a per-run budget before "
            "the scheduler can materialize future runs."
        )
    if budget_status == "admission_failed":
        return (
            "This schedule is paused after a budget admission failure. Check the "
            "per-run budget before resuming it."
        )
    if scheduler_error:
        return f"Last scheduler issue: {scheduler_error.replace('_', ' ')}."
    return None


def _optional_string(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _can_access_schedule(
    schedule: Schedule,
    *,
    installation_id: uuid.UUID | None,
    slack_user_id: str | None,
    is_admin: bool,
) -> bool:
    if installation_id is not None and schedule.installation_id != installation_id:
        return False
    if is_admin:
        return True
    return (
        schedule.owner_type == "user" and schedule.owner_slack_user_id == slack_user_id
    )


def _get_accessible_schedule(
    session: Session,
    *,
    schedule_id: uuid.UUID,
    installation_id: uuid.UUID | None,
    slack_user_id: str | None,
    is_admin: bool,
) -> Schedule:
    schedule = session.get(Schedule, schedule_id)
    if schedule is None:
        raise LookupError("Scheduled task was not found.")
    if not _can_access_schedule(
        schedule,
        installation_id=installation_id,
        slack_user_id=slack_user_id,
        is_admin=is_admin,
    ):
        raise PermissionError("You do not have access to this scheduled task.")
    return schedule


def _normalize_view(*, view: str, is_admin: bool) -> str:
    if not is_admin:
        return "my"
    return view if view in SCHEDULE_VIEWS else "all"


def _cadence_label(schedule: Schedule) -> str:
    metadata = (
        schedule.metadata_json if isinstance(schedule.metadata_json, dict) else {}
    )
    label = metadata.get("cadence_label")
    if isinstance(label, str) and label.strip():
        return label
    if schedule.spec_kind == "oneoff":
        return "One-time"
    if schedule.spec_kind == "interval" and schedule.interval_seconds is not None:
        return f"Every {schedule.interval_seconds:,} seconds"
    if schedule.cron_expr:
        return schedule.cron_expr
    return schedule.spec_kind


def _owner_identity_map(
    session: Session,
    schedules: tuple[Schedule, ...],
) -> dict[tuple[uuid.UUID, str], str]:
    owner_pairs = {
        (schedule.installation_id, schedule.owner_slack_user_id)
        for schedule in schedules
        if schedule.owner_type == "user" and schedule.owner_slack_user_id
    }
    if not owner_pairs:
        return {}
    installation_ids = {installation_id for installation_id, _slack_id in owner_pairs}
    slack_ids = {slack_id for _installation_id, slack_id in owner_pairs}
    identities = session.scalars(
        select(SlackIdentity).where(
            SlackIdentity.kind == "user",
            SlackIdentity.installation_id.in_(installation_ids),
            SlackIdentity.slack_id.in_(slack_ids),
        )
    )
    return {
        (identity.installation_id, identity.slack_id): identity.display_name
        for identity in identities
    }


def _owner_label(
    schedule: Schedule,
    *,
    identities: dict[tuple[uuid.UUID, str], str],
) -> str:
    if schedule.owner_type == "system":
        return "System"
    if not schedule.owner_slack_user_id:
        return "User"
    return (
        identities.get((schedule.installation_id, schedule.owner_slack_user_id))
        or schedule.owner_slack_user_id
    )


def _delivery_label(schedule: Schedule) -> str:
    delivery_kind = getattr(schedule, "delivery_kind", None)
    if delivery_kind == "slack_dm":
        return _delivery_with_artifact_policy("DM", schedule)
    if delivery_kind == "slack_channel":
        return _delivery_with_artifact_policy("Channel", schedule)
    if delivery_kind == "slack_thread":
        return _delivery_with_artifact_policy("Thread", schedule)
    if delivery_kind == "dashboard_only":
        return _delivery_with_artifact_policy("Dashboard", schedule)
    template = (
        schedule.task_template if isinstance(schedule.task_template, dict) else {}
    )
    surface = template.get("delivery_surface")
    if surface == "dm":
        return _delivery_with_artifact_policy("DM", schedule)
    if surface == "channel":
        return _delivery_with_artifact_policy("Channel", schedule)
    if surface == "thread":
        return _delivery_with_artifact_policy("Thread", schedule)
    channel = template.get("slack_channel_id")
    return str(channel) if channel else "Unknown"


def _delivery_with_artifact_policy(label: str, schedule: Schedule) -> str:
    template = (
        schedule.task_template if isinstance(schedule.task_template, dict) else {}
    )
    policy = (
        getattr(schedule, "artifact_delivery_policy", None)
        or template.get("artifact_delivery_policy")
        or "message_only"
    )
    if policy == "attach_files":
        return f"{label} + files"
    if policy == "link_artifacts":
        return f"{label} + links"
    return label


def _schedule_task_input(schedule: Schedule) -> str:
    template = (
        schedule.task_template if isinstance(schedule.task_template, dict) else {}
    )
    value = template.get("input")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return schedule.title


def _datetime_label(value: datetime | None) -> str:
    if value is None:
        return "Not scheduled"
    return value.strftime("%Y-%m-%d %H:%M UTC")


def _budget_label(value: Decimal | None) -> str:
    if value is None:
        return "No cap"
    return f"${value:,.4f}"


def _budget_form_value(value: Decimal | None) -> str:
    if value is None:
        return ""
    return f"{value:.4f}"


def _parse_budget(value: str) -> Decimal:
    stripped = value.strip()
    if not stripped:
        raise ValueError("Per-run budget is required.")
    try:
        budget = Decimal(stripped)
    except InvalidOperation as exc:
        raise ValueError("Per-run budget must be a valid dollar amount.") from exc
    if budget <= 0:
        raise ValueError("Per-run budget must be greater than zero.")
    return budget.quantize(Decimal("0.0001"))


def _legacy_delivery_surface(delivery_kind: str) -> str:
    return {
        "slack_dm": "dm",
        "slack_thread": "thread",
        "slack_channel": "channel",
        "dashboard_only": "dashboard",
    }[delivery_kind]


def _schedule_snapshot(schedule: Schedule) -> dict[str, Any]:
    return {
        "title": schedule.title,
        "status": schedule.status,
        "spec_kind": schedule.spec_kind,
        "cron_expr": schedule.cron_expr,
        "interval_seconds": schedule.interval_seconds,
        "run_at": schedule.run_at.isoformat() if schedule.run_at else None,
        "timezone": schedule.timezone,
        "next_run_at": schedule.next_run_at.isoformat()
        if schedule.next_run_at
        else None,
        "delivery_kind": schedule.delivery_kind,
        "delivery_slack_user_id": schedule.delivery_slack_user_id,
        "delivery_slack_channel_id": schedule.delivery_slack_channel_id,
        "delivery_slack_thread_ts": schedule.delivery_slack_thread_ts,
        "artifact_delivery_policy": schedule.artifact_delivery_policy,
        "planned_cost_ceiling_usd": _decimal_string(schedule.planned_cost_ceiling_usd),
    }


def _dashboard_edit_metadata(
    schedule: Schedule,
    *,
    actor: str,
    now: datetime,
    cadence_label: str,
    previous: dict[str, Any],
) -> dict[str, Any]:
    metadata = dict(schedule.metadata_json or {})
    history = metadata.get("dashboard_edit_history")
    if not isinstance(history, list):
        history = []
    history.append(
        {
            "at": now.isoformat(),
            "by": actor,
            "previous": previous,
        }
    )
    metadata.update(
        {
            "cadence_label": cadence_label,
            "dashboard_edited_at": now.isoformat(),
            "dashboard_edited_by": actor,
            "dashboard_edit_history": history[-20:],
        }
    )
    return metadata


def _decimal_string(value: Decimal | None) -> str | None:
    if value is None:
        return None
    return str(value)


def _coerce_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _status_tone(status: str) -> str:
    return {
        "active": "success",
        "paused": "warning",
        "proposed": "neutral",
        "completed": "accent",
        "cancelled": "danger",
    }.get(status, "neutral")


def _schedule_page_url(
    *,
    base_path: str,
    view: str,
    status: str,
    page: int,
) -> str:
    params = {"view": view, "status": status, "page": str(page)}
    return f"{base_path}?{_query(params)}"


def _query(params: dict[str, str]) -> str:
    from urllib.parse import urlencode

    return urlencode({key: value for key, value in params.items() if value})
