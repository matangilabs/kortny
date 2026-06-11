"""Natural-language schedule management commands."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sqlalchemy import Select, select
from sqlalchemy.orm import Session

from kortny.db.models import Schedule, Task, TaskEventType
from kortny.scheduler.creation import (
    DAYPART_HOURS,
    DEFAULT_TIMEZONE,
    PYTHON_WEEKDAYS,
    WEEKDAYS,
    ScheduleCreationContext,
    ScheduleDraft,
    parse_schedule_request,
)
from kortny.tasks import TaskService

SCHEDULE_ACTIVATED_MESSAGE = "schedule_activated"
SCHEDULE_CANCELLED_MESSAGE = "schedule_cancelled"
SCHEDULE_PAUSED_MESSAGE = "schedule_paused"
SCHEDULE_RESUMED_MESSAGE = "schedule_resumed"
SCHEDULE_UPDATED_MESSAGE = "schedule_updated"

MANAGEABLE_STATUSES = ("proposed", "active", "paused")
AMBIGUOUS_CONFIRMATION_WINDOW = timedelta(hours=24)

UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)
ACTIVATE_RE = re.compile(
    r"^\s*(yes|yep|yeah|confirm|confirmed|approve|approved|do it|go ahead|"
    r"set it up|activate)(?:[\s.!-]|$)",
    re.IGNORECASE,
)
EXPLICIT_ACTIVATE_RE = re.compile(
    r"\b(confirm|approve|activate|set up)\b.*\b(schedule|scheduled task)\b|"
    r"\b(schedule|scheduled task)\b.*\b(confirm|approve|activate)\b",
    re.IGNORECASE,
)
CANCEL_RE = re.compile(
    r"\b(cancel|delete|remove|stop)\b.*\b(schedule|scheduled task|that|this|it)\b|"
    r"\b(schedule|scheduled task)\b.*\b(cancel|delete|remove|stop)\b",
    re.IGNORECASE,
)
PAUSE_RE = re.compile(
    r"\b(pause|hold)\b.*\b(schedule|scheduled task|that|this|it)\b|"
    r"\b(schedule|scheduled task)\b.*\b(pause|hold)\b",
    re.IGNORECASE,
)
RESUME_RE = re.compile(
    r"\b(resume|unpause|restart)\b.*\b(schedule|scheduled task|that|this|it)\b|"
    r"\b(schedule|scheduled task)\b.*\b(resume|unpause|restart)\b",
    re.IGNORECASE,
)
EDIT_RE = re.compile(
    r"\b(make|change|move|update|edit)\b.*\b(schedule|scheduled task|that|"
    r"this|it|monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"daily|every|tomorrow|morning|afternoon|evening|night)\b",
    re.IGNORECASE,
)
WEEKDAY_EDIT_RE = re.compile(
    r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
    r"(?:\s+(morning|afternoon|evening|night))?\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class ScheduleCommand:
    """Parsed schedule command intent."""

    action: str
    ambiguous_confirmation: bool = False


@dataclass(frozen=True, slots=True)
class ScheduleCommandResult:
    """Executed schedule command result."""

    action: str
    schedule: Schedule
    response_text: str


class ScheduleCommandService:
    """Manage schedules through Slack-natural-language commands."""

    def __init__(
        self,
        session: Session,
        *,
        task_service: TaskService | None = None,
    ) -> None:
        self.session = session
        self.task_service = task_service or TaskService(session)

    def handle_text(
        self,
        *,
        task: Task,
        context: ScheduleCreationContext,
        text: str,
        now: datetime | None = None,
    ) -> ScheduleCommandResult | None:
        """Apply a schedule command when the text clearly refers to a schedule."""

        command = parse_schedule_command(text)
        if command is None:
            return None

        now = _coerce_utc(now or datetime.now(UTC))
        schedule = self._resolve_schedule(
            text=text,
            context=context,
            action=command.action,
            ambiguous_confirmation=command.ambiguous_confirmation,
            now=now,
        )
        if schedule is None:
            return None

        if command.action == "activate":
            return self._activate(
                task=task, schedule=schedule, context=context, now=now
            )
        if command.action == "cancel":
            return self._cancel(task=task, schedule=schedule, context=context, now=now)
        if command.action == "pause":
            return self._pause(task=task, schedule=schedule, context=context, now=now)
        if command.action == "resume":
            return self._resume(task=task, schedule=schedule, context=context, now=now)
        if command.action == "edit":
            return self._edit(
                task=task,
                schedule=schedule,
                context=context,
                text=text,
                now=now,
            )
        return None

    def _resolve_schedule(
        self,
        *,
        text: str,
        context: ScheduleCreationContext,
        action: str,
        ambiguous_confirmation: bool,
        now: datetime,
    ) -> Schedule | None:
        explicit_id = _schedule_id_from_text(text)
        if explicit_id is not None:
            return self.session.scalar(
                select(Schedule).where(
                    Schedule.id == explicit_id,
                    Schedule.installation_id == context.installation_id,
                    Schedule.owner_type == "user",
                    Schedule.owner_slack_user_id == context.slack_user_id,
                    Schedule.status.in_(MANAGEABLE_STATUSES),
                )
            )

        statuses = ("proposed",) if action == "activate" else MANAGEABLE_STATUSES
        statement: Select[tuple[Schedule]] = (
            select(Schedule)
            .where(
                Schedule.installation_id == context.installation_id,
                Schedule.owner_type == "user",
                Schedule.owner_slack_user_id == context.slack_user_id,
                Schedule.status.in_(statuses),
                Schedule.task_template["slack_channel_id"].as_string()
                == context.slack_channel_id,
            )
            .order_by(Schedule.created_at.desc(), Schedule.id.desc())
            .limit(1)
        )
        if context.delivery_surface != "dm":
            statement = statement.where(
                Schedule.task_template["slack_thread_ts"].as_string()
                == context.slack_thread_ts
            )
        if ambiguous_confirmation:
            statement = statement.where(
                Schedule.created_at >= now - AMBIGUOUS_CONFIRMATION_WINDOW
            )
        return self.session.scalar(statement)

    def _activate(
        self,
        *,
        task: Task,
        schedule: Schedule,
        context: ScheduleCreationContext,
        now: datetime,
    ) -> ScheduleCommandResult:
        previous_status = schedule.status
        if schedule.status == "proposed":
            schedule.status = "active"
        self._stamp_metadata(
            schedule,
            {
                "activated_at": now.isoformat(),
                "activated_by": context.slack_user_id,
                "activated_from_task_id": str(task.id),
            },
        )
        self._record_command_event(
            task=task,
            schedule=schedule,
            message=SCHEDULE_ACTIVATED_MESSAGE,
            action="activate",
            previous_status=previous_status,
            now=now,
        )
        _sync_witness_candidate(
            self.session,
            schedule,
            action="activate",
            by_user_id=context.slack_user_id,
            now=now,
        )
        return ScheduleCommandResult(
            action="activate",
            schedule=schedule,
            response_text=(
                "Done, I activated that scheduled task.\n\n"
                f"*Cadence:* {_cadence_label(schedule)}\n"
                f"*Next run:* {_next_run_label(schedule)}\n"
                f"*Delivery:* {_delivery_label(schedule)}"
            ),
        )

    def _cancel(
        self,
        *,
        task: Task,
        schedule: Schedule,
        context: ScheduleCreationContext,
        now: datetime,
    ) -> ScheduleCommandResult:
        previous_status = schedule.status
        schedule.status = "cancelled"
        schedule.next_run_at = None
        self._stamp_metadata(
            schedule,
            {
                "cancelled_at": now.isoformat(),
                "cancelled_by": context.slack_user_id,
                "cancelled_from_task_id": str(task.id),
            },
        )
        self._record_command_event(
            task=task,
            schedule=schedule,
            message=SCHEDULE_CANCELLED_MESSAGE,
            action="cancel",
            previous_status=previous_status,
            now=now,
        )
        _sync_witness_candidate(
            self.session,
            schedule,
            action="cancel",
            by_user_id=context.slack_user_id,
            now=now,
        )
        return ScheduleCommandResult(
            action="cancel",
            schedule=schedule,
            response_text="Cancelled that scheduled task. It will not run again.",
        )

    def _pause(
        self,
        *,
        task: Task,
        schedule: Schedule,
        context: ScheduleCreationContext,
        now: datetime,
    ) -> ScheduleCommandResult:
        previous_status = schedule.status
        if schedule.status == "active":
            schedule.status = "paused"
        self._stamp_metadata(
            schedule,
            {
                "paused_at": now.isoformat(),
                "paused_by": context.slack_user_id,
                "paused_from_task_id": str(task.id),
            },
        )
        self._record_command_event(
            task=task,
            schedule=schedule,
            message=SCHEDULE_PAUSED_MESSAGE,
            action="pause",
            previous_status=previous_status,
            now=now,
        )
        return ScheduleCommandResult(
            action="pause",
            schedule=schedule,
            response_text="Paused that scheduled task. It will not run until resumed.",
        )

    def _resume(
        self,
        *,
        task: Task,
        schedule: Schedule,
        context: ScheduleCreationContext,
        now: datetime,
    ) -> ScheduleCommandResult:
        previous_status = schedule.status
        if schedule.status == "paused":
            schedule.status = "active"
        self._stamp_metadata(
            schedule,
            {
                "resumed_at": now.isoformat(),
                "resumed_by": context.slack_user_id,
                "resumed_from_task_id": str(task.id),
            },
        )
        self._record_command_event(
            task=task,
            schedule=schedule,
            message=SCHEDULE_RESUMED_MESSAGE,
            action="resume",
            previous_status=previous_status,
            now=now,
        )
        return ScheduleCommandResult(
            action="resume",
            schedule=schedule,
            response_text=(
                "Resumed that scheduled task.\n\n"
                f"*Next run:* {_next_run_label(schedule)}"
            ),
        )

    def _edit(
        self,
        *,
        task: Task,
        schedule: Schedule,
        context: ScheduleCreationContext,
        text: str,
        now: datetime,
    ) -> ScheduleCommandResult | None:
        draft = parse_schedule_edit(text, schedule=schedule, now=now)
        if draft is None:
            return None

        previous_status = schedule.status
        schedule.title = draft.title
        schedule.spec_kind = draft.spec_kind
        schedule.cron_expr = draft.cron_expr
        schedule.interval_seconds = draft.interval_seconds
        schedule.run_at = draft.run_at
        schedule.timezone = draft.timezone
        schedule.next_run_at = draft.next_run_at
        template = dict(schedule.task_template or {})
        template["input"] = draft.task_input
        schedule.task_template = template
        self._stamp_metadata(
            schedule,
            {
                "cadence_label": draft.cadence_label,
                "edited_at": now.isoformat(),
                "edited_by": context.slack_user_id,
                "edited_from_task_id": str(task.id),
                "last_edit_input": text,
            },
        )
        self._record_command_event(
            task=task,
            schedule=schedule,
            message=SCHEDULE_UPDATED_MESSAGE,
            action="edit",
            previous_status=previous_status,
            now=now,
        )
        prefix = (
            "Updated the proposed schedule."
            if schedule.status == "proposed"
            else "Updated that scheduled task."
        )
        return ScheduleCommandResult(
            action="edit",
            schedule=schedule,
            response_text=(
                f"{prefix}\n\n"
                f"*Cadence:* {draft.cadence_label}\n"
                f"*Next run:* {draft.next_run_at.isoformat()} ({draft.timezone})"
            ),
        )

    def _record_command_event(
        self,
        *,
        task: Task,
        schedule: Schedule,
        message: str,
        action: str,
        previous_status: str,
        now: datetime,
    ) -> None:
        schedule.updated_at = now
        self.session.flush()
        self.task_service.append_event(
            task,
            TaskEventType.log,
            {
                "message": message,
                "schedule_id": str(schedule.id),
                "action": action,
                "from_status": previous_status,
                "to_status": schedule.status,
                "next_run_at": (
                    schedule.next_run_at.isoformat()
                    if schedule.next_run_at is not None
                    else None
                ),
                "cadence_label": _cadence_label(schedule),
            },
        )

    def _stamp_metadata(self, schedule: Schedule, values: dict[str, str]) -> None:
        schedule.metadata_json = {**dict(schedule.metadata_json or {}), **values}


def _sync_witness_candidate(
    session: Session,
    schedule: Schedule,
    *,
    action: str,
    by_user_id: str,
    now: datetime,
) -> None:
    """Link witness-drafted schedule confirmations back to their candidate.

    Imported lazily: kortny.witness.automation depends on this package, so a
    module-level import would be circular.
    """

    from kortny.witness.automation import sync_candidate_for_schedule_action

    sync_candidate_for_schedule_action(
        session,
        schedule,
        action=action,
        by_user_id=by_user_id,
        now=now,
    )


def parse_schedule_command(text: str) -> ScheduleCommand | None:
    """Classify a small set of schedule management commands."""

    if CANCEL_RE.search(text):
        return ScheduleCommand(action="cancel")
    if PAUSE_RE.search(text):
        return ScheduleCommand(action="pause")
    if RESUME_RE.search(text):
        return ScheduleCommand(action="resume")
    if EDIT_RE.search(text):
        return ScheduleCommand(action="edit")
    if EXPLICIT_ACTIVATE_RE.search(text):
        return ScheduleCommand(action="activate")
    if ACTIVATE_RE.search(text):
        return ScheduleCommand(action="activate", ambiguous_confirmation=True)
    return None


def parse_schedule_edit(
    text: str,
    *,
    schedule: Schedule,
    now: datetime,
) -> ScheduleDraft | None:
    """Parse cadence edits while preserving the existing task body by default."""

    preserved_input = _schedule_task_input(schedule)
    timezone = _schedule_timezone(schedule)
    full_draft = parse_schedule_request(text, now=now, timezone=timezone)
    if full_draft is not None:
        return ScheduleDraft(
            title=_schedule_title(schedule, preserved_input),
            spec_kind=full_draft.spec_kind,
            timezone=full_draft.timezone,
            next_run_at=full_draft.next_run_at,
            cadence_label=full_draft.cadence_label,
            task_input=preserved_input,
            cron_expr=full_draft.cron_expr,
            interval_seconds=full_draft.interval_seconds,
            run_at=full_draft.run_at,
            needs_confirmation=full_draft.needs_confirmation,
            parse_strategy="rules_edit",
        )

    weekday_match = WEEKDAY_EDIT_RE.search(text)
    if weekday_match is None:
        return None

    weekday = weekday_match.group(1).casefold()
    daypart = (weekday_match.group(2) or "morning").casefold()
    hour = DAYPART_HOURS[daypart]
    normalized_timezone, tzinfo = _timezone(timezone)
    next_run_at = _next_weekly(
        _coerce_utc(now).astimezone(tzinfo),
        weekday=weekday,
        hour=hour,
    )
    return ScheduleDraft(
        title=_schedule_title(schedule, preserved_input),
        spec_kind="cron",
        cron_expr=f"0 {hour} * * {WEEKDAYS[weekday]}",
        timezone=normalized_timezone,
        next_run_at=next_run_at,
        cadence_label=f"Every {weekday.title()} {daypart}",
        task_input=preserved_input,
        needs_confirmation=True,
        parse_strategy="rules_edit",
    )


def _schedule_id_from_text(text: str) -> uuid.UUID | None:
    match = UUID_RE.search(text)
    if match is None:
        return None
    try:
        return uuid.UUID(match.group(0))
    except ValueError:
        return None


def _schedule_task_input(schedule: Schedule) -> str:
    template = (
        schedule.task_template if isinstance(schedule.task_template, dict) else {}
    )
    value = template.get("input")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return schedule.title


def _schedule_title(schedule: Schedule, fallback_input: str) -> str:
    if schedule.title:
        return schedule.title
    return fallback_input[:80] or "Scheduled task"


def _schedule_timezone(schedule: Schedule) -> str:
    return schedule.timezone or DEFAULT_TIMEZONE


def _cadence_label(schedule: Schedule) -> str:
    metadata = (
        schedule.metadata_json if isinstance(schedule.metadata_json, dict) else {}
    )
    cadence = metadata.get("cadence_label")
    if isinstance(cadence, str) and cadence.strip():
        return cadence.strip()
    if schedule.spec_kind == "interval" and schedule.interval_seconds is not None:
        return f"Every {schedule.interval_seconds} seconds"
    if schedule.spec_kind == "oneoff":
        return "One-time"
    if schedule.cron_expr:
        return schedule.cron_expr
    return schedule.spec_kind


def _next_run_label(schedule: Schedule) -> str:
    if schedule.next_run_at is None:
        return "not scheduled"
    return f"{schedule.next_run_at.isoformat()} ({_schedule_timezone(schedule)})"


def _delivery_label(schedule: Schedule) -> str:
    template = (
        schedule.task_template if isinstance(schedule.task_template, dict) else {}
    )
    delivery_kind = getattr(schedule, "delivery_kind", None)
    if delivery_kind == "slack_dm":
        return "this DM"
    if delivery_kind == "slack_channel":
        return "this channel"
    if delivery_kind == "slack_thread":
        return "this thread"
    if delivery_kind == "dashboard_only":
        return "the dashboard"
    delivery_surface = template.get("delivery_surface")
    if delivery_surface == "dm":
        return "this DM"
    if delivery_surface == "channel":
        return "this channel"
    return "this thread"


def _timezone(value: str) -> tuple[str, ZoneInfo]:
    normalized = value.strip() or DEFAULT_TIMEZONE
    try:
        return normalized, ZoneInfo(normalized)
    except ZoneInfoNotFoundError:
        return DEFAULT_TIMEZONE, ZoneInfo(DEFAULT_TIMEZONE)


def _next_weekly(now: datetime, *, weekday: str, hour: int) -> datetime:
    target_weekday = PYTHON_WEEKDAYS[weekday]
    days_ahead = (target_weekday - now.weekday()) % 7
    target = (now + timedelta(days=days_ahead)).replace(
        hour=hour,
        minute=0,
        second=0,
        microsecond=0,
    )
    if target <= now:
        target += timedelta(days=7)
    return target.astimezone(UTC)


def _coerce_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
