"""Built-in schedule truth and management tools."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sqlalchemy import select
from sqlalchemy.orm import Session

from kortny.db.models import Schedule, Task, TaskEventType
from kortny.scheduler.commands import (
    SCHEDULE_CANCELLED_MESSAGE,
    SCHEDULE_PAUSED_MESSAGE,
    SCHEDULE_RESUMED_MESSAGE,
    SCHEDULE_UPDATED_MESSAGE,
    parse_schedule_edit,
)
from kortny.scheduler.creation import (
    DEFAULT_SCHEDULE_BUDGET_USD,
    DEFAULT_TIMEZONE,
    ScheduleCreationContext,
    ScheduleCreationService,
)
from kortny.tasks import TaskService
from kortny.tools.types import JsonObject, JsonSchema, RecoverableToolError, ToolResult

OPEN_SCHEDULE_STATUSES = ("proposed", "active", "paused")
ALL_SCHEDULE_STATUSES = ("proposed", "active", "paused", "completed", "cancelled")
DEFAULT_SCHEDULE_LIMIT = 10
MAX_SCHEDULE_LIMIT = 50

_MUTATION_PARAMETERS: JsonSchema = {
    "type": "object",
    "properties": {
        "schedule_id": {
            "type": "string",
            "description": "Exact schedule UUID when known.",
        },
        "query": {
            "type": "string",
            "description": "Fallback text to match schedule title or task body.",
        },
    },
    "additionalProperties": False,
}


class ListSchedulesTool:
    """List real schedules visible from the current Slack task."""

    name = "list_schedules"
    description = (
        "Lists real scheduled tasks from Kortny's scheduler database. Use this "
        "before answering whether a schedule exists, is active, is paused, where "
        "it will deliver, or when it will run next. Do not answer schedule-truth "
        "questions from memory, Slack history, or the workspace graph when this "
        "tool is available."
    )
    parameters: JsonSchema = {
        "type": "object",
        "properties": {
            "scope": {
                "type": "string",
                "enum": ["visible", "mine", "this_surface"],
                "description": (
                    "visible returns schedules the current user can see here; "
                    "mine returns schedules owned by the current Slack user; "
                    "this_surface returns schedules delivered to this DM, "
                    "channel, or thread."
                ),
            },
            "status": {
                "type": "string",
                "enum": ["open", "active", "paused", "proposed", "completed", "cancelled", "all"],
                "description": "Which schedule status set to return.",
            },
            "query": {
                "type": "string",
                "description": "Optional text to match schedule title or task body.",
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_SCHEDULE_LIMIT,
                "description": "Maximum schedules to return.",
            },
        },
        "additionalProperties": False,
    }

    def __init__(
        self,
        *,
        session: Session,
        task: Task,
    ) -> None:
        self.session = session
        self.task = task

    def invoke(self, args: JsonObject) -> ToolResult:
        scope = _choice(
            args.get("scope"),
            valid={"visible", "mine", "this_surface"},
            default="visible",
        )
        statuses = _status_filter(args.get("status"))
        query = _optional_string(args.get("query"))
        limit = _bounded_int(
            args.get("limit"), default=DEFAULT_SCHEDULE_LIMIT, minimum=1, maximum=MAX_SCHEDULE_LIMIT
        )
        schedules = _find_visible_schedules(
            self.session,
            task=self.task,
            scope=scope,
            statuses=statuses,
            query=query,
            limit=limit,
        )
        return ToolResult(
            output={
                "successful": True,
                "scope": scope,
                "status_filter": list(statuses),
                "query": query,
                "count": len(schedules),
                "assistant_summary": _list_summary(schedules, task=self.task),
                "schedules": [_schedule_payload(schedule, task=self.task) for schedule in schedules],
                "scope_note": (
                    "Only schedules owned by this Slack user or delivered to this "
                    "Slack surface are returned."
                ),
            }
        )


class GetScheduleTool:
    """Return one visible schedule by id or search text."""

    name = "get_schedule"
    description = (
        "Gets one real scheduled task from the scheduler database. Use this when "
        "the user asks about the state, cadence, destination, next run, or last "
        "run for a specific schedule."
    )
    parameters: JsonSchema = {
        "type": "object",
        "properties": {
            "schedule_id": {
                "type": "string",
                "description": "Exact schedule UUID when known.",
            },
            "query": {
                "type": "string",
                "description": "Fallback text to match schedule title or task body.",
            },
        },
        "additionalProperties": False,
    }

    def __init__(
        self,
        *,
        session: Session,
        task: Task,
    ) -> None:
        self.session = session
        self.task = task

    def invoke(self, args: JsonObject) -> ToolResult:
        schedule = _resolve_schedule_for_read(self.session, task=self.task, args=args)
        return ToolResult(
            output={
                "successful": True,
                "assistant_summary": _single_summary(schedule, task=self.task),
                "schedule": _schedule_payload(schedule, task=self.task),
            }
        )


class CreateScheduleTool:
    """Create a schedule using the existing Slack schedule creation service."""

    name = "create_schedule"
    description = (
        "Creates a real scheduled task in Kortny's scheduler database from the "
        "user's natural-language request. Use this only when the user clearly "
        "asked Kortny to schedule recurring or future work. New schedules default "
        "to active unless the user explicitly asked for confirmation first."
    )
    parameters: JsonSchema = {
        "type": "object",
        "properties": {
            "request": {
                "type": "string",
                "description": (
                    "The natural-language schedule request, including cadence, "
                    "time, timezone, task, and delivery wording when present."
                ),
            },
            "timezone": {
                "type": "string",
                "description": "Optional default IANA timezone, such as America/Chicago.",
            },
        },
        "required": ["request"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        *,
        session: Session,
        task: Task,
        task_service: TaskService | None = None,
    ) -> None:
        self.session = session
        self.task = task
        self.task_service = task_service or TaskService(session)

    def invoke(self, args: JsonObject) -> ToolResult:
        request = _required_string(args, "request")
        timezone = _optional_string(args.get("timezone")) or DEFAULT_TIMEZONE
        proposal = ScheduleCreationService(
            self.session,
            task_service=self.task_service,
        ).propose_from_text(
            task=self.task,
            context=_creation_context(self.task, timezone=timezone),
            text=request,
            now=datetime.now(UTC),
        )
        if proposal is None:
            raise RecoverableToolError(
                code="schedule_parse_failed",
                message="I could not turn that into a supported schedule.",
                hint=(
                    "Ask the user for a clearer cadence, time, timezone, and "
                    "what Kortny should do at each run."
                ),
            )
        return ToolResult(
            output={
                "successful": True,
                "action": "created",
                "assistant_summary": proposal.response_text,
                "schedule": _schedule_payload(proposal.schedule, task=self.task),
            }
        )


class PauseScheduleTool:
    """Pause a schedule owned by the current Slack user."""

    name = "pause_schedule"
    description = (
        "Pauses a real scheduled task. Use when the user asks Kortny to pause, "
        "hold, or stop an owned schedule temporarily."
    )
    parameters = _MUTATION_PARAMETERS

    def __init__(
        self,
        *,
        session: Session,
        task: Task,
        task_service: TaskService | None = None,
    ) -> None:
        self.session = session
        self.task = task
        self.task_service = task_service or TaskService(session)

    def invoke(self, args: JsonObject) -> ToolResult:
        schedule = _resolve_schedule_for_mutation(self.session, task=self.task, args=args)
        _ensure_schedule_status(
            schedule,
            allowed={"active", "paused", "proposed"},
            action="pause",
        )
        previous_status = schedule.status
        if schedule.status == "active":
            schedule.status = "paused"
        _stamp_schedule_action(
            self.session,
            task=self.task,
            task_service=self.task_service,
            schedule=schedule,
            message=SCHEDULE_PAUSED_MESSAGE,
            action="pause",
            previous_status=previous_status,
        )
        return ToolResult(
            output=_mutation_output(
                action="paused",
                schedule=schedule,
                task=self.task,
                assistant_summary=(
                    f"Done, I paused *{_schedule_title(schedule)}*. It will stay "
                    "quiet until you tell me to resume it."
                ),
            )
        )


class ResumeScheduleTool:
    """Resume a paused schedule owned by the current Slack user."""

    name = "resume_schedule"
    description = (
        "Resumes a paused scheduled task. Use when the user asks Kortny to "
        "resume, unpause, or restart an owned schedule."
    )
    parameters = _MUTATION_PARAMETERS

    def __init__(
        self,
        *,
        session: Session,
        task: Task,
        task_service: TaskService | None = None,
    ) -> None:
        self.session = session
        self.task = task
        self.task_service = task_service or TaskService(session)

    def invoke(self, args: JsonObject) -> ToolResult:
        schedule = _resolve_schedule_for_mutation(self.session, task=self.task, args=args)
        _ensure_schedule_status(
            schedule,
            allowed={"active", "paused", "proposed"},
            action="resume",
        )
        if schedule.next_run_at is None:
            raise RecoverableToolError(
                code="schedule_has_no_next_run",
                message="That schedule does not have a future run to resume.",
                hint="Create a new schedule or update this one with a future cadence.",
            )
        previous_status = schedule.status
        if schedule.status == "paused":
            schedule.status = "active"
        _stamp_schedule_action(
            self.session,
            task=self.task,
            task_service=self.task_service,
            schedule=schedule,
            message=SCHEDULE_RESUMED_MESSAGE,
            action="resume",
            previous_status=previous_status,
        )
        return ToolResult(
            output=_mutation_output(
                action="resumed",
                schedule=schedule,
                task=self.task,
                assistant_summary=(
                    f"Done, I resumed *{_schedule_title(schedule)}*. Next run is "
                    f"{_human_datetime(schedule.next_run_at, schedule.timezone)}."
                ),
            )
        )


class CancelScheduleTool:
    """Cancel a schedule owned by the current Slack user."""

    name = "cancel_schedule"
    description = (
        "Cancels a real scheduled task. Use when the user asks Kortny to cancel, "
        "delete, remove, or permanently stop an owned schedule."
    )
    parameters = _MUTATION_PARAMETERS

    def __init__(
        self,
        *,
        session: Session,
        task: Task,
        task_service: TaskService | None = None,
    ) -> None:
        self.session = session
        self.task = task
        self.task_service = task_service or TaskService(session)

    def invoke(self, args: JsonObject) -> ToolResult:
        schedule = _resolve_schedule_for_mutation(self.session, task=self.task, args=args)
        _ensure_schedule_status(
            schedule,
            allowed={"active", "paused", "proposed"},
            action="cancel",
        )
        previous_status = schedule.status
        schedule.status = "cancelled"
        schedule.next_run_at = None
        _stamp_schedule_action(
            self.session,
            task=self.task,
            task_service=self.task_service,
            schedule=schedule,
            message=SCHEDULE_CANCELLED_MESSAGE,
            action="cancel",
            previous_status=previous_status,
        )
        return ToolResult(
            output=_mutation_output(
                action="cancelled",
                schedule=schedule,
                task=self.task,
                assistant_summary=(
                    f"Cancelled *{_schedule_title(schedule)}*. It will not run again."
                ),
            )
        )


class UpdateScheduleTool:
    """Edit cadence/task details for an owned schedule."""

    name = "update_schedule"
    description = (
        "Updates an owned scheduled task using natural language, such as changing "
        "it to weekdays at 8 AM Central. Use when the user asks to change or edit "
        "an existing schedule."
    )
    parameters: JsonSchema = {
        "type": "object",
        "properties": {
            "schedule_id": {
                "type": "string",
                "description": "Exact schedule UUID when known.",
            },
            "query": {
                "type": "string",
                "description": "Fallback text to match schedule title or task body.",
            },
            "update_request": {
                "type": "string",
                "description": "Natural-language change request for the schedule.",
            },
        },
        "required": ["update_request"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        *,
        session: Session,
        task: Task,
        task_service: TaskService | None = None,
    ) -> None:
        self.session = session
        self.task = task
        self.task_service = task_service or TaskService(session)

    def invoke(self, args: JsonObject) -> ToolResult:
        update_request = _required_string(args, "update_request")
        schedule = _resolve_schedule_for_mutation(self.session, task=self.task, args=args)
        _ensure_schedule_status(
            schedule,
            allowed={"active", "paused", "proposed"},
            action="update",
        )
        draft = parse_schedule_edit(
            update_request,
            schedule=schedule,
            now=datetime.now(UTC),
        )
        if draft is None:
            raise RecoverableToolError(
                code="schedule_update_parse_failed",
                message="I could not safely understand that schedule change.",
                hint="Ask for a clearer cadence, day, time, or timezone.",
            )
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
        schedule.metadata_json = {
            **dict(schedule.metadata_json or {}),
            "cadence_label": draft.cadence_label,
            "edited_at": datetime.now(UTC).isoformat(),
            "edited_by": self.task.slack_user_id,
            "edited_from_task_id": str(self.task.id),
            "last_edit_input": update_request,
            "parse_strategy": draft.parse_strategy,
        }
        _stamp_schedule_action(
            self.session,
            task=self.task,
            task_service=self.task_service,
            schedule=schedule,
            message=SCHEDULE_UPDATED_MESSAGE,
            action="update",
            previous_status=previous_status,
        )
        return ToolResult(
            output=_mutation_output(
                action="updated",
                schedule=schedule,
                task=self.task,
                assistant_summary=(
                    f"Done, I updated *{_schedule_title(schedule)}*. It now runs "
                    f"{_cadence_label(schedule).lower()}; next run is "
                    f"{_human_datetime(schedule.next_run_at, schedule.timezone)}."
                ),
            )
        )


def _find_visible_schedules(
    session: Session,
    *,
    task: Task,
    scope: str,
    statuses: tuple[str, ...],
    query: str | None,
    limit: int,
) -> tuple[Schedule, ...]:
    rows = tuple(
        session.scalars(
            select(Schedule)
            .where(
                Schedule.installation_id == task.installation_id,
                Schedule.status.in_(statuses),
            )
            .order_by(
                Schedule.status.asc(),
                Schedule.next_run_at.asc().nullslast(),
                Schedule.created_at.desc(),
            )
            .limit(max(limit * 4, 25))
        )
    )
    visible = [
        schedule
        for schedule in rows
        if _matches_scope(schedule, task=task, scope=scope)
        and _matches_query(schedule, query)
    ]
    return tuple(visible[:limit])


def _resolve_schedule_for_read(
    session: Session,
    *,
    task: Task,
    args: JsonObject,
) -> Schedule:
    return _resolve_schedule(session, task=task, args=args, require_owner=False)


def _resolve_schedule_for_mutation(
    session: Session,
    *,
    task: Task,
    args: JsonObject,
) -> Schedule:
    return _resolve_schedule(session, task=task, args=args, require_owner=True)


def _resolve_schedule(
    session: Session,
    *,
    task: Task,
    args: JsonObject,
    require_owner: bool,
) -> Schedule:
    schedule_id = _optional_string(args.get("schedule_id"))
    query = _optional_string(args.get("query"))
    if schedule_id:
        try:
            parsed_id = uuid.UUID(schedule_id)
        except ValueError as exc:
            raise RecoverableToolError(
                code="invalid_schedule_id",
                message="That schedule id is not a valid UUID.",
                hint="Call list_schedules first if you do not know the exact id.",
            ) from exc
        schedule = session.get(Schedule, parsed_id)
        if schedule is None or schedule.installation_id != task.installation_id:
            raise RecoverableToolError(
                code="schedule_not_found",
                message="I could not find that schedule.",
                hint="Call list_schedules to see the schedules visible here.",
            )
        if not _is_visible(schedule, task=task):
            raise RecoverableToolError(
                code="schedule_not_visible",
                message="That schedule is not visible from this Slack surface.",
                hint="Ask from the owning DM/thread or use a schedule you can see here.",
            )
        if require_owner and not _is_owner(schedule, task=task):
            raise RecoverableToolError(
                code="schedule_not_owned",
                message="I can see that schedule, but I can't change it for you.",
                hint="Only the Slack user who owns a schedule can pause, resume, edit, or cancel it.",
            )
        return schedule

    if not query:
        raise RecoverableToolError(
            code="schedule_reference_missing",
            message="I need to know which schedule you mean.",
            hint="Call list_schedules first, then use schedule_id or a specific query.",
        )

    matches = _find_visible_schedules(
        session,
        task=task,
        scope="mine" if require_owner else "visible",
        statuses=OPEN_SCHEDULE_STATUSES,
        query=query,
        limit=5,
    )
    if not matches:
        raise RecoverableToolError(
            code="schedule_not_found",
            message="I did not find a matching active, paused, or proposed schedule.",
            hint="Call list_schedules with status=all if the schedule may be completed or cancelled.",
        )
    if len(matches) > 1:
        raise RecoverableToolError(
            code="schedule_reference_ambiguous",
            message="I found more than one matching schedule.",
            hint="Ask the user which one, or call get_schedule with a specific schedule_id.",
            details={"matches": [_schedule_choice(schedule) for schedule in matches]},
        )
    return matches[0]


def _ensure_schedule_status(
    schedule: Schedule,
    *,
    allowed: set[str],
    action: str,
) -> None:
    if schedule.status in allowed:
        return
    raise RecoverableToolError(
        code="schedule_not_mutable",
        message=f"I can't {action} a schedule that is already {schedule.status}.",
        hint="List active or paused schedules and choose one that is still open.",
    )


def _stamp_schedule_action(
    session: Session,
    *,
    task: Task,
    task_service: TaskService,
    schedule: Schedule,
    message: str,
    action: str,
    previous_status: str,
) -> None:
    now = datetime.now(UTC)
    schedule.updated_at = now
    metadata = dict(schedule.metadata_json or {})
    past_tense = _action_past_tense(action)
    metadata[f"{past_tense}_at"] = now.isoformat()
    metadata[f"{past_tense}_by"] = task.slack_user_id
    metadata[f"{past_tense}_from_task_id"] = str(task.id)
    schedule.metadata_json = metadata
    session.flush()
    task_service.append_event(
        task,
        TaskEventType.log,
        {
            "message": message,
            "schedule_id": str(schedule.id),
            "action": action,
            "from_status": previous_status,
            "to_status": schedule.status,
            "next_run_at": schedule.next_run_at.isoformat()
            if schedule.next_run_at is not None
            else None,
            "cadence_label": _cadence_label(schedule),
        },
    )


def _mutation_output(
    *,
    action: str,
    schedule: Schedule,
    task: Task,
    assistant_summary: str,
) -> JsonObject:
    return {
        "successful": True,
        "action": action,
        "assistant_summary": assistant_summary,
        "schedule": _schedule_payload(schedule, task=task),
    }


def _schedule_payload(schedule: Schedule, *, task: Task) -> JsonObject:
    return {
        "id": str(schedule.id),
        "title": schedule.title,
        "status": schedule.status,
        "owner": {
            "type": schedule.owner_type,
            "slack_user_id": schedule.owner_slack_user_id,
            "is_current_user": _is_owner(schedule, task=task),
        },
        "cadence": {
            "label": _cadence_label(schedule),
            "spec_kind": schedule.spec_kind,
            "cron_expr": schedule.cron_expr,
            "interval_seconds": schedule.interval_seconds,
            "run_at": _iso(schedule.run_at),
            "timezone": schedule.timezone,
        },
        "next_run_at": _iso(schedule.next_run_at),
        "next_run_human": _human_datetime(schedule.next_run_at, schedule.timezone),
        "last_run_at": _iso(schedule.last_run_at),
        "last_run_human": _human_datetime(schedule.last_run_at, schedule.timezone),
        "delivery": {
            "kind": schedule.delivery_kind,
            "label": _delivery_label(schedule, task=task),
            "slack_user_id": schedule.delivery_slack_user_id,
            "slack_channel_id": schedule.delivery_slack_channel_id,
            "slack_thread_ts": schedule.delivery_slack_thread_ts,
            "artifact_policy": schedule.artifact_delivery_policy,
        },
        "task": {
            "input": _schedule_task_input(schedule),
        },
        "budget": {
            "planned_cost_ceiling_usd": _decimal_string(
                schedule.planned_cost_ceiling_usd
            ),
            "default_cost_ceiling_usd": str(DEFAULT_SCHEDULE_BUDGET_USD),
        },
        "created_at": _iso(schedule.created_at),
        "updated_at": _iso(schedule.updated_at),
        "can_manage": _is_owner(schedule, task=task),
    }


def _schedule_choice(schedule: Schedule) -> JsonObject:
    return {
        "id": str(schedule.id),
        "title": schedule.title,
        "status": schedule.status,
        "cadence": _cadence_label(schedule),
        "next_run_at": _iso(schedule.next_run_at),
    }


def _list_summary(schedules: tuple[Schedule, ...], *, task: Task) -> str:
    if not schedules:
        return "I checked the scheduler and don't see any matching schedules."
    active = [schedule for schedule in schedules if schedule.status == "active"]
    paused = [schedule for schedule in schedules if schedule.status == "paused"]
    proposed = [schedule for schedule in schedules if schedule.status == "proposed"]
    lead = f"I found {len(schedules)} schedule{'s' if len(schedules) != 1 else ''}"
    details: list[str] = []
    if active:
        details.append(f"{len(active)} active")
    if paused:
        details.append(f"{len(paused)} paused")
    if proposed:
        details.append(f"{len(proposed)} waiting")
    if details:
        lead += f" ({', '.join(details)})"
    first = schedules[0]
    return (
        f"{lead}. Closest next run: *{_schedule_title(first)}* "
        f"{_next_run_phrase(first)}; delivery is {_delivery_label(first, task=task)}."
    )


def _single_summary(schedule: Schedule, *, task: Task) -> str:
    status = {
        "active": "active",
        "paused": "paused",
        "proposed": "waiting for confirmation",
        "completed": "completed",
        "cancelled": "cancelled",
    }.get(schedule.status, schedule.status)
    return (
        f"*{_schedule_title(schedule)}* is {status}. It runs "
        f"{_cadence_label(schedule).lower()}; {_next_run_phrase(schedule)}. "
        f"Delivery is {_delivery_label(schedule, task=task)}."
    )


def _matches_scope(schedule: Schedule, *, task: Task, scope: str) -> bool:
    if scope == "mine":
        return _is_owner(schedule, task=task)
    if scope == "this_surface":
        return _delivers_to_current_surface(schedule, task=task)
    return _is_visible(schedule, task=task)


def _is_visible(schedule: Schedule, *, task: Task) -> bool:
    return _is_owner(schedule, task=task) or _delivers_to_current_surface(
        schedule, task=task
    )


def _is_owner(schedule: Schedule, *, task: Task) -> bool:
    return (
        schedule.owner_type == "user"
        and schedule.owner_slack_user_id is not None
        and schedule.owner_slack_user_id == task.slack_user_id
    )


def _delivers_to_current_surface(schedule: Schedule, *, task: Task) -> bool:
    if schedule.delivery_slack_channel_id != task.slack_channel_id:
        return False
    if task.slack_channel_id.startswith("D"):
        return schedule.delivery_slack_user_id == task.slack_user_id
    if schedule.delivery_kind == "slack_thread":
        return schedule.delivery_slack_thread_ts == task.slack_thread_ts
    return schedule.delivery_kind == "slack_channel"


def _matches_query(schedule: Schedule, query: str | None) -> bool:
    if query is None:
        return True
    haystack = " ".join(
        part
        for part in (
            schedule.title,
            _schedule_task_input(schedule),
            _cadence_label(schedule),
            schedule.status,
        )
        if part
    ).casefold()
    return query.casefold() in haystack


def _creation_context(task: Task, *, timezone: str) -> ScheduleCreationContext:
    return ScheduleCreationContext(
        installation_id=task.installation_id,
        slack_channel_id=task.slack_channel_id,
        slack_user_id=task.slack_user_id,
        slack_thread_ts=_thread_ts_for_task(task),
        source_surface="dm" if task.slack_channel_id.startswith("D") else "channel",
        source_task_id=task.id,
        timezone=timezone,
    )


def _thread_ts_for_task(task: Task) -> str:
    if task.slack_thread_ts:
        return task.slack_thread_ts
    if task.slack_message_ts:
        return task.slack_message_ts
    if task.slack_channel_id.startswith("D"):
        return task.slack_channel_id
    return task.slack_channel_id


def _schedule_title(schedule: Schedule) -> str:
    return schedule.title or _schedule_task_input(schedule) or "Scheduled task"


def _schedule_task_input(schedule: Schedule) -> str:
    template = schedule.task_template if isinstance(schedule.task_template, dict) else {}
    value = template.get("input")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return schedule.title


def _cadence_label(schedule: Schedule) -> str:
    metadata = schedule.metadata_json if isinstance(schedule.metadata_json, dict) else {}
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


def _next_run_phrase(schedule: Schedule) -> str:
    if schedule.next_run_at is None:
        return "no future run is scheduled"
    return f"next run is {_human_datetime(schedule.next_run_at, schedule.timezone)}"


def _delivery_label(schedule: Schedule, *, task: Task) -> str:
    if schedule.delivery_kind == "slack_dm":
        return (
            "this DM"
            if schedule.delivery_slack_channel_id == task.slack_channel_id
            else "a DM"
        )
    if schedule.delivery_kind == "slack_channel":
        return (
            "this channel"
            if schedule.delivery_slack_channel_id == task.slack_channel_id
            else "a channel"
        )
    if schedule.delivery_kind == "slack_thread":
        return (
            "this thread"
            if schedule.delivery_slack_thread_ts == task.slack_thread_ts
            else "a thread"
        )
    if schedule.delivery_kind == "dashboard_only":
        return "the dashboard"
    return schedule.delivery_kind


def _human_datetime(value: datetime | None, timezone: str | None) -> str | None:
    if value is None:
        return None
    normalized_timezone, tzinfo = _timezone(timezone or DEFAULT_TIMEZONE)
    local = _coerce_utc(value).astimezone(tzinfo)
    today = datetime.now(UTC).astimezone(tzinfo).date()
    time_text = _format_time(hour=local.hour, minute=local.minute)
    timezone_label = _timezone_label(normalized_timezone)
    if local.date() == today:
        return f"today at {time_text} {timezone_label}"
    if local.date() == today + timedelta(days=1):
        return f"tomorrow at {time_text} {timezone_label}"
    return f"{local.strftime('%A, %b')} {local.day} at {time_text} {timezone_label}"


def _format_time(*, hour: int, minute: int) -> str:
    meridiem = "AM" if hour < 12 else "PM"
    display_hour = hour % 12 or 12
    return f"{display_hour}:{minute:02d} {meridiem}"


def _timezone(value: str) -> tuple[str, ZoneInfo]:
    normalized = value.strip() or DEFAULT_TIMEZONE
    try:
        return normalized, ZoneInfo(normalized)
    except ZoneInfoNotFoundError:
        return DEFAULT_TIMEZONE, ZoneInfo(DEFAULT_TIMEZONE)


def _timezone_label(timezone: str) -> str:
    return {
        "America/Chicago": "Central",
        "America/New_York": "Eastern",
        "America/Los_Angeles": "Pacific",
        "America/Denver": "Mountain",
        "UTC": "UTC",
    }.get(timezone, timezone)


def _status_filter(value: object) -> tuple[str, ...]:
    status = _choice(
        value,
        valid={"open", "active", "paused", "proposed", "completed", "cancelled", "all"},
        default="open",
    )
    if status == "open":
        return OPEN_SCHEDULE_STATUSES
    if status == "all":
        return ALL_SCHEDULE_STATUSES
    return (status,)


def _action_past_tense(action: str) -> str:
    return {
        "pause": "paused",
        "resume": "resumed",
        "cancel": "cancelled",
        "update": "updated",
    }.get(action, f"{action}d")


def _choice(value: object, *, valid: set[str], default: str) -> str:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in valid:
            return normalized
    return default


def _required_string(args: JsonObject, key: str) -> str:
    value = args.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Expected non-empty string '{key}'")
    return " ".join(value.split()).strip()


def _optional_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = " ".join(value.split()).strip()
    return normalized or None


def _bounded_int(
    value: object,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    try:
        parsed = int(str(value))
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _coerce_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _iso(value: datetime | None) -> str | None:
    return _coerce_utc(value).isoformat() if value is not None else None


def _decimal_string(value: Decimal | None) -> str | None:
    if value is None:
        return None
    try:
        return str(Decimal(value))
    except (InvalidOperation, ValueError, TypeError):
        return str(value)
