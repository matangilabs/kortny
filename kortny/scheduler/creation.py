"""Schedule creation and proposal contracts."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Protocol
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sqlalchemy.orm import Session

from kortny.db.models import Schedule, Task, TaskEventType
from kortny.tasks import TaskService

SCHEDULE_CREATED_MESSAGE = "schedule_created"
SCHEDULE_PROPOSAL_CREATED_MESSAGE = "schedule_proposal_created"

DEFAULT_SCHEDULE_BUDGET_USD = Decimal("0.2500")
DEFAULT_CATCHUP_WINDOW_SECONDS = 300
DEFAULT_TIMEZONE = "UTC"

WEEKDAYS = {
    "sunday": 0,
    "monday": 1,
    "tuesday": 2,
    "wednesday": 3,
    "thursday": 4,
    "friday": 5,
    "saturday": 6,
}
PYTHON_WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}
DAYPART_HOURS = {
    "morning": 9,
    "afternoon": 13,
    "evening": 17,
    "night": 20,
}
INTERVAL_UNITS = {
    "minute": 60,
    "minutes": 60,
    "hour": 3600,
    "hours": 3600,
    "day": 86400,
    "days": 86400,
    "week": 604800,
    "weeks": 604800,
}
TIMEZONE_ALIASES = (
    (
        "America/Chicago",
        "Central time",
        re.compile(r"\b(?:central|ct|cst|cdt)\s*(?:time)?\b", re.IGNORECASE),
    ),
    (
        "America/New_York",
        "Eastern time",
        re.compile(r"\b(?:eastern|et|est|edt)\s*(?:time)?\b", re.IGNORECASE),
    ),
    (
        "America/Los_Angeles",
        "Pacific time",
        re.compile(r"\b(?:pacific|pt|pst|pdt)\s*(?:time)?\b", re.IGNORECASE),
    ),
    (
        "America/Denver",
        "Mountain time",
        re.compile(r"\b(?:mountain|mt|mst|mdt)\s*(?:time)?\b", re.IGNORECASE),
    ),
    ("UTC", "UTC", re.compile(r"\b(?:utc|gmt)\b", re.IGNORECASE)),
)

SCHEDULE_RE = re.compile(
    r"\b("
    r"every|daily|weekly|monthly|schedule|scheduled|recurring|remind|reminder|"
    r"tomorrow|later|weekday|weekdays?|business\s+days?|trading\s+days?|"
    r"market\s+(?:open|close)|twice\s+a\s+day|"
    r"in\s+\d+\s+(?:minutes?|hours?|days?|weeks?)"
    r")\b",
    re.IGNORECASE,
)
CONFIRMATION_REQUEST_RE = re.compile(
    r"\b(propose|draft|ask me|confirm with me|for approval|before you run|"
    r"before running|do not run|don't run|wait for confirmation)\b",
    re.IGNORECASE,
)
WEEKLY_RE = re.compile(
    r"\bevery\s+"
    r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
    r"(?:\s+(morning|afternoon|evening|night))?\b",
    re.IGNORECASE,
)
DAILY_RE = re.compile(
    r"\b(?:every\s+day|daily|every\s+(morning|afternoon|evening|night))\b",
    re.IGNORECASE,
)
INTERVAL_RE = re.compile(
    r"\bevery\s+(\d+)\s+(minutes?|hours?|days?|weeks?)\b",
    re.IGNORECASE,
)
IN_RE = re.compile(
    r"\bin\s+(\d+)\s+(minutes?|hours?|days?|weeks?)\b",
    re.IGNORECASE,
)
TOMORROW_RE = re.compile(
    r"\btomorrow(?:\s+(morning|afternoon|evening|night))?\b",
    re.IGNORECASE,
)
EXPLICIT_TIME_RE = re.compile(
    r"\b(?:at|around|by)?\s*(\d{1,2})(?::(\d{2}))?\s*"
    r"(a\.?m\.?|p\.?m\.?)\b",
    re.IGNORECASE,
)
LEADING_TIME_RE = re.compile(
    r"^\s*(?:at|around|by)?\s*\d{1,2}(?::\d{2})?\s*"
    r"(?:a\.?m\.?|p\.?m\.?)\s*"
    r"(?:(?:central|eastern|pacific|mountain|utc|gmt|ct|et|pt|mt|cst|cdt|"
    r"est|edt|pst|pdt|mst|mdt)\s*(?:time)?\s*)?"
    r",?\s*",
    re.IGNORECASE,
)
LEADING_CADENCE_RE = re.compile(
    r"^\s*(?:"
    r"every\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
    r"(?:\s+(?:morning|afternoon|evening|night))?|"
    r"every\s+(?:morning|afternoon|evening|night)|"
    r"every\s+day|"
    r"daily|"
    r"tomorrow(?:\s+(?:morning|afternoon|evening|night))?|"
    r"in\s+\d+\s+(?:minutes?|hours?|days?|weeks?)|"
    r"every\s+\d+\s+(?:minutes?|hours?|days?|weeks?)"
    r")\s*,?\s*",
    re.IGNORECASE,
)
THREAD_DELIVERY_RE = re.compile(
    r"\b(this thread|same thread|reply here|reply in this thread|"
    r"in this thread|keep it here)\b",
    re.IGNORECASE,
)
CHANNEL_DELIVERY_RE = re.compile(
    r"\b(?:post|send|deliver|share)\b.{0,40}\b(?:here|this channel|the channel)\b|"
    r"\b(?:in|to)\s+this\s+channel\b",
    re.IGNORECASE,
)
ATTACH_ARTIFACTS_RE = re.compile(
    r"\b(attach|upload|send)\b.{0,32}\b(files?|artifacts?|pdfs?|reports?)\b",
    re.IGNORECASE,
)
LINK_ARTIFACTS_RE = re.compile(
    r"\b(link|links only|artifact links?|report links?)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class ScheduleDraft:
    """Parsed schedule fields ready for persistence."""

    title: str
    spec_kind: str
    timezone: str
    next_run_at: datetime
    cadence_label: str
    task_input: str
    cron_expr: str | None = None
    interval_seconds: int | None = None
    run_at: datetime | None = None
    needs_confirmation: bool = False
    parse_strategy: str = "rules"


@dataclass(frozen=True, slots=True)
class ScheduleProposal:
    """Created schedule proposal and Slack-facing copy."""

    schedule: Schedule
    draft: ScheduleDraft
    response_text: str


@dataclass(frozen=True, slots=True)
class ScheduleDelivery:
    """Explicit delivery destination for a scheduled run."""

    kind: str
    slack_user_id: str | None
    slack_channel_id: str | None
    slack_thread_ts: str | None
    artifact_policy: str = "message_only"

    @property
    def legacy_surface(self) -> str:
        return {
            "slack_dm": "dm",
            "slack_channel": "channel",
            "slack_thread": "thread",
            "dashboard_only": "dashboard",
        }[self.kind]

    @property
    def response_label(self) -> str:
        return {
            "slack_dm": "in this DM",
            "slack_channel": "in this channel",
            "slack_thread": "in this thread",
            "dashboard_only": "in the dashboard",
        }[self.kind]


@dataclass(frozen=True, slots=True)
class ScheduleCreationContext:
    """Slack context needed to propose a schedule."""

    installation_id: uuid.UUID
    slack_channel_id: str
    slack_user_id: str
    slack_thread_ts: str
    source_surface: str
    source_task_id: uuid.UUID
    timezone: str = DEFAULT_TIMEZONE

    @property
    def delivery_surface(self) -> str:
        return infer_schedule_delivery(context=self, text="").legacy_surface


class ScheduleFallbackParser(Protocol):
    """Optional parser used when deterministic schedule parsing cannot decide."""

    def parse(
        self,
        *,
        task: Task,
        context: ScheduleCreationContext,
        text: str,
        now: datetime,
    ) -> ScheduleDraft | None:
        """Return a validated schedule draft, or None when parsing is unsafe."""


class ScheduleCreationService:
    """Creates schedules from explicit scheduling requests."""

    def __init__(
        self,
        session: Session,
        *,
        task_service: TaskService | None = None,
        fallback_parser: ScheduleFallbackParser | None = None,
    ) -> None:
        self.session = session
        self.task_service = task_service or TaskService(session)
        self.fallback_parser = fallback_parser

    def propose_from_text(
        self,
        *,
        task: Task,
        context: ScheduleCreationContext,
        text: str,
        now: datetime | None = None,
    ) -> ScheduleProposal | None:
        """Create a schedule if the text has a supported schedule."""

        parse_time = now or datetime.now(UTC)
        draft = parse_schedule_request(
            text,
            now=parse_time,
            timezone=context.timezone,
        )
        if draft is None and self.fallback_parser is not None:
            draft = self.fallback_parser.parse(
                task=task,
                context=context,
                text=text,
                now=parse_time,
            )
        if draft is None:
            return None

        needs_confirmation = draft.needs_confirmation or _needs_confirmation(
            text=text,
            draft=draft,
        )
        status = "proposed" if needs_confirmation else "active"
        delivery = infer_schedule_delivery(context=context, text=text)
        schedule = Schedule(
            installation_id=context.installation_id,
            owner_type="user",
            owner_slack_user_id=context.slack_user_id,
            title=draft.title,
            spec_kind=draft.spec_kind,
            cron_expr=draft.cron_expr,
            interval_seconds=draft.interval_seconds,
            run_at=draft.run_at,
            timezone=draft.timezone,
            next_run_at=draft.next_run_at,
            catchup_policy="skip",
            catchup_window_seconds=DEFAULT_CATCHUP_WINDOW_SECONDS,
            overlap_policy="skip",
            status=status,
            delivery_kind=delivery.kind,
            delivery_slack_user_id=delivery.slack_user_id,
            delivery_slack_channel_id=delivery.slack_channel_id,
            delivery_slack_thread_ts=delivery.slack_thread_ts,
            artifact_delivery_policy=delivery.artifact_policy,
            task_template={
                "input": draft.task_input,
                "slack_channel_id": delivery.slack_channel_id,
                "slack_user_id": delivery.slack_user_id,
                "slack_thread_ts": delivery.slack_thread_ts,
                "delivery_surface": delivery.legacy_surface,
                "artifact_delivery_policy": delivery.artifact_policy,
            },
            planned_cost_ceiling_usd=DEFAULT_SCHEDULE_BUDGET_USD,
            created_by_slack_user_id=context.slack_user_id,
            metadata_json={
                "source_task_id": str(context.source_task_id),
                "source_surface": context.source_surface,
                "original_input": text,
                "cadence_label": draft.cadence_label,
                "delivery_surface": delivery.legacy_surface,
                "delivery_kind": delivery.kind,
                "artifact_delivery_policy": delivery.artifact_policy,
                "confirmation_required": needs_confirmation,
                "parse_strategy": draft.parse_strategy,
            },
        )
        self.session.add(schedule)
        self.session.flush()
        self.task_service.append_event(
            task,
            TaskEventType.log,
            {
                "message": SCHEDULE_CREATED_MESSAGE,
                "schedule_id": str(schedule.id),
                "schedule_status": schedule.status,
                "cadence_label": draft.cadence_label,
                "next_run_at": draft.next_run_at.isoformat(),
                "timezone": draft.timezone,
                "delivery_surface": delivery.legacy_surface,
                "delivery_kind": delivery.kind,
                "artifact_delivery_policy": delivery.artifact_policy,
                "planned_cost_ceiling_usd": str(DEFAULT_SCHEDULE_BUDGET_USD),
                "confirmation_required": needs_confirmation,
            },
        )
        if needs_confirmation:
            self.task_service.append_event(
                task,
                TaskEventType.log,
                {
                    "message": SCHEDULE_PROPOSAL_CREATED_MESSAGE,
                    "schedule_id": str(schedule.id),
                    "schedule_status": schedule.status,
                    "cadence_label": draft.cadence_label,
                    "next_run_at": draft.next_run_at.isoformat(),
                },
            )
        return ScheduleProposal(
            schedule=schedule,
            draft=draft,
            response_text=format_schedule_proposal(
                schedule=schedule,
                draft=draft,
                delivery_surface=delivery.legacy_surface,
                needs_confirmation=needs_confirmation,
                delivery_label=delivery.response_label,
                now=parse_time,
            ),
        )


def looks_like_schedule_request(text: str) -> bool:
    """Return whether a user message explicitly asks for scheduled work."""

    return bool(SCHEDULE_RE.search(text))


def infer_schedule_delivery(
    *,
    context: ScheduleCreationContext,
    text: str,
) -> ScheduleDelivery:
    """Infer a deliverable Slack destination from the request surface."""

    artifact_policy = _artifact_delivery_policy(text)
    if context.source_surface == "dm" or context.slack_channel_id.startswith("D"):
        return ScheduleDelivery(
            kind="slack_dm",
            slack_user_id=context.slack_user_id,
            slack_channel_id=context.slack_channel_id,
            slack_thread_ts=context.slack_thread_ts,
            artifact_policy=artifact_policy,
        )

    if THREAD_DELIVERY_RE.search(text):
        return ScheduleDelivery(
            kind="slack_thread",
            slack_user_id=context.slack_user_id,
            slack_channel_id=context.slack_channel_id,
            slack_thread_ts=context.slack_thread_ts,
            artifact_policy=artifact_policy,
        )

    if CHANNEL_DELIVERY_RE.search(text):
        return ScheduleDelivery(
            kind="slack_channel",
            slack_user_id=context.slack_user_id,
            slack_channel_id=context.slack_channel_id,
            slack_thread_ts=None,
            artifact_policy=artifact_policy,
        )

    # Channel-origin schedules are recurring by definition, so the safe default
    # is the request thread instead of the channel root.
    return ScheduleDelivery(
        kind="slack_thread",
        slack_user_id=context.slack_user_id,
        slack_channel_id=context.slack_channel_id,
        slack_thread_ts=context.slack_thread_ts,
        artifact_policy=artifact_policy,
    )


def parse_schedule_request(
    text: str,
    *,
    now: datetime,
    timezone: str = DEFAULT_TIMEZONE,
) -> ScheduleDraft | None:
    """Parse a small, deterministic subset of schedule requests."""

    requested_timezone = _timezone_from_text(text) or timezone
    normalized_timezone, tzinfo = _timezone(requested_timezone)
    local_now = _coerce_utc(now).astimezone(tzinfo)
    task_input = _task_input_from_schedule_text(text)

    weekly_match = WEEKLY_RE.search(text)
    if weekly_match is not None:
        weekday = weekly_match.group(1).casefold()
        daypart = (weekly_match.group(2) or "morning").casefold()
        hour, minute, time_label = _schedule_time(
            text,
            default_hour=DAYPART_HOURS[daypart],
            timezone=normalized_timezone,
        )
        next_run_at = _next_weekly(
            local_now,
            weekday=weekday,
            hour=hour,
            minute=minute,
        )
        cadence_label = _cadence_label(
            f"Every {weekday.title()} {daypart}",
            time_label=time_label,
        )
        return ScheduleDraft(
            title=_title_from_task_input(task_input),
            spec_kind="cron",
            cron_expr=f"{minute} {hour} * * {WEEKDAYS[weekday]}",
            timezone=normalized_timezone,
            next_run_at=next_run_at,
            cadence_label=cadence_label,
            task_input=task_input,
        )

    daily_match = DAILY_RE.search(text)
    if daily_match is not None:
        daypart = (daily_match.group(1) or "morning").casefold()
        hour, minute, time_label = _schedule_time(
            text,
            default_hour=DAYPART_HOURS[daypart],
            timezone=normalized_timezone,
        )
        next_run_at = _next_daily(local_now, hour=hour, minute=minute)
        cadence_label = _cadence_label(
            f"Every {daypart}",
            time_label=time_label,
        )
        return ScheduleDraft(
            title=_title_from_task_input(task_input),
            spec_kind="cron",
            cron_expr=f"{minute} {hour} * * *",
            timezone=normalized_timezone,
            next_run_at=next_run_at,
            cadence_label=cadence_label,
            task_input=task_input,
        )

    interval_match = INTERVAL_RE.search(text)
    if interval_match is not None:
        amount = int(interval_match.group(1))
        unit = interval_match.group(2).casefold()
        interval_seconds = amount * INTERVAL_UNITS[unit]
        next_run_at = _coerce_utc(now) + timedelta(seconds=interval_seconds)
        return ScheduleDraft(
            title=_title_from_task_input(task_input),
            spec_kind="interval",
            interval_seconds=interval_seconds,
            timezone=normalized_timezone,
            next_run_at=next_run_at,
            cadence_label=f"Every {amount} {unit}",
            task_input=task_input,
        )

    oneoff = _parse_oneoff(text, now=_coerce_utc(now), local_now=local_now)
    if oneoff is None:
        return None
    run_at, cadence_label = oneoff
    return ScheduleDraft(
        title=_title_from_task_input(task_input),
        spec_kind="oneoff",
        run_at=run_at,
        timezone=normalized_timezone,
        next_run_at=run_at,
        cadence_label=cadence_label,
        task_input=task_input,
    )


def format_schedule_proposal(
    *,
    schedule: Schedule,
    draft: ScheduleDraft,
    delivery_surface: str,
    needs_confirmation: bool,
    delivery_label: str | None = None,
    now: datetime | None = None,
) -> str:
    """Render a Slack-native schedule response."""

    destination = delivery_label or (
        "in this DM" if delivery_surface == "dm" else "in this thread"
    )
    first_run = _human_next_run(draft.next_run_at, timezone=draft.timezone, now=now)
    task_summary = _human_task_summary(draft.task_input)
    cadence = draft.cadence_label[:1].lower() + draft.cadence_label[1:]
    if needs_confirmation:
        return (
            "I can do that. I drafted it and will wait for you to confirm "
            "before I start.\n\n"
            f"I'd {task_summary} {cadence}. First check would be {first_run}; "
            f"I'll send it {destination}.\n\n"
            "Say `yes, set it up` when you want me to turn it on."
        )
    return (
        "Done, I'll take care of that.\n\n"
        f"I'll {task_summary} {cadence}. First check is {first_run}; "
        f"I'll send it {destination}.\n\n"
        "You can tell me to pause, change, or cancel it anytime."
    )


def _parse_oneoff(
    text: str,
    *,
    now: datetime,
    local_now: datetime,
) -> tuple[datetime, str] | None:
    in_match = IN_RE.search(text)
    if in_match is not None:
        amount = int(in_match.group(1))
        unit = in_match.group(2).casefold()
        seconds = amount * INTERVAL_UNITS[unit]
        return now + timedelta(seconds=seconds), f"In {amount} {unit}"

    tomorrow_match = TOMORROW_RE.search(text)
    if tomorrow_match is None:
        return None
    daypart = (tomorrow_match.group(1) or "morning").casefold()
    hour = DAYPART_HOURS[daypart]
    target = (local_now + timedelta(days=1)).replace(
        hour=hour,
        minute=0,
        second=0,
        microsecond=0,
    )
    return target.astimezone(UTC), f"Tomorrow {daypart}"


def _next_daily(now: datetime, *, hour: int, minute: int = 0) -> datetime:
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return target.astimezone(UTC)


def _next_weekly(
    now: datetime,
    *,
    weekday: str,
    hour: int,
    minute: int = 0,
) -> datetime:
    target_weekday = PYTHON_WEEKDAYS[weekday]
    days_ahead = (target_weekday - now.weekday()) % 7
    target = (now + timedelta(days=days_ahead)).replace(
        hour=hour,
        minute=minute,
        second=0,
        microsecond=0,
    )
    if target <= now:
        target += timedelta(days=7)
    return target.astimezone(UTC)


def _task_input_from_schedule_text(text: str) -> str:
    stripped = " ".join(text.split())
    task_input = LEADING_CADENCE_RE.sub("", stripped).strip(" ,")
    task_input = LEADING_TIME_RE.sub("", task_input).strip(" ,")
    task_input = _normalize_requested_task(task_input)
    return task_input or stripped


def _needs_confirmation(*, text: str, draft: ScheduleDraft) -> bool:
    del draft
    return bool(CONFIRMATION_REQUEST_RE.search(text))


def _artifact_delivery_policy(text: str) -> str:
    if ATTACH_ARTIFACTS_RE.search(text):
        return "attach_files"
    if LINK_ARTIFACTS_RE.search(text):
        return "link_artifacts"
    return "message_only"


def _human_next_run(
    value: datetime,
    *,
    timezone: str,
    now: datetime | None = None,
) -> str:
    normalized_timezone, tzinfo = _timezone(timezone)
    local = _coerce_utc(value).astimezone(tzinfo)
    reference = _coerce_utc(now or datetime.now(UTC)).astimezone(tzinfo)
    local_date = local.date()
    today = reference.date()
    time_text = _format_time(hour=local.hour, minute=local.minute)
    timezone_label = _timezone_label(normalized_timezone)
    if local_date == today:
        return f"today at {time_text} {timezone_label}"
    if local_date == today + timedelta(days=1):
        return f"tomorrow at {time_text} {timezone_label}"
    return f"{local.strftime('%A')} at {time_text} {timezone_label}"


def _schedule_time(
    text: str,
    *,
    default_hour: int,
    timezone: str,
) -> tuple[int, int, str | None]:
    match = EXPLICIT_TIME_RE.search(text)
    if match is None:
        return default_hour, 0, None

    hour = int(match.group(1))
    minute = int(match.group(2) or "0")
    meridiem = match.group(3).replace(".", "").casefold()
    if meridiem == "pm" and hour != 12:
        hour += 12
    if meridiem == "am" and hour == 12:
        hour = 0
    return (
        hour,
        minute,
        f"at {_format_time(hour=hour, minute=minute)} {_timezone_label(timezone)}",
    )


def _cadence_label(base: str, *, time_label: str | None) -> str:
    if time_label is None:
        return base
    return f"{base} {time_label}"


def _format_time(*, hour: int, minute: int) -> str:
    meridiem = "AM" if hour < 12 else "PM"
    display_hour = hour % 12 or 12
    return f"{display_hour}:{minute:02d} {meridiem}"


def _timezone_from_text(text: str) -> str | None:
    for timezone, _label, pattern in TIMEZONE_ALIASES:
        if pattern.search(text):
            return timezone
    return None


def _timezone_label(timezone: str) -> str:
    for alias_timezone, label, _pattern in TIMEZONE_ALIASES:
        if alias_timezone == timezone:
            return label
    return timezone


def _human_task_summary(task_input: str) -> str:
    summary = task_input.strip().rstrip(".")
    summary = re.sub(
        r"^(?:can you|could you|please|can u|could u)\s+",
        "",
        summary,
        flags=re.IGNORECASE,
    )
    summary = re.sub(r"\bgive me\b", "send you", summary, flags=re.IGNORECASE)
    summary = re.sub(r"\bDM me\b", "DM you", summary, flags=re.IGNORECASE)
    return summary[:1].lower() + summary[1:] if summary else "run this"


def _normalize_requested_task(task_input: str) -> str:
    match = re.match(
        r"^(?:i\s+want|i'd\s+like|i\s+would\s+like)\s+(?:you\s+to\s+)?(.+)$",
        task_input,
        flags=re.IGNORECASE,
    )
    if match is None:
        return task_input

    requested = match.group(1).strip()
    if re.match(
        r"^(?:check|summarize|research|review|send|create|draft|give|look|"
        r"find|watch|monitor|remind)\b",
        requested,
        flags=re.IGNORECASE,
    ):
        return requested
    return f"send {requested}"


def _title_from_task_input(task_input: str) -> str:
    clean = task_input.rstrip(".")
    if len(clean) > 80:
        clean = clean[:77].rstrip() + "..."
    return clean[:1].upper() + clean[1:] if clean else "Scheduled task"


def _timezone(value: str) -> tuple[str, ZoneInfo]:
    normalized = value.strip() or DEFAULT_TIMEZONE
    try:
        return normalized, ZoneInfo(normalized)
    except ZoneInfoNotFoundError:
        return DEFAULT_TIMEZONE, ZoneInfo(DEFAULT_TIMEZONE)


def _coerce_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
