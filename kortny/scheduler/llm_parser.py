"""LLM fallback parser for schedule requests."""

from __future__ import annotations

import json
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Literal, Protocol
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, Field, ValidationError

from kortny.db.models import Task
from kortny.llm import ChatMessage, Completion
from kortny.scheduler.creation import ScheduleCreationContext, ScheduleDraft
from kortny.tools.types import JsonObject, JsonSchema

SCHEDULE_PARSER_RESPONSE_FORMAT: JsonObject = {"type": "json_object"}
SCHEDULE_PARSER_PROMPT_NAME = "kortny.schedule_parser"
MIN_CONFIDENCE = 0.72

SCHEDULE_PARSER_SYSTEM_PROMPT = """
You convert explicit Slack scheduling requests into a safe schedule draft.

Return only a JSON object.

Allowed output fields:
- is_schedule: boolean
- schedule_kind: "cron", "interval", "oneoff", or "unsupported"
- cron_expr: string or null
- interval_seconds: integer or null
- run_at: ISO-8601 datetime string or null
- timezone: IANA timezone string, for example "America/Chicago" or "UTC"
- cadence_label: short human-readable cadence label
- task_input: the task Kortny should run when the schedule fires
- needs_confirmation: boolean
- confidence: number from 0 to 1
- clarifying_question: string or null

Rules:
- Only parse requests where the user is clearly asking for scheduled work.
- Put the recurrence in cron_expr when the request is daily/weekly/monthly-like.
- Use five-field cron only: minute hour day_of_month month day_of_week.
- Do not use seconds, years, named months, or named weekdays in cron.
- For weekdays use 1-5. Cron weekday mapping is 0=Sunday, 1=Monday, ..., 6=Saturday.
- For every day use "* *" in day_of_month/month and "*" in day_of_week.
- Use interval_seconds only for fixed intervals such as every 2 hours.
- Use run_at only for one-off future dates.
- The cron expression is evaluated in the returned timezone.
- If the request has a timezone phrase, return the matching IANA timezone.
- If no timezone is provided, use the supplied default timezone.
- Remove scheduling language from task_input. Keep the real task instruction.
- If the timing is ambiguous or unsupported, set schedule_kind="unsupported",
  is_schedule=true, confidence below 0.72, and include a clarifying_question.
- If the user asks to draft, propose, wait for confirmation, or asks for approval,
  set needs_confirmation=true.

Examples:
- "every weekday at 9am summarize my unread email" -> is_schedule=true,
  schedule_kind="cron", cron_expr="0 9 * * 1-5", cadence_label="weekdays at 9am",
  task_input="summarize my unread email".
- "every 2 hours check the deploy status" -> schedule_kind="interval",
  interval_seconds=7200, task_input="check the deploy status".
- "remind me about the review sometime next week" -> timing ambiguous:
  schedule_kind="unsupported", is_schedule=true, confidence below 0.72,
  clarifying_question asking for the exact day and time.
- "what's on my calendar tomorrow?" -> not a scheduling request:
  is_schedule=false.
""".strip()


class ScheduleParserLLMClient(Protocol):
    """Subset of LLMService used by the schedule parser."""

    def complete(
        self,
        *,
        task_id: uuid.UUID,
        messages: Sequence[ChatMessage],
        tools: Sequence[JsonSchema] = (),
        response_format: JsonObject | None = None,
        prompt_name: str | None = None,
        prompt_source: str = "code",
    ) -> Completion:
        """Complete one schedule parser turn."""


class ScheduleParserPayload(BaseModel):
    """Model-produced schedule parser payload."""

    is_schedule: bool = False
    schedule_kind: Literal["cron", "interval", "oneoff", "unsupported"] = "unsupported"
    cron_expr: str | None = None
    interval_seconds: int | None = None
    run_at: str | None = None
    timezone: str = "UTC"
    cadence_label: str = ""
    task_input: str = ""
    needs_confirmation: bool = False
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    clarifying_question: str | None = None


class LLMScheduleParser:
    """LLM-backed fallback parser for schedule requests."""

    def __init__(
        self,
        *,
        llm: ScheduleParserLLMClient,
        min_confidence: float = MIN_CONFIDENCE,
    ) -> None:
        self.llm = llm
        self.min_confidence = min_confidence
        self.last_clarifying_question: str | None = None

    def parse(
        self,
        *,
        task: Task,
        context: ScheduleCreationContext,
        text: str,
        now: datetime,
    ) -> ScheduleDraft | None:
        """Parse with an LLM and return only validated schedule drafts."""

        parse_time = _coerce_utc(now)
        self.last_clarifying_question = None
        completion = self.llm.complete(
            task_id=task.id,
            messages=(
                ChatMessage(role="system", content=SCHEDULE_PARSER_SYSTEM_PROMPT),
                ChatMessage(
                    role="user",
                    content=json.dumps(
                        {
                            "message": text,
                            "default_timezone": context.timezone,
                            "now_utc": parse_time.isoformat(),
                            "source_surface": context.source_surface,
                        },
                        sort_keys=True,
                    ),
                ),
            ),
            response_format=SCHEDULE_PARSER_RESPONSE_FORMAT,
            prompt_name=SCHEDULE_PARSER_PROMPT_NAME,
        )
        try:
            payload = ScheduleParserPayload.model_validate_json(
                _extract_json_object(completion.content)
            )
        except (ValueError, ValidationError):
            return None
        question = (payload.clarifying_question or "").strip()
        self.last_clarifying_question = question or None
        return _draft_from_payload(
            payload,
            now=parse_time,
            min_confidence=self.min_confidence,
        )


def _draft_from_payload(
    payload: ScheduleParserPayload,
    *,
    now: datetime,
    min_confidence: float,
) -> ScheduleDraft | None:
    if (
        not payload.is_schedule
        or payload.schedule_kind == "unsupported"
        or payload.confidence < min_confidence
    ):
        return None

    timezone, tzinfo = _timezone(payload.timezone)
    task_input = payload.task_input.strip()
    if not task_input:
        return None
    cadence_label = payload.cadence_label.strip() or _fallback_cadence_label(payload)

    if payload.schedule_kind == "cron":
        cron_expr = (payload.cron_expr or "").strip()
        if not _cron_supported(cron_expr):
            return None
        next_run_at = _next_cron_after(
            cron_expr,
            timezone=timezone,
            tzinfo=tzinfo,
            after=now,
        )
        return ScheduleDraft(
            title=_title_from_task_input(task_input),
            spec_kind="cron",
            cron_expr=cron_expr,
            timezone=timezone,
            next_run_at=next_run_at,
            cadence_label=cadence_label,
            task_input=task_input,
            needs_confirmation=payload.needs_confirmation,
            parse_strategy="llm_schedule_parser",
        )

    if payload.schedule_kind == "interval":
        interval_seconds = payload.interval_seconds
        if interval_seconds is None or interval_seconds <= 0:
            return None
        return ScheduleDraft(
            title=_title_from_task_input(task_input),
            spec_kind="interval",
            interval_seconds=interval_seconds,
            timezone=timezone,
            next_run_at=now + timedelta(seconds=interval_seconds),
            cadence_label=cadence_label,
            task_input=task_input,
            needs_confirmation=payload.needs_confirmation,
            parse_strategy="llm_schedule_parser",
        )

    if payload.schedule_kind == "oneoff":
        run_at = _parse_datetime(payload.run_at)
        if run_at is None or run_at <= now:
            return None
        return ScheduleDraft(
            title=_title_from_task_input(task_input),
            spec_kind="oneoff",
            run_at=run_at,
            timezone=timezone,
            next_run_at=run_at,
            cadence_label=cadence_label,
            task_input=task_input,
            needs_confirmation=payload.needs_confirmation,
            parse_strategy="llm_schedule_parser",
        )

    return None


def _extract_json_object(content: str | None) -> str:
    if content is None or not content.strip():
        raise ValueError("empty schedule parser response")
    stripped = content.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object found")
    candidate = stripped[start : end + 1]
    json.loads(candidate)
    return candidate


def _cron_supported(cron_expr: str) -> bool:
    try:
        _parse_simple_cron(cron_expr)
    except ValueError:
        return False
    return True


def _next_cron_after(
    cron_expr: str,
    *,
    timezone: str,
    tzinfo: ZoneInfo,
    after: datetime,
) -> datetime:
    del timezone
    minute, hour, weekdays = _parse_simple_cron(cron_expr)
    local_after = _coerce_utc(after).astimezone(tzinfo)
    candidate = local_after.replace(
        hour=hour,
        minute=minute,
        second=0,
        microsecond=0,
    )

    for day_offset in range(0, 370):
        current = candidate + timedelta(days=day_offset)
        if current <= local_after:
            continue
        if weekdays is None or current.weekday() in weekdays:
            return current.astimezone(UTC)
    raise ValueError("cron expression produced no next run")


def _parse_simple_cron(cron_expr: str) -> tuple[int, int, frozenset[int] | None]:
    fields = cron_expr.strip().split()
    if len(fields) != 5:
        raise ValueError("unsupported cron expression")

    minute_field, hour_field, dom_field, month_field, weekday_field = fields
    if dom_field != "*" or month_field != "*":
        raise ValueError("unsupported cron expression")
    try:
        minute = int(minute_field)
        hour = int(hour_field)
    except ValueError as exc:
        raise ValueError("unsupported cron expression") from exc
    if not 0 <= minute <= 59 or not 0 <= hour <= 23:
        raise ValueError("unsupported cron expression")
    return minute, hour, _parse_weekday_field(weekday_field)


def _parse_weekday_field(value: str) -> frozenset[int] | None:
    if value == "*":
        return None
    weekdays: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            raise ValueError("unsupported cron expression")
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = _cron_weekday_to_python(start_text)
            end = _cron_weekday_to_python(end_text)
            if start > end:
                raise ValueError("unsupported cron expression")
            weekdays.update(range(start, end + 1))
        else:
            weekdays.add(_cron_weekday_to_python(part))
    return frozenset(weekdays)


def _cron_weekday_to_python(value: str) -> int:
    try:
        cron_weekday = int(value)
    except ValueError as exc:
        raise ValueError("unsupported cron expression") from exc
    if not 0 <= cron_weekday <= 6:
        raise ValueError("unsupported cron expression")
    return (cron_weekday + 6) % 7


def _parse_datetime(value: str | None) -> datetime | None:
    if value is None or not value.strip():
        return None
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    return _coerce_utc(parsed)


def _fallback_cadence_label(payload: ScheduleParserPayload) -> str:
    if payload.schedule_kind == "cron" and payload.cron_expr:
        return payload.cron_expr
    if payload.schedule_kind == "interval" and payload.interval_seconds:
        return f"Every {payload.interval_seconds} seconds"
    if payload.schedule_kind == "oneoff" and payload.run_at:
        return "One-time"
    return "Scheduled task"


def _title_from_task_input(task_input: str) -> str:
    clean = task_input.rstrip(".")
    if len(clean) > 80:
        clean = clean[:77].rstrip() + "..."
    return clean[:1].upper() + clean[1:] if clean else "Scheduled task"


def _timezone(value: str) -> tuple[str, ZoneInfo]:
    normalized = value.strip() or "UTC"
    try:
        return normalized, ZoneInfo(normalized)
    except ZoneInfoNotFoundError:
        return "UTC", ZoneInfo("UTC")


def _coerce_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
