"""Postgres-native schedule materialization service."""

from __future__ import annotations

import argparse
import logging
import os
import socket
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any, cast
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from kortny.ambient.system_drives import SYSTEM_DRIVE_METADATA_KEY
from kortny.config import load_settings
from kortny.db.models import Schedule, Task, TaskEventType, TaskStatus
from kortny.db.session import make_session_factory
from kortny.logging_config import configure_logging
from kortny.observability import configure_tracing, start_span
from kortny.tasks import TaskIdentity, TaskService

DEFAULT_CATCHUP_WINDOW_SECONDS = 300
DEFAULT_MATERIALIZE_LIMIT = 50
DEFAULT_POLL_INTERVAL_SECONDS = 5.0
DEFAULT_ADVISORY_LOCK_KEY = 759340185
SCHEDULE_BUDGET_ADMISSION_FAILED_MESSAGE = "schedule_budget_admission_failed"
SCHEDULED_TASK_BUDGET_ADMITTED_MESSAGE = "scheduled_task_budget_admitted"

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ScheduleMaterialization:
    """Outcome for one due schedule row."""

    schedule_id: uuid.UUID
    action: str
    fire_time: datetime | None = None
    task_id: uuid.UUID | None = None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class SchedulerRunResult:
    """Outcome from one scheduler poll cycle."""

    scheduler_id: str
    status: str
    materializations: tuple[ScheduleMaterialization, ...] = ()
    leader_acquired: bool = True

    @property
    def materialized_count(self) -> int:
        return sum(1 for item in self.materializations if item.task_id is not None)


class ScheduleMaterializer:
    """Turn due schedule rows into ordinary pending task rows."""

    def __init__(
        self,
        session: Session,
        *,
        advisory_lock_key: int = DEFAULT_ADVISORY_LOCK_KEY,
        default_catchup_window_seconds: int = DEFAULT_CATCHUP_WINDOW_SECONDS,
    ) -> None:
        self.session = session
        self.tasks = TaskService(session)
        self.advisory_lock_key = advisory_lock_key
        self.default_catchup_window_seconds = default_catchup_window_seconds

    def materialize_due_schedules(
        self,
        *,
        now: datetime | None = None,
        limit: int = DEFAULT_MATERIALIZE_LIMIT,
        use_advisory_lock: bool = True,
    ) -> tuple[ScheduleMaterialization, ...]:
        """Materialize due schedules, returning one outcome per due row."""

        if limit < 1:
            raise ValueError("limit must be positive")

        materialize_time = _coerce_utc(now)
        lock_acquired = True
        if use_advisory_lock:
            lock_acquired = self._try_advisory_lock()
        if not lock_acquired:
            return ()

        try:
            schedules = self._claim_due_schedules(now=materialize_time, limit=limit)
            results = [
                self._materialize_schedule(schedule, now=materialize_time)
                for schedule in schedules
            ]
            self.session.flush()
            return tuple(results)
        finally:
            if use_advisory_lock:
                self._release_advisory_lock()

    def _claim_due_schedules(
        self, *, now: datetime, limit: int
    ) -> tuple[Schedule, ...]:
        return tuple(
            self.session.scalars(
                select(Schedule)
                .where(
                    Schedule.status == "active",
                    Schedule.next_run_at.is_not(None),
                    Schedule.next_run_at <= now,
                    # System drives (HIG-233) are the control/visibility surface
                    # for the ambient loops; they execute in the loops, never via
                    # materialization. They carry no next_run_at, but the filter
                    # is defensive even if that representation ever changes.
                    ~Schedule.metadata_json.has_key(SYSTEM_DRIVE_METADATA_KEY),
                )
                .order_by(Schedule.next_run_at, Schedule.created_at)
                .with_for_update(skip_locked=True)
                .limit(limit)
            )
        )

    def _materialize_schedule(
        self,
        schedule: Schedule,
        *,
        now: datetime,
    ) -> ScheduleMaterialization:
        due_at = _coerce_utc(schedule.next_run_at)
        if self._overlap_active(schedule):
            return ScheduleMaterialization(
                schedule_id=schedule.id,
                action="skipped",
                fire_time=due_at,
                reason="active_materialized_task_exists",
            )

        if schedule.spec_kind == "cron" and not _cron_supported(schedule):
            schedule.status = "paused"
            schedule.metadata_json = {
                **dict(schedule.metadata_json or {}),
                "last_scheduler_error": "unsupported_cron_expression",
                "paused_at": now.isoformat(),
            }
            schedule.updated_at = now
            return ScheduleMaterialization(
                schedule_id=schedule.id,
                action="paused",
                fire_time=due_at,
                reason="unsupported_cron_expression",
            )

        if self._missed_window(schedule, due_at=due_at, now=now):
            self._advance_without_run(schedule, due_at=due_at, now=now)
            return ScheduleMaterialization(
                schedule_id=schedule.id,
                action="skipped",
                fire_time=due_at,
                reason="missed_catchup_window",
            )

        if not _has_valid_run_budget(schedule):
            self._pause_for_budget_admission_failure(schedule, now=now)
            return ScheduleMaterialization(
                schedule_id=schedule.id,
                action="paused",
                fire_time=due_at,
                reason="missing_planned_cost_ceiling",
            )

        task = self._create_materialized_task(schedule, fire_time=due_at, now=now)
        self._advance_after_run(schedule, fire_time=due_at, now=now)
        return ScheduleMaterialization(
            schedule_id=schedule.id,
            action="materialized",
            fire_time=due_at,
            task_id=task.id,
        )

    def _create_materialized_task(
        self,
        schedule: Schedule,
        *,
        fire_time: datetime,
        now: datetime,
    ) -> Task:
        template = dict(schedule.task_template or {})
        input_text = _required_template_string(template, "input", schedule=schedule)
        delivery = _schedule_delivery(schedule, template=template)
        identity_payload: dict[str, Any] = {
            "schedule_title": schedule.title,
            "owner_type": schedule.owner_type,
            "owner_slack_user_id": schedule.owner_slack_user_id,
            "spec_kind": schedule.spec_kind,
            "delivery_kind": delivery["kind"],
            "delivery_slack_user_id": delivery["slack_user_id"],
            "delivery_slack_channel_id": delivery["slack_channel_id"],
            "delivery_slack_thread_ts": delivery["slack_thread_ts"],
            "artifact_delivery_policy": delivery["artifact_policy"],
        }
        if schedule.planned_cost_ceiling_usd is not None:
            identity_payload["planned_cost_ceiling_usd"] = str(
                schedule.planned_cost_ceiling_usd
            )
            identity_payload["runtime_cost_ceiling_usd"] = str(
                schedule.planned_cost_ceiling_usd
            )

        identity = TaskIdentity.scheduled(
            schedule_id=str(schedule.id),
            fire_time=fire_time.isoformat(),
            input_text=input_text,
            payload=identity_payload,
        )
        task = self.tasks.create_task(
            installation_id=schedule.installation_id,
            slack_channel_id=cast(str, delivery["slack_channel_id"]),
            slack_user_id=cast(str, delivery["slack_user_id"]),
            input=input_text,
            slack_thread_ts=delivery["slack_thread_ts"],
            slack_message_ts=_optional_template_string(template, "slack_message_ts"),
            parent_task_id=None,
            identity=identity,
            source_surface="schedule",
        )
        task.available_at = now
        self.tasks.append_event(
            task,
            TaskEventType.log,
            {
                "message": SCHEDULED_TASK_BUDGET_ADMITTED_MESSAGE,
                "schedule_id": str(schedule.id),
                "cost_ceiling_usd": _decimal_string(schedule.planned_cost_ceiling_usd),
                "behavior": "admit_with_per_run_ceiling",
            },
        )
        self.tasks.append_event(
            task,
            TaskEventType.log,
            {
                "message": "scheduled_task_materialized",
                "schedule_id": str(schedule.id),
                "schedule_title": schedule.title,
                "fire_time": fire_time.isoformat(),
                "planned_cost_ceiling_usd": _decimal_string(
                    schedule.planned_cost_ceiling_usd
                ),
                "delivery_kind": delivery["kind"],
                "delivery_slack_channel_id": delivery["slack_channel_id"],
                "delivery_slack_thread_ts": delivery["slack_thread_ts"],
                "artifact_delivery_policy": delivery["artifact_policy"],
            },
        )
        return task

    def _pause_for_budget_admission_failure(
        self,
        schedule: Schedule,
        *,
        now: datetime,
    ) -> None:
        metadata = dict(schedule.metadata_json or {})
        metadata["last_scheduler_error"] = "missing_planned_cost_ceiling"
        metadata["last_budget_status"] = "admission_failed"
        metadata["last_budget_admission_failed_at"] = now.isoformat()
        schedule.metadata_json = metadata
        schedule.status = "paused"
        schedule.updated_at = now

    def _advance_after_run(
        self,
        schedule: Schedule,
        *,
        fire_time: datetime,
        now: datetime,
    ) -> None:
        schedule.last_run_at = fire_time
        if schedule.spec_kind == "oneoff":
            schedule.status = "completed"
            schedule.next_run_at = None
        elif schedule.spec_kind == "interval":
            schedule.next_run_at = _next_interval_after(schedule, after=now)
        elif schedule.spec_kind == "cron":
            schedule.next_run_at = _next_cron_after(schedule, after=now)
        schedule.updated_at = now

    def _advance_without_run(
        self,
        schedule: Schedule,
        *,
        due_at: datetime,
        now: datetime,
    ) -> None:
        schedule.last_run_at = due_at
        if schedule.spec_kind == "oneoff":
            schedule.status = "completed"
            schedule.next_run_at = None
        elif schedule.spec_kind == "interval":
            schedule.next_run_at = _next_interval_after(schedule, after=now)
        elif schedule.spec_kind == "cron":
            schedule.next_run_at = _next_cron_after(schedule, after=now)
        schedule.updated_at = now

    def _missed_window(
        self,
        schedule: Schedule,
        *,
        due_at: datetime,
        now: datetime,
    ) -> bool:
        if schedule.catchup_policy != "skip":
            return False
        window_seconds = (
            schedule.catchup_window_seconds
            if schedule.catchup_window_seconds is not None
            else self.default_catchup_window_seconds
        )
        if window_seconds < 0:
            return False
        return due_at < now - timedelta(seconds=window_seconds)

    def _overlap_active(self, schedule: Schedule) -> bool:
        if schedule.overlap_policy != "skip":
            return False
        return (
            self.session.scalar(
                select(Task.id)
                .where(
                    Task.installation_id == schedule.installation_id,
                    Task.identity_kind == "scheduled",
                    Task.identity_payload["schedule_id"].as_string()
                    == str(schedule.id),
                    Task.status.in_(
                        [
                            TaskStatus.pending,
                            TaskStatus.running,
                            TaskStatus.waiting_approval,
                        ]
                    ),
                )
                .limit(1)
            )
            is not None
        )

    def _try_advisory_lock(self) -> bool:
        return bool(
            self.session.scalar(
                select(func.pg_try_advisory_lock(self.advisory_lock_key))
            )
        )

    def _release_advisory_lock(self) -> None:
        self.session.execute(select(func.pg_advisory_unlock(self.advisory_lock_key)))


class SchedulerWorker:
    """Poll schedules forever and materialize due tasks."""

    def __init__(
        self,
        *,
        session_factory: sessionmaker[Session] | None = None,
        scheduler_id: str | None = None,
        poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
        materialize_limit: int = DEFAULT_MATERIALIZE_LIMIT,
        advisory_lock_key: int = DEFAULT_ADVISORY_LOCK_KEY,
    ) -> None:
        self.session_factory = session_factory or make_session_factory()
        self.scheduler_id = scheduler_id or default_scheduler_id()
        self.poll_interval_seconds = poll_interval_seconds
        self.materialize_limit = materialize_limit
        self.advisory_lock_key = advisory_lock_key

    def run_once(self, *, now: datetime | None = None) -> SchedulerRunResult:
        with self.session_factory.begin() as session:
            materializer = ScheduleMaterializer(
                session,
                advisory_lock_key=self.advisory_lock_key,
            )
            with start_span(
                "scheduler.materialize",
                attributes={
                    "openinference.span.kind": "CHAIN",
                    "scheduler.id": self.scheduler_id,
                },
            ):
                materializations = materializer.materialize_due_schedules(
                    now=now,
                    limit=self.materialize_limit,
                    use_advisory_lock=True,
                )
            if not materializations:
                logger.debug("scheduler idle scheduler_id=%s", self.scheduler_id)
                return SchedulerRunResult(
                    scheduler_id=self.scheduler_id,
                    status="idle",
                    materializations=(),
                )

            logger.info(
                "scheduler materialized due schedules scheduler_id=%s materialized_count=%s outcomes=%s",
                self.scheduler_id,
                sum(1 for item in materializations if item.task_id is not None),
                [
                    {
                        "schedule_id": str(item.schedule_id),
                        "action": item.action,
                        "task_id": str(item.task_id) if item.task_id else None,
                        "reason": item.reason,
                    }
                    for item in materializations
                ],
            )
            return SchedulerRunResult(
                scheduler_id=self.scheduler_id,
                status="processed",
                materializations=materializations,
            )

    def run_forever(self) -> None:
        while True:
            self.run_once()
            time.sleep(self.poll_interval_seconds)


def default_scheduler_id() -> str:
    return f"{socket.gethostname()}-{os.getpid()}"


def _required_template_string(
    template: dict[str, Any],
    name: str,
    *,
    schedule: Schedule,
) -> str:
    value = template.get(name)
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise ValueError(f"schedule {schedule.id} task_template is missing {name!r}")


def _optional_template_string(template: dict[str, Any], name: str) -> str | None:
    value = template.get(name)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _schedule_delivery(
    schedule: Schedule,
    *,
    template: dict[str, Any],
) -> dict[str, str | None]:
    kind = schedule.delivery_kind or _legacy_delivery_kind(template)
    slack_channel_id = schedule.delivery_slack_channel_id or _optional_template_string(
        template,
        "slack_channel_id",
    )
    slack_user_id = (
        schedule.delivery_slack_user_id
        or _optional_template_string(template, "slack_user_id")
        or schedule.owner_slack_user_id
    )
    slack_thread_ts = schedule.delivery_slack_thread_ts
    if not slack_thread_ts and kind != "slack_channel":
        slack_thread_ts = _optional_template_string(template, "slack_thread_ts")
    if kind == "slack_channel":
        slack_thread_ts = None
    artifact_policy = (
        schedule.artifact_delivery_policy
        or _optional_template_string(template, "artifact_delivery_policy")
        or "message_only"
    )
    if not slack_channel_id:
        raise ValueError(f"schedule {schedule.id} delivery is missing slack_channel_id")
    if not slack_user_id:
        raise ValueError(f"schedule {schedule.id} delivery is missing slack_user_id")
    return {
        "kind": kind,
        "slack_channel_id": slack_channel_id,
        "slack_user_id": slack_user_id,
        "slack_thread_ts": slack_thread_ts,
        "artifact_policy": artifact_policy,
    }


def _legacy_delivery_kind(template: dict[str, Any]) -> str:
    surface = _optional_template_string(template, "delivery_surface")
    if surface == "thread":
        return "slack_thread"
    if surface == "channel":
        return "slack_channel"
    if surface == "dashboard":
        return "dashboard_only"
    return "slack_dm"


def _has_valid_run_budget(schedule: Schedule) -> bool:
    ceiling = schedule.planned_cost_ceiling_usd
    if ceiling is None:
        return False
    return Decimal(str(ceiling)) > 0


def _next_interval_after(schedule: Schedule, *, after: datetime) -> datetime:
    if schedule.interval_seconds is None or schedule.interval_seconds <= 0:
        raise ValueError(f"schedule {schedule.id} has invalid interval_seconds")
    next_run_at = _coerce_utc(schedule.next_run_at)
    interval = timedelta(seconds=schedule.interval_seconds)
    while next_run_at <= after:
        next_run_at += interval
    return next_run_at


def _cron_supported(schedule: Schedule) -> bool:
    try:
        _parse_simple_cron(schedule)
    except ValueError:
        return False
    return True


def _next_cron_after(schedule: Schedule, *, after: datetime) -> datetime:
    minute, hour, cron_weekdays = _parse_simple_cron(schedule)
    tzinfo = _schedule_tzinfo(schedule)
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
        if cron_weekdays is None or current.weekday() in cron_weekdays:
            return current.astimezone(UTC)
    raise ValueError(f"schedule {schedule.id} cron expression produced no next run")


def _parse_simple_cron(schedule: Schedule) -> tuple[int, int, frozenset[int] | None]:
    expr = (schedule.cron_expr or "").strip()
    fields = expr.split()
    if len(fields) != 5:
        raise ValueError(f"schedule {schedule.id} has unsupported cron expression")

    minute_field, hour_field, dom_field, month_field, weekday_field = fields
    if dom_field != "*" or month_field != "*":
        raise ValueError(f"schedule {schedule.id} has unsupported cron expression")
    try:
        minute = int(minute_field)
        hour = int(hour_field)
    except ValueError as exc:
        raise ValueError(
            f"schedule {schedule.id} has unsupported cron expression"
        ) from exc
    if not 0 <= minute <= 59 or not 0 <= hour <= 23:
        raise ValueError(f"schedule {schedule.id} has unsupported cron expression")

    return minute, hour, _parse_weekday_field(schedule, weekday_field)


def _parse_weekday_field(
    schedule: Schedule,
    value: str,
) -> frozenset[int] | None:
    if value == "*":
        return None
    weekdays: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            raise ValueError(f"schedule {schedule.id} has unsupported cron expression")
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = _cron_weekday_to_python(schedule, start_text)
            end = _cron_weekday_to_python(schedule, end_text)
            if start > end:
                raise ValueError(
                    f"schedule {schedule.id} has unsupported cron expression"
                )
            weekdays.update(range(start, end + 1))
        else:
            weekdays.add(_cron_weekday_to_python(schedule, part))
    return frozenset(weekdays)


def _cron_weekday_to_python(schedule: Schedule, value: str) -> int:
    try:
        cron_weekday = int(value)
    except ValueError as exc:
        raise ValueError(
            f"schedule {schedule.id} has unsupported cron expression"
        ) from exc
    if not 0 <= cron_weekday <= 6:
        raise ValueError(f"schedule {schedule.id} has unsupported cron expression")
    return (cron_weekday + 6) % 7


def _schedule_tzinfo(schedule: Schedule) -> ZoneInfo:
    try:
        return ZoneInfo(schedule.timezone or "UTC")
    except ZoneInfoNotFoundError:
        return ZoneInfo("UTC")


def _coerce_utc(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(UTC)
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _decimal_string(value: Decimal | None) -> str | None:
    if value is None:
        return None
    return str(value)


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entrypoint for the Postgres schedule materializer."""

    configure_logging()
    parser = argparse.ArgumentParser(description="Run the Kortny scheduler")
    parser.add_argument("--once", action="store_true", help="Run one scheduler tick")
    parser.add_argument(
        "--scheduler-id",
        default=None,
        help="Override scheduler id used in logs",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=None,
        help="Seconds to sleep between ticks",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum schedules to materialize per tick",
    )
    args = parser.parse_args(argv)

    settings = load_settings()
    configure_tracing(settings)

    worker = SchedulerWorker(
        scheduler_id=args.scheduler_id,
        poll_interval_seconds=args.poll_interval
        if args.poll_interval is not None
        else settings.scheduler_poll_interval_seconds,
        materialize_limit=args.limit
        if args.limit is not None
        else settings.scheduler_materialize_limit,
        advisory_lock_key=settings.scheduler_advisory_lock_key,
    )
    logger.info(
        "scheduler started scheduler_id=%s once=%s", worker.scheduler_id, args.once
    )
    if args.once:
        result = worker.run_once()
        print(
            f"scheduler_id={result.scheduler_id} status={result.status} "
            f"materialized_count={result.materialized_count}"
        )
        return

    worker.run_forever()
