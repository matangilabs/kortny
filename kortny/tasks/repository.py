"""SQLAlchemy-backed task domain repository."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import Select, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from kortny.db.models import LLMProvider, LLMUsage, Task, TaskEvent, TaskEventType
from kortny.db.models import TaskStatus as DbTaskStatus
from kortny.observability import sanitize_payload

TERMINAL_STATUSES = {
    DbTaskStatus.succeeded,
    DbTaskStatus.failed,
    DbTaskStatus.cancelled,
}
CANCELLABLE_STATUSES = {
    DbTaskStatus.pending,
    DbTaskStatus.running,
}


class TaskCancelledError(RuntimeError):
    """Raised when cooperative execution observes a cancelled task."""


class TaskRepository:
    """Repository for task, event, and usage persistence."""

    def __init__(self, session: Session, *, commit_after_write: bool = False) -> None:
        self.session = session
        self.commit_after_write = commit_after_write

    def create_task(
        self,
        *,
        installation_id: uuid.UUID,
        slack_channel_id: str,
        slack_user_id: str,
        input: str,
        slack_event_id: str | None = None,
        slack_thread_ts: str | None = None,
        slack_message_ts: str | None = None,
        parent_task_id: uuid.UUID | None = None,
    ) -> Task:
        """Create a task, returning the existing row for repeated Slack events."""

        if slack_event_id is not None:
            existing = self.get_by_slack_event_id(slack_event_id)
            if existing is not None:
                return existing

        task = Task(
            installation_id=installation_id,
            parent_task_id=parent_task_id,
            slack_event_id=slack_event_id,
            slack_channel_id=slack_channel_id,
            slack_thread_ts=slack_thread_ts,
            slack_message_ts=slack_message_ts,
            slack_user_id=slack_user_id,
            input=input,
            status=DbTaskStatus.pending,
        )

        try:
            with self.session.begin_nested():
                self.session.add(task)
                self.session.flush()
        except IntegrityError:
            if slack_event_id is None:
                raise
            existing = self.get_by_slack_event_id(slack_event_id)
            if existing is None:
                raise
            return existing

        self.append_event(
            task,
            TaskEventType.task_created,
            {
                "slack_event_id": slack_event_id,
                "slack_channel_id": slack_channel_id,
                "slack_thread_ts": slack_thread_ts,
                "slack_message_ts": slack_message_ts,
                "slack_user_id": slack_user_id,
            },
        )
        return task

    def get_task(self, task_id: uuid.UUID) -> Task | None:
        """Return a task by ID."""

        return self.session.scalar(select(Task).where(Task.id == task_id))

    def get_by_slack_event_id(self, slack_event_id: str) -> Task | None:
        """Return a task by Slack event ID."""

        return self.session.scalar(
            select(Task).where(Task.slack_event_id == slack_event_id)
        )

    def get_by_thread(self, channel: str, thread_ts: str) -> Task | None:
        """Return the newest task for a Slack channel/thread pair."""

        return self.session.scalar(
            select(Task)
            .where(
                Task.slack_channel_id == channel,
                Task.slack_thread_ts == thread_ts,
            )
            .order_by(Task.created_at.desc())
            .limit(1)
        )

    def get_by_slack_message(self, channel: str, message_ts: str) -> Task | None:
        """Return the newest task triggered by a Slack message."""

        return self.session.scalar(
            select(Task)
            .where(
                Task.slack_channel_id == channel,
                Task.slack_message_ts == message_ts,
            )
            .order_by(Task.created_at.desc())
            .limit(1)
        )

    def get_by_slack_reaction_target(
        self, channel: str, message_ts: str
    ) -> Task | None:
        """Return the newest task associated with a reacted Slack message."""

        direct = self.get_by_slack_message(channel, message_ts)
        posted = self.session.scalar(
            select(Task)
            .join(TaskEvent, TaskEvent.task_id == Task.id)
            .where(
                Task.slack_channel_id == channel,
                TaskEvent.type == TaskEventType.message_posted,
                TaskEvent.payload["message_ts"].as_string() == message_ts,
            )
            .order_by(TaskEvent.created_at.desc(), Task.created_at.desc())
            .limit(1)
        )
        if direct is None:
            return posted
        if posted is None:
            return direct
        if posted.created_at >= direct.created_at:
            return posted
        return direct

    def list_by_thread(self, channel: str, thread_ts: str) -> list[Task]:
        """Return tasks in a Slack channel/thread pair in creation order."""

        return list(
            self.session.scalars(
                select(Task)
                .where(
                    Task.slack_channel_id == channel,
                    Task.slack_thread_ts == thread_ts,
                )
                .order_by(Task.created_at, Task.id)
            )
        )

    def transition(self, task: Task | uuid.UUID, status: DbTaskStatus | str) -> Task:
        """Update task status and append a status_changed event."""

        task_obj = self._resolve_task(task)
        previous_status = _status_value(task_obj.status)
        next_status = DbTaskStatus(status)
        now = datetime.now(UTC)

        task_obj.status = next_status
        task_obj.updated_at = now
        if next_status is DbTaskStatus.running and task_obj.started_at is None:
            task_obj.started_at = now
        if next_status in TERMINAL_STATUSES:
            task_obj.finished_at = now

        self.session.flush()
        self.append_event(
            task_obj,
            TaskEventType.status_changed,
            {"from": previous_status, "to": next_status.value},
        )
        return task_obj

    def cancel_task(
        self,
        task: Task | uuid.UUID,
        *,
        by_user_id: str | None = None,
        reason: str = "reaction_cancel",
    ) -> Task | None:
        """Cancel a pending/running task and make the status terminal."""

        task_obj = self._resolve_task(task, for_update=True)
        previous_status = DbTaskStatus(task_obj.status)
        if previous_status not in CANCELLABLE_STATUSES:
            return None

        now = datetime.now(UTC)
        task_obj.status = DbTaskStatus.cancelled
        task_obj.finished_at = now
        task_obj.updated_at = now
        _clear_task_lease(task_obj)
        self.session.flush()
        self.append_event(
            task_obj,
            TaskEventType.status_changed,
            {
                "from": previous_status.value,
                "to": DbTaskStatus.cancelled.value,
                "reason": reason,
                "by_user_id": by_user_id,
            },
        )
        return task_obj

    def retry_failed_task(
        self,
        task: Task | uuid.UUID,
        *,
        by_user_id: str | None = None,
        reason: str = "reaction_retry",
        available_at: datetime | None = None,
    ) -> Task | None:
        """Requeue a failed task for a manual retry."""

        task_obj = self._resolve_task(task, for_update=True)
        previous_status = DbTaskStatus(task_obj.status)
        if previous_status is not DbTaskStatus.failed:
            return None

        retry_at = available_at or datetime.now(UTC)
        task_obj.status = DbTaskStatus.pending
        task_obj.attempts = 0
        task_obj.available_at = retry_at
        task_obj.started_at = None
        task_obj.finished_at = None
        task_obj.error = None
        task_obj.updated_at = retry_at
        _clear_task_lease(task_obj)
        self.session.flush()
        self.append_event(
            task_obj,
            TaskEventType.status_changed,
            {
                "from": previous_status.value,
                "to": DbTaskStatus.pending.value,
                "reason": reason,
                "by_user_id": by_user_id,
            },
        )
        return task_obj

    def raise_if_cancelled(
        self,
        task: Task | uuid.UUID,
        *,
        phase: str | None = None,
    ) -> None:
        """Refresh a task status and abort cooperative execution if cancelled."""

        task_obj = self._resolve_task(task)
        self.session.refresh(task_obj, attribute_names=["status"])
        if DbTaskStatus(task_obj.status) is DbTaskStatus.cancelled:
            suffix = f" during {phase}" if phase else ""
            raise TaskCancelledError(f"Task {task_obj.id} was cancelled{suffix}")

    def append_event(
        self,
        task: Task | uuid.UUID,
        event_type: TaskEventType | str,
        payload: dict[str, Any] | None = None,
    ) -> TaskEvent:
        """Append a task event with the next monotonic sequence number."""

        task_obj = self._resolve_task(task, for_update=True)
        event = TaskEvent(
            task_id=task_obj.id,
            seq=self._next_event_seq(task_obj.id),
            type=TaskEventType(event_type),
            payload=payload or {},
        )
        self.session.add(event)
        self.session.flush()
        self._commit_if_requested()
        return event

    def record_llm_usage(
        self,
        task: Task | uuid.UUID,
        *,
        provider: LLMProvider | str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: Decimal | int | str,
        event_id: int | None = None,
        model_tier: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LLMUsage:
        """Record LLM usage and refresh denormalized totals on the task."""

        if input_tokens < 0 or output_tokens < 0:
            raise ValueError("Token counts must be non-negative")

        task_obj = self._resolve_task(task, for_update=True)
        provider_value = LLMProvider(provider)
        cost = _coerce_decimal(cost_usd)
        if cost < 0:
            raise ValueError("LLM cost must be non-negative")

        if event_id is None:
            event_payload = {
                "message": "llm_call_completed",
                "provider": provider_value.value,
                "model": model,
                "model_tier": model_tier,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "cost_usd": str(cost),
            }
            if metadata:
                event_payload.update(sanitize_payload(metadata))
            event = self.append_event(
                task_obj,
                TaskEventType.llm_call,
                event_payload,
            )
            event_id = event.id

        usage = LLMUsage(
            task_id=task_obj.id,
            event_id=event_id,
            provider=provider_value,
            model=model,
            model_tier=model_tier,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )
        self.session.add(usage)
        self.session.flush()
        self._refresh_usage_rollup(task_obj)
        self._commit_if_requested()
        return usage

    def _resolve_task(
        self, task: Task | uuid.UUID, *, for_update: bool = False
    ) -> Task:
        if isinstance(task, Task):
            if task.id is None:
                self.session.flush()
            if not for_update:
                return task
            task_id = task.id
        else:
            task_id = task

        statement: Select[tuple[Task]] = select(Task).where(Task.id == task_id)
        if for_update:
            statement = statement.with_for_update()

        task_obj = self.session.scalar(statement)
        if task_obj is None:
            raise LookupError(f"Task not found: {task_id}")
        return task_obj

    def _next_event_seq(self, task_id: uuid.UUID) -> int:
        current_seq = self.session.scalar(
            select(func.coalesce(func.max(TaskEvent.seq), 0)).where(
                TaskEvent.task_id == task_id
            )
        )
        return int(current_seq or 0) + 1

    def _refresh_usage_rollup(self, task: Task) -> None:
        input_tokens, output_tokens, cost_usd = self.session.execute(
            select(
                func.coalesce(func.sum(LLMUsage.input_tokens), 0),
                func.coalesce(func.sum(LLMUsage.output_tokens), 0),
                func.coalesce(func.sum(LLMUsage.cost_usd), Decimal("0")),
            ).where(LLMUsage.task_id == task.id)
        ).one()

        task.total_input_tokens = int(input_tokens or 0)
        task.total_output_tokens = int(output_tokens or 0)
        task.total_cost_usd = _coerce_decimal(cost_usd or Decimal("0"))
        task.updated_at = datetime.now(UTC)
        self.session.flush()

    def _commit_if_requested(self) -> None:
        if self.commit_after_write:
            self.session.commit()


def _status_value(status: DbTaskStatus | str | None) -> str | None:
    if status is None:
        return None
    return DbTaskStatus(status).value


def _coerce_decimal(value: Decimal | int | str) -> Decimal:
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _clear_task_lease(task: Task) -> None:
    task.locked_by = None
    task.locked_at = None
    task.lease_expires_at = None
