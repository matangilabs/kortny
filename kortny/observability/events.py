"""Structured local observability helpers.

The first observability layer is intentionally small: consistent Docker log
lines plus durable task_events rows. OTEL/Langfuse adapters can build on the
same event names and payloads without leaking vendor-specific calls through the
agent code.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any, cast

from kortny.db.models import Task, TaskEvent, TaskEventType


def observe_task_event(
    task_service: Any,
    task: Task | uuid.UUID,
    event: str,
    *,
    event_type: TaskEventType | str = TaskEventType.log,
    logger: logging.Logger | None = None,
    level: int = logging.INFO,
    persist: bool = True,
    **fields: Any,
) -> TaskEvent | None:
    """Persist and log a structured task observation."""

    task_obj = task if isinstance(task, Task) else task_service.get_task(task)
    payload = observability_payload(event, task=task_obj, **fields)
    row: TaskEvent | None = None
    if persist:
        row = task_service.append_event(task, event_type, payload)
    if logger is not None:
        log_observation(
            logger,
            event,
            level=level,
            task=task_obj,
            event_type=str(event_type),
            event_id=row.id if row is not None else None,
            event_seq=row.seq if row is not None else None,
            **fields,
        )
    return row


def observability_payload(
    event: str,
    *,
    task: Task | None = None,
    **fields: Any,
) -> dict[str, Any]:
    """Return a JSON-safe payload with common task correlation fields."""

    payload: dict[str, Any] = {"message": event}
    if task is not None:
        payload.update(_task_fields(task))
    payload.update(fields)
    return cast(dict[str, Any], sanitize_payload(payload))


def log_observation(
    logger: logging.Logger,
    event: str,
    *,
    level: int = logging.INFO,
    task: Task | None = None,
    **fields: Any,
) -> None:
    """Emit a consistent key=value log line for local Docker logs."""

    if not logger.isEnabledFor(level):
        return

    payload = sanitize_payload({**_task_fields(task), **fields} if task else fields)
    suffix = " ".join(
        f"{key}={_log_value(value)}"
        for key, value in payload.items()
        if value is not None
    )
    logger.log(level, "%s%s", event, f" {suffix}" if suffix else "")


def sanitize_payload(value: Any) -> Any:
    """Return a JSON-compatible value suitable for task_events payloads."""

    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return sanitize_payload(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): sanitize_payload(child)
            for key, child in value.items()
            if child is not None
        }
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [sanitize_payload(child) for child in value]
    return str(value)


def _task_fields(task: Task | None) -> dict[str, Any]:
    if task is None:
        return {}
    return {
        "task_id": task.id,
        "installation_id": task.installation_id,
        "slack_channel_id": task.slack_channel_id,
        "slack_thread_ts": task.slack_thread_ts,
        "slack_user_id": task.slack_user_id,
    }


def _log_value(value: Any) -> str:
    if isinstance(value, str):
        if value == "":
            return '""'
        if _simple_log_value(value):
            return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _simple_log_value(value: str) -> bool:
    return all(not char.isspace() and char not in {'"', "="} for char in value)
