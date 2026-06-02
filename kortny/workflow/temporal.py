"""Temporal workflow skeleton for HIG-97.

This module intentionally contains only deterministic workflow code and small
activity boundaries. The first production slice is to make Temporal optional
and bootable; routing real Slack tasks into this workflow comes after the
handoff events have been validated.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from temporalio import activity, workflow

JsonObject = dict[str, Any]

KORTNY_TEMPORAL_TASK_QUEUE = "kortny-workflows"


@dataclass(frozen=True, slots=True)
class KortnyWorkflowInput:
    """Serializable input for a durable Kortny workflow."""

    task_id: str
    installation_id: str
    slack_channel_id: str
    slack_thread_ts: str | None
    slack_user_id: str
    input: str

    def to_payload(self) -> JsonObject:
        return {
            "task_id": self.task_id,
            "installation_id": self.installation_id,
            "slack_channel_id": self.slack_channel_id,
            "slack_thread_ts": self.slack_thread_ts,
            "slack_user_id": self.slack_user_id,
            "input": self.input,
        }


@dataclass(frozen=True, slots=True)
class KortnyWorkflowResult:
    """Serializable result from a durable Kortny workflow."""

    task_id: str
    status: str
    summary: str

    def to_payload(self) -> JsonObject:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "summary": self.summary,
        }


@activity.defn
async def record_workflow_started_activity(payload: JsonObject) -> JsonObject:
    """First activity boundary for workflow bootstrap.

    Later slices will replace this with DB-backed task event recording. Keeping
    the first activity pure makes the Temporal worker bootable without giving it
    write authority over task state yet.
    """

    task_id = str(payload.get("task_id") or "")
    return {
        "task_id": task_id,
        "status": "accepted",
        "summary": "Kortny durable workflow skeleton accepted the task.",
    }


@workflow.defn
class KortnyTaskWorkflow:
    """Durable workflow envelope for long-running Kortny tasks."""

    def __init__(self) -> None:
        self._status = "created"
        self._summary = ""

    @workflow.run
    async def run(self, payload: dict[str, Any]) -> JsonObject:
        self._status = "running"
        result = await workflow.execute_activity(
            record_workflow_started_activity,
            payload,
            start_to_close_timeout=timedelta(seconds=30),
        )
        self._status = str(result.get("status") or "completed")
        self._summary = str(result.get("summary") or "")
        return result

    @workflow.query
    def progress(self) -> JsonObject:
        """Return minimal workflow progress for future dashboard queries."""

        return {
            "status": self._status,
            "summary": self._summary,
        }
