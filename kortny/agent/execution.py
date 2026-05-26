"""Execution plan and guardrail primitives for the agent coordinator."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, field
from decimal import Decimal
from enum import StrEnum
from typing import Any

from kortny.tools.types import JsonObject


class ExecutionMode(StrEnum):
    """Coordinator execution modes."""

    inline = "inline"
    planned = "planned"


class ExecutionPlanStatus(StrEnum):
    """Lifecycle state for a task execution plan."""

    pending = "pending"
    in_progress = "in_progress"
    completed = "completed"
    failed = "failed"
    blocked = "blocked"


class ExecutionStepStatus(StrEnum):
    """Lifecycle state for a single execution step."""

    pending = "pending"
    in_progress = "in_progress"
    completed = "completed"
    failed = "failed"
    skipped = "skipped"


@dataclass(frozen=True, slots=True)
class ExecutionGuardrailLimits:
    """Runtime limits that keep the coordinator bounded."""

    max_turns: int = 6
    max_tool_calls: int = 12
    max_recoverable_failures: int = 4
    max_same_tool_call: int = 2
    max_same_recoverable_error: int = 2

    def __post_init__(self) -> None:
        if self.max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        if self.max_tool_calls < 1:
            raise ValueError("max_tool_calls must be at least 1")
        if self.max_recoverable_failures < 0:
            raise ValueError("max_recoverable_failures cannot be negative")
        if self.max_same_tool_call < 1:
            raise ValueError("max_same_tool_call must be at least 1")
        if self.max_same_recoverable_error < 1:
            raise ValueError("max_same_recoverable_error must be at least 1")


@dataclass(slots=True)
class ExecutionStep:
    """A coordinator-owned unit of work."""

    step_id: str
    description: str
    status: ExecutionStepStatus = ExecutionStepStatus.pending
    selected_tool_names: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    tool_call_count: int = 0
    recoverable_failure_count: int = 0
    observations: list[str] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)

    def to_payload(self) -> JsonObject:
        """Return a JSON-safe event payload."""

        payload = asdict(self)
        payload["status"] = self.status.value
        return payload


@dataclass(frozen=True, slots=True)
class ToolAttemptRecord:
    """Controller-owned metadata for one attempted tool invocation."""

    tool_name: str
    normalized_args_hash: str
    attempt_no: int
    status: str
    recoverable: bool = False
    error_code: str | None = None
    error_category: str | None = None
    recovery_action: str | None = None
    idempotency_key: str | None = None

    def to_payload(self) -> JsonObject:
        """Return a JSON-safe event payload."""

        return asdict(self)


@dataclass(slots=True)
class ExecutionBudgetState:
    """Mutable budget counters for a single task run."""

    tool_call_count: int = 0
    recoverable_failure_count: int = 0
    tool_call_signature_counts: dict[str, int] = field(default_factory=dict)
    recoverable_error_counts: dict[str, int] = field(default_factory=dict)

    def record_tool_attempt(
        self,
        *,
        task_id: uuid.UUID,
        step_id: str,
        tool_name: str,
        arguments: JsonObject,
    ) -> ToolAttemptRecord:
        """Track a tool attempt before the external call is made."""

        normalized_args_hash = normalized_tool_args_hash(arguments)
        signature_key = tool_signature_key(
            tool_name=tool_name,
            normalized_args_hash=normalized_args_hash,
        )
        attempt_no = self.tool_call_signature_counts.get(signature_key, 0) + 1
        self.tool_call_signature_counts[signature_key] = attempt_no
        self.tool_call_count += 1

        return ToolAttemptRecord(
            tool_name=tool_name,
            normalized_args_hash=normalized_args_hash,
            attempt_no=attempt_no,
            status="started",
            idempotency_key=_idempotency_key(
                task_id=task_id,
                step_id=step_id,
                tool_name=tool_name,
                normalized_args_hash=normalized_args_hash,
            ),
        )

    def record_recoverable_failure(
        self,
        *,
        tool_name: str,
        normalized_args_hash: str,
        error_code: str,
        error_category: str,
    ) -> int:
        """Track a recoverable failure and return same-error attempt count."""

        error_key = recoverable_error_key(
            tool_name=tool_name,
            normalized_args_hash=normalized_args_hash,
            error_code=error_code,
            error_category=error_category,
        )
        count = self.recoverable_error_counts.get(error_key, 0) + 1
        self.recoverable_error_counts[error_key] = count
        self.recoverable_failure_count += 1
        return count

    def remaining(self, limits: ExecutionGuardrailLimits) -> JsonObject:
        """Return current budget headroom."""

        return {
            "max_turns": limits.max_turns,
            "max_tool_calls": limits.max_tool_calls,
            "max_recoverable_failures": limits.max_recoverable_failures,
            "max_same_tool_call": limits.max_same_tool_call,
            "max_same_recoverable_error": limits.max_same_recoverable_error,
            "tool_calls_remaining": max(
                0, limits.max_tool_calls - self.tool_call_count
            ),
            "recoverable_failures_remaining": max(
                0, limits.max_recoverable_failures - self.recoverable_failure_count
            ),
        }


@dataclass(slots=True)
class ExecutionPlan:
    """Private execution plan envelope for a coordinator run."""

    task_id: str
    user_query_summary: str
    steps: list[ExecutionStep]
    limits: ExecutionGuardrailLimits
    plan_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    plan_version: int = 1
    mode: ExecutionMode = ExecutionMode.inline
    status: ExecutionPlanStatus = ExecutionPlanStatus.pending
    current_step_id: str | None = None
    budget: ExecutionBudgetState = field(default_factory=ExecutionBudgetState)

    @property
    def current_step(self) -> ExecutionStep:
        """Return the active step."""

        if self.current_step_id is not None:
            for step in self.steps:
                if step.step_id == self.current_step_id:
                    return step
        if not self.steps:
            raise ValueError("execution plan has no steps")
        return self.steps[0]

    def start(self) -> ExecutionStep:
        """Mark the plan and first step as in progress."""

        self.status = ExecutionPlanStatus.in_progress
        step = self.current_step
        step.status = ExecutionStepStatus.in_progress
        self.current_step_id = step.step_id
        return step

    def complete(self) -> ExecutionStep:
        """Mark the plan and active step as completed."""

        step = self.current_step
        step.status = ExecutionStepStatus.completed
        self.status = ExecutionPlanStatus.completed
        return step

    def fail(self, error: dict[str, Any] | None = None) -> ExecutionStep:
        """Mark the plan and active step as failed."""

        step = self.current_step
        step.status = ExecutionStepStatus.failed
        if error is not None:
            step.errors.append(error)
        self.status = ExecutionPlanStatus.failed
        return step

    def to_payload(self) -> JsonObject:
        """Return a compact JSON-safe representation."""

        return {
            "plan_id": self.plan_id,
            "plan_version": self.plan_version,
            "task_id": self.task_id,
            "mode": self.mode.value,
            "status": self.status.value,
            "user_query_summary": self.user_query_summary,
            "current_step_id": self.current_step_id,
            "limits": asdict(self.limits),
            "budget": {
                "tool_call_count": self.budget.tool_call_count,
                "recoverable_failure_count": self.budget.recoverable_failure_count,
            },
            "steps": [step.to_payload() for step in self.steps],
        }


def make_default_execution_plan(
    *,
    task_id: uuid.UUID,
    user_input: str,
    selected_tool_names: list[str],
    limits: ExecutionGuardrailLimits,
) -> ExecutionPlan:
    """Create the initial deterministic inline plan for a task."""

    return ExecutionPlan(
        task_id=str(task_id),
        user_query_summary=_summarize_user_input(user_input),
        limits=limits,
        steps=[
            ExecutionStep(
                step_id="step-1",
                description="Handle the Slack request using available context and tools.",
                selected_tool_names=selected_tool_names,
            )
        ],
    )


def normalized_tool_args_hash(arguments: JsonObject) -> str:
    """Return a stable hash for tool arguments."""

    canonical = json.dumps(
        arguments,
        default=_json_default,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def tool_signature_key(*, tool_name: str, normalized_args_hash: str) -> str:
    """Return the signature key used for same-call circuit breaking."""

    return f"{tool_name}:{normalized_args_hash}"


def recoverable_error_key(
    *,
    tool_name: str,
    normalized_args_hash: str,
    error_code: str,
    error_category: str,
) -> str:
    """Return the signature key used for repeated recoverable failures."""

    return f"{tool_name}:{error_category}:{error_code}:{normalized_args_hash}"


def _idempotency_key(
    *,
    task_id: uuid.UUID,
    step_id: str,
    tool_name: str,
    normalized_args_hash: str,
) -> str:
    return f"{task_id}:{step_id}:{tool_name}:{normalized_args_hash}"


def _summarize_user_input(user_input: str) -> str:
    normalized = " ".join(user_input.split())
    if len(normalized) <= 240:
        return normalized
    return f"{normalized[:237]}..."


def _json_default(value: object) -> object:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, uuid.UUID):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
