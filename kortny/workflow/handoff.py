"""Runtime handoff decisions before task execution.

This is the observable boundary for HIG-97. It classifies which tasks are
likely to need a durable workflow backend; later slices can promote the shadow
Temporal launch path into primary execution ownership.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from kortny.config import Settings
from kortny.db.models import Task
from kortny.schedule_intent import is_schedule_state_question
from kortny.tools.types import JsonObject


class TaskRuntimeClass(StrEnum):
    """Coarse execution class for a Kortny task."""

    quick_response = "quick_response"
    inline_tool_task = "inline_tool_task"
    durable_workflow_task = "durable_workflow_task"
    scheduled_workflow_task = "scheduled_workflow_task"


WorkflowBackend = Literal["inline", "temporal"]


@dataclass(frozen=True, slots=True)
class RuntimeHandoffDecision:
    """Decision emitted before an executor chooses a runtime path."""

    runtime_class: TaskRuntimeClass
    durable_candidate: bool
    recommended_backend: WorkflowBackend
    configured_backend: WorkflowBackend
    selected_backend: WorkflowBackend
    reason_codes: tuple[str, ...]
    reason: str
    fallback_reason: str | None = None

    def to_payload(self) -> JsonObject:
        payload: JsonObject = {
            "message": "runtime_handoff_evaluated",
            "runtime_class": self.runtime_class.value,
            "durable_candidate": self.durable_candidate,
            "recommended_backend": self.recommended_backend,
            "configured_backend": self.configured_backend,
            "selected_backend": self.selected_backend,
            "reason_codes": list(self.reason_codes),
            "reason": self.reason,
        }
        if self.fallback_reason:
            payload["fallback_reason"] = self.fallback_reason
        return payload


def evaluate_runtime_handoff(
    *,
    settings: Settings,
    task: Task,
) -> RuntimeHandoffDecision:
    """Classify task runtime needs without changing execution behavior yet."""

    normalized_input = _normalize(task.input)
    reason_codes = _reason_codes(task, normalized_input)

    if "schedule_state_query" in reason_codes:
        runtime_class = TaskRuntimeClass.inline_tool_task
    elif task.identity_kind == "scheduled" or "scheduled_or_recurring" in reason_codes:
        runtime_class = TaskRuntimeClass.scheduled_workflow_task
    elif _is_quick_response(normalized_input, reason_codes):
        runtime_class = TaskRuntimeClass.quick_response
    elif reason_codes:
        runtime_class = TaskRuntimeClass.durable_workflow_task
    else:
        runtime_class = TaskRuntimeClass.inline_tool_task

    durable_candidate = runtime_class in {
        TaskRuntimeClass.durable_workflow_task,
        TaskRuntimeClass.scheduled_workflow_task,
    }
    recommended_backend: WorkflowBackend = "temporal" if durable_candidate else "inline"
    configured_backend = settings.workflow_backend
    selected_backend: WorkflowBackend = "inline"
    fallback_reason = None
    if recommended_backend == "temporal" and configured_backend == "temporal":
        fallback_reason = "temporal_primary_execution_not_enabled"
    elif recommended_backend == "temporal":
        fallback_reason = "workflow_backend_inline"

    return RuntimeHandoffDecision(
        runtime_class=runtime_class,
        durable_candidate=durable_candidate,
        recommended_backend=recommended_backend,
        configured_backend=configured_backend,
        selected_backend=selected_backend,
        reason_codes=tuple(reason_codes),
        reason=_decision_reason(runtime_class, reason_codes),
        fallback_reason=fallback_reason,
    )


def _reason_codes(task: Task, normalized_input: str) -> list[str]:
    reasons: list[str] = []
    if task.identity_kind == "scheduled":
        reasons.append("scheduled_task_identity")
    if is_schedule_state_question(normalized_input):
        reasons.append("schedule_state_query")
        return reasons
    if _SCHEDULE_RE.search(normalized_input):
        reasons.append("scheduled_or_recurring")
    if _LONG_RUNNING_RE.search(normalized_input):
        reasons.append("long_running_work")
    if _MULTI_SOURCE_RE.search(normalized_input):
        reasons.append("multi_source_synthesis")
    if _INTEGRATION_RE.search(normalized_input):
        reasons.append("integration_tool_work")
    if _CRAWL_RE.search(normalized_input):
        reasons.append("crawl_or_scrape_work")
    if len(normalized_input) >= 320:
        reasons.append("large_request")
    return _dedupe(reasons)


def _is_quick_response(normalized_input: str, reason_codes: list[str]) -> bool:
    if reason_codes:
        return False
    if len(normalized_input) > 120:
        return False
    return bool(_QUICK_RESPONSE_RE.fullmatch(normalized_input))


def _decision_reason(
    runtime_class: TaskRuntimeClass,
    reason_codes: list[str],
) -> str:
    if runtime_class is TaskRuntimeClass.quick_response:
        return "Short conversational request with no durable workflow signals."
    if runtime_class is TaskRuntimeClass.inline_tool_task:
        if "schedule_state_query" in reason_codes:
            return "User is asking about existing scheduler state; keep the scheduler truth lookup inline."
        return "No durable workflow signals detected; keep current inline worker path."
    if runtime_class is TaskRuntimeClass.scheduled_workflow_task:
        return "Scheduled or recurring work should eventually be owned by a durable workflow."
    return (
        "Task has long-running, multi-source, integration, crawl, or synthesis "
        "signals that should eventually move to a durable workflow backend."
    )


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip()


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


_QUICK_RESPONSE_RE = re.compile(
    r"(?:yo|hey|hi|hello|sup|what'?s up|are you up|you up|ping|"
    r"what can you do|what tools do you have|what integrations do you have)\??"
)
_SCHEDULE_RE = re.compile(
    r"\b(every|daily|weekly|monthly|schedule|scheduled|recurring|cron|"
    r"each\s+(?:day|week|month|monday|tuesday|wednesday|thursday|friday))\b"
)
_LONG_RUNNING_RE = re.compile(
    r"\b(long[- ]running|deep research|comprehensive|full report|detailed report|"
    r"monitor|keep checking|when (?:it|this) (?:finishes|is done)|"
    r"over the next|background)\b"
)
_MULTI_SOURCE_RE = re.compile(
    r"\b(compare|synthesize|cross[- ]check|triangulate|combine|"
    r"from multiple|across (?:multiple|all)|using .* and .*)\b"
)
_INTEGRATION_RE = re.compile(
    r"\b(linear|notion|firecrawl|serpapi|alpha vantage|github|google drive|"
    r"slack files?|crm|calendar|email|docs?)\b"
)
_CRAWL_RE = re.compile(r"\b(crawl|scrape|website|site map|sitemap|all pages)\b")
