"""Observe-only planned workflow classification for HIG-179.

This classifier is deliberately separate from the runtime handoff path. Slice 0
records whether a task looks like a planned workflow candidate, but it does not
change execution behavior.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from kortny.db.models import Task, TaskEvent
from kortny.llm.routing import effective_intent_decision, latest_intent_decision
from kortny.schedule_intent import is_schedule_state_question
from kortny.tools.types import JsonObject


class PlannedWorkflowRoute(StrEnum):
    """Observe-only route recommendation before runtime execution."""

    inline = "inline"
    planned_candidate = "planned_candidate"


@dataclass(frozen=True, slots=True)
class PlannedWorkflowDecision:
    """Classifier result persisted to task events and observability logs."""

    route: PlannedWorkflowRoute
    confidence: float
    estimated_subtask_count: int
    reason_codes: tuple[str, ...]
    reason: str
    detected_integrations: tuple[str, ...]
    likely_tools: tuple[str, ...]
    needs_context: tuple[str, ...]

    @property
    def planned_candidate(self) -> bool:
        return self.route is PlannedWorkflowRoute.planned_candidate

    def to_payload(self) -> JsonObject:
        return {
            "message": "planned_workflow_classified",
            "classifier": "rules_plus_intent_metadata",
            "classifier_version": "hig_179_slice_0",
            "behavior": "observe_only",
            "route": self.route.value,
            "planned_candidate": self.planned_candidate,
            "confidence": self.confidence,
            "estimated_subtask_count": self.estimated_subtask_count,
            "reason_codes": list(self.reason_codes),
            "reason": self.reason,
            "detected_integrations": list(self.detected_integrations),
            "likely_tools": list(self.likely_tools),
            "needs_context": list(self.needs_context),
            "fallback_policy": "inline_on_low_confidence_or_classifier_failure",
        }


def classify_planned_workflow(
    *,
    task: Task,
    events: Sequence[TaskEvent] = (),
) -> PlannedWorkflowDecision:
    """Classify whether a task should later be handled by a planned workflow.

    The function is deterministic and cheap. It consumes the existing intent
    classifier metadata when present, which gives us trace visibility without
    adding another LLM call in the observe-only slice.
    """

    normalized_input = _normalize(task.input)
    intent_decision = effective_intent_decision(latest_intent_decision(events))
    likely_tools = _likely_tools(intent_decision)
    needs_context = _context_requirements(intent_decision, normalized_input)
    integrations = _detected_integrations(normalized_input, likely_tools)
    reason_codes = _reason_codes(
        task=task,
        normalized_input=normalized_input,
        likely_tools=likely_tools,
        needs_context=needs_context,
        detected_integrations=integrations,
    )

    if is_schedule_state_question(normalized_input):
        return PlannedWorkflowDecision(
            route=PlannedWorkflowRoute.inline,
            confidence=0.95,
            estimated_subtask_count=1,
            reason_codes=("schedule_state_query",),
            reason=(
                "User is asking for scheduler state; keep the schedule truth "
                "tool path inline instead of planned workflow."
            ),
            detected_integrations=integrations,
            likely_tools=likely_tools,
            needs_context=needs_context,
        )

    if _is_capability_lookup(normalized_input, likely_tools):
        return PlannedWorkflowDecision(
            route=PlannedWorkflowRoute.inline,
            confidence=0.94,
            estimated_subtask_count=1,
            reason_codes=("capability_lookup",),
            reason=(
                "User is asking what Kortny can do; answer with the bounded "
                "capability inventory path instead of planned workflow."
            ),
            detected_integrations=integrations,
            likely_tools=likely_tools,
            needs_context=needs_context,
        )

    if _is_quick_conversation(normalized_input, reason_codes):
        return PlannedWorkflowDecision(
            route=PlannedWorkflowRoute.inline,
            confidence=0.94,
            estimated_subtask_count=1,
            reason_codes=("quick_conversation",),
            reason="Short conversational request; keep the inline response path.",
            detected_integrations=integrations,
            likely_tools=likely_tools,
            needs_context=needs_context,
        )

    estimated_subtasks = _estimated_subtask_count(
        reason_codes=reason_codes,
        detected_integrations=integrations,
        likely_tools=likely_tools,
        needs_context=needs_context,
    )
    planned_reasons = _planned_reason_codes(
        reason_codes=reason_codes,
        estimated_subtask_count=estimated_subtasks,
        detected_integrations=integrations,
    )
    if planned_reasons:
        return PlannedWorkflowDecision(
            route=PlannedWorkflowRoute.planned_candidate,
            confidence=_planned_confidence(planned_reasons),
            estimated_subtask_count=estimated_subtasks,
            reason_codes=planned_reasons,
            reason=_planned_reason(planned_reasons),
            detected_integrations=integrations,
            likely_tools=likely_tools,
            needs_context=needs_context,
        )

    return PlannedWorkflowDecision(
        route=PlannedWorkflowRoute.inline,
        confidence=0.82,
        estimated_subtask_count=estimated_subtasks,
        reason_codes=tuple(reason_codes) or ("single_step_inline",),
        reason="No planning threshold crossed; keep the inline worker path.",
        detected_integrations=integrations,
        likely_tools=likely_tools,
        needs_context=needs_context,
    )


def _reason_codes(
    *,
    task: Task,
    normalized_input: str,
    likely_tools: tuple[str, ...],
    needs_context: tuple[str, ...],
    detected_integrations: tuple[str, ...],
) -> tuple[str, ...]:
    reasons: list[str] = []
    if task.identity_kind == "scheduled":
        reasons.append("scheduled_task_identity")
    if _SCHEDULE_RE.search(normalized_input):
        reasons.append("scheduled_or_recurring")
    if _WRITE_OR_DESTRUCTIVE_RE.search(normalized_input):
        reasons.append("write_or_destructive_intent")
    if _BROAD_RESEARCH_RE.search(normalized_input):
        reasons.append("broad_research")
    if _MULTI_SOURCE_RE.search(normalized_input):
        reasons.append("multi_source_synthesis")
    if _LONG_RUNNING_RE.search(normalized_input):
        reasons.append("long_running_or_monitoring")
    if _ARTIFACT_RE.search(normalized_input):
        reasons.append("artifact_or_document_output")
    if detected_integrations:
        reasons.append("integration_scope_present")
    if len(detected_integrations) >= 2:
        reasons.append("multi_integration_scope")
    if len(likely_tools) >= 2:
        reasons.append("multi_tool_likely")
    if needs_context:
        reasons.append("external_context_needed")
    if len(normalized_input) >= 420:
        reasons.append("large_request")
    return _dedupe(reasons)


def _planned_reason_codes(
    *,
    reason_codes: tuple[str, ...],
    estimated_subtask_count: int,
    detected_integrations: tuple[str, ...],
) -> tuple[str, ...]:
    planned: list[str] = []
    reason_set = set(reason_codes)
    hard_reasons = {
        "scheduled_task_identity",
        "scheduled_or_recurring",
        "write_or_destructive_intent",
        "long_running_or_monitoring",
        "large_request",
        "multi_tool_likely",
    }
    planned.extend(reason for reason in reason_codes if reason in hard_reasons)
    if len(detected_integrations) >= 2:
        planned.append("two_or_more_integration_scopes")
    if estimated_subtask_count >= 3 and (
        "broad_research" in reason_set
        or "multi_source_synthesis" in reason_set
        or "artifact_or_document_output" in reason_set
        or len(detected_integrations) >= 2
    ):
        if "broad_research" in reason_set:
            planned.append("broad_research")
        planned.append("estimated_three_or_more_subtasks")
    if {"broad_research", "multi_source_synthesis"} <= reason_set:
        planned.append("research_synthesis_work")
    return _dedupe(planned)


def _estimated_subtask_count(
    *,
    reason_codes: tuple[str, ...],
    detected_integrations: tuple[str, ...],
    likely_tools: tuple[str, ...],
    needs_context: tuple[str, ...],
) -> int:
    count = 1
    reason_set = set(reason_codes)
    if "broad_research" in reason_set:
        count += 2
    if "multi_source_synthesis" in reason_set:
        count += 1
    if "artifact_or_document_output" in reason_set:
        count += 1
    if "write_or_destructive_intent" in reason_set:
        count += 1
    if (
        "scheduled_or_recurring" in reason_set
        or "scheduled_task_identity" in reason_set
    ):
        count += 1
    count += min(3, len(detected_integrations))
    count += min(2, len(likely_tools))
    count += min(2, len(needs_context))
    return min(count, 10)


def _planned_confidence(reason_codes: tuple[str, ...]) -> float:
    hard = {
        "scheduled_task_identity",
        "scheduled_or_recurring",
        "write_or_destructive_intent",
        "long_running_or_monitoring",
        "two_or_more_integration_scopes",
    }
    if any(reason in hard for reason in reason_codes):
        return 0.9
    if "research_synthesis_work" in reason_codes:
        return 0.84
    return 0.78


def _planned_reason(reason_codes: tuple[str, ...]) -> str:
    if "write_or_destructive_intent" in reason_codes:
        return "Task may require write, destructive, or approval-sensitive steps; mark as a planned workflow candidate."
    if (
        "scheduled_or_recurring" in reason_codes
        or "scheduled_task_identity" in reason_codes
    ):
        return "Recurring or scheduled work should eventually enter the durable planned workflow path."
    if "two_or_more_integration_scopes" in reason_codes:
        return "Task mentions multiple integration scopes and should eventually be planned before execution."
    if "research_synthesis_work" in reason_codes or "broad_research" in reason_codes:
        return "Task looks like broad research plus synthesis; mark it for future planner handling."
    return "Task crosses the planned workflow threshold; observe only for now."


def _detected_integrations(
    normalized_input: str,
    likely_tools: tuple[str, ...],
) -> tuple[str, ...]:
    integrations: list[str] = []
    for name, pattern in _INTEGRATION_PATTERNS.items():
        if pattern.search(normalized_input):
            integrations.append(name)
    for tool in likely_tools:
        if tool.startswith("composio_"):
            integrations.append(tool.removeprefix("composio_"))
        if tool in _TOOL_TO_INTEGRATION:
            integrations.append(_TOOL_TO_INTEGRATION[tool])
    return tuple(_dedupe(integrations))


def _likely_tools(intent_decision: Mapping[str, Any] | None) -> tuple[str, ...]:
    if intent_decision is None:
        return ()
    value = intent_decision.get("likely_tools")
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return ()
    return tuple(_dedupe([item for item in value if isinstance(item, str) and item]))


def _context_requirements(
    intent_decision: Mapping[str, Any] | None,
    normalized_input: str,
) -> tuple[str, ...]:
    requirements: list[str] = []
    if intent_decision is not None:
        if intent_decision.get("needs_channel_context") is True:
            requirements.append("channel_context")
        if intent_decision.get("needs_thread_context") is True:
            requirements.append("thread_context")
        if intent_decision.get("needs_file_context") is True:
            requirements.append("file_context")
    if _CHANNEL_CONTEXT_RE.search(normalized_input):
        requirements.append("channel_context")
    if _FILE_CONTEXT_RE.search(normalized_input):
        requirements.append("file_context")
    return tuple(_dedupe(requirements))


def _is_quick_conversation(
    normalized_input: str,
    reason_codes: tuple[str, ...],
) -> bool:
    if reason_codes:
        return False
    if len(normalized_input) > 140:
        return False
    return bool(_QUICK_CONVERSATION_RE.fullmatch(normalized_input))


def _is_capability_lookup(
    normalized_input: str,
    likely_tools: tuple[str, ...],
) -> bool:
    if not _CAPABILITY_LOOKUP_RE.fullmatch(normalized_input):
        return False
    if not likely_tools:
        return True
    return set(likely_tools) <= CAPABILITY_LOOKUP_TOOL_HINTS


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip()


def _dedupe(values: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return tuple(result)


_QUICK_CONVERSATION_RE = re.compile(
    r"(?:yo\s+)?(?:hey\s+)?(?:kortny\s+)?(?:are you up|you up|ping|"
    r"what'?s up|what can you do|what tools do you have(?: access to)?|"
    r"what integrations do you have)\??"
)
_CAPABILITY_LOOKUP_RE = re.compile(
    r"(?:yo\s+)?(?:hey\s+)?(?:kortny\s+)?(?:what can you do|"
    r"what tools do you have(?: access to)?|what integrations do you have|"
    r"what capabilities do you have)\??"
)
CAPABILITY_LOOKUP_TOOL_HINTS = frozenset(
    {
        "capability_lookup",
        "get_capabilities",
        "describe_tools",
        "list_capabilities",
        "list_integrations",
        "list_tools",
        "native_tool_registry",
        "tool_metadata_lookup",
        "tool_registry",
    }
)
_SCHEDULE_RE = re.compile(
    r"\b(every|daily|weekly|monthly|schedule|scheduled|recurring|cron|"
    r"each\s+(?:day|week|month|monday|tuesday|wednesday|thursday|friday))\b"
)
_WRITE_OR_DESTRUCTIVE_RE = re.compile(
    r"\b(create|update|delete|remove|archive|send|post|publish|upload|write|"
    r"disconnect|revoke|approve|trigger|execute|run|generate)\b"
)
_BROAD_RESEARCH_RE = re.compile(
    r"\b(research|investigate|deep dive|look up|find current|latest|recent|"
    r"market map|landscape|benchmark|best .* tools?|compare .* tools?)\b"
)
_MULTI_SOURCE_RE = re.compile(
    r"\b(compare|synthesize|synthesise|cross[- ]check|triangulate|combine|"
    r"against|with our docs|from multiple|across (?:multiple|all)|using .* and .*)\b"
)
_LONG_RUNNING_RE = re.compile(
    r"\b(long[- ]running|comprehensive|full report|detailed report|monitor|"
    r"keep checking|over the next|background|until|when .* finishes)\b"
)
_ARTIFACT_RE = re.compile(
    r"\b(pdf|report|document|doc|deck|slides?|presentation|artifact|brief)\b"
)
_CHANNEL_CONTEXT_RE = re.compile(
    r"\b(this channel|channel context|above|earlier|thread|last few decisions)\b"
)
_FILE_CONTEXT_RE = re.compile(r"\b(file|files|attachment|attachments|csv|pdf)\b")
_INTEGRATION_PATTERNS = {
    "linear": re.compile(r"\blinear\b"),
    "notion": re.compile(r"\bnotion\b"),
    "slack": re.compile(r"\bslack|this channel|channel context\b"),
    "firecrawl": re.compile(r"\bfirecrawl|crawl|scrape\b"),
    "serpapi": re.compile(r"\bserpapi\b"),
    "alpha_vantage": re.compile(r"\balpha vantage|ticker|market data\b"),
    "github": re.compile(r"\bgithub|git hub|pr|pull request\b"),
    "docs": re.compile(r"\bdocs?|documentation\b"),
    "calendar": re.compile(r"\bcalendar|meeting\b"),
    "email": re.compile(r"\bemail|gmail|inbox\b"),
}
_TOOL_TO_INTEGRATION = {
    "web_search": "web",
    "slack_channel_history": "slack",
    "search_observed_slack_history": "slack",
    "resolve_slack_identity": "slack",
    "slack_file_read": "slack",
    "pdf_generator": "documents",
}
