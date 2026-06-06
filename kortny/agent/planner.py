"""Private execution-planning support for the coordinator."""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from kortny.agent.error_policy import ClassifiedToolError, RecoveryAction
from kortny.agent.execution import (
    ExecutionGuardrailLimits,
    ExecutionMode,
    ExecutionPlan,
    ExecutionStep,
    RecoveryPlan,
)
from kortny.db.models import Task
from kortny.llm import ChatMessage, Completion
from kortny.tools.types import JsonObject, JsonSchema

EXECUTION_PLANNER_RESPONSE_FORMAT: JsonObject = {"type": "json_object"}
EXECUTION_PLANNER_PROMPT_NAME = "kortny.execution_planner"
RECOVERY_PLANNER_RESPONSE_FORMAT: JsonObject = {"type": "json_object"}
RECOVERY_PLANNER_PROMPT_NAME = "kortny.execution_recovery_planner"

PLANNER_SYSTEM_PROMPT = """You are Kortny's private execution planner.

Return compact JSON only. Do not expose chain-of-thought. Create a practical
runtime plan for the coordinator, not a user-facing explanation.

Rules:
- Use only tool names from available_tools.
- Prefer discovery/list/search steps before tools that require unknown IDs.
- Include missing_inputs only when the runtime cannot discover the input.
- Include fallback_notes for alternative tools or narrower retries.
- Include risk_notes for side effects, privacy, scope, cost, or destructive risk.
- Never use em dashes in JSON string values. Use commas, colons, semicolons,
  periods, or simple hyphens instead.
- Keep steps concise and execution oriented.
"""

RECOVERY_PLANNER_SYSTEM_PROMPT = """You are Kortny's private recovery planner.

Return compact JSON only. Do not expose chain-of-thought. A tool just returned a
recoverable failure. Decide the safest next move for the acting model.

Rules:
- Use only tool names from available_tools.
- Never suggest repeating the same failed tool call with the same arguments.
- If required IDs or references are missing, prefer discovery/list/search/history
  tools before asking the user.
- If an integration is unavailable, prefer a safe alternate tool when it can
  still answer the task.
- If policy or destructive-risk errors occur, stop safely.
- Never use em dashes in JSON string values. Use commas, colons, semicolons,
  periods, or simple hyphens instead.
- Keep guidance concise and execution oriented.
"""


class PlannerLLMClient(Protocol):
    """Subset of LLMService needed by the execution planner."""

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
        """Complete a private planner call."""


@dataclass(frozen=True, slots=True)
class PlannerGateDecision:
    """Whether a task should use the private planner."""

    should_plan: bool
    reason: str


class PlannedStepPayload(BaseModel):
    """One model-authored private execution step."""

    model_config = ConfigDict(extra="forbid")

    description: str = Field(min_length=1, max_length=240)
    selected_tool_names: list[str] = Field(default_factory=list, max_length=8)
    depends_on: list[str] = Field(default_factory=list, max_length=8)

    @field_validator("description")
    @classmethod
    def normalize_description(cls, value: str) -> str:
        return " ".join(value.split())

    @field_validator("selected_tool_names", "depends_on")
    @classmethod
    def normalize_string_list(cls, value: list[str]) -> list[str]:
        return _clean_string_list(value)


class PlannedExecutionPayload(BaseModel):
    """Validated JSON object returned by the private planner."""

    model_config = ConfigDict(extra="forbid")

    objective: str = Field(min_length=1, max_length=300)
    steps: list[PlannedStepPayload] = Field(min_length=1, max_length=6)
    missing_inputs: list[str] = Field(default_factory=list, max_length=8)
    fallback_notes: list[str] = Field(default_factory=list, max_length=8)
    risk_notes: list[str] = Field(default_factory=list, max_length=8)

    @field_validator("objective")
    @classmethod
    def normalize_objective(cls, value: str) -> str:
        return " ".join(value.split())

    @field_validator("missing_inputs", "fallback_notes", "risk_notes")
    @classmethod
    def normalize_notes(cls, value: list[str]) -> list[str]:
        return _clean_string_list(value)


class RecoveryPlanPayload(BaseModel):
    """Validated JSON object returned by the private recovery planner."""

    model_config = ConfigDict(extra="forbid")

    recovery_goal: str = Field(min_length=1, max_length=260)
    next_action: Literal[
        "patch_arguments",
        "use_discovery_tool",
        "switch_tool",
        "retry_narrower",
        "ask_user",
        "stop_safely",
        "continue_with_available_context",
    ]
    suggested_tool_names: list[str] = Field(default_factory=list, max_length=8)
    argument_notes: list[str] = Field(default_factory=list, max_length=8)
    fallback_notes: list[str] = Field(default_factory=list, max_length=8)
    risk_notes: list[str] = Field(default_factory=list, max_length=8)
    user_message_guidance: str | None = Field(default=None, max_length=260)

    @field_validator("recovery_goal")
    @classmethod
    def normalize_goal(cls, value: str) -> str:
        return " ".join(value.split())

    @field_validator("user_message_guidance")
    @classmethod
    def normalize_user_guidance(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = " ".join(value.split())
        return normalized or None

    @field_validator(
        "suggested_tool_names", "argument_notes", "fallback_notes", "risk_notes"
    )
    @classmethod
    def normalize_list(cls, value: list[str]) -> list[str]:
        return _clean_string_list(value)


class ExecutionPlanner:
    """Decides when to plan and asks the LLM for a private plan."""

    def should_plan(
        self,
        *,
        task: Task,
        tool_schemas: Sequence[JsonSchema],
        intent_decision: Mapping[str, object] | None,
    ) -> PlannerGateDecision:
        """Return whether this task deserves an LLM-authored plan."""

        if not tool_schemas:
            return PlannerGateDecision(False, "no_tools_available")

        if intent_decision is None:
            if _external_toolkit_name_mentioned(task.input, tool_schemas):
                return PlannerGateDecision(
                    True, "external_toolkit_mentioned_without_intent"
                )
            return PlannerGateDecision(False, "no_intent_signal")

        likely_tools = _string_list(intent_decision.get("likely_tools"))
        likely_available_tools = [
            tool for tool in likely_tools if tool in _schema_names(tool_schemas)
        ]
        if len(set(likely_available_tools)) >= 2:
            return PlannerGateDecision(True, "intent_likely_multi_tool")

        if _needs_rich_context(intent_decision) and likely_available_tools:
            return PlannerGateDecision(True, "intent_context_plus_tool")

        if len(likely_available_tools) == 1 and _single_hop_read_tool(
            likely_available_tools[0]
        ):
            return PlannerGateDecision(False, "single_hop_read_tool")

        if _likely_artifact_or_integration(likely_available_tools):
            return PlannerGateDecision(True, "intent_artifact_or_integration")

        if _toolkit_name_mentioned(task.input, tool_schemas):
            return PlannerGateDecision(True, "toolkit_mentioned")

        if intent_decision.get("model_tier") == "strong":
            return PlannerGateDecision(True, "intent_strong_model")

        return PlannerGateDecision(False, "simple_or_single_step")

    def create_plan(
        self,
        *,
        task: Task,
        llm: PlannerLLMClient,
        tool_schemas: Sequence[JsonSchema],
        limits: ExecutionGuardrailLimits,
        intent_decision: Mapping[str, object] | None,
        reason: str,
    ) -> ExecutionPlan:
        """Ask the model for a private plan and convert it to an ExecutionPlan."""

        completion = llm.complete(
            task_id=task.id,
            messages=(
                ChatMessage(role="system", content=PLANNER_SYSTEM_PROMPT),
                ChatMessage(
                    role="user",
                    content=json.dumps(
                        {
                            "task_input": task.input,
                            "intent_decision": intent_decision,
                            "available_tools": _tool_summary(tool_schemas),
                        },
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                ),
            ),
            tools=(),
            response_format=EXECUTION_PLANNER_RESPONSE_FORMAT,
            prompt_name=EXECUTION_PLANNER_PROMPT_NAME,
            prompt_source="code",
        )
        payload = parse_planned_execution_payload(completion.content)
        return planned_payload_to_execution_plan(
            task=task,
            payload=payload,
            limits=limits,
            available_tool_names=_schema_names(tool_schemas),
            reason=reason,
        )

    def create_recovery_plan(
        self,
        *,
        task: Task,
        llm: PlannerLLMClient,
        tool_schemas: Sequence[JsonSchema],
        plan: ExecutionPlan,
        failed_tool_name: str,
        attempted_arguments: JsonObject,
        classification: ClassifiedToolError,
        tool_output: JsonObject,
    ) -> RecoveryPlan:
        """Ask the model for private recovery guidance after a tool failure."""

        completion = llm.complete(
            task_id=task.id,
            messages=(
                ChatMessage(role="system", content=RECOVERY_PLANNER_SYSTEM_PROMPT),
                ChatMessage(
                    role="user",
                    content=json.dumps(
                        {
                            "task_input": task.input,
                            "current_plan": _compact_plan_payload(plan),
                            "failed_tool": failed_tool_name,
                            "attempted_argument_keys": sorted(attempted_arguments),
                            "attempted_arguments": attempted_arguments,
                            "error": classification.to_payload(),
                            "tool_output": _compact_json_object(tool_output),
                            "available_tools": _tool_summary(tool_schemas),
                            "budget_remaining": plan.budget.remaining(plan.limits),
                        },
                        default=str,
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                ),
            ),
            tools=(),
            response_format=RECOVERY_PLANNER_RESPONSE_FORMAT,
            prompt_name=RECOVERY_PLANNER_PROMPT_NAME,
            prompt_source="code",
        )
        payload = parse_recovery_plan_payload(completion.content)
        return recovery_payload_to_recovery_plan(
            payload=payload,
            failed_tool_name=failed_tool_name,
            classification=classification,
            available_tool_names=_schema_names(tool_schemas),
            planner_source="llm_recovery_planner",
        )


def parse_planned_execution_payload(content: str | None) -> PlannedExecutionPayload:
    """Parse and validate planner JSON."""

    if content is None or not content.strip():
        raise ValueError("execution planner returned empty content")
    try:
        return PlannedExecutionPayload.model_validate_json(
            _extract_json_object(content)
        )
    except (ValidationError, ValueError) as exc:
        raise ValueError("execution planner returned invalid JSON") from exc


def parse_recovery_plan_payload(content: str | None) -> RecoveryPlanPayload:
    """Parse and validate recovery planner JSON."""

    if content is None or not content.strip():
        raise ValueError("execution recovery planner returned empty content")
    try:
        return RecoveryPlanPayload.model_validate_json(_extract_json_object(content))
    except (ValidationError, ValueError) as exc:
        raise ValueError("execution recovery planner returned invalid JSON") from exc


def planned_payload_to_execution_plan(
    *,
    task: Task,
    payload: PlannedExecutionPayload,
    limits: ExecutionGuardrailLimits,
    available_tool_names: Sequence[str],
    reason: str,
) -> ExecutionPlan:
    """Build an ExecutionPlan from a validated planner payload."""

    available = set(available_tool_names)
    steps = [
        ExecutionStep(
            step_id=f"step-{index}",
            description=step.description,
            selected_tool_names=[
                tool for tool in step.selected_tool_names if tool in available
            ],
            depends_on=step.depends_on,
        )
        for index, step in enumerate(payload.steps, start=1)
    ]
    return ExecutionPlan(
        task_id=str(task.id),
        user_query_summary=payload.objective,
        limits=limits,
        steps=steps,
        mode=ExecutionMode.planned,
        planner_source="llm_planner",
        planner_reason=reason,
        missing_inputs=payload.missing_inputs,
        fallback_notes=payload.fallback_notes,
        risk_notes=payload.risk_notes,
    )


def recovery_payload_to_recovery_plan(
    *,
    payload: RecoveryPlanPayload,
    failed_tool_name: str,
    classification: ClassifiedToolError,
    available_tool_names: Sequence[str],
    planner_source: str,
) -> RecoveryPlan:
    """Build a RecoveryPlan from a validated recovery payload."""

    available = set(available_tool_names)
    return RecoveryPlan(
        failed_tool_name=failed_tool_name,
        error_code=classification.code,
        error_category=classification.category.value,
        recovery_action=classification.recovery_action.value,
        recovery_goal=payload.recovery_goal,
        next_action=payload.next_action,
        suggested_tool_names=[
            tool for tool in payload.suggested_tool_names if tool in available
        ],
        argument_notes=payload.argument_notes,
        fallback_notes=payload.fallback_notes,
        risk_notes=payload.risk_notes,
        user_message_guidance=payload.user_message_guidance,
        planner_source=planner_source,
    )


def make_fallback_recovery_plan(
    *,
    failed_tool_name: str,
    classification: ClassifiedToolError,
    available_tool_names: Sequence[str],
) -> RecoveryPlan:
    """Return deterministic recovery guidance when LLM replanning is unavailable."""

    next_action = _fallback_next_action(classification.recovery_action)
    suggested_tool_names = [
        tool
        for tool in available_tool_names
        if tool != failed_tool_name and _tool_matches_recovery(tool, classification)
    ][:3]
    return RecoveryPlan(
        failed_tool_name=failed_tool_name,
        error_code=classification.code,
        error_category=classification.category.value,
        recovery_action=classification.recovery_action.value,
        recovery_goal=classification.hint or classification.message,
        next_action=next_action,
        suggested_tool_names=suggested_tool_names,
        argument_notes=[
            "Do not repeat the failed tool call with the same arguments.",
            f"Follow recovery_action={classification.recovery_action.value}.",
        ],
        fallback_notes=[
            "Ask one concise clarification only if available tools cannot infer the missing input."
        ],
        risk_notes=[],
        user_message_guidance=(
            "Explain the blocker briefly and ask for the missing input."
            if classification.user_action_required
            else None
        ),
    )


def render_execution_plan_context(plan: ExecutionPlan) -> str:
    """Render a concise private plan message for the acting model."""

    lines = [
        "<private_execution_plan>",
        "Use this private runtime plan as scaffolding. Do not mention it unless the user asks how you approached the task.",
        f"objective: {plan.user_query_summary}",
        "steps:",
    ]
    for step in plan.steps:
        tools = ", ".join(step.selected_tool_names) or "none selected yet"
        lines.append(f"- {step.step_id}: {step.description} (candidate tools: {tools})")
    if plan.missing_inputs:
        lines.append("missing_inputs:")
        lines.extend(f"- {item}" for item in plan.missing_inputs)
    if plan.fallback_notes:
        lines.append("fallback_notes:")
        lines.extend(f"- {item}" for item in plan.fallback_notes)
    if plan.risk_notes:
        lines.append("risk_notes:")
        lines.extend(f"- {item}" for item in plan.risk_notes)
    lines.append("</private_execution_plan>")
    return "\n".join(lines)


def render_recovery_plan_context(recovery_plan: RecoveryPlan) -> str:
    """Render concise private recovery guidance for the acting model."""

    lines = [
        "<private_recovery_plan>",
        "A recoverable tool failure just happened. Use this private guidance before your next action. Do not mention it unless the user asks how you recovered.",
        f"failed_tool: {recovery_plan.failed_tool_name}",
        f"error_code: {recovery_plan.error_code}",
        f"error_category: {recovery_plan.error_category}",
        f"recovery_action: {recovery_plan.recovery_action}",
        f"next_action: {recovery_plan.next_action}",
        f"recovery_goal: {recovery_plan.recovery_goal}",
    ]
    if recovery_plan.suggested_tool_names:
        lines.append(
            "suggested_tools: " + ", ".join(recovery_plan.suggested_tool_names)
        )
    if recovery_plan.argument_notes:
        lines.append("argument_notes:")
        lines.extend(f"- {note}" for note in recovery_plan.argument_notes)
    if recovery_plan.fallback_notes:
        lines.append("fallback_notes:")
        lines.extend(f"- {note}" for note in recovery_plan.fallback_notes)
    if recovery_plan.risk_notes:
        lines.append("risk_notes:")
        lines.extend(f"- {note}" for note in recovery_plan.risk_notes)
    if recovery_plan.user_message_guidance:
        lines.append(f"user_message_guidance: {recovery_plan.user_message_guidance}")
    lines.append("</private_recovery_plan>")
    return "\n".join(lines)


def _tool_summary(tool_schemas: Sequence[JsonSchema]) -> list[JsonObject]:
    summary: list[JsonObject] = []
    for schema in tool_schemas:
        parameters = schema.get("parameters")
        required: object = None
        if isinstance(parameters, Mapping):
            required = parameters.get("required")
        summary.append(
            {
                "name": schema.get("name"),
                "description": schema.get("description"),
                "required_fields": required if isinstance(required, list) else [],
            }
        )
    return summary


def _compact_plan_payload(plan: ExecutionPlan) -> JsonObject:
    return {
        "plan_id": plan.plan_id,
        "plan_version": plan.plan_version,
        "mode": plan.mode.value,
        "status": plan.status.value,
        "objective": plan.user_query_summary,
        "current_step_id": plan.current_step_id,
        "steps": [
            {
                "step_id": step.step_id,
                "description": step.description,
                "status": step.status.value,
                "selected_tool_names": step.selected_tool_names,
                "tool_call_count": step.tool_call_count,
                "recoverable_failure_count": step.recoverable_failure_count,
                "recent_observations": step.observations[-3:],
            }
            for step in plan.steps
        ],
        "fallback_notes": plan.fallback_notes,
        "risk_notes": plan.risk_notes,
    }


def _compact_json_object(payload: JsonObject, *, max_chars: int = 2000) -> JsonObject:
    serialized = json.dumps(payload, default=str, separators=(",", ":"), sort_keys=True)
    if len(serialized) <= max_chars:
        return payload
    return {"truncated": True, "preview": serialized[:max_chars]}


def _needs_rich_context(intent_decision: Mapping[str, object]) -> bool:
    return any(
        intent_decision.get(field) is True
        for field in (
            "needs_channel_context",
            "needs_thread_context",
            "needs_file_context",
        )
    )


def _likely_artifact_or_integration(tool_names: Sequence[str]) -> bool:
    return any(
        tool_name == "pdf_generator"
        or tool_name.startswith("composio_")
        or "_execute" in tool_name
        for tool_name in tool_names
    )


def _single_hop_read_tool(tool_name: str) -> bool:
    normalized = tool_name.casefold()
    if normalized in {"web_search", "recall_fact", "inspect_memory"}:
        return True
    return normalized.startswith("composio_") and any(
        token in normalized
        for token in ("find", "get", "list", "query", "read", "retrieve", "search")
    )


def _fallback_next_action(recovery_action: RecoveryAction) -> str:
    if recovery_action is RecoveryAction.patch_arguments:
        return "patch_arguments"
    if recovery_action is RecoveryAction.resolve_reference:
        return "use_discovery_tool"
    if recovery_action is RecoveryAction.retry_with_backoff:
        return "retry_narrower"
    if recovery_action is RecoveryAction.switch_tool_or_broaden_query:
        return "switch_tool"
    if recovery_action in {RecoveryAction.ask_user, RecoveryAction.wait_auth}:
        return "ask_user"
    if recovery_action is RecoveryAction.stop_safely:
        return "stop_safely"
    return "continue_with_available_context"


def _tool_matches_recovery(tool_name: str, classification: ClassifiedToolError) -> bool:
    normalized = tool_name.casefold()
    if classification.recovery_action is RecoveryAction.resolve_reference:
        return any(
            token in normalized
            for token in ("search", "list", "history", "read", "lookup", "find")
        )
    if classification.recovery_action is RecoveryAction.retry_with_backoff:
        return "search" in normalized or "composio" in normalized
    if classification.recovery_action is RecoveryAction.switch_tool_or_broaden_query:
        return True
    if classification.recovery_action is RecoveryAction.patch_arguments:
        return normalized == classification.details.get("tool_name")
    return False


def _toolkit_name_mentioned(
    input_text: str, tool_schemas: Sequence[JsonSchema]
) -> bool:
    input_tokens = set(_tokens(input_text))
    if not input_tokens:
        return False
    for schema in tool_schemas:
        tool_name = schema.get("name")
        if not isinstance(tool_name, str):
            continue
        for token in _tokens(tool_name):
            if len(token) >= 4 and token in input_tokens:
                return True
    return False


def _external_toolkit_name_mentioned(
    input_text: str,
    tool_schemas: Sequence[JsonSchema],
) -> bool:
    external_schemas = [
        schema
        for schema in tool_schemas
        if isinstance((name := schema.get("name")), str)
        and (name.startswith("composio_") or "_execute" in name)
    ]
    return _toolkit_name_mentioned(input_text, external_schemas)


def _schema_names(tool_schemas: Sequence[JsonSchema]) -> tuple[str, ...]:
    return tuple(
        name
        for schema in tool_schemas
        if isinstance((name := schema.get("name")), str) and name
    )


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return _clean_string_list([item for item in value if isinstance(item, str)])


def _clean_string_list(value: Sequence[str]) -> list[str]:
    cleaned: list[str] = []
    for item in value:
        normalized = " ".join(item.split())
        if normalized and normalized not in cleaned:
            cleaned.append(normalized)
    return cleaned


def _tokens(value: str) -> list[str]:
    normalized = value.casefold().replace("-", "_")
    raw_tokens = normalized.replace("/", "_").replace(".", "_").split("_")
    tokens: list[str] = []
    for raw_token in raw_tokens:
        tokens.extend(
            "".join(char if char.isalnum() else " " for char in raw_token).split()
        )
    return tokens


def _extract_json_object(content: str) -> str:
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
