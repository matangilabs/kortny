"""Private execution-planning support for the coordinator."""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from kortny.agent.execution import (
    ExecutionGuardrailLimits,
    ExecutionMode,
    ExecutionPlan,
    ExecutionStep,
)
from kortny.db.models import Task
from kortny.llm import ChatMessage, Completion
from kortny.tools.types import JsonObject, JsonSchema

EXECUTION_PLANNER_RESPONSE_FORMAT: JsonObject = {"type": "json_object"}
EXECUTION_PLANNER_PROMPT_NAME = "kortny.execution_planner"

PLANNER_SYSTEM_PROMPT = """You are Kortny's private execution planner.

Return compact JSON only. Do not expose chain-of-thought. Create a practical
runtime plan for the coordinator, not a user-facing explanation.

Rules:
- Use only tool names from available_tools.
- Prefer discovery/list/search steps before tools that require unknown IDs.
- Include missing_inputs only when the runtime cannot discover the input.
- Include fallback_notes for alternative tools or narrower retries.
- Include risk_notes for side effects, privacy, scope, cost, or destructive risk.
- Keep steps concise and execution oriented.
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


def parse_planned_execution_payload(content: str | None) -> PlannedExecutionPayload:
    """Parse and validate planner JSON."""

    if content is None or not content.strip():
        raise ValueError("execution planner returned empty content")
    try:
        return PlannedExecutionPayload.model_validate_json(_extract_json_object(content))
    except (ValidationError, ValueError) as exc:
        raise ValueError("execution planner returned invalid JSON") from exc


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


def _needs_rich_context(intent_decision: Mapping[str, object]) -> bool:
    return any(
        intent_decision.get(field) is True
        for field in ("needs_channel_context", "needs_thread_context", "needs_file_context")
    )


def _likely_artifact_or_integration(tool_names: Sequence[str]) -> bool:
    return any(
        tool_name == "pdf_generator"
        or tool_name.startswith("composio_")
        or "_execute" in tool_name
        for tool_name in tool_names
    )


def _toolkit_name_mentioned(input_text: str, tool_schemas: Sequence[JsonSchema]) -> bool:
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
        tokens.extend("".join(char if char.isalnum() else " " for char in raw_token).split())
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
