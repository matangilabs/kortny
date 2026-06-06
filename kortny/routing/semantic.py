"""LLM-backed semantic routing in observe-only mode."""

from __future__ import annotations

import json
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from kortny.llm import ChatMessage, Completion
from kortny.tools.types import JsonObject, JsonSchema
from kortny.workflow.handoff import TaskRuntimeClass

SEMANTIC_ROUTER_PROMPT_NAME = "kortny.semantic_router.shadow"
SEMANTIC_ROUTER_PROMPT_VERSION = "hig_187_slice_2"
SEMANTIC_ROUTER_RESPONSE_FORMAT: JsonObject = {"type": "json_object"}
SEMANTIC_ROUTER_PROMOTION_MODE = "shadow_only"


class SemanticRouterParseError(ValueError):
    """Raised when the semantic router returns unusable structured output."""


class SemanticExecutionPath(StrEnum):
    """Execution path the semantic router would choose if it controlled routing."""

    inline = "inline"
    durable_workflow = "durable_workflow"
    scheduled_workflow = "scheduled_workflow"


@dataclass(frozen=True, slots=True)
class SemanticRouteRequest:
    """Minimal semantic context for route classification."""

    user_request: str
    surface: str
    identity_kind: str | None = None


@dataclass(frozen=True, slots=True)
class SemanticRouteDecision:
    """Structured observe-only route decision from the semantic router."""

    runtime_class: TaskRuntimeClass
    intent: str
    execution_path: SemanticExecutionPath
    confidence: float
    margin: float
    candidate_capabilities: tuple[str, ...]
    needs_clarification: bool
    reason: str

    def comparison_payload(
        self,
        *,
        handoff_runtime_class: str,
        handoff_recommended_backend: str,
        selected_backend: str,
        planned_classifier_route: str | None,
        planned_candidate: bool | None,
    ) -> JsonObject:
        """Return trace metadata comparing the shadow route to current routing."""

        handoff_execution_path = (
            SemanticExecutionPath.durable_workflow.value
            if handoff_recommended_backend == "temporal"
            else SemanticExecutionPath.inline.value
        )
        return {
            "behavior": "observe_only",
            "prompt_version": SEMANTIC_ROUTER_PROMPT_VERSION,
            "execution_path": self.execution_path.value,
            "candidate_capabilities": list(self.candidate_capabilities),
            "needs_clarification": self.needs_clarification,
            "handoff_runtime_class": handoff_runtime_class,
            "handoff_recommended_backend": handoff_recommended_backend,
            "handoff_execution_path": handoff_execution_path,
            "selected_backend": selected_backend,
            "planned_classifier_route": planned_classifier_route,
            "planned_candidate": planned_candidate,
            "runtime_disagreement": (
                self.runtime_class.value != handoff_runtime_class
            ),
            "execution_path_disagreement": (
                self.execution_path.value != handoff_execution_path
            ),
            "selected_backend_disagreement": (
                selected_backend == "inline"
                and self.execution_path is not SemanticExecutionPath.inline
            ),
        }


@dataclass(frozen=True, slots=True)
class SemanticRouterPromotionDecision:
    """Whether a shadow route is eligible to control execution later."""

    threshold_eligible: bool
    control_allowed: bool
    mode: str
    min_confidence: float
    min_margin: float
    reason_codes: tuple[str, ...]

    def to_payload(self) -> JsonObject:
        """Return compact promotion-gate metadata for task traces."""

        return {
            "mode": self.mode,
            "threshold_eligible": self.threshold_eligible,
            "control_allowed": self.control_allowed,
            "min_confidence": self.min_confidence,
            "min_margin": self.min_margin,
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True, slots=True)
class SemanticRouterPromotionGate:
    """Evaluate whether shadow semantic routing is mature enough to control."""

    min_confidence: float = 0.85
    min_margin: float = 0.20
    control_enabled: bool = False
    mode: str = SEMANTIC_ROUTER_PROMOTION_MODE

    def evaluate(
        self,
        decision: SemanticRouteDecision,
    ) -> SemanticRouterPromotionDecision:
        """Return promotion eligibility without changing runtime behavior."""

        reason_codes: list[str] = []
        if decision.confidence < self.min_confidence:
            reason_codes.append("below_min_confidence")
        if decision.margin < self.min_margin:
            reason_codes.append("below_min_margin")
        if decision.needs_clarification:
            reason_codes.append("needs_clarification")

        threshold_eligible = not reason_codes
        if threshold_eligible:
            reason_codes.append("thresholds_met")
        if not self.control_enabled:
            reason_codes.append("control_disabled_shadow_mode")

        return SemanticRouterPromotionDecision(
            threshold_eligible=threshold_eligible,
            control_allowed=threshold_eligible and self.control_enabled,
            mode=self.mode,
            min_confidence=self.min_confidence,
            min_margin=self.min_margin,
            reason_codes=tuple(reason_codes),
        )


class SemanticRouterLLM(Protocol):
    """Subset of LLMService needed by the shadow semantic router."""

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
        """Complete a structured semantic routing prompt."""


class LLMSemanticRouter:
    """Ask a cheap model what route it would take, without taking control."""

    def __init__(self, llm: SemanticRouterLLM) -> None:
        self.llm = llm

    def classify(
        self,
        *,
        task_id: uuid.UUID,
        request: SemanticRouteRequest,
    ) -> SemanticRouteDecision:
        completion = self.llm.complete(
            task_id=task_id,
            messages=_semantic_router_messages(request),
            response_format=SEMANTIC_ROUTER_RESPONSE_FORMAT,
            prompt_name=SEMANTIC_ROUTER_PROMPT_NAME,
        )
        return parse_semantic_route_decision(completion.content)


def parse_semantic_route_decision(content: str | None) -> SemanticRouteDecision:
    """Parse and validate a semantic router JSON response."""

    if not content:
        raise SemanticRouterParseError("Semantic router returned empty content.")
    try:
        payload = json.loads(_extract_json_object(content))
    except json.JSONDecodeError as exc:
        raise SemanticRouterParseError("Semantic router returned invalid JSON.") from exc
    if not isinstance(payload, dict):
        raise SemanticRouterParseError("Semantic router JSON must be an object.")

    runtime_class = _runtime_class(payload.get("runtime_class"))
    execution_path = _execution_path(payload.get("execution_path"))
    confidence = _probability(payload.get("confidence"), field="confidence")
    margin = _probability(payload.get("margin"), field="margin")
    intent = _required_text(payload.get("intent"), field="intent", max_chars=120)
    reason = _required_text(payload.get("reason"), field="reason", max_chars=400)
    candidate_capabilities = _string_tuple(payload.get("candidate_capabilities"))
    needs_clarification = payload.get("needs_clarification")
    if not isinstance(needs_clarification, bool):
        raise SemanticRouterParseError(
            "Semantic router field `needs_clarification` must be a boolean."
        )

    return SemanticRouteDecision(
        runtime_class=runtime_class,
        intent=intent,
        execution_path=execution_path,
        confidence=confidence,
        margin=margin,
        candidate_capabilities=candidate_capabilities,
        needs_clarification=needs_clarification,
        reason=reason,
    )


def _semantic_router_messages(request: SemanticRouteRequest) -> tuple[ChatMessage, ...]:
    return (
        ChatMessage(
            role="system",
            content=(
                "You are Kortny's shadow semantic router. Classify what runtime "
                "Kortny should use, but do not solve the task. Return JSON only. "
                "No markdown, no chain-of-thought, no prose outside JSON. "
                "Allowed runtime_class values: quick_response, inline_tool_task, "
                "durable_workflow_task, scheduled_workflow_task. Allowed "
                "execution_path values: inline, durable_workflow, "
                "scheduled_workflow. Use quick_response only for tiny direct "
                "conversation. Use inline_tool_task for bounded single-turn work, "
                "memory, scheduler state, Slack history, workspace graph, and "
                "single scoped read integrations. Use durable_workflow_task for "
                "multi-source synthesis, broad research, website audits/crawls, "
                "artifact generation with several inputs, long-running work, or "
                "write/destructive work needing approval. Use scheduled_workflow_task "
                "for creating or running recurring/background schedules. "
                "The JSON schema is: runtime_class string, intent string, "
                "execution_path string, confidence number 0..1, margin number 0..1, "
                "candidate_capabilities array of strings, needs_clarification "
                "boolean, reason string under 400 characters. Never use em "
                "dashes in JSON string values. Use commas, colons, semicolons, "
                "periods, or simple hyphens instead."
            ),
        ),
        ChatMessage(
            role="user",
            content=json.dumps(
                {
                    "identity_kind": request.identity_kind,
                    "surface": request.surface,
                    "user_request": request.user_request,
                },
                sort_keys=True,
            ),
        ),
    )


def _extract_json_object(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end <= start:
        raise SemanticRouterParseError("Semantic router response had no JSON object.")
    return stripped[start : end + 1]


def _runtime_class(value: object) -> TaskRuntimeClass:
    if not isinstance(value, str):
        raise SemanticRouterParseError("Semantic router field `runtime_class` missing.")
    try:
        return TaskRuntimeClass(value)
    except ValueError as exc:
        raise SemanticRouterParseError(
            f"Unsupported semantic runtime_class: {value}"
        ) from exc


def _execution_path(value: object) -> SemanticExecutionPath:
    if not isinstance(value, str):
        raise SemanticRouterParseError(
            "Semantic router field `execution_path` missing."
        )
    try:
        return SemanticExecutionPath(value)
    except ValueError as exc:
        raise SemanticRouterParseError(
            f"Unsupported semantic execution_path: {value}"
        ) from exc


def _probability(value: object, *, field: str) -> float:
    if not isinstance(value, int | float):
        raise SemanticRouterParseError(
            f"Semantic router field `{field}` must be a number."
        )
    number = float(value)
    if number < 0.0 or number > 1.0:
        raise SemanticRouterParseError(
            f"Semantic router field `{field}` must be between 0 and 1."
        )
    return number


def _required_text(value: object, *, field: str, max_chars: int) -> str:
    if not isinstance(value, str):
        raise SemanticRouterParseError(
            f"Semantic router field `{field}` must be a string."
        )
    text = " ".join(value.split()).strip()
    if not text:
        raise SemanticRouterParseError(
            f"Semantic router field `{field}` cannot be blank."
        )
    if len(text) > max_chars:
        return f"{text[: max_chars - 3].rstrip()}..."
    return text


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise SemanticRouterParseError(
            "Semantic router field `candidate_capabilities` must be an array."
        )
    items: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise SemanticRouterParseError(
                "Semantic router candidate capabilities must be strings."
            )
        normalized = " ".join(item.split()).strip()
        if normalized:
            items.append(normalized[:120])
    return tuple(items[:12])
