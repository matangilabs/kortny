"""Structured task-event payloads for routing decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kortny.tools.types import JsonObject

ROUTING_DECISION_RECORDED_MESSAGE = "routing_decision_recorded"
ROUTING_CHAIN_COMPLETED_MESSAGE = "routing_chain_completed"


@dataclass(frozen=True, slots=True)
class RoutingDecisionTrace:
    """Serializable summary of one routing boundary decision."""

    stage: str
    route_tier: str
    source: str
    runtime_class: str | None = None
    intent: str | None = None
    confidence: float | None = None
    margin: float | None = None
    escalated: bool | None = None
    selected_runtime: str | None = None
    selected_backend: str | None = None
    actual_path: str | None = None
    reason: str | None = None
    reason_codes: tuple[str, ...] = ()
    candidate_tool_count: int | None = None
    selector_candidate_count: int | None = None
    selected_tool_names: tuple[str, ...] = ()
    suppressed_tool_names: tuple[str, ...] = ()
    shadow_runtime_class: str | None = None
    shadow_route: str | None = None
    shadow_planned_candidate: bool | None = None
    shadow_confidence: float | None = None
    metadata: JsonObject | None = None

    def to_payload(self) -> JsonObject:
        """Return a compact task-event payload."""

        payload: JsonObject = {
            "message": ROUTING_DECISION_RECORDED_MESSAGE,
            "stage": self.stage,
            "route_tier": self.route_tier,
            "route_tier_resolved": self.route_tier,
            "source": self.source,
        }
        _set_if_present(payload, "runtime_class", self.runtime_class)
        _set_if_present(payload, "intent", self.intent)
        _set_if_present(payload, "confidence", self.confidence)
        _set_if_present(payload, "margin", self.margin)
        _set_if_present(payload, "route_escalated", self.escalated)
        _set_if_present(payload, "selected_runtime", self.selected_runtime)
        _set_if_present(payload, "selected_backend", self.selected_backend)
        _set_if_present(payload, "actual_path", self.actual_path)
        _set_if_present(payload, "reason", self.reason)
        if self.reason_codes:
            payload["reason_codes"] = list(self.reason_codes)
        _set_if_present(payload, "candidate_tool_count", self.candidate_tool_count)
        _set_if_present(
            payload,
            "selector_candidate_count",
            self.selector_candidate_count,
        )
        if self.selected_tool_names:
            payload["selected_tool_names"] = list(self.selected_tool_names)
        if self.suppressed_tool_names:
            payload["suppressed_tool_names"] = list(self.suppressed_tool_names)
        _set_if_present(payload, "shadow_runtime_class", self.shadow_runtime_class)
        _set_if_present(payload, "shadow_route", self.shadow_route)
        _set_if_present(
            payload,
            "shadow_planned_candidate",
            self.shadow_planned_candidate,
        )
        _set_if_present(payload, "shadow_confidence", self.shadow_confidence)
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload


def _set_if_present(payload: JsonObject, key: str, value: Any | None) -> None:
    if value is not None:
        payload[key] = value
