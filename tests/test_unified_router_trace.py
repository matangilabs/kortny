"""Trace + shadow-agreement coverage for the unified router (HIG-218)."""

import pytest

from kortny.routing.trace import RoutingDecisionTrace
from kortny.worker.agent_executor import _shadow_depth_agreement


def test_routing_trace_payload_includes_unified_depth_fields() -> None:
    payload = RoutingDecisionTrace(
        stage="worker_runtime_handoff",
        route_tier="handoff_shadow",
        source="runtime_handoff",
        response_depth="deep_workflow",
        time_sensitivity="relaxed",
        toolkit_affinity=("linear", "github"),
        depth_source="deterministic_override",
        shadow_depth_agreement=True,
    ).to_payload()

    assert payload["response_depth"] == "deep_workflow"
    assert payload["time_sensitivity"] == "relaxed"
    assert payload["toolkit_affinity"] == ["linear", "github"]
    assert payload["depth_source"] == "deterministic_override"
    assert payload["shadow_depth_agreement"] is True


def test_routing_trace_omits_unset_unified_depth_fields() -> None:
    payload = RoutingDecisionTrace(
        stage="tool_scope_selected",
        route_tier="tool_scope",
        source="tool_selector",
    ).to_payload()

    assert "response_depth" not in payload
    assert "time_sensitivity" not in payload
    assert "toolkit_affinity" not in payload
    assert "depth_source" not in payload
    assert "shadow_depth_agreement" not in payload


@pytest.mark.parametrize(
    ("shadow_runtime_class", "unified_depth", "expected"),
    [
        ("quick_response", "quick_response", True),
        ("inline_tool_task", "standard_tool_task", True),
        ("durable_workflow_task", "deep_workflow", True),
        ("scheduled_workflow_task", "deep_workflow", True),
        ("inline_tool_task", "deep_workflow", False),
        ("durable_workflow_task", "standard_tool_task", False),
        ("quick_response", "standard_tool_task", False),
    ],
)
def test_shadow_depth_agreement_mapping(
    shadow_runtime_class: str,
    unified_depth: str,
    expected: bool,
) -> None:
    assert (
        _shadow_depth_agreement(
            shadow_runtime_class=shadow_runtime_class,
            unified_depth=unified_depth,
        )
        is expected
    )
