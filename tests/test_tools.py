from decimal import Decimal

import pytest

from kortny.tools import (
    DescribeToolsTool,
    DuplicateToolError,
    EchoTool,
    ForgetFactTool,
    InspectMemoryTool,
    RememberFactTool,
    ToolArtifact,
    ToolNotFoundError,
    ToolRegistry,
    ToolResult,
)
from kortny.tools.catalog import tool_descriptor_from_class, tool_metadata
from kortny.tools.types import JsonObject, JsonSchema


class CostingTool:
    name = "costing"
    description = "Returns a cost and artifact."
    parameters: JsonSchema = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }

    def invoke(self, args: JsonObject) -> ToolResult:
        return ToolResult(
            output={"ok": True},
            cost_usd=Decimal("0.012345"),
            artifacts=(
                ToolArtifact(
                    filename="report.pdf",
                    path="/tmp/report.pdf",
                    mime_type="application/pdf",
                    size_bytes=42,
                ),
            ),
        )


def test_echo_tool_invokes_through_registry() -> None:
    registry = ToolRegistry([EchoTool()])

    result = registry.invoke("echo", {"message": "hello"})

    assert result == ToolResult(output={"message": "hello"})


def test_registry_exposes_provider_neutral_schemas() -> None:
    registry = ToolRegistry([EchoTool()])

    assert registry.schemas() == (
        {
            "name": "echo",
            "description": "Echoes a message back unchanged.",
            "parameters": EchoTool.parameters,
        },
    )


def test_registry_exposes_metadata_rich_descriptors() -> None:
    registry = ToolRegistry([EchoTool()])

    descriptor = registry.descriptors()[0]

    assert descriptor.name == "echo"
    assert descriptor.namespace == "native.diagnostics"
    assert descriptor.category == "Diagnostics"
    assert descriptor.side_effect == "read"
    assert descriptor.enabled is True
    assert descriptor.required_args == ("message",)
    assert descriptor.to_payload()["parameters"] == EchoTool.parameters


def test_native_tool_metadata_captures_safety_and_scope() -> None:
    metadata = tool_metadata("slack_channel_history")

    assert metadata.namespace == "native.slack"
    assert metadata.side_effect == "read"
    assert "channels:history" in metadata.required_slack_scopes
    assert "slack_rate_limited" in metadata.plan_gates


def test_observed_slack_search_metadata_is_local_and_read_only() -> None:
    metadata = tool_metadata("search_observed_slack_history")

    assert metadata.namespace == "native.slack"
    assert metadata.category == "Slack context"
    assert metadata.side_effect == "read"
    assert metadata.required_env_vars == ("POSTGRES_URL",)
    assert "observed_history_search" in metadata.capabilities


def test_describe_tools_metadata_is_read_only_runtime_inventory() -> None:
    descriptor = tool_descriptor_from_class(DescribeToolsTool)

    assert descriptor.name == "describe_tools"
    assert descriptor.category == "Runtime"
    assert descriptor.side_effect == "read"
    assert "tool_inventory" in descriptor.capabilities


def test_remember_fact_tool_schema_requires_faithful_memory_details() -> None:
    value_text_description = RememberFactTool.parameters["properties"]["value_text"][
        "description"
    ]

    assert "Preserve every actionable detail" in RememberFactTool.description
    assert "footer/header placement" in RememberFactTool.description
    assert "placement details like footer left" in value_text_description


def test_memory_control_tool_schemas_are_user_trust_focused() -> None:
    assert "what Kortny remembers" in InspectMemoryTool.description
    assert "provenance" in InspectMemoryTool.description
    assert "audit-preserving soft delete" in ForgetFactTool.description


def test_registry_rejects_duplicate_tool_names() -> None:
    registry = ToolRegistry([EchoTool()])

    with pytest.raises(DuplicateToolError):
        registry.register(EchoTool())


def test_registry_reports_missing_tools() -> None:
    registry = ToolRegistry()

    with pytest.raises(ToolNotFoundError):
        registry.invoke("missing", {})


def test_tool_result_carries_cost_and_artifacts() -> None:
    registry = ToolRegistry([CostingTool()])

    result = registry.invoke("costing", {})

    assert result.output == {"ok": True}
    assert result.cost_usd == Decimal("0.012345")
    assert result.artifacts == (
        ToolArtifact(
            filename="report.pdf",
            path="/tmp/report.pdf",
            mime_type="application/pdf",
            size_bytes=42,
        ),
    )


def test_echo_tool_validates_required_message() -> None:
    registry = ToolRegistry([EchoTool()])

    with pytest.raises(ValueError, match="string 'message'"):
        registry.invoke("echo", {"message": 123})
