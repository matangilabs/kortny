from decimal import Decimal
from types import SimpleNamespace

import pytest

from kortny.approvals import ApprovalScope, ToolApprovalPolicy
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


def test_resolve_slack_identity_metadata_is_local_and_read_only() -> None:
    metadata = tool_metadata("resolve_slack_identity")

    assert metadata.namespace == "native.slack"
    assert metadata.category == "Slack context"
    assert metadata.side_effect == "read"
    assert metadata.required_env_vars == ("POSTGRES_URL",)
    assert "slack_identity_resolution" in metadata.capabilities


def test_slack_identity_info_metadata_is_read_only_slack_refresh() -> None:
    user = tool_metadata("slack_user_info")
    channel = tool_metadata("slack_channel_info")

    assert user.category == "Slack context"
    assert user.side_effect == "read"
    assert user.required_slack_scopes == ("users:read",)
    assert "identity_cache_refresh" in user.plan_gates
    assert channel.category == "Slack context"
    assert channel.side_effect == "read"
    assert "channels:read" in channel.required_slack_scopes
    assert "current_channel_only" in channel.plan_gates


def test_slack_action_metadata_is_current_scope_and_write_classed() -> None:
    reply = tool_metadata("slack_reply_thread")
    reaction = tool_metadata("slack_add_reaction")
    pin = tool_metadata("slack_pin_message")
    bookmark = tool_metadata("slack_add_bookmark")

    assert reply.category == "Slack actions"
    assert reply.side_effect == "write"
    assert reply.required_slack_scopes == ("chat:write",)
    assert "current_thread_only" in reply.plan_gates
    assert reaction.category == "Slack actions"
    assert reaction.side_effect == "write"
    assert reaction.required_slack_scopes == ("reactions:write",)
    assert "current_message_only" in reaction.plan_gates
    assert pin.category == "Slack actions"
    assert pin.side_effect == "write"
    assert pin.required_slack_scopes == ("pins:write",)
    assert "current_message_only" in pin.plan_gates
    assert bookmark.category == "Slack actions"
    assert bookmark.side_effect == "write"
    assert bookmark.required_slack_scopes == ("bookmarks:write",)
    assert "current_channel_only" in bookmark.plan_gates


def test_slack_action_tools_do_not_require_human_approval_by_default() -> None:
    policy = ToolApprovalPolicy()

    reply = policy.requirement_for(
        SimpleNamespace(name="slack_reply_thread", description="Post a reply"),
        {},
    )
    reaction = policy.requirement_for(
        SimpleNamespace(name="slack_add_reaction", description="Add a reaction"),
        {},
    )
    pin = policy.requirement_for(
        SimpleNamespace(name="slack_pin_message", description="Pin a message"),
        {},
    )
    bookmark = policy.requirement_for(
        SimpleNamespace(name="slack_add_bookmark", description="Add a bookmark"),
        {},
    )

    assert reply.scope is ApprovalScope.none
    assert reaction.scope is ApprovalScope.none
    assert pin.scope is ApprovalScope.none
    assert bookmark.scope is ApprovalScope.none


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
