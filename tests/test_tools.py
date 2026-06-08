from decimal import Decimal
from types import SimpleNamespace

import pytest

from kortny.approvals import ApprovalScope, ToolApprovalPolicy
from kortny.tools import (
    CodeExecTool,
    DescribeToolsTool,
    DuplicateToolError,
    EchoTool,
    ForgetFactTool,
    InspectMemoryTool,
    PdfGeneratorTool,
    RememberFactTool,
    ToolArtifact,
    ToolNotFoundError,
    ToolRegistry,
    ToolResult,
)
from kortny.tools.catalog import (
    NATIVE_TOOL_METADATA,
    dashboard_native_tool_names,
    low_risk_native_write_tool_names,
    native_slack_context_hint_names,
    native_tool_integration_map,
    native_tool_names_by_approval,
    read_only_native_tool_names,
    runtime_native_tool_names,
    tool_descriptor_from_class,
    tool_metadata,
)
from kortny.tools.list_integrations import _USER_CAPABILITY_GROUPS
from kortny.tools.native_runtime import native_tool_classes_by_name
from kortny.tools.types import JsonObject, JsonSchema
from kortny.worker.agent_executor import NATIVE_SLACK_CONTEXT_HINTS
from kortny.workflow.planning_classifier import _TOOL_TO_INTEGRATION


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
    assert descriptor.sandbox.requires_sandbox is False
    assert descriptor.to_payload()["sandbox"]["requires_sandbox"] is False


def test_native_tool_surfaces_are_derived_from_metadata() -> None:
    runtime_names = set(runtime_native_tool_names())
    class_names = set(native_tool_classes_by_name())

    assert class_names == runtime_names
    assert set(dashboard_native_tool_names()) <= runtime_names
    assert read_only_native_tool_names() == frozenset(
        name
        for name, metadata in NATIVE_TOOL_METADATA.items()
        if metadata.side_effect == "read"
    )
    assert low_risk_native_write_tool_names() == frozenset(
        name
        for name, metadata in NATIVE_TOOL_METADATA.items()
        if metadata.side_effect == "write" and metadata.approval == "none"
    )
    assert native_tool_names_by_approval("self_gated") == frozenset({"remember_fact"})
    assert native_tool_names_by_approval("user_approval") == frozenset({"forget_fact"})
    assert native_tool_names_by_approval("admin_approval") == frozenset({"code_exec"})
    assert native_tool_integration_map() == _TOOL_TO_INTEGRATION
    assert native_slack_context_hint_names() == NATIVE_SLACK_CONTEXT_HINTS


def test_native_capability_groups_cover_user_facing_metadata_categories() -> None:
    grouped_categories = {category for category, _payload in _USER_CAPABILITY_GROUPS}
    user_facing_categories = {
        metadata.category
        for metadata in NATIVE_TOOL_METADATA.values()
        if metadata.runtime_registered
        and metadata.dashboard_exposed
        and metadata.category != "Runtime"
    }

    assert user_facing_categories <= grouped_categories


def test_native_tool_metadata_captures_safety_and_scope() -> None:
    metadata = tool_metadata("slack_channel_history")

    assert metadata.namespace == "native.slack"
    assert metadata.side_effect == "read"
    assert "channels:history" in metadata.required_slack_scopes
    assert "slack_rate_limited" in metadata.plan_gates
    assert metadata.sandbox.requires_sandbox is False


def test_pdf_generator_stays_unsandboxed_until_runner_slice() -> None:
    metadata = tool_metadata("pdf_generator")
    descriptor = tool_descriptor_from_class(PdfGeneratorTool)

    assert metadata.sandbox.requires_sandbox is False
    assert metadata.sandbox.network == "none"
    assert descriptor.to_payload()["sandbox"] == metadata.sandbox.to_payload()


def test_code_exec_metadata_requires_sandbox_and_admin_approval() -> None:
    metadata = tool_metadata("code_exec")
    descriptor = tool_descriptor_from_class(CodeExecTool)
    policy = ToolApprovalPolicy()

    requirement = policy.requirement_for(
        SimpleNamespace(name="code_exec", description="Run code"),
        {"code": "print(1)"},
    )

    assert metadata.namespace == "native.execution"
    assert metadata.category == "Execution"
    assert metadata.side_effect == "destructive"
    assert metadata.approval == "admin_approval"
    assert metadata.required_env_vars == ("KORTNY_SANDBOX_RUNNER_URL",)
    assert metadata.sandbox.requires_sandbox is True
    assert metadata.sandbox.network == "none"
    assert metadata.sandbox.resource_limits.timeout_seconds == 30
    assert descriptor.to_payload()["sandbox"] == metadata.sandbox.to_payload()
    assert requirement.scope is ApprovalScope.admin
    assert requirement.risk == "sandboxed_code_execution"


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
    channel_canvas = tool_metadata("slack_create_channel_canvas")
    lookup_canvas_sections = tool_metadata("slack_lookup_canvas_sections")
    edit_canvas = tool_metadata("slack_edit_canvas")

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
    assert channel_canvas.category == "Slack actions"
    assert channel_canvas.side_effect == "write"
    assert channel_canvas.required_slack_scopes == ("canvases:write",)
    assert "current_channel_only" in channel_canvas.plan_gates
    assert "no_dm_canvas" in channel_canvas.plan_gates
    assert lookup_canvas_sections.category == "Slack context"
    assert lookup_canvas_sections.side_effect == "read"
    assert lookup_canvas_sections.required_slack_scopes == ("canvases:read",)
    assert "known_canvas_id_required" in lookup_canvas_sections.plan_gates
    assert "criteria_required" in lookup_canvas_sections.plan_gates
    assert edit_canvas.category == "Slack actions"
    assert edit_canvas.side_effect == "write"
    assert edit_canvas.required_slack_scopes == ("canvases:write",)
    assert "known_canvas_id_required" in edit_canvas.plan_gates


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
    channel_canvas = policy.requirement_for(
        SimpleNamespace(
            name="slack_create_channel_canvas",
            description="Create a channel canvas",
        ),
        {},
    )
    lookup_canvas_sections = policy.requirement_for(
        SimpleNamespace(
            name="slack_lookup_canvas_sections",
            description="Look up canvas sections",
        ),
        {},
    )
    edit_canvas = policy.requirement_for(
        SimpleNamespace(name="slack_edit_canvas", description="Edit a canvas"),
        {},
    )

    assert reply.scope is ApprovalScope.none
    assert reaction.scope is ApprovalScope.none
    assert pin.scope is ApprovalScope.none
    assert bookmark.scope is ApprovalScope.none
    assert channel_canvas.scope is ApprovalScope.none
    assert lookup_canvas_sections.scope is ApprovalScope.none
    assert edit_canvas.scope is ApprovalScope.none


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
