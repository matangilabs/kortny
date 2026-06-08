"""Tool contracts and registry helpers.

Keep this package surface light. Low-level modules import
``kortny.tools.catalog`` during task persistence and approval checks; eagerly
importing every concrete tool here creates avoidable cycles with task services.
Concrete tools are lazy-loaded through ``__getattr__`` below.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from kortny.tools.catalog import (
    ToolDescriptor,
    ToolMetadata,
    dashboard_native_tool_names,
    low_risk_native_write_tool_names,
    native_slack_context_hint_names,
    native_tool_integration_map,
    native_tool_names_by_approval,
    read_only_native_tool_names,
    runtime_native_tool_names,
    tool_descriptor,
    tool_descriptor_from_class,
    tool_descriptors,
    tool_metadata,
)
from kortny.tools.registry import DuplicateToolError, ToolNotFoundError, ToolRegistry
from kortny.tools.types import (
    JsonObject,
    JsonSchema,
    RecoverableToolError,
    Tool,
    ToolArtifact,
    ToolResult,
)

_LAZY_EXPORTS = {
    "CodeExecTool": "kortny.tools.code_exec",
    "ComposioExecuteTool": "kortny.tools.composio_execute",
    "DescribeToolsTool": "kortny.tools.list_integrations",
    "EchoTool": "kortny.tools.echo",
    "ForgetFactTool": "kortny.tools.workspace_memory",
    "InspectMemoryTool": "kortny.tools.workspace_memory",
    "ListIntegrationsTool": "kortny.tools.list_integrations",
    "ObservationChannelHistoryCache": "kortny.tools.slack_channel_history",
    "PdfGeneratorTool": "kortny.tools.pdf_generator",
    "QueryWorkspaceGraphTool": "kortny.tools.workspace_graph",
    "RecallFactTool": "kortny.tools.workspace_memory",
    "RememberFactTool": "kortny.tools.workspace_memory",
    "ResolveSlackIdentityTool": "kortny.tools.resolve_slack_identity",
    "SearchObservedSlackHistoryTool": "kortny.tools.search_observed_slack_history",
    "SlackAddBookmarkTool": "kortny.tools.slack_actions",
    "SlackAddReactionTool": "kortny.tools.slack_actions",
    "SlackCreateChannelCanvasTool": "kortny.tools.slack_actions",
    "SlackEditCanvasTool": "kortny.tools.slack_actions",
    "SlackLookupCanvasSectionsTool": "kortny.tools.slack_actions",
    "SlackChannelHistoryError": "kortny.tools.slack_channel_history",
    "SlackChannelHistoryTool": "kortny.tools.slack_channel_history",
    "SlackFileReadError": "kortny.tools.slack_file_read",
    "SlackFileReadTool": "kortny.tools.slack_file_read",
    "SlackChannelInfoTool": "kortny.tools.slack_identity_info",
    "SlackUserInfoTool": "kortny.tools.slack_identity_info",
    "SlackPinMessageTool": "kortny.tools.slack_actions",
    "SlackReplyThreadTool": "kortny.tools.slack_actions",
    "WebSearchTool": "kortny.tools.web_search",
}


def __getattr__(name: str) -> Any:
    """Lazily expose concrete tool classes without eager import cycles."""

    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


__all__ = [
    "CodeExecTool",
    "ComposioExecuteTool",
    "DescribeToolsTool",
    "DuplicateToolError",
    "EchoTool",
    "ForgetFactTool",
    "InspectMemoryTool",
    "JsonObject",
    "JsonSchema",
    "ListIntegrationsTool",
    "ObservationChannelHistoryCache",
    "PdfGeneratorTool",
    "QueryWorkspaceGraphTool",
    "RecallFactTool",
    "RecoverableToolError",
    "RememberFactTool",
    "ResolveSlackIdentityTool",
    "SearchObservedSlackHistoryTool",
    "SlackAddBookmarkTool",
    "SlackAddReactionTool",
    "SlackCreateChannelCanvasTool",
    "SlackEditCanvasTool",
    "SlackLookupCanvasSectionsTool",
    "SlackChannelHistoryError",
    "SlackChannelHistoryTool",
    "SlackFileReadError",
    "SlackFileReadTool",
    "SlackChannelInfoTool",
    "SlackUserInfoTool",
    "SlackPinMessageTool",
    "SlackReplyThreadTool",
    "ToolDescriptor",
    "ToolMetadata",
    "Tool",
    "ToolArtifact",
    "ToolNotFoundError",
    "ToolRegistry",
    "ToolResult",
    "WebSearchTool",
    "dashboard_native_tool_names",
    "low_risk_native_write_tool_names",
    "native_slack_context_hint_names",
    "native_tool_integration_map",
    "native_tool_names_by_approval",
    "read_only_native_tool_names",
    "runtime_native_tool_names",
    "tool_descriptor",
    "tool_descriptor_from_class",
    "tool_descriptors",
    "tool_metadata",
]
