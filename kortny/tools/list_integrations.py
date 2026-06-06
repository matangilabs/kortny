"""Runtime integration inventory tool."""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy.orm import Session

from kortny.composio import ComposioConnectionResolver
from kortny.db.models import Task
from kortny.tools.catalog import ToolDescriptor, tool_descriptors
from kortny.tools.types import JsonObject, JsonSchema, Tool, ToolResult


class DescribeToolsTool:
    """Describe native tools and scoped external integrations available to a task."""

    name = "describe_tools"
    description = (
        "Describes Kortny's currently available native tools and connected "
        "Composio integrations for this Slack task. Use when the user asks "
        "what Kortny can do, what tools it has, what integrations are connected, "
        "or why a capability is or is not available."
    )
    parameters: JsonSchema = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }

    def __init__(
        self,
        *,
        session: Session,
        task: Task,
        native_tools: Sequence[Tool],
    ) -> None:
        self.session = session
        self.task = task
        self.native_tools = tuple(native_tools)

    def invoke(self, args: JsonObject) -> ToolResult:
        del args
        connections = ComposioConnectionResolver(
            self.session,
            self.task,
        ).allowed_connections()
        native_descriptors = tool_descriptors(self.native_tools)
        return ToolResult(
            output={
                "native_tools": [_native_tool_payload(tool) for tool in native_descriptors],
                "native_tool_count": len(native_descriptors),
                "native_categories": sorted(
                    {descriptor.category for descriptor in native_descriptors}
                ),
                "connected_integrations": [
                    {
                        "toolkit_slug": connection.toolkit_slug,
                        "display_name": connection.display_name,
                        "scope_type": connection.visibility_scope_type,
                        "scope_id": connection.visibility_scope_id,
                        "connected_account_id": connection.connected_account_id,
                    }
                    for connection in connections
                ],
                "connected_integration_count": len(connections),
                "scope_note": (
                    "Only integrations visible to this Slack user/channel/task "
                    "are listed."
                ),
            }
        )


class ListIntegrationsTool(DescribeToolsTool):
    """Compatibility alias for older prompts that call list_integrations."""

    name = "list_integrations"
    description = (
        "Compatibility alias for describe_tools. Lists native tools and connected "
        "Composio integrations visible to this Slack task."
    )


def _native_tool_payload(descriptor: ToolDescriptor) -> JsonObject:
    payload = descriptor.to_payload()
    payload.pop("parameters", None)
    return payload
