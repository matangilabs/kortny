"""Runtime integration inventory tool."""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy.orm import Session

from kortny.composio import ComposioConnectionResolver
from kortny.db.models import Task
from kortny.tools.types import JsonObject, JsonSchema, Tool, ToolResult


class ListIntegrationsTool:
    """List native tools and scoped external integrations available to a task."""

    name = "list_integrations"
    description = (
        "Lists Kortny's currently available native tools and connected Composio "
        "integrations for this Slack task. Use when the user asks what Kortny "
        "can do, what tools it has, or what integrations are connected."
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
        return ToolResult(
            output={
                "native_tools": [
                    {
                        "name": tool.name,
                        "description": tool.description,
                    }
                    for tool in self.native_tools
                ],
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
