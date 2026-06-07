"""Runtime integration inventory tool."""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy.orm import Session

from kortny.composio import ComposioConnectionResolver, RuntimeComposioConnection
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
                "user_facing_summary": _user_facing_summary(
                    native_descriptors=native_descriptors,
                    connections=connections,
                ),
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


def _user_facing_summary(
    *,
    native_descriptors: Sequence[ToolDescriptor],
    connections: Sequence[RuntimeComposioConnection],
) -> JsonObject:
    capability_groups = _capability_groups(native_descriptors)
    connected_apps = [_connected_app_summary(connection) for connection in connections]
    limitations: list[str] = []
    if not connected_apps:
        limitations.append("No connected app accounts are visible in this conversation.")
    if not any(descriptor.name == "web_search" for descriptor in native_descriptors):
        limitations.append("Web search is not available in this conversation.")
    return {
        "preferred_opening": "Here are the things I can help with right now:",
        "voice_guidance": (
            "Answer in first person as Kortny. Use the capability groups and "
            "connected apps below. Do not expose backend field names, tool ids, "
            "connected account ids, or implementation labels unless the user asks."
        ),
        "capability_groups": capability_groups,
        "connected_apps": connected_apps,
        "limitations": limitations,
    }


def _capability_groups(
    native_descriptors: Sequence[ToolDescriptor],
) -> list[JsonObject]:
    categories = {descriptor.category for descriptor in native_descriptors}
    groups: list[JsonObject] = []
    for category, payload in _USER_CAPABILITY_GROUPS:
        if category not in categories:
            continue
        groups.append(payload)
    return groups


_USER_CAPABILITY_GROUPS: tuple[tuple[str, JsonObject], ...] = (
    (
        "Research",
        {
            "label": "Research and current info",
            "summary": "I can look up current public information when search is available.",
            "examples": [
                "research a market, company, person, or tool",
                "compare options and summarize tradeoffs",
            ],
        },
    ),
    (
        "Slack context",
        {
            "label": "Slack context",
            "summary": (
                "I can read recent channel history, search observed Slack "
                "messages, and inspect shared files I can access."
            ),
            "examples": [
                "summarize recent decisions",
                "find where a topic came up before",
                "pull context from a thread or uploaded file",
            ],
        },
    ),
    (
        "Workspace context",
        {
            "label": "Workspace knowledge",
            "summary": "I can use what I know about people, channels, projects, and relationships.",
            "examples": [
                "explain how a channel is used",
                "connect a request to known workspace context",
            ],
        },
    ),
    (
        "Memory",
        {
            "label": "Memory",
            "summary": "I can remember, recall, inspect, and forget stable preferences or facts.",
            "examples": [
                "remember a preference",
                "show or remove something I have saved",
            ],
        },
    ),
    (
        "Scheduling",
        {
            "label": "Scheduled work",
            "summary": "I can create, check, pause, resume, update, or cancel recurring work.",
            "examples": [
                "send a daily market update",
                "change or pause an existing schedule",
            ],
        },
    ),
    (
        "Documents",
        {
            "label": "Documents",
            "summary": "I can create polished PDF artifacts when you explicitly ask for one.",
            "examples": [
                "turn a researched brief into a PDF",
                "generate an artifact for a task",
            ],
        },
    ),
)


def _connected_app_summary(connection: RuntimeComposioConnection) -> JsonObject:
    toolkit_label = _display_label(connection.toolkit_slug)
    display_name = connection.display_name or f"{toolkit_label} account"
    return {
        "name": display_name,
        "app": toolkit_label,
        "scope": _scope_label(connection.visibility_scope_type),
    }


def _display_label(value: str) -> str:
    return value.replace("_", " ").replace("-", " ").title()


def _scope_label(value: str) -> str:
    return {
        "user": "personal",
        "channel": "channel",
        "workspace": "workspace",
    }.get(value, value)
