"""Kortny tool adapter for scoped Composio runtime execution."""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy.orm import Session

from kortny.composio import ComposioClient, ComposioConnectionResolver
from kortny.db.models import Task
from kortny.observability.events import log_observation
from kortny.tools.types import JsonObject, JsonSchema, ToolResult

SUPPORTED_COMPOSIO_TOOLKITS = ("firecrawl",)
SUPPORTED_COMPOSIO_TOOL_SLUGS = {
    "firecrawl": ("FIRECRAWL_SCRAPE", "FIRECRAWL_SEARCH"),
}
logger = logging.getLogger(__name__)


class ComposioExecuteTool:
    """Execute approved Composio tools through a scoped connected account."""

    name = "composio_execute"
    description = (
        "Executes an approved Composio integration tool using Kortny's scoped "
        "connected-account policy. Use this for connected Firecrawl web scraping "
        "or search when the user asks Kortny to crawl, scrape, inspect, or search "
        "web content through a connected integration. Available tool slugs: "
        "FIRECRAWL_SCRAPE with arguments like {'url': 'https://example.com', "
        "'formats': ['markdown']}; FIRECRAWL_SEARCH with arguments like "
        "{'q': 'search query', 'limit': 5}. For current/recent/latest web "
        "research or finding sources, use FIRECRAWL_SEARCH. For reading, "
        "auditing, scraping, or crawling a specific URL or website, use "
        "FIRECRAWL_SCRAPE. Do not use this for write actions."
    )
    parameters: JsonSchema = {
        "type": "object",
        "properties": {
            "toolkit_slug": {
                "type": "string",
                "enum": list(SUPPORTED_COMPOSIO_TOOLKITS),
                "description": "Connected Composio toolkit to use.",
            },
            "tool_slug": {
                "type": "string",
                "enum": [
                    slug
                    for slugs in SUPPORTED_COMPOSIO_TOOL_SLUGS.values()
                    for slug in slugs
                ],
                "description": "Approved Composio tool slug to execute.",
            },
            "arguments": {
                "type": "object",
                "description": "Tool-specific JSON arguments.",
                "additionalProperties": True,
            },
            "version": {
                "type": "string",
                "description": "Optional Composio toolkit version.",
            },
        },
        "required": ["toolkit_slug", "tool_slug", "arguments"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        *,
        session: Session,
        task: Task,
        client: ComposioClient,
    ) -> None:
        self.session = session
        self.task = task
        self.client = client
        self.resolver = ComposioConnectionResolver(session, task)

    @property
    def has_available_connections(self) -> bool:
        return self.resolver.has_allowed_connection(
            toolkit_slugs=SUPPORTED_COMPOSIO_TOOLKITS,
        )

    def invoke(self, args: JsonObject) -> ToolResult:
        toolkit_slug = _required_string(args.get("toolkit_slug"), "toolkit_slug").lower()
        tool_slug = _required_string(args.get("tool_slug"), "tool_slug").upper()
        arguments = args.get("arguments")
        if not isinstance(arguments, dict):
            raise ValueError("composio_execute 'arguments' must be an object")
        version = _optional_string(args.get("version"), "version")

        allowed_tool_slugs = SUPPORTED_COMPOSIO_TOOL_SLUGS.get(toolkit_slug)
        if allowed_tool_slugs is None:
            raise ValueError(f"Composio toolkit is not enabled for runtime: {toolkit_slug}")
        if tool_slug not in allowed_tool_slugs:
            raise ValueError(f"Composio tool is not approved for runtime: {tool_slug}")

        connection = self.resolver.best_connection(toolkit_slug=toolkit_slug)
        if connection is None:
            raise ValueError(
                f"No active Composio {toolkit_slug} connection is available "
                "for this Slack user/channel/workspace."
            )

        log_observation(
            logger,
            "composio_tool_execution_started",
            task=self.task,
            provider="composio",
            toolkit_slug=toolkit_slug,
            tool_slug=tool_slug,
            visibility_scope_type=connection.visibility_scope_type,
            argument_keys=sorted(arguments),
        )
        execution = self.client.execute_tool(
            tool_slug=tool_slug,
            user_id=connection.composio_user_id,
            connected_account_id=connection.connected_account_id,
            arguments=arguments,
            version=version,
        )
        log_observation(
            logger,
            "composio_tool_execution_completed",
            task=self.task,
            provider="composio",
            toolkit_slug=toolkit_slug,
            tool_slug=tool_slug,
            visibility_scope_type=connection.visibility_scope_type,
            successful=execution.successful,
            log_id=execution.log_id,
        )
        return ToolResult(
            output={
                "provider": "composio",
                "toolkit_slug": toolkit_slug,
                "tool_slug": tool_slug,
                "successful": execution.successful,
                "data": execution.data,
                "error": execution.error,
                "log_id": execution.log_id,
                "scope": {
                    "type": connection.visibility_scope_type,
                    "id": connection.visibility_scope_id,
                },
                "connection": {
                    "display_name": connection.display_name,
                    "connected_account_id": connection.connected_account_id,
                },
            }
        )


def _required_string(value: Any, name: str) -> str:
    text = _optional_string(value, name)
    if text is None:
        raise ValueError(f"composio_execute {name!r} must be a non-empty string")
    return text


def _optional_string(value: Any, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"composio_execute {name!r} must be a string")
    text = value.strip()
    return text or None
