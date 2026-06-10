"""Provider-neutral MCP external tool adapter (mirrors the Composio provider)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from kortny.db.models import McpServer, McpServerTool, Task
from kortny.mcp.sessions import McpSessionManager
from kortny.observability.events import log_observation
from kortny.tool_selection.models import ToolCard
from kortny.tools.mcp_execute import (
    DEFAULT_RESULT_MAX_CHARS,
    McpExecuteTool,
    mcp_runtime_tool_name,
)
from kortny.tools.types import Tool

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _ServerCatalog:
    server: McpServer
    tools: tuple[McpServerTool, ...]


class McpExternalToolProvider:
    """Expose an installation's enabled MCP server tools as selectable tools."""

    provider_name = "mcp"

    def __init__(
        self,
        *,
        session: Session,
        task: Task,
        encryption_key: str | None,
        tool_timeout_seconds: float,
        result_max_chars: int = DEFAULT_RESULT_MAX_CHARS,
    ) -> None:
        self.session = session
        self.task = task
        self.encryption_key = encryption_key
        self.tool_timeout_seconds = tool_timeout_seconds
        self.result_max_chars = result_max_chars
        self._catalog: tuple[_ServerCatalog, ...] | None = None
        self._session_manager = McpSessionManager()

    def tool_cards(self) -> tuple[ToolCard, ...]:
        cards: list[ToolCard] = []
        for entry in self._load_catalog():
            for tool in entry.tools:
                cards.append(_tool_card(entry.server, tool))
        return tuple(cards)

    def runtime_tools(self) -> tuple[Tool, ...]:
        tools: list[Tool] = []
        for entry in self._load_catalog():
            for tool in entry.tools:
                tools.append(
                    McpExecuteTool(
                        session=self.session,
                        task=self.task,
                        server=entry.server,
                        tool=tool,
                        encryption_key=self.encryption_key or "",
                        timeout_seconds=int(self.tool_timeout_seconds),
                        name=mcp_runtime_tool_name(entry.server.name, tool.name),
                        session_manager=self._session_manager,
                        result_max_chars=self.result_max_chars,
                    )
                )
        return tuple(tools)

    def close(self) -> None:
        """Close the per-task MCP session manager. Idempotent, never raises."""

        self._session_manager.close()

    def _load_catalog(self) -> tuple[_ServerCatalog, ...]:
        if self._catalog is not None:
            return self._catalog

        servers = self.session.scalars(
            select(McpServer)
            .where(
                McpServer.installation_id == self.task.installation_id,
                McpServer.status == "enabled",
            )
            .order_by(McpServer.name)
        ).all()

        entries: list[_ServerCatalog] = []
        for server in servers:
            tools = self.session.scalars(
                select(McpServerTool)
                .where(
                    McpServerTool.server_id == server.id,
                    McpServerTool.enabled.is_(True),
                )
                .order_by(McpServerTool.name)
            ).all()
            if tools:
                entries.append(_ServerCatalog(server=server, tools=tuple(tools)))

        self._catalog = tuple(entries)
        log_observation(
            logger,
            "mcp_catalog_lookup_completed",
            task=self.task,
            provider="mcp",
            server_count=len(self._catalog),
            tool_count=sum(len(entry.tools) for entry in self._catalog),
            server_names=[entry.server.name for entry in self._catalog],
        )
        return self._catalog


def _tool_card(server: McpServer, tool: McpServerTool) -> ToolCard:
    side_effect = "read" if tool.read_only_hint else "write"
    capabilities = _capabilities(server, tool)
    return ToolCard(
        registry_name=mcp_runtime_tool_name(server.name, tool.name),
        provider="mcp",
        display_name=f"{tool.name} via {server.name} (MCP)",
        description=_card_description(server, tool),
        capabilities=capabilities,
        side_effect=side_effect,
        toolkit_slug=server.name,
        tool_slugs=(tool.name,),
        tool_count=1,
        required_fields=_required_fields(tool.input_schema),
        visibility_scope_type="workspace",
        visibility_scope_id=None,
    )


def _card_description(server: McpServer, tool: McpServerTool) -> str:
    required = ", ".join(_required_fields(tool.input_schema)) or "none"
    access = "read-only" if tool.read_only_hint else "write-capable"
    # Prefer the LLM-enriched description when available (HIG-215).
    raw_body = tool.enriched_description or tool.description or tool.name
    body = raw_body.strip() or tool.name
    return (
        f"{access.capitalize()} MCP tool {tool.name} from server '{server.name}': "
        f"{body} Required fields: {required}."
    )


def _capabilities(server: McpServer, tool: McpServerTool) -> tuple[str, ...]:
    text = " ".join([server.name, tool.name, tool.description or ""]).casefold()
    capabilities = ["external_tool", "mcp_integration", f"{_slug(server.name)}_mcp"]
    if any(word in text for word in ("search", "web", "crawl", "scrape", "source")):
        capabilities.append("web_search")
    if any(word in text for word in ("scrape", "crawl", "url", "page", "website")):
        capabilities.append("web_scrape")
    if any(word in text for word in ("file", "document", "page", "database", "note")):
        capabilities.append("document_context")
    return tuple(dict.fromkeys(capabilities))


def _slug(value: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in value.casefold()).strip("_")


def _required_fields(parameters: object) -> tuple[str, ...]:
    if not isinstance(parameters, dict):
        return ()
    required = parameters.get("required")
    if not isinstance(required, list):
        return ()
    return tuple(str(item) for item in required if isinstance(item, str) and item)


__all__ = ["McpExternalToolProvider"]
