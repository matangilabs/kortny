"""Provider-neutral Composio external tool adapter."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy.orm import Session

from kortny.composio.client import ComposioCatalogError, ComposioClient, ComposioTool
from kortny.composio.runtime import (
    ComposioConnectionResolver,
    RuntimeComposioConnection,
)
from kortny.db.models import Task
from kortny.observability.events import log_observation
from kortny.tool_selection.models import ToolCard
from kortny.tools.composio_execute import (
    DEFAULT_RESULT_MAX_CHARS,
    ComposioExecuteTool,
    composio_runtime_tool_name,
)
from kortny.tools.types import Tool

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _ToolkitCatalog:
    connection: RuntimeComposioConnection
    tools: tuple[ComposioTool, ...]


class ComposioExternalToolProvider:
    """Expose scoped Composio connections as selectable external tools."""

    provider_name = "composio"

    def __init__(
        self,
        *,
        session: Session,
        task: Task,
        client: ComposioClient,
        per_toolkit_limit: int = 8,
        result_max_chars: int = DEFAULT_RESULT_MAX_CHARS,
    ) -> None:
        self.session = session
        self.task = task
        self.client = client
        self.per_toolkit_limit = per_toolkit_limit
        self.result_max_chars = result_max_chars
        self.resolver = ComposioConnectionResolver(session, task)
        self._catalog: tuple[_ToolkitCatalog, ...] | None = None

    def tool_cards(self) -> tuple[ToolCard, ...]:
        cards: list[ToolCard] = []
        for entry in self._load_catalog():
            cards.extend(_tool_cards(entry))
        return tuple(cards)

    def runtime_tools(self) -> tuple[Tool, ...]:
        tools: list[Tool] = []
        for entry in self._load_catalog():
            for tool in entry.tools:
                tools.append(
                    ComposioExecuteTool(
                        session=self.session,
                        task=self.task,
                        client=self.client,
                        tool=tool,
                        name=composio_runtime_tool_name(
                            entry.connection.toolkit_slug,
                            tool.slug,
                        ),
                        result_max_chars=self.result_max_chars,
                    )
                )
        return tuple(tools)

    def _load_catalog(self) -> tuple[_ToolkitCatalog, ...]:
        if self._catalog is not None:
            return self._catalog

        entries: list[_ToolkitCatalog] = []
        for connection in _best_connections_by_toolkit(
            self.resolver.allowed_connections()
        ):
            try:
                raw_tools = self.client.list_tools(
                    toolkit_slug=connection.toolkit_slug,
                    query=self.task.input,
                    limit=self.per_toolkit_limit,
                )
                fallback_tools = self.client.list_tools(
                    toolkit_slug=connection.toolkit_slug,
                    limit=self.per_toolkit_limit,
                )
                allowed_tools = tuple(
                    tool
                    for tool in _merge_tools(raw_tools, fallback_tools)
                    if _is_read_only(tool)
                )
            except ComposioCatalogError as exc:
                log_observation(
                    logger,
                    "composio_catalog_lookup_failed",
                    task=self.task,
                    provider="composio",
                    toolkit_slug=connection.toolkit_slug,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                continue

            if allowed_tools:
                entries.append(
                    _ToolkitCatalog(connection=connection, tools=allowed_tools)
                )

        self._catalog = tuple(entries)
        log_observation(
            logger,
            "composio_catalog_lookup_completed",
            task=self.task,
            provider="composio",
            toolkit_count=len(self._catalog),
            tool_count=sum(len(entry.tools) for entry in self._catalog),
            toolkit_slugs=[entry.connection.toolkit_slug for entry in self._catalog],
        )
        return self._catalog


def _best_connections_by_toolkit(
    connections: tuple[RuntimeComposioConnection, ...],
) -> tuple[RuntimeComposioConnection, ...]:
    best: dict[str, RuntimeComposioConnection] = {}
    for connection in connections:
        best.setdefault(connection.toolkit_slug, connection)
    return tuple(best.values())


def _merge_tools(
    *tool_groups: tuple[ComposioTool, ...],
) -> tuple[ComposioTool, ...]:
    tools_by_slug: dict[str, ComposioTool] = {}
    for group in tool_groups:
        for tool in group:
            tools_by_slug.setdefault(tool.slug, tool)
    return tuple(tools_by_slug.values())


def _tool_cards(entry: _ToolkitCatalog) -> tuple[ToolCard, ...]:
    return tuple(_tool_card(entry, tool) for tool in entry.tools)


def _tool_card(entry: _ToolkitCatalog, tool: ComposioTool) -> ToolCard:
    toolkit_slug = entry.connection.toolkit_slug
    capabilities = _capabilities(toolkit_slug=toolkit_slug, tools=(tool,))
    return ToolCard(
        registry_name=composio_runtime_tool_name(toolkit_slug, tool.slug),
        provider="composio",
        display_name=f"{tool.name} via Composio",
        description=_card_description(toolkit_slug=toolkit_slug, tool=tool),
        capabilities=capabilities,
        side_effect="read",
        toolkit_slug=toolkit_slug,
        tool_slugs=(tool.slug,),
        tool_count=1,
        required_fields=_required_fields(tool.input_parameters),
        visibility_scope_type=entry.connection.visibility_scope_type,
        visibility_scope_id=entry.connection.visibility_scope_id,
        can_replace_native_tools=_native_replacements(capabilities),
    )


def _card_description(*, toolkit_slug: str, tool: ComposioTool) -> str:
    required = ", ".join(_required_fields(tool.input_parameters)) or "none"
    return (
        f"Scoped read-only Composio tool from {toolkit_slug}: {tool.slug}. "
        f"{tool.description or tool.name} Required fields: {required}."
    )


def _capabilities(
    *,
    toolkit_slug: str,
    tools: tuple[ComposioTool, ...],
) -> tuple[str, ...]:
    text = " ".join(
        [toolkit_slug]
        + [tool.slug for tool in tools]
        + [tool.name for tool in tools]
        + [tool.description for tool in tools]
    ).casefold()
    capabilities = ["external_tool", f"{toolkit_slug}_integration"]
    if any(word in text for word in ("search", "web", "crawl", "scrape", "source")):
        capabilities.append("web_search")
        capabilities.append("current_research")
    if any(word in text for word in ("scrape", "crawl", "url", "page", "website")):
        capabilities.append("web_scrape")
    if any(word in text for word in ("file", "document", "page", "database", "notion")):
        capabilities.append("document_context")
    return tuple(dict.fromkeys(capabilities))


def _native_replacements(capabilities: tuple[str, ...]) -> tuple[str, ...]:
    if "web_search" in capabilities or "current_research" in capabilities:
        return ("web_search",)
    return ()


def _required_fields(parameters: dict) -> tuple[str, ...]:
    required = parameters.get("required")
    if not isinstance(required, list):
        return ()
    return tuple(str(item) for item in required if isinstance(item, str) and item)


def _is_read_only(tool: ComposioTool) -> bool:
    tags = {tag.casefold().replace("_", "").replace("-", "") for tag in tool.tags}
    if "readonlyhint" in tags or "readonly" in tags:
        return True

    slug_parts = {
        part.casefold()
        for part in tool.slug.replace("-", "_").split("_")
        if part.strip()
    }
    if slug_parts & WRITE_VERBS:
        return False
    return bool(slug_parts & READ_VERBS)


READ_VERBS = frozenset(
    {
        "crawl",
        "fetch",
        "find",
        "get",
        "inspect",
        "list",
        "query",
        "read",
        "retrieve",
        "scrape",
        "search",
        "summarize",
    }
)
WRITE_VERBS = frozenset(
    {
        "add",
        "archive",
        "cancel",
        "create",
        "delete",
        "disable",
        "enable",
        "invite",
        "move",
        "post",
        "publish",
        "remove",
        "send",
        "set",
        "submit",
        "update",
        "write",
    }
)
