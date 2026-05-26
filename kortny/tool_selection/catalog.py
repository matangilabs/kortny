"""Build compact tool cards for scoped tool selection."""

from __future__ import annotations

from collections.abc import Sequence

from kortny.tool_selection.models import ToolCard
from kortny.tools import Tool
from kortny.tools.composio_execute import (
    SUPPORTED_COMPOSIO_TOOL_SLUGS,
    ComposioExecuteTool,
)


class ToolCatalogService:
    """Create compact tool-selection cards from registered tool objects."""

    def native_cards(self, tools: Sequence[Tool]) -> tuple[ToolCard, ...]:
        return tuple(_native_tool_card(tool) for tool in tools)

    def external_cards(self, tools: Sequence[Tool]) -> tuple[ToolCard, ...]:
        cards: list[ToolCard] = []
        for tool in tools:
            if isinstance(tool, ComposioExecuteTool):
                cards.extend(_composio_tool_cards(tool))
        return tuple(cards)


def _native_tool_card(tool: Tool) -> ToolCard:
    return ToolCard(
        registry_name=tool.name,
        provider="native",
        display_name=tool.name,
        description=tool.description,
        capabilities=_native_capabilities(tool.name),
        side_effect=_native_side_effect(tool.name),
    )


def _composio_tool_cards(tool: ComposioExecuteTool) -> tuple[ToolCard, ...]:
    cards: list[ToolCard] = []
    firecrawl_connection = tool.resolver.best_connection(toolkit_slug="firecrawl")
    if firecrawl_connection is not None:
        cards.append(
            ToolCard(
                registry_name=tool.name,
                provider="composio",
                display_name="Firecrawl web research",
                description=(
                    "Searches the public web and scrapes website pages through "
                    "the user's scoped Firecrawl connection. Best for current "
                    "web research, source finding, website inspection, crawling, "
                    "scraping, and reading a specific URL."
                ),
                capabilities=("web_search", "web_scrape", "current_research"),
                side_effect="read",
                toolkit_slug="firecrawl",
                tool_slugs=SUPPORTED_COMPOSIO_TOOL_SLUGS["firecrawl"],
                visibility_scope_type=firecrawl_connection.visibility_scope_type,
                visibility_scope_id=firecrawl_connection.visibility_scope_id,
                can_replace_native_tools=("web_search",),
            )
        )
    return tuple(cards)


def _native_capabilities(tool_name: str) -> tuple[str, ...]:
    if tool_name == "web_search":
        return ("web_search", "current_research")
    if tool_name == "pdf_generator":
        return ("document_generation", "artifact_generation")
    if tool_name == "slack_channel_history":
        return ("slack_history", "thread_context")
    if tool_name == "slack_file_read":
        return ("slack_file_read", "file_analysis")
    if tool_name in {"remember_fact", "recall_fact", "inspect_memory", "forget_fact"}:
        return ("workspace_memory",)
    return ()


def _native_side_effect(tool_name: str) -> str:
    if tool_name == "pdf_generator":
        return "write"
    if tool_name in {"remember_fact", "forget_fact"}:
        return "write"
    return "read"
