"""Build compact tool cards for scoped tool selection."""

from __future__ import annotations

from collections.abc import Sequence

from kortny.tool_selection.models import ToolCard
from kortny.tool_selection.providers import ExternalToolProvider
from kortny.tools import Tool, tool_descriptor


class ToolCatalogService:
    """Create compact tool-selection cards from registered tool objects."""

    def native_cards(self, tools: Sequence[Tool]) -> tuple[ToolCard, ...]:
        return tuple(_native_tool_card(tool) for tool in tools)

    def external_cards(
        self,
        providers: Sequence[ExternalToolProvider],
    ) -> tuple[ToolCard, ...]:
        cards: list[ToolCard] = []
        for provider in providers:
            cards.extend(provider.tool_cards())
        return tuple(cards)


def _native_tool_card(tool: Tool) -> ToolCard:
    descriptor = tool_descriptor(tool)
    return ToolCard(
        registry_name=descriptor.name,
        provider="native",
        display_name=descriptor.display_name,
        description=descriptor.description,
        capabilities=descriptor.capabilities,
        side_effect=descriptor.side_effect,
        required_fields=descriptor.required_args,
        can_replace_native_tools=descriptor.can_replace_native_tools,
    )
