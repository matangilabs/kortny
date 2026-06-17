"""Capability overview for the agent context (HIG-219).

Summarizes what is connected (native tool categories, Composio toolkits, MCP
servers) and what is unavailable, so the model can offer setup paths instead of
flat refusals when an integration is missing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from kortny.db.models import McpServer
from kortny.tool_selection.models import ToolCard
from kortny.tools.catalog import ToolDescriptor


@dataclass(frozen=True, slots=True)
class CapabilityOverview:
    """Connected integrations and known gaps for one installation."""

    native_categories: tuple[str, ...]
    disabled_native: tuple[tuple[str, str], ...]
    composio_toolkits: tuple[str, ...]
    mcp_servers: tuple[tuple[str, str], ...]


def build_capability_overview(
    *,
    native_descriptors: Sequence[ToolDescriptor],
    external_cards: Sequence[ToolCard],
    mcp_rows: Sequence[McpServer],
    connected_composio_toolkits: Sequence[str] = (),
) -> CapabilityOverview:
    """Assemble the capability overview from already-loaded sources.

    ``connected_composio_toolkits`` is the deterministic, DB-derived set of
    active Composio toolkits for the task (HIG-274). It is authoritative: when an
    intent routes a request away from external tools, ``external_cards`` is empty
    and deriving the connected set from cards would make the agent capability-
    blind (it then fabricates "not connected"). Unioning the deterministic set in
    keeps the ``<capabilities>`` block accurate on every path.
    """

    native_categories = tuple(
        dict.fromkeys(
            descriptor.category
            for descriptor in native_descriptors
            if descriptor.enabled
        )
    )
    disabled_native = tuple(
        (descriptor.name, descriptor.disabled_reason or "unavailable")
        for descriptor in native_descriptors
        if not descriptor.enabled
    )
    composio_toolkits = tuple(
        sorted(
            {
                card.toolkit_slug
                for card in external_cards
                if card.provider == "composio" and card.toolkit_slug
            }
            | {slug for slug in connected_composio_toolkits if slug}
        )
    )
    mcp_servers = tuple((row.name, row.status) for row in mcp_rows)
    return CapabilityOverview(
        native_categories=native_categories,
        disabled_native=disabled_native,
        composio_toolkits=composio_toolkits,
        mcp_servers=mcp_servers,
    )


def render_capability_overview(overview: CapabilityOverview) -> str:
    """Render the ``<capabilities>`` context block."""

    lines = [
        "<capabilities>",
        "Current integration availability for this workspace. Use this when "
        "deciding what you can do now versus what needs setup.",
    ]
    if overview.native_categories:
        lines.append(
            "Connected: native tool categories: "
            + ", ".join(overview.native_categories)
            + "."
        )
    if overview.composio_toolkits:
        lines.append(
            "Connected: Composio toolkits: "
            + ", ".join(overview.composio_toolkits)
            + "."
        )
    enabled_mcp = [name for name, status in overview.mcp_servers if status == "enabled"]
    if enabled_mcp:
        lines.append("Connected: MCP servers: " + ", ".join(enabled_mcp) + ".")

    unavailable = [
        f"{name} ({reason})" for name, reason in overview.disabled_native
    ] + [
        f"{name} (MCP server {status})"
        for name, status in overview.mcp_servers
        if status != "enabled"
    ]
    if unavailable:
        lines.append("Unavailable (needs setup): " + "; ".join(unavailable) + ".")
    lines.append("</capabilities>")
    return "\n".join(lines)
