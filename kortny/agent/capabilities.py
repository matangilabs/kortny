"""Capability overview for the agent context (HIG-219).

Summarizes what is connected (native tool categories, Composio toolkits, MCP
servers) and what is unavailable, so the model can offer setup paths instead of
flat refusals when an integration is missing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from kortny.db.models import McpServer
from kortny.tool_selection.models import ToolCard
from kortny.tools.catalog import ToolDescriptor

TOOLKIT_TOOL_DISPLAY_CAP = 40

TOOLKIT_DESCRIPTIONS: dict[str, str] = {
    "alpha_vantage": (
        "real-time and historical stock, forex, and crypto quotes; technical "
        "indicators; company fundamentals and earnings"
    ),
    "alpaca": (
        "brokerage market data and trading: stock/crypto snapshots, bars, "
        "positions, and orders"
    ),
    "twelve_data": (
        "real-time and historical market data: stocks, forex, ETFs, crypto, "
        "and technical indicators"
    ),
    "notion": "read and write Notion pages, databases, and blocks",
    "linear": "manage Linear issues, projects, cycles, and comments",
    "confluence": "read and write Confluence pages and spaces",
    "vercel": "Vercel deployments, projects, domains, and environment variables",
    "supabase": "Supabase database queries, storage, and project management",
    "exa": "semantic web search and content retrieval",
    "firecrawl": "crawl and extract structured content from websites",
    "serpapi": "structured Google search results",
}

TOOLKIT_DISPLAY_NAMES: dict[str, str] = {
    "alpha_vantage": "Alpha Vantage",
    "alpaca": "Alpaca",
    "twelve_data": "Twelve Data",
    "notion": "Notion",
    "linear": "Linear",
    "confluence": "Confluence",
    "vercel": "Vercel",
    "supabase": "Supabase",
    "exa": "Exa",
    "firecrawl": "Firecrawl",
    "serpapi": "SerpAPI",
}


def _display_name(slug: str) -> str:
    return TOOLKIT_DISPLAY_NAMES.get(slug, slug.replace("_", " ").title())


@dataclass(frozen=True, slots=True)
class ConnectedToolkitSummary:
    """Per-toolkit summary for the connected_integrations context block."""

    toolkit_slug: str
    app_name: str
    app_description: str
    tool_names: list[str]
    total_tool_count: int


@dataclass(frozen=True, slots=True)
class CapabilityOverview:
    """Connected integrations and known gaps for one installation."""

    native_categories: tuple[str, ...]
    disabled_native: tuple[tuple[str, str], ...]
    composio_toolkits: tuple[str, ...]
    mcp_servers: tuple[tuple[str, str], ...]
    connected_toolkits: tuple[ConnectedToolkitSummary, ...] = field(default=())


def build_capability_overview(
    *,
    native_descriptors: Sequence[ToolDescriptor],
    external_cards: Sequence[ToolCard],
    mcp_rows: Sequence[McpServer],
    connected_composio_toolkits: Sequence[str] = (),
    connected_toolkits: Sequence[ConnectedToolkitSummary] = (),
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
        connected_toolkits=tuple(connected_toolkits),
    )


def render_capability_overview(overview: CapabilityOverview) -> str:
    """Render the ``<capabilities>`` block and optional ``<connected_integrations>`` block."""

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

    capabilities_block = "\n".join(lines)

    if not overview.connected_toolkits:
        return capabilities_block

    ci_lines = ["<connected_integrations>"]
    for toolkit in overview.connected_toolkits:
        display = _display_name(toolkit.toolkit_slug)
        description = TOOLKIT_DESCRIPTIONS.get(
            toolkit.toolkit_slug, toolkit.app_description
        )
        ci_lines.append(f"{display} — {description}")
        capped = toolkit.tool_names[:TOOLKIT_TOOL_DISPLAY_CAP]
        tool_line = "  tools: " + ", ".join(capped)
        remainder = toolkit.total_tool_count - len(capped)
        if remainder > 0:
            tool_line += f" [and {remainder} more (use find_tools)]"
        ci_lines.append(tool_line)
    ci_lines.append(
        "Call any connected tool directly by name — schema is fetched on demand. "
        "Prefer connected integrations over web_search for live, private, or "
        "domain-specific data."
    )
    ci_lines.append("</connected_integrations>")

    return capabilities_block + "\n" + "\n".join(ci_lines)
