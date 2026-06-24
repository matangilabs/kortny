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

# Known Composio toolkit slugs with human-readable descriptions.  Slugs not in
# this map fall back to a humanized slug name + generic description.
TOOLKIT_APP_DESCRIPTIONS: dict[str, str] = {
    "alpha_vantage": "Alpha Vantage — real-time & historical stock/forex/crypto quotes, fundamentals, technical indicators.",
    "alpaca": "Alpaca — brokerage market data + trading: snapshots, bars, positions, orders.",
    "twelve_data": "Twelve Data — real-time and historical financial market data: stocks, forex, crypto, ETFs.",
    "notion": "Notion — workspace pages, databases, blocks, and comments.",
    "linear": "Linear — issue tracker: projects, issues, cycles, team members, labels.",
    "confluence": "Confluence — wiki pages, spaces, comments, and search.",
    "supabase": "Supabase — Postgres database management, edge functions, storage, and realtime.",
    "vercel": "Vercel — project deployments, domains, environment variables, and team management.",
    "exa": "Exa — neural web search and content retrieval for research and discovery.",
    "firecrawl": "Firecrawl — web scraping and crawling: extract structured content from any URL.",
    "serpapi": "SerpApi — Google, Bing, and other search engine results via structured API.",
}


def _humanize_slug(slug: str) -> str:
    """Turn a toolkit slug like 'twelve_data' into 'Twelve Data'."""
    return " ".join(word.capitalize() for word in slug.replace("-", "_").split("_"))


@dataclass(frozen=True, slots=True)
class ConnectedToolkitSummary:
    """Summary of one connected toolkit and its available tools."""

    toolkit_slug: str
    app_description: str
    tool_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CapabilityOverview:
    """Connected integrations and known gaps for one installation."""

    native_categories: tuple[str, ...]
    disabled_native: tuple[tuple[str, str], ...]
    composio_toolkits: tuple[str, ...]
    mcp_servers: tuple[tuple[str, str], ...]
    connected_toolkits: tuple[ConnectedToolkitSummary, ...] = ()


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


def render_connected_integrations(
    overview: CapabilityOverview,
    max_chars: int = 8000,
) -> str | None:
    """Render the ``<connected_integrations>`` context block.

    Renders one line per connected toolkit (app slug + description). Per-tool
    name CSVs are intentionally omitted: task-relevant schemas are pre-loaded by
    the prewarm step, and the model must call find_tools to load any additional
    tool's schema before constructing a call. This keeps the block compact and
    prevents the model from guessing argument shapes from names alone.

    Returns ``None`` if there are no connected toolkits.
    """

    if not overview.connected_toolkits:
        return None

    lines = ["<connected_integrations>"]
    for tk in overview.connected_toolkits:
        desc = TOOLKIT_APP_DESCRIPTIONS.get(tk.toolkit_slug)
        if desc is None:
            human_name = _humanize_slug(tk.toolkit_slug)
            desc = f"{human_name} — Third-party integration via Composio."
        lines.append(f"- {tk.toolkit_slug}: {desc}")
    lines.append("</connected_integrations>")
    rendered = "\n".join(lines)
    # Hard-truncate only if the block somehow exceeds the budget (extremely
    # unlikely with app-level-only lines, but defensive).
    return rendered[:max_chars] if len(rendered) > max_chars else rendered


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
