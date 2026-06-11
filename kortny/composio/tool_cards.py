"""Shared Composio tool-card construction (HIG-222).

One definition of how a raw :class:`ComposioTool` (or a synced
``composio_tool_cards`` row) becomes a :class:`ToolCard` and how a Composio
verb maps to a coarse ``side_effect``. Both the full-catalog sync
(:mod:`kortny.composio.catalog_sync`) and the per-task provider
(:mod:`kortny.composio.provider`) import from here so the synced cards and the
degraded lexical path stay byte-compatible.
"""

from __future__ import annotations

import hashlib

from kortny.composio.client import ComposioTool
from kortny.composio.runtime import RuntimeComposioConnection
from kortny.tool_selection.models import ToolCard
from kortny.tools.composio_execute import composio_runtime_tool_name

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
DESTRUCTIVE_VERBS = frozenset({"delete", "remove", "archive", "cancel", "destroy"})


def is_read_only(tool: ComposioTool) -> bool:
    tags = {tag.casefold().replace("_", "").replace("-", "") for tag in tool.tags}
    if "readonlyhint" in tags or "readonly" in tags:
        return True

    slug_parts = _slug_parts(tool.slug)
    if slug_parts & WRITE_VERBS:
        return False
    return bool(slug_parts & READ_VERBS)


def side_effect_for_tool(tool: ComposioTool) -> str:
    """Map Composio verb/tag detection to a coarse ``side_effect``.

    Read tools surface as ``read``; destructive verbs (delete/remove/...) as
    ``destructive``; other write verbs as ``write`` (HIG-223: write tools are
    surfaced — the approval gate, not this mapping, gates them).
    """

    if is_read_only(tool):
        return "read"
    if _slug_parts(tool.slug) & DESTRUCTIVE_VERBS:
        return "destructive"
    return "write"


def card_description(*, toolkit_slug: str, tool: ComposioTool) -> str:
    required = ", ".join(required_fields(tool.input_parameters)) or "none"
    access = "read-only" if is_read_only(tool) else "write-capable"
    return (
        f"Scoped {access} Composio tool from {toolkit_slug}: {tool.slug}. "
        f"{tool.description or tool.name} Required fields: {required}."
    )


def capabilities(*, toolkit_slug: str, tool: ComposioTool) -> tuple[str, ...]:
    text = " ".join([toolkit_slug, tool.slug, tool.name, tool.description]).casefold()
    caps = ["external_tool", f"{toolkit_slug}_integration"]
    if any(word in text for word in ("search", "web", "crawl", "scrape", "source")):
        caps.append("web_search")
        caps.append("current_research")
    if any(word in text for word in ("scrape", "crawl", "url", "page", "website")):
        caps.append("web_scrape")
    if any(word in text for word in ("file", "document", "page", "database", "notion")):
        caps.append("document_context")
    return tuple(dict.fromkeys(caps))


def native_replacements(caps: tuple[str, ...]) -> tuple[str, ...]:
    if "web_search" in caps or "current_research" in caps:
        return ("web_search",)
    return ()


def required_fields(parameters: dict) -> tuple[str, ...]:
    required = parameters.get("required")
    if not isinstance(required, list):
        return ()
    return tuple(str(item) for item in required if isinstance(item, str) and item)


def tool_card_for(
    *,
    connection: RuntimeComposioConnection,
    tool: ComposioTool,
) -> ToolCard:
    """Build a :class:`ToolCard` for a tool runnable through ``connection``."""

    toolkit_slug = connection.toolkit_slug
    caps = capabilities(toolkit_slug=toolkit_slug, tool=tool)
    return ToolCard(
        registry_name=composio_runtime_tool_name(toolkit_slug, tool.slug),
        provider="composio",
        display_name=f"{tool.name} via Composio",
        description=card_description(toolkit_slug=toolkit_slug, tool=tool),
        capabilities=caps,
        side_effect=side_effect_for_tool(tool),
        toolkit_slug=toolkit_slug,
        tool_slugs=(tool.slug,),
        tool_count=1,
        required_fields=required_fields(tool.input_parameters),
        visibility_scope_type=connection.visibility_scope_type,
        visibility_scope_id=connection.visibility_scope_id,
        can_replace_native_tools=native_replacements(caps),
    )


def synced_tool_card(
    *,
    connection: RuntimeComposioConnection,
    tool_slug: str,
    name: str,
    description: str,
    side_effect: str,
) -> ToolCard:
    """Build a :class:`ToolCard` from a synced ``composio_tool_cards`` row.

    The synced row stores only name/description/side_effect (no full schema), so
    capabilities are recomputed from the stored text; the full input schema is
    fetched lazily later, only for tools that survive selection.
    """

    toolkit_slug = connection.toolkit_slug
    text = " ".join([toolkit_slug, tool_slug, name, description]).casefold()
    caps = ["external_tool", f"{toolkit_slug}_integration"]
    if any(word in text for word in ("search", "web", "crawl", "scrape", "source")):
        caps.append("web_search")
        caps.append("current_research")
    if any(word in text for word in ("scrape", "crawl", "url", "page", "website")):
        caps.append("web_scrape")
    if any(word in text for word in ("file", "document", "page", "database", "notion")):
        caps.append("document_context")
    capabilities_tuple = tuple(dict.fromkeys(caps))
    return ToolCard(
        registry_name=composio_runtime_tool_name(toolkit_slug, tool_slug),
        provider="composio",
        display_name=f"{name} via Composio",
        description=description,
        capabilities=capabilities_tuple,
        side_effect=side_effect,
        toolkit_slug=toolkit_slug,
        tool_slugs=(tool_slug,),
        tool_count=1,
        visibility_scope_type=connection.visibility_scope_type,
        visibility_scope_id=connection.visibility_scope_id,
        can_replace_native_tools=native_replacements(capabilities_tuple),
    )


def card_sha(*, name: str, description: str, side_effect: str) -> str:
    """Stable hash of the persisted card fields, gating re-embedding."""

    payload = "\x1f".join((name, description, side_effect))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _slug_parts(slug: str) -> set[str]:
    return {
        part.casefold() for part in slug.replace("-", "_").split("_") if part.strip()
    }
