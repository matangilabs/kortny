"""HIG-223 provider mapping: Composio write surfacing, MCP hint mapping."""

from __future__ import annotations

import uuid

from kortny.composio.client import ComposioTool
from kortny.composio.tool_cards import (
    is_read_only as _is_read_only,
)
from kortny.composio.tool_cards import (
    side_effect_for_tool as _side_effect,
)
from kortny.db.models import McpServer, McpServerTool
from kortny.mcp.provider import _tool_card


def _composio_tool(slug: str, *, tags: tuple[str, ...] = ()) -> ComposioTool:
    return ComposioTool(
        slug=slug,
        name=slug.replace("_", " ").title(),
        description=f"{slug} tool.",
        toolkit_slug="linear",
        input_parameters={"type": "object", "properties": {}},
        tags=tags,
        version=None,
    )


# --- Composio verb -> side_effect -------------------------------------------


def test_composio_read_tool_is_read() -> None:
    tool = _composio_tool("LINEAR_GET_ISSUE", tags=("readOnlyHint",))
    assert _is_read_only(tool) is True
    assert _side_effect(tool) == "read"


def test_composio_create_tool_is_write() -> None:
    tool = _composio_tool("LINEAR_CREATE_ISSUE")
    assert _is_read_only(tool) is False
    assert _side_effect(tool) == "write"


def test_composio_delete_tool_is_destructive() -> None:
    tool = _composio_tool("LINEAR_DELETE_ISSUE")
    assert _side_effect(tool) == "destructive"


def test_composio_update_tool_is_write() -> None:
    tool = _composio_tool("LINEAR_UPDATE_ISSUE")
    assert _side_effect(tool) == "write"


# --- MCP readOnlyHint / destructiveHint -> side_effect ----------------------


def _server() -> McpServer:
    return McpServer(
        id=uuid.uuid4(),
        installation_id=uuid.uuid4(),
        name="acme",
        transport="stdio",
        status="enabled",
        created_by="test",
    )


def _mcp_tool(
    name: str,
    *,
    read_only: bool | None,
    destructive: bool | None,
) -> McpServerTool:
    return McpServerTool(
        id=uuid.uuid4(),
        server_id=uuid.uuid4(),
        name=name,
        description=f"{name} tool.",
        input_schema={"type": "object", "properties": {}},
        read_only_hint=read_only,
        destructive_hint=destructive,
        enabled=True,
    )


def test_mcp_read_only_hint_is_read() -> None:
    card = _tool_card(_server(), _mcp_tool("lookup", read_only=True, destructive=None))
    assert card.side_effect == "read"


def test_mcp_destructive_hint_is_destructive() -> None:
    card = _tool_card(_server(), _mcp_tool("wipe", read_only=False, destructive=True))
    assert card.side_effect == "destructive"


def test_mcp_plain_write_is_write() -> None:
    card = _tool_card(_server(), _mcp_tool("note", read_only=None, destructive=None))
    assert card.side_effect == "write"
