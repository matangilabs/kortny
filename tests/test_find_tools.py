"""Unit tests for find_tools + runtime registry mutation (Linear HIG-269)."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from kortny.tools.find_tools import FindToolsTool
from kortny.tools.registry import ToolRegistry
from kortny.tools.types import (
    JsonObject,
    JsonSchema,
    RecoverableToolError,
    Tool,
    ToolResult,
)


class _StubTool:
    def __init__(self, name: str, description: str = "stub") -> None:
        self.name = name
        self.description = description
        self.parameters: JsonSchema = {"type": "object", "properties": {}}

    def invoke(self, args: JsonObject) -> ToolResult:
        return ToolResult(output={"ok": True})


def test_register_if_absent_is_idempotent() -> None:
    registry = ToolRegistry()
    tool = _StubTool("alpha")
    assert registry.register_if_absent(tool) is True
    assert registry.register_if_absent(_StubTool("alpha")) is False  # already present
    assert registry.has("alpha")
    assert registry.names() == ("alpha",)


def test_find_tools_loads_into_registry() -> None:
    registry = ToolRegistry()
    loaded_tools = [_StubTool("composio_linear_list_issues", "List Linear issues.")]

    find = FindToolsTool(
        retrieve=lambda q: ["LINEAR_LIST_ISSUES", "LINEAR_GET_ISSUE"],
        load=lambda slugs: loaded_tools,
        registry=registry,
    )
    result = find.invoke({"query": "list my open Linear issues"})

    assert registry.has("composio_linear_list_issues")  # now callable next turn
    assert result.output["newly_loaded"] == 1
    names = {entry["name"] for entry in result.output["available"]}
    assert "composio_linear_list_issues" in names


def test_find_tools_second_call_is_idempotent() -> None:
    registry = ToolRegistry()
    find = FindToolsTool(
        retrieve=lambda q: ["LINEAR_LIST_ISSUES"],
        load=lambda slugs: [_StubTool("composio_linear_list_issues")],
        registry=registry,
    )
    find.invoke({"query": "linear"})
    second = find.invoke({"query": "linear"})
    assert second.output["newly_loaded"] == 0  # already loaded, no error
    assert registry.names() == ("composio_linear_list_issues",)


def test_find_tools_empty_query_is_recoverable() -> None:
    find = FindToolsTool(
        retrieve=lambda q: [], load=lambda slugs: [], registry=ToolRegistry()
    )
    with pytest.raises(RecoverableToolError):
        find.invoke({"query": "   "})


def test_find_tools_no_matches_returns_guidance() -> None:
    find = FindToolsTool(
        retrieve=lambda q: [], load=lambda slugs: [], registry=ToolRegistry()
    )
    result = find.invoke({"query": "something with no tools"})
    assert result.output["loaded"] == []
    assert "No matching tools" in result.output["message"]


def test_find_tools_respects_top_k() -> None:
    captured: list[list[str]] = []

    def _load(slugs: Sequence[str]) -> Sequence[Tool]:
        captured.append(list(slugs))
        return []

    find = FindToolsTool(
        retrieve=lambda q: [f"TOOL_{i}" for i in range(10)],
        load=_load,
        registry=ToolRegistry(),
        top_k=3,
    )
    find.invoke({"query": "many"})
    assert captured == [["TOOL_0", "TOOL_1", "TOOL_2"]]
