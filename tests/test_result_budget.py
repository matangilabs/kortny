"""Unit tests for deterministic external tool result truncation (HIG-216)."""

from __future__ import annotations

import json

from kortny.tools.result_budget import bound_tool_result


def _chars(value: object) -> int:
    return len(json.dumps(value, ensure_ascii=False, default=str))


class TestBoundToolResultUnderLimit:
    def test_returns_same_object_when_small(self) -> None:
        output = {"provider": "mcp", "text": "tiny"}
        result = bound_tool_result(output, max_chars=16000, hint="hint")
        # Identity preserved: nothing was copied or mutated.
        assert result is output
        assert "truncated" not in result

    def test_exactly_at_limit_is_unchanged(self) -> None:
        output = {"text": "a" * 50}
        size = _chars(output)
        result = bound_tool_result(output, max_chars=size, hint="hint")
        assert result is output


class TestBoundToolResultOverLimit:
    def test_long_string_truncated_with_marker(self) -> None:
        output = {"provider": "mcp", "text": "x" * 10000}
        result = bound_tool_result(output, max_chars=2000, hint="too big")

        assert result is not output
        assert result["truncated"] is True
        assert result["original_chars"] == _chars(output)
        assert result["truncation_hint"] == "too big"
        # Keys preserved; the long string was shortened and marked.
        assert result["provider"] == "mcp"
        assert isinstance(result["text"], str)
        assert result["text"].endswith("…")
        assert len(result["text"]) < 10000
        assert _chars(result) <= 2000

    def test_long_list_keeps_head_and_tail(self) -> None:
        output = {"items": list(range(1000))}
        result = bound_tool_result(output, max_chars=1500, hint="list")

        assert result["truncated"] is True
        items = result["items"]
        assert isinstance(items, list)
        # First and last elements are preserved around an elision marker.
        assert items[0] == 0
        assert items[-1] == 999
        assert any(isinstance(part, str) and "elided" in part for part in items)
        assert _chars(result) <= 1500

    def test_nested_structure_preserves_keys(self) -> None:
        output = {
            "provider": "composio",
            "data": {
                "rows": [{"id": i, "blob": "y" * 100} for i in range(500)],
                "summary": "z" * 5000,
            },
            "log_id": "abc123",
        }
        result = bound_tool_result(output, max_chars=3000, hint="nested")

        assert result["truncated"] is True
        # Top-level keys all survive.
        assert set(result).issuperset({"provider", "data", "log_id"})
        assert result["provider"] == "composio"
        assert result["log_id"] == "abc123"
        # Nested dict keeps its keys too.
        assert set(result["data"]).issuperset({"rows", "summary"})
        assert _chars(result) <= 3000

    def test_truncation_is_deterministic(self) -> None:
        output = {"text": "q" * 9000, "items": list(range(300))}
        first = bound_tool_result(output, max_chars=2000, hint="h")
        second = bound_tool_result(output, max_chars=2000, hint="h")
        assert first == second

    def test_metadata_keys_always_present_when_truncated(self) -> None:
        output = {"a": "1" * 8000}
        result = bound_tool_result(output, max_chars=1000, hint="meta")
        assert result["truncated"] is True
        assert isinstance(result["original_chars"], int)
        assert result["truncation_hint"] == "meta"
