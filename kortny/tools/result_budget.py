"""Deterministic structural truncation of external tool result payloads.

External tools (Composio, MCP) can return arbitrarily large payloads. Feeding
the whole thing into the LLM context wastes tokens and can blow the prompt
budget. ``bound_tool_result`` shrinks an output dict to fit a character cap
while preserving as much structure as possible: dict keys are kept, long lists
keep their head and tail with an elision marker, and long strings keep a head
with an ellipsis. No LLM involved — fully deterministic.
"""

from __future__ import annotations

import json
from typing import Any

# Marker inserted in place of elided list elements.
_LIST_ELISION = "… (%d items elided) …"
# Minimum number of head/tail items kept when eliding a list.
_MIN_LIST_KEEP = 1


def bound_tool_result(output: dict, *, max_chars: int, hint: str) -> dict:
    """Return ``output`` unchanged if small enough, else a truncated copy.

    The returned dict (when truncated) always carries ``truncated=True``,
    ``original_chars`` (the size of the untruncated JSON), and
    ``truncation_hint``. Truncation is structural: dict keys are preserved
    where possible, long lists keep head+tail, and long strings keep a head.
    """

    original = _safe_dumps_len(output)
    if original <= max_chars:
        return output

    # Reserve headroom for the metadata keys we add at the top level.
    overhead = _metadata_overhead(original=original, hint=hint)
    budget = max(max_chars - overhead, 1)

    truncated_value = _truncate_value(output, budget=budget)
    if not isinstance(truncated_value, dict):
        # ``output`` is always a dict, but mypy/structure guard: wrap it.
        truncated_value = {"value": truncated_value}

    result: dict[str, Any] = dict(truncated_value)
    result["truncated"] = True
    result["original_chars"] = original
    result["truncation_hint"] = hint
    return result


def _metadata_overhead(*, original: int, hint: str) -> int:
    """Approximate JSON cost of the metadata keys appended at the top level."""

    meta = {
        "truncated": True,
        "original_chars": original,
        "truncation_hint": hint,
    }
    # ``+ 2`` accounts for the joining comma/space slack vs. an empty object.
    return len(json.dumps(meta)) + 2


def _truncate_value(value: Any, *, budget: int) -> Any:
    """Recursively shrink ``value`` so its JSON encoding fits ``budget``."""

    if _safe_dumps_len(value) <= budget:
        return value

    if isinstance(value, dict):
        return _truncate_dict(value, budget=budget)
    if isinstance(value, list):
        return _truncate_list(value, budget=budget)
    if isinstance(value, str):
        return _truncate_string(value, budget=budget)
    # Numbers/bools/None are already minimal; nothing to trim.
    return value


def _truncate_dict(value: dict, *, budget: int) -> dict:
    """Shrink a dict, preserving every key, dividing the budget across values."""

    items = list(value.items())
    if not items:
        return {}

    # Split the budget evenly across keys; the per-value share excludes the
    # rough cost of the keys themselves and JSON punctuation.
    keys_cost = sum(len(json.dumps(str(key))) + 2 for key, _ in items)
    value_budget = max((budget - keys_cost) // len(items), 1)

    truncated: dict[str, Any] = {}
    for key, item in items:
        truncated[str(key)] = _truncate_value(item, budget=value_budget)
    return truncated


def _truncate_list(value: list, *, budget: int) -> list:
    """Keep head+tail of a list with an elision marker for the middle."""

    if not value:
        return []

    # Find the largest symmetric head/tail count that fits the budget.
    keep = len(value)
    while keep > _MIN_LIST_KEEP:
        head_count = keep // 2
        tail_count = keep - head_count
        candidate = _assemble_list(value, head_count, tail_count)
        if _safe_dumps_len(candidate) <= budget:
            return candidate
        keep -= 1

    # Even minimal head+tail may not fit; fall back to a single truncated item.
    candidate = _assemble_list(value, _MIN_LIST_KEEP, _MIN_LIST_KEEP)
    if _safe_dumps_len(candidate) <= budget and len(value) > 1:
        return candidate

    first = _truncate_value(value[0], budget=max(budget - 64, 1))
    elided = len(value) - 1
    if elided > 0:
        return [first, _LIST_ELISION % elided]
    return [first]


def _assemble_list(value: list, head_count: int, tail_count: int) -> list:
    """Build a head + elision marker + tail list, or the whole list if it fits."""

    if head_count + tail_count >= len(value):
        return list(value)
    head = value[:head_count]
    tail = value[len(value) - tail_count :]
    elided = len(value) - head_count - tail_count
    return [*head, _LIST_ELISION % elided, *tail]


def _truncate_string(value: str, *, budget: int) -> str:
    """Keep a head of the string plus an ellipsis marker."""

    # JSON-encoding adds quotes/escapes; reserve a little headroom.
    keep = max(budget - 8, 1)
    if keep >= len(value):
        return value
    return value[:keep] + "…"


def _safe_dumps_len(value: Any) -> int:
    """Length of the JSON encoding of ``value`` (non-ASCII counts as-is)."""

    return len(json.dumps(value, ensure_ascii=False, default=str))


__all__ = ["bound_tool_result"]
