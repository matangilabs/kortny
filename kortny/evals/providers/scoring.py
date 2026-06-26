"""Pure scoring functions for provider contract-test probe results.

No DB, no network. Each function takes a raw LiteLLM ``ModelResponse``-like
mapping and returns True if the check passes.

These are called by the live runner (runner.py) and exercised structurally
by the CI-safe test in tests/test_providers_eval.py.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any


def response_has_content(response: Mapping[str, Any]) -> bool:
    """Return True if the response contains at least one non-empty text choice."""
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return False
    for choice in choices:
        if not isinstance(choice, Mapping):
            continue
        message = choice.get("message") or choice.get("delta")
        if not isinstance(message, Mapping):
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return True
    return False


def no_error_response(response: Mapping[str, Any]) -> bool:
    """Return True if the response does not carry a top-level error key."""
    return "error" not in response


def tool_call_present(response: Mapping[str, Any]) -> bool:
    """Return True if at least one tool/function call was returned."""
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return False
    for choice in choices:
        if not isinstance(choice, Mapping):
            continue
        message = choice.get("message")
        if not isinstance(message, Mapping):
            continue
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            return True
        # Legacy function_call field
        if message.get("function_call"):
            return True
    return False


def response_is_valid_json(response: Mapping[str, Any]) -> bool:
    """Return True if the first choice content parses as valid JSON."""
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return False
    first = choices[0]
    if not isinstance(first, Mapping):
        return False
    message = first.get("message") or first.get("delta")
    if not isinstance(message, Mapping):
        return False
    content = message.get("content")
    if not isinstance(content, str):
        return False
    try:
        json.loads(content.strip())
        return True
    except (ValueError, TypeError):
        return False


def usage_tokens_present(response: Mapping[str, Any]) -> bool:
    """Return True if usage metadata with token counts is present."""
    usage = response.get("usage")
    if not isinstance(usage, Mapping):
        return False
    return isinstance(usage.get("prompt_tokens"), int) and isinstance(
        usage.get("completion_tokens"), int
    )


# Registry for the runner to look up checks by name
SCORING_FUNCTIONS: dict[str, Callable[[Mapping[str, Any]], bool]] = {
    "response_has_content": response_has_content,
    "no_error_response": no_error_response,
    "tool_call_present": tool_call_present,
    "response_is_valid_json": response_is_valid_json,
    "usage_tokens_present": usage_tokens_present,
}
