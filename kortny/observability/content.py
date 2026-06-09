"""Render LLM prompt/response content for observability, gated by capture mode.

Capture is controlled by ``OBSERVABILITY_CAPTURE_CONTENT`` (see
``Settings.observability_capture_content``):

- ``"metadata"`` — no prompt/response content (token/cost scalars only). Default.
- ``"summaries"`` — content captured but truncated to a per-field cap.
- ``"full"`` — raw, untruncated content.

The same rendered structures feed two surfaces:

1. ``task_events`` payloads (durable DB rows the dashboard renders), and
2. OpenInference span attributes (so Phoenix/Langfuse render prompt + response).

Both engines (the custom ``LLMService`` loop and the ADK runtime) route through
these helpers so capture behavior stays consistent.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kortny.llm.types import ChatMessage, Completion

# Per-field character cap applied in "summaries" mode. Matches the OTEL span
# attribute length cap so span and DB truncation agree.
SUMMARY_MAX_CHARS = 2_000

# Cap on indexed per-message span attributes (input.value still holds the full
# JSON). Keeps the LLM span under the 96-attribute span limit.
_MAX_INDEXED_SPAN_MESSAGES = 20


def captures_content(mode: str) -> bool:
    """Return whether the given capture mode records prompt/response content."""

    return mode in ("summaries", "full")


def truncate_for_mode(text: str | None, mode: str) -> str | None:
    """Truncate ``text`` for "summaries" mode; pass through otherwise."""

    if text is None or mode != "summaries":
        return text
    if len(text) <= SUMMARY_MAX_CHARS:
        return text
    return text[: SUMMARY_MAX_CHARS - 1].rstrip() + "…"


def _truncate_json_strings(value: Any, mode: str) -> Any:
    if isinstance(value, str):
        return truncate_for_mode(value, mode)
    if isinstance(value, dict):
        return {
            key: _truncate_json_strings(child, mode) for key, child in value.items()
        }
    if isinstance(value, list):
        return [_truncate_json_strings(child, mode) for child in value]
    return value


def _render_arguments(arguments: Any, mode: str) -> Any:
    if mode != "summaries":
        return arguments
    return _truncate_json_strings(arguments, mode)


def render_chat_messages(
    messages: Sequence[ChatMessage],
    mode: str,
) -> list[dict[str, Any]] | None:
    """Render provider-neutral chat messages, or ``None`` in metadata mode."""

    if not captures_content(mode):
        return None
    rendered: list[dict[str, Any]] = []
    for message in messages:
        entry: dict[str, Any] = {"role": message.role}
        if message.content is not None:
            entry["content"] = truncate_for_mode(message.content, mode)
        if message.tool_call_id:
            entry["tool_call_id"] = message.tool_call_id
        if message.tool_calls:
            entry["tool_calls"] = [
                {
                    "id": call.id,
                    "name": call.name,
                    "arguments": _render_arguments(call.arguments, mode),
                }
                for call in message.tool_calls
            ]
        rendered.append(entry)
    return rendered


def render_completion(
    completion: Completion,
    mode: str,
) -> dict[str, Any] | None:
    """Render a completion's content + tool calls, or ``None`` in metadata mode."""

    if not captures_content(mode):
        return None
    rendered: dict[str, Any] = {}
    if completion.content is not None:
        rendered["content"] = truncate_for_mode(completion.content, mode)
    if completion.tool_calls:
        rendered["tool_calls"] = [
            {
                "id": call.id,
                "name": call.name,
                "arguments": _render_arguments(call.arguments, mode),
            }
            for call in completion.tool_calls
        ]
    return rendered or None


def render_text_messages(
    messages: Sequence[dict[str, Any]],
    mode: str,
) -> list[dict[str, Any]] | None:
    """Render already-extracted ``{"role", "content"}`` dicts (ADK side)."""

    if not captures_content(mode):
        return None
    rendered: list[dict[str, Any]] = []
    for message in messages:
        entry: dict[str, Any] = {"role": str(message.get("role") or "")}
        content = message.get("content")
        if content is not None:
            entry["content"] = truncate_for_mode(str(content), mode)
        tool_calls = message.get("tool_calls")
        if tool_calls:
            entry["tool_calls"] = _truncate_json_strings(tool_calls, mode)
        rendered.append(entry)
    return rendered


def llm_span_attributes(
    *,
    request_messages: list[dict[str, Any]] | None = None,
    response: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build OpenInference span attributes for captured prompt/response.

    Returns an empty dict when nothing is captured, so callers can splat it
    unconditionally. ``input.value`` / ``output.value`` carry the full JSON
    (the span exporter truncates to its own length cap); indexed
    ``llm.input_messages.*`` attributes are capped to keep within span limits.
    """

    attributes: dict[str, Any] = {}
    if request_messages is not None:
        attributes["input.value"] = json.dumps(request_messages, default=str)
        attributes["input.mime_type"] = "application/json"
        for index, message in enumerate(request_messages[:_MAX_INDEXED_SPAN_MESSAGES]):
            attributes[f"llm.input_messages.{index}.message.role"] = message.get(
                "role", ""
            )
            content = message.get("content")
            if content is not None:
                attributes[f"llm.input_messages.{index}.message.content"] = content
    if response is not None:
        attributes["output.value"] = json.dumps(response, default=str)
        attributes["output.mime_type"] = "application/json"
        content = response.get("content")
        if content is not None:
            attributes["llm.output_messages.0.message.role"] = "assistant"
            attributes["llm.output_messages.0.message.content"] = content
    return attributes
