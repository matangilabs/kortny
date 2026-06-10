"""Tests for prompt/response content capture gating and rendering."""

from __future__ import annotations

from decimal import Decimal

from kortny.llm.types import ChatMessage, Completion, TokenUsage, ToolCall
from kortny.observability.content import (
    SUMMARY_MAX_CHARS,
    captures_content,
    llm_span_attributes,
    render_chat_messages,
    render_completion,
    render_text_messages,
    truncate_for_mode,
)


def _completion(
    content: str | None, tool_calls: tuple[ToolCall, ...] = ()
) -> Completion:
    return Completion(
        content=content,
        tool_calls=tool_calls,
        usage=TokenUsage(input_tokens=10, output_tokens=5),
        cost_usd=Decimal("0.001"),
    )


def test_metadata_mode_captures_nothing() -> None:
    assert not captures_content("metadata")
    messages = [ChatMessage(role="user", content="hi")]
    assert render_chat_messages(messages, "metadata") is None
    assert render_completion(_completion("hello"), "metadata") is None


def test_full_mode_captures_raw_content() -> None:
    long = "x" * (SUMMARY_MAX_CHARS + 500)
    messages = [
        ChatMessage(role="system", content="be helpful"),
        ChatMessage(role="user", content=long),
    ]
    rendered = render_chat_messages(messages, "full")
    assert rendered is not None
    assert rendered[0] == {"role": "system", "content": "be helpful"}
    # full mode keeps the untruncated prompt
    assert rendered[1]["content"] == long


def test_summaries_mode_truncates() -> None:
    long = "y" * (SUMMARY_MAX_CHARS + 500)
    truncated = truncate_for_mode(long, "summaries")
    assert truncated is not None
    assert truncated.endswith("…")
    assert len(truncated) == SUMMARY_MAX_CHARS
    # short content is untouched
    assert truncate_for_mode("short", "summaries") == "short"
    rendered = render_completion(_completion(long), "summaries")
    assert rendered is not None
    assert rendered["content"].endswith("…")


def test_completion_tool_calls_captured() -> None:
    completion = _completion(
        None,
        tool_calls=(ToolCall(id="t1", name="search", arguments={"q": "kortny"}),),
    )
    rendered = render_completion(completion, "full")
    assert rendered == {
        "tool_calls": [{"id": "t1", "name": "search", "arguments": {"q": "kortny"}}]
    }


def test_render_text_messages_for_adk_shape() -> None:
    rendered = render_text_messages(
        [{"role": "user", "content": "z" * (SUMMARY_MAX_CHARS + 10)}],
        "summaries",
    )
    assert rendered is not None
    assert rendered[0]["content"].endswith("…")
    assert render_text_messages([{"role": "user", "content": "x"}], "metadata") is None


def test_span_attributes_emit_openinference_keys() -> None:
    request = [{"role": "user", "content": "hello"}]
    response = {"content": "hi there"}
    attrs = llm_span_attributes(request_messages=request, response=response)
    assert attrs["input.mime_type"] == "application/json"
    assert attrs["llm.input_messages.0.message.role"] == "user"
    assert attrs["llm.input_messages.0.message.content"] == "hello"
    assert attrs["llm.output_messages.0.message.role"] == "assistant"
    assert attrs["llm.output_messages.0.message.content"] == "hi there"
    assert "hello" in attrs["input.value"]


def test_span_attributes_empty_when_nothing_captured() -> None:
    assert llm_span_attributes() == {}
