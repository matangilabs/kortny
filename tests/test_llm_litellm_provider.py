from types import SimpleNamespace
from typing import Any

import pytest

from kortny.llm.litellm_provider import LiteLLMProvider
from kortny.llm.types import ChatMessage, TokenUsage, ToolCall


def test_litellm_provider_calls_completion_and_parses_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    response = SimpleNamespace(
        id="chatcmpl-123",
        model="openrouter/qwen/qwen3.5-flash-20260224",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="call-123",
                            function=SimpleNamespace(
                                name="web_search",
                                arguments='{"query":"kortny"}',
                            ),
                        )
                    ],
                )
            )
        ],
        usage=SimpleNamespace(prompt_tokens=12, completion_tokens=4),
    )

    def fake_completion(**kwargs: Any) -> object:
        captured.update(kwargs)
        return response

    monkeypatch.setattr(
        "kortny.llm.litellm_provider.litellm.completion",
        fake_completion,
    )
    monkeypatch.setattr(
        "kortny.llm.litellm_provider.litellm.completion_cost",
        lambda completion_response, model: 0.000123,
    )

    provider = LiteLLMProvider(
        api_key="provider-key",
        model="openrouter/qwen/qwen3.5-flash-02-23",
        api_base="https://example.test/v1",
        api_version="2026-06-01",
        extra_headers={"X-Test": "ok"},
    )

    completion = provider.complete(
        [ChatMessage(role="user", content="Search for Kortny")],
        [
            {
                "name": "web_search",
                "description": "Search the web.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            }
        ],
        response_format={"type": "json_object"},
    )

    assert captured["model"] == "openrouter/qwen/qwen3.5-flash-02-23"
    assert captured["api_key"] == "provider-key"
    assert captured["api_base"] == "https://example.test/v1"
    assert captured["api_version"] == "2026-06-01"
    assert captured["extra_headers"] == {"X-Test": "ok"}
    assert captured["messages"] == [{"role": "user", "content": "Search for Kortny"}]
    assert captured["tools"][0]["function"]["name"] == "web_search"
    assert captured["response_format"] == {"type": "json_object"}

    assert completion.response_id == "chatcmpl-123"
    assert completion.model == "openrouter/qwen/qwen3.5-flash-20260224"
    assert completion.usage == TokenUsage(input_tokens=12, output_tokens=4)
    assert completion.cost_usd is not None
    assert str(completion.cost_usd) == "0.000123"
    assert completion.tool_calls == (
        ToolCall(id="call-123", name="web_search", arguments={"query": "kortny"}),
    )


# --- HIG-196 prompt caching ---------------------------------------------------


def _make_response(usage: object, *, model: str = "anthropic/claude-x") -> object:
    return SimpleNamespace(
        id="chatcmpl-cache",
        model=model,
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="ok", tool_calls=[]),
            )
        ],
        usage=usage,
    )


def _capture_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def fake_completion(**kwargs: Any) -> object:
        captured.update(kwargs)
        return _make_response(SimpleNamespace(prompt_tokens=5000, completion_tokens=10))

    monkeypatch.setattr(
        "kortny.llm.litellm_provider.litellm.completion", fake_completion
    )
    monkeypatch.setattr(
        "kortny.llm.litellm_provider.litellm.completion_cost",
        lambda completion_response, model: 0.0,
    )
    return captured


def test_cache_control_injected_for_claude_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _capture_completion(monkeypatch)
    provider = LiteLLMProvider(api_key="k", model="anthropic/claude-opus-4-8")

    provider.complete(
        [
            ChatMessage(role="system", content="persona"),
            ChatMessage(role="user", content="hi"),
        ],
        [
            {"name": "a", "description": "A", "parameters": {"type": "object"}},
            {"name": "b", "description": "B", "parameters": {"type": "object"}},
        ],
    )

    # Last tool marked.
    assert captured["tools"][-1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in captured["tools"][0]
    # System content normalized to a block list with the marker on the last block.
    system_msg = captured["messages"][0]
    assert system_msg["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert system_msg["content"][-1]["text"] == "persona"
    # Final message marked.
    assert captured["messages"][-1]["content"][-1]["cache_control"] == {
        "type": "ephemeral"
    }


def test_cache_control_breakpoint_count_capped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _capture_completion(monkeypatch)
    provider = LiteLLMProvider(api_key="k", model="anthropic/claude-opus-4-8")

    provider.complete(
        [
            ChatMessage(role="system", content="persona"),
            ChatMessage(role="user", content="one"),
            ChatMessage(role="assistant", content="two"),
            ChatMessage(role="user", content="three"),
        ],
        [{"name": "a", "description": "A", "parameters": {"type": "object"}}],
    )

    marker_count = 0
    for tool in captured["tools"]:
        if "cache_control" in tool:
            marker_count += 1
    for message in captured["messages"]:
        content = message["content"]
        if isinstance(content, list):
            marker_count += sum(
                1
                for part in content
                if isinstance(part, dict) and "cache_control" in part
            )
    # Last tool + last system + last message = exactly 3, never above the max of 4.
    assert marker_count == 3
    assert marker_count <= 4


def test_cache_control_not_injected_for_non_claude_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _capture_completion(monkeypatch)
    provider = LiteLLMProvider(api_key="k", model="openrouter/qwen/qwen3.5")

    provider.complete(
        [ChatMessage(role="system", content="persona")],
        [{"name": "a", "description": "A", "parameters": {"type": "object"}}],
    )

    assert "cache_control" not in captured["tools"][0]
    # String content left as a plain string when not Claude-gated.
    assert captured["messages"][0]["content"] == "persona"


def test_cache_control_not_injected_when_flag_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _capture_completion(monkeypatch)
    provider = LiteLLMProvider(
        api_key="k",
        model="anthropic/claude-opus-4-8",
        prompt_cache_enabled=False,
    )

    provider.complete(
        [ChatMessage(role="system", content="persona")],
        [{"name": "a", "description": "A", "parameters": {"type": "object"}}],
    )

    assert "cache_control" not in captured["tools"][0]
    assert captured["messages"][0]["content"] == "persona"


def test_cache_control_idempotent_when_already_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kortny.llm.litellm_provider import _inject_cache_control

    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "hi",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        }
    ]
    tools: list[dict[str, Any]] = [
        {"type": "function", "cache_control": {"type": "ephemeral"}}
    ]
    first_msgs, first_tools = _inject_cache_control(messages, tools)
    second_msgs, second_tools = _inject_cache_control(first_msgs, first_tools)

    assert first_msgs == second_msgs
    assert first_tools == second_tools
    # Original inputs not mutated.
    assert messages[0]["content"][-1]["cache_control"] == {"type": "ephemeral"}


def test_inject_cache_control_does_not_mutate_inputs() -> None:
    from kortny.llm.litellm_provider import _inject_cache_control

    messages: list[dict[str, Any]] = [{"role": "user", "content": "hi"}]
    tools: list[dict[str, Any]] = [{"type": "function", "function": {"name": "a"}}]
    new_messages, new_tools = _inject_cache_control(messages, tools)

    assert messages == [{"role": "user", "content": "hi"}]
    assert "cache_control" not in tools[0]
    new_content = new_messages[0]["content"]
    assert isinstance(new_content, list)


def test_inject_cache_control_handles_empty_tools() -> None:
    from kortny.llm.litellm_provider import _inject_cache_control

    messages: list[dict[str, Any]] = [{"role": "user", "content": "hi"}]
    new_messages, new_tools = _inject_cache_control(messages, [])
    assert new_tools == []
    content = new_messages[0]["content"]
    assert isinstance(content, list)
    assert content[-1]["cache_control"] == {"type": "ephemeral"}


def test_openrouter_route_sets_usage_include(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _capture_completion(monkeypatch)
    provider = LiteLLMProvider(api_key="k", model="openrouter/anthropic/claude-opus")

    provider.complete([ChatMessage(role="user", content="hi")], [])

    assert captured["extra_body"] == {"usage": {"include": True}}


def test_non_openrouter_route_omits_usage_include(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _capture_completion(monkeypatch)
    provider = LiteLLMProvider(api_key="k", model="anthropic/claude-opus-4-8")

    provider.complete([ChatMessage(role="user", content="hi")], [])

    assert "extra_body" not in captured


def test_usage_extraction_public_cache_creation_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_completion(**kwargs: Any) -> object:
        captured.update(kwargs)
        return _make_response(
            SimpleNamespace(
                prompt_tokens=5000,
                completion_tokens=10,
                cache_creation_input_tokens=1200,
                prompt_tokens_details=SimpleNamespace(cached_tokens=3000),
            )
        )

    monkeypatch.setattr(
        "kortny.llm.litellm_provider.litellm.completion", fake_completion
    )
    monkeypatch.setattr(
        "kortny.llm.litellm_provider.litellm.completion_cost",
        lambda completion_response, model: 0.0,
    )
    provider = LiteLLMProvider(api_key="k", model="anthropic/claude-opus-4-8")
    completion = provider.complete([ChatMessage(role="user", content="hi")], [])

    assert completion.usage.input_tokens == 5000
    assert completion.usage.cache_creation_input_tokens == 1200
    assert completion.usage.cache_read_input_tokens == 3000


def test_usage_extraction_private_cache_creation_fallback() -> None:
    from kortny.llm.litellm_provider import _parse_usage

    usage = _parse_usage(
        SimpleNamespace(
            prompt_tokens=4000,
            completion_tokens=5,
            _cache_creation_input_tokens=800,
            prompt_tokens_details=SimpleNamespace(cached_tokens=1000),
        )
    )
    assert usage.cache_creation_input_tokens == 800
    assert usage.cache_read_input_tokens == 1000


def test_usage_extraction_openai_cached_tokens_only() -> None:
    from kortny.llm.litellm_provider import _parse_usage

    usage = _parse_usage(
        SimpleNamespace(
            prompt_tokens=2000,
            completion_tokens=5,
            prompt_tokens_details=SimpleNamespace(cached_tokens=512),
        )
    )
    assert usage.cache_read_input_tokens == 512
    assert usage.cache_creation_input_tokens == 0


def test_usage_extraction_absent_cache_fields() -> None:
    from kortny.llm.litellm_provider import _parse_usage

    usage = _parse_usage(SimpleNamespace(prompt_tokens=100, completion_tokens=5))
    assert usage.cache_read_input_tokens == 0
    assert usage.cache_creation_input_tokens == 0


def _text_response() -> object:
    return SimpleNamespace(
        id="chatcmpl-cap",
        model="anthropic/claude-opus-4.8",
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=[]))],
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=2),
    )


def test_litellm_provider_caps_max_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # HIG-265: cap requested completion tokens so the provider does not reserve
    # the model's full max output (e.g. 65k on Opus) per call, which inflates
    # cost-reservation and can 402 a budget-limited key.
    captured: dict[str, Any] = {}

    def fake_completion(**kwargs: Any) -> object:
        captured.update(kwargs)
        return _text_response()

    monkeypatch.setattr(
        "kortny.llm.litellm_provider.litellm.completion", fake_completion
    )
    monkeypatch.setattr(
        "kortny.llm.litellm_provider.litellm.completion_cost",
        lambda completion_response, model: 0.0,
    )

    provider = LiteLLMProvider(
        api_key="k",
        model="anthropic/claude-opus-4.8",
        max_output_tokens=16384,
    )
    provider.complete([ChatMessage(role="user", content="hi")])

    assert captured["max_tokens"] == 16384


def test_litellm_provider_omits_max_tokens_when_uncapped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_completion(**kwargs: Any) -> object:
        captured.update(kwargs)
        return _text_response()

    monkeypatch.setattr(
        "kortny.llm.litellm_provider.litellm.completion", fake_completion
    )
    monkeypatch.setattr(
        "kortny.llm.litellm_provider.litellm.completion_cost",
        lambda completion_response, model: 0.0,
    )

    provider = LiteLLMProvider(api_key="k", model="m")  # no cap
    provider.complete([ChatMessage(role="user", content="hi")])

    assert "max_tokens" not in captured
