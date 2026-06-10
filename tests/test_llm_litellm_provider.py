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
