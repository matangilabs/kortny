import json

import httpx
import pytest

from kortny.config.settings import LLMProvider as SettingsLLMProvider
from kortny.config.settings import Settings
from kortny.llm import ChatMessage, OpenRouterProvider, TokenUsage, ToolCall
from kortny.llm.openrouter import create_llm_provider


def make_settings(provider: SettingsLLMProvider, api_key: str, model: str) -> Settings:
    values: dict[str, object] = {
        "SLACK_BOT_TOKEN": "xoxb-test",
        "SLACK_APP_TOKEN": "xapp-test",
        "SLACK_SIGNING_SECRET": "signing-secret",
        "LLM_PROVIDER": provider,
        "LLM_API_KEY": api_key,
        "LLM_MODEL": model,
        "POSTGRES_URL": "postgresql://kortny:kortny@localhost/kortny",
    }
    return Settings.model_validate(values)


def test_openrouter_provider_posts_chat_completion_with_tools() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert str(request.url) == "https://openrouter.ai/api/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer openrouter-key"
        payload = json.loads(request.read().decode())
        assert payload["model"] == "openai/gpt-4o-mini"
        assert payload["tools"][0]["function"]["name"] == "web_search"
        return httpx.Response(
            200,
            json={
                "id": "gen-123",
                "model": "openai/gpt-4o-mini",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-123",
                                    "type": "function",
                                    "function": {
                                        "name": "web_search",
                                        "arguments": '{"query":"kortny"}',
                                    },
                                }
                            ],
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 120,
                    "completion_tokens": 30,
                    "total_tokens": 150,
                    "cost": 0.00042,
                },
            },
        )

    provider = OpenRouterProvider(
        api_key="openrouter-key",
        model="openai/gpt-4o-mini",
        transport=httpx.MockTransport(handler),
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
    )

    assert completion.response_id == "gen-123"
    assert completion.model == "openai/gpt-4o-mini"
    assert completion.content is None
    assert completion.usage == TokenUsage(input_tokens=120, output_tokens=30)
    assert completion.cost_usd is not None
    assert str(completion.cost_usd) == "0.00042"
    assert completion.tool_calls == (
        ToolCall(id="call-123", name="web_search", arguments={"query": "kortny"}),
    )


def test_openrouter_provider_serializes_prior_tool_messages() -> None:
    captured_payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_payloads.append(json.loads(request.read().decode()))
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "done"}}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    provider = OpenRouterProvider(
        api_key="openrouter-key",
        model="openai/gpt-4o-mini",
        transport=httpx.MockTransport(handler),
    )

    provider.complete(
        [
            ChatMessage(
                role="assistant",
                content=None,
                tool_calls=(
                    ToolCall(
                        id="call-123",
                        name="web_search",
                        arguments={"query": "kortny"},
                    ),
                ),
            ),
            ChatMessage(
                role="tool",
                content='{"results":[]}',
                tool_call_id="call-123",
            ),
        ]
    )

    assert captured_payloads[0]["messages"] == [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-123",
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "arguments": '{"query":"kortny"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": '{"results":[]}',
            "tool_call_id": "call-123",
        },
    ]


def test_openrouter_provider_sends_response_format() -> None:
    captured_payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_payloads.append(json.loads(request.read().decode()))
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"ok":true}'}}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    provider = OpenRouterProvider(
        api_key="openrouter-key",
        model="openai/gpt-4o-mini",
        transport=httpx.MockTransport(handler),
    )

    provider.complete(
        [ChatMessage(role="user", content="classify")],
        response_format={"type": "json_object"},
    )

    assert captured_payloads[0]["response_format"] == {"type": "json_object"}


def test_create_llm_provider_uses_openrouter_settings() -> None:
    settings = make_settings(
        SettingsLLMProvider.openrouter,
        "openrouter-key",
        "openai/gpt-4o-mini",
    )

    provider = create_llm_provider(
        settings,
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={})),
    )

    assert provider.model == "openai/gpt-4o-mini"


def test_create_llm_provider_rejects_unimplemented_provider() -> None:
    settings = make_settings(SettingsLLMProvider.openai, "openai-key", "gpt-4o-mini")

    with pytest.raises(NotImplementedError, match="not implemented"):
        create_llm_provider(settings)


def test_openrouter_provider_raises_for_http_errors() -> None:
    provider = OpenRouterProvider(
        api_key="openrouter-key",
        model="openai/gpt-4o-mini",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(401, json={"error": "unauthorized"})
        ),
    )

    with pytest.raises(httpx.HTTPStatusError):
        provider.complete([ChatMessage(role="user", content="hello")])
