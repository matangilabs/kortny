"""OpenRouter chat-completions provider."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from typing import Any

import httpx

from kortny.config import LLMProvider as SettingsLLMProvider
from kortny.config import Settings, load_settings
from kortny.llm.types import ChatMessage, Completion, TokenUsage, ToolCall
from kortny.tools.types import JsonObject, JsonSchema

OPENROUTER_CHAT_COMPLETIONS_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterProvider:
    """OpenRouter adapter using the OpenAI-compatible chat completions API."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        endpoint: str = OPENROUTER_CHAT_COMPLETIONS_ENDPOINT,
        timeout: float = 60.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("LLM_API_KEY is required for OpenRouter")
        if not model.strip():
            raise ValueError("LLM_MODEL is required for OpenRouter")

        self.api_key = api_key
        self.model = model
        self.endpoint = endpoint
        self.timeout = timeout
        self.transport = transport

    @classmethod
    def from_settings(
        cls,
        settings: Settings | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> OpenRouterProvider:
        """Create an OpenRouter provider from application settings."""

        resolved_settings = settings or load_settings()
        return cls(
            api_key=resolved_settings.llm_api_key,
            model=model or resolved_settings.llm_model,
            **kwargs,
        )

    def complete(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[JsonSchema] = (),
        *,
        response_format: JsonObject | None = None,
        max_output_tokens: int | None = None,
    ) -> Completion:
        payload: JsonObject = {
            "model": self.model,
            "messages": [_message_to_payload(message) for message in messages],
        }
        if tools:
            payload["tools"] = [_tool_to_openai_payload(tool) for tool in tools]
        if response_format is not None:
            payload["response_format"] = response_format
        # HIG-220 effort steering: LLMService forwards this for clamped utility
        # prompts. The LLMProvider protocol declares it, so every implementation
        # must accept it or the call raises TypeError. OpenRouter's field is
        # ``max_tokens``.
        if max_output_tokens is not None:
            payload["max_tokens"] = max_output_tokens

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-OpenRouter-Title": "Kortny",
        }

        with httpx.Client(transport=self.transport, timeout=self.timeout) as client:
            response = client.post(self.endpoint, headers=headers, json=payload)
            response.raise_for_status()
            response_payload = response.json()

        if not isinstance(response_payload, dict):
            raise ValueError("OpenRouter response must be a JSON object")
        return _parse_completion(response_payload, fallback_model=self.model)


def create_llm_provider(
    settings: Settings | None = None,
    *,
    model: str | None = None,
    provider_kind: str | None = None,
    api_key: str | None = None,
    endpoint: str | None = None,
    **kwargs: Any,
) -> OpenRouterProvider:
    """Create the configured MVP provider.

    Only OpenRouter is implemented for HIG-16. OpenAI/Anthropic can be added
    behind the same protocol later; ADK can also adapt this boundary via LiteLLM.
    """

    resolved_settings = settings or load_settings()
    resolved_provider = provider_kind or resolved_settings.llm_provider.value
    if resolved_provider != SettingsLLMProvider.openrouter.value:
        raise NotImplementedError(
            f"LLM provider {resolved_provider!r} is not "
            "implemented yet; use 'openrouter' for the MVP provider"
        )
    return OpenRouterProvider(
        api_key=api_key or resolved_settings.llm_api_key,
        model=model or resolved_settings.llm_model,
        endpoint=endpoint or OPENROUTER_CHAT_COMPLETIONS_ENDPOINT,
        **kwargs,
    )


def _message_to_payload(message: ChatMessage) -> JsonObject:
    payload: JsonObject = {
        "role": message.role,
        "content": message.content,
    }
    if message.tool_call_id is not None:
        payload["tool_call_id"] = message.tool_call_id
    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.name,
                    "arguments": json.dumps(tool_call.arguments, separators=(",", ":")),
                },
            }
            for tool_call in message.tool_calls
        ]
    return payload


def _tool_to_openai_payload(tool: Mapping[str, Any]) -> JsonObject:
    name = tool.get("name")
    description = tool.get("description")
    parameters = tool.get("parameters")
    if not isinstance(name, str) or not name:
        raise ValueError("Tool schema requires a non-empty string name")
    if not isinstance(description, str):
        raise ValueError(f"Tool {name!r} requires a string description")
    if not isinstance(parameters, dict):
        raise ValueError(f"Tool {name!r} requires object parameters")

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


def _parse_completion(payload: JsonObject, *, fallback_model: str) -> Completion:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("OpenRouter response is missing choices")

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise ValueError("OpenRouter choice must be an object")
    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise ValueError("OpenRouter choice is missing message")

    content = message.get("content")
    if content is not None and not isinstance(content, str):
        content = str(content)

    usage = _parse_usage(payload.get("usage", {}))
    response_id = payload.get("id")
    response_model = payload.get("model")
    return Completion(
        content=content,
        tool_calls=_parse_tool_calls(message.get("tool_calls", [])),
        usage=usage,
        cost_usd=_parse_usage_cost(payload.get("usage", {})),
        response_id=response_id if isinstance(response_id, str) else None,
        model=response_model if isinstance(response_model, str) else fallback_model,
    )


def _parse_usage(raw_usage: object) -> TokenUsage:
    if not isinstance(raw_usage, dict):
        return TokenUsage(input_tokens=0, output_tokens=0)

    input_tokens = raw_usage.get("prompt_tokens", raw_usage.get("input_tokens", 0))
    output_tokens = raw_usage.get(
        "completion_tokens",
        raw_usage.get("output_tokens", 0),
    )
    return TokenUsage(
        input_tokens=_coerce_token_count(input_tokens),
        output_tokens=_coerce_token_count(output_tokens),
    )


def _coerce_token_count(value: object) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return 0


def _parse_usage_cost(raw_usage: object) -> Decimal | None:
    if not isinstance(raw_usage, dict):
        return None

    raw_cost = raw_usage.get("cost")
    if raw_cost is None:
        return None
    if isinstance(raw_cost, bool):
        return None
    if isinstance(raw_cost, int | float | str):
        try:
            cost = Decimal(str(raw_cost))
        except InvalidOperation:
            return None
        if cost < 0:
            return None
        return cost
    return None


def _parse_tool_calls(raw_tool_calls: object) -> tuple[ToolCall, ...]:
    if not isinstance(raw_tool_calls, list):
        return ()

    tool_calls: list[ToolCall] = []
    for raw_tool_call in raw_tool_calls:
        if not isinstance(raw_tool_call, dict):
            continue

        function = raw_tool_call.get("function")
        if not isinstance(function, dict):
            continue

        tool_call_id = raw_tool_call.get("id")
        name = function.get("name")
        if not isinstance(tool_call_id, str) or not isinstance(name, str):
            continue

        tool_calls.append(
            ToolCall(
                id=tool_call_id,
                name=name,
                arguments=_parse_tool_arguments(function.get("arguments")),
            )
        )

    return tuple(tool_calls)


def _parse_tool_arguments(raw_arguments: object) -> JsonObject:
    if raw_arguments in (None, ""):
        return {}
    if isinstance(raw_arguments, dict):
        return raw_arguments
    if not isinstance(raw_arguments, str):
        raise ValueError("Tool call arguments must be a JSON object string")

    parsed = json.loads(raw_arguments)
    if not isinstance(parsed, dict):
        raise ValueError("Tool call arguments must decode to a JSON object")
    return parsed
