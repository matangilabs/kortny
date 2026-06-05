"""Provider-neutral LiteLLM chat-completions adapter."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from typing import Any

import litellm

from kortny.config import Settings, load_settings
from kortny.llm.types import ChatMessage, Completion, TokenUsage, ToolCall
from kortny.tools.types import JsonObject, JsonSchema


class LiteLLMProvider:
    """Adapter that calls LiteLLM with Kortny's provider-neutral LLM protocol."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        api_base: str | None = None,
        api_version: str | None = None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float = 60.0,
    ) -> None:
        if not api_key.strip():
            raise ValueError("LLM API key is required")
        if not model.strip():
            raise ValueError("LLM model is required")

        self.api_key = api_key
        self.model = model
        self.api_base = api_base
        self.api_version = api_version
        self.extra_headers = dict(extra_headers or {})
        self.timeout = timeout

    @classmethod
    def from_settings(
        cls,
        settings: Settings | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> LiteLLMProvider:
        """Create a LiteLLM provider from application settings."""

        resolved_settings = settings or load_settings()
        api_key = kwargs.pop("api_key", resolved_settings.llm_api_key)
        provider_model = model or resolved_settings.llm_model
        if resolved_settings.llm_provider.value == "openrouter" and not (
            provider_model.startswith("openrouter/")
        ):
            provider_model = f"openrouter/{provider_model}"
        return cls(
            api_key=api_key,
            model=provider_model,
            **kwargs,
        )

    def complete(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[JsonSchema] = (),
        *,
        response_format: JsonObject | None = None,
    ) -> Completion:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "api_key": self.api_key,
            "messages": [_message_to_payload(message) for message in messages],
            "request_timeout": self.timeout,
        }
        if self.api_base is not None:
            kwargs["api_base"] = self.api_base
        if self.api_version is not None:
            kwargs["api_version"] = self.api_version
        if self.extra_headers:
            kwargs["extra_headers"] = self.extra_headers
        if tools:
            kwargs["tools"] = [_tool_to_openai_payload(tool) for tool in tools]
        if response_format is not None:
            kwargs["response_format"] = response_format

        response = litellm.completion(**kwargs)
        return _parse_completion(response, fallback_model=self.model)


def create_litellm_provider(
    settings: Settings | None = None,
    *,
    model: str | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    endpoint: str | None = None,
    api_version: str | None = None,
    extra_headers: Mapping[str, str] | None = None,
    **kwargs: Any,
) -> LiteLLMProvider:
    """Create the provider-neutral direct LLM adapter."""

    resolved_settings = settings or load_settings()
    return LiteLLMProvider.from_settings(
        resolved_settings,
        model=model,
        api_key=api_key or resolved_settings.llm_api_key,
        api_base=api_base or endpoint,
        api_version=api_version,
        extra_headers=extra_headers,
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


def _parse_completion(response: object, *, fallback_model: str) -> Completion:
    choices = _get(response, "choices")
    if not isinstance(choices, Sequence) or isinstance(choices, str | bytes):
        raise ValueError("LiteLLM response is missing choices")
    if not choices:
        raise ValueError("LiteLLM response choices are empty")

    first_choice = choices[0]
    message = _get(first_choice, "message")
    if message is None:
        raise ValueError("LiteLLM choice is missing message")

    content = _get(message, "content")
    if content is not None and not isinstance(content, str):
        content = str(content)

    model = _get(response, "model")
    response_id = _get(response, "id")
    return Completion(
        content=content,
        tool_calls=_parse_tool_calls(_get(message, "tool_calls", ())),
        usage=_parse_usage(_get(response, "usage")),
        cost_usd=_completion_cost_usd(response, model=fallback_model),
        response_id=response_id if isinstance(response_id, str) else None,
        model=model if isinstance(model, str) else fallback_model,
    )


def _parse_usage(raw_usage: object) -> TokenUsage:
    if raw_usage is None:
        return TokenUsage(input_tokens=0, output_tokens=0)

    input_tokens = _get(raw_usage, "prompt_tokens", _get(raw_usage, "input_tokens", 0))
    output_tokens = _get(
        raw_usage,
        "completion_tokens",
        _get(raw_usage, "output_tokens", 0),
    )
    return TokenUsage(
        input_tokens=_coerce_token_count(input_tokens),
        output_tokens=_coerce_token_count(output_tokens),
    )


def _coerce_token_count(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return 0


def _completion_cost_usd(response: object, *, model: str) -> Decimal | None:
    usage = _get(response, "usage")
    raw_usage_cost = _get(usage, "cost")
    usage_cost = _coerce_cost(raw_usage_cost)
    if usage_cost is not None:
        return usage_cost
    try:
        cost = litellm.completion_cost(completion_response=response, model=model)
    except Exception:
        return None
    return _coerce_cost(cost)


def _coerce_cost(value: object) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float | str | Decimal):
        try:
            cost = Decimal(str(value))
        except InvalidOperation:
            return None
        if cost < 0:
            return None
        return cost
    return None


def _parse_tool_calls(raw_tool_calls: object) -> tuple[ToolCall, ...]:
    if not isinstance(raw_tool_calls, Sequence) or isinstance(
        raw_tool_calls, str | bytes
    ):
        return ()

    tool_calls: list[ToolCall] = []
    for raw_tool_call in raw_tool_calls:
        function = _get(raw_tool_call, "function")
        if function is None:
            continue

        tool_call_id = _get(raw_tool_call, "id")
        name = _get(function, "name")
        if not isinstance(tool_call_id, str) or not isinstance(name, str):
            continue

        tool_calls.append(
            ToolCall(
                id=tool_call_id,
                name=name,
                arguments=_parse_tool_arguments(_get(function, "arguments")),
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


def _get(value: object, key: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)
