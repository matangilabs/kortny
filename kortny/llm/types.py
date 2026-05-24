"""Provider-neutral LLM types.

These intentionally stay close to OpenAI/OpenRouter chat completion shapes so
the same boundary can later be adapted to ADK with LiteLLM.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol

from kortny.tools.types import JsonObject, JsonSchema


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A model-requested tool invocation."""

    id: str
    name: str
    arguments: JsonObject


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """A provider-neutral chat message."""

    role: str
    content: str | None
    tool_call_id: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Token usage returned by an LLM provider."""

    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True, slots=True)
class Completion:
    """Normalized LLM completion response."""

    content: str | None
    tool_calls: tuple[ToolCall, ...]
    usage: TokenUsage
    cost_usd: Decimal | None = None
    response_id: str | None = None
    model: str | None = None


class LLMProvider(Protocol):
    """Minimal provider protocol used by the coordinator and usage tracker."""

    model: str

    def complete(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[JsonSchema] = (),
        *,
        response_format: JsonObject | None = None,
    ) -> Completion:
        """Complete a chat turn with optional tool declarations."""
