"""Provider-neutral LLM types.

These intentionally stay close to OpenAI/OpenRouter chat completion shapes so
the same boundary forwards cleanly through LiteLLM.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
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
class ImagePart:
    """An image attached to a chat message (HIG-279 vision).

    Carries raw bytes (never base64) so a ``ChatMessage`` is always log-safe;
    base64 encoding happens only at the provider serialization boundary.
    """

    data: bytes = field(repr=False, compare=False)
    mime: str
    source: str  # short provenance label for traces/audit, e.g. "slack_file:F123" — never the bytes
    alt: str | None = None  # optional caption; untrusted, never an instruction

    @property
    def byte_size(self) -> int:
        return len(self.data)

    @property
    def sha256_prefix(self) -> str:
        return hashlib.sha256(self.data).hexdigest()[:12]


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """A provider-neutral chat message."""

    role: str
    content: str | None
    tool_call_id: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    images: tuple[ImagePart, ...] = ()


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Token usage returned by an LLM provider.

    ``input_tokens`` is the LiteLLM-normalized total prompt count (cached +
    uncached). ``cache_creation_input_tokens`` and ``cache_read_input_tokens``
    are a split *within* that total, not additions to it (HIG-196).
    """

    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

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
        max_output_tokens: int | None = None,
    ) -> Completion:
        """Complete a chat turn with optional tool declarations.

        ``max_output_tokens`` overrides the provider's default completion-token
        cap for this one call (HIG-220 effort steering) — e.g. a utility prompt
        clamps verbosity without changing the global setting.
        """
