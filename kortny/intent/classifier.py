"""LLM-backed app-wide intent classification."""

from __future__ import annotations

import json
import uuid
from collections.abc import Sequence
from typing import Protocol

from pydantic import ValidationError

from kortny.intent.models import IntentDecision, IntentRequest
from kortny.intent.prompts import INTENT_CLASSIFIER_SYSTEM_PROMPT
from kortny.llm import ChatMessage, Completion
from kortny.tools.types import JsonObject, JsonSchema

INTENT_RESPONSE_FORMAT: JsonObject = {"type": "json_object"}


class IntentClassificationError(RuntimeError):
    """Raised when intent classification cannot produce a valid decision."""


class IntentTrackedLLMClient(Protocol):
    """Subset of LLMService used by the intent classifier."""

    def complete(
        self,
        *,
        task_id: uuid.UUID,
        messages: Sequence[ChatMessage],
        tools: Sequence[JsonSchema] = (),
        response_format: JsonObject | None = None,
    ) -> Completion:
        """Complete one intent classification turn."""


class IntentChatClient(Protocol):
    """Subset of provider clients usable before a durable task exists."""

    model: str

    def complete(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[JsonSchema] = (),
        *,
        response_format: JsonObject | None = None,
    ) -> Completion:
        """Complete one intent classification turn without usage tracking."""


class IntentClassifier(Protocol):
    """Classifies inbound app messages into routing decisions."""

    def classify(
        self,
        *,
        request: IntentRequest,
        task_id: uuid.UUID | None = None,
    ) -> IntentDecision:
        """Return the structured intent decision for a message."""


class LLMIntentClassifier:
    """Provider-neutral classifier using the existing LLM service boundary."""

    def __init__(
        self,
        *,
        llm: IntentTrackedLLMClient | None = None,
        provider: IntentChatClient | None = None,
    ) -> None:
        if (llm is None) == (provider is None):
            raise ValueError("provide exactly one of llm or provider")
        self.llm = llm
        self.provider = provider

    def classify(
        self,
        *,
        request: IntentRequest,
        task_id: uuid.UUID | None = None,
    ) -> IntentDecision:
        messages = (
            ChatMessage(role="system", content=INTENT_CLASSIFIER_SYSTEM_PROMPT),
            ChatMessage(role="user", content=_request_payload(request)),
        )
        if self.llm is not None:
            if task_id is None:
                raise IntentClassificationError(
                    "usage-tracked intent classification requires task_id"
                )
            completion = self.llm.complete(
                task_id=task_id,
                messages=messages,
                response_format=INTENT_RESPONSE_FORMAT,
            )
        else:
            if self.provider is None:
                raise IntentClassificationError("intent classifier has no LLM client")
            completion = self.provider.complete(
                messages,
                response_format=INTENT_RESPONSE_FORMAT,
            )
        return parse_intent_decision(completion.content)


def parse_intent_decision(content: str | None) -> IntentDecision:
    """Parse and validate a model-produced intent decision."""

    if content is None or not content.strip():
        raise IntentClassificationError("intent classifier returned empty content")
    try:
        return IntentDecision.model_validate_json(_extract_json_object(content))
    except (ValueError, ValidationError) as exc:
        raise IntentClassificationError(
            "intent classifier returned invalid JSON"
        ) from exc


def _request_payload(request: IntentRequest) -> str:
    return request.model_dump_json()


def _extract_json_object(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object found")

    candidate = stripped[start : end + 1]
    json.loads(candidate)
    return candidate
