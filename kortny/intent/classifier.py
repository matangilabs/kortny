"""LLM-backed app-wide intent classification."""

from __future__ import annotations

import json
import uuid
from collections.abc import Sequence
from typing import Protocol

from pydantic import ValidationError

from kortny.intent.models import (
    IntentClassification,
    IntentDecision,
    IntentFragment,
    IntentRequest,
    ModelTier,
)
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
        decision = parse_intent_decision(completion.content)
        decision = _with_deterministic_overrides(request, decision)
        return _with_mixed_follow_up_memory_override(request, decision)


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


def _with_deterministic_overrides(
    request: IntentRequest,
    decision: IntentDecision,
) -> IntentDecision:
    if not _is_memory_forget_request(request.text):
        return decision
    return decision.model_copy(
        update={
            "addressed_to_kortny": True,
            "classification": IntentClassification.task_request,
            "should_create_task": True,
            "should_ack_with_reaction": True,
            "suggested_reaction": "memo",
            "needs_channel_context": False,
            "needs_thread_context": request.is_thread_follow_up,
            "needs_file_context": False,
            "likely_tools": ["inspect_memory", "forget_fact"],
            "model_tier": ModelTier.cheap,
            "reason": "User asked Kortny to forget a stored memory or preference.",
        }
    )


def _with_mixed_follow_up_memory_override(
    request: IntentRequest,
    decision: IntentDecision,
) -> IntentDecision:
    if decision.primary_intent is not None:
        return decision
    if decision.classification is not IntentClassification.memory_candidate:
        return decision
    if not _has_memory_instruction(request.text):
        return decision
    if not _has_follow_up_instruction(request):
        return decision

    primary = IntentFragment(
        type=IntentClassification.follow_up,
        objective=(
            "Continue the prior Slack thread task before handling the memory "
            "preference."
        ),
        should_execute=True,
        likely_tools=["slack_channel_history", "describe_tools"],
        route="tool_worker",
        needs_channel_context=False,
        needs_thread_context=True,
        needs_file_context=request.has_files,
    )
    secondary = IntentFragment(
        type=IntentClassification.memory_candidate,
        objective=decision.reason,
        should_execute=True,
        likely_tools=decision.likely_tools or ["remember_fact"],
        route="memory_confirmation",
        needs_channel_context=False,
        needs_thread_context=False,
        needs_file_context=False,
    )
    return decision.model_copy(
        update={
            "classification": IntentClassification.follow_up,
            "should_create_task": True,
            "needs_thread_context": True,
            "likely_tools": list(primary.likely_tools),
            "model_tier": (
                ModelTier.standard
                if decision.model_tier is ModelTier.cheap
                else decision.model_tier
            ),
            "reason": (
                "Mixed follow-up plus memory instruction; execute the follow-up "
                "as primary and preserve the memory request as secondary."
            ),
            "primary_intent": primary,
            "secondary_intents": [secondary],
        }
    )


def _is_memory_forget_request(text: str) -> bool:
    normalized = f" {text.casefold()} "
    has_forget_action = any(
        phrase in normalized
        for phrase in (
            " forget ",
            " remove ",
            " delete ",
            " clear ",
        )
    )
    if not has_forget_action:
        return False
    return any(
        phrase in normalized
        for phrase in (
            " memory",
            " memories",
            " preference",
            " preferences",
            " fact",
            " facts",
            " rule",
            " rules",
            " remembered",
            " stored",
        )
    )


def _has_memory_instruction(text: str) -> bool:
    normalized = f" {text.casefold()} "
    return any(
        phrase in normalized
        for phrase in (
            " remember ",
            " keep in mind ",
            " going forward ",
            " from now on ",
            " in the future ",
            " preference ",
        )
    )


def _has_follow_up_instruction(request: IntentRequest) -> bool:
    normalized = f" {request.text.casefold()} "
    if request.is_thread_follow_up and any(
        phrase in normalized
        for phrase in (
            " do that",
            " lets do that",
            " let's do that",
            " go ahead",
            " continue",
            " yes",
            " yeah",
            " yup",
        )
    ):
        return True
    return any(
        phrase in normalized
        for phrase in (
            " do that",
            " lets do that",
            " let's do that",
            " continue with that",
        )
    )
