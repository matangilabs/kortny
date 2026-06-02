import uuid
from collections.abc import Sequence

import pytest

from kortny.intent import (
    IntentClassification,
    IntentClassificationError,
    IntentRequest,
    IntentSurface,
    LLMIntentClassifier,
    ModelTier,
    contains_app_name,
    parse_intent_decision,
    should_classify_channel_message,
    should_create_task_from_soft_mention,
    should_react_to_rejected_soft_mention,
)
from kortny.llm import ChatMessage, Completion, TokenUsage
from kortny.tools.types import JsonObject, JsonSchema


class FakeIntentLLM:
    def __init__(self, content: str | None) -> None:
        self.content = content
        self.calls: list[
            tuple[
                uuid.UUID,
                tuple[ChatMessage, ...],
                tuple[JsonSchema, ...],
                JsonObject | None,
            ]
        ] = []

    def complete(
        self,
        *,
        task_id: uuid.UUID,
        messages: Sequence[ChatMessage],
        tools: Sequence[JsonSchema] = (),
        response_format: JsonObject | None = None,
    ) -> Completion:
        self.calls.append((task_id, tuple(messages), tuple(tools), response_format))
        return Completion(
            content=self.content,
            tool_calls=(),
            usage=TokenUsage(input_tokens=10, output_tokens=5),
            model="openai/gpt-4o-mini",
        )


class FakeIntentProvider:
    model = "openai/gpt-4o-mini"

    def __init__(self, content: str | None) -> None:
        self.content = content
        self.calls: list[
            tuple[tuple[ChatMessage, ...], tuple[JsonSchema, ...], JsonObject | None]
        ] = []

    def complete(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[JsonSchema] = (),
        *,
        response_format: JsonObject | None = None,
    ) -> Completion:
        self.calls.append((tuple(messages), tuple(tools), response_format))
        return Completion(
            content=self.content,
            tool_calls=(),
            usage=TokenUsage(input_tokens=10, output_tokens=5),
            model=self.model,
        )


def test_llm_intent_classifier_returns_typed_decision() -> None:
    task_id = uuid.uuid4()
    llm = FakeIntentLLM(
        """
        {
          "addressed_to_kortny": true,
          "classification": "task_request",
          "confidence": 0.92,
          "should_create_task": true,
          "should_ack_with_reaction": true,
          "suggested_reaction": "eyes",
          "needs_channel_context": true,
          "needs_thread_context": true,
          "needs_file_context": false,
          "likely_tools": ["slack_channel_history"],
          "model_tier": "standard",
          "reason": "User directly asks Kortny to summarize the thread."
        }
        """
    )

    decision = LLMIntentClassifier(llm=llm).classify(
        task_id=task_id,
        request=IntentRequest(
            text="Kortny can you summarize this thread?",
            surface=IntentSurface.channel_message,
            is_thread_follow_up=True,
        ),
    )

    assert decision.classification is IntentClassification.task_request
    assert decision.confidence == 0.92
    assert decision.model_tier is ModelTier.standard
    assert decision.likely_tools == ["slack_channel_history"]
    assert llm.calls[0][0] == task_id
    assert llm.calls[0][3] == {"type": "json_object"}


def test_llm_intent_classifier_can_run_before_task_creation() -> None:
    provider = FakeIntentProvider(valid_decision_json())

    decision = LLMIntentClassifier(provider=provider).classify(
        request=IntentRequest(
            text="Kortny can you summarize this channel?",
            surface=IntentSurface.channel_message,
        )
    )

    assert decision.classification is IntentClassification.task_request
    assert provider.calls[0][2] == {"type": "json_object"}


def test_intent_classifier_overrides_memory_forget_as_task_request() -> None:
    provider = FakeIntentProvider(
        """
        {
          "addressed_to_kortny": true,
          "classification": "cancel_or_retry",
          "confidence": 0.95,
          "should_create_task": false,
          "should_ack_with_reaction": true,
          "suggested_reaction": "arrows_counterclockwise",
          "needs_channel_context": false,
          "needs_thread_context": false,
          "needs_file_context": false,
          "likely_tools": ["memory_management"],
          "model_tier": "cheap",
          "reason": "User asks to forget something."
        }
        """
    )

    decision = LLMIntentClassifier(provider=provider).classify(
        request=IntentRequest(
            text="forget my PDF branding preference",
            surface=IntentSurface.app_mention,
        )
    )

    assert decision.classification is IntentClassification.task_request
    assert decision.should_create_task is True
    assert decision.suggested_reaction == "memo"
    assert decision.likely_tools == ["inspect_memory", "forget_fact"]


def test_intent_classifier_preserves_mixed_follow_up_and_memory() -> None:
    provider = FakeIntentProvider(
        """
        {
          "addressed_to_kortny": true,
          "classification": "memory_candidate",
          "confidence": 0.9,
          "should_create_task": false,
          "should_ack_with_reaction": true,
          "suggested_reaction": "brain",
          "needs_channel_context": false,
          "needs_thread_context": true,
          "needs_file_context": false,
          "likely_tools": ["set_memory", "manage_memory"],
          "model_tier": "cheap",
          "reason": "User wants Kortny to remember a stable preference."
        }
        """
    )

    decision = LLMIntentClassifier(provider=provider).classify(
        request=IntentRequest(
            text=(
                "Yeah lets do that. In the future remember to use the tools "
                "necessary and don't wait to be told."
            ),
            surface=IntentSurface.app_mention,
            is_thread_follow_up=True,
        )
    )

    assert decision.classification is IntentClassification.follow_up
    assert decision.should_create_task is True
    assert decision.routing_classification() is IntentClassification.follow_up
    assert decision.primary_intent is not None
    assert decision.primary_intent.type is IntentClassification.follow_up
    assert decision.primary_intent.needs_thread_context is True
    assert decision.secondary_intents[0].type is IntentClassification.memory_candidate
    assert decision.secondary_intents[0].route == "memory_confirmation"


def test_parse_intent_decision_accepts_explicit_decomposition() -> None:
    decision = parse_intent_decision(
        """
        {
          "addressed_to_kortny": true,
          "classification": "follow_up",
          "confidence": 0.91,
          "should_create_task": true,
          "should_ack_with_reaction": true,
          "suggested_reaction": "eyes",
          "needs_channel_context": false,
          "needs_thread_context": true,
          "needs_file_context": false,
          "likely_tools": ["slack_channel_history"],
          "model_tier": "standard",
          "reason": "User has a follow-up plus memory instruction.",
          "primary_intent": {
            "type": "follow_up",
            "objective": "Continue the prior request.",
            "should_execute": true,
            "likely_tools": ["alpha_vantage"],
            "route": "tool_worker",
            "needs_channel_context": false,
            "needs_thread_context": true,
            "needs_file_context": false
          },
          "secondary_intents": [
            {
              "type": "memory_candidate",
              "objective": "Remember the user's tool-use preference.",
              "should_execute": true,
              "likely_tools": ["remember_fact"],
              "route": "memory_confirmation",
              "needs_channel_context": false,
              "needs_thread_context": false,
              "needs_file_context": false
            }
          ]
        }
        """
    )

    assert decision.routing_likely_tools() == ["alpha_vantage"]
    assert decision.secondary_intents[0].likely_tools == ["remember_fact"]


def test_parse_intent_decision_rejects_invalid_content() -> None:
    with pytest.raises(IntentClassificationError):
        parse_intent_decision("not json")

    with pytest.raises(IntentClassificationError):
        parse_intent_decision(
            """
            {
              "addressed_to_kortny": true,
              "classification": "task_request",
              "confidence": 3,
              "should_create_task": true
            }
            """
        )


def test_soft_mention_policy_fails_closed() -> None:
    high_confidence = parse_intent_decision(
        """
        {
          "addressed_to_kortny": true,
          "classification": "task_request",
          "confidence": 0.86,
          "should_create_task": true,
          "should_ack_with_reaction": true,
          "suggested_reaction": "eyes",
          "needs_channel_context": true,
          "needs_thread_context": false,
          "needs_file_context": false,
          "likely_tools": [],
          "model_tier": "cheap",
          "reason": "Direct ask."
        }
        """
    )
    third_person = high_confidence.model_copy(
        update={
            "addressed_to_kortny": False,
            "classification": IntentClassification.third_person_reference,
            "confidence": 0.99,
        }
    )
    low_confidence = high_confidence.model_copy(update={"confidence": 0.6})

    assert should_create_task_from_soft_mention(high_confidence) is True
    assert should_create_task_from_soft_mention(third_person) is False
    assert should_create_task_from_soft_mention(low_confidence) is False


def test_rejected_soft_mention_reaction_policy_is_social_only() -> None:
    social_reference = parse_intent_decision(
        """
        {
          "addressed_to_kortny": false,
          "classification": "third_person_reference",
          "confidence": 0.9,
          "should_create_task": false,
          "should_ack_with_reaction": true,
          "suggested_reaction": "wave",
          "needs_channel_context": false,
          "needs_thread_context": false,
          "needs_file_context": false,
          "likely_tools": [],
          "model_tier": "cheap",
          "reason": "User introduces Kortny to other people."
        }
        """
    )
    silent_reference = social_reference.model_copy(
        update={"should_ack_with_reaction": False}
    )
    low_confidence = social_reference.model_copy(update={"confidence": 0.6})

    assert should_react_to_rejected_soft_mention(social_reference) is True
    assert should_react_to_rejected_soft_mention(silent_reference) is False
    assert should_react_to_rejected_soft_mention(low_confidence) is False


def test_channel_message_prefilter_only_selects_soft_name_candidates() -> None:
    assert contains_app_name("Can Kortny review this?", app_name="kortny") is True
    assert contains_app_name("kortnybot should not match", app_name="kortny") is False
    assert (
        should_classify_channel_message(
            {
                "type": "message",
                "channel_type": "channel",
                "user": "U123",
                "text": "Kortny can you compare these options?",
            },
            app_name="kortny",
        )
        is True
    )
    assert (
        should_classify_channel_message(
            {
                "type": "message",
                "channel_type": "channel",
                "user": "U123",
                "text": "This does not mention the app.",
            },
            app_name="kortny",
        )
        is False
    )
    assert (
        should_classify_channel_message(
            {
                "type": "message",
                "channel_type": "channel",
                "user": "U123",
                "text": "<@UBOT> Compare observability tools for Kortny.",
            },
            app_name="kortny",
        )
        is False
    )
    assert (
        should_classify_channel_message(
            {
                "type": "message",
                "channel_type": "channel",
                "bot_id": "B123",
                "text": "Kortny can you compare these options?",
            },
            app_name="kortny",
        )
        is False
    )


def valid_decision_json() -> str:
    return """
    {
      "addressed_to_kortny": true,
      "classification": "task_request",
      "confidence": 0.9,
      "should_create_task": true,
      "should_ack_with_reaction": true,
      "suggested_reaction": "eyes",
      "needs_channel_context": true,
      "needs_thread_context": false,
      "needs_file_context": false,
      "likely_tools": [],
      "model_tier": "cheap",
      "reason": "Direct request."
    }
    """
