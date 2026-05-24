from kortny.intent import IntentClassification, IntentDecision, ModelTier
from kortny.slack.reactions import LibraryReactionProvider


def test_library_reaction_provider_selects_relevant_ack_bucket() -> None:
    provider = LibraryReactionProvider()

    creation = provider.acknowledgement_reaction(
        input_text="turn this meeting transcript into a client-ready summary document",
        source="app_mention",
    )
    discovery = provider.acknowledgement_reaction(
        input_text="look up recent guidance and pull together the important sources",
        source="dm",
    )
    memory = provider.acknowledgement_reaction(
        input_text="remember that weekly recaps go on Fridays",
        source="app_mention",
    )
    review = provider.acknowledgement_reaction(
        input_text="review this thread and summarize the open questions",
        source="app_mention",
    )

    assert creation.intent == "creation"
    assert creation.name in {"page_facing_up", "paperclip", "open_file_folder"}
    assert discovery.intent == "discovery"
    assert discovery.name in {"mag", "newspaper", "compass"}
    assert memory.intent == "memory"
    assert memory.name in {"memo", "bookmark", "pushpin"}
    assert review.intent == "review"
    assert review.name in {"thinking_face", "bar_chart", "clipboard"}


def test_library_reaction_provider_is_stable_and_not_single_reaction() -> None:
    provider = LibraryReactionProvider()
    first = provider.acknowledgement_reaction(
        input_text="summarize this channel for me",
        source="app_mention",
    )
    second = provider.acknowledgement_reaction(
        input_text="summarize this channel for me",
        source="app_mention",
    )

    assert first == second
    assert all(len(bucket.names) > 1 for bucket in provider.buckets)


def test_library_reaction_provider_marks_completion_and_failure() -> None:
    provider = LibraryReactionProvider()

    completed = provider.completion_reaction(
        input_text="summarize this", source="worker", succeeded=True
    )
    failed = provider.completion_reaction(
        input_text="summarize this", source="worker", succeeded=False
    )

    assert completed.name == "heavy_check_mark"
    assert completed.intent == "completed"
    assert failed.name == "warning"
    assert failed.intent == "failed"


def test_acknowledgement_reaction_uses_intent_decision_when_available() -> None:
    provider = LibraryReactionProvider()
    decision = IntentDecision(
        addressed_to_kortny=True,
        classification=IntentClassification.memory_candidate,
        confidence=0.9,
        should_create_task=True,
        should_ack_with_reaction=True,
        suggested_reaction="memo",
        needs_channel_context=False,
        needs_thread_context=False,
        needs_file_context=False,
        likely_tools=[],
        model_tier=ModelTier.cheap,
        reason="User stated a durable preference.",
    )

    choice = provider.acknowledgement_reaction(
        input_text="please remember this",
        source="app_mention",
        intent_decision=decision,
    )

    assert choice.name == "memo"
    assert choice.intent == "memory_candidate"
