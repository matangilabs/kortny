"""Tests for deterministic response-depth overrides (HIG-218).

These port the scenario coverage from the retired planned-workflow classifier
tests onto the unified router's depth-override module.
"""

from kortny.intent.depth_overrides import apply_depth_overrides, classify_depth_override
from kortny.intent.models import (
    IntentClassification,
    IntentDecision,
    IntentRequest,
    IntentSurface,
    ModelTier,
)


def test_availability_check_forces_quick_response() -> None:
    override = classify_depth_override(text="Are you up?")

    assert override is not None
    assert override.depth == "quick_response"
    assert override.reason_codes == ("quick_conversation",)


def test_schedule_state_question_forces_quick_response() -> None:
    override = classify_depth_override(
        text="Do I have an active stock market update scheduled?"
    )

    assert override is not None
    assert override.depth == "quick_response"
    assert override.reason_codes == ("schedule_state_query",)


def test_capability_lookup_with_tool_hints_forces_quick_response() -> None:
    override = classify_depth_override(
        text="What tools do you have access to?",
        likely_tools=["list_tools", "get_capabilities"],
    )

    assert override is not None
    assert override.depth == "quick_response"
    assert override.reason_codes == ("capability_lookup",)


def test_broad_research_with_synthesis_forces_deep_workflow() -> None:
    override = classify_depth_override(
        text="Research best AI agents for trading and compare the options."
    )

    assert override is not None
    assert override.depth == "deep_workflow"
    assert "research_synthesis_work" in override.reason_codes


def test_write_or_destructive_verb_forces_deep_workflow() -> None:
    override = classify_depth_override(
        text="Create a Linear issue for the onboarding bug and post it.",
    )

    assert override is not None
    assert override.depth == "deep_workflow"
    assert "write_or_destructive_intent" in override.reason_codes


def test_recurring_task_forces_deep_workflow() -> None:
    override = classify_depth_override(
        text="Every Monday summarize blockers from this channel."
    )

    assert override is not None
    assert override.depth == "deep_workflow"
    assert "scheduled_or_recurring" in override.reason_codes


def test_long_running_monitoring_forces_deep_workflow() -> None:
    override = classify_depth_override(
        text="Monitor the deploy channel and keep checking for failures.",
    )

    assert override is not None
    assert override.depth == "deep_workflow"
    assert "long_running_or_monitoring" in override.reason_codes


def test_multi_integration_scope_forces_deep_workflow() -> None:
    override = classify_depth_override(
        text="Pull my open Linear work and cross-check it against GitHub PRs.",
        likely_tools=["composio_linear_execute", "composio_github_execute"],
    )

    assert override is not None
    assert override.depth == "deep_workflow"
    assert (
        "multi_integration_scope" in override.reason_codes
        or "multi_tool_likely" in override.reason_codes
    )


def test_simple_bounded_request_has_no_override() -> None:
    override = classify_depth_override(
        text="What is the weather in Tokyo right now?",
    )

    assert override is None


def test_apply_override_sets_deterministic_source() -> None:
    decision = _decision(response_depth="standard_tool_task")
    request = _request("Every Monday summarize blockers from this channel.")

    result = apply_depth_overrides(request, decision)

    assert result.response_depth == "deep_workflow"
    assert result.depth_source == "deterministic_override"


def test_apply_no_override_preserves_llm_depth_and_marks_source() -> None:
    decision = _decision(response_depth="standard_tool_task")
    request = _request("What is the weather in Tokyo right now?")

    result = apply_depth_overrides(request, decision)

    assert result.response_depth == "standard_tool_task"
    assert result.depth_source == "llm"


def _decision(*, response_depth: str) -> IntentDecision:
    return IntentDecision(
        addressed_to_kortny=True,
        classification=IntentClassification.task_request,
        confidence=0.9,
        should_create_task=True,
        should_ack_with_reaction=False,
        suggested_reaction=None,
        needs_channel_context=False,
        needs_thread_context=False,
        needs_file_context=False,
        likely_tools=[],
        model_tier=ModelTier.standard,
        reason="test",
        response_depth=response_depth,  # type: ignore[arg-type]
    )


def _request(text: str) -> IntentRequest:
    return IntentRequest(text=text, surface=IntentSurface.app_mention)
