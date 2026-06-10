from types import SimpleNamespace
from typing import Any, cast

from kortny.db.models import Task, TaskEvent
from kortny.llm.routing import INTENT_CLASSIFIED_MESSAGE
from kortny.workflow.planning_classifier import (
    PlannedWorkflowRoute,
    classify_planned_workflow,
)


def test_classifier_keeps_availability_check_inline() -> None:
    decision = classify_planned_workflow(task=_task("Are you up?"))

    assert decision.route is PlannedWorkflowRoute.inline
    assert decision.planned_candidate is False
    assert decision.estimated_subtask_count == 1
    assert decision.reason_codes == ("quick_conversation",)
    assert decision.to_payload()["behavior"] == "observe_only"


def test_classifier_marks_broad_research_as_planned_candidate() -> None:
    decision = classify_planned_workflow(
        task=_task("Research best AI agents for trading and summarize the options.")
    )

    assert decision.route is PlannedWorkflowRoute.planned_candidate
    assert decision.planned_candidate is True
    assert decision.estimated_subtask_count >= 3
    assert "estimated_three_or_more_subtasks" in decision.reason_codes
    assert "broad_research" in decision.to_payload()["reason_codes"]


def test_classifier_uses_intent_metadata_for_multi_tool_candidate() -> None:
    decision = classify_planned_workflow(
        task=_task(
            "Compare my open Linear work with recent Kortny docs and recommend next actions."
        ),
        events=(
            _intent_event(
                {
                    "likely_tools": [
                        "slack_channel_history",
                        "composio_linear_execute",
                    ],
                    "needs_channel_context": True,
                }
            ),
        ),
    )

    assert decision.route is PlannedWorkflowRoute.planned_candidate
    assert "multi_tool_likely" in decision.reason_codes
    assert "linear" in decision.detected_integrations
    assert "slack" in decision.detected_integrations
    assert "channel_context" in decision.needs_context


def test_classifier_keeps_capability_inventory_inline_with_tool_hints() -> None:
    decision = classify_planned_workflow(
        task=_task("What tools do you have access to?"),
        events=(
            _intent_event(
                {
                    "likely_tools": ["list_tools", "get_capabilities"],
                }
            ),
        ),
    )

    assert decision.route is PlannedWorkflowRoute.inline
    assert decision.planned_candidate is False
    assert decision.estimated_subtask_count == 1
    assert decision.reason_codes == ("capability_lookup",)


def test_classifier_keeps_simple_channel_context_request_inline() -> None:
    decision = classify_planned_workflow(
        task=_task("Summarize the last few decisions in this channel.")
    )

    assert decision.route is PlannedWorkflowRoute.inline
    assert decision.planned_candidate is False
    assert "external_context_needed" in decision.reason_codes


def test_classifier_marks_recurring_task_as_planned_candidate() -> None:
    decision = classify_planned_workflow(
        task=_task("Every Monday summarize blockers from this channel.")
    )

    assert decision.route is PlannedWorkflowRoute.planned_candidate
    assert "scheduled_or_recurring" in decision.reason_codes
    assert decision.confidence >= 0.9


def test_classifier_keeps_schedule_state_questions_inline() -> None:
    decision = classify_planned_workflow(
        task=_task("Do I have an active stock market update scheduled?")
    )

    assert decision.route is PlannedWorkflowRoute.inline
    assert decision.planned_candidate is False
    assert decision.estimated_subtask_count == 1
    assert decision.reason_codes == ("schedule_state_query",)


def _task(input_text: str) -> Task:
    result: Any = SimpleNamespace(
        input=input_text,
        identity_kind=None,
    )
    return cast(Task, result)


def _intent_event(decision: dict[str, Any]) -> TaskEvent:
    result: Any = SimpleNamespace(
        seq=1,
        payload={
            "message": INTENT_CLASSIFIED_MESSAGE,
            "decision": decision,
        },
    )
    return cast(TaskEvent, result)
