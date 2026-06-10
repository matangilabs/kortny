from __future__ import annotations

from dataclasses import dataclass

from kortny.routing.tool_scope import (
    NATIVE_TOOL_SCOPE_APPLIED_MESSAGE,
    NativeToolScopePolicy,
)


@dataclass(slots=True)
class FakeTool:
    name: str


def test_native_tool_scope_hides_schedule_mutations_for_unrelated_turn() -> None:
    decision = NativeToolScopePolicy().apply(
        tools=[
            FakeTool("web_search"),
            FakeTool("list_schedules"),
            FakeTool("create_schedule"),
            FakeTool("cancel_schedule"),
        ],
        task_input="Are you up?",
        intent_decision={
            "classification": "task_request",
            "likely_tools": [],
        },
    )

    assert decision.selected_tool_names == ("web_search", "list_schedules")
    assert decision.suppressed_tool_names == (
        "create_schedule",
        "cancel_schedule",
    )
    assert decision.schedule_mutation_allowed is False
    assert decision.reason_codes == ("schedule_mutation_tools_hidden",)
    assert decision.to_payload()["message"] == NATIVE_TOOL_SCOPE_APPLIED_MESSAGE


def test_native_tool_scope_does_not_treat_market_update_as_schedule_update() -> None:
    decision = NativeToolScopePolicy().apply(
        tools=[
            FakeTool("web_search"),
            FakeTool("list_schedules"),
            FakeTool("create_schedule"),
            FakeTool("update_schedule"),
        ],
        task_input="Give me a stock market update.",
        intent_decision={
            "classification": "task_request",
            "likely_tools": ["web_search"],
        },
    )

    assert decision.selected_tool_names == ("web_search", "list_schedules")
    assert decision.suppressed_tool_names == (
        "create_schedule",
        "update_schedule",
    )
    assert decision.schedule_mutation_allowed is False


def test_native_tool_scope_allows_schedule_mutations_for_schedule_request() -> None:
    decision = NativeToolScopePolicy().apply(
        tools=[
            FakeTool("list_schedules"),
            FakeTool("create_schedule"),
            FakeTool("update_schedule"),
        ],
        task_input="Every weekday at 8 AM send me a market update.",
        intent_decision={
            "classification": "task_request",
            "likely_tools": [],
        },
    )

    assert decision.selected_tool_names == (
        "list_schedules",
        "create_schedule",
        "update_schedule",
    )
    assert decision.suppressed_tool_names == ()
    assert decision.schedule_mutation_allowed is True
    assert decision.reason_codes == ("schedule_mutation_tools_allowed",)


def test_native_tool_scope_allows_schedule_mutations_from_intent_hint() -> None:
    decision = NativeToolScopePolicy().apply(
        tools=[
            FakeTool("list_schedules"),
            FakeTool("create_schedule"),
            FakeTool("pause_schedule"),
        ],
        task_input="Please do that.",
        intent_decision={
            "classification": "follow_up",
            "likely_tools": ["create_schedule"],
        },
    )

    assert decision.selected_tool_names == (
        "list_schedules",
        "create_schedule",
        "pause_schedule",
    )
    assert decision.suppressed_tool_names == ()
    assert decision.schedule_mutation_allowed is True
    assert decision.likely_tools == ("create_schedule",)
