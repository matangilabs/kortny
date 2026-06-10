from types import SimpleNamespace
from typing import Any, cast

from kortny.db.models import Task
from kortny.routing import Tier0RouteKind, Tier0Router


def test_tier0_router_routes_schedule_state_questions() -> None:
    decision = Tier0Router().route(
        _task("Do I have an active stock market update scheduled?")
    )

    assert decision is not None
    assert decision.kind is Tier0RouteKind.schedule_state_query
    assert decision.runtime_class == "inline_tool_task"
    assert decision.intent == "scheduler.query"
    assert decision.selected_runtime == "schedule_state_fast_path"
    assert decision.actual_path == "schedule_state_fast_path"
    assert decision.reason_codes == ("schedule_state_query",)
    assert decision.metadata["query"] == "stock market update"
    assert decision.metadata["status"] == "active"
    payload = decision.to_trace().to_payload()
    assert payload["message"] == "routing_decision_recorded"
    assert payload["route_tier_resolved"] == "tier0"
    assert payload["intent"] == "scheduler.query"


def test_tier0_router_ignores_normal_task_requests() -> None:
    decision = Tier0Router().route(
        _task("Research AI observability tools and compare options.")
    )

    assert decision is None


def test_tier0_router_does_not_intercept_scheduled_task_runs() -> None:
    decision = Tier0Router().route(
        _task(
            "Do I have an active stock market update scheduled?",
            identity_kind="scheduled",
        )
    )

    assert decision is None


def _task(input_text: str, *, identity_kind: str | None = None) -> Task:
    result: Any = SimpleNamespace(
        input=input_text,
        identity_kind=identity_kind,
    )
    return cast(Task, result)
