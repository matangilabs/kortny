"""Pure tests for the routing-quality signal (HIG-221)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from kortny.routing.quality import (
    RoutingQuality,
    RoutingQualityResult,
    compute_routing_quality,
)


def _q(
    status: str,
    payloads: Sequence[Mapping[str, object]] = (),
    attempts: int = 1,
) -> RoutingQualityResult:
    return compute_routing_quality(
        status=status, event_payloads=list(payloads), attempts=attempts
    )


def test_clean_success_scores_one() -> None:
    result = _q("succeeded", [{"message": "agent_completed", "partial": False}])
    assert result.quality is RoutingQuality.clean
    assert result.score == 1.0
    assert result.reason_codes == ()


def test_partial_completion_is_partial() -> None:
    result = _q("succeeded", [{"message": "agent_completed", "partial": True}])
    assert result.quality is RoutingQuality.partial
    assert result.score == 0.5
    assert "budget_exhausted" in result.reason_codes


def test_budget_exceeded_event_is_partial() -> None:
    result = _q(
        "succeeded",
        [
            {"message": "execution_budget_exceeded", "reason": "max_tool_calls"},
            {"message": "agent_completed", "partial": True},
        ],
    )
    assert result.quality is RoutingQuality.partial


def test_recoverable_failure_is_recovered() -> None:
    result = _q(
        "succeeded",
        [
            {"message": "tool_call_recoverable_failed", "tool": "x"},
            {"message": "agent_completed", "partial": False},
        ],
    )
    assert result.quality is RoutingQuality.recovered
    assert result.score == 0.75
    assert "tool_call_recoverable_failed" in result.reason_codes


def test_retry_attempts_mark_recovered() -> None:
    result = _q("succeeded", [{"message": "agent_completed"}], attempts=2)
    assert result.quality is RoutingQuality.recovered
    assert "retried" in result.reason_codes


def test_partial_beats_recovered_when_both_present() -> None:
    # A budget-exhausted run that also retried is reported as partial (the more
    # informative, lower outcome).
    result = _q(
        "succeeded",
        [
            {"message": "tool_call_recoverable_failed"},
            {"message": "agent_completed", "partial": True},
        ],
        attempts=2,
    )
    assert result.quality is RoutingQuality.partial


def test_failed_status_scores_zero() -> None:
    for status in ("failed", "crashed"):
        result = _q(status, [{"message": "task_executor_failed"}])
        assert result.quality is RoutingQuality.failed
        assert result.score == 0.0


def test_cancelled_has_no_score() -> None:
    result = _q("cancelled", [])
    assert result.quality is RoutingQuality.cancelled
    assert result.score is None


def test_non_terminal_is_not_scored() -> None:
    result = _q("running", [])
    assert result.score is None
    assert "not_terminal" in result.reason_codes


def test_every_terminal_status_yields_a_quality() -> None:
    # Acceptance: every completed task gets a routing_quality value.
    for status in ("succeeded", "failed", "crashed", "cancelled"):
        assert _q(status, []).quality in set(RoutingQuality)
