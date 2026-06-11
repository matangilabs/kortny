"""HIG-225: pure trigger decision logic for the consolidator loop."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from kortny.consolidator import evaluate_trigger

NOW = datetime(2026, 6, 11, 3, 0, 0, tzinfo=UTC)


def test_first_run_always_triggers() -> None:
    decision = evaluate_trigger(
        now=NOW,
        last_success_at=None,
        new_item_count=0,
        last_activity_at=NOW,
    )
    assert decision.should_run is True
    assert decision.reason == "first_run"


def test_threshold_path_runs_when_items_cooldown_and_quiet_align() -> None:
    decision = evaluate_trigger(
        now=NOW,
        last_success_at=NOW - timedelta(hours=9),
        new_item_count=50,
        last_activity_at=NOW - timedelta(minutes=61),
    )
    assert decision.should_run is True
    assert decision.reason == "threshold"


def test_below_min_new_items_blocks() -> None:
    decision = evaluate_trigger(
        now=NOW,
        last_success_at=NOW - timedelta(hours=9),
        new_item_count=49,
        last_activity_at=NOW - timedelta(hours=2),
    )
    assert decision.should_run is False
    assert decision.reason == "below_min_new_items"


def test_cooldown_blocks_even_with_many_items() -> None:
    decision = evaluate_trigger(
        now=NOW,
        last_success_at=NOW - timedelta(hours=7),
        new_item_count=500,
        last_activity_at=NOW - timedelta(hours=2),
    )
    assert decision.should_run is False
    assert decision.reason == "cooldown_active"


def test_recent_activity_debounces() -> None:
    decision = evaluate_trigger(
        now=NOW,
        last_success_at=NOW - timedelta(hours=9),
        new_item_count=80,
        last_activity_at=NOW - timedelta(minutes=10),
    )
    assert decision.should_run is False
    assert decision.reason == "not_quiet"


def test_no_activity_counts_as_quiet() -> None:
    decision = evaluate_trigger(
        now=NOW,
        last_success_at=NOW - timedelta(hours=9),
        new_item_count=80,
        last_activity_at=None,
    )
    assert decision.should_run is True
    assert decision.reason == "threshold"


def test_nightly_floor_runs_even_when_idle() -> None:
    decision = evaluate_trigger(
        now=NOW,
        last_success_at=NOW - timedelta(hours=25),
        new_item_count=0,
        last_activity_at=NOW - timedelta(minutes=5),
    )
    assert decision.should_run is True
    assert decision.reason == "nightly_floor"


def test_custom_thresholds_apply() -> None:
    decision = evaluate_trigger(
        now=NOW,
        last_success_at=NOW - timedelta(hours=2),
        new_item_count=10,
        last_activity_at=NOW - timedelta(minutes=31),
        min_new_items=10,
        min_interval=timedelta(hours=1),
        quiet_window=timedelta(minutes=30),
        nightly_floor=timedelta(hours=48),
    )
    assert decision.should_run is True
    assert decision.reason == "threshold"
