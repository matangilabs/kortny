"""Pure trigger decision for the consolidator loop (HIG-225).

Shipped formula (Honcho/Plastic Labs pattern): run when enough new items have
accumulated AND a cooldown has elapsed AND the workspace has been quiet for a
debounce window — with a nightly floor so consolidation always happens at
least once a day.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

DEFAULT_MIN_NEW_ITEMS = 50
DEFAULT_MIN_INTERVAL = timedelta(hours=8)
DEFAULT_QUIET_WINDOW = timedelta(minutes=60)
DEFAULT_NIGHTLY_FLOOR = timedelta(hours=24)


@dataclass(frozen=True, slots=True)
class TriggerDecision:
    """Whether to run a consolidation pass, and why (not)."""

    should_run: bool
    reason: str


def evaluate_trigger(
    *,
    now: datetime,
    last_success_at: datetime | None,
    new_item_count: int,
    last_activity_at: datetime | None,
    min_new_items: int = DEFAULT_MIN_NEW_ITEMS,
    min_interval: timedelta = DEFAULT_MIN_INTERVAL,
    quiet_window: timedelta = DEFAULT_QUIET_WINDOW,
    nightly_floor: timedelta = DEFAULT_NIGHTLY_FLOOR,
) -> TriggerDecision:
    """Decide whether the consolidator should run for one installation."""

    if last_success_at is None:
        return TriggerDecision(True, "first_run")

    if now - last_success_at >= nightly_floor:
        return TriggerDecision(True, "nightly_floor")

    if new_item_count < min_new_items:
        return TriggerDecision(False, "below_min_new_items")
    if now - last_success_at < min_interval:
        return TriggerDecision(False, "cooldown_active")
    if last_activity_at is not None and now - last_activity_at < quiet_window:
        return TriggerDecision(False, "not_quiet")
    return TriggerDecision(True, "threshold")
