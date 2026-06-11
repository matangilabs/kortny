"""Deterministic receptivity scoring for Witness delivery decisions (HIG-227).

No LLM. Features come from per-candidate ``feedback_json`` histories; the
score multiplies into ``effective_confidence`` to gate delivery. Shapes follow
the shipped art: per-category dismissal penalty with a linear 14-day recovery
(Duolingo recovering-bandit, simplified), a global dismissal cooldown, and an
acceptance boost capped at 1.0.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.orm import Session

from kortny.db.models import WitnessOpportunityCandidate

DISMISSAL_PENALTY_FACTOR = 0.6
DISMISSAL_WINDOW_DAYS = 30
DISMISSAL_RECOVERY_DAYS = 14
GLOBAL_COOLDOWN_DISMISSALS = 3
GLOBAL_COOLDOWN_WINDOW_DAYS = 7
GLOBAL_COOLDOWN_FACTOR = 0.5
ACCEPTANCE_BOOST_FACTOR = 1.15
ACCEPTANCE_WINDOW_DAYS = 30

CONFIDENCE_REINFORCEMENT_BASE = Decimal("0.6")
CONFIDENCE_REINFORCEMENT_STEP = Decimal("0.1")
CONFIDENCE_EVIDENCE_STEP = Decimal("0.05")


@dataclass(frozen=True, slots=True)
class UserFeedbackEvent:
    """One accept/dismiss data point from a user's Witness history."""

    action: str  # "accepted" | "dismissed"
    category: str
    at: datetime


def receptivity(
    user_history: Sequence[UserFeedbackEvent],
    category: str,
    now: datetime,
) -> float:
    """Score how receptive a user is to suggestions of ``category`` in [0, 1].

    - Each dismissal in ``category`` in the last 30 days multiplies by 0.6;
      the penalty decays linearly back to 1.0 over the 14 days since the most
      recent dismissal.
    - Three or more dismissals across categories in the last 7 days multiply
      the score by 0.5.
    - Each acceptance in ``category`` in the last 30 days multiplies by 1.15,
      capped at 1.0 total.
    """

    score = 1.0
    dismissal_cutoff = now - timedelta(days=DISMISSAL_WINDOW_DAYS)
    category_dismissals = [
        event.at
        for event in user_history
        if event.action == "dismissed"
        and event.category == category
        and dismissal_cutoff <= event.at <= now
    ]
    if category_dismissals:
        penalty = DISMISSAL_PENALTY_FACTOR ** len(category_dismissals)
        since_last = now - max(category_dismissals)
        recovery = min(
            1.0,
            max(
                0.0,
                since_last.total_seconds()
                / timedelta(days=DISMISSAL_RECOVERY_DAYS).total_seconds(),
            ),
        )
        score *= penalty + (1.0 - penalty) * recovery

    global_cutoff = now - timedelta(days=GLOBAL_COOLDOWN_WINDOW_DAYS)
    global_dismissals = sum(
        1
        for event in user_history
        if event.action == "dismissed" and global_cutoff <= event.at <= now
    )
    if global_dismissals >= GLOBAL_COOLDOWN_DISMISSALS:
        score *= GLOBAL_COOLDOWN_FACTOR

    acceptance_cutoff = now - timedelta(days=ACCEPTANCE_WINDOW_DAYS)
    acceptances = sum(
        1
        for event in user_history
        if event.action == "accepted"
        and event.category == category
        and acceptance_cutoff <= event.at <= now
    )
    if acceptances:
        score *= ACCEPTANCE_BOOST_FACTOR**acceptances

    return min(1.0, max(0.0, score))


def effective_confidence(
    llm_confidence: Decimal,
    *,
    reinforcement_count: int,
    evidence_count: int,
) -> Decimal:
    """Deterministic delivery-side confidence composition (HIG-197 Phase 1).

    ``llm_confidence * min(1.0, 0.6 + 0.1*reinforcement + 0.05*evidence)``,
    capped at 1.0. Replaces raw LLM self-report at delivery decisions only;
    extraction storage keeps the raw score.
    """

    multiplier = min(
        Decimal("1.0"),
        CONFIDENCE_REINFORCEMENT_BASE
        + CONFIDENCE_REINFORCEMENT_STEP * max(reinforcement_count, 0)
        + CONFIDENCE_EVIDENCE_STEP * max(evidence_count, 0),
    )
    composed = llm_confidence * multiplier
    if composed < 0:
        return Decimal("0.000")
    if composed > 1:
        return Decimal("1.000")
    return composed.quantize(Decimal("0.001"))


def collect_user_feedback_events(
    session: Session,
    *,
    installation_id: uuid.UUID,
    slack_user_id: str,
    now: datetime | None = None,
    window_days: int = DISMISSAL_WINDOW_DAYS,
) -> tuple[UserFeedbackEvent, ...]:
    """Read a user's accept/dismiss history out of candidate feedback_json."""

    observed_now = now or datetime.now(UTC)
    cutoff = observed_now - timedelta(days=window_days)
    candidates = session.scalars(
        select(WitnessOpportunityCandidate).where(
            WitnessOpportunityCandidate.installation_id == installation_id,
        )
    )
    events: list[UserFeedbackEvent] = []
    for candidate in candidates:
        feedback = candidate.feedback_json or {}
        history = feedback.get("history")
        if not isinstance(history, list):
            continue
        for entry in history:
            if not isinstance(entry, dict):
                continue
            action = entry.get("action")
            if action not in ("accepted", "dismissed"):
                continue
            if entry.get("by_user_id") != slack_user_id:
                continue
            at = _parse_datetime(entry.get("at"))
            if at is None or at < cutoff or at > observed_now:
                continue
            events.append(
                UserFeedbackEvent(
                    action=action,
                    category=candidate.candidate_type,
                    at=at,
                )
            )
    events.sort(key=lambda event: event.at)
    return tuple(events)


def _parse_datetime(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)
