"""Recency-weighted relevance scoring for memory retrieval (HIG-225).

v1 ranking is deliberately simple per the ambient-intelligence research notes:
``ranked_score = similarity * recency_decay(last_seen)`` with an exponential
half-life decay. Importance weighting is intentionally not implemented yet.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

DEFAULT_RECENCY_HALF_LIFE_DAYS = 14.0
_SECONDS_PER_DAY = 86_400.0


def recency_decay(
    last_seen: datetime | None,
    *,
    now: datetime | None = None,
    half_life_days: float = DEFAULT_RECENCY_HALF_LIFE_DAYS,
) -> float:
    """Exponential decay in (0, 1]: 1.0 when fresh, 0.5 after one half-life.

    ``last_seen=None`` (or a non-positive half-life) disables decay and
    returns 1.0 so the score degrades to pure similarity.
    """

    if last_seen is None or half_life_days <= 0:
        return 1.0
    reference = now or datetime.now(UTC)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=UTC)
    if last_seen.tzinfo is None:
        last_seen = last_seen.replace(tzinfo=UTC)
    age_days = (reference - last_seen).total_seconds() / _SECONDS_PER_DAY
    if age_days <= 0:
        return 1.0
    return math.pow(0.5, age_days / half_life_days)


def ranked_score(
    similarity: float,
    last_seen: datetime | None,
    *,
    now: datetime | None = None,
    half_life_days: float = DEFAULT_RECENCY_HALF_LIFE_DAYS,
) -> float:
    """Combine embedding similarity with recency decay."""

    return similarity * recency_decay(
        last_seen,
        now=now,
        half_life_days=half_life_days,
    )
