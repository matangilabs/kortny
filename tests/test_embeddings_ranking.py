"""HIG-225: recency-decay ranking helper math."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from kortny.embeddings import ranked_score, recency_decay

NOW = datetime(2026, 6, 11, 12, 0, 0, tzinfo=UTC)


def test_recency_decay_is_one_when_fresh() -> None:
    assert recency_decay(NOW, now=NOW, half_life_days=14) == 1.0


def test_recency_decay_halves_after_one_half_life() -> None:
    last_seen = NOW - timedelta(days=14)
    assert recency_decay(last_seen, now=NOW, half_life_days=14) == pytest.approx(0.5)


def test_recency_decay_quarters_after_two_half_lives() -> None:
    last_seen = NOW - timedelta(days=28)
    assert recency_decay(last_seen, now=NOW, half_life_days=14) == pytest.approx(0.25)


def test_recency_decay_respects_custom_half_life() -> None:
    last_seen = NOW - timedelta(days=7)
    assert recency_decay(last_seen, now=NOW, half_life_days=7) == pytest.approx(0.5)


def test_recency_decay_none_last_seen_disables_decay() -> None:
    assert recency_decay(None, now=NOW, half_life_days=14) == 1.0


def test_recency_decay_non_positive_half_life_disables_decay() -> None:
    last_seen = NOW - timedelta(days=100)
    assert recency_decay(last_seen, now=NOW, half_life_days=0) == 1.0
    assert recency_decay(last_seen, now=NOW, half_life_days=-3) == 1.0


def test_recency_decay_future_last_seen_clamps_to_one() -> None:
    last_seen = NOW + timedelta(days=2)
    assert recency_decay(last_seen, now=NOW, half_life_days=14) == 1.0


def test_recency_decay_handles_naive_datetimes() -> None:
    last_seen = (NOW - timedelta(days=14)).replace(tzinfo=None)
    assert recency_decay(last_seen, now=NOW, half_life_days=14) == pytest.approx(0.5)


def test_ranked_score_multiplies_similarity_by_decay() -> None:
    last_seen = NOW - timedelta(days=14)
    assert ranked_score(0.8, last_seen, now=NOW, half_life_days=14) == pytest.approx(
        0.4
    )


def test_ranked_score_pure_similarity_when_no_last_seen() -> None:
    assert ranked_score(0.8, None, now=NOW, half_life_days=14) == pytest.approx(0.8)


def test_ranked_score_orders_fresh_over_stale_at_equal_similarity() -> None:
    fresh = ranked_score(0.7, NOW - timedelta(days=1), now=NOW, half_life_days=14)
    stale = ranked_score(0.7, NOW - timedelta(days=60), now=NOW, half_life_days=14)
    assert fresh > stale
