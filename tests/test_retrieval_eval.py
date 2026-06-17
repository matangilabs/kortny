"""Tests for the tool-retrieval eval instrument (Linear HIG-271).

Pure — no DB, embeddings, or network. Validates the scoring math, the aggregate
runner, and the seed dataset's integrity.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import pytest

from kortny.evals.retrieval import (
    SEED_RETRIEVAL_CASES,
    RetrievalCase,
    ndcg_at_k,
    recall_at_k,
    score_retrieval,
)


def test_recall_at_k_perfect_partial_and_miss() -> None:
    assert recall_at_k(["A", "B", "C"], ["A", "C"], 3) == 1.0
    assert recall_at_k(["A", "B", "C"], ["A", "C"], 2) == 0.5  # only A in top-2
    assert recall_at_k(["X", "Y"], ["A"], 5) == 0.0
    assert recall_at_k(["A"], [], 5) == 0.0  # no expected -> 0, not div-by-zero


def test_ndcg_rewards_higher_rank() -> None:
    assert ndcg_at_k(["A", "B"], ["A"], 5) == pytest.approx(1.0)
    # A at position 2 (index 1): DCG = 1/log2(3), IDCG = 1.
    assert ndcg_at_k(["B", "A"], ["A"], 5) == pytest.approx(1.0 / math.log2(3))
    assert ndcg_at_k(["B", "C"], ["A"], 5) == 0.0


def test_ndcg_two_relevant_ideal_is_one() -> None:
    # Both expected at the top in order -> perfect nDCG.
    assert ndcg_at_k(["A", "B", "C"], ["A", "B"], 3) == pytest.approx(1.0)


def test_k_must_be_positive() -> None:
    with pytest.raises(ValueError):
        recall_at_k(["A"], ["A"], 0)
    with pytest.raises(ValueError):
        ndcg_at_k(["A"], ["A"], 0)


def test_score_retrieval_aggregates() -> None:
    cases = (
        RetrievalCase(query="find A", expected_tool_slugs=("A",)),
        RetrievalCase(query="find B", expected_tool_slugs=("B",)),
    )
    # Perfect retriever for A, miss for B.
    ranked: dict[str, Sequence[str]] = {"find A": ["A", "Z"], "find B": ["Y", "Z"]}

    report = score_retrieval(cases, lambda q: ranked[q], ks=(1, 3))

    assert report.case_count == 2
    assert report.ks == (1, 3)
    assert report.mean_recall_at_k[1] == 0.5  # A hit @1, B missed
    assert report.hit_rate == 0.5
    assert "n=2" in report.summary_line()
    assert "R@1=0.500" in report.summary_line()


def test_score_retrieval_rejects_empty_inputs() -> None:
    with pytest.raises(ValueError):
        score_retrieval((), lambda q: [], ks=(1,))
    with pytest.raises(ValueError):
        score_retrieval(
            (RetrievalCase(query="x", expected_tool_slugs=("A",)),),
            lambda q: [],
            ks=(),
        )


def test_retrieval_score_hit_detects_any_match() -> None:
    cases = (RetrievalCase(query="q", expected_tool_slugs=("A", "B")),)
    # B appears beyond k=1 but still counts as a hit.
    report = score_retrieval(cases, lambda q: ["Z", "B"], ks=(1,))
    assert report.scores[0].hit is True
    assert report.mean_recall_at_k[1] == 0.0  # nothing relevant in top-1


def test_seed_dataset_is_well_formed() -> None:
    assert len(SEED_RETRIEVAL_CASES) >= 8
    queries = [case.query for case in SEED_RETRIEVAL_CASES]
    assert len(queries) == len(set(queries)), "duplicate query in seed dataset"
    for case in SEED_RETRIEVAL_CASES:
        assert case.query.strip(), "blank query"
        assert case.expected_tool_slugs, f"no ground truth for {case.query!r}"
        for slug in case.expected_tool_slugs:
            # Composio tool slugs are non-empty uppercase identifiers.
            assert slug == slug.upper() and slug.strip(), f"bad slug {slug!r}"


def test_seed_dataset_covers_the_live_regressions() -> None:
    by_query = {case.query: case for case in SEED_RETRIEVAL_CASES}
    plate = by_query["what's on my plate today?"]
    assert plate.expected_tool_slugs == ("LINEAR_LIST_ISSUES",)
    notion = by_query["what notes can you see on Notion?"]
    assert "NOTION_SEARCH_NOTION_PAGE" in notion.expected_tool_slugs
