"""Pure scoring for tool retrieval: Recall@k and nDCG@k (binary relevance).

These are the standard IR metrics used by the tool-retrieval literature
(ToolLLM, ToolRet) so our numbers are comparable to published baselines. All
functions are pure and dependency-free, so they run in CI without a DB,
embeddings, or network — the real retriever is plugged in via ``retrieve_fn``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from kortny.evals.retrieval.cases import RetrievalCase

# A retriever under test: maps a query to a ranked list of tool slugs (best
# first). This is exactly the seam find_tools' retriever will expose.
RetrieveFn = Callable[[str], Sequence[str]]


def recall_at_k(ranked: Sequence[str], expected: Sequence[str], k: int) -> float:
    """Fraction of expected tools found in the top-k ranked results."""

    if k < 1:
        raise ValueError("k must be at least 1")
    expected_set = {slug for slug in expected if slug}
    if not expected_set:
        return 0.0
    top = {slug for slug in ranked[:k]}
    return len(expected_set & top) / len(expected_set)


def ndcg_at_k(ranked: Sequence[str], expected: Sequence[str], k: int) -> float:
    """Normalized DCG@k with binary relevance (1 if a tool is expected)."""

    if k < 1:
        raise ValueError("k must be at least 1")
    expected_set = {slug for slug in expected if slug}
    if not expected_set:
        return 0.0
    dcg = 0.0
    for index, slug in enumerate(ranked[:k]):
        if slug in expected_set:
            dcg += 1.0 / math.log2(index + 2)
    ideal_hits = min(len(expected_set), k)
    idcg = sum(1.0 / math.log2(index + 2) for index in range(ideal_hits))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


@dataclass(frozen=True, slots=True)
class RetrievalScore:
    """Per-case scores at each evaluated k."""

    query: str
    expected: tuple[str, ...]
    ranked: tuple[str, ...]
    recall_at_k: dict[int, float]
    ndcg_at_k: dict[int, float]

    @property
    def hit(self) -> bool:
        """True if at least one expected tool appears anywhere in the ranking."""

        expected = {slug for slug in self.expected if slug}
        return bool(expected & set(self.ranked))


@dataclass(frozen=True, slots=True)
class RetrievalReport:
    """Aggregate retrieval quality across the dataset."""

    case_count: int
    ks: tuple[int, ...]
    mean_recall_at_k: dict[int, float]
    mean_ndcg_at_k: dict[int, float]
    hit_rate: float
    scores: tuple[RetrievalScore, ...]

    def summary_line(self) -> str:
        parts = [f"n={self.case_count}", f"hit_rate={self.hit_rate:.3f}"]
        for k in self.ks:
            parts.append(f"R@{k}={self.mean_recall_at_k[k]:.3f}")
            parts.append(f"nDCG@{k}={self.mean_ndcg_at_k[k]:.3f}")
        return " ".join(parts)


def score_retrieval(
    cases: Sequence[RetrievalCase],
    retrieve_fn: RetrieveFn,
    *,
    ks: Sequence[int] = (1, 3, 5, 10),
) -> RetrievalReport:
    """Run ``retrieve_fn`` over every case and aggregate Recall@k / nDCG@k.

    ``retrieve_fn`` is the only impure dependency; pass the real provider
    retriever for a true measurement, or a stub for unit tests.
    """

    if not cases:
        raise ValueError("cases must not be empty")
    k_tuple = tuple(ks)
    if not k_tuple:
        raise ValueError("ks must not be empty")

    scores: list[RetrievalScore] = []
    for case in cases:
        ranked = tuple(retrieve_fn(case.query))
        scores.append(
            RetrievalScore(
                query=case.query,
                expected=case.expected_tool_slugs,
                ranked=ranked,
                recall_at_k={
                    k: recall_at_k(ranked, case.expected_tool_slugs, k) for k in k_tuple
                },
                ndcg_at_k={
                    k: ndcg_at_k(ranked, case.expected_tool_slugs, k) for k in k_tuple
                },
            )
        )

    n = len(scores)
    mean_recall = {
        k: sum(score.recall_at_k[k] for score in scores) / n for k in k_tuple
    }
    mean_ndcg = {k: sum(score.ndcg_at_k[k] for score in scores) / n for k in k_tuple}
    hit_rate = sum(1 for score in scores if score.hit) / n
    return RetrievalReport(
        case_count=n,
        ks=k_tuple,
        mean_recall_at_k=mean_recall,
        mean_ndcg_at_k=mean_ndcg,
        hit_rate=hit_rate,
        scores=tuple(scores),
    )
