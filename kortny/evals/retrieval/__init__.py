"""Tool-retrieval evaluation (Orchestration Spine slice 3a, Linear HIG-271).

The measurement instrument that gates `find_tools` (Linear HIG-269): a versioned
dataset of queries -> ground-truth tools plus Recall@k / nDCG@k scoring. Per the
deep-research pass (ToolRet, ACL 2025), off-the-shelf tool retrieval is weak and
its quality directly caps agent task success, so we measure before we trust any
retriever — never ship an unmeasured one.
"""

from kortny.evals.retrieval.cases import SEED_RETRIEVAL_CASES, RetrievalCase
from kortny.evals.retrieval.scoring import (
    RetrievalReport,
    RetrievalScore,
    ndcg_at_k,
    recall_at_k,
    score_retrieval,
)

__all__ = [
    "SEED_RETRIEVAL_CASES",
    "RetrievalCase",
    "RetrievalReport",
    "RetrievalScore",
    "ndcg_at_k",
    "recall_at_k",
    "score_retrieval",
]
