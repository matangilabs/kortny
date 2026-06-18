"""Intent-classifier eval (HIG-203).

The intent classifier is the highest-stakes prompt — it runs on every mention
and decides whether Kortny acts at all. This package is the labeled dataset +
pure scorer + offline runner so prompt edits can be evaluated instead of shipped
blind.
"""

from kortny.evals.intent.cases import SEED_INTENT_CASES, IntentCase
from kortny.evals.intent.scoring import (
    ClassifyFn,
    IntentReport,
    IntentScore,
    score_intent,
)

__all__ = [
    "ClassifyFn",
    "IntentCase",
    "IntentReport",
    "IntentScore",
    "SEED_INTENT_CASES",
    "score_intent",
]
