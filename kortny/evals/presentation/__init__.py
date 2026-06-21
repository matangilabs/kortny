"""Presentation-quality eval harness (HIG-280).

Labeled DocumentSpec cases + pure scorer + offline runner so Document Studio
prompt or IR changes can be evaluated instead of shipped blind.
"""

from kortny.evals.presentation.cases import SEED_PRESENTATION_CASES, PresentationCase
from kortny.evals.presentation.scoring import (
    EvaluateFn,
    PresentationReport,
    PresentationScore,
    score_presentation,
)

__all__ = [
    "EvaluateFn",
    "PresentationCase",
    "PresentationReport",
    "PresentationScore",
    "SEED_PRESENTATION_CASES",
    "score_presentation",
]
