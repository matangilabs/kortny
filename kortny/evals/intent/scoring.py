"""Pure scoring for the intent-classifier eval (HIG-203).

No DB/LLM — given cases + a classify function, compute accuracy and per-class
counts. CI runs the math with a dict-backed fn; the live classifier run is
offline (needs an API key), mirroring the retrieval eval.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from kortny.evals.intent.cases import IntentCase
from kortny.intent.models import IntentClassification

# Predicts the classification for a case (the live runner wraps the real LLM).
ClassifyFn = Callable[[IntentCase], IntentClassification]


@dataclass(frozen=True, slots=True)
class IntentScore:
    text: str
    expected: IntentClassification
    predicted: IntentClassification

    @property
    def correct(self) -> bool:
        return self.expected is self.predicted


@dataclass(frozen=True, slots=True)
class ClassStat:
    label: str
    support: int  # cases whose expected == label
    correct: int  # of those, predicted == label
    predicted: int  # cases predicted == label (for precision)

    @property
    def recall(self) -> float:
        return self.correct / self.support if self.support else 0.0

    @property
    def precision(self) -> float:
        return self.correct / self.predicted if self.predicted else 0.0


@dataclass(frozen=True, slots=True)
class IntentReport:
    case_count: int
    accuracy: float
    scores: tuple[IntentScore, ...]
    per_class: tuple[ClassStat, ...]

    @property
    def misses(self) -> tuple[IntentScore, ...]:
        return tuple(s for s in self.scores if not s.correct)

    def summary_line(self) -> str:
        return (
            f"accuracy={self.accuracy:.3f} "
            f"({self.case_count - len(self.misses)}/{self.case_count})"
        )


def score_intent(cases: Sequence[IntentCase], classify_fn: ClassifyFn) -> IntentReport:
    scores = tuple(
        IntentScore(case.text, case.expected, classify_fn(case)) for case in cases
    )
    correct = sum(1 for s in scores if s.correct)
    accuracy = correct / len(scores) if scores else 0.0

    labels = sorted(
        {s.expected for s in scores} | {s.predicted for s in scores},
        key=lambda c: c.value,
    )
    per_class = tuple(
        ClassStat(
            label=label.value,
            support=sum(1 for s in scores if s.expected is label),
            correct=sum(
                1 for s in scores if s.expected is label and s.predicted is label
            ),
            predicted=sum(1 for s in scores if s.predicted is label),
        )
        for label in labels
    )
    return IntentReport(
        case_count=len(scores),
        accuracy=accuracy,
        scores=scores,
        per_class=per_class,
    )
