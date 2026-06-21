"""Pure scoring for the presentation eval (HIG-280).

No DB/LLM/rendering -- given cases + an evaluate function, compute discrimination
and visual quality metrics. CI runs the math with stub evaluate functions;
the live evaluator (runner.py) needs Typst + optionally a vision model.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from kortny.evals.presentation.cases import PresentationCase


@dataclass(frozen=True, slots=True)
class PresentationScore:
    case_name: str
    visual_score: int  # 0-10; -1 when no vision model ran
    issue_count: int
    expects_defects: bool

    @property
    def flagged(self) -> bool:
        """True when this render shows a presentation problem."""
        return self.issue_count > 0 or (
            self.visual_score >= 0 and self.visual_score < 7
        )


@dataclass(frozen=True, slots=True)
class PresentationReport:
    scores: tuple[PresentationScore, ...]

    @property
    def mean_visual_score(self) -> float:
        """Mean visual_score among cases where vision actually ran (score >= 0)."""
        with_vision = [s.visual_score for s in self.scores if s.visual_score >= 0]
        return sum(with_vision) / len(with_vision) if with_vision else 0.0

    @property
    def pass_rate(self) -> float:
        """Fraction of cases with no detected defects."""
        if not self.scores:
            return 0.0
        passed = sum(1 for s in self.scores if not s.flagged)
        return passed / len(self.scores)

    @property
    def discrimination(self) -> float:
        """Fraction of cases where flagged matches expects_defects correctly.

        A perfect discriminating evaluator: all defective specs flagged AND all
        clean specs not flagged -> discrimination == 1.0.
        """
        if not self.scores:
            return 0.0
        correct = sum(1 for s in self.scores if s.flagged == s.expects_defects)
        return correct / len(self.scores)

    @property
    def summary_line(self) -> str:
        return (
            f"discrimination={self.discrimination:.3f} "
            f"pass_rate={self.pass_rate:.3f} "
            f"mean_visual_score={self.mean_visual_score:.1f}"
        )


# A callable that evaluates one PresentationCase and returns a PresentationScore.
EvaluateFn = Callable[[PresentationCase], PresentationScore]


def score_presentation(
    cases: Sequence[PresentationCase],
    evaluate_fn: EvaluateFn,
) -> PresentationReport:
    """Map evaluate_fn over cases and aggregate into a PresentationReport.

    PURE: contains zero rendering or LLM calls -- those live in evaluate_fn.
    """
    scores = tuple(evaluate_fn(c) for c in cases)
    return PresentationReport(scores=scores)
