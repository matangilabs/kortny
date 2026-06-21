"""Tests for the presentation eval harness (HIG-280).

Pure: exercises the scorer + seed-dataset integrity. The live evaluator
(kortny.evals.presentation.runner) needs Typst + optionally a vision model
and runs offline, not in CI.
"""

from __future__ import annotations

from kortny.documents.ir import DocumentSpec
from kortny.evals.presentation import (
    SEED_PRESENTATION_CASES,
    PresentationReport,
    PresentationScore,
    score_presentation,
)
from kortny.evals.presentation.cases import PresentationCase


def test_score_presentation_perfect_discriminator() -> None:
    """An evaluator that always correctly flags defective cases gets discrimination=1.0."""

    def perfect_fn(case: PresentationCase) -> PresentationScore:
        if case.expects_defects:
            return PresentationScore(
                case_name=case.name,
                visual_score=3,
                issue_count=2,
                expects_defects=case.expects_defects,
            )
        return PresentationScore(
            case_name=case.name,
            visual_score=10,
            issue_count=0,
            expects_defects=case.expects_defects,
        )

    report = score_presentation(SEED_PRESENTATION_CASES, perfect_fn)
    assert report.discrimination == 1.0


def test_score_presentation_blind_evaluator() -> None:
    """A blind evaluator that always returns clean misses the defective cases."""

    def blind_fn(case: PresentationCase) -> PresentationScore:
        return PresentationScore(
            case_name=case.name,
            visual_score=10,
            issue_count=0,
            expects_defects=case.expects_defects,
        )

    report = score_presentation(SEED_PRESENTATION_CASES, blind_fn)
    # Discrimination should be < 1.0 because defective cases are not flagged
    assert report.discrimination < 1.0


def test_seed_dataset_integrity() -> None:
    """Seed dataset must be non-empty, have unique names, and cover both polarities."""
    assert len(SEED_PRESENTATION_CASES) > 0
    names = [c.name for c in SEED_PRESENTATION_CASES]
    assert len(names) == len(set(names)), "duplicate case names"
    assert any(c.expects_defects for c in SEED_PRESENTATION_CASES)
    assert any(not c.expects_defects for c in SEED_PRESENTATION_CASES)
    for c in SEED_PRESENTATION_CASES:
        assert isinstance(c.spec, DocumentSpec)


def test_report_math() -> None:
    """PresentationReport aggregates scores correctly."""
    scores = (
        PresentationScore(
            case_name="a", visual_score=8, issue_count=0, expects_defects=False
        ),
        PresentationScore(
            case_name="b", visual_score=6, issue_count=1, expects_defects=True
        ),
    )
    report = PresentationReport(scores=scores)
    assert report.mean_visual_score == 7.0
    # "a": visual_score=8 >= 7, issue_count=0 -> not flagged -> PASS
    # "b": visual_score=6 < 7 -> flagged -> FAIL
    assert report.pass_rate == 0.5
    # "a": flagged=False, expects_defects=False -> correct
    # "b": flagged=True, expects_defects=True -> correct
    assert report.discrimination == 1.0
