"""Tests for the intent-classifier eval harness (HIG-203).

Pure: exercises the scorer + seed-dataset integrity. The live classifier run
(kortny.evals.intent.runner) needs an API key and runs offline, not in CI.
"""

from __future__ import annotations

from kortny.evals.intent import (
    SEED_INTENT_CASES,
    IntentCase,
    score_intent,
)
from kortny.intent.models import IntentClassification


def test_perfect_classifier_scores_one() -> None:
    report = score_intent(SEED_INTENT_CASES, lambda case: case.expected)
    assert report.accuracy == 1.0
    assert report.case_count == len(SEED_INTENT_CASES)
    assert report.misses == ()


def test_always_ignore_classifier_is_penalized() -> None:
    report = score_intent(SEED_INTENT_CASES, lambda _case: IntentClassification.ignore)
    assert report.accuracy < 1.0
    # Every non-ignore expected case is a miss.
    assert len(report.misses) == sum(
        1 for c in SEED_INTENT_CASES if c.expected is not IntentClassification.ignore
    )


def test_per_class_precision_recall_math() -> None:
    cases = (
        IntentCase(
            "a", SEED_INTENT_CASES[0].surface, IntentClassification.task_request
        ),
        IntentCase(
            "b", SEED_INTENT_CASES[0].surface, IntentClassification.task_request
        ),
        IntentCase("c", SEED_INTENT_CASES[0].surface, IntentClassification.ignore),
    )
    # Predict task_request for a+c (one right, one wrong), ignore for b.
    preds = {
        "a": IntentClassification.task_request,
        "b": IntentClassification.ignore,
        "c": IntentClassification.task_request,
    }
    report = score_intent(cases, lambda case: preds[case.text])
    by_label = {stat.label: stat for stat in report.per_class}
    task = by_label["task_request"]
    assert task.support == 2 and task.correct == 1 and task.predicted == 2
    assert task.recall == 0.5
    assert task.precision == 0.5


def test_seed_dataset_is_well_formed() -> None:
    assert len(SEED_INTENT_CASES) >= 12
    texts = [c.text for c in SEED_INTENT_CASES]
    assert len(texts) == len(set(texts)), "duplicate case text"
    for case in SEED_INTENT_CASES:
        assert case.text.strip()
        assert isinstance(case.expected, IntentClassification)


def test_seed_covers_the_no_trigger_failure_modes() -> None:
    # The highest-stakes class: soft mentions that must NOT become tasks.
    no_trigger = {
        IntentClassification.ignore,
        IntentClassification.ambient_observation,
        IntentClassification.third_person_reference,
    }
    assert any(c.expected in no_trigger for c in SEED_INTENT_CASES)
    assert any("no_trigger" in c.tags for c in SEED_INTENT_CASES)
