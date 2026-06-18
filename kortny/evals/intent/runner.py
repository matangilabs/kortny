"""Offline runner for the intent eval (HIG-203).

Builds a real classify function from ``LLMIntentClassifier`` and scores the seed
dataset. Needs a live LLM (an API key), so it runs on demand / offline — not in
CI. CI exercises the pure scorer (scoring.py) instead.

    uv run python -m kortny.evals.intent.runner
"""

from __future__ import annotations

from kortny.config import load_settings
from kortny.evals.intent.cases import SEED_INTENT_CASES, IntentCase
from kortny.evals.intent.scoring import IntentReport, score_intent
from kortny.intent.classifier import LLMIntentClassifier
from kortny.intent.models import IntentClassification, IntentRequest
from kortny.llm.litellm_provider import create_litellm_provider


def build_live_classify_fn() -> object:
    """A ClassifyFn backed by the real classifier (pre-task provider path)."""

    settings = load_settings()
    provider = create_litellm_provider(settings)
    classifier = LLMIntentClassifier(provider=provider)

    def classify(case: IntentCase) -> IntentClassification:
        decision = classifier.classify(
            request=IntentRequest(
                text=case.text,
                surface=case.surface,
                is_thread_follow_up=case.is_thread_follow_up,
                connected_integrations=case.connected_integrations,
            )
        )
        return decision.routing_classification()

    return classify


def run() -> IntentReport:
    report = score_intent(SEED_INTENT_CASES, build_live_classify_fn())  # type: ignore[arg-type]
    return report


def _main() -> None:
    report = run()
    print(f"\nINTENT EVAL: {report.summary_line()}\n")
    for stat in report.per_class:
        print(
            f"  {stat.label:<22} support={stat.support:<3} "
            f"recall={stat.recall:.2f} precision={stat.precision:.2f}"
        )
    if report.misses:
        print("\nMisses:")
        for miss in report.misses:
            print(
                f"  expected {miss.expected.value:<20} got {miss.predicted.value:<20} :: {miss.text}"
            )


if __name__ == "__main__":
    _main()
