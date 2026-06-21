"""Offline runner for the capability-grounding eval (HIG-274).

Builds a live classify function from ``LLMIntentClassifier`` and scores the
seed dataset. Needs a live LLM (an API key) and a database, so it runs on
demand / offline — not in CI.

The floor function is stubbed to return [] because the reachability floor
requires a full Composio + task setup that is impractical to wire end-to-end
in an offline runner. Run the floor assertions via the DB-backed integration
test suite instead.

    uv run python -m kortny.evals.capability.runner
"""

from __future__ import annotations

from kortny.config import load_settings
from kortny.evals.capability.cases import SEED_CAPABILITY_CASES, CapabilityCase
from kortny.evals.capability.scoring import (
    CapabilityReport,
    ClassifyFn,
    score_capability,
)
from kortny.intent.classifier import LLMIntentClassifier
from kortny.intent.models import IntentRequest
from kortny.llm.litellm_provider import create_litellm_provider


def build_live_classify_fn() -> ClassifyFn:
    """A ClassifyFn backed by the real intent classifier."""

    settings = load_settings()
    provider = create_litellm_provider(settings)
    classifier = LLMIntentClassifier(provider=provider)

    def classify(
        case: CapabilityCase,
    ) -> tuple[list[str], tuple[str, ...], tuple[str, ...]]:
        decision = classifier.classify(
            request=IntentRequest(
                text=case.request,
                surface=case.surface,
                connected_integrations=case.connected_toolkits,
            )
        )
        return (
            decision.routing_likely_tools(),
            decision.toolkit_affinity,
            decision.needs_connection,
        )

    return classify


def _stub_floor_fn(
    case: CapabilityCase, implied_toolkits: tuple[str, ...]
) -> list[str]:
    """Stub floor: always returns empty — full end-to-end floor needs a live DB."""
    return []


def run() -> CapabilityReport:
    report = score_capability(
        SEED_CAPABILITY_CASES,
        build_live_classify_fn(),
        _stub_floor_fn,
    )
    return report


def _main() -> None:
    report = run()
    print(f"\nCAPABILITY GROUNDING EVAL: {report.summary_line()}\n")
    if report.failures:
        print("Failures:")
        for result in report.failures:
            print(f"  case {result.case_id}: {result.request!r}")
            for failure in result.failures:
                print(f"    - {failure}")
    else:
        print("All cases passed.")


if __name__ == "__main__":
    _main()
