"""Run eval cases through candidate models and capture per-call results.

The intent classifier exposes a raw-provider seam, so the harness builds the
exact production messages (system prompt + ``IntentRequest`` payload), times the
raw ``provider.complete`` call, then validates with the production
``parse_intent_decision``. Running the raw parse (not the full ``classify``)
keeps deterministic Python overrides from masking the model's JSON quality — a
model bake-off wants the model's own output.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal

from kortny.config import Settings
from kortny.evals.cases import IntentCase
from kortny.evals.providers import ModelCandidate, build_provider
from kortny.intent.classifier import (
    INTENT_RESPONSE_FORMAT,
    IntentClassificationError,
    parse_intent_decision,
)
from kortny.intent.prompts import INTENT_CLASSIFIER_SYSTEM_PROMPT
from kortny.llm import ChatMessage, LLMProvider


@dataclass(frozen=True, slots=True)
class CaseResult:
    """Outcome of one case run against one candidate."""

    case_id: str
    candidate_name: str
    kind: str
    json_valid: bool
    label_correct: bool | None
    latency_ms: int
    cost_usd: Decimal | None
    input_tokens: int
    output_tokens: int
    error: str | None = None


def run_intent_case(
    provider: LLMProvider,
    candidate_name: str,
    case: IntentCase,
) -> CaseResult:
    """Run one intent case; never raises (provider/parse errors are recorded)."""

    messages = (
        ChatMessage(role="system", content=INTENT_CLASSIFIER_SYSTEM_PROMPT),
        ChatMessage(role="user", content=case.request.model_dump_json()),
    )
    started = time.perf_counter()
    try:
        completion = provider.complete(
            messages,
            response_format=INTENT_RESPONSE_FORMAT,
        )
    except Exception as exc:  # noqa: BLE001 — record any provider failure as a case error
        return CaseResult(
            case_id=case.id,
            candidate_name=candidate_name,
            kind="intent",
            json_valid=False,
            label_correct=None,
            latency_ms=int((time.perf_counter() - started) * 1000),
            cost_usd=None,
            input_tokens=0,
            output_tokens=0,
            error=f"{type(exc).__name__}: {exc}",
        )
    latency_ms = int((time.perf_counter() - started) * 1000)

    json_valid = True
    label_correct: bool | None = None
    parse_error: str | None = None
    try:
        decision = parse_intent_decision(completion.content)
        label_correct = decision.classification == case.expected_classification
    except IntentClassificationError as exc:
        json_valid = False
        parse_error = str(exc)

    return CaseResult(
        case_id=case.id,
        candidate_name=candidate_name,
        kind="intent",
        json_valid=json_valid,
        label_correct=label_correct,
        latency_ms=latency_ms,
        cost_usd=completion.cost_usd,
        input_tokens=completion.usage.input_tokens,
        output_tokens=completion.usage.output_tokens,
        error=parse_error,
    )


def run_matrix(
    settings: Settings,
    candidates: Sequence[ModelCandidate],
    cases: Sequence[IntentCase],
    *,
    repeats: int = 1,
) -> list[CaseResult]:
    """Run every case against every candidate ``repeats`` times (sequential)."""

    results: list[CaseResult] = []
    for candidate in candidates:
        provider = build_provider(settings, candidate)
        for case in cases:
            for _ in range(max(1, repeats)):
                results.append(run_intent_case(provider, candidate.name, case))
    return results
