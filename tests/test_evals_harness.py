"""Offline tests for the agent eval harness (HIG-258).

No live LLM and no database: a fake provider drives the runner, and the scorers
are pure functions. CI has no LLM_API_KEY, so the suite must never call out.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import replace
from decimal import Decimal

from kortny.evals.cases import IntentCase, load_cases
from kortny.evals.runner import CaseResult, run_intent_case
from kortny.evals.scoring import _percentile, aggregate, render_table
from kortny.intent.models import IntentClassification, IntentRequest
from kortny.llm import ChatMessage, Completion, TokenUsage
from kortny.tools.types import JsonObject, JsonSchema

_VALID_DECISION: dict[str, object] = {
    "addressed_to_kortny": True,
    "classification": "task_request",
    "confidence": 0.9,
    "should_create_task": True,
    "should_ack_with_reaction": False,
    "needs_channel_context": False,
    "needs_thread_context": False,
    "needs_file_context": False,
    "model_tier": "cheap",
    "reason": "user asked to do work",
}


class _FakeProvider:
    """Returns canned content; satisfies the LLMProvider protocol."""

    def __init__(
        self, content: str | None, *, raise_exc: Exception | None = None
    ) -> None:
        self.model = "fake/model"
        self._content = content
        self._raise = raise_exc

    def complete(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[JsonSchema] = (),
        *,
        response_format: JsonObject | None = None,
    ) -> Completion:
        if self._raise is not None:
            raise self._raise
        return Completion(
            content=self._content,
            tool_calls=(),
            usage=TokenUsage(input_tokens=100, output_tokens=20),
            cost_usd=Decimal("0.0001"),
        )


def _case(classification: str = "task_request") -> IntentCase:
    return IntentCase(
        id="c1",
        request=IntentRequest(text="do the thing", surface="dm"),
        expected_classification=IntentClassification(classification),
    )


def test_fixture_is_well_formed() -> None:
    case_file = load_cases()
    assert case_file.version >= 1
    assert case_file.intent_cases
    ids = [case.id for case in case_file.intent_cases]
    assert len(ids) == len(set(ids)), "case ids must be unique"
    for case in case_file.intent_cases:
        assert isinstance(case.request, IntentRequest)
        assert isinstance(case.expected_classification, IntentClassification)


def test_run_intent_case_valid_and_correct() -> None:
    provider = _FakeProvider(json.dumps(_VALID_DECISION))
    result = run_intent_case(provider, "fake", _case("task_request"))
    assert result.json_valid is True
    assert result.label_correct is True
    assert result.error is None
    assert result.input_tokens == 100
    assert result.cost_usd == Decimal("0.0001")


def test_run_intent_case_valid_but_wrong_label() -> None:
    provider = _FakeProvider(json.dumps(_VALID_DECISION))
    result = run_intent_case(provider, "fake", _case("ignore"))
    assert result.json_valid is True
    assert result.label_correct is False


def test_run_intent_case_invalid_json_is_recorded_not_raised() -> None:
    provider = _FakeProvider("not json at all")
    result = run_intent_case(provider, "fake", _case())
    assert result.json_valid is False
    assert result.label_correct is None
    assert result.error is not None


def test_run_intent_case_provider_failure_is_captured() -> None:
    provider = _FakeProvider(None, raise_exc=RuntimeError("boom"))
    result = run_intent_case(provider, "fake", _case())
    assert result.json_valid is False
    assert result.error is not None and "boom" in result.error
    assert result.input_tokens == 0


def test_percentile_interpolates() -> None:
    assert _percentile([], 0.5) == 0.0
    assert _percentile([42.0], 0.95) == 42.0
    assert _percentile([0.0, 100.0], 0.5) == 50.0
    assert _percentile([10.0, 20.0, 30.0, 40.0], 0.5) == 25.0


def _result(
    *,
    json_valid: bool,
    label_correct: bool | None,
    latency_ms: int,
    error: str | None = None,
    tokens: int = 100,
) -> CaseResult:
    return CaseResult(
        case_id="c",
        candidate_name="m",
        kind="intent",
        json_valid=json_valid,
        label_correct=label_correct,
        latency_ms=latency_ms,
        cost_usd=Decimal("0.001"),
        input_tokens=tokens,
        output_tokens=tokens,
        error=error,
    )


def test_aggregate_metrics() -> None:
    results = [
        _result(json_valid=True, label_correct=True, latency_ms=100),
        _result(json_valid=True, label_correct=False, latency_ms=200),
        _result(json_valid=False, label_correct=None, latency_ms=150, error="bad json"),
        # transport error: no content, zero tokens — excluded from validity scoring
        _result(
            json_valid=False,
            label_correct=None,
            latency_ms=50,
            error="Timeout",
            tokens=0,
        ),
    ]
    # the transport-error row carries no cost
    results[-1] = replace(results[-1], cost_usd=None)

    summaries = aggregate(results)
    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.n_calls == 4
    assert summary.n_errors == 1  # only the transport error
    # completed = 3 rows; 2 valid -> 2/3
    assert abs(summary.json_validity_rate - (2 / 3)) < 1e-9
    # labeled = 2 rows; 1 correct -> 0.5
    assert summary.intent_label_accuracy == 0.5
    assert summary.latency_p50_ms > 0
    assert "m" in render_table(summaries)
