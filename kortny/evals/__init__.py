"""Offline agent eval harness (HIG-258).

Replays recorded/synthetic cases through a chosen model configuration and scores
the typed outputs — JSON-validity, label accuracy, latency, and cost — so model
and prompt choices are measured rather than guessed. Slice 1 covers the intent
classifier (the first cheap_fast consumer) to unblock a cheap_fast bake-off.

Pure-offline by design: it talks to a provider directly and calls the consumer
in isolation, so it needs no database and no live task. The only command that
hits a provider is ``python -m kortny.evals run`` (costs money, needs
``LLM_API_KEY``); everything else is pure functions over recorded results.
"""

from __future__ import annotations

from kortny.evals.cases import EvalCaseFile, IntentCase, load_cases
from kortny.evals.providers import ModelCandidate, build_provider
from kortny.evals.runner import CaseResult, run_intent_case, run_matrix
from kortny.evals.scoring import (
    CandidateSummary,
    aggregate,
    render_table,
    summaries_to_json,
)

__all__ = [
    "CandidateSummary",
    "CaseResult",
    "EvalCaseFile",
    "IntentCase",
    "ModelCandidate",
    "aggregate",
    "build_provider",
    "load_cases",
    "render_table",
    "run_intent_case",
    "run_matrix",
    "summaries_to_json",
]
