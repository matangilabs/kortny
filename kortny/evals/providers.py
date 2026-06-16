"""Model candidates and provider construction for the eval matrix.

Each candidate is built into a bare ``LiteLLMProvider`` (no DB, no task). The
provider's ``Completion`` already carries token usage and per-call cost, so the
harness reads cost/latency straight off the response without ``record_llm_usage``.
"""

from __future__ import annotations

from dataclasses import dataclass

from kortny.config import Settings
from kortny.llm import LLMProvider
from kortny.llm.litellm_provider import create_litellm_provider


@dataclass(frozen=True, slots=True)
class ModelCandidate:
    """A model to evaluate. ``model=None`` uses the settings default model."""

    name: str
    model: str | None = None


def build_provider(settings: Settings, candidate: ModelCandidate) -> LLMProvider:
    """Construct a bare provider for one candidate model."""

    return create_litellm_provider(settings, model=candidate.model)
