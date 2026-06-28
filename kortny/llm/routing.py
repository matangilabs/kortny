"""Intent-aware model routing.

The router speaks in internal capability tiers. Provider-specific model names
come from settings so ADK or another orchestration layer can later request the
same tiers without knowing vendor IDs.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from kortny.config import Settings
from kortny.db.models import Task, TaskEvent

INTENT_CLASSIFIED_MESSAGE = "intent_classification_completed"


class ModelRouteTier(StrEnum):
    """Provider-neutral model capability tiers."""

    cheap_fast = "cheap_fast"
    standard = "standard"
    analysis = "analysis"
    document = "document"
    high_reasoning = "high_reasoning"
    humanizer = "humanizer"
    vision = "vision"
    profiler = "profiler"


@dataclass(frozen=True, slots=True)
class ModelRoute:
    """Resolved model route for a task or support call."""

    tier: ModelRouteTier
    model: str
    reason: str


class ModelRouter:
    """Resolve model tiers to deployment-configured provider model IDs."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def route_for_tier(
        self,
        tier: ModelRouteTier,
        *,
        reason: str,
    ) -> ModelRoute:
        """Return the configured model for a specific internal tier."""

        return ModelRoute(
            tier=tier,
            model=self._model_for_tier(tier),
            reason=reason,
        )

    def route_for_task(
        self,
        task: Task,
        events: Sequence[TaskEvent] = (),
    ) -> ModelRoute:
        """Select a route using intent metadata first, then task text fallback.

        Image-bearing tasks are always routed to the vision tier first,
        deterministically from the uploaded file types (HIG-279 slice 2C).
        This check runs before any LLM-based intent decision so image tasks
        never require every other tier to be vision-capable.
        """

        # Deterministic vision routing: image attachment → vision tier.
        # Pure regex over task.input — no LLM, no DB.  Text-only tasks fall
        # through to the existing intent/depth routing unchanged.
        # Lazy import avoids an import cycle: kortny.agent.__init__ imports
        # kortny.agent.context which imports kortny.llm, so a top-level import
        # of any kortny.agent module from kortny.llm.routing would deadlock on
        # partial initialisation.  attachment_parsing is a pure leaf; the
        # lazy import resolves cleanly at call time after all modules are loaded.
        from kortny.agent.attachment_parsing import (
            parse_image_attachment_pairs,  # noqa: PLC0415
        )

        if parse_image_attachment_pairs(task.input):
            return self.route_for_tier(
                ModelRouteTier.vision,
                reason="image attachment -> vision tier (HIG-279)",
            )

        decision = effective_intent_decision(latest_intent_decision(events))
        tier = _tier_from_intent_decision(decision)
        if tier is not None:
            return self.route_for_tier(
                tier,
                reason="intent_classifier",
            )

        tier = _tier_from_task_input(task.input)
        return self.route_for_tier(
            tier,
            reason="task_input_fallback",
        )

    def _model_for_tier(self, tier: ModelRouteTier) -> str:
        default = self.settings.llm_model
        if tier is ModelRouteTier.cheap_fast:
            return self.settings.llm_cheap_model or default
        if tier is ModelRouteTier.standard:
            return self.settings.llm_standard_model or default
        if tier is ModelRouteTier.analysis:
            return (
                self.settings.llm_analysis_model
                or self.settings.llm_standard_model
                or default
            )
        if tier is ModelRouteTier.document:
            return (
                self.settings.llm_document_model
                or self.settings.llm_analysis_model
                or self.settings.llm_standard_model
                or default
            )
        if tier is ModelRouteTier.humanizer:
            # The humanizer is a stylistic Slack-formatting rewrite, the cheapest
            # cognitive task in the system — yet an unset LLM_HUMANIZER_MODEL used
            # to fall back to the *standard* tier, putting a slow mid/large model
            # on the response critical path (~40s observed, the single biggest
            # chunk of a 2-minute reply — HIG-268). Prefer the cheap/fast tier so
            # the default is fast; deployments can still pin LLM_HUMANIZER_MODEL.
            return (
                self.settings.llm_humanizer_model
                or self.settings.llm_cheap_model
                or self.settings.llm_standard_model
                or default
            )
        if tier is ModelRouteTier.vision:
            # Image-understanding tasks route here (HIG-279 slice 2C). Falls back
            # to llm_model when unset; if neither is vision-capable the LLMService
            # fail-loud assert_vision_capable check fires with a clear message.
            return self.settings.llm_vision_model or default
        if tier is ModelRouteTier.profiler:
            # Capability profiler runs cheap JSON passes on tool batches; must be
            # a capable-JSON model. Falls back to standard, then llm_model.
            return (
                self.settings.llm_profiler_model
                or self.settings.llm_standard_model
                or default
            )
        return (
            self.settings.llm_high_reasoning_model
            or self.settings.llm_analysis_model
            or self.settings.llm_standard_model
            or default
        )


def latest_intent_decision(events: Sequence[TaskEvent]) -> Mapping[str, Any] | None:
    """Return the latest recorded intent decision payload, if present."""

    for event in sorted(events, key=lambda item: item.seq, reverse=True):
        payload = event.payload
        if payload.get("message") != INTENT_CLASSIFIED_MESSAGE:
            continue
        decision = payload.get("decision")
        if isinstance(decision, Mapping):
            return decision
    return None


def effective_intent_decision(
    decision: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    """Return the execution-driving view of a possibly decomposed intent."""

    if decision is None:
        return None
    primary = decision.get("primary_intent")
    if not isinstance(primary, Mapping):
        return decision
    should_execute = primary.get("should_execute")
    if isinstance(should_execute, bool) and not should_execute:
        return decision

    effective = dict(decision)
    classification = _optional_str(primary.get("type"))
    if classification:
        effective["classification"] = classification
    effective["should_create_task"] = True

    likely_tools = _string_list(primary.get("likely_tools"))
    if likely_tools:
        effective["likely_tools"] = likely_tools

    for key in (
        "needs_channel_context",
        "needs_thread_context",
        "needs_file_context",
    ):
        value = primary.get(key)
        if isinstance(value, bool):
            effective[key] = value

    objective = _optional_str(primary.get("objective"))
    if objective:
        effective["reason"] = objective
    effective["effective_intent_source"] = "primary_intent"
    return effective


def _tier_from_intent_decision(
    decision: Mapping[str, Any] | None,
) -> ModelRouteTier | None:
    if decision is None:
        return None

    likely_tools = _string_set(decision.get("likely_tools"))
    toolkit_affinity = _string_set(decision.get("toolkit_affinity"))
    # A document/report deliverable routes to the document tier (Sonnet) even
    # when the classifier rated the task "strong" — synthesis belongs on the
    # document model, not high_reasoning (Opus). Match capability hints, not just
    # the literal pdf_generator name: intents commonly signal "document_generation"
    # in likely_tools / toolkit_affinity. (HIG-265: a routine report should not
    # run every turn on Opus.)
    if (likely_tools | toolkit_affinity) & DOCUMENT_TOOL_HINTS:
        return ModelRouteTier.document
    if "slack_file_read" in likely_tools:
        return ModelRouteTier.analysis
    # The classifier's likely_tools are unreliable (it can hallucinate tool
    # names that aren't Kortny's), so a document deliverable can slip past the
    # tool-hint check above and fall through to "strong" → Opus. The objective
    # text the classifier wrote is far more reliable: if it describes producing a
    # document/report, route to the document tier (Sonnet) per HIG-265 — a
    # routine report should not run every turn on Opus.
    if _objective_signals_document(decision):
        return ModelRouteTier.document
    if "web_search" in likely_tools:
        return ModelRouteTier.analysis

    classification = _optional_str(decision.get("classification"))
    if classification in {
        "memory_candidate",
        "clarification",
        "cancel_or_retry",
    }:
        return ModelRouteTier.cheap_fast

    model_tier = _optional_str(decision.get("model_tier"))
    if model_tier == "cheap":
        return ModelRouteTier.cheap_fast
    if model_tier == "strong":
        return ModelRouteTier.high_reasoning
    if model_tier == "standard":
        return ModelRouteTier.standard

    return None


def _tier_from_task_input(input_text: str) -> ModelRouteTier:
    normalized = input_text.casefold()
    if any(keyword in normalized for keyword in DOCUMENT_KEYWORDS):
        return ModelRouteTier.document
    if any(keyword in normalized for keyword in ANALYSIS_KEYWORDS):
        return ModelRouteTier.analysis
    if len(normalized) <= 160:
        return ModelRouteTier.standard
    return ModelRouteTier.analysis


def _objective_signals_document(decision: Mapping[str, Any]) -> bool:
    """Whether the intent's objective/reason text describes a document deliverable.

    Reads the free-text objective the classifier wrote (``reason`` /
    ``objective`` / the primary intent's objective) rather than its unreliable
    ``likely_tools`` list, matching the same document keywords the task-input
    fallback uses.
    """

    parts: list[str] = []
    for key in ("reason", "objective"):
        value = decision.get(key)
        if isinstance(value, str):
            parts.append(value)
    primary = decision.get("primary_intent")
    if isinstance(primary, Mapping):
        objective = primary.get("objective")
        if isinstance(objective, str):
            parts.append(objective)
    blob = " ".join(parts).casefold()
    return any(keyword in blob for keyword in DOCUMENT_KEYWORDS)


def _string_set(value: object) -> set[str]:
    if not isinstance(value, Iterable) or isinstance(value, str | bytes):
        return set()
    return {item for item in value if isinstance(item, str)}


def _string_list(value: object) -> list[str]:
    if not isinstance(value, Iterable) or isinstance(value, str | bytes):
        return []
    return [item for item in value if isinstance(item, str) and item]


def _optional_str(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


# Capability / tool hints in an intent decision's likely_tools / toolkit_affinity
# that mean "produce a document deliverable" → route to the document tier. Covers
# both the literal native tool name and the capability words the classifier emits.
DOCUMENT_TOOL_HINTS = frozenset(
    {
        "document_studio",
        "pdf_generator",
        "document_generation",
        "report_generation",
        "deck_builder",
        "deck-builder",
        "spreadsheet_builder",
        "spreadsheet-builder",
    }
)
DOCUMENT_KEYWORDS = frozenset(
    {
        "pdf",
        "report",
        "document",
        "doc",
        "deck",
        "slides",
        "presentation",
        "brochure",
        "whitepaper",
        "make it",
        "extend this",
        "revise this",
        "version",
    }
)
ANALYSIS_KEYWORDS = frozenset(
    {
        "analyze",
        "analyse",
        "review",
        "summarize",
        "summarise",
        "compare",
        "research",
        "source",
        "latest",
        "recent",
        "market",
        "ticker",
        "earnings",
        "audit",
    }
)
