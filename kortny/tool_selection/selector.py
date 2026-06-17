"""Tool selection strategies for scoped external tools."""

from __future__ import annotations

import json
import uuid
from collections.abc import Sequence
from dataclasses import replace
from typing import Protocol

from pydantic import BaseModel, Field, ValidationError, field_validator

from kortny.llm import ChatMessage, Completion
from kortny.tool_selection.models import (
    DEFAULT_PROMPT_DESCRIPTION_CHARS,
    ToolCard,
    ToolSelection,
    ToolSelectionResult,
)
from kortny.tools.types import JsonObject, JsonSchema

TOOL_SELECTOR_RESPONSE_FORMAT: JsonObject = {"type": "json_object"}
DEFAULT_TOOL_SELECTOR_MAX_PROMPT_CHARS = 12000
MIN_PROMPT_DESCRIPTION_CHARS = 80
MIN_PROMPT_TASK_CHARS = 300
# The fit loop never trims the external candidate list below this floor: the
# native-card section alone can exceed the whole budget (observed live), and
# trimming externals to zero silently turns the selector into a no-op.
MIN_EXTERNAL_PROMPT_CANDIDATES = 8
TOOL_SELECTOR_SYSTEM_PROMPT = """You are Kortny's tool selection preflight.

Select external tools only when they are materially useful for the user's Slack task.
Native tools are always available, so do not select an external tool unless it is a better fit.
Prefer read-only tools for automatic execution. Do not select write or destructive tools.
Return strict JSON with:
{
  "selected_tools": [{"registry_name": "...", "confidence": 0.0-1.0, "reason": "..."}],
  "suppressed_native_tools": ["..."],
  "rejected_tools": [{"registry_name": "...", "confidence": 0.0-1.0, "reason": "..."}],
  "route_reason": "short_reason"
}

If a selected external tool can replace an overlapping native tool for this task, include that native tool name in suppressed_native_tools.
If the user explicitly names a connected server, toolkit, or integration, always select its matching tools — never substitute a native tool for an explicitly requested one.
Keep selected_tools to 0-3 items.
When the payload includes an "intent" block with "likely_tools", treat those as a strong prior from the upstream intent classifier. If you select nothing despite a non-empty likely_tools list that maps to available external candidates, you must explain why in route_reason.
"""


class SelectorLLMClient(Protocol):
    """Subset of LLMService used by the tool selector."""

    def complete(
        self,
        *,
        task_id: uuid.UUID,
        messages: Sequence[ChatMessage],
        tools: Sequence[JsonSchema] = (),
        response_format: JsonObject | None = None,
        prompt_name: str | None = None,
        prompt_source: str = "code",
    ) -> Completion:
        """Complete one tool-selection call."""


class ToolSelector(Protocol):
    """Select a short external-tool shortlist for one task."""

    def select(
        self,
        *,
        task_id: uuid.UUID,
        task_input: str,
        native_cards: Sequence[ToolCard],
        external_cards: Sequence[ToolCard],
        intent_classification: str | None = None,
        likely_tools: Sequence[str] = (),
        toolkit_affinity: Sequence[str] = (),
    ) -> ToolSelectionResult:
        """Return selected external tools and native suppressions."""


class LLMToolSelector:
    """Cheap-model selector with deterministic fallback outside this class."""

    def __init__(
        self,
        llm: SelectorLLMClient,
        *,
        max_prompt_chars: int = DEFAULT_TOOL_SELECTOR_MAX_PROMPT_CHARS,
    ) -> None:
        if max_prompt_chars < 1000:
            raise ValueError("max_prompt_chars must be at least 1000")
        self.llm = llm
        self.max_prompt_chars = max_prompt_chars

    def select(
        self,
        *,
        task_id: uuid.UUID,
        task_input: str,
        native_cards: Sequence[ToolCard],
        external_cards: Sequence[ToolCard],
        intent_classification: str | None = None,
        likely_tools: Sequence[str] = (),
        toolkit_affinity: Sequence[str] = (),
    ) -> ToolSelectionResult:
        if not external_cards:
            return ToolSelectionResult(route_reason="no_external_candidates")

        payload, budget = _fit_selector_payload(
            task_input=task_input,
            native_cards=native_cards,
            external_cards=external_cards,
            max_prompt_chars=self.max_prompt_chars,
            intent_classification=intent_classification,
            likely_tools=list(likely_tools),
            toolkit_affinity=list(toolkit_affinity),
        )
        completion = self.llm.complete(
            task_id=task_id,
            messages=(
                ChatMessage(role="system", content=TOOL_SELECTOR_SYSTEM_PROMPT),
                ChatMessage(role="user", content=json.dumps(payload, sort_keys=True)),
            ),
            response_format=TOOL_SELECTOR_RESPONSE_FORMAT,
            prompt_name="kortny.tool_selector",
        )
        parsed = _parse_selector_payload(
            completion.content,
            allowed_external_names={
                _payload_registry_name(candidate)
                for candidate in payload["external_candidates"]
                if isinstance(candidate, dict)
            },
            allowed_native_names={card.registry_name for card in native_cards},
        )
        forced = _explicitly_requested_selections(
            task_input=task_input,
            external_cards=external_cards,
            already_selected={
                selection.registry_name for selection in parsed.selected_tools
            },
        )
        if forced:
            parsed = replace(
                parsed,
                selected_tools=parsed.selected_tools + forced,
                route_reason=f"{parsed.route_reason}+explicit_toolkit_forced",
            )
        # Reachability floor (HIG-274): if the grounded intent named a connected
        # toolkit but the selector picked nothing from it, force it in so the
        # agent never claims a connected integration "isn't wired in".
        selected_toolkits = {
            (card.toolkit_slug or "").casefold()
            for selection in parsed.selected_tools
            for card in external_cards
            if card.registry_name == selection.registry_name and card.toolkit_slug
        }
        unmet_affinity = [
            slug
            for slug in toolkit_affinity
            if slug and slug.casefold() not in selected_toolkits
        ]
        grounded = _intent_grounded_selections(
            external_cards=external_cards,
            affinity_slugs=unmet_affinity,
            already_selected={
                selection.registry_name for selection in parsed.selected_tools
            },
        )
        if grounded:
            parsed = replace(
                parsed,
                selected_tools=parsed.selected_tools + grounded,
                route_reason=f"{parsed.route_reason}+intent_grounded_floor",
            )
        return replace(
            parsed,
            route_reason=_budgeted_route_reason(parsed.route_reason, budget),
            prompt_chars=budget.prompt_chars,
            prompt_char_budget=budget.prompt_char_budget,
            budget_omitted_candidate_names=budget.omitted_candidate_names,
        )


class HeuristicToolSelector:
    """Deterministic fallback for selector failures and tests."""

    def select(
        self,
        *,
        task_id: uuid.UUID,
        task_input: str,
        native_cards: Sequence[ToolCard],
        external_cards: Sequence[ToolCard],
        intent_classification: str | None = None,
        likely_tools: Sequence[str] = (),
        toolkit_affinity: Sequence[str] = (),
    ) -> ToolSelectionResult:
        del task_id, native_cards, intent_classification, likely_tools
        del toolkit_affinity
        if not external_cards:
            return ToolSelectionResult(route_reason="no_external_candidates")

        selections: list[ToolSelection] = []
        suppressed: list[str] = []
        rejected: list[ToolSelection] = []
        for card in external_cards:
            score = _score_tool_card(task_input, card)
            if score >= 0.42 and card.side_effect == "read":
                selections.append(
                    ToolSelection(
                        registry_name=card.registry_name,
                        confidence=min(0.95, score),
                        reason=f"Task matches {', '.join(card.capabilities)}.",
                    )
                )
                suppressed.extend(card.can_replace_native_tools)
            else:
                rejected.append(
                    ToolSelection(
                        registry_name=card.registry_name,
                        confidence=score,
                        reason="Insufficient task/capability match.",
                    )
                )

        return ToolSelectionResult(
            selected_tools=tuple(_dedupe_selections(selections)),
            suppressed_native_tools=tuple(dict.fromkeys(suppressed)),
            rejected_tools=tuple(rejected),
            route_reason="heuristic_capability_match"
            if selections
            else "heuristic_no_external_match",
            fallback_used=True,
        )


class _SelectionItem(BaseModel):
    registry_name: str
    confidence: float = Field(ge=0, le=1)
    reason: str = ""

    @field_validator("confidence", mode="before")
    @classmethod
    def _coerce_confidence(cls, value: object) -> object:
        if isinstance(value, list) and len(value) == 1:
            value = value[0]
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        return value


class _SelectorPayload(BaseModel):
    selected_tools: list[_SelectionItem] = Field(default_factory=list)
    suppressed_native_tools: list[str] = Field(default_factory=list)
    rejected_tools: list[_SelectionItem] = Field(default_factory=list)
    route_reason: str = "llm_tool_selection"


def _parse_selector_payload(
    content: str | None,
    *,
    allowed_external_names: set[str],
    allowed_native_names: set[str],
) -> ToolSelectionResult:
    if content is None or not content.strip():
        raise ValueError("tool selector returned empty content")
    try:
        payload = _SelectorPayload.model_validate_json(_extract_json(content))
    except (ValueError, ValidationError) as exc:
        raise ValueError("tool selector returned invalid JSON") from exc

    selected = [
        ToolSelection(
            registry_name=item.registry_name,
            confidence=item.confidence,
            reason=item.reason,
        )
        for item in payload.selected_tools
        if item.registry_name in allowed_external_names
    ]
    rejected = [
        ToolSelection(
            registry_name=item.registry_name,
            confidence=item.confidence,
            reason=item.reason,
        )
        for item in payload.rejected_tools
        if item.registry_name in allowed_external_names
    ]
    suppressed = tuple(
        dict.fromkeys(
            name
            for name in payload.suppressed_native_tools
            if name in allowed_native_names
        )
    )
    return ToolSelectionResult(
        selected_tools=tuple(_dedupe_selections(selected)),
        suppressed_native_tools=suppressed if selected else (),
        rejected_tools=tuple(rejected),
        route_reason=payload.route_reason or "llm_tool_selection",
    )


def _score_tool_card(task_input: str, card: ToolCard) -> float:
    words = _words(task_input)
    if not words:
        return 0.0
    score = 0.0
    if card.toolkit_slug and card.toolkit_slug.casefold() in words:
        score += 0.45
    for capability in card.capabilities:
        capability_words = set(capability.split("_"))
        if capability_words & words:
            score += 0.16
    if card.toolkit_slug == "firecrawl":
        if words & FIRECRAWL_SEARCH_WORDS:
            score += 0.42
        if words & FIRECRAWL_SCRAPE_WORDS:
            score += 0.42
    return min(1.0, score)


class _SelectorPromptBudget(BaseModel):
    prompt_chars: int
    prompt_char_budget: int
    original_candidate_count: int
    selected_candidate_count: int
    omitted_candidate_names: tuple[str, ...] = ()

    @property
    def trimmed(self) -> bool:
        return bool(self.omitted_candidate_names)


def _fit_selector_payload(
    *,
    task_input: str,
    native_cards: Sequence[ToolCard],
    external_cards: Sequence[ToolCard],
    max_prompt_chars: int,
    intent_classification: str | None = None,
    likely_tools: list[str] | None = None,
    toolkit_affinity: list[str] | None = None,
) -> tuple[JsonObject, _SelectorPromptBudget]:
    candidates = list(external_cards)
    description_chars = DEFAULT_PROMPT_DESCRIPTION_CHARS
    prompt_task_input = task_input
    payload = _selector_payload(
        task_input=prompt_task_input,
        native_cards=native_cards,
        external_cards=candidates,
        max_description_chars=description_chars,
        intent_classification=intent_classification,
        likely_tools=likely_tools,
        toolkit_affinity=toolkit_affinity,
    )
    prompt_chars = _selector_prompt_chars(payload)
    while prompt_chars > max_prompt_chars:
        # Shrink descriptions (native + external) before dropping candidates;
        # candidates are relevance-ordered so the tail goes first, but never
        # below the floor — an over-budget prompt beats an empty candidate list.
        if description_chars > MIN_PROMPT_DESCRIPTION_CHARS:
            description_chars = max(
                MIN_PROMPT_DESCRIPTION_CHARS, description_chars // 2
            )
        elif len(candidates) > MIN_EXTERNAL_PROMPT_CANDIDATES:
            candidates.pop()
        elif len(prompt_task_input) > MIN_PROMPT_TASK_CHARS:
            prompt_task_input = _truncate_text(
                prompt_task_input,
                max_chars=max(MIN_PROMPT_TASK_CHARS, len(prompt_task_input) // 2),
            )
        else:
            break
        payload = _selector_payload(
            task_input=prompt_task_input,
            native_cards=native_cards,
            external_cards=candidates,
            max_description_chars=description_chars,
            intent_classification=intent_classification,
            likely_tools=likely_tools,
            toolkit_affinity=toolkit_affinity,
        )
        prompt_chars = _selector_prompt_chars(payload)

    selected_names = {card.registry_name for card in candidates}
    omitted_names = tuple(
        card.registry_name
        for card in external_cards
        if card.registry_name not in selected_names
    )
    return payload, _SelectorPromptBudget(
        prompt_chars=prompt_chars,
        prompt_char_budget=max_prompt_chars,
        original_candidate_count=len(external_cards),
        selected_candidate_count=len(candidates),
        omitted_candidate_names=omitted_names,
    )


def _selector_payload(
    *,
    task_input: str,
    native_cards: Sequence[ToolCard],
    external_cards: Sequence[ToolCard],
    max_description_chars: int,
    intent_classification: str | None = None,
    likely_tools: list[str] | None = None,
    toolkit_affinity: list[str] | None = None,
) -> JsonObject:
    payload: JsonObject = {
        "task_input": task_input,
        "native_tools": [
            card.prompt_payload(max_description_chars=max_description_chars)
            for card in native_cards
        ],
        "external_candidates": [
            card.prompt_payload(max_description_chars=max_description_chars)
            for card in external_cards
        ],
        "note": "all external candidates listed are already connected and runnable",
        "rules": {
            "read_tools_can_run_automatically": True,
            "write_or_destructive_tools_require_approval": True,
            "max_selected_tools": 3,
        },
    }
    intent: dict[str, object] = {}
    if intent_classification is not None:
        intent["classification"] = intent_classification
    if likely_tools:
        intent["likely_tools"] = likely_tools
    if toolkit_affinity:
        intent["toolkit_affinity"] = toolkit_affinity
    if intent:
        payload["intent"] = intent
    return payload


def _selector_prompt_chars(payload: JsonObject) -> int:
    return len(TOOL_SELECTOR_SYSTEM_PROMPT) + len(json.dumps(payload, sort_keys=True))


def _payload_registry_name(payload: JsonObject) -> str:
    registry_name = payload.get("registry_name")
    return registry_name if isinstance(registry_name, str) else ""


MAX_FORCED_TOOLKIT_SELECTIONS = 4
EXPLICIT_REQUEST_REASON_PREFIX = "user explicitly named"


def _explicitly_requested_selections(
    *,
    task_input: str,
    external_cards: Sequence[ToolCard],
    already_selected: set[str],
) -> tuple[ToolSelection, ...]:
    """Force-include tools whose toolkit/server the user named verbatim.

    The selector LLM runs on the cheap tier and sometimes substitutes a
    native tool for an explicitly requested integration (observed with MCP
    servers). Naming a connected toolkit in the request is an unambiguous
    signal, so it must not be subject to LLM judgment.
    """

    words = _task_words(task_input)
    if not words:
        return ()
    forced: list[ToolSelection] = []
    for card in external_cards:
        if len(forced) >= MAX_FORCED_TOOLKIT_SELECTIONS:
            break
        slug = (card.toolkit_slug or "").casefold()
        if not slug or slug not in words:
            continue
        if card.registry_name in already_selected:
            continue
        forced.append(
            ToolSelection(
                registry_name=card.registry_name,
                confidence=0.9,
                reason=f"{EXPLICIT_REQUEST_REASON_PREFIX} {card.toolkit_slug}",
            )
        )
    return tuple(forced)


INTENT_GROUNDED_REASON_PREFIX = "intent grounded connected toolkit"


def _intent_grounded_selections(
    *,
    external_cards: Sequence[ToolCard],
    affinity_slugs: Sequence[str],
    already_selected: set[str],
) -> tuple[ToolSelection, ...]:
    """Force-include connected tools whose toolkit the grounded intent named.

    The capability-grounded classifier emits ``toolkit_affinity`` from the
    connected set, so for a query like "what's on my plate" it resolves to the
    connected work tracker (e.g. ``linear``) without the user typing it. The
    cheap selector LLM still sometimes returns ``selected_tools: []`` and routes
    to a native tool, which makes the agent claim the integration "isn't wired
    in" (task c65e7b2f). When intent names a connected toolkit, reaching it must
    not be subject to LLM judgment (HIG-274). This is the selection-time analog
    of the verbatim ``_explicitly_requested_selections`` floor.
    """

    wanted = {slug.casefold() for slug in affinity_slugs if slug}
    if not wanted:
        return ()
    forced: list[ToolSelection] = []
    for card in external_cards:
        if len(forced) >= MAX_FORCED_TOOLKIT_SELECTIONS:
            break
        slug = (card.toolkit_slug or "").casefold()
        if not slug or slug not in wanted:
            continue
        if card.registry_name in already_selected:
            continue
        forced.append(
            ToolSelection(
                registry_name=card.registry_name,
                confidence=0.85,
                reason=f"{INTENT_GROUNDED_REASON_PREFIX} {card.toolkit_slug}",
            )
        )
    return tuple(forced)


def _task_words(text: str) -> set[str]:
    return {
        "".join(char for char in raw.casefold() if char.isalnum())
        for raw in text.replace("/", " ").replace("-", " ").replace("_", " ").split()
        if raw.strip()
    } - {""}


def _budgeted_route_reason(
    route_reason: str,
    budget: _SelectorPromptBudget,
) -> str:
    if not budget.trimmed:
        return route_reason
    return f"{route_reason}+prompt_budget_trimmed"


def _truncate_text(value: str, *, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    if max_chars <= 3:
        return value[:max_chars]
    return value[: max_chars - 3].rstrip() + "..."


def _words(text: str) -> set[str]:
    return {
        "".join(char for char in raw.casefold() if char.isalnum())
        for raw in text.replace("/", " ").replace("-", " ").split()
        if raw.strip()
    } - {""}


def _dedupe_selections(selections: Sequence[ToolSelection]) -> list[ToolSelection]:
    chosen: dict[str, ToolSelection] = {}
    for selection in selections:
        current = chosen.get(selection.registry_name)
        if current is None or selection.confidence > current.confidence:
            chosen[selection.registry_name] = selection
    return sorted(chosen.values(), key=lambda item: item.confidence, reverse=True)


def _extract_json(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object found")
    candidate = stripped[start : end + 1]
    json.loads(candidate)
    return candidate


FIRECRAWL_SEARCH_WORDS = frozenset(
    {
        "ai",
        "audit",
        "current",
        "find",
        "latest",
        "recent",
        "research",
        "search",
        "source",
        "sources",
        "trend",
        "trends",
        "web",
    }
)
FIRECRAWL_SCRAPE_WORDS = frozenset(
    {
        "crawl",
        "inspect",
        "page",
        "scrape",
        "site",
        "url",
        "website",
    }
)
