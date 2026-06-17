"""Budget-aware helpers for tool-selection prompts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from kortny.tool_selection.models import ToolCard

SEMANTIC_SCORE_WEIGHT = 0.6
LEXICAL_SCORE_WEIGHT = 0.4


@dataclass(frozen=True, slots=True)
class ToolCatalogCompaction:
    """Traceable summary of a selector catalog compaction decision."""

    original_candidate_count: int
    selected_candidate_count: int
    omitted_candidate_count: int
    max_candidates: int
    selected_candidate_names: tuple[str, ...]
    omitted_candidate_names: tuple[str, ...]
    reason: str

    @property
    def compacted(self) -> bool:
        return self.omitted_candidate_count > 0

    def to_payload(self) -> dict[str, object]:
        return {
            "original_candidate_count": self.original_candidate_count,
            "selected_candidate_count": self.selected_candidate_count,
            "omitted_candidate_count": self.omitted_candidate_count,
            "max_candidates": self.max_candidates,
            "selected_candidate_names": list(self.selected_candidate_names),
            "omitted_candidate_names": list(self.omitted_candidate_names),
            "reason": self.reason,
        }


def compact_tool_cards(
    *,
    task_input: str,
    cards: tuple[ToolCard, ...],
    max_candidates: int,
    semantic_scores: Mapping[str, float] | None = None,
    protected_toolkits: frozenset[str] = frozenset(),
) -> tuple[tuple[ToolCard, ...], ToolCatalogCompaction]:
    """Return a bounded, relevance-ranked selector catalog.

    When ``semantic_scores`` (registry_name -> cosine similarity) is provided,
    each card's final score is a hybrid of semantic retrieval and the lexical
    heuristic; when it is None, behavior is exactly the legacy lexical path.

    ``protected_toolkits`` is the capability-grounding reachability floor
    (HIG-274): cards whose toolkit the intent named (toolkit_affinity /
    likely_tools) and which are connected for this user are always kept, even if
    relevance ranking would trim them. This stops the failure where a request
    that clearly implies a connected tool (e.g. "my open Linear issues") gets
    ``selected_tools: []`` because the lexical/semantic score fell below the cap.
    """

    if max_candidates < 1:
        raise ValueError("max_candidates must be at least 1")
    if semantic_scores is None and len(cards) <= max_candidates:
        return cards, ToolCatalogCompaction(
            original_candidate_count=len(cards),
            selected_candidate_count=len(cards),
            omitted_candidate_count=0,
            max_candidates=max_candidates,
            selected_candidate_names=tuple(card.registry_name for card in cards),
            omitted_candidate_names=(),
            reason="within_budget",
        )

    scored = [
        (_hybrid_tool_card_score(task_input, card, semantic_scores), index, card)
        for index, card in enumerate(cards)
    ]
    ranked = sorted(scored, key=lambda item: (-item[0], item[1]))
    if len(cards) <= max_candidates:
        # Semantic path within budget: keep every card, relevance-ordered so
        # downstream prompt tail-trimming drops the least relevant first.
        return tuple(item[2] for item in ranked), ToolCatalogCompaction(
            original_candidate_count=len(cards),
            selected_candidate_count=len(cards),
            omitted_candidate_count=0,
            max_candidates=max_candidates,
            selected_candidate_names=tuple(item[2].registry_name for item in ranked),
            omitted_candidate_names=(),
            reason="within_budget",
        )
    protected = {slug.casefold() for slug in protected_toolkits if slug}

    def _is_protected(card: ToolCard) -> bool:
        slug = card.toolkit_slug
        return slug is not None and slug.casefold() in protected

    # Keep relevance order (most relevant first): the selector prompt fitter
    # trims candidates from the tail, so tail position must mean "least
    # relevant", not "registered last" (which silently dropped MCP tools
    # because the MCP provider runs after Composio). The reachability floor
    # (protected toolkits) is always kept; remaining budget goes to the most
    # relevant non-protected cards.
    floored = False
    if protected:
        protected_ranked = [item for item in ranked if _is_protected(item[2])]
        other_ranked = [item for item in ranked if not _is_protected(item[2])]
        budget_for_other = max(0, max_candidates - len(protected_ranked))
        floored = bool(protected_ranked)
        selected_ranked = sorted(
            protected_ranked + other_ranked[:budget_for_other],
            key=lambda item: (-item[0], item[1]),
        )
    else:
        selected_ranked = ranked[:max_candidates]
    selected = tuple(item[2] for item in selected_ranked)
    selected_names = {card.registry_name for card in selected}
    omitted = tuple(card for card in cards if card.registry_name not in selected_names)
    return selected, ToolCatalogCompaction(
        original_candidate_count=len(cards),
        selected_candidate_count=len(selected),
        omitted_candidate_count=len(omitted),
        max_candidates=max_candidates,
        selected_candidate_names=tuple(card.registry_name for card in selected),
        omitted_candidate_names=tuple(card.registry_name for card in omitted),
        reason="relevance_cap_floor" if floored else "relevance_cap",
    )


def tool_card_embedding_text(card: ToolCard) -> str:
    """Compose the canonical embedding text for one external tool card."""

    return (
        f"{card.display_name}. {card.description} "
        f"Capabilities: {', '.join(card.capabilities)}. "
        f"Toolkit: {card.toolkit_slug}."
    )


def _hybrid_tool_card_score(
    task_input: str,
    card: ToolCard,
    semantic_scores: Mapping[str, float] | None,
) -> float:
    lexical = _score_tool_card(task_input, card)
    if semantic_scores is None:
        return lexical
    semantic = semantic_scores.get(card.registry_name, 0.0)
    return SEMANTIC_SCORE_WEIGHT * semantic + LEXICAL_SCORE_WEIGHT * lexical


def _score_tool_card(task_input: str, card: ToolCard) -> float:
    words = _words(task_input)
    if not words:
        return 0.0

    text_parts = [
        card.registry_name,
        card.display_name,
        card.description,
        card.toolkit_slug or "",
        " ".join(card.tool_slugs),
        " ".join(card.capabilities),
    ]
    card_words = set().union(*(_words(part) for part in text_parts if part))
    overlap = words & card_words

    score = min(0.35, len(overlap) * 0.04)
    if card.toolkit_slug and card.toolkit_slug.casefold() in words:
        score += 0.45
    if card.provider == "mcp":
        # Admin-registered MCP servers are a handful of deliberate tools;
        # never let them drown under hundreds of auto-imported catalog cards.
        score += 0.15
    if card.side_effect == "read":
        score += 0.05
    for capability in card.capabilities:
        capability_words = set(capability.casefold().split("_"))
        if words & capability_words:
            score += 0.12
    if card.toolkit_slug == "firecrawl":
        if words & FIRECRAWL_SEARCH_WORDS:
            score += 0.16
        if words & FIRECRAWL_SCRAPE_WORDS:
            score += 0.38
    return min(1.0, score)


def _words(text: str) -> set[str]:
    return {
        "".join(char for char in raw.casefold() if char.isalnum())
        for raw in text.replace("/", " ").replace("-", " ").replace("_", " ").split()
        if raw.strip()
    } - {""}


FIRECRAWL_SEARCH_WORDS = frozenset(
    {
        "latest",
        "recent",
        "research",
        "search",
        "source",
        "sources",
    }
)

FIRECRAWL_SCRAPE_WORDS = frozenset(
    {
        "crawl",
        "extract",
        "scrape",
        "url",
        "website",
    }
)
