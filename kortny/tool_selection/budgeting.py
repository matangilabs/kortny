"""Canonical embedding text for external tool cards.

The pre-flight tool-selection pipeline (budget compaction, lexical scoring,
arbitration) was removed with the move to agent-driven retrieval (HIG-269).
What survives is the one helper the retrieval path still needs: the canonical
text used to embed a tool card, shared by the Composio catalog sync, the
provider re-ranker, and the find_tools catalog retriever so all three embed
cards identically.
"""

from __future__ import annotations

from kortny.tool_selection.models import ToolCard


def tool_card_embedding_text(card: ToolCard) -> str:
    """Compose the canonical embedding text for one external tool card."""

    return (
        f"{card.display_name}. {card.description} "
        f"Capabilities: {', '.join(card.capabilities)}. "
        f"Toolkit: {card.toolkit_slug}."
    )
