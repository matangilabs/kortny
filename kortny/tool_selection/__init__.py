"""Tool card models + the external-tool provider seam.

Agent-driven retrieval (HIG-269) replaced the pre-flight selection pipeline, so
the selector, arbitration, and budget-compaction modules are gone. What remains
is the shared vocabulary: the tool-card models, the canonical embedding text,
and the ``ExternalToolProvider`` protocol that Composio/MCP providers implement.
"""

from kortny.tool_selection.budgeting import tool_card_embedding_text
from kortny.tool_selection.models import (
    ToolCard,
    ToolSelection,
    ToolSelectionResult,
)
from kortny.tool_selection.providers import ExternalToolProvider

__all__ = [
    "ExternalToolProvider",
    "ToolCard",
    "ToolSelection",
    "ToolSelectionResult",
    "tool_card_embedding_text",
]
