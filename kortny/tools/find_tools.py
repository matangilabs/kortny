"""Agent-driven tool retrieval (Orchestration Spine slice 1, Linear HIG-269).

`find_tools` lets the agent search the external catalog and load the matching
tools into the live registry mid-task, instead of relying on a pre-flight
narrowing pipeline to guess the toolset before the agent reasons. The loaded
tools become callable on the agent's next turn (the coordinator re-reads the
registry each turn).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from kortny.tools.registry import ToolRegistry
from kortny.tools.types import (
    JsonObject,
    JsonSchema,
    RecoverableToolError,
    Tool,
    ToolResult,
)

# Ranked tool slugs for a query (best first).
ToolRetriever = Callable[[str], Sequence[str]]
# Build executable tools for an explicit set of tool slugs.
ToolLoader = Callable[[Sequence[str]], Sequence[Tool]]

DEFAULT_FIND_TOOLS_TOP_K = 5

FIND_TOOLS_DESCRIPTION = (
    "Search for and load external integration tools by what you want to do. "
    "Call this when the task needs an integration (Linear, Notion, a data "
    "source, etc.) and you do not already have a matching tool available. "
    "Pass a short natural-language description of the capability you need "
    "(e.g. 'list my open Linear issues', 'search Notion pages', 'latest stock "
    "price'). The best-matching connected tools are loaded and become callable "
    "on your next step; their names and signatures are returned here. If "
    "nothing relevant loads, refine the query or proceed with what you have."
)


class FindToolsTool:
    """Retrieve and runtime-load external tools the agent asks for."""

    name = "find_tools"
    description = FIND_TOOLS_DESCRIPTION

    def __init__(
        self,
        *,
        retrieve: ToolRetriever,
        load: ToolLoader,
        registry: ToolRegistry,
        top_k: int = DEFAULT_FIND_TOOLS_TOP_K,
    ) -> None:
        self._retrieve = retrieve
        self._load = load
        self._registry = registry
        self._top_k = top_k

    @property
    def parameters(self) -> JsonSchema:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "What you want to do, in natural language; the "
                        "capability to find tools for."
                    ),
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        }

    def invoke(self, args: JsonObject) -> ToolResult:
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            raise RecoverableToolError(
                code="invalid_arguments",
                message="find_tools requires a non-empty 'query' string.",
                hint="Pass a short description of the capability you need.",
            )

        slugs = [slug for slug in self._retrieve(query)][: self._top_k]
        loaded = self._load(slugs) if slugs else ()

        available: list[JsonObject] = []
        newly_loaded = 0
        for tool in loaded:
            if self._registry.register_if_absent(tool):
                newly_loaded += 1
            available.append(
                {"name": tool.name, "description": _first_line(tool.description)}
            )

        if not available:
            return ToolResult(
                output={
                    "query": query,
                    "loaded": [],
                    "message": (
                        "No matching tools found for that query. Try rephrasing, "
                        "or answer with the tools you already have."
                    ),
                }
            )

        return ToolResult(
            output={
                "query": query,
                "newly_loaded": newly_loaded,
                "available": available,
                "message": (
                    f"Loaded {len(available)} tool(s); they are callable on your "
                    "next step. Call the one that fits."
                ),
            }
        )


def _first_line(text: str) -> str:
    line = (text or "").strip().splitlines()[0] if text and text.strip() else ""
    return line[:200]
