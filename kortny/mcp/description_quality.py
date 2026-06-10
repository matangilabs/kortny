"""MCP tool description quality scoring and LLM enrichment (HIG-215).

Research context: 97% of MCP tool descriptions have quality defects; fixing
them yields +5.85pp task-success improvement. This module scores descriptions
on a deterministic 4-criterion rubric and enriches poor ones with one cheap-tier
LLM call.

Rubric (0.25 each, summing to 1.0):

a) Purpose clarity — description is non-empty, ≥ 40 chars, and the first
   sentence contains at least one verb-like token (word ending in common verb
   suffixes, or an explicit verb keyword) that is not just the tool name.

b) Parameter coverage — every property marked as "required" in the input_schema
   has either a "description" field in the schema's properties OR its name is
   mentioned explicitly in the description text. Full credit when there are no
   required parameters.

c) Limitations stated — the description mentions at least one of a curated set
   of limiting keywords ("only", "cannot", "does not", "limit", "max",
   "requires", "not supported", "must", "only works", "minimum", "maximum").

d) Usage criteria — the description contains at least one phrase that tells the
   agent *when* to use this tool ("use this when", "use this for", "use when",
   "use for", "when you need", "to retrieve", "to create", "to update",
   "to delete", "to list", "to search", "to fetch", "to get", "to send",
   "to post", "to submit").

Enrichment prompt instructs the LLM to produce: one-sentence purpose +
limitations + when-to-use guidance. Usage examples are intentionally excluded
(per research: token cost, no gain). Output is plain text, ≤ 600 chars.
"""

from __future__ import annotations

import hashlib
import logging
import re
import uuid
from collections.abc import Sequence
from typing import Protocol

from kortny.llm.types import ChatMessage, Completion
from kortny.tools.types import JsonObject, JsonSchema

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scoring constants
# ---------------------------------------------------------------------------

_PURPOSE_MIN_CHARS = 40

# Common English verb suffixes (rough heuristic for first-sentence verb detection)
_VERB_SUFFIXES = re.compile(
    r"\b\w+(?:e[sd]?|s|ing|tion|ize[sd]?|ise[sd]?|ate[sd]?|ify|ies|ied)\b",
    re.IGNORECASE,
)
# Explicit verb keywords that clearly indicate action
_VERB_KEYWORDS = frozenset(
    {
        "get",
        "set",
        "list",
        "search",
        "fetch",
        "create",
        "update",
        "delete",
        "remove",
        "add",
        "send",
        "post",
        "read",
        "write",
        "execute",
        "run",
        "check",
        "query",
        "retrieve",
        "find",
        "return",
        "returns",
        "call",
        "invoke",
        "submit",
        "upload",
        "download",
        "parse",
        "generate",
        "convert",
        "transform",
        "validate",
        "compute",
        "calculate",
        "extract",
        "import",
        "export",
        "open",
        "close",
        "start",
        "stop",
        "enable",
        "disable",
        "manage",
        "handle",
        "process",
        "trigger",
        "subscribe",
        "unsubscribe",
        "register",
        "deregister",
        "scan",
        "resolve",
        "lookup",
        "move",
        "copy",
        "rename",
        "replace",
        "insert",
        "append",
        "filter",
        "sort",
        "paginate",
        "load",
        "save",
        "store",
        "collect",
        "analyze",
        "inspect",
        "monitor",
        "log",
        "notify",
        "access",
        "push",
        "pull",
        "merge",
        "split",
        "join",
        "wrap",
        "unwrap",
        "encode",
        "decode",
        "translate",
    }
)

# Keywords suggesting limitations
_LIMITATION_PHRASES = (
    "only",
    "cannot",
    "does not",
    "do not",
    "doesn't",
    "don't",
    "limit",
    "max",
    "requires",
    "not supported",
    "must",
    "only works",
    "minimum",
    "maximum",
    "restricted",
    "unavailable",
    "not available",
    "will not",
    "won't",
    "unable",
    "fails if",
    "error if",
    "throws if",
    "raises if",
)

# Phrases indicating usage criteria / when-to-use guidance
_USAGE_PHRASES = (
    "use this when",
    "use this for",
    "use when",
    "use for",
    "when you need",
    "when you want",
    "to retrieve",
    "to create",
    "to update",
    "to delete",
    "to list",
    "to search",
    "to fetch",
    "to get",
    "to send",
    "to post",
    "to submit",
    "to check",
    "to find",
    "to read",
    "to write",
    "to set",
    "to add",
    "to remove",
    "to run",
    "to execute",
    "to query",
    "to upload",
    "to download",
    "to generate",
    "to convert",
    "to validate",
    "to extract",
    "to import",
    "to export",
    "to manage",
    "to process",
    "to handle",
    "to trigger",
    "to monitor",
    "to access",
    "to load",
    "to save",
    "to store",
    "to analyze",
    "to inspect",
    "to notify",
    "to push",
    "to pull",
    "to merge",
    "to encode",
    "to decode",
    "to translate",
    "to resolve",
    "to lookup",
    "allows you",
    "lets you",
    "enables you",
    "useful for",
    "ideal for",
    "call this",
    "invoke this",
    "call when",
)

# Enrichment prompt
_ENRICHMENT_SYSTEM_PROMPT = """\
You are an expert at writing concise, high-quality tool descriptions for LLM
function-calling schemas.

Given a tool name, its original description, and its input schema, produce an
improved description that includes:
1. A single opening sentence that clearly states what the tool does.
2. Key limitations or constraints (if any are implied by the schema or name).
3. A short "when to use" clause — e.g. "Use this when you need to ...".

Rules:
- Output plain text only. No markdown. No bullet points.
- Keep it under 600 characters total.
- Do NOT include usage examples.
- Do NOT repeat the tool name as the entire description.
- Write in the third-person imperative style, e.g. "Retrieves ...", "Creates ...",
  "Searches ...".
""".strip()

_ENRICHMENT_PROMPT_NAME = "kortny.mcp_description_enricher"


# ---------------------------------------------------------------------------
# Protocol for the LLM client accepted by enrich_tool_description
# ---------------------------------------------------------------------------


class DescriptionEnricherLLMClient(Protocol):
    """Subset of LLMService used by the description enricher."""

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
        """Complete one enrichment turn."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def score_tool_description(
    name: str,
    description: str,
    input_schema: object,
) -> float:
    """Deterministic quality rubric for one MCP tool description.

    Returns a float in [0.0, 1.0] (increments of 0.25).

    Criteria:
      a) purpose_clarity  — non-empty, ≥ 40 chars, first sentence has a verb.
      b) parameter_coverage — required params have descriptions or are mentioned.
      c) limitations_stated — limiting keyword present.
      d) usage_criteria  — "when/use for" style phrase present.
    """
    score = 0.0
    desc_lower = description.casefold()

    # (a) Purpose clarity
    if _passes_purpose_clarity(name, description, desc_lower):
        score += 0.25

    # (b) Parameter coverage
    if _passes_parameter_coverage(description, desc_lower, input_schema):
        score += 0.25

    # (c) Limitations stated
    if any(phrase in desc_lower for phrase in _LIMITATION_PHRASES):
        score += 0.25

    # (d) Usage criteria
    if any(phrase in desc_lower for phrase in _USAGE_PHRASES):
        score += 0.25

    return round(score, 3)


def enrich_tool_description(
    llm: DescriptionEnricherLLMClient,
    *,
    name: str,
    description: str,
    input_schema: object,
    task_id: uuid.UUID | None = None,
) -> str | None:
    """Call the cheap LLM tier to produce an improved tool description.

    Returns the improved plain-text description (≤ 600 chars), or None if the
    call fails or returns empty content.  All calls go through LLMService so
    usage and cost are recorded.

    Args:
        llm: An LLMService-compatible client (must have .complete(...)).
        name: Tool name.
        description: Original raw description.
        input_schema: The tool's JSON input schema.
        task_id: UUID to attribute the LLM call to. A synthetic UUID is used
            when None is passed (e.g. during dashboard discovery).
    """
    effective_task_id = task_id or uuid.uuid4()
    schema_summary = _summarise_schema(input_schema)
    user_content = (
        f"Tool name: {name}\n"
        f"Original description: {description or '(empty)'}\n"
        f"Input schema summary: {schema_summary}"
    )
    messages: list[ChatMessage] = [
        ChatMessage(role="system", content=_ENRICHMENT_SYSTEM_PROMPT),
        ChatMessage(role="user", content=user_content),
    ]
    try:
        completion = llm.complete(
            task_id=effective_task_id,
            messages=messages,
            prompt_name=_ENRICHMENT_PROMPT_NAME,
            prompt_source="code",
        )
        content = (completion.content or "").strip()
        if not content:
            return None
        # Truncate to 600 chars as a hard safety net
        return content[:600]
    except Exception:
        logger.exception(
            "mcp_description_enrichment_failed",
            extra={"tool_name": name},
        )
        return None


def sha256_of_description(description: str) -> str:
    """Return the hex-encoded SHA-256 of the raw description string."""
    return hashlib.sha256(description.encode("utf-8", errors="replace")).hexdigest()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _passes_purpose_clarity(name: str, description: str, desc_lower: str) -> bool:
    """Return True if the description satisfies the purpose clarity criterion."""
    if not description or len(description) < _PURPOSE_MIN_CHARS:
        return False
    # Extract first sentence (split on '. ', '.\n', or '!')
    first_sentence = re.split(r"\.\s+|\.\n|!", description, maxsplit=1)[0]
    fs_lower = first_sentence.casefold()
    # Check the first sentence is not just the tool name
    if fs_lower.strip().replace("_", " ").replace(
        "-", " "
    ) == name.casefold().strip().replace("_", " ").replace("-", " "):
        return False
    # Must contain at least one verb-like word
    words = re.findall(r"\b\w+\b", fs_lower)
    for word in words:
        if word in _VERB_KEYWORDS:
            return True
    # Fallback: look for suffix-based verb patterns
    return bool(_VERB_SUFFIXES.search(first_sentence))


def _passes_parameter_coverage(
    description: str,
    desc_lower: str,
    input_schema: object,
) -> bool:
    """Return True if required parameters are described or mentioned."""
    if not isinstance(input_schema, dict):
        return True  # No schema → full credit
    required = input_schema.get("required")
    if not isinstance(required, list) or not required:
        return True  # No required params → full credit
    properties = input_schema.get("properties")
    for param_name in required:
        if not isinstance(param_name, str):
            continue
        # Credit if the param's own schema has a "description" field
        if isinstance(properties, dict):
            param_schema = properties.get(param_name)
            if isinstance(param_schema, dict) and param_schema.get("description"):
                continue  # This param is covered in schema
        # Credit if the param name appears in the description text
        if param_name.casefold() in desc_lower:
            continue
        # This required param is not covered at all
        return False
    return True


def _summarise_schema(input_schema: object) -> str:
    """Produce a compact human-readable summary of the input schema."""
    if not isinstance(input_schema, dict):
        return "(no schema)"
    properties = input_schema.get("properties")
    required = input_schema.get("required") or []
    if not isinstance(properties, dict) or not properties:
        return "(no parameters)"
    parts: list[str] = []
    for prop_name, prop_schema in properties.items():
        prop_type = "unknown"
        if isinstance(prop_schema, dict):
            prop_type = str(prop_schema.get("type", "unknown"))
        req = " [required]" if prop_name in required else ""
        parts.append(f"  {prop_name}: {prop_type}{req}")
    return "{\n" + "\n".join(parts) + "\n}"


__all__ = [
    "DescriptionEnricherLLMClient",
    "enrich_tool_description",
    "score_tool_description",
    "sha256_of_description",
]
