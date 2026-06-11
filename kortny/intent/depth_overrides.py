"""Deterministic response-depth overrides for the unified router (HIG-218).

These overrides run *after* the LLM intent decision in the intent classifier.
They port the battle-tested regex/word-list heuristics from the retired
observe-only planned workflow classifier so that high-signal requests get a
deterministic, auditable depth even when the model under- or over-classifies.

The overrides only ever force a depth; they set ``depth_source`` to
``"deterministic_override"`` so the trace -> eval loop can tell model decisions
apart from rule decisions.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from kortny.intent.models import IntentDecision, IntentRequest, ResponseDepth
from kortny.schedule_intent import is_schedule_state_question


@dataclass(frozen=True, slots=True)
class DepthOverride:
    """Result of applying deterministic depth overrides."""

    depth: ResponseDepth
    reason_codes: tuple[str, ...]


def apply_depth_overrides(
    request: IntentRequest,
    decision: IntentDecision,
) -> IntentDecision:
    """Return ``decision`` with a deterministic depth forced when warranted.

    When no deterministic signal fires, the model-authored depth is preserved
    and ``depth_source`` is set to ``"llm"``. When an override fires, the depth
    is forced and ``depth_source`` becomes ``"deterministic_override"``.
    """

    override = classify_depth_override(
        text=request.text,
        likely_tools=decision.routing_likely_tools(),
        is_scheduled_identity=False,
    )
    if override is None:
        return decision.model_copy(update={"depth_source": "llm"})
    return decision.model_copy(
        update={
            "response_depth": override.depth,
            "depth_source": "deterministic_override",
        }
    )


def classify_depth_override(
    *,
    text: str,
    likely_tools: Sequence[str] = (),
    is_scheduled_identity: bool = False,
) -> DepthOverride | None:
    """Return a forced depth for high-signal requests, or ``None``.

    ``is_scheduled_identity`` carries the equivalent of the retired classifier's
    ``scheduled_task_identity`` reason code; ingress messages never have it, but
    callers that route non-ingress tasks may pass it.
    """

    normalized = _normalize(text)

    # Force quick_response for the lightweight conversational paths.
    if is_schedule_state_question(normalized):
        return DepthOverride("quick_response", ("schedule_state_query",))
    if _is_capability_lookup(normalized, likely_tools):
        return DepthOverride("quick_response", ("capability_lookup",))
    if _is_quick_conversation(normalized):
        return DepthOverride("quick_response", ("quick_conversation",))

    # Force deep_workflow when hard planning signals are present.
    deep_reasons = _deep_reason_codes(
        normalized=normalized,
        likely_tools=likely_tools,
        is_scheduled_identity=is_scheduled_identity,
    )
    if deep_reasons == ("write_or_destructive_intent",):
        # A lone everyday write verb ("remove the part about...", "post this")
        # is not a planning signal by itself — forcing the planner + parallel
        # branches on short follow-ups added minutes of latency and an Opus
        # call. The verb still matters for approvals; depth stays with the LLM.
        return None
    if deep_reasons:
        return DepthOverride("deep_workflow", deep_reasons)

    return None


def _deep_reason_codes(
    *,
    normalized: str,
    likely_tools: Sequence[str],
    is_scheduled_identity: bool,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if is_scheduled_identity:
        reasons.append("scheduled_task_identity")
    if _SCHEDULE_RE.search(normalized):
        reasons.append("scheduled_or_recurring")
    if _WRITE_OR_DESTRUCTIVE_RE.search(normalized):
        reasons.append("write_or_destructive_intent")
    if _LONG_RUNNING_RE.search(normalized):
        reasons.append("long_running_or_monitoring")
    if len(normalized) >= 420:
        reasons.append("large_request")
    if len({tool for tool in likely_tools if tool}) >= 2:
        reasons.append("multi_tool_likely")

    detected_integrations = _detected_integrations(normalized, likely_tools)
    if len(detected_integrations) >= 2:
        reasons.append("multi_integration_scope")

    broad_research = bool(_BROAD_RESEARCH_RE.search(normalized))
    multi_source = bool(_MULTI_SOURCE_RE.search(normalized))
    if broad_research and multi_source:
        reasons.append("research_synthesis_work")

    return _dedupe(reasons)


def _detected_integrations(
    normalized: str,
    likely_tools: Sequence[str],
) -> tuple[str, ...]:
    integrations: list[str] = []
    for name, pattern in _INTEGRATION_PATTERNS.items():
        if pattern.search(normalized):
            integrations.append(name)
    for tool in likely_tools:
        if tool.startswith("composio_"):
            integrations.append(tool.removeprefix("composio_"))
    return _dedupe(integrations)


def _is_quick_conversation(normalized: str) -> bool:
    if len(normalized) > 140:
        return False
    return bool(_QUICK_CONVERSATION_RE.fullmatch(normalized))


def _is_capability_lookup(
    normalized: str,
    likely_tools: Sequence[str],
) -> bool:
    if not _CAPABILITY_LOOKUP_RE.fullmatch(normalized):
        return False
    if not likely_tools:
        return True
    return set(likely_tools) <= CAPABILITY_LOOKUP_TOOL_HINTS


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip()


def _dedupe(values: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return tuple(result)


_QUICK_CONVERSATION_RE = re.compile(
    r"(?:yo\s+)?(?:hey\s+)?(?:kortny\s+)?(?:are you up|you up|ping|"
    r"what'?s up|what can you do|what tools do you have(?: access to)?|"
    r"what integrations do you have)\??"
)
_CAPABILITY_LOOKUP_RE = re.compile(
    r"(?:yo\s+)?(?:hey\s+)?(?:kortny\s+)?(?:what can you do|"
    r"what tools do you have(?: access to)?|what integrations do you have|"
    r"what capabilities do you have)\??"
)
CAPABILITY_LOOKUP_TOOL_HINTS = frozenset(
    {
        "capability_lookup",
        "get_capabilities",
        "describe_tools",
        "list_capabilities",
        "list_integrations",
        "list_tools",
        "native_tool_registry",
        "tool_metadata_lookup",
        "tool_registry",
    }
)
_SCHEDULE_RE = re.compile(
    r"\b(every|daily|weekly|monthly|schedule|scheduled|recurring|cron|"
    r"each\s+(?:day|week|month|monday|tuesday|wednesday|thursday|friday))\b"
)
_WRITE_OR_DESTRUCTIVE_RE = re.compile(
    r"\b(create|update|delete|remove|archive|send|post|publish|upload|write|"
    r"disconnect|revoke|approve|trigger|execute|run|generate)\b"
)
_BROAD_RESEARCH_RE = re.compile(
    r"\b(research|investigate|deep dive|look up|find current|latest|recent|"
    r"market map|landscape|benchmark|best .* tools?|compare .* tools?)\b"
)
_MULTI_SOURCE_RE = re.compile(
    r"\b(compare|synthesize|synthesise|cross[- ]check|triangulate|combine|"
    r"against|with our docs|from multiple|across (?:multiple|all)|using .* and .*)\b"
)
_LONG_RUNNING_RE = re.compile(
    r"\b(long[- ]running|comprehensive|full report|detailed report|monitor|"
    r"keep checking|over the next|background|until|when .* finishes)\b"
)
_INTEGRATION_PATTERNS = {
    "linear": re.compile(r"\blinear\b"),
    "notion": re.compile(r"\bnotion\b"),
    "slack": re.compile(r"\bslack|this channel|channel context\b"),
    "firecrawl": re.compile(r"\bfirecrawl|crawl|scrape\b"),
    "serpapi": re.compile(r"\bserpapi\b"),
    "alpha_vantage": re.compile(r"\balpha vantage|ticker|market data\b"),
    "github": re.compile(r"\bgithub|git hub|pr|pull request\b"),
    "docs": re.compile(r"\bdocs?|documentation\b"),
    "calendar": re.compile(r"\bcalendar|meeting\b"),
    "email": re.compile(r"\bemail|gmail|inbox\b"),
}
