"""Deterministic persona-relevance gate (HIG-277).

Persona framing only helps role-relative asks and reliably damages factual
knowledge retrieval (PRISM). So persona is injected only when a request is
persona-relevant. This module is the cheap, deterministic half of that gate —
first/second-person possessive over a work surface, or a canonical "my
plate / what should I focus on" phrasing. It fails safe to False (neutral path);
the LLM classifier may additionally set ``persona_relevant`` to catch cases the
heuristic misses.
"""

from __future__ import annotations

import re

# Canonical role-relative phrasings.
_PHRASES = (
    "my plate",
    "on my plate",
    "my work",
    "my tasks",
    "my queue",
    "my stuff",
    "assigned to me",
    "for me to",
    "what should i focus on",
    "what should i work on",
    "what do i work on",
    "what's on my",
    "whats on my",
    "what is on my",
    "what i should",
)

# "my <work surface>" — possessive over a surface noun the persona maps to.
_SURFACE_NOUNS = (
    "issues",
    "tickets",
    "prs",
    "pull requests",
    "pull-requests",
    "reviews",
    "inbox",
    "email",
    "emails",
    "calendar",
    "meetings",
    "schedule",
    "pipeline",
    "deals",
    "accounts",
    "docs",
    "designs",
    "dashboards",
)
# "my [adjective ...] <surface>" — allow up to two words between the possessive
# and the surface noun ("my open PRs", "my assigned issues", "my upcoming meetings").
_POSSESSIVE_SURFACE = re.compile(
    r"\bmy\s+(?:\w+\s+){0,2}(?:"
    + "|".join(re.escape(n) for n in _SURFACE_NOUNS)
    + r")\b"
)


def persona_relevant_for_text(text: str) -> bool:
    """Whether ``text`` is a role-relative ask where persona framing helps."""

    if not text:
        return False
    lowered = text.casefold()
    if any(phrase in lowered for phrase in _PHRASES):
        return True
    return bool(_POSSESSIVE_SURFACE.search(lowered))


__all__ = ["persona_relevant_for_text"]
