"""Small import-safe helpers for schedule-related task routing."""

from __future__ import annotations

import re


def normalize_schedule_intent_text(value: str) -> str:
    """Normalize Slack text before schedule intent checks."""

    return re.sub(r"\s+", " ", value.casefold()).strip()


def is_schedule_state_question(value: str) -> bool:
    """Return true when a user is asking about existing scheduler state."""

    text = normalize_schedule_intent_text(value)
    if not _SCHEDULE_REF_RE.search(text):
        return False
    return bool(_SCHEDULE_STATE_RE.search(text))


def schedule_state_query_text(value: str) -> str | None:
    """Extract a loose scheduler search phrase from a state question."""

    text = normalize_schedule_intent_text(value)
    text = _SCHEDULE_STATE_RE.sub(" ", text)
    text = _SCHEDULE_REF_RE.sub(" ", text)
    text = _FILLER_RE.sub(" ", text)
    normalized = re.sub(r"\s+", " ", text).strip(" ?.!,")
    return normalized or None


def schedule_state_status_filter(value: str) -> str:
    """Infer the list_schedules status filter from a schedule state question."""

    text = normalize_schedule_intent_text(value)
    if re.search(r"\bactive|running|enabled\b", text):
        return "active"
    if re.search(r"\bpaused|disabled|stopped\b", text):
        return "paused"
    if re.search(r"\bproposed|draft|waiting|unconfirmed\b", text):
        return "proposed"
    if re.search(r"\bcancelled|canceled|completed|inactive\b", text):
        return "all"
    return "open"


_SCHEDULE_REF_RE = re.compile(r"\b(schedule|scheduled|schedules|recurring|cron)\b")
_SCHEDULE_STATE_RE = re.compile(
    r"\b(do i have|have i got|is there|are there|what(?:'s| is)|which|"
    r"show|list|tell me|check|active|paused|running|enabled|disabled|"
    r"next run|last run|when (?:does|will)|where (?:does|will)|"
    r"delivery|deliver|delivers|status)\b"
)
_FILLER_RE = re.compile(
    r"\b(a|an|the|my|me|for|of|about|any|existing|currently|right now|"
    r"please|can you|could you|would you|whether|if)\b"
)
