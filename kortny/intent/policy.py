"""Deterministic intent prefilters and action policy."""

from __future__ import annotations

import re
from collections.abc import Mapping

from kortny.intent.models import IntentClassification, IntentDecision

SOFT_MENTION_TASK_CONFIDENCE = 0.85
SOFT_MENTION_REACTION_CONFIDENCE = 0.85
TASK_CREATING_CLASSIFICATIONS = frozenset(
    {
        IntentClassification.task_request,
        IntentClassification.follow_up,
        IntentClassification.memory_candidate,
    }
)
REACTION_ONLY_CLASSIFICATIONS = frozenset(
    {
        IntentClassification.third_person_reference,
    }
)


def contains_app_name(text: str, *, app_name: str) -> bool:
    """Return true when app_name appears as a standalone text token."""

    normalized_app_name = app_name.strip()
    if not normalized_app_name:
        return False
    pattern = re.compile(
        rf"(?<![A-Za-z0-9_]){re.escape(normalized_app_name)}(?![A-Za-z0-9_])",
        re.IGNORECASE,
    )
    return pattern.search(text) is not None


def should_classify_channel_message(
    event: Mapping[str, object],
    *,
    app_name: str,
) -> bool:
    """Deterministic candidate filter for future soft-name mention handling."""

    if event.get("type") != "message":
        return False
    if event.get("channel_type") == "im":
        return False
    if _optional_nonempty_str(event.get("bot_id")) is not None:
        return False
    subtype = _optional_nonempty_str(event.get("subtype"))
    if subtype is not None:
        return False
    text = _optional_nonempty_str(event.get("text"))
    if text is None:
        return False
    return contains_app_name(text, app_name=app_name)


def should_create_task_from_soft_mention(
    decision: IntentDecision,
    *,
    threshold: float = SOFT_MENTION_TASK_CONFIDENCE,
) -> bool:
    """Fail-closed policy for bare app-name mentions in channels."""

    return (
        decision.addressed_to_kortny
        and decision.classification in TASK_CREATING_CLASSIFICATIONS
        and decision.confidence >= threshold
        and decision.should_create_task
    )


def should_react_to_rejected_soft_mention(
    decision: IntentDecision,
    *,
    threshold: float = SOFT_MENTION_REACTION_CONFIDENCE,
) -> bool:
    """Return true for high-confidence no-task messages worth a quiet reaction."""

    return (
        not decision.addressed_to_kortny
        and decision.classification in REACTION_ONLY_CLASSIFICATIONS
        and decision.confidence >= threshold
        and not decision.should_create_task
        and decision.should_ack_with_reaction
    )


def _optional_nonempty_str(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None
