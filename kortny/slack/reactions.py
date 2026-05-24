"""Slack reaction selection for lightweight coworker presence."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol

from kortny.intent import IntentClassification, IntentDecision

COMPLETED_REACTION = "heavy_check_mark"
FAILED_REACTION = "warning"
ACK_REACTION_ADDED_MESSAGE = "slack_ack_reaction_added"
ACK_REACTION_ADD_FAILED_MESSAGE = "slack_ack_reaction_failed"
ACK_REACTION_UNAVAILABLE_MESSAGE = "slack_ack_reaction_unavailable"
ACK_REACTION_REMOVED_MESSAGE = "slack_ack_reaction_removed"
ACK_REACTION_REMOVE_FAILED_MESSAGE = "slack_ack_reaction_remove_failed"
COMPLETION_REACTION_ADDED_MESSAGE = "slack_completion_reaction_added"
COMPLETION_REACTION_FAILED_MESSAGE = "slack_completion_reaction_failed"


@dataclass(frozen=True, slots=True)
class ReactionChoice:
    """A selected Slack reaction and its coarse intent label."""

    name: str
    intent: str


class ReactionProvider(Protocol):
    """Selects Slack reactions for task lifecycle feedback."""

    def acknowledgement_reaction(
        self,
        *,
        input_text: str,
        source: str,
        intent_decision: IntentDecision | None = None,
    ) -> ReactionChoice:
        """Return the reaction to add when a Slack task is accepted."""

    def completion_reaction(
        self, *, input_text: str, source: str, succeeded: bool
    ) -> ReactionChoice:
        """Return the reaction to add when a Slack task finishes."""


@dataclass(frozen=True, slots=True)
class ReactionBucket:
    """Curated reaction candidates for one coarse task intent."""

    intent: str
    names: tuple[str, ...]
    keywords: tuple[str, ...] = ()


class LibraryReactionProvider:
    """Fast curated reaction selector with deterministic variation."""

    buckets: tuple[ReactionBucket, ...] = (
        ReactionBucket(
            "memory",
            (
                "memo",
                "bookmark",
                "pushpin",
                "label",
                "spiral_note_pad",
                "card_index_dividers",
            ),
            (
                "remember",
                "keep in mind",
                "from now on",
                "going forward",
                "preference",
                "note that",
            ),
        ),
        ReactionBucket(
            "creation",
            (
                "page_facing_up",
                "paperclip",
                "open_file_folder",
                "writing_hand",
                "art",
                "hammer_and_wrench",
            ),
            (
                "create",
                "draft",
                "generate",
                "write",
                "make",
                "turn into",
                "document",
                "report",
                "pdf",
                "deck",
                "slides",
                "file",
            ),
        ),
        ReactionBucket(
            "discovery",
            (
                "mag",
                "newspaper",
                "compass",
                "bulb",
                "dart",
                "satellite",
            ),
            (
                "research",
                "search",
                "find",
                "look up",
                "source",
                "sources",
                "latest",
                "recent",
                "crawl",
            ),
        ),
        ReactionBucket(
            "review",
            (
                "thinking_face",
                "bar_chart",
                "clipboard",
                "mag_right",
                "brain",
                "memo",
            ),
            (
                "analyze",
                "analyse",
                "review",
                "summarize",
                "summarise",
                "compare",
                "explain",
                "check",
                "audit",
                "evaluate",
            ),
        ),
        ReactionBucket(
            "social_presence",
            (
                "wave",
                "sparkles",
                "raised_hands",
                "tada",
                "star",
                "handshake",
                "clap",
                "smile",
            ),
            (
                "coworker",
                "team",
                "welcome",
                "introduced",
                "good work",
                "help us",
                "helping",
            ),
        ),
        ReactionBucket(
            "working",
            (
                "eyes",
                "hourglass_flowing_sand",
                "speech_balloon",
                "gear",
                "zap",
                "hourglass",
            ),
        ),
    )

    def acknowledgement_reaction(
        self,
        *,
        input_text: str,
        source: str,
        intent_decision: IntentDecision | None = None,
    ) -> ReactionChoice:
        intent_choice = _choice_from_intent_decision(
            input_text=input_text,
            source=source,
            buckets=self.buckets,
            decision=intent_decision,
        )
        if intent_choice is not None:
            return intent_choice

        bucket = _intent_bucket(input_text, self.buckets)
        index = _stable_index(f"{source}\n{input_text}", len(bucket.names))
        return ReactionChoice(name=bucket.names[index], intent=bucket.intent)

    def completion_reaction(
        self, *, input_text: str, source: str, succeeded: bool
    ) -> ReactionChoice:
        del input_text, source
        if succeeded:
            return ReactionChoice(name=COMPLETED_REACTION, intent="completed")
        return ReactionChoice(name=FAILED_REACTION, intent="failed")


def _intent_bucket(
    input_text: str,
    buckets: tuple[ReactionBucket, ...],
) -> ReactionBucket:
    normalized = _normalize(input_text)
    default = buckets[-1]
    for bucket in buckets:
        if not bucket.keywords:
            default = bucket
            continue
        if any(_normalize(keyword) in normalized for keyword in bucket.keywords):
            return bucket
    return default


def _choice_from_intent_decision(
    *,
    input_text: str,
    source: str,
    buckets: tuple[ReactionBucket, ...],
    decision: IntentDecision | None,
) -> ReactionChoice | None:
    if decision is None:
        return None

    suggested = decision.suggested_reaction
    if suggested in APPROVED_ACK_REACTIONS:
        return ReactionChoice(name=suggested, intent=decision.classification.value)

    mapped = REACTION_BY_CLASSIFICATION.get(decision.classification)
    if mapped is not None:
        return ReactionChoice(name=mapped, intent=decision.classification.value)

    bucket_name = REACTION_BUCKET_BY_CLASSIFICATION.get(decision.classification)
    if bucket_name is None:
        return None
    bucket = _bucket_by_intent(bucket_name, buckets)
    if bucket is None:
        return None
    index = _stable_index(
        f"{source}\n{decision.classification.value}\n{input_text}",
        len(bucket.names),
    )
    return ReactionChoice(
        name=bucket.names[index],
        intent=decision.classification.value,
    )


def _bucket_by_intent(
    intent: str,
    buckets: tuple[ReactionBucket, ...],
) -> ReactionBucket | None:
    for bucket in buckets:
        if bucket.intent == intent:
            return bucket
    return None


def _normalize(value: str) -> str:
    return " ".join(value.casefold().split())


def _stable_index(value: str, modulo: int) -> int:
    if modulo < 1:
        raise ValueError("modulo must be at least 1")
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % modulo


REACTION_EXTRA_ALLOWLIST = frozenset(
    {
        "arrows_counterclockwise",
        COMPLETED_REACTION,
        FAILED_REACTION,
    }
)
APPROVED_ACK_REACTIONS = (
    frozenset(
        {name for bucket in LibraryReactionProvider.buckets for name in bucket.names}
    )
    | REACTION_EXTRA_ALLOWLIST
)
REACTION_BY_CLASSIFICATION = {
    IntentClassification.task_request: "eyes",
    IntentClassification.follow_up: "speech_balloon",
    IntentClassification.memory_candidate: "memo",
    IntentClassification.clarification: "thinking_face",
    IntentClassification.cancel_or_retry: "arrows_counterclockwise",
}
REACTION_BUCKET_BY_CLASSIFICATION = {
    IntentClassification.third_person_reference: "social_presence",
}
