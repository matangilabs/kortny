"""Labeled intent-classification cases (HIG-203).

Each case is a realistic Slack message + the IntentClassification it should map
to. Covers the high-stakes failure modes: soft mentions that must NOT trigger,
memory requests vs tasks, cancel/retry phrasing, third-person references, and
clarifications. Expand from anonymized real history over time; this seed is the
floor the classifier prompt is held to.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from kortny.intent.models import IntentClassification, IntentSurface


@dataclass(frozen=True, slots=True)
class IntentCase:
    text: str
    surface: IntentSurface
    expected: IntentClassification
    is_thread_follow_up: bool = False
    connected_integrations: tuple[str, ...] = ()
    note: str = ""
    tags: tuple[str, ...] = field(default=())


_C = IntentClassification
_APP = IntentSurface.app_mention
_DM = IntentSurface.dm
_CHAN = IntentSurface.channel_message

SEED_INTENT_CASES: tuple[IntentCase, ...] = (
    # --- direct task requests ---
    IntentCase(
        "Kortny, summarize this thread and post the key decisions.",
        _APP,
        _C.task_request,
        tags=("task",),
    ),
    IntentCase(
        "what's on my plate today?",
        _DM,
        _C.task_request,
        connected_integrations=("linear",),
        tags=("task", "grounding"),
    ),
    IntentCase(
        "research our top 3 competitors and write a one-pager with sources",
        _APP,
        _C.task_request,
        tags=("task", "research"),
    ),
    IntentCase(
        "can you pull last month's Stripe payouts and total them?",
        _DM,
        _C.task_request,
        connected_integrations=("stripe",),
        tags=("task",),
    ),
    # --- soft mentions that should NOT trigger a task ---
    IntentCase(
        "honestly kortny has been super helpful this week",
        _CHAN,
        _C.ambient_observation,
        note="praise about Kortny, not a request",
        tags=("no_trigger",),
    ),
    IntentCase(
        "we should probably ask kortny to help with the rollout at some point",
        _CHAN,
        _C.third_person_reference,
        note="talking about Kortny in third person, not addressing it",
        tags=("no_trigger", "third_person"),
    ),
    IntentCase(
        "lol same",
        _CHAN,
        _C.ignore,
        note="chatter, unrelated",
        tags=("no_trigger",),
    ),
    IntentCase(
        "the deploy finished, all green",
        _CHAN,
        _C.ambient_observation,
        note="status chatter, no ask",
        tags=("no_trigger",),
    ),
    # --- memory ---
    IntentCase(
        "Kortny, remember that our fiscal year starts in April",
        _APP,
        _C.memory_candidate,
        tags=("memory",),
    ),
    IntentCase(
        "from now on always use British spelling in docs",
        _DM,
        _C.memory_candidate,
        note="durable preference",
        tags=("memory",),
    ),
    # --- cancel / retry ---
    IntentCase(
        "Kortny stop, cancel that",
        _APP,
        _C.cancel_or_retry,
        tags=("control",),
    ),
    IntentCase(
        "that's wrong, try again",
        _APP,
        _C.cancel_or_retry,
        is_thread_follow_up=True,
        tags=("control",),
    ),
    # --- clarification ---
    IntentCase(
        "which channel did you mean?",
        _DM,
        _C.clarification,
        is_thread_follow_up=True,
        tags=("clarify",),
    ),
    # --- follow-up ---
    IntentCase(
        "yes do that, and also add the Q3 numbers",
        _APP,
        _C.follow_up,
        is_thread_follow_up=True,
        tags=("follow_up",),
    ),
    IntentCase(
        "perfect, now make it a PDF",
        _DM,
        _C.follow_up,
        is_thread_follow_up=True,
        tags=("follow_up",),
    ),
)
