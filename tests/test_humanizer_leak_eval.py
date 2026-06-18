"""Humanizer leak eval (HIG-203).

Adversarial inputs that smuggle internal scratchpad / planner-role text before
the real answer must be stripped to the Slack-facing answer. Pure: exercises
``strip_internal_response_preamble`` directly, no LLM.
"""

from __future__ import annotations

import pytest

from kortny.slack.humanizer import (
    HUMANIZER_LEAK_MARKERS,
    strip_internal_response_preamble,
)

_FINAL = "Q3 revenue was $1.2M, up 14% from Q2."

LEAKY_INPUTS = (
    (
        "I'm the planned_workflow_merger. I'll present this as Kortny's final "
        f"answer.\n\nFinal answer: {_FINAL}"
    ),
    (
        "Let me write the response. The user is asking for the revenue number.\n"
        f"Here's the final Slack-ready answer:\n{_FINAL}"
    ),
    (
        "According to my guidelines I should keep it short. response_record "
        f"shows the figure.\n\nBottom line: {_FINAL}"
    ),
)


@pytest.mark.parametrize("leaky", LEAKY_INPUTS)
def test_strips_internal_preamble_to_final_answer(leaky: str) -> None:
    cleaned = strip_internal_response_preamble(leaky)
    lowered = cleaned.casefold()
    # No internal marker survives.
    for marker in HUMANIZER_LEAK_MARKERS:
        assert marker not in lowered, f"leak marker survived: {marker!r}"
    # The real answer is preserved.
    assert "1.2M" in cleaned


def test_clean_answer_is_unchanged() -> None:
    assert strip_internal_response_preamble(_FINAL) == _FINAL


@pytest.mark.xfail(
    reason="known gap (HIG-203): a leak preamble followed by a bare sentence "
    "with no recognized final-answer boundary is not stripped — the function "
    "keys on markers/openers. Documented so a future fix flips this green.",
    strict=True,
)
def test_bare_answer_after_preamble_is_a_known_gap() -> None:
    leaky = (
        "According to my guidelines I should keep it short. response_record "
        f"shows the figure.\n\n{_FINAL}"
    )
    cleaned = strip_internal_response_preamble(leaky).casefold()
    assert not any(marker in cleaned for marker in HUMANIZER_LEAK_MARKERS)


def test_empty_stays_empty() -> None:
    assert strip_internal_response_preamble("   ") == ""
