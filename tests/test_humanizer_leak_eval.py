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


def test_bare_answer_after_preamble_is_stripped() -> None:
    # HIG-255 leak-gap fix: a leak preamble followed by a bare answer sentence
    # with no recognized boundary marker must still be stripped to the answer.
    leaky = (
        "According to my guidelines I should keep it short. response_record "
        f"shows the figure.\n\n{_FINAL}"
    )
    cleaned = strip_internal_response_preamble(leaky)
    assert not any(marker in cleaned.casefold() for marker in HUMANIZER_LEAK_MARKERS)
    assert "1.2M" in cleaned


def test_bare_answer_keeps_contiguous_clean_tail() -> None:
    # The whole clean tail survives, not just the last paragraph.
    leaky = (
        "Let me write the response. The user is asking for the numbers.\n\n"
        "Q3 revenue was $1.2M, up 14% from Q2.\n\n"
        "Margins held steady at 38%."
    )
    cleaned = strip_internal_response_preamble(leaky)
    assert not any(marker in cleaned.casefold() for marker in HUMANIZER_LEAK_MARKERS)
    assert "1.2M" in cleaned
    assert "Margins held steady" in cleaned


def test_preamble_with_only_a_bare_ack_is_not_treated_as_answer() -> None:
    # A clean tail of only "Done." is not substantive; don't promote it. The
    # function returns raw (the caller's safety net handles a leak-only output).
    leaky = "I should keep it short. response_record shows the figure.\n\nDone."
    cleaned = strip_internal_response_preamble(leaky)
    assert cleaned == leaky.strip()


def test_empty_stays_empty() -> None:
    assert strip_internal_response_preamble("   ") == ""
