"""Humanizer leak eval (HIG-203 + fence-leak fix).

Adversarial inputs that smuggle internal scratchpad / planner-role text before
the real answer must be stripped to the Slack-facing answer. Pure: exercises
``strip_internal_response_preamble``, ``_strip_json_code_fence``,
``_json_message``, ``_parse_presentation_hint``, and
``sanitize_humanized_response`` directly, no LLM.
"""

from __future__ import annotations

import pytest

from kortny.slack.humanizer import (
    HUMANIZER_LEAK_MARKERS,
    _json_message,
    _looks_like_raw_humanizer_json,
    _parse_presentation_hint,
    _strip_json_code_fence,
    sanitize_humanized_response,
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


# --------------------------------------------------------------------------- #
# _strip_json_code_fence
# --------------------------------------------------------------------------- #

_INNER_JSON = '{"message": "Hello there", "presentation": {}}'


def test_strip_fence_json_tagged() -> None:
    fenced = f"```json\n{_INNER_JSON}\n```"
    assert _strip_json_code_fence(fenced).strip() == _INNER_JSON


def test_strip_fence_untagged() -> None:
    fenced = f"```\n{_INNER_JSON}\n```"
    assert _strip_json_code_fence(fenced).strip() == _INNER_JSON


def test_strip_fence_plain_json_unchanged() -> None:
    assert _strip_json_code_fence(_INNER_JSON) == _INNER_JSON


def test_strip_fence_prose_unchanged() -> None:
    prose = "Here is the answer to your question."
    assert _strip_json_code_fence(prose) == prose


def test_strip_fence_leading_whitespace() -> None:
    fenced = f"  ```json\n{_INNER_JSON}\n```  "
    result = _strip_json_code_fence(fenced).strip()
    assert result == _INNER_JSON


# --------------------------------------------------------------------------- #
# _json_message / sanitize_humanized_response — fenced payload
# --------------------------------------------------------------------------- #

# The exact real-leak payload: a code-fenced humanizer JSON blob.
_FENCED_PAYLOAD_WITH_PRESENTATION = (
    "```json\n"
    '{"message": "Hello there", "presentation": {"version": 1, "elements": '
    '[{"type": "fields", "items": [{"label": "Status", "value": "Active"}]}]}}\n'
    "```"
)

_FENCED_PAYLOAD_SIMPLE = '```json\n{"message": "Hello there"}\n```'
_UNFENCED_PAYLOAD = '{"message": "hi"}'


def test_json_message_fenced_returns_message() -> None:
    # The core fence-leak fix: fenced JSON must yield the message string.
    assert _json_message(_FENCED_PAYLOAD_SIMPLE) == "Hello there"


def test_json_message_fenced_with_presentation_returns_message() -> None:
    # The exact real-leak payload with a fields/context presentation.
    assert _json_message(_FENCED_PAYLOAD_WITH_PRESENTATION) == "Hello there"


def test_json_message_unfenced_still_works() -> None:
    # Regression: unfenced JSON must still parse correctly.
    assert _json_message(_UNFENCED_PAYLOAD) == "hi"


def test_sanitize_fenced_payload_returns_message_text() -> None:
    # End-to-end: a fenced humanizer blob should yield the prose, not the fence.
    result = sanitize_humanized_response(
        _FENCED_PAYLOAD_WITH_PRESENTATION, fallback="raw fallback"
    )
    assert result == "Hello there"
    assert "```" not in result
    assert '"message"' not in result


def test_sanitize_unfenced_json_returns_message_text() -> None:
    result = sanitize_humanized_response(_UNFENCED_PAYLOAD, fallback="raw fallback")
    assert result == "hi"


def test_sanitize_plain_prose_unchanged() -> None:
    prose = "Here is your answer."
    assert sanitize_humanized_response(prose, fallback="raw fallback") == prose


# --------------------------------------------------------------------------- #
# _parse_presentation_hint — fenced payload
# --------------------------------------------------------------------------- #


def test_parse_presentation_hint_fenced_returns_hint() -> None:
    hint = _parse_presentation_hint(_FENCED_PAYLOAD_WITH_PRESENTATION)
    assert hint is not None
    assert len(hint.elements) == 1


def test_parse_presentation_hint_unfenced_still_works() -> None:
    payload = (
        '{"message": "ok", "presentation": {"version": 1, "elements": '
        '[{"type": "context", "items": ["note"]}]}}'
    )
    hint = _parse_presentation_hint(payload)
    assert hint is not None


def test_parse_presentation_hint_none_input_returns_none() -> None:
    assert _parse_presentation_hint(None) is None


# --------------------------------------------------------------------------- #
# Defense-in-depth: _looks_like_raw_humanizer_json + fallback in sanitize
# --------------------------------------------------------------------------- #


def test_looks_like_raw_humanizer_json_bare() -> None:
    assert _looks_like_raw_humanizer_json('{"message": "hi"}')


def test_looks_like_raw_humanizer_json_fenced() -> None:
    assert _looks_like_raw_humanizer_json('```json\n{"presentation": {}}\n```')


def test_looks_like_raw_humanizer_json_plain_prose_false() -> None:
    assert not _looks_like_raw_humanizer_json("Here is your answer.")


def test_sanitize_defense_in_depth_empty_message_falls_back() -> None:
    # If the JSON parses but the "message" value is empty/missing, the normalized
    # text is the raw JSON blob. The defense-in-depth guard must return the
    # fallback instead of posting the JSON.
    bad_payload = '{"message": "", "presentation": {"version": 1, "elements": []}}'
    result = sanitize_humanized_response(bad_payload, fallback="raw fallback answer")
    assert result == "raw fallback answer"
