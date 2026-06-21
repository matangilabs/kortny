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
    _extract_last_json_object,
    _json_message,
    _looks_like_humanizer_scratchpad,
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


# --------------------------------------------------------------------------- #
# Scratchpad / reasoning-model leak — _extract_last_json_object,
# _looks_like_humanizer_scratchpad, and end-to-end sanitize
# --------------------------------------------------------------------------- #

# Faithful fixture derived from the real /tmp/leak_local.txt scratchpad.
# The model echoed the template, wrote reasoning paragraphs, then emitted the
# final JSON without a newline separator after "Now output the JSON."
_FINAL_MSG = (
    "I saw USIM02's note about the Q2 pipeline numbers doc needing a check "
    "before Thursday. I searched Notion across pages and databases - nothing "
    'was titled "Q2 pipeline numbers" and no databases matched "pipeline". '
    "Here's what I'm seeing and the smartest path forward.\n\n"
    "*Bottom line:* The doc almost certainly isn't in our connected Notion "
    "workspace under that name. The three most likely scenarios are below. "
    "My recommendation: don't wait - confirm where it actually lives before "
    "tomorrow. If it's in another tool, drop me a link. If it doesn't exist "
    "yet, let's stand it up today. And if the Quarterly Review Q2 2024 page "
    "is the one finance is waiting on, I can verify it for staleness - just "
    "say the word."
)

_FINAL_JSON = (
    '{"message": '
    + '"'
    + _FINAL_MSG.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    + '"'
    + ', "presentation": {"elements": ['
    '{"type": "items", "title": "Scenarios", "items": ['
    '{"title": "The doc lives in another tool", '
    '"facts": [{"label": "What\'s needed", "value": "A link here in Slack"}, '
    '{"label": "Risk if ignored", "value": "Finance blocked while we search the wrong place"}], '
    '"context": ["Most likely if your team uses Google Sheets, Confluence, or a private Notion page."]},'
    '{"title": "The doc hasn\'t been created yet", '
    '"facts": [{"label": "What\'s needed", "value": "Build a new Q2 pipeline numbers page today or tomorrow"}, '
    '{"label": "Risk if ignored", "value": "Thursday deadline missed because there\'s nothing to review"}], '
    '"context": ["I can create a fresh Notion page and structure it if needed."]},'
    '{"title": "It\'s the old Quarterly Review Q2 2024 page", '
    '"facts": [{"label": "What\'s needed", "value": "Confirm if that\'s what finance means"}, '
    '{"label": "Risk if ignored", "value": "Using year-old data could mislead decisions"}], '
    '"context": ["Last edited April 2025 - could be stale."]}]},'
    '{"type": "sources", "items": [{"source_ref": "source:9", '
    '"title": "Notion Q2 2024 review page", '
    '"body": "Only page with Q2 in title in connected Notion workspace."}]},'
    '{"type": "context", "items": ["Searched Notion (pages and databases) via '
    'connected integration. Slack channel history also checked for links."]}]}}'
)

# The full scratchpad fixture: echoed template + reasoning + final JSON (no
# newline between "Now output the JSON." and the JSON object, matching the
# real leak shape).
_LEAK_FIXTURE = (
    ': {"message": "...", "presentation": {...}}\n\n'
    "Now craft message: Catch the flag from Slack channel (the observed message "
    'from USIM02: "Someone needs to double-check the Q2 pipeline numbers doc '
    "before Thursday - I haven't had time and finance is waiting on it.\"). "
    "We can mention that. So start with \"I saw USIM02's note about the Q2 "
    'pipeline numbers doc and finance waiting. I searched Notion..." but user '
    "request says start from what you noticed and what you checked, so that's "
    "perfect.\n\n"
    'Now build presentation: items, with title "Scenarios & tradeoffs" maybe. '
    'Each item: title (scenario name), facts: "What it means", "Risk", '
    '"Action".\n\n'
    "Final structure: Message: lead-in, then items presentation, then a closing "
    "recommendation in message?\n\n"
    "We'll produce:\n"
    '{"message": "draft placeholder", "presentation": {"elements": []}}\n\n'
    "But message already includes bottom line and recommendation. So it's good.\n\n"
    "Now output the JSON." + _FINAL_JSON
)


def test_extract_last_json_object_returns_last_with_message() -> None:
    # Basic: two JSON objects in text, returns the last one with a message field.
    text = 'noise {"a":1} more {"message":"hi"} tail'
    result = _extract_last_json_object(text)
    assert result == {"message": "hi"}


def test_extract_last_json_object_skips_no_message_objects() -> None:
    # Objects without a message field are not returned.
    text = '{"a": 1} {"b": 2}'
    assert _extract_last_json_object(text) is None


def test_extract_last_json_object_prefers_later_object() -> None:
    text = '{"message": "first"} some text {"message": "second"}'
    result = _extract_last_json_object(text)
    assert result is not None
    assert result["message"] == "second"


def test_looks_like_humanizer_scratchpad_echoed_template() -> None:
    # Starts with echoed template containing both keys.
    assert _looks_like_humanizer_scratchpad(
        ': {"message": "...", "presentation": {...}}'
    )


def test_looks_like_humanizer_scratchpad_now_craft_message() -> None:
    assert _looks_like_humanizer_scratchpad("Now craft message: write something")


def test_looks_like_humanizer_scratchpad_now_output_json() -> None:
    assert _looks_like_humanizer_scratchpad('Now output the JSON.{"message": "hi"}')


def test_looks_like_humanizer_scratchpad_clean_prose_false() -> None:
    assert not _looks_like_humanizer_scratchpad(
        "I checked Notion and couldn't find the doc. Here's what I know."
    )


def test_sanitize_leak_fixture_extracts_message_or_falls_back() -> None:
    # The end-to-end test: the scratchpad leak must resolve to EITHER the
    # extracted final message (starting with "I saw USIM02") OR the safe
    # fallback — it must never contain the internal reasoning phrases or start
    # with the raw JSON template.
    result = sanitize_humanized_response(_LEAK_FIXTURE, fallback="RAW")
    assert not result.startswith('{"message"'), "must not post raw JSON to Slack"
    assert "Now craft message" not in result, "internal reasoning must not leak"
    assert "Now build presentation" not in result, "internal reasoning must not leak"
    # Either we extracted the real answer or we fell back cleanly.
    assert result.startswith("I saw USIM02") or result == "RAW", (
        f"unexpected result: {result[:120]!r}"
    )


def test_parse_presentation_hint_from_leak_fixture() -> None:
    # _parse_presentation_hint must find the presentation in the LAST JSON
    # object in the scratchpad, not give up because the whole text isn't JSON.
    hint = _parse_presentation_hint(_LEAK_FIXTURE)
    assert hint is not None, "presentation hint must be extracted from the scratchpad"
    assert len(hint.elements) > 0, "extracted hint must have at least one element"
