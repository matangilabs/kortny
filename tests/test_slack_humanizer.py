from kortny.slack.humanizer import (
    ResponseMode,
    ResponseRecord,
    ResponseShape,
    ResponseShapeProfile,
    ResponseStyleProfile,
    SlackSurface,
    _parse_presentation_hint,
    _synthesis_payload,
    sanitize_humanized_response,
)
from kortny.slack.presentation import FieldsElement
from kortny.slack.synthesis import SynthesisContext, SynthesisOutcome


def test_parse_presentation_hint_extracts_elements() -> None:
    raw = (
        '{"message":"Your schedule is set.","presentation":{"elements":'
        '[{"type":"fields","items":[{"label":"Cadence","value":"Daily"}]}]}}'
    )
    hint = _parse_presentation_hint(raw)
    assert hint is not None
    assert len(hint.elements) == 1
    assert isinstance(hint.elements[0], FieldsElement)


def test_parse_presentation_hint_none_when_absent_or_garbled() -> None:
    assert _parse_presentation_hint('{"message":"hi"}') is None
    assert _parse_presentation_hint("not json") is None
    assert _parse_presentation_hint(None) is None
    # Unknown element type → dropped → no usable hint.
    assert (
        _parse_presentation_hint(
            '{"message":"x","presentation":{"elements":[{"type":"chart"}]}}'
        )
        is None
    )


def test_sanitize_humanized_response_falls_back_when_empty() -> None:
    assert sanitize_humanized_response("   ", fallback="Use **bold** here.") == (
        "Use **bold** here."
    )


def test_sanitize_humanized_response_keeps_formatting_for_send_boundary() -> None:
    assert (
        sanitize_humanized_response(
            "## Findings\nRead [Slack docs](https://docs.slack.dev).",
            fallback="fallback",
        )
        == "## Findings\nRead [Slack docs](https://docs.slack.dev)."
    )


def test_sanitize_humanized_response_keeps_em_dash_for_send_boundary() -> None:
    em_dash = chr(0x2014)

    assert (
        sanitize_humanized_response(
            f'{{"message":"I checked {em_dash} it is active."}}',
            fallback="fallback",
        )
        == f"I checked {em_dash} it is active."
    )


def test_sanitize_humanized_response_accepts_json_message_contract() -> None:
    assert (
        sanitize_humanized_response(
            '{"message":"**Ready:** I can help with research."}',
            fallback="fallback",
        )
        == "**Ready:** I can help with research."
    )


def test_sanitize_humanized_response_falls_back_on_humanizer_leak() -> None:
    leaked = (
        "_mode is quick_answer, so we just present the answer.\n\n"
        "Let me write:\n\n"
        "*Search & Research*\n"
        "• Web search"
    )

    assert (
        sanitize_humanized_response(
            leaked,
            fallback="*Search & Research*\n• Web search",
        )
        == "*Search & Research*\n• Web search"
    )


def test_synthesis_payload_does_not_normalize_raw_answer_prompt_content() -> None:
    em_dash = chr(0x2014)
    response_record = ResponseRecord(
        user_request="Summarize this",
        raw_answer=f"Raw answer {em_dash} keep internal prompt content unchanged.",
        response_mode=ResponseMode.quick_answer,
        response_shape=ResponseShapeProfile(
            shape=ResponseShape.quick_reply,
            label="Quick reply",
            selected_reason="test",
            required_elements=[],
            quality_checks=[],
            avoid=[],
        ),
        task_status="succeeded",
        slack_surface=SlackSurface(kind="dm", threaded=False),
        style_profile=ResponseStyleProfile(),
        actions_taken=[],
        evidence=[],
        artifacts=[],
        failures=[],
        uncertainties=[],
        suggested_next_actions=[],
        procedural_skills=[],
    )
    synthesis_context = SynthesisContext(
        user_intent="Summarize this",
        outcome=SynthesisOutcome.ok,
        outcome_reason="test",
        slack_surface="dm",
        threaded=False,
    )

    payload = _synthesis_payload(response_record, synthesis_context)

    assert payload["response_record"]["raw_answer"] == (
        f"Raw answer {em_dash} keep internal prompt content unchanged."
    )


def test_sanitize_humanized_response_strips_planned_workflow_preamble() -> None:
    leaked = (
        'The user said "research the top James Bond movies" and provided branch '
        "context. I'm the planned_workflow_merger, so my job is to merge branch "
        "outputs.\n\n"
        "I'll present this as Kortny's final answer.\n"
        ":clapper: James Bond Films\nCasino Royale is the best modern entry."
    )

    assert (
        sanitize_humanized_response(leaked, fallback=leaked)
        == ":clapper: James Bond Films\nCasino Royale is the best modern entry."
    )


def test_sanitize_humanized_response_strips_quick_response_scratchpad() -> None:
    leaked = (
        "The user is asking if I'm up, which is a simple check for my availability. "
        "According to my guidelines, I should be concise and avoid internal routing.\n\n"
        "I'll keep it brief and natural.\n"
        "Yep, I'm up and ready to help with anything lightweight in our Slack threads."
    )

    assert (
        sanitize_humanized_response(leaked, fallback=leaked)
        == "Yep, I'm up and ready to help with anything lightweight in our Slack threads."
    )


def test_sanitize_humanized_response_leaves_slack_formatting_to_send_boundary() -> None:
    cases = [
        (
            "Here are the **two tools** that matter most.",
            "Here are the **two tools** that matter most.",
        ),
        (
            "# Quick take\nLangfuse is the stronger default.",
            "# Quick take\nLangfuse is the stronger default.",
        ),
        (
            "See [Langfuse](https://langfuse.com) for tracing.",
            "See [Langfuse](https://langfuse.com) for tracing.",
        ),
        (
            "I checked the channel. **Nothing urgent** changed.",
            "I checked the channel. **Nothing urgent** changed.",
        ),
        (
            "### Summary\n- Firecrawl worked\n- Brave is rate-limited",
            "### Summary\n- Firecrawl worked\n- Brave is rate-limited",
        ),
        (
            "The report is ready: [download](https://files.example/report.pdf).",
            "The report is ready: [download](https://files.example/report.pdf).",
        ),
        (
            "Use `**literal**` in code.",
            "Use `**literal**` in code.",
        ),
        (
            "```md\n**keep this**\n```",
            "```md\n**keep this**\n```",
        ),
        (
            "No formatting changes needed.",
            "No formatting changes needed.",
        ),
        (
            "I found **3 themes** across [results](https://example.com).",
            "I found **3 themes** across [results](https://example.com).",
        ),
    ]

    for raw, expected in cases:
        assert sanitize_humanized_response(raw, fallback="fallback") == expected
