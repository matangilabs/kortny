from kortny.slack.humanizer import sanitize_humanized_response


def test_sanitize_humanized_response_falls_back_when_empty() -> None:
    assert sanitize_humanized_response("   ", fallback="Use **bold** here.") == (
        "Use *bold* here."
    )


def test_sanitize_humanized_response_normalizes_slack_mrkdwn() -> None:
    assert (
        sanitize_humanized_response(
            "## Findings\nRead [Slack docs](https://docs.slack.dev).",
            fallback="fallback",
        )
        == "*Findings*\nRead <https://docs.slack.dev|Slack docs>."
    )


def test_sanitize_humanized_response_replaces_em_dash() -> None:
    em_dash = chr(0x2014)

    assert (
        sanitize_humanized_response(
            f'{{"message":"I checked {em_dash} it is active."}}',
            fallback="fallback",
        )
        == "I checked - it is active."
    )


def test_sanitize_humanized_response_accepts_json_message_contract() -> None:
    assert (
        sanitize_humanized_response(
            '{"message":"**Ready:** I can help with research."}',
            fallback="fallback",
        )
        == "*Ready:* I can help with research."
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


def test_sanitize_humanized_response_golden_slack_cases() -> None:
    cases = [
        (
            "Here are the **two tools** that matter most.",
            "Here are the *two tools* that matter most.",
        ),
        (
            "# Quick take\nLangfuse is the stronger default.",
            "*Quick take*\nLangfuse is the stronger default.",
        ),
        (
            "See [Langfuse](https://langfuse.com) for tracing.",
            "See <https://langfuse.com|Langfuse> for tracing.",
        ),
        (
            "I checked the channel. **Nothing urgent** changed.",
            "I checked the channel. *Nothing urgent* changed.",
        ),
        (
            "### Summary\n- Firecrawl worked\n- Brave is rate-limited",
            "*Summary*\n- Firecrawl worked\n- Brave is rate-limited",
        ),
        (
            "The report is ready: [download](https://files.example/report.pdf).",
            "The report is ready: <https://files.example/report.pdf|download>.",
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
            "I found *3 themes* across <https://example.com|results>.",
        ),
    ]

    for raw, expected in cases:
        assert sanitize_humanized_response(raw, fallback="fallback") == expected
