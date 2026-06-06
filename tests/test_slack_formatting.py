from kortny.slack import normalize_slack_mrkdwn


def test_normalize_slack_mrkdwn_converts_bold() -> None:
    assert normalize_slack_mrkdwn("Use **Web Search** first.") == (
        "Use *Web Search* first."
    )


def test_normalize_slack_mrkdwn_converts_markdown_links() -> None:
    assert normalize_slack_mrkdwn("Read [Slack docs](https://docs.slack.dev).") == (
        "Read <https://docs.slack.dev|Slack docs>."
    )


def test_normalize_slack_mrkdwn_softens_markdown_headings() -> None:
    assert normalize_slack_mrkdwn("### Capabilities\nI can help.") == (
        "*Capabilities*\nI can help."
    )


def test_normalize_slack_mrkdwn_preserves_lists() -> None:
    text = "1. **Web Searches:** Find sources.\n2. **PDF Generation:** Create reports."

    assert normalize_slack_mrkdwn(text) == (
        "1. *Web Searches:* Find sources.\n2. *PDF Generation:* Create reports."
    )


def test_normalize_slack_mrkdwn_preserves_code() -> None:
    text = (
        "Use `**literal**` here.\n\n```md\n### Title\n[Link](https://example.com)\n```"
    )

    assert normalize_slack_mrkdwn(text) == text


def test_normalize_slack_mrkdwn_replaces_em_dash_outside_code() -> None:
    em_dash = chr(0x2014)
    text = f"Use this {em_dash} not that. Keep `{em_dash}` literal."

    assert normalize_slack_mrkdwn(text) == "Use this - not that. Keep `\u2014` literal."


def test_normalize_slack_mrkdwn_preserves_plain_urls() -> None:
    assert normalize_slack_mrkdwn("Open https://docs.slack.dev for details.") == (
        "Open https://docs.slack.dev for details."
    )
