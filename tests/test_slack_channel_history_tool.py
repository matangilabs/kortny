from __future__ import annotations

from typing import Any

import pytest
from slack_sdk.errors import SlackApiError

from kortny.tools import SlackChannelHistoryTool


class FakeSlackHistoryClient:
    def __init__(
        self,
        *,
        history_pages: list[dict[str, Any]],
        reply_pages_by_ts: dict[str, list[dict[str, Any]]] | None = None,
        history_error: Exception | None = None,
    ) -> None:
        self.history_pages = history_pages
        self.reply_pages_by_ts = reply_pages_by_ts or {}
        self.history_error = history_error
        self.history_calls: list[dict[str, Any]] = []
        self.reply_calls: list[dict[str, Any]] = []

    def conversations_history(
        self,
        *,
        channel: str,
        cursor: str | None = None,
        inclusive: bool | None = None,
        limit: int | None = None,
        latest: str | None = None,
        oldest: str | None = None,
    ) -> dict[str, Any]:
        self.history_calls.append(
            {
                "channel": channel,
                "cursor": cursor,
                "inclusive": inclusive,
                "limit": limit,
                "latest": latest,
                "oldest": oldest,
            }
        )
        if self.history_error is not None:
            raise self.history_error
        return self.history_pages.pop(0)

    def conversations_replies(
        self,
        *,
        channel: str,
        ts: str,
        cursor: str | None = None,
        inclusive: bool | None = None,
        limit: int | None = None,
        latest: str | None = None,
        oldest: str | None = None,
    ) -> dict[str, Any]:
        self.reply_calls.append(
            {
                "channel": channel,
                "ts": ts,
                "cursor": cursor,
                "inclusive": inclusive,
                "limit": limit,
                "latest": latest,
                "oldest": oldest,
            }
        )
        return self.reply_pages_by_ts[ts].pop(0)


def test_slack_channel_history_paginates_with_cursor() -> None:
    client = FakeSlackHistoryClient(
        history_pages=[
            {
                "ok": True,
                "messages": [
                    {"ts": "3.000000", "user": "U3", "text": "third"},
                    {"ts": "2.000000", "user": "U2", "text": "second"},
                ],
                "response_metadata": {"next_cursor": "cursor-2"},
            },
            {
                "ok": True,
                "messages": [{"ts": "1.000000", "user": "U1", "text": "first"}],
                "response_metadata": {"next_cursor": ""},
            },
        ]
    )

    result = SlackChannelHistoryTool(
        client,
        default_channel_id="C123",
        page_limit=2,
    ).invoke({"limit": 3, "oldest_ts": "1.000000"})

    assert result.output["message_count"] == 3
    assert [message["text"] for message in result.output["messages"]] == [
        "first",
        "second",
        "third",
    ]
    assert client.history_calls == [
        {
            "channel": "C123",
            "cursor": None,
            "inclusive": True,
            "limit": 2,
            "latest": None,
            "oldest": "1.000000",
        },
        {
            "channel": "C123",
            "cursor": "cursor-2",
            "inclusive": True,
            "limit": 1,
            "latest": None,
            "oldest": "1.000000",
        },
    ]


def test_slack_channel_history_includes_root_message_file_metadata() -> None:
    client = FakeSlackHistoryClient(
        history_pages=[
            {
                "ok": True,
                "messages": [
                    {
                        "ts": "10.000000",
                        "user": "U1",
                        "text": "Please review this report",
                        "files": [
                            {
                                "id": "F123",
                                "name": "report.pdf",
                                "title": "Q2 report",
                                "mimetype": "application/pdf",
                                "filetype": "pdf",
                                "size": 2048,
                                "created": 1779562349,
                                "user": "U1",
                                "url_private_download": "https://files.slack.com/private",
                            }
                        ],
                    }
                ],
                "response_metadata": {"next_cursor": ""},
            }
        ]
    )

    result = SlackChannelHistoryTool(client, default_channel_id="C123").invoke(
        {"limit": 1}
    )

    assert result.output["messages"][0]["files"] == [
        {
            "id": "F123",
            "name": "report.pdf",
            "title": "Q2 report",
            "mimetype": "application/pdf",
            "filetype": "pdf",
            "user": "U1",
            "size_bytes": 2048,
            "created": 1779562349,
        }
    ]
    assert "url_private_download" not in result.output["messages"][0]["files"][0]


def test_slack_channel_history_fans_out_active_threads() -> None:
    client = FakeSlackHistoryClient(
        history_pages=[
            {
                "ok": True,
                "messages": [
                    {
                        "ts": "10.000000",
                        "user": "U1",
                        "text": "Root message",
                        "reply_count": 2,
                    }
                ],
                "response_metadata": {"next_cursor": ""},
            }
        ],
        reply_pages_by_ts={
            "10.000000": [
                {
                    "ok": True,
                    "messages": [
                        {
                            "ts": "10.000000",
                            "user": "U1",
                            "text": "Root message",
                            "reply_count": 2,
                        },
                        {
                            "ts": "10.000001",
                            "user": "U2",
                            "text": "First reply",
                            "thread_ts": "10.000000",
                        },
                        {
                            "ts": "10.000002",
                            "bot_id": "B1",
                            "text": "Second reply",
                            "thread_ts": "10.000000",
                            "files": [
                                {
                                    "id": "F456",
                                    "name": "reply.csv",
                                    "mimetype": "text/csv",
                                    "filetype": "csv",
                                    "size": 512,
                                }
                            ],
                        },
                    ],
                    "response_metadata": {"next_cursor": ""},
                }
            ]
        },
    )

    result = SlackChannelHistoryTool(client, default_channel_id="C123").invoke(
        {"limit": 5, "include_threads": True, "latest_ts": "11.000000"}
    )

    assert result.output["messages"] == [
        {
            "user": "U1",
            "ts": "10.000000",
            "text": "Root message",
            "thread_ts": None,
            "reply_count": 2,
        },
        {
            "user": "U2",
            "ts": "10.000001",
            "text": "First reply",
            "thread_ts": "10.000000",
            "reply_count": 0,
        },
        {
            "user": None,
            "ts": "10.000002",
            "text": "Second reply",
            "thread_ts": "10.000000",
            "reply_count": 0,
            "bot_id": "B1",
            "files": [
                {
                    "id": "F456",
                    "name": "reply.csv",
                    "mimetype": "text/csv",
                    "filetype": "csv",
                    "size_bytes": 512,
                }
            ],
        },
    ]
    assert client.reply_calls == [
        {
            "channel": "C123",
            "ts": "10.000000",
            "cursor": None,
            "inclusive": True,
            "limit": 5,
            "latest": "11.000000",
            "oldest": None,
        }
    ]


def test_slack_channel_history_treats_blank_optional_args_as_omitted() -> None:
    client = FakeSlackHistoryClient(
        history_pages=[
            {
                "ok": True,
                "messages": [{"ts": "1.000000", "user": "U1", "text": "first"}],
                "response_metadata": {"next_cursor": ""},
            }
        ]
    )

    result = SlackChannelHistoryTool(client, default_channel_id="C123").invoke(
        {
            "channel_id": "",
            "oldest_ts": " ",
            "latest_ts": "",
            "limit": 10,
        }
    )

    assert result.output["channel_id"] == "C123"
    assert result.output["oldest_ts"] is None
    assert result.output["latest_ts"] is None
    assert result.output["message_count"] == 1
    assert client.history_calls[0]["channel"] == "C123"
    assert client.history_calls[0]["oldest"] is None
    assert client.history_calls[0]["latest"] is None


def test_slack_channel_history_reports_inaccessible_channel_as_recoverable() -> None:
    client = FakeSlackHistoryClient(
        history_pages=[
            {
                "ok": False,
                "error": "not_in_channel",
            }
        ]
    )

    result = SlackChannelHistoryTool(client, default_channel_id="C_PRIVATE").invoke({})

    assert result.output == {
        "channel_id": "C_PRIVATE",
        "oldest_ts": None,
        "latest_ts": None,
        "limit": 200,
        "include_threads": False,
        "message_count": 0,
        "messages": [],
        "error": {
            "code": "not_in_channel",
            "message": "conversations.history failed: not_in_channel",
            "recoverable": True,
            "hint": (
                "Use prior thread context if it is sufficient. Otherwise ask the user "
                "to add Kortny to the channel or provide an accessible Slack channel."
            ),
        },
    }


def test_slack_channel_history_reports_hallucinated_channel_id_as_recoverable() -> None:
    client = FakeSlackHistoryClient(
        history_pages=[],
        history_error=SlackApiError(
            "The request to the Slack API failed.",
            {"ok": False, "error": "channel_not_found"},
        ),
    )

    result = SlackChannelHistoryTool(client, default_channel_id="C_CURRENT").invoke(
        {"channel_id": "C_HALLUCINATED", "limit": 100}
    )

    assert result.output == {
        "channel_id": "C_HALLUCINATED",
        "oldest_ts": None,
        "latest_ts": None,
        "limit": 100,
        "include_threads": False,
        "message_count": 0,
        "messages": [],
        "error": {
            "code": "channel_not_found",
            "message": "Slack API failed: channel_not_found",
            "recoverable": True,
            "hint": (
                "If the user means the current Slack channel, retry "
                "slack_channel_history without channel_id."
            ),
        },
    }


def test_slack_channel_history_rejects_missing_channel_without_default() -> None:
    client = FakeSlackHistoryClient(history_pages=[])

    with pytest.raises(ValueError, match="channel_id"):
        SlackChannelHistoryTool(client).invoke({})
