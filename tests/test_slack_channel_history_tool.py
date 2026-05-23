from __future__ import annotations

from typing import Any

import pytest

from kortny.tools import SlackChannelHistoryError, SlackChannelHistoryTool


class FakeSlackHistoryClient:
    def __init__(
        self,
        *,
        history_pages: list[dict[str, Any]],
        reply_pages_by_ts: dict[str, list[dict[str, Any]]] | None = None,
    ) -> None:
        self.history_pages = history_pages
        self.reply_pages_by_ts = reply_pages_by_ts or {}
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


def test_slack_channel_history_requires_accessible_channel() -> None:
    client = FakeSlackHistoryClient(
        history_pages=[
            {
                "ok": False,
                "error": "not_in_channel",
            }
        ]
    )

    with pytest.raises(
        SlackChannelHistoryError,
        match="conversations.history failed: not_in_channel",
    ):
        SlackChannelHistoryTool(client, default_channel_id="C_PRIVATE").invoke({})


def test_slack_channel_history_rejects_missing_channel_without_default() -> None:
    client = FakeSlackHistoryClient(history_pages=[])

    with pytest.raises(ValueError, match="channel_id"):
        SlackChannelHistoryTool(client).invoke({})
