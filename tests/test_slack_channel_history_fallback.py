"""Test slack_channel_history 404 auto-fallback (HIG-295 Part 3)."""

from __future__ import annotations

from unittest.mock import MagicMock

from slack_sdk.errors import SlackApiError

from kortny.tools.slack_channel_history import SlackChannelHistoryTool


def _make_channel_not_found_error() -> SlackApiError:
    response = MagicMock()
    response.get.side_effect = lambda k, *a: {
        "ok": False,
        "error": "channel_not_found",
    }.get(k, a[0] if a else None)
    response.data = {"ok": False, "error": "channel_not_found"}
    response.status_code = 404
    err = SlackApiError(message="channel_not_found", response=response)
    return err


def test_slack_channel_history_fallback_on_channel_not_found() -> None:
    """When model supplies bad channel_id, tool auto-retries with default channel."""
    default_channel = "C_DEFAULT"
    bad_channel = "C_HALLUCINATED"

    client = MagicMock()
    good_response = {
        "ok": True,
        "messages": [{"ts": "1234567890.000001", "text": "hello", "type": "message"}],
        "has_more": False,
    }

    def conversations_history_side_effect(**kwargs: object) -> object:
        if kwargs.get("channel") == bad_channel:
            raise _make_channel_not_found_error()
        return good_response

    client.conversations_history.side_effect = conversations_history_side_effect

    tool = SlackChannelHistoryTool(
        client,
        default_channel_id=default_channel,
    )
    result = tool.invoke({"channel_id": bad_channel})

    output = result.output
    assert isinstance(output, dict)
    # Should have fallen back and returned messages
    assert output.get("channel_id") == default_channel
    assert "fallback_note" in output
    assert bad_channel in output["fallback_note"]
    assert output.get("message_count", 0) >= 0


def test_slack_channel_history_no_fallback_when_no_default() -> None:
    """When there's no default channel, channel_not_found returns recoverable error."""
    bad_channel = "C_HALLUCINATED"

    client = MagicMock()

    def conversations_history_raise(**kwargs: object) -> object:
        raise _make_channel_not_found_error()

    client.conversations_history.side_effect = conversations_history_raise

    tool = SlackChannelHistoryTool(client, default_channel_id=None)
    result = tool.invoke({"channel_id": bad_channel})

    output = result.output
    assert isinstance(output, dict)
    error = output.get("error", {})
    assert error.get("recoverable") is True
    assert "fallback_note" not in output
