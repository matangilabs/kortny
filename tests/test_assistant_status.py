"""Unit tests for live assistant status narration (HIG-247 follow-up)."""

from __future__ import annotations

from typing import Any

from kortny.slack.assistant_status import (
    AssistantStatusReporter,
    NullStatusReporter,
    status_for_tool,
)


class RecordingStatusClient:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls: list[dict[str, Any]] = []
        self._fail = fail

    def assistant_threads_setStatus(
        self,
        *,
        channel_id: str,
        thread_ts: str,
        status: str,
        loading_messages: list[str] | None = None,
    ) -> dict[str, Any]:
        if self._fail:
            raise RuntimeError("slack down")
        self.calls.append(
            {
                "channel_id": channel_id,
                "thread_ts": thread_ts,
                "status": status,
                "loading_messages": loading_messages,
            }
        )
        return {"ok": True}


def test_status_for_tool_native_verbs() -> None:
    assert status_for_tool("web_search") == "Searching the web…"
    assert status_for_tool("code_exec") == "Running code…"
    assert (
        status_for_tool("query_workspace_graph")
        == "Searching your workspace knowledge…"
    )


def test_status_for_tool_mcp_derives_server() -> None:
    assert status_for_tool("mcp__context7__get_docs") == "Querying Context7…"


def test_status_for_tool_composio_derives_toolkit() -> None:
    assert status_for_tool("composio_linear_create_issue") == "Checking Linear…"


def test_status_for_tool_display_name_fallback() -> None:
    assert (
        status_for_tool("some_unknown_tool", display_name="Custom Thing")
        == "Using Custom Thing…"
    )


def test_status_for_tool_generic_fallback() -> None:
    assert status_for_tool("totally_unknown") == "Working through it…"


def test_reporter_sets_status() -> None:
    client = RecordingStatusClient()
    reporter = AssistantStatusReporter(
        client=client, channel_id="D1", thread_ts="123.45"
    )
    reporter.report("Searching the web…")
    assert client.calls == [
        {
            "channel_id": "D1",
            "thread_ts": "123.45",
            "status": "Searching the web…",
            # Single-item list — Slack rejects an empty loading_messages; the one
            # item replaces the app's static intro loop with the current step.
            "loading_messages": ["Searching the web…"],
        }
    ]


def test_reporter_throttles_repeats() -> None:
    client = RecordingStatusClient()
    reporter = AssistantStatusReporter(
        client=client, channel_id="D1", thread_ts="123.45"
    )
    reporter.report("Running code…")
    reporter.report("Running code…")  # identical → no second call
    reporter.report("Writing the response…")
    assert [c["status"] for c in client.calls] == [
        "Running code…",
        "Writing the response…",
    ]


def test_reporter_ignores_empty_status() -> None:
    client = RecordingStatusClient()
    reporter = AssistantStatusReporter(
        client=client, channel_id="D1", thread_ts="123.45"
    )
    reporter.report("   ")
    assert client.calls == []


def test_reporter_swallows_client_errors() -> None:
    client = RecordingStatusClient(fail=True)
    reporter = AssistantStatusReporter(
        client=client, channel_id="D1", thread_ts="123.45"
    )
    # Must not raise — a status failure cannot fail the task.
    reporter.report("Running code…")


def test_null_reporter_is_noop() -> None:
    NullStatusReporter().report("anything")  # no error, no effect
