"""Unit tests for live assistant status narration (HIG-247 follow-up)."""

from __future__ import annotations

from typing import Any

from kortny.slack.assistant_status import (
    PHASE_RESEARCHING,
    PHASE_WORKING,
    AssistantStatusReporter,
    ChannelProgressReporter,
    NullStatusReporter,
    phase_for_tool,
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


def test_reporter_sets_two_level_status() -> None:
    client = RecordingStatusClient()
    reporter = AssistantStatusReporter(
        client=client, channel_id="D1", thread_ts="123.45"
    )
    reporter.report("Searching the web…", phase=PHASE_RESEARCHING)
    assert client.calls == [
        {
            "channel_id": "D1",
            "thread_ts": "123.45",
            # Composer line = coarse phase; prominent bubble = granular step.
            "status": PHASE_RESEARCHING,
            "loading_messages": ["Searching the web…"],
        }
    ]


def test_reporter_falls_back_to_step_without_phase() -> None:
    client = RecordingStatusClient()
    reporter = AssistantStatusReporter(
        client=client, channel_id="D1", thread_ts="123.45"
    )
    reporter.report("Searching the web…")
    assert client.calls[0]["status"] == "Searching the web…"
    assert client.calls[0]["loading_messages"] == ["Searching the web…"]


def test_reporter_updates_bubble_when_only_step_changes() -> None:
    client = RecordingStatusClient()
    reporter = AssistantStatusReporter(
        client=client, channel_id="D1", thread_ts="123.45"
    )
    # Same phase, different granular step → still re-issued so the bubble updates.
    reporter.report("Searching the web…", phase=PHASE_RESEARCHING)
    reporter.report("Loading a skill…", phase=PHASE_RESEARCHING)
    assert [c["loading_messages"] for c in client.calls] == [
        ["Searching the web…"],
        ["Loading a skill…"],
    ]


def test_reporter_throttles_identical_step_and_phase() -> None:
    client = RecordingStatusClient()
    reporter = AssistantStatusReporter(
        client=client, channel_id="D1", thread_ts="123.45"
    )
    reporter.report("Running code…", phase=PHASE_WORKING)
    reporter.report("Running code…", phase=PHASE_WORKING)  # identical → no 2nd call
    reporter.report("Writing the response…", phase=PHASE_WORKING)
    assert [c["loading_messages"][0] for c in client.calls] == [
        "Running code…",
        "Writing the response…",
    ]


def test_phase_for_tool_research_vs_work() -> None:
    assert phase_for_tool("web_search") == PHASE_RESEARCHING
    assert phase_for_tool("load_skill") == PHASE_RESEARCHING
    assert phase_for_tool("mcp__context7__get_docs") == PHASE_RESEARCHING
    assert phase_for_tool("code_exec") == PHASE_WORKING
    assert phase_for_tool("run_skill_script") == PHASE_WORKING
    assert phase_for_tool("totally_unknown") == PHASE_WORKING


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


class RecordingUpdateClient:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls: list[dict[str, Any]] = []
        self._fail = fail

    def chat_update(self, *, channel: str, ts: str, text: str) -> dict[str, Any]:
        if self._fail:
            raise RuntimeError("slack down")
        self.calls.append({"channel": channel, "ts": ts, "text": text})
        return {"ok": True}


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_channel_reporter_edits_ack_with_progress_line() -> None:
    client = RecordingUpdateClient()
    reporter = ChannelProgressReporter(
        client=client,
        channel_id="C1",
        message_ts="111.22",
        base_text="On it.",
        clock=FakeClock(),
    )
    reporter.report("Searching the web…", phase=PHASE_RESEARCHING)
    assert client.calls == [
        {"channel": "C1", "ts": "111.22", "text": "On it.\n_Searching the web…_"}
    ]


def test_channel_reporter_without_base_text() -> None:
    client = RecordingUpdateClient()
    reporter = ChannelProgressReporter(
        client=client,
        channel_id="C1",
        message_ts="111.22",
        base_text="",
        clock=FakeClock(),
    )
    reporter.report("Reading the channel…")
    assert client.calls[0]["text"] == "_Reading the channel…_"


def test_channel_reporter_throttles_within_interval() -> None:
    clock = FakeClock()
    client = RecordingUpdateClient()
    reporter = ChannelProgressReporter(
        client=client,
        channel_id="C1",
        message_ts="111.22",
        base_text="On it.",
        min_interval_seconds=5.0,
        clock=clock,
    )
    reporter.report("Searching the web…")  # t=0, posts
    clock.now = 2.0
    reporter.report("Reading the channel…")  # within 5s → throttled
    clock.now = 6.0
    reporter.report("Writing the response…")  # past interval → posts
    assert [c["text"] for c in client.calls] == [
        "On it.\n_Searching the web…_",
        "On it.\n_Writing the response…_",
    ]


def test_channel_reporter_dedupes_identical_line() -> None:
    clock = FakeClock()
    client = RecordingUpdateClient()
    reporter = ChannelProgressReporter(
        client=client,
        channel_id="C1",
        message_ts="111.22",
        base_text="On it.",
        clock=clock,
    )
    reporter.report("Reading the channel…")
    clock.now = 100.0  # past any throttle window
    reporter.report("Reading the channel…")  # identical content → no 2nd edit
    assert len(client.calls) == 1


def test_channel_reporter_ignores_empty_status() -> None:
    client = RecordingUpdateClient()
    reporter = ChannelProgressReporter(
        client=client,
        channel_id="C1",
        message_ts="111.22",
        base_text="On it.",
        clock=FakeClock(),
    )
    reporter.report("   ")
    assert client.calls == []


def test_channel_reporter_swallows_client_errors() -> None:
    client = RecordingUpdateClient(fail=True)
    reporter = ChannelProgressReporter(
        client=client,
        channel_id="C1",
        message_ts="111.22",
        base_text="On it.",
        clock=FakeClock(),
    )
    # Must not raise — a progress edit cannot fail the task.
    reporter.report("Running code…")
