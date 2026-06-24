"""Unit tests for the Claude-Tag-style progress checklist (HIG-289).

All tests are pure unit tests — no DB, no fixtures.
"""

from __future__ import annotations

from typing import Any

from kortny.slack.assistant_status import (
    ChannelProgressReporter,
    _is_internal_step,
    _render_checklist,
)


class RecordingUpdateClient:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls: list[dict[str, Any]] = []
        self._fail = fail
        # Verify native streaming methods are never called.
        self.start_stream_calls: list[Any] = []
        self.append_stream_calls: list[Any] = []
        self.stop_stream_calls: list[Any] = []

    def chat_update(self, *, channel: str, ts: str, text: str) -> dict[str, Any]:
        if self._fail:
            raise RuntimeError("slack down")
        self.calls.append({"channel": channel, "ts": ts, "text": text})
        return {"ok": True}

    # These must never be called from ChannelProgressReporter.
    def chat_startStream(self, **kwargs: Any) -> dict[str, Any]:  # pragma: no cover
        self.start_stream_calls.append(kwargs)
        return {"ok": True, "ts": "1.0"}

    def chat_appendStream(self, **kwargs: Any) -> dict[str, Any]:  # pragma: no cover
        self.append_stream_calls.append(kwargs)
        return {"ok": True}

    def chat_stopStream(self, **kwargs: Any) -> dict[str, Any]:  # pragma: no cover
        self.stop_stream_calls.append(kwargs)
        return {"ok": True}


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _make_reporter(
    client: RecordingUpdateClient,
    base_text: str = "On it.",
) -> ChannelProgressReporter:
    return ChannelProgressReporter(
        client=client,
        channel_id="C1",
        message_ts="111.22",
        base_text=base_text,
        min_interval_seconds=5.0,
        clock=FakeClock(),
    )


# ---------------------------------------------------------------------------
# _render_checklist unit tests
# ---------------------------------------------------------------------------


def test_render_checklist_all_pending() -> None:
    result = _render_checklist(["Step A", "Step B", "Step C"], -1, all_done=False)
    assert result == "○ Step A\n○ Step B\n○ Step C"


def test_render_checklist_first_in_progress() -> None:
    result = _render_checklist(["Step A", "Step B", "Step C"], 0, all_done=False)
    assert result == "✱ Step A\n○ Step B\n○ Step C"


def test_render_checklist_second_in_progress() -> None:
    result = _render_checklist(["Step A", "Step B", "Step C"], 1, all_done=False)
    assert result == "✓ Step A\n✱ Step B\n○ Step C"


def test_render_checklist_all_done() -> None:
    result = _render_checklist(["Step A", "Step B"], 0, all_done=True)
    assert result == "✓ Step A\n✓ Step B"


# ---------------------------------------------------------------------------
# _is_internal_step unit tests
# ---------------------------------------------------------------------------


def test_is_internal_step_exact_match() -> None:
    assert _is_internal_step(
        "Handle the Slack request using available context and tools."
    )
    assert _is_internal_step("Planning")
    assert _is_internal_step("Finalizing")
    assert _is_internal_step("Compiling")


def test_is_internal_step_prefix_match() -> None:
    assert _is_internal_step("Formatting the document")
    assert _is_internal_step("Finalizing the output")
    assert _is_internal_step("Planning the approach")
    assert _is_internal_step("handle the slack request and do something")
    assert _is_internal_step("Synthesizing the results")
    assert _is_internal_step("Compiling findings")


def test_is_internal_step_case_insensitive() -> None:
    assert _is_internal_step("FORMATTING output")
    assert _is_internal_step("finalizing")


def test_is_internal_step_non_internal() -> None:
    assert not _is_internal_step("Research aggressive ETFs")
    assert not _is_internal_step("Compare funds by leverage ratio")
    assert not _is_internal_step("Summarize findings")
    assert not _is_internal_step("Draft the report")


# ---------------------------------------------------------------------------
# ChannelProgressReporter checklist mode tests
# ---------------------------------------------------------------------------


def test_checklist_rendered_for_complex_plan() -> None:
    client = RecordingUpdateClient()
    reporter = _make_reporter(client)

    reporter.notify_plan(["Research ETFs", "Compare by leverage", "Summarize"])

    assert len(client.calls) == 1
    text = client.calls[0]["text"]
    assert "○ Research ETFs" in text
    assert "○ Compare by leverage" in text
    assert "○ Summarize" in text
    assert "On it." in text
    # Streaming methods must never be called.
    assert client.start_stream_calls == []
    assert client.append_stream_calls == []
    assert client.stop_stream_calls == []


def test_checklist_step_started_advances_marker() -> None:
    client = RecordingUpdateClient()
    reporter = _make_reporter(client)

    reporter.notify_plan(["Research ETFs", "Compare by leverage", "Summarize"])
    reporter.notify_step_started("Research ETFs")

    last_text = client.calls[-1]["text"]
    assert "✱ Research ETFs" in last_text
    assert "○ Compare by leverage" in last_text
    assert "○ Summarize" in last_text

    reporter.notify_step_started("Compare by leverage")

    last_text = client.calls[-1]["text"]
    assert "✓ Research ETFs" in last_text
    assert "✱ Compare by leverage" in last_text
    assert "○ Summarize" in last_text


def test_checklist_all_done_on_completed() -> None:
    client = RecordingUpdateClient()
    reporter = _make_reporter(client)

    reporter.notify_plan(["Research ETFs", "Compare by leverage", "Summarize"])
    reporter.notify_step_started("Research ETFs")
    reporter.notify_completed()

    last_text = client.calls[-1]["text"]
    assert "✓ Research ETFs" in last_text
    assert "✓ Compare by leverage" in last_text
    assert "✓ Summarize" in last_text
    # No ○ or ✱ remain.
    assert "○" not in last_text
    assert "✱" not in last_text


def test_simple_task_no_checklist() -> None:
    """A plan with only 1 real step must NOT activate checklist mode."""
    client = RecordingUpdateClient()
    clock = FakeClock()
    reporter = ChannelProgressReporter(
        client=client,
        channel_id="C1",
        message_ts="111.22",
        base_text="On it.",
        min_interval_seconds=0.0,
        clock=clock,
    )

    reporter.notify_plan(["Summarize"])  # only 1 step — no checklist

    # No chat_update from notify_plan (checklist mode not entered).
    assert client.calls == []

    # report() should still work normally (single-line progress).
    reporter.report("Searching…")
    assert len(client.calls) == 1
    assert "_Searching…_" in client.calls[0]["text"]


def test_zero_steps_no_checklist() -> None:
    """A plan with 0 real steps must NOT activate checklist mode."""
    client = RecordingUpdateClient()
    reporter = _make_reporter(client)

    reporter.notify_plan([])

    assert client.calls == []
    assert reporter._checklist_mode is False


def test_internal_steps_filtered_before_notify() -> None:
    """The coordinator filters internal steps; only real ones reach notify_plan."""
    client = RecordingUpdateClient()
    reporter = _make_reporter(client)

    # Simulate: coordinator already filtered internal steps before calling notify_plan.
    # If 0 real steps remain, no checklist.
    reporter.notify_plan(
        ["Handle the Slack request using available context and tools."]
    )
    # Only 1 item — no checklist mode.
    assert reporter._checklist_mode is False
    assert client.calls == []


def test_report_suppressed_in_checklist_mode() -> None:
    """In checklist mode, calling report() must be a no-op."""
    client = RecordingUpdateClient()
    reporter = _make_reporter(client)

    reporter.notify_plan(["Research ETFs", "Summarize"])
    initial_calls = len(client.calls)

    reporter.report("Searching the web…")
    reporter.report("Writing the response…")

    # No additional calls — report() is suppressed in checklist mode.
    assert len(client.calls) == initial_calls


def test_checklist_never_calls_startstream() -> None:
    """All checklist lifecycle methods must never invoke streaming primitives."""
    client = RecordingUpdateClient()
    reporter = _make_reporter(client)

    reporter.notify_plan(["Step One", "Step Two"])
    reporter.notify_step_started("Step One")
    reporter.notify_step_started("Step Two")
    reporter.notify_completed()

    assert client.start_stream_calls == []
    assert client.append_stream_calls == []
    assert client.stop_stream_calls == []


def test_throttle_and_dedup_preserved_in_simple_mode() -> None:
    """Throttle and dedup still apply when NOT in checklist mode."""
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

    reporter.report("Searching…")  # t=0, posts
    clock.now = 2.0
    reporter.report("Reading…")  # within 5 s → throttled
    clock.now = 6.0
    reporter.report("Writing…")  # past interval → posts

    assert len(client.calls) == 2
    assert "_Searching…_" in client.calls[0]["text"]
    assert "_Writing…_" in client.calls[1]["text"]


def test_dedup_identical_line_in_simple_mode() -> None:
    """Identical consecutive lines are deduped in simple mode."""
    clock = FakeClock()
    client = RecordingUpdateClient()
    reporter = ChannelProgressReporter(
        client=client,
        channel_id="C1",
        message_ts="111.22",
        base_text="On it.",
        min_interval_seconds=0.0,
        clock=clock,
    )

    reporter.report("Searching…")
    clock.now = 100.0
    reporter.report("Searching…")  # same → deduped

    assert len(client.calls) == 1


def test_notify_step_unknown_label_is_noop() -> None:
    """notify_step_started with a label not in plan steps is a no-op."""
    client = RecordingUpdateClient()
    reporter = _make_reporter(client)

    reporter.notify_plan(["Step One", "Step Two"])
    initial_calls = len(client.calls)

    reporter.notify_step_started("Some Internal Step")  # not in plan_steps

    # No additional update.
    assert len(client.calls) == initial_calls


def test_notify_completed_noop_when_not_in_checklist_mode() -> None:
    """notify_completed is a no-op when checklist mode is not active."""
    client = RecordingUpdateClient()
    reporter = _make_reporter(client)

    reporter.notify_completed()  # never called notify_plan first

    assert client.calls == []


def test_checklist_swallows_client_errors() -> None:
    """Checklist updates must never raise even when the Slack client fails."""
    client = RecordingUpdateClient(fail=True)
    reporter = _make_reporter(client)

    # Must not raise.
    reporter.notify_plan(["Step One", "Step Two"])
    reporter.notify_step_started("Step One")
    reporter.notify_completed()


def test_checklist_base_text_empty() -> None:
    """Checklist renders correctly when base_text is empty."""
    client = RecordingUpdateClient()
    reporter = ChannelProgressReporter(
        client=client,
        channel_id="C1",
        message_ts="111.22",
        base_text="",
        min_interval_seconds=0.0,
        clock=FakeClock(),
    )

    reporter.notify_plan(["Step One", "Step Two"])

    text = client.calls[0]["text"]
    assert text.startswith("○ Step One")
    assert "On it." not in text
