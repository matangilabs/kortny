"""HIG-234: ambient supervisor hosts poller loops with crash isolation.

These tests use fake in-process loops (no Postgres, no docker). Crash-restart
behavior is verified with condition-based waits and tiny injected backoff
values, never wall-clock sleeps.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

import pytest

from kortny.ambient.supervisor import (
    AmbientSupervisor,
    BackoffPolicy,
    LoopSpec,
)


def _wait_until(
    predicate: Callable[[], bool],
    *,
    timeout: float = 5.0,
    interval: float = 0.005,
) -> bool:
    """Poll ``predicate`` until true or timeout; condition-based, not a sleep."""

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def _tiny_backoff() -> BackoffPolicy:
    return BackoffPolicy(
        initial_seconds=0.01,
        max_seconds=0.04,
        reset_after_seconds=0.05,
    )


def test_supervisor_hosts_enabled_loops() -> None:
    started = {"a": threading.Event(), "b": threading.Event()}

    def make_target(name: str) -> Callable[[], None]:
        def target() -> None:
            started[name].set()
            # Block until the supervisor stops so the thread stays alive.
            while True:
                time.sleep(0.01)

        return target

    supervisor = AmbientSupervisor(
        [
            LoopSpec(name="a", target=make_target("a"), backoff=_tiny_backoff()),
            LoopSpec(name="b", target=make_target("b"), backoff=_tiny_backoff()),
        ]
    )
    live = supervisor.start()
    try:
        assert set(live) == {"a", "b"}
        assert _wait_until(lambda: started["a"].is_set())
        assert _wait_until(lambda: started["b"].is_set())
    finally:
        supervisor.request_stop()
        supervisor.join(timeout=0.0)


def test_disabled_loops_are_skipped() -> None:
    ran = threading.Event()

    def target() -> None:
        ran.set()
        while True:
            time.sleep(0.01)

    supervisor = AmbientSupervisor(
        [
            LoopSpec(name="on", target=target, enabled=True, backoff=_tiny_backoff()),
            LoopSpec(
                name="off",
                target=lambda: pytest.fail("disabled loop must not run"),
                enabled=False,
                backoff=_tiny_backoff(),
            ),
        ]
    )
    live = supervisor.start()
    try:
        assert live == ["on"]
        assert _wait_until(lambda: ran.is_set())
    finally:
        supervisor.request_stop()
        supervisor.join(timeout=0.0)


def test_crashing_loop_restarts_while_sibling_keeps_ticking() -> None:
    crash_counter = {"count": 0}
    crash_lock = threading.Lock()
    sibling_ticks = {"count": 0}
    sibling_lock = threading.Lock()

    def crasher() -> None:
        with crash_lock:
            crash_counter["count"] += 1
        # Raise immediately so the supervisor exercises its restart path.
        raise RuntimeError("boom")

    def sibling() -> None:
        while True:
            with sibling_lock:
                sibling_ticks["count"] += 1
            time.sleep(0.005)

    supervisor = AmbientSupervisor(
        [
            LoopSpec(name="crasher", target=crasher, backoff=_tiny_backoff()),
            LoopSpec(name="sibling", target=sibling, backoff=_tiny_backoff()),
        ]
    )
    supervisor.start()
    try:
        # The crasher must restart several times (proving backoff-restart, not
        # a one-shot death).
        assert _wait_until(lambda: crash_counter["count"] >= 3)
        # The sibling keeps advancing throughout the crasher's restarts.
        baseline = sibling_ticks["count"]
        assert _wait_until(lambda: sibling_ticks["count"] > baseline)
    finally:
        supervisor.request_stop()
        supervisor.join(timeout=0.0)


def test_backoff_doubles_and_caps() -> None:
    policy = BackoffPolicy(
        initial_seconds=5.0,
        max_seconds=300.0,
        reset_after_seconds=600.0,
    )
    assert policy.next_delay(None) == 5.0
    assert policy.next_delay(5.0) == 10.0
    assert policy.next_delay(10.0) == 20.0
    # Caps at max_seconds.
    assert policy.next_delay(200.0) == 300.0
    assert policy.next_delay(300.0) == 300.0


def test_backoff_policy_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        BackoffPolicy(initial_seconds=0)
    with pytest.raises(ValueError):
        BackoffPolicy(initial_seconds=10.0, max_seconds=5.0)
    with pytest.raises(ValueError):
        BackoffPolicy(reset_after_seconds=0)


def test_backoff_resets_after_healthy_run() -> None:
    # A loop healthy past reset_after_seconds resets its backoff to initial.
    # We drive this through the supervisor with a fake monotonic clock so the
    # "healthy_for" computation is deterministic and no real time elapses.
    clock = {"now": 0.0}

    def fake_monotonic() -> float:
        return clock["now"]

    run_count = {"count": 0}
    recorded_delays: list[float] = []
    barrier = threading.Event()

    policy = BackoffPolicy(
        initial_seconds=4.0,
        max_seconds=64.0,
        reset_after_seconds=100.0,
    )

    def target() -> None:
        run_count["count"] += 1
        # First run: short (no reset). Subsequent runs advance the clock past
        # reset_after_seconds so backoff resets.
        if run_count["count"] == 1:
            clock["now"] += 1.0  # healthy_for = 1.0 -> below reset threshold
        else:
            clock["now"] += 200.0  # healthy_for >= reset -> reset to initial
        raise RuntimeError("boom")

    spec = LoopSpec(name="x", target=target, backoff=policy)
    supervisor = AmbientSupervisor([spec], monotonic=fake_monotonic)

    # Wrap the stop event's wait to record requested backoff delays, then stop
    # after we've seen a reset.
    real_wait = supervisor.stop_event.wait

    def recording_wait(timeout: float | None = None) -> bool:
        if timeout is not None:
            recorded_delays.append(timeout)
            # After the second crash's delay (which should be reset to
            # initial), release the test.
            if len(recorded_delays) >= 2:
                barrier.set()
        # Do not actually sleep; return immediately so the loop spins fast.
        return supervisor.stop_event.is_set()

    supervisor.stop_event.wait = recording_wait  # type: ignore[method-assign]
    supervisor.start()
    try:
        assert _wait_until(barrier.is_set)
        # First crash after a 1s healthy run: delay = initial (4.0).
        assert recorded_delays[0] == 4.0
        # Second crash after a 200s healthy run: backoff reset, so delay is
        # again the initial 4.0 rather than doubled to 8.0.
        assert recorded_delays[1] == 4.0
    finally:
        supervisor.stop_event.wait = real_wait  # type: ignore[method-assign]
        supervisor.request_stop()
        supervisor.join(timeout=0.0)


def test_returning_target_is_restarted() -> None:
    # A run_forever that returns (rather than raising) is also restarted.
    counter = {"count": 0}

    def target() -> None:
        counter["count"] += 1
        return

    supervisor = AmbientSupervisor(
        [LoopSpec(name="r", target=target, backoff=_tiny_backoff())]
    )
    supervisor.start()
    try:
        assert _wait_until(lambda: counter["count"] >= 3)
    finally:
        supervisor.request_stop()
        supervisor.join(timeout=0.0)


def _settings(**overrides: object):  # type: ignore[no-untyped-def]
    from kortny.config.settings import LLMProvider, Settings

    kwargs: dict[str, object] = {
        "SLACK_BOT_TOKEN": "xoxb-test-token",
        "SLACK_APP_TOKEN": "xapp-test-token",
        "SLACK_SIGNING_SECRET": "test-signing-secret",
        "LLM_PROVIDER": LLMProvider.openrouter,
        "LLM_API_KEY": "test-llm-key",
        "LLM_MODEL": "openai/gpt-5.4-mini",
        "COMPOSIO_API_KEY": "composio-key",
        "POSTGRES_URL": "postgresql://kortny:kortny@localhost:5432/kortny_test",
    }
    kwargs.update(overrides)
    return Settings(**kwargs)  # type: ignore[arg-type]


def test_build_default_loops_includes_composio_sync_when_configured() -> None:
    from kortny.ambient.supervisor import build_default_loops

    loops = {loop.name: loop for loop in build_default_loops(_settings())}
    assert "composio_catalog_sync" in loops
    assert loops["composio_catalog_sync"].enabled is True


def test_build_default_loops_disables_composio_sync_when_catalog_off() -> None:
    from kortny.ambient.supervisor import build_default_loops

    loops = {
        loop.name: loop
        for loop in build_default_loops(_settings(COMPOSIO_CATALOG_ENABLED=False))
    }
    assert loops["composio_catalog_sync"].enabled is False


def test_main_list_loops_exits_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    # The entrypoint's --list-loops path must not start any threads.
    from kortny.ambient import __main__ as ambient_main
    from kortny.ambient.supervisor import LoopSpec as _LoopSpec

    monkeypatch.setattr(ambient_main, "load_settings", lambda: object())
    monkeypatch.setattr(ambient_main, "configure_tracing", lambda settings: None)
    monkeypatch.setattr(
        ambient_main,
        "build_default_loops",
        lambda settings: [
            _LoopSpec(name="scheduler", target=lambda: None, enabled=True),
            _LoopSpec(name="witness", target=lambda: None, enabled=False),
        ],
    )

    started = {"called": False}

    def fail_start(self: AmbientSupervisor) -> list[str]:
        started["called"] = True
        return []

    monkeypatch.setattr(AmbientSupervisor, "start", fail_start)

    ambient_main.main(["--list-loops"])
    assert started["called"] is False
