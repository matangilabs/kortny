"""Unit tests for sandbox container GC (HIG-200).

The GC predicate is pure logic over the Docker container listing plus the set of
live SessionManager container ids; no DB and no real Docker daemon are involved,
so these mock the Docker client.
"""

from __future__ import annotations

from kortny.sandbox_runner.docker_api import DockerSandboxContainer
from kortny.sandbox_runner.gc import (
    SandboxContainerGc,
    select_containers_to_reap,
    should_reap_container,
)

NOW = 1_000_000.0
MAX_AGE = 3600.0  # 60 min
ORPHAN_RUNNING_MAX_AGE = 86_400.0  # 24h


def _container(
    *,
    container_id: str,
    state: str,
    age_seconds: float,
    kind: str = "ephemeral",
    session_id: str = "",
) -> DockerSandboxContainer:
    return DockerSandboxContainer(
        container_id=container_id,
        kind=kind,
        session_id=session_id,
        state=state,
        created_at_epoch=int(NOW - age_seconds),
    )


def _predicate(container: DockerSandboxContainer, live: frozenset[str]) -> bool:
    return should_reap_container(
        container,
        now_epoch=NOW,
        max_age_seconds=MAX_AGE,
        orphan_running_max_age_seconds=ORPHAN_RUNNING_MAX_AGE,
        live_container_ids=live,
    )


class FakeDockerGcClient:
    def __init__(self, listed: list[DockerSandboxContainer]) -> None:
        self.listed = listed
        self.removed: list[str] = []
        self.remove_errors: dict[str, str] = {}

    def list_sandbox_containers(self) -> tuple[DockerSandboxContainer, ...]:
        return tuple(self.listed)

    def remove_session_container(self, container_id: str) -> str | None:
        error = self.remove_errors.get(container_id)
        if error is not None:
            return error
        self.removed.append(container_id)
        return None


def test_reaps_old_exited_ephemeral_container() -> None:
    container = _container(
        container_id="leaked-exited", state="exited", age_seconds=MAX_AGE + 1
    )
    assert _predicate(container, frozenset()) is True


def test_keeps_recent_exited_container() -> None:
    container = _container(
        container_id="recent-exited", state="exited", age_seconds=MAX_AGE - 1
    )
    assert _predicate(container, frozenset()) is False


def test_reaps_old_created_and_dead_containers() -> None:
    for state in ("created", "dead"):
        container = _container(
            container_id=f"old-{state}", state=state, age_seconds=MAX_AGE + 10
        )
        assert _predicate(container, frozenset()) is True


def test_never_reaps_live_session_container_even_when_ancient() -> None:
    container = _container(
        container_id="live-session",
        state="running",
        age_seconds=ORPHAN_RUNNING_MAX_AGE + 100_000,
        kind="session",
        session_id="abc",
    )
    assert _predicate(container, frozenset({"live-session"})) is False


def test_never_reaps_live_session_container_in_terminal_state() -> None:
    # Even an exited container is protected if the live manager still owns it
    # (e.g. it crashed but the record has not been pruned yet).
    container = _container(
        container_id="live-but-exited",
        state="exited",
        age_seconds=MAX_AGE + 100_000,
        kind="session",
        session_id="abc",
    )
    assert _predicate(container, frozenset({"live-but-exited"})) is False


def test_keeps_running_container_below_orphan_threshold() -> None:
    # A running, unowned container younger than the big orphan threshold is left
    # alone — it could be a session this process simply has not adopted yet.
    container = _container(
        container_id="running-young",
        state="running",
        age_seconds=ORPHAN_RUNNING_MAX_AGE - 1,
    )
    assert _predicate(container, frozenset()) is False


def test_reaps_orphaned_running_container_over_large_threshold() -> None:
    container = _container(
        container_id="running-orphan",
        state="running",
        age_seconds=ORPHAN_RUNNING_MAX_AGE + 1,
    )
    assert _predicate(container, frozenset()) is True


def test_unknown_state_only_reaped_when_ancient() -> None:
    young = _container(
        container_id="restarting-young",
        state="restarting",
        age_seconds=ORPHAN_RUNNING_MAX_AGE - 1,
    )
    old = _container(
        container_id="restarting-old",
        state="restarting",
        age_seconds=ORPHAN_RUNNING_MAX_AGE + 1,
    )
    assert _predicate(young, frozenset()) is False
    assert _predicate(old, frozenset()) is True


def test_select_filters_mixed_listing() -> None:
    containers = [
        _container(container_id="a", state="exited", age_seconds=MAX_AGE + 1),
        _container(container_id="b", state="exited", age_seconds=MAX_AGE - 1),
        _container(
            container_id="c",
            state="running",
            age_seconds=ORPHAN_RUNNING_MAX_AGE + 1,
        ),
        _container(
            container_id="live",
            state="running",
            age_seconds=ORPHAN_RUNNING_MAX_AGE + 1,
            kind="session",
        ),
    ]
    reapable = select_containers_to_reap(
        containers,
        now_epoch=NOW,
        max_age_seconds=MAX_AGE,
        orphan_running_max_age_seconds=ORPHAN_RUNNING_MAX_AGE,
        live_container_ids=frozenset({"live"}),
    )
    assert {c.container_id for c in reapable} == {"a", "c"}


def test_sweep_removes_eligible_and_reports_ids() -> None:
    client = FakeDockerGcClient(
        [
            _container(container_id="old-a", state="exited", age_seconds=MAX_AGE + 5),
            _container(container_id="old-b", state="dead", age_seconds=MAX_AGE + 5),
            _container(container_id="fresh", state="exited", age_seconds=10),
        ]
    )
    gc = SandboxContainerGc(
        docker_client=client,
        max_age_seconds=MAX_AGE,
        interval_seconds=600,
        orphan_running_max_age_seconds=ORPHAN_RUNNING_MAX_AGE,
        clock=lambda: NOW,
    )
    removed = gc.sweep()
    assert sorted(removed) == ["old-a", "old-b"]
    assert sorted(client.removed) == ["old-a", "old-b"]


def test_sweep_protects_live_session_ids_via_callable() -> None:
    client = FakeDockerGcClient(
        [
            _container(
                container_id="live-session",
                state="running",
                age_seconds=ORPHAN_RUNNING_MAX_AGE + 100,
                kind="session",
            ),
            _container(
                container_id="orphan-session",
                state="running",
                age_seconds=ORPHAN_RUNNING_MAX_AGE + 100,
                kind="session",
            ),
        ]
    )
    gc = SandboxContainerGc(
        docker_client=client,
        max_age_seconds=MAX_AGE,
        interval_seconds=600,
        orphan_running_max_age_seconds=ORPHAN_RUNNING_MAX_AGE,
        live_container_ids=lambda: frozenset({"live-session"}),
        clock=lambda: NOW,
    )
    removed = gc.sweep()
    assert removed == ["orphan-session"]
    assert client.removed == ["orphan-session"]


def test_sweep_does_not_count_failed_removals() -> None:
    client = FakeDockerGcClient(
        [
            _container(container_id="ok", state="exited", age_seconds=MAX_AGE + 5),
            _container(container_id="stuck", state="exited", age_seconds=MAX_AGE + 5),
        ]
    )
    client.remove_errors["stuck"] = "409 Conflict: removal in progress"
    gc = SandboxContainerGc(
        docker_client=client,
        max_age_seconds=MAX_AGE,
        interval_seconds=600,
        orphan_running_max_age_seconds=ORPHAN_RUNNING_MAX_AGE,
        clock=lambda: NOW,
    )
    removed = gc.sweep()
    assert removed == ["ok"]
    assert client.removed == ["ok"]
