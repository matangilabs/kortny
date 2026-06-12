"""Sandbox container garbage collection for the runner service.

The sandbox-runner spins up two kinds of ``kortny.sandbox=true``-labeled
containers: ephemeral one-shot ``/run`` containers (auto-removed on the happy
path) and long-lived workbench session containers tracked in-process by the
:class:`~kortny.sandbox_runner.sessions.SessionManager`. Both can leak:

* an ephemeral container leaks if the runner crashes between create and the
  ``finally`` remove, or if the remove call itself errors;
* a session container leaks if the runner process is replaced (deploy/restart)
  while a container is still up and no operator ever closes the session.

The session reaper already prunes idle/expired sessions it knows about, but it
only sees containers the *current* process tracks. This GC is the wider safety
net: it sweeps every sandbox-labeled container, protects the ones the live
SessionManager still owns, and reaps the rest by age + state.

The reaping predicate is pure (:func:`should_reap_container`) so it is unit
testable with a mocked Docker client and no real daemon.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Callable, Iterable

from kortny.sandbox_runner.docker_api import (
    DockerApiGcClient,
    DockerSandboxContainer,
)

# Docker container States that mean "not doing any work" — always safe to reap
# once old enough, since nothing live can be running in them.
_TERMINAL_STATES = frozenset({"exited", "created", "dead"})


def should_reap_container(
    container: DockerSandboxContainer,
    *,
    now_epoch: float,
    max_age_seconds: float,
    orphan_running_max_age_seconds: float,
    live_container_ids: frozenset[str],
) -> bool:
    """Return whether one sandbox container is eligible for GC removal.

    Predicate (pure; the safe, conservative choice):

    * Never reap a container the live SessionManager still owns
      (``live_container_ids``) — that is an in-use workbench, regardless of age.
    * Terminal containers (exited/created/dead) are reaped once their age
      exceeds ``max_age_seconds`` — nothing is running in them.
    * Running containers are only reaped when they are *orphaned* (not owned by
      the live SessionManager) AND far older than the much larger
      ``orphan_running_max_age_seconds`` — this catches a container leaked by a
      previous runner process while never touching an active session whose
      record the current process still holds.
    * Anything else (recent, or running-and-young) is left alone.
    """

    if container.container_id in live_container_ids:
        return False

    age_seconds = now_epoch - float(container.created_at_epoch)
    state = container.state.strip().lower()

    if state in _TERMINAL_STATES:
        return age_seconds > max_age_seconds

    if state == "running":
        return age_seconds > orphan_running_max_age_seconds

    # Unknown/transitional states (restarting, paused, removing, ...): only
    # reap if they are clearly ancient under the orphan threshold, to avoid
    # racing a container that is mid-startup.
    return age_seconds > orphan_running_max_age_seconds


def select_containers_to_reap(
    containers: Iterable[DockerSandboxContainer],
    *,
    now_epoch: float,
    max_age_seconds: float,
    orphan_running_max_age_seconds: float,
    live_container_ids: frozenset[str],
) -> list[DockerSandboxContainer]:
    """Filter a container listing down to the GC-eligible ones."""

    return [
        container
        for container in containers
        if should_reap_container(
            container,
            now_epoch=now_epoch,
            max_age_seconds=max_age_seconds,
            orphan_running_max_age_seconds=orphan_running_max_age_seconds,
            live_container_ids=live_container_ids,
        )
    ]


class SandboxContainerGc:
    """Sweeps leaked sandbox-labeled containers on a daemon thread.

    The GC is intentionally decoupled from the SessionManager: it takes a
    callable that returns the currently-live container ids so it can protect
    in-use workbench sessions without importing session internals.
    """

    def __init__(
        self,
        *,
        docker_client: DockerApiGcClient,
        max_age_seconds: float,
        interval_seconds: float,
        orphan_running_max_age_seconds: float,
        live_container_ids: Callable[[], frozenset[str]] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._docker_client = docker_client
        self._max_age_seconds = max_age_seconds
        self._interval_seconds = interval_seconds
        self._orphan_running_max_age_seconds = orphan_running_max_age_seconds
        self._live_container_ids = live_container_ids or (lambda: frozenset())
        self._clock = clock

    def sweep(self) -> list[str]:
        """Run one GC pass; return the ids of removed containers.

        Crash-isolated by the caller's loop, but also defensive here: a Docker
        listing error returns an empty tuple from the client, so a transient
        daemon hiccup simply yields an empty sweep rather than raising.
        """

        live = self._live_container_ids()
        containers = self._docker_client.list_sandbox_containers()
        reapable = select_containers_to_reap(
            containers,
            now_epoch=self._clock(),
            max_age_seconds=self._max_age_seconds,
            orphan_running_max_age_seconds=self._orphan_running_max_age_seconds,
            live_container_ids=live,
        )
        removed: list[str] = []
        for container in reapable:
            error = self._docker_client.remove_session_container(container.container_id)
            if error is None:
                removed.append(container.container_id)
        return removed


def gc_loop(
    gc: SandboxContainerGc,
    stop_event: threading.Event,
    *,
    interval_seconds: float,
    run_startup_sweep: bool = True,
) -> None:
    """Daemon-thread body: startup sweep then a periodic re-sweep.

    Each sweep is crash-isolated — a Docker hiccup logs nothing and continues so
    the loop survives daemon restarts and transient socket errors.
    """

    if run_startup_sweep:
        # GC must survive Docker hiccups; a transient daemon error skips one
        # sweep rather than killing the loop.
        with contextlib.suppress(Exception):
            gc.sweep()
    while not stop_event.wait(interval_seconds):
        with contextlib.suppress(Exception):
            gc.sweep()
