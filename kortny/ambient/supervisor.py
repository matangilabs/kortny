"""Thread supervisor for Kortny's ambient poller loops (HIG-234).

Each ambient loop (scheduler, witness, consolidator) is an existing
``run_forever`` poller. The supervisor hosts each enabled loop in its own
thread and isolates crashes: an exception escaping one loop is logged with a
traceback and that loop is restarted with exponential backoff, while sibling
loops keep ticking untouched.

Threading model
---------------
The loops are sleep-based pollers that spend almost all their time in
``time.sleep`` between Postgres polls, so a thread-per-loop model (rather than
asyncio) reuses their synchronous ``run_forever`` bodies verbatim — no loop
logic is reimplemented here. Worker threads are daemon threads; the main
thread parks on a ``stop`` event and a SIGTERM/SIGINT handler sets it. On
shutdown the supervisor signals every loop and gives the threads a short join
window; because the loops are sleep-bound and all mutating work is guarded by
short DB transactions + advisory locks, leaving any still-sleeping daemon
thread to be torn down with the process is safe.

Backoff
-------
Per loop, a crash restarts after ``initial_seconds`` (default 5s), doubling on
each consecutive crash up to ``max_seconds`` (default 300s). A loop that stays
healthy for ``reset_after_seconds`` (default 600s) resets its backoff to the
initial delay, so transient blips don't permanently inflate the restart delay.
Backoff sleeps wait on the ``stop`` event, so shutdown stays responsive even
mid-backoff.
"""

from __future__ import annotations

import logging
import signal
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from types import FrameType

from kortny.config import Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BackoffPolicy:
    """Exponential backoff with a healthy-run reset for one loop."""

    initial_seconds: float = 5.0
    max_seconds: float = 300.0
    reset_after_seconds: float = 600.0

    def __post_init__(self) -> None:
        if self.initial_seconds <= 0:
            raise ValueError("initial_seconds must be positive")
        if self.max_seconds < self.initial_seconds:
            raise ValueError("max_seconds must be >= initial_seconds")
        if self.reset_after_seconds <= 0:
            raise ValueError("reset_after_seconds must be positive")

    def next_delay(self, current: float | None) -> float:
        """Delay to wait after a crash given the previous delay (or ``None``)."""

        if current is None:
            return self.initial_seconds
        return min(current * 2.0, self.max_seconds)


@dataclass(frozen=True, slots=True)
class LoopSpec:
    """One supervised loop: a name, an enable flag, and a blocking target.

    ``target`` is the existing ``run_forever`` poller (or any callable that
    blocks until it raises or returns); it is invoked with no arguments. The
    supervisor passes nothing into it, so per-loop configuration is bound by
    the caller (e.g. via ``functools.partial`` or a closure).
    """

    name: str
    target: Callable[[], None]
    enabled: bool = True
    backoff: BackoffPolicy = field(default_factory=BackoffPolicy)


class AmbientSupervisor:
    """Host a set of loops as supervised, crash-isolated daemon threads."""

    def __init__(
        self,
        loops: Sequence[LoopSpec],
        *,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._loops = tuple(loops)
        self._monotonic = monotonic
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    @property
    def stop_event(self) -> threading.Event:
        return self._stop

    def _supervise_loop(self, spec: LoopSpec) -> None:
        """Run one loop forever, restarting it with backoff on any crash."""

        delay: float | None = None
        while not self._stop.is_set():
            started_at = self._monotonic()
            try:
                spec.target()
            except Exception:
                logger.exception(
                    "ambient loop %r crashed; will restart with backoff",
                    spec.name,
                )
            else:
                # A run_forever target should never return; if one does, treat
                # it like a crash so the loop is restarted rather than silently
                # going dark.
                logger.warning(
                    "ambient loop %r returned unexpectedly; restarting",
                    spec.name,
                )
            if self._stop.is_set():
                break
            healthy_for = self._monotonic() - started_at
            if healthy_for >= spec.backoff.reset_after_seconds:
                delay = None
            delay = spec.backoff.next_delay(delay)
            logger.info(
                "ambient loop %r restarting in %.1fs (healthy_for=%.1fs)",
                spec.name,
                delay,
                healthy_for,
            )
            # Wait on the stop event so shutdown interrupts the backoff sleep.
            if self._stop.wait(timeout=delay):
                break

    def start(self) -> list[str]:
        """Start a thread per enabled loop; return the live loop names."""

        live: list[str] = []
        for spec in self._loops:
            if not spec.enabled:
                logger.info("ambient loop %r disabled; skipping", spec.name)
                continue
            thread = threading.Thread(
                target=self._supervise_loop,
                args=(spec,),
                name=f"ambient-{spec.name}",
                daemon=True,
            )
            thread.start()
            self._threads.append(thread)
            live.append(spec.name)
        return live

    def request_stop(self) -> None:
        self._stop.set()

    def join(self, timeout: float | None = None) -> None:
        """Block the caller until stop is requested, then drain threads.

        With no ``timeout`` this parks on the stop event indefinitely (the
        service entrypoint). With a ``timeout`` it waits that long for the
        stop event — used by tests that drive shutdown explicitly.
        """

        self._stop.wait(timeout=timeout)
        for thread in self._threads:
            # Short join: loops are sleep-bound daemon threads, so any that are
            # mid-sleep are torn down with the process. Mutating work is short
            # and advisory-lock guarded, so this is safe.
            thread.join(timeout=2.0)

    def install_signal_handlers(self) -> None:
        """Wire SIGTERM/SIGINT to request a graceful stop (main thread only)."""

        def _handle(signum: int, _frame: FrameType | None) -> None:
            logger.info("ambient received signal %s; stopping loops", signum)
            self.request_stop()

        signal.signal(signal.SIGTERM, _handle)
        signal.signal(signal.SIGINT, _handle)


def build_default_loops(settings: Settings) -> list[LoopSpec]:
    """Build the scheduler/witness/consolidator loop specs from settings.

    Each spec wraps the existing worker's ``run_forever`` so loop logic is
    reused, not reimplemented. The scheduler has no enable flag (always on);
    witness and consolidator honor their existing settings flags.
    """

    # Imported lazily so importing the supervisor (e.g. in tests) does not pull
    # in the heavy worker dependency graphs.
    from kortny.composio.catalog_sync import ComposioCatalogSyncWorker
    from kortny.consolidator.runner import ConsolidatorWorker
    from kortny.scheduler.service import SchedulerWorker
    from kortny.witness.runner import WitnessWorker

    def _scheduler() -> None:
        SchedulerWorker(
            poll_interval_seconds=settings.scheduler_poll_interval_seconds,
            materialize_limit=settings.scheduler_materialize_limit,
            advisory_lock_key=settings.scheduler_advisory_lock_key,
        ).run_forever()

    def _witness() -> None:
        WitnessWorker(
            settings=settings,
            poll_interval_seconds=settings.witness_poll_interval_seconds,
            profile_limit=settings.witness_profile_scan_limit,
            delivery_limit=settings.witness_delivery_limit,
            scan_interval=timedelta(seconds=settings.witness_scan_interval_seconds),
            deliver_private=settings.witness_deliver_private,
        ).run_forever()

    def _consolidator() -> None:
        ConsolidatorWorker(
            settings=settings,
            poll_interval_seconds=settings.consolidator_poll_interval_seconds,
        ).run_forever()

    def _composio_catalog_sync() -> None:
        ComposioCatalogSyncWorker(
            settings=settings,
            poll_interval_seconds=settings.composio_sync_interval_hours * 3600.0,
            advisory_lock_key=settings.composio_sync_advisory_lock_key,
        ).run_forever()

    return [
        LoopSpec(name="scheduler", target=_scheduler, enabled=True),
        LoopSpec(
            name="witness",
            target=_witness,
            enabled=settings.witness_enabled,
        ),
        LoopSpec(
            name="consolidator",
            target=_consolidator,
            enabled=settings.consolidator_enabled,
        ),
        LoopSpec(
            name="composio_catalog_sync",
            target=_composio_catalog_sync,
            enabled=_composio_configured(settings),
        ),
    ]


def _composio_configured(settings: Settings) -> bool:
    """Composio sync runs only when Composio is configured + catalog enabled."""

    return bool(settings.composio_api_key) and settings.composio_catalog_enabled
