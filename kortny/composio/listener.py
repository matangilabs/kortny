"""Outbound Composio trigger listener — the realtime SDK ingress (Slice C1).

Self-hosted Kortny stacks have no public URL, so trigger delivery cannot use
Composio's webhook mechanism in this slice. Instead this module subscribes
outbound to Composio's private realtime channel via the ``composio`` Python
SDK (``composio.triggers.subscribe()``), the same "phone home" shape as
Slack Socket Mode. There is no HTTP route here.

SDK transport (pinned by reading the installed ``composio==0.17.1`` source,
``composio/core/models/triggers.py``):

* ``Composio(api_key=...).triggers.subscribe(timeout=15.0) -> TriggerSubscription``
  authenticates against ``{base_url}/api/v3/internal/sdk/realtime/auth`` and
  opens a Pusher (websocket, via the ``pysher`` dependency) connection to a
  private per-project channel (``private-{project_id}_triggers``). The call
  blocks only until the socket handshake completes (bounded by ``timeout``;
  raises ``ComposioSDKTimeoutError`` past it) — it does not block forever.
* Filtering is client-side: ``subscription.handle(**filters)`` is a decorator
  that registers a callback against a filter set (``trigger_slug``,
  ``trigger_id``, ``toolkit``, ``user_id``, ``auth_config_id``,
  ``connected_account_id``, all optional). Every project trigger event
  arrives on the single channel; multiple callbacks may be registered on one
  subscription. There is no server-side per-trigger subscribe call for the
  socket — REST ``create_trigger``/``disable_trigger`` (``kortny/composio/
  client.py``) remain the only way to turn a given trigger on/off upstream.
* ``subscription.is_alive()`` / ``has_errored()`` are point-in-time checks;
  ``wait_forever()`` polls them in an uninterruptible ``while`` loop, which
  is exactly the "silent death" shape this module avoids — we run our own
  bounded poll instead (see ``_run_connected_session``).
* Payload shape delivered to a ``.handle()`` callback is the SDK's
  ``TriggerEvent`` TypedDict: ``id`` (str), ``uuid`` (str), ``user_id`` (str),
  ``trigger_slug`` (str), ``toolkit_slug`` (str), ``payload`` (dict),
  ``metadata`` (dict, with a nested ``connected_account`` dict carrying
  ``id``), ``original_payload`` (dict | None).

KNOWN SDK LIMITATION (load-bearing for the adapter below): the realtime
``TriggerEvent.id`` is normalized to the trigger INSTANCE id (``ti_*``,
constant per subscription), not a unique per-message id — the V3 envelope's
top-level ``id``/``log_id`` fields are read by the SDK's Pusher frame parser
but are not surfaced through the public ``.handle()`` callback. Dedup keyed
on ``TriggerEvent["id"]`` directly would collapse every distinct event on one
trigger into a single row. ``adapt_sdk_event`` below works around this by
synthesizing a content-hash dedup id from ``(trigger_id, connected_account_id,
canonical_payload)`` instead — a real, distinct event still gets its own row;
an identical re-delivery (e.g. a websocket hiccup) still collapses to one.

Reviewed-and-accepted tradeoff of that content-hash scheme: two genuinely
distinct events whose ``(trigger_id, connected_account_id, payload)`` triple
is byte-identical collapse into one row by design. The three launch triggers'
payloads all carry distinguishing fields (PR/action metadata, message ids,
event start times), so an exact byte-level repeat is overwhelmingly a
re-delivery, not a new event — and because this slice is shadow-mode only,
the recorded event stream will surface any real-world collapse pattern before
delivery is ever turned on.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import CursorResult, create_engine, delete, func, select
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool

from kortny.composio.client import ParsedTriggerEvent
from kortny.composio.trigger_ingress import ingest_trigger_event
from kortny.composio.triggers import LAUNCH_TRIGGERS
from kortny.config import Settings
from kortny.db.models import (
    ComposioTriggerEvent,
    ComposioTriggerSubscription,
    Installation,
)
from kortny.db.session import make_session_factory, normalize_database_url

logger = logging.getLogger(__name__)

# Serialized payload size cap — an oversize or malformed frame is dropped and
# logged rather than persisted (untrusted socket boundary, no HMAC available
# on this transport the way the REST webhook path has one).
MAX_PAYLOAD_BYTES = 262_144

_DEFAULT_CONNECT_TIMEOUT_SECONDS = 20.0
_RETENTION_CHECK_INTERVAL_SECONDS = 3600.0
_INITIAL_BACKOFF_SECONDS = 2.0
# A session that stays healthy at least this long resets the reconnect
# backoff to its initial value, so one bad patch doesn't inflate every
# subsequent reconnect for the life of the process.
_BACKOFF_RESET_AFTER_SECONDS = 600.0


def adapt_sdk_event(raw: Any) -> ParsedTriggerEvent | None:
    """Validate + normalize one raw SDK ``TriggerEvent`` into a ``ParsedTriggerEvent``.

    Defensive by construction: no exception escapes this function. Any
    unrecognized or malformed shape is logged and dropped (returns ``None``)
    so one bad frame can never break the listener loop.
    """

    try:
        return _adapt_sdk_event_unsafe(raw)
    except Exception:
        logger.exception(
            "composio listener: unexpected error adapting SDK event; dropping"
        )
        return None


def _adapt_sdk_event_unsafe(raw: Any) -> ParsedTriggerEvent | None:
    if not isinstance(raw, dict):
        logger.warning("composio listener: dropping non-mapping SDK event frame")
        return None

    trigger_slug = raw.get("trigger_slug")
    if not isinstance(trigger_slug, str) or not trigger_slug.strip():
        logger.warning(
            "composio listener: dropping SDK event with missing trigger_slug"
        )
        return None

    trigger_id = raw.get("id")
    if not isinstance(trigger_id, str) or not trigger_id.strip():
        logger.warning(
            "composio listener: dropping SDK event with missing id trigger_slug=%s",
            trigger_slug,
        )
        return None

    payload = raw.get("payload")
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        logger.warning(
            "composio listener: dropping SDK event with non-dict payload trigger_slug=%s",
            trigger_slug,
        )
        return None

    try:
        # No `default=` coercion: a payload straight off the websocket JSON
        # frame is already JSON-primitive-only, so anything that fails to
        # serialize here is a genuinely malformed/unknown shape to drop, not
        # a legitimate value (e.g. a datetime) to coerce.
        canonical_payload = json.dumps(payload, sort_keys=True)
    except (TypeError, ValueError):
        logger.warning(
            "composio listener: dropping SDK event with non-serializable payload "
            "trigger_slug=%s",
            trigger_slug,
        )
        return None

    payload_bytes = len(canonical_payload.encode("utf-8"))
    if payload_bytes > MAX_PAYLOAD_BYTES:
        logger.warning(
            "composio listener: dropping oversize SDK event payload trigger_slug=%s bytes=%d",
            trigger_slug,
            payload_bytes,
        )
        return None

    metadata = raw.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    connected_account = metadata.get("connected_account")
    connected_account = connected_account if isinstance(connected_account, dict) else {}
    connected_account_id = _clean_str(connected_account.get("id"))
    user_id = _clean_str(raw.get("user_id"))

    dedup_id = hashlib.sha256(
        f"{trigger_id}:{connected_account_id}:{canonical_payload}".encode()
    ).hexdigest()

    return ParsedTriggerEvent(
        id=dedup_id,
        type="composio.trigger.message",
        trigger_slug=trigger_slug,
        trigger_id=trigger_id,
        connected_account_id=connected_account_id,
        user_id=user_id,
        data=payload,
        timestamp=datetime.now(UTC).isoformat(),
    )


def _clean_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _resolve_installation_id(
    session: Session, parsed: ParsedTriggerEvent
) -> uuid.UUID | None:
    """Resolve which installation a realtime SDK event belongs to.

    The SDK's Pusher channel is project-wide (one channel per Composio
    project), so unlike the REST webhook path — which is already dispatched
    against a specific installation — realtime events arrive with no
    installation context. Resolve it with the same precision as
    ``ingest_trigger_event``'s own subscription routing (trigger-id first,
    connected-account fallback) but without an installation filter, so two
    installations sharing one Composio project can't silently
    cross-attribute an event to the wrong tenant.
    """

    candidates: set[uuid.UUID] = set()
    if parsed.trigger_id:
        candidates = set(
            session.execute(
                select(ComposioTriggerSubscription.installation_id).where(
                    ComposioTriggerSubscription.composio_trigger_id
                    == parsed.trigger_id,
                    ComposioTriggerSubscription.status == "active",
                )
            ).scalars()
        )
    if not candidates and parsed.connected_account_id:
        candidates = set(
            session.execute(
                select(ComposioTriggerSubscription.installation_id).where(
                    ComposioTriggerSubscription.connected_account_id
                    == parsed.connected_account_id,
                    ComposioTriggerSubscription.trigger_slug == parsed.trigger_slug,
                    ComposioTriggerSubscription.status == "active",
                )
            ).scalars()
        )
    if len(candidates) == 1:
        return next(iter(candidates))
    if len(candidates) > 1:
        logger.warning(
            "composio listener: event matches subscriptions across multiple "
            "installations; dropping trigger_id=%s connected_account_id=%s",
            parsed.trigger_id,
            parsed.connected_account_id,
        )
        return None

    # No subscription anywhere matched this event. Fall back to the single
    # self-hosted installation (the overwhelmingly common deployment shape)
    # so an unmatched event still gets an auditable, metadata-only row.
    installation_ids = list(session.execute(select(Installation.id)).scalars())
    if len(installation_ids) == 1:
        return installation_ids[0]
    logger.warning(
        "composio listener: cannot resolve installation for unmatched event "
        "(%d installations on file); dropping trigger_slug=%s",
        len(installation_ids),
        parsed.trigger_slug,
    )
    return None


def _next_backoff(current: float | None, *, max_seconds: float) -> float:
    base = (
        _INITIAL_BACKOFF_SECONDS if current is None else min(current * 2.0, max_seconds)
    )
    jitter = base * random.uniform(0.0, 0.25)
    return min(base + jitter, max_seconds)


def _default_composio_factory(api_key: str) -> Any:
    # Imported lazily: importing the SDK pulls in pysher/websocket-client and
    # is only needed once the listener actually starts (mirrors the lazy
    # imports already used by kortny.ambient.supervisor.build_default_loops).
    from composio import Composio

    return Composio(api_key=api_key)


@dataclass
class _ListenerState:
    last_connected_at: float | None = None
    last_event_at: float | None = None


class ComposioTriggerListener:
    """Supervised loop body: subscribe, poll health, ingest, reconnect."""

    def __init__(
        self,
        *,
        settings: Settings,
        session_factory: sessionmaker[Session] | None = None,
        lock_engine: Engine | None = None,
        composio_factory: Callable[[str], Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        connect_timeout_seconds: float = _DEFAULT_CONNECT_TIMEOUT_SECONDS,
    ) -> None:
        self._settings = settings
        self._session_factory = session_factory or make_session_factory()
        # Dedicated, non-pooled connection for the leader-election advisory
        # lock: session-level pg_advisory_lock is connection-bound, and this
        # lock must be held for the whole socket-connection lifetime, not
        # swapped out from under a pool.
        self._lock_engine = lock_engine or create_engine(
            normalize_database_url(settings.postgres_url), poolclass=NullPool
        )
        self._composio_factory = composio_factory or _default_composio_factory
        self._clock = clock
        self._sleep = sleep
        self._connect_timeout_seconds = connect_timeout_seconds
        self._lock_key = settings.composio_listener_advisory_lock_key
        self._health_tick_seconds = settings.composio_listener_health_tick_seconds
        self._stale_reconnect_seconds = (
            settings.composio_listener_stale_reconnect_seconds
        )
        self._reconnect_max_seconds = settings.composio_listener_reconnect_max_seconds
        self._retention_days = settings.composio_trigger_retention_days
        self._last_retention_run: float | None = None
        self._state = _ListenerState()

    # -- leader election -----------------------------------------------

    def _try_acquire_lock(self, conn: Connection) -> bool:
        acquired = conn.execute(
            select(func.pg_try_advisory_lock(self._lock_key))
        ).scalar()
        return bool(acquired)

    def _release_lock(self, conn: Connection) -> None:
        try:
            conn.execute(select(func.pg_advisory_unlock(self._lock_key)))
        except Exception:
            logger.exception("composio listener: failed to release advisory lock")

    # -- retention --------------------------------------------------------

    def _maybe_run_retention(self) -> None:
        now = self._clock()
        if (
            self._last_retention_run is not None
            and now - self._last_retention_run < _RETENTION_CHECK_INTERVAL_SECONDS
        ):
            return
        self._last_retention_run = now
        cutoff = datetime.now(UTC) - timedelta(days=self._retention_days)
        try:
            with self._session_factory() as session:
                result = session.execute(
                    delete(ComposioTriggerEvent).where(
                        ComposioTriggerEvent.received_at < cutoff
                    )
                )
                session.commit()
                if isinstance(result, CursorResult) and result.rowcount:
                    logger.info(
                        "composio listener retention: deleted %d event rows "
                        "older than %d days",
                        result.rowcount,
                        self._retention_days,
                    )
        except Exception:
            logger.exception("composio listener: retention sweep failed; continuing")

    # -- event handling ----------------------------------------------------

    def _handle_raw_event(self, raw: Any) -> None:
        # Never let one bad event tear down the subscription or the health
        # loop — every branch below is guarded.
        self._state.last_event_at = self._clock()
        try:
            parsed = adapt_sdk_event(raw)
            if parsed is None:
                return
            with self._session_factory() as session:
                installation_id = _resolve_installation_id(session, parsed)
                if installation_id is None:
                    return
                ingest_trigger_event(
                    session,
                    installation_id=installation_id,
                    parsed=parsed,
                    shadow=True,
                )
                session.commit()
        except Exception:
            logger.exception(
                "composio listener: failed to ingest a trigger event; dropping"
            )

    # -- connection lifecycle ----------------------------------------------

    def _run_connected_session(self) -> None:
        """Subscribe, register handlers, and poll health until reconnect is needed.

        Never blocks indefinitely inside the SDK: the SDK connect call is
        bounded by ``connect_timeout_seconds`` and every subsequent wait is
        our own bounded ``health_tick_seconds`` sleep, so a health tick (and
        the retention sweep) runs on a steady cadence for the life of the
        connection.
        """

        composio: Any = self._composio_factory(self._settings.composio_api_key)
        try:
            subscription: Any = composio.triggers.subscribe(
                timeout=self._connect_timeout_seconds
            )
        except Exception:
            logger.exception("composio listener state=connect_failed")
            return

        now = self._clock()
        self._state = _ListenerState(last_connected_at=now, last_event_at=None)
        logger.info("composio listener state=connected")

        for slug in LAUNCH_TRIGGERS:
            subscription.handle(trigger_slug=slug)(self._handle_raw_event)

        try:
            while True:
                self._sleep(self._health_tick_seconds)
                self._maybe_run_retention()

                if subscription.has_errored():
                    logger.warning(
                        "composio listener state=errored transition=reconnect"
                    )
                    return
                if not subscription.is_alive():
                    logger.warning(
                        "composio listener state=not_alive transition=reconnect"
                    )
                    return

                last_activity = max(
                    self._state.last_connected_at or 0.0,
                    self._state.last_event_at or 0.0,
                )
                idle_for = self._clock() - last_activity
                if idle_for > self._stale_reconnect_seconds:
                    logger.warning(
                        "composio listener state=stale idle_seconds=%.0f transition=reconnect",
                        idle_for,
                    )
                    return
        finally:
            try:
                subscription.stop()
            except Exception:
                logger.exception("composio listener: error stopping subscription")
            logger.info("composio listener state=disconnected")

    def run_forever(self) -> None:
        """Supervised-loop body: acquire leadership, run one connected session, repeat.

        Never blocks forever inside a single call — reconnect attempts loop
        with bounded sleeps and exponential backoff (capped, jittered). Only
        an unhandled exception (e.g. the lock connection itself dying) exits
        this method, at which point the ambient supervisor restarts it with
        its own backoff and a fresh lock connection.
        """

        backoff: float | None = None
        lock_conn = self._lock_engine.connect().execution_options(
            isolation_level="AUTOCOMMIT"
        )
        try:
            while True:
                if not self._try_acquire_lock(lock_conn):
                    # Another replica is leader; park cheaply and retry.
                    self._sleep(self._health_tick_seconds)
                    continue
                started_at = self._clock()
                try:
                    self._run_connected_session()
                finally:
                    self._release_lock(lock_conn)
                healthy_for = self._clock() - started_at
                if healthy_for >= _BACKOFF_RESET_AFTER_SECONDS:
                    backoff = None
                backoff = _next_backoff(
                    backoff, max_seconds=self._reconnect_max_seconds
                )
                self._sleep(backoff)
        finally:
            lock_conn.close()


def build_listener_loop(settings: Settings) -> Callable[[], None]:
    """Build the ``run_forever`` target for the ambient supervisor's LoopSpec."""

    def _run() -> None:
        ComposioTriggerListener(settings=settings).run_forever()

    return _run


def composio_triggers_configured(settings: Settings) -> bool:
    """Gate: the listener loop only starts when the flag is on and a key is set."""

    return bool(settings.composio_triggers_enabled) and bool(settings.composio_api_key)
