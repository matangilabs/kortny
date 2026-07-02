"""Tests for the Composio realtime trigger listener (kortny/composio/listener.py).

``adapt_sdk_event`` tests are pure-function (no DB). The end-to-end tests drive
``ComposioTriggerListener`` against real Postgres with a FAKE subscription
source standing in for the Composio SDK's ``TriggerSubscription`` — the SDK
itself is never imported or contacted in tests.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, delete, select
from sqlalchemy.orm import Session, sessionmaker

from kortny.composio.listener import (
    MAX_PAYLOAD_BYTES,
    ComposioTriggerListener,
    adapt_sdk_event,
)
from kortny.config import Settings
from kortny.db.models import (
    ComposioConnection,
    ComposioTriggerEvent,
    ComposioTriggerSubscription,
    Installation,
)
from kortny.db.session import make_engine, make_session_factory, normalize_database_url

TEST_POSTGRES_URL = os.environ.get("KORTNY_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    TEST_POSTGRES_URL is None,
    reason="KORTNY_TEST_POSTGRES_URL is required for listener tests",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def engine() -> Iterator[Engine]:
    assert TEST_POSTGRES_URL is not None
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", normalize_database_url(TEST_POSTGRES_URL))
    command.upgrade(config, "head")
    engine = make_engine(TEST_POSTGRES_URL)
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture
def session_factory(engine: Engine) -> sessionmaker[Session]:
    return make_session_factory(engine=engine)


@pytest.fixture
def db_session(
    engine: Engine, session_factory: sessionmaker[Session]
) -> Iterator[Session]:
    with session_factory() as session:
        _cleanup(session)
        session.commit()
        yield session
        session.rollback()
        _cleanup(session)
        session.commit()


def _cleanup(session: Session) -> None:
    session.execute(delete(ComposioTriggerEvent))
    session.execute(delete(ComposioTriggerSubscription))
    for model in (ComposioConnection, Installation):
        session.execute(delete(model))


def _installation(session: Session) -> uuid.UUID:
    inst = Installation(slack_team_id=f"T{uuid.uuid4().hex[:8]}")
    session.add(inst)
    session.flush()
    return inst.id


def _subscription(
    session: Session,
    installation_id: uuid.UUID,
    *,
    connected_account_id: str = "ca_test",
    trigger_slug: str = "GITHUB_PULL_REQUEST_EVENT",
    toolkit_slug: str = "github",
    status: str = "active",
    composio_trigger_id: str | None = None,
) -> ComposioTriggerSubscription:
    sub = ComposioTriggerSubscription(
        installation_id=installation_id,
        connected_account_id=connected_account_id,
        composio_user_id=f"user_{installation_id.hex[:8]}",
        toolkit_slug=toolkit_slug,
        trigger_slug=trigger_slug,
        status=status,
        composio_trigger_id=composio_trigger_id,
    )
    session.add(sub)
    session.flush()
    return sub


def _test_settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "SLACK_BOT_TOKEN": "xoxb-test",
        "SLACK_APP_TOKEN": "xapp-test",
        "SLACK_SIGNING_SECRET": "test-secret",
        "LLM_PROVIDER": "openai",
        "LLM_API_KEY": "sk-test",
        "LLM_MODEL": "gpt-4o",
        "POSTGRES_URL": TEST_POSTGRES_URL or "postgresql://test:test@localhost/test",
        "COMPOSIO_API_KEY": "composio-test",
        "KORTNY_COMPOSIO_TRIGGERS_ENABLED": "true",
        **{k.upper(): v for k, v in overrides.items()},
    }
    return Settings.model_validate(base)


def _github_event(
    *,
    trigger_id: str = "ti_abc123",
    connected_account_id: str = "ca_test",
    action: str = "review_requested",
    draft: bool = False,
    reviewer: str | None = "alice",
) -> dict[str, Any]:
    payload: dict[str, Any] = {"action": action, "pull_request": {"draft": draft}}
    if reviewer:
        payload["requested_reviewer"] = {"login": reviewer}
    return {
        "id": trigger_id,
        "uuid": trigger_id,
        "user_id": "user_1",
        "trigger_slug": "GITHUB_PULL_REQUEST_EVENT",
        "toolkit_slug": "GITHUB",
        "payload": payload,
        "metadata": {
            "id": trigger_id,
            "connected_account": {"id": connected_account_id},
        },
        "original_payload": None,
    }


class FakeSubscription:
    """Stands in for composio.core.models.triggers.TriggerSubscription.

    ``has_errored()`` is used as the health-tick hook: on its first call it
    fires every queued raw event through the registered ``.handle()``
    callbacks (mirroring how the real Pusher channel delivers frames between
    health ticks), then reports healthy; on the second call it reports
    errored so the driven session ends deterministically.
    """

    def __init__(self, events: list[dict[str, Any]]) -> None:
        self._callbacks: list[tuple[Callable[[Any], None], dict[str, Any]]] = []
        self._pending_events = list(events)
        self._error_calls = 0
        self.stopped = False

    def handle(
        self, **filters: Any
    ) -> Callable[[Callable[[Any], None]], Callable[[Any], None]]:
        def _wrap(fn: Callable[[Any], None]) -> Callable[[Any], None]:
            self._callbacks.append((fn, filters))
            return fn

        return _wrap

    def is_alive(self) -> bool:
        return True

    def has_errored(self) -> bool:
        self._error_calls += 1
        if self._error_calls == 1:
            for raw in self._pending_events:
                for fn, filters in self._callbacks:
                    wanted_slug = filters.get("trigger_slug")
                    if wanted_slug is not None and wanted_slug != raw.get(
                        "trigger_slug"
                    ):
                        continue
                    fn(raw)
            return False
        return True

    def stop(self) -> None:
        self.stopped = True


def _fake_composio_factory(subscription: FakeSubscription) -> Callable[[str], Any]:
    class _FakeTriggers:
        def subscribe(self, timeout: float) -> FakeSubscription:
            del timeout
            return subscription

    class _FakeComposio:
        def __init__(self) -> None:
            self.triggers = _FakeTriggers()

    def _factory(api_key: str) -> Any:
        del api_key
        return _FakeComposio()

    return _factory


def _build_listener(
    *,
    session_factory: sessionmaker[Session],
    events: list[dict[str, Any]],
    settings: Settings | None = None,
) -> tuple[ComposioTriggerListener, FakeSubscription]:
    subscription = FakeSubscription(events)
    listener = ComposioTriggerListener(
        settings=settings or _test_settings(),
        session_factory=session_factory,
        composio_factory=_fake_composio_factory(subscription),
        sleep=lambda _seconds: None,
    )
    return listener, subscription


# ---------------------------------------------------------------------------
# adapt_sdk_event
# ---------------------------------------------------------------------------


def test_adapt_sdk_event_valid_frame() -> None:
    raw = _github_event()
    parsed = adapt_sdk_event(raw)
    assert parsed is not None
    assert parsed.trigger_slug == "GITHUB_PULL_REQUEST_EVENT"
    assert parsed.trigger_id == "ti_abc123"
    assert parsed.connected_account_id == "ca_test"
    assert parsed.data["action"] == "review_requested"
    # id is a synthesized content-hash dedup id, not the raw trigger id.
    assert parsed.id != raw["id"]
    assert len(parsed.id) == 64  # sha256 hexdigest


def test_adapt_sdk_event_same_content_hashes_identically() -> None:
    first = adapt_sdk_event(_github_event())
    second = adapt_sdk_event(_github_event())
    assert first is not None
    assert second is not None
    assert first.id == second.id


def test_adapt_sdk_event_different_content_hashes_differently() -> None:
    first = adapt_sdk_event(_github_event(action="review_requested"))
    second = adapt_sdk_event(_github_event(action="closed"))
    assert first is not None
    assert second is not None
    assert first.id != second.id


def test_adapt_sdk_event_rejects_non_mapping() -> None:
    assert adapt_sdk_event("not a dict") is None
    assert adapt_sdk_event(None) is None
    assert adapt_sdk_event([1, 2, 3]) is None


def test_adapt_sdk_event_rejects_missing_trigger_slug() -> None:
    raw = _github_event()
    del raw["trigger_slug"]
    assert adapt_sdk_event(raw) is None


def test_adapt_sdk_event_rejects_blank_trigger_slug() -> None:
    raw = _github_event()
    raw["trigger_slug"] = "   "
    assert adapt_sdk_event(raw) is None


def test_adapt_sdk_event_rejects_missing_id() -> None:
    raw = _github_event()
    del raw["id"]
    assert adapt_sdk_event(raw) is None


def test_adapt_sdk_event_rejects_data_not_dict() -> None:
    raw = _github_event()
    raw["payload"] = ["not", "a", "dict"]
    assert adapt_sdk_event(raw) is None


def test_adapt_sdk_event_treats_missing_payload_as_empty_dict() -> None:
    raw = _github_event()
    del raw["payload"]
    parsed = adapt_sdk_event(raw)
    assert parsed is not None
    assert parsed.data == {}


def test_adapt_sdk_event_rejects_oversize_payload() -> None:
    raw = _github_event()
    raw["payload"] = {"blob": "x" * (MAX_PAYLOAD_BYTES + 1)}
    assert adapt_sdk_event(raw) is None


def test_adapt_sdk_event_rejects_non_serializable_payload() -> None:
    raw = _github_event()
    raw["payload"] = {"blob": object()}
    assert adapt_sdk_event(raw) is None


def test_adapt_sdk_event_missing_connected_account_is_optional() -> None:
    raw = _github_event()
    raw["metadata"] = {"id": raw["id"]}
    parsed = adapt_sdk_event(raw)
    assert parsed is not None
    assert parsed.connected_account_id is None


# ---------------------------------------------------------------------------
# End to end: FakeSubscription -> ComposioTriggerListener -> DB
# ---------------------------------------------------------------------------


def test_listener_persists_distinct_events(
    db_session: Session, session_factory: sessionmaker[Session]
) -> None:
    iid = _installation(db_session)
    _subscription(db_session, iid)
    db_session.commit()

    listener, subscription = _build_listener(
        session_factory=session_factory,
        events=[
            _github_event(trigger_id="ti_one", action="review_requested"),
            _github_event(trigger_id="ti_one", action="closed"),
        ],
    )
    listener._run_connected_session()

    assert subscription.stopped
    rows = db_session.scalars(select(ComposioTriggerEvent)).all()
    assert len(rows) == 2
    decisions = {row.decision for row in rows}
    assert decisions == {"ask", "silent"}


def test_listener_deduplicates_identical_redelivery(
    db_session: Session, session_factory: sessionmaker[Session]
) -> None:
    iid = _installation(db_session)
    _subscription(db_session, iid)
    db_session.commit()

    same_event = _github_event(trigger_id="ti_dup")
    listener, _subscription_obj = _build_listener(
        session_factory=session_factory,
        events=[same_event, dict(same_event)],
    )
    listener._run_connected_session()

    rows = db_session.scalars(select(ComposioTriggerEvent)).all()
    assert len(rows) == 1


def test_listener_trigger_id_routing_beats_account_fallback(
    db_session: Session, session_factory: sessionmaker[Session]
) -> None:
    iid = _installation(db_session)
    precise = _subscription(
        db_session,
        iid,
        connected_account_id="ca_other",
        composio_trigger_id="ti_precise",
    )
    _subscription(db_session, iid, connected_account_id="ca_test")
    db_session.commit()

    listener, _sub = _build_listener(
        session_factory=session_factory,
        events=[_github_event(trigger_id="ti_precise", connected_account_id="ca_test")],
    )
    listener._run_connected_session()

    rows = db_session.scalars(select(ComposioTriggerEvent)).all()
    assert len(rows) == 1
    assert rows[0].subscription_id == precise.id


def test_listener_ambiguous_trigger_id_is_dropped(
    db_session: Session, session_factory: sessionmaker[Session]
) -> None:
    iid = _installation(db_session)
    _subscription(
        db_session, iid, connected_account_id="ca_one", composio_trigger_id="ti_dupe"
    )
    _subscription(
        db_session, iid, connected_account_id="ca_two", composio_trigger_id="ti_dupe"
    )
    db_session.commit()

    listener, _sub = _build_listener(
        session_factory=session_factory,
        events=[_github_event(trigger_id="ti_dupe", connected_account_id="ca_one")],
    )
    listener._run_connected_session()

    rows = db_session.scalars(select(ComposioTriggerEvent)).all()
    assert rows == []


def test_listener_unmatched_event_persists_metadata_only(
    db_session: Session, session_factory: sessionmaker[Session]
) -> None:
    iid = _installation(db_session)
    db_session.commit()
    assert iid  # single installation on file -> unmatched events still resolve to it

    listener, _sub = _build_listener(
        session_factory=session_factory,
        events=[_github_event(trigger_id="ti_no_sub", connected_account_id="ca_none")],
    )
    listener._run_connected_session()

    rows = db_session.scalars(select(ComposioTriggerEvent)).all()
    assert len(rows) == 1
    assert rows[0].decision == "unmatched"
    assert rows[0].subscription_id is None
    assert "data" not in rows[0].raw_payload_json


def test_listener_one_bad_event_does_not_break_the_next(
    db_session: Session, session_factory: sessionmaker[Session]
) -> None:
    iid = _installation(db_session)
    _subscription(db_session, iid)
    db_session.commit()

    malformed = {"trigger_slug": "GITHUB_PULL_REQUEST_EVENT", "payload": "not-a-dict"}
    good = _github_event(trigger_id="ti_good")

    listener, subscription = _build_listener(
        session_factory=session_factory,
        events=[malformed, good],
    )
    # Should not raise despite the malformed frame.
    listener._run_connected_session()

    assert subscription.stopped
    rows = db_session.scalars(select(ComposioTriggerEvent)).all()
    assert len(rows) == 1
    assert rows[0].composio_trigger_id == "ti_good"


def test_listener_subscribe_failure_does_not_raise(
    session_factory: sessionmaker[Session],
) -> None:
    class _RaisingTriggers:
        def subscribe(self, timeout: float) -> Any:
            del timeout
            raise RuntimeError("connect refused")

    class _RaisingComposio:
        def __init__(self) -> None:
            self.triggers = _RaisingTriggers()

    listener = ComposioTriggerListener(
        settings=_test_settings(),
        session_factory=session_factory,
        composio_factory=lambda api_key: _RaisingComposio(),
        sleep=lambda _seconds: None,
    )
    # Must return quietly, not raise, so the reconnect loop can back off.
    listener._run_connected_session()


# ---------------------------------------------------------------------------
# Retention
# ---------------------------------------------------------------------------


def test_listener_retention_deletes_old_rows_and_throttles(
    db_session: Session, session_factory: sessionmaker[Session]
) -> None:
    iid = _installation(db_session)
    old = ComposioTriggerEvent(
        installation_id=iid,
        event_id="evt_old",
        trigger_slug="GITHUB_PULL_REQUEST_EVENT",
        raw_payload_json={},
        decision="silent",
    )
    db_session.add(old)
    db_session.flush()
    # received_at has a server_default of now(); backdate it past retention.
    old.received_at = datetime.now(UTC) - timedelta(days=200)
    db_session.commit()

    listener = ComposioTriggerListener(
        settings=_test_settings(KORTNY_COMPOSIO_TRIGGER_RETENTION_DAYS="90"),
        session_factory=session_factory,
        sleep=lambda _seconds: None,
    )

    # Throttled: pretend a sweep "just ran" so this call is a no-op.
    listener._last_retention_run = listener._clock()
    listener._maybe_run_retention()
    remaining = db_session.scalars(select(ComposioTriggerEvent)).all()
    assert len(remaining) == 1

    # Force the sweep to actually run.
    listener._last_retention_run = None
    listener._maybe_run_retention()
    remaining = db_session.scalars(select(ComposioTriggerEvent)).all()
    assert remaining == []


# ---------------------------------------------------------------------------
# Leader election (advisory lock on a dedicated connection)
# ---------------------------------------------------------------------------


def test_listener_advisory_lock_is_exclusive(
    engine: Engine, session_factory: sessionmaker[Session]
) -> None:
    settings = _test_settings(KORTNY_COMPOSIO_LISTENER_ADVISORY_LOCK_KEY="918273645")
    listener_a = ComposioTriggerListener(
        settings=settings, session_factory=session_factory, sleep=lambda _s: None
    )
    listener_b = ComposioTriggerListener(
        settings=settings, session_factory=session_factory, sleep=lambda _s: None
    )
    try:
        conn_a = listener_a._lock_engine.connect()
        conn_b = listener_b._lock_engine.connect()
        try:
            assert listener_a._try_acquire_lock(conn_a) is True
            # A second connection cannot acquire the same session-level key.
            assert listener_b._try_acquire_lock(conn_b) is False
            listener_a._release_lock(conn_a)
            # Now it is free again.
            assert listener_b._try_acquire_lock(conn_b) is True
            listener_b._release_lock(conn_b)
        finally:
            conn_a.close()
            conn_b.close()
    finally:
        listener_a._lock_engine.dispose()
        listener_b._lock_engine.dispose()
