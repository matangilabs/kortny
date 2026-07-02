"""Tests for kortny/composio/trigger_subscriptions.py — dashboard opt-in.

REST calls are faked (a ``ComposioClient`` subclass overriding the two
methods used here); no network, no real Composio API.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from typing import Any

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, delete
from sqlalchemy.orm import Session

from kortny.composio.client import ComposioClient, ComposioConnectionError
from kortny.composio.trigger_subscriptions import (
    disable_trigger_subscription,
    enable_trigger_subscription,
)
from kortny.db.models import (
    ComposioConnection,
    ComposioTriggerSubscription,
    Installation,
)
from kortny.db.session import make_engine, make_session_factory, normalize_database_url

TEST_POSTGRES_URL = os.environ.get("KORTNY_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    TEST_POSTGRES_URL is None,
    reason="KORTNY_TEST_POSTGRES_URL is required for trigger subscription tests",
)


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
def db_session(engine: Engine) -> Iterator[Session]:
    session_factory = make_session_factory(engine=engine)
    with session_factory() as session:
        _cleanup(session)
        session.commit()
        yield session
        session.rollback()
        _cleanup(session)
        session.commit()


def _cleanup(session: Session) -> None:
    session.execute(delete(ComposioTriggerSubscription))
    for model in (ComposioConnection, Installation):
        session.execute(delete(model))


def _installation(session: Session) -> uuid.UUID:
    inst = Installation(slack_team_id=f"T{uuid.uuid4().hex[:8]}")
    session.add(inst)
    session.flush()
    return inst.id


def _connection(
    session: Session,
    installation_id: uuid.UUID,
    *,
    connected_account_id: str | None = "ca_test",
    toolkit_slug: str = "github",
) -> ComposioConnection:
    connection = ComposioConnection(
        installation_id=installation_id,
        toolkit_slug=toolkit_slug,
        connected_account_id=connected_account_id,
        composio_user_id="user_1",
        owner_slack_user_id="U123",
        visibility_scope_type="user",
        visibility_scope_id="U123",
        status="active",
    )
    session.add(connection)
    session.flush()
    return connection


class FakeComposioClient(ComposioClient):
    """A ``ComposioClient`` whose REST calls are faked for tests."""

    def __init__(
        self,
        *,
        create_response: dict[str, Any] | None = None,
        create_error: Exception | None = None,
        disable_error: Exception | None = None,
    ) -> None:
        super().__init__(api_key="test-key")
        self._create_response = (
            create_response if create_response is not None else {"id": "ti_new_001"}
        )
        self._create_error = create_error
        self._disable_error = disable_error
        self.create_calls: list[tuple[str, str, dict[str, Any], str | None]] = []
        self.disable_calls: list[str] = []

    def create_trigger(
        self,
        slug: str,
        user_id: str,
        trigger_config: dict[str, Any],
        connected_account_id: str | None = None,
    ) -> dict[str, Any]:
        self.create_calls.append((slug, user_id, trigger_config, connected_account_id))
        if self._create_error is not None:
            raise self._create_error
        return self._create_response

    def disable_trigger(self, trigger_id: str) -> dict[str, Any]:
        self.disable_calls.append(trigger_id)
        if self._disable_error is not None:
            raise self._disable_error
        return {}


# ---------------------------------------------------------------------------
# enable
# ---------------------------------------------------------------------------


def test_enable_creates_active_subscription(db_session: Session) -> None:
    iid = _installation(db_session)
    connection = _connection(db_session, iid)
    db_session.commit()

    client = FakeComposioClient(create_response={"id": "ti_created"})
    result = enable_trigger_subscription(
        db_session,
        connection=connection,
        trigger_slug="GITHUB_PULL_REQUEST_EVENT",
        client=client,
    )

    assert result.ok is True
    assert result.subscription is not None
    assert result.subscription.status == "active"
    assert result.subscription.composio_trigger_id == "ti_created"
    assert result.subscription.installation_id == iid
    assert client.create_calls == [
        ("GITHUB_PULL_REQUEST_EVENT", "user_1", {}, "ca_test")
    ]


def test_enable_without_connected_account_fails_fast(db_session: Session) -> None:
    iid = _installation(db_session)
    connection = _connection(db_session, iid, connected_account_id=None)
    db_session.commit()

    client = FakeComposioClient()
    result = enable_trigger_subscription(
        db_session,
        connection=connection,
        trigger_slug="GITHUB_PULL_REQUEST_EVENT",
        client=client,
    )

    assert result.ok is False
    assert result.subscription is None
    assert client.create_calls == []


def test_enable_rest_failure_leaves_no_active_row(db_session: Session) -> None:
    iid = _installation(db_session)
    connection = _connection(db_session, iid)
    db_session.commit()

    client = FakeComposioClient(create_error=ComposioConnectionError("boom"))
    result = enable_trigger_subscription(
        db_session,
        connection=connection,
        trigger_slug="GITHUB_PULL_REQUEST_EVENT",
        client=client,
    )

    assert result.ok is False
    assert result.error is not None
    assert result.subscription is None
    active = (
        db_session.query(ComposioTriggerSubscription)
        .filter(ComposioTriggerSubscription.status == "active")
        .count()
    )
    assert active == 0
    # The caller (dashboard route) rolls back on failure, which also discards
    # the just-claimed placeholder row.
    db_session.rollback()
    remaining = db_session.query(ComposioTriggerSubscription).count()
    assert remaining == 0


def test_enable_missing_trigger_instance_id_fails_without_activating(
    db_session: Session,
) -> None:
    """A create response with no ti_* id must NOT activate the row: an active
    subscription with composio_trigger_id=None would strand the upstream
    trigger (disable skips the REST call) and degrade trigger-id routing."""

    iid = _installation(db_session)
    connection = _connection(db_session, iid)
    db_session.commit()

    client = FakeComposioClient(create_response={"status": "success"})
    result = enable_trigger_subscription(
        db_session,
        connection=connection,
        trigger_slug="GITHUB_PULL_REQUEST_EVENT",
        client=client,
    )

    assert result.ok is False
    assert result.subscription is None
    assert result.error is not None
    assert "trigger instance id" in result.error
    # The REST create DID run (once) but nothing was activated.
    assert len(client.create_calls) == 1
    active = (
        db_session.query(ComposioTriggerSubscription)
        .filter(ComposioTriggerSubscription.status == "active")
        .count()
    )
    assert active == 0
    db_session.rollback()
    remaining = db_session.query(ComposioTriggerSubscription).count()
    assert remaining == 0


def test_enable_serializes_concurrent_enables_single_rest_create(
    db_session: Session,
) -> None:
    """The local row is claimed + row-locked BEFORE the remote create; the
    enable that loses the race finds the active row under the lock and
    returns already-enabled without firing another REST create."""

    iid = _installation(db_session)
    connection = _connection(db_session, iid)
    db_session.commit()

    client = FakeComposioClient(create_response={"id": "ti_once"})

    first = enable_trigger_subscription(
        db_session,
        connection=connection,
        trigger_slug="GITHUB_PULL_REQUEST_EVENT",
        client=client,
    )
    assert first.ok is True
    assert first.already_enabled is False
    assert first.subscription is not None
    db_session.commit()

    second = enable_trigger_subscription(
        db_session,
        connection=connection,
        trigger_slug="GITHUB_PULL_REQUEST_EVENT",
        client=client,
    )
    assert second.ok is True
    assert second.already_enabled is True
    assert second.subscription is not None
    assert second.subscription.id == first.subscription.id

    # Exactly one remote create across both calls.
    assert len(client.create_calls) == 1
    rows = db_session.query(ComposioTriggerSubscription).count()
    assert rows == 1


def test_enable_re_enables_existing_disabled_subscription(db_session: Session) -> None:
    iid = _installation(db_session)
    connection = _connection(db_session, iid)
    existing = ComposioTriggerSubscription(
        installation_id=iid,
        connected_account_id="ca_test",
        composio_user_id="user_1",
        toolkit_slug="github",
        trigger_slug="GITHUB_PULL_REQUEST_EVENT",
        composio_trigger_id="ti_old",
        status="disabled",
    )
    db_session.add(existing)
    db_session.commit()

    client = FakeComposioClient(create_response={"id": "ti_fresh"})
    result = enable_trigger_subscription(
        db_session,
        connection=connection,
        trigger_slug="GITHUB_PULL_REQUEST_EVENT",
        client=client,
    )

    assert result.ok is True
    assert result.subscription is not None
    assert result.subscription.id == existing.id
    assert result.subscription.status == "active"
    assert result.subscription.composio_trigger_id == "ti_fresh"
    remaining = db_session.query(ComposioTriggerSubscription).count()
    assert remaining == 1


# ---------------------------------------------------------------------------
# disable
# ---------------------------------------------------------------------------


def test_disable_sets_status_and_calls_rest(db_session: Session) -> None:
    iid = _installation(db_session)
    subscription = ComposioTriggerSubscription(
        installation_id=iid,
        connected_account_id="ca_test",
        composio_user_id="user_1",
        toolkit_slug="github",
        trigger_slug="GITHUB_PULL_REQUEST_EVENT",
        composio_trigger_id="ti_live",
        status="active",
    )
    db_session.add(subscription)
    db_session.commit()

    client = FakeComposioClient()
    result = disable_trigger_subscription(
        db_session, subscription=subscription, client=client
    )

    assert result.ok is True
    assert subscription.status == "disabled"
    assert client.disable_calls == ["ti_live"]


def test_disable_tolerates_rest_failure(db_session: Session) -> None:
    iid = _installation(db_session)
    subscription = ComposioTriggerSubscription(
        installation_id=iid,
        connected_account_id="ca_test",
        composio_user_id="user_1",
        toolkit_slug="github",
        trigger_slug="GITHUB_PULL_REQUEST_EVENT",
        composio_trigger_id="ti_live",
        status="active",
    )
    db_session.add(subscription)
    db_session.commit()

    client = FakeComposioClient(disable_error=ComposioConnectionError("remote gone"))
    result = disable_trigger_subscription(
        db_session, subscription=subscription, client=client
    )

    # The local row is disabled regardless of the remote failure — a stale
    # "active" row would keep routing realtime events to it.
    assert result.ok is True
    assert subscription.status == "disabled"


def test_disable_without_remote_trigger_id_skips_rest_call(db_session: Session) -> None:
    iid = _installation(db_session)
    subscription = ComposioTriggerSubscription(
        installation_id=iid,
        connected_account_id="ca_test",
        composio_user_id="user_1",
        toolkit_slug="github",
        trigger_slug="GITHUB_PULL_REQUEST_EVENT",
        composio_trigger_id=None,
        status="active",
    )
    db_session.add(subscription)
    db_session.commit()

    client = FakeComposioClient()
    disable_trigger_subscription(db_session, subscription=subscription, client=client)

    assert subscription.status == "disabled"
    assert client.disable_calls == []
