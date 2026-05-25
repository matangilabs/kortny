import os
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, delete, select
from sqlalchemy.orm import Session

from kortny.db.models import Installation, SlackIdentity
from kortny.db.session import make_engine, make_session_factory, normalize_database_url
from kortny.slack.identity import SlackIdentityService

TEST_POSTGRES_URL = os.environ.get("KORTNY_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    TEST_POSTGRES_URL is None,
    reason="KORTNY_TEST_POSTGRES_URL is required for Slack identity tests",
)


class SlackResponseLike:
    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data


class FakeIdentityClient:
    def __init__(
        self,
        *,
        user_response: object | None = None,
        channel_response: object | None = None,
        error: Exception | None = None,
    ) -> None:
        self.user_response = user_response
        self.channel_response = channel_response
        self.error = error
        self.calls: list[dict[str, str]] = []

    def users_info(self, *, user: str) -> object:
        self.calls.append({"method": "users_info", "id": user})
        if self.error is not None:
            raise self.error
        if self.user_response is None:
            raise RuntimeError("users_info not configured")
        return self.user_response

    def conversations_info(self, *, channel: str) -> object:
        self.calls.append({"method": "conversations_info", "id": channel})
        if self.error is not None:
            raise self.error
        if self.channel_response is None:
            raise RuntimeError("conversations_info not configured")
        return self.channel_response


class FailIfCalledClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    def users_info(self, *, user: str) -> object:
        self.calls.append({"method": "users_info", "id": user})
        raise AssertionError("users_info should not be called for fresh cache")

    def conversations_info(self, *, channel: str) -> object:
        self.calls.append({"method": "conversations_info", "id": channel})
        raise AssertionError("conversations_info should not be called for fresh cache")


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
        cleanup_database(session)
        session.commit()
        yield session
        session.rollback()
        cleanup_database(session)
        session.commit()


def test_identity_service_refreshes_missing_user_and_channel(
    db_session: Session,
) -> None:
    installation = create_installation(db_session)
    client = FakeIdentityClient(
        user_response={
            "ok": True,
            "user": {
                "id": "U123",
                "name": "aneesh",
                "real_name": "Aneesh Melkot",
                "profile": {"display_name": "Aneesh"},
            },
        },
        channel_response=SlackResponseLike(
            {
                "ok": True,
                "channel": {
                    "id": "C123",
                    "name": "research-room",
                    "is_private": False,
                },
            }
        ),
    )
    service = SlackIdentityService(db_session)
    now = datetime(2026, 5, 25, 10, 0, tzinfo=UTC)

    user_result = service.ensure_user(
        installation_id=installation.id,
        user_id="U123",
        client=client,
        now=now,
    )
    channel_result = service.ensure_channel(
        installation_id=installation.id,
        channel_id="C123",
        client=client,
        now=now,
    )

    user = db_session.scalar(
        select(SlackIdentity).where(
            SlackIdentity.installation_id == installation.id,
            SlackIdentity.kind == "user",
            SlackIdentity.slack_id == "U123",
        )
    )
    channel = db_session.scalar(
        select(SlackIdentity).where(
            SlackIdentity.installation_id == installation.id,
            SlackIdentity.kind == "channel",
            SlackIdentity.slack_id == "C123",
        )
    )

    assert user_result.refreshed is True
    assert channel_result.refreshed is True
    assert user is not None
    assert user.display_name == "Aneesh Melkot"
    assert user.raw_name == "Aneesh Melkot"
    assert channel is not None
    assert channel.display_name == "#research-room"
    assert channel.raw_name == "research-room"
    assert client.calls == [
        {"method": "users_info", "id": "U123"},
        {"method": "conversations_info", "id": "C123"},
    ]


def test_identity_service_uses_fresh_cache_without_api_call(
    db_session: Session,
) -> None:
    installation = create_installation(db_session)
    refreshed_at = datetime(2026, 5, 25, 9, 0, tzinfo=UTC)
    identity = SlackIdentity(
        installation_id=installation.id,
        kind="user",
        slack_id="U123",
        display_name="Cached User",
        raw_name="Cached User",
        refreshed_at=refreshed_at,
        last_seen_at=refreshed_at,
    )
    db_session.add(identity)
    db_session.flush()
    client = FailIfCalledClient()
    now = datetime(2026, 5, 25, 10, 0, tzinfo=UTC)

    result = SlackIdentityService(db_session).ensure_user(
        installation_id=installation.id,
        user_id="U123",
        client=client,
        now=now,
    )

    assert result.identity == identity
    assert result.refreshed is False
    assert result.reason is None
    assert identity.last_seen_at == now
    assert identity.refreshed_at == refreshed_at
    assert client.calls == []


def test_identity_service_normalizes_fresh_cached_user_to_full_name(
    db_session: Session,
) -> None:
    installation = create_installation(db_session)
    refreshed_at = datetime(2026, 5, 25, 9, 0, tzinfo=UTC)
    identity = SlackIdentity(
        installation_id=installation.id,
        kind="user",
        slack_id="U123",
        display_name="aneesh",
        raw_name="aneesh",
        raw_json={
            "id": "U123",
            "name": "aneesh",
            "real_name": "Aneesh Melkot",
            "profile": {"display_name": "aneesh"},
        },
        refreshed_at=refreshed_at,
        last_seen_at=refreshed_at,
    )
    db_session.add(identity)
    db_session.flush()
    client = FailIfCalledClient()
    now = datetime(2026, 5, 25, 10, 0, tzinfo=UTC)

    result = SlackIdentityService(db_session).ensure_user(
        installation_id=installation.id,
        user_id="U123",
        client=client,
        now=now,
    )

    assert result.identity == identity
    assert result.refreshed is False
    assert identity.display_name == "Aneesh Melkot"
    assert identity.raw_name == "Aneesh Melkot"
    assert identity.last_seen_at == now
    assert identity.refreshed_at == now
    assert client.calls == []


def test_identity_service_refreshes_stale_cache(
    db_session: Session,
) -> None:
    installation = create_installation(db_session)
    old_time = datetime(2026, 5, 23, 10, 0, tzinfo=UTC)
    identity = SlackIdentity(
        installation_id=installation.id,
        kind="user",
        slack_id="U123",
        display_name="Old User",
        raw_name="Old User",
        refreshed_at=old_time,
        last_seen_at=old_time,
    )
    db_session.add(identity)
    db_session.flush()
    client = FakeIdentityClient(
        user_response={
            "ok": True,
            "user": {
                "id": "U123",
                "name": "new-user",
                "profile": {"real_name": "New User"},
            },
        },
    )
    now = old_time + timedelta(days=2)

    result = SlackIdentityService(db_session).ensure_user(
        installation_id=installation.id,
        user_id="U123",
        client=client,
        now=now,
    )

    assert result.identity == identity
    assert result.refreshed is True
    assert identity.display_name == "New User"
    assert identity.raw_name == "New User"
    assert identity.refreshed_at == now
    assert client.calls == [{"method": "users_info", "id": "U123"}]


def test_identity_service_keeps_stale_cache_when_slack_api_fails(
    db_session: Session,
) -> None:
    installation = create_installation(db_session)
    old_time = datetime(2026, 5, 23, 10, 0, tzinfo=UTC)
    identity = SlackIdentity(
        installation_id=installation.id,
        kind="channel",
        slack_id="C123",
        display_name="#cached-channel",
        raw_name="cached-channel",
        refreshed_at=old_time,
        last_seen_at=old_time,
    )
    db_session.add(identity)
    db_session.flush()
    now = old_time + timedelta(days=2)

    result = SlackIdentityService(db_session).ensure_channel(
        installation_id=installation.id,
        channel_id="C123",
        client=FakeIdentityClient(error=RuntimeError("rate_limited")),
        now=now,
    )

    assert result.identity == identity
    assert result.refreshed is False
    assert result.reason == "RuntimeError"
    assert identity.display_name == "#cached-channel"
    assert identity.refreshed_at == old_time
    assert identity.last_seen_at == now


def cleanup_database(session: Session) -> None:
    for model in (SlackIdentity, Installation):
        session.execute(delete(model))


def create_installation(session: Session) -> Installation:
    installation = Installation(slack_team_id=f"T{uuid.uuid4().hex}")
    session.add(installation)
    session.flush()
    return installation
