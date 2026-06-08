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

from kortny.db.models import (
    Installation,
    SlackChannelMembership,
    SlackIdentity,
    Task,
    TaskStatus,
)
from kortny.db.session import make_engine, make_session_factory, normalize_database_url
from kortny.slack.identity import SlackIdentityService
from kortny.tools.resolve_slack_identity import ResolveSlackIdentityTool
from kortny.tools.slack_identity_info import SlackChannelInfoTool, SlackUserInfoTool

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


def test_resolve_slack_identity_tool_resolves_cached_user_and_channel(
    db_session: Session,
) -> None:
    installation = create_installation(db_session)
    now = datetime(2026, 5, 25, 10, 0, tzinfo=UTC)
    db_session.add_all(
        [
            SlackIdentity(
                installation_id=installation.id,
                kind="user",
                slack_id="U123",
                display_name="Aneesh Melkot",
                raw_name="aneesh",
                refreshed_at=now,
                last_seen_at=now,
            ),
            SlackIdentity(
                installation_id=installation.id,
                kind="channel",
                slack_id="C123",
                display_name="#research-room",
                raw_name="research-room",
                refreshed_at=now,
                last_seen_at=now,
            ),
        ]
    )
    task = create_task(db_session, installation, user_id="U123", channel_id="C123")
    db_session.flush()

    result = ResolveSlackIdentityTool(session=db_session, task=task).invoke(
        {"slack_ids": ["U123", "C123"]}
    )

    assert result.output["resolved_count"] == 2
    identities = {
        identity["slack_id"]: identity for identity in result.output["identities"]
    }
    assert identities["U123"]["kind"] == "user"
    assert identities["U123"]["display_name"] == "Aneesh Melkot"
    assert identities["U123"]["source"] == "slack_identities"
    assert identities["C123"]["kind"] == "channel"
    assert identities["C123"]["display_name"] == "#research-room"
    assert identities["C123"]["source"] == "slack_identities"
    assert "No Slack API call was made" in result.output["source_note"]


def test_resolve_slack_identity_tool_falls_back_to_active_channel_membership(
    db_session: Session,
) -> None:
    installation = create_installation(db_session)
    now = datetime(2026, 5, 25, 10, 0, tzinfo=UTC)
    db_session.add(
        SlackChannelMembership(
            installation_id=installation.id,
            channel_id="C456",
            channel_name="rag",
            channel_type="public_channel",
            membership_status="active",
            discovered_via="manual_backfill",
            first_seen_at=now,
            last_seen_at=now,
        )
    )
    task = create_task(db_session, installation)
    db_session.flush()

    result = ResolveSlackIdentityTool(session=db_session, task=task).invoke(
        {"slack_ids": ["C456"]}
    )

    identity = result.output["identities"][0]
    assert identity["resolved"] is True
    assert identity["display_name"] == "#rag"
    assert identity["source"] == "slack_channel_memberships"


def test_resolve_slack_identity_tool_does_not_leak_other_installation_cache(
    db_session: Session,
) -> None:
    current_installation = create_installation(db_session)
    other_installation = create_installation(db_session)
    now = datetime(2026, 5, 25, 10, 0, tzinfo=UTC)
    db_session.add(
        SlackIdentity(
            installation_id=other_installation.id,
            kind="user",
            slack_id="U999",
            display_name="Other Workspace User",
            raw_name="Other Workspace User",
            refreshed_at=now,
            last_seen_at=now,
        )
    )
    task = create_task(db_session, current_installation)
    db_session.flush()

    result = ResolveSlackIdentityTool(session=db_session, task=task).invoke(
        {"slack_ids": ["U999"]}
    )

    identity = result.output["identities"][0]
    assert result.output["resolved_count"] == 0
    assert identity["resolved"] is False
    assert identity["display_name"] == "U999"
    assert identity["source"] == "unresolved"


def test_resolve_slack_identity_tool_defaults_to_current_task_context(
    db_session: Session,
) -> None:
    installation = create_installation(db_session)
    now = datetime(2026, 5, 25, 10, 0, tzinfo=UTC)
    db_session.add_all(
        [
            SlackIdentity(
                installation_id=installation.id,
                kind="user",
                slack_id="U123",
                display_name="Aneesh Melkot",
                raw_name="aneesh",
                refreshed_at=now,
                last_seen_at=now,
            ),
            SlackIdentity(
                installation_id=installation.id,
                kind="channel",
                slack_id="C123",
                display_name="#research-room",
                raw_name="research-room",
                refreshed_at=now,
                last_seen_at=now,
            ),
        ]
    )
    task = create_task(db_session, installation, user_id="U123", channel_id="C123")
    db_session.flush()

    result = ResolveSlackIdentityTool(session=db_session, task=task).invoke({})

    assert result.output["requested_ids"] == ["U123", "C123"]
    assert result.output["resolved_count"] == 2


def test_slack_user_info_tool_refreshes_missing_current_user(
    db_session: Session,
) -> None:
    installation = create_installation(db_session)
    task = create_task(db_session, installation, user_id="U123", channel_id="C123")
    client = FakeIdentityClient(
        user_response={
            "ok": True,
            "user": {
                "id": "U123",
                "name": "aneesh",
                "profile": {"real_name": "Aneesh Melkot"},
            },
        }
    )

    result = SlackUserInfoTool(
        client=client,
        session=db_session,
        task=task,
    ).invoke({})

    identity = db_session.scalar(
        select(SlackIdentity).where(
            SlackIdentity.installation_id == installation.id,
            SlackIdentity.kind == "user",
            SlackIdentity.slack_id == "U123",
        )
    )
    assert result.output["successful"] is True
    assert result.output["display_name"] == "Aneesh Melkot"
    assert result.output["refreshed"] is True
    assert identity is not None
    assert identity.display_name == "Aneesh Melkot"
    assert client.calls == [{"method": "users_info", "id": "U123"}]


def test_slack_user_info_tool_force_refreshes_fresh_cache(
    db_session: Session,
) -> None:
    installation = create_installation(db_session)
    refreshed_at = datetime(2026, 5, 25, 9, 0, tzinfo=UTC)
    db_session.add(
        SlackIdentity(
            installation_id=installation.id,
            kind="user",
            slack_id="U123",
            display_name="Cached User",
            raw_name="Cached User",
            refreshed_at=refreshed_at,
            last_seen_at=refreshed_at,
        )
    )
    task = create_task(db_session, installation, user_id="U123", channel_id="C123")
    client = FakeIdentityClient(
        user_response={
            "ok": True,
            "user": {
                "id": "U123",
                "name": "aneesh",
                "profile": {"real_name": "Aneesh Melkot"},
            },
        }
    )

    result = SlackUserInfoTool(
        client=client,
        session=db_session,
        task=task,
    ).invoke({"force_refresh": True})

    assert result.output["display_name"] == "Aneesh Melkot"
    assert result.output["refreshed"] is True
    assert client.calls == [{"method": "users_info", "id": "U123"}]


def test_slack_channel_info_tool_refreshes_current_channel(
    db_session: Session,
) -> None:
    installation = create_installation(db_session)
    task = create_task(db_session, installation, user_id="U123", channel_id="C123")
    client = FakeIdentityClient(
        channel_response={
            "ok": True,
            "channel": {
                "id": "C123",
                "name": "research-room",
                "is_private": True,
            },
        }
    )

    result = SlackChannelInfoTool(
        client=client,
        session=db_session,
        task=task,
    ).invoke({})

    identity = db_session.scalar(
        select(SlackIdentity).where(
            SlackIdentity.installation_id == installation.id,
            SlackIdentity.kind == "channel",
            SlackIdentity.slack_id == "C123",
        )
    )
    assert result.output["successful"] is True
    assert result.output["display_name"] == "#research-room"
    assert result.output["is_private"] is True
    assert result.output["refreshed"] is True
    assert identity is not None
    assert identity.display_name == "#research-room"
    assert identity.is_private is True
    assert client.calls == [{"method": "conversations_info", "id": "C123"}]


def test_slack_channel_info_tool_rejects_other_channel(
    db_session: Session,
) -> None:
    installation = create_installation(db_session)
    task = create_task(db_session, installation, user_id="U123", channel_id="C123")
    client = FakeIdentityClient(
        channel_response={
            "ok": True,
            "channel": {"id": "C999", "name": "other-channel"},
        }
    )

    with pytest.raises(ValueError, match="current Slack channel"):
        SlackChannelInfoTool(
            client=client,
            session=db_session,
            task=task,
        ).invoke({"channel_id": "C999"})

    assert client.calls == []


def cleanup_database(session: Session) -> None:
    for model in (Task, SlackChannelMembership, SlackIdentity, Installation):
        session.execute(delete(model))


def create_installation(session: Session) -> Installation:
    installation = Installation(slack_team_id=f"T{uuid.uuid4().hex}")
    session.add(installation)
    session.flush()
    return installation


def create_task(
    session: Session,
    installation: Installation,
    *,
    user_id: str = "U123",
    channel_id: str = "C123",
) -> Task:
    task = Task(
        installation_id=installation.id,
        slack_event_id=f"Ev{uuid.uuid4().hex}",
        slack_channel_id=channel_id,
        slack_thread_ts="1779673337.889359",
        slack_message_ts="1779673337.889359",
        slack_user_id=user_id,
        input="resolve Slack identities",
        status=TaskStatus.pending,
    )
    session.add(task)
    session.flush()
    return task
