from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, delete
from sqlalchemy.orm import Session

from kortny.db.models import (
    Installation,
    ObservationEvent,
    ObservePolicy,
    SlackChannelMembership,
    Task,
    TaskEvent,
    TaskEventType,
    TaskStatus,
)
from kortny.db.session import make_engine, make_session_factory, normalize_database_url
from kortny.tools import ObservationChannelHistoryCache, SearchObservedSlackHistoryTool

TEST_POSTGRES_URL = os.environ.get("KORTNY_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    TEST_POSTGRES_URL is None,
    reason="KORTNY_TEST_POSTGRES_URL is required for Slack history cache tests",
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
        cleanup_database(session)
        session.commit()
        yield session
        session.rollback()
        cleanup_database(session)
        session.commit()


def test_observation_history_cache_returns_recent_channel_messages(
    db_session: Session,
) -> None:
    installation = create_installation(db_session)
    observed_at = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
    db_session.add_all(
        [
            observation(
                installation=installation,
                channel_id="C123",
                user_id="U1",
                event_id="EvRoot",
                message_ts="100.000000",
                text_preview="root message",
                observed_at=observed_at,
            ),
            observation(
                installation=installation,
                channel_id="C123",
                user_id="U2",
                event_id="EvReply",
                message_ts="101.000000",
                thread_ts="100.000000",
                text_preview="thread reply",
                observed_at=observed_at,
            ),
            observation(
                installation=installation,
                channel_id="C_OTHER",
                user_id="U3",
                event_id="EvOther",
                message_ts="102.000000",
                text_preview="other channel",
                observed_at=observed_at,
            ),
        ]
    )
    db_session.commit()

    cache = ObservationChannelHistoryCache(db_session, installation_id=installation.id)

    root_only = cache.fetch_messages(
        channel_id="C123",
        oldest_ts=None,
        latest_ts=None,
        limit=10,
        include_threads=False,
    )
    with_threads = cache.fetch_messages(
        channel_id="C123",
        oldest_ts=None,
        latest_ts=None,
        limit=10,
        include_threads=True,
    )

    assert [message["text"] for message in root_only] == ["root message"]
    assert [message["text"] for message in with_threads] == [
        "root message",
        "thread reply",
    ]
    assert with_threads[1]["thread_ts"] == "100.000000"
    assert with_threads[0]["source"] == "observation_cache"


def test_observation_history_cache_includes_kortny_posted_replies(
    db_session: Session,
) -> None:
    installation = create_installation(db_session)
    observed_at = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
    db_session.add(
        observation(
            installation=installation,
            channel_id="C123",
            user_id="U1",
            event_id="EvRoot",
            message_ts="100.000000",
            text_preview="where did the HIG data come from?",
            observed_at=observed_at,
        )
    )
    task = Task(
        installation_id=installation.id,
        slack_event_id="EvTask",
        slack_channel_id="C123",
        slack_thread_ts="100.000000",
        slack_message_ts="100.000000",
        slack_user_id="U1",
        input="where did the HIG data come from?",
        status=TaskStatus.succeeded,
    )
    db_session.add(task)
    db_session.flush()
    db_session.add(
        TaskEvent(
            task_id=task.id,
            seq=1,
            type=TaskEventType.message_posted,
            payload={
                "channel": "C123",
                "purpose": "result",
                "thread_ts": "100.000000",
                "message_ts": "101.000000",
                "text": "The HIG items came from prior messages in this thread.",
            },
        )
    )
    db_session.commit()

    cache = ObservationChannelHistoryCache(db_session, installation_id=installation.id)

    root_only = cache.fetch_messages(
        channel_id="C123",
        oldest_ts=None,
        latest_ts=None,
        limit=10,
        include_threads=False,
    )
    with_threads = cache.fetch_messages(
        channel_id="C123",
        oldest_ts=None,
        latest_ts=None,
        limit=10,
        include_threads=True,
    )

    assert [message["text"] for message in root_only] == [
        "where did the HIG data come from?"
    ]
    assert [message["text"] for message in with_threads] == [
        "where did the HIG data come from?",
        "The HIG items came from prior messages in this thread.",
    ]
    assert with_threads[1]["source"] == "task_events"
    assert with_threads[1]["author"] == "Kortny"
    assert with_threads[1]["thread_ts"] == "100.000000"


def test_search_observed_slack_history_defaults_to_current_channel(
    db_session: Session,
) -> None:
    installation = create_installation(db_session)
    observed_at = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
    db_session.add_all(
        [
            observation(
                installation=installation,
                channel_id="C123",
                user_id="U1",
                event_id="EvCurrent",
                message_ts="100.000000",
                text_preview="Langfuse decision belongs in this channel",
                observed_at=observed_at,
            ),
            observation(
                installation=installation,
                channel_id="C_OTHER",
                user_id="U2",
                event_id="EvOther",
                message_ts="101.000000",
                text_preview="Langfuse decision belongs in another channel",
                observed_at=observed_at,
            ),
        ]
    )
    task = task_for_channel(installation=installation, channel_id="C123")
    db_session.add(task)
    db_session.commit()

    result = SearchObservedSlackHistoryTool(session=db_session, task=task).invoke(
        {"query": "Langfuse decision"}
    )

    assert result.output["scope"] == "current_channel"
    assert result.output["channel_ids"] == ["C123"]
    assert result.output["match_count"] == 1
    assert result.output["results"][0]["channel_id"] == "C123"
    assert result.output["results"][0]["text"] == (
        "Langfuse decision belongs in this channel"
    )


def test_search_observed_slack_history_blocks_other_channel_without_broad_scope(
    db_session: Session,
) -> None:
    installation = create_installation(db_session)
    task = task_for_channel(installation=installation, channel_id="C123")
    db_session.add(task)
    db_session.commit()

    result = SearchObservedSlackHistoryTool(session=db_session, task=task).invoke(
        {"query": "decision", "channel_id": "C_OTHER"}
    )

    assert result.output["blocked"] is True
    assert result.output["scope"] == "blocked_channel"
    assert result.output["match_count"] == 0
    assert result.output["results"] == []


def test_search_observed_slack_history_broad_scope_uses_active_memberships(
    db_session: Session,
) -> None:
    installation = create_installation(db_session)
    observed_at = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
    db_session.add_all(
        [
            membership(
                installation=installation,
                channel_id="C123",
                status="active",
            ),
            membership(
                installation=installation,
                channel_id="C_VISIBLE",
                status="active",
            ),
            membership(
                installation=installation,
                channel_id="C_LEFT",
                status="left",
            ),
            observation(
                installation=installation,
                channel_id="C123",
                user_id="U1",
                event_id="EvCurrent",
                message_ts="100.000000",
                text_preview="pricing decision in current channel",
                observed_at=observed_at,
            ),
            observation(
                installation=installation,
                channel_id="C_VISIBLE",
                user_id="U2",
                event_id="EvVisible",
                message_ts="101.000000",
                text_preview="pricing decision in visible channel",
                observed_at=observed_at,
            ),
            observation(
                installation=installation,
                channel_id="C_LEFT",
                user_id="U3",
                event_id="EvLeft",
                message_ts="102.000000",
                text_preview="pricing decision in left channel",
                observed_at=observed_at,
            ),
            observation(
                installation=installation,
                channel_id="C_UNKNOWN",
                user_id="U4",
                event_id="EvUnknown",
                message_ts="103.000000",
                text_preview="pricing decision in unknown channel",
                observed_at=observed_at,
            ),
        ]
    )
    task = task_for_channel(installation=installation, channel_id="C123")
    db_session.add(task)
    db_session.commit()

    result = SearchObservedSlackHistoryTool(session=db_session, task=task).invoke(
        {
            "query": "pricing decision",
            "include_all_visible_channels": True,
            "limit": 10,
        }
    )

    assert result.output["scope"] == "visible_channels"
    assert set(result.output["channel_ids"]) == {"C123", "C_VISIBLE"}
    assert {item["channel_id"] for item in result.output["results"]} == {
        "C123",
        "C_VISIBLE",
    }


def cleanup_database(session: Session) -> None:
    for model in (
        TaskEvent,
        Task,
        ObservationEvent,
        SlackChannelMembership,
        ObservePolicy,
        Installation,
    ):
        session.execute(delete(model))


def create_installation(session: Session) -> Installation:
    installation = Installation(slack_team_id=f"T{uuid.uuid4().hex}")
    session.add(installation)
    session.flush()
    return installation


def observation(
    *,
    installation: Installation,
    channel_id: str,
    user_id: str,
    event_id: str,
    message_ts: str,
    text_preview: str,
    observed_at: datetime,
    thread_ts: str | None = None,
) -> ObservationEvent:
    return ObservationEvent(
        installation_id=installation.id,
        slack_team_id=installation.slack_team_id,
        channel_id=channel_id,
        user_id=user_id,
        event_type="message",
        slack_event_id=event_id,
        message_ts=message_ts,
        thread_ts=thread_ts,
        file_id=None,
        raw_payload_checksum=f"checksum-{event_id}",
        text_preview=text_preview,
        visibility_metadata={"scope_type": "channel", "scope_id": channel_id},
        observed_at=observed_at,
    )


def membership(
    *,
    installation: Installation,
    channel_id: str,
    status: str,
) -> SlackChannelMembership:
    return SlackChannelMembership(
        installation_id=installation.id,
        channel_id=channel_id,
        membership_status=status,
        discovered_via="message_observation",
    )


def task_for_channel(
    *,
    installation: Installation,
    channel_id: str,
) -> Task:
    return Task(
        installation_id=installation.id,
        slack_event_id=f"EvTask-{uuid.uuid4().hex}",
        slack_channel_id=channel_id,
        slack_thread_ts="100.000000",
        slack_message_ts="100.000000",
        slack_user_id="U1",
        input="search observed Slack history",
        status=TaskStatus.pending,
    )
