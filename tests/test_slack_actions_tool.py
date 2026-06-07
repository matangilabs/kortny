import os
import uuid
from collections.abc import Iterator
from typing import Any

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, delete, select
from sqlalchemy.orm import Session

from kortny.db.models import (
    Installation,
    SlackSideEffect,
    Task,
    TaskEvent,
    TaskEventType,
    TaskStatus,
)
from kortny.db.session import make_engine, make_session_factory, normalize_database_url
from kortny.tools.slack_actions import SlackAddReactionTool, SlackReplyThreadTool

TEST_POSTGRES_URL = os.environ.get("KORTNY_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    TEST_POSTGRES_URL is None,
    reason="KORTNY_TEST_POSTGRES_URL is required for Slack action tool tests",
)


class FakeSlackActionClient:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []
        self.reactions: list[dict[str, Any]] = []

    def chat_postMessage(
        self,
        *,
        channel: str,
        text: str,
        thread_ts: str | None = None,
    ) -> dict[str, Any]:
        self.messages.append(
            {
                "channel": channel,
                "text": text,
                "thread_ts": thread_ts,
            }
        )
        return {"ok": True, "ts": f"1716400001.{len(self.messages):06d}"}

    def reactions_add(
        self,
        *,
        channel: str,
        timestamp: str,
        name: str,
    ) -> dict[str, Any]:
        self.reactions.append(
            {
                "channel": channel,
                "timestamp": timestamp,
                "name": name,
            }
        )
        return {"ok": True}


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


def test_slack_reply_thread_posts_current_thread_and_records_side_effect(
    db_session: Session,
) -> None:
    task = create_task(db_session)
    client = FakeSlackActionClient()

    result = SlackReplyThreadTool(
        client=client,
        session=db_session,
        task=task,
    ).invoke({"text": "I left a note here.", "purpose": "status_update"})

    event = db_session.scalar(
        select(TaskEvent).where(
            TaskEvent.task_id == task.id,
            TaskEvent.type == TaskEventType.message_posted,
        )
    )
    side_effect = db_session.scalar(
        select(SlackSideEffect).where(SlackSideEffect.task_id == task.id)
    )

    assert result.output["successful"] is True
    assert result.output["message_ts"] == "1716400001.000001"
    assert client.messages == [
        {
            "channel": "C123",
            "text": "I left a note here.",
            "thread_ts": "1716400000.000001",
        }
    ]
    assert event is not None
    assert event.payload["tool"] == "slack_reply_thread"
    assert event.payload["purpose"] == "tool_status_update"
    assert event.payload["message_ts"] == "1716400001.000001"
    assert side_effect is not None
    assert side_effect.operation == "chat_postMessage"
    assert side_effect.status == "succeeded"
    assert side_effect.request_json["thread_ts"] == "1716400000.000001"


def test_slack_reply_thread_does_not_thread_dm_replies(
    db_session: Session,
) -> None:
    task = create_task(
        db_session,
        channel_id="D123",
        thread_ts="D123",
        message_ts="1716400000.000001",
    )
    client = FakeSlackActionClient()

    result = SlackReplyThreadTool(
        client=client,
        session=db_session,
        task=task,
    ).invoke({"text": "Yep, I see it."})

    assert result.output["thread_ts"] is None
    assert client.messages == [
        {
            "channel": "D123",
            "text": "Yep, I see it.",
            "thread_ts": None,
        }
    ]


def test_slack_reply_thread_rejects_other_channel(
    db_session: Session,
) -> None:
    task = create_task(db_session)
    client = FakeSlackActionClient()

    with pytest.raises(ValueError, match="current Slack channel"):
        SlackReplyThreadTool(
            client=client,
            session=db_session,
            task=task,
        ).invoke({"text": "Nope.", "channel_id": "C999"})

    assert client.messages == []


def test_slack_add_reaction_uses_current_message_and_records_side_effect(
    db_session: Session,
) -> None:
    task = create_task(db_session)
    client = FakeSlackActionClient()

    result = SlackAddReactionTool(
        client=client,
        session=db_session,
        task=task,
    ).invoke({"name": ":eyes:"})

    event = db_session.scalar(
        select(TaskEvent).where(
            TaskEvent.task_id == task.id,
            TaskEvent.type == TaskEventType.log,
            TaskEvent.payload["message"].as_string() == "slack_reaction_added",
        )
    )
    side_effect = db_session.scalar(
        select(SlackSideEffect).where(SlackSideEffect.task_id == task.id)
    )

    assert result.output["successful"] is True
    assert result.output["reaction"] == "eyes"
    assert client.reactions == [
        {
            "channel": "C123",
            "timestamp": "1716400000.000001",
            "name": "eyes",
        }
    ]
    assert event is not None
    assert event.payload["tool"] == "slack_add_reaction"
    assert event.payload["reaction"] == "eyes"
    assert side_effect is not None
    assert side_effect.operation == "reactions_add"
    assert side_effect.target_message_ts == "1716400000.000001"


def test_slack_add_reaction_rejects_other_message(
    db_session: Session,
) -> None:
    task = create_task(db_session)
    client = FakeSlackActionClient()

    with pytest.raises(ValueError, match="current Slack message"):
        SlackAddReactionTool(
            client=client,
            session=db_session,
            task=task,
        ).invoke({"name": "eyes", "message_ts": "1716400099.000001"})

    assert client.reactions == []


def cleanup_database(session: Session) -> None:
    for model in (SlackSideEffect, TaskEvent, Task, Installation):
        session.execute(delete(model))


def create_task(
    session: Session,
    *,
    channel_id: str = "C123",
    thread_ts: str | None = "1716400000.000001",
    message_ts: str | None = "1716400000.000001",
) -> Task:
    installation = Installation(slack_team_id=f"T{uuid.uuid4().hex}")
    session.add(installation)
    session.flush()
    task = Task(
        installation_id=installation.id,
        slack_event_id=f"Ev{uuid.uuid4().hex}",
        slack_channel_id=channel_id,
        slack_thread_ts=thread_ts,
        slack_message_ts=message_ts,
        slack_user_id="U123",
        input="perform Slack action",
        status=TaskStatus.pending,
    )
    session.add(task)
    session.flush()
    return task
