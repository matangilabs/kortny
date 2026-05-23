import os
import uuid
from collections.abc import Iterator
from typing import Any

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, delete, func, select
from sqlalchemy.orm import Session

from kortny.db.models import (
    Artifact,
    EncryptedSecret,
    Installation,
    LLMUsage,
    ModelPricing,
    Task,
    TaskEvent,
    TaskEventType,
)
from kortny.db.session import make_engine, make_session_factory, normalize_database_url
from kortny.slack import SlackIngress, acknowledge_then_handle
from kortny.slack.ingress import ON_IT_TEXT

TEST_POSTGRES_URL = os.environ.get("KORTNY_TEST_POSTGRES_URL")


class FakeSlackClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def chat_postMessage(
        self,
        *,
        channel: str,
        text: str,
        thread_ts: str | None = None,
    ) -> dict[str, Any]:
        call = {
            "channel": channel,
            "text": text,
            "thread_ts": thread_ts,
        }
        self.calls.append(call)
        return {
            "ok": True,
            "channel": channel,
            "ts": f"1716400000.{len(self.calls):06d}",
        }


def test_acknowledge_then_handle_acks_before_work() -> None:
    calls: list[str] = []

    def ack() -> None:
        calls.append("ack")

    def handle() -> str:
        calls.append("handle")
        return "done"

    assert acknowledge_then_handle(ack, handle) == "done"
    assert calls == ["ack", "handle"]


@pytest.fixture(scope="session")
def engine() -> Iterator[Engine]:
    if TEST_POSTGRES_URL is None:
        pytest.skip("KORTNY_TEST_POSTGRES_URL is required for Slack ingress tests")

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


def test_app_mention_creates_task_and_posts_ack_reply(db_session: Session) -> None:
    client = FakeSlackClient()
    result = SlackIngress(session=db_session, client=client).handle_app_mention(
        body=app_mention_body(event_id="EvMention1"),
        event=app_mention_event(text="<@UBOT> research pandas and make a PDF"),
    )
    db_session.commit()

    task = db_session.scalar(select(Task).where(Task.id == result.task.id))
    installation = db_session.scalar(select(Installation))
    message_event = db_session.scalar(
        select(TaskEvent).where(
            TaskEvent.task_id == result.task.id,
            TaskEvent.type == TaskEventType.message_posted,
        )
    )

    assert result.created is True
    assert result.thread_ts == "1716400000.000001"
    assert result.acknowledgement_ts == "1716400000.000001"
    assert installation is not None
    assert installation.slack_team_id == "T123"
    assert task is not None
    assert task.slack_event_id == "EvMention1"
    assert task.slack_channel_id == "C123"
    assert task.slack_thread_ts == "1716400000.000001"
    assert task.slack_message_ts == "1716400000.000001"
    assert task.slack_user_id == "U123"
    assert task.input == "research pandas and make a PDF"
    assert client.calls == [
        {
            "channel": "C123",
            "text": ON_IT_TEXT,
            "thread_ts": "1716400000.000001",
        }
    ]
    assert message_event is not None
    assert message_event.payload["purpose"] == "acknowledgement"
    assert message_event.payload["message_ts"] == "1716400000.000001"


def test_app_mention_uses_existing_thread_ts(db_session: Session) -> None:
    client = FakeSlackClient()
    result = SlackIngress(session=db_session, client=client).handle_app_mention(
        body=app_mention_body(event_id="EvMentionThread"),
        event=app_mention_event(thread_ts="1716300000.000999"),
    )

    assert result.thread_ts == "1716300000.000999"
    assert result.task.slack_thread_ts == "1716300000.000999"
    assert client.calls[0]["thread_ts"] == "1716300000.000999"


def test_redelivered_app_mention_is_idempotent(db_session: Session) -> None:
    client = FakeSlackClient()
    ingress = SlackIngress(session=db_session, client=client)
    body = app_mention_body(event_id="EvMentionDuplicate")
    event = app_mention_event(text="<@UBOT> search duplicate delivery")

    first = ingress.handle_app_mention(body=body, event=event)
    second = ingress.handle_app_mention(body=body, event=event)
    db_session.commit()

    task_count = db_session.scalar(select(func.count()).select_from(Task))
    message_event_count = db_session.scalar(
        select(func.count())
        .select_from(TaskEvent)
        .where(TaskEvent.type == TaskEventType.message_posted)
    )

    assert first.created is True
    assert second.created is False
    assert second.task.id == first.task.id
    assert task_count == 1
    assert message_event_count == 1
    assert len(client.calls) == 1


def test_app_mention_dedupes_by_slack_message_timestamp(db_session: Session) -> None:
    client = FakeSlackClient()
    ingress = SlackIngress(session=db_session, client=client)

    first = ingress.handle_app_mention(
        body=app_mention_body(event_id="EvMessageFirst"),
        event=app_mention_event(text="<@UBOT> search duplicate event shapes"),
    )
    second = ingress.handle_app_mention(
        body=app_mention_body(event_id="EvAppMentionSecond"),
        event=app_mention_event(text="<@UBOT> search duplicate event shapes"),
    )
    db_session.commit()

    task_count = db_session.scalar(select(func.count()).select_from(Task))

    assert first.created is True
    assert second.created is False
    assert second.task.id == first.task.id
    assert task_count == 1
    assert len(client.calls) == 1


def cleanup_database(session: Session) -> None:
    for model in (
        Artifact,
        LLMUsage,
        TaskEvent,
        Task,
        ModelPricing,
        EncryptedSecret,
        Installation,
    ):
        session.execute(delete(model))


def app_mention_body(*, event_id: str | None = None) -> dict[str, Any]:
    return {
        "event_id": event_id or f"Ev{uuid.uuid4().hex}",
        "team_id": "T123",
    }


def app_mention_event(
    *,
    text: str = "<@UBOT> research a topic",
    thread_ts: str | None = None,
) -> dict[str, Any]:
    event = {
        "type": "app_mention",
        "channel": "C123",
        "user": "U123",
        "text": text,
        "ts": "1716400000.000001",
    }
    if thread_ts is not None:
        event["thread_ts"] = thread_ts
    return event
