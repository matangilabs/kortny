import os
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, delete, select
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
from kortny.slack import SlackPoster, SlackThread
from kortny.tasks import TaskService

TEST_POSTGRES_URL = os.environ.get("KORTNY_TEST_POSTGRES_URL")


class FakeSlackClient:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []
        self.uploads: list[dict[str, Any]] = []

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

    def files_upload_v2(
        self,
        *,
        file: str,
        filename: str | None = None,
        title: str | None = None,
        channel: str | None = None,
        initial_comment: str | None = None,
        thread_ts: str | None = None,
    ) -> dict[str, Any]:
        self.uploads.append(
            {
                "file": file,
                "filename": filename,
                "title": title,
                "channel": channel,
                "initial_comment": initial_comment,
                "thread_ts": thread_ts,
            }
        )
        return {"ok": True, "files": [{"id": f"F{len(self.uploads):06d}"}]}


@pytest.fixture(scope="session")
def engine() -> Iterator[Engine]:
    if TEST_POSTGRES_URL is None:
        pytest.skip("KORTNY_TEST_POSTGRES_URL is required for Slack posting tests")

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


def test_post_message_posts_to_thread_and_logs_event(db_session: Session) -> None:
    task = create_task(db_session)
    client = FakeSlackClient()
    message_ts = SlackPoster(session=db_session, client=client).post_message(
        SlackThread.from_task(task),
        "Done.",
    )

    event = db_session.scalar(
        select(TaskEvent).where(
            TaskEvent.task_id == task.id,
            TaskEvent.type == TaskEventType.message_posted,
        )
    )

    assert message_ts == "1716400001.000001"
    assert client.messages == [
        {
            "channel": "C123",
            "text": "Done.",
            "thread_ts": "1716400000.000001",
        }
    ]
    assert event is not None
    assert event.payload == {
        "channel": "C123",
        "thread_ts": "1716400000.000001",
        "message_ts": "1716400001.000001",
        "text": "Done.",
        "purpose": "result",
    }


def test_post_message_in_dm_posts_without_thread_ts(db_session: Session) -> None:
    task = create_task(db_session, channel_id="D123")
    client = FakeSlackClient()

    SlackPoster(session=db_session, client=client).post_message(
        SlackThread.from_task(task),
        "Done.",
    )

    event = db_session.scalar(
        select(TaskEvent)
        .where(
            TaskEvent.task_id == task.id,
            TaskEvent.type == TaskEventType.message_posted,
        )
        .order_by(TaskEvent.seq.desc())
        .limit(1)
    )

    assert client.messages == [
        {
            "channel": "D123",
            "text": "Done.",
            "thread_ts": None,
        }
    ]
    assert event is not None
    assert event.payload["thread_ts"] is None


def test_post_message_normalizes_slack_mrkdwn(db_session: Session) -> None:
    task = create_task(db_session)
    client = FakeSlackClient()

    SlackPoster(session=db_session, client=client).post_message(
        SlackThread.from_task(task),
        "### Capabilities\n1. **Web Searches:** Read [docs](https://docs.slack.dev).",
    )

    event = db_session.scalar(
        select(TaskEvent)
        .where(
            TaskEvent.task_id == task.id,
            TaskEvent.type == TaskEventType.message_posted,
        )
        .order_by(TaskEvent.seq.desc())
        .limit(1)
    )

    expected_text = (
        "*Capabilities*\n1. *Web Searches:* Read <https://docs.slack.dev|docs>."
    )
    assert client.messages[0]["text"] == expected_text
    assert event is not None
    assert event.payload["text"] == expected_text


def test_upload_file_updates_artifact_and_logs_event(
    db_session: Session,
    tmp_path: Path,
) -> None:
    task = create_task(db_session)
    report_path = tmp_path / "report.pdf"
    report_path.write_bytes(b"%PDF-1.4 test")
    artifact = Artifact(
        task_id=task.id,
        filename="report.pdf",
        mime_type="application/pdf",
        size_bytes=report_path.stat().st_size,
        storage_path=str(report_path),
    )
    db_session.add(artifact)
    db_session.flush()
    client = FakeSlackClient()

    slack_file_id = SlackPoster(session=db_session, client=client).upload_file(
        SlackThread.from_task(task),
        report_path,
        initial_comment="Here is the report.",
        now=datetime(2026, 5, 23, 3, 45, tzinfo=UTC),
    )

    event = db_session.scalar(
        select(TaskEvent)
        .where(
            TaskEvent.task_id == task.id,
            TaskEvent.type == TaskEventType.message_posted,
        )
        .order_by(TaskEvent.seq.desc())
        .limit(1)
    )

    assert slack_file_id == "F000001"
    assert artifact.slack_file_id == "F000001"
    assert artifact.posted_at == datetime(2026, 5, 23, 3, 45, tzinfo=UTC)
    assert client.uploads == [
        {
            "file": str(report_path),
            "filename": "report.pdf",
            "title": "report.pdf",
            "channel": "C123",
            "initial_comment": "Here is the report.",
            "thread_ts": "1716400000.000001",
        }
    ]
    assert event is not None
    assert event.payload["slack_file_id"] == "F000001"
    assert event.payload["artifact_id"] == str(artifact.id)
    assert event.payload["purpose"] == "file_upload"


def test_upload_file_in_dm_posts_without_thread_ts(
    db_session: Session,
    tmp_path: Path,
) -> None:
    task = create_task(db_session, channel_id="D123")
    report_path = tmp_path / "report.pdf"
    report_path.write_bytes(b"%PDF-1.4 test")
    artifact = Artifact(
        task_id=task.id,
        filename="report.pdf",
        mime_type="application/pdf",
        size_bytes=report_path.stat().st_size,
        storage_path=str(report_path),
    )
    db_session.add(artifact)
    db_session.flush()
    client = FakeSlackClient()

    SlackPoster(session=db_session, client=client).upload_file(
        SlackThread.from_task(task),
        report_path,
        initial_comment="Here is the report.",
    )

    event = db_session.scalar(
        select(TaskEvent)
        .where(
            TaskEvent.task_id == task.id,
            TaskEvent.type == TaskEventType.message_posted,
        )
        .order_by(TaskEvent.seq.desc())
        .limit(1)
    )

    assert client.uploads == [
        {
            "file": str(report_path),
            "filename": "report.pdf",
            "title": "report.pdf",
            "channel": "D123",
            "initial_comment": "Here is the report.",
            "thread_ts": None,
        }
    ]
    assert event is not None
    assert event.payload["thread_ts"] is None


def test_upload_file_uses_posted_at_as_dedup_guard(
    db_session: Session,
    tmp_path: Path,
) -> None:
    task = create_task(db_session)
    report_path = tmp_path / "report.pdf"
    report_path.write_bytes(b"%PDF-1.4 test")
    artifact = Artifact(
        task_id=task.id,
        filename="report.pdf",
        mime_type="application/pdf",
        size_bytes=report_path.stat().st_size,
        storage_path=str(report_path),
        slack_file_id="FALREADY",
        posted_at=datetime(2026, 5, 23, 3, 50, tzinfo=UTC),
    )
    db_session.add(artifact)
    db_session.flush()
    client = FakeSlackClient()

    slack_file_id = SlackPoster(session=db_session, client=client).upload_file(
        SlackThread.from_task(task),
        report_path,
    )

    assert slack_file_id == "FALREADY"
    assert client.uploads == []


def test_upload_file_creates_artifact_if_needed(
    db_session: Session,
    tmp_path: Path,
) -> None:
    task = create_task(db_session)
    report_path = tmp_path / "report.pdf"
    report_path.write_bytes(b"%PDF-1.4 test")
    client = FakeSlackClient()

    slack_file_id = SlackPoster(session=db_session, client=client).upload_file(
        SlackThread.from_task(task),
        report_path,
    )

    artifact = db_session.scalar(select(Artifact).where(Artifact.task_id == task.id))
    artifact_event = db_session.scalar(
        select(TaskEvent).where(
            TaskEvent.task_id == task.id,
            TaskEvent.type == TaskEventType.artifact_created,
        )
    )

    assert slack_file_id == "F000001"
    assert artifact is not None
    assert artifact.filename == "report.pdf"
    assert artifact.mime_type == "application/pdf"
    assert artifact.storage_path == str(report_path)
    assert artifact.slack_file_id == "F000001"
    assert artifact.posted_at is not None
    assert artifact_event is not None
    assert artifact_event.payload["artifact_id"] == str(artifact.id)


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


def create_task(session: Session, *, channel_id: str = "C123") -> Task:
    installation = Installation(slack_team_id=f"T{uuid.uuid4().hex}")
    session.add(installation)
    session.flush()
    return TaskService(session).create_task(
        installation_id=installation.id,
        slack_event_id=f"Ev{uuid.uuid4().hex}",
        slack_channel_id=channel_id,
        slack_thread_ts="1716400000.000001",
        slack_message_ts="1716400000.000001",
        slack_user_id="U123",
        input="make a report",
    )
