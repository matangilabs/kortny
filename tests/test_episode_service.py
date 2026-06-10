import os
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, delete, select
from sqlalchemy.orm import Session

from kortny.db.models import (
    Artifact,
    EncryptedSecret,
    Episode,
    Installation,
    LLMUsage,
    ModelPricing,
    Task,
    TaskEvent,
    TaskEventType,
    TaskStatus,
    WorkspaceState,
)
from kortny.db.session import make_engine, make_session_factory, normalize_database_url
from kortny.memory import EpisodeService
from kortny.tasks import TaskService

TEST_POSTGRES_URL = os.environ.get("KORTNY_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    TEST_POSTGRES_URL is None,
    reason="KORTNY_TEST_POSTGRES_URL is required for episode memory tests",
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


def test_episode_service_records_compact_task_provenance(
    db_session: Session,
) -> None:
    installation = create_installation(db_session)
    task = create_task(
        db_session,
        installation=installation,
        input_text="research tempfile and make a PDF",
        event_id="EvEpisode",
    )
    service = TaskService(db_session)
    task.result_summary = "Generated a concise tempfile report."
    service.append_event(
        task,
        TaskEventType.tool_call,
        {
            "turn": 1,
            "tool_call_id": "call-search",
            "tool": "web_search",
            "arguments": {"query": "Python tempfile best practices"},
        },
    )
    service.append_event(
        task,
        TaskEventType.tool_result,
        {
            "turn": 1,
            "tool_call_id": "call-search",
            "tool": "web_search",
            "output": {
                "provider": "brave",
                "query": "Python tempfile best practices",
                "results": [
                    {
                        "title": "tempfile docs",
                        "url": "https://docs.python.org/3/library/tempfile.html",
                        "snippet": "Temporary files and directories.",
                    }
                ],
            },
            "cost_usd": "0",
            "artifacts": [],
        },
    )
    service.append_event(
        task,
        TaskEventType.tool_call,
        {
            "turn": 2,
            "tool_call_id": "call-pdf",
            "tool": "pdf_generator",
            "arguments": {"filename": "tempfile_report.pdf"},
        },
    )
    db_session.add(
        Artifact(
            task_id=task.id,
            filename="tempfile_report.pdf",
            mime_type="application/pdf",
            size_bytes=1234,
            slack_file_id="FEPISODE",
        )
    )
    service.transition(task, TaskStatus.succeeded)

    episode = EpisodeService(db_session).record_task(task)

    assert episode is not None
    assert episode.task_id == task.id
    assert episode.outcome == "succeeded"
    assert episode.summary == "Generated a concise tempfile report."
    assert episode.tools_used == ("web_search", "pdf_generator")
    assert len(episode.artifacts_created) == 1
    assert isinstance(episode.artifacts_created[0]["artifact_id"], str)
    assert episode.artifacts_created[0]["filename"] == "tempfile_report.pdf"
    assert episode.artifacts_created[0]["slack_file_id"] == "FEPISODE"
    assert episode.artifacts_created[0]["mime_type"] == "application/pdf"
    assert episode.artifacts_created[0]["size_bytes"] == 1234
    assert episode.source_refs[0]["url"] == (
        "https://docs.python.org/3/library/tempfile.html"
    )

    task.result_summary = "Updated summary."
    refreshed = EpisodeService(db_session).record_task(task)
    row_count = db_session.scalar(select(Episode).where(Episode.task_id == task.id))

    assert refreshed is not None
    assert refreshed.id == episode.id
    assert refreshed.summary == "Updated summary."
    assert row_count is not None


def test_episode_service_records_failed_task_and_retrieves_by_scope(
    db_session: Session,
) -> None:
    installation = create_installation(db_session)
    first = create_task(
        db_session,
        installation=installation,
        input_text="research AAPL earnings",
        event_id="EvFailedEpisode",
        thread_ts="1716500000.000001",
    )
    first.error = {"type": "RuntimeError", "message": "Brave Search 429"}
    TaskService(db_session).transition(first, TaskStatus.failed)
    recorded = EpisodeService(db_session).record_task(first)
    assert recorded is not None

    follow_up = create_task(
        db_session,
        installation=installation,
        input_text="did this fail before?",
        event_id="EvEpisodeFollowup",
        thread_ts="1716500000.000001",
        message_ts="1716500000.000002",
    )

    relevant = EpisodeService(db_session).relevant_for_task(follow_up)

    assert len(relevant) == 1
    assert relevant[0].relation == "same_thread"
    assert relevant[0].episode.outcome == "failed"
    assert relevant[0].episode.summary == "Task failed: RuntimeError: Brave Search 429"
    assert relevant[0].episode.error == {
        "type": "RuntimeError",
        "message": "Brave Search 429",
    }


def cleanup_database(session: Session) -> None:
    session.execute(delete(WorkspaceState))
    for model in (
        Episode,
        Artifact,
        LLMUsage,
        TaskEvent,
        Task,
        ModelPricing,
        EncryptedSecret,
        Installation,
    ):
        session.execute(delete(model))


def create_installation(session: Session) -> Installation:
    installation = Installation(slack_team_id=f"T{uuid.uuid4().hex}")
    session.add(installation)
    session.flush()
    return installation


def create_task(
    session: Session,
    *,
    installation: Installation,
    input_text: str,
    event_id: str,
    thread_ts: str = "1716400000.000001",
    message_ts: str | None = None,
) -> Task:
    # message_ts must be unique per task: TaskService dedups on the
    # slack-message identity key (channel:thread_ts:message_ts).
    task = TaskService(session).create_task(
        installation_id=installation.id,
        slack_event_id=event_id,
        slack_channel_id="C123",
        slack_thread_ts=thread_ts,
        slack_message_ts=message_ts or thread_ts,
        slack_user_id="U123",
        input=input_text,
    )
    task.created_at = datetime(2026, 5, 24, 9, 0, tzinfo=UTC)
    session.flush()
    return task
