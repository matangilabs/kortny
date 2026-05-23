import os
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, delete, select
from sqlalchemy.orm import Session, sessionmaker

from kortny.db.models import (
    Artifact,
    EncryptedSecret,
    Installation,
    LLMUsage,
    ModelPricing,
    Task,
    TaskEvent,
    TaskEventType,
    TaskStatus,
)
from kortny.db.session import make_engine, make_session_factory, normalize_database_url
from kortny.tasks import TaskService
from kortny.worker import TaskWorker

TEST_POSTGRES_URL = os.environ.get("KORTNY_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    TEST_POSTGRES_URL is None,
    reason="KORTNY_TEST_POSTGRES_URL is required for worker integration tests",
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


@pytest.fixture
def worker_session_factory(engine: Engine) -> sessionmaker[Session]:
    return make_session_factory(engine=engine)


def test_worker_run_once_processes_pending_task(
    db_session: Session,
    worker_session_factory: sessionmaker[Session],
) -> None:
    claim_time = datetime(2026, 5, 23, 9, 20, tzinfo=UTC)
    task = create_task(db_session, event_id="EvWorker")
    task.available_at = claim_time - timedelta(seconds=1)
    db_session.commit()

    result = TaskWorker(
        session_factory=worker_session_factory,
        worker_id="worker-test",
        lease_for=timedelta(seconds=60),
    ).run_once(now=claim_time)

    db_session.refresh(task)
    assert result.worker_id == "worker-test"
    assert result.status == TaskStatus.succeeded.value
    assert result.task_id == task.id
    assert result.handled_task is True
    assert task.status is TaskStatus.succeeded
    assert task.result_summary == (
        f"Walking skeleton processed task {task.id}: task EvWorker"
    )
    assert task.locked_by is None
    assert task.locked_at is None
    assert task.lease_expires_at is None
    assert task.finished_at is not None

    events = task_events(db_session, task)
    assert [(event.type, event.payload.get("to")) for event in events] == [
        (TaskEventType.task_created, None),
        (TaskEventType.status_changed, TaskStatus.running.value),
        (TaskEventType.log, None),
        (TaskEventType.log, None),
        (TaskEventType.status_changed, TaskStatus.succeeded.value),
    ]
    assert events[2].payload == {
        "message": "walking_skeleton_handler_started",
        "worker_id": "worker-test",
    }
    assert events[3].payload == {
        "message": "walking_skeleton_handler_completed",
        "worker_id": "worker-test",
    }


def test_worker_run_once_is_idle_without_pending_task(
    db_session: Session,
    worker_session_factory: sessionmaker[Session],
) -> None:
    result = TaskWorker(
        session_factory=worker_session_factory,
        worker_id="worker-test",
    ).run_once(now=datetime(2026, 5, 23, 9, 25, tzinfo=UTC))

    assert result.worker_id == "worker-test"
    assert result.status == "idle"
    assert result.task_id is None
    assert result.handled_task is False


def test_worker_marks_task_failed_when_handler_raises(
    db_session: Session,
    worker_session_factory: sessionmaker[Session],
) -> None:
    claim_time = datetime(2026, 5, 23, 9, 30, tzinfo=UTC)
    task = create_task(db_session, event_id="EvWorkerFailure")
    task.available_at = claim_time - timedelta(seconds=1)
    db_session.commit()

    def fail_handler(task: Task) -> str:
        raise RuntimeError(f"boom {task.id}")

    result = TaskWorker(
        session_factory=worker_session_factory,
        worker_id="worker-test",
        handler=fail_handler,
    ).run_once(now=claim_time)

    db_session.refresh(task)
    assert result.status == TaskStatus.failed.value
    assert result.task_id == task.id
    assert task.status is TaskStatus.failed
    assert task.error is not None
    assert task.error["type"] == "RuntimeError"
    assert task.locked_by is None

    events = task_events(db_session, task)
    assert events[-2].type is TaskEventType.error
    assert events[-2].payload["message"] == "walking_skeleton_handler_failed"
    assert events[-1].type is TaskEventType.status_changed
    assert events[-1].payload["to"] == TaskStatus.failed.value


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


def create_installation(session: Session) -> Installation:
    installation = Installation(slack_team_id=f"T{uuid.uuid4().hex}")
    session.add(installation)
    session.flush()
    return installation


def create_task(session: Session, *, event_id: str) -> Task:
    installation = create_installation(session)
    return TaskService(session).create_task(
        installation_id=installation.id,
        slack_event_id=event_id,
        slack_channel_id="C123",
        slack_thread_ts=event_id,
        slack_message_ts=event_id,
        slack_user_id="U123",
        input=f"task {event_id}",
    )


def task_events(session: Session, task: Task) -> list[TaskEvent]:
    return list(
        session.scalars(
            select(TaskEvent)
            .where(TaskEvent.task_id == task.id)
            .order_by(TaskEvent.seq)
        )
    )
