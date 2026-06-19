"""HIG-244: in-thread document spec injected into context for conversational edits.

When a thread already holds a Document Studio doc, the agent should see its
stored spec + lineage so "shorten that" / "add a chart" revises it in place
rather than regenerating from scratch.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, delete
from sqlalchemy.orm import Session

from kortny.agent.context import ContextAssembler
from kortny.db.models import Artifact, Installation, Task, TaskEvent
from kortny.db.session import make_engine, make_session_factory, normalize_database_url
from kortny.tasks import TaskService
from kortny.tasks.identity import TaskIdentity

TEST_POSTGRES_URL = os.environ.get("KORTNY_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    TEST_POSTGRES_URL is None,
    reason="KORTNY_TEST_POSTGRES_URL is required for document context tests",
)

CHANNEL = "C_MAIN"
USER = "U_DEV"
THREAD = "1780000000.000100"


@pytest.fixture(scope="session")
def engine() -> Iterator[Engine]:
    assert TEST_POSTGRES_URL is not None
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", normalize_database_url(TEST_POSTGRES_URL))
    command.upgrade(config, "heads")
    eng = make_engine(TEST_POSTGRES_URL)
    try:
        yield eng
    finally:
        eng.dispose()


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
    for model in (Artifact, TaskEvent, Task, Installation):
        session.execute(delete(model))


def _installation(session: Session) -> Installation:
    installation = Installation(slack_team_id=f"T{uuid.uuid4().hex}")
    session.add(installation)
    session.flush()
    return installation


def _task(
    session: Session, installation: Installation, *, thread_ts: str | None
) -> Task:
    return TaskService(session).create_task(
        installation_id=installation.id,
        slack_channel_id=CHANNEL,
        slack_user_id=USER,
        slack_thread_ts=thread_ts,
        input="add a Q3 chart",
        identity=TaskIdentity.manual(
            channel_id=CHANNEL,
            thread_ts=thread_ts,
            user_id=USER,
            input_text=f"add a Q3 chart {uuid.uuid4().hex}",
        ),
    )


def _seed_doc(
    session: Session, parent: Task, *, group: uuid.UUID, version: int
) -> None:
    session.add(
        Artifact(
            task_id=parent.id,
            filename="report.pdf",
            mime_type="application/pdf",
            doc_group_id=group,
            doc_version=version,
            spec_json={
                "title": "Q2 Report",
                "blocks": [{"type": "prose", "text": "x"}],
            },
        )
    )
    session.flush()


def _doc_messages(session: Session, task: Task) -> list[str]:
    package = ContextAssembler(session=session).build_for_task(task)
    return [
        m.content
        for m in package.messages
        if m.content and "doc_group_id=" in m.content
    ]


def test_thread_doc_spec_injected_for_followup(db_session: Session) -> None:
    installation = _installation(db_session)
    group = uuid.uuid4()
    doc_task = _task(db_session, installation, thread_ts=THREAD)
    _seed_doc(db_session, doc_task, group=group, version=2)
    followup = _task(db_session, installation, thread_ts=THREAD)

    blocks = _doc_messages(db_session, followup)

    assert len(blocks) == 1
    assert str(group) in blocks[0]
    assert "base_version=2" in blocks[0]
    assert "Q2 Report" in blocks[0]


def test_no_doc_context_without_thread_doc(db_session: Session) -> None:
    installation = _installation(db_session)
    task = _task(db_session, installation, thread_ts=THREAD)
    assert _doc_messages(db_session, task) == []
