"""HIG-277: gated persona "who you're helping" block in context assembly.

The user persona fact is never injected as a plain known-fact. It surfaces
only through the dedicated persona block, and only on persona-relevant asks
(PRISM gating) so a confirmed persona can't colour every factual answer.
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
from kortny.db.models import (
    Installation,
    Task,
    TaskEvent,
    TaskEventType,
    WorkspaceState,
)
from kortny.db.session import make_engine, make_session_factory, normalize_database_url
from kortny.llm.routing import INTENT_CLASSIFIED_MESSAGE
from kortny.tasks import TaskService

TEST_POSTGRES_URL = os.environ.get("KORTNY_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    TEST_POSTGRES_URL is None,
    reason="KORTNY_TEST_POSTGRES_URL is required for persona context tests",
)

CHANNEL = "C_MAIN"
USER = "U_DEV"
PERSONA_TEXT = "Role: Software Engineer. Work surfaces: issues, prs."


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
    for model in (WorkspaceState, TaskEvent, Task, Installation):
        session.execute(delete(model))


def _installation(session: Session) -> Installation:
    installation = Installation(slack_team_id=f"T{uuid.uuid4().hex}")
    session.add(installation)
    session.flush()
    return installation


def _task(session: Session, installation: Installation) -> Task:
    return TaskService(session).create_task(
        installation_id=installation.id,
        slack_event_id=f"Ev{uuid.uuid4().hex}",
        slack_channel_id=CHANNEL,
        slack_user_id=USER,
        slack_message_ts=f"1780000000.{uuid.uuid4().hex[:6]}",
        input="what's on my plate?",
    )


def _seed_persona_fact(session: Session, installation: Installation) -> None:
    session.add(
        WorkspaceState(
            installation_id=installation.id,
            scope_type="user",
            scope_id=USER,
            key="user_profile",
            value_json={
                "role": "Software Engineer",
                "work_surfaces": ["issues", "prs"],
            },
            value_text=PERSONA_TEXT,
            status="active",
            source_kind="observer_proposed",
            proposed_by=USER,
            confirmed_by_user_id=USER,
        )
    )
    session.flush()


def _record_intent(session: Session, task: Task, *, persona_relevant: bool) -> None:
    TaskService(session).append_event(
        task,
        TaskEventType.log,
        {
            "message": INTENT_CLASSIFIED_MESSAGE,
            "decision": {"persona_relevant": persona_relevant},
        },
    )


def _persona_messages(session: Session, task: Task) -> list[str]:
    package = ContextAssembler(session=session).build_for_task(task)
    return [
        m.content
        for m in package.messages
        if m.content and m.content.startswith("Who you're helping")
    ]


def test_persona_block_injected_on_relevant_ask(db_session: Session) -> None:
    installation = _installation(db_session)
    _seed_persona_fact(db_session, installation)
    task = _task(db_session, installation)
    _record_intent(db_session, task, persona_relevant=True)

    blocks = _persona_messages(db_session, task)

    assert len(blocks) == 1
    assert "Software Engineer" in blocks[0]
    # Carries the resolution directive, not a stereotype.
    assert "find_tools" in blocks[0]


def test_persona_block_absent_when_not_relevant(db_session: Session) -> None:
    installation = _installation(db_session)
    _seed_persona_fact(db_session, installation)
    task = _task(db_session, installation)
    _record_intent(db_session, task, persona_relevant=False)

    assert _persona_messages(db_session, task) == []


def test_persona_block_absent_without_fact(db_session: Session) -> None:
    installation = _installation(db_session)
    task = _task(db_session, installation)
    _record_intent(db_session, task, persona_relevant=True)

    assert _persona_messages(db_session, task) == []


def test_persona_fact_never_rides_in_known_facts(db_session: Session) -> None:
    """Even on a relevant ask, the persona must not appear as a plain fact."""

    installation = _installation(db_session)
    _seed_persona_fact(db_session, installation)
    task = _task(db_session, installation)
    _record_intent(db_session, task, persona_relevant=True)

    package = ContextAssembler(session=db_session).build_for_task(task)
    carrying = [
        m.content for m in package.messages if m.content and PERSONA_TEXT in m.content
    ]
    # The persona text appears in exactly one message: the dedicated block.
    assert len(carrying) == 1
    assert carrying[0].startswith("Who you're helping")
