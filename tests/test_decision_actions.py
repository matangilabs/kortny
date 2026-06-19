"""DB-backed tests for the approval decision flow (HIG-255 slice 2a).

Exercises the button path end-to-end through process_decision_action: claim →
TaskService transition → complete + supersede, plus idempotency and the
wrong-user denial leaving the task untouched.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from types import SimpleNamespace
from typing import cast

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, delete
from sqlalchemy.orm import Session

from kortny.approvals import TOOL_APPROVAL_REQUIRED_MESSAGE
from kortny.config import Settings
from kortny.db.models import (
    Installation,
    InteractiveAction,
    Task,
    TaskEvent,
    TaskEventType,
)
from kortny.db.models import TaskStatus as DbTaskStatus
from kortny.db.session import make_engine, make_session_factory, normalize_database_url
from kortny.slack.decisions import (
    ROUTE_APPROVAL_APPROVE,
    ROUTE_APPROVAL_REJECT,
    DecisionOutcome,
    approval_decision,
    process_decision_action,
    render_decision,
)
from kortny.slack.interactions import (
    STATUS_CONSUMED,
    STATUS_SUPERSEDED,
    InteractiveActionService,
)
from kortny.tasks import TaskService

TEST_POSTGRES_URL = os.environ.get("KORTNY_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    TEST_POSTGRES_URL is None,
    reason="KORTNY_TEST_POSTGRES_URL is required for decision-action tests",
)

OWNER = "U_OWNER"
TEAM = "T_ACME"
CHANNEL = "C_MAIN"
APPROVAL_KEY = "appr-123"
SETTINGS = cast(Settings, SimpleNamespace(encryption_key="test-signing-key"))


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
    for model in (InteractiveAction, TaskEvent, Task, Installation):
        session.execute(delete(model))


def _status(session: Session, task_id: uuid.UUID) -> DbTaskStatus:
    task = TaskService(session).get_task(task_id)
    assert task is not None
    return DbTaskStatus(task.status)


def _waiting_task(session: Session) -> tuple[Installation, Task]:
    installation = Installation(slack_team_id=TEAM, team_name="Acme")
    session.add(installation)
    session.flush()
    service = TaskService(session)
    task = service.create_task(
        installation_id=installation.id,
        slack_event_id=f"Ev{uuid.uuid4().hex}",
        slack_channel_id=CHANNEL,
        slack_user_id=OWNER,
        slack_message_ts=f"178100.{uuid.uuid4().hex[:6]}",
        input="run the thing",
    )
    # The coordinator records the REQUIRED event (carries the request that
    # latest_pending_tool_approval matches); mark_waiting parks the task.
    service.append_event(
        task,
        TaskEventType.log,
        {
            "message": TOOL_APPROVAL_REQUIRED_MESSAGE,
            "request": {"approval_key": APPROVAL_KEY, "tool_name": "code_exec"},
        },
    )
    service.mark_waiting_for_tool_approval(
        task,
        request={"approval_key": APPROVAL_KEY, "tool": "code_exec"},
        prompt_message_ts="178100.aaaaaa",
    )
    return installation, task


def test_approve_button_transitions_task_and_supersedes_reject(
    db_session: Session,
) -> None:
    installation, task = _waiting_task(db_session)
    service = InteractiveActionService(db_session, signing_key="test-signing-key")
    spec = approval_decision(
        approval_key=APPROVAL_KEY,
        tool_name="code_exec",
        statement="Approve running code_exec?",
        fallback_text="Approve / Reject below.",
    )
    _, minted = render_decision(
        spec,
        service,
        installation_id=installation.id,
        task_id=task.id,
        allowed_user_id=OWNER,
        allowed_channel_id=CHANNEL,
        slack_team_id=TEAM,
    )
    for m in minted:
        service.mark_sent(
            m.action,
            channel_id=CHANNEL,
            message_ts="178100.bbbbbb",
            block_id="b1",
            slack_action_id=m.action.route,
        )
    approve_key = next(
        m.raw_key for m in minted if m.action.route == ROUTE_APPROVAL_APPROVE
    )
    approve_id = next(
        m.action.id for m in minted if m.action.route == ROUTE_APPROVAL_APPROVE
    )
    reject_id = next(
        m.action.id for m in minted if m.action.route == ROUTE_APPROVAL_REJECT
    )

    outcome = process_decision_action(
        db_session,
        SETTINGS,
        action_id=ROUTE_APPROVAL_APPROVE,
        raw_key=approve_key,
        actor_user_id=OWNER,
        team_id=TEAM,
        channel_id=CHANNEL,
    )
    assert outcome is DecisionOutcome.applied
    db_session.expire_all()
    assert _status(db_session, task.id) == DbTaskStatus.pending
    approve_row = db_session.get(InteractiveAction, approve_id)
    reject_row = db_session.get(InteractiveAction, reject_id)
    assert approve_row is not None and approve_row.status == STATUS_CONSUMED
    assert reject_row is not None and reject_row.status == STATUS_SUPERSEDED

    # Second click is a no-op.
    again = process_decision_action(
        db_session,
        SETTINGS,
        action_id=ROUTE_APPROVAL_APPROVE,
        raw_key=approve_key,
        actor_user_id=OWNER,
        team_id=TEAM,
        channel_id=CHANNEL,
    )
    assert again is DecisionOutcome.already_handled


def test_wrong_user_click_does_not_transition_task(db_session: Session) -> None:
    installation, task = _waiting_task(db_session)
    service = InteractiveActionService(db_session, signing_key="test-signing-key")
    spec = approval_decision(
        approval_key=APPROVAL_KEY,
        tool_name="code_exec",
        statement="Approve?",
        fallback_text="Approve / Reject below.",
    )
    _, minted = render_decision(
        spec,
        service,
        installation_id=installation.id,
        task_id=task.id,
        allowed_user_id=OWNER,
        allowed_channel_id=CHANNEL,
        slack_team_id=TEAM,
    )
    for m in minted:
        service.mark_sent(
            m.action,
            channel_id=CHANNEL,
            message_ts="178100.bbbbbb",
            block_id="b1",
            slack_action_id=m.action.route,
        )
    approve_key = next(
        m.raw_key for m in minted if m.action.route == ROUTE_APPROVAL_APPROVE
    )

    outcome = process_decision_action(
        db_session,
        SETTINGS,
        action_id=ROUTE_APPROVAL_APPROVE,
        raw_key=approve_key,
        actor_user_id="U_INTRUDER",
        team_id=TEAM,
        channel_id=CHANNEL,
    )
    assert outcome is DecisionOutcome.denied
    db_session.expire_all()
    # Task is untouched — still waiting.
    assert _status(db_session, task.id) == DbTaskStatus.waiting_approval


def test_forged_key_is_a_no_op(db_session: Session) -> None:
    _waiting_task(db_session)
    outcome = process_decision_action(
        db_session,
        SETTINGS,
        action_id=ROUTE_APPROVAL_APPROVE,
        raw_key="iact_v1_forged",
        actor_user_id=OWNER,
        team_id=TEAM,
        channel_id=CHANNEL,
    )
    assert outcome is DecisionOutcome.not_found
