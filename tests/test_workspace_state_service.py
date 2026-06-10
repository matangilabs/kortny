import os
import uuid
from collections.abc import Iterator
from typing import Any

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, delete, func, select, update
from sqlalchemy.orm import Session

from kortny.db.models import (
    Artifact,
    EncryptedSecret,
    Installation,
    LLMUsage,
    ModelPricing,
    SlackInboundEvent,
    SlackSideEffect,
    Task,
    TaskEvent,
    TaskEventType,
    WorkspaceState,
)
from kortny.db.session import make_engine, make_session_factory, normalize_database_url
from kortny.memory import Fact, WorkspaceStateSecretError, WorkspaceStateService
from kortny.slack import SlackIngress
from kortny.slack.posting import SlackThread
from kortny.tasks import TaskService
from kortny.tools import (
    ForgetFactTool,
    InspectMemoryTool,
    RecallFactTool,
    RememberFactTool,
)

TEST_POSTGRES_URL = os.environ.get("KORTNY_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    TEST_POSTGRES_URL is None,
    reason="KORTNY_TEST_POSTGRES_URL is required for workspace memory tests",
)


class FakeConfirmationPoster:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def post_message(
        self,
        thread: SlackThread,
        text: str,
        *,
        purpose: str = "result",
    ) -> str:
        message_ts = f"1716600000.{len(self.calls) + 1:06d}"
        self.calls.append(
            {
                "channel": thread.channel_id,
                "thread_ts": thread.thread_ts,
                "task_id": thread.task_id,
                "text": text,
                "purpose": purpose,
                "message_ts": message_ts,
            }
        )
        return message_ts


class FakeSlackClient:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []

    def chat_postMessage(
        self,
        *,
        channel: str,
        text: str,
        thread_ts: str | None = None,
        blocks: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        self.messages.append(
            {
                "channel": channel,
                "text": text,
                "thread_ts": thread_ts,
                "blocks": blocks,
            }
        )
        return {"ok": True, "channel": channel, "ts": "1716600100.000001"}


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


def test_propose_confirm_round_trip_materializes_active_fact(
    db_session: Session,
) -> None:
    task = create_task(db_session)
    poster = FakeConfirmationPoster()
    service = WorkspaceStateService(db_session, poster=poster)

    pending = service.propose(
        task.installation_id,
        "channel",
        "C123",
        "Preferred Report Style",
        {"tone": "concise", "format": "pdf"},
        task.id,
        value_text="Concise PDF reports",
        proposed_reason="User asked Kortny to remember it.",
        confidence_score="0.9",
    )

    state_count = db_session.scalar(select(func.count()).select_from(WorkspaceState))
    proposal_event = db_session.scalar(
        select(TaskEvent).where(
            TaskEvent.type == TaskEventType.log,
            TaskEvent.payload["message"].as_string()
            == "workspace_state_proposal_created",
        )
    )

    assert state_count == 0
    assert poster.calls[0]["purpose"] == "memory_confirmation"
    assert poster.calls[0]["text"] == (
        "Should I remember this for this channel?\n"
        "Concise PDF reports\n\n"
        "React with :white_check_mark: to save it or :no_entry_sign: to skip."
    )
    assert "C123" not in poster.calls[0]["text"]
    assert "preferred_report_style" not in poster.calls[0]["text"]
    assert pending.prompt_message_ts == "1716600000.000001"
    assert proposal_event is not None
    assert proposal_event.payload["status"] == "pending"

    slack_client = FakeSlackClient()
    result = SlackIngress(
        session=db_session,
        client=slack_client,
    ).handle_reaction_added(
        body={"event_id": "EvMemoryConfirm", "team_id": "T123"},
        event=reaction_event(
            reaction="white_check_mark",
            user="UConfirming",
            channel="C123",
            ts=pending.prompt_message_ts,
        ),
    )

    fact = service.get(
        task.installation_id,
        "channel",
        "C123",
        "preferred_report_style",
    )
    db_session.refresh(proposal_event)

    assert result.handled is True
    assert result.action == "confirm_memory"
    assert fact is not None
    assert fact.status == "active"
    assert fact.value == {"tone": "concise", "format": "pdf"}
    assert fact.value_text == "Concise PDF reports"
    assert fact.confirmed_by_user_id == "UConfirming"
    assert proposal_event.payload["status"] == "confirmed"
    assert proposal_event.payload["workspace_state_id"] == str(fact.id)
    assert slack_client.messages == [
        {
            "channel": "C123",
            "text": "Saved. I'll use this going forward: Concise PDF reports",
            "thread_ts": "1716500000.000001",
            "blocks": None,
        }
    ]


def test_propose_preserves_structured_details_when_value_text_is_lossy(
    db_session: Session,
) -> None:
    task = create_task(db_session)
    poster = FakeConfirmationPoster()
    service = WorkspaceStateService(db_session, poster=poster)

    pending = service.propose(
        task.installation_id,
        "user",
        "U123",
        "pdf_branding",
        {
            "document_type": "PDF",
            "style": "clean and professional",
            "footer_left": "Longboard Asset Management",
            "brand_color": "blue",
        },
        task.id,
        value_text="Clean and professional style for all PDFs",
    )

    expected_value_text = (
        "Clean and professional style for all PDFs; "
        "footer left: Longboard Asset Management; brand color: blue"
    )

    assert pending.value_text == expected_value_text
    assert poster.calls[0]["text"] == (
        "Should I remember this for you?\n"
        f"{expected_value_text}\n\n"
        "React with :white_check_mark: to save it or :no_entry_sign: to skip."
    )


def test_propose_reject_does_not_write_workspace_state(db_session: Session) -> None:
    task = create_task(db_session)
    poster = FakeConfirmationPoster()
    service = WorkspaceStateService(db_session, poster=poster)
    pending = service.propose(
        task.installation_id,
        "workspace",
        None,
        "company_name",
        {"name": "Highbrow"},
        task.id,
    )

    slack_client = FakeSlackClient()
    result = SlackIngress(
        session=db_session,
        client=slack_client,
    ).handle_reaction_added(
        body={"event_id": "EvMemoryReject", "team_id": "T123"},
        event=reaction_event(
            reaction="no_entry_sign",
            user="URejecting",
            channel="C123",
            ts=pending.prompt_message_ts,
        ),
    )

    state_count = db_session.scalar(select(func.count()).select_from(WorkspaceState))
    proposal_event = db_session.scalar(
        select(TaskEvent).where(TaskEvent.id == pending.event_id)
    )

    assert result.handled is True
    assert result.action == "reject_memory"
    assert state_count == 0
    assert proposal_event is not None
    assert proposal_event.payload["status"] == "rejected"
    assert proposal_event.payload["rejected_by_user_id"] == "URejecting"
    assert slack_client.messages == [
        {
            "channel": "C123",
            "text": "No problem, I won't save that.",
            "thread_ts": "1716500000.000001",
            "blocks": None,
        }
    ]


def test_confirm_supersedes_existing_active_fact_and_history_keeps_chain(
    db_session: Session,
) -> None:
    task = create_task(db_session)
    poster = FakeConfirmationPoster()
    service = WorkspaceStateService(db_session, poster=poster)

    first = propose_and_confirm(
        service,
        task=task,
        key="preferred_model",
        value={"model": "haiku"},
        user_id="U1",
    )
    second = propose_and_confirm(
        service,
        task=task,
        key="preferred_model",
        value={"model": "sonnet"},
        user_id="U2",
    )
    current = service.get(task.installation_id, "user", "U123", "preferred_model")
    listed = service.list(task.installation_id, scope_type="user", scope_id="U123")
    history = service.list_history(
        task.installation_id,
        scope_type="user",
        scope_id="U123",
        key="preferred_model",
    )
    superseded_row = db_session.get(WorkspaceState, history[0].id)
    active_row = db_session.get(WorkspaceState, history[1].id)

    assert first.status == "active"
    assert second.status == "active"
    assert [fact.status for fact in history] == ["superseded", "active"]
    assert superseded_row is not None
    assert active_row is not None
    assert superseded_row.superseded_by_id == active_row.id
    assert current is not None
    assert current.value == {"model": "sonnet"}
    assert [fact.id for fact in listed] == [second.id]


def test_forget_removes_only_current_active_fact(db_session: Session) -> None:
    task = create_task(db_session)
    poster = FakeConfirmationPoster()
    service = WorkspaceStateService(db_session, poster=poster)

    propose_and_confirm(
        service,
        task=task,
        key="report_cadence",
        value={"cadence": "weekly"},
        user_id="U1",
    )
    active = propose_and_confirm(
        service,
        task=task,
        key="report_cadence",
        value={"cadence": "daily"},
        user_id="U2",
    )

    forgotten = service.forget(
        task.installation_id,
        "user",
        "U123",
        "report_cadence",
        "UForget",
    )
    listed = service.list(task.installation_id, scope_type="user", scope_id="U123")
    history = service.list_history(
        task.installation_id,
        scope_type="user",
        scope_id="U123",
        key="report_cadence",
    )

    assert forgotten == 1
    assert listed == []
    assert [fact.status for fact in history] == ["superseded", "forgotten"]
    assert history[-1].id == active.id


def test_memory_tool_inspect_and_forget_preserve_audit_history(
    db_session: Session,
) -> None:
    task = create_task(db_session)
    poster = FakeConfirmationPoster()
    service = WorkspaceStateService(db_session, poster=poster)
    propose_and_confirm(
        service,
        task=task,
        key="pdf_style",
        value={"style": "concise"},
        user_id="U1",
    )
    current = propose_and_confirm(
        service,
        task=task,
        key="pdf_style",
        value={"style": "board-ready"},
        user_id="U2",
    )

    inspect = InspectMemoryTool(service=service, task=task)
    forget = ForgetFactTool(service=service, task=task)

    listed = inspect.invoke({"scope": "user"}).output
    forgotten = forget.invoke({"scope": "user", "key": "pdf_style"}).output
    after_forget = inspect.invoke({"scope": "user"}).output
    history = inspect.invoke(
        {"scope": "user", "key": "pdf_style", "include_history": True}
    ).output
    audit_events = list(
        db_session.scalars(
            select(TaskEvent)
            .where(TaskEvent.task_id == task.id)
            .order_by(TaskEvent.seq)
        )
    )

    assert listed["count"] == 1
    assert listed["facts"][0]["id"] == str(current.id)
    assert listed["facts"][0]["status"] == "active"
    assert listed["facts"][0]["source_task_id"] == str(task.id)
    assert forgotten["forgotten_count"] == 1
    assert after_forget["count"] == 0
    assert history["count"] == 2
    assert [fact["status"] for fact in history["facts"]] == [
        "superseded",
        "forgotten",
    ]
    assert history["facts"][-1]["id"] == str(current.id)
    assert [
        event.payload.get("message")
        for event in audit_events
        if event.payload.get("message")
        in {"workspace_state_inspected", "workspace_state_forget_requested"}
    ] == [
        "workspace_state_inspected",
        "workspace_state_forget_requested",
        "workspace_state_inspected",
        "workspace_state_inspected",
    ]


def test_propose_blocks_secret_like_memory_and_records_audit_event(
    db_session: Session,
) -> None:
    task = create_task(db_session)
    poster = FakeConfirmationPoster()
    service = WorkspaceStateService(db_session, poster=poster)

    with pytest.raises(WorkspaceStateSecretError) as exc_info:
        service.propose(
            task.installation_id,
            "user",
            "U123",
            "openrouter_api_key",
            {"value": "sk-or-v1-this-is-a-secret-value"},
            task.id,
        )

    state_count = db_session.scalar(select(func.count()).select_from(WorkspaceState))
    blocked_event = db_session.scalar(
        select(TaskEvent).where(
            TaskEvent.payload["message"].as_string() == "workspace_state_secret_blocked"
        )
    )

    assert "secret" in exc_info.value.reason
    assert state_count == 0
    assert poster.calls == []
    assert blocked_event is not None
    assert blocked_event.payload["key"] == "openrouter_api_key"
    assert "sk-or" not in str(blocked_event.payload)


def test_remember_fact_tool_returns_recoverable_error_for_secret(
    db_session: Session,
) -> None:
    task = create_task(db_session)
    poster = FakeConfirmationPoster()
    service = WorkspaceStateService(db_session, poster=poster)
    remember = RememberFactTool(service=service, task=task)

    result = remember.invoke(
        {
            "scope": "user",
            "key": "api_key",
            "value": {"value": "sk-or-v1-this-is-a-secret-value"},
        }
    )

    assert result.output["status"] == "blocked"
    assert result.output["error"]["code"] == "secret_not_stored"
    assert result.output["error"]["recoverable"] is True
    assert poster.calls == []


def test_memory_tools_propose_and_recall_fact(db_session: Session) -> None:
    task = create_task(db_session)
    poster = FakeConfirmationPoster()
    service = WorkspaceStateService(db_session, poster=poster)

    remember = RememberFactTool(service=service, task=task)
    recall = RecallFactTool(service=service, task=task)
    pending_result = remember.invoke(
        {
            "scope": "user",
            "key": "preferred_slack_style",
            "value": {"style": "direct"},
            "value_text": "Direct Slack replies",
        }
    )
    pending_ts = pending_result.output["prompt_message_ts"]

    assert pending_result.output["status"] == "pending_confirmation"
    assert recall.invoke({"scope": "user", "key": "preferred_slack_style"}).output == {
        "found": False,
        "scope": "user",
        "scope_id": "U123",
        "key": "preferred_slack_style",
    }

    service.confirm(str(pending_ts), "UConfirm", channel_id="C123")
    recalled = recall.invoke({"scope": "user", "key": "preferred_slack_style"})

    assert recalled.output["found"] is True
    assert recalled.output["value"] == {"style": "direct"}
    assert recalled.output["value_text"] == "Direct Slack replies"


def propose_and_confirm(
    service: WorkspaceStateService,
    *,
    task: Task,
    key: str,
    value: dict[str, Any],
    user_id: str,
) -> Fact:
    pending = service.propose(
        task.installation_id,
        "user",
        "U123",
        key,
        value,
        task.id,
    )
    return service.confirm(pending.prompt_message_ts, user_id, channel_id="C123")


def create_task(session: Session) -> Task:
    installation = Installation(slack_team_id=f"T{uuid.uuid4().hex}")
    session.add(installation)
    session.flush()
    return TaskService(session).create_task(
        installation_id=installation.id,
        slack_event_id=f"Ev{uuid.uuid4().hex}",
        slack_channel_id="C123",
        slack_thread_ts="1716500000.000001",
        slack_message_ts="1716500000.000001",
        slack_user_id="U123",
        input="remember something",
    )


def reaction_event(
    *,
    reaction: str,
    user: str,
    channel: str,
    ts: str,
) -> dict[str, Any]:
    return {
        "type": "reaction_added",
        "reaction": reaction,
        "user": user,
        "item": {
            "type": "message",
            "channel": channel,
            "ts": ts,
        },
    }


def cleanup_database(session: Session) -> None:
    session.execute(update(WorkspaceState).values(superseded_by_id=None))
    for model in (
        WorkspaceState,
        Artifact,
        LLMUsage,
        SlackInboundEvent,
        TaskEvent,
        SlackSideEffect,
        Task,
        ModelPricing,
        EncryptedSecret,
        Installation,
    ):
        session.execute(delete(model))
