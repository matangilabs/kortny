import json
import os
import uuid
from collections.abc import Iterator, Sequence
from decimal import Decimal

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, delete, select
from sqlalchemy.orm import Session

from kortny.agent import AgentCoordinator, AgentTurnLimitError
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
from kortny.llm import ChatMessage, Completion, TokenUsage, ToolCall
from kortny.tasks import TaskService
from kortny.tools import ToolArtifact, ToolRegistry, ToolResult
from kortny.tools.types import JsonObject, JsonSchema

TEST_POSTGRES_URL = os.environ.get("KORTNY_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    TEST_POSTGRES_URL is None,
    reason="KORTNY_TEST_POSTGRES_URL is required for agent coordinator tests",
)


class FakeLLM:
    def __init__(self, completions: Sequence[Completion]) -> None:
        self.completions = list(completions)
        self.calls: list[
            tuple[uuid.UUID, tuple[ChatMessage, ...], tuple[JsonSchema, ...]]
        ] = []

    def complete(
        self,
        *,
        task_id: uuid.UUID,
        messages: Sequence[ChatMessage],
        tools: Sequence[JsonSchema] = (),
    ) -> Completion:
        self.calls.append((task_id, tuple(messages), tuple(tools)))
        if not self.completions:
            raise AssertionError("FakeLLM received more calls than expected")
        return self.completions.pop(0)


class EchoJsonTool:
    name = "echo_json"
    description = "Echoes JSON arguments."
    parameters: JsonSchema = {
        "type": "object",
        "properties": {"message": {"type": "string"}},
        "required": ["message"],
        "additionalProperties": False,
    }

    def invoke(self, args: JsonObject) -> ToolResult:
        return ToolResult(output={"echoed": args["message"]}, cost_usd=Decimal("0.1"))


class ArtifactTool:
    name = "make_artifact"
    description = "Returns an artifact."
    parameters: JsonSchema = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }

    def invoke(self, args: JsonObject) -> ToolResult:
        return ToolResult(
            output={"created": True},
            artifacts=(
                ToolArtifact(
                    filename="report.pdf",
                    path="/tmp/report.pdf",
                    mime_type="application/pdf",
                    size_bytes=42,
                ),
            ),
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


def test_coordinator_finishes_with_final_answer(db_session: Session) -> None:
    task = create_task(db_session, input_text="summarize this")
    llm = FakeLLM(
        [
            Completion(
                content="Here is the summary.",
                tool_calls=(),
                usage=TokenUsage(input_tokens=10, output_tokens=5),
                response_id="gen-final",
                model="openai/gpt-4o-mini",
            )
        ]
    )

    result = AgentCoordinator(
        session=db_session,
        llm=llm,
        registry=ToolRegistry(),
    ).run(task)

    assert result.result_summary == "Here is the summary."
    assert result.turns == 1
    assert result.artifact_count == 0
    assert task.result_summary == "Here is the summary."
    assert llm.calls[0][1] == (ChatMessage(role="user", content="summarize this"),)

    events = task_events(db_session, task)
    assert event_messages(events) == [
        "agent_started",
        "agent_llm_turn_started",
        "agent_llm_turn_completed",
        "agent_completed",
    ]
    assert events[-1].payload["reason"] == "final_answer"


def test_coordinator_invokes_tool_and_repeats_until_final_answer(
    db_session: Session,
) -> None:
    task = create_task(db_session, input_text="echo hi")
    llm = FakeLLM(
        [
            Completion(
                content=None,
                tool_calls=(
                    ToolCall(
                        id="call-1",
                        name="echo_json",
                        arguments={"message": "hi"},
                    ),
                ),
                usage=TokenUsage(input_tokens=20, output_tokens=3),
            ),
            Completion(
                content="Echoed hi.",
                tool_calls=(),
                usage=TokenUsage(input_tokens=25, output_tokens=6),
            ),
        ]
    )

    result = AgentCoordinator(
        session=db_session,
        llm=llm,
        registry=ToolRegistry([EchoJsonTool()]),
    ).run(task)

    assert result.result_summary == "Echoed hi."
    assert result.turns == 2
    assert len(llm.calls) == 2
    assert llm.calls[0][2][0]["name"] == "echo_json"

    second_turn_messages = llm.calls[1][1]
    assert second_turn_messages[0] == ChatMessage(role="user", content="echo hi")
    assert second_turn_messages[1].tool_calls == (
        ToolCall(id="call-1", name="echo_json", arguments={"message": "hi"}),
    )
    tool_message = second_turn_messages[2]
    assert tool_message.role == "tool"
    assert tool_message.tool_call_id == "call-1"
    assert json.loads(tool_message.content or "{}")["output"] == {"echoed": "hi"}

    events = task_events(db_session, task)
    assert [event.type for event in events if event.type in tool_event_types()] == [
        TaskEventType.tool_call,
        TaskEventType.tool_result,
    ]
    tool_result = next(
        event for event in events if event.type is TaskEventType.tool_result
    )
    assert tool_result.payload["output"] == {"echoed": "hi"}
    assert tool_result.payload["cost_usd"] == "0.1"


def test_coordinator_stops_when_tool_returns_artifact(db_session: Session) -> None:
    task = create_task(db_session, input_text="make a report")
    llm = FakeLLM(
        [
            Completion(
                content=None,
                tool_calls=(ToolCall(id="call-1", name="make_artifact", arguments={}),),
                usage=TokenUsage(input_tokens=20, output_tokens=3),
            ),
            Completion(
                content="This should not be called.",
                tool_calls=(),
                usage=TokenUsage(input_tokens=1, output_tokens=1),
            ),
        ]
    )

    result = AgentCoordinator(
        session=db_session,
        llm=llm,
        registry=ToolRegistry([ArtifactTool()]),
    ).run(task)

    assert result.result_summary == "Generated 1 artifact."
    assert result.turns == 1
    assert result.artifact_count == 1
    assert task.result_summary == "Generated 1 artifact."
    assert len(llm.calls) == 1

    events = task_events(db_session, task)
    tool_result = next(
        event for event in events if event.type is TaskEventType.tool_result
    )
    assert tool_result.payload["artifacts"] == [
        {
            "filename": "report.pdf",
            "path": "/tmp/report.pdf",
            "mime_type": "application/pdf",
            "size_bytes": 42,
        }
    ]
    assert events[-1].payload["reason"] == "artifact"


def test_coordinator_raises_after_turn_limit(db_session: Session) -> None:
    task = create_task(db_session, input_text="loop")
    llm = FakeLLM(
        [
            Completion(
                content=None,
                tool_calls=(
                    ToolCall(
                        id="call-1",
                        name="echo_json",
                        arguments={"message": "again"},
                    ),
                ),
                usage=TokenUsage(input_tokens=1, output_tokens=1),
            )
        ]
    )

    with pytest.raises(AgentTurnLimitError):
        AgentCoordinator(
            session=db_session,
            llm=llm,
            registry=ToolRegistry([EchoJsonTool()]),
            max_turns=1,
        ).run(task)

    events = task_events(db_session, task)
    assert events[-1].type is TaskEventType.error
    assert events[-1].payload["type"] == "AgentTurnLimitError"


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


def create_task(session: Session, *, input_text: str) -> Task:
    installation = Installation(slack_team_id=f"T{uuid.uuid4().hex}")
    session.add(installation)
    session.flush()
    return TaskService(session).create_task(
        installation_id=installation.id,
        slack_event_id=f"Ev{uuid.uuid4().hex}",
        slack_channel_id="C123",
        slack_thread_ts="1716400000.000001",
        slack_message_ts="1716400000.000001",
        slack_user_id="U123",
        input=input_text,
    )


def task_events(session: Session, task: Task) -> list[TaskEvent]:
    return list(
        session.scalars(
            select(TaskEvent)
            .where(TaskEvent.task_id == task.id)
            .order_by(TaskEvent.seq)
        )
    )


def event_messages(events: Sequence[TaskEvent]) -> list[str]:
    return [
        event.payload["message"] for event in events if event.type is TaskEventType.log
    ]


def tool_event_types() -> set[TaskEventType]:
    return {TaskEventType.tool_call, TaskEventType.tool_result}
