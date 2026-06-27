"""DB-backed coordinator integration tests for the HIG-169 P0.4 trifecta gate.

Verifies the gate escalates an outward/write tool to approval ONLY after an
untrusted-origin tool result has armed the task, that it does not fire before,
that it only RAISES the approval floor (HIG-223), and that disabling it via the
flag restores ungated behavior.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator, Sequence
from decimal import Decimal

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, delete, select
from sqlalchemy.orm import Session

from kortny.agent.coordinator import TRIFECTA_GATE_MESSAGE, AgentCoordinator
from kortny.agent.planner import ExecutionPlanner, PlannerGateDecision
from kortny.agent.trifecta import is_outward_or_write_tool
from kortny.approvals import ToolApprovalRequired
from kortny.db.models import (
    Artifact,
    Installation,
    LLMUsage,
    Task,
    TaskEvent,
    TaskEventType,
)
from kortny.db.session import make_engine, make_session_factory, normalize_database_url
from kortny.llm import ChatMessage, Completion, TokenUsage, ToolCall
from kortny.tasks import TaskService
from kortny.tools import ToolRegistry, ToolResult
from kortny.tools.types import JsonObject, JsonSchema

TEST_POSTGRES_URL = os.environ.get("KORTNY_TEST_POSTGRES_URL")


def test_run_skill_script_is_not_outward() -> None:
    # Vetted skill scripts run in the same network-none sandbox as code_exec,
    # so the trifecta gate must treat them as local compute, not egress (HIG-248
    # follow-up): no per-call approval even after untrusted content armed the task.
    assert is_outward_or_write_tool("run_skill_script") is False
    assert is_outward_or_write_tool("code_exec") is False
    assert is_outward_or_write_tool("sandbox_bash") is False
    # Read-only skill tools are never escalated.
    assert is_outward_or_write_tool("load_skill") is False
    assert is_outward_or_write_tool("load_skill_resource") is False
    # Pin/reaction carry no new outbound payload — not egress.
    assert is_outward_or_write_tool("slack_add_reaction") is False
    assert is_outward_or_write_tool("slack_pin_message") is False
    # HIG-266: exporting the requested artifact delivers it to the originating
    # thread (the requester), not outward to a third party — so it is NOT egress
    # and must not be gated. (Pausing to "approve" showing a user their own
    # report is not coworker behavior.)
    assert is_outward_or_write_tool("sandbox_export_artifact") is False
    # The genuine egress legs stay outward.
    assert is_outward_or_write_tool("deploy_site") is True
    assert is_outward_or_write_tool("sandbox_publish_preview") is True
    assert is_outward_or_write_tool("slack_reply_thread") is True
    assert is_outward_or_write_tool("composio_notion_create_page") is True


class FakeLLM:
    def __init__(self, completions: Sequence[Completion]) -> None:
        self.completions = list(completions)

    def complete(
        self,
        *,
        task_id: uuid.UUID,
        messages: Sequence[ChatMessage],
        tools: Sequence[JsonSchema] = (),
        response_format: JsonObject | None = None,
        prompt_name: str | None = None,
        prompt_source: str = "code",
    ) -> Completion:
        del task_id, messages, tools, response_format, prompt_name, prompt_source
        if not self.completions:
            raise AssertionError("FakeLLM received more calls than expected")
        return self.completions.pop(0)


class NoopExecutionPlanner(ExecutionPlanner):
    def should_plan(self, *, task, tool_schemas, intent_decision):  # type: ignore[no-untyped-def]
        del task, tool_schemas, intent_decision
        return PlannerGateDecision(False, "test_no_plan")


class WebSearchTool:
    """Untrusted-origin tool: its result arms the trifecta gate."""

    name = "web_search"
    description = "Search the web."
    parameters: JsonSchema = {
        "type": "object",
        "properties": {"q": {"type": "string"}},
        "required": ["q"],
        "additionalProperties": False,
    }

    def invoke(self, args: JsonObject) -> ToolResult:
        del args
        return ToolResult(
            output={"results": [{"title": "x", "snippet": "y"}]},
            cost_usd=Decimal("0"),
        )


class OutwardWriteTool:
    """Outward write tool the gate escalates once the task is armed."""

    name = "composio_notion_create_page"
    description = "Create a page in an external Notion workspace."
    parameters: JsonSchema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
        "additionalProperties": False,
    }

    def __init__(self) -> None:
        self.calls: list[JsonObject] = []

    def invoke(self, args: JsonObject) -> ToolResult:
        self.calls.append(args)
        return ToolResult(output={"sent": True})


@pytest.fixture(scope="session")
def engine() -> Iterator[Engine]:
    if TEST_POSTGRES_URL is None:
        pytest.skip("KORTNY_TEST_POSTGRES_URL is required for trifecta tests")
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
        _cleanup(session)
        session.commit()
        yield session
        session.rollback()
        _cleanup(session)
        session.commit()


def _cleanup(session: Session) -> None:
    for model in (Artifact, LLMUsage, TaskEvent, Task, Installation):
        session.execute(delete(model))


def _task(session: Session, *, input_text: str = "do a thing") -> Task:
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


def _trifecta_events(session: Session, task: Task) -> list[TaskEvent]:
    return [
        event
        for event in session.scalars(
            select(TaskEvent).where(TaskEvent.task_id == task.id)
        )
        if event.type is TaskEventType.log
        and event.payload.get("message") == TRIFECTA_GATE_MESSAGE
    ]


def test_gate_escalates_outward_after_untrusted_result(db_session: Session) -> None:
    # web_search (untrusted) -> arms; then the outward write must gate even at
    # the balanced default that would otherwise auto-approve the external write.
    task = _task(db_session)
    search = WebSearchTool()
    outward = OutwardWriteTool()
    llm = FakeLLM(
        [
            Completion(
                content=None,
                tool_calls=(
                    ToolCall(id="c1", name="web_search", arguments={"q": "hi"}),
                ),
                usage=TokenUsage(input_tokens=10, output_tokens=2),
            ),
            Completion(
                content=None,
                tool_calls=(
                    ToolCall(
                        id="c2",
                        name="composio_notion_create_page",
                        arguments={"text": "leak"},
                    ),
                ),
                usage=TokenUsage(input_tokens=10, output_tokens=2),
            ),
        ]
    )

    with pytest.raises(ToolApprovalRequired) as exc_info:
        AgentCoordinator(
            session=db_session,
            llm=llm,
            registry=ToolRegistry([search, outward]),
            execution_planner=NoopExecutionPlanner(),
        ).run(task)

    assert exc_info.value.request.tool_name == "composio_notion_create_page"
    assert exc_info.value.request.risk == "trifecta_outward_after_untrusted"
    assert outward.calls == []
    trifecta = _trifecta_events(db_session, task)
    events_by_kind = {e.payload.get("event") for e in trifecta}
    assert "armed" in events_by_kind
    assert "escalated" in events_by_kind


def test_gate_does_not_fire_before_untrusted_content(db_session: Session) -> None:
    # The FIRST outward write runs free: nothing has armed the gate before it,
    # so it is not escalated even though it auto-approves under the ladder. (Its
    # own result then arms the gate for any SUBSEQUENT outward call — that is the
    # correct flow; we only assert no escalation occurred here.)
    task = _task(db_session)
    outward = OutwardWriteTool()
    llm = FakeLLM(
        [
            Completion(
                content=None,
                tool_calls=(
                    ToolCall(
                        id="c1",
                        name="composio_notion_create_page",
                        arguments={"text": "ok"},
                    ),
                ),
                usage=TokenUsage(input_tokens=10, output_tokens=2),
            ),
            Completion(
                content="Sent.",
                tool_calls=(),
                usage=TokenUsage(input_tokens=10, output_tokens=2),
            ),
        ]
    )

    result = AgentCoordinator(
        session=db_session,
        llm=llm,
        registry=ToolRegistry([outward]),
        execution_planner=NoopExecutionPlanner(),
    ).run(task)

    assert result.result_summary == "Sent."
    assert outward.calls == [{"text": "ok"}]
    # No escalation: the gate was disarmed when the write was approval-checked.
    escalations = [
        e
        for e in _trifecta_events(db_session, task)
        if e.payload.get("event") == "escalated"
    ]
    assert escalations == []


def test_gate_disabled_does_not_escalate(db_session: Session) -> None:
    task = _task(db_session)
    search = WebSearchTool()
    outward = OutwardWriteTool()
    llm = FakeLLM(
        [
            Completion(
                content=None,
                tool_calls=(
                    ToolCall(id="c1", name="web_search", arguments={"q": "hi"}),
                ),
                usage=TokenUsage(input_tokens=10, output_tokens=2),
            ),
            Completion(
                content=None,
                tool_calls=(
                    ToolCall(
                        id="c2",
                        name="composio_notion_create_page",
                        arguments={"text": "ok"},
                    ),
                ),
                usage=TokenUsage(input_tokens=10, output_tokens=2),
            ),
            Completion(
                content="Done.",
                tool_calls=(),
                usage=TokenUsage(input_tokens=10, output_tokens=2),
            ),
        ]
    )

    result = AgentCoordinator(
        session=db_session,
        llm=llm,
        registry=ToolRegistry([search, outward]),
        execution_planner=NoopExecutionPlanner(),
        trifecta_gate_enabled=False,
    ).run(task)

    assert result.result_summary == "Done."
    assert outward.calls == [{"text": "ok"}]
    assert not _trifecta_events(db_session, task)
