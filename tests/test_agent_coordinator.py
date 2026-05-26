import json
import os
import uuid
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from decimal import Decimal

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, delete, select, update
from sqlalchemy.orm import Session

from kortny.agent import (
    AgentCoordinator,
    AgentExecutionGuardrailError,
    AgentTurnLimitError,
    ContextAssembler,
    ExecutionErrorCategory,
    ExecutionGuardrailLimits,
    RecoveryAction,
)
from kortny.agent.thread_context import ThreadTranscriptMessage
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
from kortny.llm import ChatMessage, Completion, TokenUsage, ToolCall
from kortny.memory import EpisodeService
from kortny.tasks import TaskService
from kortny.tools import RecoverableToolError, ToolArtifact, ToolRegistry, ToolResult
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


class FakeThreadTranscriptProvider:
    def __init__(self, messages: Sequence[ThreadTranscriptMessage]) -> None:
        self.messages = tuple(messages)
        self.calls: list[tuple[str, str, int]] = []

    def fetch_thread_messages(
        self,
        *,
        channel_id: str,
        thread_ts: str,
        limit: int,
    ) -> tuple[ThreadTranscriptMessage, ...]:
        self.calls.append((channel_id, thread_ts, limit))
        return self.messages[:limit]


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


class MissingRequiredContextTool:
    name = "query_database"
    description = "Requires a database id."
    parameters: JsonSchema = {
        "type": "object",
        "properties": {"database_id": {"type": "string"}},
        "required": ["database_id"],
        "additionalProperties": False,
    }

    def invoke(self, args: JsonObject) -> ToolResult:
        raise RecoverableToolError(
            code="missing_required_arguments",
            message="query_database is missing required argument(s): database_id.",
            hint="Use search_database first or ask the user for the database link.",
            details={"missing_fields": ["database_id"]},
        )


class RecordingSearchTool:
    name = "web_search"
    description = "Records web search arguments."
    parameters: JsonSchema = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
        "additionalProperties": False,
    }

    def __init__(self) -> None:
        self.calls: list[JsonObject] = []

    def invoke(self, args: JsonObject) -> ToolResult:
        self.calls.append(args)
        return ToolResult(output={"results": []})


class RecoverableResultTool:
    name = "slack_file_read"
    description = "Returns a recoverable result error."
    parameters: JsonSchema = {
        "type": "object",
        "properties": {"file_id": {"type": "string"}},
        "required": ["file_id"],
        "additionalProperties": False,
    }

    def __init__(self, code: str = "invalid_file_id") -> None:
        self.code = code
        self.calls: list[JsonObject] = []

    def invoke(self, args: JsonObject) -> ToolResult:
        self.calls.append(args)
        return ToolResult(
            output={
                "file_id": args["file_id"],
                "error": {
                    "code": self.code,
                    "message": f"Recoverable {self.code}",
                    "recoverable": True,
                },
            }
        )


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


class RecordingPdfTool:
    name = "pdf_generator"
    description = "Records PDF arguments."
    parameters: JsonSchema = {
        "type": "object",
        "properties": {},
        "additionalProperties": True,
    }

    def __init__(self) -> None:
        self.calls: list[JsonObject] = []

    def invoke(self, args: JsonObject) -> ToolResult:
        self.calls.append(args)
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
        "episode_retrieval_completed",
        "context_assembled",
        "execution_plan_created",
        "execution_step_started",
        "agent_started",
        "agent_llm_turn_started",
        "agent_llm_turn_completed",
        "execution_step_completed",
        "agent_completed",
    ]
    assert events[-1].payload["reason"] == "final_answer"
    plan_event = next(
        event
        for event in events
        if event.payload.get("message") == "execution_plan_created"
    )
    assert plan_event.payload["mode"] == "inline"
    assert plan_event.payload["plan_version"] == 1
    assert plan_event.payload["plan"]["steps"][0]["step_id"] == "step-1"
    context_event = next(
        event for event in events if event.payload.get("message") == "context_assembled"
    )
    assert context_event.payload["selected_fact_ids"] == []
    assert context_event.payload["selected_episode_ids"] == []
    assert context_event.payload["context_budget"]["thread_context_max_chars"] == 12000


def test_coordinator_injects_workspace_facts(db_session: Session) -> None:
    installation = create_installation(db_session)
    create_workspace_fact(
        db_session,
        installation=installation,
        scope_type="workspace",
        scope_id=None,
        key="default_report_template",
        value_text="Use the Longboard report template with blue accents.",
    )
    task = create_task(
        db_session,
        installation=installation,
        input_text="what's our default report template?",
    )
    llm = FakeLLM(
        [
            Completion(
                content="Use the Longboard report template with blue accents.",
                tool_calls=(),
                usage=TokenUsage(input_tokens=30, output_tokens=10),
            )
        ]
    )

    AgentCoordinator(
        session=db_session,
        llm=llm,
        registry=ToolRegistry(),
    ).run(task)

    known_facts = known_facts_message(llm.calls[0][1])

    assert "<known_facts>" in known_facts
    assert "Workspace facts:" in known_facts
    assert (
        '- default_report_template = "Use the Longboard report template with blue accents."'
        in known_facts
    )


def test_coordinator_injects_channel_facts_only_for_current_channel(
    db_session: Session,
) -> None:
    installation = create_installation(db_session)
    create_workspace_fact(
        db_session,
        installation=installation,
        scope_type="channel",
        scope_id="C123",
        key="channel_report_style",
        value_text="Use concise market commentary in this channel.",
    )
    create_workspace_fact(
        db_session,
        installation=installation,
        scope_type="channel",
        scope_id="C999",
        key="other_channel_report_style",
        value_text="Use detailed engineering notes in the other channel.",
    )
    task = create_task(
        db_session,
        installation=installation,
        input_text="how should you format this channel's reports?",
    )
    llm = FakeLLM(
        [
            Completion(
                content="I'll use concise market commentary here.",
                tool_calls=(),
                usage=TokenUsage(input_tokens=30, output_tokens=10),
            )
        ]
    )

    AgentCoordinator(
        session=db_session,
        llm=llm,
        registry=ToolRegistry(),
    ).run(task)

    known_facts = known_facts_message(llm.calls[0][1])

    assert "Channel facts:" in known_facts
    assert "Use concise market commentary in this channel." in known_facts
    assert "Use detailed engineering notes in the other channel." not in known_facts


def test_coordinator_injects_user_facts_for_requesting_user(
    db_session: Session,
) -> None:
    installation = create_installation(db_session)
    create_workspace_fact(
        db_session,
        installation=installation,
        scope_type="user",
        scope_id="U123",
        key="pdf_preference",
        value_text="Skip PDFs unless explicitly requested.",
    )
    create_workspace_fact(
        db_session,
        installation=installation,
        scope_type="user",
        scope_id="U999",
        key="other_user_pdf_preference",
        value_text="Always include a PDF.",
    )
    task = create_task(
        db_session,
        installation=installation,
        input_text="research the latest Python tempfile practices",
    )
    llm = FakeLLM(
        [
            Completion(
                content="I'll answer inline unless you ask for a PDF.",
                tool_calls=(),
                usage=TokenUsage(input_tokens=30, output_tokens=10),
            )
        ]
    )

    AgentCoordinator(
        session=db_session,
        llm=llm,
        registry=ToolRegistry(),
    ).run(task)

    known_facts = known_facts_message(llm.calls[0][1])

    assert "User facts:" in known_facts
    assert "Skip PDFs unless explicitly requested." in known_facts
    assert "Always include a PDF." not in known_facts


def test_coordinator_known_facts_precedence_uses_user_over_lower_scopes(
    db_session: Session,
) -> None:
    installation = create_installation(db_session)
    create_workspace_fact(
        db_session,
        installation=installation,
        scope_type="workspace",
        scope_id=None,
        key="pdf_policy",
        value_text="Workspace PDF policy.",
    )
    create_workspace_fact(
        db_session,
        installation=installation,
        scope_type="channel",
        scope_id="C123",
        key="pdf_policy",
        value_text="Channel PDF policy.",
    )
    create_workspace_fact(
        db_session,
        installation=installation,
        scope_type="user",
        scope_id="U123",
        key="pdf_policy",
        value_text="User PDF policy.",
    )
    task = create_task(
        db_session,
        installation=installation,
        input_text="what is the PDF policy?",
    )
    llm = FakeLLM(
        [
            Completion(
                content="User PDF policy.",
                tool_calls=(),
                usage=TokenUsage(input_tokens=30, output_tokens=10),
            )
        ]
    )

    AgentCoordinator(
        session=db_session,
        llm=llm,
        registry=ToolRegistry(),
    ).run(task)

    known_facts = known_facts_message(llm.calls[0][1])

    assert '- pdf_policy = "User PDF policy."' in known_facts
    assert "Workspace PDF policy." not in known_facts
    assert "Channel PDF policy." not in known_facts


def test_coordinator_known_facts_budget_drops_oldest(db_session: Session) -> None:
    installation = create_installation(db_session)
    create_workspace_fact(
        db_session,
        installation=installation,
        scope_type="workspace",
        scope_id=None,
        key="old_fact",
        value_text="old " * 300,
        created_at=datetime(2026, 5, 20, 10, 0, tzinfo=UTC),
    )
    create_workspace_fact(
        db_session,
        installation=installation,
        scope_type="workspace",
        scope_id=None,
        key="new_fact",
        value_text="Keep this newer fact.",
        created_at=datetime(2026, 5, 21, 10, 0, tzinfo=UTC),
    )
    task = create_task(
        db_session,
        installation=installation,
        input_text="what do you know?",
    )
    llm = FakeLLM(
        [
            Completion(
                content="Keep this newer fact.",
                tool_calls=(),
                usage=TokenUsage(input_tokens=30, output_tokens=10),
            )
        ]
    )

    AgentCoordinator(
        session=db_session,
        llm=llm,
        registry=ToolRegistry(),
        known_facts_max_chars=360,
    ).run(task)

    known_facts = known_facts_message(llm.calls[0][1])

    assert "Keep this newer fact." in known_facts
    assert "old_fact" not in known_facts
    assert len(known_facts) <= 360


def test_context_assembler_builds_minimal_package(db_session: Session) -> None:
    task = create_task(db_session, input_text="summarize this")

    package = ContextAssembler(session=db_session).build_for_task(task)

    assert package.messages == (ChatMessage(role="user", content="summarize this"),)
    assert package.selected_facts == ()
    assert package.selected_prior_tasks == ()
    assert package.selected_artifacts == ()
    assert package.acknowledgement is None
    assert package.budget.known_facts_chars == 0
    assert package.budget.prior_context_chars == 0
    assert package.omissions == ()


def test_coordinator_injects_visible_acknowledgement_context(
    db_session: Session,
) -> None:
    task = create_task(
        db_session,
        input_text="what can you do for me?",
    )
    TaskService(db_session).append_event(
        task,
        TaskEventType.message_posted,
        {
            "channel": "C123",
            "thread_ts": "1716400000.000001",
            "message_ts": "1716400000.000002",
            "text": "I'll outline where I can help.",
            "purpose": "acknowledgement",
        },
    )
    llm = FakeLLM(
        [
            Completion(
                content="Here are the areas where I can help.",
                tool_calls=(),
                usage=TokenUsage(input_tokens=40, output_tokens=10),
            )
        ]
    )

    AgentCoordinator(
        session=db_session,
        llm=llm,
        registry=ToolRegistry(),
    ).run(task)

    acknowledgement_context = visible_acknowledgement_message(llm.calls[0][1])

    assert "Kortny already posted this visible Slack acknowledgement" in (
        acknowledgement_context
    )
    assert '"I\'ll outline where I can help."' in acknowledgement_context
    assert "natural continuation" in acknowledgement_context
    assert llm.calls[0][1][-1] == ChatMessage(
        role="user", content="what can you do for me?"
    )


def test_context_assembler_exposes_known_fact_metadata(
    db_session: Session,
) -> None:
    installation = create_installation(db_session)
    fact = create_workspace_fact(
        db_session,
        installation=installation,
        scope_type="workspace",
        scope_id=None,
        key="default_report_template",
        value_text="Use the Longboard report template with blue accents.",
    )
    task = create_task(
        db_session,
        installation=installation,
        input_text="what's our default report template?",
    )

    package = ContextAssembler(session=db_session).build_for_task(task)
    known_facts = known_facts_message(package.messages)

    assert "Use the Longboard report template with blue accents." in known_facts
    assert package.selected_facts[0].fact_id == fact.id
    assert package.selected_facts[0].scope_type == "workspace"
    assert package.selected_facts[0].scope_id is None
    assert package.selected_facts[0].key == "default_report_template"
    assert package.budget.known_facts_chars == len(known_facts)
    events = task_events(db_session, task)
    context_event = next(
        event for event in events if event.payload.get("message") == "context_assembled"
    )
    assert context_event.payload["selected_fact_ids"] == [str(fact.id)]
    assert context_event.payload["selected_fact_keys"] == ["default_report_template"]


def test_context_assembler_exposes_prior_task_and_artifact_metadata(
    db_session: Session,
) -> None:
    installation = create_installation(db_session)
    prior = create_task(
        db_session,
        input_text="Enhance the report",
        installation=installation,
        slack_event_id="EvPriorArtifactContext",
        created_at=datetime(2026, 5, 23, 13, 34, tzinfo=UTC),
    )
    prior.result_summary = "Generated 1 artifact."
    artifact = Artifact(
        task_id=prior.id,
        filename="pypl_report_v2.pdf",
        mime_type="application/pdf",
        size_bytes=4096,
        storage_path=None,
        slack_file_id="FGENV2",
    )
    db_session.add(artifact)
    db_session.flush()
    current = create_task(
        db_session,
        input_text="make it more elaborate",
        installation=installation,
        slack_event_id="EvCurrentArtifactContext",
        slack_message_ts="1716500040.000001",
        created_at=datetime(2026, 5, 23, 13, 35, tzinfo=UTC),
    )

    package = ContextAssembler(session=db_session).build_for_task(current)
    prior_context = prior_context_message(package.messages)

    assert "pypl_report_v2.pdf" in prior_context
    assert package.selected_prior_tasks[0].task_id == prior.id
    assert package.selected_prior_tasks[0].status == "pending"
    assert package.selected_artifacts[0].artifact_id == artifact.id
    assert package.selected_artifacts[0].task_id == prior.id
    assert package.selected_artifacts[0].filename == "pypl_report_v2.pdf"
    assert package.selected_artifacts[0].slack_file_id == "FGENV2"
    assert package.budget.prior_context_chars == len(prior_context)
    events = task_events(db_session, current)
    context_event = next(
        event for event in events if event.payload.get("message") == "context_assembled"
    )
    assert context_event.payload["selected_prior_task_ids"] == [str(prior.id)]
    assert context_event.payload["selected_artifact_ids"] == [str(artifact.id)]


def test_context_assembler_records_context_omissions(
    db_session: Session,
) -> None:
    installation = create_installation(db_session)
    first = create_task(
        db_session,
        input_text="research a very long topic",
        installation=installation,
        slack_event_id="EvLongPriorContext",
        created_at=datetime(2026, 5, 23, 14, 0, tzinfo=UTC),
    )
    first.result_summary = "Summary survives compaction."
    TaskService(db_session).append_event(
        first,
        TaskEventType.tool_result,
        {"output": {"large": "x" * 2_000}},
    )
    current = create_task(
        db_session,
        input_text="refine that",
        installation=installation,
        slack_event_id="EvLongCurrentContext",
        slack_message_ts="1716400500.000001",
        created_at=datetime(2026, 5, 23, 14, 1, tzinfo=UTC),
    )

    package = ContextAssembler(
        session=db_session,
        thread_context_max_chars=500,
    ).build_for_task(current)
    prior_context = prior_context_message(package.messages)

    assert len(prior_context) <= 500
    assert "Summary survives compaction." in prior_context
    assert ("prior_context", "compacted_to_budget", 1) in {
        (omission.kind, omission.reason, omission.count)
        for omission in package.omissions
    }
    assert ("prior_task_events", "compacted_context_omits_event_details", 1) in {
        (omission.kind, omission.reason, omission.count)
        for omission in package.omissions
    }


def test_coordinator_uses_known_fact_without_tool_call(
    db_session: Session,
) -> None:
    installation = create_installation(db_session)
    create_workspace_fact(
        db_session,
        installation=installation,
        scope_type="workspace",
        scope_id=None,
        key="default_report_template",
        value_text="Use the Longboard template with blue headings.",
    )
    task = create_task(
        db_session,
        installation=installation,
        input_text="what's our default report template?",
    )
    search_tool = RecordingSearchTool()
    llm = FakeLLM(
        [
            Completion(
                content="Use the Longboard template with blue headings.",
                tool_calls=(),
                usage=TokenUsage(input_tokens=30, output_tokens=10),
            )
        ]
    )

    AgentCoordinator(
        session=db_session,
        llm=llm,
        registry=ToolRegistry([search_tool]),
    ).run(task)

    known_facts = known_facts_message(llm.calls[0][1])
    events = task_events(db_session, task)

    assert "Use the Longboard template with blue headings." in known_facts
    assert search_tool.calls == []
    assert not [event for event in events if event.type is TaskEventType.tool_call]


def test_coordinator_includes_prior_thread_context_for_follow_up(
    db_session: Session,
) -> None:
    installation = create_installation(db_session)
    service = TaskService(db_session)
    first = create_task(
        db_session,
        input_text="research Python tempfile best practices and make a PDF",
        installation=installation,
        slack_event_id="EvResearchTurn",
        created_at=datetime(2026, 5, 23, 11, 0, tzinfo=UTC),
    )
    first.result_summary = "Generated a PDF about Python tempfile best practices."
    service.transition(first, TaskStatus.succeeded)
    service.append_event(
        first,
        TaskEventType.tool_call,
        {
            "turn": 1,
            "tool": "web_search",
            "arguments": {"query": "Python tempfile best practices"},
        },
    )
    service.append_event(
        first,
        TaskEventType.tool_result,
        {
            "turn": 1,
            "tool": "web_search",
            "output": {
                "results": [{"url": "https://docs.python.org/3/library/tempfile.html"}]
            },
        },
    )
    follow_up = create_task(
        db_session,
        input_text="make it punchier",
        installation=installation,
        slack_event_id="EvFollowUp",
        slack_message_ts="1716400100.000001",
        created_at=datetime(2026, 5, 23, 11, 1, tzinfo=UTC),
    )
    transcript_provider = FakeThreadTranscriptProvider(
        (
            ThreadTranscriptMessage(
                ts="1716400000.000001",
                user_id="U123",
                text="research Python tempfile best practices and make a PDF",
            ),
            ThreadTranscriptMessage(
                ts="1716400100.000001",
                user_id="U123",
                text="make it punchier",
            ),
        )
    )
    llm = FakeLLM(
        [
            Completion(
                content="Punchier version ready.",
                tool_calls=(),
                usage=TokenUsage(input_tokens=100, output_tokens=20),
            )
        ]
    )

    AgentCoordinator(
        session=db_session,
        llm=llm,
        registry=ToolRegistry(),
        thread_transcript_provider=transcript_provider,
    ).run(follow_up)

    first_call_messages = llm.calls[0][1]
    prior_context = first_call_messages[0].content or ""

    assert first_call_messages[0].role == "system"
    assert "<prior_context>" in prior_context
    assert "Generated a PDF about Python tempfile best practices." in prior_context
    assert "web_search" in prior_context
    assert "https://docs.python.org/3/library/tempfile.html" in prior_context
    assert "Slack thread transcript" in prior_context
    assert first_call_messages[-1] == ChatMessage(
        role="user", content="make it punchier"
    )
    assert transcript_provider.calls == [("C123", "1716400000.000001", 30)]


def test_context_assembler_includes_relevant_episode_context(
    db_session: Session,
) -> None:
    installation = create_installation(db_session)
    service = TaskService(db_session)
    prior = create_task(
        db_session,
        input_text="research PYPL market sentiment and make a report",
        installation=installation,
        slack_event_id="EvEpisodePrior",
        slack_thread_ts="1716500000.000001",
        slack_message_ts="1716500000.000001",
        created_at=datetime(2026, 5, 23, 15, 0, tzinfo=UTC),
    )
    prior.result_summary = "Generated a PYPL report with market sentiment."
    service.append_event(
        prior,
        TaskEventType.tool_call,
        {
            "turn": 1,
            "tool_call_id": "call-search",
            "tool": "web_search",
            "arguments": {"query": "PYPL market sentiment"},
        },
    )
    service.append_event(
        prior,
        TaskEventType.tool_result,
        {
            "turn": 1,
            "tool_call_id": "call-search",
            "tool": "web_search",
            "output": {
                "provider": "brave",
                "query": "PYPL market sentiment",
                "results": [
                    {
                        "title": "PayPal Holdings Market News",
                        "url": "https://example.com/pypl-market-news",
                        "snippet": "Analysts discussed PayPal sentiment.",
                    }
                ],
            },
            "cost_usd": "0",
            "artifacts": [],
        },
    )
    db_session.add(
        Artifact(
            task_id=prior.id,
            filename="pypl_report_v2.pdf",
            mime_type="application/pdf",
            size_bytes=8192,
            slack_file_id="FPYPLV2",
        )
    )
    service.transition(prior, TaskStatus.succeeded)
    episode = EpisodeService(db_session).record_task(prior)
    assert episode is not None

    current = create_task(
        db_session,
        input_text="what did you do for the PYPL report recently?",
        installation=installation,
        slack_event_id="EvEpisodeCurrent",
        slack_thread_ts="1716600000.000001",
        slack_message_ts="1716600000.000001",
        created_at=datetime(2026, 5, 23, 15, 5, tzinfo=UTC),
    )

    package = ContextAssembler(session=db_session).build_for_task(current)
    episode_context = episode_context_message(package.messages)

    assert "<recent_episodes>" in episode_context
    assert "relation=same_channel" in episode_context
    assert "Generated a PYPL report with market sentiment." in episode_context
    assert "pypl_report_v2.pdf" in episode_context
    assert "https://example.com/pypl-market-news" in episode_context
    assert package.selected_episodes[0].episode_id == episode.id
    assert package.selected_episodes[0].task_id == prior.id
    assert package.selected_episodes[0].relation == "same_channel"
    assert package.budget.episode_context_chars == len(episode_context)
    events = task_events(db_session, current)
    retrieval_event = next(
        event
        for event in events
        if event.payload.get("message") == "episode_retrieval_completed"
    )
    assert retrieval_event.payload["selected_episode_ids"] == [str(episode.id)]
    assert retrieval_event.payload["selected_episode_relations"] == ["same_channel"]
    context_event = next(
        event for event in events if event.payload.get("message") == "context_assembled"
    )
    assert context_event.payload["selected_episode_ids"] == [str(episode.id)]


def test_coordinator_orders_three_turn_thread_context(db_session: Session) -> None:
    installation = create_installation(db_session)
    first = create_task(
        db_session,
        input_text="research FastAPI deployment",
        installation=installation,
        slack_event_id="EvThreadOne",
        created_at=datetime(2026, 5, 23, 12, 0, tzinfo=UTC),
    )
    first.result_summary = "FastAPI can run behind a reverse proxy."
    second = create_task(
        db_session,
        input_text="turn that into a PDF",
        installation=installation,
        slack_event_id="EvThreadTwo",
        slack_message_ts="1716400200.000001",
        created_at=datetime(2026, 5, 23, 12, 1, tzinfo=UTC),
    )
    second.result_summary = "Generated 1 artifact."
    third = create_task(
        db_session,
        input_text="what was the key takeaway?",
        installation=installation,
        slack_event_id="EvThreadThree",
        slack_message_ts="1716400300.000001",
        created_at=datetime(2026, 5, 23, 12, 2, tzinfo=UTC),
    )
    llm = FakeLLM(
        [
            Completion(
                content="The key takeaway was deployment behind a reverse proxy.",
                tool_calls=(),
                usage=TokenUsage(input_tokens=80, output_tokens=15),
            )
        ]
    )

    AgentCoordinator(
        session=db_session,
        llm=llm,
        registry=ToolRegistry(),
    ).run(third)

    context = llm.calls[0][1][0].content or ""
    recent_context = context.split("Recent prior task details:", maxsplit=1)[1]

    assert recent_context.index("research FastAPI deployment") < recent_context.index(
        "turn that into a PDF"
    )
    assert "what was the key takeaway?" not in context


def test_coordinator_includes_failed_prior_task_context(db_session: Session) -> None:
    installation = create_installation(db_session)
    service = TaskService(db_session)
    failed = create_task(
        db_session,
        input_text="research unavailable source",
        installation=installation,
        slack_event_id="EvFailedPrior",
        created_at=datetime(2026, 5, 23, 13, 0, tzinfo=UTC),
    )
    failed.error = {"type": "ValueError", "message": "source unavailable"}
    service.append_event(
        failed,
        TaskEventType.error,
        {"type": "ValueError", "message": "source unavailable"},
    )
    service.transition(failed, TaskStatus.failed)
    current = create_task(
        db_session,
        input_text="try a different source",
        installation=installation,
        slack_event_id="EvAfterFailedPrior",
        slack_message_ts="1716400400.000001",
        created_at=datetime(2026, 5, 23, 13, 1, tzinfo=UTC),
    )
    llm = FakeLLM(
        [
            Completion(
                content="I will use a different source.",
                tool_calls=(),
                usage=TokenUsage(input_tokens=80, output_tokens=15),
            )
        ]
    )

    AgentCoordinator(
        session=db_session,
        llm=llm,
        registry=ToolRegistry(),
    ).run(current)

    context = llm.calls[0][1][0].content or ""

    assert "status=failed" in context
    assert "ValueError: source unavailable" in context
    assert "try a different source" not in context


def test_coordinator_preserves_prior_slack_file_ids_for_follow_up(
    db_session: Session,
) -> None:
    installation = create_installation(db_session)
    long_preview = "This document preview is long. " * 20
    first = create_task(
        db_session,
        input_text=(
            f"Give me a summary of this report\n{long_preview}\n\n"
            "<slack_files>\n"
            "- id: F123\n"
            "  name: pypl_report.pdf\n"
            "  mimetype: application/pdf\n"
            "  size_bytes: 2048\n"
            "</slack_files>"
        ),
        installation=installation,
        slack_event_id="EvDmReport",
        slack_channel_id="D123",
        slack_thread_ts="D123",
        slack_message_ts="1716500000.000001",
        created_at=datetime(2026, 5, 23, 13, 30, tzinfo=UTC),
    )
    first.result_summary = "The report summarizes PayPal Holdings."
    current = create_task(
        db_session,
        input_text="Can you extend this report with more context?",
        installation=installation,
        slack_event_id="EvDmReportFollowUp",
        slack_channel_id="D123",
        slack_thread_ts="D123",
        slack_message_ts="1716500010.000001",
        created_at=datetime(2026, 5, 23, 13, 31, tzinfo=UTC),
    )
    llm = FakeLLM(
        [
            Completion(
                content="I can reuse file F123.",
                tool_calls=(),
                usage=TokenUsage(input_tokens=80, output_tokens=15),
            )
        ]
    )

    AgentCoordinator(
        session=db_session,
        llm=llm,
        registry=ToolRegistry(),
    ).run(current)

    context = llm.calls[0][1][0].content or ""

    assert "slack_file_ids=F123" in context
    assert "attached Slack files from original request" in context
    assert "pypl_report.pdf" in context
    assert "Can you extend this report" not in context


def test_coordinator_highlights_immediate_previous_exchange_for_short_follow_up(
    db_session: Session,
) -> None:
    installation = create_installation(db_session)
    previous = create_task(
        db_session,
        input_text="Try now",
        installation=installation,
        slack_event_id="EvDmTryNow",
        slack_channel_id="D123",
        slack_thread_ts="D123",
        slack_message_ts="1716500020.000001",
        created_at=datetime(2026, 5, 23, 13, 32, tzinfo=UTC),
    )
    previous.result_summary = (
        "I've accessed the PDF file, which is a report on PayPal Holdings, Inc. "
        "(PYPL). Now, I'll extend this report to include more context and make "
        "it at least 3 pages long. Do you have any specific topics or additional "
        "details you want included, or should I proceed with general research?"
    )
    current = create_task(
        db_session,
        input_text="general research and market sentiment",
        installation=installation,
        slack_event_id="EvDmMarketSentiment",
        slack_channel_id="D123",
        slack_thread_ts="D123",
        slack_message_ts="1716500030.000001",
        created_at=datetime(2026, 5, 23, 13, 33, tzinfo=UTC),
    )
    transcript_provider = FakeThreadTranscriptProvider(())
    llm = FakeLLM(
        [
            Completion(
                content="I'll expand the PYPL report with general research and market sentiment.",
                tool_calls=(),
                usage=TokenUsage(input_tokens=100, output_tokens=18),
            )
        ]
    )

    AgentCoordinator(
        session=db_session,
        llm=llm,
        registry=ToolRegistry(),
        thread_transcript_provider=transcript_provider,
    ).run(current)

    context = llm.calls[0][1][0].content or ""

    assert "Immediate previous exchange:" in context
    assert "Do you have any specific topics" in context
    assert "should I proceed with general research?" in context
    assert "general research and market sentiment" not in context
    assert transcript_provider.calls == []


def test_coordinator_includes_prior_generated_artifacts_for_revision_follow_up(
    db_session: Session,
) -> None:
    installation = create_installation(db_session)
    prior = create_task(
        db_session,
        input_text="Enhance the report",
        installation=installation,
        slack_event_id="EvPriorArtifact",
        created_at=datetime(2026, 5, 23, 13, 34, tzinfo=UTC),
    )
    prior.result_summary = "Generated 1 artifact."
    db_session.add(
        Artifact(
            task_id=prior.id,
            filename="pypl_report_v2.pdf",
            mime_type="application/pdf",
            size_bytes=4096,
            storage_path=None,
            slack_file_id="FGENV2",
        )
    )
    current = create_task(
        db_session,
        input_text="make it more elaborate",
        installation=installation,
        slack_event_id="EvCurrentArtifactRevision",
        slack_message_ts="1716500040.000001",
        created_at=datetime(2026, 5, 23, 13, 35, tzinfo=UTC),
    )
    llm = FakeLLM(
        [
            Completion(
                content="I will revise the latest generated artifact.",
                tool_calls=(),
                usage=TokenUsage(input_tokens=100, output_tokens=18),
            )
        ]
    )

    AgentCoordinator(
        session=db_session,
        llm=llm,
        registry=ToolRegistry(),
    ).run(current)

    context = llm.calls[0][1][0].content or ""

    assert "generated artifacts:" in context
    assert "pypl_report_v2.pdf" in context
    assert "slack_file_id=FGENV2" in context
    assert "prefer the newest generated artifact" in context


def test_coordinator_compacts_prior_context_when_over_budget(
    db_session: Session,
) -> None:
    installation = create_installation(db_session)
    first = create_task(
        db_session,
        input_text="research a very long topic",
        installation=installation,
        slack_event_id="EvLongPrior",
        created_at=datetime(2026, 5, 23, 14, 0, tzinfo=UTC),
    )
    first.result_summary = "Summary survives compaction."
    TaskService(db_session).append_event(
        first,
        TaskEventType.tool_result,
        {"output": {"large": "x" * 2_000}},
    )
    current = create_task(
        db_session,
        input_text="refine that",
        installation=installation,
        slack_event_id="EvLongCurrent",
        slack_message_ts="1716400500.000001",
        created_at=datetime(2026, 5, 23, 14, 1, tzinfo=UTC),
    )
    llm = FakeLLM(
        [
            Completion(
                content="Refined.",
                tool_calls=(),
                usage=TokenUsage(input_tokens=40, output_tokens=8),
            )
        ]
    )

    AgentCoordinator(
        session=db_session,
        llm=llm,
        registry=ToolRegistry(),
        thread_context_max_chars=500,
    ).run(current)

    context = llm.calls[0][1][0].content or ""

    assert "Context was compacted" in context
    assert "Summary survives compaction." in context
    assert '"large"' not in context


def test_coordinator_injects_pdf_min_pages_from_user_request(
    db_session: Session,
) -> None:
    task = create_task(
        db_session,
        input_text="make it more elaborate. I want 3 pages of data",
    )
    pdf_tool = RecordingPdfTool()
    llm = FakeLLM(
        [
            Completion(
                content=None,
                tool_calls=(
                    ToolCall(
                        id="call-1",
                        name="pdf_generator",
                        arguments={
                            "title": "PYPL Report",
                            "sections": [{"heading": "Summary", "body": "Short."}],
                            "filename": "comprehensive_pypl_report.pdf",
                        },
                    ),
                ),
                usage=TokenUsage(input_tokens=20, output_tokens=3),
            ),
        ]
    )

    AgentCoordinator(
        session=db_session,
        llm=llm,
        registry=ToolRegistry([pdf_tool]),
    ).run(task)

    assert pdf_tool.calls == [
        {
            "title": "PYPL Report",
            "sections": [{"heading": "Summary", "body": "Short."}],
            "filename": "comprehensive_pypl_report.pdf",
            "min_pages": 3,
        }
    ]


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


def test_coordinator_feeds_recoverable_tool_errors_back_to_llm(
    db_session: Session,
) -> None:
    task = create_task(db_session, input_text="check notion for open items")
    llm = FakeLLM(
        [
            Completion(
                content=None,
                tool_calls=(
                    ToolCall(
                        id="call-1",
                        name="query_database",
                        arguments={"page_size": 10},
                    ),
                ),
                usage=TokenUsage(input_tokens=20, output_tokens=3),
            ),
            Completion(
                content=None,
                tool_calls=(
                    ToolCall(
                        id="call-2",
                        name="echo_json",
                        arguments={"message": "used fallback discovery"},
                    ),
                ),
                usage=TokenUsage(input_tokens=30, output_tokens=5),
            ),
            Completion(
                content="I recovered by using the fallback discovery path.",
                tool_calls=(),
                usage=TokenUsage(input_tokens=35, output_tokens=8),
            ),
        ]
    )

    result = AgentCoordinator(
        session=db_session,
        llm=llm,
        registry=ToolRegistry([MissingRequiredContextTool(), EchoJsonTool()]),
    ).run(task)

    assert result.result_summary == "I recovered by using the fallback discovery path."
    assert result.turns == 3

    second_turn_tool_message = llm.calls[1][1][-1]
    assert second_turn_tool_message.role == "tool"
    assert second_turn_tool_message.tool_call_id == "call-1"
    recoverable_payload = json.loads(second_turn_tool_message.content or "{}")
    assert recoverable_payload["output"]["successful"] is False
    assert recoverable_payload["output"]["error"] == {
        "code": "missing_required_arguments",
        "message": "query_database is missing required argument(s): database_id.",
        "recoverable": True,
        "hint": "Use search_database first or ask the user for the database link.",
        "details": {"missing_fields": ["database_id"]},
        "category": "schema_argument_validation",
        "recovery_action": "patch_arguments",
        "retryable": False,
        "user_action_required": False,
    }
    assert recoverable_payload["output"]["attempted_argument_keys"] == ["page_size"]

    events = task_events(db_session, task)
    assert [event.type for event in events if event.type in tool_event_types()] == [
        TaskEventType.tool_call,
        TaskEventType.tool_result,
        TaskEventType.tool_call,
        TaskEventType.tool_result,
    ]
    first_tool_result = next(
        event
        for event in events
        if event.type is TaskEventType.tool_result
        and event.payload["tool"] == "query_database"
    )
    assert first_tool_result.payload["recoverable"] is True
    assert first_tool_result.payload["output"]["error"]["code"] == (
        "missing_required_arguments"
    )
    assert first_tool_result.payload["error_category"] == (
        ExecutionErrorCategory.schema_argument_validation.value
    )
    assert first_tool_result.payload["recovery_action"] == (
        RecoveryAction.patch_arguments.value
    )
    recoverable_event = next(
        event
        for event in events
        if event.payload.get("message") == "execution_recoverable_failure_recorded"
    )
    assert recoverable_event.payload["error_category"] == (
        ExecutionErrorCategory.schema_argument_validation.value
    )
    assert recoverable_event.payload["recovery_action"] == (
        RecoveryAction.patch_arguments.value
    )
    assert not any(event.type is TaskEventType.error for event in events)


def test_coordinator_records_tool_attempt_metadata(db_session: Session) -> None:
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

    AgentCoordinator(
        session=db_session,
        llm=llm,
        registry=ToolRegistry([EchoJsonTool()]),
    ).run(task)

    events = task_events(db_session, task)
    budget_event = next(
        event
        for event in events
        if event.payload.get("message") == "execution_budget_updated"
    )
    tool_call_event = next(
        event for event in events if event.type is TaskEventType.tool_call
    )

    assert budget_event.payload["mode"] == "inline"
    assert budget_event.payload["current_step_id"] == "step-1"
    assert budget_event.payload["attempt"]["tool_name"] == "echo_json"
    assert budget_event.payload["attempt"]["attempt_no"] == 1
    assert len(budget_event.payload["attempt"]["normalized_args_hash"]) == 64
    assert tool_call_event.payload["step_id"] == "step-1"
    assert tool_call_event.payload["attempt_no"] == 1
    assert tool_call_event.payload["normalized_args_hash"] == (
        budget_event.payload["attempt"]["normalized_args_hash"]
    )


def test_coordinator_classifies_recoverable_tool_result_errors(
    db_session: Session,
) -> None:
    task = create_task(db_session, input_text="summarize the attached report")
    file_tool = RecoverableResultTool(code="invalid_file_id")
    llm = FakeLLM(
        [
            Completion(
                content=None,
                tool_calls=(
                    ToolCall(
                        id="call-1",
                        name="slack_file_read",
                        arguments={"file_id": "1716400000.123456"},
                    ),
                ),
                usage=TokenUsage(input_tokens=20, output_tokens=3),
            ),
            Completion(
                content="I need the actual Slack file ID or attachment.",
                tool_calls=(),
                usage=TokenUsage(input_tokens=25, output_tokens=6),
            ),
        ]
    )

    result = AgentCoordinator(
        session=db_session,
        llm=llm,
        registry=ToolRegistry([file_tool]),
    ).run(task)

    assert result.result_summary == "I need the actual Slack file ID or attachment."
    second_turn_tool_message = llm.calls[1][1][-1]
    tool_payload = json.loads(second_turn_tool_message.content or "{}")
    error = tool_payload["output"]["error"]
    assert error["category"] == ExecutionErrorCategory.reference_resolution.value
    assert error["recovery_action"] == RecoveryAction.resolve_reference.value
    assert error["retryable"] is False
    assert file_tool.calls == [{"file_id": "1716400000.123456"}]

    events = task_events(db_session, task)
    tool_result = next(
        event for event in events if event.type is TaskEventType.tool_result
    )
    recoverable_event = next(
        event
        for event in events
        if event.payload.get("message") == "execution_recoverable_failure_recorded"
    )
    assert tool_result.payload["error_category"] == (
        ExecutionErrorCategory.reference_resolution.value
    )
    assert tool_result.payload["recovery_action"] == (
        RecoveryAction.resolve_reference.value
    )
    assert recoverable_event.payload["error_code"] == "invalid_file_id"
    assert recoverable_event.payload["recoverable_failure_count"] == 1
    assert not any(event.type is TaskEventType.error for event in events)


def test_coordinator_trips_circuit_breaker_for_repeated_tool_call(
    db_session: Session,
) -> None:
    task = create_task(db_session, input_text="keep searching")
    search_tool = RecordingSearchTool()
    llm = FakeLLM(
        [
            Completion(
                content=None,
                tool_calls=(
                    ToolCall(
                        id="call-1",
                        name="web_search",
                        arguments={"query": "same"},
                    ),
                ),
                usage=TokenUsage(input_tokens=20, output_tokens=3),
            ),
            Completion(
                content=None,
                tool_calls=(
                    ToolCall(
                        id="call-2",
                        name="web_search",
                        arguments={"query": "same"},
                    ),
                ),
                usage=TokenUsage(input_tokens=25, output_tokens=3),
            ),
            Completion(
                content=None,
                tool_calls=(
                    ToolCall(
                        id="call-3",
                        name="web_search",
                        arguments={"query": "same"},
                    ),
                ),
                usage=TokenUsage(input_tokens=30, output_tokens=3),
            ),
        ]
    )

    with pytest.raises(AgentExecutionGuardrailError):
        AgentCoordinator(
            session=db_session,
            llm=llm,
            registry=ToolRegistry([search_tool]),
        ).run(task)

    assert search_tool.calls == [{"query": "same"}, {"query": "same"}]
    events = task_events(db_session, task)
    circuit_event = next(
        event
        for event in events
        if event.payload.get("message") == "execution_circuit_breaker_tripped"
    )
    assert circuit_event.payload["tool"] == "web_search"
    assert circuit_event.payload["attempt"]["attempt_no"] == 3
    assert circuit_event.payload["reason"] == "same_tool_call_repeated"
    assert events[-1].type is TaskEventType.error
    assert events[-1].payload["type"] == "AgentExecutionGuardrailError"


def test_coordinator_bounds_recoverable_tool_failures(db_session: Session) -> None:
    task = create_task(db_session, input_text="check notion")
    llm = FakeLLM(
        [
            Completion(
                content=None,
                tool_calls=(
                    ToolCall(
                        id="call-1",
                        name="query_database",
                        arguments={"page_size": 10},
                    ),
                ),
                usage=TokenUsage(input_tokens=20, output_tokens=3),
            ),
            Completion(
                content=None,
                tool_calls=(
                    ToolCall(
                        id="call-2",
                        name="query_database",
                        arguments={"page_size": 10},
                    ),
                ),
                usage=TokenUsage(input_tokens=25, output_tokens=3),
            ),
        ]
    )

    with pytest.raises(AgentExecutionGuardrailError):
        AgentCoordinator(
            session=db_session,
            llm=llm,
            registry=ToolRegistry([MissingRequiredContextTool()]),
            guardrail_limits=ExecutionGuardrailLimits(
                max_turns=6,
                max_tool_calls=12,
                max_recoverable_failures=1,
                max_same_tool_call=5,
                max_same_recoverable_error=5,
            ),
        ).run(task)

    events = task_events(db_session, task)
    recoverable_events = [
        event
        for event in events
        if event.payload.get("message") == "execution_recoverable_failure_recorded"
    ]
    assert len(recoverable_events) == 2
    budget_event = next(
        event
        for event in events
        if event.payload.get("message") == "execution_budget_exceeded"
    )
    assert budget_event.payload["reason"] == "max_recoverable_failures_exceeded"
    assert budget_event.payload["recoverable_failure_count"] == 2
    assert events[-1].type is TaskEventType.error


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
    session.execute(update(WorkspaceState).values(superseded_by_id=None))
    for model in (
        WorkspaceState,
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
    input_text: str,
    installation: Installation | None = None,
    slack_event_id: str | None = None,
    slack_channel_id: str = "C123",
    slack_thread_ts: str = "1716400000.000001",
    slack_message_ts: str = "1716400000.000001",
    slack_user_id: str = "U123",
    created_at: datetime | None = None,
) -> Task:
    installation = installation or create_installation(session)
    task = TaskService(session).create_task(
        installation_id=installation.id,
        slack_event_id=slack_event_id or f"Ev{uuid.uuid4().hex}",
        slack_channel_id=slack_channel_id,
        slack_thread_ts=slack_thread_ts,
        slack_message_ts=slack_message_ts,
        slack_user_id=slack_user_id,
        input=input_text,
    )
    if created_at is not None:
        task.created_at = created_at
        session.flush()
    return task


def create_workspace_fact(
    session: Session,
    *,
    installation: Installation,
    scope_type: str,
    scope_id: str | None,
    key: str,
    value_text: str,
    value_json: dict[str, object] | None = None,
    created_at: datetime | None = None,
) -> WorkspaceState:
    state = WorkspaceState(
        installation_id=installation.id,
        scope_type=scope_type,
        scope_id=scope_id,
        key=key,
        value_json=value_json or {"value": value_text},
        value_text=value_text,
        status="active",
        source_kind="user_explicit",
        proposed_by="U123",
        confirmed_by_user_id="U123",
        confirmed_at=created_at or datetime.now(UTC),
    )
    if created_at is not None:
        state.created_at = created_at
    session.add(state)
    session.flush()
    return state


def known_facts_message(messages: Sequence[ChatMessage]) -> str:
    for message in messages:
        if message.content and "<known_facts>" in message.content:
            return message.content
    raise AssertionError("No known_facts message found")


def prior_context_message(messages: Sequence[ChatMessage]) -> str:
    for message in messages:
        if message.content and "<prior_context>" in message.content:
            return message.content
    raise AssertionError("No prior_context message found")


def episode_context_message(messages: Sequence[ChatMessage]) -> str:
    for message in messages:
        if message.content and "<recent_episodes>" in message.content:
            return message.content
    raise AssertionError("No recent_episodes message found")


def visible_acknowledgement_message(messages: Sequence[ChatMessage]) -> str:
    for message in messages:
        if message.content and "<visible_acknowledgement>" in message.content:
            return message.content
    raise AssertionError("No visible_acknowledgement message found")


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
