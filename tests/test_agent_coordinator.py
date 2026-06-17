import json
import os
import uuid
from collections.abc import Iterator, Mapping, Sequence
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
    ContextBudget,
    ContextEngineInfo,
    ContextPackage,
    ExecutionErrorCategory,
    ExecutionGuardrailLimits,
    PlannerGateDecision,
    RecoveryAction,
    ToolApprovalRequired,
)
from kortny.agent import coordinator as coordinator_module
from kortny.agent.planner import ExecutionPlanner
from kortny.agent.thread_context import ThreadTranscriptMessage
from kortny.approvals import TOOL_APPROVAL_REQUIRED_MESSAGE
from kortny.db.models import (
    Artifact,
    AutonomyPolicy,
    EncryptedSecret,
    Episode,
    Installation,
    KnowledgeGraphEdge,
    KnowledgeGraphEntity,
    KnowledgeGraphEvidence,
    LLMUsage,
    ModelPricing,
    SlackChannelMembership,
    Task,
    TaskEvent,
    TaskEventType,
    TaskStatus,
    WorkspaceState,
)
from kortny.db.session import make_engine, make_session_factory, normalize_database_url
from kortny.knowledge_graph import EvidenceInput, GraphService, VisibilityScope
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
        response_format: JsonObject | None = None,
        prompt_name: str | None = None,
        prompt_source: str = "code",
    ) -> Completion:
        del response_format, prompt_name, prompt_source
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


class FakeContextEngine:
    def __init__(self, messages: Sequence[ChatMessage]) -> None:
        self._messages = tuple(messages)
        self.info = ContextEngineInfo(
            id="test.fake_context_engine",
            name="Fake Context Engine",
        )
        self.ingested_task_ids: list[uuid.UUID] = []
        self.assembled_task_ids: list[uuid.UUID] = []
        self.after_turn_calls: list[tuple[uuid.UUID, str]] = []

    def ingest(self, task: Task) -> None:
        self.ingested_task_ids.append(task.id)

    def assemble(self, task: Task) -> ContextPackage:
        self.assembled_task_ids.append(task.id)
        return ContextPackage(
            messages=self._messages,
            selected_facts=(),
            selected_prior_tasks=(),
            selected_episodes=(),
            selected_artifacts=(),
            selected_graph_entities=(),
            selected_graph_edges=(),
            acknowledgement=None,
            budget=ContextBudget(
                system_prompt_chars=0,
                known_facts_max_chars=0,
                known_facts_chars=0,
                thread_context_max_chars=1,
                prior_context_chars=0,
                thread_context_recent_tasks=1,
                thread_transcript_limit=0,
                episode_context_max_chars=0,
                episode_context_chars=0,
                episode_context_limit=0,
                graph_context_max_chars=0,
                graph_context_chars=0,
                graph_context_max_items=0,
                graph_context_max_hops=0,
            ),
            omissions=(),
            context_engine_id=self.info.id,
            context_engine_name=self.info.name,
        )

    def compact(self, task: Task, *, force: bool = False) -> ContextPackage | None:
        del task, force
        return None

    def after_turn(
        self,
        task: Task,
        package: ContextPackage,
        *,
        outcome: str,
    ) -> None:
        del package
        self.after_turn_calls.append((task.id, outcome))


class NoopExecutionPlanner(ExecutionPlanner):
    def should_plan(
        self,
        *,
        task: Task,
        tool_schemas: Sequence[JsonSchema],
        intent_decision: Mapping[str, object] | None,
    ) -> PlannerGateDecision:
        del task, tool_schemas, intent_decision
        return PlannerGateDecision(False, "test_no_plan")


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


class InspectMemoryDifferentTool:
    name = "inspect_memory"
    description = "Returns active memories that do not match the requested one."
    parameters: JsonSchema = {
        "type": "object",
        "properties": {"scope": {"type": "string"}},
        "required": ["scope"],
        "additionalProperties": False,
    }

    def __init__(self) -> None:
        self.calls: list[JsonObject] = []

    def invoke(self, args: JsonObject) -> ToolResult:
        self.calls.append(args)
        return ToolResult(
            output={
                "scope": args["scope"],
                "count": 1,
                "facts": [
                    {
                        "key": "pdf_generation_policy",
                        "value_text": "Do not generate PDFs unless explicitly asked.",
                        "status": "active",
                    }
                ],
            }
        )


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


class VerboseSearchTool:
    name = "web_search"
    description = "Returns many long web search results."
    parameters: JsonSchema = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
        "additionalProperties": False,
    }

    def invoke(self, args: JsonObject) -> ToolResult:
        return ToolResult(
            output={
                "provider": "brave",
                "query": args["query"],
                "results": [
                    {
                        "title": f"Result {index}",
                        "url": f"https://example.com/{index}",
                        "snippet": "x" * 1000,
                    }
                    for index in range(12)
                ],
                "raw_payload": "y" * 20000,
            }
        )


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


class RecordingNotionComposioTool:
    name = "composio_notion_execute"
    description = "Executes selected Notion tools through Composio."
    parameters: JsonSchema = {
        "type": "object",
        "properties": {
            "tool_slug": {"type": "string"},
            "arguments": {"type": "object"},
        },
        "required": ["tool_slug"],
        "additionalProperties": False,
    }

    def invoke(self, args: JsonObject) -> ToolResult:
        return ToolResult(output={"received": args})


class DangerousExternalTool:
    name = "composio_linear_create_issue"
    description = "Creates a Linear issue."
    parameters: JsonSchema = {
        "type": "object",
        "properties": {"title": {"type": "string"}},
        "required": ["title"],
        "additionalProperties": False,
    }

    def __init__(self) -> None:
        self.calls: list[JsonObject] = []

    def invoke(self, args: JsonObject) -> ToolResult:
        self.calls.append(args)
        return ToolResult(output={"created": True})


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
        "kg_retrieval_started",
        "kg_retrieval_completed",
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
    assert context_event.payload["context_engine_id"] == "kortny.default_context_engine"
    assert context_event.payload["context_engine_name"] == "Default Context Engine"
    assert context_event.payload["context_budget"]["thread_context_max_chars"] == 12000


def _empty_completion(response_id: str) -> Completion:
    return Completion(
        content="",
        tool_calls=(),
        usage=TokenUsage(input_tokens=8, output_tokens=0),
        response_id=response_id,
        model="openrouter/google/gemini-2.5-flash-lite",
    )


def test_coordinator_retries_empty_completion_then_succeeds(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    # HIG-270: a transient empty completion (cheap-model blip) must be retried at
    # the call layer and not consume an agent turn or fail the task.
    monkeypatch.setattr("kortny.agent.coordinator.EMPTY_COMPLETION_BACKOFF_SECONDS", 0)
    task = create_task(db_session, input_text="summarize this")
    llm = FakeLLM(
        [
            _empty_completion("gen-empty-1"),
            Completion(
                content="Here is the summary.",
                tool_calls=(),
                usage=TokenUsage(input_tokens=10, output_tokens=5),
                response_id="gen-final",
                model="openai/gpt-4o-mini",
            ),
        ]
    )

    result = AgentCoordinator(
        session=db_session,
        llm=llm,
        registry=ToolRegistry(),
    ).run(task)

    assert result.result_summary == "Here is the summary."
    assert result.turns == 1  # the empty retry did NOT consume a turn
    assert len(llm.calls) == 2  # one empty + one good, both at the call layer
    events = task_events(db_session, task)
    assert "agent_empty_completion_retry" in event_messages(events)


def test_coordinator_degrades_gracefully_on_persistent_empty(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    # HIG-270: persistently empty even after retries -> graceful fallback message,
    # NOT a hard crash surfaced to the user as "Something went wrong".
    monkeypatch.setattr("kortny.agent.coordinator.EMPTY_COMPLETION_BACKOFF_SECONDS", 0)
    task = create_task(db_session, input_text="is notion connected?")
    llm = FakeLLM([_empty_completion(f"gen-empty-{i}") for i in range(3)])

    result = AgentCoordinator(
        session=db_session,
        llm=llm,
        registry=ToolRegistry(),
        max_turns=1,
    ).run(task)

    assert result.result_summary == coordinator_module.EMPTY_FINAL_FALLBACK_MESSAGE
    assert len(llm.calls) == 3  # EMPTY_COMPLETION_RETRIES + 1 attempts
    events = task_events(db_session, task)
    assert "agent_empty_final_fallback" in event_messages(events)


def test_coordinator_reports_assistant_status(db_session: Session) -> None:
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
    statuses: list[str] = []
    phases: list[str | None] = []

    class Recorder:
        def report(self, status: str, *, phase: str | None = None) -> None:
            statuses.append(status)
            phases.append(phase)

    AgentCoordinator(
        session=db_session,
        llm=llm,
        registry=ToolRegistry(),
        status_reporter=Recorder(),
    ).run(task)

    assert "Getting up to speed…" in statuses
    assert "Writing the response…" in statuses
    # Coarse macro-phases accompany the granular steps so the two assistant-pane
    # status lines complement rather than mirror each other.
    assert "is getting started…" in phases
    assert "is writing the reply…" in phases


def test_coordinator_tool_schemas_identical_across_turns(
    db_session: Session,
) -> None:
    """HIG-196: tool schemas must be byte-stable across turns of one run.

    Prompt caching keys on the rendered tools/system prefix; a reordered or
    rebuilt schema between turns silently busts the within-task cache.
    """

    task = create_task(db_session, input_text="echo something")
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
                usage=TokenUsage(input_tokens=10, output_tokens=5),
                response_id="gen-1",
                model="openai/gpt-4o-mini",
            ),
            Completion(
                content="Done.",
                tool_calls=(),
                usage=TokenUsage(input_tokens=12, output_tokens=4),
                response_id="gen-2",
                model="openai/gpt-4o-mini",
            ),
        ]
    )

    AgentCoordinator(
        session=db_session,
        llm=llm,
        registry=ToolRegistry([EchoJsonTool()]),
    ).run(task)

    assert len(llm.calls) == 2
    first_turn_tools = llm.calls[0][2]
    second_turn_tools = llm.calls[1][2]
    assert first_turn_tools == second_turn_tools
    # Stable JSON serialization across turns (the actual cache-key invariant).
    assert json.dumps(list(first_turn_tools), sort_keys=True) == json.dumps(
        list(second_turn_tools), sort_keys=True
    )


def test_coordinator_recovers_from_unregistered_tool_call(
    db_session: Session,
) -> None:
    """A tool_call naming a tool not registered for this task must be a
    recoverable error fed back to the model, not an uncaught crash.

    Regression: the model hallucinated `web_search` (suppressed by per-task
    selection), `registry.get` raised an uncaught ToolNotFoundError, and the
    whole task crashed. Now it should recover into a follow-up turn.
    """

    task = create_task(db_session, input_text="research nvidia earnings")
    llm = FakeLLM(
        [
            Completion(
                content=None,
                tool_calls=(
                    ToolCall(
                        id="call-unknown",
                        name="web_search",  # NOT registered below
                        arguments={"query": "nvidia earnings"},
                    ),
                ),
                usage=TokenUsage(input_tokens=10, output_tokens=5),
                response_id="gen-1",
                model="openai/gpt-4o-mini",
            ),
            Completion(
                content="Here is the earnings summary.",
                tool_calls=(),
                usage=TokenUsage(input_tokens=12, output_tokens=4),
                response_id="gen-2",
                model="openai/gpt-4o-mini",
            ),
        ]
    )

    # Only echo_json is registered; web_search is absent on purpose.
    AgentCoordinator(
        session=db_session,
        llm=llm,
        registry=ToolRegistry([EchoJsonTool()]),
    ).run(task)

    # Recovered into a second turn instead of crashing on turn 1.
    assert len(llm.calls) == 2
    # The model received a tool result explaining the tool is unavailable.
    second_turn_messages = llm.calls[1][1]
    tool_messages = [
        message
        for message in second_turn_messages
        if message.role == "tool" and message.tool_call_id == "call-unknown"
    ]
    assert tool_messages, "expected a tool result for the unregistered tool call"
    assert "not available" in (tool_messages[0].content or "").lower()


def test_coordinator_soft_caps_repeated_web_search(db_session: Session) -> None:
    """web_search called with a fresh query each turn must be soft-capped.

    Regression (HIG-267): a one-page report task burned 9 of 16 turns on 14
    distinct web searches and never reached the deliverable. The same-call
    circuit breaker missed it (each query hashes differently), so a per-tool
    NAME cap now nudges the model off the tool after the ceiling and the
    over-cap call is NOT actually invoked.
    """

    task = create_task(db_session, input_text="research nvidia earnings deeply")
    search = RecordingSearchTool()
    # Five distinct-query web_search calls, then a final answer. The 5th call is
    # over the cap of 4 and must be intercepted before the tool runs.
    completions = [
        Completion(
            content=None,
            tool_calls=(
                ToolCall(
                    id=f"call-{i}",
                    name="web_search",
                    arguments={"query": f"nvidia earnings angle {i}"},
                ),
            ),
            usage=TokenUsage(input_tokens=10, output_tokens=3),
        )
        for i in range(5)
    ]
    completions.append(
        Completion(
            content="Here is the earnings summary from what I gathered.",
            tool_calls=(),
            usage=TokenUsage(input_tokens=12, output_tokens=6),
        )
    )
    llm = FakeLLM(completions)

    AgentCoordinator(
        session=db_session,
        llm=llm,
        registry=ToolRegistry([search]),
        execution_planner=NoopExecutionPlanner(),
        guardrail_limits=ExecutionGuardrailLimits(max_turns=10),
    ).run(task)

    # Only the first four searches actually ran; the fifth was nudged, not run.
    assert len(search.calls) == 4
    # The over-cap call's tool result steered the model to stop and deliver.
    final_turn_messages = llm.calls[-1][1]
    nudge = next(
        message
        for message in final_turn_messages
        if message.role == "tool" and message.tool_call_id == "call-4"
    )
    nudge_text = (nudge.content or "").lower()
    assert "web_search" in nudge_text
    assert "do not call it again" in nudge_text or "enough" in nudge_text


def test_coordinator_compacts_large_search_tool_result_for_next_turn(
    db_session: Session,
) -> None:
    task = create_task(db_session, input_text="compare observability tools")
    llm = FakeLLM(
        [
            Completion(
                content=None,
                tool_calls=(
                    ToolCall(
                        id="call-search",
                        name="web_search",
                        arguments={"query": "AI observability tools"},
                    ),
                ),
                usage=TokenUsage(input_tokens=20, output_tokens=5),
            ),
            Completion(
                content="Langfuse and Phoenix are the two to compare.",
                tool_calls=(),
                usage=TokenUsage(input_tokens=40, output_tokens=8),
            ),
        ]
    )

    AgentCoordinator(
        session=db_session,
        llm=llm,
        registry=ToolRegistry([VerboseSearchTool()]),
        execution_planner=NoopExecutionPlanner(),
        tool_result_prompt_max_chars=4000,
    ).run(task)

    tool_message = llm.calls[1][1][-1]
    prompt_payload = json.loads(tool_message.content or "{}")
    prompt_output = prompt_payload["output"]
    assert prompt_output["compacted"] is True
    assert prompt_output["compaction_kind"] == "search_results"
    assert 1 <= len(prompt_output["results"]) <= 8
    assert prompt_output["omitted_result_count"] == 12 - len(prompt_output["results"])
    assert "raw_payload" not in prompt_output
    assert all(len(result["snippet"]) <= 263 for result in prompt_output["results"])

    events = task_events(db_session, task)
    raw_tool_result = next(
        event for event in events if event.type is TaskEventType.tool_result
    )
    assert raw_tool_result.payload["output"]["raw_payload"] == "y" * 20000
    compacted_event = next(
        event
        for event in events
        if event.payload.get("message") == "tool_result_compacted"
    )
    assert (
        compacted_event.payload["raw_chars"] > compacted_event.payload["prompt_chars"]
    )
    assert compacted_event.payload["reason"] == "search_result_compaction"


def test_execution_planner_skips_single_hop_read_integration(
    db_session: Session,
) -> None:
    task = create_task(db_session, input_text="search current AI observability tools")

    decision = {
        "classification": "task_request",
        "likely_tools": ["composio_firecrawl_search"],
        "needs_channel_context": False,
        "needs_thread_context": False,
        "needs_file_context": False,
        "model_tier": "standard",
    }
    gate = ExecutionPlanner().should_plan(
        task=task,
        tool_schemas=(
            {
                "name": "composio_firecrawl_search",
                "description": "Search the web.",
                "parameters": {"type": "object"},
            },
        ),
        intent_decision=decision,
    )

    assert gate.should_plan is False
    assert gate.reason == "single_hop_read_tool"


_DEPTH_SCHEMAS: tuple[JsonSchema, ...] = (
    {
        "name": "composio_firecrawl_search",
        "description": "Search the web.",
        "parameters": {"type": "object"},
    },
)


def test_execution_planner_plans_for_deep_workflow_depth(
    db_session: Session,
) -> None:
    task = create_task(db_session, input_text="research and write a brief")
    gate = ExecutionPlanner().should_plan(
        task=task,
        tool_schemas=_DEPTH_SCHEMAS,
        intent_decision={
            "likely_tools": ["composio_firecrawl_search"],
            "model_tier": "standard",
            "response_depth": "deep_workflow",
        },
    )

    assert gate.should_plan is True
    assert gate.reason == "unified_depth_deep_workflow"


def test_execution_planner_skips_planning_for_standard_and_quick_depth(
    db_session: Session,
) -> None:
    task = create_task(db_session, input_text="look up the AAPL price")
    for depth in ("standard_tool_task", "quick_response"):
        gate = ExecutionPlanner().should_plan(
            task=task,
            tool_schemas=_DEPTH_SCHEMAS,
            intent_decision={
                "likely_tools": ["composio_firecrawl_search"],
                "model_tier": "standard",
                "response_depth": depth,
            },
        )
        assert gate.should_plan is False
        assert gate.reason == f"unified_depth_{depth}"


def test_execution_planner_falls_back_to_legacy_without_intent(
    db_session: Session,
) -> None:
    task = create_task(db_session, input_text="how is the team doing today")
    gate = ExecutionPlanner().should_plan(
        task=task,
        tool_schemas=_DEPTH_SCHEMAS,
        intent_decision=None,
    )

    # Legacy heuristic path: no intent signal and no toolkit name match.
    assert gate.should_plan is False
    assert gate.reason == "no_intent_signal"


def test_guardrail_limits_for_depth_scales_by_depth() -> None:
    quick = ExecutionGuardrailLimits.for_depth("quick_response")
    assert quick.max_turns == 2
    assert quick.max_tool_calls == 3

    # Multi-step research + document tasks need more than the old 6-turn default
    # (HIG-250): standard gets a wider budget, deep wider still.
    standard = ExecutionGuardrailLimits.for_depth("standard_tool_task")
    assert standard.max_turns == 10
    assert standard.max_tool_calls == 20

    deep = ExecutionGuardrailLimits.for_depth("deep_workflow")
    assert deep.max_turns == 16
    assert deep.max_tool_calls == 40
    # HIG-267: deep workflows get extra recoverable-failure headroom now that a
    # per-tool soft cap bounds runaway research independently.
    assert deep.max_recoverable_failures == 8

    # An unrecognized depth falls back to the standard (not quick) budget.
    assert ExecutionGuardrailLimits.for_depth("???") == standard


def test_guardrail_limits_carry_default_web_search_soft_cap() -> None:
    # HIG-267: web_search is capped per task NAME (regardless of query), so a
    # fresh-query-each-turn research loop can't monopolize the turn budget.
    for depth in ("quick_response", "standard_tool_task", "deep_workflow"):
        limits = ExecutionGuardrailLimits.for_depth(depth)
        assert limits.soft_cap_for("web_search") == 4
    # Build/export tools carry no cap (they legitimately repeat).
    assert (
        ExecutionGuardrailLimits.for_depth("deep_workflow").soft_cap_for(
            "sandbox_export_artifact"
        )
        is None
    )
    # HIG-269 follow-up: external runtime tools (Composio/MCP, find_tools-loaded)
    # are capped by name prefix so a search/fetch tool can't loop the budget away
    # (observed: notion_search called 8+ times -> max_tool_calls crash).
    standard_limits = ExecutionGuardrailLimits.for_depth("standard_tool_task")
    assert standard_limits.soft_cap_for("composio_notion_search_notion_page") == 6
    assert standard_limits.soft_cap_for("mcp__context7__query_docs") == 6
    # A native tool without an explicit cap is still uncapped.
    assert standard_limits.soft_cap_for("slack_channel_history") is None


def test_execution_budget_counts_tool_calls_by_name() -> None:
    # The per-name counter (drives the soft cap) increments on every call to a
    # tool regardless of arguments, unlike attempt_no which is per (name, args).
    import uuid as _uuid

    from kortny.agent.execution import ExecutionBudgetState

    budget = ExecutionBudgetState()
    task_id = _uuid.uuid4()
    first = budget.record_tool_attempt(
        task_id=task_id, step_id="s1", tool_name="web_search", arguments={"q": "a"}
    )
    second = budget.record_tool_attempt(
        task_id=task_id, step_id="s1", tool_name="web_search", arguments={"q": "b"}
    )
    # Distinct args -> attempt_no stays 1 (signature differs) but the name
    # counter climbs, which is exactly what the same-call breaker misses.
    assert first.attempt_no == 1
    assert second.attempt_no == 1
    assert first.tool_name_attempt_no == 1
    assert second.tool_name_attempt_no == 2


def test_coordinator_gates_sensitive_tool_before_invocation(
    db_session: Session,
) -> None:
    task = create_task(db_session, input_text="create an issue")
    # HIG-223: an external create auto-approves at the balanced default, so pin
    # this workspace to conservative to exercise the explicit approval gate.
    _set_conservative_autonomy(db_session, task)
    tool = DangerousExternalTool()
    llm = FakeLLM(
        [
            Completion(
                content=None,
                tool_calls=(
                    ToolCall(
                        id="call-create",
                        name="composio_linear_create_issue",
                        arguments={"title": "Follow up"},
                    ),
                ),
                usage=TokenUsage(input_tokens=30, output_tokens=5),
            )
        ]
    )

    with pytest.raises(ToolApprovalRequired) as exc_info:
        AgentCoordinator(
            session=db_session,
            llm=llm,
            registry=ToolRegistry([tool]),
            execution_planner=NoopExecutionPlanner(),
        ).run(task)

    assert tool.calls == []
    assert exc_info.value.request.tool_name == "composio_linear_create_issue"
    assert exc_info.value.request.argument_keys == ("title",)

    events = task_events(db_session, task)
    required = next(
        event
        for event in events
        if event.payload.get("message") == TOOL_APPROVAL_REQUIRED_MESSAGE
    )
    assert required.payload["request"]["tool"] == "composio_linear_create_issue"
    assert required.payload["request"]["scope"] == "user"
    assert required.payload["request"]["argument_keys"] == ["title"]
    assert not any(event.type is TaskEventType.tool_call for event in events)


def test_coordinator_uses_prior_approval_for_same_tool_signature(
    db_session: Session,
) -> None:
    task = create_task(db_session, input_text="create an issue")
    _set_conservative_autonomy(db_session, task)
    tool = DangerousExternalTool()
    tool_call = ToolCall(
        id="call-create",
        name="composio_linear_create_issue",
        arguments={"title": "Follow up"},
    )

    with pytest.raises(ToolApprovalRequired) as exc_info:
        AgentCoordinator(
            session=db_session,
            llm=FakeLLM(
                [
                    Completion(
                        content=None,
                        tool_calls=(tool_call,),
                        usage=TokenUsage(input_tokens=30, output_tokens=5),
                    )
                ]
            ),
            registry=ToolRegistry([tool]),
            execution_planner=NoopExecutionPlanner(),
        ).run(task)
    TaskService(db_session).append_event(
        task,
        TaskEventType.log,
        {
            "message": "tool_approval_decision",
            "decision": "approved",
            "approval_key": exc_info.value.request.approval_key,
            "tool": exc_info.value.request.tool_name,
            "by_user_id": "U123",
        },
    )

    result = AgentCoordinator(
        session=db_session,
        llm=FakeLLM(
            [
                Completion(
                    content=None,
                    tool_calls=(tool_call,),
                    usage=TokenUsage(input_tokens=30, output_tokens=5),
                ),
                Completion(
                    content="Created it.",
                    tool_calls=(),
                    usage=TokenUsage(input_tokens=40, output_tokens=8),
                ),
            ]
        ),
        registry=ToolRegistry([tool]),
        execution_planner=NoopExecutionPlanner(),
    ).run(task)

    assert result.result_summary == "Created it."
    assert tool.calls == [{"title": "Follow up"}]


def test_coordinator_retries_empty_response_after_tool_result(
    db_session: Session,
) -> None:
    task = create_task(db_session, input_text="inspect this and answer")
    llm = FakeLLM(
        [
            Completion(
                content=None,
                tool_calls=(
                    ToolCall(
                        id="call-echo",
                        name="echo_json",
                        arguments={"message": "memory"},
                    ),
                ),
                usage=TokenUsage(input_tokens=30, output_tokens=5),
            ),
            Completion(
                content="",
                tool_calls=(),
                usage=TokenUsage(input_tokens=20, output_tokens=0),
            ),
            Completion(
                content="I found the memory topic.",
                tool_calls=(),
                usage=TokenUsage(input_tokens=25, output_tokens=7),
            ),
        ]
    )

    result = AgentCoordinator(
        session=db_session,
        llm=llm,
        registry=ToolRegistry([EchoJsonTool()]),
        execution_planner=NoopExecutionPlanner(),
    ).run(task)

    # HIG-270: the empty completion after the tool result is now absorbed at the
    # CALL layer (agent_empty_completion_retry) — it is retried with the same
    # messages without consuming an agent turn, so the loop-level repair prompt
    # is not injected for a single transient blip. The final answer still lands.
    assert result.result_summary == "I found the memory topic."
    assert len(llm.calls) == 3
    assert not any(
        message.role == "system"
        and "previous response was empty" in (message.content or "")
        for message in llm.calls[2][1]
    )
    assert any(
        event.payload.get("message") == "agent_empty_completion_retry"
        for event in task_events(db_session, task)
    )


def test_coordinator_loop_repair_after_call_retries_exhausted(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    # HIG-270 defense-in-depth: when call-level retries are exhausted (3 empties),
    # the loop-level repair prompt still fires and a later turn recovers.
    monkeypatch.setattr("kortny.agent.coordinator.EMPTY_COMPLETION_BACKOFF_SECONDS", 0)
    task = create_task(db_session, input_text="answer this")
    llm = FakeLLM(
        [
            _empty_completion("gen-empty-1"),
            _empty_completion("gen-empty-2"),
            _empty_completion("gen-empty-3"),
            Completion(
                content="Recovered answer.",
                tool_calls=(),
                usage=TokenUsage(input_tokens=12, output_tokens=4),
                response_id="gen-good",
                model="openai/gpt-4o-mini",
            ),
        ]
    )

    result = AgentCoordinator(
        session=db_session,
        llm=llm,
        registry=ToolRegistry(),
    ).run(task)

    assert result.result_summary == "Recovered answer."
    assert len(llm.calls) == 4  # 3 exhausted call-retries + 1 recovered turn
    assert any(
        message.role == "system"
        and "previous response was empty" in (message.content or "")
        for message in llm.calls[3][1]
    )
    messages = event_messages(task_events(db_session, task))
    assert "agent_empty_completion_retry" in messages
    assert "agent_empty_response_retry" in messages


def test_coordinator_exposes_runtime_loaded_tool_next_turn(
    db_session: Session,
) -> None:
    # HIG-269 core mechanism: a tool registered mid-loop (as find_tools does)
    # must become visible in the schemas AND callable on the next turn.
    task = create_task(db_session, input_text="do the thing")
    registry = ToolRegistry()

    class LateTool:
        name = "late_tool"
        description = "added at runtime"
        parameters: JsonSchema = {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }
        invoked = False

        def invoke(self, args: JsonObject) -> ToolResult:
            LateTool.invoked = True
            return ToolResult(output={"done": True})

    class LoaderTool:
        name = "loader"
        description = "loads late_tool at runtime"
        parameters: JsonSchema = {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }

        def __init__(self, reg: ToolRegistry) -> None:
            self._reg = reg

        def invoke(self, args: JsonObject) -> ToolResult:
            self._reg.register_if_absent(LateTool())
            return ToolResult(output={"loaded": "late_tool"})

    registry.register(LoaderTool(registry))

    llm = FakeLLM(
        [
            Completion(
                content=None,
                tool_calls=(ToolCall(id="c1", name="loader", arguments={}),),
                usage=TokenUsage(input_tokens=10, output_tokens=2),
            ),
            Completion(
                content=None,
                tool_calls=(ToolCall(id="c2", name="late_tool", arguments={}),),
                usage=TokenUsage(input_tokens=10, output_tokens=2),
            ),
            Completion(
                content="All done.",
                tool_calls=(),
                usage=TokenUsage(input_tokens=10, output_tokens=2),
            ),
        ]
    )

    result = AgentCoordinator(
        session=db_session,
        llm=llm,
        registry=registry,
        execution_planner=NoopExecutionPlanner(),
    ).run(task)

    assert LateTool.invoked is True
    assert result.result_summary == "All done."
    # Turn 2's schemas (llm.calls[1][2]) must include the runtime-loaded tool.
    turn2_tool_names = {schema["name"] for schema in llm.calls[1][2]}
    assert "late_tool" in turn2_tool_names
    # Turn 1 did not have it yet.
    turn1_tool_names = {schema["name"] for schema in llm.calls[0][2]}
    assert "late_tool" not in turn1_tool_names


def test_coordinator_humanizes_memory_no_match_final_text(
    db_session: Session,
) -> None:
    task = create_task(db_session, input_text="forget my PDF branding preference")
    inspect_memory = InspectMemoryDifferentTool()
    llm = FakeLLM(
        [
            Completion(
                content=None,
                tool_calls=(
                    ToolCall(
                        id="call-inspect-memory",
                        name="inspect_memory",
                        arguments={"scope": "user"},
                    ),
                ),
                usage=TokenUsage(input_tokens=30, output_tokens=5),
            ),
            Completion(
                content="No active memory fact matched that scope and key.",
                tool_calls=(),
                usage=TokenUsage(input_tokens=40, output_tokens=8),
            ),
        ]
    )

    result = AgentCoordinator(
        session=db_session,
        llm=llm,
        registry=ToolRegistry([inspect_memory]),
        execution_planner=NoopExecutionPlanner(),
    ).run(task)

    assert inspect_memory.calls == [{"scope": "user"}]
    assert result.result_summary == (
        "I checked what I remember and don't see anything matching "
        '"PDF branding preference" saved right now, so there is nothing for me '
        "to remove."
    )
    assert "No active memory fact matched" not in result.result_summary


def test_coordinator_uses_context_engine_lifecycle(db_session: Session) -> None:
    task = create_task(db_session, input_text="this should be replaced")
    context_engine = FakeContextEngine(
        (ChatMessage(role="user", content="from context engine"),)
    )
    llm = FakeLLM(
        [
            Completion(
                content="Handled via context engine.",
                tool_calls=(),
                usage=TokenUsage(input_tokens=10, output_tokens=5),
                response_id="gen-context-engine",
                model="openai/gpt-4o-mini",
            )
        ]
    )

    result = AgentCoordinator(
        session=db_session,
        llm=llm,
        registry=ToolRegistry(),
        context_engine=context_engine,
    ).run(task)

    assert result.result_summary == "Handled via context engine."
    assert context_engine.ingested_task_ids == [task.id]
    assert context_engine.assembled_task_ids == [task.id]
    assert context_engine.after_turn_calls == [(task.id, "succeeded")]
    assert llm.calls[0][1] == (ChatMessage(role="user", content="from context engine"),)


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


def test_context_assembler_injects_scope_safe_graph_context(
    db_session: Session,
) -> None:
    installation = create_installation(db_session)
    task = create_task(
        db_session,
        installation=installation,
        input_text="what do you know about this channel?",
        slack_channel_id="CGraph",
        slack_event_id="EvGraphContext",
    )
    db_session.add(
        SlackChannelMembership(
            installation_id=installation.id,
            channel_id="CGraph",
            channel_name="graph-test",
            channel_type="public_channel",
            membership_status="active",
            discovered_via="member_joined_channel",
            onboarding_status="posted",
        )
    )
    graph = GraphService(db_session)
    channel = graph.create_entity(
        installation_id=installation.id,
        entity_type="channel",
        canonical_key="slack_channel:CGraph",
        display_name="#graph-test",
        visibility_scope=VisibilityScope.channel("CGraph"),
        source_type="slack_authoritative",
        lifecycle_state="active",
        evidence=EvidenceInput(
            source_type="slack_authoritative",
            extracted_by="test",
            source_task_id=task.id,
            source_slack_channel_id="CGraph",
            raw_snippet="Channel membership recorded.",
        ),
    )
    graph.create_entity(
        installation_id=installation.id,
        entity_type="firm_fact",
        canonical_key="channel_profile:CGraph",
        display_name="Candidate profile",
        visibility_scope=VisibilityScope.channel("CGraph"),
        source_type="onboarding_scan",
        lifecycle_state="candidate",
        evidence=EvidenceInput(
            source_type="onboarding_scan",
            extracted_by="test",
            source_task_id=task.id,
            source_slack_channel_id="CGraph",
            raw_snippet="Candidate summary should stay out of current context.",
        ),
    )
    db_session.flush()

    package = ContextAssembler(session=db_session).build_for_task(task)
    graph_context = graph_context_message(package.messages)

    assert "<workspace_graph_context>" in graph_context
    assert "slack_channel:CGraph" in graph_context
    assert "channel_profile:CGraph" not in graph_context
    assert package.selected_graph_entities[0].entity_id == channel.id
    assert package.selected_graph_entities[0].canonical_key == "slack_channel:CGraph"
    assert package.selected_graph_edges == ()
    assert package.budget.graph_context_chars == len(graph_context)

    events = task_events(db_session, task)
    assert "kg_retrieval_started" in event_messages(events)
    retrieval_event = next(
        event
        for event in events
        if event.payload.get("message") == "kg_retrieval_completed"
    )
    assert retrieval_event.payload["entity_count"] == 1
    assert retrieval_event.payload["edge_count"] == 0
    assert retrieval_event.payload["anchor_keys"] == [
        "slack_channel:CGraph",
        "slack_user:U123",
    ]
    context_event = next(
        event for event in events if event.payload.get("message") == "context_assembled"
    )
    assert context_event.payload["selected_graph_entity_ids"] == [str(channel.id)]
    assert context_event.payload["selected_graph_entity_keys"] == [
        "slack_channel:CGraph"
    ]


def test_context_assembler_does_not_leak_private_graph_context_to_public_channel(
    db_session: Session,
) -> None:
    installation = create_installation(db_session)
    task = create_task(
        db_session,
        installation=installation,
        input_text="what do you know here?",
        slack_channel_id="CGraphPublic",
        slack_event_id="EvGraphNoLeak",
    )
    db_session.add(
        SlackChannelMembership(
            installation_id=installation.id,
            channel_id="CGraphPublic",
            channel_name="public-graph",
            channel_type="public_channel",
            membership_status="active",
            discovered_via="member_joined_channel",
            onboarding_status="posted",
        )
    )
    graph = GraphService(db_session)
    public_channel = graph.create_entity(
        installation_id=installation.id,
        entity_type="channel",
        canonical_key="slack_channel:CGraphPublic",
        display_name="#public-graph",
        visibility_scope=VisibilityScope.channel("CGraphPublic"),
        source_type="slack_authoritative",
        lifecycle_state="active",
        evidence=EvidenceInput(
            source_type="slack_authoritative",
            extracted_by="test",
            source_task_id=task.id,
            source_slack_channel_id="CGraphPublic",
            raw_snippet="Public channel membership recorded.",
        ),
    )
    private_project = graph.create_entity(
        installation_id=installation.id,
        entity_type="project",
        canonical_key="project:private-alpha",
        display_name="Private Alpha",
        visibility_scope=VisibilityScope.private_channel("GSecret"),
        source_type="onboarding_scan",
        lifecycle_state="active",
        evidence=EvidenceInput(
            source_type="onboarding_scan",
            extracted_by="test",
            source_task_id=task.id,
            source_slack_channel_id="GSecret",
            raw_snippet="Private project should not leak.",
        ),
    )
    graph.create_edge(
        installation_id=installation.id,
        source_entity_id=public_channel.id,
        target_entity_id=private_project.id,
        relationship_type="relates_to",
        visibility_scope=VisibilityScope.private_channel("GSecret"),
        source_type="onboarding_scan",
        lifecycle_state="active",
        evidence=EvidenceInput(
            source_type="onboarding_scan",
            extracted_by="test",
            source_task_id=task.id,
            source_slack_channel_id="GSecret",
            raw_snippet="Private edge should not leak.",
        ),
    )
    db_session.flush()

    package = ContextAssembler(session=db_session).build_for_task(task)
    graph_context = graph_context_message(package.messages)

    assert "slack_channel:CGraphPublic" in graph_context
    assert "project:private-alpha" not in graph_context
    assert "Private Alpha" not in graph_context
    assert package.selected_graph_entities[0].entity_id == public_channel.id
    assert package.selected_graph_edges == ()

    events = task_events(db_session, task)
    retrieval_event = next(
        event
        for event in events
        if event.payload.get("message") == "kg_retrieval_completed"
    )
    assert retrieval_event.payload["entity_count"] == 1
    assert retrieval_event.payload["edge_count"] == 0
    assert not [
        event
        for event in events
        if event.payload.get("message") == "kg_scope_guard_failed"
    ]


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
    assert (
        tool_call_event.payload["normalized_args_hash"]
        == (budget_event.payload["attempt"]["normalized_args_hash"])
    )


def test_coordinator_keeps_simple_tasks_on_inline_plan(db_session: Session) -> None:
    task = create_task(db_session, input_text="answer this directly")
    llm = FakeLLM(
        [
            Completion(
                content="Direct answer.",
                tool_calls=(),
                usage=TokenUsage(input_tokens=20, output_tokens=4),
            )
        ]
    )

    AgentCoordinator(
        session=db_session,
        llm=llm,
        registry=ToolRegistry([EchoJsonTool()]),
    ).run(task)

    assert len(llm.calls) == 1
    assert llm.calls[0][1] == (
        ChatMessage(role="user", content="answer this directly"),
    )
    events = task_events(db_session, task)
    plan_event = next(
        event
        for event in events
        if event.payload.get("message") == "execution_plan_created"
    )
    assert plan_event.payload["mode"] == "inline"
    assert plan_event.payload["plan"]["planner_source"] == "inline_default"
    assert plan_event.payload["plan"]["planner_reason"] == "no_intent_signal"


def test_coordinator_uses_private_planner_for_complex_tool_tasks(
    db_session: Session,
) -> None:
    task = create_task(
        db_session,
        input_text="Check Notion for open action items and summarize the risky ones.",
    )
    record_intent_decision(
        db_session,
        task,
        likely_tools=["composio_notion_execute", "web_search"],
        needs_channel_context=False,
        model_tier="strong",
    )
    llm = FakeLLM(
        [
            Completion(
                content=json.dumps(
                    {
                        "objective": (
                            "Find open Notion action items and summarize risk."
                        ),
                        "steps": [
                            {
                                "description": (
                                    "Discover relevant Notion task databases or pages."
                                ),
                                "selected_tool_names": ["composio_notion_execute"],
                            },
                            {
                                "description": "Summarize open items and call out risk.",
                                "selected_tool_names": [],
                            },
                        ],
                        "missing_inputs": [],
                        "fallback_notes": [
                            "If a database id is missing, use Notion discovery before asking."
                        ],
                        "risk_notes": ["Do not mutate Notion content."],
                    }
                ),
                tool_calls=(),
                usage=TokenUsage(input_tokens=80, output_tokens=40),
            ),
            Completion(
                content="I found the risky open items.",
                tool_calls=(),
                usage=TokenUsage(input_tokens=120, output_tokens=8),
            ),
        ]
    )

    result = AgentCoordinator(
        session=db_session,
        llm=llm,
        registry=ToolRegistry([RecordingNotionComposioTool(), RecordingSearchTool()]),
    ).run(task)

    assert result.result_summary == "I found the risky open items."
    assert len(llm.calls) == 2
    actor_messages = llm.calls[1][1]
    plan_message = next(
        message
        for message in actor_messages
        if message.content and "<private_execution_plan>" in message.content
    )
    assert plan_message.content is not None
    assert "Find open Notion action items" in plan_message.content
    assert "composio_notion_execute" in plan_message.content
    assert actor_messages[-1] == ChatMessage(role="user", content=task.input)

    events = task_events(db_session, task)
    plan_event = next(
        event
        for event in events
        if event.payload.get("message") == "execution_plan_created"
    )
    assert plan_event.payload["mode"] == "planned"
    assert plan_event.payload["plan"]["planner_source"] == "llm_planner"
    assert plan_event.payload["plan"]["planner_reason"] == "intent_likely_multi_tool"
    assert len(plan_event.payload["plan"]["steps"]) == 2
    assert plan_event.payload["plan"]["fallback_notes"] == [
        "If a database id is missing, use Notion discovery before asking."
    ]


def test_coordinator_falls_back_to_inline_plan_when_planner_fails(
    db_session: Session,
) -> None:
    task = create_task(
        db_session,
        input_text="Use Notion and web search to compare open action items.",
    )
    record_intent_decision(
        db_session,
        task,
        likely_tools=["composio_notion_execute", "web_search"],
        needs_channel_context=True,
        model_tier="strong",
    )
    llm = FakeLLM(
        [
            Completion(
                content="not json",
                tool_calls=(),
                usage=TokenUsage(input_tokens=80, output_tokens=4),
            ),
            Completion(
                content="I can still answer from the inline path.",
                tool_calls=(),
                usage=TokenUsage(input_tokens=100, output_tokens=8),
            ),
        ]
    )

    result = AgentCoordinator(
        session=db_session,
        llm=llm,
        registry=ToolRegistry([RecordingNotionComposioTool(), RecordingSearchTool()]),
    ).run(task)

    assert result.result_summary == "I can still answer from the inline path."
    assert len(llm.calls) == 2
    assert not any(
        message.content and "<private_execution_plan>" in message.content
        for message in llm.calls[1][1]
    )

    events = task_events(db_session, task)
    planner_failure = next(
        event
        for event in events
        if event.payload.get("message") == "execution_planner_failed"
    )
    assert planner_failure.payload["reason"] == "intent_likely_multi_tool"
    plan_event = next(
        event
        for event in events
        if event.payload.get("message") == "execution_plan_created"
    )
    assert plan_event.payload["mode"] == "inline"
    assert plan_event.payload["plan"]["planner_source"] == "planner_fallback"
    assert plan_event.payload["plan"]["planner_reason"] == "intent_likely_multi_tool"


def test_coordinator_replans_after_recoverable_failure_in_planned_mode(
    db_session: Session,
) -> None:
    task = create_task(
        db_session,
        input_text="Check the task database and use search if the database is missing.",
    )
    record_intent_decision(
        db_session,
        task,
        likely_tools=["query_database", "web_search"],
        model_tier="strong",
    )
    search_tool = RecordingSearchTool()
    llm = FakeLLM(
        [
            Completion(
                content=json.dumps(
                    {
                        "objective": "Find task database items with a fallback.",
                        "steps": [
                            {
                                "description": "Try the database query first.",
                                "selected_tool_names": ["query_database"],
                            },
                            {
                                "description": "Use search if the database id is missing.",
                                "selected_tool_names": ["web_search"],
                            },
                        ],
                        "missing_inputs": [],
                        "fallback_notes": [
                            "Use search before asking the user for the database id."
                        ],
                        "risk_notes": [],
                    }
                ),
                tool_calls=(),
                usage=TokenUsage(input_tokens=80, output_tokens=40),
            ),
            Completion(
                content=None,
                tool_calls=(
                    ToolCall(
                        id="call-1",
                        name="query_database",
                        arguments={"page_size": 10},
                    ),
                ),
                usage=TokenUsage(input_tokens=100, output_tokens=4),
            ),
            Completion(
                content=json.dumps(
                    {
                        "recovery_goal": (
                            "Recover by discovering context before asking the user."
                        ),
                        "next_action": "use_discovery_tool",
                        "suggested_tool_names": ["web_search"],
                        "argument_notes": [
                            "Do not retry query_database without database_id."
                        ],
                        "fallback_notes": [
                            "Ask for a database link only if search cannot help."
                        ],
                        "risk_notes": [],
                    }
                ),
                tool_calls=(),
                usage=TokenUsage(input_tokens=90, output_tokens=30),
            ),
            Completion(
                content=None,
                tool_calls=(
                    ToolCall(
                        id="call-2",
                        name="web_search",
                        arguments={"query": "task database actionable items"},
                    ),
                ),
                usage=TokenUsage(input_tokens=130, output_tokens=5),
            ),
            Completion(
                content="I recovered by switching to search.",
                tool_calls=(),
                usage=TokenUsage(input_tokens=140, output_tokens=8),
            ),
        ]
    )

    result = AgentCoordinator(
        session=db_session,
        llm=llm,
        registry=ToolRegistry([MissingRequiredContextTool(), search_tool]),
    ).run(task)

    assert result.result_summary == "I recovered by switching to search."
    assert result.turns == 3
    assert search_tool.calls == [{"query": "task database actionable items"}]

    second_actor_messages = llm.calls[3][1]
    recovery_message = next(
        message
        for message in second_actor_messages
        if message.content and "<private_recovery_plan>" in message.content
    )
    assert recovery_message.content is not None
    assert "failed_tool: query_database" in recovery_message.content
    assert "next_action: use_discovery_tool" in recovery_message.content
    assert "suggested_tools: web_search" in recovery_message.content

    events = task_events(db_session, task)
    recovery_event = next(
        event
        for event in events
        if event.payload.get("message") == "execution_recovery_plan_created"
    )
    assert recovery_event.payload["plan_version"] == 2
    assert recovery_event.payload["recovery_plan"]["planner_source"] == (
        "llm_recovery_planner"
    )
    assert recovery_event.payload["recovery_plan"]["next_action"] == (
        "use_discovery_tool"
    )
    assert recovery_event.payload["recovery_plan"]["suggested_tool_names"] == [
        "web_search"
    ]


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
        KnowledgeGraphEvidence,
        KnowledgeGraphEdge,
        KnowledgeGraphEntity,
        SlackChannelMembership,
        WorkspaceState,
        Episode,
        Artifact,
        LLMUsage,
        TaskEvent,
        Task,
        ModelPricing,
        EncryptedSecret,
        AutonomyPolicy,
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


def _set_conservative_autonomy(session: Session, task: Task) -> None:
    session.add(
        AutonomyPolicy(
            installation_id=task.installation_id,
            scope_type="workspace",
            scope_id=None,
            level="conservative",
        )
    )
    session.flush()


def record_intent_decision(
    session: Session,
    task: Task,
    *,
    likely_tools: list[str],
    needs_channel_context: bool = False,
    needs_thread_context: bool = False,
    needs_file_context: bool = False,
    model_tier: str = "standard",
) -> None:
    TaskService(session).append_event(
        task,
        TaskEventType.log,
        {
            "message": "intent_classification_completed",
            "source": "test",
            "decision": {
                "addressed_to_kortny": True,
                "classification": "task_request",
                "confidence": 0.95,
                "should_create_task": True,
                "should_ack_with_reaction": True,
                "suggested_reaction": "eyes",
                "needs_channel_context": needs_channel_context,
                "needs_thread_context": needs_thread_context,
                "needs_file_context": needs_file_context,
                "likely_tools": likely_tools,
                "model_tier": model_tier,
                "reason": "test intent",
            },
        },
    )


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


def graph_context_message(messages: Sequence[ChatMessage]) -> str:
    for message in messages:
        if message.content and "<workspace_graph_context>" in message.content:
            return message.content
    raise AssertionError("No workspace_graph_context message found")


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
