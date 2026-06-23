"""Tests for honest failure recovery on terminal exhaustion (HIG-288 B).

Coverage:
- _guardrail_failure_reason classifies exception messages correctly
- _summarize_tool_history builds a compact event summary
- Circuit breaker trip -> honest message logged in task events
- Max recoverable failures -> honest message logged in task events
- LLM synthesis success -> custom message used (not generic fallback)
- LLM synthesis error -> deterministic fallback used
- Suppressed post (background assessment) -> no Slack post attempted
"""

import os
import uuid
from collections.abc import Iterator, Sequence

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, delete, select
from sqlalchemy.orm import Session

from kortny.agent import (
    AgentCoordinator,
    AgentExecutionGuardrailError,
    ExecutionGuardrailLimits,
)
from kortny.agent import coordinator as coordinator_module
from kortny.config.settings import LLMProvider as SettingsLLMProvider
from kortny.config.settings import Settings
from kortny.db.models import (
    Artifact,
    EncryptedSecret,
    Episode,
    Installation,
    KnowledgeGraphEdge,
    KnowledgeGraphEntity,
    KnowledgeGraphEvidence,
    LLMProvider,
    LLMUsage,
    ModelPricing,
    SlackChannelMembership,
    Task,
    TaskEvent,
    TaskEventType,
    WorkspaceState,
)
from kortny.db.session import make_engine, make_session_factory, normalize_database_url
from kortny.llm import ChatMessage, Completion, TokenUsage, ToolCall
from kortny.tasks import TaskService
from kortny.tools import RecoverableToolError, ToolRegistry, ToolResult
from kortny.tools.types import JsonObject, JsonSchema
from kortny.worker.agent_executor import (
    GENERIC_FAILURE_TEXT,
    AgentTaskExecutor,
    _guardrail_failure_reason,
    _summarize_tool_history,
)

TEST_POSTGRES_URL = os.environ.get("KORTNY_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    TEST_POSTGRES_URL is None,
    reason="KORTNY_TEST_POSTGRES_URL is required",
)


# ---------------------------------------------------------------------------
# FakeLLM for coordinator tests (same pattern as test_agent_coordinator.py)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Tool that raises RecoverableToolError on every call
# ---------------------------------------------------------------------------


class AlwaysFailTool:
    name = "query_database"
    description = "Always fails with a recoverable error."
    parameters: JsonSchema = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
        "additionalProperties": False,
    }

    def invoke(self, args: JsonObject) -> ToolResult:
        raise RecoverableToolError(
            code="missing_required_arguments",
            message="query_database is missing required argument(s): query.",
            hint="Provide a query string.",
        )


class AlwaysRepeatSearchTool:
    name = "web_search"
    description = "Records web search calls."
    parameters: JsonSchema = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
        "additionalProperties": False,
    }

    def invoke(self, args: JsonObject) -> ToolResult:
        return ToolResult(output={"results": []})


# ---------------------------------------------------------------------------
# DB fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def engine() -> Iterator[Engine]:
    assert TEST_POSTGRES_URL is not None
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", normalize_database_url(TEST_POSTGRES_URL))
    command.upgrade(config, "head")
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
    from sqlalchemy import update

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
        Installation,
    ):
        session.execute(delete(model))


def _make_installation(session: Session) -> Installation:
    inst = Installation(slack_team_id=f"T{uuid.uuid4().hex}")
    session.add(inst)
    session.flush()
    return inst


def _create_task(session: Session, *, input_text: str = "search something") -> Task:
    inst = _make_installation(session)
    return TaskService(session).create_task(
        installation_id=inst.id,
        slack_event_id=f"Ev{uuid.uuid4().hex}",
        slack_channel_id="C123",
        slack_thread_ts="1716400000.000001",
        slack_message_ts=f"{uuid.uuid4().hex}",
        slack_user_id="U123",
        input=input_text,
    )


def _task_log_messages(session: Session, task: Task) -> list[str]:
    events = session.scalars(
        select(TaskEvent).where(TaskEvent.task_id == task.id).order_by(TaskEvent.seq)
    ).all()
    return [
        event.payload["message"]
        for event in events
        if event.type is TaskEventType.log and isinstance(event.payload, dict)
    ]


# ---------------------------------------------------------------------------
# Unit tests for module-level helpers
# ---------------------------------------------------------------------------


def test_guardrail_failure_reason_circuit_breaker() -> None:
    exc = AgentExecutionGuardrailError(
        "Execution circuit breaker tripped for repeated tool call web_search"
    )
    assert _guardrail_failure_reason(exc) == "same_tool_call_repeated"


def test_guardrail_failure_reason_recoverable_budget() -> None:
    exc = AgentExecutionGuardrailError(
        "Recoverable tool failure budget exceeded for query_database:missing_required_arguments"
    )
    assert _guardrail_failure_reason(exc) == "recoverable_failure_budget_exceeded"


def test_guardrail_failure_reason_unknown() -> None:
    exc = AgentExecutionGuardrailError("Something totally unexpected happened")
    assert _guardrail_failure_reason(exc) == "unknown"


def test_summarize_tool_history_empty(db_session: Session) -> None:
    task = _create_task(db_session)
    summary = _summarize_tool_history(db_session, task)
    assert summary == "(no tool calls recorded)"


def test_summarize_tool_history_records_tool_calls(db_session: Session) -> None:
    task = _create_task(db_session)
    task_service = TaskService(db_session)
    task_service.append_event(
        task,
        TaskEventType.tool_call,
        {"tool": "web_search", "tool_call_id": "tc-1"},
    )
    task_service.append_event(
        task,
        TaskEventType.error,
        {"error": "Something broke", "type": "SomeError"},
    )
    db_session.flush()

    summary = _summarize_tool_history(db_session, task)
    assert "- called web_search" in summary
    assert "- error: Something broke" in summary


# ---------------------------------------------------------------------------
# Coordinator-level: circuit breaker trips and raises guardrail error
# (Existing behavior preserved -- the coordinator still raises.)
# ---------------------------------------------------------------------------


def test_circuit_breaker_still_raises_guardrail_error(db_session: Session) -> None:
    """Coordinator raises AgentExecutionGuardrailError on circuit breaker trip."""
    task = _create_task(
        db_session, input_text="search for the same thing over and over"
    )
    llm = FakeLLM(
        [
            Completion(
                content=None,
                tool_calls=(
                    ToolCall(id="c1", name="web_search", arguments={"query": "same"}),
                ),
                usage=TokenUsage(input_tokens=10, output_tokens=3),
            ),
            Completion(
                content=None,
                tool_calls=(
                    ToolCall(id="c2", name="web_search", arguments={"query": "same"}),
                ),
                usage=TokenUsage(input_tokens=10, output_tokens=3),
            ),
            Completion(
                content=None,
                tool_calls=(
                    ToolCall(id="c3", name="web_search", arguments={"query": "same"}),
                ),
                usage=TokenUsage(input_tokens=10, output_tokens=3),
            ),
        ]
    )

    with pytest.raises(AgentExecutionGuardrailError):
        AgentCoordinator(
            session=db_session,
            llm=llm,
            registry=ToolRegistry([AlwaysRepeatSearchTool()]),
        ).run(task)


def test_max_recoverable_failures_still_raises_guardrail_error(
    db_session: Session,
) -> None:
    """Coordinator raises AgentExecutionGuardrailError when recoverable budget exceeded."""
    task = _create_task(db_session, input_text="query the database")
    llm = FakeLLM(
        [
            Completion(
                content=None,
                tool_calls=(
                    ToolCall(id="c1", name="query_database", arguments={"query": "x"}),
                ),
                usage=TokenUsage(input_tokens=10, output_tokens=3),
            ),
            Completion(
                content=None,
                tool_calls=(
                    ToolCall(id="c2", name="query_database", arguments={"query": "x"}),
                ),
                usage=TokenUsage(input_tokens=10, output_tokens=3),
            ),
        ]
    )

    with pytest.raises(AgentExecutionGuardrailError):
        AgentCoordinator(
            session=db_session,
            llm=llm,
            registry=ToolRegistry([AlwaysFailTool()]),
            guardrail_limits=ExecutionGuardrailLimits(
                max_turns=6,
                max_tool_calls=12,
                max_recoverable_failures=1,
                max_same_tool_call=5,
                max_same_recoverable_error=5,
            ),
        ).run(task)


# ---------------------------------------------------------------------------
# Worker _synthesize_honest_failure: LLM synthesis path
# ---------------------------------------------------------------------------


def test_synthesize_honest_failure_uses_llm_result(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the synthesis LLM call succeeds, its text is used as the failure message."""
    task = _create_task(db_session, input_text="find something")

    # Patch LLMService.complete directly so we control what the synthesis call
    # returns without needing a real provider or pricing rows.
    expected_message = "I tried searching but couldn't find what you needed."

    from kortny.llm import LLMService

    def _fake_complete(
        self: LLMService,
        *,
        task_id: uuid.UUID,
        messages: Sequence[ChatMessage],
        tools: Sequence[JsonSchema] = (),
        response_format: JsonObject | None = None,
        prompt_name: str | None = None,
        prompt_source: str = "code",
        prompt_label: str | None = None,
        prompt_version: str | None = None,
        max_output_tokens: int | None = None,
    ) -> Completion:
        return Completion(
            content=expected_message,
            tool_calls=(),
            usage=TokenUsage(input_tokens=20, output_tokens=10),
            response_id="synth-1",
            model="openai/gpt-4o-mini",
        )

    monkeypatch.setattr(LLMService, "complete", _fake_complete)

    # Also stub out select_runtime_model / create_provider_for_selection so no
    # real DB config or LiteLLM provider is constructed.
    from kortny.llm import ModelRoute
    from kortny.llm.routing import ModelRouteTier

    def _fake_select(**_kwargs: object) -> object:
        from kortny.llm.runtime_config import RuntimeModelSelection

        class _M:
            model = "openai/gpt-4o-mini"
            provider_kind = "openrouter"
            litellm_provider_kwargs: dict[str, object] = {}
            provider_account_id = None
            model_catalog_id = None
            tier_assignment_id = None
            credential_source = "env"

        class _C:
            primary = _M()
            source = "env"
            fallback_reason = None
            skipped_candidate_count = 0

        return RuntimeModelSelection(
            model_route=ModelRoute(
                tier=ModelRouteTier.cheap_fast,
                model="openai/gpt-4o-mini",
                reason="honest_failure_synthesis",
            ),
            chain=_C(),  # type: ignore[arg-type]
            model=_M(),  # type: ignore[arg-type]
            provider_name=LLMProvider.openrouter,
        )

    monkeypatch.setattr(
        "kortny.worker.agent_executor.select_runtime_model", _fake_select
    )
    monkeypatch.setattr(
        "kortny.worker.agent_executor.create_provider_for_selection",
        lambda **_kwargs: None,  # provider unused when complete is patched
    )

    settings = _make_settings()
    executor = AgentTaskExecutor(settings=settings)
    result = executor._synthesize_honest_failure(
        settings=settings,
        session=db_session,
        task=task,
        task_service=TaskService(db_session),
        failure_reason="same_tool_call_repeated",
    )

    assert result == expected_message
    log_messages = _task_log_messages(db_session, task)
    assert "honest_failure_synthesized" in log_messages


def test_synthesize_honest_failure_falls_back_when_llm_raises(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the synthesis LLM call raises, the deterministic fallback is used."""
    task = _create_task(db_session, input_text="find something")

    def _raise(**_kwargs: object) -> object:
        raise RuntimeError("provider down")

    monkeypatch.setattr(
        "kortny.worker.agent_executor.select_runtime_model",
        _raise,
    )

    settings = _make_settings()
    executor = AgentTaskExecutor(settings=settings)
    result = executor._synthesize_honest_failure(
        settings=settings,
        session=db_session,
        task=task,
        task_service=TaskService(db_session),
        failure_reason="same_tool_call_repeated",
    )

    expected_fallback = coordinator_module.HONEST_FAILURE_FALLBACK[
        "same_tool_call_repeated"
    ]
    assert result == expected_fallback
    assert result != GENERIC_FAILURE_TEXT
    log_messages = _task_log_messages(db_session, task)
    assert "honest_failure_fallback_used" in log_messages


def test_synthesize_honest_failure_uses_default_fallback_for_unknown_reason(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unknown failure reasons fall back to the default (non-generic) text."""
    task = _create_task(db_session, input_text="do something")

    def _raise(**_kwargs: object) -> object:
        raise RuntimeError("provider down")

    monkeypatch.setattr(
        "kortny.worker.agent_executor.select_runtime_model",
        _raise,
    )

    settings = _make_settings()
    executor = AgentTaskExecutor(settings=settings)
    result = executor._synthesize_honest_failure(
        settings=settings,
        session=db_session,
        task=task,
        task_service=TaskService(db_session),
        failure_reason="unknown",
    )

    assert result == coordinator_module.HONEST_FAILURE_FALLBACK_DEFAULT
    assert result != GENERIC_FAILURE_TEXT


def test_synthesize_honest_failure_empty_llm_response_uses_fallback(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the synthesis LLM returns an empty string, use the deterministic fallback."""
    task = _create_task(db_session, input_text="find something")

    class _EmptyProvider:
        model = "openai/gpt-4o-mini"

        def complete(
            self,
            messages: Sequence[ChatMessage],
            tools: Sequence[JsonSchema] = (),
            *,
            response_format: JsonObject | None = None,
            max_output_tokens: int | None = None,
        ) -> Completion:
            return Completion(
                content="",
                tool_calls=(),
                usage=TokenUsage(input_tokens=5, output_tokens=0),
                response_id="empty-1",
                model="openai/gpt-4o-mini",
            )

    from kortny.llm import ModelRoute
    from kortny.llm.routing import ModelRouteTier
    from kortny.llm.runtime_config import RuntimeModelSelection

    fake_route = ModelRoute(
        tier=ModelRouteTier.cheap_fast,
        model="openai/gpt-4o-mini",
        reason="honest_failure_synthesis",
    )

    class _FakeModel:
        model = "openai/gpt-4o-mini"
        provider_kind = "openrouter"
        litellm_provider_kwargs: dict[str, object] = {}

    class _FakeChain:
        primary = _FakeModel()

    fake_selection = RuntimeModelSelection(
        model_route=fake_route,
        chain=_FakeChain(),  # type: ignore[arg-type]
        model=_FakeModel(),  # type: ignore[arg-type]
        provider_name=LLMProvider.openrouter,
    )

    monkeypatch.setattr(
        "kortny.worker.agent_executor.select_runtime_model",
        lambda **_kwargs: fake_selection,
    )
    monkeypatch.setattr(
        "kortny.worker.agent_executor.create_provider_for_selection",
        lambda **_kwargs: _EmptyProvider(),
    )

    settings = _make_settings()
    executor = AgentTaskExecutor(settings=settings)
    result = executor._synthesize_honest_failure(
        settings=settings,
        session=db_session,
        task=task,
        task_service=TaskService(db_session),
        failure_reason="recoverable_failure_budget_exceeded",
    )

    expected_fallback = coordinator_module.HONEST_FAILURE_FALLBACK[
        "recoverable_failure_budget_exceeded"
    ]
    assert result == expected_fallback
    log_messages = _task_log_messages(db_session, task)
    assert "honest_failure_fallback_used" in log_messages


# ---------------------------------------------------------------------------
# _post_honest_failure_notice: suppression gate
# ---------------------------------------------------------------------------


def test_post_honest_failure_suppressed_for_background_assessment(
    db_session: Session,
) -> None:
    """Background assessment tasks suppress the Slack post but log the suppression."""

    task = _create_task(db_session, input_text="assess channel")
    # Mark as playground to trigger suppression (simplest path through
    # _should_suppress_slack_post).
    task.slack_channel_id = "playground"
    db_session.flush()

    messages_posted: list[str] = []

    class _NoSlack:
        def chat_postMessage(self, **_kwargs: object) -> dict[str, object]:
            messages_posted.append("posted")
            return {"ok": True, "ts": "1.0"}

    exc = AgentExecutionGuardrailError(
        "Execution circuit breaker tripped for repeated tool call x"
    )
    settings = _make_settings()
    executor = AgentTaskExecutor(settings=settings, slack_client=_NoSlack())  # type: ignore[arg-type]
    executor._post_honest_failure_notice(
        settings=settings,
        session=db_session,
        task=task,
        task_service=TaskService(db_session),
        exc=exc,
    )

    # No Slack message was posted
    assert messages_posted == []
    # Suppression was logged
    log_messages = _task_log_messages(db_session, task)
    assert "slack_honest_failure_suppressed" in log_messages


def test_post_honest_failure_posts_to_slack(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_post_honest_failure_notice calls SlackPoster with the synthesized text."""
    task = _create_task(db_session, input_text="search for something")

    # Monkeypatch synthesis to return a deterministic message without an LLM call.
    deterministic_msg = "I tried to search but kept hitting the same dead end."
    monkeypatch.setattr(
        AgentTaskExecutor,
        "_synthesize_honest_failure",
        lambda self, **_kw: deterministic_msg,
    )

    posted_messages: list[str] = []

    class _RecordingSlack:
        def chat_postMessage(
            self,
            *,
            channel: str,
            text: str,
            thread_ts: str | None = None,
            blocks: object = None,
            unfurl_links: bool = True,
            unfurl_media: bool = True,
        ) -> dict[str, object]:
            posted_messages.append(text)
            return {"ok": True, "ts": "1716400100.000001"}

    exc = AgentExecutionGuardrailError(
        "Execution circuit breaker tripped for repeated tool call x"
    )
    settings = _make_settings()
    executor = AgentTaskExecutor(settings=settings, slack_client=_RecordingSlack())  # type: ignore[arg-type]
    executor._post_honest_failure_notice(
        settings=settings,
        session=db_session,
        task=task,
        task_service=TaskService(db_session),
        exc=exc,
    )

    assert len(posted_messages) == 1
    assert posted_messages[0] == deterministic_msg
    # Generic failure text was NOT used
    assert posted_messages[0] != GENERIC_FAILURE_TEXT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings() -> Settings:
    return Settings.model_validate(
        {
            "SLACK_BOT_TOKEN": "xoxb-test",
            "SLACK_APP_TOKEN": "xapp-test",
            "SLACK_SIGNING_SECRET": "signing-secret",
            "LLM_PROVIDER": SettingsLLMProvider.openrouter,
            "LLM_API_KEY": "openrouter-key",
            "LLM_MODEL": "openai/gpt-4o-mini",
            "AGENT_RUNTIME": "custom",
            "KORTNY_WORKFLOW_BACKEND": "inline",
            "POSTGRES_URL": "postgresql://kortny:kortny@localhost/kortny",
            "KORTNY_EMBEDDINGS_BACKEND": "disabled",
        }
    )
