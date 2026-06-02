import json
import os
import uuid
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from alembic import command
from alembic.config import Config
from google.adk.models.llm_response import LlmResponse
from google.genai import types as genai_types
from sqlalchemy import Engine, delete, select
from sqlalchemy.orm import Session, sessionmaker

from kortny.agent.adk_runtime import AdkAgentRuntime
from kortny.agent.coordinator import AgentRunResult
from kortny.approvals import (
    TOOL_APPROVAL_PROMPT_PURPOSE,
    TOOL_APPROVAL_REQUIRED_MESSAGE,
    TOOL_APPROVAL_WAITING_MESSAGE,
)
from kortny.config.settings import LLMProvider as SettingsLLMProvider
from kortny.config.settings import Settings
from kortny.db.models import (
    Artifact,
    EncryptedSecret,
    Episode,
    Installation,
    LLMProvider,
    LLMUsage,
    ModelPricing,
    ObserveChannelProfile,
    ProceduralSkillInvocation,
    SlackChannelMembership,
    SlackSideEffect,
    Task,
    TaskEvent,
    TaskEventType,
    TaskStatus,
)
from kortny.db.session import make_engine, make_session_factory, normalize_database_url
from kortny.llm import ChatMessage, Completion, TokenUsage, ToolCall
from kortny.llm.routing import ModelRoute, ModelRouteTier
from kortny.observe.assessment import (
    CHANNEL_ASSESSMENT_COMPLETED_MESSAGE,
    CHANNEL_ASSESSMENT_REQUESTED_MESSAGE,
)
from kortny.slack.comments import ARTIFACT_COMMENT_FALLBACK_TEXT
from kortny.slack.humanizer import build_response_record, build_synthesis_context
from kortny.slack.reactions import (
    ACK_REACTION_ADDED_MESSAGE,
    ACK_REACTION_REMOVED_MESSAGE,
)
from kortny.slack.synthesis import SynthesisOutcome
from kortny.tasks import TaskService
from kortny.tools import ToolResult
from kortny.tools.types import JsonObject, JsonSchema
from kortny.worker import (
    AgentTaskExecutor,
    TaskExecutionResult,
    TaskWorker,
    WalkingSkeletonExecutor,
)

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
        executor=WalkingSkeletonExecutor(),
        lease_for=timedelta(seconds=60),
    ).run_once(now=claim_time)

    db_session.refresh(task)
    assert result.worker_id == "worker-test"
    assert result.status == TaskStatus.succeeded.value, task.error
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
        (TaskEventType.log, None),
    ]
    assert events[2].payload == {
        "message": "task_executor_started",
        "worker_id": "worker-test",
    }
    assert events[3].payload == {
        "message": "task_executor_completed",
        "worker_id": "worker-test",
    }
    assert events[5].payload["message"] == "episode_recorded"
    assert events[5].payload["outcome"] == TaskStatus.succeeded.value
    episode = db_session.scalar(select(Episode).where(Episode.task_id == task.id))
    assert episode is not None
    assert episode.outcome == TaskStatus.succeeded.value
    assert episode.summary == task.result_summary


def test_worker_run_once_is_idle_without_pending_task(
    db_session: Session,
    worker_session_factory: sessionmaker[Session],
) -> None:
    result = TaskWorker(
        session_factory=worker_session_factory,
        worker_id="worker-test",
        executor=WalkingSkeletonExecutor(),
    ).run_once(now=datetime(2026, 5, 23, 9, 25, tzinfo=UTC))

    assert result.worker_id == "worker-test"
    assert result.status == "idle"
    assert result.task_id is None
    assert result.handled_task is False


def test_worker_recovers_stale_slack_side_effects_before_poll(
    db_session: Session,
    worker_session_factory: sessionmaker[Session],
) -> None:
    now = datetime(2026, 5, 31, 12, 15, tzinfo=UTC)
    installation = Installation(slack_team_id=f"T{uuid.uuid4().hex}")
    db_session.add(installation)
    db_session.flush()
    side_effect = SlackSideEffect(
        installation_id=installation.id,
        idempotency_key=f"stale:{uuid.uuid4()}",
        operation="chat_postMessage",
        purpose="result",
        request_json={"channel": "C123"},
        status="in_progress",
        attempts=1,
        started_at=now - timedelta(minutes=7),
        available_at=now - timedelta(minutes=7),
    )
    db_session.add(side_effect)
    db_session.commit()

    result = TaskWorker(
        session_factory=worker_session_factory,
        worker_id="worker-test",
        executor=WalkingSkeletonExecutor(),
    ).run_once(now=now)

    db_session.refresh(side_effect)
    assert result.status == "idle"
    assert result.recovered_side_effect_ids == (side_effect.id,)
    assert side_effect.status == "failed"
    assert side_effect.last_error is not None
    assert side_effect.last_error["type"] == "StaleSideEffectLease"
    assert side_effect.last_error["delivery_state"] == "unknown"


def test_worker_marks_task_failed_when_handler_raises(
    db_session: Session,
    worker_session_factory: sessionmaker[Session],
) -> None:
    claim_time = datetime(2026, 5, 23, 9, 30, tzinfo=UTC)
    task = create_task(db_session, event_id="EvWorkerFailure")
    task.available_at = claim_time - timedelta(seconds=1)
    db_session.commit()

    class FailingExecutor:
        def execute(
            self,
            *,
            session: Session,
            task: Task,
            task_service: TaskService,
        ) -> TaskExecutionResult:
            raise RuntimeError(f"boom {task.id}")

    result = TaskWorker(
        session_factory=worker_session_factory,
        worker_id="worker-test",
        executor=FailingExecutor(),
    ).run_once(now=claim_time)

    db_session.refresh(task)
    assert result.status == TaskStatus.failed.value
    assert result.task_id == task.id
    assert task.status is TaskStatus.failed
    assert task.error is not None
    assert task.error["type"] == "RuntimeError"
    assert task.locked_by is None
    episode = db_session.scalar(select(Episode).where(Episode.task_id == task.id))
    assert episode is not None
    assert episode.outcome == TaskStatus.failed.value
    assert episode.error_json == {
        "type": "RuntimeError",
        "message": f"boom {task.id}",
        "worker_id": "worker-test",
    }

    events = task_events(db_session, task)
    assert events[-3].type is TaskEventType.error
    assert events[-3].payload["message"] == "task_executor_failed"
    assert events[-2].type is TaskEventType.status_changed
    assert events[-2].payload["to"] == TaskStatus.failed.value
    assert events[-1].type is TaskEventType.log
    assert events[-1].payload["message"] == "episode_recorded"
    assert events[-1].payload["outcome"] == TaskStatus.failed.value


def test_worker_exits_cleanly_when_task_is_cancelled_cooperatively(
    db_session: Session,
    worker_session_factory: sessionmaker[Session],
) -> None:
    claim_time = datetime(2026, 5, 23, 9, 35, tzinfo=UTC)
    task = create_task(db_session, event_id="EvWorkerCancel")
    task.available_at = claim_time - timedelta(seconds=1)
    db_session.commit()

    class CancellingExecutor:
        def execute(
            self,
            *,
            session: Session,
            task: Task,
            task_service: TaskService,
        ) -> TaskExecutionResult:
            task_service.cancel_task(task, by_user_id=task.slack_user_id)
            task_service.raise_if_cancelled(task, phase="test_executor")
            return TaskExecutionResult(result_summary="should not be used")

    result = TaskWorker(
        session_factory=worker_session_factory,
        worker_id="worker-test",
        executor=CancellingExecutor(),
    ).run_once(now=claim_time)

    db_session.refresh(task)
    events = task_events(db_session, task)

    assert result.status == TaskStatus.cancelled.value
    assert result.task_id == task.id
    assert task.status is TaskStatus.cancelled
    assert task.result_summary is None
    assert task.error is None
    assert task.locked_by is None
    assert task.lease_expires_at is None
    assert any(
        event.payload.get("message") == "task_executor_cancelled"
        for event in events
        if event.type is TaskEventType.log
    )
    assert not any(
        event.payload.get("message") == "task_executor_failed"
        for event in events
        if event.type is TaskEventType.error
    )


def test_worker_runs_agent_flow_and_posts_pdf(
    db_session: Session,
    worker_session_factory: sessionmaker[Session],
    tmp_path: Path,
) -> None:
    claim_time = datetime(2026, 5, 23, 9, 45, tzinfo=UTC)
    task = create_task(db_session, event_id="EvAgentWorker")
    task.input = "research Python temp files and make a PDF"
    task.available_at = claim_time - timedelta(seconds=1)
    db_session.add(
        ModelPricing(
            provider=LLMProvider.openrouter,
            model="openai/gpt-4o-mini",
            input_price_per_mtok=Decimal("1.000000"),
            output_price_per_mtok=Decimal("2.000000"),
            effective_from=datetime(2026, 1, 1, tzinfo=UTC),
        )
    )
    db_session.commit()

    slack_client = FakeSlackClient()
    provider = FakeAgentProvider(
        [
            Completion(
                content=None,
                tool_calls=(
                    ToolCall(
                        id="call-search",
                        name="web_search",
                        arguments={"query": "Python temp files", "count": 2},
                    ),
                ),
                usage=TokenUsage(input_tokens=1000, output_tokens=500),
                model="openai/gpt-4o-mini",
            ),
            Completion(
                content=None,
                tool_calls=(
                    ToolCall(
                        id="call-pdf",
                        name="pdf_generator",
                        arguments={
                            "title": "Python Temp Files Research",
                            "filename": "python-temp-files.pdf",
                            "sections": [
                                {
                                    "heading": "Summary",
                                    "body": "Python tempfile creates temporary files and directories safely.",
                                    "bullets": [
                                        "TemporaryDirectory cleans up automatically.",
                                        "NamedTemporaryFile gives a visible file name.",
                                    ],
                                }
                            ],
                        },
                    ),
                ),
                usage=TokenUsage(input_tokens=1200, output_tokens=600),
                model="openai/gpt-4o-mini",
            ),
        ]
    )

    result = TaskWorker(
        session_factory=worker_session_factory,
        worker_id="agent-worker-test",
        executor=AgentTaskExecutor(
            settings=make_settings(),
            llm_provider=provider,
            provider_name=LLMProvider.openrouter,
            web_search_tool=StaticWebSearchTool(),
            slack_client=slack_client,
            artifact_comment_generator=FakeArtifactCommentGenerator(
                "Here's the Python tempfile report."
            ),
            workspace_base_dir=tmp_path,
        ),
    ).run_once(now=claim_time)

    db_session.refresh(task)
    artifact = db_session.scalar(select(Artifact).where(Artifact.task_id == task.id))
    usage_rows = list(
        db_session.scalars(select(LLMUsage).where(LLMUsage.task_id == task.id))
    )
    events = task_events(db_session, task)

    assert result.status == TaskStatus.succeeded.value, task.error
    assert task.status is TaskStatus.succeeded
    assert task.result_summary == "Generated 1 artifact."
    assert task.total_input_tokens == 2200
    assert task.total_output_tokens == 1100
    assert task.total_cost_usd == Decimal("0.004400")
    assert len(usage_rows) == 2

    assert artifact is not None
    assert artifact.filename == "python-temp-files.pdf"
    assert artifact.mime_type == "application/pdf"
    assert artifact.slack_file_id == "F000001"
    assert artifact.posted_at is not None

    assert slack_client.messages == []
    assert len(slack_client.uploads) == 1
    assert slack_client.uploads[0]["channel"] == "C123"
    assert slack_client.uploads[0]["thread_ts"] == "EvAgentWorker"
    assert slack_client.uploads[0]["filename"] == "python-temp-files.pdf"
    assert slack_client.uploads[0]["initial_comment"] == (
        "Here's the Python tempfile report."
    )
    assert slack_client.uploads[0]["file_bytes"].startswith(b"%PDF-")

    event_types = [event.type for event in events]
    assert event_types.count(TaskEventType.llm_call) == 2
    assert event_types.count(TaskEventType.tool_call) == 2
    assert event_types.count(TaskEventType.tool_result) == 2
    assert TaskEventType.artifact_created in event_types
    assert TaskEventType.message_posted in event_types
    assert [
        event.payload["to"] for event in events if event.type == "status_changed"
    ] == [
        "running",
        "succeeded",
    ]


def test_worker_posts_approval_prompt_for_sensitive_tool(
    db_session: Session,
    worker_session_factory: sessionmaker[Session],
    tmp_path: Path,
) -> None:
    claim_time = datetime(2026, 5, 31, 14, 10, tzinfo=UTC)
    task = create_task(db_session, event_id="EvApprovalGate")
    task.input = "forget my pdf policy"
    task.available_at = claim_time - timedelta(seconds=1)
    db_session.add(
        ModelPricing(
            provider=LLMProvider.openrouter,
            model="openai/gpt-4o-mini",
            input_price_per_mtok=Decimal("1.000000"),
            output_price_per_mtok=Decimal("2.000000"),
            effective_from=datetime(2026, 1, 1, tzinfo=UTC),
        )
    )
    db_session.commit()

    slack_client = FakeSlackClient()
    provider = FakeAgentProvider(
        [
            Completion(
                content=None,
                tool_calls=(
                    ToolCall(
                        id="call-forget",
                        name="forget_fact",
                        arguments={"scope": "user", "key": "pdf_policy"},
                    ),
                ),
                usage=TokenUsage(input_tokens=500, output_tokens=100),
                model="openai/gpt-4o-mini",
            )
        ]
    )

    result = TaskWorker(
        session_factory=worker_session_factory,
        worker_id="approval-worker-test",
        executor=AgentTaskExecutor(
            settings=make_settings(),
            llm_provider=provider,
            provider_name=LLMProvider.openrouter,
            slack_client=slack_client,
            workspace_base_dir=tmp_path,
        ),
    ).run_once(now=claim_time)

    db_session.refresh(task)
    events = task_events(db_session, task)
    posted_events = [
        event for event in events if event.type is TaskEventType.message_posted
    ]

    assert result.status == TaskStatus.waiting_approval.value
    assert task.status is TaskStatus.waiting_approval
    assert task.locked_by is None
    assert task.lease_expires_at is None
    assert len(slack_client.messages) == 1
    assert "approval before I run *forget_fact*" in slack_client.messages[0]["text"]
    assert "React with :white_check_mark:" in slack_client.messages[0]["text"]
    assert posted_events[0].payload["purpose"] == TOOL_APPROVAL_PROMPT_PURPOSE
    assert any(
        event.payload.get("message") == TOOL_APPROVAL_REQUIRED_MESSAGE
        for event in events
    )
    assert any(
        event.payload.get("message") == TOOL_APPROVAL_WAITING_MESSAGE
        for event in events
    )
    assert not any(event.type is TaskEventType.tool_call for event in events)
    assert [
        event.payload["to"] for event in events if event.type == "status_changed"
    ] == [
        "running",
        "waiting_approval",
    ]


def test_agent_executor_falls_back_when_artifact_comment_generation_fails(
    db_session: Session,
    worker_session_factory: sessionmaker[Session],
    tmp_path: Path,
) -> None:
    claim_time = datetime(2026, 5, 23, 9, 50, tzinfo=UTC)
    task = create_task(db_session, event_id="EvAgentWorkerCommentFallback")
    task.input = "generate a short PDF"
    task.available_at = claim_time - timedelta(seconds=1)
    db_session.add(
        ModelPricing(
            provider=LLMProvider.openrouter,
            model="openai/gpt-4o-mini",
            input_price_per_mtok=Decimal("1.000000"),
            output_price_per_mtok=Decimal("2.000000"),
            effective_from=datetime(2026, 1, 1, tzinfo=UTC),
        )
    )
    db_session.commit()

    slack_client = FakeSlackClient()
    provider = FakeAgentProvider(
        [
            Completion(
                content=None,
                tool_calls=(
                    ToolCall(
                        id="call-pdf",
                        name="pdf_generator",
                        arguments={
                            "title": "Short Report",
                            "filename": "short-report.pdf",
                            "sections": [{"heading": "Summary", "body": "Done."}],
                        },
                    ),
                ),
                usage=TokenUsage(input_tokens=50, output_tokens=25),
                model="openai/gpt-4o-mini",
            ),
        ]
    )

    result = TaskWorker(
        session_factory=worker_session_factory,
        worker_id="agent-worker-test",
        executor=AgentTaskExecutor(
            settings=make_settings(),
            llm_provider=provider,
            provider_name=LLMProvider.openrouter,
            web_search_tool=StaticWebSearchTool(),
            slack_client=slack_client,
            artifact_comment_generator=FakeArtifactCommentGenerator(
                error=RuntimeError("comment failed")
            ),
            workspace_base_dir=tmp_path,
        ),
    ).run_once(now=claim_time)

    db_session.refresh(task)
    events = task_events(db_session, task)

    assert result.status == TaskStatus.succeeded.value, task.error
    assert slack_client.uploads[0]["initial_comment"] == ARTIFACT_COMMENT_FALLBACK_TEXT
    assert any(
        event.payload.get("message") == "artifact_comment_generation_failed"
        for event in events
    )


def test_agent_executor_builds_routed_llm_from_task_input(
    db_session: Session,
) -> None:
    task = create_task(db_session, event_id="EvRouteDocument")
    task.input = "make a detailed PDF report about this discussion"
    task_service = TaskService(db_session)
    settings = Settings.model_validate(
        {
            "SLACK_BOT_TOKEN": "xoxb-test",
            "SLACK_APP_TOKEN": "xapp-test",
            "SLACK_SIGNING_SECRET": "signing-secret",
            "LLM_PROVIDER": SettingsLLMProvider.openrouter,
            "LLM_API_KEY": "openrouter-key",
            "LLM_MODEL": "openai/default",
            "LLM_DOCUMENT_MODEL": "anthropic/document-model",
            "BRAVE_SEARCH_API_KEY": "brave-key",
            "POSTGRES_URL": "postgresql://kortny:kortny@localhost/kortny",
        }
    )

    llm = AgentTaskExecutor(settings=settings)._build_llm(
        settings=settings,
        session=db_session,
        task=task,
        task_service=task_service,
    )

    events = task_events(db_session, task)

    assert llm.provider.model == "anthropic/document-model"
    assert llm.model_tier == "document"
    assert any(
        event.payload.get("message") == "model_route_selected"
        and event.payload.get("tier") == "document"
        and event.payload.get("model") == "anthropic/document-model"
        for event in events
    )


def test_agent_executor_passes_routed_model_to_adk_runtime(
    db_session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = create_task(db_session, event_id="EvAdkRouteCheap")
    task.input = "are you up?"
    task_service = TaskService(db_session)
    task_service.append_event(
        task,
        TaskEventType.log,
        {
            "message": "intent_classification_completed",
            "decision": {
                "classification": "task_request",
                "model_tier": "cheap",
                "likely_tools": [],
                "needs_channel_context": False,
                "needs_thread_context": False,
                "needs_file_context": False,
            },
        },
    )
    settings = Settings.model_validate(
        {
            "SLACK_BOT_TOKEN": "xoxb-test",
            "SLACK_APP_TOKEN": "xapp-test",
            "SLACK_SIGNING_SECRET": "signing-secret",
            "LLM_PROVIDER": SettingsLLMProvider.openrouter,
            "LLM_API_KEY": "openrouter-key",
            "LLM_MODEL": "anthropic/sonnet-default",
            "LLM_CHEAP_MODEL": "deepseek/deepseek-v4-flash",
            "LLM_STANDARD_MODEL": "openai/gpt-5.4-mini",
            "POSTGRES_URL": "postgresql://kortny:kortny@localhost/kortny",
            "AGENT_RUNTIME": "adk",
        }
    )
    captured: dict[str, Any] = {}

    class FakeAdkRuntime:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        def run(self, task_arg: Task) -> AgentRunResult:
            return AgentRunResult(
                task_id=task_arg.id,
                result_summary="Yep, up.",
                turns=1,
                artifact_count=0,
            )

    monkeypatch.setattr(
        "kortny.agent.adk_runtime.AdkAgentRuntime",
        FakeAdkRuntime,
    )

    result = AgentTaskExecutor(settings=settings)._run_agent_runtime(
        settings=settings,
        session=db_session,
        task=task,
        task_service=task_service,
        working_dir=tmp_path,
    )

    events = task_events(db_session, task)

    assert result.result_summary == "Yep, up."
    assert captured["model"] == "deepseek/deepseek-v4-flash"
    assert captured["model_route"].model == "deepseek/deepseek-v4-flash"
    assert captured["model_route"].tier is ModelRouteTier.cheap_fast
    assert captured["registry_factory"] is not None
    assert any(
        event.payload.get("message") == "model_route_selected"
        and event.payload.get("runtime") == "adk"
        and event.payload.get("tier") == "cheap_fast"
        and event.payload.get("model") == "deepseek/deepseek-v4-flash"
        for event in events
    )


def test_adk_model_callback_records_llm_usage(
    db_session: Session,
) -> None:
    task = create_task(db_session, event_id="EvAdkUsage")
    task_service = TaskService(db_session)
    db_session.add(
        ModelPricing(
            provider=LLMProvider.openrouter,
            model="anthropic/claude-sonnet-4.6",
            input_price_per_mtok=Decimal("3.00"),
            output_price_per_mtok=Decimal("15.00"),
            effective_from=datetime(2026, 1, 1, tzinfo=UTC),
        )
    )
    db_session.commit()
    settings = Settings.model_validate(
        {
            "SLACK_BOT_TOKEN": "xoxb-test",
            "SLACK_APP_TOKEN": "xapp-test",
            "SLACK_SIGNING_SECRET": "signing-secret",
            "LLM_PROVIDER": SettingsLLMProvider.openrouter,
            "LLM_API_KEY": "openrouter-key",
            "LLM_MODEL": "anthropic/sonnet-default",
            "POSTGRES_URL": "postgresql://kortny:kortny@localhost/kortny",
            "AGENT_RUNTIME": "adk",
        }
    )
    runtime = AdkAgentRuntime(
        settings=settings,
        session=db_session,
        task_service=task_service,
        model_route=ModelRoute(
            tier=ModelRouteTier.analysis,
            model="anthropic/claude-sonnet-4.6",
            reason="intent_classifier",
        ),
    )
    context = FakeAdkContext(
        agent_name="kortny_root_orchestrator",
        invocation_id="adk-invocation-1",
        task_id=str(task.id),
    )
    response = LlmResponse(
        model_version="openrouter/anthropic/claude-sonnet-4.6",
        usage_metadata=genai_types.GenerateContentResponseUsageMetadata(
            prompt_token_count=100,
            tool_use_prompt_token_count=10,
            candidates_token_count=20,
            total_token_count=130,
        ),
    )

    runtime._record_adk_model_usage(callback_context=context, llm_response=response)
    usage = db_session.scalar(select(LLMUsage).where(LLMUsage.task_id == task.id))

    assert usage is not None
    assert usage.provider is LLMProvider.openrouter
    assert usage.model == "anthropic/claude-sonnet-4.6"
    assert usage.model_tier == "analysis"
    assert usage.input_tokens == 110
    assert usage.output_tokens == 20
    assert usage.cost_usd == Decimal("0.000630")
    db_session.refresh(task)
    assert task.total_input_tokens == 110
    assert task.total_output_tokens == 20
    assert task.total_cost_usd == Decimal("0.000630")
    event = next(
        event
        for event in task_events(db_session, task)
        if event.payload.get("prompt_name") == "kortny.adk.kortny_root_orchestrator"
    )
    assert event.payload["runtime"] == "adk"
    assert event.payload["route_reason"] == "intent_classifier"
    assert event.payload["pricing_missing"] is False


def test_agent_executor_skips_humanizer_for_adk_quick_fast_path(
    db_session: Session,
) -> None:
    task = create_task(db_session, event_id="EvAdkQuickSkipHumanizer")
    task.input = "are you up?"
    task_service = TaskService(db_session)
    task_service.append_event(
        task,
        TaskEventType.log,
        {
            "message": "model_route_selected",
            "runtime": "adk",
            "tier": "cheap_fast",
            "model": "deepseek/deepseek-v4-flash",
            "reason": "intent_classifier",
        },
    )
    task_service.append_event(
        task,
        TaskEventType.log,
        {
            "message": "adk_runtime_completed",
            "runtime": "adk",
            "mode": "orchestrated",
            "event_count": 3,
            "final_author": "quick_response_agent",
            "authors": ["kortny_root_orchestrator", "quick_response_agent"],
            "result_chars": 21,
        },
    )
    settings = Settings.model_validate(
        {
            "SLACK_BOT_TOKEN": "xoxb-test",
            "SLACK_APP_TOKEN": "xapp-test",
            "SLACK_SIGNING_SECRET": "signing-secret",
            "LLM_PROVIDER": SettingsLLMProvider.openrouter,
            "LLM_API_KEY": "openrouter-key",
            "LLM_MODEL": "anthropic/sonnet-default",
            "LLM_CHEAP_MODEL": "deepseek/deepseek-v4-flash",
            "POSTGRES_URL": "postgresql://kortny:kortny@localhost/kortny",
            "AGENT_RUNTIME": "adk",
            "RESPONSE_HUMANIZER_ENABLED": True,
        }
    )
    slack_client = FakeSlackClient()

    AgentTaskExecutor(
        settings=settings,
        llm_provider=FakeAgentProvider([]),
        slack_client=slack_client,
    )._post_outputs(
        settings=settings,
        session=db_session,
        task=task,
        task_service=task_service,
        result_summary="Yep, up and ready!",
    )

    events = task_events(db_session, task)

    assert slack_client.messages[-1]["text"] == "Yep, up and ready!"
    assert any(
        event.payload.get("message") == "response_humanizer_skipped"
        and event.payload.get("reason") == "adk_quick_fast_path"
        for event in events
    )
    assert not any(
        event.payload.get("message") == "response_humanizer_started" for event in events
    )


def test_agent_executor_humanizes_final_text_before_posting(
    db_session: Session,
    worker_session_factory: sessionmaker[Session],
    tmp_path: Path,
) -> None:
    claim_time = datetime(2026, 5, 23, 9, 50, tzinfo=UTC)
    task = create_task(db_session, event_id="EvAgentWorkerHumanizer")
    task.input = (
        "Compare current AI observability tools for Kortny and tell me which "
        "two matter most."
    )
    task.available_at = claim_time - timedelta(seconds=1)
    db_session.add(
        ModelPricing(
            provider=LLMProvider.openrouter,
            model="openai/gpt-4o-mini",
            input_price_per_mtok=Decimal("1.000000"),
            output_price_per_mtok=Decimal("2.000000"),
            effective_from=datetime(2026, 1, 1, tzinfo=UTC),
        )
    )
    db_session.commit()

    raw_answer = (
        "Here is a detailed comparison of AI observability tooling. "
        "1. Langfuse is open-source and strong for traces, prompts, and evals. "
        "2. Arize Phoenix is strong for OpenTelemetry-native traces and evals. "
        "If you want, I can turn this into a PDF report."
    )
    humanized_answer = (
        "*Quick take:* Kortny should care most about Langfuse and Arize Phoenix.\n\n"
        "- *Langfuse* is the better default for traces, prompts, and evals.\n"
        "- *Arize Phoenix* is the strongest complement for OTel-native debugging."
    )
    slack_client = FakeSlackClient()
    provider = FakeAgentProvider(
        [
            Completion(
                content=raw_answer,
                tool_calls=(),
                usage=TokenUsage(input_tokens=200, output_tokens=60),
                model="openai/gpt-4o-mini",
            ),
            Completion(
                content=humanized_answer,
                tool_calls=(),
                usage=TokenUsage(input_tokens=260, output_tokens=70),
                model="openai/gpt-4o-mini",
            ),
        ]
    )

    result = TaskWorker(
        session_factory=worker_session_factory,
        worker_id="agent-worker-test",
        executor=AgentTaskExecutor(
            settings=make_settings(),
            llm_provider=provider,
            provider_name=LLMProvider.openrouter,
            web_search_tool=StaticWebSearchTool(),
            slack_client=slack_client,
            workspace_base_dir=tmp_path,
        ),
    ).run_once(now=claim_time)

    db_session.refresh(task)
    events = task_events(db_session, task)
    usage_rows = list(
        db_session.scalars(select(LLMUsage).where(LLMUsage.task_id == task.id))
    )

    assert result.status == TaskStatus.succeeded.value, task.error
    assert task.result_summary == raw_answer
    assert slack_client.messages == [
        {
            "channel": "C123",
            "text": humanized_answer,
            "thread_ts": "EvAgentWorkerHumanizer",
        }
    ]
    assert len(usage_rows) == 2
    assert any(
        event.type is TaskEventType.llm_call
        and event.payload.get("prompt_name") == "kortny.response_humanizer"
        for event in events
    )
    assert any(
        event.payload.get("message") == "response_humanizer_started" for event in events
    )
    completed = next(
        event
        for event in events
        if event.payload.get("message") == "response_humanizer_completed"
    )
    assert completed.payload["changed"] is True
    assert completed.payload["reason"] == "llm_humanizer"
    record_event = next(
        event
        for event in events
        if event.payload.get("message") == "response_record_built"
    )
    assert record_event.payload["response_mode"] == "quick_answer"
    assert record_event.payload["response_shape"] == "comparison_memo"
    context_event = next(
        event
        for event in events
        if event.payload.get("message") == "synthesis_context_built"
    )
    assert context_event.payload["synthesis_outcome"] == "ok"
    assert context_event.payload["synthesis_evidence_count"] == 0
    humanizer_payload = provider.calls[1][0][1].content
    assert humanizer_payload is not None
    humanizer_contract = json.loads(humanizer_payload)
    response_record = humanizer_contract["response_record"]
    synthesis_context = humanizer_contract["synthesis_context"]
    assert response_record["response_mode"] == "quick_answer"
    assert response_record["response_shape"]["shape"] == "comparison_memo"
    assert synthesis_context["outcome"] == "ok"
    assert synthesis_context["user_intent"] == task.input
    assert synthesis_context["addressee_user_id"] == "U123"
    assert response_record["response_shape"]["required_elements"] == [
        "recommendation",
        "scope",
        "tradeoffs",
        "when_to_choose_each",
        "decision_risk",
        "next_step",
    ]
    assert response_record["user_request"] == task.input
    assert [skill["slug"] for skill in response_record["procedural_skills"]] == [
        "slack-humanizer",
        "analyst-grade-synthesis",
    ]
    skill_events = [
        event
        for event in events
        if event.payload.get("message") == "procedural_skill_invoked"
    ]
    assert {event.payload["slug"] for event in skill_events} == {
        "slack-humanizer",
        "analyst-grade-synthesis",
    }
    skill_invocations = list(
        db_session.scalars(
            select(ProceduralSkillInvocation).where(
                ProceduralSkillInvocation.task_id == task.id
            )
        )
    )
    assert len(skill_invocations) == 2


def test_agent_executor_builds_research_response_record_for_tool_results(
    db_session: Session,
    worker_session_factory: sessionmaker[Session],
    tmp_path: Path,
) -> None:
    claim_time = datetime(2026, 5, 23, 9, 51, tzinfo=UTC)
    task = create_task(db_session, event_id="EvAgentWorkerResearchHumanizer")
    task.input = (
        "Research current Python tempfile guidance and summarize the practical points."
    )
    task.available_at = claim_time - timedelta(seconds=1)
    db_session.add(
        ModelPricing(
            provider=LLMProvider.openrouter,
            model="openai/gpt-4o-mini",
            input_price_per_mtok=Decimal("1.000000"),
            output_price_per_mtok=Decimal("2.000000"),
            effective_from=datetime(2026, 1, 1, tzinfo=UTC),
        )
    )
    db_session.commit()

    raw_answer = (
        "I searched for Python tempfile guidance and found that the standard "
        "library documentation is still the best source for the main practical "
        "points. TemporaryDirectory is useful for scoped workspace cleanup, "
        "NamedTemporaryFile is useful when a visible path is required, and code "
        "should avoid hard-coded shared temp paths."
    )
    humanized_answer = (
        "*Practical take:* use the stdlib helpers and keep temp file lifetime "
        "explicit.\n\n"
        "- Use `TemporaryDirectory` for scoped cleanup.\n"
        "- Use `NamedTemporaryFile` when another process needs a visible path.\n"
        "- Avoid shared hard-coded temp paths."
    )
    slack_client = FakeSlackClient()
    provider = FakeAgentProvider(
        [
            Completion(
                content=None,
                tool_calls=(
                    ToolCall(
                        id="call-search-tempfile",
                        name="web_search",
                        arguments={
                            "query": "Python tempfile best practices",
                            "count": 2,
                        },
                    ),
                ),
                usage=TokenUsage(input_tokens=220, output_tokens=40),
                model="openai/gpt-4o-mini",
            ),
            Completion(
                content=raw_answer,
                tool_calls=(),
                usage=TokenUsage(input_tokens=300, output_tokens=90),
                model="openai/gpt-4o-mini",
            ),
            Completion(
                content=humanized_answer,
                tool_calls=(),
                usage=TokenUsage(input_tokens=420, output_tokens=85),
                model="openai/gpt-4o-mini",
            ),
        ]
    )

    result = TaskWorker(
        session_factory=worker_session_factory,
        worker_id="agent-worker-test",
        executor=AgentTaskExecutor(
            settings=make_settings(),
            llm_provider=provider,
            provider_name=LLMProvider.openrouter,
            web_search_tool=StaticWebSearchTool(),
            slack_client=slack_client,
            workspace_base_dir=tmp_path,
        ),
    ).run_once(now=claim_time)

    db_session.refresh(task)
    events = task_events(db_session, task)

    assert result.status == TaskStatus.succeeded.value, task.error
    assert slack_client.messages == [
        {
            "channel": "C123",
            "text": humanized_answer,
            "thread_ts": "EvAgentWorkerResearchHumanizer",
        }
    ]
    record_event = next(
        event
        for event in events
        if event.payload.get("message") == "response_record_built"
    )
    assert record_event.payload["response_mode"] == "research_summary"
    assert record_event.payload["response_shape"] == "research_brief"
    assert record_event.payload["action_count"] == 1
    assert record_event.payload["evidence_count"] == 1
    context_event = next(
        event
        for event in events
        if event.payload.get("message") == "synthesis_context_built"
    )
    assert context_event.payload["synthesis_outcome"] == "ok"
    assert context_event.payload["synthesis_evidence_count"] == 1
    assert context_event.payload["synthesis_evidence_kinds"] == ["tool_result"]
    assert context_event.payload["synthesis_evidence_trust"] == ["untrusted"]
    humanizer_payload = provider.calls[2][0][1].content
    assert humanizer_payload is not None
    humanizer_contract = json.loads(humanizer_payload)
    response_record = humanizer_contract["response_record"]
    synthesis_context = humanizer_contract["synthesis_context"]
    assert response_record["response_mode"] == "research_summary"
    assert response_record["response_shape"]["shape"] == "research_brief"
    assert synthesis_context["outcome"] == "ok"
    assert synthesis_context["evidence"][0]["tool"] == "web_search"
    assert synthesis_context["evidence"][0]["trust"] == "untrusted"
    assert "tempfile" in synthesis_context["evidence"][0]["content"]
    assert [skill["slug"] for skill in response_record["procedural_skills"]] == [
        "slack-humanizer",
        "research-synthesis",
    ]
    assert response_record["actions_taken"][0]["tool"] == "web_search"
    assert response_record["evidence"][0]["urls"] == [
        "https://docs.python.org/3/library/tempfile.html"
    ]


def test_agent_executor_builds_analyst_audit_response_record(
    db_session: Session,
    worker_session_factory: sessionmaker[Session],
    tmp_path: Path,
) -> None:
    claim_time = datetime(2026, 5, 23, 9, 52, tzinfo=UTC)
    task = create_task(db_session, event_id="EvAgentWorkerAnalystAudit")
    task.input = (
        "Review our website copy using the CPT framework. Point out the biggest "
        "gaps and what you would change first."
    )
    task.available_at = claim_time - timedelta(seconds=1)
    db_session.add(
        ModelPricing(
            provider=LLMProvider.openrouter,
            model="openai/gpt-4o-mini",
            input_price_per_mtok=Decimal("1.000000"),
            output_price_per_mtok=Decimal("2.000000"),
            effective_from=datetime(2026, 1, 1, tzinfo=UTC),
        )
    )
    db_session.commit()

    raw_answer = (
        "The site has several issues. It does not establish a clear loss "
        "reference point, it underplays drawdown protection, and the calls to "
        "action are framed as upside rather than risk reduction. The first "
        "thing I would change is the homepage hero so the advisor immediately "
        "sees the cost of staying with the default bond allocation."
    )
    humanized_answer = (
        "*Bottom line:* I would fix the homepage loss anchor first.\n\n"
        "*Scope:* based on the copy described in the prompt, not a fresh crawl.\n\n"
        "*Top gaps:*\n"
        "- The page does not establish the loss clients already feel.\n"
        "- Drawdown protection is buried instead of leading the story.\n"
        "- CTAs ask users to explore upside instead of reduce risk.\n\n"
        "*Highest-leverage move:* rewrite the hero around bond-risk loss."
    )
    slack_client = FakeSlackClient()
    provider = FakeAgentProvider(
        [
            Completion(
                content=raw_answer,
                tool_calls=(),
                usage=TokenUsage(input_tokens=260, output_tokens=85),
                model="openai/gpt-4o-mini",
            ),
            Completion(
                content=humanized_answer,
                tool_calls=(),
                usage=TokenUsage(input_tokens=580, output_tokens=140),
                model="openai/gpt-4o-mini",
            ),
        ]
    )

    result = TaskWorker(
        session_factory=worker_session_factory,
        worker_id="agent-worker-test",
        executor=AgentTaskExecutor(
            settings=make_settings(),
            llm_provider=provider,
            provider_name=LLMProvider.openrouter,
            web_search_tool=StaticWebSearchTool(),
            slack_client=slack_client,
            workspace_base_dir=tmp_path,
        ),
    ).run_once(now=claim_time)

    db_session.refresh(task)
    events = task_events(db_session, task)

    assert result.status == TaskStatus.succeeded.value, task.error
    assert slack_client.messages == [
        {
            "channel": "C123",
            "text": humanized_answer,
            "thread_ts": "EvAgentWorkerAnalystAudit",
        }
    ]
    record_event = next(
        event
        for event in events
        if event.payload.get("message") == "response_record_built"
    )
    assert record_event.payload["response_shape"] == "analyst_audit"
    assert record_event.payload["required_element_count"] == 8
    humanizer_payload = provider.calls[1][0][1].content
    assert humanizer_payload is not None
    response_record = json.loads(humanizer_payload)["response_record"]
    assert response_record["response_shape"]["shape"] == "analyst_audit"
    assert response_record["response_shape"]["framework_hint"] is not None
    assert response_record["response_shape"]["required_elements"] == [
        "bottom_line",
        "scope",
        "evaluation_lens",
        "ranked_findings",
        "evidence_or_limits",
        "concrete_recommendations",
        "highest_leverage_move",
        "next_step",
    ]
    assert [skill["slug"] for skill in response_record["procedural_skills"]] == [
        "slack-humanizer",
        "analyst-grade-synthesis",
    ]


def test_synthesis_context_marks_memory_no_result_from_tool_evidence(
    db_session: Session,
) -> None:
    task = create_task(db_session, event_id="EvMemoryNoResultContext")
    task.input = "forget my old report style preference"
    service = TaskService(db_session)
    service.append_event(
        task,
        TaskEventType.tool_call,
        {
            "tool_call_id": "call-forget-style",
            "tool": "forget_fact",
            "argument_keys": ["scope", "key"],
        },
    )
    service.append_event(
        task,
        TaskEventType.tool_result,
        {
            "tool_call_id": "call-forget-style",
            "tool": "forget_fact",
            "output": {
                "scope": "user",
                "scope_id": "U123",
                "key": "old_report_style_preference",
                "forgotten_count": 0,
                "message": "No active memory fact matched that scope and key.",
            },
            "artifact_count": 0,
        },
    )
    raw_answer = (
        "I checked what I remember and don't see that saved right now, "
        "so there is nothing for me to remove."
    )

    response_record = build_response_record(
        session=db_session,
        task=task,
        raw_text=raw_answer,
    )
    synthesis_context = build_synthesis_context(
        session=db_session,
        task=task,
        raw_text=raw_answer,
        response_record=response_record,
    )

    assert synthesis_context.outcome is SynthesisOutcome.no_result
    assert (
        synthesis_context.outcome_reason == "tool evidence reported no matching result"
    )
    assert synthesis_context.evidence[0].kind == "memory"
    assert synthesis_context.evidence[0].trust == "trusted"
    assert synthesis_context.evidence[0].metadata["forgotten_count"] == 0
    assert "Do not claim the requested item was found or changed." in (
        synthesis_context.forbidden_claims
    )


def test_agent_executor_removes_ack_reaction_after_success(
    db_session: Session,
    worker_session_factory: sessionmaker[Session],
    tmp_path: Path,
) -> None:
    claim_time = datetime(2026, 5, 23, 9, 51, tzinfo=UTC)
    task = create_task(db_session, event_id="EvAgentWorkerReaction")
    task.input = "summarize this thread"
    task.available_at = claim_time - timedelta(seconds=1)
    TaskService(db_session).append_event(
        task,
        TaskEventType.log,
        {
            "message": ACK_REACTION_ADDED_MESSAGE,
            "source": "app_mention",
            "channel": "C123",
            "message_ts": "EvAgentWorkerReaction",
            "reaction": "thinking_face",
            "reaction_intent": "review",
        },
    )
    db_session.add(
        ModelPricing(
            provider=LLMProvider.openrouter,
            model="openai/gpt-4o-mini",
            input_price_per_mtok=Decimal("1.000000"),
            output_price_per_mtok=Decimal("2.000000"),
            effective_from=datetime(2026, 1, 1, tzinfo=UTC),
        )
    )
    db_session.commit()

    slack_client = FakeSlackClient()
    result = TaskWorker(
        session_factory=worker_session_factory,
        worker_id="agent-worker-test",
        executor=AgentTaskExecutor(
            settings=make_settings(),
            llm_provider=FakeAgentProvider(
                [
                    Completion(
                        content="Here is the summary.",
                        tool_calls=(),
                        usage=TokenUsage(input_tokens=50, output_tokens=15),
                        model="openai/gpt-4o-mini",
                    ),
                ]
            ),
            provider_name=LLMProvider.openrouter,
            web_search_tool=StaticWebSearchTool(),
            slack_client=slack_client,
            workspace_base_dir=tmp_path,
        ),
    ).run_once(now=claim_time)

    db_session.refresh(task)
    events = task_events(db_session, task)

    assert result.status == TaskStatus.succeeded.value, task.error
    assert slack_client.reaction_removes == [
        {
            "channel": "C123",
            "name": "thinking_face",
            "timestamp": "EvAgentWorkerReaction",
        }
    ]
    assert slack_client.reaction_adds == []
    assert any(
        event.payload.get("message") == ACK_REACTION_REMOVED_MESSAGE for event in events
    )


def test_agent_executor_suppresses_final_message_after_memory_prompt(
    db_session: Session,
    worker_session_factory: sessionmaker[Session],
    tmp_path: Path,
) -> None:
    claim_time = datetime(2026, 5, 23, 9, 53, tzinfo=UTC)
    task = create_task(db_session, event_id="EvAgentWorkerMemory")
    task.input = "remember that I do not want PDFs unless I explicitly ask"
    task.available_at = claim_time - timedelta(seconds=1)
    db_session.add(
        ModelPricing(
            provider=LLMProvider.openrouter,
            model="openai/gpt-4o-mini",
            input_price_per_mtok=Decimal("1.000000"),
            output_price_per_mtok=Decimal("2.000000"),
            effective_from=datetime(2026, 1, 1, tzinfo=UTC),
        )
    )
    db_session.commit()

    slack_client = FakeSlackClient()
    provider = FakeAgentProvider(
        [
            Completion(
                content=None,
                tool_calls=(
                    ToolCall(
                        id="call-memory",
                        name="remember_fact",
                        arguments={
                            "scope": "user",
                            "key": "no_auto_pdfs",
                            "value": {
                                "preference": (
                                    "Do not generate PDFs unless explicitly requested"
                                )
                            },
                            "value_text": (
                                "Do not generate PDFs unless explicitly requested"
                            ),
                        },
                    ),
                ),
                usage=TokenUsage(input_tokens=100, output_tokens=40),
                model="openai/gpt-4o-mini",
            ),
            Completion(
                content="I've posted a confirmation to save this preference.",
                tool_calls=(),
                usage=TokenUsage(input_tokens=80, output_tokens=20),
                model="openai/gpt-4o-mini",
            ),
        ]
    )

    result = TaskWorker(
        session_factory=worker_session_factory,
        worker_id="agent-worker-test",
        executor=AgentTaskExecutor(
            settings=make_settings(),
            llm_provider=provider,
            provider_name=LLMProvider.openrouter,
            web_search_tool=StaticWebSearchTool(),
            slack_client=slack_client,
            artifact_comment_generator=FakeArtifactCommentGenerator(),
            workspace_base_dir=tmp_path,
        ),
    ).run_once(now=claim_time)

    db_session.refresh(task)
    events = task_events(db_session, task)
    posted_events = [
        event for event in events if event.type is TaskEventType.message_posted
    ]

    assert result.status == TaskStatus.succeeded.value, task.error
    assert task.result_summary == "I've posted a confirmation to save this preference."
    assert len(slack_client.messages) == 1
    assert slack_client.messages[0]["channel"] == "C123"
    assert slack_client.messages[0]["thread_ts"] == "EvAgentWorkerMemory"
    assert slack_client.messages[0]["text"] == (
        "Should I remember this for you?\n"
        "Do not generate PDFs unless explicitly requested\n\n"
        "React with :white_check_mark: to save it or :no_entry_sign: to skip."
    )
    assert posted_events[0].payload["purpose"] == "memory_confirmation"
    assert not any(event.payload.get("purpose") == "result" for event in posted_events)


def test_agent_executor_records_missing_brave_key_and_continues_without_web_search(
    db_session: Session,
    worker_session_factory: sessionmaker[Session],
    tmp_path: Path,
) -> None:
    claim_time = datetime(2026, 5, 23, 9, 55, tzinfo=UTC)
    task = create_task(db_session, event_id="EvAgentWorkerMissingSearch")
    task.input = "research Python tempfile and make a PDF"
    task.available_at = claim_time - timedelta(seconds=1)
    TaskService(db_session).append_event(
        task,
        TaskEventType.log,
        {
            "message": ACK_REACTION_ADDED_MESSAGE,
            "source": "app_mention",
            "channel": "C123",
            "message_ts": "EvAgentWorkerMissingSearch",
            "reaction": "mag",
            "reaction_intent": "discovery",
        },
    )
    db_session.add(
        ModelPricing(
            provider=LLMProvider.openrouter,
            model="openai/gpt-4o-mini",
            input_price_per_mtok=Decimal("1.000000"),
            output_price_per_mtok=Decimal("2.000000"),
            effective_from=datetime(2026, 1, 1, tzinfo=UTC),
        )
    )
    db_session.commit()

    slack_client = FakeSlackClient()
    result = TaskWorker(
        session_factory=worker_session_factory,
        worker_id="agent-worker-test",
        executor=AgentTaskExecutor(
            settings=make_settings(brave_search_api_key=""),
            llm_provider=FakeAgentProvider(
                [
                    Completion(
                        content="I don't have web search configured yet.",
                        tool_calls=(),
                        usage=TokenUsage(input_tokens=100, output_tokens=12),
                        model="openai/gpt-4o-mini",
                    )
                ]
            ),
            provider_name=LLMProvider.openrouter,
            slack_client=slack_client,
            workspace_base_dir=tmp_path,
        ),
    ).run_once(now=claim_time)

    db_session.refresh(task)
    events = task_events(db_session, task)

    assert result.status == TaskStatus.succeeded.value
    assert task.status is TaskStatus.succeeded
    assert task.error is None
    assert slack_client.messages == [
        {
            "channel": "C123",
            "text": "I don't have web search configured yet.",
            "thread_ts": "EvAgentWorkerMissingSearch",
        }
    ]
    assert slack_client.reaction_removes == [
        {
            "channel": "C123",
            "name": "mag",
            "timestamp": "EvAgentWorkerMissingSearch",
        }
    ]
    assert slack_client.reaction_adds == []
    unavailable_event = next(
        event
        for event in events
        if event.payload.get("message") == "native_tool_unavailable"
    )
    assert unavailable_event.payload["tool"] == "web_search"
    assert unavailable_event.payload["reason"] == "missing_brave_api_key"
    posted_event = next(
        event
        for event in events
        if event.type is TaskEventType.message_posted
        and event.payload.get("purpose") == "result"
    )
    assert posted_event.payload["text"] == "I don't have web search configured yet."


def test_agent_executor_records_channel_profile_when_assessment_completes(
    db_session: Session,
) -> None:
    task = create_task(db_session, event_id="EvChannelAssessment")
    task.slack_channel_id = "CObserve"
    membership = SlackChannelMembership(
        installation_id=task.installation_id,
        channel_id="CObserve",
        channel_name="observe",
        channel_type="public_channel",
        membership_status="active",
        discovered_via="member_joined_channel",
        added_by_user_id="UInvite",
        onboarding_status="posted",
        onboarding_message_ts="1779900000.000000",
        metadata_json={
            "assessment_task_id": str(task.id),
            "assessment_status": "queued",
        },
    )
    db_session.add(membership)
    db_session.flush()
    task_service = TaskService(db_session)
    task_service.append_event(
        task,
        TaskEventType.log,
        {
            "message": CHANNEL_ASSESSMENT_REQUESTED_MESSAGE,
            "source": "member_joined_channel",
            "channel_id": "CObserve",
            "membership_id": str(membership.id),
        },
    )
    task_service.append_event(
        task,
        TaskEventType.tool_result,
        {
            "tool": "slack_channel_history",
            "tool_call_id": "call-history",
            "output": {
                "channel_id": "CObserve",
                "message_count": 2,
                "messages": [
                    {
                        "ts": "1779900001.000001",
                        "user": "U1",
                        "text": "Daily report posted.",
                    },
                    {
                        "ts": "1779900002.000002",
                        "user": "U2",
                        "text": "Please review the attached file.",
                        "files": [{"id": "F1", "name": "report.pdf"}],
                    },
                ],
            },
            "cost_usd": "0",
            "artifacts": [],
        },
    )
    db_session.commit()

    AgentTaskExecutor()._mark_channel_assessment_completed(
        session=db_session,
        task=task,
        task_service=task_service,
        result_summary="This channel appears to handle daily report review.",
    )
    db_session.commit()

    db_session.refresh(membership)
    profile = db_session.scalar(
        select(ObserveChannelProfile).where(
            ObserveChannelProfile.installation_id == task.installation_id,
            ObserveChannelProfile.channel_id == "CObserve",
        )
    )
    completed_event = next(
        event
        for event in task_events(db_session, task)
        if event.payload.get("message") == CHANNEL_ASSESSMENT_COMPLETED_MESSAGE
    )

    assert membership.metadata_json["assessment_status"] == "posted"
    assert profile is not None
    assert profile.summary == "This channel appears to handle daily report review."
    assert profile.message_count == 2
    assert profile.file_count == 1
    assert profile.source_task_id == task.id
    assert completed_event.payload["profile_id"] == str(profile.id)
    assert completed_event.payload["profile_version"] == 1


def cleanup_database(session: Session) -> None:
    for model in (
        Episode,
        ObserveChannelProfile,
        SlackChannelMembership,
        Artifact,
        LLMUsage,
        TaskEvent,
        SlackSideEffect,
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


class FakeAdkContext:
    def __init__(self, *, agent_name: str, invocation_id: str, task_id: str) -> None:
        self.agent_name = agent_name
        self.invocation_id = invocation_id
        self.state = {"task_id": task_id}


class FakeAgentProvider:
    model = "openai/gpt-4o-mini"

    def __init__(self, completions: list[Completion]) -> None:
        self.completions = completions
        self.calls: list[tuple[tuple[ChatMessage, ...], tuple[JsonSchema, ...]]] = []

    def complete(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[JsonSchema] = (),
        *,
        response_format: JsonObject | None = None,
    ) -> Completion:
        del response_format
        self.calls.append((tuple(messages), tuple(tools)))
        if not self.completions:
            raise AssertionError("FakeAgentProvider received too many calls")
        return self.completions.pop(0)


class StaticWebSearchTool:
    name = "web_search"
    description = "Searches the public web and returns structured search results."
    parameters: JsonSchema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "count": {"type": "integer"},
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def invoke(self, args: JsonObject) -> ToolResult:
        return ToolResult(
            output={
                "provider": "test",
                "query": args["query"],
                "results": [
                    {
                        "title": "tempfile",
                        "url": "https://docs.python.org/3/library/tempfile.html",
                        "snippet": "Temporary file and directory helpers.",
                    }
                ],
            }
        )


class FakeArtifactCommentGenerator:
    def __init__(
        self,
        text: str = "Here's the report.",
        *,
        error: Exception | None = None,
    ) -> None:
        self.text = text
        self.error = error
        self.calls: list[tuple[str, str]] = []

    def generate(
        self,
        *,
        session: Session,
        task: Task,
        artifact: Artifact,
        task_service: TaskService,
    ) -> str:
        self.calls.append((task.input, artifact.filename))
        if self.error is not None:
            raise self.error
        return self.text


class FakeSlackClient:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []
        self.uploads: list[dict[str, Any]] = []
        self.reaction_adds: list[dict[str, Any]] = []
        self.reaction_removes: list[dict[str, Any]] = []

    def chat_postMessage(
        self,
        *,
        channel: str,
        text: str,
        thread_ts: str | None = None,
    ) -> dict[str, Any]:
        self.messages.append(
            {
                "channel": channel,
                "text": text,
                "thread_ts": thread_ts,
            }
        )
        return {"ok": True, "ts": f"1716400100.{len(self.messages):06d}"}

    def files_upload_v2(
        self,
        *,
        file: str,
        filename: str | None = None,
        title: str | None = None,
        channel: str | None = None,
        initial_comment: str | None = None,
        thread_ts: str | None = None,
    ) -> dict[str, Any]:
        self.uploads.append(
            {
                "file": file,
                "file_bytes": Path(file).read_bytes(),
                "filename": filename,
                "title": title,
                "channel": channel,
                "initial_comment": initial_comment,
                "thread_ts": thread_ts,
            }
        )
        return {"ok": True, "files": [{"id": f"F{len(self.uploads):06d}"}]}

    def reactions_add(
        self,
        *,
        channel: str,
        name: str,
        timestamp: str,
    ) -> dict[str, Any]:
        self.reaction_adds.append(
            {
                "channel": channel,
                "name": name,
                "timestamp": timestamp,
            }
        )
        return {"ok": True}

    def reactions_remove(
        self,
        *,
        channel: str,
        name: str,
        timestamp: str,
    ) -> dict[str, Any]:
        self.reaction_removes.append(
            {
                "channel": channel,
                "name": name,
                "timestamp": timestamp,
            }
        )
        return {"ok": True}


def task_events(session: Session, task: Task) -> list[TaskEvent]:
    return list(
        session.scalars(
            select(TaskEvent)
            .where(TaskEvent.task_id == task.id)
            .order_by(TaskEvent.seq)
        )
    )


def make_settings(brave_search_api_key: str = "brave-key") -> Settings:
    return Settings.model_validate(
        {
            "SLACK_BOT_TOKEN": "xoxb-test",
            "SLACK_APP_TOKEN": "xapp-test",
            "SLACK_SIGNING_SECRET": "signing-secret",
            "LLM_PROVIDER": SettingsLLMProvider.openrouter,
            "LLM_API_KEY": "openrouter-key",
            "LLM_MODEL": "openai/gpt-4o-mini",
            "BRAVE_SEARCH_API_KEY": brave_search_api_key,
            "POSTGRES_URL": "postgresql://kortny:kortny@localhost/kortny",
        }
    )
