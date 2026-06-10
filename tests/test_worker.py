import json
import os
import uuid
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import pytest
from alembic import command
from alembic.config import Config
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_response import LlmResponse
from google.genai import types as genai_types
from sqlalchemy import Engine, delete, func, select
from sqlalchemy.orm import Session, sessionmaker

from kortny.agent.adk_runtime import AdkAgentRuntime
from kortny.agent.coordinator import AgentRunResult
from kortny.approvals import (
    TOOL_APPROVAL_PROMPT_PURPOSE,
    TOOL_APPROVAL_REQUIRED_MESSAGE,
    TOOL_APPROVAL_WAITING_MESSAGE,
    ApprovalScope,
    ToolApprovalRequest,
    ToolApprovalRequired,
)
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
    ObservationEvent,
    ObserveChannelProfile,
    ProceduralSkillInvocation,
    Schedule,
    SlackChannelMembership,
    SlackSideEffect,
    Task,
    TaskEvent,
    TaskEventType,
    TaskStatus,
    WitnessOpportunityCandidate,
)
from kortny.db.session import make_engine, make_session_factory, normalize_database_url
from kortny.knowledge_graph import (
    KG_CHANNEL_PROFILE_PROJECTED_MESSAGE,
    KG_CHANNEL_REFRESH_HISTORY_LOADED_MESSAGE,
    KG_CHANNEL_REFRESH_PIPELINE_STARTED_MESSAGE,
    KG_CHANNEL_REFRESH_PROFILE_SYNTHESIZED_MESSAGE,
    KG_CHANNEL_REFRESH_SEMANTIC_EXTRACTED_MESSAGE,
    KG_CHANNEL_REFRESH_SEMANTIC_FALLBACK_MESSAGE,
    KG_REFRESH_SOURCE,
    KG_RUNTIME_CONTEXT_REINFORCED_MESSAGE,
    KG_TASK_SUMMARY_PROJECTED_MESSAGE,
    DestinationSurface,
    EvidenceInput,
    GraphService,
    KnowledgeGraphExtractionService,
    VisibilityScope,
)
from kortny.llm import ChatMessage, Completion, TokenUsage, ToolCall
from kortny.llm.routing import ModelRoute, ModelRouteTier
from kortny.observe.assessment import (
    CHANNEL_ASSESSMENT_COMPLETED_MESSAGE,
    CHANNEL_ASSESSMENT_REQUESTED_MESSAGE,
    CHANNEL_ASSESSMENT_SUPPRESS_SLACK_POST_KEY,
)
from kortny.slack.comments import ARTIFACT_COMMENT_FALLBACK_TEXT
from kortny.slack.humanizer import (
    ResponseSynthesisResult,
    StaticResponseSynthesizer,
    build_response_record,
    build_synthesis_context,
)
from kortny.slack.reactions import (
    ACK_REACTION_ADDED_MESSAGE,
    ACK_REACTION_REMOVED_MESSAGE,
)
from kortny.slack.synthesis import EvidenceKind, EvidenceTrust, SynthesisOutcome
from kortny.tasks import TaskService
from kortny.tools import ToolRegistry, ToolResult
from kortny.tools.types import JsonObject, JsonSchema
from kortny.witness import WITNESS_OPPORTUNITY_CANDIDATES_PROJECTED_MESSAGE
from kortny.worker import (
    AgentTaskExecutor,
    TaskExecutionResult,
    TaskWorker,
    WalkingSkeletonExecutor,
)
from kortny.workflow.launcher import TemporalWorkflowLaunch

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


def test_worker_synthesizes_approval_prompt_with_cheap_fast_model(
    db_session: Session,
) -> None:
    task = create_task(db_session, event_id="EvApprovalPromptSynthesis")
    task.input = "Can you verify the gross margin percentage with a quick code check?"
    task_service = TaskService(db_session)
    settings = make_settings()
    slack_client = FakeSlackClient()
    provider = FakeAgentProvider(
        [
            Completion(
                content=json.dumps(
                    {
                        "text": (
                            "I can verify that in a locked-down Python sandbox before "
                            "I run it.\n*Safety:* no network or host filesystem access."
                        )
                    }
                ),
                tool_calls=(),
                usage=TokenUsage(input_tokens=120, output_tokens=40),
                cost_usd=Decimal("0.0001"),
                model="openai/gpt-4o-mini",
            )
        ]
    )
    request = ToolApprovalRequest(
        approval_key="code_exec:abc123",
        tool_name="code_exec",
        tool_call_id="call-code",
        normalized_args_hash="abc123",
        argument_keys=("code", "language", "timeout_seconds"),
        scope=ApprovalScope.user,
        reason="code_exec can execute code in Kortny's isolated sandbox.",
        risk="sandboxed_code_execution",
        arguments={
            "code": "print((128.4 - 91.7) / 128.4 * 100)",
            "language": "python",
            "timeout_seconds": 5,
        },
    )

    AgentTaskExecutor(
        settings=settings,
        llm_provider=provider,
        provider_name=LLMProvider.openrouter,
        slack_client=slack_client,
    )._post_approval_request(
        settings=settings,
        session=db_session,
        task=task,
        task_service=task_service,
        approval=ToolApprovalRequired(request),
    )

    assert len(slack_client.messages) == 1
    posted_text = slack_client.messages[0]["text"]
    assert "locked-down Python sandbox" in posted_text
    assert "no network or host filesystem access" in posted_text
    assert "React with :white_check_mark:" in posted_text
    assert "code_exec" not in posted_text
    msg_content = provider.calls[0][0][1].content
    assert msg_content is not None
    assert "print((128.4 - 91.7)" not in msg_content
    assert provider.calls[0][2] == {"type": "json_object"}
    events = task_events(db_session, task)
    assert any(
        event.payload.get("message") == "tool_approval_prompt_synthesized"
        and event.payload.get("model_tier") == "cheap_fast"
        and event.payload.get("tool") == "code_exec"
        for event in events
    )
    assert any(
        event.type is TaskEventType.llm_call
        and event.payload.get("prompt_name") == "kortny.tool_approval_prompt"
        for event in events
    )


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

    # LiteLLM runtime identifiers carry the openrouter/ prefix (see
    # litellm_catalog model identifier normalization).
    assert llm.provider.model == "openrouter/anthropic/document-model"
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
            "KORTNY_PLANNED_WORKFLOW_PROGRESS_UPDATES_ENABLED": False,
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
        event.payload.get("message") == "runtime_handoff_evaluated"
        and event.payload.get("runtime_class") == "quick_response"
        and event.payload.get("selected_backend") == "inline"
        for event in events
    )
    assert any(
        event.payload.get("message") == "model_route_selected"
        and event.payload.get("runtime") == "adk"
        and event.payload.get("tier") == "cheap_fast"
        and event.payload.get("model") == "deepseek/deepseek-v4-flash"
        for event in events
    )


def test_agent_executor_memoizes_adk_registry_factory_per_execution(
    db_session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = create_task(db_session, event_id="EvAdkMemoizedRegistry")
    task.input = "research AI observability, check Linear, and summarize"
    task_service = TaskService(db_session)
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
    build_count = 0
    captured: dict[str, Any] = {}

    def fake_build_registry(self: AgentTaskExecutor, **kwargs: Any) -> ToolRegistry:
        del self, kwargs
        nonlocal build_count
        build_count += 1
        return ToolRegistry([StaticWebSearchTool()])

    class FakeAdkRuntime:
        def __init__(self, **kwargs: Any) -> None:
            registry_factory = kwargs["registry_factory"]
            first = registry_factory()
            second = registry_factory()
            third = registry_factory()
            captured["same_registry"] = first is second is third
            captured["tool_names"] = list(first.names())

        def run(self, task_arg: Task) -> AgentRunResult:
            return AgentRunResult(
                task_id=task_arg.id,
                result_summary="Done.",
                turns=1,
                artifact_count=0,
            )

    monkeypatch.setattr(AgentTaskExecutor, "_build_registry", fake_build_registry)
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

    assert result.result_summary == "Done."
    assert build_count == 1
    assert captured["same_registry"] is True
    assert captured["tool_names"] == ["web_search"]


def test_agent_executor_records_planned_workflow_classifier_event_for_complex_task(
    db_session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = create_task(db_session, event_id="EvPlannedWorkflowClassified")
    task.input = "Research best AI agents for trading and summarize the options."
    db_session.commit()
    task_service = TaskService(db_session)
    settings = Settings.model_validate(
        {
            "SLACK_BOT_TOKEN": "xoxb-test",
            "SLACK_APP_TOKEN": "xapp-test",
            "SLACK_SIGNING_SECRET": "signing-secret",
            "LLM_PROVIDER": SettingsLLMProvider.openrouter,
            "LLM_API_KEY": "openrouter-key",
            "LLM_MODEL": "anthropic/sonnet-default",
            "LLM_CHEAP_MODEL": "deepseek/deepseek-v4-flash",
            "LLM_ANALYSIS_MODEL": "anthropic/claude-sonnet-4.6",
            "POSTGRES_URL": "postgresql://kortny:kortny@localhost/kortny",
            "AGENT_RUNTIME": "adk",
            "KORTNY_PLANNED_WORKFLOW_PROGRESS_UPDATES_ENABLED": False,
        }
    )

    class FakeAdkRuntime:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def run(self, task_arg: Task) -> AgentRunResult:
            return AgentRunResult(
                task_id=task_arg.id,
                result_summary="Complex task still executed by the current runtime.",
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

    assert (
        result.result_summary == "Complex task still executed by the current runtime."
    )
    assert any(
        event.payload.get("message") == "planned_workflow_classified"
        and event.payload.get("behavior") == "observe_only"
        and event.payload.get("route") == "planned_candidate"
        and event.payload.get("planned_candidate") is True
        and (event.payload.get("estimated_subtask_count") or 0) >= 3
        for event in events
    )
    assert any(
        event.payload.get("message") == "runtime_handoff_evaluated"
        and event.payload.get("selected_backend") == "inline"
        for event in events
    )
    route_events = [
        event.payload
        for event in events
        if event.payload.get("message") == "routing_decision_recorded"
    ]
    assert any(
        payload.get("stage") == "worker_runtime_handoff"
        and payload.get("runtime_class") == "inline_tool_task"
        and payload.get("shadow_route") == "planned_candidate"
        and payload.get("shadow_planned_candidate") is True
        for payload in route_events
    )
    assert any(
        payload.get("stage") == "worker_runtime_selected"
        and payload.get("selected_runtime") == "adk"
        and payload.get("actual_path") == "adk"
        for payload in route_events
    )


def test_agent_executor_answers_schedule_state_question_with_fast_path(
    db_session: Session,
) -> None:
    task = create_task(db_session, event_id="EvScheduleStateFastPath")
    task.input = "Do I have an active stock market update scheduled?"
    task.slack_channel_id = "D123"
    task.slack_thread_ts = "D123"
    task.slack_message_ts = "1780764480.613429"
    task.slack_user_id = "U123"
    schedule = Schedule(
        installation_id=task.installation_id,
        owner_type="user",
        owner_slack_user_id="U123",
        title="Daily stock market update",
        spec_kind="cron",
        cron_expr="0 8 * * *",
        timezone="America/Chicago",
        next_run_at=datetime(2026, 6, 7, 13, 0, tzinfo=UTC),
        catchup_policy="skip",
        catchup_window_seconds=300,
        overlap_policy="skip",
        status="active",
        delivery_kind="slack_dm",
        delivery_slack_user_id="U123",
        delivery_slack_channel_id="D123",
        delivery_slack_thread_ts="D123",
        artifact_delivery_policy="message_only",
        task_template={
            "input": "send me a stock market update",
            "slack_user_id": "U123",
            "slack_channel_id": "D123",
            "slack_thread_ts": "D123",
            "delivery_surface": "dm",
        },
        planned_cost_ceiling_usd=Decimal("0.2500"),
        created_by_slack_user_id="U123",
        metadata_json={"cadence_label": "Every morning at 8:00 AM Central"},
    )
    db_session.add(schedule)
    db_session.commit()
    task_service = TaskService(db_session)
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
            "KORTNY_WORKFLOW_BACKEND": "temporal",
            "RESPONSE_HUMANIZER_ENABLED": True,
        }
    )
    slack_client = FakeSlackClient()
    humanized_answer = (
        "Yup! I have a stock market update set for you every morning at "
        "8 AM Central. The next one comes your way tomorrow."
    )

    class RecordingSynthesizer:
        uses_procedural_skills = False

        def __init__(self) -> None:
            self.raw_answers: list[str] = []

        def synthesize(
            self,
            *,
            session: Session,
            task: Task,
            response_record: Any,
            synthesis_context: Any,
            task_service: TaskService,
        ) -> ResponseSynthesisResult:
            del session, task, synthesis_context, task_service
            self.raw_answers.append(response_record.raw_answer)
            return ResponseSynthesisResult(
                text=humanized_answer,
                changed=True,
                reason="recording_humanizer",
            )

    synthesizer = RecordingSynthesizer()

    result = AgentTaskExecutor(
        settings=settings,
        llm_provider=FakeAgentProvider([]),
        slack_client=slack_client,
        response_synthesizer=synthesizer,
    ).execute(
        session=db_session,
        task=task,
        task_service=task_service,
    )

    events = task_events(db_session, task)
    event_messages = [event.payload.get("message") for event in events]

    assert result.result_summary.startswith("Yes")
    assert slack_client.messages[-1]["text"] == humanized_answer
    assert "Daily stock market update" in synthesizer.raw_answers[0]
    assert "Scheduler DB is the source of truth here" in synthesizer.raw_answers[0]
    assert "schedule_state_fast_path_completed" in event_messages
    assert "routing_decision_recorded" in event_messages
    assert "routing_chain_completed" in event_messages
    assert "response_humanizer_started" in event_messages
    assert "response_humanizer_completed" in event_messages
    assert "response_humanizer_skipped" not in event_messages
    assert "planned_workflow_classified" not in event_messages
    assert "runtime_handoff_evaluated" not in event_messages
    assert "adk_runtime_started" not in event_messages
    assert "witness_opportunity_candidates_projected" not in event_messages
    route_event = next(
        event
        for event in events
        if event.payload.get("message") == "routing_decision_recorded"
    )
    assert route_event.payload["route_tier_resolved"] == "tier0"
    assert route_event.payload["runtime_class"] == "inline_tool_task"
    assert route_event.payload["intent"] == "scheduler.query"
    assert route_event.payload["actual_path"] == "schedule_state_fast_path"
    completed_route = next(
        event
        for event in events
        if event.payload.get("message") == "routing_chain_completed"
    )
    assert completed_route.payload["selected_runtime"] == "schedule_state_fast_path"
    assert completed_route.payload["final_actual_path"] == "schedule_state_fast_path"
    assert completed_route.payload["final_intent"] == "scheduler.query"


def test_agent_executor_posts_planned_workflow_progress_update(
    db_session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = create_task(db_session, event_id="EvPlannedWorkflowProgress")
    task.input = "Research best AI agents for trading and summarize the options."
    db_session.commit()
    task_service = TaskService(db_session)
    settings = Settings.model_validate(
        {
            "SLACK_BOT_TOKEN": "xoxb-test",
            "SLACK_APP_TOKEN": "xapp-test",
            "SLACK_SIGNING_SECRET": "signing-secret",
            "LLM_PROVIDER": SettingsLLMProvider.openrouter,
            "LLM_API_KEY": "openrouter-key",
            "LLM_MODEL": "anthropic/sonnet-default",
            "LLM_CHEAP_MODEL": "deepseek/deepseek-v4-flash",
            "LLM_ANALYSIS_MODEL": "anthropic/claude-sonnet-4.6",
            "POSTGRES_URL": "postgresql://kortny:kortny@localhost/kortny",
            "AGENT_RUNTIME": "adk",
            "KORTNY_PLANNED_WORKFLOW_PROGRESS_UPDATES_ENABLED": True,
        }
    )
    slack_client = FakeSlackClient()
    provider = FakeAgentProvider(
        [
            Completion(
                content=json.dumps(
                    {
                        "message": (
                            "I'll check the channel pattern and recent context, "
                            "then call out what I would watch for."
                        )
                    }
                ),
                tool_calls=(),
                usage=TokenUsage(input_tokens=90, output_tokens=24),
                cost_usd=Decimal("0.000001"),
                model="deepseek/deepseek-v4-flash-20260423",
            )
        ]
    )

    class FakeAdkRuntime:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def run(self, task_arg: Task) -> AgentRunResult:
            return AgentRunResult(
                task_id=task_arg.id,
                result_summary="Complex task still executed by the current runtime.",
                turns=1,
                artifact_count=0,
            )

    monkeypatch.setattr(
        "kortny.agent.adk_runtime.AdkAgentRuntime",
        FakeAdkRuntime,
    )

    result = AgentTaskExecutor(
        settings=settings,
        llm_provider=provider,
        provider_name="openrouter",
        slack_client=slack_client,
    )._run_agent_runtime(
        settings=settings,
        session=db_session,
        task=task,
        task_service=task_service,
        working_dir=tmp_path,
    )

    events = task_events(db_session, task)

    assert (
        result.result_summary == "Complex task still executed by the current runtime."
    )
    assert len(slack_client.messages) == 1
    assert slack_client.messages[0]["channel"] == "C123"
    assert slack_client.messages[0]["thread_ts"] == "EvPlannedWorkflowProgress"
    assert "channel pattern and recent context" in slack_client.messages[0]["text"]
    assert "workstreams" not in slack_client.messages[0]["text"].casefold()
    assert len(provider.calls) == 1
    assert provider.calls[0][2] == {"type": "json_object"}
    assert any(
        event.payload.get("message") == "planned_task_started"
        and event.payload.get("progress_updates_enabled") is True
        for event in events
    )
    assert any(
        event.type is TaskEventType.message_posted
        and event.payload.get("purpose") == "planned_progress_start"
        for event in events
    )
    assert any(
        event.payload.get("message") == "planned_task_progress_posted"
        and event.payload.get("purpose") == "planned_progress_start"
        and event.payload.get("text_source") == "llm"
        for event in events
    )
    assert any(
        event.payload.get("message") == "llm_call_started"
        and event.payload.get("prompt_name") == "kortny.planned_progress_status"
        and event.payload.get("model_tier") == "cheap_fast"
        for event in events
    )
    usage = db_session.scalar(select(LLMUsage).where(LLMUsage.task_id == task.id))
    assert usage is not None
    assert usage.model_tier == "cheap_fast"
    assert usage.model == "deepseek/deepseek-v4-flash-20260423"


def test_agent_executor_shadow_starts_temporal_for_planned_candidate(
    db_session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = create_task(db_session, event_id="EvPlannedTemporalShadow")
    task.input = "Research best AI agents for trading and summarize the options."
    db_session.commit()
    task_service = TaskService(db_session)
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
            "KORTNY_WORKFLOW_BACKEND": "temporal",
            "KORTNY_PLANNED_WORKFLOW_PROGRESS_UPDATES_ENABLED": False,
        }
    )
    launch_calls: list[str] = []

    def fake_start_temporal_task_workflow_sync(
        *,
        settings: Settings,
        task: Task,
    ) -> TemporalWorkflowLaunch:
        launch_calls.append(str(task.id))
        return TemporalWorkflowLaunch(
            workflow_id=f"kortny-task-{task.id}",
            run_id="run-planned-1",
            first_execution_run_id="first-run-planned-1",
            result_run_id=None,
            namespace=settings.temporal_namespace,
            task_queue=settings.temporal_task_queue,
        )

    class FakeAdkRuntime:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def run(self, task_arg: Task) -> AgentRunResult:
            return AgentRunResult(
                task_id=task_arg.id,
                result_summary="Planned task still answered by the current runtime.",
                turns=1,
                artifact_count=0,
            )

    monkeypatch.setattr(
        "kortny.workflow.launcher.start_temporal_task_workflow_sync",
        fake_start_temporal_task_workflow_sync,
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

    assert (
        result.result_summary == "Planned task still answered by the current runtime."
    )
    assert launch_calls == [str(task.id)]
    assert any(
        event.payload.get("message") == "planned_workflow_classified"
        and event.payload.get("planned_candidate") is True
        for event in events
    )
    assert any(
        event.payload.get("message") == "runtime_handoff_evaluated"
        and event.payload.get("runtime_class") == "inline_tool_task"
        for event in events
    )
    assert any(
        event.payload.get("message") == "temporal_workflow_shadow_started"
        and event.payload.get("mode") == "shadow"
        and event.payload.get("run_id") == "run-planned-1"
        for event in events
    )


def test_agent_executor_shadow_starts_temporal_workflow_for_durable_candidate(
    db_session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = create_task(db_session, event_id="EvTemporalShadow")
    task.input = (
        "Research AI observability tools, check Linear and Notion, compare with "
        "our docs, and recommend next actions."
    )
    db_session.commit()
    task_service = TaskService(db_session)
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
            "KORTNY_WORKFLOW_BACKEND": "temporal",
            "KORTNY_PLANNED_WORKFLOW_PROGRESS_UPDATES_ENABLED": False,
        }
    )
    launch_calls: list[str] = []

    def fake_start_temporal_task_workflow_sync(
        *,
        settings: Settings,
        task: Task,
    ) -> TemporalWorkflowLaunch:
        launch_calls.append(str(task.id))
        return TemporalWorkflowLaunch(
            workflow_id=f"kortny-task-{task.id}",
            run_id="run-1",
            first_execution_run_id="first-run-1",
            result_run_id=None,
            namespace=settings.temporal_namespace,
            task_queue=settings.temporal_task_queue,
        )

    class FakeAdkRuntime:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def run(self, task_arg: Task) -> AgentRunResult:
            return AgentRunResult(
                task_id=task_arg.id,
                result_summary="Durable task still answered inline.",
                turns=1,
                artifact_count=0,
            )

    monkeypatch.setattr(
        "kortny.workflow.launcher.start_temporal_task_workflow_sync",
        fake_start_temporal_task_workflow_sync,
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

    assert result.result_summary == "Durable task still answered inline."
    assert launch_calls == [str(task.id)]
    assert any(
        event.payload.get("message") == "runtime_handoff_evaluated"
        and event.payload.get("runtime_class") == "durable_workflow_task"
        and event.payload.get("configured_backend") == "temporal"
        and event.payload.get("selected_backend") == "inline"
        and event.payload.get("fallback_reason")
        == "temporal_primary_execution_not_enabled"
        for event in events
    )
    assert any(
        event.payload.get("message") == "temporal_workflow_shadow_started"
        and event.payload.get("mode") == "shadow"
        and event.payload.get("workflow_id") == f"kortny-task-{task.id}"
        and event.payload.get("run_id") == "run-1"
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
        model_version="openrouter/anthropic/claude-4.6-sonnet-20260217",
        usage_metadata=genai_types.GenerateContentResponseUsageMetadata(
            prompt_token_count=100,
            tool_use_prompt_token_count=10,
            candidates_token_count=20,
            total_token_count=130,
        ),
    )

    runtime._record_adk_model_usage(
        callback_context=cast(CallbackContext, context), llm_response=response
    )
    usage = db_session.scalar(select(LLMUsage).where(LLMUsage.task_id == task.id))

    assert usage is not None
    assert usage.provider is LLMProvider.openrouter
    assert usage.model == "anthropic/claude-4.6-sonnet-20260217"
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


def test_adk_model_callback_records_planned_workflow_cost_ceiling_event(
    db_session: Session,
) -> None:
    task = create_task(db_session, event_id="EvAdkPlannedCostCeiling")
    task_service = TaskService(db_session)
    task_service.append_event(
        task,
        TaskEventType.log,
        {
            "message": "planned_workflow_classified",
            "route": "planned_candidate",
            "planned_candidate": True,
            "confidence": 0.9,
        },
    )
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
            "KORTNY_PLANNED_WORKFLOW_COST_CEILING_USD": "0.0001",
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
        agent_name="planned_workflow_planner",
        invocation_id="adk-invocation-planned-budget",
        task_id=str(task.id),
    )
    response = LlmResponse(
        model_version="openrouter/anthropic/claude-4.6-sonnet-20260217",
        usage_metadata=genai_types.GenerateContentResponseUsageMetadata(
            prompt_token_count=100,
            candidates_token_count=20,
            total_token_count=120,
        ),
    )

    runtime._record_adk_model_usage(
        callback_context=cast(CallbackContext, context), llm_response=response
    )
    events = task_events(db_session, task)

    assert any(
        event.payload.get("message") == "planned_workflow_cost_ceiling_exceeded"
        and event.payload.get("runtime") == "adk"
        and event.payload.get("behavior") == "observe_only"
        and event.payload.get("cost_ceiling_usd") == "0.0001"
        for event in events
    )


def test_adk_model_callback_uses_openrouter_catalog_pricing_when_litellm_misses(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = create_task(db_session, event_id="EvAdkOpenRouterCatalogUsage")
    task_service = TaskService(db_session)
    monkeypatch.setattr(
        "kortny.agent.adk_runtime._openrouter_model_pricing_catalog",
        lambda: {
            "deepseek/deepseek-v4-pro": (
                Decimal("0.000000435"),
                Decimal("0.00000087"),
            ),
            "deepseek/deepseek-v4-pro-20260423": (
                Decimal("0.000000435"),
                Decimal("0.00000087"),
            ),
        },
    )
    settings = Settings.model_validate(
        {
            "SLACK_BOT_TOKEN": "xoxb-test",
            "SLACK_APP_TOKEN": "xapp-test",
            "SLACK_SIGNING_SECRET": "signing-secret",
            "LLM_PROVIDER": SettingsLLMProvider.openrouter,
            "LLM_API_KEY": "openrouter-key",
            "LLM_MODEL": "deepseek/deepseek-v4-pro",
            "POSTGRES_URL": "postgresql://kortny:kortny@localhost/kortny",
            "AGENT_RUNTIME": "adk",
        }
    )
    runtime = AdkAgentRuntime(
        settings=settings,
        session=db_session,
        task_service=task_service,
        model_route=ModelRoute(
            tier=ModelRouteTier.standard,
            model="deepseek/deepseek-v4-pro",
            reason="intent_classifier",
        ),
    )
    context = FakeAdkContext(
        agent_name="kortny_root_orchestrator",
        invocation_id="adk-invocation-catalog",
        task_id=str(task.id),
    )
    response = LlmResponse(
        model_version="openrouter/deepseek/deepseek-v4-pro-20260423",
        usage_metadata=genai_types.GenerateContentResponseUsageMetadata(
            prompt_token_count=4238,
            candidates_token_count=343,
            total_token_count=4581,
        ),
    )

    runtime._record_adk_model_usage(
        callback_context=cast(CallbackContext, context), llm_response=response
    )
    usage = db_session.scalar(select(LLMUsage).where(LLMUsage.task_id == task.id))

    assert usage is not None
    assert usage.model == "deepseek/deepseek-v4-pro-20260423"
    assert usage.cost_usd == Decimal("0.002142")
    db_session.refresh(task)
    assert task.total_cost_usd == Decimal("0.002142")
    event = next(
        event
        for event in task_events(db_session, task)
        if event.payload.get("prompt_name") == "kortny.adk.kortny_root_orchestrator"
    )
    assert event.payload["pricing_missing"] is False
    assert event.payload["cost_usd"] == "0.002142"


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


def test_agent_executor_sanitizes_adk_quick_scratchpad_before_skip(
    db_session: Session,
) -> None:
    task = create_task(db_session, event_id="EvAdkQuickScratchpad")
    task.input = "are you up?"
    task_service = TaskService(db_session)
    task_service.append_event(
        task,
        TaskEventType.log,
        {
            "message": "model_route_selected",
            "runtime": "adk",
            "tier": "cheap_fast",
            "model": "qwen/qwen3.5-flash-02-23",
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
            "result_chars": 260,
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
            "LLM_CHEAP_MODEL": "qwen/qwen3.5-flash-02-23",
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
        result_summary=(
            "The user is asking if I'm up, which is a simple availability check. "
            "According to my guidelines, I should avoid internal routing details.\n\n"
            "I'll keep it brief and natural.\n"
            "Yep, I'm up and ready to help."
        ),
    )

    events = task_events(db_session, task)

    assert slack_client.messages[-1]["text"] == "Yep, I'm up and ready to help."
    assert any(
        event.payload.get("message") == "final_response_sanitized" for event in events
    )
    assert any(
        event.payload.get("message") == "response_humanizer_skipped"
        and event.payload.get("reason") == "adk_quick_fast_path"
        for event in events
    )
    assert not any(
        event.payload.get("message") == "response_humanizer_started" for event in events
    )


def test_agent_executor_humanizes_adk_planned_merger_final(
    db_session: Session,
) -> None:
    task = create_task(db_session, event_id="EvAdkPlannedHumanizer")
    task.input = "research AI observability tools, check Linear, and summarize"
    task_service = TaskService(db_session)
    task_service.append_event(
        task,
        TaskEventType.log,
        {
            "message": "model_route_selected",
            "runtime": "adk",
            "tier": "analysis",
            "model": "anthropic/claude-sonnet-4.6",
            "reason": "planned_workflow",
        },
    )
    task_service.append_event(
        task,
        TaskEventType.log,
        {
            "message": "adk_runtime_completed",
            "runtime": "adk",
            "mode": "planned_parallel",
            "event_count": 8,
            "final_author": "planned_workflow_merger",
            "authors": [
                "planned_workflow_planner",
                "planned_research_worker",
                "planned_workspace_worker",
                "planned_integration_worker",
                "planned_workflow_merger",
            ],
            "result_chars": 183,
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
    raw_answer = (
        'The user said "research AI observability tools" and provided branch '
        "context. I am the planned_workflow_merger, and my job is to merge "
        "branch outputs into one Slack-native answer.\n\n"
        "I'll present this as Kortny's final answer.\n"
        "*Bottom line:* record Langfuse as the default observability candidate "
        "and track Phoenix as the evaluation path. Capture the decision in "
        "Linear, keep pricing/current docs as follow-up evidence, and avoid "
        "creating a PDF unless the user explicitly asks for one."
    )
    humanized_answer = (
        "*Bottom line:* record Langfuse as the default observability candidate "
        "and track Phoenix as the evaluation path."
    )

    class RecordingSynthesizer:
        uses_procedural_skills = False

        def __init__(self) -> None:
            self.raw_answers: list[str] = []

        def synthesize(
            self,
            *,
            session: Session,
            task: Task,
            response_record: Any,
            synthesis_context: Any,
            task_service: TaskService,
        ) -> ResponseSynthesisResult:
            del session, task, synthesis_context, task_service
            self.raw_answers.append(response_record.raw_answer)
            return ResponseSynthesisResult(
                text=humanized_answer,
                changed=True,
                reason="recording_humanizer",
            )

    synthesizer = RecordingSynthesizer()

    AgentTaskExecutor(
        settings=settings,
        response_synthesizer=synthesizer,
        slack_client=slack_client,
    )._post_outputs(
        settings=settings,
        session=db_session,
        task=task,
        task_service=task_service,
        result_summary=raw_answer,
    )

    events = task_events(db_session, task)

    assert slack_client.messages[-1]["text"] == humanized_answer
    assert any(
        event.payload.get("message") == "final_response_sanitized"
        and event.payload.get("reason") == "internal_preamble_removed"
        for event in events
    )
    assert any(
        event.payload.get("message") == "response_humanizer_started" for event in events
    )
    assert any(
        event.payload.get("message") == "response_humanizer_completed"
        for event in events
    )
    assert not any(
        event.payload.get("message") == "response_humanizer_skipped"
        and event.payload.get("reason") == "adk_planned_merger_final"
        for event in events
    )
    assert synthesizer.raw_answers == [
        (
            "*Bottom line:* record Langfuse as the default observability "
            "candidate and track Phoenix as the evaluation path. Capture the "
            "decision in Linear, keep pricing/current docs as follow-up "
            "evidence, and avoid creating a PDF unless the user explicitly "
            "asks for one."
        )
    ]
    assert "planned_workflow_merger" not in synthesizer.raw_answers[0]


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
    humanizer_llm_event = next(
        event
        for event in events
        if event.type is TaskEventType.llm_call
        and event.payload.get("prompt_name") == "kortny.response_humanizer"
    )
    assert humanizer_llm_event.payload["model_tier"] == "humanizer"
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


def test_synthesis_context_uses_graph_profile_despite_memory_no_match(
    db_session: Session,
) -> None:
    task = create_task(db_session, event_id="EvGraphProfileWithMemoryMiss")
    task.input = "what do you know about how this channel is used, and why?"
    service = TaskService(db_session)
    service.append_event(
        task,
        TaskEventType.tool_call,
        {
            "tool_call_id": "call-graph-profile",
            "tool": "query_workspace_graph",
            "argument_keys": ["anchor_keys", "include_evidence"],
        },
    )
    service.append_event(
        task,
        TaskEventType.tool_result,
        {
            "tool_call_id": "call-graph-profile",
            "tool": "query_workspace_graph",
            "output": {
                "successful": True,
                "destination": {
                    "surface_type": "private_channel",
                    "surface_id": "C123",
                    "user_id": None,
                },
                "entity_count": 2,
                "edge_count": 1,
                "omitted_count": 0,
                "omitted_reasons": [],
                "entities": [
                    {
                        "canonical_key": "slack_channel:C123",
                        "display_name": "#rag",
                        "entity_type": "channel",
                        "confidence_score": "0.900",
                        "evidence_count": 1,
                    },
                    {
                        "canonical_key": "channel_profile:C123",
                        "display_name": "AI assistant testing and project management",
                        "entity_type": "firm_fact",
                        "confidence_score": "0.700",
                        "evidence_count": 4,
                    },
                ],
                "relationships": [
                    {
                        "source_label": "#rag",
                        "target_label": "AI assistant testing and project management",
                        "relationship_type": "relates_to",
                        "confidence_score": "0.700",
                    }
                ],
            },
            "artifact_count": 0,
        },
    )
    service.append_event(
        task,
        TaskEventType.tool_call,
        {
            "tool_call_id": "call-recall-channel-note",
            "tool": "recall_fact",
            "argument_keys": ["key"],
        },
    )
    service.append_event(
        task,
        TaskEventType.tool_result,
        {
            "tool_call_id": "call-recall-channel-note",
            "tool": "recall_fact",
            "output": {
                "key": "channel_usage_summary",
                "found": False,
            },
            "artifact_count": 0,
        },
    )
    raw_answer = (
        "This channel looks like an AI assistant testing and project management "
        "space. I believe that because the channel profile and recent messages "
        "point to Linear task lookups, tool comparisons, and policy decisions."
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

    assert response_record.response_mode.value == "context_answer"
    assert response_record.response_shape.shape.value == "context_profile"
    assert synthesis_context.outcome is SynthesisOutcome.ok
    assert synthesis_context.outcome_reason == "task completed with user-facing answer"
    assert synthesis_context.evidence[0].kind is EvidenceKind.workspace_graph
    assert synthesis_context.evidence[0].trust is EvidenceTrust.trusted
    assert "AI assistant testing" in synthesis_context.evidence[0].content
    assert "No active memory fact" in synthesis_context.evidence[1].content
    assert "Do not claim the requested item was found or changed." not in (
        synthesis_context.forbidden_claims
    )


def test_agent_executor_reinforces_graph_rows_used_in_delivered_answer(
    db_session: Session,
    worker_session_factory: sessionmaker[Session],
    tmp_path: Path,
) -> None:
    claim_time = datetime(2026, 6, 3, 10, 15, tzinfo=UTC)
    task = create_task(db_session, event_id="EvRuntimeGraphReinforcement")
    task.input = "what do you know about how this channel is used, and why?"
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
    graph = GraphService(db_session)
    channel = graph.create_entity(
        installation_id=task.installation_id,
        entity_type="channel",
        canonical_key="slack_channel:C123",
        display_name="#rag",
        visibility_scope=VisibilityScope.channel("C123"),
        source_type="slack_authoritative",
        lifecycle_state="active",
        confidence_score=Decimal("1.000"),
        confidence_reason="Slack channel membership is authoritative.",
        evidence=EvidenceInput(
            source_type="slack_authoritative",
            extracted_by="test",
            source_slack_channel_id="C123",
            raw_snippet="Kortny is active in #rag.",
            confidence_score=Decimal("1.000"),
        ),
    )
    profile = graph.create_entity(
        installation_id=task.installation_id,
        entity_type="firm_fact",
        canonical_key="channel_profile:C123",
        display_name="AI assistant testing and Linear task management",
        visibility_scope=VisibilityScope.channel("C123"),
        source_type="onboarding_scan",
        lifecycle_state="active",
        confidence_score=Decimal("0.750"),
        confidence_reason="Repeated channel assessment sample.",
        evidence=EvidenceInput(
            source_type="onboarding_scan",
            extracted_by="test",
            source_task_id=task.id,
            raw_snippet="The channel is used for AI assistant testing and Linear lookups.",
            confidence_score=Decimal("0.750"),
        ),
    )
    relation = graph.create_edge(
        installation_id=task.installation_id,
        source_entity_id=channel.id,
        target_entity_id=profile.id,
        relationship_type="relates_to",
        visibility_scope=VisibilityScope.channel("C123"),
        source_type="onboarding_scan",
        lifecycle_state="active",
        confidence_score=Decimal("0.750"),
        confidence_reason="Channel profile projection.",
        evidence=EvidenceInput(
            source_type="onboarding_scan",
            extracted_by="test",
            source_task_id=task.id,
            raw_snippet="#rag relates to AI assistant testing and Linear task work.",
            confidence_score=Decimal("0.750"),
        ),
    )
    db_session.commit()

    raw_answer = (
        "This channel looks like a Kortny testing and task-management workspace. "
        "I believe that because the graph profile says it is used for AI "
        "assistant testing and Linear task management, and the channel-profile "
        "relationship ties that profile directly to #rag."
    )
    humanized_answer = (
        "This channel reads like a Kortny testing and task-management space.\n\n"
        "Why I believe that: I have a channel profile tied to #rag that points "
        "to AI assistant testing and Linear task-management work."
    )
    slack_client = FakeSlackClient()
    provider = FakeAgentProvider(
        [
            Completion(
                content=None,
                tool_calls=(
                    ToolCall(
                        id="call-graph",
                        name="query_workspace_graph",
                        arguments={
                            "anchor_keys": ["slack_channel:C123"],
                            "include_evidence": True,
                            "max_hops": 1,
                            "limit": 10,
                        },
                    ),
                ),
                usage=TokenUsage(input_tokens=260, output_tokens=45),
                model="openai/gpt-4o-mini",
            ),
            Completion(
                content=raw_answer,
                tool_calls=(),
                usage=TokenUsage(input_tokens=520, output_tokens=80),
                model="openai/gpt-4o-mini",
            ),
            Completion(
                content=humanized_answer,
                tool_calls=(),
                usage=TokenUsage(input_tokens=640, output_tokens=90),
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
            slack_client=slack_client,
            workspace_base_dir=tmp_path,
        ),
    ).run_once(now=claim_time)

    db_session.refresh(task)
    db_session.refresh(channel)
    db_session.refresh(profile)
    db_session.refresh(relation)
    events = task_events(db_session, task)
    reinforcement_event = next(
        event
        for event in events
        if event.payload.get("message") == KG_RUNTIME_CONTEXT_REINFORCED_MESSAGE
    )
    message_event = next(
        event for event in events if event.type is TaskEventType.message_posted
    )
    profile_evidence = db_session.scalar(
        select(KnowledgeGraphEvidence).where(
            KnowledgeGraphEvidence.target_kind == "entity",
            KnowledgeGraphEvidence.target_id == profile.id,
            KnowledgeGraphEvidence.source_type == "task_summary",
            KnowledgeGraphEvidence.source_task_id == task.id,
        )
    )
    edge_evidence = db_session.scalar(
        select(KnowledgeGraphEvidence).where(
            KnowledgeGraphEvidence.target_kind == "edge",
            KnowledgeGraphEvidence.target_id == relation.id,
            KnowledgeGraphEvidence.source_type == "task_summary",
            KnowledgeGraphEvidence.source_task_id == task.id,
        )
    )

    assert result.status == TaskStatus.succeeded.value, task.error
    assert slack_client.messages == [
        {
            "channel": "C123",
            "text": humanized_answer,
            "thread_ts": "EvRuntimeGraphReinforcement",
        }
    ]
    assert channel.reinforcement_count == 1
    assert profile.reinforcement_count == 1
    assert relation.reinforcement_count == 1
    assert profile.last_reinforced_at is not None
    assert relation.last_reinforced_at is not None
    assert reinforcement_event.payload["entity_count"] == 2
    assert reinforcement_event.payload["edge_count"] == 1
    assert reinforcement_event.payload["evidence_count"] == 3
    assert reinforcement_event.payload["message_event_id"] == message_event.id
    assert profile_evidence is not None
    assert profile_evidence.source_task_event_id == message_event.id
    assert (
        profile_evidence.source_slack_message_ts == message_event.payload["message_ts"]
    )
    assert "Runtime graph context was used" in (profile_evidence.raw_snippet or "")
    assert edge_evidence is not None


def test_agent_executor_projects_task_summary_into_graph(
    db_session: Session,
    worker_session_factory: sessionmaker[Session],
    tmp_path: Path,
) -> None:
    claim_time = datetime(2026, 6, 3, 10, 35, tzinfo=UTC)
    task = create_task(db_session, event_id="EvTaskSummaryGraphProjection")
    task.input = (
        "summarize the last few decisions in this channel and call out "
        "anything unresolved"
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
        "Decisions / recommendations:\n"
        "- Use Langfuse as the default AI observability pilot for Kortny "
        "before adding another observability platform.\n"
        "- Keep Arize Phoenix as the second tool to revisit after the first "
        "observability baseline is working.\n\n"
        "Open / unresolved:\n"
        "- API key cleanup needs an owner before the integration docs are "
        "published.\n\n"
        "Commitments:\n"
        "- Weekly observability review cadence should continue until the pilot "
        "is stable."
    )
    humanized_answer = (
        "Short version: use Langfuse for the first observability pilot, keep "
        "Arize Phoenix as the next tool to revisit, and do not publish the "
        "integration docs until API key cleanup has an owner."
    )
    slack_client = FakeSlackClient()
    provider = FakeAgentProvider(
        [
            Completion(
                content=raw_answer,
                tool_calls=(),
                usage=TokenUsage(input_tokens=320, output_tokens=120),
                model="openai/gpt-4o-mini",
            ),
            Completion(
                content=humanized_answer,
                tool_calls=(),
                usage=TokenUsage(input_tokens=440, output_tokens=80),
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
            slack_client=slack_client,
            workspace_base_dir=tmp_path,
        ),
    ).run_once(now=claim_time)

    db_session.refresh(task)
    events = task_events(db_session, task)
    projection_event = next(
        event
        for event in events
        if event.payload.get("message") == KG_TASK_SUMMARY_PROJECTED_MESSAGE
    )
    message_event = next(
        event
        for event in events
        if event.type is TaskEventType.message_posted
        and event.payload.get("purpose") == "result"
    )
    decision_entity = db_session.scalar(
        select(KnowledgeGraphEntity).where(
            KnowledgeGraphEntity.installation_id == task.installation_id,
            KnowledgeGraphEntity.entity_type == "decision",
            KnowledgeGraphEntity.source_type == "task_summary",
            KnowledgeGraphEntity.lifecycle_state == "active",
            KnowledgeGraphEntity.display_name.ilike("%Langfuse%"),
            KnowledgeGraphEntity.is_current.is_(True),
            KnowledgeGraphEntity.expired_at.is_(None),
        )
    )
    open_question = db_session.scalar(
        select(KnowledgeGraphEntity).where(
            KnowledgeGraphEntity.installation_id == task.installation_id,
            KnowledgeGraphEntity.entity_type == "open_question",
            KnowledgeGraphEntity.source_type == "task_summary",
            KnowledgeGraphEntity.lifecycle_state == "candidate",
            KnowledgeGraphEntity.display_name.ilike("%API key cleanup%"),
            KnowledgeGraphEntity.is_current.is_(True),
            KnowledgeGraphEntity.expired_at.is_(None),
        )
    )

    assert result.status == TaskStatus.succeeded.value, task.error
    assert slack_client.messages == [
        {
            "channel": "C123",
            "text": humanized_answer,
            "thread_ts": "EvTaskSummaryGraphProjection",
        }
    ]
    assert projection_event.payload["active_count"] >= 3
    assert projection_event.payload["candidate_count"] == 1
    assert projection_event.payload["evidence_count"] >= 5
    assert projection_event.payload["message_event_id"] == message_event.id
    assert decision_entity is not None
    assert decision_entity.visibility_scope_type == "channel"
    assert decision_entity.visibility_scope_id == "C123"
    assert decision_entity.attrs_json["review_status"] == "auto"
    assert open_question is not None
    assert open_question.attrs_json["review_status"] == "needs_review"
    assert (
        open_question.attrs_json["review_reason"] == "sensitive_or_high_impact_language"
    )
    decision_evidence = db_session.scalar(
        select(KnowledgeGraphEvidence).where(
            KnowledgeGraphEvidence.target_kind == "entity",
            KnowledgeGraphEvidence.target_id == decision_entity.id,
            KnowledgeGraphEvidence.source_type == "task_summary",
            KnowledgeGraphEvidence.source_task_id == task.id,
        )
    )
    assert decision_evidence is not None
    assert decision_evidence.source_task_event_id == message_event.id
    assert (
        decision_evidence.source_slack_message_ts == message_event.payload["message_ts"]
    )
    graph_context = GraphService(db_session).retrieve_current_context(
        installation_id=task.installation_id,
        destination=DestinationSurface.channel("C123"),
        anchor_keys=("slack_channel:C123",),
        max_hops=2,
        max_items=20,
    )
    graph_context_keys = {entity.canonical_key for entity in graph_context.entities}
    assert decision_entity.canonical_key in graph_context_keys
    assert open_question.canonical_key not in graph_context_keys


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


def test_agent_executor_suppresses_final_message_for_background_channel_assessment(
    db_session: Session,
) -> None:
    task = create_task(db_session, event_id="EvGraphRefresh")
    task_service = TaskService(db_session)
    task_service.append_event(
        task,
        TaskEventType.log,
        {
            "message": CHANNEL_ASSESSMENT_REQUESTED_MESSAGE,
            "source": "dashboard_knowledge_graph_refresh",
            "channel_id": "C123",
            "membership_id": "membership-id",
            CHANNEL_ASSESSMENT_SUPPRESS_SLACK_POST_KEY: True,
        },
    )
    db_session.commit()
    slack_client = FakeSlackClient()

    AgentTaskExecutor(
        settings=make_settings(),
        slack_client=slack_client,
    )._post_outputs(
        settings=make_settings(),
        session=db_session,
        task=task,
        task_service=task_service,
        result_summary="Internal graph refresh assessment.",
    )

    assert slack_client.messages == []
    assert any(
        event.payload.get("message") == "slack_final_message_suppressed"
        and event.payload.get("reason") == "background_channel_assessment"
        for event in task_events(db_session, task)
    )


def test_worker_runs_dashboard_graph_refresh_without_agent_runtime(
    db_session: Session,
    worker_session_factory: sessionmaker[Session],
    tmp_path: Path,
) -> None:
    claim_time = datetime(2026, 6, 2, 11, 0, tzinfo=UTC)
    task = create_task(db_session, event_id="EvGraphPipeline")
    installation = db_session.get(Installation, task.installation_id)
    assert installation is not None
    task.slack_channel_id = "CGraph"
    task.slack_thread_ts = "1780400000.000000"
    task.slack_message_ts = "1780400000.000000"
    task.input = "Run background channel graph refresh."
    task.available_at = claim_time - timedelta(seconds=1)
    membership = SlackChannelMembership(
        installation_id=task.installation_id,
        channel_id="CGraph",
        channel_name="trading-ops",
        channel_type="public_channel",
        membership_status="active",
        discovered_via="member_joined_channel",
        added_by_user_id="UInvite",
        onboarding_status="posted",
        onboarding_message_ts="1780400000.000000",
        metadata_json={
            "assessment_task_id": str(task.id),
            "assessment_status": "queued",
        },
    )
    db_session.add_all(
        [
            membership,
            ObservationEvent(
                installation_id=installation.id,
                slack_team_id=installation.slack_team_id,
                channel_id="CGraph",
                user_id="U1",
                event_type="message",
                slack_event_id="EvObsGraph1",
                message_ts="1780400001.000001",
                thread_ts="1780400001.000001",
                file_id=None,
                raw_payload_checksum="checksum-EvObsGraph1",
                text_preview="Daily trade blotter posted for NVDA and AAPL earnings review.",
                visibility_metadata={"scope_type": "channel", "scope_id": "CGraph"},
                observed_at=claim_time,
            ),
            ObservationEvent(
                installation_id=installation.id,
                slack_team_id=installation.slack_team_id,
                channel_id="CGraph",
                user_id="U2",
                event_type="message",
                slack_event_id="EvObsGraph2",
                message_ts="1780400002.000002",
                thread_ts="1780400001.000001",
                file_id=None,
                raw_payload_checksum="checksum-EvObsGraph2",
                text_preview="Please review the entry and exit signals before the PM meeting.",
                visibility_metadata={"scope_type": "channel", "scope_id": "CGraph"},
                observed_at=claim_time + timedelta(seconds=1),
            ),
            ObservationEvent(
                installation_id=installation.id,
                slack_team_id=installation.slack_team_id,
                channel_id="CGraph",
                user_id="U3",
                event_type="file_share",
                slack_event_id="EvObsGraph3",
                message_ts="1780400003.000003",
                thread_ts="1780400003.000003",
                file_id="FBlotter",
                raw_payload_checksum="checksum-EvObsGraph3",
                text_preview="Uploaded blotter.csv with scale-down candidates and liquidation notes.",
                visibility_metadata={"scope_type": "channel", "scope_id": "CGraph"},
                observed_at=claim_time + timedelta(seconds=2),
            ),
        ]
    )
    task_service = TaskService(db_session)
    task_service.append_event(
        task,
        TaskEventType.log,
        {
            "message": CHANNEL_ASSESSMENT_REQUESTED_MESSAGE,
            "source": KG_REFRESH_SOURCE,
            "channel_id": "CGraph",
            "membership_id": str(membership.id),
            CHANNEL_ASSESSMENT_SUPPRESS_SLACK_POST_KEY: True,
        },
    )
    db_session.commit()

    provider = FakeAgentProvider(
        [
            Completion(
                content=json.dumps(
                    {
                        "likely_purpose": (
                            "Daily trading operations review for blotter updates, "
                            "earnings, and entry or exit signals."
                        ),
                        "recurring_topics": [
                            "trade blotter",
                            "entry and exit signals",
                            "earnings review",
                        ],
                        "workflows": [
                            "Review daily blotter files before PM meeting",
                            "Check scale-down and liquidation notes",
                        ],
                        "important_entities": [
                            "NVDA",
                            "AAPL",
                            "blotter.csv",
                        ],
                        "assumptions": [
                            "The channel appears focused on investment operations.",
                        ],
                        "help_opportunities": [
                            "Summarize daily blotter changes",
                            "Flag unresolved review items",
                            "Share the private API key cleanup plan",
                        ],
                        "evidence": [
                            "Daily trade blotter posted",
                            "review the entry and exit signals",
                        ],
                        "confidence": "medium",
                    }
                ),
                tool_calls=(),
                usage=TokenUsage(input_tokens=240, output_tokens=90),
                cost_usd=Decimal("0.000100"),
                model="openai/gpt-4o-mini",
            ),
            # Second call: witness channel-profile opportunity extraction that
            # runs after the assessment completes.
            Completion(
                content=json.dumps(
                    {
                        "candidates": [
                            {
                                "candidate_type": "recurring_check",
                                "title": "Summarize daily blotter changes",
                                "summary": (
                                    "The channel reviews a daily trade blotter "
                                    "before the PM meeting."
                                ),
                                "suggested_action": (
                                    "Offer a recurring blotter summary."
                                ),
                                "suggested_message": (
                                    "Want me to summarize blotter changes each morning?"
                                ),
                                "evidence": ["Daily trade blotter posted"],
                                "confidence_score": 0.7,
                                "confidence_reason": (
                                    "Recurring daily workflow in evidence."
                                ),
                            }
                        ]
                    }
                ),
                tool_calls=(),
                usage=TokenUsage(input_tokens=80, output_tokens=20),
                cost_usd=Decimal("0.000020"),
                model="openai/gpt-4o-mini",
            ),
        ]
    )
    slack_client = FakeSlackClient()
    result = TaskWorker(
        session_factory=worker_session_factory,
        worker_id="agent-worker-test",
        executor=AgentTaskExecutor(
            settings=make_settings(),
            llm_provider=provider,
            provider_name=LLMProvider.openrouter,
            slack_client=slack_client,
            workspace_base_dir=tmp_path,
        ),
    ).run_once(now=claim_time)

    db_session.refresh(task)
    db_session.refresh(membership)
    events = task_events(db_session, task)
    event_messages = [event.payload.get("message") for event in events]
    tool_results = [
        event
        for event in events
        if event.type is TaskEventType.tool_result
        and event.payload.get("tool") == "slack_channel_history"
    ]
    profile = db_session.scalar(
        select(ObserveChannelProfile).where(
            ObserveChannelProfile.installation_id == task.installation_id,
            ObserveChannelProfile.channel_id == "CGraph",
        )
    )
    profile_entity = db_session.scalar(
        select(KnowledgeGraphEntity).where(
            KnowledgeGraphEntity.installation_id == task.installation_id,
            KnowledgeGraphEntity.canonical_key == "channel_profile:CGraph",
            KnowledgeGraphEntity.is_current.is_(True),
        )
    )
    projection_event = next(
        event
        for event in events
        if event.payload.get("message") == KG_CHANNEL_PROFILE_PROJECTED_MESSAGE
    )
    witness_event = next(
        event
        for event in events
        if event.payload.get("message")
        == WITNESS_OPPORTUNITY_CANDIDATES_PROJECTED_MESSAGE
    )
    semantic_entities = tuple(
        db_session.scalars(
            select(KnowledgeGraphEntity).where(
                KnowledgeGraphEntity.installation_id == task.installation_id,
                KnowledgeGraphEntity.visibility_scope_id == "CGraph",
                KnowledgeGraphEntity.source_type == "onboarding_scan",
                KnowledgeGraphEntity.is_current.is_(True),
                KnowledgeGraphEntity.expired_at.is_(None),
            )
        )
    )
    semantic_edges = tuple(
        db_session.scalars(
            select(KnowledgeGraphEdge).where(
                KnowledgeGraphEdge.installation_id == task.installation_id,
                KnowledgeGraphEdge.source_type == "onboarding_scan",
                KnowledgeGraphEdge.is_current.is_(True),
                KnowledgeGraphEdge.expired_at.is_(None),
            )
        )
    )

    assert result.status == TaskStatus.succeeded.value, task.error
    assert task.status is TaskStatus.succeeded
    # Call 1: semantic profile extraction; call 2: witness opportunity pass
    assert len(provider.calls) == 2
    assert provider.calls[0][2] == {"type": "json_object"}
    assert slack_client.messages == []
    assert "agent_runtime_selected" in event_messages
    assert KG_CHANNEL_REFRESH_PIPELINE_STARTED_MESSAGE in event_messages
    assert KG_CHANNEL_REFRESH_HISTORY_LOADED_MESSAGE in event_messages
    assert KG_CHANNEL_REFRESH_SEMANTIC_EXTRACTED_MESSAGE in event_messages
    assert KG_CHANNEL_REFRESH_PROFILE_SYNTHESIZED_MESSAGE in event_messages
    assert KG_CHANNEL_PROFILE_PROJECTED_MESSAGE in event_messages
    assert WITNESS_OPPORTUNITY_CANDIDATES_PROJECTED_MESSAGE in event_messages
    assert "tool_selection_completed" not in event_messages
    assert "adk_runtime_started" not in event_messages
    assert "planned_workflow_classified" not in event_messages
    assert len(tool_results) == 1
    assert tool_results[0].payload["output"]["context_source"] == "observation_cache"
    assert tool_results[0].payload["output"]["message_count"] == 3
    assert membership.metadata_json["assessment_status"] == "posted"
    assert profile is not None
    assert profile.message_count == 3
    assert profile.file_count == 1
    assert profile.metadata_json["synthesis"] == "semantic_llm"
    assert profile.profile_json["semantic_extraction"]["workflows"] == [
        "Review daily blotter files before PM meeting",
        "Check scale-down and liquidation notes",
    ]
    assert profile.summary is not None
    assert "investment operations" in profile.summary.lower()
    assert "blotter" in profile.summary.lower()
    assert (
        next(
            event.payload["synthesis"]
            for event in events
            if event.payload.get("message")
            == KG_CHANNEL_REFRESH_PROFILE_SYNTHESIZED_MESSAGE
        )
        == "semantic_llm"
    )
    assert "adk" not in profile.summary.lower()
    assert "branch outputs" not in profile.summary.lower()
    assert profile_entity is not None
    assert profile_entity.attrs_json["summary"] == profile.summary
    assert profile_entity.lifecycle_state == "active"
    assert profile_entity.attrs_json["review_status"] == "auto"
    assert projection_event.payload["entity_count"] > 2
    assert projection_event.payload["edge_count"] > 1
    assert projection_event.payload["evidence_count"] > 3
    assert witness_event.payload["created_count"] >= 1
    assert witness_event.payload["updated_count"] == 0
    assert witness_event.payload["candidate_ids"]
    witness_candidates = tuple(
        db_session.scalars(
            select(WitnessOpportunityCandidate).where(
                WitnessOpportunityCandidate.installation_id == task.installation_id,
                WitnessOpportunityCandidate.channel_id == "CGraph",
            )
        )
    )
    assert witness_candidates
    assert {candidate.status for candidate in witness_candidates} == {"candidate"}
    semantic_keys = {entity.canonical_key for entity in semantic_entities}
    assert "channel_topic:CGraph:trade-blotter" in semantic_keys
    assert "channel_entity:CGraph:nvda" in semantic_keys
    assert any(
        key.startswith(
            "channel_workflow:CGraph:review-daily-blotter-files-before-pm-meeting"
        )
        for key in semantic_keys
    )
    semantic_kinds = {
        entity.attrs_json.get("semantic_kind")
        for entity in semantic_entities
        if entity.attrs_json.get("kind") == "channel_semantic_projection"
    }
    assert {
        "topic",
        "workflow",
        "important_entity",
        "assumption",
        "help_opportunity",
    } <= (semantic_kinds)
    assert any(
        edge.relationship_type == "maps_to"
        and edge.attrs_json.get("kind") == "channel_semantic_projection"
        for edge in semantic_edges
    )
    semantic_rows_by_key = {
        entity.canonical_key: entity for entity in semantic_entities
    }
    sensitive_help = semantic_rows_by_key[
        "channel_help:CGraph:share-the-private-api-key-cleanup-plan"
    ]
    assert sensitive_help.lifecycle_state == "candidate"
    assert sensitive_help.attrs_json["review_status"] == "needs_review"
    assert (
        sensitive_help.attrs_json["review_reason"]
        == "sensitive_or_high_impact_language"
    )
    active_semantic_rows = [
        entity
        for entity in semantic_entities
        if entity.attrs_json.get("kind") == "channel_semantic_projection"
        and entity.canonical_key != sensitive_help.canonical_key
    ]
    assert active_semantic_rows
    assert {entity.lifecycle_state for entity in active_semantic_rows} == {"active"}
    assert {entity.attrs_json["review_status"] for entity in active_semantic_rows} == {
        "auto"
    }
    graph_context = GraphService(db_session).retrieve_current_context(
        installation_id=task.installation_id,
        destination=DestinationSurface.channel("CGraph"),
        anchor_keys=("slack_channel:CGraph",),
        max_hops=2,
        max_items=50,
    )
    graph_context_keys = {entity.canonical_key for entity in graph_context.entities}
    assert "channel_topic:CGraph:trade-blotter" in graph_context_keys
    assert sensitive_help.canonical_key not in graph_context_keys
    trade_blotter = semantic_rows_by_key["channel_topic:CGraph:trade-blotter"]
    entry_exit = semantic_rows_by_key["channel_topic:CGraph:entry-and-exit-signals"]
    profile.profile_version += 1
    profile.summary = "Updated profile still sees trade blotter review."
    profile.profile_json = {
        "semantic_extraction": {
            "recurring_topics": ["trade blotter", "post-trade exceptions"],
            "workflows": ["Review daily blotter files before PM meeting"],
            "important_entities": ["NVDA"],
            "assumptions": [],
            "help_opportunities": ["Summarize daily blotter changes"],
            "evidence": ["Trade blotter discussion repeated"],
            "confidence": "medium",
        }
    }
    profile.assumptions_json = []
    profile.evidence_refs_json = []
    profile.confidence_score = Decimal("0.650")
    profile.confidence_reason = "Repeated graph refresh sample."
    profile.message_count = 2
    profile.file_count = 0
    profile.metadata_json = {"synthesis": "semantic_llm"}
    db_session.flush()

    second_projection = KnowledgeGraphExtractionService(
        db_session
    ).project_channel_profile(
        task=task,
        membership=membership,
        profile=profile,
    )
    db_session.flush()
    reinforced_trade_blotter = db_session.scalar(
        select(KnowledgeGraphEntity).where(
            KnowledgeGraphEntity.installation_id == task.installation_id,
            KnowledgeGraphEntity.canonical_key == "channel_topic:CGraph:trade-blotter",
            KnowledgeGraphEntity.is_current.is_(True),
            KnowledgeGraphEntity.expired_at.is_(None),
        )
    )
    db_session.refresh(entry_exit)
    assert second_projection.entity_count >= 1
    assert reinforced_trade_blotter is not None
    assert reinforced_trade_blotter.id == trade_blotter.id
    assert reinforced_trade_blotter.reinforcement_count == 1
    assert reinforced_trade_blotter.last_reinforced_at is not None
    assert entry_exit.is_current is True
    assert entry_exit.lifecycle_state == "active"
    reinforced_evidence = tuple(
        db_session.scalars(
            select(KnowledgeGraphEvidence).where(
                KnowledgeGraphEvidence.target_kind == "entity",
                KnowledgeGraphEvidence.target_id == reinforced_trade_blotter.id,
            )
        )
    )
    assert len(reinforced_evidence) >= 2


def test_worker_dashboard_graph_refresh_falls_back_on_bad_semantic_output(
    db_session: Session,
    worker_session_factory: sessionmaker[Session],
    tmp_path: Path,
) -> None:
    claim_time = datetime(2026, 6, 2, 11, 30, tzinfo=UTC)
    task = create_task(db_session, event_id="EvGraphSemanticFallback")
    installation = db_session.get(Installation, task.installation_id)
    assert installation is not None
    task.slack_channel_id = "CGraphFallback"
    task.slack_thread_ts = "1780400100.000000"
    task.slack_message_ts = "1780400100.000000"
    task.input = "Run background channel graph refresh."
    task.available_at = claim_time - timedelta(seconds=1)
    membership = SlackChannelMembership(
        installation_id=task.installation_id,
        channel_id="CGraphFallback",
        channel_name="product-launch",
        channel_type="public_channel",
        membership_status="active",
        discovered_via="member_joined_channel",
        added_by_user_id="UInvite",
        onboarding_status="posted",
        onboarding_message_ts="1780400100.000000",
        metadata_json={
            "assessment_task_id": str(task.id),
            "assessment_status": "queued",
        },
    )
    db_session.add_all(
        [
            membership,
            ObservationEvent(
                installation_id=installation.id,
                slack_team_id=installation.slack_team_id,
                channel_id="CGraphFallback",
                user_id="U1",
                event_type="message",
                slack_event_id="EvObsFallback1",
                message_ts="1780400101.000001",
                thread_ts="1780400101.000001",
                file_id=None,
                raw_payload_checksum="checksum-EvObsFallback1",
                text_preview=(
                    "Product launch checklist needs owner review and legal signoff."
                ),
                visibility_metadata={
                    "scope_type": "channel",
                    "scope_id": "CGraphFallback",
                },
                observed_at=claim_time,
            ),
        ]
    )
    task_service = TaskService(db_session)
    task_service.append_event(
        task,
        TaskEventType.log,
        {
            "message": CHANNEL_ASSESSMENT_REQUESTED_MESSAGE,
            "source": KG_REFRESH_SOURCE,
            "channel_id": "CGraphFallback",
            "membership_id": str(membership.id),
            CHANNEL_ASSESSMENT_SUPPRESS_SLACK_POST_KEY: True,
        },
    )
    db_session.commit()

    provider = FakeAgentProvider(
        [
            Completion(
                content=json.dumps(
                    {
                        "likely_purpose": "ADK branch outputs for final answer merge",
                        "recurring_topics": ["route_reason"],
                        "workflows": [],
                        "important_entities": [],
                        "assumptions": [],
                        "help_opportunities": [],
                        "evidence": [],
                        "confidence": "medium",
                    }
                ),
                tool_calls=(),
                usage=TokenUsage(input_tokens=120, output_tokens=45),
                cost_usd=Decimal("0.000050"),
                model="openai/gpt-4o-mini",
            ),
            # Second call: witness channel-profile opportunity extraction that
            # runs after the assessment completes.
            Completion(
                content=json.dumps(
                    {"candidates": [], "skipped_reason": "no actionable items"}
                ),
                tool_calls=(),
                usage=TokenUsage(input_tokens=80, output_tokens=20),
                cost_usd=Decimal("0.000020"),
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
            slack_client=FakeSlackClient(),
            workspace_base_dir=tmp_path,
        ),
    ).run_once(now=claim_time)

    db_session.refresh(task)
    events = task_events(db_session, task)
    event_messages = [event.payload.get("message") for event in events]
    profile = db_session.scalar(
        select(ObserveChannelProfile).where(
            ObserveChannelProfile.installation_id == task.installation_id,
            ObserveChannelProfile.channel_id == "CGraphFallback",
        )
    )

    assert result.status == TaskStatus.succeeded.value, task.error
    # Call 1: semantic profile extraction; call 2: witness opportunity pass
    assert len(provider.calls) == 2
    assert KG_CHANNEL_REFRESH_SEMANTIC_FALLBACK_MESSAGE in event_messages
    assert KG_CHANNEL_REFRESH_SEMANTIC_EXTRACTED_MESSAGE not in event_messages
    assert (
        next(
            event.payload["synthesis"]
            for event in events
            if event.payload.get("message")
            == KG_CHANNEL_REFRESH_PROFILE_SYNTHESIZED_MESSAGE
        )
        == "deterministic"
    )
    assert profile is not None
    assert profile.metadata_json["synthesis"] == "deterministic"
    assert profile.summary is not None
    assert "product" in profile.summary.lower()
    assert "launch" in profile.summary.lower()
    assert "adk" not in profile.summary.lower()
    # Semantic extraction (120/45) + witness opportunity pass (80/20)
    assert task.total_input_tokens == 200
    assert task.total_output_tokens == 65


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
    settings = make_settings()
    extractor_response = json.dumps(
        {
            "candidates": [
                {
                    "candidate_type": "recurring_check",
                    "title": "Daily report review",
                    "summary": "Watch for daily reports that need review.",
                    "suggested_action": "Offer report review help.",
                    "suggested_message": "I can help review the daily report here.",
                    "evidence": ["The profile names daily report review."],
                    "confidence_score": 0.74,
                    "confidence_reason": "Assessment evidence names the workflow.",
                },
                {
                    "candidate_type": "data_quality_issue",
                    "title": "Attached file quality",
                    "summary": "Flag attached report files that need review.",
                    "suggested_action": "Watch attached report quality.",
                    "suggested_message": "I can flag report file issues when I see them.",
                    "evidence": ["The profile found an attached report file."],
                    "confidence_score": 0.69,
                    "confidence_reason": "Assessment evidence includes a file.",
                },
            ],
            "skipped_reason": None,
        }
    )
    provider = FakeAgentProvider(
        [
            Completion(
                content=extractor_response,
                tool_calls=(),
                usage=TokenUsage(input_tokens=380, output_tokens=120),
                cost_usd=Decimal("0"),
                model="openai/gpt-4o-mini",
            )
        ]
    )

    AgentTaskExecutor(
        settings=settings,
        llm_provider=provider,
        provider_name=LLMProvider.openrouter,
    )._mark_channel_assessment_completed(
        settings=settings,
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
    projection_event = next(
        event
        for event in task_events(db_session, task)
        if event.payload.get("message") == KG_CHANNEL_PROFILE_PROJECTED_MESSAGE
    )
    witness_projection_event = next(
        event
        for event in task_events(db_session, task)
        if event.payload.get("message")
        == WITNESS_OPPORTUNITY_CANDIDATES_PROJECTED_MESSAGE
    )
    channel_entity = db_session.scalar(
        select(KnowledgeGraphEntity).where(
            KnowledgeGraphEntity.installation_id == task.installation_id,
            KnowledgeGraphEntity.canonical_key == "slack_channel:CObserve",
            KnowledgeGraphEntity.is_current.is_(True),
        )
    )
    profile_entity = db_session.scalar(
        select(KnowledgeGraphEntity).where(
            KnowledgeGraphEntity.installation_id == task.installation_id,
            KnowledgeGraphEntity.canonical_key == "channel_profile:CObserve",
            KnowledgeGraphEntity.is_current.is_(True),
        )
    )
    witness_candidates = tuple(
        db_session.scalars(
            select(WitnessOpportunityCandidate).order_by(
                WitnessOpportunityCandidate.candidate_type
            )
        )
    )
    usage_row = db_session.scalar(select(LLMUsage).where(LLMUsage.task_id == task.id))

    assert membership.metadata_json["assessment_status"] == "posted"
    assert profile is not None
    assert profile.summary == "This channel appears to handle daily report review."
    assert profile.message_count == 2
    assert profile.file_count == 1
    assert profile.source_task_id == task.id
    assert completed_event.payload["profile_id"] == str(profile.id)
    assert completed_event.payload["profile_version"] == 1
    assert projection_event.payload["channel_id"] == "CObserve"
    assert projection_event.payload["entity_count"] == 2
    assert projection_event.payload["edge_count"] == 1
    assert projection_event.payload["evidence_count"] == 3
    assert witness_projection_event.payload["source_type"] == "channel_profile"
    assert witness_projection_event.payload["extractor"] == "llm"
    assert witness_projection_event.payload["raw_candidate_count"] == 2
    assert witness_projection_event.payload["created_count"] == 2
    assert witness_projection_event.payload["updated_count"] == 0
    assert set(witness_projection_event.payload["candidate_ids"]) == {
        str(candidate.id) for candidate in witness_candidates
    }
    assert len(provider.calls) == 1
    assert usage_row is not None
    assert usage_row.model_tier == "cheap_fast"
    assert len(witness_candidates) == 2
    assert {candidate.candidate_type for candidate in witness_candidates} == {
        "data_quality_issue",
        "recurring_check",
    }
    assert all(
        candidate.source_type == "channel_profile" for candidate in witness_candidates
    )
    assert all(
        candidate.source_profile_id == profile.id for candidate in witness_candidates
    )
    assert all(
        candidate.metadata_json["source"] == "llm_channel_profile_extractor"
        for candidate in witness_candidates
    )
    assert channel_entity is not None
    assert channel_entity.entity_type == "channel"
    assert channel_entity.lifecycle_state == "active"
    assert channel_entity.visibility_scope_type == "channel"
    assert profile_entity is not None
    assert profile_entity.lifecycle_state == "active"
    assert profile_entity.attrs_json["review_status"] == "auto"
    current_context = GraphService(db_session).retrieve_current_context(
        installation_id=task.installation_id,
        destination=DestinationSurface.channel("CObserve"),
        anchor_keys=("slack_channel:CObserve",),
        max_hops=1,
    )
    assert {entity.canonical_key for entity in current_context.entities} == {
        "slack_channel:CObserve",
        "channel_profile:CObserve",
    }


def test_agent_executor_projects_witness_candidates_from_posted_watch_answer(
    db_session: Session,
) -> None:
    task = create_task(db_session, event_id="EvWitnessTaskAnswer")
    task.slack_channel_id = "CWitness"
    task.slack_thread_ts = "1780600000.000001"
    task.slack_message_ts = "1780600000.000001"
    task.input = "what do you know about how this channel is used?"
    membership = SlackChannelMembership(
        installation_id=task.installation_id,
        channel_id="CWitness",
        channel_name="rag",
        channel_type="private_channel",
        membership_status="active",
        discovered_via="app_mention",
        onboarding_status="posted",
        metadata_json={},
    )
    db_session.add(membership)
    db_session.commit()
    task_service = TaskService(db_session)
    settings = make_settings()
    slack_client = FakeSlackClient()
    response_text = (
        "This channel is used for Kortny testing and real project execution. "
        "I should keep an eye on unresolved Linear decisions and broken "
        "integration output."
    )
    extractor_response = json.dumps(
        {
            "candidates": [
                {
                    "candidate_type": "unresolved_decision",
                    "title": "Linear decision follow-ups",
                    "summary": "Surface unresolved Linear decisions and blockers.",
                    "suggested_action": "Track unresolved Linear decisions.",
                    "suggested_message": (
                        "I can keep an eye on unresolved Linear decisions here."
                    ),
                    "evidence": ["The answer named unresolved Linear decisions."],
                    "confidence_score": 0.72,
                    "confidence_reason": "The final answer explicitly identified it.",
                },
                {
                    "candidate_type": "data_quality_issue",
                    "title": "Integration output quality",
                    "summary": "Flag broken integration output.",
                    "suggested_action": "Watch integration output quality.",
                    "suggested_message": "I can flag broken tool output when I see it.",
                    "evidence": ["The answer named broken integration output."],
                    "confidence_score": 0.68,
                    "confidence_reason": "The final answer explicitly identified it.",
                },
            ],
            "skipped_reason": None,
        }
    )
    provider = FakeAgentProvider(
        [
            Completion(
                content=extractor_response,
                tool_calls=(),
                usage=TokenUsage(input_tokens=420, output_tokens=150),
                cost_usd=Decimal("0"),
                model="openai/gpt-4o-mini",
            )
        ]
    )
    executor = AgentTaskExecutor(
        settings=settings,
        llm_provider=provider,
        provider_name=LLMProvider.openrouter,
        response_synthesizer=StaticResponseSynthesizer(),
        slack_client=slack_client,
    )

    posted_text = executor._post_outputs(
        settings=settings,
        session=db_session,
        task=task,
        task_service=task_service,
        result_summary=response_text,
    )
    executor._project_witness_opportunities_from_result(
        settings=settings,
        session=db_session,
        task=task,
        task_service=task_service,
        posted_response_text=posted_text,
    )
    db_session.commit()

    candidates = tuple(
        db_session.scalars(
            select(WitnessOpportunityCandidate).order_by(
                WitnessOpportunityCandidate.candidate_type
            )
        )
    )
    projection_event = next(
        event
        for event in task_events(db_session, task)
        if event.payload.get("message")
        == WITNESS_OPPORTUNITY_CANDIDATES_PROJECTED_MESSAGE
    )
    usage_row = db_session.scalar(select(LLMUsage).where(LLMUsage.task_id == task.id))

    assert slack_client.messages[-1]["text"] == response_text.strip()
    assert posted_text == response_text.strip()
    assert len(provider.calls) == 1
    assert provider.calls[0][2] == {"type": "json_object"}
    assert usage_row is not None
    assert usage_row.model_tier == "cheap_fast"
    assert len(candidates) == 2
    assert {candidate.candidate_type for candidate in candidates} == {
        "data_quality_issue",
        "unresolved_decision",
    }
    assert all(
        candidate.visibility_scope_type == "private_channel" for candidate in candidates
    )
    assert all(candidate.source_type == "task_summary" for candidate in candidates)
    assert all(candidate.source_task_id == task.id for candidate in candidates)
    assert projection_event.payload["source_type"] == "task_summary"
    assert projection_event.payload["extractor"] == "llm"
    assert projection_event.payload["created_count"] == 2
    assert projection_event.payload["updated_count"] == 0
    assert set(projection_event.payload["candidate_ids"]) == {
        str(candidate.id) for candidate in candidates
    }


def test_agent_executor_skips_witness_extractor_for_adk_quick_response(
    db_session: Session,
) -> None:
    task = create_task(db_session, event_id="EvWitnessQuickSkip")
    task.input = "are you up?"
    task_service = TaskService(db_session)
    task_service.append_event(
        task,
        TaskEventType.log,
        {
            "message": "adk_quick_response_selected",
            "runtime": "adk",
            "agent": "quick_response_agent",
            "reason": "runtime_handoff_quick_conversation",
        },
    )
    db_session.commit()
    settings = make_settings()
    slack_client = FakeSlackClient()
    provider = FakeAgentProvider([])
    executor = AgentTaskExecutor(
        settings=settings,
        llm_provider=provider,
        provider_name=LLMProvider.openrouter,
        response_synthesizer=StaticResponseSynthesizer(),
        slack_client=slack_client,
    )

    posted_text = executor._post_outputs(
        settings=settings,
        session=db_session,
        task=task,
        task_service=task_service,
        result_summary="Yep, I'm up.",
    )
    executor._project_witness_opportunities_from_result(
        settings=settings,
        session=db_session,
        task=task,
        task_service=task_service,
        posted_response_text=posted_text,
    )
    db_session.commit()

    assert slack_client.messages[-1]["text"] == "Yep, I'm up."
    assert provider.calls == []
    assert (
        db_session.scalar(select(func.count()).select_from(WitnessOpportunityCandidate))
        == 0
    )
    assert not any(
        event.payload.get("message") == WITNESS_OPPORTUNITY_CANDIDATES_PROJECTED_MESSAGE
        for event in task_events(db_session, task)
    )


def test_agent_executor_skips_witness_extractor_for_witness_autopilot_task(
    db_session: Session,
) -> None:
    task = create_task(db_session, event_id="EvWitnessAutopilotSkip")
    task.identity_kind = "synthetic"
    task.identity_key = "synthetic:witness_autopilot:loop-source"
    task.identity_payload = {"source": "witness_autopilot"}
    task_service = TaskService(db_session)
    db_session.commit()
    settings = make_settings()
    slack_client = FakeSlackClient()
    provider = FakeAgentProvider([])
    executor = AgentTaskExecutor(
        settings=settings,
        llm_provider=provider,
        provider_name=LLMProvider.openrouter,
        response_synthesizer=StaticResponseSynthesizer(),
        slack_client=slack_client,
    )

    posted_text = executor._post_outputs(
        settings=settings,
        session=db_session,
        task=task,
        task_service=task_service,
        result_summary="I found a useful follow-up.",
    )
    executor._project_witness_opportunities_from_result(
        settings=settings,
        session=db_session,
        task=task,
        task_service=task_service,
        posted_response_text=posted_text,
    )
    db_session.commit()

    assert slack_client.messages[-1]["text"] == "I found a useful follow-up."
    assert provider.calls == []
    assert (
        db_session.scalar(select(func.count()).select_from(WitnessOpportunityCandidate))
        == 0
    )
    assert not any(
        event.payload.get("message") == WITNESS_OPPORTUNITY_CANDIDATES_PROJECTED_MESSAGE
        for event in task_events(db_session, task)
    )


def cleanup_database(session: Session) -> None:
    for model in (
        WitnessOpportunityCandidate,
        KnowledgeGraphEvidence,
        KnowledgeGraphEdge,
        KnowledgeGraphEntity,
        Episode,
        ObserveChannelProfile,
        ObservationEvent,
        SlackChannelMembership,
        Artifact,
        LLMUsage,
        TaskEvent,
        SlackSideEffect,
        Task,
        Schedule,
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
        self.calls: list[
            tuple[tuple[ChatMessage, ...], tuple[JsonSchema, ...], JsonObject | None]
        ] = []

    def complete(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[JsonSchema] = (),
        *,
        response_format: JsonObject | None = None,
    ) -> Completion:
        self.calls.append((tuple(messages), tuple(tools), response_format))
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
        blocks: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        message: dict[str, Any] = {
            "channel": channel,
            "text": text,
            "thread_ts": thread_ts,
        }
        if blocks is not None:
            message["blocks"] = blocks
        self.messages.append(message)
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
            "AGENT_RUNTIME": "custom",
            "KORTNY_WORKFLOW_BACKEND": "inline",
            "BRAVE_SEARCH_API_KEY": brave_search_api_key,
            "POSTGRES_URL": "postgresql://kortny:kortny@localhost/kortny",
        }
    )
