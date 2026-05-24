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
from sqlalchemy import Engine, delete, select
from sqlalchemy.orm import Session, sessionmaker

from kortny.config.settings import LLMProvider as SettingsLLMProvider
from kortny.config.settings import Settings
from kortny.db.models import (
    Artifact,
    EncryptedSecret,
    Installation,
    LLMProvider,
    LLMUsage,
    ModelPricing,
    Task,
    TaskEvent,
    TaskEventType,
    TaskStatus,
)
from kortny.db.session import make_engine, make_session_factory, normalize_database_url
from kortny.llm import ChatMessage, Completion, TokenUsage, ToolCall
from kortny.slack.comments import ARTIFACT_COMMENT_FALLBACK_TEXT
from kortny.slack.reactions import (
    ACK_REACTION_ADDED_MESSAGE,
    ACK_REACTION_REMOVED_MESSAGE,
    COMPLETION_REACTION_ADDED_MESSAGE,
)
from kortny.tasks import TaskService
from kortny.tools import ToolResult
from kortny.tools.types import JsonObject, JsonSchema
from kortny.worker import (
    AgentTaskExecutor,
    TaskExecutionResult,
    TaskWorker,
    WalkingSkeletonExecutor,
)
from kortny.worker.agent_executor import GENERIC_FAILURE_TEXT

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
    ]
    assert events[2].payload == {
        "message": "task_executor_started",
        "worker_id": "worker-test",
    }
    assert events[3].payload == {
        "message": "task_executor_completed",
        "worker_id": "worker-test",
    }


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

    events = task_events(db_session, task)
    assert events[-2].type is TaskEventType.error
    assert events[-2].payload["message"] == "task_executor_failed"
    assert events[-1].type is TaskEventType.status_changed
    assert events[-1].payload["to"] == TaskStatus.failed.value


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


def test_agent_executor_replaces_ack_reaction_after_success(
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
    assert slack_client.reaction_adds == [
        {
            "channel": "C123",
            "name": "heavy_check_mark",
            "timestamp": "EvAgentWorkerReaction",
        }
    ]
    assert any(
        event.payload.get("message") == ACK_REACTION_REMOVED_MESSAGE for event in events
    )
    assert any(
        event.payload.get("message") == COMPLETION_REACTION_ADDED_MESSAGE
        and event.payload.get("reaction") == "heavy_check_mark"
        for event in events
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


def test_agent_executor_posts_generic_failure_notice_for_setup_errors(
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
    db_session.commit()

    slack_client = FakeSlackClient()
    result = TaskWorker(
        session_factory=worker_session_factory,
        worker_id="agent-worker-test",
        executor=AgentTaskExecutor(
            settings=make_settings(brave_search_api_key=""),
            llm_provider=FakeAgentProvider([]),
            provider_name=LLMProvider.openrouter,
            slack_client=slack_client,
            workspace_base_dir=tmp_path,
        ),
    ).run_once(now=claim_time)

    db_session.refresh(task)
    events = task_events(db_session, task)

    assert result.status == TaskStatus.failed.value
    assert task.status is TaskStatus.failed
    assert task.error is not None
    assert task.error["type"] == "ValueError"
    assert task.error["message"] == "BRAVE_SEARCH_API_KEY is required for web_search"
    assert slack_client.messages == [
        {
            "channel": "C123",
            "text": GENERIC_FAILURE_TEXT,
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
    assert slack_client.reaction_adds == [
        {
            "channel": "C123",
            "name": "warning",
            "timestamp": "EvAgentWorkerMissingSearch",
        }
    ]
    posted_event = next(
        event
        for event in events
        if event.type is TaskEventType.message_posted
        and event.payload.get("purpose") == "failure"
    )
    assert posted_event.payload["text"] == GENERIC_FAILURE_TEXT


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
