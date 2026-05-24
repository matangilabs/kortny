"""Default worker executor that runs the agent coordinator."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from slack_sdk import WebClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from kortny.agent import AgentCoordinator
from kortny.agent.coordinator import DEFAULT_SYSTEM_PROMPT
from kortny.agent.thread_context import ThreadTranscriptProvider
from kortny.config import Settings, load_settings
from kortny.db.models import Artifact, Task, TaskEvent, TaskEventType
from kortny.db.models import LLMProvider as DbLLMProvider
from kortny.execution import task_workspace
from kortny.llm import LLMProvider, LLMService, create_llm_provider
from kortny.memory import WorkspaceStateService
from kortny.slack import SlackPoster, SlackThread
from kortny.slack.comments import (
    ArtifactCommentGenerator,
    LLMArtifactCommentGenerator,
    generate_artifact_comment,
)
from kortny.slack.posting import SlackPostingClient
from kortny.slack.thread_context import SlackThreadTranscriptProvider
from kortny.tasks import TaskCancelledError, TaskService
from kortny.tools import (
    PdfGeneratorTool,
    RecallFactTool,
    RememberFactTool,
    SlackChannelHistoryTool,
    SlackFileReadTool,
    Tool,
    ToolRegistry,
    WebSearchTool,
)

GENERIC_FAILURE_TEXT = (
    "Something went wrong while I was working on this. Please try again soon."
)
MEMORY_CONFIRMATION_PURPOSE = "memory_confirmation"
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TaskExecutionResult:
    """Result returned by a worker task executor."""

    result_summary: str


class TaskExecutor(Protocol):
    """Executes one already-claimed task."""

    def execute(
        self,
        *,
        session: Session,
        task: Task,
        task_service: TaskService,
    ) -> TaskExecutionResult:
        """Run the task and return a result summary."""


class AgentTaskExecutor:
    """Runs the real MVP agent flow for a task and posts outputs to Slack."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        llm_provider: LLMProvider | None = None,
        provider_name: DbLLMProvider | str | None = None,
        web_search_tool: Tool | None = None,
        slack_client: SlackPostingClient | None = None,
        thread_transcript_provider: ThreadTranscriptProvider | None = None,
        artifact_comment_generator: ArtifactCommentGenerator | None = None,
        workspace_base_dir: Path | str | None = None,
        system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        self.settings = settings
        self.llm_provider = llm_provider
        self.provider_name = DbLLMProvider(provider_name) if provider_name else None
        self.web_search_tool = web_search_tool
        self.slack_client = slack_client
        self.thread_transcript_provider = thread_transcript_provider
        self.artifact_comment_generator = artifact_comment_generator
        self.workspace_base_dir = workspace_base_dir
        self.system_prompt = system_prompt

    def execute(
        self,
        *,
        session: Session,
        task: Task,
        task_service: TaskService,
    ) -> TaskExecutionResult:
        settings = self.settings or load_settings()
        try:
            logger.info("agent executor started task_id=%s", task.id)
            with task_workspace(task.id, base_dir=self.workspace_base_dir) as workspace:
                llm = self._build_llm(
                    settings=settings,
                    session=session,
                    task_service=task_service,
                )
                registry = self._build_registry(
                    settings=settings,
                    session=session,
                    task=task,
                    task_service=task_service,
                    working_dir=workspace.path,
                )
                logger.info(
                    "agent executor registry ready task_id=%s tools=%s",
                    task.id,
                    ",".join(registry.names()),
                )
                agent_result = AgentCoordinator(
                    session=session,
                    llm=llm,
                    registry=registry,
                    task_service=task_service,
                    system_prompt=self.system_prompt,
                    thread_transcript_provider=self._build_thread_transcript_provider(
                        settings
                    ),
                ).run(task)
                task_service.raise_if_cancelled(task, phase="before_post_outputs")
                self._post_outputs(
                    settings=settings,
                    session=session,
                    task=task,
                    task_service=task_service,
                    result_summary=agent_result.result_summary,
                )
                logger.info(
                    "agent executor completed task_id=%s artifact_count=%s",
                    task.id,
                    agent_result.artifact_count,
                )
                return TaskExecutionResult(result_summary=agent_result.result_summary)
        except TaskCancelledError:
            logger.info("agent executor cancelled task_id=%s", task.id)
            raise
        except Exception:
            logger.exception("agent executor failed task_id=%s", task.id)
            self._post_failure_notice(
                settings=settings,
                session=session,
                task=task,
                task_service=task_service,
            )
            raise

    def _build_llm(
        self,
        *,
        settings: Settings,
        session: Session,
        task_service: TaskService,
    ) -> LLMService:
        provider = self.llm_provider or create_llm_provider(settings)
        provider_name = self.provider_name or DbLLMProvider(settings.llm_provider.value)
        return LLMService(
            session=session,
            provider=provider,
            provider_name=provider_name,
            task_service=task_service,
        )

    def _build_thread_transcript_provider(
        self,
        settings: Settings,
    ) -> ThreadTranscriptProvider:
        if self.thread_transcript_provider is not None:
            return self.thread_transcript_provider
        return SlackThreadTranscriptProvider(WebClient(token=settings.slack_bot_token))

    def _build_artifact_comment_generator(
        self,
        settings: Settings,
    ) -> ArtifactCommentGenerator:
        if self.artifact_comment_generator is not None:
            return self.artifact_comment_generator
        return LLMArtifactCommentGenerator(settings=settings)

    def _build_registry(
        self,
        *,
        settings: Settings,
        session: Session,
        task: Task,
        task_service: TaskService,
        working_dir: Path,
    ) -> ToolRegistry:
        web_search = self.web_search_tool or WebSearchTool.from_settings(settings)
        pdf_generator = PdfGeneratorTool(
            working_dir=working_dir,
            session=session,
            task_id=task.id,
            task_service=task_service,
        )
        slack_channel_history = SlackChannelHistoryTool(
            self._build_slack_history_client(settings),
            default_channel_id=task.slack_channel_id,
        )
        slack_file_read = SlackFileReadTool(
            client=self._build_slack_file_client(settings),
            bot_token=settings.slack_bot_token,
            working_dir=working_dir,
            max_file_size_bytes=settings.slack_file_read_max_bytes,
        )
        memory_service = WorkspaceStateService(
            session,
            task_service=task_service,
            poster=SlackPoster(
                session=session,
                client=self._build_slack_posting_client(settings),
                task_service=task_service,
            ),
        )
        remember_fact = RememberFactTool(service=memory_service, task=task)
        recall_fact = RecallFactTool(service=memory_service, task=task)
        return ToolRegistry(
            [
                web_search,
                pdf_generator,
                slack_channel_history,
                slack_file_read,
                remember_fact,
                recall_fact,
            ]
        )

    def _build_slack_history_client(self, settings: Settings) -> Any:
        if self.slack_client is not None and hasattr(
            self.slack_client, "conversations_history"
        ):
            return self.slack_client
        return WebClient(token=settings.slack_bot_token)

    def _build_slack_file_client(self, settings: Settings) -> Any:
        if self.slack_client is not None and hasattr(self.slack_client, "files_info"):
            return self.slack_client
        return WebClient(token=settings.slack_bot_token)

    def _build_slack_posting_client(self, settings: Settings) -> SlackPostingClient:
        if self.slack_client is not None:
            return self.slack_client
        return cast(SlackPostingClient, WebClient(token=settings.slack_bot_token))

    def _post_outputs(
        self,
        *,
        settings: Settings,
        session: Session,
        task: Task,
        task_service: TaskService,
        result_summary: str,
    ) -> None:
        client = self.slack_client
        if client is None:
            client = cast(
                SlackPostingClient,
                WebClient(token=settings.slack_bot_token),
            )
        poster = SlackPoster(
            session=session,
            client=client,
            task_service=task_service,
        )
        thread = SlackThread.from_task(task)
        artifacts = list(
            session.scalars(
                select(Artifact)
                .where(
                    Artifact.task_id == task.id,
                    Artifact.storage_path.is_not(None),
                    Artifact.posted_at.is_(None),
                )
                .order_by(Artifact.created_at)
            )
        )
        if not artifacts:
            if self._has_memory_confirmation_prompt(session=session, task=task):
                logger.info(
                    "suppressing final message after memory confirmation prompt task_id=%s",
                    task.id,
                )
                return
            logger.info("posting final message task_id=%s", task.id)
            poster.post_message(thread, result_summary)
            return

        for index, artifact in enumerate(artifacts):
            if artifact.storage_path is None:
                continue
            initial_comment = None
            if index == 0:
                initial_comment = generate_artifact_comment(
                    self._build_artifact_comment_generator(settings),
                    session=session,
                    task=task,
                    artifact=artifact,
                    task_service=task_service,
                )
            logger.info(
                "posting artifact task_id=%s artifact_id=%s filename=%s",
                task.id,
                artifact.id,
                artifact.filename,
            )
            poster.upload_file(
                thread,
                artifact.storage_path,
                artifact=artifact,
                initial_comment=initial_comment,
                title=artifact.filename,
            )

    def _has_memory_confirmation_prompt(
        self,
        *,
        session: Session,
        task: Task,
    ) -> bool:
        return (
            session.scalar(
                select(TaskEvent.id)
                .where(
                    TaskEvent.task_id == task.id,
                    TaskEvent.type == TaskEventType.message_posted,
                    TaskEvent.payload["purpose"].as_string()
                    == MEMORY_CONFIRMATION_PURPOSE,
                )
                .limit(1)
            )
            is not None
        )

    def _post_failure_notice(
        self,
        *,
        settings: Settings,
        session: Session,
        task: Task,
        task_service: TaskService,
    ) -> None:
        try:
            client = self.slack_client
            if client is None:
                client = cast(
                    SlackPostingClient,
                    WebClient(token=settings.slack_bot_token),
                )
            SlackPoster(
                session=session,
                client=client,
                task_service=task_service,
            ).post_message(
                SlackThread.from_task(task),
                GENERIC_FAILURE_TEXT,
                purpose="failure",
            )
            logger.info("posted generic failure notice task_id=%s", task.id)
        except Exception:
            logger.exception(
                "failed to post generic failure notice task_id=%s", task.id
            )


class WalkingSkeletonExecutor:
    """Legacy trivial executor retained for narrow worker tests."""

    def execute(
        self,
        *,
        session: Session,
        task: Task,
        task_service: TaskService,
    ) -> TaskExecutionResult:
        return TaskExecutionResult(
            result_summary=f"Walking skeleton processed task {task.id}: {task.input}"
        )
