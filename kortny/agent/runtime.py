"""Runtime boundary for Kortny agent orchestration engines."""

from __future__ import annotations

import uuid
from typing import Protocol

from sqlalchemy.orm import Session

from kortny.agent.capabilities import CapabilityOverview
from kortny.agent.context import DEFAULT_SKILL_DIRECT_THRESHOLD, ContextAssembler
from kortny.agent.context_engine import ContextEngine
from kortny.agent.coordinator import (
    DEFAULT_TOOL_RESULT_PROMPT_MAX_CHARS,
    AgentCoordinator,
    AgentRunResult,
    ConnectedToolLoader,
    LLMClient,
)
from kortny.agent.execution import ExecutionGuardrailLimits
from kortny.agent.image_attachments import ImageAttachmentResolver
from kortny.agent.planner import ExecutionPlanner
from kortny.agent.thread_context import ThreadTranscriptProvider
from kortny.approvals import ToolApprovalPolicy
from kortny.config import Settings
from kortny.db.models import Task
from kortny.embeddings import EmbeddingIndex
from kortny.slack.assistant_status import StatusReporter
from kortny.tasks import TaskService
from kortny.tools import ToolRegistry


class AgentRuntime(Protocol):
    """Execution engine behind one Kortny task."""

    def run(self, task: Task | uuid.UUID) -> AgentRunResult:
        """Run the task and return the final agent result."""


class CustomAgentRuntime:
    """Adapter for the existing custom coordinator loop."""

    def __init__(
        self,
        *,
        session: Session,
        llm: LLMClient,
        registry: ToolRegistry,
        task_service: TaskService | None = None,
        max_turns: int = 6,
        system_prompt: str | None = None,
        thread_transcript_provider: ThreadTranscriptProvider | None = None,
        thread_context_max_chars: int = 12000,
        thread_context_recent_tasks: int = 3,
        thread_transcript_limit: int = 30,
        known_facts_max_chars: int = 4000,
        context_assembler: ContextAssembler | None = None,
        context_engine: ContextEngine | None = None,
        guardrail_limits: ExecutionGuardrailLimits | None = None,
        execution_planner: ExecutionPlanner | None = None,
        approval_policy: ToolApprovalPolicy | None = None,
        autonomy_default_level: str = "balanced",
        tool_result_prompt_max_chars: int = DEFAULT_TOOL_RESULT_PROMPT_MAX_CHARS,
        capability_overview: CapabilityOverview | None = None,
        embedding_index: EmbeddingIndex | None = None,
        skill_direct_threshold: float = DEFAULT_SKILL_DIRECT_THRESHOLD,
        trifecta_gate_enabled: bool = True,
        status_reporter: StatusReporter | None = None,
        agent_display_name: str = "Kortny",
        image_resolver: ImageAttachmentResolver | None = None,
        connected_tool_loader: ConnectedToolLoader | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.coordinator = AgentCoordinator(
            session=session,
            llm=llm,
            registry=registry,
            task_service=task_service,
            max_turns=max_turns,
            system_prompt=system_prompt,
            thread_transcript_provider=thread_transcript_provider,
            thread_context_max_chars=thread_context_max_chars,
            thread_context_recent_tasks=thread_context_recent_tasks,
            thread_transcript_limit=thread_transcript_limit,
            known_facts_max_chars=known_facts_max_chars,
            context_assembler=context_assembler,
            context_engine=context_engine,
            guardrail_limits=guardrail_limits,
            execution_planner=execution_planner,
            approval_policy=approval_policy,
            autonomy_default_level=autonomy_default_level,
            tool_result_prompt_max_chars=tool_result_prompt_max_chars,
            capability_overview=capability_overview,
            embedding_index=embedding_index,
            skill_direct_threshold=skill_direct_threshold,
            trifecta_gate_enabled=trifecta_gate_enabled,
            status_reporter=status_reporter,
            agent_display_name=agent_display_name,
            image_resolver=image_resolver,
            connected_tool_loader=connected_tool_loader,
            settings=settings,
        )

    def run(self, task: Task | uuid.UUID) -> AgentRunResult:
        """Run the task with the current custom coordinator."""

        return self.coordinator.run(task)
