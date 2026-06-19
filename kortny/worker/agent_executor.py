"""Default worker executor that runs the agent coordinator."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from slack_sdk import WebClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from kortny.agent.capabilities import CapabilityOverview, build_capability_overview
from kortny.agent.coordinator import DEFAULT_SYSTEM_PROMPT, AgentRunResult
from kortny.agent.execution import ExecutionGuardrailLimits
from kortny.agent.runtime import CustomAgentRuntime
from kortny.agent.thread_context import ThreadTranscriptProvider
from kortny.approvals import (
    TOOL_APPROVAL_PROMPT_PURPOSE,
    TOOL_APPROVAL_REACTION_INSTRUCTION,
    TOOL_APPROVAL_REQUIRED_MESSAGE,
    ToolApprovalRequest,
    ToolApprovalRequired,
    approval_prompt_text,
)
from kortny.composio import ComposioClient
from kortny.composio.connect import (
    ComposioConnectionRequired,
    connect_prompt_text,
    initiate_connect_for_task,
    park_payload,
)
from kortny.composio.provider import ComposioExternalToolProvider
from kortny.composio.runtime import connected_toolkit_slugs
from kortny.config import Settings, load_settings
from kortny.db.models import (
    Artifact,
    McpServer,
    Task,
    TaskEvent,
    TaskEventType,
)
from kortny.db.models import LLMProvider as DbLLMProvider
from kortny.embeddings import EmbeddingBackend, EmbeddingIndex, create_embedding_backend
from kortny.evals.retrieval.catalog_retriever import build_catalog_retrieve_fn
from kortny.execution import task_workspace
from kortny.intent import (
    IntentClassificationService,
    IntentScope,
    LLMIntentClassifier,
)
from kortny.intent.models import IntentRequest, IntentSurface
from kortny.knowledge_graph import (
    KG_CHANNEL_PROFILE_PROJECTED_MESSAGE,
    KG_RUNTIME_CONTEXT_REINFORCED_MESSAGE,
    KG_TASK_SUMMARY_PROJECTED_MESSAGE,
    ChannelGraphRefreshPipeline,
    KnowledgeGraphExtractionService,
    RuntimeGraphReinforcementService,
    TaskSummaryGraphExtractionService,
    is_dashboard_graph_refresh_task,
)
from kortny.llm import ChatMessage, LLMProvider, LLMService, ModelRoute, ModelRouter
from kortny.llm.errors import classify_provider_failure
from kortny.llm.routing import ModelRouteTier, effective_intent_decision
from kortny.llm.runtime_config import (
    RuntimeModelSelection,
    create_provider_for_selection,
    select_runtime_model,
)
from kortny.mcp.provider import McpExternalToolProvider
from kortny.memory import WorkspaceStateService
from kortny.observability import log_observation
from kortny.observe.assessment import (
    CHANNEL_ASSESSMENT_COMPLETED_MESSAGE,
    CHANNEL_ASSESSMENT_FAILED_MESSAGE,
    CHANNEL_ASSESSMENT_SUPPRESS_SLACK_POST_KEY,
    channel_assessment_request_event,
    is_channel_assessment_task,
)
from kortny.observe.profiles import ObserveChannelProfileService
from kortny.persona import personalize
from kortny.routing import (
    ROUTING_DECISION_RECORDED_MESSAGE,
    SEMANTIC_ROUTER_PROMPT_VERSION,
    LLMSemanticRouter,
    NativeToolScopePolicy,
    RoutingDecisionTrace,
    SemanticRouteRequest,
    SemanticRouterPromotionGate,
    Tier0RouteDecision,
    Tier0RouteKind,
    Tier0Router,
)
from kortny.slack import SlackPoster, SlackThread
from kortny.slack.assistant_status import (
    AssistantStatusClient,
    AssistantStatusReporter,
    ChannelProgressReporter,
    MessageUpdateClient,
    StatusReporter,
)
from kortny.slack.comments import (
    ArtifactCommentGenerator,
    LLMArtifactCommentGenerator,
    generate_artifact_comment,
)
from kortny.slack.egress import parse_egress_allowlist
from kortny.slack.humanizer import (
    ChannelStyleCardResolver,
    ChannelStyleResolver,
    LLMResponseSynthesizer,
    ResponseSynthesizer,
    StaticResponseSynthesizer,
    strip_internal_response_preamble,
    synthesize_response,
)
from kortny.slack.membership import SlackChannelMembershipService
from kortny.slack.outbox import SlackSideEffectOutbox, slack_reaction_key
from kortny.slack.posting import SlackPostingClient
from kortny.slack.presentation import PresentationHint
from kortny.slack.reactions import (
    ACK_REACTION_ADDED_MESSAGE,
    ACK_REACTION_REMOVE_FAILED_MESSAGE,
    ACK_REACTION_REMOVED_MESSAGE,
    LibraryReactionProvider,
    ReactionProvider,
)
from kortny.slack.response_render import render_blocks
from kortny.slack.thread_context import SlackThreadTranscriptProvider
from kortny.tasks import TaskCancelledError, TaskService
from kortny.tool_selection import ExternalToolProvider, ToolCard
from kortny.tools import JsonObject, Tool, ToolRegistry, WebSearchTool
from kortny.tools.catalog import (
    native_slack_context_hint_names,
    runtime_native_tool_names,
    tool_descriptor_from_class,
)
from kortny.tools.find_tools import FindToolsTool
from kortny.tools.native_runtime import (
    NativeToolBuildContext,
    build_native_inventory_tools,
    build_native_tools,
    native_tool_classes_by_name,
)
from kortny.tools.schedules import ListSchedulesTool
from kortny.tools.slack_channel_history import (
    ObservationChannelHistoryCache,
    SlackChannelHistoryTool,
)
from kortny.witness import (
    WITNESS_OPPORTUNITY_CANDIDATES_PROJECTED_MESSAGE,
    WitnessChannelProfileExtractor,
    WitnessOpportunityService,
    WitnessTaskResponseExtractor,
)
from kortny.workflow.handoff import evaluate_runtime_handoff

GENERIC_FAILURE_TEXT = (
    "Something went wrong while I was working on this. Please try again soon."
)
MEMORY_CONFIRMATION_PURPOSE = "memory_confirmation"
PLANNED_WORKFLOW_PROGRESS_PURPOSE = "planned_progress_start"
PLANNED_WORKFLOW_PROGRESS_TEXT = "Hang on, I'll check."
TOOL_APPROVAL_PROMPT_SYNTHESIS_PROMPT_NAME = "kortny.tool_approval_prompt"

# Deterministic progress templates — picked by stable hash of task id so retries
# always get the same line. One sentence, 25-100 chars, first person, no emoji,
# no tool/agent/runtime jargon.
_PLANNED_PROGRESS_TEMPLATES: tuple[str, ...] = (
    "On it — give me a few minutes to dig in.",
    "Looking into this now, back shortly with what I find.",
    "Let me pull this together and I'll have something for you soon.",
    "Working on it — this one will take a moment.",
    "On it, I'll come back with a thorough answer.",
    "Digging in now, give me a bit.",
    "I'll gather what I need and get back to you shortly.",
)
TOOL_APPROVAL_PROMPT_RESPONSE_FORMAT: dict[str, str] = {"type": "json_object"}
ADK_QUICK_FINAL_AUTHORS = frozenset(
    {
        "quick_response_agent",
        "kortny_root_orchestrator",
    }
)
logger = logging.getLogger(__name__)


UNIFIED_DEPTH_DECISION_MESSAGE = "unified_depth_decision"


@dataclass(frozen=True, slots=True)
class TaskExecutionResult:
    """Result returned by a worker task executor."""

    result_summary: str


@dataclass(frozen=True, slots=True)
class UnifiedDepthDecision:
    """Unified router depth decision derived from the ingress intent (HIG-218).

    Tasks that never pass ingress (synthetic/scheduled/manual) default to
    ``standard_tool_task`` with ``depth_source="default"`` so legacy behavior is
    unchanged.
    """

    response_depth: str
    time_sensitivity: str
    toolkit_affinity: tuple[str, ...]
    depth_source: str

    @property
    def is_quick(self) -> bool:
        return self.response_depth == "quick_response"

    @property
    def is_deep(self) -> bool:
        return self.response_depth == "deep_workflow"

    def to_payload(self) -> JsonObject:
        return {
            "message": UNIFIED_DEPTH_DECISION_MESSAGE,
            "response_depth": self.response_depth,
            "time_sensitivity": self.time_sensitivity,
            "toolkit_affinity": list(self.toolkit_affinity),
            "depth_source": self.depth_source,
        }


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
        reaction_provider: ReactionProvider | None = None,
        composio_client: Any | None = None,
        response_synthesizer: ResponseSynthesizer | None = None,
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
        self.reaction_provider = reaction_provider or LibraryReactionProvider()
        self.composio_client = composio_client
        self.response_synthesizer = response_synthesizer
        # External tool providers created for the in-flight task; closed in the
        # ``execute`` finally so per-task resources (e.g. MCP sessions and their
        # subprocesses) never leak. Reset at the start of every ``execute``.
        self._active_external_providers: list[ExternalToolProvider] = []
        # Embedding backend is loaded once per executor process (model load is
        # expensive); the per-task EmbeddingIndex wraps it with the live session.
        self._embedding_backend: EmbeddingBackend | None = None
        self._embedding_backend_resolved = False
        # Capability overview built per task in _build_registry, consumed when
        # constructing the custom runtime's context assembler.
        self._capability_overview: CapabilityOverview | None = None

    def execute(
        self,
        *,
        session: Session,
        task: Task,
        task_service: TaskService,
    ) -> TaskExecutionResult:
        settings = self.settings or load_settings()
        self._active_external_providers = []
        self._capability_overview = None
        try:
            logger.info("agent executor started task_id=%s", task.id)
            with task_workspace(task.id, base_dir=self.workspace_base_dir) as workspace:
                tier0_decision = Tier0Router().route(task)
                if tier0_decision is not None:
                    agent_result = self._run_tier0_route(
                        session=session,
                        task=task,
                        task_service=task_service,
                        decision=tier0_decision,
                    )
                elif is_dashboard_graph_refresh_task(session, task):
                    agent_result = self._run_channel_graph_refresh_pipeline(
                        settings=settings,
                        session=session,
                        task=task,
                        task_service=task_service,
                    )
                else:
                    agent_result = self._run_agent_runtime(
                        settings=settings,
                        session=session,
                        task=task,
                        task_service=task_service,
                        working_dir=workspace.path,
                    )
                task_service.raise_if_cancelled(task, phase="before_post_outputs")
                posted_response_text = self._post_outputs(
                    settings=settings,
                    session=session,
                    task=task,
                    task_service=task_service,
                    result_summary=agent_result.result_summary,
                )
                self._record_semantic_router_shadow(
                    settings=settings,
                    session=session,
                    task=task,
                    task_service=task_service,
                )
                self._record_routing_chain_completed(
                    session=session,
                    task=task,
                    task_service=task_service,
                    result_summary=agent_result.result_summary,
                    posted_response_text=posted_response_text,
                )
                self._project_witness_opportunities_from_result(
                    settings=settings,
                    session=session,
                    task=task,
                    task_service=task_service,
                    posted_response_text=posted_response_text,
                )
                self._reinforce_runtime_graph_context(
                    session=session,
                    task=task,
                    task_service=task_service,
                )
                self._project_task_summary_graph_context(
                    session=session,
                    task=task,
                    task_service=task_service,
                    result_summary=agent_result.result_summary,
                )
                self._mark_channel_assessment_completed(
                    settings=settings,
                    session=session,
                    task=task,
                    task_service=task_service,
                    result_summary=agent_result.result_summary,
                )
                self._complete_ack_reaction(
                    settings=settings,
                    session=session,
                    task=task,
                    task_service=task_service,
                    succeeded=True,
                )
                logger.info(
                    "agent executor completed task_id=%s artifact_count=%s",
                    task.id,
                    agent_result.artifact_count,
                )
                return TaskExecutionResult(result_summary=agent_result.result_summary)
        except ComposioConnectionRequired as exc:
            logger.info(
                "agent executor needs composio connect task_id=%s toolkit=%s",
                task.id,
                exc.toolkit_slug,
            )
            self._park_for_composio_connect(
                settings=settings,
                session=session,
                task=task,
                task_service=task_service,
                connect=exc,
            )
            self._complete_ack_reaction(
                settings=settings,
                session=session,
                task=task,
                task_service=task_service,
                succeeded=True,
            )
            return TaskExecutionResult(
                result_summary=(
                    f"Waiting for you to connect {exc.toolkit_slug} before I continue."
                )
            )
        except ToolApprovalRequired as exc:
            logger.info(
                "agent executor waiting for approval task_id=%s tool=%s approval_key=%s",
                task.id,
                exc.request.tool_name,
                exc.request.approval_key,
            )
            prompt_ts = self._post_approval_request(
                settings=settings,
                session=session,
                task=task,
                task_service=task_service,
                approval=exc,
            )
            task_service.mark_waiting_for_tool_approval(
                task,
                request=exc.request.to_payload(),
                prompt_message_ts=prompt_ts,
            )
            self._complete_ack_reaction(
                settings=settings,
                session=session,
                task=task,
                task_service=task_service,
                succeeded=True,
            )
            return TaskExecutionResult(
                result_summary=(f"Waiting for approval to run {exc.request.tool_name}.")
            )
        except TaskCancelledError:
            logger.info("agent executor cancelled task_id=%s", task.id)
            raise
        except Exception as exc:
            logger.exception("agent executor failed task_id=%s", task.id)
            provider_failure = classify_provider_failure(exc)
            if provider_failure is not None:
                logger.warning(
                    "llm provider failure task_id=%s kind=%s",
                    task.id,
                    provider_failure.kind.value,
                )
            self._post_failure_notice(
                settings=settings,
                session=session,
                task=task,
                task_service=task_service,
                failure_text=(
                    provider_failure.message if provider_failure is not None else None
                ),
            )
            self._mark_channel_assessment_failed(
                session=session,
                task=task,
                task_service=task_service,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            self._complete_ack_reaction(
                settings=settings,
                session=session,
                task=task,
                task_service=task_service,
                succeeded=False,
            )
            raise
        finally:
            self._close_external_tool_providers()

    def _close_external_tool_providers(self) -> None:
        """Close any external tool providers created for the in-flight task.

        Providers may hold per-task resources (MCP sessions / subprocesses).
        Closing is duck-typed (optional ``close``) and best-effort so cleanup
        never masks the task's real outcome.
        """

        providers = self._active_external_providers
        self._active_external_providers = []
        for provider in providers:
            close = getattr(provider, "close", None)
            if not callable(close):
                continue
            try:
                close()
            except Exception:
                logger.exception(
                    "failed to close external tool provider provider=%s",
                    getattr(provider, "provider_name", type(provider).__name__),
                )

    def _run_channel_graph_refresh_pipeline(
        self,
        *,
        settings: Settings,
        session: Session,
        task: Task,
        task_service: TaskService,
    ) -> AgentRunResult:
        """Run dashboard graph refresh jobs without the general agent runtime."""

        task_service.append_event(
            task,
            TaskEventType.log,
            RoutingDecisionTrace(
                stage="tier0_system_of_record",
                route_tier="tier0",
                source="dashboard_graph_refresh",
                runtime_class="inline_tool_task",
                intent="knowledge_graph.refresh",
                confidence=1.0,
                escalated=False,
                selected_runtime="kg_channel_refresh_pipeline",
                selected_backend="inline",
                actual_path="kg_channel_refresh_pipeline",
                reason="dashboard_knowledge_graph_refresh",
            ).to_payload(),
        )
        task_service.append_event(
            task,
            TaskEventType.log,
            {
                "message": "agent_runtime_selected",
                "runtime": "kg_channel_refresh_pipeline",
                "reason": "dashboard_knowledge_graph_refresh",
            },
        )
        model_route = ModelRouter(settings).route_for_tier(
            ModelRouteTier.cheap_fast,
            reason="knowledge_graph_semantic_extraction",
        )
        selection = self._select_runtime_model(
            settings=settings,
            session=session,
            task=task,
            model_route=model_route,
        )
        provider = self.llm_provider or create_provider_for_selection(
            settings=settings,
            selection=selection,
        )
        provider_name = self.provider_name or selection.provider_name
        result = ChannelGraphRefreshPipeline(
            session=session,
            task_service=task_service,
            history_tool=SlackChannelHistoryTool(
                self._build_slack_history_client(settings),
                default_channel_id=task.slack_channel_id,
                cache=ObservationChannelHistoryCache(
                    session,
                    installation_id=task.installation_id,
                ),
            ),
            llm=LLMService(
                session=session,
                provider=provider,
                provider_name=provider_name,
                task_service=task_service,
                model_route=selection.model_route,
            ),
        ).run(task)
        return AgentRunResult(
            task_id=task.id,
            result_summary=result.result_summary,
            turns=0,
            artifact_count=result.artifact_count,
        )

    def _run_tier0_route(
        self,
        *,
        session: Session,
        task: Task,
        task_service: TaskService,
        decision: Tier0RouteDecision,
    ) -> AgentRunResult:
        """Execute a direct Tier 0 system-of-record route."""

        if decision.kind is Tier0RouteKind.schedule_state_query:
            return self._run_schedule_state_fast_path(
                session=session,
                task=task,
                task_service=task_service,
                decision=decision,
            )
        raise RuntimeError(f"Unsupported Tier 0 route: {decision.kind}")

    def _run_schedule_state_fast_path(
        self,
        *,
        session: Session,
        task: Task,
        task_service: TaskService,
        decision: Tier0RouteDecision,
    ) -> AgentRunResult:
        """Answer scheduler state questions from the scheduler DB directly."""

        query = _payload_optional_str(decision.metadata.get("query"))
        status = _payload_optional_str(decision.metadata.get("status")) or "open"
        task_service.append_event(
            task,
            TaskEventType.log,
            decision.to_trace().to_payload(),
        )
        task_service.append_event(
            task,
            TaskEventType.log,
            {
                "message": "agent_runtime_selected",
                "runtime": decision.selected_runtime,
                "reason": decision.reason,
                "query": query,
                "status": status,
            },
        )
        log_observation(
            logger,
            "agent_runtime_selected",
            task=task,
            runtime=decision.selected_runtime,
            reason=decision.reason,
            query=query,
            status=status,
        )
        output = self._invoke_list_schedules_fast_path(
            session=session,
            task=task,
            task_service=task_service,
            args={
                "scope": "visible",
                "status": status,
                **({"query": query} if query is not None else {}),
                "limit": 10,
            },
        )
        fallback_used = False
        if query is not None and output.get("count") == 0:
            fallback_used = True
            output = self._invoke_list_schedules_fast_path(
                session=session,
                task=task,
                task_service=task_service,
                args={
                    "scope": "visible",
                    "status": status,
                    "limit": 10,
                },
            )
        summary = _schedule_state_fast_path_response(
            output,
            query=query,
            status=status,
            fallback_used=fallback_used,
        )
        task_service.append_event(
            task,
            TaskEventType.log,
            {
                "message": "schedule_state_fast_path_completed",
                "runtime": "schedule_state_fast_path",
                "query": query,
                "status": status,
                "fallback_used": fallback_used,
                "schedule_count": output.get("count"),
            },
        )
        return AgentRunResult(
            task_id=task.id,
            result_summary=summary,
            turns=0,
            artifact_count=0,
        )

    def _invoke_list_schedules_fast_path(
        self,
        *,
        session: Session,
        task: Task,
        task_service: TaskService,
        args: JsonObject,
    ) -> JsonObject:
        tool = ListSchedulesTool(session=session, task=task)
        task_service.append_event(
            task,
            TaskEventType.tool_call,
            {
                "tool": tool.name,
                "runtime": "schedule_state_fast_path",
                "arguments": args,
                "argument_keys": sorted(args),
            },
        )
        started = time.perf_counter()
        result = tool.invoke(args)
        latency_ms = int((time.perf_counter() - started) * 1000)
        task_service.append_event(
            task,
            TaskEventType.tool_result,
            {
                "tool": tool.name,
                "runtime": "schedule_state_fast_path",
                "output": result.output,
                "latency_ms": latency_ms,
                "artifact_count": len(result.artifacts),
                "cost_usd": "0",
            },
        )
        return result.output

    def _build_llm(
        self,
        *,
        settings: Settings,
        session: Session,
        task: Task,
        task_service: TaskService,
    ) -> LLMService:
        model_route = None
        provider: LLMProvider
        if self.llm_provider is None:
            model_route = ModelRouter(settings).route_for_task(
                task,
                events=_task_events(session, task),
            )
            selection = self._select_runtime_model(
                settings=settings,
                session=session,
                task=task,
                model_route=model_route,
            )
            model_route = selection.model_route
            provider = create_provider_for_selection(
                settings=settings,
                selection=selection,
            )
            task_service.append_event(
                task,
                TaskEventType.log,
                {
                    "message": "model_route_selected",
                    "tier": model_route.tier.value,
                    "model": model_route.model,
                    "reason": model_route.reason,
                    **selection.event_payload,
                },
            )
            logger.info(
                "agent executor model route selected task_id=%s tier=%s model=%s reason=%s source=%s",
                task.id,
                model_route.tier.value,
                model_route.model,
                model_route.reason,
                selection.chain.source,
            )
        else:
            provider = self.llm_provider
            selection = None
        provider_name = self.provider_name or (
            selection.provider_name
            if selection is not None
            else DbLLMProvider(settings.llm_provider.value)
        )
        return LLMService(
            session=session,
            provider=provider,
            provider_name=provider_name,
            task_service=task_service,
            model_route=model_route,
        )

    def _select_runtime_model(
        self,
        *,
        settings: Settings,
        session: Session,
        task: Task,
        model_route: ModelRoute,
    ) -> RuntimeModelSelection:
        return select_runtime_model(
            session=session,
            settings=settings,
            installation_id=task.installation_id,
            model_route=model_route,
        )

    def _build_thread_transcript_provider(
        self,
        settings: Settings,
    ) -> ThreadTranscriptProvider:
        if self.thread_transcript_provider is not None:
            return self.thread_transcript_provider
        return SlackThreadTranscriptProvider(WebClient(token=settings.slack_bot_token))

    def _ensure_intent_decision(
        self,
        *,
        settings: Settings,
        session: Session,
        task: Task,
        task_service: TaskService,
    ) -> None:
        """Classify intent for surfaces that bypassed ingress classification.

        The assistant pane creates tasks directly (``slack/assistant.py``) without
        running the intent classifier, so they reach the worker with no intent
        decision and ``_resolve_unified_depth`` falls back to ``standard_tool_task``
        — a research-backed report then dies at the 10-turn cap. Run the *real*
        cheap-tier LLM classifier here (same one every other surface uses; it
        layers the deterministic depth override on top) so depth, tier, and tool
        hints come from genuine classification rather than a regex guess.

        Best-effort and idempotent: an existing decision short-circuits (retries
        don't re-classify), and any failure leaves the standard-depth default
        intact — never fails the task.
        """

        payload = (
            task.identity_payload if isinstance(task.identity_payload, dict) else {}
        )
        if payload.get("source_surface") != "assistant":
            return
        if _latest_intent_decision(session, task) is not None:
            return

        model_route = ModelRouter(settings).route_for_tier(
            ModelRouteTier.cheap_fast,
            reason="intent_classification",
        )
        try:
            selection = self._select_runtime_model(
                settings=settings,
                session=session,
                task=task,
                model_route=model_route,
            )
            classifier = LLMIntentClassifier(
                llm=LLMService(
                    session=session,
                    provider=create_provider_for_selection(
                        settings=settings,
                        selection=selection,
                    ),
                    provider_name=selection.provider_name,
                    task_service=task_service,
                    model_route=selection.model_route,
                )
            )
            # Ground + classify through the shared chokepoint (HIG-187): the DM
            # path used to attach grounding here separately and drifted; now the
            # service derives connected integrations from the task's scope.
            decision = IntentClassificationService(session, classifier).classify(
                request=IntentRequest(text=task.input, surface=IntentSurface.dm),
                scope=IntentScope(
                    installation_id=task.installation_id,
                    channel_id=task.slack_channel_id,
                    user_id=task.slack_user_id,
                ),
                task_id=task.id,
            )
        except Exception as exc:
            log_observation(
                logger,
                "intent_classification_failed",
                task=task,
                source="assistant_worker",
                error_type=type(exc).__name__,
                error=str(exc),
            )
            return

        task_service.append_event(
            task,
            TaskEventType.log,
            {
                "message": "intent_classification_completed",
                "source": "assistant_worker",
                "decision": decision.model_dump(mode="json"),
            },
        )
        log_observation(
            logger,
            "intent_classification_completed",
            task=task,
            source="assistant_worker",
            classification=decision.classification.value,
            response_depth=decision.response_depth,
            depth_source=decision.depth_source,
        )

    def _run_agent_runtime(
        self,
        *,
        settings: Settings,
        session: Session,
        task: Task,
        task_service: TaskService,
        working_dir: Path,
    ) -> AgentRunResult:
        self._ensure_intent_decision(
            settings=settings,
            session=session,
            task=task,
            task_service=task_service,
        )
        depth = _resolve_unified_depth(session, task)
        task_service.append_event(task, TaskEventType.log, depth.to_payload())
        log_observation(
            logger,
            UNIFIED_DEPTH_DECISION_MESSAGE,
            task=task,
            response_depth=depth.response_depth,
            time_sensitivity=depth.time_sensitivity,
            depth_source=depth.depth_source,
            toolkit_affinity=list(depth.toolkit_affinity),
        )
        # Deep workflows are the durable/planned candidates under the unified
        # router; quick/standard stay on the inline path.
        planned_workflow_candidate = depth.is_deep
        planned_workflow_payload = depth.to_payload()
        handoff = evaluate_runtime_handoff(settings=settings, task=task)
        task_service.append_event(task, TaskEventType.log, handoff.to_payload())
        log_observation(
            logger,
            "runtime_handoff_evaluated",
            task=task,
            runtime_class=handoff.runtime_class.value,
            durable_candidate=handoff.durable_candidate,
            recommended_backend=handoff.recommended_backend,
            configured_backend=handoff.configured_backend,
            selected_backend=handoff.selected_backend,
            reason_codes=list(handoff.reason_codes),
            fallback_reason=handoff.fallback_reason,
        )
        if handoff.configured_backend == "temporal" and (
            handoff.durable_candidate or planned_workflow_candidate
        ):
            self._shadow_start_temporal_workflow(
                settings=settings,
                task=task,
                task_service=task_service,
            )
        task_service.append_event(
            task,
            TaskEventType.log,
            RoutingDecisionTrace(
                stage="worker_runtime_handoff",
                route_tier="handoff_shadow",
                source="runtime_handoff",
                runtime_class=handoff.runtime_class.value,
                intent=_routing_intent_from_handoff(
                    handoff_reason_codes=handoff.reason_codes,
                    planned_workflow_payload=planned_workflow_payload,
                ),
                confidence=_routing_confidence_from_planned_payload(
                    planned_workflow_payload
                ),
                escalated=handoff.durable_candidate or planned_workflow_candidate,
                selected_runtime=settings.agent_runtime,
                selected_backend=handoff.selected_backend,
                actual_path="pending_runtime_selection",
                reason=handoff.reason,
                reason_codes=handoff.reason_codes,
                shadow_runtime_class=handoff.runtime_class.value,
                shadow_route=_routing_payload_str(planned_workflow_payload, "route"),
                shadow_planned_candidate=planned_workflow_candidate,
                shadow_confidence=_routing_confidence_from_planned_payload(
                    planned_workflow_payload
                ),
                response_depth=depth.response_depth,
                time_sensitivity=depth.time_sensitivity,
                toolkit_affinity=depth.toolkit_affinity,
                depth_source=depth.depth_source,
                metadata={
                    "recommended_backend": handoff.recommended_backend,
                    "configured_backend": handoff.configured_backend,
                    "fallback_reason": handoff.fallback_reason,
                },
            ).to_payload(),
        )
        task_service.append_event(
            task,
            TaskEventType.log,
            {
                "message": "agent_runtime_selected",
                "runtime": settings.agent_runtime,
            },
        )
        log_observation(
            logger,
            "agent_runtime_selected",
            task=task,
            runtime=settings.agent_runtime,
        )
        task_service.append_event(
            task,
            TaskEventType.log,
            RoutingDecisionTrace(
                stage="worker_runtime_selected",
                route_tier=(
                    "tier2_orchestrator"
                    if settings.agent_runtime == "adk"
                    else "custom_runtime"
                ),
                source="agent_executor",
                runtime_class=handoff.runtime_class.value,
                intent=_routing_intent_from_handoff(
                    handoff_reason_codes=handoff.reason_codes,
                    planned_workflow_payload=planned_workflow_payload,
                ),
                confidence=_routing_confidence_from_planned_payload(
                    planned_workflow_payload
                ),
                escalated=settings.agent_runtime == "adk",
                selected_runtime=settings.agent_runtime,
                selected_backend=handoff.selected_backend,
                actual_path=settings.agent_runtime,
                reason="Worker selected configured agent runtime.",
                reason_codes=handoff.reason_codes,
                shadow_runtime_class=handoff.runtime_class.value,
                shadow_route=_routing_payload_str(planned_workflow_payload, "route"),
                shadow_planned_candidate=planned_workflow_candidate,
                shadow_confidence=_routing_confidence_from_planned_payload(
                    planned_workflow_payload
                ),
                response_depth=depth.response_depth,
                time_sensitivity=depth.time_sensitivity,
                toolkit_affinity=depth.toolkit_affinity,
                depth_source=depth.depth_source,
            ).to_payload(),
        )
        if settings.agent_runtime == "adk":
            from kortny.agent.adk_runtime import AdkAgentRuntime

            if planned_workflow_candidate and settings.planned_workflows_enabled:
                self._record_planned_task_started(
                    task=task,
                    task_service=task_service,
                    progress_enabled=settings.planned_workflow_progress_updates_enabled,
                )
                if settings.planned_workflow_progress_updates_enabled:
                    self._post_planned_workflow_progress(
                        settings=settings,
                        session=session,
                        task=task,
                        task_service=task_service,
                    )

            model_route = ModelRouter(settings).route_for_task(
                task,
                events=_task_events(session, task),
            )
            selection = self._select_runtime_model(
                settings=settings,
                session=session,
                task=task,
                model_route=model_route,
            )
            model_route = selection.model_route
            task_service.append_event(
                task,
                TaskEventType.log,
                {
                    "message": "model_route_selected",
                    "tier": model_route.tier.value,
                    "model": model_route.model,
                    "reason": model_route.reason,
                    "runtime": "adk",
                    **selection.event_payload,
                },
            )
            logger.info(
                "agent executor model route selected task_id=%s runtime=adk tier=%s model=%s reason=%s source=%s",
                task.id,
                model_route.tier.value,
                model_route.model,
                model_route.reason,
                selection.chain.source,
            )

            cached_registry: ToolRegistry | None = None

            def registry_factory() -> ToolRegistry:
                nonlocal cached_registry
                if cached_registry is not None:
                    return cached_registry
                registry = self._build_registry(
                    settings=settings,
                    session=session,
                    task=task,
                    task_service=task_service,
                    working_dir=working_dir,
                )
                cached_registry = registry
                logger.info(
                    "agent executor registry ready task_id=%s runtime=adk_lazy tools=%s",
                    task.id,
                    ",".join(registry.names()),
                )
                return registry

            return AdkAgentRuntime(
                settings=settings,
                session=session,
                task_service=task_service,
                registry_factory=registry_factory,
                model=model_route.model,
                model_route=model_route,
                thread_transcript_provider=self._build_thread_transcript_provider(
                    settings
                ),
                tool_result_prompt_max_chars=settings.tool_result_prompt_max_chars,
                response_depth=depth.response_depth,
            ).run(task)

        llm = self._build_llm(
            settings=settings,
            session=session,
            task=task,
            task_service=task_service,
        )
        registry = self._build_registry(
            settings=settings,
            session=session,
            task=task,
            task_service=task_service,
            working_dir=working_dir,
        )
        logger.info(
            "agent executor registry ready task_id=%s tools=%s",
            task.id,
            ",".join(registry.names()),
        )
        return CustomAgentRuntime(
            session=session,
            llm=llm,
            registry=registry,
            task_service=task_service,
            system_prompt=self.system_prompt,
            tool_result_prompt_max_chars=settings.tool_result_prompt_max_chars,
            autonomy_default_level=settings.autonomy_default_level,
            thread_transcript_provider=self._build_thread_transcript_provider(settings),
            guardrail_limits=ExecutionGuardrailLimits.for_depth(depth.response_depth),
            capability_overview=self._capability_overview,
            embedding_index=self._embedding_index_for(
                settings=settings,
                session=session,
            ),
            skill_direct_threshold=settings.skill_direct_similarity_threshold,
            trifecta_gate_enabled=settings.trifecta_gate_enabled,
            status_reporter=self._build_status_reporter(
                settings=settings, task=task, session=session
            ),
            agent_display_name=settings.agent_display_name,
        ).run(task)

    def _build_status_reporter(
        self,
        *,
        settings: Settings,
        task: Task,
        session: Session,
    ) -> StatusReporter | None:
        """Live status reporter for the task's Slack surface.

        Two surfaces narrate progress; everything else returns None so the
        coordinator falls back to its no-op reporter.

        - Assistant pane (HIG-247): a native loading indicator driven by
          ``assistant.threads.setStatus``.
        - Channel / app-mention (HIG-220, default off): no native indicator
          exists, so we edit the posted acknowledgement in place as the task
          moves through phases. Requires an ack message to have been posted —
          there is nothing to edit otherwise.
        """

        payload = (
            task.identity_payload if isinstance(task.identity_payload, dict) else {}
        )
        thread_ts = task.slack_thread_ts or task.slack_message_ts
        client = self.slack_client or cast(
            SlackPostingClient, WebClient(token=settings.slack_bot_token)
        )
        if payload.get("source_surface") == "assistant":
            if not (task.slack_channel_id and thread_ts):
                return None
            return AssistantStatusReporter(
                client=cast(AssistantStatusClient, client),
                channel_id=task.slack_channel_id,
                thread_ts=thread_ts,
            )
        if not settings.channel_progress_enabled:
            return None
        if not task.slack_channel_id:
            return None
        ack = self._acknowledgement_message(session=session, task=task)
        if ack is None:
            return None
        message_ts, base_text = ack
        return ChannelProgressReporter(
            client=cast(MessageUpdateClient, client),
            channel_id=task.slack_channel_id,
            message_ts=message_ts,
            base_text=base_text,
        )

    def _acknowledgement_message(
        self,
        *,
        session: Session,
        task: Task,
    ) -> tuple[str, str] | None:
        """Return ``(message_ts, text)`` of this task's posted acknowledgement.

        The channel progress reporter edits that message in place; without an
        ack there is nothing to narrate into.
        """

        row = session.scalar(
            select(TaskEvent)
            .where(
                TaskEvent.task_id == task.id,
                TaskEvent.type == TaskEventType.message_posted,
                TaskEvent.payload["purpose"].as_string() == "acknowledgement",
            )
            .order_by(TaskEvent.created_at.desc())
            .limit(1)
        )
        if row is None or not isinstance(row.payload, dict):
            return None
        message_ts = row.payload.get("message_ts")
        text = row.payload.get("text")
        if not isinstance(message_ts, str) or not message_ts:
            return None
        return message_ts, text if isinstance(text, str) else ""

    def _record_semantic_router_shadow(
        self,
        *,
        settings: Settings,
        session: Session,
        task: Task,
        task_service: TaskService,
    ) -> None:
        """Run the Tier 1 semantic router in observe-only mode."""

        if self.llm_provider is not None:
            return

        handoff = evaluate_runtime_handoff(settings=settings, task=task)
        events = _task_events(session, task)
        depth_payload = _latest_payload_event(
            events,
            message=UNIFIED_DEPTH_DECISION_MESSAGE,
        )
        unified_depth = (
            _payload_str(depth_payload, "response_depth")
            if depth_payload is not None
            else None
        )
        planned_candidate = unified_depth == "deep_workflow"

        model_route = ModelRouter(settings).route_for_tier(
            ModelRouteTier.cheap_fast,
            reason="semantic_router_shadow",
        )
        try:
            selection = self._select_runtime_model(
                settings=settings,
                session=session,
                task=task,
                model_route=model_route,
            )
            decision = LLMSemanticRouter(
                LLMService(
                    session=session,
                    provider=create_provider_for_selection(
                        settings=settings,
                        selection=selection,
                    ),
                    provider_name=selection.provider_name,
                    task_service=task_service,
                    model_route=selection.model_route,
                )
            ).classify(
                task_id=task.id,
                request=SemanticRouteRequest(
                    user_request=task.input,
                    surface=_routing_surface_from_task(task),
                    identity_kind=task.identity_kind,
                ),
            )
        except Exception as exc:
            task_service.append_event(
                task,
                TaskEventType.log,
                {
                    "message": "semantic_router_shadow_failed",
                    "behavior": "observe_only",
                    "prompt_version": SEMANTIC_ROUTER_PROMPT_VERSION,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "fallback_policy": "ignore_shadow_router_failure",
                },
            )
            log_observation(
                logger,
                "semantic_router_shadow_failed",
                task=task,
                behavior="observe_only",
                prompt_version=SEMANTIC_ROUTER_PROMPT_VERSION,
                error_type=type(exc).__name__,
                error_summary=str(exc)[:500],
            )
            return

        metadata = decision.comparison_payload(
            handoff_runtime_class=handoff.runtime_class.value,
            handoff_recommended_backend=handoff.recommended_backend,
            selected_backend=handoff.selected_backend,
            planned_classifier_route=(
                "planned_candidate" if planned_candidate else "inline"
            )
            if unified_depth is not None
            else None,
            planned_candidate=planned_candidate if unified_depth is not None else None,
        )
        promotion = SemanticRouterPromotionGate().evaluate(decision)
        metadata["promotion_gate"] = promotion.to_payload()
        if unified_depth is not None:
            metadata["unified_response_depth"] = unified_depth
            metadata["shadow_depth_agreement"] = _shadow_depth_agreement(
                shadow_runtime_class=decision.runtime_class.value,
                unified_depth=unified_depth,
            )
        task_service.append_event(
            task,
            TaskEventType.log,
            RoutingDecisionTrace(
                stage="semantic_router_shadow",
                route_tier="tier1_shadow",
                source="llm_semantic_router",
                runtime_class=decision.runtime_class.value,
                intent=decision.intent,
                confidence=decision.confidence,
                margin=decision.margin,
                escalated=decision.execution_path.value != "inline",
                reason=decision.reason,
                shadow_runtime_class=decision.runtime_class.value,
                shadow_route=decision.execution_path.value,
                shadow_confidence=decision.confidence,
                metadata=metadata,
            ).to_payload(),
        )
        log_observation(
            logger,
            "semantic_router_shadow_completed",
            task=task,
            runtime_class=decision.runtime_class.value,
            intent=decision.intent,
            execution_path=decision.execution_path.value,
            confidence=decision.confidence,
            margin=decision.margin,
            runtime_disagreement=metadata["runtime_disagreement"],
            execution_path_disagreement=metadata["execution_path_disagreement"],
            threshold_eligible=promotion.threshold_eligible,
            control_allowed=promotion.control_allowed,
            promotion_reason_codes=list(promotion.reason_codes),
        )

    def _shadow_start_temporal_workflow(
        self,
        *,
        settings: Settings,
        task: Task,
        task_service: TaskService,
    ) -> None:
        """Start the Temporal envelope without handing over task completion yet."""

        try:
            from kortny.workflow.launcher import start_temporal_task_workflow_sync

            launch = start_temporal_task_workflow_sync(
                settings=settings,
                task=task,
            )
        except Exception as exc:
            task_service.append_event(
                task,
                TaskEventType.error,
                {
                    "message": "temporal_workflow_shadow_start_failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "mode": "shadow",
                },
            )
            log_observation(
                logger,
                "temporal_workflow_shadow_start_failed",
                task=task,
                error_type=type(exc).__name__,
                error_summary=str(exc)[:500],
                mode="shadow",
            )
            logger.exception(
                "temporal workflow shadow start failed task_id=%s",
                task.id,
            )
            return

        payload = {
            "message": "temporal_workflow_shadow_started",
            "mode": "shadow",
            **launch.to_payload(),
        }
        task_service.append_event(task, TaskEventType.log, payload)
        log_observation(
            logger,
            "temporal_workflow_shadow_started",
            task=task,
            workflow_id=launch.workflow_id,
            run_id=launch.run_id,
            task_queue=launch.task_queue,
            namespace=launch.namespace,
            mode="shadow",
        )

    def _build_artifact_comment_generator(
        self,
        settings: Settings,
    ) -> ArtifactCommentGenerator:
        if self.artifact_comment_generator is not None:
            return self.artifact_comment_generator
        return LLMArtifactCommentGenerator(settings=settings)

    def _build_response_synthesizer(
        self,
        settings: Settings,
    ) -> ResponseSynthesizer:
        if self.response_synthesizer is not None:
            return self.response_synthesizer
        if not settings.response_humanizer_enabled:
            return StaticResponseSynthesizer()
        return LLMResponseSynthesizer(
            settings=settings,
            provider=self.llm_provider,
            provider_name=self.provider_name,
        )

    @staticmethod
    def _build_style_resolver(settings: Settings) -> ChannelStyleResolver | None:
        if not settings.style_cards_enabled:
            return None
        return ChannelStyleCardResolver()

    def _build_registry(
        self,
        *,
        settings: Settings,
        session: Session,
        task: Task,
        task_service: TaskService,
        working_dir: Path,
    ) -> ToolRegistry:
        web_search = self._build_web_search_tool(
            settings=settings,
            task=task,
            task_service=task_service,
        )
        slack_identity_client = self._build_slack_identity_client(settings)
        slack_action_client = self._build_slack_posting_client(settings)
        memory_service = WorkspaceStateService(
            session,
            task_service=task_service,
            poster=SlackPoster(
                session=session,
                client=slack_action_client,
                task_service=task_service,
            ),
        )
        native_context = NativeToolBuildContext(
            settings=settings,
            session=session,
            task=task,
            task_service=task_service,
            working_dir=working_dir,
            web_search_tool=web_search,
            slack_history_client=self._build_slack_history_client(settings),
            slack_file_client=self._build_slack_file_client(settings),
            slack_identity_client=slack_identity_client,
            slack_action_client=slack_action_client,
            memory_service=memory_service,
        )
        native_tools = list(build_native_tools(native_context))
        raw_intent_decision = _latest_intent_decision(session, task)
        effective_decision = effective_intent_decision(raw_intent_decision)
        native_scope = NativeToolScopePolicy().apply(
            tools=native_tools,
            task_input=task.input,
            intent_decision=effective_decision,
        )
        task_service.append_event(task, TaskEventType.log, native_scope.to_payload())
        task_service.append_event(
            task,
            TaskEventType.log,
            RoutingDecisionTrace(
                stage="native_tool_scope",
                route_tier="tool_scope",
                source="native_tool_scope_policy",
                intent=(
                    _payload_str(effective_decision, "classification")
                    if effective_decision is not None
                    else None
                )
                or "unknown",
                confidence=None,
                escalated=False,
                selected_runtime=settings.agent_runtime,
                selected_backend="inline",
                actual_path="native_tool_scope",
                reason="Native tool exposure policy applied before runtime registry selection.",
                reason_codes=native_scope.reason_codes,
                candidate_tool_count=len(native_scope.original_tool_names),
                selected_tool_names=native_scope.selected_tool_names,
                suppressed_tool_names=native_scope.suppressed_tool_names,
                metadata=native_scope.to_payload(),
            ).to_payload(),
        )
        log_observation(
            logger,
            "native_tool_scope_applied",
            task=task,
            original_tool_count=len(native_scope.original_tool_names),
            selected_tool_count=len(native_scope.selected_tool_names),
            suppressed_tool_count=len(native_scope.suppressed_tool_names),
            suppressed_tool_names=list(native_scope.suppressed_tool_names),
            reason_codes=list(native_scope.reason_codes),
            schedule_mutation_allowed=native_scope.schedule_mutation_allowed,
            intent_classification=native_scope.intent_classification,
            likely_tools=list(native_scope.likely_tools),
        )
        inventory_tools = list(build_native_inventory_tools(native_context, ()))
        native_inventory_tools = (*tuple(native_tools), *tuple(inventory_tools))
        for inventory_tool in inventory_tools:
            cast(Any, inventory_tool).native_tools = native_inventory_tools
        native_tools = [
            cast(Tool, scoped_tool) for scoped_tool in native_scope.selected_tools
        ]
        native_tools.extend(inventory_tools)
        _record_deferred_secondary_intents(
            session=session,
            task=task,
            task_service=task_service,
            decision=raw_intent_decision,
        )
        # Agent-driven tool retrieval (HIG-269): the agent reaches external
        # tools only via find_tools, so there is no pre-flight selection
        # pipeline. Capability overview comes from the connection snapshot
        # (HIG-274), so grounding is intact without enumerating external cards.
        self._capability_overview = self._build_capability_overview(
            settings=settings,
            session=session,
            task=task,
            external_cards=(),
        )
        return self._finalize_registry(
            ToolRegistry(native_tools),
            settings=settings,
            session=session,
            task=task,
        )

    def _finalize_registry(
        self,
        registry: ToolRegistry,
        *,
        settings: Settings,
        session: Session,
        task: Task,
    ) -> ToolRegistry:
        """Add the find_tools agent-driven retrieval capability (HIG-269).

        find_tools is how the agent reaches external tools: it retrieves ranked
        tool slugs across every connected provider (Composio + MCP) in one index
        and loads the matches into this registry on demand. No-op when there is
        no embedding index, or nothing connected on either provider — the agent
        then runs native-only.
        """

        embedding_index = self._embedding_index_for(settings=settings, session=session)
        if embedding_index is None:
            return registry

        composio_provider: ComposioExternalToolProvider | None = None
        toolkits: tuple[str, ...] = ()
        if settings.composio_api_key is not None and settings.composio_catalog_enabled:
            toolkits = connected_toolkit_slugs(session, task)
            if toolkits:
                composio_provider = ComposioExternalToolProvider(
                    session=session,
                    task=task,
                    client=self._resolve_composio_connect_client(settings),
                    result_max_chars=settings.tool_result_max_chars,
                    embedding_index=embedding_index,
                    top_k=settings.tool_retrieval_top_k,
                )

        mcp_provider: McpExternalToolProvider | None = None
        mcp_cards: tuple[ToolCard, ...] = ()
        if settings.mcp_enabled:
            candidate = McpExternalToolProvider(
                session=session,
                task=task,
                encryption_key=settings.encryption_key,
                tool_timeout_seconds=settings.mcp_tool_timeout_seconds,
                result_max_chars=settings.tool_result_max_chars,
            )
            mcp_cards = candidate.tool_cards()
            if mcp_cards:
                mcp_provider = candidate
                # Track for close() — MCP providers hold per-task sessions.
                self._active_external_providers.append(candidate)

        if composio_provider is None and mcp_provider is None:
            return registry

        retrieve = build_catalog_retrieve_fn(
            session,
            toolkit_slugs=toolkits,
            embedding_index=embedding_index,
            # Grounding prior (HIG-274): boost the intent-named connected
            # toolkits so find_tools surfaces them near the top (eval: "my plate"
            # Linear tool went from rank #92 to #2 with this boost).
            boost_toolkits=frozenset(_intent_forced_toolkits(session, task)),
            extra_cards=mcp_cards,
        )

        def load(slugs: Sequence[str]) -> tuple[Tool, ...]:
            # MCP tools are keyed by runtime name (mcp__server__tool); everything
            # else is a Composio tool_slug. Dispatch each to the right loader.
            mcp_slugs = [slug for slug in slugs if slug.startswith("mcp__")]
            composio_slugs = [slug for slug in slugs if not slug.startswith("mcp__")]
            tools: list[Tool] = []
            if composio_provider is not None and composio_slugs:
                tools.extend(
                    composio_provider.load_runtime_tools_for_slugs(composio_slugs)
                )
            if mcp_provider is not None and mcp_slugs:
                tools.extend(mcp_provider.load_runtime_tools_for_slugs(mcp_slugs))
            return tuple(tools)

        registry.register_if_absent(
            cast(
                Tool,
                FindToolsTool(retrieve=retrieve, load=load, registry=registry),
            )
        )
        return registry

    def _build_web_search_tool(
        self,
        *,
        settings: Settings,
        task: Task,
        task_service: TaskService,
    ) -> Tool | None:
        if self.web_search_tool is not None:
            return self.web_search_tool
        try:
            return cast(Tool, WebSearchTool.from_settings(settings))
        except ValueError as exc:
            if "BRAVE_SEARCH_API_KEY" not in str(exc):
                raise
            task_service.append_event(
                task,
                TaskEventType.log,
                {
                    "message": "native_tool_unavailable",
                    "tool": "web_search",
                    "reason": "missing_brave_api_key",
                    "env_var": "BRAVE_SEARCH_API_KEY",
                },
            )
            log_observation(
                logger,
                "native_tool_unavailable",
                task=task,
                tool="web_search",
                reason="missing_brave_api_key",
                env_var="BRAVE_SEARCH_API_KEY",
            )
            return None

    def _build_capability_overview(
        self,
        *,
        settings: Settings,
        session: Session,
        task: Task,
        external_cards: tuple[Any, ...],
    ) -> CapabilityOverview | None:
        """Build the installation capability overview; never fails the task."""

        try:
            classes_by_name = native_tool_classes_by_name()
            native_descriptors = tuple(
                tool_descriptor_from_class(classes_by_name[name], settings=settings)
                for name in runtime_native_tool_names()
                if name in classes_by_name
            )
            mcp_rows = tuple(
                session.scalars(
                    select(McpServer)
                    .where(McpServer.installation_id == task.installation_id)
                    .order_by(McpServer.name)
                )
            )
            return build_capability_overview(
                native_descriptors=native_descriptors,
                external_cards=external_cards,
                mcp_rows=mcp_rows,
                connected_composio_toolkits=connected_toolkit_slugs(session, task),
            )
        except Exception:
            logger.warning(
                "capability overview build failed task_id=%s",
                task.id,
                exc_info=True,
            )
            return None

    def _embedding_index_for(
        self,
        *,
        settings: Settings,
        session: Session,
    ) -> EmbeddingIndex | None:
        """Return a session-bound EmbeddingIndex, or None when unavailable."""

        if not self._embedding_backend_resolved:
            self._embedding_backend = create_embedding_backend(settings)
            self._embedding_backend_resolved = True
        if self._embedding_backend is None:
            return None
        return EmbeddingIndex(session, self._embedding_backend)

    def _record_routing_chain_completed(
        self,
        *,
        session: Session,
        task: Task,
        task_service: TaskService,
        result_summary: str,
        posted_response_text: str | None,
    ) -> None:
        events = _task_events(session, task)
        route_events = [
            event.payload
            for event in events
            if event.type is TaskEventType.log
            and event.payload.get("message") == ROUTING_DECISION_RECORDED_MESSAGE
        ]
        selected_runtime = _latest_payload_event(
            events,
            message="agent_runtime_selected",
        )
        adk_completed = _latest_payload_event(events, message="adk_runtime_completed")
        tool_selection = _latest_payload_event(
            events, message="tool_selection_completed"
        )
        final_route = next(
            (
                payload
                for payload in reversed(route_events)
                if payload.get("actual_path") is not None
            ),
            route_events[-1] if route_events else {},
        )
        final_intent_route = next(
            (
                payload
                for payload in reversed(route_events)
                if payload.get("intent") is not None
            ),
            final_route,
        )
        payload: JsonObject = {
            "message": "routing_chain_completed",
            "route_event_count": len(route_events),
            "selected_runtime": (
                selected_runtime.get("runtime")
                if isinstance(selected_runtime, dict)
                else None
            ),
            "final_actual_path": final_route.get("actual_path"),
            "final_runtime_class": final_route.get("runtime_class"),
            "final_intent": final_intent_route.get("intent"),
            "result_chars": len(result_summary),
            "posted_response_chars": (
                len(posted_response_text) if posted_response_text is not None else 0
            ),
        }
        if adk_completed is not None:
            payload["adk_mode"] = adk_completed.get("mode")
            payload["adk_final_author"] = adk_completed.get("final_author")
        if tool_selection is not None:
            payload["candidate_tool_count"] = tool_selection.get("candidate_count")
            payload["selector_candidate_count"] = tool_selection.get(
                "selector_candidate_count"
            )
            payload["selected_tool_names"] = [
                item.get("registry_name")
                for item in tool_selection.get("selected_tools", [])
                if isinstance(item, dict)
            ]
        task_service.append_event(task, TaskEventType.log, payload)

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

    def _build_slack_identity_client(self, settings: Settings) -> Any:
        if self.slack_client is not None and (
            hasattr(self.slack_client, "users_info")
            or hasattr(self.slack_client, "conversations_info")
        ):
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
    ) -> str | None:
        if _should_suppress_slack_post(session, task):
            task_service.append_event(
                task,
                TaskEventType.log,
                {
                    "message": "slack_final_message_suppressed",
                    "reason": "background_channel_assessment",
                },
            )
            logger.info(
                "suppressing final message for background assessment task_id=%s",
                task.id,
            )
            return None
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
            egress_url_allowlist=parse_egress_allowlist(settings.egress_url_allowlist),
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
                return None
            logger.info("posting final message task_id=%s", task.id)
            response_source = strip_internal_response_preamble(result_summary)
            if response_source != result_summary.strip():
                task_service.append_event(
                    task,
                    TaskEventType.log,
                    {
                        "message": "final_response_sanitized",
                        "reason": "internal_preamble_removed",
                        "raw_chars": len(result_summary),
                        "output_chars": len(response_source),
                    },
                )
            skip_humanizer_reason = _response_humanizer_skip_reason(
                settings=settings,
                session=session,
                task=task,
                raw_text=response_source,
            )
            presentation: PresentationHint | None = None
            if skip_humanizer_reason is not None:
                response_text = response_source
                task_service.append_event(
                    task,
                    TaskEventType.log,
                    {
                        "message": "response_humanizer_skipped",
                        "reason": skip_humanizer_reason,
                        "runtime": "adk",
                        "raw_chars": len(result_summary),
                        "output_chars": len(response_text),
                    },
                )
            else:
                synthesis = synthesize_response(
                    self._build_response_synthesizer(settings),
                    session=session,
                    task=task,
                    raw_text=response_source,
                    task_service=task_service,
                    style_resolver=self._build_style_resolver(settings),
                )
                response_text = synthesis.text
                presentation = synthesis.presentation
            blocks = render_blocks(response_text, presentation)
            if thread.is_assistant and settings.assistant_streaming_enabled:
                poster.stream_message(thread, response_text, blocks=blocks)
            else:
                poster.post_message(thread, response_text, blocks=blocks)
            poster.clear_assistant_status(thread)
            return response_text

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
        poster.clear_assistant_status(thread)
        return None

    def _park_for_composio_connect(
        self,
        *,
        settings: Settings,
        session: Session,
        task: Task,
        task_service: TaskService,
        connect: ComposioConnectionRequired,
    ) -> None:
        """Post a connect link and park the task on waiting_approval (HIG-209).

        Creates the auth config + connect link, a pending user-scoped
        ComposioConnection row, posts the threaded connect prompt once (outbox
        idempotency via SlackPoster's task-bound message key), writes the
        ``wait_auth`` request marker, then parks the task. No new status enum
        value: the resume scan distinguishes connect-parks via the request
        payload's ``recovery_action``.
        """

        client = self._resolve_composio_connect_client(settings)
        callback_url = self._composio_connect_callback_url(settings)
        prompt = initiate_connect_for_task(
            session,
            task=task,
            toolkit_slug=connect.toolkit_slug,
            client=client,
            callback_url=callback_url,
        )
        prompt_ts = self._post_composio_connect_prompt(
            settings=settings,
            session=session,
            task=task,
            task_service=task_service,
            toolkit_slug=connect.toolkit_slug,
            redirect_url=prompt.redirect_url,
        )
        request = park_payload(
            toolkit_slug=connect.toolkit_slug,
            tool_name=connect.tool_name,
            connection_id=prompt.connection_id,
            scope_type=prompt.scope_type,
            scope_id=prompt.scope_id,
            prompt_message_ts=prompt_ts,
        )
        # Write the canonical approval-required marker so the existing
        # latest_pending_tool_approval / approve_tool_approval resume path finds
        # this connect park by its approval_key.
        task_service.append_event(
            task,
            TaskEventType.log,
            {
                "message": TOOL_APPROVAL_REQUIRED_MESSAGE,
                "request": request,
            },
        )
        task_service.mark_waiting_for_tool_approval(
            task,
            request=request,
            prompt_message_ts=prompt_ts or "",
        )

    def _resolve_composio_connect_client(self, settings: Settings) -> ComposioClient:
        if self.composio_client is not None:
            return cast(ComposioClient, self.composio_client)
        return ComposioClient(
            api_key=settings.composio_api_key,
            timeout_seconds=settings.composio_request_timeout_seconds,
        )

    def _composio_connect_callback_url(self, settings: Settings) -> str:
        base = (settings.public_base_url or "http://localhost:8080").rstrip("/")
        return f"{base}/composio/callback"

    def _post_composio_connect_prompt(
        self,
        *,
        settings: Settings,
        session: Session,
        task: Task,
        task_service: TaskService,
        toolkit_slug: str,
        redirect_url: str,
    ) -> str:
        client = self.slack_client
        if client is None:
            client = cast(
                SlackPostingClient,
                WebClient(token=settings.slack_bot_token),
            )
        return cast(
            str,
            SlackPoster(
                session=session,
                client=client,
                task_service=task_service,
            ).post_message(
                SlackThread.from_task(task),
                connect_prompt_text(
                    toolkit_slug=toolkit_slug,
                    redirect_url=redirect_url,
                ),
                purpose=TOOL_APPROVAL_PROMPT_PURPOSE,
                # Unique per toolkit so connecting a second app in one task
                # isn't deduped away by the outbox (HIG-248).
                idempotency_purpose=(
                    f"{TOOL_APPROVAL_PROMPT_PURPOSE}:connect:{toolkit_slug}"
                ),
            ),
        )

    def _post_approval_request(
        self,
        *,
        settings: Settings,
        session: Session,
        task: Task,
        task_service: TaskService,
        approval: ToolApprovalRequired,
    ) -> str:
        client = self.slack_client
        if client is None:
            client = cast(
                SlackPostingClient,
                WebClient(token=settings.slack_bot_token),
            )
        return cast(
            str,
            SlackPoster(
                session=session,
                client=client,
                task_service=task_service,
            ).post_message(
                SlackThread.from_task(task),
                self._approval_prompt_text(
                    settings=settings,
                    session=session,
                    task=task,
                    task_service=task_service,
                    approval=approval,
                ),
                purpose=TOOL_APPROVAL_PROMPT_PURPOSE,
                # Unique per approval so a task needing >1 approval doesn't have
                # its later prompts deduped away by the outbox (HIG-248).
                idempotency_purpose=(
                    f"{TOOL_APPROVAL_PROMPT_PURPOSE}:{approval.request.approval_key}"
                ),
            ),
        )

    def _approval_prompt_text(
        self,
        *,
        settings: Settings,
        session: Session,
        task: Task,
        task_service: TaskService,
        approval: ToolApprovalRequired,
    ) -> str:
        fallback = approval_prompt_text(approval.request)
        model_route = ModelRouter(settings).route_for_tier(
            ModelRouteTier.cheap_fast,
            reason="tool_approval_prompt",
        )
        try:
            provider: LLMProvider
            if self.llm_provider is None:
                selection = self._select_runtime_model(
                    settings=settings,
                    session=session,
                    task=task,
                    model_route=model_route,
                )
                model_route = selection.model_route
                provider = create_provider_for_selection(
                    settings=settings,
                    selection=selection,
                )
                provider_name: DbLLMProvider | str = selection.provider_name
            else:
                provider = self.llm_provider
                provider_name = self.provider_name or DbLLMProvider(
                    settings.llm_provider.value
                )

            completion = LLMService(
                session=session,
                provider=provider,
                provider_name=provider_name,
                task_service=task_service,
                model_route=model_route,
            ).complete(
                task_id=task.id,
                messages=(
                    ChatMessage(
                        role="system",
                        content=personalize(
                            _tool_approval_prompt_system_prompt(),
                            settings.agent_display_name,
                        ),
                    ),
                    ChatMessage(
                        role="user",
                        content=json.dumps(
                            _tool_approval_prompt_payload(
                                task=task,
                                request=approval.request,
                                fallback=fallback,
                            ),
                            default=str,
                            separators=(",", ":"),
                            sort_keys=True,
                        ),
                    ),
                ),
                response_format=TOOL_APPROVAL_PROMPT_RESPONSE_FORMAT,
                prompt_name=TOOL_APPROVAL_PROMPT_SYNTHESIS_PROMPT_NAME,
            )
            text = _tool_approval_prompt_from_completion(
                completion.content,
                fallback=fallback,
            )
            if text is None:
                return fallback
            task_service.append_event(
                task,
                TaskEventType.log,
                {
                    "message": "tool_approval_prompt_synthesized",
                    "prompt_name": TOOL_APPROVAL_PROMPT_SYNTHESIS_PROMPT_NAME,
                    "tool": approval.request.tool_name,
                    "risk": approval.request.risk,
                    "model_tier": ModelRouteTier.cheap_fast.value,
                    "text_source": "llm",
                },
            )
            return text
        except Exception as exc:
            task_service.append_event(
                task,
                TaskEventType.error,
                {
                    "message": "tool_approval_prompt_synthesis_failed",
                    "prompt_name": TOOL_APPROVAL_PROMPT_SYNTHESIS_PROMPT_NAME,
                    "tool": approval.request.tool_name,
                    "risk": approval.request.risk,
                    "model_tier": ModelRouteTier.cheap_fast.value,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            log_observation(
                logger,
                "tool_approval_prompt_synthesis_failed",
                task=task,
                tool=approval.request.tool_name,
                risk=approval.request.risk,
                model_tier=ModelRouteTier.cheap_fast.value,
                error_type=type(exc).__name__,
                error_summary=str(exc)[:500],
            )
            return fallback

    def _record_planned_task_started(
        self,
        *,
        task: Task,
        task_service: TaskService,
        progress_enabled: bool,
    ) -> None:
        task_service.append_event(
            task,
            TaskEventType.log,
            {
                "message": "planned_task_started",
                "runtime": "adk",
                "phase": "started",
                "progress_updates_enabled": progress_enabled,
            },
        )
        log_observation(
            logger,
            "planned_task_started",
            task=task,
            runtime="adk",
            phase="started",
            progress_updates_enabled=progress_enabled,
        )

    def _post_planned_workflow_progress(
        self,
        *,
        settings: Settings,
        session: Session,
        task: Task,
        task_service: TaskService,
    ) -> None:
        if _ack_reaction_already_added(session, task):
            # The ingress ack reaction is the acknowledgement; a templated
            # "On it..." message on top reads as bot filler (user feedback).
            log_observation(
                logger,
                "planned_task_progress_suppressed",
                task=task,
                runtime="adk",
                phase="started",
                reason="ack_reaction_present",
            )
            return
        client = self.slack_client
        if client is None:
            client = cast(
                SlackPostingClient,
                WebClient(token=settings.slack_bot_token),
            )
        progress_text, progress_source = self._planned_workflow_progress_text(
            settings=settings,
            session=session,
            task=task,
            task_service=task_service,
        )
        try:
            message_ts = SlackPoster(
                session=session,
                client=client,
                task_service=task_service,
            ).post_message(
                SlackThread.from_task(task),
                progress_text,
                purpose=PLANNED_WORKFLOW_PROGRESS_PURPOSE,
            )
        except Exception as exc:
            task_service.append_event(
                task,
                TaskEventType.error,
                {
                    "message": "planned_task_progress_post_failed",
                    "runtime": "adk",
                    "phase": "started",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            log_observation(
                logger,
                "planned_task_progress_post_failed",
                task=task,
                runtime="adk",
                phase="started",
                error_type=type(exc).__name__,
                error_summary=str(exc)[:500],
            )
            logger.warning(
                "planned workflow progress post failed task_id=%s",
                task.id,
                exc_info=True,
            )
            return

        task_service.append_event(
            task,
            TaskEventType.log,
            {
                "message": "planned_task_progress_posted",
                "runtime": "adk",
                "phase": "started",
                "purpose": PLANNED_WORKFLOW_PROGRESS_PURPOSE,
                "message_ts": message_ts,
                "text_chars": len(progress_text),
                "text_source": progress_source,
            },
        )
        log_observation(
            logger,
            "planned_task_progress_posted",
            task=task,
            runtime="adk",
            phase="started",
            purpose=PLANNED_WORKFLOW_PROGRESS_PURPOSE,
            message_ts=message_ts,
            text_chars=len(progress_text),
            text_source=progress_source,
        )

    def _planned_workflow_progress_text(
        self,
        *,
        settings: Settings,
        session: Session,
        task: Task,
        task_service: TaskService,
    ) -> tuple[str, str]:
        del settings, session, task_service
        index = int(task.id.hex[:8], 16) % len(_PLANNED_PROGRESS_TEMPLATES)
        return _PLANNED_PROGRESS_TEMPLATES[index], "template"

    def _project_witness_opportunities_from_result(
        self,
        *,
        settings: Settings,
        session: Session,
        task: Task,
        task_service: TaskService,
        posted_response_text: str | None,
    ) -> None:
        """Best-effort Witness candidates from delivered watch-for answers."""

        if (
            not posted_response_text
            or is_channel_assessment_task(session, task)
            or _should_skip_witness_extraction(session, task)
        ):
            return
        try:
            model_route = ModelRouter(settings).route_for_tier(
                ModelRouteTier.cheap_fast,
                reason="witness_task_response_extraction",
            )
            provider: LLMProvider
            if self.llm_provider is None:
                selection = self._select_runtime_model(
                    settings=settings,
                    session=session,
                    task=task,
                    model_route=model_route,
                )
                model_route = selection.model_route
                provider = create_provider_for_selection(
                    settings=settings,
                    selection=selection,
                )
                provider_name: DbLLMProvider | str = selection.provider_name
            else:
                provider = self.llm_provider
                provider_name = self.provider_name or DbLLMProvider(
                    settings.llm_provider.value
                )
            extraction = WitnessTaskResponseExtractor(
                LLMService(
                    session=session,
                    provider=provider,
                    provider_name=provider_name,
                    task_service=task_service,
                    model_route=model_route,
                )
            ).extract(
                task=task,
                response_text=posted_response_text,
            )
            result = WitnessOpportunityService(session).project_from_task_candidates(
                task=task,
                candidates=extraction.candidates,
                response_text=posted_response_text,
                extraction_metadata={
                    "raw_candidate_count": extraction.raw_candidate_count,
                    "skipped_reason": extraction.skipped_reason,
                },
            )
        except Exception as exc:
            task_service.append_event(
                task,
                TaskEventType.error,
                {
                    "message": "witness_opportunity_projection_failed",
                    "source_type": "task_summary",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            logger.exception(
                "witness opportunity projection failed task_id=%s", task.id
            )
            return
        if result.total_count == 0:
            return
        task_service.append_event(
            task,
            TaskEventType.log,
            {
                "message": WITNESS_OPPORTUNITY_CANDIDATES_PROJECTED_MESSAGE,
                "source_type": "task_summary",
                "extractor": "llm",
                "channel_id": task.slack_channel_id,
                "created_count": result.created_count,
                "updated_count": result.updated_count,
                "skipped_count": result.skipped_count,
                "candidate_ids": list(result.candidate_ids),
            },
        )

    def _mark_channel_assessment_completed(
        self,
        *,
        settings: Settings,
        session: Session,
        task: Task,
        task_service: TaskService,
        result_summary: str,
    ) -> None:
        membership_service = SlackChannelMembershipService(session)
        membership = membership_service.find_by_assessment_task_id(task_id=task.id)
        if membership is None:
            return
        membership_service.mark_assessment_completed(
            membership=membership,
            result_summary=result_summary,
        )
        profile = ObserveChannelProfileService(session).upsert_from_assessment(
            task=task,
            membership=membership,
            result_summary=result_summary,
        )
        projection = KnowledgeGraphExtractionService(session).project_channel_profile(
            task=task,
            membership=membership,
            profile=profile,
        )
        try:
            model_route = ModelRouter(settings).route_for_tier(
                ModelRouteTier.cheap_fast,
                reason="witness_channel_profile_extraction",
            )
            provider: LLMProvider
            if self.llm_provider is None:
                selection = self._select_runtime_model(
                    settings=settings,
                    session=session,
                    task=task,
                    model_route=model_route,
                )
                model_route = selection.model_route
                provider = create_provider_for_selection(
                    settings=settings,
                    selection=selection,
                )
                provider_name: DbLLMProvider | str = selection.provider_name
            else:
                provider = self.llm_provider
                provider_name = self.provider_name or DbLLMProvider(
                    settings.llm_provider.value
                )
            witness_extraction = WitnessChannelProfileExtractor(
                LLMService(
                    session=session,
                    provider=provider,
                    provider_name=provider_name,
                    task_service=task_service,
                    model_route=model_route,
                )
            ).extract(
                task=task,
                membership=membership,
                profile=profile,
            )
            witness_candidates = WitnessOpportunityService(
                session
            ).project_from_channel_profile(
                task=task,
                membership=membership,
                profile=profile,
                candidates=witness_extraction.candidates,
                extraction_metadata={
                    "raw_candidate_count": witness_extraction.raw_candidate_count,
                    "skipped_reason": witness_extraction.skipped_reason,
                },
            )
        except Exception as exc:
            task_service.append_event(
                task,
                TaskEventType.error,
                {
                    "message": "witness_opportunity_projection_failed",
                    "source_type": "channel_profile",
                    "channel_id": membership.channel_id,
                    "profile_id": str(profile.id),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            logger.exception(
                "channel profile witness projection failed task_id=%s profile_id=%s",
                task.id,
                profile.id,
            )
            witness_extraction = None
            witness_candidates = None
        task_service.append_event(
            task,
            TaskEventType.log,
            {
                "message": KG_CHANNEL_PROFILE_PROJECTED_MESSAGE,
                "channel_id": membership.channel_id,
                "membership_id": str(membership.id),
                "profile_id": str(profile.id),
                "channel_entity_id": projection.channel_entity_id,
                "profile_entity_id": projection.profile_entity_id,
                "profile_edge_id": projection.profile_edge_id,
                "entity_count": projection.entity_count,
                "edge_count": projection.edge_count,
                "evidence_count": projection.evidence_count,
            },
        )
        task_service.append_event(
            task,
            TaskEventType.log,
            {
                "message": WITNESS_OPPORTUNITY_CANDIDATES_PROJECTED_MESSAGE,
                "source_type": "channel_profile",
                "extractor": "llm",
                "channel_id": membership.channel_id,
                "membership_id": str(membership.id),
                "profile_id": str(profile.id),
                "raw_candidate_count": (
                    witness_extraction.raw_candidate_count
                    if witness_extraction is not None
                    else 0
                ),
                "skipped_reason": (
                    witness_extraction.skipped_reason
                    if witness_extraction is not None
                    else "extractor_failed"
                ),
                "created_count": (
                    witness_candidates.created_count
                    if witness_candidates is not None
                    else 0
                ),
                "updated_count": (
                    witness_candidates.updated_count
                    if witness_candidates is not None
                    else 0
                ),
                "skipped_count": (
                    witness_candidates.skipped_count
                    if witness_candidates is not None
                    else 1
                ),
                "candidate_ids": (
                    list(witness_candidates.candidate_ids)
                    if witness_candidates is not None
                    else []
                ),
            },
        )
        task_service.append_event(
            task,
            TaskEventType.log,
            {
                "message": CHANNEL_ASSESSMENT_COMPLETED_MESSAGE,
                "channel_id": membership.channel_id,
                "membership_id": str(membership.id),
                "profile_id": str(profile.id),
                "profile_version": profile.profile_version,
            },
        )

    def _reinforce_runtime_graph_context(
        self,
        *,
        session: Session,
        task: Task,
        task_service: TaskService,
    ) -> None:
        """Best-effort reinforcement for graph rows used in delivered answers."""

        try:
            result = RuntimeGraphReinforcementService(session).reinforce_task_context(
                task
            )
        except Exception as exc:
            task_service.append_event(
                task,
                TaskEventType.error,
                {
                    "message": "kg_runtime_context_reinforcement_failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            log_observation(
                logger,
                "kg_runtime_context_reinforcement_failed",
                level=logging.WARNING,
                task=task,
                error_type=type(exc).__name__,
                error_summary=str(exc)[:500],
            )
            return

        if result.reinforced_count <= 0:
            return
        task_service.append_event(task, TaskEventType.log, result.to_payload())
        log_observation(
            logger,
            KG_RUNTIME_CONTEXT_REINFORCED_MESSAGE,
            task=task,
            entity_count=result.entity_count,
            edge_count=result.edge_count,
            evidence_count=result.evidence_count,
            duplicate_count=result.duplicate_count,
        )

    def _project_task_summary_graph_context(
        self,
        *,
        session: Session,
        task: Task,
        task_service: TaskService,
        result_summary: str,
    ) -> None:
        """Best-effort graph growth from a successful task answer."""

        try:
            result = TaskSummaryGraphExtractionService(session).project_task_summary(
                task=task,
                result_summary=result_summary,
            )
        except Exception as exc:
            task_service.append_event(
                task,
                TaskEventType.error,
                {
                    "message": "kg_task_summary_projection_failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            log_observation(
                logger,
                "kg_task_summary_projection_failed",
                level=logging.WARNING,
                task=task,
                error_type=type(exc).__name__,
                error_summary=str(exc)[:500],
            )
            return

        if result.projected_count <= 0:
            return
        task_service.append_event(task, TaskEventType.log, result.to_payload())
        log_observation(
            logger,
            KG_TASK_SUMMARY_PROJECTED_MESSAGE,
            task=task,
            entity_count=result.entity_count,
            edge_count=result.edge_count,
            evidence_count=result.evidence_count,
            active_count=result.active_count,
            candidate_count=result.candidate_count,
        )

    def _mark_channel_assessment_failed(
        self,
        *,
        session: Session,
        task: Task,
        task_service: TaskService,
        error_type: str,
        error: str,
    ) -> None:
        membership_service = SlackChannelMembershipService(session)
        membership = membership_service.find_by_assessment_task_id(task_id=task.id)
        if membership is None:
            return
        membership_service.mark_assessment_failed(
            membership=membership,
            error_type=error_type,
            error=error,
        )
        task_service.append_event(
            task,
            TaskEventType.log,
            {
                "message": CHANNEL_ASSESSMENT_FAILED_MESSAGE,
                "channel_id": membership.channel_id,
                "membership_id": str(membership.id),
                "error_type": error_type,
                "error": error,
            },
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
        failure_text: str | None = None,
    ) -> None:
        if _should_suppress_slack_post(session, task):
            task_service.append_event(
                task,
                TaskEventType.log,
                {
                    "message": "slack_failure_notice_suppressed",
                    "reason": "background_channel_assessment",
                },
            )
            logger.info(
                "suppressing failure notice for background assessment task_id=%s",
                task.id,
            )
            return
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
                failure_text or GENERIC_FAILURE_TEXT,
                purpose="failure",
            )
            logger.info("posted generic failure notice task_id=%s", task.id)
        except Exception:
            logger.exception(
                "failed to post generic failure notice task_id=%s", task.id
            )

    def _complete_ack_reaction(
        self,
        *,
        settings: Settings,
        session: Session,
        task: Task,
        task_service: TaskService,
        succeeded: bool,
    ) -> None:
        ack_event = _latest_ack_reaction_event(session, task)
        if ack_event is None:
            return

        channel_id = _payload_str(ack_event.payload, "channel")
        message_ts = _payload_str(ack_event.payload, "message_ts")
        ack_reaction = _payload_str(ack_event.payload, "reaction")
        if channel_id is None or message_ts is None or ack_reaction is None:
            return

        client: Any = self.slack_client
        if client is None:
            client = WebClient(token=settings.slack_bot_token)

        self._remove_ack_reaction(
            client=client,
            task=task,
            task_service=task_service,
            channel_id=channel_id,
            message_ts=message_ts,
            reaction=ack_reaction,
        )
        del succeeded

    def _remove_ack_reaction(
        self,
        *,
        client: Any,
        task: Task,
        task_service: TaskService,
        channel_id: str,
        message_ts: str,
        reaction: str,
    ) -> None:
        reactions_remove = getattr(client, "reactions_remove", None)
        if not callable(reactions_remove):
            return
        try:
            outbox_result = SlackSideEffectOutbox(task_service.session).deliver(
                installation_id=task.installation_id,
                task_id=task.id,
                idempotency_key=slack_reaction_key(
                    task_id=task.id,
                    operation="reactions_remove",
                    channel_id=channel_id,
                    message_ts=message_ts,
                    reaction=reaction,
                ),
                operation="reactions_remove",
                purpose="acknowledgement_complete",
                target_channel_id=channel_id,
                target_message_ts=message_ts,
                request={
                    "channel": channel_id,
                    "name": reaction,
                    "timestamp": message_ts,
                },
                call=lambda: reactions_remove(
                    channel=channel_id,
                    name=reaction,
                    timestamp=message_ts,
                ),
            )
        except Exception as exc:
            logger.info(
                "slack ack reaction remove failed task_id=%s channel=%s message_ts=%s reaction=%s error_type=%s error=%s",
                task.id,
                channel_id,
                message_ts,
                reaction,
                type(exc).__name__,
                exc,
            )
            task_service.append_event(
                task,
                TaskEventType.log,
                {
                    "message": ACK_REACTION_REMOVE_FAILED_MESSAGE,
                    "channel": channel_id,
                    "message_ts": message_ts,
                    "reaction": reaction,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            return

        task_service.append_event(
            task,
            TaskEventType.log,
            {
                "message": ACK_REACTION_REMOVED_MESSAGE,
                "channel": channel_id,
                "message_ts": message_ts,
                "reaction": reaction,
                "slack_side_effect_id": str(outbox_result.side_effect.id),
                "idempotency_key": outbox_result.side_effect.idempotency_key,
            },
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


def _latest_ack_reaction_event(session: Session, task: Task) -> TaskEvent | None:
    return session.scalar(
        select(TaskEvent)
        .where(
            TaskEvent.task_id == task.id,
            TaskEvent.type == TaskEventType.log,
            TaskEvent.payload["message"].as_string() == ACK_REACTION_ADDED_MESSAGE,
        )
        .order_by(TaskEvent.seq.desc())
        .limit(1)
    )


def _task_events(session: Session, task: Task) -> tuple[TaskEvent, ...]:
    return tuple(
        session.scalars(
            select(TaskEvent)
            .where(TaskEvent.task_id == task.id)
            .order_by(TaskEvent.seq)
        )
    )


def _ack_reaction_already_added(session: Session, task: Task) -> bool:
    return any(
        event.payload.get("message") == ACK_REACTION_ADDED_MESSAGE
        for event in _task_events(session, task)
        if event.type == TaskEventType.log
    )


def _schedule_state_fast_path_response(
    output: JsonObject,
    *,
    query: str | None,
    status: str,
    fallback_used: bool,
) -> str:
    schedules = output.get("schedules")
    schedule_rows = schedules if isinstance(schedules, list) else []
    status_label = {
        "active": "active",
        "paused": "paused",
        "proposed": "draft",
        "open": "open",
        "all": "",
    }.get(status, status)
    query_label = f" matching `{query}`" if query and not fallback_used else ""
    if not schedule_rows:
        target = f" {status_label}" if status_label else ""
        return (
            f"I checked the scheduler and don't see any{target} schedules{query_label}."
        )

    if fallback_used and query:
        lead = (
            f"I didn't find an exact schedule match for `{query}`, but I found "
            f"{len(schedule_rows)} {status_label or 'visible'} schedule"
            f"{'' if len(schedule_rows) == 1 else 's'}."
        )
    else:
        lead = (
            f"Yes, I found {len(schedule_rows)} {status_label or 'visible'} "
            f"schedule{'' if len(schedule_rows) == 1 else 's'}{query_label}."
        )

    details = [
        _schedule_state_row(row) for row in schedule_rows[:5] if isinstance(row, dict)
    ]
    if len(schedule_rows) > 5:
        details.append(f"• Plus {len(schedule_rows) - 5} more.")
    suffix = "Scheduler DB is the source of truth here."
    return "\n".join([lead, "", *details, "", suffix]).strip()


def _schedule_state_row(row: Mapping[str, Any]) -> str:
    title = _plain_text(row.get("title")) or "Scheduled task"
    cadence = _nested_plain(row, "cadence", "label")
    next_run = _plain_text(row.get("next_run_human"))
    delivery = _nested_plain(row, "delivery", "label")
    fragments: list[str] = []
    if cadence:
        fragments.append(cadence)
    if next_run:
        fragments.append(f"next run {next_run}")
    if delivery:
        fragments.append(f"delivery: {delivery}")
    if not fragments:
        return f"• *{title}*"
    return f"• *{title}*: {'; '.join(fragments)}"


def _nested_plain(row: Mapping[str, Any], key: str, nested_key: str) -> str | None:
    value = row.get(key)
    if not isinstance(value, Mapping):
        return None
    return _plain_text(value.get(nested_key))


def _plain_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = " ".join(value.split()).strip()
    return normalized or None


def _response_humanizer_skip_reason(
    *,
    settings: Settings,
    session: Session,
    task: Task,
    raw_text: str,
) -> str | None:
    """Return why an ADK response can bypass final synthesis, if applicable."""

    if settings.agent_runtime != "adk":
        return None

    events = _task_events(session, task)

    if len(raw_text.strip()) >= settings.response_humanizer_min_chars:
        return None

    if any(
        event.type in {TaskEventType.tool_call, TaskEventType.tool_result}
        for event in events
    ):
        return None

    route = _latest_payload_event(
        events,
        message="model_route_selected",
    )
    if route is None or route.get("runtime") != "adk":
        return None
    if route.get("tier") != ModelRouteTier.cheap_fast.value:
        return None
    completed = _latest_payload_event(
        events,
        message="adk_runtime_completed",
    )
    if completed is None:
        return None
    final_author = completed.get("final_author")
    if final_author in ADK_QUICK_FINAL_AUTHORS:
        return "adk_quick_fast_path"
    return None


def _should_skip_witness_extraction(session: Session, task: Task) -> bool:
    if _is_witness_autopilot_task(task):
        return True
    events = _task_events(session, task)
    if (
        _latest_payload_event(events, message="schedule_state_fast_path_completed")
        is not None
    ):
        return True
    if _latest_payload_event(events, message="adk_quick_response_selected") is not None:
        return True
    completed = _latest_payload_event(events, message="adk_runtime_completed")
    if completed is None:
        return False
    return completed.get("final_author") in ADK_QUICK_FINAL_AUTHORS


def _is_witness_autopilot_task(task: Task) -> bool:
    if task.identity_kind != "synthetic":
        return False
    payload = task.identity_payload
    return isinstance(payload, Mapping) and payload.get("source") == "witness_autopilot"


def _latest_payload_event(
    events: tuple[TaskEvent, ...],
    *,
    message: str,
) -> dict[str, Any] | None:
    for event in reversed(events):
        if event.type is not TaskEventType.log:
            continue
        if event.payload.get("message") != message:
            continue
        return cast(dict[str, Any], event.payload)
    return None


def _routing_intent_from_handoff(
    *,
    handoff_reason_codes: tuple[str, ...],
    planned_workflow_payload: Mapping[str, Any] | None,
) -> str:
    reason_set = set(handoff_reason_codes)
    planned_reasons = (
        planned_workflow_payload.get("reason_codes", ())
        if planned_workflow_payload is not None
        else ()
    )
    if isinstance(planned_reasons, list | tuple | set):
        reason_set.update(str(reason) for reason in planned_reasons)
    if "schedule_state_query" in reason_set:
        return "scheduler.query"
    if (
        "scheduled_or_recurring" in reason_set
        or "scheduled_task_identity" in reason_set
    ):
        return "scheduler.create_or_run"
    if "quick_conversation" in reason_set:
        return "conversation.quick"
    if "write_or_destructive_intent" in reason_set:
        return "integration.write_or_approval"
    if "broad_research" in reason_set or "research_synthesis_work" in reason_set:
        return "research.synthesis"
    if "multi_source_synthesis" in reason_set:
        return "workspace.multi_source"
    if (
        "integration_tool_work" in reason_set
        or "integration_scope_present" in reason_set
    ):
        return "integration.read"
    return "task.general"


def _shadow_depth_agreement(
    *,
    shadow_runtime_class: str,
    unified_depth: str,
) -> bool:
    """Map the shadow semantic runtime_class to the unified depth (HIG-218).

    quick_response <-> quick_response;
    inline_tool_task <-> standard_tool_task;
    durable/scheduled_workflow_task <-> deep_workflow.
    """

    mapping = {
        "quick_response": "quick_response",
        "inline_tool_task": "standard_tool_task",
        "durable_workflow_task": "deep_workflow",
        "scheduled_workflow_task": "deep_workflow",
    }
    return mapping.get(shadow_runtime_class) == unified_depth


def _routing_payload_str(
    payload: Mapping[str, Any] | None,
    key: str,
) -> str | None:
    if payload is None:
        return None
    value = payload.get(key)
    return value if isinstance(value, str) and value else None


def _payload_optional_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _routing_confidence_from_planned_payload(
    payload: Mapping[str, Any] | None,
) -> float | None:
    if payload is None:
        return None
    value = payload.get("confidence")
    if isinstance(value, int | float):
        return float(value)
    return None


def _routing_surface_from_task(task: Task) -> str:
    if task.slack_channel_id.startswith("D"):
        return "dm"
    if task.identity_kind == "scheduled":
        return "scheduled"
    if task.identity_kind == "synthetic":
        return "synthetic"
    return "channel"


def _should_suppress_slack_post(session: Session, task: Task) -> bool:
    if task.slack_channel_id == "playground" or task.identity_kind == "manual":
        return True
    request_event = channel_assessment_request_event(session, task)
    if request_event is None:
        return False
    return request_event.payload.get(CHANNEL_ASSESSMENT_SUPPRESS_SLACK_POST_KEY) is True


def _intent_forced_toolkits(session: Session, task: Task) -> tuple[str, ...]:
    """Connected toolkits the grounded intent named, for the catalog floor.

    HIG-274: the capability-grounded classifier resolves "my work / my plate" to
    the connected work tracker via ``toolkit_affinity``. The Composio provider
    must keep those toolkits reachable past its top_k catalog ranking so the
    agent can actually answer instead of claiming the integration "isn't wired
    in" (task c65e7b2f).
    """

    decision = effective_intent_decision(_latest_intent_decision(session, task))
    if decision is None:
        return ()
    slugs: list[str] = []
    for key in ("toolkit_affinity", "likely_tools"):
        raw = decision.get(key)
        if isinstance(raw, list | tuple):
            slugs.extend(item for item in raw if isinstance(item, str) and item)
    return tuple(dict.fromkeys(slug.casefold() for slug in slugs if slug))


def _resolve_unified_depth(
    session: Session,
    task: Task,
) -> UnifiedDepthDecision:
    """Resolve the unified router depth for a task (HIG-218).

    Reads the execution-driving (effective) intent decision. Tasks with no
    intent decision default to ``standard_tool_task`` / ``default``.
    """

    decision = effective_intent_decision(_latest_intent_decision(session, task))
    if decision is None:
        return UnifiedDepthDecision(
            response_depth="standard_tool_task",
            time_sensitivity="interactive",
            toolkit_affinity=(),
            depth_source="default",
        )
    response_depth = _payload_str(decision, "response_depth") or "standard_tool_task"
    time_sensitivity = _payload_str(decision, "time_sensitivity") or "interactive"
    depth_source = _payload_str(decision, "depth_source") or "default"
    raw_affinity = decision.get("toolkit_affinity")
    toolkit_affinity: tuple[str, ...] = ()
    if isinstance(raw_affinity, list | tuple):
        toolkit_affinity = tuple(
            item for item in raw_affinity if isinstance(item, str) and item
        )
    return UnifiedDepthDecision(
        response_depth=response_depth,
        time_sensitivity=time_sensitivity,
        toolkit_affinity=toolkit_affinity,
        depth_source=depth_source,
    )


def _latest_intent_decision(session: Session, task: Task) -> dict[str, Any] | None:
    event = session.scalar(
        select(TaskEvent)
        .where(
            TaskEvent.task_id == task.id,
            TaskEvent.type == TaskEventType.log,
            TaskEvent.payload["message"].as_string()
            == "intent_classification_completed",
        )
        .order_by(TaskEvent.seq.desc())
        .limit(1)
    )
    if event is None:
        return None
    decision = event.payload.get("decision")
    if not isinstance(decision, dict):
        return None
    return decision


def _record_deferred_secondary_intents(
    *,
    session: Session,
    task: Task,
    task_service: TaskService,
    decision: dict[str, Any] | None,
) -> None:
    if decision is None:
        return
    secondary_intents = decision.get("secondary_intents")
    if not isinstance(secondary_intents, list):
        return
    memory_intents = [
        intent
        for intent in secondary_intents
        if isinstance(intent, dict) and intent.get("type") == "memory_candidate"
    ]
    if not memory_intents:
        return
    existing = session.scalar(
        select(TaskEvent.id)
        .where(
            TaskEvent.task_id == task.id,
            TaskEvent.type == TaskEventType.log,
            TaskEvent.payload["message"].as_string() == "secondary_intent_deferred",
            TaskEvent.payload["intent_type"].as_string() == "memory_candidate",
        )
        .limit(1)
    )
    if existing is not None:
        return
    task_service.append_event(
        task,
        TaskEventType.log,
        {
            "message": "secondary_intent_deferred",
            "intent_type": "memory_candidate",
            "route": _payload_str(memory_intents[0], "route") or "memory_confirmation",
            "objective": _payload_str(memory_intents[0], "objective")
            or "Memory candidate preserved for later confirmation.",
            "reason": "primary_task_execution_first",
        },
    )


def _tool_approval_prompt_system_prompt() -> str:
    return (
        "You write __AGENT_NAME__'s Slack approval request before a gated action. "
        "Return JSON only with a `text` string. Write as __AGENT_NAME__ in first person. "
        "Use the user's request to make the approval note specific and natural. "
        "Keep it under 450 characters before the reaction instruction. "
        "Do not say the action is already done. Do not mention backend, model, "
        "runtime, agent, tool ids, internal tool names, approval keys, account ids, "
        "or raw implementation details. Do not quote full code or secrets. "
        "For sandboxed_code_execution, mention a locked-down Python sandbox and "
        "that it has no network or host filesystem access. No emoji. No markdown "
        "headings. No em dash characters. End naturally, but do not include the "
        "final reaction instruction because the system appends it."
    )


def _tool_approval_prompt_payload(
    *,
    task: Task,
    request: ToolApprovalRequest,
    fallback: str,
) -> dict[str, Any]:
    return {
        "user_request": _compact_prompt_context(task.input, max_chars=1000),
        "approval_scope": request.scope.value,
        "risk": request.risk,
        "reason": request.reason,
        "argument_keys": list(request.argument_keys),
        "argument_summary": _tool_approval_argument_summary(request),
        "fallback_prompt": fallback,
        "required_final_line": TOOL_APPROVAL_REACTION_INSTRUCTION,
    }


def _tool_approval_argument_summary(
    request: ToolApprovalRequest,
) -> dict[str, str]:
    summary: dict[str, str] = {}
    for key in request.argument_keys:
        value = request.arguments.get(key)
        normalized_key = key.casefold()
        if normalized_key == "code" and isinstance(value, str):
            summary[key] = f"Python snippet, {len(value)} chars"
        elif _sensitive_argument_key(normalized_key):
            summary[key] = "redacted"
        elif isinstance(value, str) and len(value) <= 80:
            summary[key] = value
        elif isinstance(value, (bool, int, float)):
            summary[key] = str(value)
        else:
            summary[key] = "present"
    return summary


def _sensitive_argument_key(key: str) -> bool:
    return any(
        marker in key
        for marker in (
            "api",
            "auth",
            "credential",
            "key",
            "password",
            "secret",
            "token",
        )
    )


def _compact_prompt_context(value: str, *, max_chars: int) -> str:
    text = " ".join(value.split())
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 3].rstrip()}..."


def _tool_approval_prompt_from_completion(
    content: str | None,
    *,
    fallback: str,
) -> str | None:
    del fallback
    if not content:
        return None
    try:
        payload = json.loads(_extract_json_object(content))
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    raw_text = payload.get("text")
    if not isinstance(raw_text, str):
        return None
    body = _sanitize_tool_approval_prompt_body(raw_text)
    if body is None:
        return None
    return f"{body}\n\n{TOOL_APPROVAL_REACTION_INSTRUCTION}"


def _sanitize_tool_approval_prompt_body(text: str) -> str | None:
    lines = []
    for raw_line in text.splitlines():
        line = " ".join(raw_line.strip().strip('"').split())
        if not line:
            continue
        if line.startswith("#"):
            return None
        if "React with :white_check_mark:" in line:
            continue
        if ":no_entry_sign:" in line and "skip" in line.casefold():
            continue
        lines.append(line)
    body = "\n".join(lines).strip()
    if len(body) < 20 or len(body) > 600:
        return None
    lowered = body.casefold()
    blocked_terms = (
        "actual tool",
        "agent",
        "approval key",
        "backend",
        "cheap_fast",
        "code_exec",
        "guidelines",
        "i should",
        "llm",
        "model",
        "runtime",
        "the user",
        "tool id",
        "tool name",
        "tool_approval",
    )
    if any(term in lowered for term in blocked_terms):
        return None
    return body


def _extract_json_object(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found")
    return stripped[start : end + 1]


def _intent_needs_no_external_tools(decision: dict[str, Any]) -> bool:
    """Return whether the intent record is a cheap, no-tool conversational turn."""

    if _payload_str(decision, "model_tier") != "cheap":
        return False
    classification = _payload_str(decision, "classification")
    if classification not in {"task_request", "follow_up"}:
        return False
    if _truthy_bool(decision.get("needs_channel_context")):
        return False
    if _truthy_bool(decision.get("needs_thread_context")):
        return False
    if _truthy_bool(decision.get("needs_file_context")):
        return False
    likely_tools = _likely_tools(decision)
    return not likely_tools or likely_tools <= NO_EXTERNAL_TOOL_HINTS


def _intent_prefers_native_web_search(
    decision: dict[str, Any],
    input_text: str,
) -> bool:
    """Return whether native web search is the cheaper correct default."""

    classification = _payload_str(decision, "classification")
    if classification not in {"task_request", "follow_up"}:
        return False
    likely_tools = _likely_tools(decision)
    if not likely_tools or not likely_tools <= NATIVE_WEB_SEARCH_HINTS:
        return False
    lowered = input_text.casefold()
    return not any(trigger in lowered for trigger in EXTERNAL_WEB_TOOL_TRIGGERS)


def _intent_prefers_native_slack_context(decision: dict[str, Any]) -> bool:
    """Return whether local Slack context tools are sufficient."""

    classification = _payload_str(decision, "classification")
    if classification not in {"task_request", "follow_up"}:
        return False
    likely_tools = _likely_tools(decision)
    return bool(likely_tools) and likely_tools <= NATIVE_SLACK_CONTEXT_HINTS


def _truthy_bool(value: object) -> bool:
    return isinstance(value, bool) and value


def _payload_str(payload: Mapping[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if isinstance(value, str) and value:
        return value
    return None


def _likely_tools(decision: Mapping[str, Any]) -> set[str]:
    value = decision.get("likely_tools")
    if not isinstance(value, list):
        return set()
    return {item for item in value if isinstance(item, str) and item}


EXTERNAL_TOOL_SKIP_CLASSIFICATIONS = frozenset(
    {
        "ambient_observation",
        "cancel_or_retry",
        "clarification",
        "ignore",
        "memory_candidate",
        "third_person_reference",
    }
)

WORKER_TASK_CLASSIFICATIONS = frozenset(
    {
        "follow_up",
        "task_request",
    }
)

NO_EXTERNAL_TOOL_HINTS = frozenset(
    {
        "capability_lookup",
        "describe_tools",
        "list_capabilities",
        "list_integrations",
        "native_tool_registry",
        "tool_metadata_lookup",
        "tool_registry",
    }
)

NATIVE_WEB_SEARCH_HINTS = frozenset(
    {
        "current_research",
        "web_search",
    }
)

NATIVE_SLACK_CONTEXT_HINTS = native_slack_context_hint_names()

LINEAR_TASK_LOOKUP_WORDS = frozenset(
    {
        "assigned",
        "issue",
        "issues",
        "open",
        "task",
        "tasks",
        "todo",
        "todos",
    }
)

EXTERNAL_WEB_TOOL_TRIGGERS = frozenset(
    {
        "crawl",
        "extract",
        "firecrawl",
        "scrape",
        "url",
        "website",
    }
)
