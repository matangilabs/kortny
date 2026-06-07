"""Default worker executor that runs the agent coordinator."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol, cast

from slack_sdk import WebClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from kortny.agent.coordinator import DEFAULT_SYSTEM_PROMPT, AgentRunResult
from kortny.agent.runtime import CustomAgentRuntime
from kortny.agent.thread_context import ThreadTranscriptProvider
from kortny.approvals import (
    TOOL_APPROVAL_PROMPT_PURPOSE,
    ToolApprovalRequired,
    approval_prompt_text,
)
from kortny.composio import ComposioClient
from kortny.composio.provider import ComposioExternalToolProvider
from kortny.config import Settings, load_settings
from kortny.db.models import Artifact, Task, TaskEvent, TaskEventType, TaskStatus
from kortny.db.models import LLMProvider as DbLLMProvider
from kortny.execution import task_workspace
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
from kortny.llm.routing import ModelRouteTier, effective_intent_decision
from kortny.llm.runtime_config import (
    RuntimeModelSelection,
    create_provider_for_selection,
    select_runtime_model,
)
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
from kortny.slack.comments import (
    ArtifactCommentGenerator,
    LLMArtifactCommentGenerator,
    generate_artifact_comment,
)
from kortny.slack.formatting import normalize_slack_mrkdwn
from kortny.slack.humanizer import (
    LLMResponseSynthesizer,
    ResponseSynthesizer,
    StaticResponseSynthesizer,
    strip_internal_response_preamble,
    synthesize_response,
)
from kortny.slack.membership import SlackChannelMembershipService
from kortny.slack.outbox import SlackSideEffectOutbox, slack_reaction_key
from kortny.slack.posting import SlackPostingClient
from kortny.slack.reactions import (
    ACK_REACTION_ADDED_MESSAGE,
    ACK_REACTION_REMOVE_FAILED_MESSAGE,
    ACK_REACTION_REMOVED_MESSAGE,
    LibraryReactionProvider,
    ReactionProvider,
)
from kortny.slack.thread_context import SlackThreadTranscriptProvider
from kortny.tasks import TaskCancelledError, TaskService
from kortny.tool_selection import (
    ExternalToolProvider,
    HeuristicToolSelector,
    LLMToolSelector,
    ToolCatalogService,
    ToolSelection,
    ToolSelectionResult,
    ToolSelector,
    compact_tool_cards,
)
from kortny.tools import (
    DescribeToolsTool,
    ForgetFactTool,
    InspectMemoryTool,
    JsonObject,
    ListIntegrationsTool,
    ObservationChannelHistoryCache,
    PdfGeneratorTool,
    QueryWorkspaceGraphTool,
    RecallFactTool,
    RememberFactTool,
    ResolveSlackIdentityTool,
    SearchObservedSlackHistoryTool,
    SlackChannelHistoryTool,
    SlackFileReadTool,
    Tool,
    ToolRegistry,
    WebSearchTool,
)
from kortny.tools.schedules import (
    CancelScheduleTool,
    CreateScheduleTool,
    GetScheduleTool,
    ListSchedulesTool,
    PauseScheduleTool,
    ResumeScheduleTool,
    UpdateScheduleTool,
)
from kortny.witness import (
    WITNESS_OPPORTUNITY_CANDIDATES_PROJECTED_MESSAGE,
    WitnessChannelProfileExtractor,
    WitnessOpportunityService,
    WitnessTaskResponseExtractor,
)
from kortny.workflow.handoff import evaluate_runtime_handoff
from kortny.workflow.planning_classifier import classify_planned_workflow

GENERIC_FAILURE_TEXT = (
    "Something went wrong while I was working on this. Please try again soon."
)
MEMORY_CONFIRMATION_PURPOSE = "memory_confirmation"
PLANNED_WORKFLOW_PROGRESS_PURPOSE = "planned_progress_start"
PLANNED_WORKFLOW_PROGRESS_TEXT = "Hang on, I'll check."
PLANNED_WORKFLOW_PROGRESS_PROMPT_NAME = "kortny.planned_progress_status"
ADK_QUICK_FINAL_AUTHORS = frozenset(
    {
        "quick_response_agent",
        "kortny_root_orchestrator",
    }
)
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
        reaction_provider: ReactionProvider | None = None,
        tool_selector: ToolSelector | None = None,
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
        self.tool_selector = tool_selector
        self.composio_client = composio_client
        self.response_synthesizer = response_synthesizer

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
            self._post_failure_notice(
                settings=settings,
                session=session,
                task=task,
                task_service=task_service,
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

    def _run_agent_runtime(
        self,
        *,
        settings: Settings,
        session: Session,
        task: Task,
        task_service: TaskService,
        working_dir: Path,
    ) -> AgentRunResult:
        planned_workflow_candidate = False
        planned_workflow_payload: JsonObject | None = None
        try:
            planned_workflow = classify_planned_workflow(
                task=task,
                events=_task_events(session, task),
            )
            planned_workflow_candidate = planned_workflow.planned_candidate
            planned_workflow_payload = planned_workflow.to_payload()
            task_service.append_event(
                task,
                TaskEventType.log,
                planned_workflow_payload,
            )
            log_observation(
                logger,
                "planned_workflow_classified",
                task=task,
                classifier="rules_plus_intent_metadata",
                classifier_version="hig_179_slice_0",
                behavior="observe_only",
                route=planned_workflow.route.value,
                planned_candidate=planned_workflow.planned_candidate,
                confidence=planned_workflow.confidence,
                estimated_subtask_count=planned_workflow.estimated_subtask_count,
                reason_codes=list(planned_workflow.reason_codes),
                detected_integrations=list(planned_workflow.detected_integrations),
                likely_tools=list(planned_workflow.likely_tools),
                needs_context=list(planned_workflow.needs_context),
            )
        except Exception as exc:
            task_service.append_event(
                task,
                TaskEventType.error,
                {
                    "message": "planned_workflow_classifier_failed",
                    "classifier": "rules_plus_intent_metadata",
                    "classifier_version": "hig_179_slice_0",
                    "behavior": "observe_only",
                    "fallback_policy": "inline_on_low_confidence_or_classifier_failure",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            log_observation(
                logger,
                "planned_workflow_classifier_failed",
                task=task,
                classifier="rules_plus_intent_metadata",
                classifier_version="hig_179_slice_0",
                behavior="observe_only",
                error_type=type(exc).__name__,
                error_summary=str(exc)[:500],
            )
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
            thread_transcript_provider=self._build_thread_transcript_provider(settings),
        ).run(task)

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
        planned_workflow_payload = _latest_payload_event(
            events,
            message="planned_workflow_classified",
        )
        planned_candidate = None
        if planned_workflow_payload is not None:
            raw_planned_candidate = planned_workflow_payload.get("planned_candidate")
            if isinstance(raw_planned_candidate, bool):
                planned_candidate = raw_planned_candidate

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
            planned_classifier_route=_routing_payload_str(
                planned_workflow_payload,
                "route",
            ),
            planned_candidate=planned_candidate,
        )
        promotion = SemanticRouterPromotionGate().evaluate(decision)
        metadata["promotion_gate"] = promotion.to_payload()
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
        pdf_generator = PdfGeneratorTool(
            working_dir=working_dir,
            session=session,
            task_id=task.id,
            task_service=task_service,
        )
        slack_channel_history = SlackChannelHistoryTool(
            self._build_slack_history_client(settings),
            default_channel_id=task.slack_channel_id,
            cache=ObservationChannelHistoryCache(
                session,
                installation_id=task.installation_id,
            ),
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
        inspect_memory = InspectMemoryTool(service=memory_service, task=task)
        forget_fact = ForgetFactTool(service=memory_service, task=task)
        query_workspace_graph = QueryWorkspaceGraphTool(session=session, task=task)
        schedule_tools: tuple[Tool, ...] = (
            ListSchedulesTool(session=session, task=task),
            GetScheduleTool(session=session, task=task),
            CreateScheduleTool(
                session=session,
                task=task,
                task_service=task_service,
            ),
            UpdateScheduleTool(
                session=session,
                task=task,
                task_service=task_service,
            ),
            PauseScheduleTool(
                session=session,
                task=task,
                task_service=task_service,
            ),
            ResumeScheduleTool(
                session=session,
                task=task,
                task_service=task_service,
            ),
            CancelScheduleTool(
                session=session,
                task=task,
                task_service=task_service,
            ),
        )
        native_tools: list[Tool] = [
            pdf_generator,
            slack_channel_history,
            SearchObservedSlackHistoryTool(session=session, task=task),
            ResolveSlackIdentityTool(session=session, task=task),
            slack_file_read,
            remember_fact,
            recall_fact,
            inspect_memory,
            forget_fact,
            query_workspace_graph,
            *schedule_tools,
        ]
        if web_search is not None:
            native_tools.insert(0, web_search)
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
        describe_tools = DescribeToolsTool(
            session=session,
            task=task,
            native_tools=(),
        )
        list_integrations = ListIntegrationsTool(
            session=session,
            task=task,
            native_tools=(),
        )
        native_inventory_tools = (
            *tuple(native_tools),
            describe_tools,
            list_integrations,
        )
        describe_tools.native_tools = native_inventory_tools
        list_integrations.native_tools = native_inventory_tools
        native_tools = [
            cast(Tool, scoped_tool) for scoped_tool in native_scope.selected_tools
        ]
        native_tools.append(describe_tools)
        native_tools.append(list_integrations)
        _record_deferred_secondary_intents(
            session=session,
            task=task,
            task_service=task_service,
            decision=raw_intent_decision,
        )
        skip_external_reason = _external_tool_skip_reason(
            session,
            task,
            decision=raw_intent_decision,
            native_web_search_available=web_search is not None,
        )
        if skip_external_reason is not None:
            task_service.append_event(
                task,
                TaskEventType.log,
                {
                    "message": "external_tool_selection_skipped",
                    **skip_external_reason,
                },
            )
            log_observation(
                logger,
                "external_tool_selection_skipped",
                task=task,
                reason=skip_external_reason["reason"],
                classification=skip_external_reason.get("classification"),
            )
            return ToolRegistry(native_tools)

        external_providers = self._build_external_tool_providers(
            settings=settings,
            session=session,
            task=task,
        )
        external_tools = [
            tool for provider in external_providers for tool in provider.runtime_tools()
        ]
        external_cards = ToolCatalogService().external_cards(external_providers)
        tools = self._select_runtime_tools(
            settings=settings,
            session=session,
            task=task,
            task_service=task_service,
            native_tools=native_tools,
            external_tools=external_tools,
            external_cards=external_cards,
        )
        return ToolRegistry(tools)

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
            return WebSearchTool.from_settings(settings)
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

    def _select_runtime_tools(
        self,
        *,
        settings: Settings,
        session: Session,
        task: Task,
        task_service: TaskService,
        native_tools: list[Tool],
        external_tools: list[Tool],
        external_cards: tuple[Any, ...],
    ) -> list[Tool]:
        if not external_tools:
            return native_tools

        catalog = ToolCatalogService()
        native_cards = catalog.native_cards(native_tools)
        if not external_cards:
            return native_tools

        selector_cards, compaction = compact_tool_cards(
            task_input=_tool_selection_task_input(
                session=session,
                task=task,
                base_input=task.input,
            ),
            cards=external_cards,
            max_candidates=settings.tool_selector_max_external_candidates,
        )
        if compaction.compacted:
            task_service.append_event(
                task,
                TaskEventType.log,
                {
                    "message": "tool_catalog_compacted",
                    **compaction.to_payload(),
                },
            )
            log_observation(
                logger,
                "tool_catalog_compacted",
                task=task,
                original_candidate_count=compaction.original_candidate_count,
                selected_candidate_count=compaction.selected_candidate_count,
                omitted_candidate_count=compaction.omitted_candidate_count,
                max_candidates=compaction.max_candidates,
                reason=compaction.reason,
            )

        selector = self.tool_selector or self._build_tool_selector(
            settings=settings,
            session=session,
            task=task,
            task_service=task_service,
        )
        try:
            selection = selector.select(
                task_id=task.id,
                task_input=_tool_selection_task_input(
                    session=session,
                    task=task,
                    base_input=task.input,
                ),
                native_cards=native_cards,
                external_cards=selector_cards,
            )
        except Exception as exc:
            logger.exception("tool selector failed task_id=%s", task.id)
            task_service.append_event(
                task,
                TaskEventType.log,
                {
                    "message": "tool_selection_failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "fallback": "heuristic_tool_selector",
                },
            )
            selection = HeuristicToolSelector().select(
                task_id=task.id,
                task_input=_tool_selection_task_input(
                    session=session,
                    task=task,
                    base_input=task.input,
                ),
                native_cards=native_cards,
                external_cards=selector_cards,
            )
        selection = _expand_related_tool_selection(
            task_input=task.input,
            selection=selection,
            selector_cards=selector_cards,
        )

        self._record_tool_selection(
            task=task,
            task_service=task_service,
            selection=selection,
            candidate_count=len(external_cards),
            selector_candidate_count=len(selector_cards),
        )
        selected_external_names = set(selection.selected_names)
        suppressed_native_names = set(selection.suppressed_native_tools)
        return [
            tool for tool in native_tools if tool.name not in suppressed_native_names
        ] + [tool for tool in external_tools if tool.name in selected_external_names]

    def _build_tool_selector(
        self,
        *,
        settings: Settings,
        session: Session,
        task: Task,
        task_service: TaskService,
    ) -> ToolSelector:
        if self.llm_provider is not None:
            return HeuristicToolSelector()

        model_route = ModelRouter(settings).route_for_tier(
            ModelRouteTier.cheap_fast,
            reason="tool_selection",
        )
        selection = self._select_runtime_model(
            settings=settings,
            session=session,
            task=task,
            model_route=model_route,
        )
        return LLMToolSelector(
            LLMService(
                session=session,
                provider=create_provider_for_selection(
                    settings=settings,
                    selection=selection,
                ),
                provider_name=self.provider_name or selection.provider_name,
                task_service=task_service,
                model_route=selection.model_route,
            ),
            max_prompt_chars=settings.tool_selector_max_prompt_chars,
        )

    def _record_tool_selection(
        self,
        *,
        task: Task,
        task_service: TaskService,
        selection: ToolSelectionResult,
        candidate_count: int,
        selector_candidate_count: int,
    ) -> None:
        payload = {
            "message": "tool_selection_completed",
            "candidate_count": candidate_count,
            "selector_candidate_count": selector_candidate_count,
            "selected_tools": [
                {
                    "registry_name": item.registry_name,
                    "confidence": item.confidence,
                    "reason": item.reason,
                }
                for item in selection.selected_tools
            ],
            "suppressed_native_tools": list(selection.suppressed_native_tools),
            "rejected_tools": [
                {
                    "registry_name": item.registry_name,
                    "confidence": item.confidence,
                    "reason": item.reason,
                }
                for item in selection.rejected_tools
            ],
            "route_reason": selection.route_reason,
            "fallback_used": selection.fallback_used,
            **(
                {
                    "selector_prompt_chars": selection.prompt_chars,
                    "selector_prompt_char_budget": selection.prompt_char_budget,
                }
                if selection.prompt_chars is not None
                and selection.prompt_char_budget is not None
                else {}
            ),
            **(
                {
                    "budget_omitted_candidate_names": list(
                        selection.budget_omitted_candidate_names
                    ),
                    "budget_omitted_candidate_count": len(
                        selection.budget_omitted_candidate_names
                    ),
                }
                if selection.budget_omitted_candidate_names
                else {}
            ),
        }
        task_service.append_event(task, TaskEventType.log, payload)
        task_service.append_event(
            task,
            TaskEventType.log,
            RoutingDecisionTrace(
                stage="tool_scope_selected",
                route_tier="tool_scope",
                source="tool_selector",
                candidate_tool_count=candidate_count,
                selector_candidate_count=selector_candidate_count,
                selected_tool_names=tuple(
                    item.registry_name for item in selection.selected_tools
                ),
                suppressed_tool_names=tuple(selection.suppressed_native_tools),
                reason=selection.route_reason,
                metadata={
                    "fallback_used": selection.fallback_used,
                    "rejected_tool_count": len(selection.rejected_tools),
                    "budget_omitted_candidate_count": len(
                        selection.budget_omitted_candidate_names
                    ),
                },
            ).to_payload(),
        )
        log_observation(
            logger,
            "tool_selection_completed",
            task=task,
            candidate_count=candidate_count,
            selector_candidate_count=selector_candidate_count,
            selected_tools=[item.registry_name for item in selection.selected_tools],
            suppressed_native_tools=list(selection.suppressed_native_tools),
            route_reason=selection.route_reason,
            fallback_used=selection.fallback_used,
            selector_prompt_chars=selection.prompt_chars,
            selector_prompt_char_budget=selection.prompt_char_budget,
            budget_omitted_candidate_count=len(
                selection.budget_omitted_candidate_names
            ),
        )

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
        tool_selection = _latest_payload_event(events, message="tool_selection_completed")
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

    def _build_external_tool_providers(
        self,
        *,
        settings: Settings,
        session: Session,
        task: Task,
    ) -> list[ExternalToolProvider]:
        providers: list[ExternalToolProvider] = []
        if settings.composio_api_key is not None and settings.composio_catalog_enabled:
            providers.append(
                ComposioExternalToolProvider(
                    session=session,
                    task=task,
                    client=self.composio_client
                    or ComposioClient(
                        api_key=settings.composio_api_key,
                        timeout_seconds=settings.composio_request_timeout_seconds,
                    ),
                    per_toolkit_limit=settings.composio_catalog_limit,
                )
            )
        return providers

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
            return
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
            if skip_humanizer_reason is not None:
                response_text = normalize_slack_mrkdwn(response_source)
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
                response_text = synthesize_response(
                    self._build_response_synthesizer(settings),
                    session=session,
                    task=task,
                    raw_text=response_source,
                    task_service=task_service,
                )
            poster.post_message(thread, response_text)
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
        return None

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
                approval_prompt_text(approval.request),
                purpose=TOOL_APPROVAL_PROMPT_PURPOSE,
            ),
        )

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
        model_route = ModelRouter(settings).route_for_tier(
            ModelRouteTier.cheap_fast,
            reason="planned_progress_status",
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

        llm = LLMService(
            session=session,
            provider=provider,
            provider_name=provider_name,
            task_service=task_service,
            model_route=model_route,
        )
        messages = (
            ChatMessage(
                role="system",
                content=(
                    "You write Kortny's first Slack progress update for a task that "
                    "will take more than a quick answer. Return JSON only with a "
                    "`message` string. Constraints: one sentence, 25-100 characters, "
                    "first person as Kortny, natural and coworker-like, specific "
                    "when it helps but never stiff, no user mentions, no emoji, "
                    "no markdown headings, "
                    "no backend/agent/model/runtime/tool language, no explanation of "
                    "what the user is asking, no em dashes, no phrase "
                    "`split this into workstreams`."
                ),
            ),
            ChatMessage(
                role="user",
                content=json.dumps(
                    {
                        "slack_surface": (
                            "dm"
                            if task.slack_channel_id.startswith("D")
                            else "channel"
                        ),
                        "user_request": task.input,
                    },
                    sort_keys=True,
                ),
            ),
        )
        try:
            completion = llm.complete(
                task_id=task.id,
                messages=messages,
                response_format={"type": "json_object"},
                prompt_name=PLANNED_WORKFLOW_PROGRESS_PROMPT_NAME,
            )
            text = _planned_progress_text_from_completion(completion.content)
        except Exception as exc:
            task_service.append_event(
                task,
                TaskEventType.error,
                {
                    "message": "planned_task_progress_synthesis_failed",
                    "runtime": "adk",
                    "phase": "started",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            log_observation(
                logger,
                "planned_task_progress_synthesis_failed",
                task=task,
                runtime="adk",
                phase="started",
                error_type=type(exc).__name__,
                error_summary=str(exc)[:500],
            )
            return PLANNED_WORKFLOW_PROGRESS_TEXT, "fallback"
        if text is None:
            return PLANNED_WORKFLOW_PROGRESS_TEXT, "fallback"
        return text, "llm"

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
                GENERIC_FAILURE_TEXT,
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
            f"I checked the scheduler and don't see any{target} schedules"
            f"{query_label}."
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
        _schedule_state_row(row)
        for row in schedule_rows[:5]
        if isinstance(row, dict)
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
    if _latest_payload_event(events, message="schedule_state_fast_path_completed") is not None:
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
    if "scheduled_or_recurring" in reason_set or "scheduled_task_identity" in reason_set:
        return "scheduler.create_or_run"
    if "quick_conversation" in reason_set:
        return "conversation.quick"
    if "write_or_destructive_intent" in reason_set:
        return "integration.write_or_approval"
    if "broad_research" in reason_set or "research_synthesis_work" in reason_set:
        return "research.synthesis"
    if "multi_source_synthesis" in reason_set:
        return "workspace.multi_source"
    if "integration_tool_work" in reason_set or "integration_scope_present" in reason_set:
        return "integration.read"
    return "task.general"


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


def _external_tool_skip_reason(
    session: Session,
    task: Task,
    *,
    decision: dict[str, Any] | None = None,
    native_web_search_available: bool = True,
) -> dict[str, Any] | None:
    if is_channel_assessment_task(session, task):
        return {
            "reason": "system_observe_channel_assessment",
            "classification": None,
        }

    raw_decision = (
        decision if decision is not None else _latest_intent_decision(session, task)
    )
    effective_decision = effective_intent_decision(raw_decision)
    if effective_decision is None:
        return None
    decision = dict(effective_decision)

    classification = _payload_str(decision, "classification")
    should_create_task = decision.get("should_create_task")
    if (
        should_create_task is False
        and classification not in WORKER_TASK_CLASSIFICATIONS
    ):
        return {
            "reason": "intent_should_not_create_task",
            "classification": classification,
        }

    if classification in EXTERNAL_TOOL_SKIP_CLASSIFICATIONS:
        return {
            "reason": "intent_classification",
            "classification": classification,
        }

    if _intent_needs_no_external_tools(decision):
        return {
            "reason": "intent_no_external_tools",
            "classification": classification,
        }

    if _intent_prefers_native_slack_context(decision):
        return {
            "reason": "intent_native_slack_context_only",
            "classification": classification,
        }

    if native_web_search_available and _intent_prefers_native_web_search(
        decision,
        task.input,
    ):
        return {
            "reason": "intent_native_web_search_only",
            "classification": classification,
        }

    return None


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


def _tool_selection_task_input(
    *,
    session: Session,
    task: Task,
    base_input: str,
) -> str:
    decision = effective_intent_decision(_latest_intent_decision(session, task))
    if not _should_include_prior_context_for_tool_selection(decision):
        return base_input
    if not task.slack_thread_ts:
        return base_input

    prior_tasks = tuple(
        session.scalars(
            select(Task)
            .where(
                Task.installation_id == task.installation_id,
                Task.slack_channel_id == task.slack_channel_id,
                Task.slack_thread_ts == task.slack_thread_ts,
                Task.id != task.id,
                Task.status == TaskStatus.succeeded,
                Task.result_summary.is_not(None),
            )
            .order_by(Task.created_at.desc(), Task.id.desc())
            .limit(2)
        )
    )
    if not prior_tasks:
        return base_input

    lines = [base_input, "", "Prior Slack thread context for tool selection:"]
    for prior in reversed(prior_tasks):
        lines.append(f"- User asked: {_compact_tool_selection_text(prior.input)}")
        if prior.result_summary:
            lines.append(
                f"  Kortny answered: "
                f"{_compact_tool_selection_text(prior.result_summary)}"
            )
    return "\n".join(lines)


def _should_include_prior_context_for_tool_selection(
    decision: Mapping[str, Any] | None,
) -> bool:
    if decision is None:
        return False
    classification = _payload_str(decision, "classification")
    if classification == "follow_up":
        return True
    return _truthy_bool(decision.get("needs_thread_context"))


def _compact_tool_selection_text(value: str, *, max_chars: int = 500) -> str:
    compact = " ".join(value.split())
    if len(compact) <= max_chars:
        return compact
    return f"{compact[: max_chars - 3].rstrip()}..."


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


def _expand_related_tool_selection(
    *,
    task_input: str,
    selection: ToolSelectionResult,
    selector_cards: tuple[Any, ...],
) -> ToolSelectionResult:
    """Add obvious same-toolkit read tools needed for multi-step requests."""

    if not _looks_like_linear_task_lookup(task_input):
        return selection

    selected_names = set(selection.selected_names)
    if not any(name.startswith("composio_linear_") for name in selected_names):
        return selection

    additions: list[ToolSelection] = []
    max_selected = 3
    for card in selector_cards:
        registry_name = getattr(card, "registry_name", "")
        if registry_name in selected_names:
            continue
        if getattr(card, "provider", None) != "composio":
            continue
        if getattr(card, "toolkit_slug", None) != "linear":
            continue
        if getattr(card, "side_effect", None) != "read":
            continue
        if not _linear_issue_lookup_tool(card):
            continue

        additions.append(
            ToolSelection(
                registry_name=registry_name,
                confidence=0.86,
                reason=(
                    "Related Linear read tool needed after project discovery for "
                    "a task/issue summary request."
                ),
            )
        )
        selected_names.add(registry_name)
        if len(selection.selected_tools) + len(additions) >= max_selected:
            break

    if not additions:
        return selection

    rejected = tuple(
        item
        for item in selection.rejected_tools
        if item.registry_name not in selected_names
    )
    route_reason = selection.route_reason
    if "related_tool_expansion" not in route_reason:
        route_reason = f"{route_reason}+related_tool_expansion"
    return replace(
        selection,
        selected_tools=selection.selected_tools + tuple(additions),
        suppressed_native_tools=selection.suppressed_native_tools,
        rejected_tools=rejected,
        route_reason=route_reason,
        fallback_used=selection.fallback_used,
    )


def _looks_like_linear_task_lookup(text: str) -> bool:
    words = _input_words(text)
    if "linear" not in words:
        return False
    return bool(words & LINEAR_TASK_LOOKUP_WORDS)


def _linear_issue_lookup_tool(card: Any) -> bool:
    haystack = " ".join(
        str(part)
        for part in (
            getattr(card, "registry_name", ""),
            getattr(card, "display_name", ""),
            getattr(card, "description", ""),
            " ".join(getattr(card, "capabilities", ()) or ()),
            " ".join(getattr(card, "tool_slugs", ()) or ()),
        )
        if part
    ).casefold()
    return (
        ("issue" in haystack or "task" in haystack)
        and ("list" in haystack or "search" in haystack)
        and "project" not in haystack
    )


def _input_words(text: str) -> set[str]:
    return {
        "".join(char for char in raw.casefold() if char.isalnum())
        for raw in text.replace("/", " ").replace("-", " ").replace("_", " ").split()
        if raw.strip()
    } - {""}


def _planned_progress_text_from_completion(content: str | None) -> str | None:
    if not content:
        return None
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    message = payload.get("message")
    if not isinstance(message, str):
        return None
    text = normalize_slack_mrkdwn(message).strip().strip('"')
    text = " ".join(text.split())
    if len(text) < 20 or len(text) > 220:
        return None
    lowered = text.casefold()
    blocked_terms = (
        "according to",
        "agent",
        "backend",
        "cheap_fast",
        "guidelines",
        "inference",
        "i should",
        "model",
        "runtime",
        "split this into",
        "the user",
        "tool",
        "user asks",
        "workstreams",
        "workstream",
    )
    if any(term in lowered for term in blocked_terms):
        return None
    return text


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

NATIVE_SLACK_CONTEXT_HINTS = frozenset(
    {
        "slack_channel_history",
        "search_observed_slack_history",
        "resolve_slack_identity",
    }
)

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
