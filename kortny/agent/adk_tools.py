"""ADK tool adapters for Kortny's provider-neutral tool registry."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable
from typing import Any

from google.adk.tools.base_tool import BaseTool
from google.adk.tools.base_toolset import BaseToolset
from google.adk.tools.tool_context import ToolContext
from google.genai import types as genai_types
from sqlalchemy import select
from sqlalchemy.orm import Session

from kortny.agent.coordinator import (
    _classify_recoverable_result_error,
    _output_shape,
    _recoverable_tool_error_result,
    _recoverable_tool_result,
    _tool_result_payload,
    _tool_result_prompt_payload,
    _with_classified_error,
)
from kortny.agent.execution import normalized_tool_args_hash
from kortny.approvals import (
    TOOL_APPROVAL_DECISION_MESSAGE,
    TOOL_APPROVAL_REQUIRED_MESSAGE,
    ToolApprovalPolicy,
    ToolApprovalRequest,
    ToolApprovalRequired,
    approval_key,
)
from kortny.db.models import Task, TaskEvent, TaskEventType
from kortny.observability import log_observation
from kortny.tasks import TaskService
from kortny.tools import Tool, ToolRegistry
from kortny.tools.types import JsonObject, RecoverableToolError, ToolResult

logger = logging.getLogger(__name__)
ADK_TOOL_TURN = 1
ADK_TOOL_STEP_ID = "adk_tool_call"


class KortnyAdkTool(BaseTool):
    """Expose one Kortny registry tool as an ADK BaseTool."""

    def __init__(
        self,
        *,
        tool: Tool,
        task: Task,
        session: Session,
        task_service: TaskService,
        approval_policy: ToolApprovalPolicy | None = None,
        tool_result_prompt_max_chars: int = 8000,
    ) -> None:
        super().__init__(name=tool.name, description=tool.description)
        self.tool = tool
        self.task = task
        self.session = session
        self.task_service = task_service
        self.approval_policy = approval_policy or ToolApprovalPolicy()
        self.tool_result_prompt_max_chars = tool_result_prompt_max_chars
        # ADK treats tools with a sync `func` attribute differently and keeps
        # run_async on the runner loop for non-FunctionTool tools. Kortny tools
        # use the task SQLAlchemy session, so keep execution in this worker
        # thread instead of ADK's background tool thread.
        self.func = _sync_execution_marker

    def _get_declaration(self) -> genai_types.FunctionDeclaration:
        return genai_types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters_json_schema=self.tool.parameters,
        )

    async def run_async(
        self, *, args: dict[str, Any], tool_context: ToolContext
    ) -> JsonObject:
        arguments = dict(args)
        tool_call_id = tool_context.function_call_id or f"adk-{uuid.uuid4()}"
        normalized_args_hash = normalized_tool_args_hash(arguments)

        self._raise_if_approval_required(
            arguments=arguments,
            tool_call_id=tool_call_id,
            normalized_args_hash=normalized_args_hash,
        )
        self._record_tool_call(
            arguments=arguments,
            tool_call_id=tool_call_id,
            normalized_args_hash=normalized_args_hash,
        )
        started = time.perf_counter()
        recoverable_error = None
        result: ToolResult
        try:
            try:
                result = self.tool.invoke(arguments)
            except RecoverableToolError as exc:
                recoverable_error = exc
                classification = _classify_recoverable_result_error(
                    {"error": exc.to_payload()}
                )
                if classification is None:
                    raise
                result = _recoverable_tool_error_result(
                    arguments=arguments,
                    error=exc,
                    classification=classification,
                )
            classification = _classify_recoverable_result_error(result.output)
            if classification is not None:
                result = _with_classified_error(result, classification)
        except Exception as exc:
            latency_ms = _latency_ms(started)
            result = _recoverable_exception_tool_result(
                arguments=arguments,
                tool_name=self.name,
                error=exc,
            )
            result_payload = _tool_result_payload(self.name, result)
            prompt_result_payload, compaction_payload = _tool_result_prompt_payload(
                self.name,
                result_payload,
                max_chars=self.tool_result_prompt_max_chars,
            )
            self.task_service.append_event(
                self.task,
                TaskEventType.tool_result,
                {
                    "turn": ADK_TOOL_TURN,
                    "tool_call_id": tool_call_id,
                    "tool": self.name,
                    "runtime": "adk",
                    "step_id": ADK_TOOL_STEP_ID,
                    "normalized_args_hash": normalized_args_hash,
                    "attempt_no": 1,
                    "latency_ms": latency_ms,
                    "output_shape": _output_shape(result.output),
                    "artifact_count": len(result.artifacts),
                    "recoverable": _recoverable_tool_result(result.output),
                    **result_payload,
                },
            )
            if compaction_payload is not None:
                self.task_service.append_event(
                    self.task,
                    TaskEventType.log,
                    {
                        "message": "tool_result_compacted",
                        "runtime": "adk",
                        "turn": ADK_TOOL_TURN,
                        "tool_call_id": tool_call_id,
                        "tool": self.name,
                        **compaction_payload,
                    },
                )
            log_observation(
                logger,
                "adk_tool_call_failed",
                level=logging.WARNING,
                task=self.task,
                tool_call_id=tool_call_id,
                tool=self.name,
                latency_ms=latency_ms,
                error_type=type(exc).__name__,
                error_summary=str(exc),
                recoverable=True,
            )
            prompt_result_payload["recoverable_error"] = True
            return prompt_result_payload

        latency_ms = _latency_ms(started)
        result_payload = _tool_result_payload(self.name, result)
        prompt_result_payload, compaction_payload = _tool_result_prompt_payload(
            self.name,
            result_payload,
            max_chars=self.tool_result_prompt_max_chars,
        )
        self.task_service.append_event(
            self.task,
            TaskEventType.tool_result,
            {
                "turn": ADK_TOOL_TURN,
                "tool_call_id": tool_call_id,
                "tool": self.name,
                "runtime": "adk",
                "step_id": ADK_TOOL_STEP_ID,
                "normalized_args_hash": normalized_args_hash,
                "attempt_no": 1,
                "latency_ms": latency_ms,
                "output_shape": _output_shape(result.output),
                "artifact_count": len(result.artifacts),
                "recoverable": _recoverable_tool_result(result.output),
                **result_payload,
            },
        )
        if compaction_payload is not None:
            self.task_service.append_event(
                self.task,
                TaskEventType.log,
                {
                    "message": "tool_result_compacted",
                    "runtime": "adk",
                    "turn": ADK_TOOL_TURN,
                    "tool_call_id": tool_call_id,
                    "tool": self.name,
                    **compaction_payload,
                },
            )
        log_observation(
            logger,
            "adk_tool_call_completed",
            task=self.task,
            tool_call_id=tool_call_id,
            tool=self.name,
            latency_ms=latency_ms,
            output_shape=_output_shape(result.output),
            artifact_count=len(result.artifacts),
            recoverable=_recoverable_tool_result(result.output),
            cost_usd=str(result.cost_usd),
        )
        if recoverable_error is not None:
            prompt_result_payload["recoverable_error"] = True
        return prompt_result_payload

    def _record_tool_call(
        self,
        *,
        arguments: JsonObject,
        tool_call_id: str,
        normalized_args_hash: str,
    ) -> None:
        self.task_service.append_event(
            self.task,
            TaskEventType.tool_call,
            {
                "turn": ADK_TOOL_TURN,
                "tool_call_id": tool_call_id,
                "tool": self.name,
                "runtime": "adk",
                "step_id": ADK_TOOL_STEP_ID,
                "normalized_args_hash": normalized_args_hash,
                "attempt_no": 1,
                "argument_keys": sorted(arguments),
                "arguments": arguments,
            },
        )
        log_observation(
            logger,
            "adk_tool_call_started",
            task=self.task,
            tool_call_id=tool_call_id,
            tool=self.name,
            argument_keys=sorted(arguments),
        )

    def _raise_if_approval_required(
        self,
        *,
        arguments: JsonObject,
        tool_call_id: str,
        normalized_args_hash: str,
    ) -> None:
        requirement = self.approval_policy.requirement_for(self.tool, arguments)
        if not requirement.required:
            return

        key = approval_key(self.name, normalized_args_hash)
        if self._approval_is_granted(key):
            self.task_service.append_event(
                self.task,
                TaskEventType.log,
                {
                    "message": "tool_approval_previously_granted",
                    "runtime": "adk",
                    "turn": ADK_TOOL_TURN,
                    "tool_call_id": tool_call_id,
                    "tool": self.name,
                    "step_id": ADK_TOOL_STEP_ID,
                    "approval_key": key,
                    "normalized_args_hash": normalized_args_hash,
                    "attempt_no": 1,
                },
            )
            return

        request = ToolApprovalRequest(
            approval_key=key,
            tool_name=self.name,
            tool_call_id=tool_call_id,
            normalized_args_hash=normalized_args_hash,
            argument_keys=tuple(sorted(arguments)),
            scope=requirement.scope,
            reason=requirement.reason,
            risk=requirement.risk,
            arguments=arguments,
        )
        self.task_service.append_event(
            self.task,
            TaskEventType.log,
            {
                "message": TOOL_APPROVAL_REQUIRED_MESSAGE,
                "runtime": "adk",
                "turn": ADK_TOOL_TURN,
                "step_id": ADK_TOOL_STEP_ID,
                "request": request.to_payload(),
            },
        )
        raise ToolApprovalRequired(request)

    def _approval_is_granted(self, key: str) -> bool:
        event = self.session.scalar(
            select(TaskEvent)
            .where(
                TaskEvent.task_id == self.task.id,
                TaskEvent.type == TaskEventType.log,
                TaskEvent.payload["message"].as_string()
                == TOOL_APPROVAL_DECISION_MESSAGE,
                TaskEvent.payload["approval_key"].as_string() == key,
            )
            .order_by(TaskEvent.seq.desc())
            .limit(1)
        )
        return event is not None and event.payload.get("decision") == "approved"


class KortnyRegistryToolset(BaseToolset):
    """Lazy ADK toolset backed by Kortny's scoped tool registry."""

    def __init__(
        self,
        *,
        registry_factory: Callable[[], ToolRegistry],
        task: Task,
        session: Session,
        task_service: TaskService,
        approval_policy: ToolApprovalPolicy | None = None,
        tool_result_prompt_max_chars: int = 8000,
    ) -> None:
        super().__init__()
        self.registry_factory = registry_factory
        self.task = task
        self.session = session
        self.task_service = task_service
        self.approval_policy = approval_policy
        self.tool_result_prompt_max_chars = tool_result_prompt_max_chars
        self._registry: ToolRegistry | None = None

    async def get_tools(self, readonly_context: Any | None = None) -> list[BaseTool]:
        del readonly_context
        registry = self._registry
        if registry is None:
            registry = self.registry_factory()
            self._registry = registry
            self.task_service.append_event(
                self.task,
                TaskEventType.log,
                {
                    "message": "adk_lazy_toolset_loaded",
                    "runtime": "adk",
                    "tool_count": len(registry.names()),
                    "tool_names": list(registry.names()),
                },
            )
            log_observation(
                logger,
                "adk_lazy_toolset_loaded",
                task=self.task,
                tool_count=len(registry.names()),
                tool_names=list(registry.names()),
            )

        return adk_tools_from_registry(
            registry,
            task=self.task,
            session=self.session,
            task_service=self.task_service,
            approval_policy=self.approval_policy,
            tool_result_prompt_max_chars=self.tool_result_prompt_max_chars,
        )


def adk_tools_from_registry(
    registry: ToolRegistry,
    *,
    task: Task,
    session: Session,
    task_service: TaskService,
    approval_policy: ToolApprovalPolicy | None = None,
    tool_result_prompt_max_chars: int = 8000,
) -> list[BaseTool]:
    """Build ADK tool adapters for every selected Kortny registry tool."""

    return [
        KortnyAdkTool(
            tool=registry.get(name),
            task=task,
            session=session,
            task_service=task_service,
            approval_policy=approval_policy,
            tool_result_prompt_max_chars=tool_result_prompt_max_chars,
        )
        for name in registry.names()
    ]


def _sync_execution_marker() -> None:
    """Marker used only for ADK's sync-tool detection."""


def _recoverable_exception_tool_result(
    *,
    arguments: JsonObject,
    tool_name: str,
    error: Exception,
) -> ToolResult:
    message = str(error) or type(error).__name__
    error_payload: JsonObject = {
        "code": _recoverable_exception_code(error, message),
        "message": message,
        "recoverable": True,
        "hint": (
            "Treat this tool call as unavailable for this run. Try another "
            "available tool if one can answer the request; otherwise explain "
            "the blocker clearly."
        ),
        "details": {
            "tool": tool_name,
            "error_type": type(error).__name__,
            "attempted_argument_keys": sorted(arguments),
        },
    }
    classification = _classify_recoverable_result_error({"error": error_payload})
    if classification is None:
        return ToolResult(
            output={
                "successful": False,
                "attempted_argument_keys": sorted(arguments),
                "error": error_payload,
            }
        )
    return ToolResult(
        output={
            "successful": False,
            "attempted_argument_keys": sorted(arguments),
            "error": _with_classified_error(
                ToolResult(output={"error": error_payload}),
                classification,
            ).output["error"],
        }
    )


def _recoverable_exception_code(error: Exception, message: str) -> str:
    normalized = f"{type(error).__name__} {message}".casefold()
    if any(token in normalized for token in ("timeout", "timed out")):
        return "tool_execution_timeout"
    return "tool_execution_failed"


def _latency_ms(started: float) -> int:
    return max(0, int((time.perf_counter() - started) * 1000))
