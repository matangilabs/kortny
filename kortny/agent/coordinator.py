"""Plain MVP coordinator loop.

This is intentionally not ADK. The message and tool boundaries stay close to
OpenAI/OpenRouter chat completions so they can be adapted to ADK later.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from decimal import Decimal
from typing import Protocol

from sqlalchemy.orm import Session

from kortny.agent.context import (
    DEFAULT_KNOWN_FACTS_MAX_CHARS,
    DEFAULT_THREAD_CONTEXT_MAX_CHARS,
    DEFAULT_THREAD_CONTEXT_RECENT_TASKS,
    DEFAULT_THREAD_TRANSCRIPT_LIMIT,
    ContextAssembler,
)
from kortny.agent.thread_context import ThreadTranscriptProvider
from kortny.db.models import Task, TaskEventType
from kortny.llm import ChatMessage, Completion, ToolCall
from kortny.observability import (
    log_observation,
    record_span_exception,
    set_span_attributes,
    start_span,
)
from kortny.tasks import TaskService
from kortny.tools import ToolRegistry
from kortny.tools.types import (
    JsonObject,
    JsonSchema,
    RecoverableToolError,
    ToolArtifact,
    ToolResult,
)

DEFAULT_MAX_TURNS = 6
REQUESTED_PAGES_RE = re.compile(r"\b(\d{1,2})\s+pages?\b", re.I)
logger = logging.getLogger(__name__)
DEFAULT_SYSTEM_PROMPT = (
    "You are Kortny, a Slack-native AI coworker. Use the available tools when "
    "they are needed to complete the user's request. If the user asks for "
    "research and a PDF, search first and then generate the PDF artifact. "
    "If the user asks about an attached Slack file and the task input or prior "
    "context includes Slack file IDs, call slack_file_read before answering. "
    "For document revision requests, prefer the newest generated artifact "
    "listed in prior context over the original attachment. Preserve the source "
    "document title and filename lineage unless the user explicitly asks for a "
    "retitle, and use versioned filenames like source_v2.pdf, source_v3.pdf. "
    "If the current Slack message is a short answer to your immediately "
    "previous question, continue that pending task using the answer instead of "
    "treating it as a new standalone request. "
    "If the user explicitly asks you to remember a stable fact or preference, "
    "use remember_fact; it will ask for Slack confirmation before saving. Use "
    "inspect_memory when the user asks what you remember about them, this "
    "channel, this workspace, or why you believe a remembered fact. Use "
    "recall_fact only when you need one specific memory key. Use forget_fact "
    "when the user asks you to forget a remembered fact; tell them if no active "
    "memory matched. Never store secrets, API keys, tokens, passwords, or "
    "private keys in memory; if asked, explain that secrets belong in "
    "environment variables or a secret manager. "
    "When calling remember_fact, preserve every actionable detail from the "
    "user's request in both value and value_text: concrete names, firm names, "
    "colors, placement, formats, conditions, and exceptions. Prefer a slightly "
    "longer faithful memory proposal over a short lossy summary. "
    "If a tool result includes error.recoverable=true, keep working with the "
    "context and tool results already available. Retry only when you can change "
    "the arguments or content to address the tool feedback. If a required field "
    "or identifier is missing, first use a discovery, list, search, or history "
    "tool that can infer it. If no available tool can infer the missing input, "
    "ask one concise clarification. Never repeat the same failed tool call with "
    "the same arguments. Prefer cheap broad search for quick public discovery; "
    "use richer connected tools when the request needs authenticated workspace "
    "data, structured app actions, scraping, or deeper page extraction; combine "
    "tools when discovery and extraction are both useful. "
    "When answering with text, format for Slack mrkdwn rather than GitHub "
    "Markdown: use *bold*, <https://example.com|label> links, simple line-break "
    "lists, and avoid Markdown headings."
)


class LLMClient(Protocol):
    """Subset of LLMService used by the coordinator."""

    def complete(
        self,
        *,
        task_id: uuid.UUID,
        messages: Sequence[ChatMessage],
        tools: Sequence[JsonSchema] = (),
    ) -> Completion:
        """Complete one coordinator turn."""


class AgentLoopError(RuntimeError):
    """Raised when the coordinator cannot finish the task."""


class AgentTurnLimitError(AgentLoopError):
    """Raised when the coordinator exhausts its maximum LLM turns."""


@dataclass(frozen=True, slots=True)
class AgentRunResult:
    """Final coordinator result."""

    task_id: uuid.UUID
    result_summary: str
    turns: int
    artifact_count: int


class AgentCoordinator:
    """Calls the LLM, executes requested tools, and records task events."""

    def __init__(
        self,
        *,
        session: Session,
        llm: LLMClient,
        registry: ToolRegistry,
        task_service: TaskService | None = None,
        max_turns: int = DEFAULT_MAX_TURNS,
        system_prompt: str | None = None,
        thread_transcript_provider: ThreadTranscriptProvider | None = None,
        thread_context_max_chars: int = DEFAULT_THREAD_CONTEXT_MAX_CHARS,
        thread_context_recent_tasks: int = DEFAULT_THREAD_CONTEXT_RECENT_TASKS,
        thread_transcript_limit: int = DEFAULT_THREAD_TRANSCRIPT_LIMIT,
        known_facts_max_chars: int = DEFAULT_KNOWN_FACTS_MAX_CHARS,
        context_assembler: ContextAssembler | None = None,
    ) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        if thread_context_max_chars < 1:
            raise ValueError("thread_context_max_chars must be at least 1")
        if thread_context_recent_tasks < 1:
            raise ValueError("thread_context_recent_tasks must be at least 1")
        if thread_transcript_limit < 0:
            raise ValueError("thread_transcript_limit cannot be negative")
        if known_facts_max_chars < 0:
            raise ValueError("known_facts_max_chars cannot be negative")

        self.session = session
        self.llm = llm
        self.registry = registry
        self.task_service = task_service or TaskService(session)
        self.max_turns = max_turns
        self.context_assembler = context_assembler or ContextAssembler(
            session=session,
            task_service=self.task_service,
            system_prompt=system_prompt,
            thread_transcript_provider=thread_transcript_provider,
            thread_context_max_chars=thread_context_max_chars,
            thread_context_recent_tasks=thread_context_recent_tasks,
            thread_transcript_limit=thread_transcript_limit,
            known_facts_max_chars=known_facts_max_chars,
        )

    def run(self, task: Task | uuid.UUID) -> AgentRunResult:
        """Run the coordinator until final text or a produced artifact."""

        task_obj = self._resolve_task(task)
        messages = self._initial_messages(task_obj)
        schemas = self.registry.schemas()
        artifact_count = 0

        self._append_log(
            task_obj,
            "agent_started",
            {"tool_names": list(self.registry.names())},
        )

        for turn in range(1, self.max_turns + 1):
            self.task_service.raise_if_cancelled(task_obj, phase=f"before_turn_{turn}")
            completion = self._complete_turn(task_obj, messages, schemas, turn)
            self.task_service.raise_if_cancelled(
                task_obj, phase=f"after_turn_{turn}_completion"
            )
            messages.append(
                ChatMessage(
                    role="assistant",
                    content=completion.content,
                    tool_calls=completion.tool_calls,
                )
            )

            if not completion.tool_calls:
                return self._finish_with_text(task_obj, completion.content, turn)

            turn_artifacts = self._invoke_tool_calls(
                task_obj=task_obj,
                messages=messages,
                completion=completion,
                turn=turn,
            )
            artifact_count += turn_artifacts
            if turn_artifacts:
                return self._finish_with_artifacts(
                    task_obj,
                    turn=turn,
                    artifact_count=artifact_count,
                )

        error = AgentTurnLimitError(
            f"Agent exceeded max_turns={self.max_turns} for task {task_obj.id}"
        )
        self._append_error(task_obj, error, {"max_turns": self.max_turns})
        raise error

    def _complete_turn(
        self,
        task: Task,
        messages: list[ChatMessage],
        schemas: Sequence[JsonSchema],
        turn: int,
    ) -> Completion:
        self._append_log(
            task,
            "agent_llm_turn_started",
            {
                "turn": turn,
                "message_count": len(messages),
                "tool_count": len(schemas),
            },
        )
        try:
            completion = self.llm.complete(
                task_id=task.id,
                messages=messages,
                tools=schemas,
            )
        except Exception as exc:
            self._append_error(task, exc, {"turn": turn, "phase": "llm_complete"})
            raise

        self._append_log(
            task,
            "agent_llm_turn_completed",
            {
                "turn": turn,
                "response_id": completion.response_id,
                "model": completion.model,
                "has_content": bool(completion.content),
                "tool_calls": [
                    {"id": tool_call.id, "name": tool_call.name}
                    for tool_call in completion.tool_calls
                ],
            },
        )
        return completion

    def _invoke_tool_calls(
        self,
        *,
        task_obj: Task,
        messages: list[ChatMessage],
        completion: Completion,
        turn: int,
    ) -> int:
        artifact_count = 0
        for tool_call in completion.tool_calls:
            self.task_service.raise_if_cancelled(
                task_obj, phase=f"before_tool_{tool_call.name}"
            )
            arguments = self._tool_arguments(task_obj, tool_call)
            self.task_service.append_event(
                task_obj,
                TaskEventType.tool_call,
                {
                    "turn": turn,
                    "tool_call_id": tool_call.id,
                    "tool": tool_call.name,
                    "argument_keys": sorted(arguments),
                    "arguments": arguments,
                },
            )
            log_observation(
                logger,
                "tool_call_started",
                task=task_obj,
                turn=turn,
                tool_call_id=tool_call.id,
                tool=tool_call.name,
                argument_keys=sorted(arguments),
            )
            started = time.perf_counter()
            recoverable_error: RecoverableToolError | None = None
            result: ToolResult
            try:
                with start_span(
                    "tool.invoke",
                    task=task_obj,
                    attributes={
                        "openinference.span.kind": "TOOL",
                        "agent.turn": turn,
                        "tool.name": tool_call.name,
                        "tool.call_id": tool_call.id,
                        "tool.argument_keys": sorted(arguments),
                    },
                ):
                    try:
                        result = self.registry.invoke(tool_call.name, arguments)
                    except RecoverableToolError as exc:
                        recoverable_error = exc
                        result = _recoverable_tool_error_result(
                            arguments=arguments,
                            error=exc,
                        )
                        record_span_exception(exc)
                    set_span_attributes(
                        {
                            "tool.latency_ms": _latency_ms(started),
                            "tool.artifact_count": len(result.artifacts),
                            "tool.recoverable": _recoverable_tool_result(result.output),
                            "tool.cost_usd": str(result.cost_usd),
                        }
                    )
            except Exception as exc:
                latency_ms = _latency_ms(started)
                record_span_exception(exc)
                log_observation(
                    logger,
                    "tool_call_failed",
                    level=logging.ERROR,
                    task=task_obj,
                    turn=turn,
                    tool_call_id=tool_call.id,
                    tool=tool_call.name,
                    latency_ms=latency_ms,
                    error_type=type(exc).__name__,
                    error_summary=str(exc),
                )
                self._append_error(
                    task_obj,
                    exc,
                    {
                        "turn": turn,
                        "phase": "tool_invoke",
                        "tool_call_id": tool_call.id,
                        "tool": tool_call.name,
                        "latency_ms": latency_ms,
                    },
                )
                raise

            if recoverable_error is not None:
                log_observation(
                    logger,
                    "tool_call_recoverable_failed",
                    level=logging.WARNING,
                    task=task_obj,
                    turn=turn,
                    tool_call_id=tool_call.id,
                    tool=tool_call.name,
                    latency_ms=_latency_ms(started),
                    error_code=recoverable_error.code,
                    error_summary=recoverable_error.message,
                )

            self.task_service.raise_if_cancelled(
                task_obj, phase=f"after_tool_{tool_call.name}"
            )
            latency_ms = _latency_ms(started)
            artifact_count += len(result.artifacts)
            result_payload = _tool_result_payload(tool_call.name, result)
            self.task_service.append_event(
                task_obj,
                TaskEventType.tool_result,
                {
                    "turn": turn,
                    "tool_call_id": tool_call.id,
                    "tool": tool_call.name,
                    "latency_ms": latency_ms,
                    "output_shape": _output_shape(result.output),
                    "artifact_count": len(result.artifacts),
                    "recoverable": _recoverable_tool_result(result.output),
                    **result_payload,
                },
            )
            log_observation(
                logger,
                "tool_call_completed",
                task=task_obj,
                turn=turn,
                tool_call_id=tool_call.id,
                tool=tool_call.name,
                latency_ms=latency_ms,
                output_shape=_output_shape(result.output),
                artifact_count=len(result.artifacts),
                recoverable=_recoverable_tool_result(result.output),
                cost_usd=str(result.cost_usd),
            )
            messages.append(
                ChatMessage(
                    role="tool",
                    content=_json_dumps(result_payload),
                    tool_call_id=tool_call.id,
                )
            )

        return artifact_count

    def _tool_arguments(self, task: Task, tool_call: ToolCall) -> JsonObject:
        arguments = dict(tool_call.arguments)
        if tool_call.name == "pdf_generator" and "min_pages" not in arguments:
            min_pages = _requested_pdf_min_pages(task.input)
            if min_pages is not None:
                arguments["min_pages"] = min_pages
        return arguments

    def _finish_with_text(
        self,
        task: Task,
        content: str | None,
        turn: int,
    ) -> AgentRunResult:
        summary = (content or "").strip()
        if not summary:
            error = AgentLoopError("Agent returned no final content or tool calls")
            self._append_error(task, error, {"turn": turn, "phase": "final_content"})
            raise error

        return self._finish(
            task,
            summary=summary,
            turn=turn,
            artifact_count=0,
            reason="final_answer",
        )

    def _finish_with_artifacts(
        self,
        task: Task,
        *,
        turn: int,
        artifact_count: int,
    ) -> AgentRunResult:
        artifact_word = "artifact" if artifact_count == 1 else "artifacts"
        return self._finish(
            task,
            summary=f"Generated {artifact_count} {artifact_word}.",
            turn=turn,
            artifact_count=artifact_count,
            reason="artifact",
        )

    def _finish(
        self,
        task: Task,
        *,
        summary: str,
        turn: int,
        artifact_count: int,
        reason: str,
    ) -> AgentRunResult:
        task.result_summary = summary
        self.session.flush()
        self._append_log(
            task,
            "agent_completed",
            {
                "turns": turn,
                "reason": reason,
                "artifact_count": artifact_count,
            },
        )
        return AgentRunResult(
            task_id=task.id,
            result_summary=summary,
            turns=turn,
            artifact_count=artifact_count,
        )

    def _append_log(
        self,
        task: Task,
        message: str,
        payload: JsonObject | None = None,
    ) -> None:
        self.task_service.append_event(
            task,
            TaskEventType.log,
            {"message": message, **(payload or {})},
        )

    def _append_error(
        self,
        task: Task,
        error: Exception,
        payload: JsonObject | None = None,
    ) -> None:
        self.task_service.append_event(
            task,
            TaskEventType.error,
            {
                "type": type(error).__name__,
                "message": str(error),
                **(payload or {}),
            },
        )

    def _resolve_task(self, task: Task | uuid.UUID) -> Task:
        if isinstance(task, Task):
            return task
        task_obj = self.task_service.get_task(task)
        if task_obj is None:
            raise LookupError(f"Task not found: {task}")
        return task_obj

    def _initial_messages(self, task: Task) -> list[ChatMessage]:
        return list(self.context_assembler.build_for_task(task).messages)


def _tool_result_payload(tool_name: str, result: ToolResult) -> JsonObject:
    return {
        "tool": tool_name,
        "output": result.output,
        "cost_usd": str(result.cost_usd),
        "artifacts": [_artifact_payload(artifact) for artifact in result.artifacts],
    }


def _recoverable_tool_error_result(
    *,
    arguments: JsonObject,
    error: RecoverableToolError,
) -> ToolResult:
    return ToolResult(
        output={
            "successful": False,
            "attempted_argument_keys": sorted(arguments),
            "error": error.to_payload(),
        }
    )


def _artifact_payload(artifact: ToolArtifact) -> JsonObject:
    return asdict(artifact)


def _requested_pdf_min_pages(input_text: str) -> int | None:
    matches = [int(match.group(1)) for match in REQUESTED_PAGES_RE.finditer(input_text)]
    if not matches:
        return None
    requested = max(matches)
    if requested < 1 or requested > 50:
        return None
    return requested


def _json_dumps(payload: object) -> str:
    return json.dumps(payload, default=_json_default, separators=(",", ":"))


def _latency_ms(started: float) -> int:
    return max(0, int((time.perf_counter() - started) * 1000))


def _output_shape(output: JsonObject) -> JsonObject:
    return {
        "type": "object",
        "keys": sorted(output),
    }


def _recoverable_tool_result(output: JsonObject) -> bool | None:
    error = output.get("error")
    if not isinstance(error, dict):
        return None
    recoverable = error.get("recoverable")
    return recoverable if isinstance(recoverable, bool) else None


def _json_default(value: object) -> object:
    if isinstance(value, Decimal):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
