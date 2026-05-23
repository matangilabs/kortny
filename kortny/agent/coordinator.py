"""Plain MVP coordinator loop.

This is intentionally not ADK. The message and tool boundaries stay close to
OpenAI/OpenRouter chat completions so they can be adapted to ADK later.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from decimal import Decimal
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session

from kortny.agent.thread_context import (
    ThreadTranscriptMessage,
    ThreadTranscriptProvider,
)
from kortny.db.models import Task, TaskEvent, TaskEventType
from kortny.llm import ChatMessage, Completion
from kortny.tasks import TaskService
from kortny.tools import ToolRegistry
from kortny.tools.types import JsonObject, JsonSchema, ToolArtifact, ToolResult

DEFAULT_MAX_TURNS = 6
DEFAULT_THREAD_CONTEXT_MAX_CHARS = 12_000
DEFAULT_THREAD_CONTEXT_RECENT_TASKS = 3
DEFAULT_THREAD_TRANSCRIPT_LIMIT = 30
DEFAULT_SYSTEM_PROMPT = (
    "You are Kortny, a Slack-native AI coworker. Use the available tools when "
    "they are needed to complete the user's request. If the user asks for "
    "research and a PDF, search first and then generate the PDF artifact. "
    "When answering with text, format for Slack mrkdwn rather than GitHub "
    "Markdown: use *bold*, <https://example.com|label> links, simple line-break "
    "lists, and avoid Markdown headings."
)
THREAD_CONTEXT_EVENT_TYPES = {
    TaskEventType.llm_call,
    TaskEventType.tool_call,
    TaskEventType.tool_result,
    TaskEventType.error,
}


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
    ) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        if thread_context_max_chars < 1:
            raise ValueError("thread_context_max_chars must be at least 1")
        if thread_context_recent_tasks < 1:
            raise ValueError("thread_context_recent_tasks must be at least 1")
        if thread_transcript_limit < 0:
            raise ValueError("thread_transcript_limit cannot be negative")

        self.session = session
        self.llm = llm
        self.registry = registry
        self.task_service = task_service or TaskService(session)
        self.max_turns = max_turns
        self.system_prompt = system_prompt
        self.thread_transcript_provider = thread_transcript_provider
        self.thread_context_max_chars = thread_context_max_chars
        self.thread_context_recent_tasks = thread_context_recent_tasks
        self.thread_transcript_limit = thread_transcript_limit

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
            completion = self._complete_turn(task_obj, messages, schemas, turn)
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
            self.task_service.append_event(
                task_obj,
                TaskEventType.tool_call,
                {
                    "turn": turn,
                    "tool_call_id": tool_call.id,
                    "tool": tool_call.name,
                    "arguments": tool_call.arguments,
                },
            )
            try:
                result = self.registry.invoke(tool_call.name, tool_call.arguments)
            except Exception as exc:
                self._append_error(
                    task_obj,
                    exc,
                    {
                        "turn": turn,
                        "phase": "tool_invoke",
                        "tool_call_id": tool_call.id,
                        "tool": tool_call.name,
                    },
                )
                raise

            artifact_count += len(result.artifacts)
            result_payload = _tool_result_payload(tool_call.name, result)
            self.task_service.append_event(
                task_obj,
                TaskEventType.tool_result,
                {
                    "turn": turn,
                    "tool_call_id": tool_call.id,
                    "tool": tool_call.name,
                    **result_payload,
                },
            )
            messages.append(
                ChatMessage(
                    role="tool",
                    content=_json_dumps(result_payload),
                    tool_call_id=tool_call.id,
                )
            )

        return artifact_count

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
        messages: list[ChatMessage] = []
        if self.system_prompt:
            messages.append(ChatMessage(role="system", content=self.system_prompt))
        prior_context = self._prior_context(task)
        if prior_context:
            messages.append(ChatMessage(role="system", content=prior_context))
        messages.append(ChatMessage(role="user", content=task.input))
        return messages

    def _prior_context(self, task: Task) -> str | None:
        thread_ts = task.slack_thread_ts
        if not thread_ts:
            return None

        thread_tasks = self.task_service.list_by_thread(
            task.slack_channel_id, thread_ts
        )
        prior_tasks = _tasks_before(thread_tasks, task)
        if not prior_tasks:
            return None

        transcript = self._fetch_thread_transcript(task)
        detailed = self._render_prior_context(
            prior_tasks,
            transcript=transcript,
            include_events=True,
            compacted=False,
        )
        if len(detailed) <= self.thread_context_max_chars:
            return detailed

        compact = self._render_prior_context(
            prior_tasks,
            transcript=transcript,
            include_events=False,
            compacted=True,
        )
        return _fit_context_to_budget(compact, self.thread_context_max_chars)

    def _fetch_thread_transcript(
        self,
        task: Task,
    ) -> tuple[ThreadTranscriptMessage, ...]:
        if self.thread_transcript_provider is None or self.thread_transcript_limit == 0:
            return ()
        if not task.slack_thread_ts:
            return ()

        try:
            return self.thread_transcript_provider.fetch_thread_messages(
                channel_id=task.slack_channel_id,
                thread_ts=task.slack_thread_ts,
                limit=self.thread_transcript_limit,
            )
        except Exception as exc:
            self._append_log(
                task,
                "thread_transcript_unavailable",
                {"error_type": type(exc).__name__, "error": str(exc)},
            )
            return ()

    def _render_prior_context(
        self,
        prior_tasks: Sequence[Task],
        *,
        transcript: Sequence[ThreadTranscriptMessage],
        include_events: bool,
        compacted: bool,
    ) -> str:
        lines = [
            "<prior_context>",
            "This task is a follow-up in the same Slack thread. Use this context "
            'to resolve references like "it", "that", "the PDF", and '
            '"your source". Do not treat it as cross-thread memory.',
        ]
        if compacted:
            lines.append(
                "Context was compacted to stay within the configured token budget; "
                "older task event details were omitted."
            )

        if include_events:
            older_tasks = prior_tasks[: -self.thread_context_recent_tasks]
            recent_tasks = prior_tasks[-self.thread_context_recent_tasks :]
            if older_tasks:
                lines.append("")
                lines.append("Older prior task summaries:")
                for index, prior_task in enumerate(older_tasks, start=1):
                    lines.append(_task_summary_line(index, prior_task))
            lines.append("")
            lines.append("Recent prior task details:")
            start_index = len(older_tasks) + 1
            for index, prior_task in enumerate(recent_tasks, start=start_index):
                lines.extend(self._prior_task_detail_lines(index, prior_task))
        else:
            lines.append("")
            lines.append("Prior task summaries:")
            for index, prior_task in enumerate(prior_tasks, start=1):
                lines.append(_task_summary_line(index, prior_task))

        if transcript:
            lines.append("")
            lines.append("Slack thread transcript:")
            for message in transcript:
                lines.append(_transcript_line(message))

        lines.append("</prior_context>")
        return "\n".join(lines)

    def _prior_task_detail_lines(self, index: int, task: Task) -> list[str]:
        lines = [_task_summary_line(index, task)]
        events = self._context_events(task)
        if events:
            lines.append("  events:")
            for event in events:
                lines.append(
                    f"  - {event.type.value}: {_shorten(_json_dumps(event.payload), max_chars=600)}"
                )
        return lines

    def _context_events(self, task: Task) -> list[TaskEvent]:
        return list(
            self.session.scalars(
                select(TaskEvent)
                .where(
                    TaskEvent.task_id == task.id,
                    TaskEvent.type.in_(THREAD_CONTEXT_EVENT_TYPES),
                )
                .order_by(TaskEvent.seq)
            )
        )


def _tool_result_payload(tool_name: str, result: ToolResult) -> JsonObject:
    return {
        "tool": tool_name,
        "output": result.output,
        "cost_usd": str(result.cost_usd),
        "artifacts": [_artifact_payload(artifact) for artifact in result.artifacts],
    }


def _artifact_payload(artifact: ToolArtifact) -> JsonObject:
    return asdict(artifact)


def _tasks_before(thread_tasks: Sequence[Task], current_task: Task) -> list[Task]:
    prior_tasks: list[Task] = []
    for task in thread_tasks:
        if task.id == current_task.id:
            return prior_tasks
        prior_tasks.append(task)
    return [task for task in thread_tasks if task.id != current_task.id]


def _task_summary_line(index: int, task: Task) -> str:
    result = task.result_summary or "(no result summary yet)"
    line = (
        f"- {index}. task_id={task.id} status={task.status.value} input={_quote(_shorten(task.input, max_chars=240))} "
        f"result={_quote(_shorten(result, max_chars=360))} cost_usd={task.total_cost_usd}"
    )
    error = _error_summary(task.error)
    if error:
        line = f"{line} error={_quote(_shorten(error, max_chars=240))}"
    return line


def _error_summary(error: dict | None) -> str | None:
    if not error:
        return None
    error_type = error.get("type")
    message = error.get("message")
    if isinstance(error_type, str) and isinstance(message, str):
        return f"{error_type}: {message}"
    return _json_dumps(error)


def _transcript_line(message: ThreadTranscriptMessage) -> str:
    speaker = message.user_id
    if speaker is None and message.bot_id is not None:
        speaker = f"bot:{message.bot_id}"
    if speaker is None:
        speaker = "unknown"
    return f"- [{message.ts}] {speaker}: {_shorten(_single_line(message.text), max_chars=500)}"


def _fit_context_to_budget(content: str, max_chars: int) -> str:
    if len(content) <= max_chars:
        return content
    suffix = "\n[prior_context truncated at configured budget]\n</prior_context>"
    return content[: max(0, max_chars - len(suffix))].rstrip() + suffix


def _shorten(value: str, *, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[: max(0, max_chars - 3)].rstrip() + "..."


def _quote(value: str) -> str:
    return json.dumps(_single_line(value))


def _single_line(value: str) -> str:
    return " ".join(value.split())


def _json_dumps(payload: object) -> str:
    return json.dumps(payload, default=_json_default, separators=(",", ":"))


def _json_default(value: object) -> object:
    if isinstance(value, Decimal):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
