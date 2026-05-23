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

from sqlalchemy.orm import Session

from kortny.db.models import Task, TaskEventType
from kortny.llm import ChatMessage, Completion
from kortny.tasks import TaskService
from kortny.tools import ToolRegistry
from kortny.tools.types import JsonObject, JsonSchema, ToolArtifact, ToolResult

DEFAULT_MAX_TURNS = 6


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
    ) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1")

        self.session = session
        self.llm = llm
        self.registry = registry
        self.task_service = task_service or TaskService(session)
        self.max_turns = max_turns

    def run(self, task: Task | uuid.UUID) -> AgentRunResult:
        """Run the coordinator until final text or a produced artifact."""

        task_obj = self._resolve_task(task)
        messages = [ChatMessage(role="user", content=task_obj.input)]
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


def _tool_result_payload(tool_name: str, result: ToolResult) -> JsonObject:
    return {
        "tool": tool_name,
        "output": result.output,
        "cost_usd": str(result.cost_usd),
        "artifacts": [_artifact_payload(artifact) for artifact in result.artifacts],
    }


def _artifact_payload(artifact: ToolArtifact) -> JsonObject:
    return asdict(artifact)


def _json_dumps(payload: JsonObject) -> str:
    return json.dumps(payload, default=_json_default, separators=(",", ":"))


def _json_default(value: object) -> object:
    if isinstance(value, Decimal):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
