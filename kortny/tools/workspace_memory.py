"""Workspace memory tools backed by WorkspaceStateService."""

from __future__ import annotations

from typing import Any

from kortny.db.models import Task
from kortny.memory import WorkspaceStateService
from kortny.tools.types import JsonObject, JsonSchema, ToolResult


class RememberFactTool:
    """Propose a durable memory fact and ask the user to confirm it."""

    name = "remember_fact"
    description = (
        "Proposes a workspace, channel, or user memory fact. The fact is not "
        "saved until the user confirms the Slack prompt."
    )
    parameters: JsonSchema = {
        "type": "object",
        "properties": {
            "scope": {
                "type": "string",
                "enum": ["workspace", "channel", "user"],
                "description": "Where the memory applies.",
            },
            "key": {
                "type": "string",
                "description": "Stable snake_case key for the memory fact.",
            },
            "value": {
                "type": "object",
                "description": "Structured JSON object to remember.",
                "additionalProperties": True,
            },
            "value_text": {
                "type": "string",
                "description": "Human-readable summary shown in the confirmation prompt.",
            },
            "reason": {
                "type": "string",
                "description": "Why this fact should be remembered.",
            },
            "confidence_score": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "Optional confidence score for the proposed fact.",
            },
            "confidence_reason": {
                "type": "string",
                "description": "Optional confidence rationale.",
            },
        },
        "required": ["scope", "key", "value"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        *,
        service: WorkspaceStateService,
        task: Task,
    ) -> None:
        self.service = service
        self.task = task

    def invoke(self, args: JsonObject) -> ToolResult:
        scope = _required_string(args, "scope")
        key = _required_string(args, "key")
        value = _required_object(args, "value")
        value_text = _optional_string(args.get("value_text"))
        reason = _optional_string(args.get("reason"))
        confidence_reason = _optional_string(args.get("confidence_reason"))
        confidence_score = args.get("confidence_score")

        pending = self.service.propose(
            self.task.installation_id,
            scope,
            _scope_id_for_task(scope, self.task),
            key,
            value,
            self.task.id,
            value_text=value_text,
            proposed_reason=reason,
            confidence_score=confidence_score,
            confidence_reason=confidence_reason,
        )
        return ToolResult(
            output={
                "status": "pending_confirmation",
                "scope": pending.scope_type,
                "scope_id": pending.scope_id,
                "key": pending.key,
                "value": pending.value,
                "value_text": pending.value_text,
                "prompt_channel_id": pending.prompt_channel_id,
                "prompt_message_ts": pending.prompt_message_ts,
                "message": (
                    "A confirmation prompt was posted. The fact is not saved "
                    "until the user reacts with :white_check_mark:."
                ),
            }
        )


class RecallFactTool:
    """Read current active memory facts."""

    name = "recall_fact"
    description = "Reads a current active memory fact for this workspace/channel/user."
    parameters: JsonSchema = {
        "type": "object",
        "properties": {
            "scope": {
                "type": "string",
                "enum": ["workspace", "channel", "user"],
                "description": "Where to look for the memory fact.",
            },
            "key": {
                "type": "string",
                "description": "Stable key for the memory fact.",
            },
        },
        "required": ["scope", "key"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        *,
        service: WorkspaceStateService,
        task: Task,
    ) -> None:
        self.service = service
        self.task = task

    def invoke(self, args: JsonObject) -> ToolResult:
        scope = _required_string(args, "scope")
        key = _required_string(args, "key")
        fact = self.service.get(
            self.task.installation_id,
            scope,
            _scope_id_for_task(scope, self.task),
            key,
        )
        if fact is None:
            return ToolResult(
                output={
                    "found": False,
                    "scope": scope,
                    "scope_id": _scope_id_for_task(scope, self.task),
                    "key": key,
                }
            )
        return ToolResult(
            output={
                "found": True,
                "id": str(fact.id),
                "scope": fact.scope_type,
                "scope_id": fact.scope_id,
                "key": fact.key,
                "value": fact.value,
                "value_text": fact.value_text,
                "confirmed_by_user_id": fact.confirmed_by_user_id,
                "confirmed_at": fact.confirmed_at.isoformat()
                if fact.confirmed_at is not None
                else None,
            }
        )


def _scope_id_for_task(scope: str, task: Task) -> str | None:
    if scope == "workspace":
        return None
    if scope == "channel":
        return task.slack_channel_id
    if scope == "user":
        return task.slack_user_id
    raise ValueError(f"Unsupported scope: {scope}")


def _required_string(args: JsonObject, key: str) -> str:
    value = args.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} is required")
    return value.strip()


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("optional string argument must be a string")
    return value.strip() or None


def _required_object(args: JsonObject, key: str) -> dict[str, Any]:
    value = args.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a JSON object")
    return dict(value)
