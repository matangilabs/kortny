"""Human approval policy for sensitive tool calls."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol, TypeAlias

from kortny.tools.catalog import (
    low_risk_native_write_tool_names,
    native_tool_names_by_approval,
    read_only_native_tool_names,
)

JsonObject: TypeAlias = dict[str, Any]


class Tool(Protocol):
    """Minimal tool shape needed by approval policy."""

    name: str
    description: str


TOOL_APPROVAL_REQUIRED_MESSAGE = "tool_approval_required"
TOOL_APPROVAL_WAITING_MESSAGE = "tool_approval_waiting"
TOOL_APPROVAL_DECISION_MESSAGE = "tool_approval_decision"
TOOL_APPROVAL_PROMPT_PURPOSE = "tool_approval_request"
TOOL_APPROVAL_REJECTED_PURPOSE = "tool_approval_rejected"
TOOL_APPROVAL_REACTION_INSTRUCTION = (
    "React with :white_check_mark: to approve, or :no_entry_sign: to skip it."
)


class ApprovalScope(StrEnum):
    """Who must approve a gated tool call."""

    none = "none"
    user = "user"
    admin = "admin"


@dataclass(frozen=True, slots=True)
class ToolApprovalRequirement:
    """Approval policy output for one tool call."""

    scope: ApprovalScope
    reason: str
    risk: str

    @property
    def required(self) -> bool:
        return self.scope is not ApprovalScope.none


@dataclass(frozen=True, slots=True)
class ToolApprovalRequest:
    """A concrete approval request for a planned tool invocation."""

    approval_key: str
    tool_name: str
    tool_call_id: str
    normalized_args_hash: str
    argument_keys: tuple[str, ...]
    scope: ApprovalScope
    reason: str
    risk: str
    arguments: JsonObject

    def to_payload(self) -> JsonObject:
        return {
            "approval_key": self.approval_key,
            "tool": self.tool_name,
            "tool_call_id": self.tool_call_id,
            "normalized_args_hash": self.normalized_args_hash,
            "argument_keys": list(self.argument_keys),
            "scope": self.scope.value,
            "reason": self.reason,
            "risk": self.risk,
            "arguments": self.arguments,
        }


class ToolApprovalRequired(RuntimeError):
    """Raised when a tool call must pause for human approval."""

    def __init__(self, request: ToolApprovalRequest) -> None:
        super().__init__(
            f"Tool approval required for {request.tool_name} ({request.approval_key})"
        )
        self.request = request


class ToolApprovalPolicy:
    """Conservative default approval policy.

    The current Composio provider only exposes read-only tools, but the policy is
    intentionally provider-neutral so future write-capable integrations are gated
    before execution.
    """

    def requirement_for(self, tool: Tool, args: JsonObject) -> ToolApprovalRequirement:
        del args
        tool_name = tool.name.casefold()
        if (
            tool_name in READ_ONLY_NATIVE_TOOLS
            or tool_name in SELF_GATED_NATIVE_TOOLS
            or tool_name in LOW_RISK_NATIVE_WRITE_TOOLS
        ):
            return NO_APPROVAL_REQUIRED
        if tool_name in USER_APPROVAL_NATIVE_TOOLS:
            if tool_name in SANDBOX_WORKBENCH_TOOLS:
                return ToolApprovalRequirement(
                    scope=ApprovalScope.user,
                    risk="sandboxed_code_execution",
                    reason=(
                        f"{tool.name} works inside Kortny's isolated sandbox "
                        "workspace for this task."
                    ),
                )
            if tool_name == "deploy_site":
                return ToolApprovalRequirement(
                    scope=ApprovalScope.user,
                    risk="external_deployment",
                    reason=(
                        f"{tool.name} publishes files to an external hosting provider."
                    ),
                )
            if tool_name in SANDBOXED_CODE_NATIVE_TOOLS:
                return ToolApprovalRequirement(
                    scope=ApprovalScope.user,
                    risk="sandboxed_code_execution",
                    reason=(
                        f"{tool.name} can execute code in Kortny's isolated sandbox."
                    ),
                )
            return ToolApprovalRequirement(
                scope=ApprovalScope.user,
                risk="workspace_state_mutation",
                reason=f"{tool.name} can change Kortny's stored state.",
            )
        if tool_name in ADMIN_APPROVAL_NATIVE_TOOLS:
            return ToolApprovalRequirement(
                scope=ApprovalScope.admin,
                risk="sandboxed_code_execution",
                reason=(f"{tool.name} can execute untrusted code in Kortny's sandbox."),
            )
        if _tool_is_explicitly_read_only(tool):
            return NO_APPROVAL_REQUIRED
        risky_verbs = _risky_verbs(tool)
        if risky_verbs:
            verb = sorted(risky_verbs)[0]
            return ToolApprovalRequirement(
                scope=ApprovalScope.user,
                risk="external_side_effect",
                reason=(
                    f"{tool.name} appears to perform a {verb} action against an "
                    "external service."
                ),
            )
        return NO_APPROVAL_REQUIRED


NO_APPROVAL_REQUIRED = ToolApprovalRequirement(
    scope=ApprovalScope.none,
    risk="none",
    reason="Tool is read-only or already has its own confirmation path.",
)

READ_ONLY_NATIVE_TOOLS = read_only_native_tool_names()
SELF_GATED_NATIVE_TOOLS = native_tool_names_by_approval("self_gated")
LOW_RISK_NATIVE_WRITE_TOOLS = low_risk_native_write_tool_names()
USER_APPROVAL_NATIVE_TOOLS = native_tool_names_by_approval("user_approval")
ADMIN_APPROVAL_NATIVE_TOOLS = native_tool_names_by_approval("admin_approval")
SANDBOXED_CODE_NATIVE_TOOLS = frozenset({"code_exec"})
SANDBOX_WORKBENCH_TOOLS = frozenset(
    {
        "sandbox_bash",
        "sandbox_write_file",
        "sandbox_read_file",
        "sandbox_export_artifact",
        "sandbox_publish_preview",
    }
)
SANDBOX_WORKBENCH_APPROVAL_KEY = "sandbox_workbench:session"
WRITE_VERBS = frozenset(
    {
        "add",
        "archive",
        "cancel",
        "create",
        "delete",
        "disable",
        "enable",
        "invite",
        "move",
        "post",
        "publish",
        "remove",
        "send",
        "set",
        "submit",
        "update",
        "write",
    }
)
READ_ONLY_TAGS = frozenset({"readonly", "readonlyhint", "read_only", "read-only"})


def approval_key(tool_name: str, normalized_args_hash: str) -> str:
    """Return a stable approval key for a tool call signature."""

    return f"{tool_name}:{normalized_args_hash}"


def approval_key_for(tool_name: str, normalized_args_hash: str) -> str:
    """Return the approval key, collapsing workbench tools to one session key.

    All sandbox workbench tools share one approval per task: a single user
    confirmation unlocks the whole sandbox workspace instead of prompting on
    every command.
    """

    if tool_name.casefold() in SANDBOX_WORKBENCH_TOOLS:
        return SANDBOX_WORKBENCH_APPROVAL_KEY
    return approval_key(tool_name, normalized_args_hash)


def approval_prompt_text(request: ToolApprovalRequest) -> str:
    """Render a Slack-native approval prompt."""

    args = ", ".join(request.argument_keys) or "none"
    if request.tool_name in SANDBOX_WORKBENCH_TOOLS:
        return (
            "I need a sandboxed workspace to build and verify this. One "
            "approval covers all sandbox commands for this task.\n"
            "*Safety:* isolated container, no network, no host filesystem "
            "access, CPU/memory capped, auto-removed when the task is done.\n"
            f"*First step:* {request.tool_name} ({args})\n\n"
            f"{TOOL_APPROVAL_REACTION_INSTRUCTION}"
        )
    if request.tool_name == "deploy_site" and request.risk == "external_deployment":
        return (
            "I'm ready to deploy this site to an external hosting provider. "
            "Please approve before I publish.\n"
            f"*Inputs:* {args}\n\n"
            f"{TOOL_APPROVAL_REACTION_INSTRUCTION}"
        )
    if request.tool_name == "code_exec" and request.risk == "sandboxed_code_execution":
        return (
            "I can check this in a locked-down Python sandbox. Please approve "
            "before I run it.\n"
            "*Safety:* no network, no package installs, and no access to the "
            "host filesystem.\n"
            f"*Inputs:* {args}\n\n"
            f"{TOOL_APPROVAL_REACTION_INSTRUCTION}"
        )
    return (
        f"I need your approval before I run *{request.tool_name}*.\n"
        f"*Why:* {request.reason}\n"
        f"*Arguments:* {args}\n\n"
        f"{TOOL_APPROVAL_REACTION_INSTRUCTION}"
    )


def approval_rejected_text(request: ToolApprovalRequest | None) -> str:
    """Render a concise rejection acknowledgement."""

    if request is None:
        return "Okay, I won't run that action."
    return f"Okay, I won't run *{request.tool_name}*."


def _tool_is_explicitly_read_only(tool: Tool) -> bool:
    composio_tool = getattr(tool, "tool", None)
    tags = getattr(composio_tool, "tags", None)
    if not isinstance(tags, (tuple, list, set)):
        return False
    normalized = {
        str(tag).casefold().replace("-", "_").replace(" ", "_") for tag in tags
    }
    return bool(normalized & READ_ONLY_TAGS)


def _risky_verbs(tool: Tool) -> set[str]:
    text_parts: list[str] = [tool.name, tool.description]
    composio_tool = getattr(tool, "tool", None)
    for attr in ("slug", "name", "description"):
        value: Any = getattr(composio_tool, attr, None)
        if isinstance(value, str):
            text_parts.append(value)
    words = set[str]()
    for text in text_parts:
        words.update(
            part
            for part in text.casefold().replace("-", "_").split("_")
            if part.strip()
        )
    return words & WRITE_VERBS
