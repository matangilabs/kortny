"""Human approval policy for sensitive tool calls.

Approval is the *policy* half of the HIG-223 autonomy ladder; risk
classification (the *which tier* half) lives in :mod:`kortny.autonomy`. The
``requirement_for`` seam takes the resolved autonomy level plus a
:class:`~kortny.autonomy.RiskAssessment` and maps tier x level to a concrete
approval requirement.

Tier x level matrix (as implemented in ``_requirement_from_ladder``):

    tier \\ level    conservative   balanced       autonomous
    -----------------------------------------------------------
    free             none           none           none
    implicit         user           none+audit     none+audit
    explicit         user           user           user

Tier-1 (implicit) auto-approvals — balanced + autonomous — are recorded by the
coordinator as ``tool_autonomy_decision`` audit events. Hard native gates
(deploy_site, forget_fact, sandbox workbench session, admin untrusted skill
scripts) are unconditional and stay gated at every level; sandbox code_exec
stays auto-approved at every level.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol, TypeAlias

from kortny.autonomy import AutonomyLevel, AutonomyTier, RiskAssessment
from kortny.tools.catalog import (
    ToolSideEffect,
    auto_approved_native_tool_names,
    low_risk_native_write_tool_names,
    native_tool_names_by_approval,
    read_only_native_tool_names,
)

JsonObject: TypeAlias = dict[str, Any]
ToolSideEffectLiteral: TypeAlias = ToolSideEffect


class Tool(Protocol):
    """Minimal tool shape needed by approval policy."""

    name: str
    description: str


TOOL_APPROVAL_REQUIRED_MESSAGE = "tool_approval_required"
TOOL_APPROVAL_WAITING_MESSAGE = "tool_approval_waiting"
TOOL_APPROVAL_DECISION_MESSAGE = "tool_approval_decision"
TOOL_AUTONOMY_DECISION_MESSAGE = "tool_autonomy_decision"
TOOL_APPROVAL_PROMPT_PURPOSE = "tool_approval_request"
TOOL_APPROVAL_REJECTED_PURPOSE = "tool_approval_rejected"
# Buttons are the primary approval UI (HIG-255 s2); emoji reactions still work
# as a silent fallback, but we no longer advertise them in the prompt.
TOOL_APPROVAL_REACTION_INSTRUCTION = "Use the *Approve* / *Reject* buttons below."


class ApprovalScope(StrEnum):
    """Who must approve a gated tool call."""

    none = "none"
    user = "user"
    admin = "admin"


@dataclass(frozen=True, slots=True)
class ToolApprovalRequirement:
    """Approval policy output for one tool call.

    ``autonomy_tier`` / ``autonomy_level`` / ``autonomy_reasons`` carry the
    HIG-223 ladder decision so the coordinator can emit a ``tool_autonomy_decision``
    audit event on auto-approved Tier-1 calls. ``audit_autonomy`` marks the
    requirements that should be audited (auto-approved implicit-tier calls).
    """

    scope: ApprovalScope
    reason: str
    risk: str
    autonomy_tier: AutonomyTier | None = None
    autonomy_level: AutonomyLevel | None = None
    autonomy_reasons: tuple[str, ...] = ()
    audit_autonomy: bool = False

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

    def requirement_for(
        self,
        tool: Tool,
        args: JsonObject,
        *,
        autonomy_level: AutonomyLevel = AutonomyLevel.balanced,
        risk: RiskAssessment | None = None,
    ) -> ToolApprovalRequirement:
        tool_name = tool.name.casefold()
        if tool_name in READ_ONLY_NATIVE_TOOLS or tool_name in SELF_GATED_NATIVE_TOOLS:
            return NO_APPROVAL_REQUIRED
        # Native tools the catalog explicitly marks approval='none' (slack
        # replies, schedule mutations, sandbox writes, and auto-approved
        # sandbox/code execution) stay auto-approved at every level — their
        # blast radius is scoped or sandboxed, so they never enter the ladder.
        if (
            tool_name in AUTO_APPROVED_NATIVE_TOOLS
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

        # External / unclassified tools flow through the HIG-223 autonomy ladder.
        assessment = risk if risk is not None else assess_tool_risk(tool, args)
        return self._requirement_from_ladder(
            tool=tool,
            autonomy_level=autonomy_level,
            assessment=assessment,
        )

    def _requirement_from_ladder(
        self,
        *,
        tool: Tool,
        autonomy_level: AutonomyLevel,
        assessment: RiskAssessment,
    ) -> ToolApprovalRequirement:
        tier = assessment.tier
        if tier is AutonomyTier.free:
            return ToolApprovalRequirement(
                scope=ApprovalScope.none,
                risk="read_only",
                reason=f"{tool.name} is read-only or generates a local artifact.",
                autonomy_tier=tier,
                autonomy_level=autonomy_level,
                autonomy_reasons=assessment.reasons,
                audit_autonomy=False,
            )

        if tier is AutonomyTier.explicit:
            return ToolApprovalRequirement(
                scope=ApprovalScope.user,
                risk="irreversible_or_outward",
                reason=(
                    f"{tool.name} performs an irreversible, outward, or bulk "
                    "action that needs your go-ahead."
                ),
                autonomy_tier=tier,
                autonomy_level=autonomy_level,
                autonomy_reasons=assessment.reasons,
                audit_autonomy=False,
            )

        # tier is implicit (external create/update).
        if autonomy_level is AutonomyLevel.conservative:
            return ToolApprovalRequirement(
                scope=ApprovalScope.user,
                risk="external_write",
                reason=(
                    f"{tool.name} writes to an external service; conservative "
                    "autonomy gates every write."
                ),
                autonomy_tier=tier,
                autonomy_level=autonomy_level,
                autonomy_reasons=assessment.reasons,
                audit_autonomy=False,
            )
        # balanced / autonomous auto-approve implicit writes, with an audit trail.
        return ToolApprovalRequirement(
            scope=ApprovalScope.none,
            risk="external_write",
            reason=f"{tool.name} writes to an external service (auto-approved).",
            autonomy_tier=tier,
            autonomy_level=autonomy_level,
            autonomy_reasons=assessment.reasons,
            audit_autonomy=True,
        )


NO_APPROVAL_REQUIRED = ToolApprovalRequirement(
    scope=ApprovalScope.none,
    risk="none",
    reason="Tool is read-only or already has its own confirmation path.",
)

READ_ONLY_NATIVE_TOOLS = read_only_native_tool_names()
SELF_GATED_NATIVE_TOOLS = native_tool_names_by_approval("self_gated")
LOW_RISK_NATIVE_WRITE_TOOLS = low_risk_native_write_tool_names()
AUTO_APPROVED_NATIVE_TOOLS = auto_approved_native_tool_names()
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
# Read verbs mirror ``kortny.composio.tool_cards.READ_VERBS`` so the approval
# gate's read detection agrees with the catalog's ``side_effect`` classification.
# Without this, a tag-less Composio read (GMAIL_FETCH_EMAILS, GITHUB_LIST_ISSUES,
# GOOGLECALENDAR_EVENTS_LIST_ALL_CALENDARS) has no write verb and no read-only
# tag, so it falls through to "write" and wrongly demands approval — gating a
# read of the user's own data (violates gate-egress-not-compute).
READ_VERBS = frozenset(
    {
        "crawl",
        "fetch",
        "find",
        "get",
        "inspect",
        "list",
        "query",
        "read",
        "retrieve",
        "scrape",
        "search",
        "summarize",
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


def assess_tool_risk(tool: Tool, args: JsonObject) -> RiskAssessment:
    """Build a :class:`RiskAssessment` for any tool, native or external.

    Native tools use their catalog metadata directly. External Composio/MCP
    tools synthesise a ``ToolMetadata`` from their surfaced hints — MCP
    ``readOnlyHint``/``destructiveHint`` and Composio read-only tags / verb
    slugs map onto ``side_effect`` (read/write/destructive) and capability
    words — before running the deterministic classifier.
    """

    from kortny.autonomy import classify_tool_risk
    from kortny.tools.catalog import NATIVE_TOOL_METADATA, ToolMetadata

    native = NATIVE_TOOL_METADATA.get(tool.name)
    if native is not None:
        return classify_tool_risk(native, args)

    side_effect, capabilities = _external_tool_risk_shape(tool)
    synthetic = ToolMetadata(
        name=tool.name,
        namespace="external.tool",
        category="External",
        display_name=tool.name,
        capabilities=capabilities,
        side_effect=side_effect,
    )
    return classify_tool_risk(synthetic, args)


def _external_tool_risk_shape(
    tool: Tool,
) -> tuple[ToolSideEffectLiteral, tuple[str, ...]]:
    """Infer (side_effect, capability words) for an external tool's hints."""

    inner = getattr(tool, "tool", None)

    # HIG-169 P0.2/P0.3: ``readOnlyHint`` is attacker-asserted metadata. When a
    # tool explicitly signals its read-only claim must NOT be honored as an
    # approval bypass (untrusted server, or drifted/unpinned schema), suppress
    # every read-only signal — including the raw ``server_tool`` fallback — so
    # the tool falls back to side-effect-based gating. Tools without the flag
    # (native, Composio) keep the existing behavior.
    bypass_allowed = getattr(tool, "read_only_bypass_allowed", True)

    # MCP read-only / destructive annotation hints take precedence when present.
    read_only_hint = getattr(inner, "read_only_hint", None)
    destructive_hint = getattr(inner, "destructive_hint", None)
    server_tool = getattr(tool, "server_tool", None)
    if read_only_hint is None and server_tool is not None:
        read_only_hint = getattr(server_tool, "read_only_hint", None)
    if destructive_hint is None and server_tool is not None:
        destructive_hint = getattr(server_tool, "destructive_hint", None)

    verbs = _risky_verbs(tool)

    if destructive_hint is True or (verbs & _DESTRUCTIVE_VERBS):
        return "destructive", tuple(sorted(verbs)) or ("delete",)
    read_only_claim = _tool_is_explicitly_read_only(tool) or read_only_hint is True
    if read_only_claim and bypass_allowed:
        return "read", ()
    if verbs:
        return "write", tuple(sorted(verbs))
    # No explicit read claim and no write/destructive verb: fall back to the
    # verb lexicon. A slug whose verb is read-only (fetch/list/get/find/...) is a
    # read — matching ``tool_cards.is_read_only`` so the approval gate agrees
    # with the catalog's ``side_effect``. ``bypass_allowed`` is False only for an
    # untrusted/drifted MCP server (HIG-169), where we deliberately do NOT honor
    # a guessed read and fall through to gating.
    if bypass_allowed and _read_verbs(tool):
        return "read", ()
    # Unknown shape: conservative — unknown write-ish defaults to implicit.
    return "write", ("unknown_external_write",)


_DESTRUCTIVE_VERBS = frozenset({"delete", "remove", "archive", "cancel"})


def _tool_is_explicitly_read_only(tool: Tool) -> bool:
    composio_tool = getattr(tool, "tool", None)
    tags = getattr(composio_tool, "tags", None)
    if not isinstance(tags, (tuple, list, set)):
        return False
    normalized = {
        str(tag).casefold().replace("-", "_").replace(" ", "_") for tag in tags
    }
    return bool(normalized & READ_ONLY_TAGS)


def _tool_word_set(tool: Tool) -> set[str]:
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
    return words


def _risky_verbs(tool: Tool) -> set[str]:
    return _tool_word_set(tool) & WRITE_VERBS


def _read_verbs(tool: Tool) -> set[str]:
    return _tool_word_set(tool) & READ_VERBS
