"""Lethal-trifecta flow gate (HIG-169 P0.4).

The lethal trifecta is the combination that makes prompt injection dangerous:
**private data + untrusted content + an egress / outward action**. Spotlighting
and classifiers only lower the probability that an injected instruction is
obeyed; breaking the trifecta is the only deterministic *guarantee*.

This module is a deterministic state machine, no LLM. It tracks, per task,
whether any untrusted-origin content has entered the context — tool results
from web search, Composio, MCP, or Slack file reads (and observed channel
content, which the caller arms at task start). Once untrusted content has
entered, any subsequently-selected tool that writes, is destructive, or
communicates outward is *escalated*: it must get user approval for the rest of
the task even if the autonomy ladder would have auto-approved it.

Respecting HIG-223: the gate only ever raises the approval floor. It can ADD an
approval requirement, never remove one. At the ``autonomous`` level the user has
opted into more, but a trifecta-armed outward/destructive action still gates —
the injected-instruction blast radius is exactly what autonomy did not consent
to.
"""

from __future__ import annotations

from kortny.tools.catalog import NATIVE_TOOL_METADATA

# Native tools whose results introduce untrusted, attacker-influenceable content
# into the context. ``web_search`` returns arbitrary web pages; ``slack_file_read``
# returns file bodies a third party may have uploaded.
_UNTRUSTED_NATIVE_TOOLS = frozenset({"web_search", "slack_file_read"})

# Runtime-name prefixes for external providers. Every Composio/MCP tool result
# is third-party content by construction.
_EXTERNAL_TOOL_PREFIXES = ("mcp__", "composio__")

# Native tool namespaces that constitute an egress / outward / persistence
# action — the dangerous leg of the lethal trifecta. A native "write" tool is
# escalated ONLY when it actually moves data outward, persists it, or schedules
# a future autonomous action; purely local compute/artifact writes
# (native.documents PDF generation, sandbox-local file writes / shell / code)
# are NOT egress and stay free even when the gate is armed.
_OUTWARD_NATIVE_NAMESPACES = frozenset(
    {
        "native.slack",  # posting into Slack channels is outbound communication
        "native.deploy",  # publishes externally
        "native.memory",  # persists possibly-injected content into memory
        "native.scheduler",  # arms a future autonomous action
    }
)
# NOTE: native.skills is intentionally NOT outward. run_skill_script runs
# vetted ('trusted'-tier) skill code inside the same network-none sandbox as
# code_exec/sandbox_bash — it has no egress channel, so it's sandbox-local
# compute, not the trifecta's egress leg (see _TRIFECTA_FREE_NATIVE_TOOLS).
# load_skill/load_skill_resource are read-only and never escalate. The real
# egress for skill output is the export/preview/deploy/post tools below.

# Specific native tools that deliver content out of the sandbox to the user,
# plus the always-outward deploy tool. ``slack_add_reaction`` is intentionally
# excluded — an emoji reaction carries no exfiltration payload.
_OUTWARD_NATIVE_TOOLS = frozenset(
    {
        "deploy_site",
        "sandbox_publish_preview",
        "sandbox_export_artifact",
    }
)
# Sandbox-local compute: destructive to the throwaway container, but network-
# isolated, so not the egress leg. (They already carry their own approval gate
# at the base level; the trifecta gate must not add a second, redundant prompt.)
_TRIFECTA_FREE_NATIVE_TOOLS = frozenset(
    {
        "slack_add_reaction",
        # Pinning an existing message creates no new outbound payload (parity
        # with slack_add_reaction) — not an egress leg.
        "slack_pin_message",
        "code_exec",
        "sandbox_bash",
        "sandbox_write_file",
        "sandbox_read_file",
        # Vetted ('trusted'-tier) skill code in the same network-none sandbox as
        # code_exec — no egress channel, so not the trifecta's egress leg.
        "run_skill_script",
    }
)


def is_untrusted_origin_tool(tool_name: str) -> bool:
    """True when this tool's RESULT brings untrusted content into the context.

    External provider tools (Composio / MCP) are always untrusted-origin; among
    native tools only the small allowlist of external-content fetchers count.
    First-party native tools that act on Kortny's own state are trusted-origin.
    """

    name = tool_name.casefold()
    if name.startswith(_EXTERNAL_TOOL_PREFIXES):
        return True
    return name in _UNTRUSTED_NATIVE_TOOLS


def is_outward_or_write_tool(tool_name: str) -> bool:
    """True when this tool communicates outward, persists, or is destructive.

    These are the egress / persistence actions the trifecta gate escalates once
    untrusted content has armed the task. Read-only native tools — and purely
    local compute/artifact tools (PDF generation, sandbox-local file writes,
    shell, code execution) — stay free even when armed: they do not move data
    out, so they are not the trifecta's egress leg. External provider tools are
    treated conservatively as outward (their writes leave the workspace).
    """

    name = tool_name.casefold()
    # External (Composio / MCP) tool: its writes leave the workspace.
    if name.startswith(_EXTERNAL_TOOL_PREFIXES):
        return True
    if name in _TRIFECTA_FREE_NATIVE_TOOLS:
        return False
    if name in _OUTWARD_NATIVE_TOOLS:
        return True
    native = NATIVE_TOOL_METADATA.get(tool_name)
    if native is None:
        # Unknown non-prefixed tool: conservative — treat any write as outward.
        return False
    if native.side_effect == "destructive":
        return True
    if native.side_effect == "write":
        return native.namespace in _OUTWARD_NATIVE_NAMESPACES
    return False


class TrifectaGateState:
    """Per-task armed/disarmed state for the trifecta flow gate.

    Single-task, in-memory, no persistence: a task either has seen untrusted
    content (armed) or has not. The coordinator instantiates one per task run.
    """

    __slots__ = ("_armed", "_armed_by", "enabled")

    def __init__(self, *, enabled: bool = True, armed: bool = False) -> None:
        self.enabled = enabled
        self._armed = armed
        self._armed_by: str | None = "initial_context" if armed else None

    @property
    def armed(self) -> bool:
        return self._armed

    @property
    def armed_by(self) -> str | None:
        """The source that first armed the gate (a tool name or context note)."""

        return self._armed_by

    def note_tool_result(self, tool_name: str) -> bool:
        """Record a tool result; arm the gate if it was untrusted-origin.

        Returns True only on the transition from disarmed to armed, so the
        coordinator emits exactly one arming audit event per task.
        """

        if not self.enabled or self._armed:
            return False
        if is_untrusted_origin_tool(tool_name):
            self._armed = True
            self._armed_by = tool_name
            return True
        return False

    def should_escalate(self, tool_name: str) -> bool:
        """True when an armed gate must escalate approval for this tool call."""

        if not self.enabled or not self._armed:
            return False
        return is_outward_or_write_tool(tool_name)


__all__ = [
    "TrifectaGateState",
    "is_outward_or_write_tool",
    "is_untrusted_origin_tool",
]
