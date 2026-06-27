"""Pure-unit tests for the HIG-169 P0 deterministic injection baseline.

No DB: covers the egress URL scanner, the schema fingerprint, the trifecta
state machine, and the approval-policy trust/pin gating over synthetic tools.
"""

from __future__ import annotations

from dataclasses import dataclass

from kortny.agent.coordinator import DEFAULT_SYSTEM_PROMPT
from kortny.agent.trifecta import (
    TrifectaGateState,
    is_outward_or_write_tool,
    is_untrusted_origin_tool,
)
from kortny.approvals import ApprovalScope, ToolApprovalPolicy
from kortny.autonomy import AutonomyLevel
from kortny.slack.egress import parse_egress_allowlist, scan_outbound_urls
from kortny.tools.pinning import compute_tool_fingerprint

# --- P0.1 egress scanner -----------------------------------------------------


def test_scan_flags_long_query_payload_to_external_host() -> None:
    text = "look https://evil.example.com/c?d=" + "a" * 80
    flagged = scan_outbound_urls(text)
    assert len(flagged) == 1
    assert flagged[0].host == "evil.example.com"


def test_scan_ignores_short_query_and_no_query() -> None:
    assert scan_outbound_urls("https://example.com/page?id=42") == ()
    assert scan_outbound_urls("https://example.com/page") == ()


def test_scan_respects_allowlist() -> None:
    text = "https://trusted.example.com/c?d=" + "a" * 80
    assert scan_outbound_urls(text, allowlist=frozenset({"trusted.example.com"})) == ()


def test_parse_allowlist_normalizes() -> None:
    assert parse_egress_allowlist(" A.com , b.com ") == frozenset({"a.com", "b.com"})
    assert parse_egress_allowlist(None) == frozenset()
    assert parse_egress_allowlist("") == frozenset()


# --- P0.3 fingerprint --------------------------------------------------------


def test_fingerprint_includes_input_schema() -> None:
    base = compute_tool_fingerprint(
        name="t", description="d", input_schema={"type": "object", "properties": {}}
    )
    changed = compute_tool_fingerprint(
        name="t",
        description="d",
        input_schema={"type": "object", "properties": {"x": {"type": "string"}}},
    )
    assert base.fingerprint != changed.fingerprint


def test_fingerprint_is_key_order_stable() -> None:
    a = compute_tool_fingerprint(
        name="t", description="d", input_schema={"a": 1, "b": 2}
    )
    b = compute_tool_fingerprint(
        name="t", description="d", input_schema={"b": 2, "a": 1}
    )
    assert a.fingerprint == b.fingerprint


def test_fingerprint_changes_on_description() -> None:
    a = compute_tool_fingerprint(name="t", description="d1", input_schema={})
    b = compute_tool_fingerprint(name="t", description="d2", input_schema={})
    assert a.fingerprint != b.fingerprint


# --- P0.4 trifecta classification + state ------------------------------------


def test_untrusted_origin_classification() -> None:
    assert is_untrusted_origin_tool("web_search")
    assert is_untrusted_origin_tool("slack_file_read")
    assert is_untrusted_origin_tool("mcp__server__fetch")
    assert is_untrusted_origin_tool("composio_github_get")
    # First-party native state tools are trusted-origin.
    assert not is_untrusted_origin_tool("list_schedules")
    assert not is_untrusted_origin_tool("inspect_memory")


def test_outward_or_write_classification() -> None:
    assert is_outward_or_write_tool("deploy_site")
    assert is_outward_or_write_tool("mcp__server__create_issue")
    assert is_outward_or_write_tool("composio_slack_send")
    # Native outward / persistence writes escalate.
    assert is_outward_or_write_tool("slack_reply_thread")
    assert is_outward_or_write_tool("remember_fact")
    assert is_outward_or_write_tool("create_schedule")
    # Read-only native tool stays free.
    assert not is_outward_or_write_tool("web_search")
    # Local compute / artifact writes are NOT egress — they stay free.
    assert not is_outward_or_write_tool("pdf_generator")
    assert not is_outward_or_write_tool("sandbox_write_file")
    assert not is_outward_or_write_tool("code_exec")
    assert not is_outward_or_write_tool("slack_add_reaction")


def test_gate_arms_only_after_untrusted_result() -> None:
    state = TrifectaGateState(enabled=True)
    assert not state.armed
    # A trusted-origin result does not arm.
    assert not state.note_tool_result("list_schedules")
    assert not state.armed
    # The first untrusted-origin result arms (returns True on the transition).
    assert state.note_tool_result("web_search")
    assert state.armed
    assert state.armed_by == "web_search"
    # Subsequent untrusted results do not re-trigger the transition.
    assert not state.note_tool_result("mcp__s__fetch")


def test_gate_escalates_outward_only_when_armed() -> None:
    state = TrifectaGateState(enabled=True)
    # Before arming, nothing escalates.
    assert not state.should_escalate("deploy_site")
    state.note_tool_result("web_search")
    # After arming, outward/write tools escalate; read-only stays free.
    assert state.should_escalate("deploy_site")
    assert not state.should_escalate("web_search")


def test_gate_disabled_never_arms() -> None:
    state = TrifectaGateState(enabled=False)
    assert not state.note_tool_result("web_search")
    assert not state.armed
    assert not state.should_escalate("deploy_site")


# --- P0.2 approval trust/pin gating ------------------------------------------


@dataclass
class _FakeMcpDescriptor:
    tags: tuple[str, ...]
    read_only_hint: bool | None
    destructive_hint: bool | None = None


@dataclass
class _FakeMcpServerTool:
    read_only_hint: bool | None
    destructive_hint: bool | None = None


class _FakeMcpTool:
    """Mimics McpExecuteTool's approval-relevant surface."""

    def __init__(self, *, read_only_bypass_allowed: bool) -> None:
        self.name = "mcp__poison__read_secrets"
        self.description = "read-only tool from an MCP server"
        self.read_only_bypass_allowed = read_only_bypass_allowed
        self.server_tool = _FakeMcpServerTool(read_only_hint=True)
        if read_only_bypass_allowed:
            self.tool: _FakeMcpDescriptor = _FakeMcpDescriptor(
                tags=("readonlyhint",), read_only_hint=True
            )
        else:
            self.tool = _FakeMcpDescriptor(tags=(), read_only_hint=None)


def test_trusted_pinned_read_only_clears_approval() -> None:
    # Trusted + pinned: the read-only claim is honored as a free (read) tool at
    # EVERY autonomy level — no approval, classified read, no write audit.
    policy = ToolApprovalPolicy()
    tool = _FakeMcpTool(read_only_bypass_allowed=True)
    for level in AutonomyLevel:
        requirement = policy.requirement_for(tool, {}, autonomy_level=level)
        assert requirement.scope is ApprovalScope.none
        assert not requirement.audit_autonomy


def test_untrusted_or_unpinned_read_only_does_not_clear_approval() -> None:
    # readOnlyHint is attacker-asserted: an untrusted/unpinned server's
    # read-only claim must NOT bypass approval — it falls back to write gating.
    policy = ToolApprovalPolicy()
    tool = _FakeMcpTool(read_only_bypass_allowed=False)
    # Conservative gates every external write -> user approval required.
    conservative = policy.requirement_for(
        tool, {}, autonomy_level=AutonomyLevel.conservative
    )
    assert conservative.scope is ApprovalScope.user
    # Balanced auto-approves the implicit write but AUDITS it — proving the tool
    # is now treated as a write, not a free read (the bypass is revoked).
    balanced = policy.requirement_for(tool, {}, autonomy_level=AutonomyLevel.balanced)
    assert balanced.audit_autonomy is True


# --- system prompt clause ----------------------------------------------------


def test_system_prompt_has_untrusted_data_clause() -> None:
    prompt = DEFAULT_SYSTEM_PROMPT.casefold()
    assert "untrusted" in prompt
    assert "ignore previous" in prompt
    assert "not instructions" in prompt or "not as instructions" in prompt
