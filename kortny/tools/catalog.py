"""Native tool metadata catalog."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from kortny.config import Settings
from kortny.execution.sandbox import SandboxResourceLimits, ToolSandboxPolicy
from kortny.tools.types import JsonObject, JsonSchema, Tool

ToolSideEffect = Literal["read", "write", "destructive"]
ToolApproval = Literal["none", "self_gated", "user_approval", "admin_approval"]

# HIG-195: default registry-enforced per-tool execution deadline (seconds).
DEFAULT_TOOL_TIMEOUT_SECONDS = 60
# Margin added on top of a sandbox tool's inner timeout so the inner sandbox
# limit (which can clean up its container) fires before the outer registry
# deadline (which cannot kill the lingering worker thread) does.
SANDBOX_TIMEOUT_MARGIN_SECONDS = 30


@dataclass(frozen=True, slots=True)
class ToolMetadata:
    """Static metadata Kortny owns about a native tool."""

    name: str
    namespace: str
    category: str
    display_name: str
    capabilities: tuple[str, ...]
    side_effect: ToolSideEffect
    integration: str = ""
    approval: ToolApproval = "none"
    # HIG-195: per-tool execution deadline (seconds). The registry wraps every
    # invocation with this deadline so an unbounded in-process tool cannot hang
    # the worker until the lease expires. Sandbox tools set this above their
    # inner sandbox limit so the inner limit fires first; quick lookups set it
    # short. 0 disables the registry deadline (used only by the test echo probe).
    timeout_seconds: int = 60
    runtime_registered: bool = True
    dashboard_exposed: bool = True
    context_hint: bool = False
    required_env_vars: tuple[str, ...] = ()
    required_slack_scopes: tuple[str, ...] = ()
    plan_gates: tuple[str, ...] = ()
    result_budget: str = "normal"
    notes: tuple[str, ...] = ()
    can_replace_native_tools: tuple[str, ...] = ()
    sandbox: ToolSandboxPolicy = field(default_factory=ToolSandboxPolicy)


@dataclass(frozen=True, slots=True)
class ToolDescriptor:
    """Runtime descriptor produced from metadata plus a tool schema."""

    name: str
    namespace: str
    integration: str
    category: str
    display_name: str
    description: str
    parameters: JsonSchema
    capabilities: tuple[str, ...]
    side_effect: ToolSideEffect
    approval: ToolApproval
    timeout_seconds: int
    required_env_vars: tuple[str, ...]
    required_slack_scopes: tuple[str, ...]
    plan_gates: tuple[str, ...]
    result_budget: str
    notes: tuple[str, ...]
    can_replace_native_tools: tuple[str, ...]
    sandbox: ToolSandboxPolicy
    enabled: bool
    disabled_reason: str | None
    required_args: tuple[str, ...]
    optional_args: tuple[str, ...]

    def to_payload(self) -> JsonObject:
        """Return a JSON-safe descriptor for tools and dashboard surfaces."""

        return {
            "name": self.name,
            "namespace": self.namespace,
            "integration": self.integration,
            "category": self.category,
            "display_name": self.display_name,
            "description": self.description,
            "parameters": self.parameters,
            "capabilities": list(self.capabilities),
            "side_effect": self.side_effect,
            "approval": self.approval,
            "timeout_seconds": self.timeout_seconds,
            "required_env_vars": list(self.required_env_vars),
            "required_slack_scopes": list(self.required_slack_scopes),
            "plan_gates": list(self.plan_gates),
            "result_budget": self.result_budget,
            "notes": list(self.notes),
            "can_replace_native_tools": list(self.can_replace_native_tools),
            "sandbox": self.sandbox.to_payload(),
            "enabled": self.enabled,
            "disabled_reason": self.disabled_reason,
            "required_args": list(self.required_args),
            "optional_args": list(self.optional_args),
        }


_WORKBENCH_SANDBOX_POLICY = ToolSandboxPolicy(
    requires_sandbox=True,
    profile="workbench",
    network="none",
    resource_limits=SandboxResourceLimits(
        cpus=2.0,
        memory_mb=2048,
        pids_limit=512,
        timeout_seconds=300,
    ),
    reason="Workbench commands run in the task's persistent sandbox session.",
)
# Registry deadline for workbench tools: inner sandbox limit (300s) plus margin
# so the sandbox container limit fires before the un-killable worker thread.
_WORKBENCH_TIMEOUT_SECONDS = (
    _WORKBENCH_SANDBOX_POLICY.resource_limits.timeout_seconds
    + SANDBOX_TIMEOUT_MARGIN_SECONDS
)

NATIVE_TOOL_METADATA: dict[str, ToolMetadata] = {
    "find_tools": ToolMetadata(
        name="find_tools",
        namespace="native.meta",
        category="Runtime",
        display_name="Find tools",
        capabilities=("tool_retrieval", "capability_lookup"),
        side_effect="read",
        approval="none",
        # Constructed in the executor with runtime deps (retriever + loader +
        # the live registry), not via the native factory, so it is excluded
        # from the factory-registration invariant.
        runtime_registered=False,
        result_budget="bounded_results",
        notes=("Agent-driven tool retrieval; loads external tools on demand.",),
    ),
    # ------------------------------------------------------------------
    # Browser tools (Playwright-MCP, HIG-282)
    # runtime_registered=False because lifecycle-managed separately in
    # agent_executor (the holder must be closed at task end).
    # ------------------------------------------------------------------
    "browser_navigate": ToolMetadata(
        name="browser_navigate",
        namespace="native.browser",
        category="Browser",
        display_name="Browser navigate",
        capabilities=("browser_navigate", "web_browse"),
        side_effect="read",
        approval="none",
        timeout_seconds=60,
        runtime_registered=False,
        required_env_vars=("KORTNY_BROWSER_URL",),
        plan_gates=("external_network",),
        result_budget="small_lookup",
        notes=("Navigates the browser to a URL via Playwright-MCP.",),
    ),
    "browser_snapshot": ToolMetadata(
        name="browser_snapshot",
        namespace="native.browser",
        category="Browser",
        display_name="Browser snapshot",
        capabilities=("browser_snapshot", "web_browse", "page_extraction"),
        side_effect="read",
        approval="none",
        timeout_seconds=60,
        runtime_registered=False,
        required_env_vars=("KORTNY_BROWSER_URL",),
        plan_gates=("external_network",),
        result_budget="large_text_compaction",
        notes=(
            "Returns the accessibility-tree snapshot of the current page; use element refs from this to click/type.",
        ),
    ),
    "browser_click": ToolMetadata(
        name="browser_click",
        namespace="native.browser",
        category="Browser",
        display_name="Browser click",
        capabilities=("browser_interact", "web_browse"),
        side_effect="write",
        approval="self_gated",
        timeout_seconds=60,
        runtime_registered=False,
        required_env_vars=("KORTNY_BROWSER_URL",),
        plan_gates=("external_network",),
        result_budget="normal",
        notes=(
            "Clicks a browser element by ref from a snapshot. May trigger page navigation or form submission.",
        ),
    ),
    "browser_type": ToolMetadata(
        name="browser_type",
        namespace="native.browser",
        category="Browser",
        display_name="Browser type",
        capabilities=("browser_interact", "web_browse"),
        side_effect="write",
        approval="self_gated",
        timeout_seconds=60,
        runtime_registered=False,
        required_env_vars=("KORTNY_BROWSER_URL",),
        plan_gates=("external_network",),
        result_budget="normal",
        notes=(
            "Types text into a browser element by ref from a snapshot. Set submit=true to press Enter after typing.",
        ),
    ),
    "browser_take_screenshot": ToolMetadata(
        name="browser_take_screenshot",
        namespace="native.browser",
        category="Browser",
        display_name="Browser screenshot",
        capabilities=("browser_screenshot", "web_browse"),
        side_effect="read",
        approval="none",
        timeout_seconds=60,
        runtime_registered=False,
        required_env_vars=("KORTNY_BROWSER_URL",),
        plan_gates=("external_network",),
        result_budget="normal",
        notes=("Takes a screenshot of the current browser page.",),
    ),
    "browser_wait_for": ToolMetadata(
        name="browser_wait_for",
        namespace="native.browser",
        category="Browser",
        display_name="Browser wait",
        capabilities=("browser_wait", "web_browse"),
        side_effect="read",
        approval="none",
        timeout_seconds=120,
        runtime_registered=False,
        required_env_vars=("KORTNY_BROWSER_URL",),
        plan_gates=("external_network",),
        result_budget="normal",
        notes=("Waits for text to appear on the page or for a time duration.",),
    ),
    "browser_navigate_back": ToolMetadata(
        name="browser_navigate_back",
        namespace="native.browser",
        category="Browser",
        display_name="Browser navigate back",
        capabilities=("browser_navigate", "web_browse"),
        side_effect="read",
        approval="none",
        timeout_seconds=30,
        runtime_registered=False,
        required_env_vars=("KORTNY_BROWSER_URL",),
        plan_gates=("external_network",),
        result_budget="small_lookup",
        notes=("Navigates the browser back to the previous page.",),
    ),
    "web_search": ToolMetadata(
        name="web_search",
        namespace="native.research",
        category="Research",
        display_name="Web search",
        capabilities=("web_search", "current_research"),
        side_effect="read",
        # Brave Search client uses a 10s httpx timeout; give the registry a
        # little headroom over that inner network timeout.
        timeout_seconds=20,
        required_env_vars=("BRAVE_SEARCH_API_KEY",),
        plan_gates=("external_network",),
        result_budget="bounded_results",
        notes=("Uses Brave Search when configured.",),
    ),
    "slack_channel_history": ToolMetadata(
        name="slack_channel_history",
        namespace="native.slack",
        category="Slack context",
        display_name="Slack channel history",
        capabilities=("slack_context", "channel_summary", "decision_recall"),
        side_effect="read",
        required_env_vars=("SLACK_BOT_TOKEN",),
        required_slack_scopes=(
            "channels:history",
            "groups:history",
            "im:history",
            "mpim:history",
        ),
        context_hint=True,
        plan_gates=("slack_rate_limited",),
        result_budget="history_window",
        notes=("Uses observed local cache first when available.",),
    ),
    "search_observed_slack_history": ToolMetadata(
        name="search_observed_slack_history",
        namespace="native.slack",
        category="Slack context",
        display_name="Search observed Slack history",
        capabilities=(
            "slack_context",
            "observed_history_search",
            "decision_recall",
            "channel_memory",
        ),
        side_effect="read",
        required_env_vars=("POSTGRES_URL",),
        context_hint=True,
        plan_gates=("scope_guarded_context",),
        result_budget="bounded_results",
        notes=(
            "Searches Kortny's local observed Slack cache without Slack API calls.",
        ),
    ),
    "resolve_slack_identity": ToolMetadata(
        name="resolve_slack_identity",
        namespace="native.slack",
        category="Slack context",
        display_name="Resolve Slack identity",
        capabilities=(
            "slack_identity_resolution",
            "user_name_resolution",
            "channel_name_resolution",
            "slack_context",
        ),
        side_effect="read",
        # Local cache lookup with no external network call: keep it snappy.
        timeout_seconds=15,
        required_env_vars=("POSTGRES_URL",),
        context_hint=True,
        plan_gates=("scope_guarded_context",),
        result_budget="small_lookup",
        notes=(
            "Resolves cached Slack user and channel names without Slack API calls.",
        ),
    ),
    "slack_user_info": ToolMetadata(
        name="slack_user_info",
        namespace="native.slack",
        category="Slack context",
        display_name="Slack user info",
        capabilities=(
            "slack_identity_resolution",
            "slack_user_info",
            "user_name_resolution",
        ),
        side_effect="read",
        required_env_vars=("SLACK_BOT_TOKEN", "POSTGRES_URL"),
        required_slack_scopes=("users:read",),
        context_hint=True,
        plan_gates=("slack_rate_limited", "identity_cache_refresh"),
        result_budget="small_lookup",
        notes=(
            "Refreshes missing or stale user identity cache entries with Slack users.info.",
        ),
    ),
    "slack_channel_info": ToolMetadata(
        name="slack_channel_info",
        namespace="native.slack",
        category="Slack context",
        display_name="Slack channel info",
        capabilities=(
            "slack_identity_resolution",
            "slack_channel_info",
            "channel_name_resolution",
        ),
        side_effect="read",
        required_env_vars=("SLACK_BOT_TOKEN", "POSTGRES_URL"),
        required_slack_scopes=(
            "channels:read",
            "groups:read",
            "im:read",
            "mpim:read",
        ),
        context_hint=True,
        plan_gates=(
            "slack_rate_limited",
            "current_channel_only",
            "identity_cache_refresh",
        ),
        result_budget="small_lookup",
        notes=(
            "Refreshes the current channel identity cache entry with Slack conversations.info.",
        ),
    ),
    "slack_reply_thread": ToolMetadata(
        name="slack_reply_thread",
        namespace="native.slack",
        category="Slack actions",
        display_name="Reply in Slack thread",
        capabilities=("slack_reply", "thread_reply", "slack_write"),
        side_effect="write",
        approval="none",
        required_env_vars=("SLACK_BOT_TOKEN",),
        required_slack_scopes=("chat:write",),
        plan_gates=("current_thread_only",),
        result_budget="visible_slack_message",
        notes=(
            "Scoped to the current task channel and thread.",
            "Do not use for ordinary final answers because the final response posts automatically.",
        ),
    ),
    "slack_add_reaction": ToolMetadata(
        name="slack_add_reaction",
        namespace="native.slack",
        category="Slack actions",
        display_name="Add Slack reaction",
        capabilities=("slack_reaction", "slack_acknowledgement", "slack_write"),
        side_effect="write",
        approval="none",
        required_env_vars=("SLACK_BOT_TOKEN",),
        required_slack_scopes=("reactions:write",),
        plan_gates=("current_message_only",),
        result_budget="small_action",
        notes=("Scoped to the current triggering Slack message.",),
    ),
    "slack_pin_message": ToolMetadata(
        name="slack_pin_message",
        namespace="native.slack",
        category="Slack actions",
        display_name="Pin Slack message",
        capabilities=("slack_pin", "slack_write", "message_visibility"),
        side_effect="write",
        approval="none",
        required_env_vars=("SLACK_BOT_TOKEN",),
        required_slack_scopes=("pins:write",),
        plan_gates=("current_message_only",),
        result_budget="small_action",
        notes=("Scoped to the current triggering Slack message.",),
    ),
    "slack_add_bookmark": ToolMetadata(
        name="slack_add_bookmark",
        namespace="native.slack",
        category="Slack actions",
        display_name="Add Slack bookmark",
        capabilities=("slack_bookmark", "slack_write", "channel_resource"),
        side_effect="write",
        approval="none",
        required_env_vars=("SLACK_BOT_TOKEN",),
        required_slack_scopes=("bookmarks:write",),
        plan_gates=("current_channel_only", "link_bookmarks_only"),
        result_budget="small_action",
        notes=("Adds link bookmarks only, scoped to the current Slack channel.",),
    ),
    "slack_create_channel_canvas": ToolMetadata(
        name="slack_create_channel_canvas",
        namespace="native.slack",
        category="Slack actions",
        display_name="Create channel canvas",
        capabilities=(
            "slack_canvas",
            "slack_canvas_create",
            "channel_documentation",
            "slack_write",
        ),
        side_effect="write",
        approval="none",
        required_env_vars=("SLACK_BOT_TOKEN",),
        required_slack_scopes=("canvases:write",),
        plan_gates=("current_channel_only", "no_dm_canvas", "markdown_canvas_only"),
        result_budget="visible_channel_resource",
        notes=(
            "Creates the current Slack channel canvas only.",
            "Use when the user explicitly asks for a channel canvas or channel hub.",
        ),
    ),
    "slack_edit_canvas": ToolMetadata(
        name="slack_edit_canvas",
        namespace="native.slack",
        category="Slack actions",
        display_name="Edit Slack canvas",
        capabilities=(
            "slack_canvas",
            "slack_canvas_edit",
            "channel_documentation",
            "slack_write",
        ),
        side_effect="write",
        approval="none",
        required_env_vars=("SLACK_BOT_TOKEN",),
        required_slack_scopes=("canvases:write",),
        plan_gates=("one_canvas_change_per_call",),
        result_budget="visible_channel_resource",
        notes=(
            "Edits a Slack canvas with one operation per call; defaults "
            "to the current channel canvas when no canvas_id is given.",
            "Supports append, insert, replace, and rename operations.",
        ),
    ),
    "slack_lookup_canvas_sections": ToolMetadata(
        name="slack_lookup_canvas_sections",
        namespace="native.slack",
        category="Slack context",
        display_name="Lookup Slack canvas sections",
        capabilities=(
            "slack_canvas",
            "slack_canvas_sections",
            "channel_documentation",
            "slack_context",
        ),
        side_effect="read",
        required_env_vars=("SLACK_BOT_TOKEN",),
        required_slack_scopes=("canvases:read",),
        context_hint=True,
        plan_gates=("criteria_required",),
        result_budget="small_lookup",
        notes=(
            "Finds section IDs in a Slack canvas; defaults to the current "
            "channel canvas when no canvas_id is given.",
            "Use before targeted canvas edits when only a heading or phrase is known.",
        ),
    ),
    "slack_file_read": ToolMetadata(
        name="slack_file_read",
        namespace="native.slack",
        category="Slack context",
        display_name="Slack file reader",
        capabilities=("slack_file_read", "file_analysis"),
        side_effect="read",
        required_env_vars=("SLACK_BOT_TOKEN",),
        required_slack_scopes=("files:read",),
        plan_gates=("untrusted_file_input",),
        result_budget="large_text_compaction",
        notes=("Extracts text from supported Slack file attachments.",),
    ),
    "pdf_generator": ToolMetadata(
        name="pdf_generator",
        namespace="native.documents",
        category="Documents",
        display_name="PDF generator",
        capabilities=("document_generation", "artifact_generation"),
        side_effect="write",
        approval="none",
        # Document rendering (PDF generation) legitimately runs longer than a
        # quick tool call.
        timeout_seconds=180,
        result_budget="artifact",
        notes=("Creates local task artifacts only when explicitly useful.",),
    ),
    "document_studio": ToolMetadata(
        name="document_studio",
        namespace="native.documents",
        category="Documents",
        display_name="Document Studio",
        capabilities=(
            "document_generation",
            "artifact_generation",
            "editorial_pdf",
            "presentation_generation",
            "word_document_generation",
        ),
        side_effect="write",
        approval="none",
        # Themed multi-page rendering (Typst PDF / python-pptx / python-docx);
        # allow headroom.
        timeout_seconds=180,
        result_budget="artifact",
        notes=(
            "Editorial-grade themed PDF, PowerPoint, or Word from one structured "
            "block IR. Preferred over pdf_generator for polished deliverables.",
        ),
    ),
    "code_exec": ToolMetadata(
        name="code_exec",
        namespace="native.execution",
        category="Execution",
        display_name="Code execution",
        capabilities=("sandboxed_code_execution", "python_execution"),
        side_effect="destructive",
        # Sandbox execution runs auto-approved: the runner is network-isolated,
        # host-filesystem-free, and resource-capped, so gating it added
        # coworker friction without a safety win.
        approval="none",
        # Inner sandbox limit is 30s; the registry deadline sits above it
        # (SANDBOX_TIMEOUT_MARGIN_SECONDS) so the sandbox limit fires first.
        timeout_seconds=30 + SANDBOX_TIMEOUT_MARGIN_SECONDS,
        required_env_vars=("KORTNY_SANDBOX_RUNNER_URL",),
        plan_gates=(
            "sandbox_required",
            "network_disabled",
        ),
        result_budget="bounded_stdout",
        notes=(
            "Runs short Python snippets only through the isolated sandbox runner.",
            "No network, package installation, secrets, or host filesystem access.",
        ),
        sandbox=ToolSandboxPolicy(
            requires_sandbox=True,
            profile="code",
            network="none",
            resource_limits=SandboxResourceLimits(
                cpus=1.0,
                memory_mb=512,
                pids_limit=64,
                timeout_seconds=30,
            ),
            reason="Untrusted code must run outside the worker process.",
        ),
    ),
    "sandbox_bash": ToolMetadata(
        name="sandbox_bash",
        namespace="native.execution",
        category="Execution",
        display_name="Sandbox shell",
        capabilities=(
            "sandboxed_code_execution",
            "shell_execution",
            "build_and_test",
        ),
        side_effect="destructive",
        approval="none",
        timeout_seconds=_WORKBENCH_TIMEOUT_SECONDS,
        required_env_vars=("KORTNY_SANDBOX_RUNNER_URL",),
        plan_gates=(
            "sandbox_required",
            "network_disabled",
        ),
        result_budget="bounded_stdout",
        notes=(
            "Runs shell commands in the task's persistent sandbox workspace.",
            "Auto-approved: confined to the isolated per-task sandbox.",
        ),
        sandbox=_WORKBENCH_SANDBOX_POLICY,
    ),
    "sandbox_write_file": ToolMetadata(
        name="sandbox_write_file",
        namespace="native.execution",
        category="Execution",
        display_name="Sandbox file writer",
        capabilities=("sandboxed_code_execution", "file_generation"),
        side_effect="write",
        approval="none",
        timeout_seconds=_WORKBENCH_TIMEOUT_SECONDS,
        required_env_vars=("KORTNY_SANDBOX_RUNNER_URL",),
        plan_gates=("sandbox_required",),
        result_budget="normal",
        notes=("Writes files into the task's sandbox workspace.",),
        sandbox=_WORKBENCH_SANDBOX_POLICY,
    ),
    "sandbox_stage_file": ToolMetadata(
        name="sandbox_stage_file",
        namespace="native.execution",
        category="Execution",
        display_name="Sandbox file staging",
        capabilities=("slack_file_staging", "document_editing", "sandbox_required"),
        side_effect="write",
        approval="none",
        timeout_seconds=_WORKBENCH_TIMEOUT_SECONDS,
        required_env_vars=("KORTNY_SANDBOX_RUNNER_URL",),
        plan_gates=("sandbox_required",),
        result_budget="normal",
        notes=(
            "Downloads a Slack file's binary bytes and stages them into the sandbox workspace.",
            "Binary-safe; size-capped at 5 MB. Use before editing uploaded docs with skill scripts.",
        ),
        sandbox=_WORKBENCH_SANDBOX_POLICY,
    ),
    "sandbox_read_file": ToolMetadata(
        name="sandbox_read_file",
        namespace="native.execution",
        category="Execution",
        display_name="Sandbox file reader",
        capabilities=("sandboxed_code_execution", "file_inspection"),
        side_effect="read",
        approval="none",
        timeout_seconds=_WORKBENCH_TIMEOUT_SECONDS,
        required_env_vars=("KORTNY_SANDBOX_RUNNER_URL",),
        plan_gates=("sandbox_required",),
        result_budget="large_text_compaction",
        notes=("Reads files from the task's sandbox workspace.",),
        sandbox=_WORKBENCH_SANDBOX_POLICY,
    ),
    "sandbox_export_artifact": ToolMetadata(
        name="sandbox_export_artifact",
        namespace="native.execution",
        category="Execution",
        display_name="Sandbox artifact export",
        capabilities=(
            "sandboxed_code_execution",
            "artifact_generation",
        ),
        side_effect="write",
        approval="none",
        timeout_seconds=_WORKBENCH_TIMEOUT_SECONDS,
        required_env_vars=("KORTNY_SANDBOX_RUNNER_URL",),
        plan_gates=("sandbox_required",),
        result_budget="artifact",
        notes=("Exports sandbox files or zipped directories as task artifacts.",),
        sandbox=_WORKBENCH_SANDBOX_POLICY,
    ),
    "sandbox_publish_preview": ToolMetadata(
        name="sandbox_publish_preview",
        namespace="native.execution",
        category="Execution",
        display_name="Sandbox preview publisher",
        capabilities=(
            "sandboxed_code_execution",
            "artifact_generation",
            "web_preview",
        ),
        side_effect="write",
        approval="none",
        timeout_seconds=_WORKBENCH_TIMEOUT_SECONDS,
        required_env_vars=(
            "KORTNY_SANDBOX_RUNNER_URL",
            "KORTNY_ARTIFACTS_DIR",
            "KORTNY_PUBLIC_BASE_URL",
            "KORTNY_PREVIEW_SIGNING_SECRET",
        ),
        plan_gates=("sandbox_required",),
        result_budget="normal",
        notes=("Publishes static sites from the sandbox at signed preview URLs.",),
        sandbox=_WORKBENCH_SANDBOX_POLICY,
    ),
    "deploy_site": ToolMetadata(
        name="deploy_site",
        namespace="native.deploy",
        category="Deployment",
        display_name="Site deployer",
        capabilities=("site_deployment", "external_publishing"),
        side_effect="destructive",
        approval="user_approval",
        # Netlify/Vercel deploy posts use a 120s httpx timeout per request and
        # may chain several calls; give the registry generous headroom.
        timeout_seconds=300,
        plan_gates=(
            "explicit_user_request_required",
            "requester_approval_required",
            "external_network",
        ),
        result_budget="normal",
        notes=(
            "Deploys sandbox-built static sites to Netlify or Vercel from "
            "the trusted host; integration tokens never enter the sandbox.",
        ),
    ),
    "remember_fact": ToolMetadata(
        name="remember_fact",
        namespace="native.memory",
        category="Memory",
        display_name="Remember fact",
        capabilities=("workspace_memory", "memory_write"),
        side_effect="write",
        approval="self_gated",
        required_env_vars=("POSTGRES_URL",),
        plan_gates=("memory_confirmation",),
        notes=("Posts a Slack confirmation before saving durable memory.",),
    ),
    "recall_fact": ToolMetadata(
        name="recall_fact",
        namespace="native.memory",
        category="Memory",
        display_name="Recall fact",
        capabilities=("workspace_memory",),
        side_effect="read",
        required_env_vars=("POSTGRES_URL",),
    ),
    "inspect_memory": ToolMetadata(
        name="inspect_memory",
        namespace="native.memory",
        category="Memory",
        display_name="Inspect memory",
        capabilities=("workspace_memory", "memory_provenance"),
        side_effect="read",
        required_env_vars=("POSTGRES_URL",),
    ),
    "forget_fact": ToolMetadata(
        name="forget_fact",
        namespace="native.memory",
        category="Memory",
        display_name="Forget fact",
        capabilities=("workspace_memory", "memory_delete"),
        side_effect="write",
        approval="user_approval",
        required_env_vars=("POSTGRES_URL",),
        plan_gates=("memory_mutation",),
        notes=("Uses audit-preserving soft delete.",),
    ),
    "load_skill": ToolMetadata(
        name="load_skill",
        namespace="native.skills",
        category="Skills",
        display_name="Load skill",
        capabilities=("procedural_skills", "skill_instructions"),
        side_effect="read",
        required_env_vars=("POSTGRES_URL",),
        notes=(
            "Loads full SKILL.md instructions for a skill enabled in the "
            "task's scope; records the invocation.",
        ),
    ),
    "load_skill_resource": ToolMetadata(
        name="load_skill_resource",
        namespace="native.skills",
        category="Skills",
        display_name="Load skill resource",
        capabilities=("procedural_skills", "skill_resources"),
        side_effect="read",
        required_env_vars=("POSTGRES_URL",),
        notes=(
            "Reads one bundled reference/asset/script file from an enabled "
            "skill. Scripts are viewable, never executed.",
        ),
    ),
    "run_skill_script": ToolMetadata(
        name="run_skill_script",
        namespace="native.skills",
        category="Skills",
        display_name="Run skill script",
        capabilities=("procedural_skills", "skill_script_execution"),
        side_effect="write",
        approval="none",
        # Runs a skill script inside the persistent sandbox session.
        timeout_seconds=_WORKBENCH_TIMEOUT_SECONDS,
        required_env_vars=("POSTGRES_URL", "KORTNY_SANDBOX_RUNNER_URL"),
        notes=(
            "Runs a bundled script from a trusted skill inside the task's "
            "sandbox; only skills at trust level 'trusted' are runnable. "
            "Trust tier is the gate and the sandbox bounds blast radius, so no "
            "per-call approval is required. Never executes on the worker host.",
        ),
    ),
    "query_workspace_graph": ToolMetadata(
        name="query_workspace_graph",
        namespace="native.context",
        category="Workspace context",
        display_name="Workspace graph",
        capabilities=("workspace_graph", "firm_context", "relationship_lookup"),
        side_effect="read",
        required_env_vars=("POSTGRES_URL",),
        plan_gates=("scope_guarded_context",),
        result_budget="context_pack",
        notes=("Returns scope-safe current graph context with evidence.",),
    ),
    "declare_project": ToolMetadata(
        name="declare_project",
        namespace="native.context",
        category="Workspace context",
        display_name="Declare project",
        capabilities=("project_declaration", "workspace_graph"),
        side_effect="write",
        approval="none",
        required_env_vars=("POSTGRES_URL",),
        notes=(
            "Records a project boundary (name + channels) as graph hub + edges; "
            "internal knowledge only, no external egress.",
        ),
    ),
    "list_schedules": ToolMetadata(
        name="list_schedules",
        namespace="native.scheduler",
        category="Scheduling",
        display_name="List schedules",
        capabilities=("schedule_truth", "schedule_lookup"),
        side_effect="read",
        required_env_vars=("POSTGRES_URL",),
    ),
    "get_schedule": ToolMetadata(
        name="get_schedule",
        namespace="native.scheduler",
        category="Scheduling",
        display_name="Get schedule",
        capabilities=("schedule_truth", "schedule_lookup"),
        side_effect="read",
        required_env_vars=("POSTGRES_URL",),
    ),
    "create_schedule": ToolMetadata(
        name="create_schedule",
        namespace="native.scheduler",
        category="Scheduling",
        display_name="Create schedule",
        capabilities=("schedule_create", "recurring_work"),
        side_effect="write",
        approval="none",
        required_env_vars=("POSTGRES_URL",),
        plan_gates=("schedule_mutation",),
    ),
    "update_schedule": ToolMetadata(
        name="update_schedule",
        namespace="native.scheduler",
        category="Scheduling",
        display_name="Update schedule",
        capabilities=("schedule_update", "recurring_work"),
        side_effect="write",
        approval="none",
        required_env_vars=("POSTGRES_URL",),
        plan_gates=("schedule_mutation",),
    ),
    "pause_schedule": ToolMetadata(
        name="pause_schedule",
        namespace="native.scheduler",
        category="Scheduling",
        display_name="Pause schedule",
        capabilities=("schedule_pause", "recurring_work"),
        side_effect="write",
        approval="none",
        required_env_vars=("POSTGRES_URL",),
        plan_gates=("schedule_mutation",),
    ),
    "resume_schedule": ToolMetadata(
        name="resume_schedule",
        namespace="native.scheduler",
        category="Scheduling",
        display_name="Resume schedule",
        capabilities=("schedule_resume", "recurring_work"),
        side_effect="write",
        approval="none",
        required_env_vars=("POSTGRES_URL",),
        plan_gates=("schedule_mutation",),
    ),
    "cancel_schedule": ToolMetadata(
        name="cancel_schedule",
        namespace="native.scheduler",
        category="Scheduling",
        display_name="Cancel schedule",
        capabilities=("schedule_cancel", "recurring_work"),
        side_effect="write",
        approval="none",
        required_env_vars=("POSTGRES_URL",),
        plan_gates=("schedule_mutation",),
    ),
    "describe_tools": ToolMetadata(
        name="describe_tools",
        namespace="native.meta",
        category="Runtime",
        display_name="Describe tools",
        capabilities=("tool_inventory", "capability_lookup"),
        side_effect="read",
        notes=("Returns runtime truth about native tools and scoped integrations.",),
    ),
    "list_integrations": ToolMetadata(
        name="list_integrations",
        namespace="native.meta",
        category="Runtime",
        display_name="List integrations",
        capabilities=("tool_inventory", "capability_lookup"),
        side_effect="read",
        notes=("Compatibility alias for describe_tools.",),
    ),
    "echo": ToolMetadata(
        name="echo",
        namespace="native.diagnostics",
        category="Diagnostics",
        display_name="Echo",
        capabilities=("diagnostic",),
        side_effect="read",
        runtime_registered=False,
        dashboard_exposed=False,
        notes=("Test-only registry probe.",),
    ),
}


def tool_metadata(name: str) -> ToolMetadata:
    """Return metadata for a tool name, falling back to a conservative shape."""

    return NATIVE_TOOL_METADATA.get(
        name,
        ToolMetadata(
            name=name,
            namespace="native.other",
            category="Other",
            display_name=name.replace("_", " ").title(),
            capabilities=(),
            side_effect="read",
            notes=("No explicit metadata has been registered yet.",),
        ),
    )


def tool_timeout_seconds(name: str) -> int:
    """Return the registry-enforced execution deadline for a tool name.

    Native tools use their catalog ``timeout_seconds``. External tools
    (Composio/MCP) have no native metadata entry, so they fall back to the
    conservative default — their own provider clients still apply network-level
    timeouts underneath this deadline.
    """

    return tool_metadata(name).timeout_seconds


def read_only_native_tool_names() -> frozenset[str]:
    """Return native tools marked read-only in the metadata catalog."""

    return frozenset(
        name
        for name, metadata in NATIVE_TOOL_METADATA.items()
        if metadata.side_effect == "read"
    )


def native_tool_names_by_approval(approval: ToolApproval) -> frozenset[str]:
    """Return native tools with a specific approval classification."""

    return frozenset(
        name
        for name, metadata in NATIVE_TOOL_METADATA.items()
        if metadata.approval == approval
    )


def low_risk_native_write_tool_names() -> frozenset[str]:
    """Return write tools whose metadata declares no extra approval gate."""

    return frozenset(
        name
        for name, metadata in NATIVE_TOOL_METADATA.items()
        if metadata.side_effect == "write" and metadata.approval == "none"
    )


def auto_approved_native_tool_names() -> frozenset[str]:
    """Return native tools the catalog explicitly marks ``approval='none'``.

    Includes auto-approved sandbox/code tools whose ``side_effect`` is
    ``destructive`` but which run inside the isolated runner (code_exec,
    sandbox_*). These stay auto-approved at every autonomy level and never enter
    the HIG-223 ladder — the sandbox, not a human gate, is their safety boundary.
    """

    return frozenset(
        name
        for name, metadata in NATIVE_TOOL_METADATA.items()
        if metadata.approval == "none"
    )


def native_tool_integration_map() -> dict[str, str]:
    """Return tool name to integration family mapping derived from metadata."""

    return {
        name: _metadata_integration(metadata)
        for name, metadata in NATIVE_TOOL_METADATA.items()
    }


def dashboard_native_tool_names() -> tuple[str, ...]:
    """Return native tool names exposed on dashboard capability surfaces."""

    return tuple(
        name
        for name, metadata in NATIVE_TOOL_METADATA.items()
        if metadata.runtime_registered and metadata.dashboard_exposed
    )


def runtime_native_tool_names() -> tuple[str, ...]:
    """Return production native tool names expected in runtime registration."""

    return tuple(
        name
        for name, metadata in NATIVE_TOOL_METADATA.items()
        if metadata.runtime_registered
    )


def native_slack_context_hint_names() -> frozenset[str]:
    """Return likely-tool hints that can stay on local Slack context tools."""

    return frozenset(
        name for name, metadata in NATIVE_TOOL_METADATA.items() if metadata.context_hint
    )


def tool_descriptor(
    tool: Tool,
    *,
    settings: Settings | None = None,
    config_available: bool = True,
    enabled: bool | None = None,
    disabled_reason: str | None = None,
) -> ToolDescriptor:
    """Build a metadata-rich descriptor for a registered tool object."""

    metadata = tool_metadata(tool.name)
    return _descriptor(
        metadata=metadata,
        description=tool.description,
        parameters=tool.parameters,
        settings=settings,
        config_available=config_available,
        enabled=enabled,
        disabled_reason=disabled_reason,
    )


def tool_descriptor_from_class(
    tool: type[Any],
    *,
    settings: Settings | None = None,
    config_available: bool = True,
) -> ToolDescriptor:
    """Build a descriptor for dashboard/static inspection from a tool class."""

    metadata = tool_metadata(str(tool.name))
    return _descriptor(
        metadata=metadata,
        description=str(tool.description),
        parameters=dict(tool.parameters),
        settings=settings,
        config_available=config_available,
    )


def tool_descriptors(
    tools: Sequence[Tool],
    *,
    settings: Settings | None = None,
    config_available: bool = True,
) -> tuple[ToolDescriptor, ...]:
    """Build descriptors for registered tools in registry order."""

    return tuple(
        tool_descriptor(
            tool,
            settings=settings,
            config_available=config_available,
        )
        for tool in tools
    )


def _descriptor(
    *,
    metadata: ToolMetadata,
    description: str,
    parameters: JsonSchema,
    settings: Settings | None,
    config_available: bool,
    enabled: bool | None = None,
    disabled_reason: str | None = None,
) -> ToolDescriptor:
    required_args, optional_args = tool_argument_names(parameters)
    dynamic_disabled_reason = disabled_reason or _disabled_reason(
        metadata=metadata,
        settings=settings,
        config_available=config_available,
    )
    resolved_enabled = (
        enabled if enabled is not None else dynamic_disabled_reason is None
    )
    return ToolDescriptor(
        name=metadata.name,
        namespace=metadata.namespace,
        integration=_metadata_integration(metadata),
        category=metadata.category,
        display_name=metadata.display_name,
        description=description,
        parameters=parameters,
        capabilities=metadata.capabilities,
        side_effect=metadata.side_effect,
        approval=metadata.approval,
        timeout_seconds=metadata.timeout_seconds,
        required_env_vars=metadata.required_env_vars,
        required_slack_scopes=metadata.required_slack_scopes,
        plan_gates=metadata.plan_gates,
        result_budget=metadata.result_budget,
        notes=metadata.notes,
        can_replace_native_tools=metadata.can_replace_native_tools,
        sandbox=metadata.sandbox,
        enabled=resolved_enabled,
        disabled_reason=dynamic_disabled_reason if not resolved_enabled else None,
        required_args=required_args,
        optional_args=optional_args,
    )


def tool_argument_names(
    schema: JsonSchema,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return required and optional top-level argument names from JSON schema."""

    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return (), ()
    required_values = schema.get("required", ())
    required = tuple(
        name for name in required_values if isinstance(name, str) and name in properties
    )
    optional = tuple(
        name for name in properties if isinstance(name, str) and name not in required
    )
    return required, optional


def _disabled_reason(
    *,
    metadata: ToolMetadata,
    settings: Settings | None,
    config_available: bool,
) -> str | None:
    if not config_available:
        return "Runtime settings could not load."
    if settings is None:
        return None
    missing = _missing_required_env_vars(metadata, settings)
    if missing:
        return f"Missing required configuration: {', '.join(missing)}."
    return None


def _missing_required_env_vars(
    metadata: ToolMetadata,
    settings: Settings,
) -> tuple[str, ...]:
    return tuple(
        env_var
        for env_var in metadata.required_env_vars
        if not _settings_has_env_var(settings, env_var)
    )


def _settings_has_env_var(settings: Settings, env_var: str) -> bool:
    attr = _SETTINGS_ENV_ATTRS.get(env_var)
    if attr is None:
        return True
    value = getattr(settings, attr)
    if isinstance(value, str):
        return bool(value.strip())
    return value is not None


def _metadata_integration(metadata: ToolMetadata) -> str:
    if metadata.integration:
        return metadata.integration
    return _NAMESPACE_INTEGRATIONS.get(metadata.namespace, metadata.namespace)


_NAMESPACE_INTEGRATIONS = {
    "native.browser": "browser",
    "native.context": "workspace",
    "native.diagnostics": "diagnostics",
    "native.documents": "documents",
    "native.execution": "execution",
    "native.memory": "memory",
    "native.meta": "runtime",
    "native.research": "web",
    "native.scheduler": "scheduler",
    "native.slack": "slack",
}


_SETTINGS_ENV_ATTRS = {
    "BRAVE_SEARCH_API_KEY": "brave_search_api_key",
    "COMPOSIO_API_KEY": "composio_api_key",
    "KORTNY_BROWSER_URL": "browser_url",
    "POSTGRES_URL": "postgres_url",
    "KORTNY_SANDBOX_RUNNER_URL": "sandbox_runner_url",
    "SLACK_APP_TOKEN": "slack_app_token",
    "SLACK_BOT_TOKEN": "slack_bot_token",
    "SLACK_SIGNING_SECRET": "slack_signing_secret",
}
