"""Native tool metadata catalog."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from kortny.config import Settings
from kortny.tools.types import JsonObject, JsonSchema, Tool

ToolSideEffect = Literal["read", "write", "destructive"]
ToolApproval = Literal["none", "self_gated", "user_approval", "admin_approval"]


@dataclass(frozen=True, slots=True)
class ToolMetadata:
    """Static metadata Kortny owns about a native tool."""

    name: str
    namespace: str
    category: str
    display_name: str
    capabilities: tuple[str, ...]
    side_effect: ToolSideEffect
    approval: ToolApproval = "none"
    required_env_vars: tuple[str, ...] = ()
    required_slack_scopes: tuple[str, ...] = ()
    plan_gates: tuple[str, ...] = ()
    result_budget: str = "normal"
    notes: tuple[str, ...] = ()
    can_replace_native_tools: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ToolDescriptor:
    """Runtime descriptor produced from metadata plus a tool schema."""

    name: str
    namespace: str
    category: str
    display_name: str
    description: str
    parameters: JsonSchema
    capabilities: tuple[str, ...]
    side_effect: ToolSideEffect
    approval: ToolApproval
    required_env_vars: tuple[str, ...]
    required_slack_scopes: tuple[str, ...]
    plan_gates: tuple[str, ...]
    result_budget: str
    notes: tuple[str, ...]
    can_replace_native_tools: tuple[str, ...]
    enabled: bool
    disabled_reason: str | None
    required_args: tuple[str, ...]
    optional_args: tuple[str, ...]

    def to_payload(self) -> JsonObject:
        """Return a JSON-safe descriptor for tools and dashboard surfaces."""

        return {
            "name": self.name,
            "namespace": self.namespace,
            "category": self.category,
            "display_name": self.display_name,
            "description": self.description,
            "parameters": self.parameters,
            "capabilities": list(self.capabilities),
            "side_effect": self.side_effect,
            "approval": self.approval,
            "required_env_vars": list(self.required_env_vars),
            "required_slack_scopes": list(self.required_slack_scopes),
            "plan_gates": list(self.plan_gates),
            "result_budget": self.result_budget,
            "notes": list(self.notes),
            "can_replace_native_tools": list(self.can_replace_native_tools),
            "enabled": self.enabled,
            "disabled_reason": self.disabled_reason,
            "required_args": list(self.required_args),
            "optional_args": list(self.optional_args),
        }


NATIVE_TOOL_METADATA: dict[str, ToolMetadata] = {
    "web_search": ToolMetadata(
        name="web_search",
        namespace="native.research",
        category="Research",
        display_name="Web search",
        capabilities=("web_search", "current_research"),
        side_effect="read",
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
        plan_gates=("scope_guarded_context",),
        result_budget="bounded_results",
        notes=("Searches Kortny's local observed Slack cache without Slack API calls.",),
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
        required_env_vars=("POSTGRES_URL",),
        plan_gates=("scope_guarded_context",),
        result_budget="small_lookup",
        notes=("Resolves cached Slack user and channel names without Slack API calls.",),
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
        result_budget="artifact",
        notes=("Creates local task artifacts only when explicitly useful.",),
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
    resolved_enabled = enabled if enabled is not None else dynamic_disabled_reason is None
    return ToolDescriptor(
        name=metadata.name,
        namespace=metadata.namespace,
        category=metadata.category,
        display_name=metadata.display_name,
        description=description,
        parameters=parameters,
        capabilities=metadata.capabilities,
        side_effect=metadata.side_effect,
        approval=metadata.approval,
        required_env_vars=metadata.required_env_vars,
        required_slack_scopes=metadata.required_slack_scopes,
        plan_gates=metadata.plan_gates,
        result_budget=metadata.result_budget,
        notes=metadata.notes,
        can_replace_native_tools=metadata.can_replace_native_tools,
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


_SETTINGS_ENV_ATTRS = {
    "BRAVE_SEARCH_API_KEY": "brave_search_api_key",
    "COMPOSIO_API_KEY": "composio_api_key",
    "POSTGRES_URL": "postgres_url",
    "SLACK_APP_TOKEN": "slack_app_token",
    "SLACK_BOT_TOKEN": "slack_bot_token",
    "SLACK_SIGNING_SECRET": "slack_signing_secret",
}
