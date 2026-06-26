"""Read models for the operator dashboard."""

from __future__ import annotations

import json
import math
import uuid
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Protocol
from urllib.parse import urlencode, urlsplit, urlunsplit

from sqlalchemy import Select, Text, case, cast, exists, func, or_, select
from sqlalchemy.engine import Row
from sqlalchemy.orm import Session, aliased
from sqlalchemy.sql.elements import ColumnElement

from kortny.composio import (
    ComposioAuthConfig,
    ComposioCatalogError,
    ComposioClient,
    ComposioConnectionError,
    ComposioTool,
    ComposioToolkit,
)
from kortny.config import Settings
from kortny.dashboard.settings import DashboardSettings
from kortny.db.models import (
    Artifact,
    ComposioConnection,
    ConsolidationRun,
    Episode,
    Installation,
    KnowledgeGraphEdge,
    KnowledgeGraphEntity,
    KnowledgeGraphEvidence,
    LLMConfigAudit,
    LLMModelCatalog,
    LLMModelPricing,
    LLMProviderAccount,
    LLMTierAssignment,
    LLMUsage,
    ModelPricing,
    ObserveChannelProfile,
    ObservePolicy,
    ProceduralSkillInvocation,
    ProceduralSkillVersion,
    SlackChannelMembership,
    SlackIdentity,
    Task,
    TaskEvent,
    TaskEventType,
    TaskStatus,
    WitnessDeliveryLog,
    WitnessOpportunityCandidate,
    WorkspaceState,
)
from kortny.knowledge_graph.provenance import (
    provenance_kind,
    provenance_label,
    review_status,
)
from kortny.llm.litellm_catalog import (
    LiteLLMProviderOption,
    litellm_provider_options,
)
from kortny.llm.provider_config import CONFIG_TIERS
from kortny.observe.style_cards import (
    STYLE_CARD_UPDATED_AT_KEY,
    pinned_style_from_profile,
    style_card_from_profile,
)
from kortny.tools.catalog import ToolDescriptor, tool_descriptor_from_class
from kortny.tools.native_runtime import native_dashboard_tool_classes

_TOKENS_PER_MTOK = Decimal("1000000")
DEFAULT_PAGE_SIZE = 25
MAX_PAGE_SIZE = 100
MODEL_CATALOG_PAGE_SIZE = 25
COMPOSIO_DETAIL_TOOL_LIMIT = 12
KG_GRAPH_NODE_LIMIT = 18
KG_GRAPH_EDGE_LIMIT = 30
KG_GRAPH_WIDTH = 920
KG_GRAPH_HEIGHT = 420
PLANNED_BRANCH_AGENT_TO_KEY = {
    "planned_research_worker": "research",
    "planned_workspace_worker": "workspace",
    "planned_integration_worker": "integration",
}
PLANNED_BRANCH_LABELS = {
    "research": "Research",
    "workspace": "Workspace",
    "integration": "Integrations",
}
PLANNED_AGENT_NAMES = frozenset(
    {
        "planned_workflow_planner",
        "planned_workflow_merger",
        *PLANNED_BRANCH_AGENT_TO_KEY.keys(),
    }
)
PLANNED_TRACE_MESSAGES = frozenset(
    {
        "adk_planned_workflow_selected",
        "planned_task_started",
        "planned_task_planning_started",
        "planned_task_plan_ready",
        "planned_task_branch_started",
        "planned_task_branch_completed",
        "planned_task_budget_reached",
        "planned_task_merging",
        "planned_task_completed",
        "planned_task_progress_posted",
        "planned_workflow_cost_ceiling_exceeded",
        "final_response_sanitized",
    }
)
_NATIVE_DASHBOARD_TOOL_CLASSES: tuple[type[Any], ...] = native_dashboard_tool_classes()


@dataclass(frozen=True)
class TaskListItem:
    task: Task
    channel: IdentityLabel
    user: IdentityLabel
    models: tuple[str, ...]
    turn_count: int


@dataclass(frozen=True)
class TaskListPage:
    items: tuple[TaskListItem, ...]
    page: int
    page_size: int
    total_count: int

    @property
    def total_pages(self) -> int:
        if self.total_count == 0:
            return 1
        return math.ceil(self.total_count / self.page_size)

    @property
    def previous_page(self) -> int | None:
        if self.page <= 1:
            return None
        return self.page - 1

    @property
    def next_page(self) -> int | None:
        if self.page >= self.total_pages:
            return None
        return self.page + 1

    @property
    def first_item(self) -> int:
        if self.total_count == 0:
            return 0
        return ((self.page - 1) * self.page_size) + 1

    @property
    def last_item(self) -> int:
        return min(self.page * self.page_size, self.total_count)


@dataclass(frozen=True)
class TaskDetail:
    task: Task
    channel: IdentityLabel
    user: IdentityLabel
    events: tuple[TaskEvent, ...]
    timeline: tuple[TimelineEvent, ...]
    usage: tuple[LLMUsage, ...]
    artifacts: tuple[Artifact, ...]
    posted_response_text: str | None
    planned_trace: PlannedWorkflowTrace


@dataclass(frozen=True)
class IdentityLabel:
    name: str
    slack_id: str
    found: bool

    @property
    def secondary(self) -> str | None:
        if self.found and self.name != self.slack_id:
            return self.slack_id
        return None


@dataclass(frozen=True)
class TimelineBadge:
    label: str
    tone: str = "neutral"


@dataclass(frozen=True)
class TimelineMetric:
    label: str
    value: str


@dataclass(frozen=True)
class TimelineEvent:
    seq: int
    event_type: str
    tone: str
    title: str
    summary: str
    created_at: datetime
    badges: tuple[TimelineBadge, ...]
    metrics: tuple[TimelineMetric, ...]
    payload_json: str
    prompt_json: str | None = None
    response_json: str | None = None
    input_json: str | None = None
    output_json: str | None = None
    # Kortny's timeline is a ~100-row append-only log, not a clean span tree.
    # ``tier`` separates the handful of real model/tool *spans* (rendered
    # prominently with durations + one-click I/O) from the bulk of internal
    # *dim* log lines (thin, muted, non-expandable). See ``_event_tier``.
    tier: str = "dim"
    # For span rows: a short kind label ("LLM" / "Tool") plus the concrete
    # name (model id / tool name) shown as the title. Empty for dim rows.
    span_label: str = ""
    span_name: str = ""
    # At-a-glance metrics shown only when present; a dim/log row renders none.
    duration_ms: int | None = None
    tokens: int | None = None
    cost_usd: str | None = None
    turn: int | None = None
    # Whether expanding the row reveals anything useful (tool I/O, prompt/
    # response, or a raw payload worth inspecting). Rows with nothing to show
    # are not expandable (no disclosure caret).
    has_detail: bool = False


@dataclass(frozen=True)
class PlannedToolRollup:
    tool: str
    call_count: int
    result_count: int
    artifact_count: int
    cost_usd: Decimal
    branch_labels: tuple[str, ...]


@dataclass(frozen=True)
class PlannedBranchTrace:
    key: str
    label: str
    status: str
    tone: str
    started: bool
    completed: bool
    llm_call_count: int
    tool_call_count: int
    tool_result_count: int
    input_tokens: int
    output_tokens: int
    cost_usd: Decimal
    budget_hit_count: int
    tool_names: tuple[str, ...]
    model_names: tuple[str, ...]


@dataclass(frozen=True)
class PlannedWorkflowTrace:
    present: bool
    mode: str | None
    route: str | None
    planner_agent: str | None
    merger_agent: str | None
    max_parallel_branches: int | None
    max_branch_model_calls: int | None
    max_branch_tool_calls: int | None
    max_total_tool_calls: int | None
    cost_ceiling_usd: Decimal | None
    branch_count: int
    completed_branch_count: int
    budget_hit_count: int
    llm_call_count: int
    tool_call_count: int
    tool_result_count: int
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: Decimal
    final_sanitized: bool
    raw_chars: int | None
    posted_chars: int | None
    branches: tuple[PlannedBranchTrace, ...]
    tool_rollups: tuple[PlannedToolRollup, ...]
    budget_events: tuple[TimelineEvent, ...]
    phase_events: tuple[TimelineEvent, ...]


@dataclass(frozen=True)
class AggregateRow:
    key: str
    calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: Decimal
    label: IdentityLabel | None = None

    @property
    def display_key(self) -> str:
        if self.label is not None:
            return self.label.name
        return self.key

    @property
    def secondary_key(self) -> str | None:
        if self.label is not None:
            return self.label.secondary
        return None


@dataclass(frozen=True)
class DailyUsageRow:
    day: date
    calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: Decimal


@dataclass(frozen=True)
class DailyTaskRow:
    day: date
    task_count: int
    failed_task_count: int


@dataclass(frozen=True)
class ChartBar:
    label: str
    secondary: str | None
    value_label: str
    percent: int


@dataclass(frozen=True)
class ChartPoint:
    label: str
    value_label: str
    percent: int
    tone: str = "accent"
    detail: str | None = None


@dataclass(frozen=True)
class UsageCharts:
    daily_cost: tuple[ChartPoint, ...]
    daily_task_volume: tuple[ChartPoint, ...]
    cost_by_model: tuple[ChartBar, ...]
    cost_by_user: tuple[ChartBar, ...]


@dataclass(frozen=True)
class CacheStats:
    """Prompt-cache rollup for the usage view (HIG-196)."""

    total_input_tokens: int
    cache_read_input_tokens: int
    cache_creation_input_tokens: int
    estimated_savings_usd: Decimal

    @property
    def hit_rate(self) -> float:
        if self.total_input_tokens <= 0:
            return 0.0
        return self.cache_read_input_tokens / self.total_input_tokens

    @property
    def hit_rate_label(self) -> str:
        return f"{self.hit_rate * 100:.1f}%"

    @property
    def estimated_savings_label(self) -> str:
        return f"${self.estimated_savings_usd:.2f}"

    @property
    def has_activity(self) -> bool:
        return self.cache_read_input_tokens > 0 or self.cache_creation_input_tokens > 0


@dataclass(frozen=True)
class UsageAggregate:
    start: datetime | None
    end: datetime | None
    by_model: tuple[AggregateRow, ...]
    by_user: tuple[AggregateRow, ...]
    by_channel: tuple[AggregateRow, ...]
    by_day: tuple[DailyUsageRow, ...]
    by_task_day: tuple[DailyTaskRow, ...]
    cache: CacheStats = field(default_factory=lambda: CacheStats(0, 0, 0, Decimal("0")))

    @property
    def total_calls(self) -> int:
        return sum(row.calls for row in self.by_day)

    @property
    def total_input_tokens(self) -> int:
        return sum(row.input_tokens for row in self.by_day)

    @property
    def total_output_tokens(self) -> int:
        return sum(row.output_tokens for row in self.by_day)

    @property
    def total_cost_usd(self) -> Decimal:
        return sum((row.cost_usd for row in self.by_day), Decimal("0"))

    @property
    def total_tasks(self) -> int:
        return sum(row.task_count for row in self.by_task_day)

    @property
    def failed_tasks(self) -> int:
        return sum(row.failed_task_count for row in self.by_task_day)

    @property
    def task_failure_rate_label(self) -> str:
        if self.total_tasks == 0:
            return "0.0%"
        return f"{(self.failed_tasks / self.total_tasks) * 100:.1f}%"

    @property
    def charts(self) -> UsageCharts:
        return UsageCharts(
            daily_cost=_daily_cost_points(self.by_day),
            daily_task_volume=_daily_task_points(self.by_task_day),
            cost_by_model=_aggregate_bars(self.by_model),
            cost_by_user=_aggregate_bars(self.by_user),
        )


@dataclass(frozen=True)
class UserListItem:
    user: IdentityLabel
    task_count: int
    failed_task_count: int
    artifact_count: int
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: Decimal
    last_activity_at: datetime | None


@dataclass(frozen=True)
class UserDirectory:
    start: datetime | None
    end: datetime | None
    users: tuple[UserListItem, ...]

    @property
    def total_tasks(self) -> int:
        return sum(u.task_count for u in self.users)

    @property
    def total_failures(self) -> int:
        return sum(u.failed_task_count for u in self.users)

    @property
    def total_cost_usd(self) -> Decimal:
        return sum((u.total_cost_usd for u in self.users), Decimal(0))

    @property
    def failure_rate(self) -> float:
        return (self.total_failures / self.total_tasks) if self.total_tasks else 0.0


@dataclass(frozen=True)
class UserTaskRow:
    task: Task
    channel: IdentityLabel
    usage_count: int
    artifact_count: int


@dataclass(frozen=True)
class UserArtifactRow:
    artifact: Artifact
    task: Task


@dataclass(frozen=True)
class UserDetail:
    user: IdentityLabel
    start: datetime | None
    end: datetime | None
    task_count: int
    failed_task_count: int
    artifact_count: int
    usage_call_count: int
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: Decimal
    last_activity_at: datetime | None
    tasks: tuple[UserTaskRow, ...]
    usage: tuple[LLMUsage, ...]
    artifacts: tuple[UserArtifactRow, ...]


@dataclass(frozen=True)
class SystemMetric:
    label: str
    value: str
    detail: str | None = None
    tone: str = "neutral"


@dataclass(frozen=True)
class SystemCheck:
    group: str
    name: str
    status: str
    tone: str
    detail: str
    action: str | None = None


@dataclass(frozen=True)
class SystemConfigRow:
    name: str
    value: str
    detail: str | None = None
    tone: str = "neutral"


@dataclass(frozen=True)
class SystemConfigSection:
    title: str
    rows: tuple[SystemConfigRow, ...]


@dataclass(frozen=True)
class SystemHealth:
    overall_label: str
    overall_tone: str
    metrics: tuple[SystemMetric, ...]
    checks: tuple[SystemCheck, ...]
    config_sections: tuple[SystemConfigSection, ...]


@dataclass(frozen=True)
class OverviewAttentionItem:
    title: str
    detail: str
    tone: str
    badge: str
    href: str


@dataclass(frozen=True)
class DashboardOverview:
    metrics: tuple[SystemMetric, ...]
    attention_items: tuple[OverviewAttentionItem, ...]
    charts: UsageCharts
    top_models: tuple[AggregateRow, ...]
    top_users: tuple[AggregateRow, ...]
    top_channels: tuple[AggregateRow, ...]
    recent_tasks: tuple[TaskListItem, ...]
    system_health: SystemHealth
    window_label: str
    refreshed_at: datetime
    active_facts: tuple[MemoryFactRow, ...]
    skill_usage: tuple[OverviewSkillUsageRow, ...]
    channel_profiles: tuple[OverviewChannelProfileRow, ...]
    by_day: tuple[DailyUsageRow, ...]
    by_task_day: tuple[DailyTaskRow, ...]


@dataclass(frozen=True)
class OverviewChannelProfileRow:
    channel: IdentityLabel
    summary: str | None
    message_count: int
    updated_at: datetime


@dataclass(frozen=True)
class OverviewSkillUsageRow:
    name: str
    description: str | None
    invocation_count: int


@dataclass(frozen=True)
class MemoryFactRow:
    fact: WorkspaceState
    scope: IdentityLabel
    value_summary: str
    confirmed_by: IdentityLabel | None
    proposed_by: IdentityLabel | None
    rejected_by: IdentityLabel | None
    forgotten_by: IdentityLabel | None
    source_task: Task | None
    tone: str


@dataclass(frozen=True)
class MemoryEpisodeRow:
    episode: Episode
    channel: IdentityLabel
    user: IdentityLabel
    task: Task | None
    tools_label: str
    artifacts_label: str
    source_refs_label: str
    tone: str


@dataclass(frozen=True)
class MemoryPageInfo:
    page: int
    page_size: int
    total_count: int
    noun: str

    @property
    def total_pages(self) -> int:
        if self.total_count == 0:
            return 1
        return math.ceil(self.total_count / self.page_size)

    @property
    def previous_page(self) -> int | None:
        if self.page <= 1:
            return None
        return self.page - 1

    @property
    def next_page(self) -> int | None:
        if self.page >= self.total_pages:
            return None
        return self.page + 1

    @property
    def first_item(self) -> int:
        if self.total_count == 0:
            return 0
        return ((self.page - 1) * self.page_size) + 1

    @property
    def last_item(self) -> int:
        return min(self.page * self.page_size, self.total_count)


@dataclass(frozen=True)
class MemoryDashboard:
    active_fact_count: int
    proposed_fact_count: int
    episode_count: int
    failed_episode_count: int
    active_view: str
    query: str
    scope_filter: str
    status_filter: str
    outcome_filter: str
    sort: str
    page: MemoryPageInfo
    previous_page_url: str | None
    next_page_url: str | None
    reset_url: str
    facts: tuple[MemoryFactRow, ...]
    episodes: tuple[MemoryEpisodeRow, ...]


@dataclass(frozen=True)
class KnowledgeGraphEvidencePreview:
    evidence: KnowledgeGraphEvidence
    source_label: str
    snippet: str
    confidence_label: str


@dataclass(frozen=True)
class KnowledgeGraphEntityRow:
    entity: KnowledgeGraphEntity
    scope: IdentityLabel
    evidence_count: int
    source_edge_count: int
    target_edge_count: int
    evidence: tuple[KnowledgeGraphEvidencePreview, ...]
    tone: str
    confidence_label: str
    provenance_label: str
    provenance_tone: str
    review_status: str
    attrs_preview: str


@dataclass(frozen=True)
class KnowledgeGraphEdgeRow:
    edge: KnowledgeGraphEdge
    source_label: str
    target_label: str
    scope: IdentityLabel
    evidence_count: int
    evidence: tuple[KnowledgeGraphEvidencePreview, ...]
    tone: str
    confidence_label: str
    provenance_label: str
    provenance_tone: str
    review_status: str
    attrs_preview: str


@dataclass(frozen=True)
class KnowledgeGraphMapNode:
    id: str
    label: str
    secondary_label: str
    entity_type: str
    lifecycle_state: str
    tone: str
    x: int
    y: int
    radius: int
    evidence_count: int
    incoming_count: int
    outgoing_count: int
    confidence_label: str
    provenance_label: str
    provenance_tone: str
    review_status: str
    scope_label: str


@dataclass(frozen=True)
class KnowledgeGraphMapEdge:
    id: str
    source_id: str
    target_id: str
    label: str
    relationship_type: str
    tone: str
    x1: int
    y1: int
    x2: int
    y2: int
    label_x: int
    label_y: int


@dataclass(frozen=True)
class KnowledgeGraphMap:
    nodes: tuple[KnowledgeGraphMapNode, ...]
    edges: tuple[KnowledgeGraphMapEdge, ...]
    empty: bool
    node_count: int
    edge_count: int


@dataclass(frozen=True)
class KnowledgeGraphPageInfo:
    page: int
    page_size: int
    total_count: int
    noun: str

    @property
    def total_pages(self) -> int:
        if self.total_count == 0:
            return 1
        return math.ceil(self.total_count / self.page_size)

    @property
    def previous_page(self) -> int | None:
        if self.page <= 1:
            return None
        return self.page - 1

    @property
    def next_page(self) -> int | None:
        if self.page >= self.total_pages:
            return None
        return self.page + 1

    @property
    def first_item(self) -> int:
        if self.total_count == 0:
            return 0
        return ((self.page - 1) * self.page_size) + 1

    @property
    def last_item(self) -> int:
        return min(self.page * self.page_size, self.total_count)


@dataclass(frozen=True)
class KnowledgeGraphDashboard:
    entity_count: int
    current_entity_count: int
    candidate_entity_count: int
    known_channel_count: int
    profiled_channel_count: int
    edge_count: int
    evidence_count: int
    stale_entity_count: int
    stale_edge_count: int
    unbacked_current_entity_count: int
    unbacked_current_edge_count: int
    runtime_eligible_entity_count: int
    runtime_eligible_edge_count: int
    active_view: str
    query: str
    scope_filter: str
    state_filter: str
    kind_filter: str
    sort: str
    map: KnowledgeGraphMap
    page: KnowledgeGraphPageInfo
    previous_page_url: str | None
    next_page_url: str | None
    reset_url: str
    entities_url: str
    relationships_url: str
    entities: tuple[KnowledgeGraphEntityRow, ...]
    edges: tuple[KnowledgeGraphEdgeRow, ...]


@dataclass(frozen=True)
class WitnessEvidencePreview:
    source_label: str
    snippet: str


@dataclass(frozen=True)
class WitnessCandidateRow:
    candidate: WitnessOpportunityCandidate
    channel: IdentityLabel | None
    target_user: IdentityLabel | None
    scope: IdentityLabel
    source_task: Task | None
    source_profile: ObserveChannelProfile | None
    evidence: tuple[WitnessEvidencePreview, ...]
    tone: str
    type_label: str
    status_label: str
    source_label: str
    confidence_label: str
    cooldown_label: str | None
    can_send_private: bool
    can_snooze: bool
    can_dismiss: bool
    can_accept: bool
    can_reactivate: bool
    can_archive: bool


@dataclass(frozen=True)
class WitnessCandidatePageInfo:
    page: int
    page_size: int
    total_count: int

    @property
    def total_pages(self) -> int:
        if self.total_count == 0:
            return 1
        return math.ceil(self.total_count / self.page_size)

    @property
    def previous_page(self) -> int | None:
        if self.page <= 1:
            return None
        return self.page - 1

    @property
    def next_page(self) -> int | None:
        if self.page >= self.total_pages:
            return None
        return self.page + 1

    @property
    def first_item(self) -> int:
        if self.total_count == 0:
            return 0
        return ((self.page - 1) * self.page_size) + 1

    @property
    def last_item(self) -> int:
        return min(self.page * self.page_size, self.total_count)


@dataclass(frozen=True)
class WitnessCandidatesDashboard:
    total_count: int
    candidate_count: int
    due_candidate_count: int
    sent_count: int
    accepted_count: int
    automated_count: int
    inactive_count: int
    query: str
    status_filter: str
    type_filter: str
    scope_filter: str
    sort: str
    page: WitnessCandidatePageInfo
    previous_page_url: str | None
    next_page_url: str | None
    reset_url: str
    rows: tuple[WitnessCandidateRow, ...]


@dataclass(frozen=True)
class WitnessDecisionCount:
    """One row of the delivery-decision breakdown table."""

    decision: str
    label: str
    count: int


@dataclass(frozen=True)
class WitnessKpis:
    """Proactivity quality KPIs (HIG-227), computed over a rolling window."""

    window_days: int
    candidates_created: int
    trigger_rate_per_day: float
    delivered_count: int
    silent_count: int
    silent_rate: float | None
    acceptance_rate: float | None
    dismissal_rate: float | None
    time_to_action_median_hours: float | None
    conversion_to_automation: float | None
    decision_counts: tuple[WitnessDecisionCount, ...]

    @property
    def trigger_rate_label(self) -> str:
        return f"{self.trigger_rate_per_day:.1f}/day"

    @property
    def silent_rate_label(self) -> str:
        return _rate_label(self.silent_rate)

    @property
    def acceptance_rate_label(self) -> str:
        return _rate_label(self.acceptance_rate)

    @property
    def dismissal_rate_label(self) -> str:
        return _rate_label(self.dismissal_rate)

    @property
    def conversion_label(self) -> str:
        return _rate_label(self.conversion_to_automation)

    @property
    def time_to_action_label(self) -> str:
        if self.time_to_action_median_hours is None:
            return "-"
        return f"{self.time_to_action_median_hours:.1f}h"


def _rate_label(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.1f}%"


@dataclass(frozen=True)
class WitnessChannelRow:
    """Per-channel delivery status for the Witness activation dashboard."""

    channel_id: str
    channel_name: str
    proactivity_status: str
    full_enabled_at: datetime | None
    queued_candidate_count: int
    last_deferral_reason: str | None
    channel_posts_this_week: int
    channel_posts_per_week_budget: int
    last_sent_at: datetime | None


@dataclass(frozen=True)
class WitnessDormancyStatus:
    """Activation state for the Witness dormancy banner."""

    total_queued: int
    dm_digest_enabled: bool
    dm_digest_epoch: datetime | None
    channels_full: int
    channels_total: int
    total_deferred: int
    total_sent: int
    autopilot_db_override: bool | None  # None = using env default
    channel_rows: tuple[WitnessChannelRow, ...]


def get_witness_dormancy_status(
    session: Session,
    *,
    installation_id: uuid.UUID | None = None,
    channel_posts_per_week_budget: int = 1,
    now: datetime | None = None,
) -> WitnessDormancyStatus:
    """Compute activation state for Witness dormancy banner and per-channel table."""
    observed_now = now or datetime.now(UTC)
    one_week_ago = observed_now - timedelta(days=7)

    candidate_scope = _kg_installation_filter(
        WitnessOpportunityCandidate, installation_id
    )
    log_scope = _kg_installation_filter(WitnessDeliveryLog, installation_id)
    policy_scope = (
        [ObservePolicy.installation_id == installation_id]
        if installation_id is not None
        else []
    )

    # Total queued candidates
    total_queued = int(
        session.scalar(
            select(func.count())
            .select_from(WitnessOpportunityCandidate)
            .where(*candidate_scope, WitnessOpportunityCandidate.status == "candidate")
        )
        or 0
    )

    # DM digest state from installation
    dm_digest_enabled = False
    dm_digest_epoch: datetime | None = None
    autopilot_db_override: bool | None = None
    if installation_id is not None:
        install = session.get(Installation, installation_id)
        if install is not None:
            dm_digest_epoch = install.digest_enabled_at
            dm_digest_enabled = dm_digest_epoch is not None
            autopilot_db_override = install.autopilot_enabled

    # Channel policies with proactivity status
    channel_policies = (
        session.execute(
            select(ObservePolicy).where(
                *policy_scope, ObservePolicy.scope_type == "channel"
            )
        )
        .scalars()
        .all()
    )

    channels_full = sum(1 for p in channel_policies if p.proactivity_status == "full")
    channels_total = len(channel_policies)

    # Total deferred from delivery log
    total_deferred = int(
        session.scalar(
            select(func.count())
            .select_from(WitnessDeliveryLog)
            .where(*log_scope, WitnessDeliveryLog.decision == "channel_deferred")
        )
        or 0
    )

    # Total sent from delivery log
    total_sent = int(
        session.scalar(
            select(func.count())
            .select_from(WitnessDeliveryLog)
            .where(
                *log_scope,
                WitnessDeliveryLog.decision.in_(
                    ("channel_sent", "notify", "question", "draft", "digest")
                ),
                WitnessDeliveryLog.reason == "sent",
            )
        )
        or 0
    )

    # Per-channel rows
    channel_rows: list[WitnessChannelRow] = []
    for policy in channel_policies:
        channel_id = policy.scope_id
        if channel_id is None:
            continue
        ch_scope = _kg_installation_filter(WitnessOpportunityCandidate, installation_id)
        queued = int(
            session.scalar(
                select(func.count())
                .select_from(WitnessOpportunityCandidate)
                .where(
                    *ch_scope,
                    WitnessOpportunityCandidate.status == "candidate",
                    WitnessOpportunityCandidate.channel_id == channel_id,
                )
            )
            or 0
        )
        log_user = f"channel:{channel_id}"
        log_installation_filter = (
            [WitnessDeliveryLog.installation_id == policy.installation_id]
            if installation_id is not None
            else []
        )
        last_deferral = session.scalar(
            select(WitnessDeliveryLog.reason)
            .where(
                *log_installation_filter,
                WitnessDeliveryLog.slack_user_id == log_user,
            )
            .order_by(WitnessDeliveryLog.created_at.desc())
            .limit(1)
        )
        posts_this_week = int(
            session.scalar(
                select(func.count())
                .select_from(WitnessDeliveryLog)
                .where(
                    *log_installation_filter,
                    WitnessDeliveryLog.slack_user_id == log_user,
                    WitnessDeliveryLog.decision.in_(
                        ("channel_sent", "ambient_file_brief")
                    ),
                    WitnessDeliveryLog.created_at > one_week_ago,
                )
            )
            or 0
        )
        last_sent = session.scalar(
            select(func.max(WitnessDeliveryLog.created_at)).where(
                *log_installation_filter,
                WitnessDeliveryLog.slack_user_id == log_user,
                WitnessDeliveryLog.decision == "channel_sent",
                WitnessDeliveryLog.reason == "sent",
            )
        )
        channel_rows.append(
            WitnessChannelRow(
                channel_id=channel_id,
                channel_name=channel_id,
                proactivity_status=policy.proactivity_status,
                full_enabled_at=policy.full_enabled_at,
                queued_candidate_count=queued,
                last_deferral_reason=last_deferral,
                channel_posts_this_week=posts_this_week,
                channel_posts_per_week_budget=channel_posts_per_week_budget,
                last_sent_at=last_sent,
            )
        )

    return WitnessDormancyStatus(
        total_queued=total_queued,
        dm_digest_enabled=dm_digest_enabled,
        dm_digest_epoch=dm_digest_epoch,
        channels_full=channels_full,
        channels_total=channels_total,
        total_deferred=total_deferred,
        total_sent=total_sent,
        autopilot_db_override=autopilot_db_override,
        channel_rows=tuple(channel_rows),
    )


@dataclass(frozen=True)
class IntegrationCard:
    name: str
    category: str
    status: str
    tone: str
    description: str
    details: tuple[str, ...]
    env_vars: tuple[str, ...]
    action: str | None = None


@dataclass(frozen=True)
class ComposioToolkitRow:
    slug: str
    name: str
    description: str
    logo_url: str | None
    categories: tuple[str, ...]
    auth_schemes: tuple[str, ...]
    managed_auth_schemes: tuple[str, ...]
    tools_count: int
    triggers_count: int
    no_auth: bool
    connection_status: str
    connection_tone: str
    connected: bool


@dataclass(frozen=True)
class ComposioConnectionRow:
    id: uuid.UUID
    toolkit_slug: str
    status: str
    tone: str
    display_name: str
    scope_label: str
    visibility_scope_type: str
    visibility_scope_id: str | None
    owner: IdentityLabel
    connected_account_id: str | None
    auth_config_id: str | None
    updated_at: datetime


@dataclass(frozen=True)
class ComposioAuthConfigRow:
    id: str
    name: str
    toolkit_slug: str
    auth_scheme: str | None
    is_composio_managed: bool
    enabled: bool

    @property
    def managed_label(self) -> str:
        return "Composio managed" if self.is_composio_managed else "Custom"

    @property
    def status_label(self) -> str:
        return "Enabled" if self.enabled else "Disabled"

    @property
    def tone(self) -> str:
        return "success" if self.enabled else "neutral"


@dataclass(frozen=True)
class ComposioToolRow:
    slug: str
    name: str
    description: str
    tags: tuple[str, ...]
    version: str | None


@dataclass(frozen=True)
class ComposioCatalogView:
    enabled: bool
    configured: bool
    status: str
    tone: str
    query: str
    cursor: str
    page_size: int
    total_items: int | None
    visible_count: int
    next_cursor: str | None
    connection_count: int
    active_connection_count: int
    connected_toolkit_count: int
    pinned_connected_count: int
    error: str | None
    toolkits: tuple[ComposioToolkitRow, ...]
    connections: tuple[ComposioConnectionRow, ...]


@dataclass(frozen=True)
class ComposioScopeOption:
    name: str
    key: str
    description: str
    default: bool = False
    risk: str | None = None


@dataclass(frozen=True)
class ComposioToolkitDetail:
    slug: str
    configured: bool
    status: str
    tone: str
    toolkit: ComposioToolkitRow | None
    raw_toolkit: ComposioToolkit | None
    auth_configs: tuple[ComposioAuthConfigRow, ...]
    tools: tuple[ComposioToolRow, ...]
    tools_error: str | None
    connections: tuple[ComposioConnectionRow, ...]
    scope_options: tuple[ComposioScopeOption, ...]
    user_options: tuple[IdentityLabel, ...]
    channel_options: tuple[IdentityLabel, ...]
    error: str | None

    @property
    def active_connection(self) -> ComposioConnectionRow | None:
        return next(
            (
                connection
                for connection in self.connections
                if connection.status == "active"
            ),
            None,
        )


@dataclass(frozen=True)
class ToolCapability:
    name: str
    group: str
    status: str
    tone: str
    description: str
    required_args: tuple[str, ...]
    optional_args: tuple[str, ...]
    notes: tuple[str, ...]


@dataclass(frozen=True)
class ToolCapabilityGroup:
    name: str
    description: str
    tools: tuple[ToolCapability, ...]


@dataclass(frozen=True)
class IntegrationDashboard:
    metrics: tuple[SystemMetric, ...]
    integrations: tuple[IntegrationCard, ...]
    composio_catalog: ComposioCatalogView
    tool_groups: tuple[ToolCapabilityGroup, ...]
    runtime_error: str | None


@dataclass(frozen=True)
class LLMProviderConfigRow:
    provider: LLMProviderAccount
    model_count: int
    enabled_model_count: int
    tier_count: int
    credential_label: str
    source_label: str
    status_tone: str
    health_tone: str


@dataclass(frozen=True)
class LLMModelConfigOption:
    id: uuid.UUID
    label: str
    detail: str
    enabled: bool


@dataclass(frozen=True)
class LLMTierCatalogOption:
    tier: str
    label: str
    description: str


@dataclass(frozen=True)
class LLMTierConfigRow:
    tier: str
    label: str
    description: str
    primary_assignment: LLMTierAssignment | None
    primary_model: LLMModelCatalog | None
    primary_provider: LLMProviderAccount | None
    fallback_assignments: tuple[
        tuple[LLMTierAssignment, LLMModelCatalog, LLMProviderAccount], ...
    ]
    options: tuple[LLMModelConfigOption, ...]
    tone: str


@dataclass(frozen=True)
class LLMModelConfigRow:
    model: LLMModelCatalog
    provider: LLMProviderAccount
    assignment_labels: tuple[str, ...]
    latest_pricing: LLMModelPricing | None
    tone: str
    source_label: str
    pricing_label: str
    pricing_detail: str
    context_label: str
    output_label: str
    mode_label: str
    capability_labels: tuple[str, ...]


@dataclass(frozen=True)
class LLMAuditConfigRow:
    audit: LLMConfigAudit
    label: str
    detail: str
    tone: str


@dataclass(frozen=True)
class LLMModelConfigDashboard:
    installation_id: uuid.UUID | None
    installation_label: str
    metrics: tuple[SystemMetric, ...]
    provider_options: tuple[LiteLLMProviderOption, ...]
    providers: tuple[LLMProviderConfigRow, ...]
    tiers: tuple[LLMTierConfigRow, ...]
    models: tuple[LLMModelConfigRow, ...]
    audits: tuple[LLMAuditConfigRow, ...]
    runtime_error: str | None
    can_bootstrap: bool
    empty_message: str | None


@dataclass(frozen=True)
class LLMProviderConfigDetail:
    installation_id: uuid.UUID
    installation_label: str
    provider: LLMProviderAccount
    provider_row: LLMProviderConfigRow
    models: tuple[LLMModelConfigRow, ...]
    routed_models: tuple[LLMModelConfigRow, ...]
    attention_models: tuple[LLMModelConfigRow, ...]
    missing_pricing_count: int
    model_total_count: int
    model_page_size: int
    model_next_offset: int | None
    model_has_more: bool
    audits: tuple[LLMAuditConfigRow, ...]
    metrics: tuple[SystemMetric, ...]
    api_version_label: str
    base_url_label: str
    tier_options: tuple[LLMTierCatalogOption, ...]


@dataclass(frozen=True)
class LLMProviderModelCatalogPage:
    rows: tuple[LLMModelConfigRow, ...]
    total_count: int
    offset: int
    limit: int
    next_offset: int | None
    has_more: bool


def list_tasks(
    session: Session,
    *,
    page: int = 1,
    page_size: int = DEFAULT_PAGE_SIZE,
    installation_id: uuid.UUID | None = None,
    slack_user_id: str | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    query: str | None = None,
    status: str | None = None,
    channel: str | None = None,
    user: str | None = None,
    model: str | None = None,
) -> TaskListPage:
    """Return a paginated dashboard task list."""

    normalized_page = max(page, 1)
    normalized_size = min(max(page_size, 1), MAX_PAGE_SIZE)
    offset = (normalized_page - 1) * normalized_size
    task_filter = [
        *_task_scope_filter(
            installation_id=installation_id,
            slack_user_id=slack_user_id,
        ),
        *_task_filter(start=start, end=end),
        *_task_list_filters(
            query=query,
            status=status,
            channel=channel,
            user=user,
            model=model,
        ),
    ]

    total_count = (
        session.scalar(select(func.count()).select_from(Task).where(*task_filter)) or 0
    )
    tasks = tuple(
        session.scalars(
            select(Task)
            .where(*task_filter)
            .order_by(Task.created_at.desc(), Task.id.desc())
            .offset(offset)
            .limit(normalized_size)
        )
    )
    return TaskListPage(
        items=_task_items(session, tasks),
        page=normalized_page,
        page_size=normalized_size,
        total_count=total_count,
    )


def get_task_detail(
    session: Session,
    task_id: uuid.UUID,
    *,
    installation_id: uuid.UUID | None = None,
    slack_user_id: str | None = None,
) -> TaskDetail | None:
    """Return one task and its child rows."""

    task = session.get(Task, task_id)
    if task is None:
        return None
    if installation_id is not None and task.installation_id != installation_id:
        return None
    if slack_user_id is not None and task.slack_user_id != slack_user_id:
        return None
    events = tuple(
        session.scalars(
            select(TaskEvent)
            .where(TaskEvent.task_id == task_id)
            .order_by(TaskEvent.seq.asc())
        )
    )
    usage = tuple(
        session.scalars(
            select(LLMUsage)
            .where(LLMUsage.task_id == task_id)
            .order_by(LLMUsage.created_at.asc(), LLMUsage.id.asc())
        )
    )
    artifacts = tuple(
        session.scalars(
            select(Artifact)
            .where(Artifact.task_id == task_id)
            .order_by(Artifact.created_at.asc(), Artifact.id.asc())
        )
    )
    timeline = tuple(_timeline_event(event) for event in events)
    posted_response_text = _posted_response_text(events)
    identities = _identity_map(session, (task,))
    return TaskDetail(
        task=task,
        channel=_identity_label(
            identities,
            installation_id=task.installation_id,
            kind="channel",
            slack_id=task.slack_channel_id,
        ),
        user=_identity_label(
            identities,
            installation_id=task.installation_id,
            kind="user",
            slack_id=task.slack_user_id,
        ),
        events=events,
        timeline=timeline,
        usage=usage,
        artifacts=artifacts,
        posted_response_text=posted_response_text,
        planned_trace=_planned_workflow_trace(
            events=events,
            timeline=timeline,
            usage=usage,
            raw_response_text=task.result_summary,
            posted_response_text=posted_response_text,
        ),
    )


def get_usage_aggregate(
    session: Session,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
    installation_id: uuid.UUID | None = None,
    slack_user_id: str | None = None,
) -> UsageAggregate:
    """Return dashboard usage rollups."""

    usage_filter = _usage_filter(start=start, end=end)
    scoped_task_filter = _task_scope_filter(
        installation_id=installation_id,
        slack_user_id=slack_user_id,
    )
    by_model_rows = session.execute(
        select(
            LLMUsage.model,
            func.count(LLMUsage.id),
            func.coalesce(func.sum(LLMUsage.input_tokens), 0),
            func.coalesce(func.sum(LLMUsage.output_tokens), 0),
            func.coalesce(func.sum(LLMUsage.cost_usd), 0),
        )
        .join(Task, Task.id == LLMUsage.task_id)
        .where(*usage_filter, *scoped_task_filter)
        .group_by(LLMUsage.model)
        .order_by(func.sum(LLMUsage.cost_usd).desc())
    ).all()
    by_user_raw_rows = session.execute(
        select(
            Task.installation_id,
            Task.slack_user_id,
            func.count(LLMUsage.id),
            func.coalesce(func.sum(LLMUsage.input_tokens), 0),
            func.coalesce(func.sum(LLMUsage.output_tokens), 0),
            func.coalesce(func.sum(LLMUsage.cost_usd), 0),
        )
        .join(Task, Task.id == LLMUsage.task_id)
        .where(*usage_filter, *scoped_task_filter)
        .group_by(Task.installation_id, Task.slack_user_id)
        .order_by(func.sum(LLMUsage.cost_usd).desc())
    ).all()
    user_identities = _identity_map_from_keys(
        session,
        (
            (row[0], "user", row[1])
            for row in by_user_raw_rows
            if row[0] is not None and row[1] is not None
        ),
    )
    by_user_rows = tuple(
        _aggregate_row(
            (row[1], row[2], row[3], row[4], row[5]),
            label=_identity_label(
                user_identities,
                installation_id=row[0],
                kind="user",
                slack_id=row[1],
            ),
        )
        for row in by_user_raw_rows
    )
    by_channel_raw_rows = session.execute(
        select(
            Task.installation_id,
            Task.slack_channel_id,
            func.count(LLMUsage.id),
            func.coalesce(func.sum(LLMUsage.input_tokens), 0),
            func.coalesce(func.sum(LLMUsage.output_tokens), 0),
            func.coalesce(func.sum(LLMUsage.cost_usd), 0),
        )
        .join(Task, Task.id == LLMUsage.task_id)
        .where(
            *usage_filter, *scoped_task_filter, ~Task.slack_channel_id.startswith("D")
        )
        .group_by(Task.installation_id, Task.slack_channel_id)
        .order_by(func.sum(LLMUsage.cost_usd).desc())
    ).all()
    channel_identities = _identity_map_from_keys(
        session,
        (
            (row[0], "channel", row[1])
            for row in by_channel_raw_rows
            if row[0] is not None and row[1] is not None
        ),
    )
    by_channel_rows = tuple(
        _aggregate_row(
            (row[1], row[2], row[3], row[4], row[5]),
            label=_identity_label(
                channel_identities,
                installation_id=row[0],
                kind="channel",
                slack_id=row[1],
            ),
        )
        for row in by_channel_raw_rows
    )
    day_bucket = func.date_trunc("day", LLMUsage.created_at).label("day")
    by_day_rows = session.execute(
        select(
            day_bucket,
            func.count(LLMUsage.id),
            func.coalesce(func.sum(LLMUsage.input_tokens), 0),
            func.coalesce(func.sum(LLMUsage.output_tokens), 0),
            func.coalesce(func.sum(LLMUsage.cost_usd), 0),
        )
        .join(Task, Task.id == LLMUsage.task_id)
        .where(*usage_filter, *scoped_task_filter)
        .group_by(day_bucket)
        .order_by(day_bucket.desc())
    ).all()
    task_day_bucket = func.date_trunc("day", Task.created_at).label("day")
    by_task_day_rows = session.execute(
        select(
            task_day_bucket,
            func.count(Task.id),
            func.coalesce(func.sum(_failed_task_case()), 0),
        )
        .where(
            *_task_filter(start=start, end=end),
            *scoped_task_filter,
        )
        .group_by(task_day_bucket)
        .order_by(task_day_bucket.desc())
    ).all()
    cache = _cache_stats(
        session,
        usage_filter=usage_filter,
        scoped_task_filter=scoped_task_filter,
    )
    return UsageAggregate(
        start=start,
        end=end,
        by_model=tuple(_aggregate_row(row) for row in by_model_rows),
        by_user=by_user_rows,
        by_channel=by_channel_rows,
        by_day=tuple(_daily_row(row) for row in by_day_rows),
        by_task_day=tuple(_daily_task_row(row) for row in by_task_day_rows),
        cache=cache,
    )


def _cache_stats(
    session: Session,
    *,
    usage_filter: Sequence[Any],
    scoped_task_filter: Sequence[Any],
) -> CacheStats:
    """Prompt-cache rollup: hit rate + estimated savings USD (HIG-196).

    Savings = cache_read * base_input_price * (1 - cache_read_multiplier),
    matched per (provider, model) against the latest effective pricing row.
    Rows without a pricing match contribute tokens to the hit-rate denominator
    but $0 to savings (we can't price them).
    """

    totals = session.execute(
        select(
            func.coalesce(func.sum(LLMUsage.input_tokens), 0),
            func.coalesce(func.sum(LLMUsage.cache_read_input_tokens), 0),
            func.coalesce(func.sum(LLMUsage.cache_creation_input_tokens), 0),
        )
        .join(Task, Task.id == LLMUsage.task_id)
        .where(*usage_filter, *scoped_task_filter)
    ).one()
    total_input = int(totals[0])
    total_read = int(totals[1])
    total_creation = int(totals[2])

    # Latest pricing row per (provider, model) — savings priced against it.
    latest_pricing = (
        select(
            ModelPricing.provider.label("provider"),
            ModelPricing.model.label("model"),
            func.max(ModelPricing.effective_from).label("effective_from"),
        )
        .group_by(ModelPricing.provider, ModelPricing.model)
        .subquery()
    )
    savings_rows = session.execute(
        select(
            func.coalesce(func.sum(LLMUsage.cache_read_input_tokens), 0),
            ModelPricing.input_price_per_mtok,
            ModelPricing.cache_read_multiplier,
        )
        .join(Task, Task.id == LLMUsage.task_id)
        .join(
            ModelPricing,
            (ModelPricing.provider == LLMUsage.provider)
            & (ModelPricing.model == LLMUsage.model),
        )
        .join(
            latest_pricing,
            (latest_pricing.c.provider == ModelPricing.provider)
            & (latest_pricing.c.model == ModelPricing.model)
            & (latest_pricing.c.effective_from == ModelPricing.effective_from),
        )
        .where(*usage_filter, *scoped_task_filter)
        .group_by(
            ModelPricing.input_price_per_mtok,
            ModelPricing.cache_read_multiplier,
        )
    ).all()
    savings = Decimal("0")
    for read_tokens, input_price, read_multiplier in savings_rows:
        full = Decimal(read_tokens) * Decimal(input_price)
        discounted = full * Decimal(read_multiplier)
        savings += (full - discounted) / _TOKENS_PER_MTOK
    savings = savings.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)

    return CacheStats(
        total_input_tokens=total_input,
        cache_read_input_tokens=total_read,
        cache_creation_input_tokens=total_creation,
        estimated_savings_usd=savings,
    )


def list_users(
    session: Session,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
) -> UserDirectory:
    """Return user-level task/cost rollups."""

    task_filter = _task_filter(start=start, end=end)
    rows = session.execute(
        select(
            Task.installation_id,
            Task.slack_user_id,
            func.count(Task.id),
            func.coalesce(func.sum(_failed_task_case()), 0),
            func.coalesce(func.sum(Task.total_input_tokens), 0),
            func.coalesce(func.sum(Task.total_output_tokens), 0),
            func.coalesce(func.sum(Task.total_cost_usd), 0),
            func.max(Task.created_at),
        )
        .where(*task_filter)
        .group_by(Task.installation_id, Task.slack_user_id)
        .order_by(
            func.sum(Task.total_cost_usd).desc(), func.max(Task.created_at).desc()
        )
    ).all()
    artifact_counts = _artifact_counts_by_user(session, task_filter)
    identities = _identity_map_from_keys(
        session,
        (
            (row[0], "user", row[1])
            for row in rows
            if row[0] is not None and row[1] is not None
        ),
    )
    users = tuple(
        UserListItem(
            user=_identity_label(
                identities,
                installation_id=row[0],
                kind="user",
                slack_id=row[1],
            ),
            task_count=int(row[2]),
            failed_task_count=int(row[3]),
            total_input_tokens=int(row[4]),
            total_output_tokens=int(row[5]),
            total_cost_usd=Decimal(row[6]),
            last_activity_at=row[7],
            artifact_count=artifact_counts.get((row[0], row[1]), 0),
        )
        for row in rows
    )
    return UserDirectory(start=start, end=end, users=users)


def get_user_detail(
    session: Session,
    slack_user_id: str,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
    installation_id: uuid.UUID | None = None,
) -> UserDetail | None:
    """Return one user's tasks, usage, and artifacts."""

    task_filter = [
        Task.slack_user_id == slack_user_id,
        *_task_scope_filter(installation_id=installation_id),
        *_task_filter(start=start, end=end),
    ]
    stats = session.execute(
        select(
            func.count(Task.id),
            func.coalesce(func.sum(_failed_task_case()), 0),
            func.coalesce(func.sum(Task.total_input_tokens), 0),
            func.coalesce(func.sum(Task.total_output_tokens), 0),
            func.coalesce(func.sum(Task.total_cost_usd), 0),
            func.max(Task.created_at),
        ).where(*task_filter)
    ).one()
    task_count = int(stats[0])
    if task_count == 0:
        return None

    tasks = tuple(
        session.scalars(
            select(Task)
            .where(*task_filter)
            .order_by(Task.created_at.desc(), Task.id.desc())
            .limit(25)
        )
    )
    task_ids = [task.id for task in tasks]
    usage_by_task = _usage_by_task(session, task_ids)
    artifact_counts = _artifact_counts_by_task(session, task_ids)
    identities = _identity_map(session, tasks)
    user_label = _user_label_for_tasks(
        session,
        slack_user_id=slack_user_id,
        tasks=tasks,
    )

    usage_filter = [
        Task.slack_user_id == slack_user_id,
        *_task_scope_filter(installation_id=installation_id),
        *_usage_filter(start=start, end=end),
    ]
    usage = tuple(
        session.scalars(
            select(LLMUsage)
            .join(Task, Task.id == LLMUsage.task_id)
            .where(*usage_filter)
            .order_by(LLMUsage.created_at.desc(), LLMUsage.id.desc())
            .limit(25)
        )
    )
    artifact_rows = tuple(
        session.execute(
            select(Artifact, Task)
            .join(Task, Task.id == Artifact.task_id)
            .where(*task_filter)
            .order_by(Artifact.created_at.desc(), Artifact.id.desc())
            .limit(25)
        )
    )
    artifacts = tuple(
        UserArtifactRow(artifact=row[0], task=row[1]) for row in artifact_rows
    )

    return UserDetail(
        user=user_label,
        start=start,
        end=end,
        task_count=task_count,
        failed_task_count=int(stats[1]),
        total_input_tokens=int(stats[2]),
        total_output_tokens=int(stats[3]),
        total_cost_usd=Decimal(stats[4]),
        last_activity_at=stats[5],
        usage_call_count=len(usage),
        artifact_count=_artifact_count_for_user(session, task_filter),
        tasks=tuple(
            UserTaskRow(
                task=task,
                channel=_identity_label(
                    identities,
                    installation_id=task.installation_id,
                    kind="channel",
                    slack_id=task.slack_channel_id,
                ),
                usage_count=len(usage_by_task[task.id]),
                artifact_count=artifact_counts.get(task.id, 0),
            )
            for task in tasks
        ),
        usage=usage,
        artifacts=artifacts,
    )


def get_dashboard_overview(
    session: Session,
    *,
    system_health: SystemHealth,
    now: datetime | None = None,
) -> DashboardOverview:
    """Return the operator dashboard home read model."""

    current = now or datetime.now(UTC)
    current = current.astimezone(UTC)
    today_start = datetime.combine(current.date(), time.min, tzinfo=UTC)
    week_start = current - timedelta(days=7)
    query_end = current + timedelta(seconds=1)

    usage = get_usage_aggregate(session, start=week_start, end=query_end)
    week_stats = session.execute(
        select(
            func.count(Task.id),
            func.coalesce(func.sum(_failed_task_case()), 0),
            func.coalesce(func.sum(Task.total_cost_usd), 0),
        ).where(*_task_filter(start=week_start, end=query_end))
    ).one()
    active_tasks = (
        session.scalar(
            select(func.count())
            .select_from(Task)
            .where(Task.status.in_((TaskStatus.pending, TaskStatus.running)))
        )
        or 0
    )
    today_cost = session.scalar(
        select(func.coalesce(func.sum(LLMUsage.cost_usd), 0)).where(
            LLMUsage.created_at >= today_start,
            LLMUsage.created_at < query_end,
        )
    ) or Decimal("0")
    last_task_at = session.scalar(select(func.max(Task.created_at)))
    week_task_count = int(week_stats[0])
    week_failed_count = int(week_stats[1])

    metrics = (
        SystemMetric(
            label="Readiness",
            value=system_health.overall_label,
            detail="Worst current system check",
            tone=system_health.overall_tone,
        ),
        SystemMetric(
            label="Active Tasks",
            value=f"{active_tasks:,}",
            detail="Pending or running now",
            tone="warning" if active_tasks else "neutral",
        ),
        SystemMetric(
            label="Today Cost",
            value=_format_money(Decimal(today_cost)),
            detail="LLM usage recorded today",
        ),
        SystemMetric(
            label="7 Day Failures",
            value=_failure_rate_label(week_task_count, week_failed_count),
            detail=f"{week_failed_count:,} of {week_task_count:,} tasks",
            tone="danger" if week_failed_count else "neutral",
        ),
        SystemMetric(
            label="Last Task",
            value=_relative_label(last_task_at, current),
            detail="Most recent task creation",
        ),
    )

    attention_items = _overview_attention_items(
        session,
        system_health=system_health,
        current=current,
        today_cost=Decimal(today_cost),
        week_cost=Decimal(week_stats[2]),
    )
    recent_tasks = list_tasks(session, page=1, page_size=10)

    # 1. Fetch top 5 active facts
    _, _, active_facts = _memory_fact_rows(
        session,
        query="",
        scope_filter="all",
        status_filter="active",
        sort="newest",
        page=1,
        page_size=5,
    )

    # 2. Fetch active channel profiles
    profiles = tuple(
        session.scalars(
            select(ObserveChannelProfile)
            .where(ObserveChannelProfile.profile_status == "active")
            .order_by(ObserveChannelProfile.updated_at.desc())
            .limit(5)
        )
    )
    profile_identities = _identity_map_from_keys(
        session,
        ((cp.installation_id, "channel", cp.channel_id) for cp in profiles),
    )
    channel_profiles = tuple(
        OverviewChannelProfileRow(
            channel=_identity_label(
                profile_identities,
                installation_id=cp.installation_id,
                kind="channel",
                slack_id=cp.channel_id,
            ),
            summary=cp.summary,
            message_count=cp.message_count,
            updated_at=cp.updated_at,
        )
        for cp in profiles
    )

    # 3. Fetch skill usage stats
    skill_rows = session.execute(
        select(
            ProceduralSkillVersion.name,
            ProceduralSkillVersion.description,
            func.count(ProceduralSkillInvocation.id).label("count"),
        )
        .join(
            ProceduralSkillVersion,
            ProceduralSkillVersion.id == ProceduralSkillInvocation.skill_version_id,
        )
        .group_by(ProceduralSkillVersion.name, ProceduralSkillVersion.description)
        .order_by(func.count(ProceduralSkillInvocation.id).desc())
        .limit(5)
    ).all()
    skill_usage = tuple(
        OverviewSkillUsageRow(
            name=row[0],
            description=row[1],
            invocation_count=int(row[2]),
        )
        for row in skill_rows
    )

    return DashboardOverview(
        metrics=metrics,
        attention_items=attention_items,
        charts=usage.charts,
        top_models=usage.by_model[:5],
        top_users=usage.by_user[:5],
        top_channels=usage.by_channel[:5],
        recent_tasks=recent_tasks.items,
        system_health=system_health,
        window_label="Last 7 days",
        refreshed_at=current,
        active_facts=active_facts,
        skill_usage=skill_usage,
        channel_profiles=channel_profiles,
        by_day=usage.by_day,
        by_task_day=usage.by_task_day,
    )


def get_memory_dashboard(
    session: Session,
    *,
    view: str = "facts",
    query: str | None = None,
    scope_filter: str = "all",
    status_filter: str = "active",
    outcome_filter: str = "all",
    sort: str | None = None,
    page: int = 1,
    page_size: int = DEFAULT_PAGE_SIZE,
    installation_id: uuid.UUID | None = None,
    slack_user_id: str | None = None,
    base_path: str = "/memory",
) -> MemoryDashboard:
    """Return read-only memory state for the management console."""

    active_view = "episodes" if view == "episodes" else "facts"
    normalized_query = " ".join((query or "").split())
    normalized_page = max(page, 1)
    normalized_size = min(max(page_size, 1), MAX_PAGE_SIZE)
    normalized_scope = scope_filter if scope_filter in _MEMORY_SCOPES else "all"
    normalized_status = status_filter if status_filter in _MEMORY_STATUSES else "active"
    normalized_outcome = outcome_filter if outcome_filter in _MEMORY_OUTCOMES else "all"
    normalized_sort = _normalize_memory_sort(active_view, sort)
    memory_fact_scope = _workspace_state_scope_filter(
        installation_id=installation_id,
        slack_user_id=slack_user_id,
    )
    memory_episode_scope = _episode_scope_filter(
        installation_id=installation_id,
        slack_user_id=slack_user_id,
    )

    active_fact_count = (
        session.scalar(
            select(func.count())
            .select_from(WorkspaceState)
            .where(WorkspaceState.status == "active", *memory_fact_scope)
        )
        or 0
    )
    proposed_fact_count = (
        session.scalar(
            select(func.count())
            .select_from(WorkspaceState)
            .where(WorkspaceState.status == "proposed", *memory_fact_scope)
        )
        or 0
    )
    episode_count = (
        session.scalar(
            select(func.count()).select_from(Episode).where(*memory_episode_scope)
        )
        or 0
    )
    failed_episode_count = (
        session.scalar(
            select(func.count())
            .select_from(Episode)
            .where(Episode.outcome == "failed", *memory_episode_scope)
        )
        or 0
    )

    facts: tuple[MemoryFactRow, ...] = ()
    episodes: tuple[MemoryEpisodeRow, ...] = ()
    if active_view == "facts":
        total_count, resolved_page, facts = _memory_fact_rows(
            session,
            query=normalized_query,
            scope_filter=normalized_scope,
            status_filter=normalized_status,
            sort=normalized_sort,
            page=normalized_page,
            page_size=normalized_size,
            installation_id=installation_id,
            slack_user_id=slack_user_id,
        )
        noun = "facts"
    else:
        total_count, resolved_page, episodes = _memory_episode_rows(
            session,
            query=normalized_query,
            outcome_filter=normalized_outcome,
            sort=normalized_sort,
            page=normalized_page,
            page_size=normalized_size,
            installation_id=installation_id,
            slack_user_id=slack_user_id,
        )
        noun = "episodes"

    page_info = MemoryPageInfo(
        page=resolved_page,
        page_size=normalized_size,
        total_count=total_count,
        noun=noun,
    )

    return MemoryDashboard(
        active_fact_count=int(active_fact_count),
        proposed_fact_count=int(proposed_fact_count),
        episode_count=int(episode_count),
        failed_episode_count=int(failed_episode_count),
        active_view=active_view,
        query=normalized_query,
        scope_filter=normalized_scope,
        status_filter=normalized_status,
        outcome_filter=normalized_outcome,
        sort=normalized_sort,
        page=page_info,
        previous_page_url=(
            _memory_page_url(
                view=active_view,
                query=normalized_query,
                scope_filter=normalized_scope,
                status_filter=normalized_status,
                outcome_filter=normalized_outcome,
                sort=normalized_sort,
                page=page_info.previous_page,
                page_size=normalized_size,
                base_path=base_path,
            )
            if page_info.previous_page is not None
            else None
        ),
        next_page_url=(
            _memory_page_url(
                view=active_view,
                query=normalized_query,
                scope_filter=normalized_scope,
                status_filter=normalized_status,
                outcome_filter=normalized_outcome,
                sort=normalized_sort,
                page=page_info.next_page,
                page_size=normalized_size,
                base_path=base_path,
            )
            if page_info.next_page is not None
            else None
        ),
        reset_url=f"{base_path}?view={active_view}",
        facts=facts,
        episodes=episodes,
    )


def get_knowledge_graph_dashboard(
    session: Session,
    *,
    view: str = "entities",
    query: str | None = None,
    scope_filter: str = "all",
    state_filter: str = "current",
    kind_filter: str = "all",
    sort: str | None = None,
    page: int = 1,
    page_size: int = DEFAULT_PAGE_SIZE,
    installation_id: uuid.UUID | None = None,
    base_path: str = "/knowledge-graph",
) -> KnowledgeGraphDashboard:
    """Return read-only workspace knowledge graph rows for the dashboard."""

    active_view = "relationships" if view == "relationships" else "entities"
    normalized_query = " ".join((query or "").split())
    normalized_page = max(page, 1)
    normalized_size = min(max(page_size, 1), MAX_PAGE_SIZE)
    normalized_scope = scope_filter if scope_filter in _KG_SCOPES else "all"
    normalized_state = state_filter if state_filter in _KG_STATES else "current"
    normalized_kind = " ".join((kind_filter or "all").split()).lower() or "all"
    normalized_sort = _normalize_kg_sort(active_view, sort)

    entity_scope = _kg_installation_filter(KnowledgeGraphEntity, installation_id)
    edge_scope = _kg_installation_filter(KnowledgeGraphEdge, installation_id)
    evidence_scope = _kg_installation_filter(KnowledgeGraphEvidence, installation_id)
    entity_count = (
        session.scalar(
            select(func.count()).select_from(KnowledgeGraphEntity).where(*entity_scope)
        )
        or 0
    )
    current_entity_count = (
        session.scalar(
            select(func.count())
            .select_from(KnowledgeGraphEntity)
            .where(
                *entity_scope,
                KnowledgeGraphEntity.is_current.is_(True),
                KnowledgeGraphEntity.expired_at.is_(None),
            )
        )
        or 0
    )
    candidate_entity_count = (
        session.scalar(
            select(func.count())
            .select_from(KnowledgeGraphEntity)
            .where(
                *entity_scope,
                KnowledgeGraphEntity.lifecycle_state == "candidate",
                KnowledgeGraphEntity.is_current.is_(True),
                KnowledgeGraphEntity.expired_at.is_(None),
            )
        )
        or 0
    )
    known_channel_count = (
        session.scalar(
            select(func.count())
            .select_from(SlackChannelMembership)
            .where(
                *_kg_installation_filter(SlackChannelMembership, installation_id),
                SlackChannelMembership.membership_status == "active",
            )
        )
        or 0
    )
    profiled_channel_count = (
        session.scalar(
            select(func.count())
            .select_from(KnowledgeGraphEntity)
            .where(
                *entity_scope,
                KnowledgeGraphEntity.canonical_key.like("channel_profile:%"),
                KnowledgeGraphEntity.is_current.is_(True),
                KnowledgeGraphEntity.expired_at.is_(None),
            )
        )
        or 0
    )
    stale_entity_count = (
        session.scalar(
            select(func.count())
            .select_from(KnowledgeGraphEntity)
            .where(
                *entity_scope,
                *_kg_current_filters(KnowledgeGraphEntity),
                KnowledgeGraphEntity.lifecycle_state == "stale",
            )
        )
        or 0
    )
    edge_count = (
        session.scalar(
            select(func.count()).select_from(KnowledgeGraphEdge).where(*edge_scope)
        )
        or 0
    )
    stale_edge_count = (
        session.scalar(
            select(func.count())
            .select_from(KnowledgeGraphEdge)
            .where(
                *edge_scope,
                *_kg_current_filters(KnowledgeGraphEdge),
                KnowledgeGraphEdge.lifecycle_state == "stale",
            )
        )
        or 0
    )
    evidence_count = (
        session.scalar(
            select(func.count())
            .select_from(KnowledgeGraphEvidence)
            .where(*evidence_scope)
        )
        or 0
    )
    unbacked_current_entity_count = (
        session.scalar(
            select(func.count())
            .select_from(KnowledgeGraphEntity)
            .where(
                *entity_scope,
                *_kg_current_filters(KnowledgeGraphEntity),
                ~_kg_has_evidence_predicate(KnowledgeGraphEntity, "entity"),
            )
        )
        or 0
    )
    unbacked_current_edge_count = (
        session.scalar(
            select(func.count())
            .select_from(KnowledgeGraphEdge)
            .where(
                *edge_scope,
                *_kg_current_filters(KnowledgeGraphEdge),
                ~_kg_has_evidence_predicate(KnowledgeGraphEdge, "edge"),
            )
        )
        or 0
    )
    runtime_eligible_entity_count = (
        session.scalar(
            select(func.count())
            .select_from(KnowledgeGraphEntity)
            .where(
                *entity_scope,
                *_kg_current_filters(KnowledgeGraphEntity),
                KnowledgeGraphEntity.lifecycle_state.in_(("active", "confirmed")),
                _kg_has_evidence_predicate(KnowledgeGraphEntity, "entity"),
            )
        )
        or 0
    )
    runtime_eligible_edge_count = (
        session.scalar(
            select(func.count())
            .select_from(KnowledgeGraphEdge)
            .where(
                *edge_scope,
                *_kg_current_filters(KnowledgeGraphEdge),
                KnowledgeGraphEdge.lifecycle_state.in_(("active", "confirmed")),
                _kg_has_evidence_predicate(KnowledgeGraphEdge, "edge"),
            )
        )
        or 0
    )

    entities: tuple[KnowledgeGraphEntityRow, ...] = ()
    edges: tuple[KnowledgeGraphEdgeRow, ...] = ()
    if active_view == "entities":
        total_count, resolved_page, entities = _kg_entity_rows(
            session,
            query=normalized_query,
            scope_filter=normalized_scope,
            state_filter=normalized_state,
            kind_filter=normalized_kind,
            sort=normalized_sort,
            page=normalized_page,
            page_size=normalized_size,
            installation_id=installation_id,
        )
        noun = "entities"
    else:
        total_count, resolved_page, edges = _kg_edge_rows(
            session,
            query=normalized_query,
            scope_filter=normalized_scope,
            state_filter=normalized_state,
            kind_filter=normalized_kind,
            sort=normalized_sort,
            page=normalized_page,
            page_size=normalized_size,
            installation_id=installation_id,
        )
        noun = "relationships"

    graph_map = _kg_graph_map(
        session,
        active_view=active_view,
        query=normalized_query,
        scope_filter=normalized_scope,
        state_filter=normalized_state,
        kind_filter=normalized_kind,
        sort=normalized_sort,
        installation_id=installation_id,
    )
    page_info = KnowledgeGraphPageInfo(
        page=resolved_page,
        page_size=normalized_size,
        total_count=total_count,
        noun=noun,
    )
    entities_url = _knowledge_graph_page_url(
        view="entities",
        query=normalized_query,
        scope_filter=normalized_scope,
        state_filter=normalized_state,
        kind_filter=normalized_kind,
        sort="updated_desc",
        page=1,
        page_size=normalized_size,
        base_path=base_path,
    )
    relationships_url = _knowledge_graph_page_url(
        view="relationships",
        query=normalized_query,
        scope_filter=normalized_scope,
        state_filter=normalized_state,
        kind_filter=normalized_kind,
        sort="updated_desc",
        page=1,
        page_size=normalized_size,
        base_path=base_path,
    )
    return KnowledgeGraphDashboard(
        entity_count=int(entity_count),
        current_entity_count=int(current_entity_count),
        candidate_entity_count=int(candidate_entity_count),
        known_channel_count=int(known_channel_count),
        profiled_channel_count=int(profiled_channel_count),
        edge_count=int(edge_count),
        evidence_count=int(evidence_count),
        stale_entity_count=int(stale_entity_count),
        stale_edge_count=int(stale_edge_count),
        unbacked_current_entity_count=int(unbacked_current_entity_count),
        unbacked_current_edge_count=int(unbacked_current_edge_count),
        runtime_eligible_entity_count=int(runtime_eligible_entity_count),
        runtime_eligible_edge_count=int(runtime_eligible_edge_count),
        active_view=active_view,
        query=normalized_query,
        scope_filter=normalized_scope,
        state_filter=normalized_state,
        kind_filter=normalized_kind,
        sort=normalized_sort,
        map=graph_map,
        page=page_info,
        previous_page_url=(
            _knowledge_graph_page_url(
                view=active_view,
                query=normalized_query,
                scope_filter=normalized_scope,
                state_filter=normalized_state,
                kind_filter=normalized_kind,
                sort=normalized_sort,
                page=page_info.previous_page,
                page_size=normalized_size,
                base_path=base_path,
            )
            if page_info.previous_page is not None
            else None
        ),
        next_page_url=(
            _knowledge_graph_page_url(
                view=active_view,
                query=normalized_query,
                scope_filter=normalized_scope,
                state_filter=normalized_state,
                kind_filter=normalized_kind,
                sort=normalized_sort,
                page=page_info.next_page,
                page_size=normalized_size,
                base_path=base_path,
            )
            if page_info.next_page is not None
            else None
        ),
        reset_url=f"{base_path}?view={active_view}",
        entities_url=entities_url,
        relationships_url=relationships_url,
        entities=entities,
        edges=edges,
    )


def get_witness_candidates_dashboard(
    session: Session,
    *,
    query: str | None = None,
    status_filter: str = "candidate",
    type_filter: str = "all",
    scope_filter: str = "all",
    sort: str | None = None,
    page: int = 1,
    page_size: int = DEFAULT_PAGE_SIZE,
    installation_id: uuid.UUID | None = None,
    base_path: str = "/witness",
) -> WitnessCandidatesDashboard:
    """Return read-only proactive work candidates for the Witness dashboard."""

    normalized_query = " ".join((query or "").split())
    normalized_page = max(page, 1)
    normalized_size = min(max(page_size, 1), MAX_PAGE_SIZE)
    normalized_status = (
        status_filter if status_filter in _WITNESS_STATUSES else "candidate"
    )
    normalized_type = type_filter if type_filter in _WITNESS_TYPES else "all"
    normalized_scope = scope_filter if scope_filter in _WITNESS_SCOPES else "all"
    normalized_sort = sort if sort in _WITNESS_SORTS else "updated_desc"
    now = datetime.now(UTC)

    installation_scope = _kg_installation_filter(
        WitnessOpportunityCandidate, installation_id
    )
    total_count = (
        session.scalar(
            select(func.count())
            .select_from(WitnessOpportunityCandidate)
            .where(*installation_scope)
        )
        or 0
    )
    candidate_count = (
        session.scalar(
            select(func.count())
            .select_from(WitnessOpportunityCandidate)
            .where(
                *installation_scope,
                WitnessOpportunityCandidate.status == "candidate",
            )
        )
        or 0
    )
    due_candidate_count = (
        session.scalar(
            select(func.count())
            .select_from(WitnessOpportunityCandidate)
            .where(
                *installation_scope,
                WitnessOpportunityCandidate.status == "candidate",
                or_(
                    WitnessOpportunityCandidate.cooldown_until.is_(None),
                    WitnessOpportunityCandidate.cooldown_until <= now,
                ),
            )
        )
        or 0
    )
    sent_count = (
        session.scalar(
            select(func.count())
            .select_from(WitnessOpportunityCandidate)
            .where(*installation_scope, WitnessOpportunityCandidate.status == "sent")
        )
        or 0
    )
    accepted_count = (
        session.scalar(
            select(func.count())
            .select_from(WitnessOpportunityCandidate)
            .where(
                *installation_scope,
                WitnessOpportunityCandidate.status == "accepted",
            )
        )
        or 0
    )
    automated_count = (
        session.scalar(
            select(func.count())
            .select_from(WitnessOpportunityCandidate)
            .where(
                *installation_scope,
                WitnessOpportunityCandidate.status == "automated",
            )
        )
        or 0
    )
    inactive_count = (
        session.scalar(
            select(func.count())
            .select_from(WitnessOpportunityCandidate)
            .where(
                *installation_scope,
                WitnessOpportunityCandidate.status.in_(
                    ("dismissed", "cooldown", "superseded", "archived")
                ),
            )
        )
        or 0
    )

    row_total_count, resolved_page, rows = _witness_candidate_rows(
        session,
        query=normalized_query,
        status_filter=normalized_status,
        type_filter=normalized_type,
        scope_filter=normalized_scope,
        sort=normalized_sort,
        page=normalized_page,
        page_size=normalized_size,
        installation_id=installation_id,
        now=now,
    )
    page_info = WitnessCandidatePageInfo(
        page=resolved_page,
        page_size=normalized_size,
        total_count=row_total_count,
    )
    return WitnessCandidatesDashboard(
        total_count=int(total_count),
        candidate_count=int(candidate_count),
        due_candidate_count=int(due_candidate_count),
        sent_count=int(sent_count),
        accepted_count=int(accepted_count),
        automated_count=int(automated_count),
        inactive_count=int(inactive_count),
        query=normalized_query,
        status_filter=normalized_status,
        type_filter=normalized_type,
        scope_filter=normalized_scope,
        sort=normalized_sort,
        page=page_info,
        previous_page_url=(
            _witness_candidates_url(
                query=normalized_query,
                status_filter=normalized_status,
                type_filter=normalized_type,
                scope_filter=normalized_scope,
                sort=normalized_sort,
                page=page_info.previous_page,
                page_size=normalized_size,
                base_path=base_path,
            )
            if page_info.previous_page is not None
            else None
        ),
        next_page_url=(
            _witness_candidates_url(
                query=normalized_query,
                status_filter=normalized_status,
                type_filter=normalized_type,
                scope_filter=normalized_scope,
                sort=normalized_sort,
                page=page_info.next_page,
                page_size=normalized_size,
                base_path=base_path,
            )
            if page_info.next_page is not None
            else None
        ),
        reset_url=base_path,
        rows=rows,
    )


_WITNESS_DELIVERED_DECISIONS = ("notify", "question", "draft", "channel_sent")
_WITNESS_DECISION_LABELS = {
    "notify": "Notify (digest)",
    "question": "Question (cadence ask)",
    "draft": "Draft (say go)",
    "silent": "Silent (below threshold)",
    "digest": "Digest DMs sent",
    "channel_sent": "Channel post sent",
    "channel_deferred": "Channel deferred (policy/quiet/budget)",
    "draft_executed": "Draft executed (autopilot)",
}


def get_witness_kpis(
    session: Session,
    *,
    installation_id: uuid.UUID | None = None,
    now: datetime | None = None,
    window_days: int = 30,
) -> WitnessKpis:
    """Compute proactivity-loop KPIs from witness_delivery_log + candidates."""

    observed_now = now or datetime.now(UTC)
    cutoff = observed_now - timedelta(days=window_days)
    candidate_scope = _kg_installation_filter(
        WitnessOpportunityCandidate, installation_id
    )
    log_scope = _kg_installation_filter(WitnessDeliveryLog, installation_id)

    candidates_created = int(
        session.scalar(
            select(func.count())
            .select_from(WitnessOpportunityCandidate)
            .where(
                *candidate_scope,
                WitnessOpportunityCandidate.created_at >= cutoff,
            )
        )
        or 0
    )
    decision_rows = session.execute(
        select(WitnessDeliveryLog.decision, func.count())
        .where(*log_scope, WitnessDeliveryLog.created_at >= cutoff)
        .group_by(WitnessDeliveryLog.decision)
    ).all()
    decision_totals = {decision: int(count) for decision, count in decision_rows}
    delivered_count = int(
        session.scalar(
            select(func.count())
            .select_from(WitnessDeliveryLog)
            .where(
                *log_scope,
                WitnessDeliveryLog.created_at >= cutoff,
                WitnessDeliveryLog.decision.in_(_WITNESS_DELIVERED_DECISIONS),
                WitnessDeliveryLog.reason == "sent",
            )
        )
        or 0
    )
    silent_count = decision_totals.get("silent", 0)
    decided_count = silent_count + delivered_count
    silent_rate = silent_count / decided_count if decided_count else None

    delivered_pairs = session.execute(
        select(
            WitnessDeliveryLog.candidate_id,
            func.min(WitnessDeliveryLog.created_at),
        )
        .where(
            *log_scope,
            WitnessDeliveryLog.created_at >= cutoff,
            WitnessDeliveryLog.decision.in_(_WITNESS_DELIVERED_DECISIONS),
            WitnessDeliveryLog.reason == "sent",
            WitnessDeliveryLog.candidate_id.is_not(None),
        )
        .group_by(WitnessDeliveryLog.candidate_id)
    ).all()
    delivered_at_by_candidate = {
        candidate_id: delivered_at for candidate_id, delivered_at in delivered_pairs
    }
    delivered_candidates = (
        tuple(
            session.scalars(
                select(WitnessOpportunityCandidate).where(
                    WitnessOpportunityCandidate.id.in_(tuple(delivered_at_by_candidate))
                )
            )
        )
        if delivered_at_by_candidate
        else ()
    )
    accepted = sum(
        1
        for candidate in delivered_candidates
        if candidate.status in ("accepted", "automated")
    )
    dismissed = sum(
        1 for candidate in delivered_candidates if candidate.status == "dismissed"
    )
    delivered_total = len(delivered_candidates)
    acceptance_rate = accepted / delivered_total if delivered_total else None
    dismissal_rate = dismissed / delivered_total if delivered_total else None

    latencies_hours: list[float] = []
    for candidate in delivered_candidates:
        delivered_at = delivered_at_by_candidate.get(candidate.id)
        if delivered_at is None:
            continue
        action_at = _witness_first_action_at(candidate, after=delivered_at)
        if action_at is None:
            continue
        latencies_hours.append((action_at - delivered_at).total_seconds() / 3600.0)
    time_to_action = _median(latencies_hours)

    accepted_total = int(
        session.scalar(
            select(func.count())
            .select_from(WitnessOpportunityCandidate)
            .where(
                *candidate_scope,
                WitnessOpportunityCandidate.status.in_(("accepted", "automated")),
                WitnessOpportunityCandidate.updated_at >= cutoff,
            )
        )
        or 0
    )
    automated_total = int(
        session.scalar(
            select(func.count())
            .select_from(WitnessOpportunityCandidate)
            .where(
                *candidate_scope,
                WitnessOpportunityCandidate.status == "automated",
                WitnessOpportunityCandidate.updated_at >= cutoff,
            )
        )
        or 0
    )
    conversion = automated_total / accepted_total if accepted_total else None

    decision_counts = tuple(
        WitnessDecisionCount(
            decision=decision,
            label=_WITNESS_DECISION_LABELS[decision],
            count=decision_totals.get(decision, 0),
        )
        for decision in (
            "notify",
            "question",
            "draft",
            "silent",
            "digest",
            "channel_sent",
            "channel_deferred",
            "draft_executed",
        )
    )
    return WitnessKpis(
        window_days=window_days,
        candidates_created=candidates_created,
        trigger_rate_per_day=candidates_created / window_days if window_days else 0.0,
        delivered_count=delivered_count,
        silent_count=silent_count,
        silent_rate=silent_rate,
        acceptance_rate=acceptance_rate,
        dismissal_rate=dismissal_rate,
        time_to_action_median_hours=time_to_action,
        conversion_to_automation=conversion,
        decision_counts=decision_counts,
    )


def _witness_first_action_at(
    candidate: WitnessOpportunityCandidate,
    *,
    after: datetime,
) -> datetime | None:
    feedback = candidate.feedback_json or {}
    history = feedback.get("history")
    if not isinstance(history, list):
        return None
    for entry in history:
        if not isinstance(entry, dict):
            continue
        if entry.get("action") not in ("accepted", "dismissed"):
            continue
        at_value = entry.get("at")
        if not isinstance(at_value, str):
            continue
        try:
            at = datetime.fromisoformat(at_value)
        except ValueError:
            continue
        if at.tzinfo is None:
            at = at.replace(tzinfo=UTC)
        if at >= after:
            return at
    return None


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def get_system_health(
    session: Session,
    *,
    dashboard_settings: DashboardSettings,
    runtime_settings: Settings | None = None,
    runtime_error: str | None = None,
) -> SystemHealth:
    """Return a read-only operator snapshot for setup and system health."""

    total_tasks = session.scalar(select(func.count()).select_from(Task)) or 0
    active_tasks = (
        session.scalar(
            select(func.count())
            .select_from(Task)
            .where(Task.status.in_((TaskStatus.pending, TaskStatus.running)))
        )
        or 0
    )
    failed_tasks = (
        session.scalar(
            select(func.count())
            .select_from(Task)
            .where(Task.status.in_((TaskStatus.failed, TaskStatus.crashed)))
        )
        or 0
    )
    llm_calls = session.scalar(select(func.count()).select_from(LLMUsage)) or 0
    last_task_at = session.scalar(select(func.max(Task.created_at)))

    checks: list[SystemCheck] = [
        SystemCheck(
            group="Core",
            name="Database",
            status="Connected",
            tone="success",
            detail=f"{total_tasks:,} tasks and {llm_calls:,} LLM calls recorded.",
        ),
    ]

    if runtime_settings is None:
        checks.append(
            SystemCheck(
                group="Core",
                name="Runtime settings",
                status="Needs setup",
                tone="danger",
                detail=runtime_error or "Runtime configuration could not be loaded.",
                action="Set the required Slack, LLM, and Postgres environment variables.",
            )
        )
    else:
        checks.extend(_runtime_checks(runtime_settings))

    checks.append(_dashboard_auth_check(dashboard_settings))

    metrics = (
        SystemMetric(
            label="Overall",
            value=_overall_label(checks),
            detail="Worst current setup check.",
            tone=_overall_tone(checks),
        ),
        SystemMetric(
            label="Tasks",
            value=f"{total_tasks:,}",
            detail=f"{active_tasks:,} active",
        ),
        SystemMetric(
            label="Failures",
            value=f"{failed_tasks:,}",
            detail="Failed or crashed tasks",
            tone="danger" if failed_tasks else "neutral",
        ),
        SystemMetric(
            label="Last Task",
            value=_datetime_label(last_task_at),
            detail="Most recent task creation",
        ),
    )

    return SystemHealth(
        overall_label=_overall_label(checks),
        overall_tone=_overall_tone(checks),
        metrics=metrics,
        checks=tuple(checks),
        config_sections=_config_sections(
            dashboard_settings=dashboard_settings,
            runtime_settings=runtime_settings,
            runtime_error=runtime_error,
        ),
    )


def get_integration_dashboard(
    *,
    session: Session | None = None,
    runtime_settings: Settings | None = None,
    runtime_error: str | None = None,
    installation_id: uuid.UUID | None = None,
    owner_slack_user_id: str | None = None,
) -> IntegrationDashboard:
    """Return configured integration and tool-registry state."""

    integrations = _integration_cards(runtime_settings, runtime_error)
    composio_catalog = _composio_catalog_summary_view(
        session=session,
        settings=runtime_settings,
        installation_id=installation_id,
        owner_slack_user_id=owner_slack_user_id,
    )
    tool_groups = _tool_capability_groups(runtime_settings)
    configured_count = sum(1 for card in integrations if card.tone == "success")
    setup_gap_count = sum(
        1 for card in integrations if card.tone in {"warning", "danger"}
    )
    tool_count = sum(len(group.tools) for group in tool_groups)

    metrics = (
        SystemMetric(
            label="Configured",
            value=f"{configured_count:,}",
            detail="Providers ready for this deployment",
            tone="success" if configured_count else "warning",
        ),
        SystemMetric(
            label="Setup Gaps",
            value=f"{setup_gap_count:,}",
            detail="Missing or planned configuration",
            tone="warning" if setup_gap_count else "success",
        ),
        SystemMetric(
            label="Native Tools",
            value=f"{tool_count:,}",
            detail="Tool contracts exposed to the agent loop",
        ),
        SystemMetric(
            label="External Adapters",
            value=f"{composio_catalog.active_connection_count:,} active",
            detail="Composio app catalog lives under Tools",
            tone=composio_catalog.tone,
        ),
    )
    return IntegrationDashboard(
        metrics=metrics,
        integrations=integrations,
        composio_catalog=composio_catalog,
        tool_groups=tool_groups,
        runtime_error=runtime_error,
    )


def get_llm_model_config_dashboard(
    *,
    session: Session,
    runtime_settings: Settings | None = None,
    runtime_error: str | None = None,
    installation_id: uuid.UUID | None = None,
) -> LLMModelConfigDashboard:
    """Return DB-managed LLM provider/model/tier configuration state."""

    resolved_installation_id = installation_id or _single_installation_id(session)
    installation_label = _installation_label(session, resolved_installation_id)
    if resolved_installation_id is None:
        return LLMModelConfigDashboard(
            installation_id=None,
            installation_label="No Slack workspace installed",
            metrics=(
                SystemMetric(
                    label="Providers",
                    value="0",
                    detail="Install Kortny in Slack first",
                    tone="warning",
                ),
                SystemMetric(
                    label="Models",
                    value="0",
                    detail="No model catalog yet",
                    tone="neutral",
                ),
                SystemMetric(
                    label="Assigned Tiers",
                    value="0 / 5",
                    detail="No active tier assignments",
                    tone="warning",
                ),
                SystemMetric(
                    label="Config Source",
                    value="Unavailable",
                    detail="No installation scope",
                    tone="warning",
                ),
            ),
            provider_options=litellm_provider_options(),
            providers=(),
            tiers=_empty_tier_rows(),
            models=(),
            audits=(),
            runtime_error=runtime_error,
            can_bootstrap=False,
            empty_message="Kortny needs a Slack installation before model routing can be configured.",
        )

    providers = tuple(
        session.scalars(
            select(LLMProviderAccount)
            .where(LLMProviderAccount.installation_id == resolved_installation_id)
            .order_by(
                LLMProviderAccount.status.asc(),
                LLMProviderAccount.provider_kind.asc(),
                LLMProviderAccount.display_name.asc(),
            )
        )
    )
    provider_ids = tuple(provider.id for provider in providers)
    models = (
        tuple(
            session.scalars(
                select(LLMModelCatalog)
                .where(LLMModelCatalog.provider_account_id.in_(provider_ids))
                .order_by(
                    LLMModelCatalog.is_enabled.desc(),
                    LLMModelCatalog.display_name.asc(),
                    LLMModelCatalog.model_identifier.asc(),
                )
            )
        )
        if provider_ids
        else ()
    )
    model_ids = tuple(model.id for model in models)
    assignments = (
        tuple(
            session.scalars(
                select(LLMTierAssignment)
                .where(
                    LLMTierAssignment.installation_id == resolved_installation_id,
                    LLMTierAssignment.model_catalog_id.in_(model_ids),
                )
                .order_by(
                    LLMTierAssignment.tier.asc(),
                    LLMTierAssignment.priority.asc(),
                    LLMTierAssignment.created_at.asc(),
                )
            )
        )
        if model_ids
        else ()
    )
    audits = tuple(
        session.scalars(
            select(LLMConfigAudit)
            .where(LLMConfigAudit.installation_id == resolved_installation_id)
            .order_by(LLMConfigAudit.created_at.desc())
            .limit(12)
        )
    )

    provider_by_id = {provider.id: provider for provider in providers}
    model_by_id = {model.id: model for model in models}
    models_by_provider: dict[uuid.UUID, list[LLMModelCatalog]] = defaultdict(list)
    for model in models:
        models_by_provider[model.provider_account_id].append(model)
    assignments_by_model: dict[uuid.UUID, list[LLMTierAssignment]] = defaultdict(list)
    assignments_by_tier: dict[str, list[LLMTierAssignment]] = defaultdict(list)
    for assignment in assignments:
        assignments_by_model[assignment.model_catalog_id].append(assignment)
        assignments_by_tier[assignment.tier].append(assignment)

    pricing_by_model = _latest_llm_pricing_by_model(session, provider_ids)
    provider_rows = tuple(
        LLMProviderConfigRow(
            provider=provider,
            model_count=len(models_by_provider[provider.id]),
            enabled_model_count=sum(
                1 for model in models_by_provider[provider.id] if model.is_enabled
            ),
            tier_count=sum(
                1
                for assignment in assignments
                if assignment.is_active
                and model_by_id.get(assignment.model_catalog_id) is not None
                and model_by_id[assignment.model_catalog_id].provider_account_id
                == provider.id
            ),
            credential_label=_provider_credential_label(provider),
            source_label=_provider_source_label(provider),
            status_tone=_provider_status_tone(provider.status),
            health_tone=_provider_health_tone(provider.health_status),
        )
        for provider in providers
    )
    model_rows = tuple(
        _llm_model_config_row(
            model=model,
            provider=provider_by_id[model.provider_account_id],
            assignments=assignments_by_model.get(model.id, ()),
            latest_pricing=pricing_by_model.get(
                (model.provider_account_id, model.model_identifier)
            ),
        )
        for model in models
        if model.provider_account_id in provider_by_id
    )
    enabled_options = tuple(
        LLMModelConfigOption(
            id=model.id,
            label=model.display_name or model.model_identifier,
            detail=(
                f"{provider_by_id[model.provider_account_id].display_name} / "
                f"{model.model_identifier}"
            ),
            enabled=model.is_enabled
            and provider_by_id[model.provider_account_id].status == "active",
        )
        for model in models
        if model.provider_account_id in provider_by_id
    )
    tier_rows = tuple(
        _tier_config_row(
            tier=tier.value,
            assignments=assignments_by_tier.get(tier.value, ()),
            model_by_id=model_by_id,
            provider_by_id=provider_by_id,
            options=enabled_options,
        )
        for tier in CONFIG_TIERS
    )
    active_tier_count = sum(1 for row in tier_rows if row.primary_assignment)
    config_source = _llm_config_source_label(providers, runtime_settings, runtime_error)
    metrics = (
        SystemMetric(
            label="Providers",
            value=f"{len(providers):,}",
            detail=f"{sum(1 for provider in providers if provider.status == 'active'):,} active",
            tone="success" if providers else "warning",
        ),
        SystemMetric(
            label="Models",
            value=f"{len(models):,}",
            detail=f"{sum(1 for model in models if model.is_enabled):,} enabled",
            tone="success" if models else "warning",
        ),
        SystemMetric(
            label="Assigned Tiers",
            value=f"{active_tier_count:,} / {len(CONFIG_TIERS):,}",
            detail="Primary tier routes configured",
            tone="success" if active_tier_count == len(CONFIG_TIERS) else "warning",
        ),
        SystemMetric(
            label="Config Source",
            value=config_source,
            detail=installation_label,
            tone="success" if providers else "warning",
        ),
    )
    return LLMModelConfigDashboard(
        installation_id=resolved_installation_id,
        installation_label=installation_label,
        metrics=metrics,
        provider_options=litellm_provider_options(),
        providers=provider_rows,
        tiers=tier_rows,
        models=model_rows,
        audits=tuple(_audit_config_row(audit) for audit in audits),
        runtime_error=runtime_error,
        can_bootstrap=runtime_settings is not None,
        empty_message=None
        if providers
        else "No DB-managed model config exists yet. Bootstrap from env to seed the current provider and tier assignments.",
    )


def get_llm_provider_config_detail(
    *,
    session: Session,
    provider_account_id: uuid.UUID,
    installation_id: uuid.UUID | None = None,
) -> LLMProviderConfigDetail | None:
    """Return one LLM provider account with diagnostic model/config detail."""

    provider = session.get(LLMProviderAccount, provider_account_id)
    if provider is None:
        return None
    if installation_id is not None and provider.installation_id != installation_id:
        return None

    installation_label = _installation_label(session, provider.installation_id)
    models = tuple(
        session.scalars(
            select(LLMModelCatalog)
            .where(LLMModelCatalog.provider_account_id == provider.id)
            .order_by(
                LLMModelCatalog.is_enabled.desc(),
                LLMModelCatalog.display_name.asc(),
                LLMModelCatalog.model_identifier.asc(),
            )
        )
    )
    model_ids = tuple(model.id for model in models)
    assignments = (
        tuple(
            session.scalars(
                select(LLMTierAssignment)
                .where(
                    LLMTierAssignment.installation_id == provider.installation_id,
                    LLMTierAssignment.model_catalog_id.in_(model_ids),
                )
                .order_by(
                    LLMTierAssignment.tier.asc(),
                    LLMTierAssignment.priority.asc(),
                    LLMTierAssignment.created_at.asc(),
                )
            )
        )
        if model_ids
        else ()
    )
    model_rows = _llm_provider_model_rows(
        session,
        provider=provider,
        models=models,
    )
    pricing_count = int(
        session.scalar(
            select(func.count())
            .select_from(LLMModelPricing)
            .where(LLMModelPricing.provider_account_id == provider.id)
        )
        or 0
    )
    model_page = get_llm_provider_model_catalog_page(
        session=session,
        provider_account_id=provider.id,
        installation_id=provider.installation_id,
        offset=0,
        limit=MODEL_CATALOG_PAGE_SIZE,
    )
    routed_models = tuple(row for row in model_rows if row.assignment_labels)
    missing_pricing_count = sum(1 for row in model_rows if row.latest_pricing is None)
    attention_models = tuple(
        row
        for row in model_rows
        if row.assignment_labels
        and (row.latest_pricing is None or not row.model.is_enabled)
    )
    if not attention_models:
        attention_models = tuple(
            row
            for row in model_rows
            if row.model.is_enabled and row.latest_pricing is None
        )[:5]
    active_assignment_count = sum(
        1 for assignment in assignments if assignment.is_active
    )
    provider_row = LLMProviderConfigRow(
        provider=provider,
        model_count=len(models),
        enabled_model_count=sum(1 for model in models if model.is_enabled),
        tier_count=active_assignment_count,
        credential_label=_provider_credential_label(provider),
        source_label=_provider_source_label(provider),
        status_tone=_provider_status_tone(provider.status),
        health_tone=_provider_health_tone(provider.health_status),
    )

    audit_entity_ids = {
        str(provider.id),
        *(str(model.id) for model in models),
        *(str(assignment.id) for assignment in assignments),
    }
    audits = tuple(
        session.scalars(
            select(LLMConfigAudit)
            .where(
                LLMConfigAudit.installation_id == provider.installation_id,
                LLMConfigAudit.entity_id.in_(tuple(audit_entity_ids)),
            )
            .order_by(LLMConfigAudit.created_at.desc())
            .limit(12)
        )
    )
    metadata = (
        provider.metadata_json if isinstance(provider.metadata_json, dict) else {}
    )
    metrics = (
        SystemMetric(
            label="Status",
            value=provider.status.title(),
            detail="Provider runtime availability",
            tone=provider_row.status_tone,
        ),
        SystemMetric(
            label="Health",
            value=provider.health_status.title(),
            detail="Latest dashboard credential test",
            tone=provider_row.health_tone,
        ),
        SystemMetric(
            label="Models",
            value=f"{len(models):,}",
            detail=f"{provider_row.enabled_model_count:,} enabled, {pricing_count:,} priced",
            tone="success" if models else "warning",
        ),
        SystemMetric(
            label="Tier Routes",
            value=f"{active_assignment_count:,}",
            detail="Active assignments using this provider",
            tone="success" if active_assignment_count else "neutral",
        ),
    )
    return LLMProviderConfigDetail(
        installation_id=provider.installation_id,
        installation_label=installation_label,
        provider=provider,
        provider_row=provider_row,
        models=model_page.rows if model_page is not None else (),
        routed_models=routed_models,
        attention_models=attention_models[:5],
        missing_pricing_count=missing_pricing_count,
        model_total_count=model_page.total_count if model_page is not None else 0,
        model_page_size=model_page.limit
        if model_page is not None
        else MODEL_CATALOG_PAGE_SIZE,
        model_next_offset=model_page.next_offset if model_page is not None else None,
        model_has_more=model_page.has_more if model_page is not None else False,
        audits=tuple(_audit_config_row(audit) for audit in audits),
        metrics=metrics,
        api_version_label=str(metadata.get("api_version") or "Not configured"),
        base_url_label=provider.base_url or "Default LiteLLM endpoint",
        tier_options=_tier_catalog_options(),
    )


def get_llm_provider_model_catalog_page(
    *,
    session: Session,
    provider_account_id: uuid.UUID,
    installation_id: uuid.UUID | None = None,
    offset: int = 0,
    limit: int = MODEL_CATALOG_PAGE_SIZE,
    query: str | None = None,
) -> LLMProviderModelCatalogPage | None:
    """Return a lazy-load page of provider model catalog rows."""

    provider = session.get(LLMProviderAccount, provider_account_id)
    if provider is None:
        return None
    if installation_id is not None and provider.installation_id != installation_id:
        return None

    normalized_offset = max(offset, 0)
    normalized_limit = min(max(limit, 1), MAX_PAGE_SIZE)
    cleaned_query = (query or "").strip()
    model_stmt = select(LLMModelCatalog).where(
        LLMModelCatalog.provider_account_id == provider.id
    )
    if cleaned_query:
        pattern = f"%{cleaned_query}%"
        model_stmt = model_stmt.where(
            or_(
                LLMModelCatalog.display_name.ilike(pattern),
                LLMModelCatalog.model_identifier.ilike(pattern),
                LLMModelCatalog.source.ilike(pattern),
            )
        )

    total_count = int(
        session.scalar(
            select(func.count()).select_from(model_stmt.order_by(None).subquery())
        )
        or 0
    )
    models = tuple(
        session.scalars(
            model_stmt.order_by(
                LLMModelCatalog.is_enabled.desc(),
                LLMModelCatalog.display_name.asc(),
                LLMModelCatalog.model_identifier.asc(),
            )
            .offset(normalized_offset)
            .limit(normalized_limit)
        )
    )
    next_offset = normalized_offset + len(models)
    has_more = next_offset < total_count
    return LLMProviderModelCatalogPage(
        rows=_llm_provider_model_rows(
            session,
            provider=provider,
            models=models,
        ),
        total_count=total_count,
        offset=normalized_offset,
        limit=normalized_limit,
        next_offset=next_offset if has_more else None,
        has_more=has_more,
    )


def _llm_provider_model_rows(
    session: Session,
    *,
    provider: LLMProviderAccount,
    models: Sequence[LLMModelCatalog],
) -> tuple[LLMModelConfigRow, ...]:
    model_ids = tuple(model.id for model in models)
    assignments = (
        tuple(
            session.scalars(
                select(LLMTierAssignment)
                .where(
                    LLMTierAssignment.installation_id == provider.installation_id,
                    LLMTierAssignment.model_catalog_id.in_(model_ids),
                )
                .order_by(
                    LLMTierAssignment.tier.asc(),
                    LLMTierAssignment.priority.asc(),
                    LLMTierAssignment.created_at.asc(),
                )
            )
        )
        if model_ids
        else ()
    )
    assignments_by_model: dict[uuid.UUID, list[LLMTierAssignment]] = defaultdict(list)
    for assignment in assignments:
        assignments_by_model[assignment.model_catalog_id].append(assignment)
    pricing_by_model = _latest_llm_pricing_by_model(session, (provider.id,))
    return tuple(
        _llm_model_config_row(
            model=model,
            provider=provider,
            assignments=assignments_by_model.get(model.id, ()),
            latest_pricing=pricing_by_model.get((provider.id, model.model_identifier)),
        )
        for model in models
    )


def get_composio_catalog_dashboard(
    session: Session,
    *,
    runtime_settings: Settings | None = None,
    query: str | None = None,
    cursor: str | None = None,
    limit: int | None = None,
    composio_client: ComposioClient | None = None,
    installation_id: uuid.UUID | None = None,
    owner_slack_user_id: str | None = None,
) -> ComposioCatalogView:
    """Return the Composio catalog view for the dedicated management page."""

    return _composio_catalog_view(
        session=session,
        settings=runtime_settings,
        query=query,
        cursor=cursor,
        limit=limit,
        client=composio_client,
        installation_id=installation_id,
        owner_slack_user_id=owner_slack_user_id,
    )


def get_composio_toolkit_detail(
    session: Session,
    *,
    slug: str,
    runtime_settings: Settings | None = None,
    composio_client: ComposioClient | None = None,
    installation_id: uuid.UUID | None = None,
    owner_slack_user_id: str | None = None,
) -> ComposioToolkitDetail:
    """Return one Composio toolkit and local scoped connection metadata."""

    normalized_slug = slug.strip().lower()
    connections = tuple(
        connection
        for connection in _composio_connection_rows(
            session,
            installation_id=installation_id,
            owner_slack_user_id=owner_slack_user_id,
        )
        if connection.toolkit_slug == normalized_slug
    )
    user_options = _slack_identity_options(session, kind="user")
    channel_options = _slack_identity_options(session, kind="channel")
    if runtime_settings is None or not runtime_settings.composio_api_key:
        return ComposioToolkitDetail(
            slug=normalized_slug,
            configured=False,
            status="Not configured",
            tone="neutral",
            toolkit=None,
            raw_toolkit=None,
            auth_configs=(),
            tools=(),
            tools_error=None,
            connections=connections,
            scope_options=_composio_scope_options(),
            user_options=user_options,
            channel_options=channel_options,
            error=None,
        )
    if not runtime_settings.composio_catalog_enabled:
        return ComposioToolkitDetail(
            slug=normalized_slug,
            configured=True,
            status="Catalog disabled",
            tone="warning",
            toolkit=None,
            raw_toolkit=None,
            auth_configs=(),
            tools=(),
            tools_error=None,
            connections=connections,
            scope_options=_composio_scope_options(),
            user_options=user_options,
            channel_options=channel_options,
            error="COMPOSIO_CATALOG_ENABLED is false.",
        )

    client = composio_client or ComposioClient(
        api_key=runtime_settings.composio_api_key,
        timeout_seconds=runtime_settings.composio_request_timeout_seconds,
    )
    try:
        toolkit = client.get_toolkit(normalized_slug)
    except ComposioCatalogError as exc:
        return ComposioToolkitDetail(
            slug=normalized_slug,
            configured=True,
            status="Unavailable",
            tone="danger",
            toolkit=None,
            raw_toolkit=None,
            auth_configs=(),
            tools=(),
            tools_error=None,
            connections=connections,
            scope_options=_composio_scope_options(),
            user_options=user_options,
            channel_options=channel_options,
            error=_short_error(str(exc)),
        )
    auth_configs: tuple[ComposioAuthConfigRow, ...] = ()
    auth_config_error = None
    try:
        auth_configs = tuple(
            _composio_auth_config_row(auth_config)
            for auth_config in client.list_auth_configs(toolkit_slug=normalized_slug)
        )
    except (ComposioCatalogError, ComposioConnectionError) as exc:
        auth_config_error = f"Auth configs unavailable: {_short_error(str(exc))}"
    tools: tuple[ComposioToolRow, ...] = ()
    tools_error: str | None = None
    try:
        tools = tuple(
            _composio_tool_row(tool)
            for tool in client.list_tools(
                toolkit_slug=normalized_slug,
                limit=COMPOSIO_DETAIL_TOOL_LIMIT,
            )
        )
    except ComposioCatalogError as exc:
        tools_error = f"Tools unavailable: {_short_error(str(exc))}"

    status_map = _composio_status_by_toolkit(connections)
    return ComposioToolkitDetail(
        slug=normalized_slug,
        configured=True,
        status="Connected" if status_map.get(toolkit.slug) == "active" else "Available",
        tone="success" if status_map.get(toolkit.slug) == "active" else "neutral",
        toolkit=_composio_toolkit_row(toolkit, status_map),
        raw_toolkit=toolkit,
        auth_configs=auth_configs,
        tools=tools,
        tools_error=tools_error,
        connections=connections,
        scope_options=_composio_scope_options(),
        user_options=user_options,
        channel_options=channel_options,
        error=auth_config_error,
    )


def parse_date_bound(
    value: str | None, *, inclusive_end: bool = False
) -> datetime | None:
    """Parse dashboard date filters.

    Date-only upper bounds are treated as inclusive by moving to the next day.
    """

    if value is None or value.strip() == "":
        return None
    stripped = value.strip()
    parsed_date = date.fromisoformat(stripped)
    parsed = datetime.combine(parsed_date, time.min, tzinfo=UTC)
    if inclusive_end:
        return parsed + timedelta(days=1)
    return parsed


def _runtime_checks(settings: Settings) -> tuple[SystemCheck, ...]:
    model_tier_count = len(
        tuple(
            model
            for model in (
                settings.llm_cheap_model,
                settings.llm_standard_model,
                settings.llm_analysis_model,
                settings.llm_document_model,
                settings.llm_high_reasoning_model,
                settings.llm_humanizer_model,
            )
            if model
        )
    )
    return (
        SystemCheck(
            group="Core",
            name="Slack app",
            status="Configured",
            tone="success",
            detail=f"Socket mode credentials are present for app name {settings.slack_app_name!r}.",
        ),
        SystemCheck(
            group="Core",
            name="LLM provider",
            status="Configured",
            tone="success",
            detail=f"{settings.llm_provider.value} using {settings.llm_model}.",
        ),
        SystemCheck(
            group="Models",
            name="Model routing",
            status="Specialized" if model_tier_count else "Fallback only",
            tone="success" if model_tier_count else "warning",
            detail=(
                f"{model_tier_count:,} specialized model tiers configured."
                if model_tier_count
                else "All model tiers fall back to LLM_MODEL."
            ),
            action=(
                None
                if model_tier_count
                else "Set LLM_CHEAP_MODEL, LLM_ANALYSIS_MODEL, or document/high reasoning tiers."
            ),
        ),
        SystemCheck(
            group="Tools",
            name="Web search",
            status="Configured" if settings.brave_search_api_key else "Unavailable",
            tone="success" if settings.brave_search_api_key else "warning",
            detail=(
                "Brave Search API key is present."
                if settings.brave_search_api_key
                else "Web search tool will fail until BRAVE_SEARCH_API_KEY is set."
            ),
        ),
        SystemCheck(
            group="Observability",
            name="Tracing export",
            status=_observability_status(settings),
            tone=_observability_tone(settings),
            detail=_observability_detail(settings),
            action=(
                None
                if settings.otel_exporter_otlp_endpoint
                else "Run the observability profile or configure an OTLP endpoint for external traces."
            ),
        ),
    )


def _dashboard_auth_check(settings: DashboardSettings) -> SystemCheck:
    default_password = settings.password == "change-me"
    default_secret = settings.session_secret == "change-me-dashboard-session-secret"
    if default_password or default_secret:
        return SystemCheck(
            group="Dashboard",
            name="Dashboard auth",
            status="Needs hardening",
            tone="danger",
            detail="Default dashboard credentials or session secret are still in use.",
            action="Set DASHBOARD_PASSWORD and DASHBOARD_SESSION_SECRET before exposing the dashboard.",
        )
    if not settings.secure_cookies:
        return SystemCheck(
            group="Dashboard",
            name="Dashboard auth",
            status="Local mode",
            tone="warning",
            detail="Secure cookies are disabled, which is acceptable for local HTTP only.",
            action="Enable DASHBOARD_SECURE_COOKIES when serving over HTTPS.",
        )
    return SystemCheck(
        group="Dashboard",
        name="Dashboard auth",
        status="Hardened",
        tone="success",
        detail="Custom credentials and secure cookies are configured.",
    )


def _config_sections(
    *,
    dashboard_settings: DashboardSettings,
    runtime_settings: Settings | None,
    runtime_error: str | None,
) -> tuple[SystemConfigSection, ...]:
    dashboard_rows = (
        SystemConfigRow("Dashboard user", dashboard_settings.username),
        SystemConfigRow(
            "Dashboard password",
            "Default" if dashboard_settings.password == "change-me" else "Custom",
            tone=(
                "danger" if dashboard_settings.password == "change-me" else "success"
            ),
        ),
        SystemConfigRow(
            "Session secret",
            (
                "Default"
                if dashboard_settings.session_secret
                == "change-me-dashboard-session-secret"
                else "Custom"
            ),
            tone=(
                "danger"
                if dashboard_settings.session_secret
                == "change-me-dashboard-session-secret"
                else "success"
            ),
        ),
        SystemConfigRow(
            "Secure cookies",
            "Enabled" if dashboard_settings.secure_cookies else "Disabled",
            detail="Use enabled when served over HTTPS.",
            tone="success" if dashboard_settings.secure_cookies else "warning",
        ),
        SystemConfigRow(
            "Postgres URL",
            _redact_url(dashboard_settings.postgres_url),
            detail="Password is always hidden.",
        ),
    )

    sections: list[SystemConfigSection] = [
        SystemConfigSection("Dashboard", dashboard_rows),
    ]

    if runtime_settings is None:
        sections.append(
            SystemConfigSection(
                "Runtime",
                (
                    SystemConfigRow(
                        "Configuration",
                        "Invalid",
                        detail=runtime_error or "Runtime settings could not load.",
                        tone="danger",
                    ),
                ),
            )
        )
        return tuple(sections)

    sections.extend(
        (
            SystemConfigSection(
                "Runtime",
                (
                    SystemConfigRow(
                        "App Postgres URL",
                        _redact_url(runtime_settings.postgres_url),
                        detail="Runtime database target with password hidden.",
                    ),
                    SystemConfigRow(
                        "Release",
                        runtime_settings.kortny_release
                        or runtime_settings.kortny_version
                        or "Not set",
                    ),
                ),
            ),
            SystemConfigSection(
                "Slack",
                (
                    SystemConfigRow("App name", runtime_settings.slack_app_name),
                    SystemConfigRow("Bot token", "Configured", tone="success"),
                    SystemConfigRow("Socket app token", "Configured", tone="success"),
                    SystemConfigRow("Signing secret", "Configured", tone="success"),
                    SystemConfigRow(
                        "File read limit",
                        f"{runtime_settings.slack_file_read_max_bytes:,} bytes",
                    ),
                ),
            ),
            SystemConfigSection(
                "Models",
                (
                    SystemConfigRow("Provider", runtime_settings.llm_provider.value),
                    SystemConfigRow("Default model", runtime_settings.llm_model),
                    _model_row("Cheap model", runtime_settings.llm_cheap_model),
                    _model_row("Standard model", runtime_settings.llm_standard_model),
                    _model_row("Analysis model", runtime_settings.llm_analysis_model),
                    _model_row("Document model", runtime_settings.llm_document_model),
                    _model_row(
                        "High reasoning model",
                        runtime_settings.llm_high_reasoning_model,
                    ),
                    _model_row(
                        "Humanizer model",
                        runtime_settings.llm_humanizer_model,
                    ),
                ),
            ),
            SystemConfigSection(
                "Tools",
                (
                    SystemConfigRow(
                        "Brave Search",
                        (
                            "Configured"
                            if runtime_settings.brave_search_api_key
                            else "Missing"
                        ),
                        tone=(
                            "success"
                            if runtime_settings.brave_search_api_key
                            else "warning"
                        ),
                    ),
                    SystemConfigRow(
                        "Composio",
                        (
                            "Configured"
                            if runtime_settings.composio_api_key
                            else "Not configured"
                        ),
                        tone=(
                            "success"
                            if runtime_settings.composio_api_key
                            else "neutral"
                        ),
                    ),
                ),
            ),
            SystemConfigSection(
                "Observability",
                (
                    SystemConfigRow(
                        "Enabled",
                        "Yes" if runtime_settings.observability_enabled else "No",
                        tone=(
                            "success"
                            if runtime_settings.observability_enabled
                            else "warning"
                        ),
                    ),
                    SystemConfigRow(
                        "Capture mode",
                        runtime_settings.observability_capture_content,
                    ),
                    SystemConfigRow(
                        "OTLP endpoint",
                        runtime_settings.otel_exporter_otlp_endpoint
                        or "Not configured",
                        tone=(
                            "success"
                            if runtime_settings.otel_exporter_otlp_endpoint
                            else "warning"
                        ),
                    ),
                    SystemConfigRow(
                        "Trace sampling",
                        f"{runtime_settings.otel_trace_sampling_ratio:.2f}",
                    ),
                ),
            ),
        )
    )
    return tuple(sections)


def _integration_cards(
    settings: Settings | None,
    runtime_error: str | None,
) -> tuple[IntegrationCard, ...]:
    if settings is None:
        return (
            IntegrationCard(
                name="Runtime configuration",
                category="Core",
                status="Needs setup",
                tone="danger",
                description="Kortny cannot load runtime settings for integrations.",
                details=(
                    runtime_error or "Required environment variables are missing.",
                    "Set Slack, LLM, and Postgres values before checking tools.",
                ),
                env_vars=("SLACK_BOT_TOKEN", "LLM_API_KEY", "POSTGRES_URL"),
                action="Open System for the redacted configuration error.",
            ),
            IntegrationCard(
                name="Native tool registry",
                category="Tools",
                status="Blocked",
                tone="warning",
                description="Native tools depend on a valid runtime configuration.",
                details=(
                    "Tool metadata is visible, but runtime invocation is blocked.",
                ),
                env_vars=(),
            ),
        )

    model_tiers = tuple(
        model
        for model in (
            settings.llm_cheap_model,
            settings.llm_standard_model,
            settings.llm_analysis_model,
            settings.llm_document_model,
            settings.llm_high_reasoning_model,
            settings.llm_humanizer_model,
        )
        if model
    )
    integrations = [
        IntegrationCard(
            name="Slack workspace",
            category="Transport",
            status="Configured",
            tone="success",
            description="Socket Mode transport for DMs, mentions, reactions, files, and channel context.",
            details=(
                f"App name: {settings.slack_app_name}",
                f"File read limit: {settings.slack_file_read_max_bytes:,} bytes",
                "Bot, app, and signing credentials are present.",
            ),
            env_vars=(
                "SLACK_BOT_TOKEN",
                "SLACK_APP_TOKEN",
                "SLACK_SIGNING_SECRET",
                "SLACK_APP_NAME",
            ),
        ),
        IntegrationCard(
            name="LLM provider",
            category="Inference",
            status="Configured",
            tone="success",
            description="Primary inference backend used by the coordinator, intent classifier, and model router.",
            details=(
                f"Provider: {settings.llm_provider.value}",
                f"Default model: {settings.llm_model}",
                (
                    f"{len(model_tiers):,} specialized routing tiers configured."
                    if model_tiers
                    else "Specialized tiers fall back to LLM_MODEL."
                ),
            ),
            env_vars=(
                "LLM_PROVIDER",
                "LLM_API_KEY",
                "LLM_MODEL",
                "LLM_CHEAP_MODEL",
                "LLM_STANDARD_MODEL",
                "LLM_ANALYSIS_MODEL",
                "LLM_DOCUMENT_MODEL",
                "LLM_HIGH_REASONING_MODEL",
            ),
            action=(
                None
                if model_tiers
                else "Set tier-specific model env vars to make routing explicit."
            ),
        ),
        IntegrationCard(
            name="Brave Search",
            category="Research",
            status="Configured" if settings.brave_search_api_key else "Missing",
            tone="success" if settings.brave_search_api_key else "warning",
            description="Public web search provider used by the native web_search tool.",
            details=(
                (
                    "API key is present. The tool still respects Brave API rate limits."
                    if settings.brave_search_api_key
                    else "The web_search tool needs BRAVE_SEARCH_API_KEY before it can run."
                ),
            ),
            env_vars=("BRAVE_SEARCH_API_KEY",),
            action=(
                None
                if settings.brave_search_api_key
                else "Add BRAVE_SEARCH_API_KEY to enable web research."
            ),
        ),
        IntegrationCard(
            name="PDF generation",
            category="Documents",
            status="Built in",
            tone="success",
            description="ReportLab-backed document generation running inside the worker container.",
            details=(
                "No external account required.",
                "Uses task workspace storage and records generated artifacts.",
            ),
            env_vars=(),
        ),
        IntegrationCard(
            name="Workspace memory",
            category="Memory",
            status="Available",
            tone="success",
            description="Confirm-gated workspace_state memory tools backed by Postgres.",
            details=(
                "Facts, proposals, supersession, and forget events are stored with audit metadata.",
                "Episodic recall is recorded separately from durable facts.",
            ),
            env_vars=("POSTGRES_URL",),
        ),
        IntegrationCard(
            name="Observability",
            category="Operations",
            status=_integration_observability_status(settings),
            tone=_integration_observability_tone(settings),
            description="Structured logs, task events, LLM usage, and optional OTLP export.",
            details=(
                f"Capture mode: {settings.observability_capture_content}",
                (
                    f"OTLP endpoint: {settings.otel_exporter_otlp_endpoint}"
                    if settings.otel_exporter_otlp_endpoint
                    else "No OTLP endpoint configured; dashboard still reads local task and usage rows."
                ),
                f"Trace sampling: {settings.otel_trace_sampling_ratio:.2f}",
            ),
            env_vars=(
                "OBSERVABILITY_ENABLED",
                "OBSERVABILITY_CAPTURE_CONTENT",
                "OTEL_EXPORTER_OTLP_ENDPOINT",
                "OTEL_TRACE_SAMPLING_RATIO",
            ),
            action=(
                None
                if settings.otel_exporter_otlp_endpoint
                else "Run the observability profile or connect an OTLP endpoint for external traces."
            ),
        ),
        IntegrationCard(
            name="Langfuse",
            category="Prompts",
            status="Enabled" if settings.langfuse_enabled else "Optional",
            tone="success" if settings.langfuse_enabled else "neutral",
            description="Optional hosted prompt and trace backend for teams that want cloud prompt management.",
            details=(
                (
                    f"Host: {settings.langfuse_host or 'not set'}"
                    if settings.langfuse_enabled
                    else "Not required for local self-hosting."
                ),
                (
                    "Prompt fetching is enabled."
                    if settings.langfuse_prompts_enabled
                    else "Prompt fetching is disabled."
                ),
            ),
            env_vars=(
                "LANGFUSE_ENABLED",
                "LANGFUSE_HOST",
                "LANGFUSE_PUBLIC_KEY",
                "LANGFUSE_SECRET_KEY",
                "LANGFUSE_PROMPTS_ENABLED",
            ),
        ),
        IntegrationCard(
            name="Composio",
            category="External tools",
            status="Key present, catalog available"
            if settings.composio_api_key
            else "Planned",
            tone="warning" if settings.composio_api_key else "neutral",
            description="Third-party app catalog and scoped connected-account adapter.",
            details=(
                (
                    "COMPOSIO_API_KEY is present; HIG-35 is wiring catalog and scoped-account metadata before runtime tool use."
                    if settings.composio_api_key
                    else "No key configured. HIG-35 tracks the actual integration adapter."
                ),
                "Runtime tool execution stays disabled until per-task visibility gates are in place.",
            ),
            env_vars=(
                "COMPOSIO_API_KEY",
                "COMPOSIO_CATALOG_ENABLED",
                "COMPOSIO_CATALOG_LIMIT",
            ),
            action="Catalog is read-only first; OAuth connect and runtime execution are follow-up slices.",
        ),
    ]
    return tuple(integrations)


def _composio_catalog_view(
    *,
    session: Session | None,
    settings: Settings | None,
    query: str | None,
    cursor: str | None = None,
    limit: int | None = None,
    client: ComposioClient | None = None,
    installation_id: uuid.UUID | None = None,
    owner_slack_user_id: str | None = None,
) -> ComposioCatalogView:
    normalized_query = (query or "").strip()
    normalized_cursor = (cursor or "").strip()
    default_limit = settings.composio_catalog_limit if settings is not None else 60
    page_size = max(1, min(limit or default_limit, 100))
    connections = _composio_connection_rows(
        session,
        installation_id=installation_id,
        owner_slack_user_id=owner_slack_user_id,
    )
    active_count = sum(1 for connection in connections if connection.status == "active")
    active_toolkit_slugs = _active_composio_toolkit_slugs(connections)
    connected_toolkit_count = len(active_toolkit_slugs)
    if settings is None or not settings.composio_api_key:
        return ComposioCatalogView(
            enabled=False,
            configured=False,
            status="Not configured",
            tone="neutral",
            query=normalized_query,
            cursor=normalized_cursor,
            page_size=page_size,
            total_items=None,
            visible_count=0,
            next_cursor=None,
            connection_count=len(connections),
            active_connection_count=active_count,
            connected_toolkit_count=connected_toolkit_count,
            pinned_connected_count=0,
            error=None,
            toolkits=(),
            connections=connections,
        )
    if not settings.composio_catalog_enabled:
        return ComposioCatalogView(
            enabled=False,
            configured=True,
            status="Catalog disabled",
            tone="warning",
            query=normalized_query,
            cursor=normalized_cursor,
            page_size=page_size,
            total_items=None,
            visible_count=0,
            next_cursor=None,
            connection_count=len(connections),
            active_connection_count=active_count,
            connected_toolkit_count=connected_toolkit_count,
            pinned_connected_count=0,
            error="COMPOSIO_CATALOG_ENABLED is false.",
            toolkits=(),
            connections=connections,
        )

    resolved_client = client or ComposioClient(
        api_key=settings.composio_api_key,
        timeout_seconds=settings.composio_request_timeout_seconds,
    )
    try:
        catalog = resolved_client.list_toolkits(
            search=normalized_query or None,
            limit=page_size,
            cursor=normalized_cursor or None,
        )
    except ComposioCatalogError as exc:
        return ComposioCatalogView(
            enabled=True,
            configured=True,
            status="Catalog unavailable",
            tone="danger",
            query=normalized_query,
            cursor=normalized_cursor,
            page_size=page_size,
            total_items=None,
            visible_count=0,
            next_cursor=None,
            connection_count=len(connections),
            active_connection_count=active_count,
            connected_toolkit_count=connected_toolkit_count,
            pinned_connected_count=0,
            error=_short_error(str(exc)),
            toolkits=(),
            connections=connections,
        )

    connection_statuses = _composio_status_by_toolkit(connections)
    catalog_items, pinned_connected_count = _composio_catalog_items_with_connected(
        catalog_items=tuple(
            item[1]
            for item in sorted(
                enumerate(catalog.items),
                key=lambda item: (
                    connection_statuses.get(item[1].slug.lower()) != "active",
                    item[0],
                ),
            )
        ),
        active_toolkit_slugs=active_toolkit_slugs,
        connections=connections,
        client=resolved_client,
        query=normalized_query,
    )
    toolkits = tuple(
        _composio_toolkit_row(toolkit, connection_statuses) for toolkit in catalog_items
    )
    return ComposioCatalogView(
        enabled=True,
        configured=True,
        status="Catalog synced",
        tone="success",
        query=normalized_query,
        cursor=normalized_cursor,
        page_size=page_size,
        total_items=catalog.total_items,
        visible_count=len(toolkits),
        next_cursor=catalog.next_cursor,
        connection_count=len(connections),
        active_connection_count=active_count,
        connected_toolkit_count=connected_toolkit_count,
        pinned_connected_count=pinned_connected_count,
        error=None,
        toolkits=toolkits,
        connections=connections,
    )


def _composio_catalog_summary_view(
    *,
    session: Session | None,
    settings: Settings | None,
    installation_id: uuid.UUID | None = None,
    owner_slack_user_id: str | None = None,
) -> ComposioCatalogView:
    connections = _composio_connection_rows(
        session,
        installation_id=installation_id,
        owner_slack_user_id=owner_slack_user_id,
    )
    active_count = sum(1 for connection in connections if connection.status == "active")
    active_toolkit_slugs = _active_composio_toolkit_slugs(connections)
    configured = settings is not None and bool(settings.composio_api_key)
    enabled = configured and settings is not None and settings.composio_catalog_enabled
    if enabled:
        status = "Open catalog"
    elif configured:
        status = "Catalog disabled"
    else:
        status = "Not configured"
    return ComposioCatalogView(
        enabled=enabled,
        configured=configured,
        status=status,
        tone="success" if active_count else ("neutral" if enabled else "warning"),
        query="",
        cursor="",
        page_size=0,
        total_items=None,
        visible_count=0,
        next_cursor=None,
        connection_count=len(connections),
        active_connection_count=active_count,
        connected_toolkit_count=len(active_toolkit_slugs),
        pinned_connected_count=0,
        error=None,
        toolkits=(),
        connections=connections,
    )


def _composio_connection_rows(
    session: Session | None,
    *,
    installation_id: uuid.UUID | None = None,
    owner_slack_user_id: str | None = None,
) -> tuple[ComposioConnectionRow, ...]:
    if session is None:
        return ()
    filters: list[ColumnElement[bool]] = []
    if installation_id is not None:
        filters.append(ComposioConnection.installation_id == installation_id)
    if owner_slack_user_id:
        filters.append(ComposioConnection.owner_slack_user_id == owner_slack_user_id)
    rows = tuple(
        session.scalars(
            select(ComposioConnection)
            .where(*filters)
            .order_by(
                ComposioConnection.updated_at.desc(),
                ComposioConnection.id.desc(),
            )
        )
    )
    identity_keys: list[IdentityKey] = []
    for row in rows:
        identity_keys.append((row.installation_id, "user", row.owner_slack_user_id))
        if row.visibility_scope_type == "channel" and row.visibility_scope_id:
            identity_keys.append(
                (row.installation_id, "channel", row.visibility_scope_id)
            )
        elif row.visibility_scope_type == "user" and row.visibility_scope_id:
            identity_keys.append((row.installation_id, "user", row.visibility_scope_id))
    identities = _identity_map_from_keys(session, identity_keys)
    return tuple(_composio_connection_row(row, identities) for row in rows)


def _composio_connection_row(
    row: ComposioConnection,
    identities: dict[IdentityKey, SlackIdentity],
) -> ComposioConnectionRow:
    owner = _identity_label(
        identities,
        installation_id=row.installation_id,
        kind="user",
        slack_id=row.owner_slack_user_id,
    )
    return ComposioConnectionRow(
        id=row.id,
        toolkit_slug=row.toolkit_slug,
        status=row.status,
        tone=_composio_connection_tone(row.status),
        display_name=row.display_name or row.external_account_label or row.toolkit_slug,
        scope_label=_composio_scope_label(row, identities),
        visibility_scope_type=row.visibility_scope_type,
        visibility_scope_id=row.visibility_scope_id,
        owner=owner,
        connected_account_id=row.connected_account_id,
        auth_config_id=row.auth_config_id,
        updated_at=row.updated_at,
    )


def _composio_auth_config_row(auth_config: ComposioAuthConfig) -> ComposioAuthConfigRow:
    return ComposioAuthConfigRow(
        id=auth_config.id,
        name=auth_config.name,
        toolkit_slug=auth_config.toolkit_slug,
        auth_scheme=auth_config.auth_scheme,
        is_composio_managed=auth_config.is_composio_managed,
        enabled=auth_config.enabled,
    )


def _slack_identity_options(
    session: Session,
    *,
    kind: str,
) -> tuple[IdentityLabel, ...]:
    rows = tuple(
        session.scalars(
            select(SlackIdentity)
            .where(SlackIdentity.kind == kind)
            .order_by(
                SlackIdentity.display_name.asc(),
                SlackIdentity.slack_id.asc(),
                SlackIdentity.updated_at.desc(),
            )
        )
    )
    options: list[IdentityLabel] = []
    seen: set[str] = set()
    for row in rows:
        if row.slack_id in seen:
            continue
        seen.add(row.slack_id)
        options.append(
            IdentityLabel(
                name=row.display_name or row.raw_name or row.slack_id,
                slack_id=row.slack_id,
                found=True,
            )
        )
    return tuple(options)


def _composio_scope_label(
    row: ComposioConnection,
    identities: dict[IdentityKey, SlackIdentity],
) -> str:
    if row.visibility_scope_type == "workspace":
        return "Workspace"
    if row.visibility_scope_id is None:
        return row.visibility_scope_type.title()
    kind = "channel" if row.visibility_scope_type == "channel" else "user"
    label = _identity_label(
        identities,
        installation_id=row.installation_id,
        kind=kind,
        slack_id=row.visibility_scope_id,
    )
    return label.name


def _composio_status_by_toolkit(
    connections: tuple[ComposioConnectionRow, ...],
) -> dict[str, str]:
    statuses: dict[str, str] = {}
    priority = {"active": 4, "pending": 3, "expired": 2, "failed": 1, "disabled": 0}
    for connection in connections:
        slug = connection.toolkit_slug.strip().lower()
        current = statuses.get(slug)
        if current is None or priority.get(connection.status, -1) > priority.get(
            current, -1
        ):
            statuses[slug] = connection.status
    return statuses


def _active_composio_toolkit_slugs(
    connections: tuple[ComposioConnectionRow, ...],
) -> tuple[str, ...]:
    slugs: dict[str, None] = {}
    for connection in connections:
        slug = connection.toolkit_slug.strip().lower()
        if connection.status == "active" and slug:
            slugs.setdefault(slug, None)
    return tuple(slugs)


def _composio_catalog_items_with_connected(
    *,
    catalog_items: tuple[ComposioToolkit, ...],
    active_toolkit_slugs: tuple[str, ...],
    connections: tuple[ComposioConnectionRow, ...],
    client: ComposioClient,
    query: str,
) -> tuple[tuple[ComposioToolkit, ...], int]:
    page_slugs = {toolkit.slug.lower() for toolkit in catalog_items}
    connection_by_slug = {
        connection.toolkit_slug.lower(): connection
        for connection in connections
        if connection.status == "active"
    }
    pinned: list[ComposioToolkit] = []
    for slug in active_toolkit_slugs:
        if slug in page_slugs:
            continue
        try:
            toolkit = client.get_toolkit(slug)
        except ComposioCatalogError:
            toolkit = _composio_toolkit_placeholder(
                slug,
                connection_by_slug.get(slug),
            )
        if not _composio_toolkit_matches_query(toolkit, query):
            continue
        pinned.append(toolkit)
        page_slugs.add(slug)
    return (*pinned, *catalog_items), len(pinned)


def _composio_toolkit_placeholder(
    slug: str,
    connection: ComposioConnectionRow | None,
) -> ComposioToolkit:
    name = (
        connection.display_name
        if connection is not None
        else _humanize_composio_slug(slug)
    )
    return ComposioToolkit(
        slug=slug,
        name=name,
        description="Connected account. Catalog details were not available from Composio.",
        categories=(),
        auth_schemes=(),
        managed_auth_schemes=(),
        tools_count=0,
        triggers_count=0,
        logo_url=None,
        app_url=None,
        auth_guide_url=None,
        base_url=None,
        enabled=True,
        no_auth=False,
        is_local_toolkit=False,
    )


def _composio_toolkit_matches_query(toolkit: ComposioToolkit, query: str) -> bool:
    if not query:
        return True
    normalized = query.casefold()
    haystack = " ".join(
        (
            toolkit.slug,
            toolkit.name,
            toolkit.description,
            " ".join(toolkit.categories),
            " ".join(toolkit.auth_schemes),
        )
    ).casefold()
    return normalized in haystack


def _composio_toolkit_row(
    toolkit: ComposioToolkit,
    connection_statuses: dict[str, str],
) -> ComposioToolkitRow:
    status = connection_statuses.get(toolkit.slug.lower())
    if status is None:
        connection_status = "Available"
        connection_tone = "neutral"
    else:
        connection_status = status.title()
        connection_tone = _composio_connection_tone(status)
    return ComposioToolkitRow(
        slug=toolkit.slug,
        name=toolkit.name,
        description=toolkit.description,
        logo_url=toolkit.logo_url,
        categories=toolkit.categories,
        auth_schemes=toolkit.auth_schemes,
        managed_auth_schemes=toolkit.managed_auth_schemes,
        tools_count=toolkit.tools_count,
        triggers_count=toolkit.triggers_count,
        no_auth=toolkit.no_auth,
        connection_status=connection_status,
        connection_tone=connection_tone,
        connected=status == "active",
    )


def _composio_tool_row(tool: ComposioTool) -> ComposioToolRow:
    name = tool.name.strip()
    if not name or name.upper() == tool.slug.upper():
        name = _humanize_composio_tool_slug(tool.slug, tool.toolkit_slug)
    return ComposioToolRow(
        slug=tool.slug,
        name=name,
        description=tool.description,
        tags=tool.tags,
        version=tool.version,
    )


def _humanize_composio_tool_slug(slug: str, toolkit_slug: str) -> str:
    normalized_slug = slug.upper()
    toolkit_prefix = f"{toolkit_slug.upper()}_"
    if normalized_slug.startswith(toolkit_prefix):
        normalized_slug = normalized_slug[len(toolkit_prefix) :]
    return _humanize_composio_slug(normalized_slug)


def _humanize_composio_slug(slug: str) -> str:
    return slug.replace("_", " ").replace("-", " ").title()


def _composio_scope_options() -> tuple[ComposioScopeOption, ...]:
    return (
        ComposioScopeOption(
            name="Personal",
            key="user",
            description="Only the Slack user who connected the account can use it.",
            default=True,
        ),
        ComposioScopeOption(
            name="Channel",
            key="channel",
            description="Tasks in one Slack channel can use the connected account.",
            risk="Good for shared project apps; risky for personal inboxes or calendars.",
        ),
        ComposioScopeOption(
            name="Workspace",
            key="workspace",
            description="Any task in the Slack workspace can use the connected account.",
            risk="Requires explicit admin-level intent before enabling.",
        ),
    )


def _composio_connection_tone(status: str) -> str:
    return {
        "active": "success",
        "pending": "warning",
        "expired": "warning",
        "failed": "danger",
        "disabled": "neutral",
    }.get(status, "neutral")


def _short_error(value: str, *, max_chars: int = 220) -> str:
    cleaned = " ".join(value.split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 1].rstrip() + "..."


def _tool_capability_groups(
    settings: Settings | None,
) -> tuple[ToolCapabilityGroup, ...]:
    config_available = settings is not None
    grouped: dict[str, list[ToolCapability]] = defaultdict(list)
    for tool in _NATIVE_DASHBOARD_TOOL_CLASSES:
        descriptor = tool_descriptor_from_class(
            tool,
            settings=settings,
            config_available=config_available,
        )
        grouped[descriptor.category].append(_tool_capability(descriptor))

    return tuple(
        ToolCapabilityGroup(
            name=category,
            description=_tool_group_description(category),
            tools=tuple(tools),
        )
        for category, tools in grouped.items()
    )


def _tool_capability(descriptor: ToolDescriptor) -> ToolCapability:
    return ToolCapability(
        name=descriptor.name,
        group=descriptor.category,
        status="Available" if descriptor.enabled else "Needs setup",
        tone="success" if descriptor.enabled else "warning",
        description=descriptor.description,
        required_args=descriptor.required_args,
        optional_args=descriptor.optional_args,
        notes=(
            f"Side effect: {descriptor.side_effect}.",
            f"Approval: {descriptor.approval}.",
            (
                f"Scopes: {', '.join(descriptor.required_slack_scopes)}."
                if descriptor.required_slack_scopes
                else "No Slack scope beyond runtime context."
            ),
            (
                descriptor.disabled_reason
                if not descriptor.enabled and descriptor.disabled_reason
                else "Registered through the native tool catalog."
            ),
            *descriptor.notes,
        ),
    )


def _tool_group_description(category: str) -> str:
    descriptions = {
        "Documents": "Tools that create or manage task artifacts.",
        "Memory": "Confirm-gated fact memory and operator-visible recall tools.",
        "Research": "Tools that gather external context.",
        "Runtime": "Meta-tools that describe current capabilities and integrations.",
        "Scheduling": "Tools that create and manage recurring or future work.",
        "Slack actions": "Tools that perform bounded Slack replies and reactions.",
        "Slack context": "Tools that read Slack messages, threads, and files.",
        "Workspace context": "Tools that query Kortny's workspace knowledge graph.",
    }
    return descriptions.get(category, "Native tools registered by Kortny.")


def _integration_observability_status(settings: Settings) -> str:
    if not settings.observability_enabled:
        return "Disabled"
    if settings.otel_exporter_otlp_endpoint:
        return "OTLP export"
    return "Local metadata"


def _integration_observability_tone(settings: Settings) -> str:
    if not settings.observability_enabled:
        return "warning"
    if settings.otel_exporter_otlp_endpoint:
        return "success"
    return "neutral"


def _model_row(label: str, value: str | None) -> SystemConfigRow:
    if value:
        return SystemConfigRow(label, value, tone="success")
    return SystemConfigRow(
        label,
        "Fallback to default",
        detail="Set the tier-specific env var to override.",
        tone="warning",
    )


def _overall_tone(checks: Sequence[SystemCheck]) -> str:
    tones = {check.tone for check in checks}
    if "danger" in tones:
        return "danger"
    if "warning" in tones:
        return "warning"
    return "success"


def _overall_label(checks: Sequence[SystemCheck]) -> str:
    tone = _overall_tone(checks)
    if tone == "danger":
        return "Needs setup"
    if tone == "warning":
        return "Needs attention"
    return "Ready"


def _observability_status(settings: Settings) -> str:
    if not settings.observability_enabled:
        return "Disabled"
    if settings.otel_exporter_otlp_endpoint:
        return "Exporting"
    return "Local only"


def _observability_tone(settings: Settings) -> str:
    if not settings.observability_enabled:
        return "warning"
    if settings.otel_exporter_otlp_endpoint:
        return "success"
    return "warning"


def _observability_detail(settings: Settings) -> str:
    if not settings.observability_enabled:
        return "Task events still exist, but OTEL instrumentation is disabled."
    if settings.otel_exporter_otlp_endpoint:
        return f"Traces export to {_redact_url(settings.otel_exporter_otlp_endpoint)}."
    return "Structured logs and task events are available; no external trace sink is configured."


def _datetime_label(value: datetime | None) -> str:
    if value is None:
        return "-"
    return value.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")


def _relative_label(value: datetime | None, now: datetime) -> str:
    if value is None:
        return "Never"
    moment = value if value.tzinfo else value.replace(tzinfo=UTC)
    secs = (now - moment.astimezone(UTC)).total_seconds()
    secs = max(secs, 0.0)
    if secs < 90:
        return "just now"
    if secs < 3600:
        return f"{int(secs // 60)}m ago"
    if secs < 86400:
        return f"{int(secs // 3600)}h ago"
    return f"{int(secs // 86400)}d ago"


def _redact_url(value: str) -> str:
    parsed = urlsplit(value)
    if not parsed.scheme or not parsed.netloc:
        return value

    host = parsed.hostname or ""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"

    if parsed.username:
        user = parsed.username
        auth = f"{user}:***@"
    else:
        auth = ""

    return urlunsplit((parsed.scheme, f"{auth}{host}", parsed.path, "", ""))


def _failure_rate_label(total: int, failed: int) -> str:
    if total <= 0:
        return "0.0%"
    return f"{(failed / total) * 100:.1f}%"


def _overview_attention_items(
    session: Session,
    *,
    system_health: SystemHealth,
    current: datetime,
    today_cost: Decimal,
    week_cost: Decimal,
) -> tuple[OverviewAttentionItem, ...]:
    items: list[OverviewAttentionItem] = []
    for check in system_health.checks:
        if check.tone not in {"danger", "warning"}:
            continue
        items.append(
            OverviewAttentionItem(
                title=f"{check.group}: {check.name}",
                detail=check.action or check.detail,
                tone=check.tone,
                badge=check.status,
                href="/system",
            )
        )

    proposed_fact_count = (
        session.scalar(
            select(func.count())
            .select_from(WorkspaceState)
            .where(WorkspaceState.status == "proposed")
        )
        or 0
    )
    if proposed_fact_count:
        items.append(
            OverviewAttentionItem(
                title="Memory: proposed facts",
                detail=(
                    f"{int(proposed_fact_count):,} memory "
                    f"{'fact needs' if proposed_fact_count == 1 else 'facts need'} review."
                ),
                tone="warning",
                badge="review",
                href="/memory?status=proposed",
            )
        )

    stale_threshold = current - timedelta(minutes=30)
    stale_active_count = (
        session.scalar(
            select(func.count())
            .select_from(Task)
            .where(
                Task.status.in_(
                    (
                        TaskStatus.pending,
                        TaskStatus.running,
                        TaskStatus.waiting_approval,
                    )
                ),
                Task.created_at < stale_threshold,
            )
        )
        or 0
    )
    if stale_active_count:
        items.append(
            OverviewAttentionItem(
                title="Tasks: active work is stale",
                detail=(
                    f"{int(stale_active_count):,} pending, running, or approval "
                    f"{'task is' if stale_active_count == 1 else 'tasks are'} older than 30 minutes."
                ),
                tone="warning",
                badge="stale",
                href="/tasks?status=running",
            )
        )

    baseline_daily_cost = (
        (week_cost - today_cost) / Decimal("6") if week_cost else Decimal("0")
    )
    if (
        today_cost > Decimal("0")
        and baseline_daily_cost > Decimal("0")
        and today_cost >= baseline_daily_cost * Decimal("2")
    ):
        items.append(
            OverviewAttentionItem(
                title="Usage: cost spike today",
                detail=(
                    f"Today is {_format_money(today_cost)} versus "
                    f"{_format_money(baseline_daily_cost)} daily baseline."
                ),
                tone="warning",
                badge="spend",
                href="/usage",
            )
        )

    tasks = tuple(
        session.scalars(
            select(Task)
            .where(
                Task.status.in_(
                    (
                        TaskStatus.failed,
                        TaskStatus.crashed,
                        TaskStatus.waiting_approval,
                        TaskStatus.pending,
                        TaskStatus.running,
                    )
                )
            )
            .order_by(
                case(
                    (Task.status.in_((TaskStatus.failed, TaskStatus.crashed)), 0),
                    (Task.status == TaskStatus.waiting_approval, 1),
                    else_=1,
                ),
                Task.created_at.desc(),
                Task.id.desc(),
            )
            .limit(5)
        )
    )
    for item in _task_items(session, tasks):
        tone = (
            "danger"
            if item.task.status in (TaskStatus.failed, TaskStatus.crashed)
            else "warning"
        )
        items.append(
            OverviewAttentionItem(
                title=f"{item.task.status.value.capitalize()}: {_truncate(item.task.input, 86)}",
                detail=(
                    f"{item.user.name} in {item.channel.name} - "
                    f"{_datetime_label(item.task.created_at)}"
                ),
                tone=tone,
                badge=item.task.status.value,
                href=f"/tasks/{item.task.id}",
            )
        )

    return tuple(items[:8])


def _truncate(value: str, max_length: int) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= max_length:
        return normalized
    return normalized[: max_length - 3].rstrip() + "..."


def _memory_fact_identity_keys(
    facts: Sequence[WorkspaceState],
) -> tuple[IdentityKey, ...]:
    keys: list[IdentityKey] = []
    for fact in facts:
        if fact.scope_type in {"channel", "user"} and fact.scope_id:
            keys.append((fact.installation_id, fact.scope_type, fact.scope_id))
        for user_id in (
            fact.proposed_by,
            fact.confirmed_by_user_id,
            fact.rejected_by_user_id,
            fact.forgotten_by_user_id,
        ):
            if user_id:
                keys.append((fact.installation_id, "user", user_id))
    return tuple(keys)


_MEMORY_SCOPES = frozenset({"all", "workspace", "channel", "user"})
_MEMORY_STATUSES = frozenset(
    {"all", "active", "proposed", "rejected", "superseded", "forgotten"}
)
_MEMORY_OUTCOMES = frozenset({"all", "succeeded", "failed", "cancelled"})
_MEMORY_FACT_SORTS = frozenset({"updated_desc", "created_desc", "key_asc", "scope_asc"})
_MEMORY_EPISODE_SORTS = frozenset({"created_desc", "created_asc", "outcome_asc"})
_KG_SCOPES = frozenset({"all", "workspace", "channel", "private_channel", "dm", "user"})
_KG_STATES = frozenset(
    {
        "all",
        "current",
        "candidate",
        "active",
        "confirmed",
        "stale",
        "superseded",
        "contradicted",
        "archived",
        "forgotten",
    }
)
_KG_ENTITY_SORTS = frozenset(
    {"updated_desc", "created_desc", "confidence_desc", "key_asc"}
)
_KG_EDGE_SORTS = frozenset(
    {"updated_desc", "created_desc", "confidence_desc", "relationship_asc"}
)
_WITNESS_STATUSES = frozenset(
    {
        "all",
        "candidate",
        "sent",
        "accepted",
        "automated",
        "dismissed",
        "cooldown",
        "superseded",
        "archived",
    }
)
_WITNESS_TYPES = frozenset(
    {
        "all",
        "workflow_gap",
        "artifact_followup",
        "unresolved_decision",
        "data_quality_issue",
        "recurring_check",
        "project_status_gap",
        "general_help",
    }
)
_WITNESS_SCOPES = frozenset(
    {"all", "workspace", "channel", "private_channel", "dm", "user"}
)
_WITNESS_SORTS = frozenset(
    {"updated_desc", "created_desc", "confidence_desc", "status_asc", "type_asc"}
)


def _normalize_memory_sort(view: str, sort: str | None) -> str:
    if view == "episodes":
        return sort if sort in _MEMORY_EPISODE_SORTS else "created_desc"
    return sort if sort in _MEMORY_FACT_SORTS else "updated_desc"


def _normalize_kg_sort(view: str, sort: str | None) -> str:
    if view == "relationships":
        return sort if sort in _KG_EDGE_SORTS else "updated_desc"
    return sort if sort in _KG_ENTITY_SORTS else "updated_desc"


def _kg_installation_filter(
    model: type[Any],
    installation_id: uuid.UUID | None,
) -> list[ColumnElement[bool]]:
    if installation_id is None:
        return []
    return [model.installation_id == installation_id]


def _kg_current_filters(model: type[Any]) -> list[ColumnElement[bool]]:
    return [model.is_current.is_(True), model.expired_at.is_(None)]


def _kg_has_evidence_predicate(
    model: type[Any],
    target_kind: str,
) -> ColumnElement[bool]:
    return exists().where(
        KnowledgeGraphEvidence.target_kind == target_kind,
        KnowledgeGraphEvidence.target_id == model.id,
        KnowledgeGraphEvidence.installation_id == model.installation_id,
    )


def _kg_entity_rows(
    session: Session,
    *,
    query: str,
    scope_filter: str,
    state_filter: str,
    kind_filter: str,
    sort: str,
    page: int,
    page_size: int,
    installation_id: uuid.UUID | None,
) -> tuple[int, int, tuple[KnowledgeGraphEntityRow, ...]]:
    filters = _kg_entity_filters(
        query=query,
        scope_filter=scope_filter,
        state_filter=state_filter,
        kind_filter=kind_filter,
        installation_id=installation_id,
    )
    total_count = (
        session.scalar(
            select(func.count()).select_from(KnowledgeGraphEntity).where(*filters)
        )
        or 0
    )
    resolved_page = _resolved_page(
        page=page, page_size=page_size, total_count=total_count
    )
    entities = tuple(
        session.scalars(
            select(KnowledgeGraphEntity)
            .where(*filters)
            .order_by(*_kg_entity_order(sort))
            .offset((resolved_page - 1) * page_size)
            .limit(page_size)
        )
    )
    entity_ids = [entity.id for entity in entities]
    evidence_counts = _kg_evidence_counts(session, "entity", entity_ids)
    evidence_previews = _kg_evidence_previews(session, "entity", entity_ids)
    source_edge_counts, target_edge_counts = _kg_edge_counts_by_entity(
        session, entity_ids
    )
    identities = _identity_map_from_keys(
        session,
        tuple(
            key
            for entity in entities
            if (key := _kg_scope_identity_key(entity)) is not None
        ),
    )
    rows = tuple(
        KnowledgeGraphEntityRow(
            entity=entity,
            scope=_kg_scope_label(
                identities,
                installation_id=entity.installation_id,
                scope_type=entity.visibility_scope_type,
                scope_id=entity.visibility_scope_id,
            ),
            evidence_count=evidence_counts.get(entity.id, 0),
            source_edge_count=source_edge_counts.get(entity.id, 0),
            target_edge_count=target_edge_counts.get(entity.id, 0),
            evidence=evidence_previews.get(entity.id, ()),
            tone=_kg_lifecycle_tone(entity.lifecycle_state),
            confidence_label=_confidence_label(entity.confidence_score),
            provenance_label=provenance_label(
                provenance_kind(entity.source_type, entity.attrs_json)
            ),
            provenance_tone=_kg_provenance_tone(
                provenance_kind(entity.source_type, entity.attrs_json)
            ),
            review_status=review_status(entity.attrs_json, entity.lifecycle_state),
            attrs_preview=_json_preview(entity.attrs_json),
        )
        for entity in entities
    )
    return int(total_count), resolved_page, rows


def _kg_edge_rows(
    session: Session,
    *,
    query: str,
    scope_filter: str,
    state_filter: str,
    kind_filter: str,
    sort: str,
    page: int,
    page_size: int,
    installation_id: uuid.UUID | None,
) -> tuple[int, int, tuple[KnowledgeGraphEdgeRow, ...]]:
    source = aliased(KnowledgeGraphEntity)
    target = aliased(KnowledgeGraphEntity)
    filters = _kg_edge_filters(
        source=source,
        target=target,
        query=query,
        scope_filter=scope_filter,
        state_filter=state_filter,
        kind_filter=kind_filter,
        installation_id=installation_id,
    )
    total_count = (
        session.scalar(
            select(func.count())
            .select_from(KnowledgeGraphEdge)
            .join(source, KnowledgeGraphEdge.source_entity_id == source.id)
            .join(target, KnowledgeGraphEdge.target_entity_id == target.id)
            .where(*filters)
        )
        or 0
    )
    resolved_page = _resolved_page(
        page=page, page_size=page_size, total_count=total_count
    )
    edges = tuple(
        session.scalars(
            select(KnowledgeGraphEdge)
            .join(source, KnowledgeGraphEdge.source_entity_id == source.id)
            .join(target, KnowledgeGraphEdge.target_entity_id == target.id)
            .where(*filters)
            .order_by(*_kg_edge_order(sort))
            .offset((resolved_page - 1) * page_size)
            .limit(page_size)
        )
    )
    edge_ids = [edge.id for edge in edges]
    entity_ids = {
        entity_id
        for edge in edges
        for entity_id in (edge.source_entity_id, edge.target_entity_id)
    }
    entities_by_id = _kg_entities_by_id(session, entity_ids)
    evidence_counts = _kg_evidence_counts(session, "edge", edge_ids)
    evidence_previews = _kg_evidence_previews(session, "edge", edge_ids)
    identities = _identity_map_from_keys(
        session,
        tuple(
            key for edge in edges if (key := _kg_scope_identity_key(edge)) is not None
        ),
    )
    rows = tuple(
        KnowledgeGraphEdgeRow(
            edge=edge,
            source_label=_kg_entity_label(entities_by_id.get(edge.source_entity_id)),
            target_label=_kg_entity_label(entities_by_id.get(edge.target_entity_id)),
            scope=_kg_scope_label(
                identities,
                installation_id=edge.installation_id,
                scope_type=edge.visibility_scope_type,
                scope_id=edge.visibility_scope_id,
            ),
            evidence_count=evidence_counts.get(edge.id, 0),
            evidence=evidence_previews.get(edge.id, ()),
            tone=_kg_lifecycle_tone(edge.lifecycle_state),
            confidence_label=_confidence_label(edge.confidence_score),
            provenance_label=provenance_label(
                provenance_kind(edge.source_type, edge.attrs_json)
            ),
            provenance_tone=_kg_provenance_tone(
                provenance_kind(edge.source_type, edge.attrs_json)
            ),
            review_status=review_status(edge.attrs_json, edge.lifecycle_state),
            attrs_preview=_json_preview(edge.attrs_json),
        )
        for edge in edges
    )
    return int(total_count), resolved_page, rows


def _kg_graph_map(
    session: Session,
    *,
    active_view: str,
    query: str,
    scope_filter: str,
    state_filter: str,
    kind_filter: str,
    sort: str,
    installation_id: uuid.UUID | None,
) -> KnowledgeGraphMap:
    if active_view == "relationships":
        entities, edges = _kg_graph_from_relationship_filters(
            session,
            query=query,
            scope_filter=scope_filter,
            state_filter=state_filter,
            kind_filter=kind_filter,
            sort=sort,
            installation_id=installation_id,
        )
    else:
        entities, edges = _kg_graph_from_entity_filters(
            session,
            query=query,
            scope_filter=scope_filter,
            state_filter=state_filter,
            kind_filter=kind_filter,
            sort=sort,
            installation_id=installation_id,
        )

    if not entities:
        return KnowledgeGraphMap(
            nodes=(), edges=(), empty=True, node_count=0, edge_count=0
        )

    positions = _kg_graph_positions(len(entities))
    entity_ids = [entity.id for entity in entities]
    entity_id_set = set(entity_ids)
    evidence_counts = _kg_evidence_counts(session, "entity", entity_ids)
    source_edge_counts, target_edge_counts = _kg_edge_counts_by_entity(
        session, entity_ids
    )
    identities = _identity_map_from_keys(
        session,
        tuple(
            key
            for entity in entities
            if (key := _kg_scope_identity_key(entity)) is not None
        ),
    )
    node_by_id: dict[uuid.UUID, KnowledgeGraphMapNode] = {}
    for index, entity in enumerate(entities):
        outgoing_count = source_edge_counts.get(entity.id, 0)
        incoming_count = target_edge_counts.get(entity.id, 0)
        evidence_count = evidence_counts.get(entity.id, 0)
        degree = outgoing_count + incoming_count
        scope = _kg_scope_label(
            identities,
            installation_id=entity.installation_id,
            scope_type=entity.visibility_scope_type,
            scope_id=entity.visibility_scope_id,
        )
        node_by_id[entity.id] = KnowledgeGraphMapNode(
            id=str(entity.id),
            label=_kg_graph_node_label(entity),
            secondary_label=_kg_graph_node_secondary(entity),
            entity_type=entity.entity_type,
            lifecycle_state=entity.lifecycle_state,
            tone=_kg_lifecycle_tone(entity.lifecycle_state),
            x=positions[index][0],
            y=positions[index][1],
            radius=min(24, 13 + min(degree + evidence_count, 8)),
            evidence_count=evidence_count,
            incoming_count=incoming_count,
            outgoing_count=outgoing_count,
            confidence_label=_confidence_label(entity.confidence_score),
            provenance_label=provenance_label(
                provenance_kind(entity.source_type, entity.attrs_json)
            ),
            provenance_tone=_kg_provenance_tone(
                provenance_kind(entity.source_type, entity.attrs_json)
            ),
            review_status=review_status(entity.attrs_json, entity.lifecycle_state),
            scope_label=_kg_graph_scope_label(scope),
        )

    map_edges: list[KnowledgeGraphMapEdge] = []
    for edge in edges:
        if edge.source_entity_id not in entity_id_set:
            continue
        if edge.target_entity_id not in entity_id_set:
            continue
        source_node = node_by_id[edge.source_entity_id]
        target_node = node_by_id[edge.target_entity_id]
        map_edges.append(
            KnowledgeGraphMapEdge(
                id=str(edge.id),
                source_id=source_node.id,
                target_id=target_node.id,
                label=_truncate(edge.relationship_type.replace("_", " "), 22),
                relationship_type=edge.relationship_type,
                tone=_kg_lifecycle_tone(edge.lifecycle_state),
                x1=source_node.x,
                y1=source_node.y,
                x2=target_node.x,
                y2=target_node.y,
                label_x=(source_node.x + target_node.x) // 2,
                label_y=(source_node.y + target_node.y) // 2,
            )
        )
        if len(map_edges) >= KG_GRAPH_EDGE_LIMIT:
            break

    nodes = tuple(node_by_id[entity.id] for entity in entities)
    return KnowledgeGraphMap(
        nodes=nodes,
        edges=tuple(map_edges),
        empty=False,
        node_count=len(nodes),
        edge_count=len(map_edges),
    )


def _kg_graph_from_entity_filters(
    session: Session,
    *,
    query: str,
    scope_filter: str,
    state_filter: str,
    kind_filter: str,
    sort: str,
    installation_id: uuid.UUID | None,
) -> tuple[tuple[KnowledgeGraphEntity, ...], tuple[KnowledgeGraphEdge, ...]]:
    filters = _kg_entity_filters(
        query=query,
        scope_filter=scope_filter,
        state_filter=state_filter,
        kind_filter=kind_filter,
        installation_id=installation_id,
    )
    seed_entities = list(
        session.scalars(
            select(KnowledgeGraphEntity)
            .where(*filters)
            .order_by(*_kg_entity_order(sort))
            .limit(KG_GRAPH_NODE_LIMIT)
        )
    )
    if not seed_entities:
        return (), ()

    seed_ids = [entity.id for entity in seed_entities]
    edge_filters = _kg_installation_filter(KnowledgeGraphEdge, installation_id)
    edge_filters.extend(_kg_state_filters(KnowledgeGraphEdge, state_filter))
    if scope_filter != "all":
        edge_filters.append(KnowledgeGraphEdge.visibility_scope_type == scope_filter)
    edge_filters.append(
        or_(
            KnowledgeGraphEdge.source_entity_id.in_(seed_ids),
            KnowledgeGraphEdge.target_entity_id.in_(seed_ids),
        )
    )
    edges = list(
        session.scalars(
            select(KnowledgeGraphEdge)
            .where(*edge_filters)
            .order_by(
                KnowledgeGraphEdge.updated_at.desc(), KnowledgeGraphEdge.id.desc()
            )
            .limit(KG_GRAPH_EDGE_LIMIT)
        )
    )
    entity_ids = list(seed_ids)
    for edge in edges:
        for entity_id in (edge.source_entity_id, edge.target_entity_id):
            if entity_id in entity_ids:
                continue
            if len(entity_ids) >= KG_GRAPH_NODE_LIMIT:
                continue
            entity_ids.append(entity_id)
    entities_by_id = _kg_entities_by_id(session, entity_ids)
    entities = tuple(
        entity for entity_id in entity_ids if (entity := entities_by_id.get(entity_id))
    )
    return entities, tuple(edges)


def _kg_graph_from_relationship_filters(
    session: Session,
    *,
    query: str,
    scope_filter: str,
    state_filter: str,
    kind_filter: str,
    sort: str,
    installation_id: uuid.UUID | None,
) -> tuple[tuple[KnowledgeGraphEntity, ...], tuple[KnowledgeGraphEdge, ...]]:
    source = aliased(KnowledgeGraphEntity)
    target = aliased(KnowledgeGraphEntity)
    filters = _kg_edge_filters(
        source=source,
        target=target,
        query=query,
        scope_filter=scope_filter,
        state_filter=state_filter,
        kind_filter=kind_filter,
        installation_id=installation_id,
    )
    edges = list(
        session.scalars(
            select(KnowledgeGraphEdge)
            .join(source, KnowledgeGraphEdge.source_entity_id == source.id)
            .join(target, KnowledgeGraphEdge.target_entity_id == target.id)
            .where(*filters)
            .order_by(*_kg_edge_order(sort))
            .limit(KG_GRAPH_EDGE_LIMIT)
        )
    )
    entity_ids: list[uuid.UUID] = []
    for edge in edges:
        for entity_id in (edge.source_entity_id, edge.target_entity_id):
            if entity_id in entity_ids:
                continue
            if len(entity_ids) >= KG_GRAPH_NODE_LIMIT:
                continue
            entity_ids.append(entity_id)
    entities_by_id = _kg_entities_by_id(session, entity_ids)
    entities = tuple(
        entity for entity_id in entity_ids if (entity := entities_by_id.get(entity_id))
    )
    return entities, tuple(edges)


def _kg_graph_positions(count: int) -> tuple[tuple[int, int], ...]:
    if count <= 0:
        return ()
    center_x = KG_GRAPH_WIDTH // 2
    center_y = KG_GRAPH_HEIGHT // 2
    if count == 1:
        return ((center_x, center_y),)
    radius_x = 310 if count > 8 else 260
    radius_y = 150 if count > 8 else 128
    return tuple(
        (
            int(
                center_x
                + math.cos((-math.pi / 2) + ((2 * math.pi * index) / count)) * radius_x
            ),
            int(
                center_y
                + math.sin((-math.pi / 2) + ((2 * math.pi * index) / count)) * radius_y
            ),
        )
        for index in range(count)
    )


def _kg_graph_node_label(entity: KnowledgeGraphEntity) -> str:
    return _truncate(entity.display_name or entity.canonical_key, 28)


def _kg_graph_node_secondary(entity: KnowledgeGraphEntity) -> str:
    if entity.display_name:
        return _truncate(entity.canonical_key, 42)
    return _truncate(entity.entity_type, 42)


def _kg_graph_scope_label(scope: IdentityLabel) -> str:
    if scope.secondary:
        return f"{scope.name} / {scope.secondary}"
    return scope.name


def _kg_entity_filters(
    *,
    query: str,
    scope_filter: str,
    state_filter: str,
    kind_filter: str,
    installation_id: uuid.UUID | None,
) -> list[ColumnElement[bool]]:
    filters = _kg_installation_filter(KnowledgeGraphEntity, installation_id)
    filters.extend(_kg_state_filters(KnowledgeGraphEntity, state_filter))
    if scope_filter != "all":
        filters.append(KnowledgeGraphEntity.visibility_scope_type == scope_filter)
    if kind_filter != "all":
        filters.append(KnowledgeGraphEntity.entity_type == kind_filter)
    if query:
        pattern = f"%{query}%"
        filters.append(
            or_(
                KnowledgeGraphEntity.canonical_key.ilike(pattern),
                KnowledgeGraphEntity.display_name.ilike(pattern),
                KnowledgeGraphEntity.entity_type.ilike(pattern),
                KnowledgeGraphEntity.source_type.ilike(pattern),
                KnowledgeGraphEntity.visibility_scope_id.ilike(pattern),
                cast(KnowledgeGraphEntity.attrs_json, Text).ilike(pattern),
            )
        )
    return filters


def _kg_edge_filters(
    *,
    source: type[KnowledgeGraphEntity],
    target: type[KnowledgeGraphEntity],
    query: str,
    scope_filter: str,
    state_filter: str,
    kind_filter: str,
    installation_id: uuid.UUID | None,
) -> list[ColumnElement[bool]]:
    filters = _kg_installation_filter(KnowledgeGraphEdge, installation_id)
    filters.extend(_kg_state_filters(KnowledgeGraphEdge, state_filter))
    if scope_filter != "all":
        filters.append(KnowledgeGraphEdge.visibility_scope_type == scope_filter)
    if kind_filter != "all":
        filters.append(KnowledgeGraphEdge.relationship_type == kind_filter)
    if query:
        pattern = f"%{query}%"
        # source and target are aliased ORM classes; attribute access returns column
        # descriptors at runtime even though mypy sees the field types.
        src: Any = source
        tgt: Any = target
        filters.append(
            or_(
                KnowledgeGraphEdge.relationship_type.ilike(pattern),
                KnowledgeGraphEdge.source_type.ilike(pattern),
                KnowledgeGraphEdge.visibility_scope_id.ilike(pattern),
                cast(KnowledgeGraphEdge.attrs_json, Text).ilike(pattern),
                src.canonical_key.ilike(pattern),
                src.display_name.ilike(pattern),
                tgt.canonical_key.ilike(pattern),
                tgt.display_name.ilike(pattern),
            )
        )
    return filters


def _kg_state_filters(model: type[Any], state_filter: str) -> list[ColumnElement[bool]]:
    if state_filter == "all":
        return []
    if state_filter == "current":
        return [model.is_current.is_(True), model.expired_at.is_(None)]
    return [model.lifecycle_state == state_filter]


def _kg_entity_order(sort: str) -> tuple[Any, ...]:
    if sort == "created_desc":
        return (KnowledgeGraphEntity.created_at.desc(), KnowledgeGraphEntity.id.desc())
    if sort == "confidence_desc":
        return (
            KnowledgeGraphEntity.confidence_score.desc(),
            KnowledgeGraphEntity.updated_at.desc(),
            KnowledgeGraphEntity.id.desc(),
        )
    if sort == "key_asc":
        return (KnowledgeGraphEntity.canonical_key.asc(), KnowledgeGraphEntity.id.asc())
    return (KnowledgeGraphEntity.updated_at.desc(), KnowledgeGraphEntity.id.desc())


def _kg_edge_order(sort: str) -> tuple[Any, ...]:
    if sort == "created_desc":
        return (KnowledgeGraphEdge.created_at.desc(), KnowledgeGraphEdge.id.desc())
    if sort == "confidence_desc":
        return (
            KnowledgeGraphEdge.confidence_score.desc(),
            KnowledgeGraphEdge.updated_at.desc(),
            KnowledgeGraphEdge.id.desc(),
        )
    if sort == "relationship_asc":
        return (
            KnowledgeGraphEdge.relationship_type.asc(),
            KnowledgeGraphEdge.updated_at.desc(),
            KnowledgeGraphEdge.id.desc(),
        )
    return (KnowledgeGraphEdge.updated_at.desc(), KnowledgeGraphEdge.id.desc())


def _kg_evidence_counts(
    session: Session,
    target_kind: str,
    target_ids: Sequence[uuid.UUID],
) -> dict[uuid.UUID, int]:
    if not target_ids:
        return {}
    rows = session.execute(
        select(KnowledgeGraphEvidence.target_id, func.count())
        .where(
            KnowledgeGraphEvidence.target_kind == target_kind,
            KnowledgeGraphEvidence.target_id.in_(target_ids),
        )
        .group_by(KnowledgeGraphEvidence.target_id)
    )
    return {target_id: int(count) for target_id, count in rows}


def _kg_evidence_previews(
    session: Session,
    target_kind: str,
    target_ids: Sequence[uuid.UUID],
    *,
    limit_per_target: int = 2,
) -> dict[uuid.UUID, tuple[KnowledgeGraphEvidencePreview, ...]]:
    if not target_ids:
        return {}
    grouped: dict[uuid.UUID, list[KnowledgeGraphEvidencePreview]] = defaultdict(list)
    rows = session.scalars(
        select(KnowledgeGraphEvidence)
        .where(
            KnowledgeGraphEvidence.target_kind == target_kind,
            KnowledgeGraphEvidence.target_id.in_(target_ids),
        )
        .order_by(
            KnowledgeGraphEvidence.created_at.desc(), KnowledgeGraphEvidence.id.desc()
        )
    )
    for evidence in rows:
        previews = grouped[evidence.target_id]
        if len(previews) >= limit_per_target:
            continue
        previews.append(
            KnowledgeGraphEvidencePreview(
                evidence=evidence,
                source_label=_kg_evidence_source_label(evidence),
                snippet=_kg_evidence_snippet(evidence),
                confidence_label=_confidence_label(evidence.confidence_score),
            )
        )
    return {target_id: tuple(previews) for target_id, previews in grouped.items()}


def _kg_edge_counts_by_entity(
    session: Session,
    entity_ids: Sequence[uuid.UUID],
) -> tuple[dict[uuid.UUID, int], dict[uuid.UUID, int]]:
    if not entity_ids:
        return {}, {}
    source_rows = session.execute(
        select(KnowledgeGraphEdge.source_entity_id, func.count())
        .where(KnowledgeGraphEdge.source_entity_id.in_(entity_ids))
        .group_by(KnowledgeGraphEdge.source_entity_id)
    )
    target_rows = session.execute(
        select(KnowledgeGraphEdge.target_entity_id, func.count())
        .where(KnowledgeGraphEdge.target_entity_id.in_(entity_ids))
        .group_by(KnowledgeGraphEdge.target_entity_id)
    )
    return (
        {entity_id: int(count) for entity_id, count in source_rows},
        {entity_id: int(count) for entity_id, count in target_rows},
    )


def _kg_entities_by_id(
    session: Session,
    entity_ids: Iterable[uuid.UUID],
) -> dict[uuid.UUID, KnowledgeGraphEntity]:
    ids = tuple(entity_ids)
    if not ids:
        return {}
    return {
        entity.id: entity
        for entity in session.scalars(
            select(KnowledgeGraphEntity).where(KnowledgeGraphEntity.id.in_(ids))
        )
    }


class _HasScopeFields(Protocol):
    installation_id: uuid.UUID
    visibility_scope_type: str
    visibility_scope_id: str | None


def _kg_scope_identity_key(model: _HasScopeFields) -> IdentityKey | None:
    scope_type = model.visibility_scope_type
    scope_id = model.visibility_scope_id
    if not scope_id:
        return None
    if scope_type in {"channel", "private_channel"}:
        return (model.installation_id, "channel", scope_id)
    if scope_type == "user":
        return (model.installation_id, "user", scope_id)
    return None


def _kg_scope_label(
    identities: dict[IdentityKey, SlackIdentity],
    *,
    installation_id: uuid.UUID,
    scope_type: str,
    scope_id: str | None,
) -> IdentityLabel:
    if scope_type == "workspace":
        return IdentityLabel(name="Workspace", slack_id="workspace", found=True)
    if scope_id is None:
        return IdentityLabel(name=scope_type, slack_id=scope_type, found=False)
    if scope_type in {"channel", "private_channel"}:
        return _identity_label(
            identities,
            installation_id=installation_id,
            kind="channel",
            slack_id=scope_id,
        )
    if scope_type == "user":
        return _identity_label(
            identities,
            installation_id=installation_id,
            kind="user",
            slack_id=scope_id,
        )
    return IdentityLabel(
        name=f"{scope_type}:{scope_id}", slack_id=scope_id, found=False
    )


def _kg_entity_label(entity: KnowledgeGraphEntity | None) -> str:
    if entity is None:
        return "Unknown entity"
    if entity.display_name:
        return f"{entity.display_name} ({entity.canonical_key})"
    return entity.canonical_key


def _kg_evidence_source_label(evidence: KnowledgeGraphEvidence) -> str:
    if evidence.source_task_id:
        return f"Task {evidence.source_task_id}"
    if evidence.source_observation_id:
        return f"Observation {evidence.source_observation_id}"
    if evidence.source_episode_id:
        return f"Episode {evidence.source_episode_id}"
    if evidence.source_slack_channel_id:
        return f"Slack {evidence.source_slack_channel_id}"
    if evidence.source_slack_file_id:
        return f"Slack file {evidence.source_slack_file_id}"
    if evidence.source_url:
        return evidence.source_url
    return evidence.source_type


def _kg_evidence_snippet(evidence: KnowledgeGraphEvidence) -> str:
    if evidence.raw_snippet:
        return _truncate(evidence.raw_snippet, 220)
    if evidence.confidence_reason:
        return _truncate(evidence.confidence_reason, 220)
    if evidence.source_url:
        return _truncate(evidence.source_url, 220)
    return "No snippet recorded."


def _kg_lifecycle_tone(state: str) -> str:
    if state in {"active", "confirmed"}:
        return "success"
    if state in {"candidate", "stale"}:
        return "warning"
    if state in {"contradicted", "forgotten"}:
        return "danger"
    return "neutral"


def _kg_provenance_tone(kind: str) -> str:
    if kind == "observed":
        return "success"
    if kind == "extracted":
        return "accent"
    if kind == "inferred":
        return "warning"
    return "neutral"


def _confidence_label(value: Decimal | None) -> str:
    if value is None:
        return "-"
    return f"{int((value * Decimal('100')).quantize(Decimal('1')))}%"


def _json_preview(payload: dict | None) -> str:
    if not payload:
        return "No attributes"
    return _truncate(json.dumps(payload, sort_keys=True, default=str), 180)


def _knowledge_graph_page_url(
    *,
    view: str,
    query: str,
    scope_filter: str,
    state_filter: str,
    kind_filter: str,
    sort: str,
    page: int | None,
    page_size: int,
    base_path: str = "/knowledge-graph",
) -> str:
    params: dict[str, str | int] = {
        "view": view,
        "state": state_filter,
        "sort": sort,
        "page": page or 1,
        "page_size": page_size,
    }
    if query:
        params["q"] = query
    if scope_filter != "all":
        params["scope"] = scope_filter
    if kind_filter != "all":
        params["kind"] = kind_filter
    return f"{base_path}?{urlencode(params)}"


def _witness_candidate_rows(
    session: Session,
    *,
    query: str,
    status_filter: str,
    type_filter: str,
    scope_filter: str,
    sort: str,
    page: int,
    page_size: int,
    installation_id: uuid.UUID | None,
    now: datetime,
) -> tuple[int, int, tuple[WitnessCandidateRow, ...]]:
    filters = _witness_candidate_filters(
        query=query,
        status_filter=status_filter,
        type_filter=type_filter,
        scope_filter=scope_filter,
        installation_id=installation_id,
    )
    total_count = (
        session.scalar(
            select(func.count())
            .select_from(WitnessOpportunityCandidate)
            .where(*filters)
        )
        or 0
    )
    resolved_page = _resolved_page(
        page=page, page_size=page_size, total_count=total_count
    )
    candidates = tuple(
        session.scalars(
            select(WitnessOpportunityCandidate)
            .where(*filters)
            .order_by(*_witness_candidate_order(sort))
            .offset((resolved_page - 1) * page_size)
            .limit(page_size)
        )
    )
    tasks = _tasks_by_id(
        session, [candidate.source_task_id for candidate in candidates]
    )
    profile_ids = tuple(
        {
            candidate.source_profile_id
            for candidate in candidates
            if candidate.source_profile_id
        }
    )
    profiles = (
        {
            profile.id: profile
            for profile in session.scalars(
                select(ObserveChannelProfile).where(
                    ObserveChannelProfile.id.in_(profile_ids)
                )
            )
        }
        if profile_ids
        else {}
    )
    identities = _identity_map_from_keys(
        session, _witness_candidate_identity_keys(candidates)
    )
    return (
        int(total_count),
        resolved_page,
        tuple(
            WitnessCandidateRow(
                candidate=candidate,
                channel=(
                    _identity_label(
                        identities,
                        installation_id=candidate.installation_id,
                        kind="channel",
                        slack_id=candidate.channel_id,
                    )
                    if candidate.channel_id
                    else None
                ),
                target_user=(
                    _identity_label(
                        identities,
                        installation_id=candidate.installation_id,
                        kind="user",
                        slack_id=candidate.target_slack_user_id,
                    )
                    if candidate.target_slack_user_id
                    else None
                ),
                scope=_kg_scope_label(
                    identities,
                    installation_id=candidate.installation_id,
                    scope_type=candidate.visibility_scope_type,
                    scope_id=candidate.visibility_scope_id,
                ),
                source_task=(
                    tasks.get(candidate.source_task_id)
                    if candidate.source_task_id is not None
                    else None
                ),
                source_profile=(
                    profiles.get(candidate.source_profile_id)
                    if candidate.source_profile_id is not None
                    else None
                ),
                evidence=_witness_evidence_preview(candidate),
                tone=_witness_status_tone(
                    candidate.status, candidate.cooldown_until, now
                ),
                type_label=_labelize(candidate.candidate_type),
                status_label=_labelize(candidate.status),
                source_label=_labelize(candidate.source_type),
                confidence_label=_confidence_label(candidate.confidence_score),
                cooldown_label=_witness_cooldown_label(candidate.cooldown_until, now),
                can_send_private=_witness_can_send_private(candidate, now),
                can_snooze=candidate.status == "candidate",
                can_dismiss=candidate.status in {"candidate", "sent", "cooldown"},
                can_accept=candidate.status in {"candidate", "sent", "cooldown"},
                can_reactivate=candidate.status in {"dismissed", "cooldown"},
                can_archive=candidate.status != "archived",
            )
            for candidate in candidates
        ),
    )


def _witness_candidate_filters(
    *,
    query: str,
    status_filter: str,
    type_filter: str,
    scope_filter: str,
    installation_id: uuid.UUID | None,
) -> list[ColumnElement[bool]]:
    filters = _kg_installation_filter(WitnessOpportunityCandidate, installation_id)
    if status_filter != "all":
        filters.append(WitnessOpportunityCandidate.status == status_filter)
    if type_filter != "all":
        filters.append(WitnessOpportunityCandidate.candidate_type == type_filter)
    if scope_filter != "all":
        filters.append(
            WitnessOpportunityCandidate.visibility_scope_type == scope_filter
        )
    if query:
        pattern = f"%{query}%"
        channel_identity_match = (
            select(SlackIdentity.id)
            .where(
                SlackIdentity.installation_id
                == WitnessOpportunityCandidate.installation_id,
                SlackIdentity.kind == "channel",
                SlackIdentity.slack_id == WitnessOpportunityCandidate.channel_id,
                or_(
                    SlackIdentity.display_name.ilike(pattern),
                    SlackIdentity.raw_name.ilike(pattern),
                ),
            )
            .exists()
        )
        user_identity_match = (
            select(SlackIdentity.id)
            .where(
                SlackIdentity.installation_id
                == WitnessOpportunityCandidate.installation_id,
                SlackIdentity.kind == "user",
                SlackIdentity.slack_id
                == WitnessOpportunityCandidate.target_slack_user_id,
                or_(
                    SlackIdentity.display_name.ilike(pattern),
                    SlackIdentity.raw_name.ilike(pattern),
                ),
            )
            .exists()
        )
        filters.append(
            or_(
                WitnessOpportunityCandidate.title.ilike(pattern),
                WitnessOpportunityCandidate.summary.ilike(pattern),
                WitnessOpportunityCandidate.suggested_action.ilike(pattern),
                WitnessOpportunityCandidate.suggested_message.ilike(pattern),
                WitnessOpportunityCandidate.channel_id.ilike(pattern),
                WitnessOpportunityCandidate.target_slack_user_id.ilike(pattern),
                WitnessOpportunityCandidate.visibility_scope_id.ilike(pattern),
                WitnessOpportunityCandidate.source_id.ilike(pattern),
                WitnessOpportunityCandidate.dedupe_key.ilike(pattern),
                cast(WitnessOpportunityCandidate.evidence_json, Text).ilike(pattern),
                cast(WitnessOpportunityCandidate.metadata_json, Text).ilike(pattern),
                channel_identity_match,
                user_identity_match,
            )
        )
    return filters


def _witness_candidate_order(sort: str) -> tuple[Any, ...]:
    if sort == "created_desc":
        return (
            WitnessOpportunityCandidate.created_at.desc(),
            WitnessOpportunityCandidate.id.desc(),
        )
    if sort == "confidence_desc":
        return (
            WitnessOpportunityCandidate.confidence_score.desc(),
            WitnessOpportunityCandidate.updated_at.desc(),
        )
    if sort == "status_asc":
        return (
            WitnessOpportunityCandidate.status.asc(),
            WitnessOpportunityCandidate.updated_at.desc(),
        )
    if sort == "type_asc":
        return (
            WitnessOpportunityCandidate.candidate_type.asc(),
            WitnessOpportunityCandidate.updated_at.desc(),
        )
    return (
        WitnessOpportunityCandidate.updated_at.desc(),
        WitnessOpportunityCandidate.created_at.desc(),
    )


def _witness_candidate_identity_keys(
    candidates: Sequence[WitnessOpportunityCandidate],
) -> tuple[IdentityKey, ...]:
    keys: list[IdentityKey] = []
    for candidate in candidates:
        if candidate.channel_id:
            keys.append((candidate.installation_id, "channel", candidate.channel_id))
        if candidate.target_slack_user_id:
            keys.append(
                (candidate.installation_id, "user", candidate.target_slack_user_id)
            )
        scope_key = _kg_scope_identity_key(candidate)
        if scope_key is not None:
            keys.append(scope_key)
    return tuple(keys)


def _witness_evidence_preview(
    candidate: WitnessOpportunityCandidate,
) -> tuple[WitnessEvidencePreview, ...]:
    evidence = (
        candidate.evidence_json if isinstance(candidate.evidence_json, list) else []
    )
    previews: list[WitnessEvidencePreview] = []
    for item in evidence:
        if not isinstance(item, dict):
            continue
        source_label = _labelize(str(item.get("type") or "evidence"))
        snippet = _witness_evidence_snippet(item)
        previews.append(
            WitnessEvidencePreview(source_label=source_label, snippet=snippet)
        )
        if len(previews) >= 3:
            break
    if not previews and candidate.confidence_reason:
        previews.append(
            WitnessEvidencePreview(
                source_label="Confidence",
                snippet=_truncate(candidate.confidence_reason, 220),
            )
        )
    return tuple(previews)


def _witness_evidence_snippet(item: Mapping[str, Any]) -> str:
    for key in ("snippet", "summary", "message", "text", "title"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return _truncate(value, 220)
    return _truncate(json.dumps(item, sort_keys=True, default=str), 220)


def _witness_status_tone(
    status: str,
    cooldown_until: datetime | None,
    now: datetime,
) -> str:
    if status == "accepted":
        return "success"
    if status == "automated":
        return "accent"
    if status == "sent":
        return "accent"
    if status == "candidate":
        if cooldown_until is not None and cooldown_until > now:
            return "warning"
        return "success"
    if status == "cooldown":
        return "warning"
    if status in {"dismissed", "superseded", "archived"}:
        return "neutral"
    return "neutral"


def _witness_can_send_private(
    candidate: WitnessOpportunityCandidate,
    now: datetime,
) -> bool:
    return (
        candidate.status == "candidate"
        and (candidate.cooldown_until is None or candidate.cooldown_until <= now)
        and candidate.visibility_scope_type == "dm"
        and candidate.channel_id is not None
        and candidate.channel_id.startswith("D")
        and bool(candidate.target_slack_user_id)
    )


def _witness_cooldown_label(
    cooldown_until: datetime | None,
    now: datetime,
) -> str | None:
    if cooldown_until is None:
        return None
    if cooldown_until > now:
        return f"Cooling down until {cooldown_until:%Y-%m-%d %H:%M UTC}"
    return "Cooldown elapsed"


def _witness_candidates_url(
    *,
    query: str,
    status_filter: str,
    type_filter: str,
    scope_filter: str,
    sort: str,
    page: int | None,
    page_size: int,
    base_path: str = "/witness",
) -> str:
    params: dict[str, str | int] = {
        "status": status_filter,
        "sort": sort,
        "page": page or 1,
        "page_size": page_size,
    }
    if query:
        params["q"] = query
    if type_filter != "all":
        params["type"] = type_filter
    if scope_filter != "all":
        params["scope"] = scope_filter
    return f"{base_path}?{urlencode(params)}"


def _labelize(value: str) -> str:
    return value.replace("_", " ").strip().title()


def _memory_fact_rows(
    session: Session,
    *,
    query: str,
    scope_filter: str,
    status_filter: str,
    sort: str,
    page: int,
    page_size: int,
    installation_id: uuid.UUID | None = None,
    slack_user_id: str | None = None,
) -> tuple[int, int, tuple[MemoryFactRow, ...]]:
    filters = _memory_fact_filters(
        query=query,
        scope_filter=scope_filter,
        status_filter=status_filter,
        installation_id=installation_id,
        slack_user_id=slack_user_id,
    )
    total_count = (
        session.scalar(select(func.count()).select_from(WorkspaceState).where(*filters))
        or 0
    )
    resolved_page = _resolved_page(
        page=page, page_size=page_size, total_count=total_count
    )
    facts = tuple(
        session.scalars(
            select(WorkspaceState)
            .where(*filters)
            .order_by(*_memory_fact_order(sort))
            .offset((resolved_page - 1) * page_size)
            .limit(page_size)
        )
    )
    source_tasks = _tasks_by_id(
        session,
        [fact.source_task_id for fact in facts if fact.source_task_id],
    )
    fact_identities = _identity_map_from_keys(
        session,
        _memory_fact_identity_keys(facts),
    )
    rows = tuple(
        MemoryFactRow(
            fact=fact,
            scope=_memory_scope_label(fact_identities, fact),
            value_summary=_memory_value_summary(fact),
            confirmed_by=_optional_user_label(
                fact_identities,
                installation_id=fact.installation_id,
                slack_id=fact.confirmed_by_user_id,
            ),
            proposed_by=_optional_user_label(
                fact_identities,
                installation_id=fact.installation_id,
                slack_id=fact.proposed_by,
            ),
            rejected_by=_optional_user_label(
                fact_identities,
                installation_id=fact.installation_id,
                slack_id=fact.rejected_by_user_id,
            ),
            forgotten_by=_optional_user_label(
                fact_identities,
                installation_id=fact.installation_id,
                slack_id=fact.forgotten_by_user_id,
            ),
            source_task=(
                source_tasks.get(fact.source_task_id)
                if fact.source_task_id is not None
                else None
            ),
            tone=_memory_status_tone(fact.status),
        )
        for fact in facts
    )
    return int(total_count), resolved_page, rows


def _memory_fact_filters(
    *,
    query: str,
    scope_filter: str,
    status_filter: str,
    installation_id: uuid.UUID | None = None,
    slack_user_id: str | None = None,
) -> list[ColumnElement[bool]]:
    filters = _workspace_state_scope_filter(
        installation_id=installation_id,
        slack_user_id=slack_user_id,
    )
    if status_filter != "all":
        filters.append(WorkspaceState.status == status_filter)
    if scope_filter != "all":
        filters.append(WorkspaceState.scope_type == scope_filter)
    if query:
        pattern = f"%{query}%"
        scope_identity_match = (
            select(SlackIdentity.id)
            .where(
                SlackIdentity.installation_id == WorkspaceState.installation_id,
                SlackIdentity.kind == WorkspaceState.scope_type,
                SlackIdentity.slack_id == WorkspaceState.scope_id,
                or_(
                    SlackIdentity.display_name.ilike(pattern),
                    SlackIdentity.raw_name.ilike(pattern),
                ),
            )
            .exists()
        )
        filters.append(
            or_(
                WorkspaceState.key.ilike(pattern),
                WorkspaceState.scope_id.ilike(pattern),
                WorkspaceState.value_text.ilike(pattern),
                cast(WorkspaceState.value_json, Text).ilike(pattern),
                scope_identity_match,
            )
        )
    return filters


def _memory_fact_order(sort: str) -> tuple[Any, ...]:
    if sort == "created_desc":
        return (WorkspaceState.created_at.desc(), WorkspaceState.id.desc())
    if sort == "key_asc":
        return (WorkspaceState.key.asc(), WorkspaceState.updated_at.desc())
    if sort == "scope_asc":
        return (
            WorkspaceState.scope_type.asc(),
            WorkspaceState.scope_id.asc().nullsfirst(),
            WorkspaceState.key.asc(),
        )
    return (WorkspaceState.updated_at.desc(), WorkspaceState.created_at.desc())


def _memory_episode_rows(
    session: Session,
    *,
    query: str,
    outcome_filter: str,
    sort: str,
    page: int,
    page_size: int,
    installation_id: uuid.UUID | None = None,
    slack_user_id: str | None = None,
) -> tuple[int, int, tuple[MemoryEpisodeRow, ...]]:
    filters = _memory_episode_filters(
        query=query,
        outcome_filter=outcome_filter,
        installation_id=installation_id,
        slack_user_id=slack_user_id,
    )
    base = select(Episode)
    count_base = select(func.count()).select_from(Episode)
    if query:
        base = base.join(Task, Task.id == Episode.task_id)
        count_base = count_base.join(Task, Task.id == Episode.task_id)
    total_count = session.scalar(count_base.where(*filters)) or 0
    resolved_page = _resolved_page(
        page=page, page_size=page_size, total_count=total_count
    )
    episodes = tuple(
        session.scalars(
            base.where(*filters)
            .order_by(*_memory_episode_order(sort))
            .offset((resolved_page - 1) * page_size)
            .limit(page_size)
        )
    )
    episode_tasks = _tasks_by_id(session, [episode.task_id for episode in episodes])
    episode_identities = _identity_map_from_keys(
        session,
        (
            key
            for episode in episodes
            for key in (
                (episode.installation_id, "channel", episode.channel_id),
                (episode.installation_id, "user", episode.user_id),
            )
        ),
    )
    rows = tuple(
        MemoryEpisodeRow(
            episode=episode,
            channel=_identity_label(
                episode_identities,
                installation_id=episode.installation_id,
                kind="channel",
                slack_id=episode.channel_id,
            ),
            user=_identity_label(
                episode_identities,
                installation_id=episode.installation_id,
                kind="user",
                slack_id=episode.user_id,
            ),
            task=episode_tasks.get(episode.task_id),
            tools_label=_list_count_label(episode.tools_used, "tool"),
            artifacts_label=_list_count_label(episode.artifacts_created, "artifact"),
            source_refs_label=_list_count_label(episode.source_refs, "source"),
            tone=_episode_tone(episode.outcome),
        )
        for episode in episodes
    )
    return int(total_count), resolved_page, rows


def _memory_episode_filters(
    *,
    query: str,
    outcome_filter: str,
    installation_id: uuid.UUID | None = None,
    slack_user_id: str | None = None,
) -> list[ColumnElement[bool]]:
    filters = _episode_scope_filter(
        installation_id=installation_id,
        slack_user_id=slack_user_id,
    )
    if outcome_filter != "all":
        filters.append(Episode.outcome == outcome_filter)
    if query:
        pattern = f"%{query}%"
        channel_identity_match = (
            select(SlackIdentity.id)
            .where(
                SlackIdentity.installation_id == Episode.installation_id,
                SlackIdentity.kind == "channel",
                SlackIdentity.slack_id == Episode.channel_id,
                or_(
                    SlackIdentity.display_name.ilike(pattern),
                    SlackIdentity.raw_name.ilike(pattern),
                ),
            )
            .exists()
        )
        user_identity_match = (
            select(SlackIdentity.id)
            .where(
                SlackIdentity.installation_id == Episode.installation_id,
                SlackIdentity.kind == "user",
                SlackIdentity.slack_id == Episode.user_id,
                or_(
                    SlackIdentity.display_name.ilike(pattern),
                    SlackIdentity.raw_name.ilike(pattern),
                ),
            )
            .exists()
        )
        filters.append(
            or_(
                Episode.summary.ilike(pattern),
                Episode.channel_id.ilike(pattern),
                Episode.user_id.ilike(pattern),
                Task.input.ilike(pattern),
                channel_identity_match,
                user_identity_match,
            )
        )
    return filters


def _memory_episode_order(sort: str) -> tuple[Any, ...]:
    if sort == "created_asc":
        return (Episode.created_at.asc(), Episode.id.asc())
    if sort == "outcome_asc":
        return (Episode.outcome.asc(), Episode.created_at.desc())
    return (Episode.created_at.desc(), Episode.id.desc())


def _memory_page_url(
    *,
    view: str,
    query: str,
    scope_filter: str,
    status_filter: str,
    outcome_filter: str,
    sort: str,
    page: int | None,
    page_size: int,
    base_path: str = "/memory",
) -> str:
    params: dict[str, str | int] = {
        "view": view,
        "sort": sort,
        "page": page or 1,
        "page_size": page_size,
    }
    if query:
        params["q"] = query
    if view == "facts":
        if scope_filter != "all":
            params["scope"] = scope_filter
        if status_filter != "active":
            params["status"] = status_filter
    else:
        if outcome_filter != "all":
            params["outcome"] = outcome_filter
    return f"{base_path}?{urlencode(params)}"


def _resolved_page(*, page: int, page_size: int, total_count: int) -> int:
    total_pages = max(1, math.ceil(total_count / page_size)) if total_count else 1
    return min(max(page, 1), total_pages)


def _memory_scope_label(
    identities: dict[IdentityKey, SlackIdentity],
    fact: WorkspaceState,
) -> IdentityLabel:
    if fact.scope_type == "workspace":
        return IdentityLabel(name="Workspace", slack_id="workspace", found=True)
    if fact.scope_id is None:
        return IdentityLabel(name="-", slack_id="-", found=False)
    return _identity_label(
        identities,
        installation_id=fact.installation_id,
        kind=fact.scope_type,
        slack_id=fact.scope_id,
    )


def _optional_user_label(
    identities: dict[IdentityKey, SlackIdentity],
    *,
    installation_id: uuid.UUID,
    slack_id: str | None,
) -> IdentityLabel | None:
    if not slack_id:
        return None
    return _identity_label(
        identities,
        installation_id=installation_id,
        kind="user",
        slack_id=slack_id,
    )


def _memory_value_summary(fact: WorkspaceState) -> str:
    if fact.value_text:
        return _truncate(fact.value_text, 180)
    return _truncate(json.dumps(fact.value_json, sort_keys=True, default=str), 180)


def _memory_status_tone(status: str) -> str:
    if status == "active":
        return "success"
    if status == "proposed":
        return "warning"
    if status in {"rejected", "forgotten"}:
        return "danger"
    return "neutral"


def _episode_tone(outcome: str) -> str:
    if outcome == "succeeded":
        return "success"
    if outcome == "failed":
        return "danger"
    return "warning"


def _list_count_label(items: object, singular: str) -> str:
    if not isinstance(items, list):
        return f"0 {singular}s"
    count = len(items)
    suffix = "" if count == 1 else "s"
    return f"{count:,} {singular}{suffix}"


def _tasks_by_id(
    session: Session,
    task_ids: Sequence[uuid.UUID | None],
) -> dict[uuid.UUID, Task]:
    normalized = tuple({task_id for task_id in task_ids if task_id is not None})
    if not normalized:
        return {}
    return {
        task.id: task
        for task in session.scalars(select(Task).where(Task.id.in_(normalized)))
    }


IdentityKey = tuple[uuid.UUID, str, str]


def _task_items(session: Session, tasks: Sequence[Task]) -> tuple[TaskListItem, ...]:
    usage_by_task = _usage_by_task(session, [task.id for task in tasks])
    identities = _identity_map(session, tasks)
    return tuple(
        TaskListItem(
            task=task,
            channel=_identity_label(
                identities,
                installation_id=task.installation_id,
                kind="channel",
                slack_id=task.slack_channel_id,
            ),
            user=_identity_label(
                identities,
                installation_id=task.installation_id,
                kind="user",
                slack_id=task.slack_user_id,
            ),
            models=tuple(sorted({usage.model for usage in usage_by_task[task.id]})),
            turn_count=len(usage_by_task[task.id]),
        )
        for task in tasks
    )


def _identity_map(
    session: Session,
    tasks: Sequence[Task],
) -> dict[IdentityKey, SlackIdentity]:
    keys: list[IdentityKey] = []
    for task in tasks:
        keys.append((task.installation_id, "channel", task.slack_channel_id))
        keys.append((task.installation_id, "user", task.slack_user_id))
    return _identity_map_from_keys(session, keys)


def _identity_map_from_keys(
    session: Session,
    keys: Iterable[IdentityKey],
) -> dict[IdentityKey, SlackIdentity]:
    normalized = tuple({key for key in keys if key[2]})
    if not normalized:
        return {}
    installation_ids = tuple({key[0] for key in normalized})
    slack_ids = tuple({key[2] for key in normalized})
    rows = session.scalars(
        select(SlackIdentity).where(
            SlackIdentity.installation_id.in_(installation_ids),
            SlackIdentity.slack_id.in_(slack_ids),
        )
    )
    return {
        (row.installation_id, row.kind, row.slack_id): row
        for row in rows
        if (row.installation_id, row.kind, row.slack_id) in normalized
    }


def _identity_label(
    identities: dict[IdentityKey, SlackIdentity],
    *,
    installation_id: uuid.UUID,
    kind: str,
    slack_id: str,
) -> IdentityLabel:
    identity = identities.get((installation_id, kind, slack_id))
    if identity is None:
        return IdentityLabel(name=slack_id, slack_id=slack_id, found=False)
    return IdentityLabel(
        name=identity.display_name,
        slack_id=identity.slack_id,
        found=True,
    )


def _user_label_for_tasks(
    session: Session,
    *,
    slack_user_id: str,
    tasks: Sequence[Task],
) -> IdentityLabel:
    keys = [
        (task.installation_id, "user", slack_user_id)
        for task in tasks
        if task.slack_user_id == slack_user_id
    ]
    identities = _identity_map_from_keys(session, keys)
    for key in keys:
        identity = identities.get(key)
        if identity is not None:
            return IdentityLabel(
                name=identity.display_name,
                slack_id=identity.slack_id,
                found=True,
            )
    return IdentityLabel(name=slack_user_id, slack_id=slack_user_id, found=False)


def _failed_task_case() -> Any:
    return case(
        (Task.status.in_((TaskStatus.failed, TaskStatus.crashed)), 1),
        else_=0,
    )


def _artifact_counts_by_user(
    session: Session,
    task_filter: Sequence[ColumnElement[bool]],
) -> dict[tuple[uuid.UUID, str], int]:
    rows = session.execute(
        select(Task.installation_id, Task.slack_user_id, func.count(Artifact.id))
        .join(Artifact, Artifact.task_id == Task.id)
        .where(*task_filter)
        .group_by(Task.installation_id, Task.slack_user_id)
    )
    return {(row[0], row[1]): int(row[2]) for row in rows}


def _artifact_counts_by_task(
    session: Session,
    task_ids: Sequence[uuid.UUID],
) -> dict[uuid.UUID, int]:
    if not task_ids:
        return {}
    rows = session.execute(
        select(Artifact.task_id, func.count(Artifact.id))
        .where(Artifact.task_id.in_(task_ids))
        .group_by(Artifact.task_id)
    )
    return {row[0]: int(row[1]) for row in rows}


def _artifact_count_for_user(
    session: Session,
    task_filter: Sequence[ColumnElement[bool]],
) -> int:
    return int(
        session.scalar(
            select(func.count(Artifact.id))
            .join(Task, Task.id == Artifact.task_id)
            .where(*task_filter)
        )
        or 0
    )


# Event types rendered as prominent "spans": the handful of real model/tool
# steps that carry durations + inspectable I/O. Everything else is a dim log
# line. ``error`` rides along as a span so failures never recede into noise.
_SPAN_EVENT_TYPES = frozenset({"llm_call", "tool_call", "tool_result", "error"})


def _event_tier(event_type: str) -> str:
    return "span" if event_type in _SPAN_EVENT_TYPES else "dim"


def _span_identity(event_type: str, payload: dict[str, Any]) -> tuple[str, str]:
    """Return ``(label, name)`` for a span row, e.g. ``("LLM", "openai/gpt")``.

    ``name`` falls back to an empty string when the concrete model/tool is
    unknown; the template then shows the generic title instead.
    """

    if event_type == "llm_call":
        return "LLM", _payload_string(payload, "model")
    if event_type in {"tool_call", "tool_result"}:
        return "Tool", _payload_string(payload, "tool")
    if event_type == "error":
        return "Error", _payload_string(payload, "error_type")
    return "", ""


def _timeline_event(event: TaskEvent) -> TimelineEvent:
    payload = event.payload if isinstance(event.payload, dict) else {}
    message = _payload_string(payload, "message")
    event_type = event.type.value
    title = _event_title(event_type, message)
    summary = _event_summary(event_type, payload, message)
    tone = _event_tone(event_type, message)
    badges = _event_badges(event_type, payload, message, tone)
    metrics = _event_metrics(event_type, payload)
    tier = _event_tier(event_type)
    span_label, span_name = _span_identity(event_type, payload)
    tokens = _payload_int_or_none(payload, "total_tokens")
    if tokens is None:
        input_tokens = _payload_int_or_none(payload, "input_tokens")
        output_tokens = _payload_int_or_none(payload, "output_tokens")
        if input_tokens is not None or output_tokens is not None:
            tokens = (input_tokens or 0) + (output_tokens or 0)
    input_json = _tool_io_json(payload.get("arguments"))
    output_json = _tool_io_json(payload.get("output"))
    prompt_json = _captured_content_json(payload.get("request_messages"))
    response_json = _captured_content_json(payload.get("response"))
    # A row is expandable only when expanding reveals real content: tool I/O,
    # captured prompt/response, or (for spans) a raw payload worth inspecting.
    # Dim log lines with only their one-line summary stay non-expandable.
    has_detail = bool(input_json or output_json or prompt_json or response_json) or (
        tier == "span" and bool(payload)
    )
    return TimelineEvent(
        seq=event.seq,
        event_type=event_type,
        tone=tone,
        title=title,
        summary=summary,
        created_at=event.created_at,
        badges=badges,
        metrics=metrics,
        payload_json=json.dumps(payload, indent=2, sort_keys=True, default=str),
        prompt_json=prompt_json,
        response_json=response_json,
        input_json=input_json,
        output_json=output_json,
        tier=tier,
        span_label=span_label,
        span_name=span_name,
        duration_ms=_payload_int_or_none(payload, "latency_ms"),
        tokens=tokens,
        cost_usd=_optional_payload_string(payload, "cost_usd"),
        turn=_payload_int_or_none(payload, "turn"),
        has_detail=has_detail,
    )


# Cap tool I/O so a big result (e.g. a search dump) doesn't bloat the page; the
# full value is always in the Raw payload below.
_TOOL_IO_MAX_CHARS = 20_000


def _tool_io_json(value: Any) -> str | None:
    """Pretty-print a tool call's input (arguments) or output for the timeline."""
    if not value:
        return None
    text = json.dumps(value, indent=2, sort_keys=True, default=str)
    if len(text) > _TOOL_IO_MAX_CHARS:
        return text[:_TOOL_IO_MAX_CHARS] + "\n… (truncated — see Raw payload)"
    return text


def _captured_content_json(value: Any) -> str | None:
    """Pretty-print captured prompt/response content for the timeline.

    Present only when ``OBSERVABILITY_CAPTURE_CONTENT`` is ``summaries``/``full``;
    ``None`` (the default metadata mode) hides the dedicated UI block.
    """

    if not value:
        return None
    return json.dumps(value, indent=2, sort_keys=True, default=str)


def _posted_response_text(events: Sequence[TaskEvent]) -> str | None:
    for event in reversed(events):
        if event.type is not TaskEventType.message_posted:
            continue
        payload = event.payload if isinstance(event.payload, dict) else {}
        purpose = _payload_string(payload, "purpose")
        if purpose not in {None, "", "result"}:
            continue
        text = _payload_string(payload, "text")
        if text:
            return text
    return None


def _planned_workflow_trace(
    *,
    events: Sequence[TaskEvent],
    timeline: Sequence[TimelineEvent],
    usage: Sequence[LLMUsage],
    raw_response_text: str | None,
    posted_response_text: str | None,
) -> PlannedWorkflowTrace:
    selected_payload: dict[str, Any] = {}
    classifier_payload: dict[str, Any] = {}
    present = False
    timeline_by_seq = {
        event.seq: item for event, item in zip(events, timeline, strict=True)
    }
    phase_events: list[TimelineEvent] = []
    budget_events: list[TimelineEvent] = []

    for event in events:
        payload = event.payload if isinstance(event.payload, dict) else {}
        message = _payload_string(payload, "message")
        if _is_planned_trace_payload(payload, event.type):
            present = True
        if message == "adk_planned_workflow_selected":
            selected_payload = payload
        if message == "unified_depth_decision":
            classifier_payload = payload
        if message in PLANNED_TRACE_MESSAGES and event.seq in timeline_by_seq:
            phase_events.append(timeline_by_seq[event.seq])
        if (
            message
            in {
                "planned_task_budget_reached",
                "planned_workflow_cost_ceiling_exceeded",
                "scheduled_task_cost_ceiling_exceeded",
            }
            and event.seq in timeline_by_seq
        ):
            budget_events.append(timeline_by_seq[event.seq])

    if not present:
        return PlannedWorkflowTrace(
            present=False,
            mode=None,
            route=None,
            planner_agent=None,
            merger_agent=None,
            max_parallel_branches=None,
            max_branch_model_calls=None,
            max_branch_tool_calls=None,
            max_total_tool_calls=None,
            cost_ceiling_usd=None,
            branch_count=0,
            completed_branch_count=0,
            budget_hit_count=0,
            llm_call_count=0,
            tool_call_count=0,
            tool_result_count=0,
            total_input_tokens=0,
            total_output_tokens=0,
            total_cost_usd=Decimal("0"),
            final_sanitized=False,
            raw_chars=len(raw_response_text) if raw_response_text else None,
            posted_chars=len(posted_response_text) if posted_response_text else None,
            branches=(),
            tool_rollups=(),
            budget_events=(),
            phase_events=(),
        )

    branch_state: dict[str, dict[str, Any]] = {}
    tool_state: dict[str, dict[str, Any]] = {}

    for agent_name in _payload_list(selected_payload, "branch_agents"):
        branch_key = PLANNED_BRANCH_AGENT_TO_KEY.get(str(agent_name))
        if branch_key is not None:
            _planned_branch_state(branch_state, branch_key)

    final_sanitized = False
    raw_chars = len(raw_response_text) if raw_response_text else None
    posted_chars = len(posted_response_text) if posted_response_text else None
    planned_llm_call_count = 0
    planned_input_tokens = 0
    planned_output_tokens = 0
    planned_tool_call_count = 0
    planned_tool_result_count = 0

    for event in events:
        payload = event.payload if isinstance(event.payload, dict) else {}
        message = _payload_string(payload, "message")
        branch_key = _planned_branch_key(payload)

        if message == "final_response_sanitized":
            final_sanitized = True
            raw_chars = _payload_int_or_none(payload, "raw_chars") or raw_chars
            posted_chars = _payload_int_or_none(payload, "output_chars") or posted_chars

        if branch_key is not None:
            branch = _planned_branch_state(branch_state, branch_key)
            if message == "planned_task_branch_started":
                branch["started"] = True
            elif message == "planned_task_branch_completed":
                branch["completed"] = True
            elif message == "planned_task_budget_reached":
                branch["budget_hit_count"] += 1

        if event.type is TaskEventType.llm_call and _is_planned_llm_payload(payload):
            planned_llm_call_count += 1
            input_tokens = _payload_int(payload, "input_tokens")
            output_tokens = _payload_int(payload, "output_tokens")
            cost_usd = _payload_decimal(payload, "cost_usd")
            planned_input_tokens += input_tokens
            planned_output_tokens += output_tokens
            if branch_key is not None:
                branch = _planned_branch_state(branch_state, branch_key)
                branch["llm_call_count"] += 1
                branch["input_tokens"] += input_tokens
                branch["output_tokens"] += output_tokens
                branch["cost_usd"] += cost_usd
                if model := _payload_string(payload, "model"):
                    branch["model_names"].append(model)

        if event.type in {TaskEventType.tool_call, TaskEventType.tool_result}:
            runtime = _payload_string(payload, "runtime")
            if runtime != "adk" and branch_key is None:
                continue
            tool = _payload_string(payload, "tool") or "tool"
            tool_rollup = tool_state.setdefault(
                tool,
                {
                    "call_count": 0,
                    "result_count": 0,
                    "artifact_count": 0,
                    "cost_usd": Decimal("0"),
                    "branch_labels": [],
                },
            )
            if branch_key is not None:
                label = PLANNED_BRANCH_LABELS[branch_key]
                if label not in tool_rollup["branch_labels"]:
                    tool_rollup["branch_labels"].append(label)
            if event.type is TaskEventType.tool_call:
                planned_tool_call_count += 1
                tool_rollup["call_count"] += 1
                if branch_key is not None:
                    branch = _planned_branch_state(branch_state, branch_key)
                    branch["tool_call_count"] += 1
                    branch["tool_names"].append(tool)
            else:
                planned_tool_result_count += 1
                tool_rollup["result_count"] += 1
                tool_rollup["artifact_count"] += _payload_int(payload, "artifact_count")
                tool_rollup["cost_usd"] += _payload_decimal(payload, "cost_usd")
                if branch_key is not None:
                    branch = _planned_branch_state(branch_state, branch_key)
                    branch["tool_result_count"] += 1
                    branch["tool_names"].append(tool)

    branches = tuple(
        _planned_branch_trace(branch_key, values)
        for branch_key, values in sorted(
            branch_state.items(), key=lambda item: _planned_branch_sort_key(item[0])
        )
    )
    tool_rollups = tuple(
        PlannedToolRollup(
            tool=tool,
            call_count=int(values["call_count"]),
            result_count=int(values["result_count"]),
            artifact_count=int(values["artifact_count"]),
            cost_usd=values["cost_usd"],
            branch_labels=tuple(values["branch_labels"]),
        )
        for tool, values in sorted(
            tool_state.items(),
            key=lambda item: (-int(item[1]["call_count"]), item[0]),
        )
    )
    classifier = selected_payload.get("classifier_payload")
    route = ""
    if isinstance(classifier, dict):
        route = _payload_string(classifier, "route")
    if not route and classifier_payload:
        route = _payload_string(classifier_payload, "route")
    return PlannedWorkflowTrace(
        present=True,
        mode=_optional_payload_string(selected_payload, "mode"),
        route=route or None,
        planner_agent=_optional_payload_string(selected_payload, "planner_agent"),
        merger_agent=_optional_payload_string(selected_payload, "merger_agent"),
        max_parallel_branches=_payload_int_or_none(
            selected_payload, "max_parallel_branches"
        ),
        max_branch_model_calls=_payload_int_or_none(
            selected_payload, "max_branch_model_calls"
        ),
        max_branch_tool_calls=_payload_int_or_none(
            selected_payload, "max_branch_tool_calls"
        ),
        max_total_tool_calls=_payload_int_or_none(
            selected_payload, "max_total_tool_calls"
        ),
        cost_ceiling_usd=_payload_decimal_or_none(selected_payload, "cost_ceiling_usd"),
        branch_count=len(branches),
        completed_branch_count=sum(1 for branch in branches if branch.completed),
        budget_hit_count=len(budget_events),
        llm_call_count=planned_llm_call_count,
        tool_call_count=planned_tool_call_count,
        tool_result_count=planned_tool_result_count,
        total_input_tokens=planned_input_tokens,
        total_output_tokens=planned_output_tokens,
        total_cost_usd=sum((row.cost_usd for row in usage), Decimal("0")),
        final_sanitized=final_sanitized,
        raw_chars=raw_chars,
        posted_chars=posted_chars,
        branches=branches,
        tool_rollups=tool_rollups,
        budget_events=tuple(budget_events),
        phase_events=tuple(phase_events),
    )


def _planned_branch_state(
    branch_state: dict[str, dict[str, Any]], branch_key: str
) -> dict[str, Any]:
    return branch_state.setdefault(
        branch_key,
        {
            "started": False,
            "completed": False,
            "llm_call_count": 0,
            "tool_call_count": 0,
            "tool_result_count": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": Decimal("0"),
            "budget_hit_count": 0,
            "tool_names": [],
            "model_names": [],
        },
    )


def _planned_branch_trace(
    branch_key: str, values: dict[str, Any]
) -> PlannedBranchTrace:
    budget_hit_count = int(values["budget_hit_count"])
    completed = bool(values["completed"])
    started = bool(values["started"])
    if budget_hit_count:
        status = "Budget hit"
        tone = "warning"
    elif completed:
        status = "Completed"
        tone = "success"
    elif started:
        status = "Started"
        tone = "accent"
    else:
        status = "Waiting"
        tone = "neutral"
    return PlannedBranchTrace(
        key=branch_key,
        label=PLANNED_BRANCH_LABELS[branch_key],
        status=status,
        tone=tone,
        started=started,
        completed=completed,
        llm_call_count=int(values["llm_call_count"]),
        tool_call_count=int(values["tool_call_count"]),
        tool_result_count=int(values["tool_result_count"]),
        input_tokens=int(values["input_tokens"]),
        output_tokens=int(values["output_tokens"]),
        cost_usd=values["cost_usd"],
        budget_hit_count=budget_hit_count,
        tool_names=tuple(_unique_strings(values["tool_names"])[:6]),
        model_names=tuple(_unique_strings(values["model_names"])[:4]),
    )


def _is_planned_trace_payload(
    payload: dict[str, Any], event_type: TaskEventType
) -> bool:
    message = _payload_string(payload, "message")
    if message in PLANNED_TRACE_MESSAGES or message == "unified_depth_decision":
        return True
    if _payload_string(payload, "adk_agent_name") in PLANNED_AGENT_NAMES:
        return True
    prompt_name = _payload_string(payload, "prompt_name")
    return (
        prompt_name.startswith("kortny.adk.planned_")
        and event_type is TaskEventType.llm_call
    )


def _is_planned_llm_payload(payload: dict[str, Any]) -> bool:
    agent_name = _payload_string(payload, "adk_agent_name")
    if agent_name in PLANNED_AGENT_NAMES:
        return True
    prompt_name = _payload_string(payload, "prompt_name")
    return prompt_name.startswith("kortny.adk.planned_")


def _planned_branch_key(payload: dict[str, Any]) -> str | None:
    branch = _payload_string(payload, "branch").lower()
    if branch in PLANNED_BRANCH_LABELS:
        return branch
    agent_name = _payload_string(payload, "adk_agent_name")
    if agent_name in PLANNED_BRANCH_AGENT_TO_KEY:
        return PLANNED_BRANCH_AGENT_TO_KEY[agent_name]
    prompt_name = _payload_string(payload, "prompt_name")
    for planned_agent_name, branch_key in PLANNED_BRANCH_AGENT_TO_KEY.items():
        if planned_agent_name in prompt_name:
            return branch_key
    return None


def _planned_branch_sort_key(branch_key: str) -> int:
    order = {"research": 0, "workspace": 1, "integration": 2}
    return order.get(branch_key, 99)


def _event_title(event_type: str, message: str) -> str:
    if event_type == "log" and message:
        return _message_title(message)
    titles = {
        "task_created": "Task created",
        "status_changed": "Status changed",
        "llm_call": "LLM call completed",
        "tool_call": "Tool call started",
        "tool_result": "Tool result recorded",
        "artifact_created": "Artifact created",
        "message_posted": "Slack message posted",
        "error": "Error recorded",
        "log": "Log event",
    }
    return titles.get(event_type, _humanize_slug(event_type))


def _message_title(message: str) -> str:
    titles = {
        "agent_executor_completed": "Agent executor completed",
        "agent_executor_started": "Agent executor started",
        "context_assembled": "Context assembled",
        "episode_recorded": "Episode recorded",
        "episode_retrieval_completed": "Episode retrieval completed",
        "llm_call_failed": "LLM call failed",
        "llm_call_started": "LLM call started",
        "memory_confirmation_posted": "Memory confirmation posted",
        "memory_write_confirmed": "Memory saved",
        "memory_write_skipped": "Memory skipped",
        "planned_task_branch_completed": "Planned branch completed",
        "planned_task_branch_started": "Planned branch started",
        "planned_task_budget_reached": "Planned budget reached",
        "planned_task_completed": "Planned task completed",
        "planned_task_merging": "Merging planned work",
        "planned_task_plan_ready": "Plan ready",
        "planned_task_planning_started": "Planning started",
        "planned_task_progress_posted": "Progress posted",
        "planned_task_started": "Planned task started",
        "task_executor_started": "Worker started task",
        "tool_call_completed": "Tool call completed",
        "tool_call_failed": "Tool call failed",
        "tool_call_started": "Tool call started",
    }
    return titles.get(message, _humanize_slug(message))


def _event_summary(event_type: str, payload: dict[str, Any], message: str) -> str:
    if event_type == "task_created":
        return _summary_from_fields(
            "Created from Slack",
            payload,
            ("slack_channel_id", "channel"),
            ("slack_user_id", "user"),
            ("slack_thread_ts", "thread_ts"),
            ("slack_event_id", "event_id"),
        )
    if event_type == "status_changed":
        from_status = _payload_string(payload, "from")
        to_status = _payload_string(payload, "to") or _payload_string(payload, "status")
        if from_status and to_status:
            return f"Moved from {from_status} to {to_status}."
        if to_status:
            return f"Task status is now {to_status}."
        return "Task status changed."
    if event_type == "llm_call":
        model = _payload_string(payload, "model") or "model"
        total_tokens = _payload_number(payload, "total_tokens")
        cost = _payload_string(payload, "cost_usd")
        pieces = [f"Completed by {model}"]
        if total_tokens:
            pieces.append(f"{total_tokens} tokens")
        if cost:
            pieces.append(f"${cost} recorded cost")
        return ". ".join(pieces) + "."
    if event_type == "tool_call":
        tool = _payload_string(payload, "tool") or "tool"
        argument_keys = payload.get("argument_keys")
        if isinstance(argument_keys, list) and argument_keys:
            return f"Invoked {tool} with {', '.join(map(str, argument_keys))}."
        return f"Invoked {tool}."
    if event_type == "tool_result":
        tool = _payload_string(payload, "tool") or "tool"
        latency = _payload_string(payload, "latency_ms")
        artifacts = _payload_string(payload, "artifact_count")
        pieces = [f"{tool} returned a result"]
        if latency:
            pieces.append(f"{latency} ms")
        if artifacts:
            pieces.append(f"{artifacts} artifacts")
        return ". ".join(pieces) + "."
    if event_type == "artifact_created":
        filename = _payload_string(payload, "filename") or "file"
        return f"Created artifact {filename}."
    if event_type == "message_posted":
        purpose = _payload_string(payload, "purpose") or "Slack update"
        channel = _payload_string(payload, "channel")
        if channel:
            return f"Posted {purpose} to {channel}."
        return f"Posted {purpose}."
    if event_type == "error":
        error_type = _payload_string(payload, "error_type") or "Error"
        error_summary = _payload_string(payload, "error_summary")
        if error_summary:
            return f"{error_type}: {error_summary}"
        return f"{error_type} recorded."
    if message == "context_assembled":
        fact_count = len(_payload_list(payload, "selected_fact_ids"))
        episode_count = len(_payload_list(payload, "selected_episode_ids"))
        artifact_count = len(_payload_list(payload, "selected_artifact_ids"))
        return (
            "Built the prompt context with "
            f"{fact_count} facts, {episode_count} episodes, "
            f"and {artifact_count} artifacts."
        )
    if message == "episode_retrieval_completed":
        selected_count = _payload_string(payload, "selected_count")
        if selected_count:
            return f"Retrieved {selected_count} relevant prior episodes."
    if message == "llm_call_started":
        model = _payload_string(payload, "model") or "model"
        prompt = _payload_string(payload, "prompt_name")
        if prompt:
            return f"Started {model} with prompt {prompt}."
        return f"Started {model}."
    if message in {
        "planned_task_started",
        "planned_task_planning_started",
        "planned_task_plan_ready",
        "planned_task_branch_started",
        "planned_task_branch_completed",
        "planned_task_budget_reached",
        "planned_task_merging",
        "planned_task_completed",
        "planned_task_progress_posted",
    }:
        return _planned_task_event_summary(payload, message)
    if message:
        return _humanize_slug(message) + "."
    return "Recorded execution metadata."


def _summary_from_fields(
    prefix: str, payload: dict[str, Any], *keys: tuple[str, str]
) -> str:
    fields = [
        f"{label}={value}"
        for key, label in keys
        if (value := _payload_string(payload, key))
    ]
    if not fields:
        return f"{prefix}."
    return f"{prefix}: {', '.join(fields)}."


def _planned_task_event_summary(payload: dict[str, Any], message: str) -> str:
    branch = _payload_string(payload, "branch")
    budget_type = _payload_string(payload, "budget_type")
    limit = _payload_string(payload, "limit")
    observed = _payload_string(payload, "observed")
    if message == "planned_task_started":
        return "Kortny marked this as planned work."
    if message == "planned_task_progress_posted":
        return "Posted a lightweight progress update in Slack."
    if message == "planned_task_planning_started":
        return "Started building the execution plan."
    if message == "planned_task_plan_ready":
        return "The planner produced a working plan."
    if message == "planned_task_branch_started":
        if branch:
            return f"Started the {branch} branch."
        return "Started a planned branch."
    if message == "planned_task_branch_completed":
        if branch:
            return f"Completed the {branch} branch."
        return "Completed a planned branch."
    if message == "planned_task_budget_reached":
        pieces = ["A planned branch reached its budget"]
        if budget_type:
            pieces.append(budget_type.replace("_", " "))
        if observed and limit:
            pieces.append(f"{observed}/{limit}")
        return ". ".join(pieces) + "."
    if message == "planned_task_merging":
        return "Started merging planned branch findings."
    if message == "planned_task_completed":
        return "Completed the planned workflow result."
    return _humanize_slug(message) + "."


def _event_tone(event_type: str, message: str) -> str:
    if event_type == "error" or message.endswith("_failed"):
        return "danger"
    if message == "planned_task_budget_reached":
        return "warning"
    if message.startswith("planned_task_"):
        return "accent"
    if event_type in {"llm_call", "tool_call", "tool_result"}:
        return "accent"
    if event_type in {"artifact_created", "message_posted"}:
        return "success"
    if event_type == "status_changed":
        return "warning"
    return "neutral"


def _event_badges(
    event_type: str, payload: dict[str, Any], message: str, tone: str
) -> tuple[TimelineBadge, ...]:
    badges = [TimelineBadge(label=event_type, tone=tone)]
    if message and message != event_type:
        badges.append(TimelineBadge(label=message, tone="neutral"))
    for key, badge_tone in (
        ("model_tier", "accent"),
        ("provider", "neutral"),
        ("tool", "accent"),
        ("status", "warning"),
        ("to", "warning"),
        ("phase", "accent"),
        ("branch", "accent"),
        ("budget_type", "warning"),
        ("purpose", "neutral"),
        ("error_type", "danger"),
    ):
        value = _payload_string(payload, key)
        if value:
            badges.append(TimelineBadge(label=value, tone=badge_tone))
    return tuple(_unique_badges(badges)[:5])


def _event_metrics(
    event_type: str, payload: dict[str, Any]
) -> tuple[TimelineMetric, ...]:
    message = _payload_string(payload, "message")
    keys = (
        "model",
        "prompt_name",
        "route_reason",
        "latency_ms",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cost_usd",
        "tool_call_count",
        "artifact_count",
        "selected_count",
        "worker_id",
        "filename",
        "mime_type",
        "size_bytes",
        "channel",
        "thread_ts",
        "message_ts",
        "phase",
        "branch",
        "budget_type",
        "reason",
        "limit",
        "observed",
    )
    metrics: list[TimelineMetric] = []
    for key in keys:
        value = (
            _payload_number(payload, key)
            if key in _NUMERIC_PAYLOAD_KEYS
            else _payload_string(payload, key)
        )
        if value:
            metrics.append(TimelineMetric(label=_humanize_slug(key), value=value))
    if event_type == "context_assembled" or message == "context_assembled":
        metrics.extend(
            [
                TimelineMetric(
                    label="facts",
                    value=str(len(_payload_list(payload, "selected_fact_ids"))),
                ),
                TimelineMetric(
                    label="episodes",
                    value=str(len(_payload_list(payload, "selected_episode_ids"))),
                ),
            ]
        )
    return tuple(metrics[:8])


def _payload_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if value is None or value == "":
        return ""
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, default=str)
    return str(value)


def _optional_payload_string(payload: dict[str, Any], key: str) -> str | None:
    value = _payload_string(payload, key)
    return value or None


_NUMERIC_PAYLOAD_KEYS = frozenset(
    {
        "latency_ms",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "tool_call_count",
        "artifact_count",
        "selected_count",
        "size_bytes",
        "turn",
        "limit",
        "observed",
    }
)


def _payload_int(payload: dict[str, Any], key: str) -> int:
    return _payload_int_or_none(payload, key) or 0


def _payload_int_or_none(payload: dict[str, Any], key: str) -> int | None:
    value = payload.get(key)
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _payload_decimal(payload: dict[str, Any], key: str) -> Decimal:
    return _payload_decimal_or_none(payload, key) or Decimal("0")


def _payload_decimal_or_none(payload: dict[str, Any], key: str) -> Decimal | None:
    value = payload.get(key)
    if value is None or value == "":
        return None
    try:
        return Decimal(str(value))
    except Exception:
        return None


def _payload_number(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if value is None or value == "":
        return ""
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return _payload_string(payload, key)


def _payload_list(payload: dict[str, Any], key: str) -> list[Any]:
    value = payload.get(key)
    return value if isinstance(value, list) else []


def _unique_badges(badges: list[TimelineBadge]) -> list[TimelineBadge]:
    seen: set[tuple[str, str]] = set()
    unique: list[TimelineBadge] = []
    for badge in badges:
        marker = (badge.label, badge.tone)
        if marker in seen:
            continue
        seen.add(marker)
        unique.append(badge)
    return unique


def _unique_strings(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def _humanize_slug(value: str) -> str:
    return value.replace("_", " ").replace("-", " ").strip().capitalize()


def _usage_by_task(
    session: Session, task_ids: list[uuid.UUID]
) -> dict[uuid.UUID, list[LLMUsage]]:
    usage_by_task: dict[uuid.UUID, list[LLMUsage]] = defaultdict(list)
    if not task_ids:
        return usage_by_task
    usage_rows = session.scalars(
        select(LLMUsage)
        .where(LLMUsage.task_id.in_(task_ids))
        .order_by(LLMUsage.created_at.asc(), LLMUsage.id.asc())
    )
    for usage in usage_rows:
        usage_by_task[usage.task_id].append(usage)
    return usage_by_task


def _usage_filter(
    *, start: datetime | None, end: datetime | None
) -> list[ColumnElement[bool]]:
    filters: list[ColumnElement[bool]] = []
    if start is not None:
        filters.append(LLMUsage.created_at >= start)
    if end is not None:
        filters.append(LLMUsage.created_at < end)
    return filters


def _task_filter(
    *, start: datetime | None, end: datetime | None
) -> list[ColumnElement[bool]]:
    filters: list[ColumnElement[bool]] = []
    if start is not None:
        filters.append(Task.created_at >= start)
    if end is not None:
        filters.append(Task.created_at < end)
    return filters


def _task_list_filters(
    *,
    query: str | None,
    status: str | None,
    channel: str | None,
    user: str | None,
    model: str | None,
) -> list[ColumnElement[bool]]:
    filters: list[ColumnElement[bool]] = []
    normalized_status = _normalize_task_status(status)
    if normalized_status is not None:
        filters.append(Task.status == normalized_status)
    normalized_query = _like_query(query)
    if normalized_query is not None:
        filters.append(
            or_(
                Task.input.ilike(normalized_query),
                Task.result_summary.ilike(normalized_query),
                Task.slack_channel_id.ilike(normalized_query),
                Task.slack_channel_id.in_(
                    _identity_match_ids("channel", normalized_query)
                ),
                Task.slack_user_id.ilike(normalized_query),
                Task.slack_user_id.in_(_identity_match_ids("user", normalized_query)),
            )
        )
    normalized_channel = _like_query(channel)
    if normalized_channel is not None:
        filters.append(
            or_(
                Task.slack_channel_id.ilike(normalized_channel),
                Task.slack_channel_id.in_(
                    _identity_match_ids("channel", normalized_channel)
                ),
            )
        )
    normalized_user = _like_query(user)
    if normalized_user is not None:
        filters.append(
            or_(
                Task.slack_user_id.ilike(normalized_user),
                Task.slack_user_id.in_(_identity_match_ids("user", normalized_user)),
            )
        )
    normalized_model = _like_query(model)
    if normalized_model is not None:
        filters.append(
            Task.id.in_(
                select(LLMUsage.task_id).where(LLMUsage.model.ilike(normalized_model))
            )
        )
    return filters


def _identity_match_ids(kind: str, pattern: str) -> Select[tuple[str]]:
    return select(SlackIdentity.slack_id).where(
        SlackIdentity.kind == kind,
        SlackIdentity.installation_id == Task.installation_id,
        or_(
            SlackIdentity.slack_id.ilike(pattern),
            SlackIdentity.display_name.ilike(pattern),
            SlackIdentity.raw_name.ilike(pattern),
        ),
    )


def _normalize_task_status(value: str | None) -> TaskStatus | None:
    if value is None or value.strip() in {"", "all"}:
        return None
    normalized = value.strip().lower()
    for status in TaskStatus:
        if status.value == normalized:
            return status
    return None


def _like_query(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    return f"%{stripped}%"


def _task_scope_filter(
    *,
    installation_id: uuid.UUID | None = None,
    slack_user_id: str | None = None,
) -> list[ColumnElement[bool]]:
    filters: list[ColumnElement[bool]] = []
    if installation_id is not None:
        filters.append(Task.installation_id == installation_id)
    if slack_user_id:
        filters.append(Task.slack_user_id == slack_user_id)
    return filters


def _single_installation_id(session: Session) -> uuid.UUID | None:
    installation_ids = tuple(
        session.scalars(
            select(Installation.id).order_by(Installation.created_at.asc()).limit(2)
        )
    )
    if len(installation_ids) == 1:
        return installation_ids[0]
    return None


def _installation_label(session: Session, installation_id: uuid.UUID | None) -> str:
    if installation_id is None:
        return "All workspaces"
    installation = session.get(Installation, installation_id)
    if installation is None:
        return "Unknown workspace"
    if installation.team_name:
        return str(installation.team_name)
    return str(installation.slack_team_id)


def _empty_tier_rows() -> tuple[LLMTierConfigRow, ...]:
    return tuple(
        LLMTierConfigRow(
            tier=tier.value,
            label=_tier_label(tier.value),
            description=_tier_description(tier.value),
            primary_assignment=None,
            primary_model=None,
            primary_provider=None,
            fallback_assignments=(),
            options=(),
            tone="warning",
        )
        for tier in CONFIG_TIERS
    )


def _tier_config_row(
    *,
    tier: str,
    assignments: Sequence[LLMTierAssignment],
    model_by_id: Mapping[uuid.UUID, LLMModelCatalog],
    provider_by_id: Mapping[uuid.UUID, LLMProviderAccount],
    options: Sequence[LLMModelConfigOption],
) -> LLMTierConfigRow:
    active_assignments = tuple(
        assignment
        for assignment in sorted(assignments, key=lambda row: row.priority)
        if assignment.is_active and assignment.model_catalog_id in model_by_id
    )
    joined: list[tuple[LLMTierAssignment, LLMModelCatalog, LLMProviderAccount]] = []
    for assignment in active_assignments:
        model = model_by_id[assignment.model_catalog_id]
        provider = provider_by_id.get(model.provider_account_id)
        if provider is None:
            continue
        joined.append((assignment, model, provider))
    primary = joined[0] if joined else None
    primary_assignment = primary[0] if primary else None
    primary_model = primary[1] if primary else None
    primary_provider = primary[2] if primary else None
    return LLMTierConfigRow(
        tier=tier,
        label=_tier_label(tier),
        description=_tier_description(tier),
        primary_assignment=primary_assignment,
        primary_model=primary_model,
        primary_provider=primary_provider,
        fallback_assignments=tuple(joined[1:]),
        options=tuple(options),
        tone=_tier_tone(primary_model, primary_provider),
    )


def _latest_llm_pricing_by_model(
    session: Session,
    provider_ids: Sequence[uuid.UUID],
) -> dict[tuple[uuid.UUID, str], LLMModelPricing]:
    if not provider_ids:
        return {}
    pricing_rows = tuple(
        session.scalars(
            select(LLMModelPricing)
            .where(LLMModelPricing.provider_account_id.in_(provider_ids))
            .order_by(LLMModelPricing.effective_from.desc())
        )
    )
    latest: dict[tuple[uuid.UUID, str], LLMModelPricing] = {}
    for pricing in pricing_rows:
        key = (pricing.provider_account_id, pricing.model_identifier)
        latest.setdefault(key, pricing)
    return latest


def _llm_model_config_row(
    *,
    model: LLMModelCatalog,
    provider: LLMProviderAccount,
    assignments: Sequence[LLMTierAssignment],
    latest_pricing: LLMModelPricing | None,
) -> LLMModelConfigRow:
    return LLMModelConfigRow(
        model=model,
        provider=provider,
        assignment_labels=tuple(
            _tier_assignment_label(assignment) for assignment in assignments
        ),
        latest_pricing=latest_pricing,
        tone="success" if model.is_enabled else "neutral",
        source_label=model.source.replace("_", " ").title(),
        pricing_label=_model_pricing_label(latest_pricing),
        pricing_detail=_model_pricing_detail(latest_pricing),
        context_label=_model_token_label(
            model,
            keys=("max_input_tokens", "context_length", "context_window"),
            fallback="Context unknown",
        ),
        output_label=_model_token_label(
            model,
            keys=("max_output_tokens", "max_tokens"),
            fallback="Output unknown",
        ),
        mode_label=_model_mode_label(model),
        capability_labels=_model_capability_labels(model),
    )


def _format_price_value(value: object) -> str:
    if value is None:
        return ""
    try:
        num = float(str(value))
        if num.is_integer():
            return f"{int(num):.2f}"

        s = f"{num:.6f}"
        s = s.rstrip("0")
        if s.endswith("."):
            s += "00"
        elif len(s.split(".")[1]) == 1:
            s += "0"
        return s
    except Exception:
        return str(value)


def _model_pricing_label(pricing: LLMModelPricing | None) -> str:
    if pricing is None:
        return "Missing pricing"
    input_price = (
        f"${_format_price_value(pricing.input_price_per_mtok)}"
        if pricing.input_price_per_mtok is not None
        else "? input"
    )
    output_price = (
        f"${_format_price_value(pricing.output_price_per_mtok)}"
        if pricing.output_price_per_mtok is not None
        else "? output"
    )
    return f"{input_price} in / {output_price} out"


def _model_pricing_detail(pricing: LLMModelPricing | None) -> str:
    if pricing is None:
        return "Cost tracking will show missing until pricing is synced or entered."
    return f"per 1M tokens · {pricing.currency}"


def _model_token_label(
    model: LLMModelCatalog,
    *,
    keys: Sequence[str],
    fallback: str,
) -> str:
    for key in keys:
        value = _model_metadata_value(model, key)
        numeric = _intish(value)
        if numeric is not None and numeric > 0:
            return f"{numeric:,} tokens"
    return fallback


def _model_mode_label(model: LLMModelCatalog) -> str:
    mode = _model_metadata_value(model, "mode")
    if isinstance(mode, str) and mode.strip():
        return mode.strip().replace("_", " ").title()
    input_modalities = _model_metadata_value(model, "input_modalities")
    output_modalities = _model_metadata_value(model, "output_modalities")
    if _contains_str(input_modalities, "text") and _contains_str(
        output_modalities, "text"
    ):
        return "Text"
    return "Mode unknown"


def _model_capability_labels(model: LLMModelCatalog) -> tuple[str, ...]:
    labels: list[str] = []
    supported_parameters = _model_metadata_value(model, "supported_parameters")
    input_modalities = _model_metadata_value(model, "input_modalities")
    output_modalities = _model_metadata_value(model, "output_modalities")
    capability_checks = (
        ("Tools", _truthy_model_value(model, "supports_function_calling")),
        (
            "Parallel tools",
            _truthy_model_value(model, "supports_parallel_function_calling"),
        ),
        ("Vision", _truthy_model_value(model, "supports_vision")),
        ("Structured output", _truthy_model_value(model, "supports_response_schema")),
        ("System prompts", _truthy_model_value(model, "supports_system_messages")),
    )
    for label, enabled in capability_checks:
        if enabled:
            labels.append(label)
    if _contains_str(supported_parameters, "tools"):
        labels.append("Tools")
    if _contains_str(supported_parameters, "response_format"):
        labels.append("Structured output")
    if _contains_str(input_modalities, "image"):
        labels.append("Vision")
    if _contains_str(input_modalities, "audio"):
        labels.append("Audio in")
    if _contains_str(output_modalities, "image"):
        labels.append("Image out")
    if _model_metadata_value(model, "runtime_routable") is False:
        labels.append("Not routable")
    unique = tuple(dict.fromkeys(labels))
    return unique[:5] if unique else ("Metadata pending",)


def _model_metadata_value(model: LLMModelCatalog, key: str) -> Any:
    capabilities = _mapping(model.capabilities_json)
    metadata = _mapping(model.metadata_json)
    litellm_metadata = _mapping(metadata.get("litellm_metadata"))
    openrouter_architecture = _mapping(litellm_metadata.get("openrouter_architecture"))
    for source in (capabilities, litellm_metadata, metadata, openrouter_architecture):
        value = source.get(key)
        if value is not None:
            return value
    return None


def _truthy_model_value(model: LLMModelCatalog, key: str) -> bool:
    return _model_metadata_value(model, key) is True


def _contains_str(value: object, expected: str) -> bool:
    if isinstance(value, str):
        return value == expected
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(item == expected for item in value)
    return False


def _intish(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _provider_credential_label(provider: LLMProviderAccount) -> str:
    metadata = (
        provider.metadata_json if isinstance(provider.metadata_json, dict) else {}
    )
    source = metadata.get("credential_source")
    if source == "env":
        return "Env managed"
    if provider.encrypted_secret_id is not None:
        return "Encrypted secret"
    return "Not stored"


def _provider_source_label(provider: LLMProviderAccount) -> str:
    metadata = (
        provider.metadata_json if isinstance(provider.metadata_json, dict) else {}
    )
    source = metadata.get("source") or metadata.get("credential_source")
    return str(source).replace("_", " ").title() if source else "Manual"


def _provider_status_tone(status: str) -> str:
    if status == "active":
        return "success"
    if status == "testing":
        return "warning"
    return "danger"


def _provider_health_tone(health_status: str) -> str:
    if health_status == "ok":
        return "success"
    if health_status == "degraded":
        return "warning"
    if health_status == "down":
        return "danger"
    return "neutral"


def _tier_label(tier: str) -> str:
    return tier.replace("_", " ").title()


def _tier_description(tier: str) -> str:
    descriptions = {
        "cheap_fast": "Fast path, acknowledgements, selection, and low-risk small tasks.",
        "standard": "Default worker reasoning for normal Slack requests.",
        "analysis": "Planner and deeper synthesis work where quality matters more.",
        "document": "Long-form document and artifact generation.",
        "high_reasoning": "Highest-effort planning, audits, and difficult reasoning.",
        "humanizer": "Final Slack response synthesis and tone polishing.",
        "vision": "Reading uploaded images and visual documents; image tasks route here.",
    }
    return descriptions.get(tier, "Model tier route.")


def _tier_catalog_options() -> tuple[LLMTierCatalogOption, ...]:
    return tuple(
        LLMTierCatalogOption(
            tier=tier.value,
            label=_tier_label(tier.value),
            description=_tier_description(tier.value),
        )
        for tier in CONFIG_TIERS
    )


def llm_tier_catalog_options() -> tuple[LLMTierCatalogOption, ...]:
    """Return runtime tier choices for provider catalog assignment controls."""

    return _tier_catalog_options()


def _tier_assignment_label(assignment: LLMTierAssignment) -> str:
    label = f"{_tier_label(assignment.tier)} P{assignment.priority}"
    if not assignment.is_active:
        return f"{label} inactive"
    return label


def _tier_tone(
    model: LLMModelCatalog | None,
    provider: LLMProviderAccount | None,
) -> str:
    if model is None or provider is None:
        return "warning"
    if not model.is_enabled or provider.status != "active":
        return "danger"
    return "success"


def _llm_config_source_label(
    providers: Sequence[LLMProviderAccount],
    runtime_settings: Settings | None,
    runtime_error: str | None,
) -> str:
    if providers:
        sources = {
            _provider_source_label(provider)
            for provider in providers
            if _provider_source_label(provider)
        }
        if len(sources) == 1:
            return next(iter(sources))
        return "DB Managed"
    if runtime_settings is not None:
        return "Env Ready"
    if runtime_error:
        return "Config Error"
    return "Unconfigured"


def _audit_config_row(audit: LLMConfigAudit) -> LLMAuditConfigRow:
    entity_label = audit.entity_type.replace("_", " ").title()
    action_label = audit.action.replace("_", " ").title()
    actor = audit.actor_slack_user_id or "dashboard"
    return LLMAuditConfigRow(
        audit=audit,
        label=f"{action_label} {entity_label}",
        detail=f"{actor} / {audit.entity_id or 'unknown'}",
        tone="danger" if audit.action in {"disable", "delete"} else "success",
    )


def _workspace_state_scope_filter(
    *,
    installation_id: uuid.UUID | None = None,
    slack_user_id: str | None = None,
) -> list[ColumnElement[bool]]:
    filters: list[ColumnElement[bool]] = []
    if installation_id is not None:
        filters.append(WorkspaceState.installation_id == installation_id)
    if slack_user_id:
        filters.append(WorkspaceState.scope_type == "user")
        filters.append(WorkspaceState.scope_id == slack_user_id)
    return filters


def _episode_scope_filter(
    *,
    installation_id: uuid.UUID | None = None,
    slack_user_id: str | None = None,
) -> list[ColumnElement[bool]]:
    filters: list[ColumnElement[bool]] = []
    if installation_id is not None:
        filters.append(Episode.installation_id == installation_id)
    if slack_user_id:
        filters.append(Episode.user_id == slack_user_id)
    return filters


def _aggregate_query(key: Any, filters: list[ColumnElement[bool]]) -> Select[Any]:
    return (
        select(
            key,
            func.count(LLMUsage.id),
            func.coalesce(func.sum(LLMUsage.input_tokens), 0),
            func.coalesce(func.sum(LLMUsage.output_tokens), 0),
            func.coalesce(func.sum(LLMUsage.cost_usd), 0),
        )
        .where(*filters)
        .group_by(key)
    )


def _aggregate_row(
    row: Row[Any] | tuple[Any, ...],
    label: IdentityLabel | None = None,
) -> AggregateRow:
    key, calls, input_tokens, output_tokens, cost_usd = row
    return AggregateRow(
        key=str(key),
        calls=int(calls),
        input_tokens=int(input_tokens),
        output_tokens=int(output_tokens),
        cost_usd=Decimal(cost_usd),
        label=label,
    )


def _daily_row(row: Row[Any]) -> DailyUsageRow:
    day_value, calls, input_tokens, output_tokens, cost_usd = row
    if isinstance(day_value, datetime):
        day = day_value.date()
    elif isinstance(day_value, date):
        day = day_value
    else:
        day = date.fromisoformat(str(day_value)[:10])
    return DailyUsageRow(
        day=day,
        calls=int(calls),
        input_tokens=int(input_tokens),
        output_tokens=int(output_tokens),
        cost_usd=Decimal(cost_usd),
    )


def _daily_task_row(row: Row[Any]) -> DailyTaskRow:
    day_value, task_count, failed_task_count = row
    if isinstance(day_value, datetime):
        day = day_value.date()
    elif isinstance(day_value, date):
        day = day_value
    else:
        day = date.fromisoformat(str(day_value)[:10])
    return DailyTaskRow(
        day=day,
        task_count=int(task_count),
        failed_task_count=int(failed_task_count),
    )


def _daily_cost_points(rows: Sequence[DailyUsageRow]) -> tuple[ChartPoint, ...]:
    ordered = tuple(reversed(rows))
    max_value = max((row.cost_usd for row in ordered), default=Decimal("0"))
    return tuple(
        ChartPoint(
            label=row.day.isoformat(),
            value_label=_format_money(row.cost_usd),
            percent=_percent_of(row.cost_usd, max_value),
        )
        for row in ordered
    )


def _daily_task_points(rows: Sequence[DailyTaskRow]) -> tuple[ChartPoint, ...]:
    ordered = tuple(reversed(rows))
    max_value = max((row.task_count for row in ordered), default=0)
    return tuple(
        ChartPoint(
            label=row.day.isoformat(),
            value_label=_format_number(row.task_count),
            percent=_percent_of(row.task_count, max_value),
            tone="danger" if row.failed_task_count else "accent",
            detail=(
                f"{_format_number(row.failed_task_count)} failed"
                if row.failed_task_count
                else "0 failed"
            ),
        )
        for row in ordered
    )


def _aggregate_bars(rows: Sequence[AggregateRow]) -> tuple[ChartBar, ...]:
    limited = tuple(rows[:8])
    max_value = max((row.cost_usd for row in limited), default=Decimal("0"))
    return tuple(
        ChartBar(
            label=row.display_key,
            secondary=row.secondary_key,
            value_label=_format_money(row.cost_usd),
            percent=_percent_of(row.cost_usd, max_value),
        )
        for row in limited
    )


def _percent_of(value: Decimal | int, max_value: Decimal | int) -> int:
    if value <= 0 or max_value <= 0:
        return 0
    percent = int((Decimal(value) / Decimal(max_value)) * 100)
    return max(4, min(100, percent))


def _format_money(value: Decimal) -> str:
    if value == 0:
        return "$0.00"
    if Decimal(0) < value < Decimal("0.01"):
        return "<$0.01"
    return f"${value:,.2f}"


def _format_number(value: int) -> str:
    return f"{value:,}"


# --- Consolidation (HIG-225) -------------------------------------------------


@dataclass(frozen=True, slots=True)
class ConsolidationConflictRow:
    """A user-confirmed-knowledge conflict flagged by the promotion pass."""

    run_id: uuid.UUID
    run_started_at: datetime
    entity_id: str
    canonical_key: str
    episode_id: str
    reason: str


@dataclass(frozen=True, slots=True)
class ConsolidationRunRow:
    """One consolidation run for the dashboard."""

    id: uuid.UUID
    installation_label: str
    started_at: datetime
    finished_at: datetime | None
    status: str
    status_tone: str
    promoted: int
    updated: int
    merged: int
    invalidated: int
    archived: int
    purged_observations: int
    profiles_refreshed: int
    style_cards_derived: int
    embedded: int
    conflict_count: int
    pass_errors: tuple[str, ...]
    cost_label: str


@dataclass(frozen=True, slots=True)
class ChannelStyleCardRow:
    """One channel's learned style card for the consolidation page."""

    profile_id: uuid.UUID
    channel: IdentityLabel
    has_card: bool
    formality: str | None
    brevity: str | None
    emoji_culture: str | None
    punctuation: str | None
    threading_norm: str | None
    notes: str | None
    common_phrases_label: str
    updated_at_label: str | None
    pinned_style: str | None


@dataclass(frozen=True, slots=True)
class ConsolidationDashboard:
    """Read model for the consolidation dashboard page."""

    runs: tuple[ConsolidationRunRow, ...]
    conflicts: tuple[ConsolidationConflictRow, ...]
    style_cards: tuple[ChannelStyleCardRow, ...]
    total_runs: int
    total_cost_label: str
    last_run_at: datetime | None


def get_consolidation_dashboard(
    session: Session,
    *,
    limit: int = 25,
) -> ConsolidationDashboard:
    """Recent consolidation runs with counters, cost, and flagged conflicts."""

    rows = list(
        session.scalars(
            select(ConsolidationRun)
            .order_by(ConsolidationRun.started_at.desc())
            .limit(limit)
        )
    )
    total_runs = int(
        session.scalar(select(func.count()).select_from(ConsolidationRun)) or 0
    )
    total_cost = session.scalar(
        select(func.coalesce(func.sum(ConsolidationRun.cost_usd), 0))
    )
    runs = tuple(_consolidation_run_row(session, run) for run in rows)
    conflicts: list[ConsolidationConflictRow] = []
    for run in rows:
        for conflict in _run_conflicts(run):
            conflicts.append(conflict)
    return ConsolidationDashboard(
        runs=runs,
        conflicts=tuple(conflicts[:50]),
        style_cards=_channel_style_card_rows(session),
        total_runs=total_runs,
        total_cost_label=_format_money(Decimal(total_cost or 0)),
        last_run_at=rows[0].started_at if rows else None,
    )


def _channel_style_card_rows(
    session: Session,
    *,
    limit: int = 50,
) -> tuple[ChannelStyleCardRow, ...]:
    profiles = list(
        session.scalars(
            select(ObserveChannelProfile)
            .where(ObserveChannelProfile.profile_status == "active")
            .order_by(ObserveChannelProfile.channel_id)
            .limit(limit)
        )
    )
    identities = _identity_map_from_keys(
        session,
        ((cp.installation_id, "channel", cp.channel_id) for cp in profiles),
    )
    rows: list[ChannelStyleCardRow] = []
    for profile in profiles:
        payload = profile.profile_json if isinstance(profile.profile_json, dict) else {}
        card = style_card_from_profile(payload)
        rows.append(
            ChannelStyleCardRow(
                profile_id=profile.id,
                channel=_identity_label(
                    identities,
                    installation_id=profile.installation_id,
                    kind="channel",
                    slack_id=profile.channel_id,
                ),
                has_card=card is not None,
                formality=card.formality if card else None,
                brevity=card.brevity if card else None,
                emoji_culture=card.emoji_culture if card else None,
                punctuation=card.punctuation if card else None,
                threading_norm=card.threading_norm if card else None,
                notes=(card.notes or None) if card else None,
                common_phrases_label=(", ".join(card.common_phrases) if card else ""),
                updated_at_label=_style_card_updated_label(
                    payload.get(STYLE_CARD_UPDATED_AT_KEY)
                ),
                pinned_style=pinned_style_from_profile(payload),
            )
        )
    return tuple(rows)


def _style_card_updated_label(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return value
    return parsed.strftime("%Y-%m-%d %H:%M UTC")


def _consolidation_run_row(
    session: Session, run: ConsolidationRun
) -> ConsolidationRunRow:
    counters = run.counters_json if isinstance(run.counters_json, dict) else {}
    pass_errors = counters.get("pass_errors")
    error_labels: list[str] = []
    if isinstance(pass_errors, dict):
        for name, message in sorted(pass_errors.items()):
            error_labels.append(f"{name}: {message}")
    if run.error:
        error_labels.append(run.error)
    conflicts = counters.get("conflicts")
    return ConsolidationRunRow(
        id=run.id,
        installation_label=_installation_label(session, run.installation_id),
        started_at=run.started_at,
        finished_at=run.finished_at,
        status=run.status,
        status_tone=_consolidation_status_tone(run.status),
        promoted=_counter_int(counters, "promoted"),
        updated=_counter_int(counters, "updated"),
        merged=_counter_int(counters, "merged"),
        invalidated=_counter_int(counters, "invalidated"),
        archived=_counter_int(counters, "archived"),
        purged_observations=_counter_int(counters, "purged_observations"),
        profiles_refreshed=_counter_int(counters, "profiles_refreshed"),
        style_cards_derived=_counter_int(counters, "style_cards_derived"),
        embedded=_counter_int(counters, "embedded"),
        conflict_count=len(conflicts) if isinstance(conflicts, list) else 0,
        pass_errors=tuple(error_labels),
        cost_label=_format_money(Decimal(run.cost_usd or 0)),
    )


def _run_conflicts(run: ConsolidationRun) -> tuple[ConsolidationConflictRow, ...]:
    counters = run.counters_json if isinstance(run.counters_json, dict) else {}
    raw_conflicts = counters.get("conflicts")
    if not isinstance(raw_conflicts, list):
        return ()
    rows: list[ConsolidationConflictRow] = []
    for item in raw_conflicts:
        if not isinstance(item, dict):
            continue
        rows.append(
            ConsolidationConflictRow(
                run_id=run.id,
                run_started_at=run.started_at,
                entity_id=str(item.get("entity_id", "")),
                canonical_key=str(item.get("canonical_key", "")),
                episode_id=str(item.get("episode_id", "")),
                reason=str(item.get("reason", "")),
            )
        )
    return tuple(rows)


def _counter_int(counters: Mapping[str, Any], key: str) -> int:
    value = counters.get(key)
    return value if isinstance(value, int) else 0


def _consolidation_status_tone(status: str) -> str:
    if status == "succeeded":
        return "success"
    if status == "failed":
        return "danger"
    return "neutral"
