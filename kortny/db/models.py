"""
Kortny database models.

Postgres-specific types are used throughout (UUID, JSONB, BYTEA, timestamptz,
native enums), so the canonical path to a live database is the Alembic
migration, not Base.metadata.create_all().
"""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import BYTEA, ENUM, JSONB, TIMESTAMP, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


# --- Enums (native Postgres enum types) -------------------------------------


class TaskStatus(StrEnum):
    pending = "pending"
    running = "running"
    waiting_approval = "waiting_approval"
    succeeded = "succeeded"
    failed = "failed"  # TERMINAL: agent error, or crashes exhausted attempts
    crashed = "crashed"  # TRANSIENT: reclaim sweep found an expired lease
    cancelled = "cancelled"


class LLMProvider(StrEnum):
    openai = "openai"
    anthropic = "anthropic"
    openrouter = "openrouter"


class TaskEventType(StrEnum):
    task_created = "task_created"
    status_changed = "status_changed"
    llm_call = "llm_call"
    tool_call = "tool_call"
    tool_result = "tool_result"
    artifact_created = "artifact_created"
    message_posted = "message_posted"
    error = "error"
    log = "log"


def _pg_enum(py_enum: type[StrEnum], name: str) -> ENUM:
    # create_type=False: the migration creates the type; columns just reference it.
    return ENUM(
        py_enum,
        name=name,
        create_type=False,
        values_callable=lambda obj: [e.value for e in obj],
    )


TASK_STATUS = _pg_enum(TaskStatus, "task_status")
LLM_PROVIDER = _pg_enum(LLMProvider, "llm_provider")
TASK_EVENT_TYPE = _pg_enum(TaskEventType, "task_event_type")

TZ = TIMESTAMP(timezone=True)


# --- Tables -----------------------------------------------------------------


class Installation(Base):
    __tablename__ = "installations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    slack_team_id: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    team_name: Mapped[str | None] = mapped_column(String)
    bot_user_id: Mapped[str | None] = mapped_column(String)
    # The Slack user who installed Kortny (first captured at the install-intro
    # DM). Ambient passes — e.g. the org-profile proposer (HIG-271) — have no
    # originating user, so they DM this admin to confirm workspace-level facts.
    primary_admin_user_id: Mapped[str | None] = mapped_column(String)
    digest_enabled_at: Mapped[datetime | None] = mapped_column(TZ)
    autopilot_enabled: Mapped[bool | None] = mapped_column(Boolean)
    created_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now(), onupdate=func.now()
    )


class EncryptedSecret(Base):
    __tablename__ = "encrypted_secrets"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    installation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("installations.id"), nullable=False
    )
    secret_type: Mapped[str] = mapped_column(String, nullable=False)
    ciphertext: Mapped[bytes] = mapped_column(BYTEA, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        UniqueConstraint("installation_id", "secret_type", name="idx_secret_lookup"),
    )


class LLMProviderAccount(Base):
    __tablename__ = "llm_provider_accounts"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    installation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("installations.id", ondelete="CASCADE"), nullable=False
    )
    provider_kind: Mapped[str] = mapped_column(String, nullable=False)
    display_name: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(
        String, nullable=False, server_default=text("'active'")
    )
    health_status: Mapped[str] = mapped_column(
        String, nullable=False, server_default=text("'unknown'")
    )
    base_url: Mapped[str | None] = mapped_column(Text)
    encrypted_secret_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("encrypted_secrets.id", ondelete="SET NULL")
    )
    metadata_json: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    created_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        CheckConstraint(
            "status in ('active', 'disabled', 'testing')",
            name="ck_llm_provider_accounts_status",
        ),
        CheckConstraint(
            "health_status in ('ok', 'degraded', 'down', 'unknown')",
            name="ck_llm_provider_accounts_health_status",
        ),
        Index(
            "idx_llm_provider_accounts_installation",
            "installation_id",
            "status",
        ),
        Index(
            "idx_llm_provider_accounts_kind",
            "installation_id",
            "provider_kind",
        ),
    )


class LLMModelCatalog(Base):
    __tablename__ = "llm_model_catalog"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    provider_account_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("llm_provider_accounts.id", ondelete="CASCADE"), nullable=False
    )
    model_identifier: Mapped[str] = mapped_column(String, nullable=False)
    display_name: Mapped[str] = mapped_column(String, nullable=False)
    is_enabled: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("true")
    )
    capabilities_json: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    source: Mapped[str] = mapped_column(
        String, nullable=False, server_default=text("'manual'")
    )
    metadata_json: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    created_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        CheckConstraint(
            "source in ('manual', 'env_bootstrap', 'litellm_catalog', 'provider_api')",
            name="ck_llm_model_catalog_source",
        ),
        UniqueConstraint(
            "provider_account_id",
            "model_identifier",
            name="idx_llm_model_catalog_provider_model",
        ),
        Index("idx_llm_model_catalog_enabled", "provider_account_id", "is_enabled"),
    )


class LLMTierAssignment(Base):
    __tablename__ = "llm_tier_assignments"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    installation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("installations.id", ondelete="CASCADE"), nullable=False
    )
    tier: Mapped[str] = mapped_column(String, nullable=False)
    model_catalog_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("llm_model_catalog.id", ondelete="CASCADE"), nullable=False
    )
    priority: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("1")
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("true")
    )
    routing_json: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    created_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        CheckConstraint(
            "tier in ('cheap_fast', 'standard', 'analysis', 'document', "
            "'high_reasoning', 'humanizer', 'vision')",
            name="ck_llm_tier_assignments_tier",
        ),
        CheckConstraint("priority >= 1", name="ck_llm_tier_assignments_priority"),
        UniqueConstraint(
            "installation_id",
            "tier",
            "priority",
            name="idx_llm_tier_assignment_priority",
        ),
        Index(
            "idx_llm_tier_assignments_active",
            "installation_id",
            "tier",
            "is_active",
        ),
    )


class LLMModelPricing(Base):
    __tablename__ = "llm_model_pricing"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    provider_account_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("llm_provider_accounts.id", ondelete="CASCADE"), nullable=False
    )
    model_identifier: Mapped[str] = mapped_column(String, nullable=False)
    input_price_per_mtok: Mapped[Decimal | None] = mapped_column(Numeric(12, 6))
    output_price_per_mtok: Mapped[Decimal | None] = mapped_column(Numeric(12, 6))
    currency: Mapped[str] = mapped_column(
        String, nullable=False, server_default=text("'USD'")
    )
    pricing_source: Mapped[str] = mapped_column(String, nullable=False)
    effective_from: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now()
    )
    metadata_json: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    created_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint(
            "provider_account_id",
            "model_identifier",
            "effective_from",
            name="idx_llm_model_pricing_lookup",
        ),
        Index("idx_llm_model_pricing_model", "provider_account_id", "model_identifier"),
    )


class LLMBudgetPolicy(Base):
    __tablename__ = "llm_budget_policies"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    installation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("installations.id", ondelete="CASCADE"), nullable=False
    )
    tier: Mapped[str | None] = mapped_column(String)
    daily_budget_usd: Mapped[Decimal | None] = mapped_column(Numeric(12, 2))
    monthly_budget_usd: Mapped[Decimal | None] = mapped_column(Numeric(12, 2))
    alert_threshold_pct: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("80")
    )
    behavior: Mapped[str] = mapped_column(
        String, nullable=False, server_default=text("'soft_stop'")
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("true")
    )
    metadata_json: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    created_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        CheckConstraint(
            "tier is null or tier in ('cheap_fast', 'standard', 'analysis', "
            "'document', 'high_reasoning', 'humanizer', 'vision')",
            name="ck_llm_budget_policies_tier",
        ),
        CheckConstraint(
            "alert_threshold_pct between 1 and 100",
            name="ck_llm_budget_policies_alert_threshold",
        ),
        CheckConstraint(
            "behavior in ('soft_stop', 'hard_stop')",
            name="ck_llm_budget_policies_behavior",
        ),
        Index("idx_llm_budget_policies_installation", "installation_id", "is_active"),
    )


class LLMConfigAudit(Base):
    __tablename__ = "llm_config_audit"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    installation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("installations.id", ondelete="CASCADE"), nullable=False
    )
    actor_slack_user_id: Mapped[str | None] = mapped_column(String)
    action: Mapped[str] = mapped_column(String, nullable=False)
    entity_type: Mapped[str] = mapped_column(String, nullable=False)
    entity_id: Mapped[str | None] = mapped_column(String)
    previous_value: Mapped[dict | None] = mapped_column(JSONB)
    new_value: Mapped[dict | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now()
    )

    __table_args__ = (
        CheckConstraint(
            "action in ('create', 'update', 'disable', 'enable', 'delete', "
            "'bootstrap')",
            name="ck_llm_config_audit_action",
        ),
        Index("idx_llm_config_audit_installation", "installation_id", "created_at"),
    )


class Task(Base):
    __tablename__ = "tasks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    installation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("installations.id"), nullable=False
    )
    parent_task_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("tasks.id"))

    # Slack trigger context
    slack_event_id: Mapped[str | None] = mapped_column(String, unique=True)
    slack_channel_id: Mapped[str] = mapped_column(String, nullable=False)
    slack_thread_ts: Mapped[str | None] = mapped_column(String)
    slack_message_ts: Mapped[str | None] = mapped_column(String)
    slack_user_id: Mapped[str] = mapped_column(String, nullable=False)
    identity_kind: Mapped[str | None] = mapped_column(String)
    identity_key: Mapped[str | None] = mapped_column(String)
    identity_payload: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    identity_fingerprint: Mapped[str | None] = mapped_column(String)

    # Request + result
    input: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[TaskStatus] = mapped_column(
        TASK_STATUS, nullable=False, server_default=text("'pending'::task_status")
    )
    result_summary: Mapped[str | None] = mapped_column(Text)
    error: Mapped[dict | None] = mapped_column(JSONB)
    # Routing outcome label computed at completion (HIG-221 learning loop):
    # clean | recovered | partial | failed | cancelled, + a 0..1 score.
    routing_quality: Mapped[str | None] = mapped_column(String)
    routing_quality_score: Mapped[Decimal | None] = mapped_column(Numeric(4, 3))

    # Queue / lease
    available_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now()
    )
    locked_by: Mapped[str | None] = mapped_column(String)
    locked_at: Mapped[datetime | None] = mapped_column(TZ)
    lease_expires_at: Mapped[datetime | None] = mapped_column(TZ)
    attempts: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    max_attempts: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("3")
    )

    # Cost rollup (denormalized cache of llm_usage)
    total_input_tokens: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    total_output_tokens: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    total_cost_usd: Mapped[Decimal] = mapped_column(
        Numeric(12, 6), nullable=False, server_default=text("0")
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now()
    )
    started_at: Mapped[datetime | None] = mapped_column(TZ)
    finished_at: Mapped[datetime | None] = mapped_column(TZ)
    updated_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        CheckConstraint(
            "identity_kind is null or identity_kind in "
            "('slack_message', 'slack_event', 'synthetic', 'scheduled', 'manual')",
            name="ck_tasks_identity_kind",
        ),
        Index(
            "idx_tasks_identity_unique",
            "installation_id",
            "identity_key",
            unique=True,
            postgresql_where=text("identity_key IS NOT NULL"),
        ),
        Index("idx_tasks_claim", "status", "available_at"),
        Index("idx_tasks_history", "installation_id", "created_at"),
        Index("idx_tasks_thread", "slack_channel_id", "slack_thread_ts"),
    )


class TaskEvent(Base):
    __tablename__ = "task_events"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    task_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tasks.id"), nullable=False)
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    type: Mapped[TaskEventType] = mapped_column(TASK_EVENT_TYPE, nullable=False)
    payload: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    created_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now()
    )

    __table_args__ = (UniqueConstraint("task_id", "seq", name="idx_events_task_seq"),)


class SlackInboundEvent(Base):
    __tablename__ = "slack_inbound_events"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    installation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("installations.id", ondelete="CASCADE"), nullable=False
    )
    slack_team_id: Mapped[str] = mapped_column(String, nullable=False)
    slack_event_id: Mapped[str | None] = mapped_column(String)
    event_type: Mapped[str] = mapped_column(String, nullable=False)
    event_subtype: Mapped[str | None] = mapped_column(String)
    surface: Mapped[str] = mapped_column(String, nullable=False)
    channel_id: Mapped[str | None] = mapped_column(String)
    user_id: Mapped[str | None] = mapped_column(String)
    message_ts: Mapped[str | None] = mapped_column(String)
    thread_ts: Mapped[str | None] = mapped_column(String)
    event_time: Mapped[datetime | None] = mapped_column(TZ)
    retry_num: Mapped[int | None] = mapped_column(Integer)
    retry_reason: Mapped[str | None] = mapped_column(String)
    raw_body: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    raw_event: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    processing_status: Mapped[str] = mapped_column(
        String, nullable=False, server_default=text("'received'")
    )
    processing_attempts: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    task_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("tasks.id", ondelete="SET NULL")
    )
    observation_event_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("observation_events.id", ondelete="SET NULL")
    )
    metadata_json: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    last_error: Mapped[dict | None] = mapped_column(JSONB)
    received_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now()
    )
    processed_at: Mapped[datetime | None] = mapped_column(TZ)
    created_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        CheckConstraint(
            "processing_status in "
            "('received', 'ignored', 'task_created', 'observed', "
            "'failed', 'dead_lettered', 'replayed')",
            name="ck_slack_inbound_events_processing_status",
        ),
        Index(
            "idx_slack_inbound_events_event_unique",
            "installation_id",
            "slack_event_id",
            unique=True,
            postgresql_where=text("slack_event_id IS NOT NULL"),
        ),
        Index(
            "idx_slack_inbound_events_status",
            "installation_id",
            "processing_status",
            "received_at",
        ),
        Index(
            "idx_slack_inbound_events_channel",
            "installation_id",
            "channel_id",
            "received_at",
        ),
        Index("idx_slack_inbound_events_task", "task_id"),
        Index("idx_slack_inbound_events_observation", "observation_event_id"),
    )


class SlackSideEffect(Base):
    __tablename__ = "slack_side_effects"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    installation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("installations.id", ondelete="CASCADE"), nullable=False
    )
    task_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("tasks.id", ondelete="SET NULL")
    )
    idempotency_key: Mapped[str] = mapped_column(String, nullable=False)
    operation: Mapped[str] = mapped_column(String, nullable=False)
    purpose: Mapped[str | None] = mapped_column(String)
    target_channel_id: Mapped[str | None] = mapped_column(String)
    target_thread_ts: Mapped[str | None] = mapped_column(String)
    target_message_ts: Mapped[str | None] = mapped_column(String)
    request_json: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    response_json: Mapped[dict | None] = mapped_column(JSONB)
    status: Mapped[str] = mapped_column(
        String, nullable=False, server_default=text("'pending'")
    )
    attempts: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    last_error: Mapped[dict | None] = mapped_column(JSONB)
    available_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now()
    )
    started_at: Mapped[datetime | None] = mapped_column(TZ)
    delivered_at: Mapped[datetime | None] = mapped_column(TZ)
    created_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        CheckConstraint(
            "operation in "
            "('chat_postMessage', 'files_upload_v2', 'reactions_add', "
            "'reactions_remove', 'pins_add', 'bookmarks_add', "
            "'conversations_canvases_create', 'canvases_edit')",
            name="ck_slack_side_effects_operation",
        ),
        CheckConstraint(
            "status in ('pending', 'in_progress', 'succeeded', 'failed')",
            name="ck_slack_side_effects_status",
        ),
        UniqueConstraint(
            "installation_id",
            "idempotency_key",
            name="idx_slack_side_effects_idempotency",
        ),
        Index(
            "idx_slack_side_effects_status",
            "installation_id",
            "status",
            "available_at",
        ),
        Index("idx_slack_side_effects_task", "task_id", "created_at"),
        Index(
            "idx_slack_side_effects_target",
            "installation_id",
            "target_channel_id",
            "target_message_ts",
        ),
    )


class Schedule(Base):
    __tablename__ = "schedules"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    installation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("installations.id", ondelete="CASCADE"), nullable=False
    )
    owner_type: Mapped[str] = mapped_column(String, nullable=False)
    owner_slack_user_id: Mapped[str | None] = mapped_column(String)
    title: Mapped[str] = mapped_column(String, nullable=False)
    spec_kind: Mapped[str] = mapped_column(String, nullable=False)
    cron_expr: Mapped[str | None] = mapped_column(String)
    interval_seconds: Mapped[int | None] = mapped_column(Integer)
    run_at: Mapped[datetime | None] = mapped_column(TZ)
    timezone: Mapped[str] = mapped_column(
        String, nullable=False, server_default=text("'UTC'")
    )
    next_run_at: Mapped[datetime | None] = mapped_column(TZ)
    last_run_at: Mapped[datetime | None] = mapped_column(TZ)
    catchup_policy: Mapped[str] = mapped_column(
        String, nullable=False, server_default=text("'skip'")
    )
    catchup_window_seconds: Mapped[int | None] = mapped_column(Integer)
    overlap_policy: Mapped[str] = mapped_column(
        String, nullable=False, server_default=text("'skip'")
    )
    status: Mapped[str] = mapped_column(
        String, nullable=False, server_default=text("'active'")
    )
    task_template: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    delivery_kind: Mapped[str] = mapped_column(
        String, nullable=False, server_default=text("'slack_dm'")
    )
    delivery_slack_user_id: Mapped[str | None] = mapped_column(String)
    delivery_slack_channel_id: Mapped[str | None] = mapped_column(String)
    delivery_slack_thread_ts: Mapped[str | None] = mapped_column(String)
    artifact_delivery_policy: Mapped[str] = mapped_column(
        String, nullable=False, server_default=text("'message_only'")
    )
    planned_cost_ceiling_usd: Mapped[Decimal | None] = mapped_column(Numeric(10, 4))
    idempotency_key_template: Mapped[str | None] = mapped_column(String)
    created_by_slack_user_id: Mapped[str | None] = mapped_column(String)
    metadata_json: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    created_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        CheckConstraint(
            "owner_type in ('user', 'system')",
            name="ck_schedules_owner_type",
        ),
        CheckConstraint(
            "(owner_type = 'user' and owner_slack_user_id is not null) or "
            "(owner_type = 'system')",
            name="ck_schedules_owner",
        ),
        CheckConstraint(
            "spec_kind in ('oneoff', 'interval', 'cron')",
            name="ck_schedules_spec_kind",
        ),
        CheckConstraint(
            "(spec_kind = 'oneoff' and run_at is not null) or "
            "(spec_kind = 'interval' and interval_seconds is not null and "
            "interval_seconds > 0) or "
            "(spec_kind = 'cron' and cron_expr is not null and cron_expr <> '')",
            name="ck_schedules_spec",
        ),
        CheckConstraint(
            "catchup_policy in ('skip', 'run_once', 'backfill')",
            name="ck_schedules_catchup_policy",
        ),
        CheckConstraint(
            "catchup_window_seconds is null or catchup_window_seconds >= 0",
            name="ck_schedules_catchup_window",
        ),
        CheckConstraint(
            "overlap_policy in ('skip', 'allow')",
            name="ck_schedules_overlap_policy",
        ),
        CheckConstraint(
            "status in ('proposed', 'active', 'paused', 'completed', 'cancelled')",
            name="ck_schedules_status",
        ),
        CheckConstraint(
            "planned_cost_ceiling_usd is null or planned_cost_ceiling_usd > 0",
            name="ck_schedules_cost_ceiling",
        ),
        CheckConstraint(
            "delivery_kind in "
            "('slack_dm', 'slack_channel', 'slack_thread', 'dashboard_only')",
            name="ck_schedules_delivery_kind",
        ),
        CheckConstraint(
            "artifact_delivery_policy in "
            "('message_only', 'attach_files', 'link_artifacts')",
            name="ck_schedules_artifact_delivery_policy",
        ),
        Index(
            "idx_schedules_due",
            "next_run_at",
            postgresql_where=text("status = 'active'"),
        ),
        Index(
            "idx_schedules_owner",
            "installation_id",
            "owner_type",
            "owner_slack_user_id",
            "status",
        ),
        Index(
            "idx_schedules_status",
            "installation_id",
            "status",
            "next_run_at",
        ),
    )


class WorkspaceState(Base):
    __tablename__ = "workspace_state"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    installation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("installations.id", ondelete="CASCADE"), nullable=False
    )
    scope_type: Mapped[str] = mapped_column(String, nullable=False)
    scope_id: Mapped[str | None] = mapped_column(String)
    key: Mapped[str] = mapped_column(String, nullable=False)
    value_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    value_text: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String, nullable=False)
    source_kind: Mapped[str] = mapped_column(String, nullable=False)
    source_task_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("tasks.id", ondelete="SET NULL")
    )
    source_event_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("task_events.id", ondelete="SET NULL")
    )
    source_slack_channel_id: Mapped[str | None] = mapped_column(String)
    source_slack_message_ts: Mapped[str | None] = mapped_column(String)
    source_slack_file_id: Mapped[str | None] = mapped_column(String)
    source_url: Mapped[str | None] = mapped_column(Text)
    proposed_by: Mapped[str] = mapped_column(String, nullable=False)
    proposed_reason: Mapped[str | None] = mapped_column(Text)
    confidence_score: Mapped[Decimal | None] = mapped_column(Numeric(4, 3))
    confidence_reason: Mapped[str | None] = mapped_column(Text)
    confirmed_by_user_id: Mapped[str | None] = mapped_column(String)
    confirmed_at: Mapped[datetime | None] = mapped_column(TZ)
    rejected_by_user_id: Mapped[str | None] = mapped_column(String)
    rejected_at: Mapped[datetime | None] = mapped_column(TZ)
    superseded_by_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("workspace_state.id")
    )
    superseded_at: Mapped[datetime | None] = mapped_column(TZ)
    forgotten_by_user_id: Mapped[str | None] = mapped_column(String)
    forgotten_at: Mapped[datetime | None] = mapped_column(TZ)
    expires_at: Mapped[datetime | None] = mapped_column(TZ)
    created_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        CheckConstraint(
            "scope_type in ('workspace', 'channel', 'user')",
            name="ck_workspace_state_scope_type",
        ),
        CheckConstraint(
            "status in ('proposed', 'active', 'rejected', 'superseded', 'forgotten')",
            name="ck_workspace_state_status",
        ),
        CheckConstraint(
            "source_kind in "
            "('user_explicit', 'agent_proposed', 'summarizer_proposed', "
            "'observer_proposed', 'import')",
            name="ck_workspace_state_source_kind",
        ),
        CheckConstraint(
            "(scope_type = 'workspace' and scope_id is null) or "
            "(scope_type in ('channel', 'user') and scope_id is not null)",
            name="ck_workspace_state_scope_id",
        ),
        Index(
            "idx_workspace_state_active_unique",
            "installation_id",
            "scope_type",
            text("coalesce(scope_id, '')"),
            "key",
            unique=True,
            postgresql_where=text("status = 'active' AND expires_at IS NULL"),
        ),
        Index(
            "idx_workspace_state_active_lookup",
            "installation_id",
            "status",
            "scope_type",
            "scope_id",
        ),
        Index(
            "idx_workspace_state_source",
            "source_task_id",
            "source_event_id",
        ),
        Index("idx_workspace_state_expires_at", "expires_at"),
    )


class SlackIdentity(Base):
    __tablename__ = "slack_identities"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    installation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("installations.id", ondelete="CASCADE"), nullable=False
    )
    kind: Mapped[str] = mapped_column(String, nullable=False)
    slack_id: Mapped[str] = mapped_column(String, nullable=False)
    display_name: Mapped[str] = mapped_column(String, nullable=False)
    raw_name: Mapped[str | None] = mapped_column(String)
    is_deleted: Mapped[bool] = mapped_column(
        nullable=False, server_default=text("false")
    )
    is_bot: Mapped[bool] = mapped_column(nullable=False, server_default=text("false"))
    is_private: Mapped[bool] = mapped_column(
        nullable=False, server_default=text("false")
    )
    raw_json: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    last_seen_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now()
    )
    refreshed_at: Mapped[datetime | None] = mapped_column(TZ)
    created_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        CheckConstraint("kind in ('user', 'channel')", name="ck_slack_identity_kind"),
        UniqueConstraint(
            "installation_id",
            "kind",
            "slack_id",
            name="idx_slack_identity_unique",
        ),
        Index("idx_slack_identity_lookup", "installation_id", "kind", "slack_id"),
        Index("idx_slack_identity_seen", "installation_id", "kind", "last_seen_at"),
    )


class SlackChannelMembership(Base):
    __tablename__ = "slack_channel_memberships"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    installation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("installations.id", ondelete="CASCADE"), nullable=False
    )
    channel_id: Mapped[str] = mapped_column(String, nullable=False)
    channel_name: Mapped[str | None] = mapped_column(String)
    channel_type: Mapped[str | None] = mapped_column(String)
    membership_status: Mapped[str] = mapped_column(
        String, nullable=False, server_default=text("'active'")
    )
    discovered_via: Mapped[str] = mapped_column(String, nullable=False)
    added_by_user_id: Mapped[str | None] = mapped_column(String)
    first_seen_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now()
    )
    last_seen_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now()
    )
    onboarding_status: Mapped[str] = mapped_column(
        String, nullable=False, server_default=text("'pending'")
    )
    onboarding_posted_at: Mapped[datetime | None] = mapped_column(TZ)
    onboarding_message_ts: Mapped[str | None] = mapped_column(String)
    last_event_id: Mapped[str | None] = mapped_column(String)
    metadata_json: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    created_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        CheckConstraint(
            "membership_status in ('active', 'left', 'unknown')",
            name="ck_slack_channel_memberships_status",
        ),
        CheckConstraint(
            "discovered_via in "
            "('member_joined_channel', 'app_mention', 'message_observation', "
            "'channel_history', 'manual_backfill')",
            name="ck_slack_channel_memberships_discovered_via",
        ),
        CheckConstraint(
            "onboarding_status in ('pending', 'posted', 'skipped')",
            name="ck_slack_channel_memberships_onboarding_status",
        ),
        UniqueConstraint(
            "installation_id",
            "channel_id",
            name="idx_slack_channel_memberships_unique",
        ),
        Index(
            "idx_slack_channel_memberships_lookup",
            "installation_id",
            "channel_id",
        ),
        Index(
            "idx_slack_channel_memberships_status",
            "installation_id",
            "membership_status",
            "last_seen_at",
        ),
        Index(
            "idx_slack_channel_memberships_onboarding",
            "installation_id",
            "onboarding_status",
            "last_seen_at",
        ),
    )


class DashboardUser(Base):
    __tablename__ = "dashboard_users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    installation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("installations.id", ondelete="CASCADE"), nullable=False
    )
    slack_user_id: Mapped[str] = mapped_column(String, nullable=False)
    email: Mapped[str | None] = mapped_column(String)
    display_name: Mapped[str] = mapped_column(String, nullable=False)
    avatar_url: Mapped[str | None] = mapped_column(String)
    role: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    last_login_at: Mapped[datetime | None] = mapped_column(TZ)
    created_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        CheckConstraint(
            "role in ('admin', 'member')",
            name="ck_dashboard_users_role",
        ),
        CheckConstraint(
            "status in ('active', 'disabled')",
            name="ck_dashboard_users_status",
        ),
        UniqueConstraint(
            "installation_id",
            "slack_user_id",
            name="idx_dashboard_users_slack_user_unique",
        ),
        Index("idx_dashboard_users_installation_role", "installation_id", "role"),
        Index("idx_dashboard_users_status", "installation_id", "status"),
        Index(
            "idx_dashboard_users_email_unique",
            "installation_id",
            "email",
            unique=True,
            postgresql_where=text("email IS NOT NULL"),
        ),
    )


class DashboardOAuthState(Base):
    __tablename__ = "dashboard_oauth_states"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    provider: Mapped[str] = mapped_column(String, nullable=False)
    state: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    redirect_path: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now()
    )
    expires_at: Mapped[datetime] = mapped_column(TZ, nullable=False)
    used_at: Mapped[datetime | None] = mapped_column(TZ)

    __table_args__ = (
        CheckConstraint(
            "provider in ('slack')",
            name="ck_dashboard_oauth_states_provider",
        ),
        Index(
            "idx_dashboard_oauth_states_lookup",
            "provider",
            "state",
            "expires_at",
        ),
    )


class ComposioConnection(Base):
    __tablename__ = "composio_connections"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    installation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("installations.id", ondelete="CASCADE"), nullable=False
    )
    toolkit_slug: Mapped[str] = mapped_column(String, nullable=False)
    auth_config_id: Mapped[str | None] = mapped_column(String)
    connected_account_id: Mapped[str | None] = mapped_column(String)
    connection_request_id: Mapped[str | None] = mapped_column(String)
    composio_user_id: Mapped[str] = mapped_column(String, nullable=False)
    owner_slack_user_id: Mapped[str] = mapped_column(String, nullable=False)
    visibility_scope_type: Mapped[str] = mapped_column(String, nullable=False)
    visibility_scope_id: Mapped[str | None] = mapped_column(String)
    status: Mapped[str] = mapped_column(String, nullable=False)
    display_name: Mapped[str | None] = mapped_column(String)
    external_account_label: Mapped[str | None] = mapped_column(String)
    metadata_json: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    created_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        CheckConstraint(
            "visibility_scope_type in ('workspace', 'channel', 'user')",
            name="ck_composio_connections_visibility_scope_type",
        ),
        CheckConstraint(
            "status in ('pending', 'active', 'expired', 'failed', 'disabled')",
            name="ck_composio_connections_status",
        ),
        CheckConstraint(
            "(visibility_scope_type = 'workspace' and visibility_scope_id is null) or "
            "(visibility_scope_type in ('channel', 'user') and visibility_scope_id is not null)",
            name="ck_composio_connections_visibility_scope_id",
        ),
        Index(
            "idx_composio_connections_connected_account",
            "installation_id",
            "connected_account_id",
            unique=True,
            postgresql_where=text("connected_account_id IS NOT NULL"),
        ),
        Index(
            "idx_composio_connections_allowed_lookup",
            "installation_id",
            "status",
            "visibility_scope_type",
            "visibility_scope_id",
        ),
        Index(
            "idx_composio_connections_owner",
            "installation_id",
            "owner_slack_user_id",
        ),
        Index(
            "idx_composio_connections_toolkit",
            "installation_id",
            "toolkit_slug",
            "status",
        ),
    )


class ComposioToolCard(Base):
    """One synced Composio tool card for an installation (HIG-222).

    The full catalog of every connected toolkit is synced into this table by
    ``kortny.composio.catalog_sync`` so per-task tool retrieval is purely
    semantic (rank over ``tool_embeddings`` kind ``tool_card``) with no hot-path
    Composio HTTP for candidate listing. ``side_effect`` is the verb-mapped
    coarse class (read/write/destructive); full input schemas are fetched lazily
    only for tools that survive selection, so they are intentionally not stored
    here. ``card_sha`` gates re-embedding the same way the embedding index does.
    """

    __tablename__ = "composio_tool_cards"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    installation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("installations.id", ondelete="CASCADE"), nullable=False
    )
    toolkit_slug: Mapped[str] = mapped_column(String, nullable=False)
    tool_slug: Mapped[str] = mapped_column(String, nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    side_effect: Mapped[str] = mapped_column(String, nullable=False)
    card_sha: Mapped[str] = mapped_column(String(64), nullable=False)
    synced_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now()
    )
    created_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        CheckConstraint(
            "side_effect in ('read', 'write', 'destructive')",
            name="ck_composio_tool_cards_side_effect",
        ),
        UniqueConstraint(
            "installation_id",
            "tool_slug",
            name="uq_composio_tool_cards_installation_tool",
        ),
        Index(
            "idx_composio_tool_cards_toolkit",
            "installation_id",
            "toolkit_slug",
        ),
    )


class ObservePolicy(Base):
    __tablename__ = "observe_policies"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    installation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("installations.id", ondelete="CASCADE"), nullable=False
    )
    scope_type: Mapped[str] = mapped_column(String, nullable=False)
    scope_id: Mapped[str | None] = mapped_column(String)
    observation_status: Mapped[str] = mapped_column(String, nullable=False)
    proactivity_status: Mapped[str] = mapped_column(String, nullable=False)
    retention_days: Mapped[int | None] = mapped_column(Integer)
    quiet_hours_json: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    cooldown_seconds: Mapped[int | None] = mapped_column(Integer)
    enabled_by_user_id: Mapped[str | None] = mapped_column(String)
    enabled_at: Mapped[datetime | None] = mapped_column(TZ)
    paused_by_user_id: Mapped[str | None] = mapped_column(String)
    paused_at: Mapped[datetime | None] = mapped_column(TZ)
    pause_reason: Mapped[str | None] = mapped_column(Text)
    full_enabled_at: Mapped[datetime | None] = mapped_column(TZ)
    metadata_json: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    created_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        CheckConstraint(
            "scope_type in ('workspace', 'channel', 'user')",
            name="ck_observe_policies_scope_type",
        ),
        CheckConstraint(
            "observation_status in ('off', 'passive', 'active')",
            name="ck_observe_policies_observation_status",
        ),
        CheckConstraint(
            "proactivity_status in ('off', 'digest_only', 'full')",
            name="ck_observe_policies_proactivity_status",
        ),
        CheckConstraint(
            "(scope_type = 'workspace' and scope_id is null) or "
            "(scope_type in ('channel', 'user') and scope_id is not null)",
            name="ck_observe_policies_scope_id",
        ),
        Index(
            "idx_observe_policies_scope_unique",
            "installation_id",
            "scope_type",
            text("coalesce(scope_id, '')"),
            unique=True,
        ),
        Index(
            "idx_observe_policies_lookup",
            "installation_id",
            "scope_type",
            "scope_id",
            "observation_status",
        ),
    )


class AutonomyPolicy(Base):
    """Scoped autonomy-ladder level (HIG-223).

    Resolution: channel override -> workspace default -> 'balanced'. Mirrors the
    ObservePolicy scoping shape (unique per installation + scope_type + scope_id).
    """

    __tablename__ = "autonomy_policies"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    installation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("installations.id", ondelete="CASCADE"), nullable=False
    )
    scope_type: Mapped[str] = mapped_column(String, nullable=False)
    scope_id: Mapped[str | None] = mapped_column(String)
    level: Mapped[str] = mapped_column(String, nullable=False)
    updated_by_user_id: Mapped[str | None] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        CheckConstraint(
            "scope_type in ('workspace', 'channel')",
            name="ck_autonomy_policies_scope_type",
        ),
        CheckConstraint(
            "level in ('conservative', 'balanced', 'autonomous')",
            name="ck_autonomy_policies_level",
        ),
        CheckConstraint(
            "(scope_type = 'workspace' and scope_id is null) or "
            "(scope_type = 'channel' and scope_id is not null)",
            name="ck_autonomy_policies_scope_id",
        ),
        Index(
            "idx_autonomy_policies_scope_unique",
            "installation_id",
            "scope_type",
            text("coalesce(scope_id, '')"),
            unique=True,
        ),
        Index(
            "idx_autonomy_policies_lookup",
            "installation_id",
            "scope_type",
            "scope_id",
        ),
    )


class ObservationEvent(Base):
    __tablename__ = "observation_events"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    installation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("installations.id", ondelete="CASCADE"), nullable=False
    )
    slack_team_id: Mapped[str] = mapped_column(String, nullable=False)
    channel_id: Mapped[str] = mapped_column(String, nullable=False)
    user_id: Mapped[str | None] = mapped_column(String)
    event_type: Mapped[str] = mapped_column(String, nullable=False)
    slack_event_id: Mapped[str | None] = mapped_column(String)
    message_ts: Mapped[str | None] = mapped_column(String)
    thread_ts: Mapped[str | None] = mapped_column(String)
    file_id: Mapped[str | None] = mapped_column(String)
    raw_payload_checksum: Mapped[str] = mapped_column(String, nullable=False)
    text_preview: Mapped[str | None] = mapped_column(Text)
    visibility_metadata: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    observed_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now()
    )
    purged_at: Mapped[datetime | None] = mapped_column(TZ)
    created_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now()
    )

    __table_args__ = (
        CheckConstraint(
            "event_type in ('message', 'file_share', 'channel_join', 'channel_onboarding_intro')",
            name="ck_observation_events_event_type",
        ),
        Index(
            "idx_observation_events_event_unique",
            "installation_id",
            "slack_event_id",
            unique=True,
            postgresql_where=text("slack_event_id IS NOT NULL"),
        ),
        Index(
            "idx_observation_events_channel",
            "installation_id",
            "channel_id",
            "observed_at",
        ),
        Index(
            "idx_observation_events_user",
            "installation_id",
            "user_id",
            "observed_at",
        ),
        Index("idx_observation_events_purged", "purged_at"),
    )


class ObserveChannelProfile(Base):
    __tablename__ = "observe_channel_profiles"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    installation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("installations.id", ondelete="CASCADE"), nullable=False
    )
    channel_id: Mapped[str] = mapped_column(String, nullable=False)
    profile_status: Mapped[str] = mapped_column(
        String, nullable=False, server_default=text("'active'")
    )
    profile_version: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("1")
    )
    summary: Mapped[str | None] = mapped_column(Text)
    profile_json: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    assumptions_json: Mapped[list] = mapped_column(
        JSONB, nullable=False, server_default=text("'[]'::jsonb")
    )
    evidence_refs_json: Mapped[list] = mapped_column(
        JSONB, nullable=False, server_default=text("'[]'::jsonb")
    )
    confidence_score: Mapped[Decimal | None] = mapped_column(Numeric(4, 3))
    confidence_reason: Mapped[str | None] = mapped_column(Text)
    fresh_window_days: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("30")
    )
    archive_window_days: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("365")
    )
    observed_range_start_ts: Mapped[str | None] = mapped_column(String)
    observed_range_end_ts: Mapped[str | None] = mapped_column(String)
    message_count: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    file_count: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    last_scanned_message_ts: Mapped[str | None] = mapped_column(String)
    last_profiled_at: Mapped[datetime | None] = mapped_column(TZ)
    source_task_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("tasks.id", ondelete="SET NULL")
    )
    metadata_json: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    created_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        CheckConstraint(
            "profile_status in ('active', 'stale', 'disabled', 'failed')",
            name="ck_observe_channel_profiles_status",
        ),
        CheckConstraint(
            "fresh_window_days > 0",
            name="ck_observe_channel_profiles_fresh_window",
        ),
        CheckConstraint(
            "archive_window_days >= fresh_window_days",
            name="ck_observe_channel_profiles_archive_window",
        ),
        UniqueConstraint(
            "installation_id",
            "channel_id",
            name="idx_observe_channel_profiles_unique",
        ),
        Index(
            "idx_observe_channel_profiles_lookup",
            "installation_id",
            "channel_id",
            "profile_status",
        ),
        Index(
            "idx_observe_channel_profiles_last_profiled",
            "installation_id",
            "last_profiled_at",
        ),
        Index("idx_observe_channel_profiles_source_task", "source_task_id"),
    )


class KnowledgeGraphEntity(Base):
    __tablename__ = "kg_entities"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    installation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("installations.id", ondelete="CASCADE"), nullable=False
    )
    entity_type: Mapped[str] = mapped_column(String, nullable=False)
    canonical_key: Mapped[str] = mapped_column(String, nullable=False)
    display_name: Mapped[str | None] = mapped_column(String)
    external_ref_type: Mapped[str | None] = mapped_column(String)
    external_ref_id: Mapped[str | None] = mapped_column(String)
    attrs_json: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    visibility_scope_type: Mapped[str] = mapped_column(String, nullable=False)
    visibility_scope_id: Mapped[str | None] = mapped_column(String)
    source_type: Mapped[str] = mapped_column(String, nullable=False)
    confidence_score: Mapped[Decimal] = mapped_column(
        Numeric(4, 3), nullable=False, server_default=text("0.500")
    )
    confidence_reason: Mapped[str | None] = mapped_column(Text)
    lifecycle_state: Mapped[str] = mapped_column(
        String, nullable=False, server_default=text("'candidate'")
    )
    valid_from: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now()
    )
    valid_to: Mapped[datetime | None] = mapped_column(TZ)
    recorded_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now()
    )
    expired_at: Mapped[datetime | None] = mapped_column(TZ)
    # Bi-temporal validity (HIG-225): contradiction sets invalid_at instead of
    # deleting; system_expired_at records system-side archival.
    valid_at: Mapped[datetime | None] = mapped_column(TZ)
    invalid_at: Mapped[datetime | None] = mapped_column(TZ)
    system_expired_at: Mapped[datetime | None] = mapped_column(TZ)
    is_current: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("true")
    )
    freshness_window_days: Mapped[int | None] = mapped_column(Integer)
    last_reinforced_at: Mapped[datetime | None] = mapped_column(TZ)
    reinforcement_count: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    created_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        CheckConstraint(
            "entity_type in "
            "('person', 'channel', 'project', 'firm_fact', 'artifact', "
            "'decision', 'open_question', 'commitment', 'integration', "
            "'external_entity')",
            name="ck_kg_entities_type",
        ),
        CheckConstraint(
            "visibility_scope_type in "
            "('workspace', 'channel', 'private_channel', 'dm', 'user')",
            name="ck_kg_entities_visibility_scope_type",
        ),
        CheckConstraint(
            "(visibility_scope_type = 'workspace' and visibility_scope_id is null) or "
            "(visibility_scope_type in ('channel', 'private_channel', 'dm', 'user') "
            "and visibility_scope_id is not null)",
            name="ck_kg_entities_visibility_scope_id",
        ),
        CheckConstraint(
            "source_type in "
            "('slack_authoritative', 'user_explicit', 'user_confirmed', "
            "'agent_inferred', 'onboarding_scan', 'task_summary', "
            "'integration_result', 'workspace_state', 'admin_import')",
            name="ck_kg_entities_source_type",
        ),
        CheckConstraint(
            "lifecycle_state in "
            "('candidate', 'active', 'confirmed', 'stale', 'superseded', "
            "'contradicted', 'archived', 'forgotten')",
            name="ck_kg_entities_lifecycle_state",
        ),
        CheckConstraint(
            "confidence_score >= 0 and confidence_score <= 1",
            name="ck_kg_entities_confidence_score",
        ),
        CheckConstraint(
            "reinforcement_count >= 0",
            name="ck_kg_entities_reinforcement_count",
        ),
        Index(
            "idx_kg_entities_current_unique_key",
            "installation_id",
            "canonical_key",
            unique=True,
            postgresql_where=text("is_current = true AND expired_at IS NULL"),
        ),
        Index(
            "idx_kg_entities_lookup",
            "installation_id",
            "entity_type",
            "lifecycle_state",
            "is_current",
        ),
        Index(
            "idx_kg_entities_scope",
            "installation_id",
            "visibility_scope_type",
            "visibility_scope_id",
        ),
        Index("idx_kg_entities_external_ref", "external_ref_type", "external_ref_id"),
        Index("idx_kg_entities_attrs", "attrs_json", postgresql_using="gin"),
        Index("idx_kg_entities_invalid_at", "invalid_at"),
    )


class KnowledgeGraphEdge(Base):
    __tablename__ = "kg_edges"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    installation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("installations.id", ondelete="CASCADE"), nullable=False
    )
    source_entity_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("kg_entities.id", ondelete="CASCADE"), nullable=False
    )
    target_entity_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("kg_entities.id", ondelete="CASCADE"), nullable=False
    )
    relationship_type: Mapped[str] = mapped_column(String, nullable=False)
    attrs_json: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    visibility_scope_type: Mapped[str] = mapped_column(String, nullable=False)
    visibility_scope_id: Mapped[str | None] = mapped_column(String)
    source_type: Mapped[str] = mapped_column(String, nullable=False)
    confidence_score: Mapped[Decimal] = mapped_column(
        Numeric(4, 3), nullable=False, server_default=text("0.500")
    )
    confidence_reason: Mapped[str | None] = mapped_column(Text)
    lifecycle_state: Mapped[str] = mapped_column(
        String, nullable=False, server_default=text("'candidate'")
    )
    valid_from: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now()
    )
    valid_to: Mapped[datetime | None] = mapped_column(TZ)
    recorded_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now()
    )
    expired_at: Mapped[datetime | None] = mapped_column(TZ)
    # Bi-temporal validity (HIG-225): contradiction sets invalid_at instead of
    # deleting; system_expired_at records system-side archival.
    valid_at: Mapped[datetime | None] = mapped_column(TZ)
    invalid_at: Mapped[datetime | None] = mapped_column(TZ)
    system_expired_at: Mapped[datetime | None] = mapped_column(TZ)
    is_current: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("true")
    )
    freshness_window_days: Mapped[int | None] = mapped_column(Integer)
    last_reinforced_at: Mapped[datetime | None] = mapped_column(TZ)
    reinforcement_count: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    created_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        CheckConstraint(
            "relationship_type in "
            "('member_of', 'maps_to', 'works_on', 'owns', 'belongs_to', "
            "'referenced_in', 'made_in', 'affects', 'relates_to', 'available_for', "
            "'project_includes_channel', 'project_includes_entity')",
            name="ck_kg_edges_relationship_type",
        ),
        CheckConstraint(
            "visibility_scope_type in "
            "('workspace', 'channel', 'private_channel', 'dm', 'user')",
            name="ck_kg_edges_visibility_scope_type",
        ),
        CheckConstraint(
            "(visibility_scope_type = 'workspace' and visibility_scope_id is null) or "
            "(visibility_scope_type in ('channel', 'private_channel', 'dm', 'user') "
            "and visibility_scope_id is not null)",
            name="ck_kg_edges_visibility_scope_id",
        ),
        CheckConstraint(
            "source_type in "
            "('slack_authoritative', 'user_explicit', 'user_confirmed', "
            "'agent_inferred', 'onboarding_scan', 'task_summary', "
            "'integration_result', 'workspace_state', 'admin_import')",
            name="ck_kg_edges_source_type",
        ),
        CheckConstraint(
            "lifecycle_state in "
            "('candidate', 'active', 'confirmed', 'stale', 'superseded', "
            "'contradicted', 'archived', 'forgotten')",
            name="ck_kg_edges_lifecycle_state",
        ),
        CheckConstraint(
            "confidence_score >= 0 and confidence_score <= 1",
            name="ck_kg_edges_confidence_score",
        ),
        CheckConstraint(
            "reinforcement_count >= 0",
            name="ck_kg_edges_reinforcement_count",
        ),
        Index(
            "idx_kg_edges_current_unique",
            "installation_id",
            "source_entity_id",
            "target_entity_id",
            "relationship_type",
            "visibility_scope_type",
            text("coalesce(visibility_scope_id, '')"),
            unique=True,
            postgresql_where=text("is_current = true AND expired_at IS NULL"),
        ),
        Index(
            "idx_kg_edges_source_lookup",
            "installation_id",
            "source_entity_id",
            "relationship_type",
            "is_current",
        ),
        Index(
            "idx_kg_edges_target_lookup",
            "installation_id",
            "target_entity_id",
            "relationship_type",
            "is_current",
        ),
        Index(
            "idx_kg_edges_scope",
            "installation_id",
            "visibility_scope_type",
            "visibility_scope_id",
        ),
        Index("idx_kg_edges_attrs", "attrs_json", postgresql_using="gin"),
        Index("idx_kg_edges_invalid_at", "invalid_at"),
    )


class WitnessOpportunityCandidate(Base):
    __tablename__ = "witness_opportunity_candidates"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    installation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("installations.id", ondelete="CASCADE"), nullable=False
    )
    channel_id: Mapped[str | None] = mapped_column(String)
    target_slack_user_id: Mapped[str | None] = mapped_column(String)
    visibility_scope_type: Mapped[str] = mapped_column(String, nullable=False)
    visibility_scope_id: Mapped[str | None] = mapped_column(String)
    candidate_type: Mapped[str] = mapped_column(String, nullable=False)
    title: Mapped[str] = mapped_column(String, nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    suggested_action: Mapped[str | None] = mapped_column(Text)
    suggested_message: Mapped[str | None] = mapped_column(Text)
    evidence_json: Mapped[list] = mapped_column(
        JSONB, nullable=False, server_default=text("'[]'::jsonb")
    )
    source_type: Mapped[str] = mapped_column(String, nullable=False)
    source_id: Mapped[str | None] = mapped_column(String)
    source_task_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("tasks.id", ondelete="SET NULL")
    )
    source_profile_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("observe_channel_profiles.id", ondelete="SET NULL")
    )
    dedupe_key: Mapped[str] = mapped_column(String, nullable=False)
    confidence_score: Mapped[Decimal] = mapped_column(
        Numeric(4, 3), nullable=False, server_default=text("0.500")
    )
    confidence_reason: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(
        String, nullable=False, server_default=text("'candidate'")
    )
    automation_kind: Mapped[str | None] = mapped_column(Text)
    cadence_suggestion: Mapped[str | None] = mapped_column(Text)
    deliverable: Mapped[str | None] = mapped_column(Text)
    automated_schedule_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("schedules.id", ondelete="SET NULL")
    )
    automated_task_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("tasks.id", ondelete="SET NULL")
    )
    cooldown_until: Mapped[datetime | None] = mapped_column(TZ)
    last_suggested_at: Mapped[datetime | None] = mapped_column(TZ)
    reinforcement_count: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("1")
    )
    first_observed_at: Mapped[datetime | None] = mapped_column(TZ)
    last_decision: Mapped[str | None] = mapped_column(Text)
    receptivity_score: Mapped[Decimal | None] = mapped_column(Numeric(4, 3))
    feedback_json: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    metadata_json: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    created_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        CheckConstraint(
            "visibility_scope_type in "
            "('workspace', 'channel', 'private_channel', 'dm', 'user')",
            name="ck_witness_opportunity_candidates_scope_type",
        ),
        CheckConstraint(
            "(visibility_scope_type = 'workspace' and visibility_scope_id is null) or "
            "(visibility_scope_type in ('channel', 'private_channel', 'dm', 'user') "
            "and visibility_scope_id is not null)",
            name="ck_witness_opportunity_candidates_scope_id",
        ),
        CheckConstraint(
            "candidate_type in "
            "('workflow_gap', 'artifact_followup', 'unresolved_decision', "
            "'data_quality_issue', 'recurring_check', 'project_status_gap', "
            "'general_help')",
            name="ck_witness_opportunity_candidates_type",
        ),
        CheckConstraint(
            "source_type in "
            "('channel_profile', 'knowledge_graph', 'task_summary', "
            "'scheduled_witness', 'manual')",
            name="ck_witness_opportunity_candidates_source_type",
        ),
        CheckConstraint(
            "status in "
            "('candidate', 'sent', 'accepted', 'automated', 'dismissed', "
            "'cooldown', 'superseded', 'archived')",
            name="ck_witness_opportunity_candidates_status",
        ),
        CheckConstraint(
            "automation_kind is null or "
            "automation_kind in ('recurring', 'one_shot', 'watch')",
            name="ck_witness_opportunity_candidates_automation_kind",
        ),
        CheckConstraint(
            "confidence_score >= 0 and confidence_score <= 1",
            name="ck_witness_opportunity_candidates_confidence",
        ),
        CheckConstraint(
            "last_decision is null or "
            "last_decision in ('notify', 'question', 'draft', 'silent')",
            name="ck_witness_opportunity_candidates_last_decision",
        ),
        CheckConstraint(
            "receptivity_score is null or "
            "(receptivity_score >= 0 and receptivity_score <= 1)",
            name="ck_witness_opportunity_candidates_receptivity",
        ),
        CheckConstraint(
            "reinforcement_count >= 0",
            name="ck_witness_opportunity_candidates_reinforcement_count",
        ),
        Index(
            "idx_witness_opportunity_candidates_unique",
            "installation_id",
            "visibility_scope_type",
            text("coalesce(visibility_scope_id, '')"),
            "candidate_type",
            "dedupe_key",
            unique=True,
        ),
        Index(
            "idx_witness_opportunity_candidates_status",
            "installation_id",
            "status",
            "cooldown_until",
        ),
        Index(
            "idx_witness_opportunity_candidates_channel",
            "installation_id",
            "channel_id",
            "status",
        ),
        Index(
            "idx_witness_opportunity_candidates_scope",
            "installation_id",
            "visibility_scope_type",
            "visibility_scope_id",
        ),
        Index("idx_witness_opportunity_candidates_task", "source_task_id"),
        Index("idx_witness_opportunity_candidates_profile", "source_profile_id"),
        Index(
            "idx_witness_opportunity_candidates_evidence",
            "evidence_json",
            postgresql_using="gin",
        ),
    )


class WitnessDeliveryLog(Base):
    """Append-only record of every Witness delivery decision (incl. silence)."""

    __tablename__ = "witness_delivery_log"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    installation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("installations.id", ondelete="CASCADE"), nullable=False
    )
    slack_user_id: Mapped[str] = mapped_column(Text, nullable=False)
    candidate_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("witness_opportunity_candidates.id", ondelete="CASCADE")
    )
    decision: Mapped[str] = mapped_column(Text, nullable=False)
    reason: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now()
    )

    __table_args__ = (
        CheckConstraint(
            "decision in ('notify', 'question', 'draft', 'silent', 'digest', "
            "'channel_sent', 'channel_deferred', 'draft_executed', "
            "'ambient_file_brief')",
            name="ck_witness_delivery_log_decision",
        ),
        Index(
            "idx_witness_delivery_log_user_window",
            "installation_id",
            "slack_user_id",
            "created_at",
        ),
        Index("idx_witness_delivery_log_candidate", "candidate_id"),
        Index(
            "idx_witness_delivery_log_decision",
            "installation_id",
            "decision",
            "created_at",
        ),
    )


class KnowledgeGraphEvidence(Base):
    __tablename__ = "kg_evidence"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    installation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("installations.id", ondelete="CASCADE"), nullable=False
    )
    target_kind: Mapped[str] = mapped_column(String, nullable=False)
    target_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    source_type: Mapped[str] = mapped_column(String, nullable=False)
    source_task_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("tasks.id", ondelete="SET NULL")
    )
    source_episode_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("episodes.id", ondelete="SET NULL")
    )
    source_task_event_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("task_events.id", ondelete="SET NULL")
    )
    source_observation_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("observation_events.id", ondelete="SET NULL")
    )
    source_slack_channel_id: Mapped[str | None] = mapped_column(String)
    source_slack_message_ts: Mapped[str | None] = mapped_column(String)
    source_slack_file_id: Mapped[str | None] = mapped_column(String)
    source_url: Mapped[str | None] = mapped_column(Text)
    extracted_by: Mapped[str] = mapped_column(String, nullable=False)
    raw_snippet: Mapped[str | None] = mapped_column(Text)
    confidence_score: Mapped[Decimal | None] = mapped_column(Numeric(4, 3))
    confidence_reason: Mapped[str | None] = mapped_column(Text)
    consensus_count: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("1")
    )
    created_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now()
    )

    __table_args__ = (
        CheckConstraint(
            "target_kind in ('entity', 'edge')",
            name="ck_kg_evidence_target_kind",
        ),
        CheckConstraint(
            "source_type in "
            "('slack_authoritative', 'user_explicit', 'agent_inferred', "
            "'onboarding_scan', 'task_summary', 'integration_result', "
            "'workspace_state', 'admin_import')",
            name="ck_kg_evidence_source_type",
        ),
        CheckConstraint(
            "consensus_count > 0",
            name="ck_kg_evidence_consensus_count",
        ),
        CheckConstraint(
            "confidence_score is null or "
            "(confidence_score >= 0 and confidence_score <= 1)",
            name="ck_kg_evidence_confidence_score",
        ),
        CheckConstraint(
            "source_type = 'admin_import' or "
            "source_task_id is not null or "
            "source_episode_id is not null or "
            "source_task_event_id is not null or "
            "source_observation_id is not null or "
            "source_slack_channel_id is not null or "
            "source_slack_file_id is not null or "
            "source_url is not null",
            name="ck_kg_evidence_source_reference",
        ),
        Index(
            "idx_kg_evidence_target",
            "target_kind",
            "target_id",
        ),
        Index("idx_kg_evidence_task", "source_task_id"),
        Index("idx_kg_evidence_episode", "source_episode_id"),
        Index("idx_kg_evidence_observation", "source_observation_id"),
        Index(
            "idx_kg_evidence_slack_message",
            "installation_id",
            "source_slack_channel_id",
            "source_slack_message_ts",
        ),
    )


class ProceduralSkill(Base):
    __tablename__ = "procedural_skills"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    slug: Mapped[str] = mapped_column(String, nullable=False)
    owner_type: Mapped[str] = mapped_column(String, nullable=False)
    owner_id: Mapped[str | None] = mapped_column(String)
    status: Mapped[str] = mapped_column(String, nullable=False)
    trust_level: Mapped[str] = mapped_column(String, nullable=False)
    visibility: Mapped[str] = mapped_column(String, nullable=False)
    provenance: Mapped[str] = mapped_column(
        String, nullable=False, server_default="kortny"
    )
    created_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        CheckConstraint(
            "owner_type in ('system', 'workspace', 'user')",
            name="ck_procedural_skills_owner_type",
        ),
        CheckConstraint(
            "status in ('draft', 'active', 'deprecated', 'disabled', 'archived')",
            name="ck_procedural_skills_status",
        ),
        CheckConstraint(
            "trust_level in ('trusted', 'community', 'untrusted', 'quarantined')",
            name="ck_procedural_skills_trust_level",
        ),
        CheckConstraint(
            "visibility in ('catalog', 'explicit_only', 'disabled')",
            name="ck_procedural_skills_visibility",
        ),
        CheckConstraint(
            "(owner_type = 'system' and owner_id is null) or "
            "(owner_type in ('workspace', 'user') and owner_id is not null)",
            name="ck_procedural_skills_owner_id",
        ),
        Index(
            "idx_procedural_skills_unique_slug",
            "owner_type",
            text("coalesce(owner_id, '')"),
            "slug",
            unique=True,
        ),
        Index(
            "idx_procedural_skills_catalog",
            "owner_type",
            "status",
            "visibility",
            "slug",
        ),
    )


class ProceduralSkillVersion(Base):
    __tablename__ = "procedural_skill_versions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    skill_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("procedural_skills.id", ondelete="CASCADE"), nullable=False
    )
    version: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    instructions_md: Mapped[str] = mapped_column(Text, nullable=False)
    intent_tags: Mapped[list] = mapped_column(
        JSONB, nullable=False, server_default=text("'[]'::jsonb")
    )
    response_modes: Mapped[list] = mapped_column(
        JSONB, nullable=False, server_default=text("'[]'::jsonb")
    )
    trigger_phrases: Mapped[list] = mapped_column(
        JSONB, nullable=False, server_default=text("'[]'::jsonb")
    )
    allowed_tools: Mapped[list] = mapped_column(
        JSONB, nullable=False, server_default=text("'[]'::jsonb")
    )
    metadata_json: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    content_sha256: Mapped[str] = mapped_column(String, nullable=False)
    created_by: Mapped[str] = mapped_column(String, nullable=False)
    approved_by: Mapped[str | None] = mapped_column(String)
    published_at: Mapped[datetime | None] = mapped_column(TZ)
    created_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        CheckConstraint(
            "status in ('draft', 'active', 'deprecated', 'archived')",
            name="ck_procedural_skill_versions_status",
        ),
        UniqueConstraint(
            "skill_id",
            "version",
            name="idx_procedural_skill_versions_unique",
        ),
        Index(
            "idx_procedural_skill_versions_active",
            "skill_id",
            "status",
            "version",
        ),
        Index(
            "idx_procedural_skill_versions_tags",
            "intent_tags",
            postgresql_using="gin",
        ),
        Index(
            "idx_procedural_skill_versions_modes",
            "response_modes",
            postgresql_using="gin",
        ),
    )


class ProceduralSkillInvocation(Base):
    __tablename__ = "procedural_skill_invocations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    installation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("installations.id", ondelete="CASCADE"), nullable=False
    )
    task_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False
    )
    skill_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("procedural_skills.id", ondelete="CASCADE"), nullable=False
    )
    skill_version_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("procedural_skill_versions.id", ondelete="CASCADE"),
        nullable=False,
    )
    invocation_kind: Mapped[str] = mapped_column(String, nullable=False)
    response_mode: Mapped[str | None] = mapped_column(String)
    selected_reason: Mapped[str | None] = mapped_column(Text)
    payload: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    created_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index(
            "idx_procedural_skill_invocations_task",
            "task_id",
            "created_at",
        ),
        Index(
            "idx_procedural_skill_invocations_skill",
            "skill_id",
            "skill_version_id",
            "created_at",
        ),
        Index(
            "idx_procedural_skill_invocations_installation",
            "installation_id",
            "created_at",
        ),
    )


class SkillFile(Base):
    __tablename__ = "skill_files"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    skill_version_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("procedural_skill_versions.id", ondelete="CASCADE"),
        nullable=False,
    )
    path: Mapped[str] = mapped_column(String, nullable=False)
    kind: Mapped[str] = mapped_column(String, nullable=False)
    content_text: Mapped[str | None] = mapped_column(Text)
    content_bytes: Mapped[bytes | None] = mapped_column(BYTEA)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    sha256: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now()
    )

    __table_args__ = (
        CheckConstraint(
            "kind in ('reference', 'asset', 'script')",
            name="ck_skill_files_kind",
        ),
        UniqueConstraint(
            "skill_version_id",
            "path",
            name="uq_skill_files_version_path",
        ),
        Index("idx_skill_files_version", "skill_version_id", "kind"),
    )


class SkillEnablement(Base):
    __tablename__ = "skill_enablements"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    installation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("installations.id", ondelete="CASCADE"), nullable=False
    )
    skill_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("procedural_skills.id", ondelete="CASCADE"), nullable=False
    )
    scope_type: Mapped[str] = mapped_column(String, nullable=False)
    scope_id: Mapped[str | None] = mapped_column(String)
    status: Mapped[str] = mapped_column(
        String, nullable=False, server_default="enabled"
    )
    added_by: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        CheckConstraint(
            "scope_type in ('workspace', 'channel', 'user')",
            name="ck_skill_enablements_scope_type",
        ),
        CheckConstraint(
            "status in ('enabled', 'disabled')",
            name="ck_skill_enablements_status",
        ),
        CheckConstraint(
            "(scope_type = 'workspace' and scope_id is null) or "
            "(scope_type in ('channel', 'user') and scope_id is not null)",
            name="ck_skill_enablements_scope_id",
        ),
        Index(
            "idx_skill_enablements_unique",
            "installation_id",
            "skill_id",
            "scope_type",
            text("coalesce(scope_id, '')"),
            unique=True,
        ),
        Index(
            "idx_skill_enablements_scope",
            "installation_id",
            "scope_type",
            "scope_id",
            "status",
        ),
    )


class McpServer(Base):
    __tablename__ = "mcp_servers"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    installation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("installations.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String, nullable=False)
    transport: Mapped[str] = mapped_column(String, nullable=False)
    command: Mapped[str | None] = mapped_column(String)
    args: Mapped[list] = mapped_column(
        JSONB, nullable=False, server_default=text("'[]'::jsonb")
    )
    url: Mapped[str | None] = mapped_column(String)
    env_json: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    headers_json: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    secret_env: Mapped[bytes | None] = mapped_column(BYTEA)
    status: Mapped[str] = mapped_column(
        String, nullable=False, server_default="enabled"
    )
    # HIG-169 P0.2: trust tier mirrors the skills ladder. ``readOnlyHint`` is
    # attacker-asserted metadata, so a tool's read-only claim only clears its
    # approval requirement when its server is trusted (and the tool is
    # pinned-unchanged). Newly registered servers default to ``untrusted``.
    trust_tier: Mapped[str] = mapped_column(
        String, nullable=False, server_default="untrusted"
    )
    last_discovery_at: Mapped[datetime | None] = mapped_column(TZ)
    last_discovery_error: Mapped[str | None] = mapped_column(Text)
    created_by: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        CheckConstraint(
            "transport in ('stdio', 'streamable_http', 'sse')",
            name="ck_mcp_servers_transport",
        ),
        CheckConstraint(
            "status in ('enabled', 'disabled')",
            name="ck_mcp_servers_status",
        ),
        CheckConstraint(
            "trust_tier in ('trusted', 'community', 'untrusted')",
            name="ck_mcp_servers_trust_tier",
        ),
        CheckConstraint(
            "(transport = 'stdio' and command is not null) or "
            "(transport in ('streamable_http', 'sse') and url is not null)",
            name="ck_mcp_servers_transport_target",
        ),
        UniqueConstraint(
            "installation_id",
            "name",
            name="uq_mcp_servers_installation_name",
        ),
        Index(
            "idx_mcp_servers_enabled_lookup",
            "installation_id",
            "status",
        ),
    )


class McpServerTool(Base):
    __tablename__ = "mcp_server_tools"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    server_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("mcp_servers.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, server_default="")
    input_schema: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    read_only_hint: Mapped[bool | None] = mapped_column(Boolean)
    destructive_hint: Mapped[bool | None] = mapped_column(Boolean)
    enabled: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("true")
    )
    # Description quality columns (migration 0028 / HIG-215)
    description_quality_score: Mapped[Decimal | None] = mapped_column(
        Numeric(precision=4, scale=3), nullable=True
    )
    enriched_description: Mapped[str | None] = mapped_column(Text, nullable=True)
    description_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        UniqueConstraint(
            "server_id",
            "name",
            name="uq_mcp_server_tools_server_name",
        ),
        Index(
            "idx_mcp_server_tools_enabled_lookup",
            "server_id",
            "enabled",
        ),
    )


class ToolPin(Base):
    """Pinned fingerprint of one external tool's schema (HIG-169 P0.3).

    Defends against tool rug-pulls: a server (MCP) or toolkit (Composio) that
    silently mutates a tool's ``inputSchema`` after first approval. The
    fingerprint is the sha256 of the canonical JSON of the tool's
    ``{name, description, inputSchema, ...}`` — crucially including the input
    schema, which the existing ``card_sha`` / ``description_sha256`` columns do
    NOT cover. ``status`` flips to ``drifted`` when the live fingerprint diverges
    from the pinned one, which revokes the read-only approval bypass until an
    admin re-pins. Pin-on-first-sight: the first registration is the admin's
    implicit approval, consistent with the existing trust model.
    """

    __tablename__ = "tool_pins"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    installation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("installations.id", ondelete="CASCADE"), nullable=False
    )
    provider: Mapped[str] = mapped_column(String, nullable=False)
    # MCP server id (as string) or Composio toolkit slug.
    server_ref: Mapped[str] = mapped_column(String, nullable=False)
    tool_name: Mapped[str] = mapped_column(String, nullable=False)
    fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    prior_description: Mapped[str | None] = mapped_column(Text, nullable=True)
    prior_schema_json: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    status: Mapped[str] = mapped_column(String, nullable=False, server_default="active")
    approved_by: Mapped[str | None] = mapped_column(String, nullable=True)
    approved_at: Mapped[datetime | None] = mapped_column(TZ)
    created_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        CheckConstraint(
            "provider in ('mcp', 'composio')",
            name="ck_tool_pins_provider",
        ),
        CheckConstraint(
            "status in ('active', 'drifted')",
            name="ck_tool_pins_status",
        ),
        UniqueConstraint(
            "installation_id",
            "provider",
            "server_ref",
            "tool_name",
            name="uq_tool_pins_identity",
        ),
        Index(
            "idx_tool_pins_lookup",
            "installation_id",
            "provider",
            "server_ref",
        ),
    )


class ToolEmbedding(Base):
    """Semantic embedding for one tool card, skill, or memory row.

    The table name is historical (HIG-219 shipped it for tool cards/skills);
    HIG-225 reuses it as the general memory index with kinds ``fact``,
    ``episode``, and ``kg_entity`` (``ref_key`` = source row UUID as string).

    ``embedding`` is an untyped ``vector`` so rows from models with different
    dimensions can coexist; queries cast the query vector at runtime and always
    filter on ``model``.
    """

    __tablename__ = "tool_embeddings"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    kind: Mapped[str] = mapped_column(Text, nullable=False)
    ref_key: Mapped[str] = mapped_column(Text, nullable=False)
    model: Mapped[str] = mapped_column(Text, nullable=False)
    dim: Mapped[int] = mapped_column(Integer, nullable=False)
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        CheckConstraint(
            "kind in ('tool_card', 'skill', 'fact', 'episode', 'kg_entity')",
            name="ck_tool_embeddings_kind",
        ),
        UniqueConstraint(
            "kind",
            "ref_key",
            "model",
            name="uq_tool_embeddings_kind_ref_key_model",
        ),
        Index("idx_tool_embeddings_kind_model", "kind", "model"),
    )


class ConsolidationRun(Base):
    """One memory-consolidation run for an installation (HIG-225)."""

    __tablename__ = "consolidation_runs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    installation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("installations.id", ondelete="CASCADE"), nullable=False
    )
    started_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now()
    )
    finished_at: Mapped[datetime | None] = mapped_column(TZ)
    status: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=text("'running'")
    )
    counters_json: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    cost_usd: Mapped[Decimal | None] = mapped_column(Numeric(12, 6))
    error: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (
        CheckConstraint(
            "status in ('running', 'succeeded', 'failed')",
            name="ck_consolidation_runs_status",
        ),
        Index(
            "idx_consolidation_runs_installation_started",
            "installation_id",
            "started_at",
        ),
    )


class Episode(Base):
    __tablename__ = "episodes"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    installation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("installations.id", ondelete="CASCADE"), nullable=False
    )
    task_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False
    )
    channel_id: Mapped[str] = mapped_column(String, nullable=False)
    user_id: Mapped[str] = mapped_column(String, nullable=False)
    thread_ts: Mapped[str | None] = mapped_column(String)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    tools_used: Mapped[list] = mapped_column(
        JSONB, nullable=False, server_default=text("'[]'::jsonb")
    )
    artifacts_created: Mapped[list] = mapped_column(
        JSONB, nullable=False, server_default=text("'[]'::jsonb")
    )
    source_refs: Mapped[list] = mapped_column(
        JSONB, nullable=False, server_default=text("'[]'::jsonb")
    )
    outcome: Mapped[str] = mapped_column(String, nullable=False)
    error_json: Mapped[dict | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        CheckConstraint(
            "outcome in ('succeeded', 'failed', 'cancelled')",
            name="ck_episodes_outcome",
        ),
        UniqueConstraint("task_id", name="idx_episodes_task_unique"),
        Index("idx_episodes_thread", "installation_id", "channel_id", "thread_ts"),
        Index("idx_episodes_channel", "installation_id", "channel_id", "created_at"),
        Index("idx_episodes_user", "installation_id", "user_id", "created_at"),
    )


class LLMUsage(Base):
    __tablename__ = "llm_usage"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    task_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tasks.id"), nullable=False)
    event_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("task_events.id")
    )
    provider: Mapped[LLMProvider] = mapped_column(LLM_PROVIDER, nullable=False)
    model: Mapped[str] = mapped_column(String, nullable=False)
    model_tier: Mapped[str | None] = mapped_column(String)
    input_tokens: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    output_tokens: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    # Prompt-cache split of input_tokens (HIG-196). These are a partition *within*
    # input_tokens (the total prompt count), not additions to it.
    cache_creation_input_tokens: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    cache_read_input_tokens: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    cost_usd: Mapped[Decimal] = mapped_column(
        Numeric(12, 6), nullable=False, server_default=text("0")
    )
    created_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("idx_usage_task", "task_id"),
        Index("idx_usage_time", "created_at"),
    )


class Artifact(Base):
    __tablename__ = "artifacts"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    task_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tasks.id"), nullable=False)
    filename: Mapped[str] = mapped_column(String, nullable=False)
    mime_type: Mapped[str | None] = mapped_column(String)
    size_bytes: Mapped[int | None] = mapped_column(BigInteger)
    storage_path: Mapped[str | None] = mapped_column(String)
    slack_file_id: Mapped[str | None] = mapped_column(String)
    posted_at: Mapped[datetime | None] = mapped_column(TZ)
    # Living-document fields (HIG-244): the canonical post-critique spec is kept
    # so a doc can be re-rendered/edited from buttons. All versions of one doc
    # share doc_group_id; doc_version increments per visible revision.
    doc_group_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    doc_version: Mapped[int | None] = mapped_column(Integer)
    spec_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("idx_artifacts_task", "task_id"),
        Index("idx_artifacts_doc_group", "doc_group_id", "doc_version"),
    )


class ModelPricing(Base):
    __tablename__ = "model_pricing"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    provider: Mapped[LLMProvider] = mapped_column(LLM_PROVIDER, nullable=False)
    model: Mapped[str] = mapped_column(String, nullable=False)
    input_price_per_mtok: Mapped[Decimal] = mapped_column(
        Numeric(12, 6), nullable=False
    )
    output_price_per_mtok: Mapped[Decimal] = mapped_column(
        Numeric(12, 6), nullable=False
    )
    # Prompt-cache cost multipliers vs the base input price (HIG-196 D5):
    # cache writes default to 1.25x (5-minute TTL), cache reads to 0.1x.
    cache_write_multiplier: Mapped[Decimal] = mapped_column(
        Numeric(6, 4), nullable=False, server_default=text("1.25")
    )
    cache_read_multiplier: Mapped[Decimal] = mapped_column(
        Numeric(6, 4), nullable=False, server_default=text("0.10")
    )
    effective_from: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint(
            "provider", "model", "effective_from", name="idx_pricing_lookup"
        ),
    )


class AssistantThreadContext(Base):
    """Per-assistant-thread context store (HIG-236).

    Slack assistant-thread ``message.im`` events do not carry the channel the
    user was viewing when they opened the assistant; the app must persist it.
    One row per (channel_id, thread_ts) — the Postgres-backed implementation of
    slack_bolt's ``AssistantThreadContextStore`` upserts here.
    """

    __tablename__ = "assistant_thread_context"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    channel_id: Mapped[str] = mapped_column(String, nullable=False)
    thread_ts: Mapped[str] = mapped_column(String, nullable=False)
    context_json: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    created_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        UniqueConstraint(
            "channel_id",
            "thread_ts",
            name="uq_assistant_thread_context_channel_thread",
        ),
    )


class InteractiveAction(Base):
    """A Block Kit interactive action bound into a posted Slack message (HIG-255
    slice 2). The Slack button carries only an opaque raw key; this row stores
    the key's HMAC hash plus everything needed to authorize a click — the actor,
    workspace, container, target, and lifecycle. The opaque key is the lookup,
    not the security model: a click is honored only when key + Slack-authed actor
    + workspace/container match + the target is still actionable, under a row
    lock. Wrong-user/forged clicks increment the denial audit and leave the row
    usable.
    """

    __tablename__ = "interactive_actions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    installation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("installations.id", ondelete="CASCADE"), nullable=False
    )
    task_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("tasks.id", ondelete="CASCADE")
    )
    # The task created by acting on this action (retry/run_again), if any.
    result_task_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("tasks.id", ondelete="SET NULL")
    )

    action_key_hash: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    action_kind: Mapped[str] = mapped_column(String, nullable=False)
    route: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(
        String, nullable=False, server_default=text("'pending_send'")
    )

    target_type: Mapped[str] = mapped_column(String, nullable=False)
    target_id: Mapped[str] = mapped_column(String, nullable=False)
    payload_json: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )

    slack_team_id: Mapped[str | None] = mapped_column(String)
    slack_channel_id: Mapped[str | None] = mapped_column(String)
    slack_thread_ts: Mapped[str | None] = mapped_column(String)
    slack_message_ts: Mapped[str | None] = mapped_column(String)
    slack_block_id: Mapped[str | None] = mapped_column(String)
    slack_action_id: Mapped[str | None] = mapped_column(String)

    created_for_user_id: Mapped[str | None] = mapped_column(String)
    allowed_user_id: Mapped[str | None] = mapped_column(String)
    required_role: Mapped[str | None] = mapped_column(String)
    allowed_channel_id: Mapped[str | None] = mapped_column(String)

    expires_at: Mapped[datetime] = mapped_column(TZ, nullable=False)
    sent_at: Mapped[datetime | None] = mapped_column(TZ)
    consumed_at: Mapped[datetime | None] = mapped_column(TZ)
    consumed_by_user_id: Mapped[str | None] = mapped_column(String)
    last_denied_at: Mapped[datetime | None] = mapped_column(TZ)
    denied_count: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )

    metadata_json: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    created_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        Index("idx_interactive_actions_installation", "installation_id"),
        Index("idx_interactive_actions_task", "task_id"),
        Index(
            "idx_interactive_actions_target",
            "installation_id",
            "target_type",
            "target_id",
            "status",
        ),
    )


class FileExtractionCache(Base):
    """Content-addressed cache for extracted text from Slack files (HIG-279).

    The SHA-256 of the raw file bytes is the primary key, so identical content
    is extracted once and reused across retries, follow-up questions, and
    multiple users referencing the same document.
    """

    __tablename__ = "file_extraction_cache"

    content_sha256: Mapped[str] = mapped_column(String(64), primary_key=True)
    backend: Mapped[str] = mapped_column(String, nullable=False)
    extraction_supported: Mapped[bool] = mapped_column(Boolean, nullable=False)
    extracted_text: Mapped[str | None] = mapped_column(Text)
    truncated: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )
    page_count: Mapped[int | None] = mapped_column(Integer)
    byte_size: Mapped[int] = mapped_column(BigInteger, nullable=False)
    warnings: Mapped[list] = mapped_column(
        JSONB, nullable=False, server_default=text("'[]'::jsonb")
    )
    created_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now()
    )
    last_accessed_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("idx_file_extraction_cache_last_accessed", "last_accessed_at"),
    )
