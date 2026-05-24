"""
Kortny — database models.

Postgres-specific types are used throughout (UUID, JSONB, BYTEA, timestamptz,
native enums), so the canonical path to a live database is the Alembic
migration, not Base.metadata.create_all().
"""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from sqlalchemy import (
    BigInteger,
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

    # Request + result
    input: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[TaskStatus] = mapped_column(
        TASK_STATUS, nullable=False, server_default=text("'pending'::task_status")
    )
    result_summary: Mapped[str | None] = mapped_column(Text)
    error: Mapped[dict | None] = mapped_column(JSONB)

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
    created_at: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now()
    )

    __table_args__ = (Index("idx_artifacts_task", "task_id"),)


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
    effective_from: Mapped[datetime] = mapped_column(
        TZ, nullable=False, server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint(
            "provider", "model", "effective_from", name="idx_pricing_lookup"
        ),
    )
