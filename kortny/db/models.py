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
            "trust_level in ('trusted', 'reviewed', 'unreviewed', 'quarantined')",
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
