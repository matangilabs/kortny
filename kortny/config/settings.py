"""Typed application settings loaded from environment variables."""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(StrEnum):
    """Supported model-provider routing options."""

    openai = "openai"
    anthropic = "anthropic"
    openrouter = "openrouter"


class Settings(BaseSettings):
    """Runtime configuration sourced from the environment or `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    slack_bot_token: str = Field(validation_alias="SLACK_BOT_TOKEN", min_length=1)
    slack_app_token: str = Field(validation_alias="SLACK_APP_TOKEN", min_length=1)
    slack_signing_secret: str = Field(
        validation_alias="SLACK_SIGNING_SECRET", min_length=1
    )
    slack_app_name: str = Field(
        default="kortny", validation_alias="SLACK_APP_NAME", min_length=1
    )

    llm_provider: LLMProvider = Field(validation_alias="LLM_PROVIDER")
    llm_api_key: str = Field(validation_alias="LLM_API_KEY", min_length=1)
    llm_model: str = Field(validation_alias="LLM_MODEL", min_length=1)
    llm_cheap_model: str | None = Field(
        default=None, validation_alias="LLM_CHEAP_MODEL"
    )
    llm_standard_model: str | None = Field(
        default=None, validation_alias="LLM_STANDARD_MODEL"
    )
    llm_analysis_model: str | None = Field(
        default=None, validation_alias="LLM_ANALYSIS_MODEL"
    )
    llm_document_model: str | None = Field(
        default=None, validation_alias="LLM_DOCUMENT_MODEL"
    )
    llm_high_reasoning_model: str | None = Field(
        default=None, validation_alias="LLM_HIGH_REASONING_MODEL"
    )
    llm_humanizer_model: str | None = Field(
        default=None, validation_alias="LLM_HUMANIZER_MODEL"
    )
    llm_vision_model: str | None = Field(
        default=None, validation_alias="LLM_VISION_MODEL"
    )
    llm_config_force_env: bool = Field(
        default=False, validation_alias="LLM_CONFIG_FORCE_ENV"
    )
    # Cap on completion tokens requested per LLM call. Providers (incl.
    # OpenRouter) reserve credit for the requested max_tokens up front; left
    # unset, litellm defaults to the model's max output (e.g. 65k on Opus),
    # which inflates cost-reservation and can hard-fail with a 402 on a
    # budget-limited key (HIG-265). 16k is ample for any single agent turn.
    llm_max_output_tokens: int = Field(
        default=16384, validation_alias="LLM_MAX_OUTPUT_TOKENS", ge=256
    )
    response_humanizer_enabled: bool = Field(
        default=True, validation_alias="RESPONSE_HUMANIZER_ENABLED"
    )
    response_humanizer_min_chars: int = Field(
        default=120, validation_alias="RESPONSE_HUMANIZER_MIN_CHARS"
    )
    assistant_streaming_enabled: bool = Field(
        default=True, validation_alias="KORTNY_ASSISTANT_STREAMING_ENABLED"
    )
    channel_progress_enabled: bool = Field(
        default=False, validation_alias="KORTNY_CHANNEL_PROGRESS_ENABLED"
    )
    # AGENT_RUNTIME is still accepted for env compatibility, but the Google ADK
    # runtime was retired (HIG-281); custom is the only runtime.
    agent_runtime: Literal["custom"] = Field(
        default="custom", validation_alias="AGENT_RUNTIME"
    )
    workflow_backend: Literal["inline", "temporal"] = Field(
        default="inline", validation_alias="KORTNY_WORKFLOW_BACKEND"
    )
    planned_workflows_enabled: bool = Field(
        default=True, validation_alias="KORTNY_PLANNED_WORKFLOWS_ENABLED"
    )
    planned_workflow_max_parallel_branches: int = Field(
        default=3, validation_alias="KORTNY_PLANNED_WORKFLOW_MAX_PARALLEL_BRANCHES"
    )
    planned_workflow_cost_ceiling_usd: float = Field(
        default=0.75, validation_alias="KORTNY_PLANNED_WORKFLOW_COST_CEILING_USD"
    )
    ambient_task_cost_ceiling_usd: float = Field(
        default=0.25, validation_alias="KORTNY_AMBIENT_TASK_COST_CEILING_USD"
    )
    consolidator_run_cost_ceiling_usd: float = Field(
        default=2.00, validation_alias="KORTNY_CONSOLIDATOR_RUN_COST_CEILING_USD"
    )
    planned_workflow_max_branch_model_calls: int = Field(
        default=3,
        validation_alias="KORTNY_PLANNED_WORKFLOW_MAX_BRANCH_MODEL_CALLS",
    )
    planned_workflow_max_branch_tool_calls: int = Field(
        default=8,
        validation_alias="KORTNY_PLANNED_WORKFLOW_MAX_BRANCH_TOOL_CALLS",
    )
    planned_workflow_max_total_tool_calls: int = Field(
        default=12,
        validation_alias="KORTNY_PLANNED_WORKFLOW_MAX_TOTAL_TOOL_CALLS",
    )
    planned_workflow_progress_updates_enabled: bool = Field(
        default=True,
        validation_alias="KORTNY_PLANNED_WORKFLOW_PROGRESS_UPDATES_ENABLED",
    )
    sandbox_runner_url: str | None = Field(
        default=None,
        validation_alias="KORTNY_SANDBOX_RUNNER_URL",
    )
    sandbox_runner_timeout_seconds: float = Field(
        default=70.0,
        validation_alias="KORTNY_SANDBOX_RUNNER_TIMEOUT_SECONDS",
    )
    sandbox_default_image: str = Field(
        default="ghcr.io/astral-sh/uv:python3.11-bookworm-slim",
        validation_alias="KORTNY_SANDBOX_DEFAULT_IMAGE",
        min_length=1,
    )
    # --- Document Studio (HIG-244) -------------------------------------------
    # Typst is the PDF beauty engine, vendored via the ``typst`` PyPI wheel (no
    # system binary). Font dirs (colon-separated) let a deployment ship the
    # theme fonts so renders are deterministic rather than relying on whatever
    # system fonts exist.
    document_font_paths: str = Field(
        default="",
        validation_alias="KORTNY_DOCUMENT_FONT_PATHS",
    )
    # --- Sandbox container GC (HIG-200) ---------------------------------------
    # Consumed by the sandbox-runner service; declared here so every
    # env-var-backed knob has one authoritative home. The runner reads these
    # from its own env loader, but they are mirrored here for discoverability
    # and so config tooling/docs surface them.
    sandbox_gc_enabled: bool = Field(
        default=True,
        validation_alias="KORTNY_SANDBOX_GC_ENABLED",
    )
    sandbox_gc_max_age_minutes: int = Field(
        default=60,
        validation_alias="KORTNY_SANDBOX_GC_MAX_AGE_MINUTES",
        gt=0,
    )
    sandbox_gc_interval_seconds: int = Field(
        default=600,
        validation_alias="KORTNY_SANDBOX_GC_INTERVAL_SECONDS",
        gt=0,
    )
    sandbox_gc_orphan_running_max_age_hours: int = Field(
        default=24,
        validation_alias="KORTNY_SANDBOX_GC_ORPHAN_RUNNING_MAX_AGE_HOURS",
        gt=0,
    )
    artifacts_dir: str | None = Field(
        default=None,
        validation_alias="KORTNY_ARTIFACTS_DIR",
    )
    public_base_url: str | None = Field(
        default=None,
        validation_alias="KORTNY_PUBLIC_BASE_URL",
    )
    preview_signing_secret: str | None = Field(
        default=None,
        validation_alias="KORTNY_PREVIEW_SIGNING_SECRET",
    )
    netlify_auth_token: str | None = Field(
        default=None,
        validation_alias="NETLIFY_AUTH_TOKEN",
    )
    vercel_token: str | None = Field(
        default=None,
        validation_alias="VERCEL_TOKEN",
    )
    vercel_team_id: str | None = Field(
        default=None,
        validation_alias="VERCEL_TEAM_ID",
    )
    temporal_address: str = Field(
        default="temporal:7233",
        validation_alias="TEMPORAL_ADDRESS",
        min_length=1,
    )
    temporal_namespace: str = Field(
        default="default",
        validation_alias="TEMPORAL_NAMESPACE",
        min_length=1,
    )
    temporal_task_queue: str = Field(
        default="kortny-workflows",
        validation_alias="TEMPORAL_TASK_QUEUE",
        min_length=1,
    )
    scheduler_poll_interval_seconds: float = Field(
        default=5.0,
        validation_alias="KORTNY_SCHEDULER_POLL_INTERVAL_SECONDS",
    )
    scheduler_materialize_limit: int = Field(
        default=50,
        validation_alias="KORTNY_SCHEDULER_MATERIALIZE_LIMIT",
    )
    scheduler_advisory_lock_key: int = Field(
        default=759340185,
        validation_alias="KORTNY_SCHEDULER_ADVISORY_LOCK_KEY",
    )
    witness_enabled: bool = Field(
        default=True,
        validation_alias="KORTNY_WITNESS_ENABLED",
    )
    autonomy_default_level: str = Field(
        default="balanced",
        validation_alias="KORTNY_AUTONOMY_DEFAULT_LEVEL",
    )
    app_home_enabled: bool = Field(
        default=True,
        validation_alias="KORTNY_APP_HOME_ENABLED",
    )
    assistant_enabled: bool = Field(
        default=True,
        validation_alias="KORTNY_ASSISTANT_ENABLED",
    )
    witness_deliver_private: bool = Field(
        default=False,
        validation_alias="KORTNY_WITNESS_DELIVER_PRIVATE",
    )
    witness_poll_interval_seconds: float = Field(
        default=300.0,
        validation_alias="KORTNY_WITNESS_POLL_INTERVAL_SECONDS",
    )
    witness_profile_scan_limit: int = Field(
        default=10,
        validation_alias="KORTNY_WITNESS_PROFILE_SCAN_LIMIT",
    )
    witness_delivery_limit: int = Field(
        default=5,
        validation_alias="KORTNY_WITNESS_DELIVERY_LIMIT",
        description=(
            "Deprecated: the per-tick drip limit is retired. Digest batching "
            "(KORTNY_WITNESS_DIGEST_MAX_ITEMS) is the delivery budget now."
        ),
    )
    witness_delivery_threshold: Decimal = Field(
        default=Decimal("0.55"),
        validation_alias="KORTNY_WITNESS_DELIVERY_THRESHOLD",
    )
    witness_digest_interval_hours: int = Field(
        default=24,
        validation_alias="KORTNY_WITNESS_DIGEST_INTERVAL_HOURS",
    )
    witness_digest_max_items: int = Field(
        default=5,
        validation_alias="KORTNY_WITNESS_DIGEST_MAX_ITEMS",
    )
    witness_quiet_hours_start: int | None = Field(
        default=None,
        validation_alias="KORTNY_WITNESS_QUIET_HOURS_START",
    )
    witness_quiet_hours_end: int | None = Field(
        default=None,
        validation_alias="KORTNY_WITNESS_QUIET_HOURS_END",
    )
    witness_scan_interval_seconds: int = Field(
        default=21_600,
        validation_alias="KORTNY_WITNESS_SCAN_INTERVAL_SECONDS",
    )
    witness_autopilot_enabled: bool = Field(
        default=True,
        validation_alias="KORTNY_WITNESS_AUTOPILOT_ENABLED",
    )
    witness_autopilot_limit: int = Field(
        default=1,
        validation_alias="KORTNY_WITNESS_AUTOPILOT_LIMIT",
    )
    witness_autopilot_min_confidence: Decimal = Field(
        default=Decimal("0.600"),
        validation_alias="KORTNY_WITNESS_AUTOPILOT_MIN_CONFIDENCE",
    )
    witness_automation_enabled: bool = Field(
        default=True,
        validation_alias="KORTNY_WITNESS_AUTOMATION_ENABLED",
    )
    witness_recurring_min_reinforcements: int = Field(
        default=3,
        validation_alias="KORTNY_WITNESS_RECURRING_MIN_REINFORCEMENTS",
        description=(
            "Re-observations a recurring candidate needs before its copy may "
            "claim recurrence (HIG-197). A single scan never claims it."
        ),
    )
    witness_recurring_min_span_days: int = Field(
        default=14,
        validation_alias="KORTNY_WITNESS_RECURRING_MIN_SPAN_DAYS",
        description=(
            "Minimum observation span (days, first_observed_at -> now) before "
            "a recurring candidate may claim recurrence in copy (HIG-197)."
        ),
    )
    witness_channel_posts_per_week: int = Field(
        default=1,
        validation_alias="KORTNY_WITNESS_CHANNEL_POSTS_PER_WEEK",
        description=(
            "Sliding-window budget for proactive Witness posts per channel "
            "(7-day window). 0 disables channel delivery entirely."
        ),
    )
    witness_drafts_per_channel_per_day: int = Field(
        default=1,
        validation_alias="KORTNY_WITNESS_DRAFTS_PER_CHANNEL_PER_DAY",
        description=(
            "Sliding-window budget for autopilot draft deliverables per "
            "channel (24-hour window). 0 disables the draft tier."
        ),
    )
    ambient_files_enabled: bool = Field(
        default=True,
        validation_alias="KORTNY_AMBIENT_FILES_ENABLED",
    )
    ambient_file_max_mb: int = Field(
        default=15,
        validation_alias="KORTNY_AMBIENT_FILE_MAX_MB",
    )
    ambient_file_briefs_per_day: int = Field(
        default=1,
        validation_alias="KORTNY_AMBIENT_FILE_BRIEFS_PER_DAY",
    )
    consolidator_enabled: bool = Field(
        default=True,
        validation_alias="KORTNY_CONSOLIDATOR_ENABLED",
    )
    consolidator_poll_interval_seconds: float = Field(
        default=600.0,
        validation_alias="KORTNY_CONSOLIDATOR_POLL_INTERVAL_SECONDS",
    )
    consolidator_min_new_items: int = Field(
        default=50,
        validation_alias="KORTNY_CONSOLIDATOR_MIN_NEW_ITEMS",
    )
    consolidator_min_interval_hours: float = Field(
        default=8.0,
        validation_alias="KORTNY_CONSOLIDATOR_MIN_INTERVAL_HOURS",
    )
    consolidator_quiet_minutes: float = Field(
        default=60.0,
        validation_alias="KORTNY_CONSOLIDATOR_QUIET_MINUTES",
    )
    consolidator_nightly_floor_hours: float = Field(
        default=24.0,
        validation_alias="KORTNY_CONSOLIDATOR_NIGHTLY_FLOOR_HOURS",
    )
    consolidator_advisory_lock_key: int = Field(
        default=759340187,
        validation_alias="KORTNY_CONSOLIDATOR_ADVISORY_LOCK_KEY",
    )
    skills_seed_advisory_lock_key: int = Field(
        default=759340188,
        validation_alias="KORTNY_SKILLS_SEED_ADVISORY_LOCK_KEY",
    )
    memory_recency_half_life_days: float = Field(
        default=14.0,
        validation_alias="KORTNY_MEMORY_RECENCY_HALF_LIFE_DAYS",
    )
    style_cards_enabled: bool = Field(
        default=True,
        validation_alias="KORTNY_STYLE_CARDS_ENABLED",
    )
    style_card_min_messages: int = Field(
        default=30,
        validation_alias="KORTNY_STYLE_CARD_MIN_MESSAGES",
    )
    kg_stale_days: int = Field(
        default=45,
        validation_alias="KORTNY_KG_STALE_DAYS",
    )
    embeddings_backend: Literal["local", "disabled"] = Field(
        default="local", validation_alias="KORTNY_EMBEDDINGS_BACKEND"
    )
    embeddings_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        validation_alias="KORTNY_EMBEDDINGS_MODEL",
        min_length=1,
    )
    tool_retrieval_top_k: int = Field(
        default=15, validation_alias="KORTNY_TOOL_RETRIEVAL_TOP_K"
    )
    skill_direct_similarity_threshold: float = Field(
        default=0.60, validation_alias="KORTNY_SKILL_DIRECT_THRESHOLD"
    )
    tool_selector_max_external_candidates: int = Field(
        default=24, validation_alias="TOOL_SELECTOR_MAX_EXTERNAL_CANDIDATES"
    )
    # 12k proved too small live: ~35 native cards alone exceed it, which used
    # to trim every external candidate out of the selector prompt.
    tool_selector_max_prompt_chars: int = Field(
        default=24000, validation_alias="TOOL_SELECTOR_MAX_PROMPT_CHARS"
    )
    tool_result_prompt_max_chars: int = Field(
        default=8000, validation_alias="TOOL_RESULT_PROMPT_MAX_CHARS"
    )
    tool_result_max_chars: int = Field(
        default=16000, validation_alias="KORTNY_TOOL_RESULT_MAX_CHARS"
    )

    composio_api_key: str = Field(validation_alias="COMPOSIO_API_KEY", min_length=1)
    composio_catalog_enabled: bool = Field(
        default=True, validation_alias="COMPOSIO_CATALOG_ENABLED"
    )
    composio_catalog_limit: int = Field(
        default=60, validation_alias="COMPOSIO_CATALOG_LIMIT"
    )
    composio_request_timeout_seconds: float = Field(
        # 30s: scrape/fetch-style Composio tools (e.g. firecrawl) routinely take
        # longer than 10s on heavy pages; a too-short read timeout was the
        # proximate cause of a hard task failure (the classification fix in
        # error_policy makes such a timeout recoverable regardless).
        default=30.0,
        validation_alias="COMPOSIO_REQUEST_TIMEOUT_SECONDS",
    )
    composio_sync_interval_hours: float = Field(
        default=6.0, validation_alias="KORTNY_COMPOSIO_SYNC_INTERVAL_HOURS"
    )
    composio_sync_page_size: int = Field(
        default=20, validation_alias="KORTNY_COMPOSIO_SYNC_PAGE_SIZE"
    )
    composio_sync_advisory_lock_key: int = Field(
        default=759340222, validation_alias="KORTNY_COMPOSIO_SYNC_ADVISORY_LOCK_KEY"
    )
    prompt_cache_enabled: bool = Field(
        default=True, validation_alias="KORTNY_PROMPT_CACHE_ENABLED"
    )
    mcp_enabled: bool = Field(default=True, validation_alias="KORTNY_MCP_ENABLED")
    mcp_tool_timeout_seconds: float = Field(
        default=60.0, validation_alias="KORTNY_MCP_TOOL_TIMEOUT_SECONDS"
    )
    # --- Prompt-injection / tool-trust baseline (HIG-169 P0) ------------------
    # Egress allowlist for outbound-post URL flagging. CSV of bare hostnames
    # (e.g. "example.com,docs.kortny.ai"). Empty (default) flags nothing by
    # name; the unfurl-off change at the posting boundary is the real exfil
    # fix, the URL flagging is observability only.
    egress_url_allowlist: str | None = Field(
        default=None, validation_alias="KORTNY_EGRESS_URL_ALLOWLIST"
    )
    # Trifecta flow gate: once untrusted-origin content has entered a task's
    # context, escalate approval for any subsequently-selected write/outward
    # tool. Default on; deterministic, no LLM.
    trifecta_gate_enabled: bool = Field(
        default=True, validation_alias="KORTNY_TRIFECTA_GATE_ENABLED"
    )
    brave_search_api_key: str | None = Field(
        default=None, validation_alias="BRAVE_SEARCH_API_KEY"
    )
    slack_file_read_max_bytes: int = Field(
        default=25 * 1024 * 1024,
        validation_alias="SLACK_FILE_READ_MAX_BYTES",
    )

    # --- Vision (HIG-279) ----------------------------------------------------
    vision_enabled: bool = Field(
        default=True,
        validation_alias="KORTNY_VISION_ENABLED",
    )
    vision_max_images_per_request: int = Field(
        default=5,
        validation_alias="KORTNY_VISION_MAX_IMAGES_PER_REQUEST",
    )
    vision_max_image_bytes: int = Field(
        default=10_485_760,  # 10 MB
        validation_alias="KORTNY_VISION_MAX_IMAGE_BYTES",
    )
    vision_max_total_image_bytes: int = Field(
        default=26_214_400,  # 25 MB
        validation_alias="KORTNY_VISION_MAX_TOTAL_IMAGE_BYTES",
    )
    # Comma-separated list of allowed MIME types for image attachments.
    vision_allowed_image_mimes: str = Field(
        default="image/png,image/jpeg,image/webp",
        validation_alias="KORTNY_VISION_ALLOWED_IMAGE_MIMES",
    )
    # Maximum number of pages to OCR when a scanned/image-only PDF is detected.
    # Pages beyond this cap are skipped and a truncation warning is recorded.
    pdf_ocr_max_pages: int = Field(
        default=20,
        validation_alias="KORTNY_PDF_OCR_MAX_PAGES",
    )

    # --- Document Studio VLM visual critic (HIG-244 slice 1) ------------------
    doc_visual_critic_enabled: bool = Field(
        default=False,
        validation_alias="KORTNY_DOC_VISUAL_CRITIC_ENABLED",
    )
    doc_visual_critic_max_pages: int = Field(
        default=8,
        validation_alias="KORTNY_DOC_VISUAL_CRITIC_MAX_PAGES",
    )
    document_visual_revision_enabled: bool = Field(
        default=False,
        validation_alias="KORTNY_DOCUMENT_VISUAL_REVISION_ENABLED",
    )
    document_visual_revision_max_iterations: int = Field(
        default=2,
        validation_alias="KORTNY_DOCUMENT_VISUAL_REVISION_MAX_ITERATIONS",
    )

    observability_enabled: bool = Field(
        default=True, validation_alias="OBSERVABILITY_ENABLED"
    )
    observability_capture_content: Literal["metadata", "summaries", "full"] = Field(
        default="summaries", validation_alias="OBSERVABILITY_CAPTURE_CONTENT"
    )
    otel_exporter_otlp_endpoint: str | None = Field(
        default=None, validation_alias="OTEL_EXPORTER_OTLP_ENDPOINT"
    )
    otel_exporter_otlp_headers: str | None = Field(
        default=None, validation_alias="OTEL_EXPORTER_OTLP_HEADERS"
    )
    otel_service_name: str = Field(
        default="kortny", validation_alias="OTEL_SERVICE_NAME", min_length=1
    )
    otel_trace_sampling_ratio: float = Field(
        default=1.0, validation_alias="OTEL_TRACE_SAMPLING_RATIO"
    )
    langfuse_enabled: bool = Field(default=False, validation_alias="LANGFUSE_ENABLED")
    langfuse_host: str | None = Field(default=None, validation_alias="LANGFUSE_HOST")
    langfuse_public_key: str | None = Field(
        default=None, validation_alias="LANGFUSE_PUBLIC_KEY"
    )
    langfuse_secret_key: str | None = Field(
        default=None, validation_alias="LANGFUSE_SECRET_KEY"
    )
    langfuse_prompts_enabled: bool = Field(
        default=False, validation_alias="LANGFUSE_PROMPTS_ENABLED"
    )
    langfuse_prompt_label: str | None = Field(
        default=None, validation_alias="LANGFUSE_PROMPT_LABEL"
    )
    kortny_release: str | None = Field(default=None, validation_alias="KORTNY_RELEASE")
    kortny_version: str | None = Field(default=None, validation_alias="KORTNY_VERSION")

    postgres_url: str = Field(validation_alias="POSTGRES_URL", min_length=1)
    encryption_key: str | None = Field(default=None, validation_alias="ENCRYPTION_KEY")

    # --- Browser automation (HIG-282) -----------------------------------------
    # When set, KORTNY_BROWSER_URL points at a running Playwright-MCP server
    # (e.g. http://browser:8931/mcp). The worker opens one persistent MCP
    # session per task and exposes browser tools through the normal tool pipeline.
    # Leave unset to disable browser entirely (default).
    browser_url: str | None = Field(default=None, validation_alias="KORTNY_BROWSER_URL")
    browser_session_idle_timeout_seconds: int = Field(
        default=120, validation_alias="KORTNY_BROWSER_SESSION_IDLE_TIMEOUT_SECONDS"
    )

    @field_validator(
        "brave_search_api_key",
        "encryption_key",
        "browser_url",
        "llm_cheap_model",
        "llm_standard_model",
        "llm_analysis_model",
        "llm_document_model",
        "llm_high_reasoning_model",
        "llm_humanizer_model",
        "llm_vision_model",
        "otel_exporter_otlp_endpoint",
        "otel_exporter_otlp_headers",
        "langfuse_host",
        "langfuse_public_key",
        "langfuse_secret_key",
        "langfuse_prompt_label",
        "kortny_release",
        "kortny_version",
        "sandbox_runner_url",
        "artifacts_dir",
        "public_base_url",
        "preview_signing_secret",
        "netlify_auth_token",
        "vercel_token",
        "vercel_team_id",
        "egress_url_allowlist",
        mode="before",
    )
    @classmethod
    def _blank_optional_strings_to_none(cls, value: Any) -> Any:
        if isinstance(value, str) and value.strip() == "":
            return None
        return value

    @field_validator("composio_api_key")
    @classmethod
    def _strip_required_composio_api_key(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("COMPOSIO_API_KEY cannot be blank")
        return stripped

    @field_validator("agent_runtime", mode="before")
    @classmethod
    def _reject_retired_adk_runtime(cls, value: object) -> object:
        if value == "adk":
            raise ValueError(
                "AGENT_RUNTIME=adk was removed (HIG-281). The custom runtime is "
                "now the only runtime; unset AGENT_RUNTIME or set it to 'custom'."
            )
        return value

    @field_validator(
        "llm_cheap_model",
        "llm_standard_model",
        "llm_analysis_model",
        "llm_document_model",
        "llm_high_reasoning_model",
        "llm_humanizer_model",
        "llm_vision_model",
    )
    @classmethod
    def _strip_optional_model(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @field_validator("slack_app_name")
    @classmethod
    def _normalize_app_name(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("SLACK_APP_NAME cannot be blank")
        return stripped

    @property
    def agent_display_name(self) -> str:
        """Human-facing name for the agent in prompts and Slack copy.

        Self-hosters set SLACK_APP_NAME to brand their install; everything the
        agent says about itself flows from here instead of a hardcoded "Kortny".
        An all-lowercase value (like the "kortny" default) is title-cased so it
        reads as a name; a value that already carries casing is left as authored
        (e.g. "ACME Bot" stays "ACME Bot").
        """

        name = self.slack_app_name
        return name.title() if name == name.lower() else name

    @property
    def browser_enabled(self) -> bool:
        """True when KORTNY_BROWSER_URL is configured."""
        return self.browser_url is not None

    @field_validator("slack_file_read_max_bytes")
    @classmethod
    def _positive_file_read_limit(cls, value: int) -> int:
        if value < 1:
            raise ValueError("SLACK_FILE_READ_MAX_BYTES must be at least 1")
        return value

    @field_validator("composio_catalog_limit")
    @classmethod
    def _valid_composio_catalog_limit(cls, value: int) -> int:
        if value < 1 or value > 1000:
            raise ValueError("COMPOSIO_CATALOG_LIMIT must be between 1 and 1000")
        return value

    @field_validator("composio_request_timeout_seconds")
    @classmethod
    def _valid_composio_timeout(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("COMPOSIO_REQUEST_TIMEOUT_SECONDS must be positive")
        return value

    @field_validator("response_humanizer_min_chars")
    @classmethod
    def _valid_response_humanizer_min_chars(cls, value: int) -> int:
        if value < 0:
            raise ValueError("RESPONSE_HUMANIZER_MIN_CHARS cannot be negative")
        return value

    @field_validator("planned_workflow_max_parallel_branches")
    @classmethod
    def _valid_planned_workflow_parallel_branches(cls, value: int) -> int:
        if value < 1 or value > 5:
            raise ValueError(
                "KORTNY_PLANNED_WORKFLOW_MAX_PARALLEL_BRANCHES must be between 1 and 5"
            )
        return value

    @field_validator("planned_workflow_cost_ceiling_usd")
    @classmethod
    def _valid_planned_workflow_cost_ceiling_usd(cls, value: float) -> float:
        if value <= 0:
            raise ValueError(
                "KORTNY_PLANNED_WORKFLOW_COST_CEILING_USD must be positive"
            )
        return value

    @field_validator("planned_workflow_max_branch_model_calls")
    @classmethod
    def _valid_planned_workflow_max_branch_model_calls(cls, value: int) -> int:
        if value < 1 or value > 20:
            raise ValueError(
                "KORTNY_PLANNED_WORKFLOW_MAX_BRANCH_MODEL_CALLS must be between 1 and 20"
            )
        return value

    @field_validator("planned_workflow_max_branch_tool_calls")
    @classmethod
    def _valid_planned_workflow_max_branch_tool_calls(cls, value: int) -> int:
        if value < 0 or value > 100:
            raise ValueError(
                "KORTNY_PLANNED_WORKFLOW_MAX_BRANCH_TOOL_CALLS must be between 0 and 100"
            )
        return value

    @field_validator("planned_workflow_max_total_tool_calls")
    @classmethod
    def _valid_planned_workflow_max_total_tool_calls(cls, value: int) -> int:
        if value < 0 or value > 200:
            raise ValueError(
                "KORTNY_PLANNED_WORKFLOW_MAX_TOTAL_TOOL_CALLS must be between 0 and 200"
            )
        return value

    @field_validator("sandbox_runner_url")
    @classmethod
    def _strip_optional_sandbox_runner_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip().rstrip("/")
        return stripped or None

    @field_validator("sandbox_runner_timeout_seconds")
    @classmethod
    def _valid_sandbox_runner_timeout_seconds(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("KORTNY_SANDBOX_RUNNER_TIMEOUT_SECONDS must be positive")
        return value

    @field_validator("browser_session_idle_timeout_seconds")
    @classmethod
    def _valid_browser_session_idle_timeout_seconds(cls, value: int) -> int:
        if value < 10:
            raise ValueError(
                "KORTNY_BROWSER_SESSION_IDLE_TIMEOUT_SECONDS must be at least 10"
            )
        return value

    @field_validator("sandbox_default_image")
    @classmethod
    def _strip_sandbox_default_image(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("KORTNY_SANDBOX_DEFAULT_IMAGE cannot be blank")
        return stripped

    @field_validator("scheduler_poll_interval_seconds")
    @classmethod
    def _valid_scheduler_poll_interval_seconds(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("KORTNY_SCHEDULER_POLL_INTERVAL_SECONDS must be positive")
        return value

    @field_validator("scheduler_materialize_limit")
    @classmethod
    def _valid_scheduler_materialize_limit(cls, value: int) -> int:
        if value < 1 or value > 500:
            raise ValueError(
                "KORTNY_SCHEDULER_MATERIALIZE_LIMIT must be between 1 and 500"
            )
        return value

    @field_validator("witness_poll_interval_seconds")
    @classmethod
    def _valid_witness_poll_interval_seconds(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("KORTNY_WITNESS_POLL_INTERVAL_SECONDS must be positive")
        return value

    @field_validator("witness_profile_scan_limit")
    @classmethod
    def _valid_witness_profile_scan_limit(cls, value: int) -> int:
        if value < 0 or value > 500:
            raise ValueError(
                "KORTNY_WITNESS_PROFILE_SCAN_LIMIT must be between 0 and 500"
            )
        return value

    @field_validator("witness_delivery_limit")
    @classmethod
    def _valid_witness_delivery_limit(cls, value: int) -> int:
        if value < 0 or value > 100:
            raise ValueError("KORTNY_WITNESS_DELIVERY_LIMIT must be between 0 and 100")
        return value

    @field_validator("witness_scan_interval_seconds")
    @classmethod
    def _valid_witness_scan_interval_seconds(cls, value: int) -> int:
        if value < 0:
            raise ValueError("KORTNY_WITNESS_SCAN_INTERVAL_SECONDS cannot be negative")
        return value

    @field_validator("witness_delivery_threshold")
    @classmethod
    def _valid_witness_delivery_threshold(cls, value: Decimal) -> Decimal:
        if value < 0 or value > 1:
            raise ValueError(
                "KORTNY_WITNESS_DELIVERY_THRESHOLD must be between 0 and 1"
            )
        return value

    @field_validator("witness_digest_interval_hours")
    @classmethod
    def _valid_witness_digest_interval_hours(cls, value: int) -> int:
        if value < 1 or value > 168:
            raise ValueError(
                "KORTNY_WITNESS_DIGEST_INTERVAL_HOURS must be between 1 and 168"
            )
        return value

    @field_validator("witness_digest_max_items")
    @classmethod
    def _valid_witness_digest_max_items(cls, value: int) -> int:
        if value < 1 or value > 25:
            raise ValueError("KORTNY_WITNESS_DIGEST_MAX_ITEMS must be between 1 and 25")
        return value

    @field_validator("witness_quiet_hours_start", "witness_quiet_hours_end")
    @classmethod
    def _valid_witness_quiet_hours(cls, value: int | None) -> int | None:
        if value is not None and (value < 0 or value > 23):
            raise ValueError(
                "KORTNY_WITNESS_QUIET_HOURS_START/END must be an hour 0-23 (UTC)"
            )
        return value

    @field_validator("witness_channel_posts_per_week")
    @classmethod
    def _valid_witness_channel_posts_per_week(cls, value: int) -> int:
        if value < 0 or value > 25:
            raise ValueError(
                "KORTNY_WITNESS_CHANNEL_POSTS_PER_WEEK must be between 0 and 25"
            )
        return value

    @field_validator("witness_drafts_per_channel_per_day")
    @classmethod
    def _valid_witness_drafts_per_channel_per_day(cls, value: int) -> int:
        if value < 0 or value > 25:
            raise ValueError(
                "KORTNY_WITNESS_DRAFTS_PER_CHANNEL_PER_DAY must be between 0 and 25"
            )
        return value

    @field_validator("witness_recurring_min_reinforcements")
    @classmethod
    def _valid_witness_recurring_min_reinforcements(cls, value: int) -> int:
        if value < 1:
            raise ValueError(
                "KORTNY_WITNESS_RECURRING_MIN_REINFORCEMENTS must be at least 1"
            )
        return value

    @field_validator("witness_recurring_min_span_days")
    @classmethod
    def _valid_witness_recurring_min_span_days(cls, value: int) -> int:
        if value < 0:
            raise ValueError(
                "KORTNY_WITNESS_RECURRING_MIN_SPAN_DAYS cannot be negative"
            )
        return value

    @field_validator("ambient_file_max_mb")
    @classmethod
    def _valid_ambient_file_max_mb(cls, value: int) -> int:
        if value < 1 or value > 200:
            raise ValueError("KORTNY_AMBIENT_FILE_MAX_MB must be between 1 and 200")
        return value

    @field_validator("ambient_file_briefs_per_day")
    @classmethod
    def _valid_ambient_file_briefs_per_day(cls, value: int) -> int:
        if value < 0 or value > 20:
            raise ValueError(
                "KORTNY_AMBIENT_FILE_BRIEFS_PER_DAY must be between 0 and 20"
            )
        return value

    @field_validator(
        "consolidator_poll_interval_seconds",
        "consolidator_min_interval_hours",
        "consolidator_quiet_minutes",
        "consolidator_nightly_floor_hours",
        "memory_recency_half_life_days",
    )
    @classmethod
    def _positive_consolidator_interval(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("Consolidator intervals and half-life must be positive")
        return value

    @field_validator("consolidator_min_new_items", "kg_stale_days")
    @classmethod
    def _positive_consolidator_count(cls, value: int) -> int:
        if value < 1:
            raise ValueError(
                "KORTNY_CONSOLIDATOR_MIN_NEW_ITEMS and KORTNY_KG_STALE_DAYS "
                "must be at least 1"
            )
        return value

    @field_validator("tool_retrieval_top_k")
    @classmethod
    def _valid_tool_retrieval_top_k(cls, value: int) -> int:
        if value < 1 or value > 200:
            raise ValueError("KORTNY_TOOL_RETRIEVAL_TOP_K must be between 1 and 200")
        return value

    @field_validator("skill_direct_similarity_threshold")
    @classmethod
    def _valid_skill_direct_similarity_threshold(cls, value: float) -> float:
        if value < 0 or value > 1:
            raise ValueError("KORTNY_SKILL_DIRECT_THRESHOLD must be between 0 and 1")
        return value

    @field_validator("tool_selector_max_external_candidates")
    @classmethod
    def _valid_tool_selector_max_external_candidates(cls, value: int) -> int:
        if value < 1 or value > 200:
            raise ValueError(
                "TOOL_SELECTOR_MAX_EXTERNAL_CANDIDATES must be between 1 and 200"
            )
        return value

    @field_validator("tool_selector_max_prompt_chars")
    @classmethod
    def _valid_tool_selector_max_prompt_chars(cls, value: int) -> int:
        if value < 1000 or value > 100000:
            raise ValueError(
                "TOOL_SELECTOR_MAX_PROMPT_CHARS must be between 1000 and 100000"
            )
        return value

    @field_validator("tool_result_prompt_max_chars")
    @classmethod
    def _valid_tool_result_prompt_max_chars(cls, value: int) -> int:
        if value < 1000:
            raise ValueError("TOOL_RESULT_PROMPT_MAX_CHARS must be at least 1000")
        return value

    @field_validator("tool_result_max_chars")
    @classmethod
    def _valid_tool_result_max_chars(cls, value: int) -> int:
        if value < 1000:
            raise ValueError("KORTNY_TOOL_RESULT_MAX_CHARS must be at least 1000")
        return value

    @field_validator("otel_trace_sampling_ratio")
    @classmethod
    def _valid_trace_sampling_ratio(cls, value: float) -> float:
        if value < 0 or value > 1:
            raise ValueError("OTEL_TRACE_SAMPLING_RATIO must be between 0 and 1")
        return value


class SettingsError(RuntimeError):
    """Raised when application settings are missing or invalid."""


def load_settings(env_file: str | Path | None = ".env") -> Settings:
    """Load settings and raise a concise startup error on validation failure."""

    try:
        settings_kwargs: dict[str, Any] = {"_env_file": env_file}
        return Settings(**settings_kwargs)
    except ValidationError as exc:
        raise SettingsError(_format_validation_error(exc)) from exc


def _format_validation_error(exc: ValidationError) -> str:
    failed_fields = sorted({str(error["loc"][0]) for error in exc.errors()})
    return "Missing or invalid configuration: " + ", ".join(failed_fields)
