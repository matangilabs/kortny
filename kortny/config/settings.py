"""Typed application settings loaded from environment variables."""

from __future__ import annotations

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

    composio_api_key: str | None = Field(
        default=None, validation_alias="COMPOSIO_API_KEY"
    )
    brave_search_api_key: str | None = Field(
        default=None, validation_alias="BRAVE_SEARCH_API_KEY"
    )
    slack_file_read_max_bytes: int = Field(
        default=25 * 1024 * 1024,
        validation_alias="SLACK_FILE_READ_MAX_BYTES",
    )

    observability_enabled: bool = Field(
        default=True, validation_alias="OBSERVABILITY_ENABLED"
    )
    observability_capture_content: Literal["metadata", "summaries", "full"] = Field(
        default="metadata", validation_alias="OBSERVABILITY_CAPTURE_CONTENT"
    )
    otel_exporter_otlp_endpoint: str | None = Field(
        default=None, validation_alias="OTEL_EXPORTER_OTLP_ENDPOINT"
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

    @field_validator(
        "composio_api_key",
        "brave_search_api_key",
        "llm_cheap_model",
        "llm_standard_model",
        "llm_analysis_model",
        "llm_document_model",
        "llm_high_reasoning_model",
        "otel_exporter_otlp_endpoint",
        "langfuse_host",
        "langfuse_public_key",
        "langfuse_secret_key",
        "langfuse_prompt_label",
        "kortny_release",
        "kortny_version",
        mode="before",
    )
    @classmethod
    def _blank_optional_strings_to_none(cls, value: Any) -> Any:
        if isinstance(value, str) and value.strip() == "":
            return None
        return value

    @field_validator(
        "llm_cheap_model",
        "llm_standard_model",
        "llm_analysis_model",
        "llm_document_model",
        "llm_high_reasoning_model",
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

    @field_validator("slack_file_read_max_bytes")
    @classmethod
    def _positive_file_read_limit(cls, value: int) -> int:
        if value < 1:
            raise ValueError("SLACK_FILE_READ_MAX_BYTES must be at least 1")
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
