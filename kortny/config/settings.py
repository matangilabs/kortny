"""Typed application settings loaded from environment variables."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

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

    llm_provider: LLMProvider = Field(validation_alias="LLM_PROVIDER")
    llm_api_key: str = Field(validation_alias="LLM_API_KEY", min_length=1)
    llm_model: str = Field(validation_alias="LLM_MODEL", min_length=1)

    composio_api_key: str | None = Field(
        default=None, validation_alias="COMPOSIO_API_KEY"
    )
    brave_search_api_key: str | None = Field(
        default=None, validation_alias="BRAVE_SEARCH_API_KEY"
    )

    postgres_url: str = Field(validation_alias="POSTGRES_URL", min_length=1)

    @field_validator("composio_api_key", "brave_search_api_key", mode="before")
    @classmethod
    def _blank_optional_strings_to_none(cls, value: Any) -> Any:
        if isinstance(value, str) and value.strip() == "":
            return None
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
