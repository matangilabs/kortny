"""Settings for the read-only dashboard service."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import Field, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DashboardAuthMode(StrEnum):
    """Supported dashboard authentication modes."""

    bootstrap = "bootstrap"
    slack = "slack"
    hybrid = "hybrid"


class DashboardSettings(BaseSettings):
    """Dashboard-only settings.

    The dashboard intentionally avoids loading the full Slack/LLM runtime settings so
    it can stay a small read-only operational surface over the database.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    postgres_url: str = Field(validation_alias="POSTGRES_URL", min_length=1)
    username: str = Field(
        default="kortny", validation_alias="DASHBOARD_USERNAME", min_length=1
    )
    password: str = Field(
        default="change-me", validation_alias="DASHBOARD_PASSWORD", min_length=1
    )
    session_secret: str = Field(
        default="change-me-dashboard-session-secret",
        validation_alias="DASHBOARD_SESSION_SECRET",
        min_length=16,
    )
    secure_cookies: bool = Field(
        default=False, validation_alias="DASHBOARD_SECURE_COOKIES"
    )
    auth_mode: DashboardAuthMode = Field(
        default=DashboardAuthMode.bootstrap,
        validation_alias="DASHBOARD_AUTH_MODE",
    )
    slack_client_id: str | None = Field(
        default=None, validation_alias="DASHBOARD_SLACK_CLIENT_ID"
    )
    slack_client_secret: str | None = Field(
        default=None, validation_alias="DASHBOARD_SLACK_CLIENT_SECRET"
    )
    slack_redirect_uri: str | None = Field(
        default=None, validation_alias="DASHBOARD_SLACK_REDIRECT_URI"
    )
    slack_oauth_state_ttl_minutes: int = Field(
        default=10,
        validation_alias="DASHBOARD_SLACK_OAUTH_STATE_TTL_MINUTES",
        ge=1,
        le=60,
    )
    artifacts_dir: str | None = Field(
        default=None, validation_alias="KORTNY_ARTIFACTS_DIR"
    )
    preview_signing_secret: str | None = Field(
        default=None, validation_alias="KORTNY_PREVIEW_SIGNING_SECRET"
    )

    @field_validator("username", "password", "session_secret")
    @classmethod
    def _strip_required_string(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("cannot be blank")
        return stripped

    @field_validator(
        "slack_client_id",
        "slack_client_secret",
        "slack_redirect_uri",
        "artifacts_dir",
        "preview_signing_secret",
    )
    @classmethod
    def _strip_optional_string(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @model_validator(mode="after")
    def _require_slack_login_config_for_slack_modes(self) -> DashboardSettings:
        if (
            self.auth_mode in {DashboardAuthMode.slack, DashboardAuthMode.hybrid}
            and not self.slack_login_configured
        ):
            raise ValueError(
                "DASHBOARD_SLACK_CLIENT_ID, DASHBOARD_SLACK_CLIENT_SECRET, "
                "and DASHBOARD_SLACK_REDIRECT_URI are required when "
                "DASHBOARD_AUTH_MODE is slack or hybrid"
            )
        return self

    @property
    def bootstrap_login_enabled(self) -> bool:
        """Whether the environment username/password login should be shown."""

        return self.auth_mode in {
            DashboardAuthMode.bootstrap,
            DashboardAuthMode.hybrid,
        }

    @property
    def slack_login_configured(self) -> bool:
        """Whether Slack OpenID settings are complete enough to start login."""

        return bool(
            self.slack_client_id
            and self.slack_client_secret
            and self.slack_redirect_uri
        )

    @property
    def slack_login_enabled(self) -> bool:
        """Whether Slack login is enabled and configured."""

        return (
            self.auth_mode in {DashboardAuthMode.slack, DashboardAuthMode.hybrid}
            and self.slack_login_configured
        )


class DashboardSettingsError(RuntimeError):
    """Raised when dashboard settings are missing or invalid."""


def load_dashboard_settings(
    env_file: str | Path | None = ".env",
) -> DashboardSettings:
    """Load dashboard settings with concise errors."""

    try:
        settings_kwargs: dict[str, Any] = {"_env_file": env_file}
        return DashboardSettings(**settings_kwargs)
    except ValidationError as exc:
        failed_fields = sorted({str(error["loc"][0]) for error in exc.errors()})
        message = "Missing or invalid dashboard configuration: " + ", ".join(
            failed_fields
        )
        raise DashboardSettingsError(message) from exc
