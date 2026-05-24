"""Typed intent classification contracts."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class IntentSurface(StrEnum):
    """Where an inbound user message came from."""

    app_mention = "app_mention"
    dm = "dm"
    channel_message = "channel_message"


class IntentClassification(StrEnum):
    """Coarse app-wide message intent."""

    task_request = "task_request"
    follow_up = "follow_up"
    memory_candidate = "memory_candidate"
    clarification = "clarification"
    cancel_or_retry = "cancel_or_retry"
    third_person_reference = "third_person_reference"
    ambient_observation = "ambient_observation"
    ignore = "ignore"


class ModelTier(StrEnum):
    """Initial model tier hint before the full model router exists."""

    cheap = "cheap"
    standard = "standard"
    strong = "strong"


class IntentRequest(BaseModel):
    """Input sent to the intent classifier."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    surface: IntentSurface
    app_name: str = Field(default="kortny", min_length=1)
    is_thread_follow_up: bool = False
    has_files: bool = False

    @field_validator("text", "app_name")
    @classmethod
    def strip_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("value cannot be blank")
        return stripped


class IntentDecision(BaseModel):
    """Structured routing decision returned by the classifier."""

    model_config = ConfigDict(extra="forbid")

    addressed_to_kortny: bool
    classification: IntentClassification
    confidence: float = Field(ge=0.0, le=1.0)
    should_create_task: bool
    should_ack_with_reaction: bool
    suggested_reaction: str | None = None
    needs_channel_context: bool
    needs_thread_context: bool
    needs_file_context: bool
    likely_tools: list[str] = Field(default_factory=list)
    model_tier: ModelTier
    reason: str = Field(min_length=1, max_length=500)

    @field_validator("reason")
    @classmethod
    def normalize_reason(cls, value: str) -> str:
        return " ".join(value.split())

    @field_validator("suggested_reaction")
    @classmethod
    def normalize_reaction(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().strip(":").casefold()
        return normalized or None
