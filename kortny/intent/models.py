"""Typed intent classification contracts."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ResponseDepth = Literal["quick_response", "standard_tool_task", "deep_workflow"]
TimeSensitivity = Literal["interactive", "relaxed"]
DepthSource = Literal["llm", "deterministic_override", "default"]


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


class IntentFragment(BaseModel):
    """One actionable intent inside a single Slack message."""

    model_config = ConfigDict(extra="forbid")

    type: IntentClassification
    objective: str = Field(min_length=1, max_length=500)
    should_execute: bool = True
    likely_tools: list[str] = Field(default_factory=list)
    route: str | None = None
    needs_channel_context: bool | None = None
    needs_thread_context: bool | None = None
    needs_file_context: bool | None = None

    @field_validator("objective")
    @classmethod
    def normalize_objective(cls, value: str) -> str:
        return " ".join(value.split())

    @field_validator("route")
    @classmethod
    def normalize_route(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


class IntentRequest(BaseModel):
    """Input sent to the intent classifier."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    surface: IntentSurface
    app_name: str = Field(default="kortny", min_length=1)
    is_thread_follow_up: bool = False
    has_files: bool = False
    connected_integrations: tuple[str, ...] = ()

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
    # HIG-277: gate for persona injection. Persona framing only helps
    # role-relative asks ("my plate") and hurts factual lookups (PRISM), so it
    # is injected only when this is set. Defaults False — fail safe to neutral.
    persona_relevant: bool = False
    response_depth: ResponseDepth = "standard_tool_task"
    time_sensitivity: TimeSensitivity = "interactive"
    toolkit_affinity: tuple[str, ...] = ()
    depth_source: DepthSource = "default"
    primary_intent: IntentFragment | None = None
    secondary_intents: list[IntentFragment] = Field(default_factory=list)

    @field_validator("toolkit_affinity", mode="before")
    @classmethod
    def normalize_toolkit_affinity(cls, value: object) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str | bytes):
            return ()
        if not isinstance(value, list | tuple):
            return ()
        cleaned: list[str] = []
        for item in value:
            if not isinstance(item, str):
                continue
            normalized = " ".join(item.split()).strip()
            if normalized and normalized not in cleaned:
                cleaned.append(normalized)
        return tuple(cleaned)

    def routing_classification(self) -> IntentClassification:
        """Return the intent classification that should drive execution."""

        if self.primary_intent is not None and self.primary_intent.should_execute:
            return self.primary_intent.type
        return self.classification

    def routing_should_create_task(self) -> bool:
        """Return whether the execution-driving intent should create work."""

        if self.primary_intent is not None:
            return self.primary_intent.should_execute
        return self.should_create_task

    def routing_likely_tools(self) -> list[str]:
        """Return likely tools for the execution-driving intent."""

        if (
            self.primary_intent is not None
            and self.primary_intent.should_execute
            and self.primary_intent.likely_tools
        ):
            return list(self.primary_intent.likely_tools)
        return list(self.likely_tools)

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
