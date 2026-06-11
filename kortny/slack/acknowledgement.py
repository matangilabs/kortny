"""Slack acknowledgement text generation."""

from __future__ import annotations

import logging
from typing import Protocol

from sqlalchemy.orm import Session

from kortny.config import Settings
from kortny.db.models import LLMProvider as DbLLMProvider
from kortny.db.models import Task
from kortny.llm import (
    ChatMessage,
    LLMProvider,
    LLMService,
    ModelRouter,
    ModelRouteTier,
)
from kortny.llm.runtime_config import (
    create_provider_for_selection,
    select_runtime_model,
)
from kortny.observe.style_cards import load_channel_style
from kortny.tasks import TaskService

ROOT_ACK_FALLBACK_TEXT = "I'll take a look and post back here."
ACK_SYSTEM_PROMPT = (
    "Write one short Slack bridge acknowledgement for the user's request. "
    "Make it specific to what they asked, natural, and calm, but do not answer "
    "the request. "
    "Use first person. Maximum 14 words. No emoji. No markdown. "
    "For capability questions, say that you'll outline where you can help. "
    "For simple questions, say that you'll answer directly. "
    "Do not mention tasks, queues, workers, tools, or internal systems. "
    "Do not ask a question. Do not say 'On it'. Do not list capabilities, "
    "sources, facts, or conclusions."
)
ACK_MAX_CHARS = 140
logger = logging.getLogger(__name__)


class AcknowledgementGenerator(Protocol):
    """Generates a visible acknowledgement for a root Slack request."""

    def generate(
        self,
        *,
        session: Session,
        task: Task,
        task_service: TaskService,
    ) -> str:
        """Return acknowledgement text for a root request."""


class StaticAcknowledgementGenerator:
    """Deterministic fallback acknowledgement generator."""

    def generate(
        self,
        *,
        session: Session,
        task: Task,
        task_service: TaskService,
    ) -> str:
        return ROOT_ACK_FALLBACK_TEXT


class LLMAcknowledgementGenerator:
    """Generates short, query-aware Slack acknowledgements with the LLM."""

    def __init__(
        self,
        *,
        settings: Settings,
        provider: LLMProvider | None = None,
        provider_name: DbLLMProvider | str | None = None,
    ) -> None:
        self.settings = settings
        self.provider = provider
        self.provider_name = DbLLMProvider(provider_name) if provider_name else None

    def generate(
        self,
        *,
        session: Session,
        task: Task,
        task_service: TaskService,
    ) -> str:
        model_route = ModelRouter(self.settings).route_for_tier(
            ModelRouteTier.cheap_fast,
            reason="acknowledgement",
        )
        if self.provider is None:
            selection = select_runtime_model(
                session=session,
                settings=self.settings,
                installation_id=task.installation_id,
                model_route=model_route,
            )
            provider = create_provider_for_selection(
                settings=self.settings,
                selection=selection,
            )
            provider_name = self.provider_name or selection.provider_name
            model_route = selection.model_route
        else:
            provider = self.provider
            provider_name = self.provider_name or DbLLMProvider(
                self.settings.llm_provider
            )
        completion = LLMService(
            session=session,
            provider=provider,
            provider_name=provider_name,
            task_service=task_service,
            model_route=model_route,
        ).complete(
            task_id=task.id,
            messages=(
                ChatMessage(
                    role="system",
                    content=ACK_SYSTEM_PROMPT
                    + _channel_register_line(
                        session=session,
                        task=task,
                        settings=self.settings,
                    ),
                ),
                ChatMessage(role="user", content=task.input),
            ),
            prompt_name="kortny.ack_generator",
        )
        return sanitize_acknowledgement(completion.content)


def _channel_register_line(
    *,
    session: Session,
    task: Task,
    settings: Settings,
) -> str:
    """One register hint line when the channel has a learned style card.

    Flag off, DM surface, or no card => empty string, so the ack prompt is
    byte-identical to the pre-style-card behavior.
    """

    if not settings.style_cards_enabled:
        return ""
    if task.slack_channel_id.startswith("D"):
        return ""
    try:
        style = load_channel_style(
            session,
            installation_id=task.installation_id,
            channel_id=task.slack_channel_id,
        )
    except Exception:
        logger.exception("channel register lookup failed task_id=%s", task.id)
        return ""
    if style.card is None:
        return ""
    return f"\nChannel register: {style.card.formality}, {style.card.brevity}."


def generate_acknowledgement(
    generator: AcknowledgementGenerator,
    *,
    session: Session,
    task: Task,
    task_service: TaskService,
) -> str:
    """Generate acknowledgement text, falling back if generation fails."""

    try:
        return generator.generate(
            session=session,
            task=task,
            task_service=task_service,
        )
    except Exception as exc:
        logger.exception("slack acknowledgement generation failed task_id=%s", task.id)
        task_service.append_event(
            task,
            "log",
            {
                "message": "acknowledgement_generation_failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        return ROOT_ACK_FALLBACK_TEXT


def sanitize_acknowledgement(text: str | None) -> str:
    """Normalize model-generated acknowledgement text."""

    if text is None:
        return ROOT_ACK_FALLBACK_TEXT
    normalized = " ".join(text.strip().strip('"').strip("'").split())
    if not normalized:
        return ROOT_ACK_FALLBACK_TEXT
    if len(normalized) > ACK_MAX_CHARS:
        normalized = normalized[: ACK_MAX_CHARS - 1].rstrip() + "."
    return normalized
