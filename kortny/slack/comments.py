"""Slack-facing result comment generation."""

from __future__ import annotations

import logging
from typing import Protocol

from sqlalchemy.orm import Session

from kortny.config import Settings
from kortny.db.models import Artifact, Task
from kortny.db.models import LLMProvider as DbLLMProvider
from kortny.llm import (
    ChatMessage,
    LLMProvider,
    LLMService,
    ModelRouter,
    ModelRouteTier,
    create_llm_provider,
)
from kortny.tasks import TaskService

ARTIFACT_COMMENT_FALLBACK_TEXT = "I finished this and attached it here."
ARTIFACT_COMMENT_SYSTEM_PROMPT = (
    "Write one short Slack message to accompany a generated file attachment. "
    "Make it specific to the user's request and natural. Maximum 14 words. "
    "No emoji. No markdown. Do not mention tasks, artifacts, workers, or tools. "
    "Do not ask a question."
)
ARTIFACT_COMMENT_MAX_CHARS = 160
logger = logging.getLogger(__name__)


class ArtifactCommentGenerator(Protocol):
    """Generates Slack-facing comments for uploaded artifacts."""

    def generate(
        self,
        *,
        session: Session,
        task: Task,
        artifact: Artifact,
        task_service: TaskService,
    ) -> str:
        """Return text to post with an uploaded artifact."""


class StaticArtifactCommentGenerator:
    """Deterministic artifact comment fallback."""

    def generate(
        self,
        *,
        session: Session,
        task: Task,
        artifact: Artifact,
        task_service: TaskService,
    ) -> str:
        return ARTIFACT_COMMENT_FALLBACK_TEXT


class LLMArtifactCommentGenerator:
    """Generates short, human Slack comments for artifact uploads."""

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
        artifact: Artifact,
        task_service: TaskService,
    ) -> str:
        model_route = ModelRouter(self.settings).route_for_tier(
            ModelRouteTier.cheap_fast,
            reason="artifact_comment",
        )
        provider = self.provider or create_llm_provider(
            self.settings,
            model=model_route.model,
        )
        provider_name = self.provider_name or DbLLMProvider(self.settings.llm_provider)
        completion = LLMService(
            session=session,
            provider=provider,
            provider_name=provider_name,
            task_service=task_service,
            model_route=model_route,
        ).complete(
            task_id=task.id,
            messages=(
                ChatMessage(role="system", content=ARTIFACT_COMMENT_SYSTEM_PROMPT),
                ChatMessage(
                    role="user",
                    content=(
                        f"User request: {task.input}\n"
                        f"Generated filename: {artifact.filename}\n"
                        f"Internal result summary: {task.result_summary or ''}"
                    ),
                ),
            ),
        )
        return sanitize_artifact_comment(completion.content)


def generate_artifact_comment(
    generator: ArtifactCommentGenerator,
    *,
    session: Session,
    task: Task,
    artifact: Artifact,
    task_service: TaskService,
) -> str:
    """Generate artifact upload text, falling back if generation fails."""

    try:
        return generator.generate(
            session=session,
            task=task,
            artifact=artifact,
            task_service=task_service,
        )
    except Exception as exc:
        logger.exception("slack artifact comment generation failed task_id=%s", task.id)
        task_service.append_event(
            task,
            "log",
            {
                "message": "artifact_comment_generation_failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "artifact_id": str(artifact.id),
                "filename": artifact.filename,
            },
        )
        return ARTIFACT_COMMENT_FALLBACK_TEXT


def sanitize_artifact_comment(text: str | None) -> str:
    """Normalize model-generated artifact comments."""

    if text is None:
        return ARTIFACT_COMMENT_FALLBACK_TEXT
    normalized = " ".join(text.strip().strip('"').strip("'").split())
    if not normalized:
        return ARTIFACT_COMMENT_FALLBACK_TEXT
    if len(normalized) > ARTIFACT_COMMENT_MAX_CHARS:
        normalized = normalized[: ARTIFACT_COMMENT_MAX_CHARS - 1].rstrip() + "."
    return normalized
