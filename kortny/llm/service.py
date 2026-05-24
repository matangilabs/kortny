"""LLM usage tracking and model-pricing cost calculation."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import ROUND_HALF_UP, Decimal

from sqlalchemy import select
from sqlalchemy.orm import Session

from kortny.db.models import LLMProvider as DbLLMProvider
from kortny.db.models import ModelPricing
from kortny.llm.types import ChatMessage, Completion, LLMProvider, TokenUsage
from kortny.tasks import TaskService
from kortny.tools.types import JsonObject, JsonSchema

USD_QUANTUM = Decimal("0.000001")
TOKENS_PER_MTOK = Decimal("1000000")


class ModelPricingNotFoundError(LookupError):
    """Raised when no model pricing row can price an LLM call."""


class LLMService:
    """Calls an LLM provider and records usage through the task service."""

    def __init__(
        self,
        *,
        session: Session,
        provider: LLMProvider,
        provider_name: DbLLMProvider | str,
        task_service: TaskService | None = None,
    ) -> None:
        self.session = session
        self.provider = provider
        self.provider_name = DbLLMProvider(provider_name)
        self.task_service = task_service or TaskService(session)

    def complete(
        self,
        *,
        task_id: uuid.UUID,
        messages: Sequence[ChatMessage],
        tools: Sequence[JsonSchema] = (),
        response_format: JsonObject | None = None,
    ) -> Completion:
        """Complete a turn, price it, and persist the usage rollup."""

        completion = self.provider.complete(
            messages,
            tools,
            response_format=response_format,
        )
        model = completion.model or self.provider.model
        cost_usd = completion.cost_usd
        if cost_usd is None:
            pricing = self.get_pricing(model)
            cost_usd = calculate_cost_usd(completion.usage, pricing)

        self.task_service.record_llm_usage(
            task_id,
            provider=self.provider_name,
            model=model,
            input_tokens=completion.usage.input_tokens,
            output_tokens=completion.usage.output_tokens,
            cost_usd=cost_usd,
        )
        return completion

    def get_pricing(
        self,
        model: str,
        *,
        at: datetime | None = None,
    ) -> ModelPricing:
        """Return the most recent pricing row effective at the given time."""

        effective_at = at or datetime.now(UTC)
        pricing = self.session.scalar(
            select(ModelPricing)
            .where(
                ModelPricing.provider == self.provider_name,
                ModelPricing.model == model,
                ModelPricing.effective_from <= effective_at,
            )
            .order_by(ModelPricing.effective_from.desc())
            .limit(1)
        )
        if pricing is None:
            raise ModelPricingNotFoundError(
                f"No model_pricing row for {self.provider_name.value}/{model}"
            )
        return pricing


def calculate_cost_usd(usage: TokenUsage, pricing: ModelPricing) -> Decimal:
    """Calculate USD cost from token usage and per-million-token pricing."""

    input_cost = Decimal(usage.input_tokens) * pricing.input_price_per_mtok
    output_cost = Decimal(usage.output_tokens) * pricing.output_price_per_mtok
    return ((input_cost + output_cost) / TOKENS_PER_MTOK).quantize(
        USD_QUANTUM,
        rounding=ROUND_HALF_UP,
    )
