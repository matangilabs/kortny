"""LLM usage tracking and model-pricing cost calculation."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import ROUND_HALF_UP, Decimal

from sqlalchemy import select
from sqlalchemy.orm import Session

from kortny.db.models import LLMProvider as DbLLMProvider
from kortny.db.models import ModelPricing
from kortny.llm.routing import ModelRoute, ModelRouteTier
from kortny.llm.types import ChatMessage, Completion, LLMProvider, TokenUsage
from kortny.observability import log_observation, observe_task_event
from kortny.tasks import TaskService
from kortny.tools.types import JsonObject, JsonSchema

USD_QUANTUM = Decimal("0.000001")
TOKENS_PER_MTOK = Decimal("1000000")
logger = logging.getLogger(__name__)


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
        model_route: ModelRoute | None = None,
    ) -> None:
        self.session = session
        self.provider = provider
        self.provider_name = DbLLMProvider(provider_name)
        self.task_service = task_service or TaskService(session)
        self.model_route = model_route

    def complete(
        self,
        *,
        task_id: uuid.UUID,
        messages: Sequence[ChatMessage],
        tools: Sequence[JsonSchema] = (),
        response_format: JsonObject | None = None,
        prompt_name: str | None = None,
        prompt_source: str = "code",
        prompt_label: str | None = None,
        prompt_version: str | None = None,
    ) -> Completion:
        """Complete a turn, price it, and persist the usage rollup."""

        prompt_name = prompt_name or _default_prompt_name(
            tools=tools,
            response_format=response_format,
        )
        started = time.perf_counter()
        start_fields = self._base_observation_fields(
            messages=messages,
            tools=tools,
            response_format=response_format,
            prompt_name=prompt_name,
            prompt_source=prompt_source,
            prompt_label=prompt_label,
            prompt_version=prompt_version,
        )
        observe_task_event(
            self.task_service,
            task_id,
            "llm_call_started",
            logger=logger,
            **start_fields,
        )

        try:
            completion = self.provider.complete(
                messages,
                tools,
                response_format=response_format,
            )
        except Exception as exc:
            latency_ms = _latency_ms(started)
            observe_task_event(
                self.task_service,
                task_id,
                "llm_call_failed",
                event_type="error",
                logger=logger,
                level=logging.ERROR,
                latency_ms=latency_ms,
                error_type=type(exc).__name__,
                error_summary=str(exc),
                **start_fields,
            )
            raise

        model = completion.model or self.provider.model
        cost_usd = completion.cost_usd
        if cost_usd is None:
            pricing = self.get_pricing(model)
            cost_usd = calculate_cost_usd(completion.usage, pricing)

        latency_ms = _latency_ms(started)
        metadata = {
            **start_fields,
            "model": model,
            "response_id": completion.response_id,
            "latency_ms": latency_ms,
            "has_content": bool(completion.content),
            "tool_call_count": len(completion.tool_calls),
            "tool_call_names": [tool_call.name for tool_call in completion.tool_calls],
        }
        self.task_service.record_llm_usage(
            task_id,
            provider=self.provider_name,
            model=model,
            model_tier=self.model_tier,
            input_tokens=completion.usage.input_tokens,
            output_tokens=completion.usage.output_tokens,
            cost_usd=cost_usd,
            metadata=metadata,
        )
        log_observation(
            logger,
            "llm_call_completed",
            task=self.task_service.get_task(task_id),
            provider=self.provider_name.value,
            model=model,
            model_tier=self.model_tier,
            route_reason=self.route_reason,
            prompt_name=prompt_name,
            prompt_source=prompt_source,
            prompt_label=prompt_label,
            prompt_version=prompt_version,
            response_id=completion.response_id,
            input_tokens=completion.usage.input_tokens,
            output_tokens=completion.usage.output_tokens,
            total_tokens=completion.usage.input_tokens + completion.usage.output_tokens,
            cost_usd=str(cost_usd),
            latency_ms=latency_ms,
            tool_count=len(tools),
            tool_call_count=len(completion.tool_calls),
        )
        return completion

    @property
    def model_tier(self) -> str | None:
        if self.model_route is None:
            return None
        tier = self.model_route.tier
        if isinstance(tier, ModelRouteTier):
            return tier.value
        return str(tier)

    @property
    def route_reason(self) -> str | None:
        if self.model_route is None:
            return None
        return self.model_route.reason

    def _base_observation_fields(
        self,
        *,
        messages: Sequence[ChatMessage],
        tools: Sequence[JsonSchema],
        response_format: JsonObject | None,
        prompt_name: str | None,
        prompt_source: str,
        prompt_label: str | None,
        prompt_version: str | None,
    ) -> JsonObject:
        return {
            "provider": self.provider_name.value,
            "model": self.provider.model,
            "model_tier": self.model_tier,
            "route_reason": self.route_reason,
            "prompt_name": prompt_name,
            "prompt_source": prompt_source,
            "prompt_label": prompt_label,
            "prompt_version": prompt_version,
            "message_count": len(messages),
            "tool_count": len(tools),
            "response_format_type": response_format.get("type")
            if response_format
            else None,
        }

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


def _latency_ms(started: float) -> int:
    return max(0, int((time.perf_counter() - started) * 1000))


def _default_prompt_name(
    *,
    tools: Sequence[JsonSchema],
    response_format: JsonObject | None,
) -> str | None:
    if response_format and response_format.get("type") == "json_object":
        return "kortny.intent_classifier"
    if tools:
        return "kortny.agent_coordinator.system"
    return None


def calculate_cost_usd(usage: TokenUsage, pricing: ModelPricing) -> Decimal:
    """Calculate USD cost from token usage and per-million-token pricing."""

    input_cost = Decimal(usage.input_tokens) * pricing.input_price_per_mtok
    output_cost = Decimal(usage.output_tokens) * pricing.output_price_per_mtok
    return ((input_cost + output_cost) / TOKENS_PER_MTOK).quantize(
        USD_QUANTUM,
        rounding=ROUND_HALF_UP,
    )
