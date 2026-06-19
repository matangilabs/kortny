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

from kortny.config.settings import Settings
from kortny.db.models import LLMProvider as DbLLMProvider
from kortny.db.models import ModelPricing
from kortny.llm.litellm_catalog import model_supports_vision
from kortny.llm.routing import ModelRoute, ModelRouteTier
from kortny.llm.types import ChatMessage, Completion, ImagePart, LLMProvider, TokenUsage
from kortny.observability import (
    capture_content_mode,
    log_observation,
    observe_task_event,
    record_span_exception,
    set_span_attributes,
    start_span,
)
from kortny.observability.content import (
    llm_span_attributes,
    render_chat_messages,
    render_completion,
)
from kortny.prompts.registry import prompt_version as registered_prompt_version
from kortny.tasks import TaskService
from kortny.tools.types import JsonObject, JsonSchema

USD_QUANTUM = Decimal("0.000001")
TOKENS_PER_MTOK = Decimal("1000000")
DEFAULT_CACHE_WRITE_MULTIPLIER = Decimal("1.25")
DEFAULT_CACHE_READ_MULTIPLIER = Decimal("0.10")
logger = logging.getLogger(__name__)

# HIG-220 effort steering: one-shot utility prompts get a per-call completion-
# token cap so they cannot burn hundreds of reasoning tokens for a tiny
# structured/short output. Caps are deliberately GENEROUS — comfortably larger
# than the expected output — so they bound verbosity without ever truncating the
# answer. Aggressive per-prompt tuning (and Anthropic thinking-effort config) is
# a live-validated follow-up; this is the safe floor.
UTILITY_PROMPT_OUTPUT_CLAMP: dict[str, int] = {
    "kortny.intent_classifier": 1024,
    "kortny.project_inference_namer": 256,
    "kortny.ack_generator": 256,
}


class ModelPricingNotFoundError(LookupError):
    """Raised when no model pricing row can price an LLM call."""


class VisionUnsupportedError(Exception):
    """Raised when a request contains images but the configured model cannot accept them."""


class ImageGuardError(Exception):
    """Raised when image attachments violate a size, count, or policy guard (HIG-279)."""


def enforce_image_guards(
    messages: Sequence[ChatMessage],
    settings: Settings,
) -> None:
    """Validate image attachments against workspace policy guards.

    Call this BEFORE forwarding messages to any provider.  If ``messages``
    contain no images the function returns immediately and is a no-op — text-
    only requests are completely unaffected.

    Raises:
        ImageGuardError: if any policy limit is exceeded or vision is disabled.
    """
    images: list[ImagePart] = [img for m in messages for img in m.images]
    if not images:
        return

    if not settings.vision_enabled:
        raise ImageGuardError("Vision is disabled for this workspace.")

    if len(images) > settings.vision_max_images_per_request:
        raise ImageGuardError(
            f"Too many images: {len(images)} (max {settings.vision_max_images_per_request})."
        )

    for img in images:
        if img.byte_size > settings.vision_max_image_bytes:
            raise ImageGuardError("An attached image is too large.")

    if sum(img.byte_size for img in images) > settings.vision_max_total_image_bytes:
        raise ImageGuardError("Attached images exceed the total size limit.")

    allowed_mimes = {
        m.strip() for m in settings.vision_allowed_image_mimes.split(",") if m.strip()
    }
    for img in images:
        if img.mime not in allowed_mimes:
            raise ImageGuardError(f"Unsupported image type {img.mime}.")


def assert_vision_capable(
    provider_kind: str,
    model_identifier: str,
    catalog_row: object = None,
) -> None:
    """Assert that the provider/model combination supports vision.

    Call this AFTER ``enforce_image_guards`` when the message list contains
    images.  Raises ``VisionUnsupportedError`` if the model is not known to
    accept image input.
    """
    if not model_supports_vision(
        provider_kind,
        model_identifier,
        catalog_row=catalog_row,
    ):
        raise VisionUnsupportedError(
            "No vision-capable model is configured to read images."
        )


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
        settings: Settings | None = None,
    ) -> None:
        self.session = session
        self.provider = provider
        self.provider_name = DbLLMProvider(provider_name)
        self.task_service = task_service or TaskService(session)
        self.model_route = model_route
        self._settings = settings

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
        max_output_tokens: int | None = None,
    ) -> Completion:
        """Complete a turn, price it, and persist the usage rollup.

        ``max_output_tokens`` caps this call's completion tokens; when not given,
        a small utility prompt (intent, project namer, ack) is clamped by name
        from ``UTILITY_PROMPT_OUTPUT_CLAMP`` (HIG-220 effort steering). Caps are
        generous enough never to truncate the structured output — they only
        bound runaway verbosity/reasoning on one-shot utility calls.
        """

        # --- Vision guards (HIG-279) -----------------------------------------
        # Pure helpers run only when images are present; text-only requests are
        # completely unaffected.  Settings are optional so callers that don't
        # have them (e.g. tests for the pricing logic) are not affected.
        if self._settings is not None:
            enforce_image_guards(messages, self._settings)
            _images = [img for m in messages for img in m.images]
            if _images:
                assert_vision_capable(
                    self.provider_name.value,
                    self.provider.model,
                )

        prompt_name = prompt_name or _default_prompt_name(
            tools=tools,
            response_format=response_format,
        )
        if max_output_tokens is None and prompt_name:
            max_output_tokens = UTILITY_PROMPT_OUTPUT_CLAMP.get(prompt_name)
        # Stamp the registered prompt version (HIG-203) so usage rows correlate
        # quality with prompt changes; explicit caller version wins.
        if prompt_version is None:
            prompt_version = registered_prompt_version(prompt_name)
        started = time.perf_counter()
        capture_mode = capture_content_mode()
        request_messages = render_chat_messages(messages, capture_mode)
        start_fields = self._base_observation_fields(
            messages=messages,
            tools=tools,
            response_format=response_format,
            prompt_name=prompt_name,
            prompt_source=prompt_source,
            prompt_label=prompt_label,
            prompt_version=prompt_version,
        )
        started_fields = dict(start_fields)
        if request_messages is not None:
            started_fields["request_messages"] = request_messages
        with start_span(
            "llm.complete",
            task=self.task_service.get_task(task_id),
            attributes={
                **start_fields,
                **llm_span_attributes(request_messages=request_messages),
                "openinference.span.kind": "LLM",
                "llm.provider": self.provider_name.value,
                "llm.model_name": self.provider.model,
            },
        ):
            observe_task_event(
                self.task_service,
                task_id,
                "llm_call_started",
                logger=logger,
                **started_fields,
            )

            try:
                # Only forward the clamp when set, so providers that don't take
                # the kwarg are unaffected on the common (unclamped) path.
                provider_kwargs: dict[str, object] = {}
                if max_output_tokens is not None:
                    provider_kwargs["max_output_tokens"] = max_output_tokens
                completion = self.provider.complete(
                    messages,
                    tools,
                    response_format=response_format,
                    **provider_kwargs,  # type: ignore[arg-type]
                )
            except Exception as exc:
                record_span_exception(exc)
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
            response_content = render_completion(completion, capture_mode)
            metadata = {
                **start_fields,
                "model": model,
                "response_id": completion.response_id,
                "latency_ms": latency_ms,
                "has_content": bool(completion.content),
                "cache_creation_input_tokens": (
                    completion.usage.cache_creation_input_tokens
                ),
                "cache_read_input_tokens": completion.usage.cache_read_input_tokens,
                "tool_call_count": len(completion.tool_calls),
                "tool_call_names": [
                    tool_call.name for tool_call in completion.tool_calls
                ],
            }
            image_count = sum(len(m.images) for m in messages)
            if image_count:
                metadata["image_count"] = image_count
                metadata["vision_request"] = True
            if response_content is not None:
                metadata["response"] = response_content
            self.task_service.record_llm_usage(
                task_id,
                provider=self.provider_name,
                model=model,
                model_tier=self.model_tier,
                input_tokens=completion.usage.input_tokens,
                output_tokens=completion.usage.output_tokens,
                cache_creation_input_tokens=(
                    completion.usage.cache_creation_input_tokens
                ),
                cache_read_input_tokens=completion.usage.cache_read_input_tokens,
                cost_usd=cost_usd,
                metadata=metadata,
            )
            total_tokens = (
                completion.usage.input_tokens + completion.usage.output_tokens
            )
            set_span_attributes(
                {
                    "langfuse.observation.type": "generation",
                    "langfuse.observation.model.name": model,
                    "langfuse.observation.usage_details": {
                        "input": completion.usage.input_tokens,
                        "output": completion.usage.output_tokens,
                        "total": total_tokens,
                    },
                    "langfuse.observation.cost_details": {"total": str(cost_usd)},
                    "langfuse.observation.prompt.name": prompt_name,
                    "gen_ai.response.model": model,
                    "gen_ai.usage.input_tokens": completion.usage.input_tokens,
                    "gen_ai.usage.output_tokens": completion.usage.output_tokens,
                    "gen_ai.usage.total_tokens": total_tokens,
                    "gen_ai.usage.cost": str(cost_usd),
                    "openinference.span.kind": "LLM",
                    "llm.provider": self.provider_name.value,
                    "llm.model": model,
                    "llm.model_name": model,
                    "llm.response_id": completion.response_id,
                    "llm.input_tokens": completion.usage.input_tokens,
                    "llm.output_tokens": completion.usage.output_tokens,
                    "llm.total_tokens": total_tokens,
                    "llm.cache_creation_input_tokens": (
                        completion.usage.cache_creation_input_tokens
                    ),
                    "llm.cache_read_input_tokens": (
                        completion.usage.cache_read_input_tokens
                    ),
                    "llm.token_count.prompt": completion.usage.input_tokens,
                    "llm.token_count.completion": completion.usage.output_tokens,
                    "llm.token_count.total": total_tokens,
                    "llm.cost_usd": str(cost_usd),
                    "llm.latency_ms": latency_ms,
                    "llm.tool_count": len(tools),
                    "llm.tool_call_count": len(completion.tool_calls),
                    **llm_span_attributes(response=response_content),
                }
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
                total_tokens=total_tokens,
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
    """Calculate USD cost from token usage and per-million-token pricing.

    Splits the prompt total into uncached / cache-creation / cache-read bands
    (HIG-196 D5). ``input_tokens`` is the *total* prompt count; cache-creation
    and cache-read are partitions within it. The uncached remainder is clamped
    at 0 so a provider reporting more cache tokens than the total can never
    produce a negative charge. Cache-write/read multipliers come from the
    pricing row (defaults 1.25x / 0.1x).
    """

    base_input = pricing.input_price_per_mtok
    # server_default only fires on a DB insert; an in-memory pricing row built
    # without these columns leaves them None — fall back to the D5 defaults.
    write_multiplier = (
        pricing.cache_write_multiplier
        if pricing.cache_write_multiplier is not None
        else DEFAULT_CACHE_WRITE_MULTIPLIER
    )
    read_multiplier = (
        pricing.cache_read_multiplier
        if pricing.cache_read_multiplier is not None
        else DEFAULT_CACHE_READ_MULTIPLIER
    )
    cache_creation = Decimal(usage.cache_creation_input_tokens)
    cache_read = Decimal(usage.cache_read_input_tokens)
    uncached = Decimal(usage.input_tokens) - cache_creation - cache_read
    if uncached < 0:
        uncached = Decimal(0)

    input_cost = (
        uncached * base_input
        + cache_creation * write_multiplier * base_input
        + cache_read * read_multiplier * base_input
    )
    output_cost = Decimal(usage.output_tokens) * pricing.output_price_per_mtok
    return ((input_cost + output_cost) / TOKENS_PER_MTOK).quantize(
        USD_QUANTUM,
        rounding=ROUND_HALF_UP,
    )
