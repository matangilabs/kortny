"""LiteLLM-backed provider and model catalog helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from decimal import Decimal
from functools import lru_cache
from typing import Any, cast


@dataclass(frozen=True, slots=True)
class LiteLLMProviderOption:
    kind: str
    label: str
    description: str
    default_probe_model: str
    default_base_url: str | None = None
    supports_endpoint_discovery: bool = False
    needs_base_url: bool = False


@dataclass(frozen=True, slots=True)
class LiteLLMModelCandidate:
    model_identifier: str
    display_name: str
    provider_kind: str
    source: str
    capabilities: dict[str, object]
    metadata: dict[str, object]
    input_price_per_mtok: Decimal | None
    output_price_per_mtok: Decimal | None


OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
OPENROUTER_MODELS_TIMEOUT_SECONDS = 5.0
PRICE_PER_MTOK_QUANTUM = Decimal("0.000001")
MAX_PRICE_PER_MTOK = Decimal("999999.999999")


LITELLM_PROVIDER_OPTIONS: tuple[LiteLLMProviderOption, ...] = (
    LiteLLMProviderOption(
        kind="openrouter",
        label="OpenRouter",
        description="Multi-provider gateway with one API key.",
        default_probe_model="openrouter/openai/gpt-4o-mini",
    ),
    LiteLLMProviderOption(
        kind="openai",
        label="OpenAI",
        description="OpenAI models and OpenAI-compatible endpoints.",
        default_probe_model="gpt-4o-mini",
        supports_endpoint_discovery=True,
    ),
    LiteLLMProviderOption(
        kind="anthropic",
        label="Anthropic",
        description="Claude models through Anthropic's API.",
        default_probe_model="claude-3-5-haiku-20241022",
        supports_endpoint_discovery=True,
    ),
    LiteLLMProviderOption(
        kind="gemini",
        label="Google Gemini",
        description="Gemini API models.",
        default_probe_model="gemini/gemini-2.0-flash",
        supports_endpoint_discovery=True,
    ),
    LiteLLMProviderOption(
        kind="xai",
        label="xAI",
        description="Grok models through xAI.",
        default_probe_model="xai/grok-2-latest",
        supports_endpoint_discovery=True,
    ),
    LiteLLMProviderOption(
        kind="fireworks_ai",
        label="Fireworks AI",
        description="Hosted open model inference.",
        default_probe_model="fireworks_ai/accounts/fireworks/models/llama-v3p1-8b-instruct",
        supports_endpoint_discovery=True,
    ),
    LiteLLMProviderOption(
        kind="azure",
        label="Azure OpenAI",
        description="Azure-hosted OpenAI deployments.",
        default_probe_model="azure/gpt-4o-mini",
        needs_base_url=True,
    ),
    LiteLLMProviderOption(
        kind="bedrock",
        label="Amazon Bedrock",
        description="AWS Bedrock models. Usually needs AWS environment or role config.",
        default_probe_model="bedrock/anthropic.claude-3-5-haiku-20241022-v1:0",
    ),
    LiteLLMProviderOption(
        kind="ollama",
        label="Ollama",
        description="Self-hosted local models through Ollama.",
        default_probe_model="ollama/llama3.1",
        default_base_url="http://localhost:11434",
        needs_base_url=True,
    ),
)

_PROVIDER_BY_KIND = {option.kind: option for option in LITELLM_PROVIDER_OPTIONS}


def litellm_provider_options() -> tuple[LiteLLMProviderOption, ...]:
    """Return curated provider options for the dashboard."""

    return LITELLM_PROVIDER_OPTIONS


def litellm_provider_option(kind: str) -> LiteLLMProviderOption | None:
    """Return a provider option by LiteLLM provider kind."""

    return _PROVIDER_BY_KIND.get(kind)


def litellm_model_candidates(
    provider_kind: str,
    *,
    limit: int | None = 24,
) -> tuple[LiteLLMModelCandidate, ...]:
    """Return local LiteLLM model-cost-map candidates for a provider."""

    model_cost = _litellm_model_cost()
    candidates = [
        _candidate_from_model_cost(
            model_identifier=model_identifier,
            provider_kind=provider_kind,
            info=info,
            source="litellm_catalog",
        )
        for model_identifier, info in model_cost.items()
        if _model_cost_row_matches(provider_kind, model_identifier, info)
    ]
    ranked = _rank_candidates(provider_kind, candidates)
    if limit is None:
        return tuple(ranked)
    return tuple(ranked[:limit])


def model_candidate_for_identifier(
    provider_kind: str,
    model_identifier: str,
    *,
    include_provider_catalog: bool = False,
) -> LiteLLMModelCandidate | None:
    """Return pricing/capability metadata for one provider model identifier.

    DB-backed model config stores the provider-native identifier. OpenRouter is
    the main wrinkle: LiteLLM runtime calls need an ``openrouter/`` prefix, but
    env tier config commonly stores ``anthropic/...`` or ``deepseek/...``. This
    helper preserves the caller's identifier while searching both shapes.
    """

    local = _local_model_candidate_for_identifier(
        provider_kind=provider_kind,
        model_identifier=model_identifier,
    )
    if local is not None:
        return local
    if include_provider_catalog and provider_kind == "openrouter":
        return _openrouter_model_candidate_for_identifier(model_identifier)
    return None


def litellm_endpoint_model_candidates(
    provider_kind: str,
    *,
    api_key: str,
    api_base: str | None = None,
    limit: int | None = 24,
) -> tuple[LiteLLMModelCandidate, ...]:
    """Ask LiteLLM/provider endpoint for valid models when supported."""

    if provider_kind == "openrouter":
        return openrouter_model_candidates(limit=limit)

    option = litellm_provider_option(provider_kind)
    if option is None or not option.supports_endpoint_discovery:
        return ()
    import litellm

    models = litellm.get_valid_models(
        check_provider_endpoint=True,
        custom_llm_provider=provider_kind,
        api_key=api_key,
        api_base=api_base,
    )
    local_by_model = {
        candidate.model_identifier: candidate
        for candidate in litellm_model_candidates(provider_kind, limit=500)
    }
    candidates: list[LiteLLMModelCandidate] = []
    for model_identifier in _unique_strings(models):
        local = local_by_model.get(model_identifier)
        if local is not None:
            candidates.append(
                LiteLLMModelCandidate(
                    model_identifier=local.model_identifier,
                    display_name=local.display_name,
                    provider_kind=local.provider_kind,
                    source="provider_api",
                    capabilities=local.capabilities,
                    metadata=local.metadata,
                    input_price_per_mtok=local.input_price_per_mtok,
                    output_price_per_mtok=local.output_price_per_mtok,
                )
            )
        else:
            candidates.append(
                LiteLLMModelCandidate(
                    model_identifier=model_identifier,
                    display_name=_display_name(model_identifier),
                    provider_kind=provider_kind,
                    source="provider_api",
                    capabilities={},
                    metadata={"litellm_provider": provider_kind},
                    input_price_per_mtok=None,
                    output_price_per_mtok=None,
                )
            )
        if limit is not None and len(candidates) >= limit:
            break
    return tuple(candidates)


def openrouter_model_candidates(
    *,
    limit: int | None = None,
) -> tuple[LiteLLMModelCandidate, ...]:
    """Return full OpenRouter catalog candidates from the OpenRouter models API."""

    candidates: list[LiteLLMModelCandidate] = []
    for item in _openrouter_model_items():
        candidate = _openrouter_model_candidate_from_item(item)
        if candidate is None:
            continue
        candidates.append(candidate)
        if limit is not None and len(candidates) >= limit:
            break
    return tuple(_rank_candidates("openrouter", candidates))


def check_litellm_provider_key(
    *,
    provider_kind: str,
    api_key: str,
    model: str,
    api_base: str | None = None,
) -> bool:
    """Validate a provider key through LiteLLM helpers."""

    import litellm

    option = litellm_provider_option(provider_kind)
    if api_base or (option is not None and option.supports_endpoint_discovery):
        models = litellm.get_valid_models(
            check_provider_endpoint=True,
            custom_llm_provider=provider_kind,
            api_key=api_key,
            api_base=api_base,
        )
        return bool(models)
    return bool(litellm.check_valid_key(model=model, api_key=api_key))


def default_probe_model(provider_kind: str, fallback: str | None = None) -> str:
    """Return a reasonable provider-specific model for credential tests."""

    option = litellm_provider_option(provider_kind)
    if option is not None:
        return option.default_probe_model
    if fallback:
        return fallback
    return provider_kind


def _litellm_model_cost() -> Mapping[str, Mapping[str, Any]]:
    import litellm

    return cast(Mapping[str, Mapping[str, Any]], litellm.model_cost)


def _local_model_candidate_for_identifier(
    *,
    provider_kind: str,
    model_identifier: str,
) -> LiteLLMModelCandidate | None:
    model_cost = _litellm_model_cost()
    for lookup_identifier in _provider_lookup_identifiers(
        provider_kind, model_identifier
    ):
        info = model_cost.get(lookup_identifier)
        if not isinstance(info, Mapping):
            continue
        if not _model_cost_row_matches(provider_kind, lookup_identifier, info):
            continue
        candidate = _candidate_from_model_cost(
            model_identifier=model_identifier,
            provider_kind=provider_kind,
            info=info,
            source="litellm_catalog",
        )
        if (
            lookup_identifier == model_identifier
            and candidate.model_identifier == model_identifier
        ):
            return candidate
        return LiteLLMModelCandidate(
            model_identifier=model_identifier,
            display_name=candidate.display_name,
            provider_kind=candidate.provider_kind,
            source=candidate.source,
            capabilities=candidate.capabilities,
            metadata={
                **candidate.metadata,
                "litellm_model_identifier": lookup_identifier,
            },
            input_price_per_mtok=candidate.input_price_per_mtok,
            output_price_per_mtok=candidate.output_price_per_mtok,
        )
    return None


def _provider_lookup_identifiers(
    provider_kind: str, model_identifier: str
) -> tuple[str, ...]:
    normalized = model_identifier.strip()
    if provider_kind != "openrouter":
        return (normalized,)
    if normalized.startswith("openrouter/"):
        return (normalized, normalized.removeprefix("openrouter/"))
    return (normalized, f"openrouter/{normalized}")


def _model_cost_row_matches(
    provider_kind: str,
    model_identifier: str,
    info: Mapping[str, Any],
) -> bool:
    if model_identifier == "sample_spec":
        return False
    if info.get("litellm_provider") != provider_kind:
        return False
    return info.get("mode") in {None, "chat", "completion"}


def _openrouter_model_candidate_for_identifier(
    model_identifier: str,
) -> LiteLLMModelCandidate | None:
    lookup_values = set(_provider_lookup_identifiers("openrouter", model_identifier))
    lookup_values.update(
        value.removeprefix("openrouter/")
        for value in tuple(lookup_values)
        if value.startswith("openrouter/")
    )
    for item in _openrouter_model_items():
        if not isinstance(item, dict):
            continue
        keys = {
            value.strip()
            for value in (item.get("id"), item.get("canonical_slug"))
            if isinstance(value, str) and value.strip()
        }
        keys.update(f"openrouter/{value}" for value in tuple(keys))
        if keys.isdisjoint(lookup_values):
            continue
        return _openrouter_model_candidate_from_item(
            item,
            model_identifier=model_identifier,
        )
    return None


def _openrouter_model_candidate_from_item(
    item: Mapping[str, Any],
    *,
    model_identifier: str | None = None,
) -> LiteLLMModelCandidate | None:
    raw_identifier = model_identifier or item.get("id") or item.get("canonical_slug")
    if not isinstance(raw_identifier, str) or not raw_identifier.strip():
        return None
    resolved_identifier = raw_identifier.strip()
    if model_identifier is None:
        resolved_identifier = resolved_identifier.removeprefix("openrouter/")
    raw_name = item.get("name")
    display_name = (
        raw_name.strip()
        if isinstance(raw_name, str) and raw_name.strip()
        else _display_name(resolved_identifier)
    )
    pricing = item.get("pricing")
    pricing_map = pricing if isinstance(pricing, Mapping) else {}
    return LiteLLMModelCandidate(
        model_identifier=resolved_identifier,
        display_name=display_name,
        provider_kind="openrouter",
        source="provider_api",
        capabilities=_openrouter_capabilities(item),
        metadata=_openrouter_metadata(item),
        input_price_per_mtok=_price_per_mtok(pricing_map.get("prompt")),
        output_price_per_mtok=_price_per_mtok(pricing_map.get("completion")),
    )


@lru_cache(maxsize=1)
def _openrouter_model_items() -> tuple[Mapping[str, Any], ...]:
    try:
        import httpx
    except ImportError:
        return ()
    try:
        response = httpx.get(
            OPENROUTER_MODELS_URL,
            params={"output_modalities": "all"},
            timeout=OPENROUTER_MODELS_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except Exception:
        return ()
    payload = response.json()
    if not isinstance(payload, dict):
        return ()
    data = payload.get("data")
    if not isinstance(data, list):
        return ()
    return tuple(item for item in data if isinstance(item, Mapping))


def _openrouter_capabilities(item: Mapping[str, Any]) -> dict[str, object]:
    capabilities: dict[str, object] = {}
    context_length = item.get("context_length")
    if isinstance(context_length, int):
        capabilities["max_input_tokens"] = context_length
    supported_parameters = item.get("supported_parameters")
    if isinstance(supported_parameters, list):
        capabilities["supported_parameters"] = [
            value for value in supported_parameters if isinstance(value, str)
        ]
    architecture = item.get("architecture")
    if isinstance(architecture, dict):
        input_modalities = architecture.get("input_modalities")
        output_modalities = architecture.get("output_modalities")
        for key in (
            "input_modalities",
            "output_modalities",
            "tokenizer",
            "instruct_type",
        ):
            value = architecture.get(key)
            if value is not None:
                capabilities[key] = value
        supports_text_input = (
            isinstance(input_modalities, list) and "text" in input_modalities
        )
        supports_text_output = (
            isinstance(output_modalities, list) and "text" in output_modalities
        )
        capabilities["runtime_routable"] = supports_text_input and supports_text_output
        capabilities["runtime_routing_reason"] = (
            "text_input_output"
            if supports_text_input and supports_text_output
            else "non_text_modalities"
        )
    return capabilities


def _openrouter_metadata(item: Mapping[str, Any]) -> dict[str, object]:
    metadata: dict[str, object] = {"litellm_provider": "openrouter"}
    for key in (
        "id",
        "canonical_slug",
        "created",
        "description",
        "architecture",
        "top_provider",
        "per_request_limits",
    ):
        value = item.get(key)
        if value is not None:
            metadata[f"openrouter_{key}"] = value
    return metadata


def _candidate_from_model_cost(
    *,
    model_identifier: str,
    provider_kind: str,
    info: Mapping[str, Any],
    source: str,
) -> LiteLLMModelCandidate:
    resolved_identifier = (
        model_identifier.removeprefix("openrouter/")
        if provider_kind == "openrouter"
        else model_identifier
    )
    metadata = {
        key: value
        for key, value in info.items()
        if key
        in {
            "litellm_provider",
            "max_input_tokens",
            "max_output_tokens",
            "max_tokens",
            "mode",
            "source",
            "supported_endpoints",
            "supported_modalities",
            "supported_output_modalities",
        }
    }
    if resolved_identifier != model_identifier:
        metadata["litellm_model_identifier"] = model_identifier
    return LiteLLMModelCandidate(
        model_identifier=resolved_identifier,
        display_name=_display_name(resolved_identifier),
        provider_kind=provider_kind,
        source=source,
        capabilities=_capabilities_from_model_cost(info),
        metadata=metadata,
        input_price_per_mtok=_price_per_mtok(info.get("input_cost_per_token")),
        output_price_per_mtok=_price_per_mtok(info.get("output_cost_per_token")),
    )


def _capabilities_from_model_cost(info: Mapping[str, Any]) -> dict[str, object]:
    return {
        key: value
        for key, value in info.items()
        if key.startswith("supports_") and isinstance(value, bool)
    }


def _price_per_mtok(value: object) -> Decimal | None:
    if value is None:
        return None
    try:
        price = (Decimal(str(value)) * Decimal("1000000")).quantize(
            PRICE_PER_MTOK_QUANTUM
        )
    except Exception:
        return None
    if price < 0 or price > MAX_PRICE_PER_MTOK:
        return None
    return price


def _rank_candidates(
    provider_kind: str,
    candidates: Iterable[LiteLLMModelCandidate],
) -> list[LiteLLMModelCandidate]:
    preferred = {
        "openrouter": (
            "openai/gpt-4o-mini",
            "anthropic/claude-sonnet-4",
            "deepseek/deepseek-chat",
            "openrouter/openai/gpt-4o-mini",
            "openrouter/anthropic/claude-sonnet-4",
            "openrouter/deepseek/deepseek-chat",
        ),
        "openai": ("gpt-4o-mini", "gpt-4o", "gpt-5.1"),
        "anthropic": (
            "claude-3-5-haiku-20241022",
            "claude-sonnet-4-20250514",
            "claude-3-7-sonnet-20250219",
        ),
        "gemini": ("gemini/gemini-2.0-flash", "gemini/gemini-2.5-flash"),
        "xai": ("xai/grok-2-latest", "xai/grok-3"),
    }.get(provider_kind, ())
    preferred_rank = {model: index for index, model in enumerate(preferred)}
    return sorted(
        candidates,
        key=lambda candidate: (
            preferred_rank.get(candidate.model_identifier, len(preferred_rank) + 1),
            candidate.display_name.lower(),
            candidate.model_identifier.lower(),
        ),
    )


def _display_name(model_identifier: str) -> str:
    return model_identifier.removeprefix("openrouter/")


def _unique_strings(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value or value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output
