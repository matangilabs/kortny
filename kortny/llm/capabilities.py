"""Capability metadata normalization for LiteLLM model identifiers.

Given a LiteLLM model identifier, normalize the raw ``litellm.model_cost``
flags into a consistent capability dict. Pure function — no DB, no network.
"""

from __future__ import annotations

from typing import Any


def normalize_model_capabilities(model_identifier: str) -> dict[str, object]:
    """Return normalized capability flags for a LiteLLM model identifier.

    Resolution order:
    1. ``litellm.model_cost[model_identifier]`` — exact match.
    2. ``litellm.get_model_info(model_identifier)`` — provider lookup fallback.
    3. Partial result (unknown fields omitted) on KeyError or any error.

    The returned dict always contains only keys that could be resolved. Callers
    should treat absent keys as unknown, not False.

    Returned keys (all optional):
    - ``tools`` (bool): model supports function/tool calling.
    - ``parallel_tools`` (bool): model supports parallel tool calls.
    - ``structured_output`` (bool): model supports JSON schema output.
    - ``vision`` (bool): model accepts image input.
    - ``reasoning`` (bool): model has extended reasoning / thinking mode.
    - ``prompt_caching`` (bool): model supports prompt caching.
    - ``context_window`` (int): max input tokens.
    - ``max_output_tokens`` (int): max output tokens.
    """
    try:
        import litellm
    except ImportError:
        return {}

    info: dict[str, Any] | None = None

    # 1. Direct model_cost lookup
    model_cost: dict[str, Any] = getattr(litellm, "model_cost", {})
    raw = model_cost.get(model_identifier)
    if isinstance(raw, dict):
        info = raw
    else:
        # 2. get_model_info fallback
        try:
            result: Any = litellm.get_model_info(model_identifier)
            if isinstance(result, dict):
                info = dict(result)
        except Exception:
            pass

    if info is None:
        return {}

    return _extract_capabilities(info)


def _extract_capabilities(info: dict[str, Any]) -> dict[str, object]:
    """Extract normalized capability flags from a raw LiteLLM model-cost row."""
    result: dict[str, object] = {}

    _bool_flag(result, info, "tools", "supports_function_calling")
    _bool_flag(result, info, "parallel_tools", "supports_parallel_function_calling")
    _bool_flag(
        result,
        info,
        "structured_output",
        "supports_response_schema",
        fallback_key="supports_json_mode",
    )
    _bool_flag(result, info, "vision", "supports_vision")
    _bool_flag(result, info, "reasoning", "supports_reasoning")
    _bool_flag(result, info, "prompt_caching", "supports_prompt_caching")

    # Numeric fields
    for out_key, raw_key in (
        ("context_window", "max_input_tokens"),
        ("max_output_tokens", "max_output_tokens"),
    ):
        val = info.get(raw_key)
        if isinstance(val, int) and val > 0:
            result[out_key] = val

    return result


def _bool_flag(
    result: dict[str, object],
    info: dict[str, Any],
    out_key: str,
    primary_key: str,
    *,
    fallback_key: str | None = None,
) -> None:
    """Write a bool capability flag if the source key is present."""
    val = info.get(primary_key)
    if isinstance(val, bool):
        result[out_key] = val
        return
    if fallback_key is not None:
        fallback_val = info.get(fallback_key)
        if isinstance(fallback_val, bool):
            result[out_key] = fallback_val
