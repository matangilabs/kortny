"""Provider-agnostic model identifier normalization for pricing lookups."""

from __future__ import annotations

# Providers whose model identifiers may be prefixed as "<provider>/<model>".
_PREFIXED_PROVIDERS = frozenset(
    {
        "anthropic",
        "azure",
        "bedrock",
        "bedrock_converse",
        "fireworks_ai",
        "gemini",
        "ollama",
        "openai",
        "xai",
    }
)


def normalize_model_id(model: str) -> str:
    """Strip all provider prefixes to get the bare model name.

    E.g.:
      "openrouter/openai/gpt-4o-mini" -> "gpt-4o-mini"
      "anthropic/claude-3-5-haiku-20241022" -> "claude-3-5-haiku-20241022"
      "gpt-4o-mini" -> "gpt-4o-mini"
    """
    stripped = model.strip()
    # Strip leading provider prefixes iteratively until no more slashes remain
    # or the segment before the slash is not a known provider prefix.
    while "/" in stripped:
        prefix, rest = stripped.split("/", 1)
        # Remove if it looks like a provider prefix (known set or "openrouter").
        if prefix in _PREFIXED_PROVIDERS or prefix == "openrouter":
            stripped = rest
        else:
            break
    return stripped


def pricing_lookup_candidates(
    model: str,
    *,
    provider_kind: str | None = None,
) -> tuple[str, ...]:
    """Return lookup variants to try against litellm.model_cost, most-specific first.

    Covers:
    - openrouter/google/gemini-2.5-flash-lite  <-> google/gemini-2.5-flash-lite
    - anthropic/claude-...  <-> claude-...
    - azure/<deployment>
    - bedrock/... and bedrock_converse/...
    - openai/<custom>
    - raw names

    Order: original, then prefix-stripped variant, then bare name.
    Deduplicates — no repeated candidates.
    """
    normalized = model.strip()
    candidates: list[str] = [normalized]

    if provider_kind == "openrouter":
        # OpenRouter identifiers are either "openrouter/sub/model" or "sub/model".
        if normalized.startswith("openrouter/"):
            # Strip the openrouter/ prefix to get "sub/model"
            without_prefix = normalized.removeprefix("openrouter/")
            _append_unique(candidates, without_prefix)
        else:
            # Add the "openrouter/" prefixed form
            _append_unique(candidates, f"openrouter/{normalized}")

        # Always add the bare name (strip everything up to and including last
        # provider segment). For "google/gemini-2.5-flash-lite" or
        # "openrouter/google/gemini-2.5-flash-lite", bare is "gemini-2.5-flash-lite".
        bare = normalize_model_id(normalized)
        _append_unique(candidates, bare)

    elif provider_kind in _PREFIXED_PROVIDERS:
        prefix = f"{provider_kind}/"
        if normalized.startswith(prefix):
            # Has prefix — add without-prefix variant
            _append_unique(candidates, normalized.removeprefix(prefix))
        elif "/" not in normalized:
            # Bare name — add prefixed variant
            _append_unique(candidates, f"{prefix}{normalized}")
        else:
            # Has a slash but not this provider's prefix (e.g. "bedrock_converse/..."
            # when provider_kind="bedrock") — still try stripping the first segment.
            _without = normalized.split("/", 1)[1]
            _append_unique(candidates, _without)

    return tuple(candidates)


def _append_unique(lst: list[str], value: str) -> None:
    if value not in lst:
        lst.append(value)
