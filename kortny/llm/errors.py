"""Provider-agnostic classification of LLM provider failures.

When the configured language-model provider rejects a request for billing,
authentication, or rate-limit reasons, the raw exception is an opaque
``litellm`` error whose message leaks account URLs and key fragments. Those
failures are not bugs and not transient in the usual sense — an operator must
act (top up credits, fix the key, wait out a limit). This module turns such an
exception into a clear, secret-free, admin-actionable message that any
self-hosted deployment can show in Slack, regardless of which provider
(OpenAI, Anthropic, OpenRouter, Azure, …) is configured.

The classifier never imports ``litellm`` so it stays decoupled and unit
testable: it inspects the exception's ``status_code`` (LiteLLM sets this on its
mapped exceptions) plus the lowercased class name and message text. Substring
scanning is deliberately broad because providers word these errors differently.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ProviderFailureKind(StrEnum):
    """Categories of non-bug LLM provider failures an operator must resolve."""

    billing = "billing"
    auth = "auth"
    rate_limit = "rate_limit"
    context_window = "context_window"


@dataclass(frozen=True, slots=True)
class ProviderFailure:
    """A classified provider failure plus the message to show the user."""

    kind: ProviderFailureKind
    message: str


# Admin-actionable, secret-free copy. Provider-neutral wording ("the language
# model provider") because a self-hosted deployment may run any provider.
_BILLING_MESSAGE = (
    "I couldn't finish that: the language-model provider rejected the request "
    "for billing reasons (out of credits, or over the configured spending "
    "limit). An admin needs to top up the provider account or raise the API "
    "key's limit, then you can retry."
)
_AUTH_MESSAGE = (
    "I couldn't reach the language model: the provider rejected the API key "
    "(authentication failed). An admin needs to check the LLM provider "
    "credentials in the Kortny configuration."
)
_RATE_LIMIT_MESSAGE = (
    "The language-model provider is rate-limiting requests right now. Give it a "
    "moment and try again — if it keeps happening, an admin may need to raise "
    "the account's rate limits."
)
_CONTEXT_WINDOW_MESSAGE = (
    "This request was too large for the configured model's context window. Try "
    "a shorter request, or an admin can switch to a model with a larger context."
)

_MESSAGES: dict[ProviderFailureKind, str] = {
    ProviderFailureKind.billing: _BILLING_MESSAGE,
    ProviderFailureKind.auth: _AUTH_MESSAGE,
    ProviderFailureKind.rate_limit: _RATE_LIMIT_MESSAGE,
    ProviderFailureKind.context_window: _CONTEXT_WINDOW_MESSAGE,
}

# Substrings that signal a billing/credit/quota rejection across providers.
# OpenRouter: "requires more credits"; OpenAI: "insufficient_quota" /
# "billing"; Anthropic: "credit balance is too low"; generic: "payment".
_BILLING_MARKERS: tuple[str, ...] = (
    "credit",
    "insufficient_quota",
    "insufficient quota",
    "billing",
    "payment required",
    "quota",
    "exceeded your current quota",
    "spending limit",
    "budget",
)
_AUTH_MARKERS: tuple[str, ...] = (
    "api key",
    "api-key",
    "authentication",
    "unauthorized",
    "invalid_api_key",
    "no auth credentials",
    "incorrect api key",
)
_CONTEXT_WINDOW_MARKERS: tuple[str, ...] = (
    "context window",
    "context_length_exceeded",
    "maximum context length",
    "too many tokens",
    "reduce the length",
)


def _status_code(exc: BaseException) -> int | None:
    code = getattr(exc, "status_code", None)
    if isinstance(code, bool):  # bool is an int subclass; never a status code
        return None
    if isinstance(code, int):
        return code
    if isinstance(code, str) and code.isdigit():
        return int(code)
    return None


def classify_provider_failure(exc: BaseException) -> ProviderFailure | None:
    """Classify ``exc`` as a known provider failure, or return ``None``.

    ``None`` means "not a recognized provider config/billing/limit failure" —
    the caller should fall back to its generic error handling, because the
    failure is more likely an actual bug.
    """

    status = _status_code(exc)
    haystack = f"{type(exc).__name__} {exc}".lower()

    # Billing first: a 402, or any provider's credit/quota wording. Quota
    # rejections often arrive as 429s, so check markers before the rate-limit
    # status branch below.
    if status == 402 or any(marker in haystack for marker in _BILLING_MARKERS):
        return ProviderFailure(ProviderFailureKind.billing, _BILLING_MESSAGE)

    if status in (401, 403) or any(marker in haystack for marker in _AUTH_MARKERS):
        return ProviderFailure(ProviderFailureKind.auth, _AUTH_MESSAGE)

    if any(marker in haystack for marker in _CONTEXT_WINDOW_MARKERS):
        return ProviderFailure(
            ProviderFailureKind.context_window, _CONTEXT_WINDOW_MESSAGE
        )

    if status == 429 or "rate limit" in haystack or "ratelimit" in haystack:
        return ProviderFailure(ProviderFailureKind.rate_limit, _RATE_LIMIT_MESSAGE)

    return None
