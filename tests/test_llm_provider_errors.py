"""Tests for provider-agnostic LLM failure classification."""

from __future__ import annotations

import pytest

from kortny.llm.errors import (
    ProviderFailureKind,
    classify_provider_failure,
)


class _FakeProviderError(Exception):
    """Stand-in for a litellm exception carrying a status_code."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        if status_code is not None:
            self.status_code = status_code


# Real OpenRouter 402 wording observed in production logs.
_OPENROUTER_402 = (
    "litellm.APIError: APIError: OpenrouterException - "
    '{"error":{"message":"This request requires more credits, or fewer '
    'max_tokens. You requested up to 16384 tokens, but can only afford 7040.",'
    '"code":402}}'
)


def test_openrouter_credit_exhaustion_classifies_as_billing() -> None:
    failure = classify_provider_failure(_FakeProviderError(_OPENROUTER_402, 402))
    assert failure is not None
    assert failure.kind is ProviderFailureKind.billing
    assert "billing" in failure.message.lower()
    # Never leak the account URL or raw key fragment.
    assert "openrouter.ai" not in failure.message
    assert "http" not in failure.message.lower()


def test_billing_detected_from_message_without_status_code() -> None:
    failure = classify_provider_failure(
        _FakeProviderError("You exceeded your current quota, please check billing")
    )
    assert failure is not None
    assert failure.kind is ProviderFailureKind.billing


def test_openai_insufficient_quota_429_is_billing_not_rate_limit() -> None:
    # OpenAI surfaces credit exhaustion as a 429 with insufficient_quota; it is
    # a billing problem, not a transient rate limit.
    failure = classify_provider_failure(
        _FakeProviderError("Rate limit reached: insufficient_quota", 429)
    )
    assert failure is not None
    assert failure.kind is ProviderFailureKind.billing


def test_auth_failure_classifies_as_auth() -> None:
    failure = classify_provider_failure(
        _FakeProviderError("Incorrect API key provided", 401)
    )
    assert failure is not None
    assert failure.kind is ProviderFailureKind.auth
    assert "credentials" in failure.message.lower()


def test_plain_rate_limit_classifies_as_rate_limit() -> None:
    failure = classify_provider_failure(
        _FakeProviderError("Rate limit exceeded, slow down", 429)
    )
    assert failure is not None
    assert failure.kind is ProviderFailureKind.rate_limit


def test_context_window_classifies() -> None:
    failure = classify_provider_failure(
        _FakeProviderError("This model's maximum context length is 8192 tokens")
    )
    assert failure is not None
    assert failure.kind is ProviderFailureKind.context_window


def test_unrelated_error_is_not_classified() -> None:
    # A real bug should fall through to generic handling, not be mislabeled.
    assert classify_provider_failure(KeyError("missing field")) is None
    assert classify_provider_failure(ValueError("bad value")) is None


def test_bool_status_code_is_ignored() -> None:
    exc = _FakeProviderError("something odd")
    exc.status_code = True  # bool is an int subclass; must not be read as 1
    assert classify_provider_failure(exc) is None


@pytest.mark.parametrize("status", [402, 401, 403, 429])
def test_known_status_codes_always_classify(status: int) -> None:
    failure = classify_provider_failure(_FakeProviderError("opaque", status))
    assert failure is not None
