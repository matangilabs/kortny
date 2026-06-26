"""Tests for model identifier normalization."""

from __future__ import annotations

from kortny.llm.model_ids import normalize_model_id, pricing_lookup_candidates


def test_openrouter_with_prefix() -> None:
    result = pricing_lookup_candidates(
        "openrouter/google/gemini-2.5-flash-lite", provider_kind="openrouter"
    )
    assert result[0] == "openrouter/google/gemini-2.5-flash-lite"
    assert "google/gemini-2.5-flash-lite" in result


def test_openrouter_without_prefix() -> None:
    result = pricing_lookup_candidates(
        "google/gemini-2.5-flash-lite", provider_kind="openrouter"
    )
    assert result[0] == "google/gemini-2.5-flash-lite"
    assert "openrouter/google/gemini-2.5-flash-lite" in result


def test_anthropic_with_prefix() -> None:
    result = pricing_lookup_candidates(
        "anthropic/claude-3-5-haiku-20241022", provider_kind="anthropic"
    )
    assert result[0] == "anthropic/claude-3-5-haiku-20241022"
    assert "claude-3-5-haiku-20241022" in result


def test_anthropic_without_prefix() -> None:
    result = pricing_lookup_candidates(
        "claude-3-5-haiku-20241022", provider_kind="anthropic"
    )
    assert result[0] == "claude-3-5-haiku-20241022"
    assert "anthropic/claude-3-5-haiku-20241022" in result


def test_azure_deployment() -> None:
    result = pricing_lookup_candidates(
        "azure/my-gpt4-deployment", provider_kind="azure"
    )
    assert "azure/my-gpt4-deployment" in result
    assert "my-gpt4-deployment" in result


def test_bedrock_model() -> None:
    result = pricing_lookup_candidates(
        "bedrock/anthropic.claude-3-5-haiku-20241022-v1:0", provider_kind="bedrock"
    )
    assert "bedrock/anthropic.claude-3-5-haiku-20241022-v1:0" in result
    assert "anthropic.claude-3-5-haiku-20241022-v1:0" in result


def test_bedrock_converse_model() -> None:
    result = pricing_lookup_candidates(
        "bedrock_converse/anthropic.claude-3-5-haiku-20241022-v1:0",
        provider_kind="bedrock",
    )
    assert "bedrock_converse/anthropic.claude-3-5-haiku-20241022-v1:0" in result
    assert "anthropic.claude-3-5-haiku-20241022-v1:0" in result


def test_openai_bare_name() -> None:
    result = pricing_lookup_candidates("gpt-4o-mini", provider_kind="openai")
    assert "gpt-4o-mini" in result


def test_openai_with_prefix() -> None:
    result = pricing_lookup_candidates("openai/gpt-4o-custom", provider_kind="openai")
    assert "openai/gpt-4o-custom" in result
    assert "gpt-4o-custom" in result


def test_no_provider_kind() -> None:
    result = pricing_lookup_candidates("gpt-4o-mini", provider_kind=None)
    assert "gpt-4o-mini" in result


def test_no_duplicates() -> None:
    # When model already is "openrouter/x", adding openrouter/ prefix again would be a dup
    result = pricing_lookup_candidates(
        "openrouter/openai/gpt-4o-mini", provider_kind="openrouter"
    )
    assert len(result) == len(set(result)), "No duplicates allowed"


def test_normalize_model_id_strips_prefix() -> None:
    assert normalize_model_id("openrouter/openai/gpt-4o-mini") == "gpt-4o-mini"
    assert (
        normalize_model_id("anthropic/claude-3-5-haiku-20241022")
        == "claude-3-5-haiku-20241022"
    )
    assert normalize_model_id("gpt-4o-mini") == "gpt-4o-mini"


def test_openrouter_bare_name_round_trip() -> None:
    # "gemini-2.5-flash-lite" with openrouter should produce both prefixed variants
    result = pricing_lookup_candidates(
        "gemini-2.5-flash-lite", provider_kind="openrouter"
    )
    assert result[0] == "gemini-2.5-flash-lite"
    assert "openrouter/gemini-2.5-flash-lite" in result
    # bare name is the same as original for a bare name
    assert len(result) == len(set(result))
