"""Tests for the capability normalization helper."""

from __future__ import annotations

from unittest.mock import patch


class TestNormalizeModelCapabilities:
    def test_returns_dict(self) -> None:
        from kortny.llm.capabilities import normalize_model_capabilities

        fake_model_cost = {
            "test-model": {
                "litellm_provider": "openai",
                "supports_function_calling": True,
                "supports_vision": False,
                "max_input_tokens": 128000,
                "max_output_tokens": 4096,
            }
        }
        with patch("litellm.model_cost", fake_model_cost):
            result = normalize_model_capabilities("test-model")

        assert isinstance(result, dict)

    def test_extracts_tools_flag(self) -> None:
        from kortny.llm.capabilities import normalize_model_capabilities

        fake_model_cost = {
            "test-model": {
                "supports_function_calling": True,
            }
        }
        with patch("litellm.model_cost", fake_model_cost):
            result = normalize_model_capabilities("test-model")

        assert result.get("tools") is True

    def test_extracts_vision_flag(self) -> None:
        from kortny.llm.capabilities import normalize_model_capabilities

        fake_model_cost = {
            "gpt-4o": {
                "supports_vision": True,
                "supports_function_calling": True,
                "max_input_tokens": 128000,
                "max_output_tokens": 16384,
            }
        }
        with patch("litellm.model_cost", fake_model_cost):
            result = normalize_model_capabilities("gpt-4o")

        assert result.get("vision") is True

    def test_extracts_context_window(self) -> None:
        from kortny.llm.capabilities import normalize_model_capabilities

        fake_model_cost = {
            "claude-test": {
                "max_input_tokens": 200000,
                "max_output_tokens": 8192,
            }
        }
        with patch("litellm.model_cost", fake_model_cost):
            result = normalize_model_capabilities("claude-test")

        assert result.get("context_window") == 200000
        assert result.get("max_output_tokens") == 8192

    def test_unknown_model_returns_empty(self) -> None:
        from kortny.llm.capabilities import normalize_model_capabilities

        with (
            patch("litellm.model_cost", {}),
            patch("litellm.get_model_info", side_effect=Exception("not found")),
        ):
            result = normalize_model_capabilities("unknown-model-xyz")

        assert result == {}

    def test_structured_output_fallback_to_json_mode(self) -> None:
        from kortny.llm.capabilities import normalize_model_capabilities

        fake_model_cost = {
            "test-model": {
                "supports_json_mode": True,
            }
        }
        with patch("litellm.model_cost", fake_model_cost):
            result = normalize_model_capabilities("test-model")

        assert result.get("structured_output") is True

    def test_missing_bool_fields_absent_not_false(self) -> None:
        """Fields that are missing from model_cost should be absent, not False."""
        from kortny.llm.capabilities import normalize_model_capabilities

        fake_model_cost = {
            "minimal-model": {
                "supports_function_calling": True,
            }
        }
        with patch("litellm.model_cost", fake_model_cost):
            result = normalize_model_capabilities("minimal-model")

        assert "tools" in result
        assert "vision" not in result
        assert "reasoning" not in result
