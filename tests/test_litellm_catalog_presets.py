"""Tests for new provider presets in litellm_catalog."""

from __future__ import annotations

from kortny.llm.litellm_catalog import (
    LITELLM_PROVIDER_OPTIONS,
    litellm_provider_option,
)


class TestNewPresets:
    def test_groq_preset_present(self) -> None:
        option = litellm_provider_option("groq")
        assert option is not None
        assert option.label == "Groq"
        assert option.auth_type == "api_key"

    def test_together_ai_preset_present(self) -> None:
        option = litellm_provider_option("together_ai")
        assert option is not None
        assert option.label == "Together AI"
        assert option.auth_type == "api_key"

    def test_deepseek_preset_present(self) -> None:
        option = litellm_provider_option("deepseek")
        assert option is not None
        assert option.label == "DeepSeek"
        assert option.auth_type == "api_key"

    def test_mistral_preset_present(self) -> None:
        option = litellm_provider_option("mistral")
        assert option is not None
        assert option.label == "Mistral AI"
        assert option.auth_type == "api_key"

    def test_vertex_ai_preset_present(self) -> None:
        option = litellm_provider_option("vertex_ai")
        assert option is not None
        assert option.auth_type == "instance_role"

    def test_bedrock_converse_preset_present(self) -> None:
        option = litellm_provider_option("bedrock_converse")
        assert option is not None
        assert option.auth_type == "instance_role"

    def test_openai_compatible_preset(self) -> None:
        option = litellm_provider_option("openai_compatible")
        assert option is not None
        assert option.auth_type == "api_key_optional"
        assert option.needs_base_url is True

    def test_ollama_auth_type_no_key(self) -> None:
        option = litellm_provider_option("ollama")
        assert option is not None
        assert option.auth_type == "no_key"

    def test_bedrock_auth_type_instance_role(self) -> None:
        option = litellm_provider_option("bedrock")
        assert option is not None
        assert option.auth_type == "instance_role"

    def test_all_options_have_auth_type(self) -> None:
        valid_auth_types = {"api_key", "api_key_optional", "instance_role", "no_key"}
        for option in LITELLM_PROVIDER_OPTIONS:
            assert option.auth_type in valid_auth_types, (
                f"Provider '{option.kind}' has invalid auth_type '{option.auth_type}'"
            )
