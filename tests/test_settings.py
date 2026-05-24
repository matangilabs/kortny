import pytest

from kortny.config import LLMProvider, SettingsError, load_settings

SETTINGS_ENV_VARS = {
    "SLACK_BOT_TOKEN",
    "SLACK_APP_TOKEN",
    "SLACK_SIGNING_SECRET",
    "SLACK_APP_NAME",
    "LLM_PROVIDER",
    "LLM_API_KEY",
    "LLM_MODEL",
    "LLM_CHEAP_MODEL",
    "LLM_STANDARD_MODEL",
    "LLM_ANALYSIS_MODEL",
    "LLM_DOCUMENT_MODEL",
    "LLM_HIGH_REASONING_MODEL",
    "COMPOSIO_API_KEY",
    "BRAVE_SEARCH_API_KEY",
    "POSTGRES_URL",
}


def clear_settings_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in SETTINGS_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def set_required_settings_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    monkeypatch.setenv("SLACK_APP_TOKEN", "xapp-test")
    monkeypatch.setenv("SLACK_SIGNING_SECRET", "signing-secret")
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("LLM_API_KEY", "llm-key")
    monkeypatch.setenv("LLM_MODEL", "openai/gpt-4o")
    monkeypatch.setenv("POSTGRES_URL", "postgresql://kortny:kortny@localhost/kortny")


def test_settings_loads_required_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_settings_env(monkeypatch)
    set_required_settings_env(monkeypatch)

    settings = load_settings(env_file=None)

    assert settings.slack_bot_token == "xoxb-test"
    assert settings.slack_app_token == "xapp-test"
    assert settings.llm_provider is LLMProvider.openrouter
    assert settings.postgres_url == "postgresql://kortny:kortny@localhost/kortny"
    assert settings.slack_file_read_max_bytes == 25 * 1024 * 1024
    assert settings.slack_app_name == "kortny"


def test_settings_loads_optional_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_settings_env(monkeypatch)
    set_required_settings_env(monkeypatch)
    monkeypatch.setenv("COMPOSIO_API_KEY", "composio-key")
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "brave-key")
    monkeypatch.setenv("SLACK_FILE_READ_MAX_BYTES", "1024")
    monkeypatch.setenv("SLACK_APP_NAME", "Courtney")
    monkeypatch.setenv("LLM_CHEAP_MODEL", "anthropic/claude-haiku-test")
    monkeypatch.setenv("LLM_DOCUMENT_MODEL", "anthropic/claude-sonnet-test")

    settings = load_settings(env_file=None)

    assert settings.composio_api_key == "composio-key"
    assert settings.brave_search_api_key == "brave-key"
    assert settings.slack_file_read_max_bytes == 1024
    assert settings.slack_app_name == "Courtney"
    assert settings.llm_cheap_model == "anthropic/claude-haiku-test"
    assert settings.llm_document_model == "anthropic/claude-sonnet-test"


def test_blank_optional_environment_values_are_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_settings_env(monkeypatch)
    set_required_settings_env(monkeypatch)
    monkeypatch.setenv("COMPOSIO_API_KEY", "")
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "")
    monkeypatch.setenv("LLM_CHEAP_MODEL", "")

    settings = load_settings(env_file=None)

    assert settings.composio_api_key is None
    assert settings.brave_search_api_key is None
    assert settings.llm_cheap_model is None


def test_load_settings_reports_missing_required_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_settings_env(monkeypatch)

    with pytest.raises(SettingsError) as exc_info:
        load_settings(env_file=None)

    message = str(exc_info.value)
    assert "SLACK_BOT_TOKEN" in message
    assert "SLACK_APP_TOKEN" in message
    assert "POSTGRES_URL" in message


def test_settings_rejects_unknown_llm_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_settings_env(monkeypatch)
    set_required_settings_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "unknown-provider")

    with pytest.raises(SettingsError) as exc_info:
        load_settings(env_file=None)

    assert "LLM_PROVIDER" in str(exc_info.value)
