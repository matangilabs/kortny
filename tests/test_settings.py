from decimal import Decimal

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
    "LLM_HUMANIZER_MODEL",
    "AGENT_RUNTIME",
    "KORTNY_PLANNED_WORKFLOWS_ENABLED",
    "KORTNY_PLANNED_WORKFLOW_MAX_PARALLEL_BRANCHES",
    "KORTNY_PLANNED_WORKFLOW_COST_CEILING_USD",
    "KORTNY_PLANNED_WORKFLOW_MAX_BRANCH_MODEL_CALLS",
    "KORTNY_PLANNED_WORKFLOW_MAX_BRANCH_TOOL_CALLS",
    "KORTNY_PLANNED_WORKFLOW_MAX_TOTAL_TOOL_CALLS",
    "KORTNY_PLANNED_WORKFLOW_PROGRESS_UPDATES_ENABLED",
    "KORTNY_SANDBOX_RUNNER_URL",
    "KORTNY_SANDBOX_RUNNER_TIMEOUT_SECONDS",
    "KORTNY_SANDBOX_DEFAULT_IMAGE",
    "KORTNY_WORKFLOW_BACKEND",
    "TEMPORAL_ADDRESS",
    "TEMPORAL_NAMESPACE",
    "TEMPORAL_TASK_QUEUE",
    "KORTNY_SCHEDULER_POLL_INTERVAL_SECONDS",
    "KORTNY_SCHEDULER_MATERIALIZE_LIMIT",
    "KORTNY_SCHEDULER_ADVISORY_LOCK_KEY",
    "KORTNY_WITNESS_ENABLED",
    "KORTNY_WITNESS_DELIVER_PRIVATE",
    "KORTNY_WITNESS_POLL_INTERVAL_SECONDS",
    "KORTNY_WITNESS_PROFILE_SCAN_LIMIT",
    "KORTNY_WITNESS_DELIVERY_LIMIT",
    "KORTNY_WITNESS_SCAN_INTERVAL_SECONDS",
    "KORTNY_WITNESS_AUTOPILOT_ENABLED",
    "KORTNY_WITNESS_AUTOPILOT_LIMIT",
    "KORTNY_WITNESS_AUTOPILOT_MIN_CONFIDENCE",
    "COMPOSIO_API_KEY",
    "COMPOSIO_CATALOG_ENABLED",
    "COMPOSIO_CATALOG_LIMIT",
    "COMPOSIO_REQUEST_TIMEOUT_SECONDS",
    "TOOL_SELECTOR_MAX_PROMPT_CHARS",
    "BRAVE_SEARCH_API_KEY",
    "OBSERVABILITY_ENABLED",
    "OBSERVABILITY_CAPTURE_CONTENT",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_EXPORTER_OTLP_HEADERS",
    "OTEL_SERVICE_NAME",
    "OTEL_TRACE_SAMPLING_RATIO",
    "LANGFUSE_ENABLED",
    "LANGFUSE_HOST",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_PROMPTS_ENABLED",
    "LANGFUSE_PROMPT_LABEL",
    "KORTNY_RELEASE",
    "KORTNY_VERSION",
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
    monkeypatch.setenv("COMPOSIO_API_KEY", "composio-key")
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
    assert settings.agent_runtime == "custom"
    assert settings.planned_workflows_enabled is True
    assert settings.planned_workflow_max_parallel_branches == 3
    assert settings.planned_workflow_cost_ceiling_usd == 0.75
    assert settings.planned_workflow_max_branch_model_calls == 3
    assert settings.planned_workflow_max_branch_tool_calls == 8
    assert settings.planned_workflow_max_total_tool_calls == 12
    assert settings.sandbox_runner_url is None
    assert settings.sandbox_runner_timeout_seconds == 70.0
    assert (
        settings.sandbox_default_image
        == "ghcr.io/astral-sh/uv:python3.11-bookworm-slim"
    )
    assert settings.workflow_backend == "inline"
    assert settings.temporal_address == "temporal:7233"
    assert settings.temporal_namespace == "default"
    assert settings.temporal_task_queue == "kortny-workflows"
    assert settings.scheduler_poll_interval_seconds == 5.0
    assert settings.scheduler_materialize_limit == 50
    assert settings.scheduler_advisory_lock_key == 759340185
    assert settings.witness_enabled is True
    assert settings.witness_deliver_private is False
    assert settings.witness_poll_interval_seconds == 300.0
    assert settings.witness_profile_scan_limit == 10
    assert settings.witness_delivery_limit == 5
    assert settings.witness_scan_interval_seconds == 21600
    assert settings.witness_autopilot_enabled is True
    assert settings.witness_autopilot_limit == 1
    assert settings.witness_autopilot_min_confidence == Decimal("0.600")
    assert settings.composio_api_key == "composio-key"
    assert settings.composio_catalog_enabled is True
    assert settings.composio_catalog_limit == 60
    assert settings.composio_request_timeout_seconds == 10.0
    assert settings.tool_selector_max_external_candidates == 24
    assert settings.tool_selector_max_prompt_chars == 12000
    assert settings.tool_result_prompt_max_chars == 8000
    assert settings.observability_enabled is True
    assert settings.observability_capture_content == "summaries"
    assert settings.otel_service_name == "kortny"
    assert settings.langfuse_enabled is False


def test_settings_loads_optional_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_settings_env(monkeypatch)
    set_required_settings_env(monkeypatch)
    monkeypatch.setenv("COMPOSIO_API_KEY", "composio-key")
    monkeypatch.setenv("COMPOSIO_CATALOG_ENABLED", "false")
    monkeypatch.setenv("COMPOSIO_CATALOG_LIMIT", "120")
    monkeypatch.setenv("COMPOSIO_REQUEST_TIMEOUT_SECONDS", "2.5")
    monkeypatch.setenv("TOOL_SELECTOR_MAX_EXTERNAL_CANDIDATES", "12")
    monkeypatch.setenv("TOOL_SELECTOR_MAX_PROMPT_CHARS", "6000")
    monkeypatch.setenv("TOOL_RESULT_PROMPT_MAX_CHARS", "4000")
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "brave-key")
    monkeypatch.setenv("SLACK_FILE_READ_MAX_BYTES", "1024")
    monkeypatch.setenv("SLACK_APP_NAME", "Courtney")
    monkeypatch.setenv("LLM_CHEAP_MODEL", "anthropic/claude-haiku-test")
    monkeypatch.setenv("LLM_DOCUMENT_MODEL", "anthropic/claude-sonnet-test")
    monkeypatch.setenv("LLM_HUMANIZER_MODEL", "anthropic/claude-haiku-humanizer")
    monkeypatch.setenv("AGENT_RUNTIME", "adk")
    monkeypatch.setenv("KORTNY_PLANNED_WORKFLOWS_ENABLED", "false")
    monkeypatch.setenv("KORTNY_PLANNED_WORKFLOW_MAX_PARALLEL_BRANCHES", "4")
    monkeypatch.setenv("KORTNY_PLANNED_WORKFLOW_COST_CEILING_USD", "1.25")
    monkeypatch.setenv("KORTNY_PLANNED_WORKFLOW_MAX_BRANCH_MODEL_CALLS", "5")
    monkeypatch.setenv("KORTNY_PLANNED_WORKFLOW_MAX_BRANCH_TOOL_CALLS", "13")
    monkeypatch.setenv("KORTNY_PLANNED_WORKFLOW_MAX_TOTAL_TOOL_CALLS", "21")
    monkeypatch.setenv("KORTNY_PLANNED_WORKFLOW_PROGRESS_UPDATES_ENABLED", "false")
    monkeypatch.setenv("KORTNY_SANDBOX_RUNNER_URL", "http://sandbox-runner:8090/")
    monkeypatch.setenv("KORTNY_SANDBOX_RUNNER_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("KORTNY_SANDBOX_DEFAULT_IMAGE", "python:3.11-slim")
    monkeypatch.setenv("KORTNY_WORKFLOW_BACKEND", "temporal")
    monkeypatch.setenv("TEMPORAL_ADDRESS", "temporal.example:7233")
    monkeypatch.setenv("TEMPORAL_NAMESPACE", "kortny-dev")
    monkeypatch.setenv("TEMPORAL_TASK_QUEUE", "kortny-dev-workflows")
    monkeypatch.setenv("KORTNY_SCHEDULER_POLL_INTERVAL_SECONDS", "2.5")
    monkeypatch.setenv("KORTNY_SCHEDULER_MATERIALIZE_LIMIT", "25")
    monkeypatch.setenv("KORTNY_SCHEDULER_ADVISORY_LOCK_KEY", "123456")
    monkeypatch.setenv("KORTNY_WITNESS_ENABLED", "false")
    monkeypatch.setenv("KORTNY_WITNESS_DELIVER_PRIVATE", "true")
    monkeypatch.setenv("KORTNY_WITNESS_POLL_INTERVAL_SECONDS", "90")
    monkeypatch.setenv("KORTNY_WITNESS_PROFILE_SCAN_LIMIT", "7")
    monkeypatch.setenv("KORTNY_WITNESS_DELIVERY_LIMIT", "2")
    monkeypatch.setenv("KORTNY_WITNESS_SCAN_INTERVAL_SECONDS", "3600")
    monkeypatch.setenv("KORTNY_WITNESS_AUTOPILOT_ENABLED", "false")
    monkeypatch.setenv("KORTNY_WITNESS_AUTOPILOT_LIMIT", "3")
    monkeypatch.setenv("KORTNY_WITNESS_AUTOPILOT_MIN_CONFIDENCE", "0.725")
    monkeypatch.setenv("OBSERVABILITY_CAPTURE_CONTENT", "summaries")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel:4318")
    monkeypatch.setenv(
        "OTEL_EXPORTER_OTLP_HEADERS",
        "Authorization=Basic token,x-langfuse-ingestion-version=4",
    )
    monkeypatch.setenv("OTEL_TRACE_SAMPLING_RATIO", "0.25")
    monkeypatch.setenv("LANGFUSE_ENABLED", "true")
    monkeypatch.setenv("LANGFUSE_HOST", "http://langfuse:3000")
    monkeypatch.setenv("LANGFUSE_PROMPTS_ENABLED", "true")
    monkeypatch.setenv("LANGFUSE_PROMPT_LABEL", "staging")
    monkeypatch.setenv("KORTNY_RELEASE", "2026.05.24")

    settings = load_settings(env_file=None)

    assert settings.composio_api_key == "composio-key"
    assert settings.composio_catalog_enabled is False
    assert settings.composio_catalog_limit == 120
    assert settings.composio_request_timeout_seconds == 2.5
    assert settings.tool_selector_max_external_candidates == 12
    assert settings.tool_selector_max_prompt_chars == 6000
    assert settings.tool_result_prompt_max_chars == 4000
    assert settings.brave_search_api_key == "brave-key"
    assert settings.slack_file_read_max_bytes == 1024
    assert settings.slack_app_name == "Courtney"
    assert settings.llm_cheap_model == "anthropic/claude-haiku-test"
    assert settings.llm_document_model == "anthropic/claude-sonnet-test"
    assert settings.llm_humanizer_model == "anthropic/claude-haiku-humanizer"
    assert settings.agent_runtime == "adk"
    assert settings.planned_workflows_enabled is False
    assert settings.planned_workflow_max_parallel_branches == 4
    assert settings.planned_workflow_cost_ceiling_usd == 1.25
    assert settings.planned_workflow_max_branch_model_calls == 5
    assert settings.planned_workflow_max_branch_tool_calls == 13
    assert settings.planned_workflow_max_total_tool_calls == 21
    assert settings.planned_workflow_progress_updates_enabled is False
    assert settings.sandbox_runner_url == "http://sandbox-runner:8090"
    assert settings.sandbox_runner_timeout_seconds == 12.5
    assert settings.sandbox_default_image == "python:3.11-slim"
    assert settings.workflow_backend == "temporal"
    assert settings.temporal_address == "temporal.example:7233"
    assert settings.temporal_namespace == "kortny-dev"
    assert settings.temporal_task_queue == "kortny-dev-workflows"
    assert settings.scheduler_poll_interval_seconds == 2.5
    assert settings.scheduler_materialize_limit == 25
    assert settings.scheduler_advisory_lock_key == 123456
    assert settings.witness_enabled is False
    assert settings.witness_deliver_private is True
    assert settings.witness_poll_interval_seconds == 90
    assert settings.witness_profile_scan_limit == 7
    assert settings.witness_delivery_limit == 2
    assert settings.witness_scan_interval_seconds == 3600
    assert settings.witness_autopilot_enabled is False
    assert settings.witness_autopilot_limit == 3
    assert settings.witness_autopilot_min_confidence == Decimal("0.725")
    assert settings.observability_capture_content == "summaries"
    assert settings.otel_exporter_otlp_endpoint == "http://otel:4318"
    assert (
        settings.otel_exporter_otlp_headers
        == "Authorization=Basic token,x-langfuse-ingestion-version=4"
    )
    assert settings.otel_trace_sampling_ratio == 0.25
    assert settings.langfuse_enabled is True
    assert settings.langfuse_host == "http://langfuse:3000"
    assert settings.langfuse_prompts_enabled is True
    assert settings.langfuse_prompt_label == "staging"
    assert settings.kortny_release == "2026.05.24"


def test_blank_optional_environment_values_are_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_settings_env(monkeypatch)
    set_required_settings_env(monkeypatch)
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "")
    monkeypatch.setenv("LLM_CHEAP_MODEL", "")
    monkeypatch.setenv("LLM_HUMANIZER_MODEL", "")
    monkeypatch.setenv("LANGFUSE_HOST", "")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")
    monkeypatch.setenv("LANGFUSE_PROMPT_LABEL", "")
    monkeypatch.setenv("KORTNY_SANDBOX_RUNNER_URL", "")

    settings = load_settings(env_file=None)

    assert settings.brave_search_api_key is None
    assert settings.llm_cheap_model is None
    assert settings.llm_humanizer_model is None
    assert settings.langfuse_host is None
    assert settings.langfuse_public_key is None
    assert settings.langfuse_secret_key is None
    assert settings.langfuse_prompt_label is None
    assert settings.sandbox_runner_url is None


def test_load_settings_reports_missing_required_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_settings_env(monkeypatch)

    with pytest.raises(SettingsError) as exc_info:
        load_settings(env_file=None)

    message = str(exc_info.value)
    assert "SLACK_BOT_TOKEN" in message
    assert "SLACK_APP_TOKEN" in message
    assert "COMPOSIO_API_KEY" in message
    assert "POSTGRES_URL" in message


def test_load_settings_rejects_blank_required_composio_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_settings_env(monkeypatch)
    set_required_settings_env(monkeypatch)
    monkeypatch.setenv("COMPOSIO_API_KEY", "")

    with pytest.raises(SettingsError) as exc_info:
        load_settings(env_file=None)

    assert "COMPOSIO_API_KEY" in str(exc_info.value)


def test_settings_rejects_unknown_llm_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_settings_env(monkeypatch)
    set_required_settings_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "unknown-provider")

    with pytest.raises(SettingsError) as exc_info:
        load_settings(env_file=None)

    assert "LLM_PROVIDER" in str(exc_info.value)
