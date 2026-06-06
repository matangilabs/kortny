from types import SimpleNamespace
from typing import Any, cast

from kortny.config import Settings
from kortny.db.models import Task
from kortny.workflow.handoff import TaskRuntimeClass, evaluate_runtime_handoff


def test_handoff_classifies_short_availability_check_as_quick_response() -> None:
    decision = evaluate_runtime_handoff(
        settings=_settings(),
        task=_task("Are you up?"),
    )

    assert decision.runtime_class is TaskRuntimeClass.quick_response
    assert decision.durable_candidate is False
    assert decision.recommended_backend == "inline"
    assert decision.selected_backend == "inline"
    assert decision.fallback_reason is None


def test_handoff_classifies_multi_integration_research_as_durable_candidate() -> None:
    decision = evaluate_runtime_handoff(
        settings=_settings(),
        task=_task(
            "Research AI observability tools, check Linear for existing HIGs, "
            "compare with our docs, and recommend the next two actions."
        ),
    )

    assert decision.runtime_class is TaskRuntimeClass.durable_workflow_task
    assert decision.durable_candidate is True
    assert decision.recommended_backend == "temporal"
    assert decision.selected_backend == "inline"
    assert "multi_source_synthesis" in decision.reason_codes
    assert "integration_tool_work" in decision.reason_codes
    assert decision.fallback_reason == "workflow_backend_inline"


def test_handoff_records_temporal_configured_but_primary_execution_not_enabled() -> (
    None
):
    decision = evaluate_runtime_handoff(
        settings=_settings(KORTNY_WORKFLOW_BACKEND="temporal"),
        task=_task("Use Firecrawl to crawl the whole website and summarize it."),
    )

    assert decision.runtime_class is TaskRuntimeClass.durable_workflow_task
    assert decision.recommended_backend == "temporal"
    assert decision.configured_backend == "temporal"
    assert decision.selected_backend == "inline"
    assert decision.fallback_reason == "temporal_primary_execution_not_enabled"


def test_handoff_classifies_scheduled_identity_as_scheduled_workflow() -> None:
    decision = evaluate_runtime_handoff(
        settings=_settings(),
        task=_task("Summarize this channel.", identity_kind="scheduled"),
    )

    assert decision.runtime_class is TaskRuntimeClass.scheduled_workflow_task
    assert decision.durable_candidate is True
    assert decision.recommended_backend == "temporal"
    assert "scheduled_task_identity" in decision.reason_codes


def _task(input_text: str, *, identity_kind: str | None = None) -> Task:
    return cast(Any, SimpleNamespace(input=input_text, identity_kind=identity_kind))


def _settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "SLACK_BOT_TOKEN": "xoxb-test",
        "SLACK_APP_TOKEN": "xapp-test",
        "SLACK_SIGNING_SECRET": "secret",
        "LLM_PROVIDER": "openrouter",
        "LLM_API_KEY": "llm-key",
        "LLM_MODEL": "openai/gpt-4o",
        "COMPOSIO_API_KEY": "composio-key",
        "POSTGRES_URL": "postgresql://kortny:kortny@localhost/kortny",
        "KORTNY_WORKFLOW_BACKEND": "inline",
    }
    values.update(overrides)
    return Settings(**values)
