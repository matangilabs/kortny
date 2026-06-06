import asyncio
from types import SimpleNamespace
from typing import Any, cast

from temporalio.common import WorkflowIDConflictPolicy, WorkflowIDReusePolicy

from kortny.config import Settings
from kortny.db.models import Task
from kortny.workflow.launcher import (
    build_temporal_workflow_input,
    start_temporal_task_workflow,
    temporal_workflow_id,
)
from kortny.workflow.temporal import KortnyTaskWorkflow


def test_build_temporal_workflow_input_from_task() -> None:
    task = _task(input_text="Research this and summarize it.")

    workflow_input = build_temporal_workflow_input(task)

    assert temporal_workflow_id(task.id) == f"kortny-task-{task.id}"
    assert workflow_input.to_payload() == {
        "task_id": str(task.id),
        "installation_id": str(task.installation_id),
        "slack_channel_id": "C123",
        "slack_thread_ts": "123.456",
        "slack_user_id": "U123",
        "input": "Research this and summarize it.",
    }


def test_start_temporal_task_workflow_uses_stable_id_and_safe_duplicate_policy() -> (
    None
):
    task = _task(input_text="Compare Linear, Notion, and docs.")
    settings = _settings()
    client = FakeTemporalClient()

    launch = asyncio.run(
        start_temporal_task_workflow(settings=settings, task=task, client=client)
    )

    assert launch.workflow_id == f"kortny-task-{task.id}"
    assert launch.run_id == "run-1"
    assert launch.namespace == "default"
    assert launch.task_queue == "kortny-workflows"
    assert client.calls == [
        {
            "workflow": KortnyTaskWorkflow.run,
            "arg": {
                "task_id": str(task.id),
                "installation_id": str(task.installation_id),
                "slack_channel_id": "C123",
                "slack_thread_ts": "123.456",
                "slack_user_id": "U123",
                "input": "Compare Linear, Notion, and docs.",
            },
            "id": f"kortny-task-{task.id}",
            "task_queue": "kortny-workflows",
            "id_reuse_policy": WorkflowIDReusePolicy.ALLOW_DUPLICATE_FAILED_ONLY,
            "id_conflict_policy": WorkflowIDConflictPolicy.USE_EXISTING,
            "static_summary": f"Kortny task {task.id}",
        }
    ]


class FakeTemporalClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def start_workflow(self, workflow: Any, arg: Any, **kwargs: Any) -> Any:
        self.calls.append({"workflow": workflow, "arg": arg, **kwargs})
        return SimpleNamespace(
            id=kwargs["id"],
            run_id="run-1",
            first_execution_run_id="first-run-1",
            result_run_id=None,
        )


def _task(*, input_text: str) -> Task:
    return cast(
        Task,
        SimpleNamespace(
            id="42c9f71e-5ae7-4236-a8f8-6459b1f204e0",
            installation_id="6ff3b1aa-4a1a-43cf-835a-eaa354670492",
            slack_channel_id="C123",
            slack_thread_ts="123.456",
            slack_user_id="U123",
            input=input_text,
        ),
    )


def _settings() -> Settings:
    return Settings(
        SLACK_BOT_TOKEN="xoxb-test",
        SLACK_APP_TOKEN="xapp-test",
        SLACK_SIGNING_SECRET="secret",
        LLM_PROVIDER="openrouter",
        LLM_API_KEY="llm-key",
        LLM_MODEL="openai/gpt-4o",
        COMPOSIO_API_KEY="composio-key",
        POSTGRES_URL="postgresql://kortny:kortny@localhost/kortny",
        KORTNY_WORKFLOW_BACKEND="temporal",
    )
