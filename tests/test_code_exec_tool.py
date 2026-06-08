from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

import pytest

from kortny.db.models import TaskEventType
from kortny.execution import (
    SANDBOX_RESULT_MESSAGE,
    SandboxLifecycleEvent,
    SandboxResult,
    SandboxSpec,
    SandboxUnavailableError,
)
from kortny.tools.code_exec import CodeExecTool


def test_code_exec_runs_python_through_sandbox_runner_and_records_events() -> None:
    runner = FakeSandboxRunner(
        SandboxResult(
            exit_code=0,
            stdout="3\n",
            usage={"duration_ms": 12},
            events=(
                SandboxLifecycleEvent(
                    phase="started",
                    message="sandbox started",
                    details={"container_id": "sandbox-123"},
                ),
            ),
        )
    )
    task = FakeTask(id=uuid.uuid4())
    task_service = RecordingTaskService()
    tool = CodeExecTool(
        runner=runner,
        image="kortny/sandbox-python:latest",
        task=task,
        task_service=task_service,
    )

    result = tool.invoke({"code": "print(1 + 2)", "timeout_seconds": 7})

    assert result.output["successful"] is True
    assert result.output["stdout"] == "3\n"
    assert result.output["stderr"] == ""
    assert result.output["exit_code"] == 0
    assert runner.specs[0].image == "kortny/sandbox-python:latest"
    assert runner.specs[0].command == ("python", "-c", "print(1 + 2)")
    assert runner.specs[0].network == "none"
    assert runner.specs[0].resource_limits.timeout_seconds == 7
    assert runner.specs[0].resource_limits.pids_limit == 64
    assert str(runner.specs[0].workspace_path).startswith("/workspace/code-exec-")
    assert [event_type for event_type, _payload in task_service.events] == [
        TaskEventType.log,
        TaskEventType.log,
    ]
    assert task_service.events[0][1]["message"] == "sandbox_lifecycle"
    assert task_service.events[1][1]["message"] == SANDBOX_RESULT_MESSAGE
    assert task_service.events[1][1]["tool"] == "code_exec"
    assert task_service.events[1][1]["stdout_preview"] == "3\n"


def test_code_exec_returns_recoverable_unavailable_result_without_runner() -> None:
    tool = CodeExecTool(
        runner=None,
        image="kortny/sandbox-python:latest",
    )

    result = tool.invoke({"code": "print('hello')"})

    assert result.output["successful"] is False
    assert result.output["error"]["code"] == "sandbox_service_unavailable"
    assert result.output["error"]["recoverable"] is True
    assert result.output["error"]["details"] == {"execution_attempted": False}


def test_code_exec_maps_runner_unavailable_to_recoverable_result() -> None:
    runner = FakeSandboxRunner(
        SandboxUnavailableError("Sandbox runner request failed: ConnectError")
    )
    tool = CodeExecTool(
        runner=runner,
        image="kortny/sandbox-python:latest",
    )

    result = tool.invoke({"code": "print('hello')"})

    assert result.output["successful"] is False
    assert result.output["error"]["code"] == "sandbox_service_unavailable"
    assert "ConnectError" in result.output["error"]["message"]


def test_code_exec_returns_recoverable_failed_exit() -> None:
    runner = FakeSandboxRunner(SandboxResult(exit_code=2, stdout="", stderr="boom\n"))
    tool = CodeExecTool(
        runner=runner,
        image="kortny/sandbox-python:latest",
    )

    result = tool.invoke({"code": "raise SystemExit(2)"})

    assert result.output["successful"] is False
    assert result.output["exit_code"] == 2
    assert result.output["stderr"] == "boom\n"
    assert result.output["error"]["code"] == "sandbox_execution_failed"
    assert result.output["error"]["recoverable"] is True


def test_code_exec_validates_language_timeout_and_code_size() -> None:
    tool = CodeExecTool(
        runner=FakeSandboxRunner(SandboxResult(exit_code=0)),
        image="kortny/sandbox-python:latest",
        max_code_chars=5,
    )

    with pytest.raises(ValueError, match="non-empty string 'code'"):
        tool.invoke({"code": " "})
    with pytest.raises(ValueError, match="language='python'"):
        tool.invoke({"code": "1", "language": "javascript"})
    with pytest.raises(ValueError, match="timeout_seconds"):
        tool.invoke({"code": "1", "timeout_seconds": 61})
    with pytest.raises(ValueError, match="at most 5 chars"):
        tool.invoke({"code": "print(1)"})


class FakeSandboxRunner:
    def __init__(self, result: SandboxResult | SandboxUnavailableError) -> None:
        self.result = result
        self.specs: list[SandboxSpec] = []

    def run(self, spec: SandboxSpec) -> SandboxResult:
        self.specs.append(spec)
        if isinstance(self.result, SandboxUnavailableError):
            raise self.result
        return self.result


@dataclass(frozen=True, slots=True)
class FakeTask:
    id: uuid.UUID


class RecordingTaskService:
    def __init__(self) -> None:
        self.events: list[tuple[TaskEventType, dict[str, Any]]] = []

    def append_event(
        self,
        task: object,
        event_type: TaskEventType | str,
        payload: dict[str, Any] | None = None,
    ) -> object:
        del task
        self.events.append((TaskEventType(event_type), payload or {}))
        return object()
