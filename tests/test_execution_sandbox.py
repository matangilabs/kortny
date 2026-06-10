from pathlib import Path
from typing import Any, cast

import pytest

from kortny.execution import (
    SANDBOX_LIFECYCLE_MESSAGE,
    SANDBOX_RESULT_MESSAGE,
    SandboxArtifact,
    SandboxEventRecorder,
    SandboxLifecycleEvent,
    SandboxResourceLimits,
    SandboxResult,
    SandboxSpec,
    ToolSandboxPolicy,
    sandbox_lifecycle_event_payload,
    sandbox_result_event_payload,
)


def test_tool_sandbox_policy_defaults_to_no_sandbox() -> None:
    policy = ToolSandboxPolicy()

    assert policy.requires_sandbox is False
    assert policy.network == "none"
    assert policy.to_payload() == {
        "requires_sandbox": False,
        "profile": "default",
        "network": "none",
        "egress_allowlist": [],
        "resource_limits": {
            "cpus": 1.0,
            "memory_mb": 512,
            "pids_limit": 128,
            "timeout_seconds": 60,
        },
        "reason": "",
    }


def test_allowlist_network_requires_explicit_hosts() -> None:
    with pytest.raises(ValueError, match="allowlist network requires egress hosts"):
        ToolSandboxPolicy(network="allowlist")

    with pytest.raises(ValueError, match="allowlist network requires egress hosts"):
        SandboxSpec(
            image="ghcr.io/astral-sh/uv:python3.11-bookworm-slim",
            command=("python", "-c", "print('ok')"),
            workspace_path=Path("/tmp/task"),
            network="allowlist",
        )


def test_resource_limits_reject_non_positive_values() -> None:
    with pytest.raises(ValueError, match="CPU limit"):
        SandboxResourceLimits(cpus=0)
    with pytest.raises(ValueError, match="memory limit"):
        SandboxResourceLimits(memory_mb=0)
    with pytest.raises(ValueError, match="PID limit"):
        SandboxResourceLimits(pids_limit=0)
    with pytest.raises(ValueError, match="timeout"):
        SandboxResourceLimits(timeout_seconds=0)


def test_sandbox_spec_payload_redacts_env_values() -> None:
    spec = SandboxSpec(
        image="ghcr.io/astral-sh/uv:python3.11-bookworm-slim",
        command=("python", "-c", "print('ok')"),
        workspace_path=Path("/tmp/task"),
        artifacts_path=Path("/tmp/task/artifacts"),
        env={"SAFE_FLAG": "1", "SECRET_TOKEN": "do-not-log"},
    )

    payload = spec.to_payload()

    assert payload["image"] == "ghcr.io/astral-sh/uv:python3.11-bookworm-slim"
    assert payload["command"] == ["python", "-c", "print('ok')"]
    assert payload["workspace_path"] == "/tmp/task"
    assert payload["artifacts_path"] == "/tmp/task/artifacts"
    assert payload["network"] == "none"
    assert payload["env_keys"] == ["SAFE_FLAG", "SECRET_TOKEN"]
    assert "do-not-log" not in str(payload)


def test_sandbox_result_payload_preserves_artifact_and_lifecycle_summary() -> None:
    result = SandboxResult(
        exit_code=0,
        stdout="done",
        artifacts=(
            SandboxArtifact(
                filename="report.pdf",
                path="/tmp/task/artifacts/report.pdf",
                mime_type="application/pdf",
                size_bytes=42,
            ),
        ),
        usage={"wall_ms": 123},
        events=(
            SandboxLifecycleEvent(
                phase="started",
                message="sandbox started",
                details={"container_id": "abc123"},
            ),
        ),
    )

    payload = result.to_payload()

    assert payload["exit_code"] == 0
    assert payload["artifact_count"] == 1
    assert cast(list[Any], payload["artifacts"])[0]["filename"] == "report.pdf"
    assert payload["usage"] == {"wall_ms": 123}
    assert payload["events"] == [
        {
            "phase": "started",
            "message": "sandbox started",
            "details": {"container_id": "abc123"},
        }
    ]


def test_lifecycle_event_payload_includes_redacted_spec_and_tool_context() -> None:
    spec = SandboxSpec(
        image="ghcr.io/astral-sh/uv:python3.11-bookworm-slim",
        command=("python", "task.py"),
        workspace_path=Path("/tmp/task"),
        env={"SECRET_TOKEN": "do-not-log"},
    )
    event = SandboxLifecycleEvent(
        phase="created",
        message="container created",
        details={"sandbox_id": "sandbox-123"},
    )

    payload = sandbox_lifecycle_event_payload(
        event,
        spec=spec,
        runner="docker",
        tool_name="code_exec",
        tool_call_id="call-123",
    )

    assert payload["message"] == SANDBOX_LIFECYCLE_MESSAGE
    assert payload["runner"] == "docker"
    assert payload["phase"] == "created"
    assert payload["tool"] == "code_exec"
    assert payload["tool_call_id"] == "call-123"
    assert cast(dict[str, Any], payload["spec"])["env_keys"] == ["SECRET_TOKEN"]
    assert "do-not-log" not in str(payload)


def test_result_event_payload_bounds_stdout_and_stderr_previews() -> None:
    result = SandboxResult(
        exit_code=137,
        stdout="x" * 10,
        stderr="memory limit exceeded",
        usage={"wall_ms": 999, "max_rss_mb": 512},
    )

    payload = sandbox_result_event_payload(
        result,
        runner="docker",
        output_preview_chars=4,
    )

    assert payload["message"] == SANDBOX_RESULT_MESSAGE
    assert payload["status"] == "failed"
    assert payload["exit_code"] == 137
    assert payload["stdout_chars"] == 10
    assert payload["stdout_preview"] == "xxxx"
    assert payload["stderr_chars"] == len("memory limit exceeded")
    assert payload["stderr_preview"] == "memo"
    assert payload["usage"] == {"wall_ms": 999, "max_rss_mb": 512}


def test_event_recorder_appends_lifecycle_and_result_logs() -> None:
    sink = FakeEventSink()
    recorder = SandboxEventRecorder(sink, runner="docker", output_preview_chars=3)
    task = object()

    lifecycle_event = SandboxLifecycleEvent(
        phase="started",
        message="sandbox started",
    )
    result = SandboxResult(exit_code=0, stdout="abcdef")

    recorder.record_lifecycle(
        task,
        lifecycle_event,
        tool_name="code_exec",
        tool_call_id="call-123",
    )
    recorder.record_result(
        task,
        result,
        tool_name="code_exec",
        tool_call_id="call-123",
    )

    assert sink.events == [
        (
            task,
            "log",
            {
                "message": SANDBOX_LIFECYCLE_MESSAGE,
                "source": "execution.sandbox",
                "runner": "docker",
                "phase": "started",
                "event": {
                    "phase": "started",
                    "message": "sandbox started",
                    "details": {},
                },
                "tool": "code_exec",
                "tool_call_id": "call-123",
            },
        ),
        (
            task,
            "log",
            {
                "message": SANDBOX_RESULT_MESSAGE,
                "source": "execution.sandbox",
                "runner": "docker",
                "status": "succeeded",
                "exit_code": 0,
                "stdout_chars": 6,
                "stderr_chars": 0,
                "stdout_preview": "abc",
                "stderr_preview": "",
                "artifact_count": 0,
                "artifacts": [],
                "usage": {},
                "lifecycle_event_count": 0,
                "lifecycle_events": [],
                "tool": "code_exec",
                "tool_call_id": "call-123",
            },
        ),
    ]


class FakeEventSink:
    def __init__(self) -> None:
        self.events: list[tuple[object, str, dict[str, object] | None]] = []

    def append_event(
        self,
        task: object,
        event_type: str,
        payload: dict[str, object] | None = None,
    ) -> object:
        event = (task, event_type, payload)
        self.events.append(event)
        return event
