from pathlib import Path

import httpx
import pytest

from kortny.config import load_settings
from kortny.execution import (
    HttpSandboxRunner,
    SandboxResourceLimits,
    SandboxSpec,
    SandboxUnavailableError,
    create_sandbox_runner_from_settings,
)


def test_http_sandbox_runner_posts_spec_and_maps_success() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["payload"] = request.read().decode()
        return httpx.Response(
            200,
            json={
                "ok": True,
                "status": "succeeded",
                "execution_attempted": True,
                "result": {
                    "ok": True,
                    "status": "succeeded",
                    "execution_attempted": True,
                    "container_id": "sandbox-123",
                    "exit_code": 0,
                    "stdout": "hello\n",
                    "stderr": "",
                    "duration_ms": 42,
                    "status_code": 200,
                },
            },
        )

    runner = HttpSandboxRunner(
        base_url="http://sandbox-runner:8090/",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    spec = SandboxSpec(
        image="kortny/sandbox-python:latest",
        command=("python", "-c", "print('hello')"),
        workspace_path=Path("/workspace/task-123"),
        artifacts_path=Path("/workspace/task-123/artifacts"),
        env={"SAFE_FLAG": "1", "SECRET_TOKEN": "do-not-log"},
        resource_limits=SandboxResourceLimits(memory_mb=256, timeout_seconds=10),
    )

    result = runner.run(spec)

    assert captured["url"] == "http://sandbox-runner:8090/run"
    assert '"SECRET_TOKEN":"do-not-log"' in str(captured["payload"])
    assert result.exit_code == 0
    assert result.stdout == "hello\n"
    assert result.stderr == ""
    assert result.usage == {
        "runner_status": "succeeded",
        "runner_ok": True,
        "execution_attempted": True,
        "container_id": "sandbox-123",
        "duration_ms": 42,
        "status_code": 200,
    }
    assert [event.phase for event in result.events] == ["started", "exited"]
    assert result.events[0].details == {"container_id": "sandbox-123"}
    assert result.events[1].details == {
        "container_id": "sandbox-123",
        "exit_code": 0,
    }


def test_http_sandbox_runner_returns_failed_exit_without_raising() -> None:
    runner = HttpSandboxRunner(
        base_url="http://sandbox-runner:8090",
        http_client=httpx.Client(
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(
                    200,
                    json={
                        "ok": False,
                        "status": "failed",
                        "execution_attempted": True,
                        "result": {
                            "ok": False,
                            "status": "failed",
                            "execution_attempted": True,
                            "container_id": "sandbox-123",
                            "exit_code": 2,
                            "stdout": "",
                            "stderr": "boom\n",
                            "duration_ms": 12,
                        },
                    },
                )
            )
        ),
    )

    result = runner.run(_sandbox_spec())

    assert result.exit_code == 2
    assert result.stderr == "boom\n"
    assert result.usage["runner_status"] == "failed"
    assert result.events[-1].details == {
        "container_id": "sandbox-123",
        "exit_code": 2,
    }


def test_http_sandbox_runner_maps_timeout_without_exit_code() -> None:
    runner = HttpSandboxRunner(
        base_url="http://sandbox-runner:8090",
        http_client=httpx.Client(
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(
                    200,
                    json={
                        "ok": False,
                        "status": "timed_out",
                        "execution_attempted": True,
                        "result": {
                            "ok": False,
                            "status": "timed_out",
                            "execution_attempted": True,
                            "container_id": "sandbox-123",
                            "error_type": "TimeoutException",
                            "error": "deadline exceeded",
                        },
                    },
                )
            )
        ),
    )

    result = runner.run(_sandbox_spec())

    assert result.exit_code == 124
    assert result.usage["runner_status"] == "timed_out"
    assert result.usage["error_type"] == "TimeoutException"
    assert [event.phase for event in result.events] == ["started", "killed"]


def test_http_sandbox_runner_raises_when_runner_does_not_attempt_execution() -> None:
    runner = HttpSandboxRunner(
        base_url="http://sandbox-runner:8090",
        http_client=httpx.Client(
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(
                    200,
                    json={
                        "ok": False,
                        "status": "execution_disabled",
                        "execution_attempted": False,
                        "reason": "not enabled",
                    },
                )
            )
        ),
    )

    with pytest.raises(
        SandboxUnavailableError,
        match="execution_disabled: not enabled",
    ):
        runner.run(_sandbox_spec())


def test_http_sandbox_runner_raises_on_http_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    runner = HttpSandboxRunner(
        base_url="http://sandbox-runner:8090",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(SandboxUnavailableError, match="ConnectError"):
        runner.run(_sandbox_spec())


def test_create_sandbox_runner_from_settings_uses_optional_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_required_settings(monkeypatch)
    monkeypatch.setenv("KORTNY_SANDBOX_RUNNER_URL", "http://sandbox-runner:8090/")
    monkeypatch.setenv("KORTNY_SANDBOX_RUNNER_TIMEOUT_SECONDS", "11.5")

    runner = create_sandbox_runner_from_settings(load_settings(env_file=None))

    assert isinstance(runner, HttpSandboxRunner)
    assert runner.base_url == "http://sandbox-runner:8090"
    assert runner.timeout_seconds == 11.5


def test_create_sandbox_runner_from_settings_returns_none_without_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_required_settings(monkeypatch)

    assert create_sandbox_runner_from_settings(load_settings(env_file=None)) is None


def _sandbox_spec() -> SandboxSpec:
    return SandboxSpec(
        image="kortny/sandbox-python:latest",
        command=("python", "-c", "print('hello')"),
        workspace_path=Path("/workspace/task-123"),
    )


def _set_required_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KORTNY_SANDBOX_RUNNER_URL", raising=False)
    monkeypatch.delenv("KORTNY_SANDBOX_RUNNER_TIMEOUT_SECONDS", raising=False)
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    monkeypatch.setenv("SLACK_APP_TOKEN", "xapp-test")
    monkeypatch.setenv("SLACK_SIGNING_SECRET", "signing-secret")
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("LLM_API_KEY", "llm-key")
    monkeypatch.setenv("LLM_MODEL", "openai/gpt-4o")
    monkeypatch.setenv("COMPOSIO_API_KEY", "composio-key")
    monkeypatch.setenv("POSTGRES_URL", "postgresql://kortny:kortny@localhost/kortny")
