import base64

import httpx
import pytest

from kortny.config import load_settings
from kortny.execution import SandboxUnavailableError
from kortny.execution.sandbox_sessions import (
    HttpSandboxSessionClient,
    SandboxSessionError,
    create_sandbox_session_client_from_settings,
)


def _client(handler: httpx.MockTransport) -> HttpSandboxSessionClient:
    return HttpSandboxSessionClient(
        base_url="http://sandbox-runner:8090",
        http_client=httpx.Client(transport=handler),
    )


def test_open_session_maps_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/sessions"
        return httpx.Response(
            200,
            json={
                "ok": True,
                "session_id": "s-1",
                "task_id": "task-1",
                "container_id": "c-1",
                "profile": "workbench",
                "reused": False,
            },
        )

    info = _client(httpx.MockTransport(handler)).open_session("task-1")

    assert info.session_id == "s-1"
    assert info.container_id == "c-1"
    assert info.reused is False


def test_exec_maps_result_and_timeout_fallback() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/sessions/s-1/exec"
        return httpx.Response(
            200,
            json={
                "ok": False,
                "timed_out": True,
                "stdout": "",
                "stderr": "",
            },
        )

    result = _client(httpx.MockTransport(handler)).exec("s-1", "sleep 999")

    assert result.exit_code == 124
    assert result.timed_out is True
    assert result.ok is False


def test_write_and_read_file_round_trip_base64() -> None:
    stored: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "PUT":
            import json

            body = json.loads(request.content)
            stored["content_b64"] = body["content_b64"]
            return httpx.Response(
                200, json={"ok": True, "size_bytes": 5, "path": body["path"]}
            )
        return httpx.Response(
            200,
            json={"ok": True, "content_b64": stored["content_b64"]},
        )

    client = _client(httpx.MockTransport(handler))
    written = client.write_file("s-1", "/workspace/a.txt", b"hello")
    content = client.read_file("s-1", "/workspace/a.txt")

    assert written == 5
    assert content == b"hello"


def test_export_archive_returns_raw_bytes() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/sessions/s-1/archive"
        return httpx.Response(
            200,
            content=b"tar-bytes",
            headers={"content-type": "application/x-tar"},
        )

    content = _client(httpx.MockTransport(handler)).export_archive(
        "s-1", "/workspace/dist"
    )

    assert content == b"tar-bytes"


def test_503_maps_to_sandbox_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"detail": "sessions disabled"})

    with pytest.raises(SandboxUnavailableError, match="sessions disabled"):
        _client(httpx.MockTransport(handler)).open_session("task-1")


def test_client_error_maps_to_session_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json={"detail": "bad path"})

    with pytest.raises(SandboxSessionError, match="bad path") as exc_info:
        _client(httpx.MockTransport(handler)).write_file("s-1", "/etc/passwd", b"x")

    assert exc_info.value.status_code == 422


def test_transport_error_maps_to_sandbox_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with pytest.raises(SandboxUnavailableError):
        _client(httpx.MockTransport(handler)).open_session("task-1")


def test_factory_returns_none_without_runner_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_required_settings(monkeypatch)

    settings = load_settings(env_file=None)

    assert create_sandbox_session_client_from_settings(settings) is None


def test_factory_builds_client_from_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_required_settings(monkeypatch)
    monkeypatch.setenv("KORTNY_SANDBOX_RUNNER_URL", "http://sandbox-runner:8090/")
    monkeypatch.setenv("KORTNY_SANDBOX_RUNNER_TIMEOUT_SECONDS", "42.0")

    client = create_sandbox_session_client_from_settings(load_settings(env_file=None))

    assert isinstance(client, HttpSandboxSessionClient)
    assert client.base_url == "http://sandbox-runner:8090"
    assert client.timeout_seconds == 42.0


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


def test_read_file_decodes_runner_content() -> None:
    payload_b64 = base64.b64encode(b"data").decode("ascii")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True, "content_b64": payload_b64})

    assert (
        _client(httpx.MockTransport(handler)).read_file("s-1", "/workspace/x")
        == b"data"
    )
