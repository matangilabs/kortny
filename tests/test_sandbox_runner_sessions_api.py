import base64
import io
import tarfile

from fastapi.testclient import TestClient

from kortny.sandbox_runner import SandboxRunnerSettings, create_app
from kortny.sandbox_runner.sessions import SessionConfig, SessionManager
from tests.test_sandbox_sessions import FakeDockerSessionClient


def _client(
    *,
    execution_enabled: bool = True,
    sessions_enabled: bool = True,
    docker: FakeDockerSessionClient | None = None,
) -> tuple[TestClient, FakeDockerSessionClient]:
    settings = SandboxRunnerSettings(
        runner_name="test-runner",
        docker_host="tcp://sandbox-docker-proxy:2375",
        execution_enabled=execution_enabled,
        sessions_enabled=sessions_enabled,
    )
    resolved_docker = docker or FakeDockerSessionClient()
    manager = SessionManager(
        docker_client=resolved_docker,
        config=SessionConfig(image=settings.default_image),
    )
    app = create_app(settings=settings, session_manager=manager)
    return TestClient(app), resolved_docker


def test_create_session_returns_session_info() -> None:
    client, _ = _client()

    with client:
        response = client.post("/sessions", json={"task_id": "task-1"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["task_id"] == "task-1"
    assert payload["reused"] is False
    assert payload["session_id"]


def test_create_session_rejected_when_execution_disabled() -> None:
    client, _ = _client(execution_enabled=False)

    with client:
        response = client.post("/sessions", json={"task_id": "task-1"})

    assert response.status_code == 503


def test_create_session_rejected_when_sessions_disabled() -> None:
    client, _ = _client(sessions_enabled=False)

    with client:
        response = client.post("/sessions", json={"task_id": "task-1"})

    assert response.status_code == 503


def test_session_exec_runs_command() -> None:
    client, docker = _client()

    with client:
        created = client.post("/sessions", json={"task_id": "task-1"}).json()
        response = client.post(
            f"/sessions/{created['session_id']}/exec",
            json={"command": "echo hi", "timeout_seconds": 30},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["exit_code"] == 0
    assert payload["stdout"] == "ok"
    assert docker.execs[-1][1] == ("timeout", "30", "/bin/sh", "-lc", "echo hi")


def test_session_exec_unknown_session_is_404() -> None:
    client, _ = _client()

    with client:
        response = client.post("/sessions/missing/exec", json={"command": "echo hi"})

    assert response.status_code == 404


def test_session_file_write_and_read_round_trip() -> None:
    client, docker = _client()
    content = b"<html>hello</html>"

    with client:
        created = client.post("/sessions", json={"task_id": "task-1"}).json()
        session_id = created["session_id"]
        write_response = client.put(
            f"/sessions/{session_id}/files",
            json={
                "path": "/workspace/app/index.html",
                "content_b64": base64.b64encode(content).decode("ascii"),
            },
        )
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as tar:
            info = tarfile.TarInfo(name="index.html")
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
        docker.archive_content = buffer.getvalue()
        read_response = client.get(
            f"/sessions/{session_id}/files",
            params={"path": "/workspace/app/index.html"},
        )

    assert write_response.status_code == 200
    assert write_response.json()["size_bytes"] == len(content)
    assert read_response.status_code == 200
    assert base64.b64decode(read_response.json()["content_b64"]) == content


def test_session_file_write_rejects_unsafe_path() -> None:
    client, _ = _client()

    with client:
        created = client.post("/sessions", json={"task_id": "task-1"}).json()
        response = client.put(
            f"/sessions/{created['session_id']}/files",
            json={
                "path": "/etc/passwd",
                "content_b64": base64.b64encode(b"x").decode("ascii"),
            },
        )

    assert response.status_code == 422


def test_session_archive_returns_tar_bytes() -> None:
    client, docker = _client()
    docker.archive_content = b"fake-tar-bytes"

    with client:
        created = client.post("/sessions", json={"task_id": "task-1"}).json()
        response = client.get(
            f"/sessions/{created['session_id']}/archive",
            params={"path": "/workspace/dist"},
        )

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/x-tar"
    assert response.content == b"fake-tar-bytes"


def test_session_close_removes_container() -> None:
    client, docker = _client()

    with client:
        created = client.post("/sessions", json={"task_id": "task-1"}).json()
        response = client.delete(f"/sessions/{created['session_id']}")

    assert response.status_code == 200
    assert docker.removed
