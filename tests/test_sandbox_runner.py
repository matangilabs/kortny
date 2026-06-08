from pathlib import Path

from fastapi.testclient import TestClient

from kortny.sandbox_runner import (
    DockerApiProbe,
    SandboxRunnerSettings,
    create_app,
    load_sandbox_runner_settings,
)
from kortny.sandbox_runner.docker_api import DockerApiClient, _docker_host_base_url


def test_sandbox_runner_health_reports_control_plane_only() -> None:
    settings = SandboxRunnerSettings(
        runner_name="test-runner",
        docker_host="tcp://sandbox-docker-proxy:2375",
    )

    with TestClient(create_app(settings=settings)) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "service": "kortny-sandbox-runner",
        "runner": "test-runner",
        "docker_host_configured": True,
        "execution_enabled": False,
        "mode": "control_plane_smoke",
    }


def test_sandbox_runner_smoke_does_not_execute_code() -> None:
    settings = SandboxRunnerSettings(
        runner_name="test-runner",
        docker_host="tcp://sandbox-docker-proxy:2375",
    )

    with TestClient(create_app(settings=settings)) as client:
        response = client.post("/smoke", json={"message": "hello"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["message"] == "hello"
    assert payload["execution_enabled"] is False
    assert payload["execution_attempted"] is False
    assert payload["sandbox_policy"]["requires_sandbox"] is True
    assert payload["sandbox_policy"]["network"] == "none"
    assert payload["sandbox_policy"]["resource_limits"] == {
        "cpus": 1.0,
        "memory_mb": 512,
        "pids_limit": 128,
        "timeout_seconds": 60,
    }


def test_sandbox_runner_docker_smoke_reports_reachable_proxy() -> None:
    settings = SandboxRunnerSettings(
        runner_name="test-runner",
        docker_host="tcp://sandbox-docker-proxy:2375",
    )
    docker_client = FakeDockerClient(
        DockerApiProbe(
            ok=True,
            configured=True,
            api_version="1.45",
            docker_version="26.1.0",
            platform_name="Docker Engine",
            status_code=200,
        )
    )

    with TestClient(
        create_app(settings=settings, docker_client=docker_client)
    ) as client:
        response = client.get("/docker-smoke")

    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "service": "kortny-sandbox-runner",
        "runner": "test-runner",
        "docker_host_configured": True,
        "execution_enabled": False,
        "execution_attempted": False,
        "docker_api": {
            "ok": True,
            "configured": True,
            "endpoint": "GET /version",
            "api_version": "1.45",
            "docker_version": "26.1.0",
            "platform_name": "Docker Engine",
            "status_code": 200,
            "error_type": None,
            "error": None,
        },
    }
    assert docker_client.calls == 1


def test_sandbox_runner_docker_smoke_reports_unavailable_proxy() -> None:
    settings = SandboxRunnerSettings(
        runner_name="test-runner",
        docker_host="tcp://sandbox-docker-proxy:2375",
    )
    docker_client = FakeDockerClient(
        DockerApiProbe(
            ok=False,
            configured=True,
            error_type="ConnectError",
            error="connection refused",
        )
    )

    with TestClient(
        create_app(settings=settings, docker_client=docker_client)
    ) as client:
        response = client.get("/docker-smoke")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is False
    assert payload["execution_enabled"] is False
    assert payload["execution_attempted"] is False
    assert payload["docker_api"]["configured"] is True
    assert payload["docker_api"]["error_type"] == "ConnectError"
    assert payload["docker_api"]["error"] == "connection refused"


def test_sandbox_runner_run_contract_rejects_execution_while_disabled() -> None:
    settings = SandboxRunnerSettings(
        runner_name="test-runner",
        docker_host="tcp://sandbox-docker-proxy:2375",
    )

    with TestClient(create_app(settings=settings)) as client:
        response = client.post(
            "/run",
            json={
                "image": "kortny/sandbox-python:latest",
                "command": ["python", "-c", "print('hello')"],
                "workspace_path": "/tmp/kortny-task-123",
                "artifacts_path": "/tmp/kortny-task-123/artifacts",
                "network": "none",
                "env": {"SAFE_FLAG": "1", "SECRET_TOKEN": "do-not-leak"},
                "resource_limits": {
                    "cpus": 1.5,
                    "memory_mb": 256,
                    "pids_limit": 32,
                    "timeout_seconds": 10,
                },
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is False
    assert payload["status"] == "execution_disabled"
    assert payload["execution_enabled"] is False
    assert payload["execution_attempted"] is False
    assert payload["request"] == {
        "image": "kortny/sandbox-python:latest",
        "command": ["python", "-c", "print('hello')"],
        "workspace_path": "/tmp/kortny-task-123",
        "artifacts_path": "/tmp/kortny-task-123/artifacts",
        "network": "none",
        "egress_allowlist": [],
        "env_keys": ["SAFE_FLAG", "SECRET_TOKEN"],
        "resource_limits": {
            "cpus": 1.5,
            "memory_mb": 256,
            "pids_limit": 32,
            "timeout_seconds": 10,
        },
    }
    assert "do-not-leak" not in str(payload)


def test_sandbox_runner_run_contract_reports_enabled_execution_as_unimplemented() -> (
    None
):
    settings = SandboxRunnerSettings(execution_enabled=True)

    with TestClient(create_app(settings=settings)) as client:
        response = client.post(
            "/run",
            json={
                "image": "kortny/sandbox-python:latest",
                "command": ["python", "-c", "print('hello')"],
                "workspace_path": "/tmp/kortny-task-123",
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is False
    assert payload["status"] == "execution_not_implemented"
    assert payload["execution_enabled"] is True
    assert payload["execution_attempted"] is False


def test_sandbox_runner_run_contract_accepts_allowlist_network_shape() -> None:
    with TestClient(create_app(settings=SandboxRunnerSettings())) as client:
        response = client.post(
            "/run",
            json={
                "image": "kortny/sandbox-python:latest",
                "command": ["python", "-c", "print('hello')"],
                "workspace_path": "/tmp/kortny-task-123",
                "network": "allowlist",
                "egress_allowlist": ["pypi.org", "files.pythonhosted.org"],
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["execution_attempted"] is False
    assert payload["request"]["network"] == "allowlist"
    assert payload["request"]["egress_allowlist"] == [
        "pypi.org",
        "files.pythonhosted.org",
    ]


def test_sandbox_runner_run_contract_rejects_allowlist_without_hosts() -> None:
    with TestClient(create_app(settings=SandboxRunnerSettings())) as client:
        response = client.post(
            "/run",
            json={
                "image": "kortny/sandbox-python:latest",
                "command": ["python", "-c", "print('hello')"],
                "workspace_path": "/tmp/kortny-task-123",
                "network": "allowlist",
            },
        )

    assert response.status_code == 422


def test_sandbox_runner_run_contract_rejects_allowlist_hosts_on_no_network() -> None:
    with TestClient(create_app(settings=SandboxRunnerSettings())) as client:
        response = client.post(
            "/run",
            json={
                "image": "kortny/sandbox-python:latest",
                "command": ["python", "-c", "print('hello')"],
                "workspace_path": "/tmp/kortny-task-123",
                "network": "none",
                "egress_allowlist": ["pypi.org"],
            },
        )

    assert response.status_code == 422


def test_sandbox_runner_run_contract_rejects_empty_command() -> None:
    with TestClient(create_app(settings=SandboxRunnerSettings())) as client:
        response = client.post(
            "/run",
            json={
                "image": "kortny/sandbox-python:latest",
                "command": [],
                "workspace_path": "/tmp/kortny-task-123",
            },
        )

    assert response.status_code == 422


def test_sandbox_runner_run_contract_rejects_empty_command_part() -> None:
    with TestClient(create_app(settings=SandboxRunnerSettings())) as client:
        response = client.post(
            "/run",
            json={
                "image": "kortny/sandbox-python:latest",
                "command": ["python", ""],
                "workspace_path": "/tmp/kortny-task-123",
            },
        )

    assert response.status_code == 422


def test_sandbox_runner_run_contract_rejects_empty_env_key() -> None:
    with TestClient(create_app(settings=SandboxRunnerSettings())) as client:
        response = client.post(
            "/run",
            json={
                "image": "kortny/sandbox-python:latest",
                "command": ["python", "-c", "print('hello')"],
                "workspace_path": "/tmp/kortny-task-123",
                "env": {"": "nope"},
            },
        )

    assert response.status_code == 422


def test_sandbox_runner_run_contract_rejects_bad_resource_limits() -> None:
    with TestClient(create_app(settings=SandboxRunnerSettings())) as client:
        response = client.post(
            "/run",
            json={
                "image": "kortny/sandbox-python:latest",
                "command": ["python", "-c", "print('hello')"],
                "workspace_path": "/tmp/kortny-task-123",
                "resource_limits": {"cpus": 0},
            },
        )

    assert response.status_code == 422


def test_sandbox_runner_settings_load_from_env_without_secrets() -> None:
    settings = load_sandbox_runner_settings(
        {
            "KORTNY_SANDBOX_RUNNER_NAME": "env-runner",
            "DOCKER_HOST": "tcp://sandbox-docker-proxy:2375",
            "KORTNY_SANDBOX_CPUS": "2.5",
            "KORTNY_SANDBOX_MEMORY_MB": "1024",
            "KORTNY_SANDBOX_PIDS_LIMIT": "64",
            "KORTNY_SANDBOX_TIMEOUT_SECONDS": "30",
            "IGNORED_SECRET": "do-not-read",
        }
    )

    assert settings.runner_name == "env-runner"
    assert settings.docker_host_configured is True
    assert settings.execution_enabled is False
    assert settings.default_cpus == 2.5
    assert settings.default_memory_mb == 1024
    assert settings.default_pids_limit == 64
    assert settings.default_timeout_seconds == 30


def test_docker_api_client_reports_missing_host_without_network_call() -> None:
    probe = DockerApiClient(docker_host="").version()

    assert probe.ok is False
    assert probe.configured is False
    assert probe.error_type == "DockerHostMissing"


def test_docker_host_base_url_supports_proxy_tcp_urls() -> None:
    assert (
        _docker_host_base_url("tcp://sandbox-docker-proxy:2375")
        == "http://sandbox-docker-proxy:2375"
    )
    assert _docker_host_base_url("http://localhost:2375") == "http://localhost:2375"


def test_compose_sandbox_services_are_profiled_and_do_not_use_env_file() -> None:
    compose = Path("compose.yaml").read_text()
    sandbox_runner_block = compose.split("  sandbox-runner:", maxsplit=1)[1].split(
        "\n\nvolumes:",
        maxsplit=1,
    )[0]
    socket_proxy_block = compose.split(
        "  sandbox-docker-proxy:",
        maxsplit=1,
    )[1].split("  sandbox-runner:", maxsplit=1)[0]

    assert "profiles:\n      - sandbox" in sandbox_runner_block
    assert "profiles:\n      - sandbox" in socket_proxy_block
    assert "env_file:" not in sandbox_runner_block
    assert "env_file:" not in socket_proxy_block
    assert "ports:" not in sandbox_runner_block
    assert "ports:" not in socket_proxy_block
    assert "DOCKER_HOST: tcp://sandbox-docker-proxy:2375" in sandbox_runner_block
    assert "/var/run/docker.sock:/var/run/docker.sock:ro" in socket_proxy_block
    assert "networks:\n  sandbox-control:\n    internal: true" in compose


class FakeDockerClient:
    def __init__(self, probe: DockerApiProbe) -> None:
        self.probe = probe
        self.calls = 0

    def version(self) -> DockerApiProbe:
        self.calls += 1
        return self.probe
