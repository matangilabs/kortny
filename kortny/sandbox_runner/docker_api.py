"""Minimal Docker API client for sandbox-runner checks and execution."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from time import monotonic
from typing import Any, Protocol

import httpx

from kortny.execution import SandboxResourceLimits

JsonObject = dict[str, Any]


@dataclass(frozen=True, slots=True)
class DockerApiProbe:
    """Result of a safe Docker API reachability probe."""

    ok: bool
    configured: bool
    endpoint: str = "GET /version"
    api_version: str | None = None
    docker_version: str | None = None
    platform_name: str | None = None
    status_code: int | None = None
    error_type: str | None = None
    error: str | None = None

    def to_payload(self) -> JsonObject:
        return {
            "ok": self.ok,
            "configured": self.configured,
            "endpoint": self.endpoint,
            "api_version": self.api_version,
            "docker_version": self.docker_version,
            "platform_name": self.platform_name,
            "status_code": self.status_code,
            "error_type": self.error_type,
            "error": self.error,
        }


class DockerApiProbeClient(Protocol):
    """Shape used by the runner app to probe Docker safely."""

    def version(self) -> DockerApiProbe:
        """Probe Docker Engine `/version`."""
        ...


@dataclass(frozen=True, slots=True)
class DockerContainerRunSpec:
    """One hardened sibling-container execution request."""

    image: str
    command: tuple[str, ...]
    workspace_path: str
    env: dict[str, str]
    resource_limits: SandboxResourceLimits


@dataclass(frozen=True, slots=True)
class DockerContainerRunResult:
    """Result of one Docker-backed sandbox execution attempt."""

    ok: bool
    status: str
    execution_attempted: bool
    container_id: str | None = None
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    duration_ms: int | None = None
    status_code: int | None = None
    error_type: str | None = None
    error: str | None = None
    cleanup_error: str | None = None

    def to_payload(self) -> JsonObject:
        return {
            "ok": self.ok,
            "status": self.status,
            "execution_attempted": self.execution_attempted,
            "container_id": self.container_id,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_ms": self.duration_ms,
            "status_code": self.status_code,
            "error_type": self.error_type,
            "error": self.error,
            "cleanup_error": self.cleanup_error,
        }


class DockerApiRunnerClient(DockerApiProbeClient, Protocol):
    """Shape used by the runner app to probe and launch sandbox containers."""

    def run_container(self, spec: DockerContainerRunSpec) -> DockerContainerRunResult:
        """Create, start, wait, collect logs, and remove a sandbox container."""
        ...


@dataclass(frozen=True, slots=True)
class DockerApiClient:
    """Tiny Docker Engine API client for safe control-plane checks."""

    docker_host: str
    timeout_seconds: float = 2.0

    def version(self) -> DockerApiProbe:
        """Return Docker Engine version metadata through the configured endpoint."""

        if not self.docker_host.strip():
            return DockerApiProbe(
                ok=False,
                configured=False,
                error_type="DockerHostMissing",
                error="DOCKER_HOST is not configured.",
            )

        try:
            base_url = _docker_host_base_url(self.docker_host)
            response = httpx.get(
                f"{base_url}/version",
                timeout=self.timeout_seconds,
            )
            payload = response.json() if response.content else {}
            if not isinstance(payload, dict):
                payload = {}
            return DockerApiProbe(
                ok=response.is_success,
                configured=True,
                api_version=_optional_str(payload.get("ApiVersion")),
                docker_version=_optional_str(payload.get("Version")),
                platform_name=_platform_name(payload),
                status_code=response.status_code,
                error=None if response.is_success else response.text[:500],
            )
        except Exception as exc:
            return DockerApiProbe(
                ok=False,
                configured=True,
                error_type=type(exc).__name__,
                error=str(exc),
            )

    def run_container(self, spec: DockerContainerRunSpec) -> DockerContainerRunResult:
        """Run one hardened sibling container through the Docker Engine API."""

        if not self.docker_host.strip():
            return DockerContainerRunResult(
                ok=False,
                status="docker_host_missing",
                execution_attempted=False,
                error_type="DockerHostMissing",
                error="DOCKER_HOST is not configured.",
            )

        base_url = _docker_host_base_url(self.docker_host)
        container_id: str | None = None
        cleanup_error: str | None = None
        started_at = monotonic()

        try:
            create_response = httpx.post(
                f"{base_url}/containers/create",
                json=_container_create_payload(spec),
                timeout=self.timeout_seconds,
            )
            create_payload = create_response.json() if create_response.content else {}
            if not create_response.is_success:
                return _failed_run_result(
                    status="container_create_failed",
                    execution_attempted=False,
                    response=create_response,
                    payload=create_payload,
                    duration_ms=_elapsed_ms(started_at),
                )
            if not isinstance(create_payload, dict) or not isinstance(
                create_payload.get("Id"), str
            ):
                return DockerContainerRunResult(
                    ok=False,
                    status="container_create_failed",
                    execution_attempted=False,
                    status_code=create_response.status_code,
                    error_type="DockerResponseInvalid",
                    error="Docker create response did not include a container id.",
                    duration_ms=_elapsed_ms(started_at),
                )
            container_id = create_payload["Id"]

            start_response = httpx.post(
                f"{base_url}/containers/{container_id}/start",
                timeout=self.timeout_seconds,
            )
            if not start_response.is_success:
                return _failed_run_result(
                    status="container_start_failed",
                    execution_attempted=True,
                    response=start_response,
                    duration_ms=_elapsed_ms(started_at),
                    container_id=container_id,
                )

            wait_response = httpx.post(
                f"{base_url}/containers/{container_id}/wait",
                timeout=spec.resource_limits.timeout_seconds + self.timeout_seconds,
            )
            wait_payload = wait_response.json() if wait_response.content else {}
            logs = _container_logs(base_url=base_url, container_id=container_id)
            if not wait_response.is_success:
                return _failed_run_result(
                    status="container_wait_failed",
                    execution_attempted=True,
                    response=wait_response,
                    payload=wait_payload,
                    duration_ms=_elapsed_ms(started_at),
                    container_id=container_id,
                    stdout=logs,
                )

            exit_code = _exit_code(wait_payload)
            return DockerContainerRunResult(
                ok=exit_code == 0,
                status="succeeded" if exit_code == 0 else "failed",
                execution_attempted=True,
                container_id=container_id,
                exit_code=exit_code,
                stdout=logs,
                duration_ms=_elapsed_ms(started_at),
                status_code=wait_response.status_code,
                cleanup_error=cleanup_error,
            )
        except httpx.TimeoutException as exc:
            cleanup_error = _kill_container(
                base_url=base_url, container_id=container_id
            )
            return DockerContainerRunResult(
                ok=False,
                status="timed_out",
                execution_attempted=container_id is not None,
                container_id=container_id,
                duration_ms=_elapsed_ms(started_at),
                error_type=type(exc).__name__,
                error=str(exc),
                cleanup_error=cleanup_error,
            )
        except Exception as exc:
            return DockerContainerRunResult(
                ok=False,
                status="docker_api_error",
                execution_attempted=container_id is not None,
                container_id=container_id,
                duration_ms=_elapsed_ms(started_at),
                error_type=type(exc).__name__,
                error=str(exc),
                cleanup_error=cleanup_error,
            )
        finally:
            remove_error = _remove_container(
                base_url=base_url,
                container_id=container_id,
            )
            cleanup_error = cleanup_error or remove_error


def _docker_host_base_url(docker_host: str) -> str:
    value = docker_host.strip().rstrip("/")
    if value.startswith("tcp://"):
        return f"http://{value.removeprefix('tcp://')}"
    if value.startswith(("http://", "https://")):
        return value
    raise ValueError(
        "Only tcp://, http://, and https:// Docker hosts are supported by "
        "the sandbox-runner smoke client."
    )


def _container_create_payload(spec: DockerContainerRunSpec) -> JsonObject:
    resource_limits = spec.resource_limits
    tmpfs_options = "rw,nosuid,nodev,size=67108864,mode=1777"
    return {
        "Image": spec.image,
        "Cmd": list(spec.command),
        "Env": [f"{key}={value}" for key, value in sorted(spec.env.items())],
        "WorkingDir": spec.workspace_path,
        "Tty": True,
        "OpenStdin": False,
        "AttachStdout": True,
        "AttachStderr": True,
        "Labels": {
            "kortny.sandbox": "true",
            "kortny.sandbox.profile": "default",
        },
        "HostConfig": {
            "AutoRemove": False,
            "Binds": [],
            "CapDrop": ["ALL"],
            "Memory": resource_limits.memory_mb * 1024 * 1024,
            "NanoCpus": _nano_cpus(resource_limits.cpus),
            "NetworkMode": "none",
            "PidsLimit": resource_limits.pids_limit,
            "Privileged": False,
            "ReadonlyRootfs": True,
            "SecurityOpt": ["no-new-privileges"],
            "Tmpfs": {
                spec.workspace_path: tmpfs_options,
                "/tmp": tmpfs_options,
            },
        },
    }


def _failed_run_result(
    *,
    status: str,
    execution_attempted: bool,
    response: httpx.Response,
    payload: object | None = None,
    duration_ms: int,
    container_id: str | None = None,
    stdout: str = "",
) -> DockerContainerRunResult:
    return DockerContainerRunResult(
        ok=False,
        status=status,
        execution_attempted=execution_attempted,
        container_id=container_id,
        stdout=stdout,
        duration_ms=duration_ms,
        status_code=response.status_code,
        error=_docker_error_text(response=response, payload=payload),
    )


def _container_logs(*, base_url: str, container_id: str) -> str:
    response = httpx.get(
        f"{base_url}/containers/{container_id}/logs",
        params={"stdout": "1", "stderr": "1", "timestamps": "0"},
        timeout=5,
    )
    if not response.is_success:
        return ""
    return response.text[-64000:]


def _kill_container(*, base_url: str, container_id: str | None) -> str | None:
    if not container_id:
        return None
    try:
        response = httpx.post(f"{base_url}/containers/{container_id}/kill", timeout=2)
        if response.status_code in {204, 304, 404}:
            return None
        return response.text[:500]
    except Exception as exc:
        return str(exc)


def _remove_container(*, base_url: str, container_id: str | None) -> str | None:
    if not container_id:
        return None
    try:
        response = httpx.delete(
            f"{base_url}/containers/{container_id}",
            params={"force": "1", "v": "1"},
            timeout=2,
        )
        if response.status_code in {204, 404}:
            return None
        return response.text[:500]
    except Exception as exc:
        return str(exc)


def _docker_error_text(*, response: httpx.Response, payload: object | None) -> str:
    if isinstance(payload, dict) and isinstance(payload.get("message"), str):
        return payload["message"][:500]
    return response.text[:500]


def _exit_code(payload: object) -> int | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get("StatusCode")
    return value if isinstance(value, int) else None


def _elapsed_ms(started_at: float) -> int:
    return int((monotonic() - started_at) * 1000)


def _nano_cpus(cpus: float) -> int:
    return int(Decimal(str(cpus)) * Decimal("1000000000"))


def _platform_name(payload: JsonObject) -> str | None:
    platform = payload.get("Platform")
    if not isinstance(platform, dict):
        return None
    return _optional_str(platform.get("Name"))


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None
