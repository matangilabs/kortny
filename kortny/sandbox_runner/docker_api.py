"""Minimal Docker API client for sandbox-runner checks and execution."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
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
class DockerSessionCreateSpec:
    """One hardened long-lived session container request."""

    image: str
    session_id: str
    task_id: str
    profile: str
    workspace_path: str = "/workspace"
    env: dict[str, str] = field(default_factory=dict)
    cpus: float = 2.0
    memory_mb: int = 2048
    pids_limit: int = 512
    workspace_mb: int = 1024


@dataclass(frozen=True, slots=True)
class DockerSessionCreateResult:
    """Result of one session container create+start attempt."""

    ok: bool
    status: str
    container_id: str | None = None
    status_code: int | None = None
    error_type: str | None = None
    error: str | None = None

    def to_payload(self) -> JsonObject:
        return {
            "ok": self.ok,
            "status": self.status,
            "container_id": self.container_id,
            "status_code": self.status_code,
            "error_type": self.error_type,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class DockerExecResult:
    """Result of one `docker exec` style command in a session container."""

    ok: bool
    status: str
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    duration_ms: int | None = None
    timed_out: bool = False
    truncated: bool = False
    status_code: int | None = None
    error_type: str | None = None
    error: str | None = None

    def to_payload(self) -> JsonObject:
        return {
            "ok": self.ok,
            "status": self.status,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_ms": self.duration_ms,
            "timed_out": self.timed_out,
            "truncated": self.truncated,
            "status_code": self.status_code,
            "error_type": self.error_type,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class DockerArchiveResult:
    """Result of one archive read/write against a session container."""

    ok: bool
    status: str
    content: bytes = b""
    status_code: int | None = None
    error_type: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class DockerSessionContainer:
    """One labeled session container as listed from the Docker API."""

    container_id: str
    session_id: str
    task_id: str
    created_at_epoch: int
    running: bool


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


class DockerApiSessionClient(Protocol):
    """Shape used by the session manager to drive long-lived containers."""

    def create_session_container(
        self, spec: DockerSessionCreateSpec
    ) -> DockerSessionCreateResult:
        """Create and start one hardened session container."""
        ...

    def exec_in_container(
        self,
        container_id: str,
        command: tuple[str, ...],
        *,
        workdir: str,
        timeout_seconds: int,
        max_output_bytes: int = 65536,
    ) -> DockerExecResult:
        """Run one command in a running session container."""
        ...

    def get_archive(self, container_id: str, path: str) -> DockerArchiveResult:
        """Fetch a tar archive of a path inside the container."""
        ...

    def put_archive(
        self, container_id: str, path: str, tar_bytes: bytes
    ) -> DockerArchiveResult:
        """Extract a tar archive into a directory inside the container."""
        ...

    def list_session_containers(self) -> tuple[DockerSessionContainer, ...]:
        """List containers labeled as Kortny sandbox sessions."""
        ...

    def remove_session_container(self, container_id: str) -> str | None:
        """Force-remove one session container; return an error string if any."""
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


    def create_session_container(
        self, spec: DockerSessionCreateSpec
    ) -> DockerSessionCreateResult:
        """Create and start one hardened long-lived session container."""

        if not self.docker_host.strip():
            return DockerSessionCreateResult(
                ok=False,
                status="docker_host_missing",
                error_type="DockerHostMissing",
                error="DOCKER_HOST is not configured.",
            )

        base_url = _docker_host_base_url(self.docker_host)
        container_id: str | None = None
        try:
            create_response = httpx.post(
                f"{base_url}/containers/create",
                json=_session_create_payload(spec),
                timeout=self.timeout_seconds,
            )
            create_payload = create_response.json() if create_response.content else {}
            if not create_response.is_success:
                return DockerSessionCreateResult(
                    ok=False,
                    status="session_create_failed",
                    status_code=create_response.status_code,
                    error=_docker_error_text(
                        response=create_response, payload=create_payload
                    ),
                )
            if not isinstance(create_payload, dict) or not isinstance(
                create_payload.get("Id"), str
            ):
                return DockerSessionCreateResult(
                    ok=False,
                    status="session_create_failed",
                    status_code=create_response.status_code,
                    error_type="DockerResponseInvalid",
                    error="Docker create response did not include a container id.",
                )
            container_id = create_payload["Id"]

            start_response = httpx.post(
                f"{base_url}/containers/{container_id}/start",
                timeout=self.timeout_seconds,
            )
            if not start_response.is_success:
                self.remove_session_container(container_id)
                return DockerSessionCreateResult(
                    ok=False,
                    status="session_start_failed",
                    container_id=container_id,
                    status_code=start_response.status_code,
                    error=_docker_error_text(response=start_response, payload=None),
                )
            return DockerSessionCreateResult(
                ok=True,
                status="running",
                container_id=container_id,
                status_code=start_response.status_code,
            )
        except Exception as exc:
            if container_id:
                self.remove_session_container(container_id)
            return DockerSessionCreateResult(
                ok=False,
                status="docker_api_error",
                container_id=container_id,
                error_type=type(exc).__name__,
                error=str(exc),
            )

    def exec_in_container(
        self,
        container_id: str,
        command: tuple[str, ...],
        *,
        workdir: str,
        timeout_seconds: int,
        max_output_bytes: int = 65536,
    ) -> DockerExecResult:
        """Run one command in a running session container via the exec API."""

        if not self.docker_host.strip():
            return DockerExecResult(
                ok=False,
                status="docker_host_missing",
                error_type="DockerHostMissing",
                error="DOCKER_HOST is not configured.",
            )

        base_url = _docker_host_base_url(self.docker_host)
        started_at = monotonic()
        try:
            exec_create = httpx.post(
                f"{base_url}/containers/{container_id}/exec",
                json={
                    "AttachStdout": True,
                    "AttachStderr": True,
                    "Tty": False,
                    "Cmd": list(command),
                    "WorkingDir": workdir,
                },
                timeout=self.timeout_seconds,
            )
            exec_payload = exec_create.json() if exec_create.content else {}
            if not exec_create.is_success or not isinstance(
                exec_payload.get("Id"), str
            ):
                return DockerExecResult(
                    ok=False,
                    status="exec_create_failed",
                    duration_ms=_elapsed_ms(started_at),
                    status_code=exec_create.status_code,
                    error=_docker_error_text(
                        response=exec_create, payload=exec_payload
                    ),
                )
            exec_id = exec_payload["Id"]

            raw, truncated = _exec_start_stream(
                base_url=base_url,
                exec_id=exec_id,
                timeout_seconds=timeout_seconds + 10,
                max_output_bytes=max_output_bytes,
            )
            stdout, stderr = _demux_docker_stream(raw)

            inspect_response = httpx.get(
                f"{base_url}/exec/{exec_id}/json",
                timeout=self.timeout_seconds,
            )
            inspect_payload = (
                inspect_response.json() if inspect_response.content else {}
            )
            exit_code = (
                inspect_payload.get("ExitCode")
                if isinstance(inspect_payload, dict)
                else None
            )
            if not isinstance(exit_code, int):
                exit_code = None
            timed_out = exit_code == 124
            return DockerExecResult(
                ok=exit_code == 0,
                status=_exec_status(exit_code),
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                duration_ms=_elapsed_ms(started_at),
                timed_out=timed_out,
                truncated=truncated,
                status_code=inspect_response.status_code,
            )
        except httpx.TimeoutException as exc:
            return DockerExecResult(
                ok=False,
                status="timed_out",
                duration_ms=_elapsed_ms(started_at),
                timed_out=True,
                error_type=type(exc).__name__,
                error=str(exc),
            )
        except Exception as exc:
            return DockerExecResult(
                ok=False,
                status="docker_api_error",
                duration_ms=_elapsed_ms(started_at),
                error_type=type(exc).__name__,
                error=str(exc),
            )

    def get_archive(self, container_id: str, path: str) -> DockerArchiveResult:
        """Fetch a tar archive of one path inside a session container."""

        base_url = _docker_host_base_url(self.docker_host)
        try:
            response = httpx.get(
                f"{base_url}/containers/{container_id}/archive",
                params={"path": path},
                timeout=30.0,
            )
            if not response.is_success:
                return DockerArchiveResult(
                    ok=False,
                    status="archive_read_failed",
                    status_code=response.status_code,
                    error=response.text[:500],
                )
            return DockerArchiveResult(
                ok=True,
                status="succeeded",
                content=response.content,
                status_code=response.status_code,
            )
        except Exception as exc:
            return DockerArchiveResult(
                ok=False,
                status="docker_api_error",
                error_type=type(exc).__name__,
                error=str(exc),
            )

    def put_archive(
        self, container_id: str, path: str, tar_bytes: bytes
    ) -> DockerArchiveResult:
        """Extract one tar archive into a directory inside a session container."""

        base_url = _docker_host_base_url(self.docker_host)
        try:
            response = httpx.put(
                f"{base_url}/containers/{container_id}/archive",
                params={"path": path},
                content=tar_bytes,
                headers={"Content-Type": "application/x-tar"},
                timeout=30.0,
            )
            if not response.is_success:
                return DockerArchiveResult(
                    ok=False,
                    status="archive_write_failed",
                    status_code=response.status_code,
                    error=response.text[:500],
                )
            return DockerArchiveResult(
                ok=True,
                status="succeeded",
                status_code=response.status_code,
            )
        except Exception as exc:
            return DockerArchiveResult(
                ok=False,
                status="docker_api_error",
                error_type=type(exc).__name__,
                error=str(exc),
            )

    def list_session_containers(self) -> tuple[DockerSessionContainer, ...]:
        """List containers labeled as Kortny sandbox sessions."""

        base_url = _docker_host_base_url(self.docker_host)
        response = httpx.get(
            f"{base_url}/containers/json",
            params={
                "all": "1",
                "filters": json.dumps({"label": ["kortny.sandbox.kind=session"]}),
            },
            timeout=self.timeout_seconds,
        )
        if not response.is_success:
            return ()
        payload = response.json()
        if not isinstance(payload, list):
            return ()
        containers: list[DockerSessionContainer] = []
        for entry in payload:
            if not isinstance(entry, dict):
                continue
            labels = entry.get("Labels") or {}
            container_id = entry.get("Id")
            if not isinstance(container_id, str) or not isinstance(labels, dict):
                continue
            containers.append(
                DockerSessionContainer(
                    container_id=container_id,
                    session_id=str(labels.get("kortny.sandbox.session", "")),
                    task_id=str(labels.get("kortny.sandbox.task", "")),
                    created_at_epoch=int(entry.get("Created") or 0),
                    running=entry.get("State") == "running",
                )
            )
        return tuple(containers)

    def remove_session_container(self, container_id: str) -> str | None:
        """Force-remove one session container; return an error string if any."""

        base_url = _docker_host_base_url(self.docker_host)
        return _remove_container(base_url=base_url, container_id=container_id)


def _exec_start_stream(
    *,
    base_url: str,
    exec_id: str,
    timeout_seconds: float,
    max_output_bytes: int,
) -> tuple[bytes, bool]:
    """Start an exec and collect its multiplexed output, bounded."""

    chunks: list[bytes] = []
    collected = 0
    truncated = False
    with httpx.stream(
        "POST",
        f"{base_url}/exec/{exec_id}/start",
        json={"Detach": False, "Tty": False},
        timeout=timeout_seconds,
    ) as response:
        response.raise_for_status()
        for chunk in response.iter_bytes():
            if collected >= max_output_bytes:
                truncated = True
                break
            if collected + len(chunk) > max_output_bytes:
                chunk = chunk[: max_output_bytes - collected]
                truncated = True
            chunks.append(chunk)
            collected += len(chunk)
            if truncated:
                break
    return b"".join(chunks), truncated


def _demux_docker_stream(raw: bytes) -> tuple[str, str]:
    """Split a multiplexed Docker attach stream into stdout and stderr text."""

    stdout = bytearray()
    stderr = bytearray()
    offset = 0
    total = len(raw)
    while offset + 8 <= total:
        stream_type = raw[offset]
        padding = raw[offset + 1 : offset + 4]
        if stream_type not in (0, 1, 2) or padding != b"\x00\x00\x00":
            # Not a multiplexed stream (TTY mode); treat the rest as stdout.
            stdout.extend(raw[offset:])
            offset = total
            break
        size = int.from_bytes(raw[offset + 4 : offset + 8], "big")
        frame_end = min(offset + 8 + size, total)
        frame = raw[offset + 8 : frame_end]
        if stream_type == 2:
            stderr.extend(frame)
        else:
            stdout.extend(frame)
        offset += 8 + size
    if offset < total:
        stdout.extend(raw[offset:])
    return (
        stdout.decode("utf-8", errors="replace"),
        stderr.decode("utf-8", errors="replace"),
    )


def _exec_status(exit_code: int | None) -> str:
    if exit_code == 0:
        return "succeeded"
    if exit_code == 124:
        return "timed_out"
    return "failed"


def _session_create_payload(spec: DockerSessionCreateSpec) -> JsonObject:
    # The workspace is an anonymous volume, not tmpfs: the Docker archive
    # API (file read/write/export) cannot see tmpfs mounts, which exist only
    # in the container's mount namespace. The volume is removed with the
    # container (DELETE ?v=1). Disk quota is advisory until volume quotas
    # are supported; memory/CPU/pids limits still apply.
    tmp_tmpfs = "rw,nosuid,nodev,size=268435456,mode=1777"
    memory_bytes = spec.memory_mb * 1024 * 1024
    return {
        "Image": spec.image,
        "Cmd": ["sleep", "infinity"],
        "Env": [
            f"{key}={value}"
            for key, value in sorted(
                {**spec.env, "HOME": f"{spec.workspace_path}/home"}.items()
            )
        ],
        "WorkingDir": spec.workspace_path,
        "Tty": False,
        "OpenStdin": False,
        "Volumes": {spec.workspace_path: {}},
        "Labels": {
            "kortny.sandbox": "true",
            "kortny.sandbox.kind": "session",
            "kortny.sandbox.session": spec.session_id,
            "kortny.sandbox.task": spec.task_id,
            "kortny.sandbox.profile": spec.profile,
        },
        "HostConfig": {
            "AutoRemove": False,
            "Binds": [],
            "CapDrop": ["ALL"],
            "Init": True,
            "IpcMode": "private",
            "Memory": memory_bytes,
            "MemorySwap": memory_bytes,
            "NanoCpus": _nano_cpus(spec.cpus),
            "NetworkMode": "none",
            "PidsLimit": spec.pids_limit,
            "Privileged": False,
            "ReadonlyRootfs": True,
            "SecurityOpt": ["no-new-privileges"],
            "Tmpfs": {
                "/tmp": tmp_tmpfs,
            },
        },
    }


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
