"""Internal FastAPI app for sandbox-runner health checks and execution."""

from __future__ import annotations

import base64
import binascii
import contextlib
import os
import threading
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Literal

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field, model_validator

from kortny.execution import SandboxResourceLimits, ToolSandboxPolicy
from kortny.sandbox_runner.docker_api import (
    DockerApiClient,
    DockerApiRunnerClient,
    DockerContainerRunSpec,
)
from kortny.sandbox_runner.sessions import (
    SessionConfig,
    SessionDockerError,
    SessionManager,
    SessionNotFoundError,
    SessionPathError,
)

SERVICE_NAME = "kortny-sandbox-runner"
SESSION_REAPER_INTERVAL_SECONDS = 60.0


@dataclass(frozen=True, slots=True)
class SandboxRunnerSettings:
    """Runtime settings for the sandbox runner control plane."""

    runner_name: str = SERVICE_NAME
    docker_host: str = ""
    execution_enabled: bool = False
    default_image: str = "ghcr.io/astral-sh/uv:python3.11-bookworm-slim"
    default_network: str = "none"
    default_cpus: float = 1.0
    default_memory_mb: int = 512
    default_pids_limit: int = 128
    default_timeout_seconds: int = 60
    sessions_enabled: bool = True
    session_cpus: float = 2.0
    session_memory_mb: int = 2048
    session_pids_limit: int = 512
    session_workspace_mb: int = 1024
    session_idle_seconds: int = 1800
    session_max_age_seconds: int = 14400
    session_exec_max_timeout_seconds: int = 300

    @property
    def docker_host_configured(self) -> bool:
        """Whether this runner has a Docker endpoint configured."""

        return bool(self.docker_host.strip())

    def session_config(self) -> SessionConfig:
        """Return the session-container configuration for this runner."""

        return SessionConfig(
            image=self.default_image,
            cpus=self.session_cpus,
            memory_mb=self.session_memory_mb,
            pids_limit=self.session_pids_limit,
            workspace_mb=self.session_workspace_mb,
            idle_seconds=self.session_idle_seconds,
            max_age_seconds=self.session_max_age_seconds,
            exec_max_timeout_seconds=self.session_exec_max_timeout_seconds,
        )

    def default_policy(self) -> ToolSandboxPolicy:
        """Return the default future execution policy advertised by smoke checks."""

        return ToolSandboxPolicy(
            requires_sandbox=True,
            profile="default",
            network="none",
            resource_limits=SandboxResourceLimits(
                cpus=self.default_cpus,
                memory_mb=self.default_memory_mb,
                pids_limit=self.default_pids_limit,
                timeout_seconds=self.default_timeout_seconds,
            ),
            reason="Default sandbox-runner policy for isolated code execution.",
        )


class SandboxSmokeRequest(BaseModel):
    """Smoke-test request that does not execute user code."""

    message: str = Field(default="ping", max_length=200)


class SandboxResourceLimitsRequest(BaseModel):
    """Resource limit shape accepted by the worker-facing run contract."""

    cpus: float = Field(default=1.0, gt=0)
    memory_mb: int = Field(default=512, gt=0)
    pids_limit: int = Field(default=128, gt=0)
    timeout_seconds: int = Field(default=60, gt=0)

    def to_contract(self) -> SandboxResourceLimits:
        return SandboxResourceLimits(
            cpus=self.cpus,
            memory_mb=self.memory_mb,
            pids_limit=self.pids_limit,
            timeout_seconds=self.timeout_seconds,
        )


class SandboxRunRequest(BaseModel):
    """Validated worker-facing sandbox run request."""

    image: str = Field(min_length=1, max_length=256)
    command: list[str] = Field(min_length=1, max_length=64)
    workspace_path: str = Field(min_length=1, max_length=1024)
    artifacts_path: str | None = Field(default=None, max_length=1024)
    network: Literal["none", "allowlist"] = "none"
    egress_allowlist: list[str] = Field(default_factory=list, max_length=64)
    env: dict[str, str] = Field(default_factory=dict, max_length=64)
    resource_limits: SandboxResourceLimitsRequest = Field(
        default_factory=SandboxResourceLimitsRequest
    )

    @model_validator(mode="after")
    def validate_run_shape(self) -> SandboxRunRequest:
        if any(not part.strip() for part in self.command):
            raise ValueError("command entries must be non-empty")
        if not _is_safe_workspace_path(self.workspace_path):
            raise ValueError("workspace_path must be /workspace or a child path")
        if self.artifacts_path is not None and not _is_safe_workspace_path(
            self.artifacts_path
        ):
            raise ValueError("artifacts_path must be /workspace or a child path")
        if self.artifacts_path is not None and not _is_child_path(
            child=self.artifacts_path,
            parent=self.workspace_path,
        ):
            raise ValueError("artifacts_path must be inside workspace_path")
        if any(not host.strip() for host in self.egress_allowlist):
            raise ValueError("egress_allowlist entries must be non-empty")
        if self.network == "allowlist" and not self.egress_allowlist:
            raise ValueError("allowlist network requires egress hosts")
        if self.network == "none" and self.egress_allowlist:
            raise ValueError("egress_allowlist requires allowlist network")
        if any(not key.strip() for key in self.env):
            raise ValueError("env keys must be non-empty")
        return self

    def redacted_payload(self) -> dict[str, object]:
        """Return a JSON-safe request summary without env values."""

        return {
            "image": self.image,
            "command": self.command,
            "workspace_path": self.workspace_path,
            "artifacts_path": self.artifacts_path,
            "network": self.network,
            "egress_allowlist": self.egress_allowlist,
            "env_keys": sorted(self.env),
            "resource_limits": self.resource_limits.to_contract().to_payload(),
        }

    def to_run_spec(self) -> DockerContainerRunSpec:
        """Return the Docker runner spec for an already-validated request."""

        return DockerContainerRunSpec(
            image=self.image,
            command=tuple(self.command),
            workspace_path=self.workspace_path,
            env=dict(self.env),
            resource_limits=self.resource_limits.to_contract(),
        )


class SessionCreateRequest(BaseModel):
    """Worker-facing request to open (or reuse) a task session."""

    task_id: str = Field(min_length=1, max_length=128)
    profile: str = Field(default="workbench", min_length=1, max_length=64)


class SessionExecRequest(BaseModel):
    """Worker-facing request to run one command in a session."""

    command: str = Field(min_length=1, max_length=20_000)
    workdir: str = Field(default="/workspace", min_length=1, max_length=1024)
    timeout_seconds: int = Field(default=120, ge=1, le=3600)


class SessionFileWriteRequest(BaseModel):
    """Worker-facing request to write one file into a session workspace."""

    path: str = Field(min_length=1, max_length=1024)
    content_b64: str = Field(max_length=8_000_000)


def load_sandbox_runner_settings(
    env: Mapping[str, str] | None = None,
) -> SandboxRunnerSettings:
    """Load sandbox-runner settings from environment variables."""

    source = env or os.environ
    return SandboxRunnerSettings(
        runner_name=source.get("KORTNY_SANDBOX_RUNNER_NAME", SERVICE_NAME),
        docker_host=source.get("DOCKER_HOST", ""),
        execution_enabled=_env_bool(
            source.get("KORTNY_SANDBOX_EXECUTION_ENABLED"),
            default=False,
        ),
        default_image=source.get(
            "KORTNY_SANDBOX_DEFAULT_IMAGE",
            "ghcr.io/astral-sh/uv:python3.11-bookworm-slim",
        ),
        default_network=source.get("KORTNY_SANDBOX_DEFAULT_NETWORK", "none"),
        default_cpus=_env_float(source.get("KORTNY_SANDBOX_CPUS"), default=1.0),
        default_memory_mb=_env_int(
            source.get("KORTNY_SANDBOX_MEMORY_MB"),
            default=512,
        ),
        default_pids_limit=_env_int(
            source.get("KORTNY_SANDBOX_PIDS_LIMIT"),
            default=128,
        ),
        default_timeout_seconds=_env_int(
            source.get("KORTNY_SANDBOX_TIMEOUT_SECONDS"),
            default=60,
        ),
        sessions_enabled=_env_bool(
            source.get("KORTNY_SANDBOX_SESSIONS_ENABLED"),
            default=True,
        ),
        session_cpus=_env_float(
            source.get("KORTNY_SANDBOX_SESSION_CPUS"),
            default=2.0,
        ),
        session_memory_mb=_env_int(
            source.get("KORTNY_SANDBOX_SESSION_MEMORY_MB"),
            default=2048,
        ),
        session_pids_limit=_env_int(
            source.get("KORTNY_SANDBOX_SESSION_PIDS_LIMIT"),
            default=512,
        ),
        session_workspace_mb=_env_int(
            source.get("KORTNY_SANDBOX_SESSION_WORKSPACE_MB"),
            default=1024,
        ),
        session_idle_seconds=_env_int(
            source.get("KORTNY_SANDBOX_SESSION_IDLE_SECONDS"),
            default=1800,
        ),
        session_max_age_seconds=_env_int(
            source.get("KORTNY_SANDBOX_SESSION_MAX_AGE_SECONDS"),
            default=14400,
        ),
        session_exec_max_timeout_seconds=_env_int(
            source.get("KORTNY_SANDBOX_SESSION_EXEC_MAX_TIMEOUT_SECONDS"),
            default=300,
        ),
    )


def create_app(
    settings: SandboxRunnerSettings | None = None,
    docker_client: DockerApiRunnerClient | None = None,
    session_manager: SessionManager | None = None,
) -> FastAPI:
    """Create the internal sandbox-runner control-plane app."""

    resolved_settings = settings or load_sandbox_runner_settings()
    resolved_docker_client = docker_client or DockerApiClient(
        docker_host=resolved_settings.docker_host
    )
    resolved_session_manager = session_manager
    if resolved_session_manager is None and hasattr(
        resolved_docker_client, "create_session_container"
    ):
        resolved_session_manager = SessionManager(
            docker_client=resolved_docker_client,  # type: ignore[arg-type]
            config=resolved_settings.session_config(),
        )

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        stop_event = threading.Event()
        reaper: threading.Thread | None = None
        if (
            resolved_session_manager is not None
            and resolved_settings.execution_enabled
            and resolved_settings.sessions_enabled
            and resolved_settings.docker_host_configured
        ):
            reaper = threading.Thread(
                target=_session_reaper_loop,
                args=(resolved_session_manager, stop_event),
                name="sandbox-session-reaper",
                daemon=True,
            )
            reaper.start()
        try:
            yield
        finally:
            stop_event.set()
            if reaper is not None:
                reaper.join(timeout=2)

    app = FastAPI(
        title="Kortny Sandbox Runner",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )
    app.state.sandbox_runner_settings = resolved_settings
    app.state.docker_client = resolved_docker_client
    app.state.session_manager = resolved_session_manager

    @app.get("/health")
    def health() -> dict[str, object]:
        return {
            "ok": True,
            "service": SERVICE_NAME,
            "runner": resolved_settings.runner_name,
            "docker_host_configured": resolved_settings.docker_host_configured,
            "execution_enabled": resolved_settings.execution_enabled,
            "mode": "control_plane_smoke",
        }

    @app.post("/smoke")
    def smoke(request: SandboxSmokeRequest) -> dict[str, object]:
        policy = resolved_settings.default_policy()
        return {
            "ok": True,
            "service": SERVICE_NAME,
            "runner": resolved_settings.runner_name,
            "message": request.message,
            "execution_enabled": resolved_settings.execution_enabled,
            "execution_attempted": False,
            "default_image": resolved_settings.default_image,
            "default_network": resolved_settings.default_network,
            "sandbox_policy": policy.to_payload(),
        }

    @app.get("/docker-smoke")
    def docker_smoke() -> dict[str, object]:
        probe = resolved_docker_client.version()
        return {
            "ok": probe.ok,
            "service": SERVICE_NAME,
            "runner": resolved_settings.runner_name,
            "docker_host_configured": resolved_settings.docker_host_configured,
            "execution_enabled": resolved_settings.execution_enabled,
            "execution_attempted": False,
            "docker_api": probe.to_payload(),
        }

    @app.post("/run")
    def run(request: SandboxRunRequest) -> dict[str, object]:
        if not resolved_settings.execution_enabled:
            return {
                "ok": False,
                "service": SERVICE_NAME,
                "runner": resolved_settings.runner_name,
                "execution_enabled": False,
                "execution_attempted": False,
                "status": "execution_disabled",
                "reason": ("Sandbox execution is not enabled for this runner."),
                "request": request.redacted_payload(),
            }
        if request.image != resolved_settings.default_image:
            return _rejected_run_payload(
                settings=resolved_settings,
                request=request,
                status="image_not_allowed",
                reason="Sandbox execution only allows the configured default image.",
            )
        if request.network != "none":
            return _rejected_run_payload(
                settings=resolved_settings,
                request=request,
                status="network_not_implemented",
                reason="Sandbox execution currently supports only network: none.",
            )

        result = resolved_docker_client.run_container(request.to_run_spec())
        return {
            "ok": result.ok,
            "service": SERVICE_NAME,
            "runner": resolved_settings.runner_name,
            "execution_enabled": resolved_settings.execution_enabled,
            "execution_attempted": result.execution_attempted,
            "status": result.status,
            "request": request.redacted_payload(),
            "result": result.to_payload(),
        }

    def _require_session_manager() -> SessionManager:
        if not resolved_settings.execution_enabled:
            raise HTTPException(
                status_code=503,
                detail="Sandbox execution is not enabled for this runner.",
            )
        if resolved_session_manager is None or not resolved_settings.sessions_enabled:
            raise HTTPException(
                status_code=503,
                detail="Sandbox sessions are not enabled for this runner.",
            )
        return resolved_session_manager

    @app.post("/sessions")
    def create_session(request: SessionCreateRequest) -> dict[str, object]:
        manager = _require_session_manager()
        try:
            info = manager.create_or_get(request.task_id, request.profile)
        except SessionDockerError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return {"ok": True, "service": SERVICE_NAME, **info.to_payload()}

    @app.post("/sessions/{session_id}/exec")
    def session_exec(session_id: str, request: SessionExecRequest) -> dict[str, object]:
        manager = _require_session_manager()
        try:
            result = manager.exec(
                session_id,
                request.command,
                workdir=request.workdir,
                timeout_seconds=request.timeout_seconds,
            )
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Unknown session.") from exc
        except SessionPathError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"ok": result.ok, "service": SERVICE_NAME, **result.to_payload()}

    @app.put("/sessions/{session_id}/files")
    def session_write_file(
        session_id: str, request: SessionFileWriteRequest
    ) -> dict[str, object]:
        manager = _require_session_manager()
        try:
            content = base64.b64decode(request.content_b64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise HTTPException(
                status_code=422, detail="content_b64 is not valid base64."
            ) from exc
        try:
            written = manager.write_file(session_id, request.path, content)
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Unknown session.") from exc
        except SessionPathError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except SessionDockerError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return {
            "ok": True,
            "service": SERVICE_NAME,
            "path": request.path,
            "size_bytes": written,
        }

    @app.get("/sessions/{session_id}/files")
    def session_read_file(session_id: str, path: str) -> dict[str, object]:
        manager = _require_session_manager()
        try:
            content = manager.read_file(session_id, path)
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Unknown session.") from exc
        except SessionPathError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except SessionDockerError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return {
            "ok": True,
            "service": SERVICE_NAME,
            "path": path,
            "size_bytes": len(content),
            "content_b64": base64.b64encode(content).decode("ascii"),
        }

    @app.get("/sessions/{session_id}/archive")
    def session_archive(session_id: str, path: str) -> Response:
        manager = _require_session_manager()
        try:
            tar_bytes = manager.export_archive(session_id, path)
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Unknown session.") from exc
        except SessionPathError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except SessionDockerError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return Response(content=tar_bytes, media_type="application/x-tar")

    @app.delete("/sessions/{session_id}")
    def session_close(session_id: str) -> dict[str, object]:
        manager = _require_session_manager()
        try:
            manager.close(session_id)
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Unknown session.") from exc
        return {"ok": True, "service": SERVICE_NAME, "closed": session_id}

    return app


def _session_reaper_loop(manager: SessionManager, stop_event: threading.Event) -> None:
    while not stop_event.wait(SESSION_REAPER_INTERVAL_SECONDS):
        try:
            manager.reap()
        except Exception:  # noqa: BLE001 - reaper must survive Docker hiccups
            continue


def _rejected_run_payload(
    *,
    settings: SandboxRunnerSettings,
    request: SandboxRunRequest,
    status: str,
    reason: str,
) -> dict[str, object]:
    return {
        "ok": False,
        "service": SERVICE_NAME,
        "runner": settings.runner_name,
        "execution_enabled": settings.execution_enabled,
        "execution_attempted": False,
        "status": status,
        "reason": reason,
        "request": request.redacted_payload(),
    }


def _is_safe_workspace_path(value: str) -> bool:
    if value != "/workspace" and not value.startswith("/workspace/"):
        return False
    return ".." not in PurePosixPath(value).parts


def _is_child_path(*, child: str, parent: str) -> bool:
    normalized_parent = parent.rstrip("/")
    return child == normalized_parent or child.startswith(f"{normalized_parent}/")


def _env_bool(value: str | None, *, default: bool) -> bool:
    if value is None or not value.strip():
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(value: str | None, *, default: int) -> int:
    if value is None or not value.strip():
        return default
    return int(value)


def _env_float(value: str | None, *, default: float) -> float:
    if value is None or not value.strip():
        return default
    return float(value)
