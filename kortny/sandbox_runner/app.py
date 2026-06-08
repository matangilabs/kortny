"""Internal FastAPI app for sandbox-runner health and smoke checks."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field, model_validator

from kortny.execution import SandboxResourceLimits, ToolSandboxPolicy
from kortny.sandbox_runner.docker_api import DockerApiClient, DockerApiProbeClient

SERVICE_NAME = "kortny-sandbox-runner"


@dataclass(frozen=True, slots=True)
class SandboxRunnerSettings:
    """Runtime settings for the sandbox runner control plane."""

    runner_name: str = SERVICE_NAME
    docker_host: str = ""
    execution_enabled: bool = False
    default_image: str = "kortny/sandbox-python:latest"
    default_network: str = "none"
    default_cpus: float = 1.0
    default_memory_mb: int = 512
    default_pids_limit: int = 128
    default_timeout_seconds: int = 60

    @property
    def docker_host_configured(self) -> bool:
        """Whether this runner has a Docker endpoint configured."""

        return bool(self.docker_host.strip())

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
            reason="Sandbox-runner profile smoke check; execution is disabled in this slice.",
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
    """Validated worker-facing sandbox run request.

    This contract is intentionally non-executing until a later slice enables
    container creation behind `KORTNY_SANDBOX_EXECUTION_ENABLED`.
    """

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
            "kortny/sandbox-python:latest",
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
    )


def create_app(
    settings: SandboxRunnerSettings | None = None,
    docker_client: DockerApiProbeClient | None = None,
) -> FastAPI:
    """Create the internal sandbox-runner control-plane app."""

    resolved_settings = settings or load_sandbox_runner_settings()
    resolved_docker_client = docker_client or DockerApiClient(
        docker_host=resolved_settings.docker_host
    )
    app = FastAPI(title="Kortny Sandbox Runner", docs_url=None, redoc_url=None)
    app.state.sandbox_runner_settings = resolved_settings
    app.state.docker_client = resolved_docker_client

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
        status = (
            "execution_not_implemented"
            if resolved_settings.execution_enabled
            else "execution_disabled"
        )
        reason = (
            "Sandbox execution is enabled in settings, but container launch is "
            "not implemented in this slice."
            if resolved_settings.execution_enabled
            else (
                "Sandbox execution is not enabled. This endpoint currently "
                "validates the request contract only."
            )
        )
        return {
            "ok": False,
            "service": SERVICE_NAME,
            "runner": resolved_settings.runner_name,
            "execution_enabled": resolved_settings.execution_enabled,
            "execution_attempted": False,
            "status": status,
            "reason": reason,
            "request": request.redacted_payload(),
        }

    return app


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
