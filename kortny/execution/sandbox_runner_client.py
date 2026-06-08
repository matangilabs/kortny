"""HTTP client for the sandbox-runner service."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from kortny.config import Settings
from kortny.execution.sandbox import (
    SandboxLifecycleEvent,
    SandboxResult,
    SandboxRunner,
    SandboxSpec,
    SandboxUnavailableError,
)

JsonObject = dict[str, object]


@dataclass(frozen=True, slots=True)
class HttpSandboxRunner(SandboxRunner):
    """Run sandbox specs through the internal sandbox-runner HTTP API."""

    base_url: str
    timeout_seconds: float = 70.0
    http_client: httpx.Client | None = None

    def __post_init__(self) -> None:
        base_url = self.base_url.strip().rstrip("/")
        if not base_url:
            raise ValueError("Sandbox runner URL is required")
        if self.timeout_seconds <= 0:
            raise ValueError("Sandbox runner timeout must be positive")
        object.__setattr__(self, "base_url", base_url)

    def run(self, spec: SandboxSpec) -> SandboxResult:
        """Execute a sandbox spec through `POST /run`."""

        try:
            response = self._post(spec)
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPError as exc:
            raise SandboxUnavailableError(
                f"Sandbox runner request failed: {type(exc).__name__}: {exc}"
            ) from exc
        except ValueError as exc:
            raise SandboxUnavailableError(
                "Sandbox runner returned invalid JSON."
            ) from exc

        if not isinstance(payload, dict):
            raise SandboxUnavailableError(
                "Sandbox runner returned a non-object payload."
            )

        result_payload = payload.get("result")
        if not isinstance(result_payload, dict):
            raise SandboxUnavailableError(_runner_unavailable_message(payload))

        return _sandbox_result_from_payload(payload, result_payload)

    def _post(self, spec: SandboxSpec) -> httpx.Response:
        client = self.http_client or httpx.Client(timeout=self.timeout_seconds)
        return client.post(
            f"{self.base_url}/run",
            json=_spec_request_payload(spec),
            timeout=self.timeout_seconds,
        )


def create_sandbox_runner_from_settings(settings: Settings) -> SandboxRunner | None:
    """Return the configured sandbox runner, if enabled for this process."""

    if settings.sandbox_runner_url is None:
        return None
    return HttpSandboxRunner(
        base_url=settings.sandbox_runner_url,
        timeout_seconds=settings.sandbox_runner_timeout_seconds,
    )


def _spec_request_payload(spec: SandboxSpec) -> JsonObject:
    return {
        "image": spec.image,
        "command": list(spec.command),
        "workspace_path": _posix_path(spec.workspace_path),
        "artifacts_path": _posix_path(spec.artifacts_path)
        if spec.artifacts_path is not None
        else None,
        "network": spec.network,
        "egress_allowlist": list(spec.egress_allowlist),
        "env": dict(spec.env),
        "resource_limits": spec.resource_limits.to_payload(),
    }


def _sandbox_result_from_payload(
    payload: dict[str, Any],
    result_payload: dict[str, Any],
) -> SandboxResult:
    exit_code = _result_exit_code(result_payload)
    stdout = _optional_str(result_payload.get("stdout")) or ""
    stderr = _optional_str(result_payload.get("stderr")) or ""
    status = _optional_str(payload.get("status")) or _optional_str(
        result_payload.get("status")
    )
    usage: JsonObject = {
        "runner_status": status,
        "runner_ok": _optional_bool(payload.get("ok")),
        "execution_attempted": _optional_bool(payload.get("execution_attempted")),
        "container_id": _optional_str(result_payload.get("container_id")),
        "duration_ms": _optional_int(result_payload.get("duration_ms")),
        "status_code": _optional_int(result_payload.get("status_code")),
        "cleanup_error": _optional_str(result_payload.get("cleanup_error")),
        "error_type": _optional_str(result_payload.get("error_type")),
        "error": _optional_str(result_payload.get("error")),
    }
    return SandboxResult(
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        usage={key: value for key, value in usage.items() if value is not None},
        events=_lifecycle_events(result_payload, exit_code=exit_code),
    )


def _lifecycle_events(
    result_payload: dict[str, Any],
    *,
    exit_code: int,
) -> tuple[SandboxLifecycleEvent, ...]:
    container_id = _optional_str(result_payload.get("container_id"))
    status = _optional_str(result_payload.get("status"))
    events: list[SandboxLifecycleEvent] = []
    if container_id:
        events.append(
            SandboxLifecycleEvent(
                phase="started",
                message="sandbox runner started container",
                details={"container_id": container_id},
            )
        )
    if status == "timed_out":
        events.append(
            SandboxLifecycleEvent(
                phase="killed",
                message="sandbox runner timed out execution",
                details={"container_id": container_id} if container_id else {},
            )
        )
    else:
        events.append(
            SandboxLifecycleEvent(
                phase="exited",
                message="sandbox runner completed execution",
                details={
                    "container_id": container_id,
                    "exit_code": exit_code,
                }
                if container_id
                else {"exit_code": exit_code},
            )
        )
    return tuple(events)


def _runner_unavailable_message(payload: dict[str, Any]) -> str:
    status = _optional_str(payload.get("status")) or "unavailable"
    reason = _optional_str(payload.get("reason"))
    if reason:
        return f"Sandbox runner unavailable: {status}: {reason}"
    return f"Sandbox runner unavailable: {status}"


def _result_exit_code(result_payload: dict[str, Any]) -> int:
    exit_code = _optional_int(result_payload.get("exit_code"))
    if exit_code is not None:
        return exit_code
    if result_payload.get("status") == "timed_out":
        return 124
    return 1


def _posix_path(value: Path) -> str:
    return value.as_posix()


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) else None


def _optional_bool(value: object) -> bool | None:
    return value if isinstance(value, bool) else None
