"""Sandbox-backed code execution tool."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from kortny.db.models import TaskEventType
from kortny.execution import (
    SandboxEventRecorder,
    SandboxResourceLimits,
    SandboxRunner,
    SandboxSpec,
    SandboxUnavailableError,
)
from kortny.tools.types import JsonObject, JsonSchema, ToolResult

DEFAULT_CODE_EXEC_TIMEOUT_SECONDS = 30
MAX_CODE_EXEC_TIMEOUT_SECONDS = 60
MAX_CODE_EXEC_CHARS = 20_000


class TaskEventSink(Protocol):
    """Subset of TaskService needed for sandbox event recording."""

    def append_event(
        self,
        task: Any,
        event_type: TaskEventType | str,
        payload: dict[str, Any] | None = None,
    ) -> object:
        """Append an event for a task."""


class CodeExecTool:
    """Execute short Python snippets in Kortny's per-task sandbox runner."""

    name = "code_exec"
    description = (
        "Executes a short Python 3 snippet in Kortny's isolated sandbox runner. "
        "Use only when the user explicitly asks to run code, verify a calculation, "
        "or test a tiny script. No network, package installation, secrets, or "
        "host filesystem access are available."
    )
    parameters: JsonSchema = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": (
                    "Python 3 code to execute. Keep it self-contained and do not "
                    "include secrets or network calls."
                ),
            },
            "language": {
                "type": "string",
                "enum": ["python"],
                "default": "python",
                "description": "Execution language. Only python is supported.",
            },
            "timeout_seconds": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_CODE_EXEC_TIMEOUT_SECONDS,
                "default": DEFAULT_CODE_EXEC_TIMEOUT_SECONDS,
                "description": "Wall-clock timeout for the sandboxed run.",
            },
        },
        "required": ["code"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        *,
        runner: SandboxRunner | None,
        image: str,
        task: Any | None = None,
        task_service: TaskEventSink | None = None,
        max_code_chars: int = MAX_CODE_EXEC_CHARS,
    ) -> None:
        if not image.strip():
            raise ValueError("Sandbox image is required for code_exec")
        if max_code_chars < 1:
            raise ValueError("max_code_chars must be positive")
        if (task is None) != (task_service is None):
            raise ValueError("task and task_service must be provided together")

        self.runner = runner
        self.image = image.strip()
        self.task = task
        self.task_service = task_service
        self.max_code_chars = max_code_chars

    def invoke(self, args: JsonObject) -> ToolResult:
        code = _required_code(args, max_code_chars=self.max_code_chars)
        language = _language(args)
        timeout_seconds = _timeout_seconds(args)

        if language != "python":
            raise ValueError("code_exec only supports language='python'")
        if self.runner is None:
            return _recoverable_error_result(
                code="sandbox_service_unavailable",
                message="The code execution sandbox is not configured for this worker.",
                hint=(
                    "Start the sandbox profile and set KORTNY_SANDBOX_RUNNER_URL "
                    "before asking Kortny to run code."
                ),
                details={"execution_attempted": False},
            )

        spec = SandboxSpec(
            image=self.image,
            command=("python", "-c", code),
            workspace_path=_workspace_path(self.task),
            network="none",
            resource_limits=SandboxResourceLimits(
                cpus=1.0,
                memory_mb=512,
                pids_limit=64,
                timeout_seconds=timeout_seconds,
            ),
        )
        try:
            result = self.runner.run(spec)
        except SandboxUnavailableError as exc:
            return _recoverable_error_result(
                code="sandbox_service_unavailable",
                message=str(exc),
                hint="Check that the sandbox-runner container is healthy.",
                details={"execution_attempted": False},
            )

        self._record_sandbox_events(spec=spec, result=result)
        output: JsonObject = {
            "successful": result.exit_code == 0,
            "language": language,
            "exit_code": result.exit_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "artifact_count": len(result.artifacts),
            "usage": result.usage,
        }
        if result.exit_code != 0:
            output["error"] = {
                "code": "sandbox_execution_failed",
                "message": f"Sandboxed code exited with status {result.exit_code}.",
                "recoverable": True,
                "details": {
                    "exit_code": result.exit_code,
                    "stderr_chars": len(result.stderr),
                    "stdout_chars": len(result.stdout),
                },
            }
        return ToolResult(output=output)

    def _record_sandbox_events(
        self,
        *,
        spec: SandboxSpec,
        result: Any,
    ) -> None:
        if self.task is None or self.task_service is None:
            return
        recorder = SandboxEventRecorder(
            event_sink=self.task_service,
            runner="sandbox-runner",
        )
        for event in result.events:
            recorder.record_lifecycle(
                self.task,
                event,
                spec=spec,
                tool_name=self.name,
            )
        recorder.record_result(
            self.task,
            result,
            spec=spec,
            tool_name=self.name,
        )


def _required_code(args: JsonObject, *, max_code_chars: int) -> str:
    value = args.get("code")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("code_exec requires a non-empty string 'code' argument")
    code = value.strip()
    if len(code) > max_code_chars:
        raise ValueError(f"code_exec 'code' must be at most {max_code_chars} chars")
    return code


def _language(args: JsonObject) -> str:
    value = args.get("language", "python")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("code_exec 'language' must be a non-empty string")
    return value.strip().casefold()


def _timeout_seconds(args: JsonObject) -> int:
    value = args.get("timeout_seconds", DEFAULT_CODE_EXEC_TIMEOUT_SECONDS)
    if not isinstance(value, int):
        raise ValueError("code_exec 'timeout_seconds' must be an integer")
    if value < 1 or value > MAX_CODE_EXEC_TIMEOUT_SECONDS:
        raise ValueError(
            "code_exec 'timeout_seconds' must be between 1 and "
            f"{MAX_CODE_EXEC_TIMEOUT_SECONDS}"
        )
    return value


def _workspace_path(task: Any | None) -> Path:
    task_id = getattr(task, "id", None)
    suffix = str(task_id) if task_id is not None else "ad-hoc"
    return Path("/workspace") / f"code-exec-{suffix}"


def _recoverable_error_result(
    *,
    code: str,
    message: str,
    hint: str,
    details: JsonObject,
) -> ToolResult:
    return ToolResult(
        output={
            "successful": False,
            "error": {
                "code": code,
                "message": message,
                "hint": hint,
                "recoverable": True,
                "details": details,
            },
        }
    )
