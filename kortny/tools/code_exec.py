"""Sandbox-backed code execution tools (code_exec and codeact_exec placeholder)."""

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
                    "Ensure the sandbox-runner service is healthy and "
                    "KORTNY_SANDBOX_RUNNER_URL is configured before asking "
                    "Kortny to run code."
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


# ---------------------------------------------------------------------------
# CodeActExecTool — public schema placeholder for codeact_exec (HIG-301 B)
# ---------------------------------------------------------------------------
# This class defines the tool's NAME, DESCRIPTION, and PARAMETERS schema —
# the parts the LLM sees.  The actual execution is coordinator-owned (see
# AgentCoordinator._handle_codeact_exec) because it needs the approval policy,
# registry, and task context that a standalone tool.invoke() cannot access.
#
# invoke() is intentionally unreachable in production: the coordinator
# intercepts codeact_exec before the registry.invoke() path.  If somehow
# called directly (e.g. in a unit test for the stub alone), it returns a
# structured error rather than silently succeeding.
# ---------------------------------------------------------------------------


class CodeActExecTool:
    """Execute model-written Python that calls the task's tools as library functions.

    The model writes a Python script that calls the tools listed in
    ``allowed_tools`` as regular function calls (via the auto-generated
    ``kortny_tools`` library).  The script runs in Kortny's isolated sandbox;
    tool calls are dispatched host-side through the RPC bridge.

    Security notes:
    - Declare every tool the script will call in ``allowed_tools``; calls to
      unlisted tools are rejected before the sandbox starts.
    - The sandbox container has no outbound network (``NetworkMode=none``), so
      the script cannot exfiltrate data via HTTP/DNS.  Its only side-effects
      are the allowlisted tool calls, which are gated by the same approval
      policy as direct tool calls.
    - Secrets (API keys, Slack tokens) never enter the sandbox; the RPC bridge
      dispatches tool calls host-side with credentials.
    - Approval is collected once for the whole script and allowlist (approve-
      once model).  Mid-script approval does not exist in v1.
    - Only the script's final stdout is returned to the LLM; intermediate RPC
      results stay in-process and never enter the message context.
    """

    name = "codeact_exec"
    description = (
        "Execute model-written Python that calls the task's tools as library functions. "
        "Write a self-contained script that imports from kortny_tools and calls the "
        "functions listed in allowed_tools. Only stdout is returned to you; "
        "intermediate tool results are dispatched host-side and never pollute context. "
        "Declare every tool the script will call in allowed_tools; calls to unlisted "
        "tools are blocked before the sandbox starts. The sandbox has no outbound "
        "network and no access to secrets or host filesystem."
    )
    parameters: JsonSchema = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": (
                    "Self-contained Python 3 script. Import from kortny_tools to call "
                    "the tools listed in allowed_tools. Print the final result to stdout."
                ),
            },
            "allowed_tools": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Exact runtime names of every tool the script will call "
                    "(e.g. 'composio_linear_list_issues'). The script cannot call "
                    "any tool not listed here."
                ),
            },
            "timeout_seconds": {
                "type": "integer",
                "minimum": 1,
                "maximum": 300,
                "default": 60,
                "description": "Wall-clock timeout for the sandboxed run.",
            },
        },
        "required": ["code", "allowed_tools"],
        "additionalProperties": False,
    }

    def invoke(self, args: JsonObject) -> ToolResult:
        # The coordinator intercepts codeact_exec before the registry.invoke()
        # path, so this method is intentionally unreachable in production.
        # If called directly (e.g. in an integration test), return a
        # structured error rather than a confusing exception.
        return ToolResult(
            output={
                "successful": False,
                "error": {
                    "code": "codeact_exec_not_routed",
                    "message": (
                        "codeact_exec must be dispatched by the coordinator, not "
                        "invoked directly via the tool registry."
                    ),
                    "recoverable": False,
                },
            }
        )
