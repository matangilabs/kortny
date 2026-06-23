"""Sandbox workbench tools: bash, files, artifact export, and previews.

These tools share one long-lived sandbox session per task. The filesystem
under /workspace persists across calls; shell environment does not, so the
model is told to use absolute paths and re-activate environments per call.
"""

from __future__ import annotations

import hashlib
import mimetypes
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import unquote, urlparse

import httpx
from sqlalchemy.orm import Session

from kortny.db.models import Artifact, TaskEventType
from kortny.execution.preview import (
    UnsafeArchiveError,
    extract_tar_to_dir,
    preview_url,
    safe_slug,
)
from kortny.execution.sandbox import SandboxUnavailableError
from kortny.execution.sandbox_sessions import (
    SandboxExecResult,
    SandboxSessionClient,
    SandboxSessionError,
    SandboxSessionInfo,
)
from kortny.tools.types import (
    JsonObject,
    JsonSchema,
    RecoverableToolError,
    ToolArtifact,
    ToolResult,
)

DEFAULT_BASH_TIMEOUT_SECONDS = 120
MAX_BASH_TIMEOUT_SECONDS = 300
MAX_BASH_COMMAND_CHARS = 20_000
MAX_WRITE_FILE_CHARS = 262_144
SANDBOX_SESSION_MESSAGE = "sandbox_session"
WORKBENCH_STATE_NOTE = (
    "The /workspace filesystem persists across calls in this task; shell "
    "environment variables and the working directory do not. Use absolute "
    "paths under /workspace."
)
SANDBOX_STAGE_FILE_MAX_BYTES = 5 * 1024 * 1024  # 5 MB


class TaskEventSink(Protocol):
    """Subset of TaskService needed for workbench event recording."""

    def append_event(
        self,
        task: Any,
        event_type: TaskEventType | str,
        payload: dict[str, Any] | None = None,
    ) -> object:
        """Append an event for a task."""


@dataclass(slots=True)
class WorkbenchSession:
    """Lazily opens and shares one sandbox session for a task."""

    client: SandboxSessionClient
    task: Any
    task_service: TaskEventSink | None = None
    _info: SandboxSessionInfo | None = field(default=None, init=False)

    def ensure(self) -> SandboxSessionInfo:
        """Open (or reuse) the sandbox session for this task."""

        if self._info is not None:
            return self._info
        task_id = str(getattr(self.task, "id", None) or "ad-hoc")
        info = self.client.open_session(task_id, profile="workbench")
        self._info = info
        if not info.reused and self.task_service is not None and self.task is not None:
            self.task_service.append_event(
                self.task,
                "log",
                {
                    "message": SANDBOX_SESSION_MESSAGE,
                    "source": "execution.sandbox",
                    "phase": "opened",
                    "session_id": info.session_id,
                    "container_id": info.container_id,
                    "profile": info.profile,
                },
            )
        return info


class _WorkbenchToolBase:
    """Shared session plumbing for all workbench tools."""

    def __init__(self, workbench: WorkbenchSession) -> None:
        self.workbench = workbench

    def _session(self) -> SandboxSessionInfo:
        return self.workbench.ensure()


class SandboxBashTool(_WorkbenchToolBase):
    """Run one shell command in the task's persistent sandbox workspace."""

    name = "sandbox_bash"
    description = (
        "Runs a shell command in this task's isolated sandbox workspace. Use it "
        "to build, test, transform data, and verify outputs instead of guessing "
        "results. " + WORKBENCH_STATE_NOTE + " No network access is available; "
        "rely on the Python 3.11 + uv toolchain already in the image."
    )
    parameters: JsonSchema = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Shell command to run with /bin/sh -lc.",
            },
            "workdir": {
                "type": "string",
                "description": "Working directory (must be under /workspace).",
                "default": "/workspace",
            },
            "timeout_seconds": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_BASH_TIMEOUT_SECONDS,
                "default": DEFAULT_BASH_TIMEOUT_SECONDS,
                "description": "Wall-clock timeout for the command.",
            },
        },
        "required": ["command"],
        "additionalProperties": False,
    }

    def invoke(self, args: JsonObject) -> ToolResult:
        command = _required_string(args, "command", max_chars=MAX_BASH_COMMAND_CHARS)
        workdir = _optional_string(args, "workdir") or "/workspace"
        timeout_seconds = _timeout_seconds(args)
        try:
            session = self._session()
            result = self.workbench.client.exec(
                session.session_id,
                command,
                workdir=workdir,
                timeout_seconds=timeout_seconds,
            )
        except (SandboxUnavailableError, SandboxSessionError) as exc:
            return _sandbox_error_result(exc)
        return ToolResult(output=_exec_output(result))


class SandboxWriteFileTool(_WorkbenchToolBase):
    """Create or overwrite one file in the sandbox workspace."""

    name = "sandbox_write_file"
    description = (
        "Creates or overwrites one text file in this task's sandbox workspace. "
        "Use it to materialize source code, HTML, configs, or data files "
        "before running them with sandbox_bash. " + WORKBENCH_STATE_NOTE
    )
    parameters: JsonSchema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute file path under /workspace.",
            },
            "content": {
                "type": "string",
                "description": "Full file content (UTF-8 text).",
            },
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    }

    def invoke(self, args: JsonObject) -> ToolResult:
        path = _required_string(args, "path", max_chars=1024)
        content = args.get("content")
        if not isinstance(content, str):
            raise ValueError("sandbox_write_file requires a string 'content'")
        if len(content) > MAX_WRITE_FILE_CHARS:
            raise ValueError(
                f"sandbox_write_file 'content' must be at most "
                f"{MAX_WRITE_FILE_CHARS} chars"
            )
        try:
            session = self._session()
            written = self.workbench.client.write_file(
                session.session_id, path, content.encode("utf-8")
            )
        except (SandboxUnavailableError, SandboxSessionError) as exc:
            return _sandbox_error_result(exc)
        return ToolResult(
            output={"successful": True, "path": path, "size_bytes": written}
        )


class SandboxStageFileTool(_WorkbenchToolBase):
    """Download a Slack file's bytes and stage them into the sandbox workspace."""

    name = "sandbox_stage_file"
    description = (
        "Downloads a Slack file's bytes (binary-safe) and stages them directly "
        "into this task's sandbox workspace. Use this before editing uploaded "
        "files (e.g. .docx) with a skill script — unlike slack_file_read, this "
        "preserves the binary format so python-docx and similar tools can open it."
    )
    parameters: JsonSchema = {
        "type": "object",
        "properties": {
            "file_id": {
                "type": "string",
                "description": "Slack file ID (e.g. F123ABC). Provide this or file_url.",
            },
            "file_url": {
                "type": "string",
                "description": "Private Slack download URL. Provide this or file_id.",
            },
            "dest_path": {
                "type": "string",
                "description": (
                    "Destination path under /workspace. "
                    "Defaults to /workspace/original.<ext> derived from the filename."
                ),
            },
        },
        "required": [],
        "additionalProperties": False,
    }

    def __init__(
        self,
        workbench: WorkbenchSession,
        slack_file_client: Any,
        bot_token: str,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        super().__init__(workbench)
        self.slack_file_client = slack_file_client
        self.bot_token = bot_token
        self.timeout = timeout
        self.transport = transport

    def invoke(self, args: JsonObject) -> ToolResult:
        file_id = args.get("file_id")
        file_url = args.get("file_url")

        if not file_id and not file_url:
            raise ValueError(
                "sandbox_stage_file requires either 'file_id' or 'file_url'"
            )
        if file_id and file_url:
            raise ValueError(
                "sandbox_stage_file accepts 'file_id' or 'file_url', not both"
            )

        # Resolve filename and download URL
        filename: str
        download_url: str
        if file_url:
            if not isinstance(file_url, str):
                raise ValueError("'file_url' must be a string")
            filename = unquote(Path(urlparse(file_url).path).name) or "slack-file"
            download_url = file_url
        else:
            if not isinstance(file_id, str):
                raise ValueError("'file_id' must be a string")
            filename, download_url = self._lookup_slack_file(file_id)

        # Determine dest_path
        dest_path_arg = args.get("dest_path")
        if dest_path_arg and isinstance(dest_path_arg, str):
            dest_path = dest_path_arg.strip()
        else:
            ext = Path(filename).suffix
            dest_path = f"original{ext}"

        if not dest_path.startswith("/workspace/"):
            dest_path = f"/workspace/{dest_path.lstrip('/')}"

        # Download binary content
        try:
            with httpx.Client(transport=self.transport, timeout=self.timeout) as client:
                resp = client.get(
                    download_url,
                    headers={"Authorization": f"Bearer {self.bot_token}"},
                    follow_redirects=True,
                )
                resp.raise_for_status()
                content = resp.content
        except httpx.HTTPError as exc:
            raise RecoverableToolError(
                code="slack_file_download_failed",
                message=f"Failed to download file: {exc}",
                hint="Check the file URL and bot token.",
            ) from exc

        # Size guard
        if len(content) > SANDBOX_STAGE_FILE_MAX_BYTES:
            raise RecoverableToolError(
                code="file_too_large_for_sandbox",
                message=(
                    f"File is {len(content)} bytes; limit is "
                    f"{SANDBOX_STAGE_FILE_MAX_BYTES} bytes (5 MB). "
                    "Split or reduce the file."
                ),
                hint="Use a smaller file or split it first.",
            )

        sha256 = hashlib.sha256(content).hexdigest()
        mime_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"

        # Write to sandbox
        try:
            session_info = self.workbench.ensure()
            written = self.workbench.client.write_file(
                session_info.session_id, dest_path, content
            )
        except (SandboxUnavailableError, SandboxSessionError) as exc:
            raise RecoverableToolError(
                code="sandbox_unavailable",
                message=f"Sandbox session error: {exc}",
                hint=(
                    "Check that KORTNY_SANDBOX_RUNNER_URL is set and the "
                    "sandbox-runner container is healthy."
                ),
            ) from exc

        return ToolResult(
            output={
                "path": dest_path,
                "filename": filename,
                "mime_type": mime_type,
                "size_bytes": len(content),
                "sha256": sha256,
                "written_bytes": written,
            }
        )

    def _lookup_slack_file(self, file_id: str) -> tuple[str, str]:
        """Fetch filename + download URL from Slack files.info."""
        from slack_sdk.errors import SlackApiError  # noqa: PLC0415

        try:
            response = self.slack_file_client.files_info(file=file_id)
        except SlackApiError as exc:
            raise RecoverableToolError(
                code="slack_file_lookup_failed",
                message=f"Slack files.info failed: {exc}",
                hint="Check the file_id and bot token scopes.",
            ) from exc
        raw_file = response.get("file") or {}
        name: str = raw_file.get("name") or raw_file.get("title") or file_id
        url: str = raw_file.get("url_private_download") or raw_file.get(
            "url_private", ""
        )
        return name, url


class SandboxReadFileTool(_WorkbenchToolBase):
    """Read one file from the sandbox workspace."""

    name = "sandbox_read_file"
    description = (
        "Reads one file from this task's sandbox workspace and returns its "
        "text content. Use it to inspect generated files, logs, or outputs."
    )
    parameters: JsonSchema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute file path under /workspace.",
            },
            "max_chars": {
                "type": "integer",
                "minimum": 1,
                "maximum": 200_000,
                "default": 50_000,
                "description": "Truncate the returned content to this length.",
            },
        },
        "required": ["path"],
        "additionalProperties": False,
    }

    def invoke(self, args: JsonObject) -> ToolResult:
        path = _required_string(args, "path", max_chars=1024)
        max_chars = args.get("max_chars", 50_000)
        if not isinstance(max_chars, int) or max_chars < 1:
            raise ValueError("sandbox_read_file 'max_chars' must be positive")
        try:
            session = self._session()
            content_bytes = self.workbench.client.read_file(session.session_id, path)
        except (SandboxUnavailableError, SandboxSessionError) as exc:
            return _sandbox_error_result(exc)
        text = content_bytes.decode("utf-8", errors="replace")
        truncated = len(text) > max_chars
        return ToolResult(
            output={
                "successful": True,
                "path": path,
                "size_bytes": len(content_bytes),
                "truncated": truncated,
                "content": text[:max_chars],
            }
        )


class SandboxExportArtifactTool(_WorkbenchToolBase):
    """Export a sandbox file or directory as a task artifact."""

    name = "sandbox_export_artifact"
    description = (
        "Exports one file or directory from the sandbox workspace as a task "
        "artifact that gets delivered to the user in Slack. Directories are "
        "zipped. Use this for finished deliverables: reports, datasets, "
        "charts, or app bundles."
    )
    parameters: JsonSchema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Sandbox path under /workspace to export.",
            },
            "filename": {
                "type": "string",
                "description": "Optional artifact filename override.",
            },
        },
        "required": ["path"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        *,
        workbench: WorkbenchSession,
        working_dir: Path,
        session: Session | None = None,
        task_id: uuid.UUID | None = None,
        task_service: TaskEventSink | None = None,
    ) -> None:
        super().__init__(workbench)
        self.working_dir = working_dir
        self.session = session
        self.task_id = task_id
        self.task_service = task_service

    def invoke(self, args: JsonObject) -> ToolResult:
        path = _required_string(args, "path", max_chars=1024)
        filename_override = _optional_string(args, "filename")
        try:
            session_info = self._session()
            tar_bytes = self.workbench.client.export_archive(
                session_info.session_id, path
            )
        except (SandboxUnavailableError, SandboxSessionError) as exc:
            return _sandbox_error_result(exc)

        staging_dir = Path(self.working_dir) / f"sandbox-export-{uuid.uuid4().hex[:8]}"
        staging_dir.mkdir(parents=True, exist_ok=True)
        staging_dir = staging_dir.resolve()
        try:
            extracted = extract_tar_to_dir(tar_bytes, staging_dir)
        except UnsafeArchiveError as exc:
            return _sandbox_error_result(
                SandboxSessionError(f"Unsafe sandbox archive: {exc}")
            )
        if not extracted:
            return _sandbox_error_result(
                SandboxSessionError(f"No files found at sandbox path {path}")
            )

        artifact_path, filename = _materialize_artifact(
            extracted,
            staging_dir=staging_dir,
            working_dir=Path(self.working_dir),
            source_path=path,
            filename_override=filename_override,
        )
        size_bytes = artifact_path.stat().st_size
        mime_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        artifact = ToolArtifact(
            filename=filename,
            path=str(artifact_path),
            mime_type=mime_type,
            size_bytes=size_bytes,
        )
        artifact_id = self._record_artifact(
            filename=filename,
            path=artifact_path,
            mime_type=mime_type,
            size_bytes=size_bytes,
        )
        return ToolResult(
            output={
                "successful": True,
                "filename": filename,
                "path": str(artifact_path),
                "mime_type": mime_type,
                "size_bytes": size_bytes,
                "artifact_id": artifact_id,
            },
            artifacts=(artifact,),
        )

    def _record_artifact(
        self,
        *,
        filename: str,
        path: Path,
        mime_type: str,
        size_bytes: int,
    ) -> str | None:
        if self.session is None or self.task_id is None:
            return None
        artifact = Artifact(
            task_id=self.task_id,
            filename=filename,
            mime_type=mime_type,
            size_bytes=size_bytes,
            storage_path=str(path),
        )
        self.session.add(artifact)
        self.session.flush()
        if self.task_service is not None:
            self.task_service.append_event(
                self.task_id,
                TaskEventType.artifact_created,
                {
                    "artifact_id": str(artifact.id),
                    "filename": filename,
                    "mime_type": mime_type,
                    "size_bytes": size_bytes,
                    "storage_path": str(path),
                },
            )
        return str(artifact.id)


class SandboxPublishPreviewTool(_WorkbenchToolBase):
    """Publish a static directory from the sandbox at a shareable URL."""

    name = "sandbox_publish_preview"
    description = (
        "Publishes a static site directory (HTML/CSS/JS) from the sandbox "
        "workspace at a shareable preview URL the user can open. The "
        "directory should contain an index.html. Use this to deliver "
        "dashboards, reports, and small web apps."
    )
    parameters: JsonSchema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": (
                    "Sandbox directory under /workspace containing the built "
                    "static site (with index.html)."
                ),
            },
            "slug": {
                "type": "string",
                "description": "Optional short name used in the preview URL.",
            },
        },
        "required": ["path"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        *,
        workbench: WorkbenchSession,
        artifacts_dir: Path,
        public_base_url: str,
        signing_secret: str,
    ) -> None:
        super().__init__(workbench)
        self.artifacts_dir = artifacts_dir
        self.public_base_url = public_base_url
        self.signing_secret = signing_secret

    def invoke(self, args: JsonObject) -> ToolResult:
        path = _required_string(args, "path", max_chars=1024)
        slug = safe_slug(_optional_string(args, "slug") or Path(path).name)
        task_id = str(getattr(self.workbench.task, "id", None) or "ad-hoc")
        try:
            session_info = self._session()
            tar_bytes = self.workbench.client.export_archive(
                session_info.session_id, path
            )
        except (SandboxUnavailableError, SandboxSessionError) as exc:
            return _sandbox_error_result(exc)

        destination = Path(self.artifacts_dir) / task_id / slug
        try:
            extracted = extract_tar_to_dir(tar_bytes, destination)
        except UnsafeArchiveError as exc:
            return _sandbox_error_result(
                SandboxSessionError(f"Unsafe sandbox archive: {exc}")
            )
        if not extracted:
            return _sandbox_error_result(
                SandboxSessionError(f"No files found at sandbox path {path}")
            )

        url = preview_url(self.public_base_url, self.signing_secret, task_id, slug)
        has_index = (destination / "index.html").is_file()
        output: JsonObject = {
            "successful": True,
            "preview_url": url,
            "slug": slug,
            "file_count": len(extracted),
            "has_index_html": has_index,
        }
        if not has_index:
            output["warning"] = (
                "No index.html found at the published root; the preview URL "
                "will 404 until one exists."
            )
        return ToolResult(output=output)


def _materialize_artifact(
    extracted: list[Path],
    *,
    staging_dir: Path,
    working_dir: Path,
    source_path: str,
    filename_override: str | None,
) -> tuple[Path, str]:
    """Return (artifact_path, filename) for one export, zipping directories."""

    if len(extracted) == 1:
        source = extracted[0]
        filename = _safe_filename(filename_override or source.name)
        target = working_dir / filename
        if target != source:
            working_dir.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
        return target, filename

    base_name = filename_override or f"{Path(source_path).name or 'export'}.zip"
    filename = _safe_filename(base_name)
    if not filename.endswith(".zip"):
        filename = f"{filename}.zip"
    target = working_dir / filename
    working_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for file_path in extracted:
            archive.write(file_path, file_path.relative_to(staging_dir))
    return target, filename


def _safe_filename(value: str) -> str:
    cleaned = "".join(
        char if char.isalnum() or char in "._-" else "-" for char in value.strip()
    ).strip("-.")
    return cleaned[:128] or "artifact"


def _exec_output(result: SandboxExecResult) -> JsonObject:
    output: JsonObject = {
        "successful": result.ok,
        "exit_code": result.exit_code,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "duration_ms": result.duration_ms,
        "timed_out": result.timed_out,
        "truncated": result.truncated,
    }
    if not result.ok:
        output["error"] = {
            "code": "sandbox_command_failed",
            "message": (
                "Command timed out."
                if result.timed_out
                else f"Command exited with status {result.exit_code}."
            ),
            "recoverable": True,
        }
    return output


def _sandbox_error_result(exc: Exception) -> ToolResult:
    code = (
        "sandbox_service_unavailable"
        if isinstance(exc, SandboxUnavailableError)
        else "sandbox_session_error"
    )
    return ToolResult(
        output={
            "successful": False,
            "error": {
                "code": code,
                "message": str(exc),
                "recoverable": True,
                "hint": (
                    "Check that the sandbox-runner service is healthy and "
                    "sessions are enabled."
                ),
            },
        }
    )


def _required_string(args: JsonObject, key: str, *, max_chars: int) -> str:
    value = args.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'{key}' must be a non-empty string")
    stripped = value.strip()
    if len(stripped) > max_chars:
        raise ValueError(f"'{key}' must be at most {max_chars} chars")
    return stripped


def _optional_string(args: JsonObject, key: str) -> str | None:
    value = args.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"'{key}' must be a string")
    stripped = value.strip()
    return stripped or None


def _timeout_seconds(args: JsonObject) -> int:
    value = args.get("timeout_seconds", DEFAULT_BASH_TIMEOUT_SECONDS)
    if not isinstance(value, int):
        raise ValueError("'timeout_seconds' must be an integer")
    if value < 1 or value > MAX_BASH_TIMEOUT_SECONDS:
        raise ValueError(
            f"'timeout_seconds' must be between 1 and {MAX_BASH_TIMEOUT_SECONDS}"
        )
    return value
