"""Long-lived sandbox session management for the runner service.

A session is one hardened container per task that stays alive across tool
calls. Commands run through the Docker exec API; files move through the
Docker archive API so no shell-escaping surface exists for file content.
"""

from __future__ import annotations

import io
import shlex
import tarfile
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from kortny.sandbox_runner.docker_api import (
    DockerApiSessionClient,
    DockerExecResult,
    DockerSessionCreateSpec,
)

WORKSPACE_ROOT = "/workspace"


class SessionNotFoundError(LookupError):
    """Raised when a session id is unknown to this runner."""


class SessionPathError(ValueError):
    """Raised when a session file path escapes the workspace."""


class SessionDockerError(RuntimeError):
    """Raised when the Docker API rejects a session operation."""

    def __init__(self, status: str, error: str | None = None) -> None:
        super().__init__(error or status)
        self.status = status
        self.error = error


@dataclass(frozen=True, slots=True)
class SessionConfig:
    """Static configuration for session containers and their lifecycle."""

    image: str
    cpus: float = 2.0
    memory_mb: int = 2048
    pids_limit: int = 512
    workspace_mb: int = 1024
    idle_seconds: int = 1800
    max_age_seconds: int = 14400
    exec_max_timeout_seconds: int = 300
    exec_max_output_bytes: int = 65536
    file_max_bytes: int = 5 * 1024 * 1024
    archive_max_bytes: int = 50 * 1024 * 1024

    def __post_init__(self) -> None:
        if not self.image.strip():
            raise ValueError("Session image is required")
        for name in (
            "cpus",
            "memory_mb",
            "pids_limit",
            "workspace_mb",
            "idle_seconds",
            "max_age_seconds",
            "exec_max_timeout_seconds",
            "exec_max_output_bytes",
            "file_max_bytes",
            "archive_max_bytes",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"Session config {name} must be positive")


@dataclass(slots=True)
class SessionRecord:
    """In-memory state for one live session."""

    session_id: str
    task_id: str
    container_id: str
    profile: str
    created_at: float
    last_used: float


@dataclass(frozen=True, slots=True)
class SessionInfo:
    """API-facing summary of one session."""

    session_id: str
    task_id: str
    container_id: str
    profile: str
    reused: bool

    def to_payload(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "task_id": self.task_id,
            "container_id": self.container_id,
            "profile": self.profile,
            "reused": self.reused,
        }


@dataclass(slots=True)
class SessionManager:
    """Owns the registry of live session containers for this runner."""

    docker_client: DockerApiSessionClient
    config: SessionConfig
    clock: Callable[[], float] = time.time
    _sessions: dict[str, SessionRecord] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def create_or_get(self, task_id: str, profile: str = "workbench") -> SessionInfo:
        """Return the live session for a task, creating one if needed."""

        with self._lock:
            for record in self._sessions.values():
                if record.task_id == task_id:
                    record.last_used = self.clock()
                    return _session_info(record, reused=True)

            adopted = self._adopt_existing_locked(task_id)
            if adopted is not None:
                return _session_info(adopted, reused=True)

            session_id = uuid.uuid4().hex
            result = self.docker_client.create_session_container(
                DockerSessionCreateSpec(
                    image=self.config.image,
                    session_id=session_id,
                    task_id=task_id,
                    profile=profile,
                    cpus=self.config.cpus,
                    memory_mb=self.config.memory_mb,
                    pids_limit=self.config.pids_limit,
                    workspace_mb=self.config.workspace_mb,
                )
            )
            if not result.ok or result.container_id is None:
                raise SessionDockerError(result.status, result.error)
            now = self.clock()
            record = SessionRecord(
                session_id=session_id,
                task_id=task_id,
                container_id=result.container_id,
                profile=profile,
                created_at=now,
                last_used=now,
            )
            self._sessions[record.session_id] = record
            return _session_info(record, reused=False)

    def live_container_ids(self) -> frozenset[str]:
        """Return the container ids of all sessions this process still owns.

        Used by the container GC to protect in-use workbench containers from
        being reaped no matter their age.
        """

        with self._lock:
            return frozenset(record.container_id for record in self._sessions.values())

    def get(self, session_id: str) -> SessionRecord:
        """Return the registry record for a session id."""

        with self._lock:
            record = self._sessions.get(session_id)
        if record is None:
            raise SessionNotFoundError(session_id)
        return record

    def exec(
        self,
        session_id: str,
        command: str,
        *,
        workdir: str = WORKSPACE_ROOT,
        timeout_seconds: int = 120,
    ) -> DockerExecResult:
        """Run one shell command in the session container."""

        record = self.get(session_id)
        _require_safe_workspace_path(workdir)
        timeout = min(max(timeout_seconds, 1), self.config.exec_max_timeout_seconds)
        result = self.docker_client.exec_in_container(
            record.container_id,
            ("timeout", str(timeout), "/bin/sh", "-lc", command),
            workdir=workdir,
            timeout_seconds=timeout,
            max_output_bytes=self.config.exec_max_output_bytes,
        )
        with self._lock:
            record.last_used = self.clock()
        return result

    def write_file(self, session_id: str, path: str, content: bytes) -> int:
        """Write one file into the session workspace via the archive API."""

        record = self.get(session_id)
        _require_safe_workspace_path(path)
        if len(content) > self.config.file_max_bytes:
            raise SessionPathError(
                f"File content exceeds {self.config.file_max_bytes} bytes"
            )
        parent = str(PurePosixPath(path).parent)
        filename = PurePosixPath(path).name
        if not filename:
            raise SessionPathError("File path must include a filename")
        mkdir = self.docker_client.exec_in_container(
            record.container_id,
            ("mkdir", "-p", parent),
            workdir=WORKSPACE_ROOT,
            timeout_seconds=10,
        )
        if not mkdir.ok:
            raise SessionDockerError(mkdir.status, mkdir.error or mkdir.stderr)
        archive = self.docker_client.put_archive(
            record.container_id,
            parent,
            _single_file_tar(filename, content),
        )
        if not archive.ok:
            raise SessionDockerError(archive.status, archive.error)
        with self._lock:
            record.last_used = self.clock()
        return len(content)

    def read_file(self, session_id: str, path: str) -> bytes:
        """Read one file from the session workspace via the archive API."""

        record = self.get(session_id)
        _require_safe_workspace_path(path)
        archive = self.docker_client.get_archive(record.container_id, path)
        if not archive.ok:
            raise SessionDockerError(archive.status, archive.error)
        content = _first_file_from_tar(archive.content, self.config.file_max_bytes)
        with self._lock:
            record.last_used = self.clock()
        return content

    def export_archive(self, session_id: str, path: str) -> bytes:
        """Return a raw tar archive of one workspace path."""

        record = self.get(session_id)
        _require_safe_workspace_path(path)
        archive = self.docker_client.get_archive(record.container_id, path)
        if not archive.ok:
            raise SessionDockerError(archive.status, archive.error)
        if len(archive.content) > self.config.archive_max_bytes:
            raise SessionPathError(
                f"Archive exceeds {self.config.archive_max_bytes} bytes"
            )
        with self._lock:
            record.last_used = self.clock()
        return archive.content

    def close(self, session_id: str) -> None:
        """Remove one session container and forget it."""

        record = self.get(session_id)
        self.docker_client.remove_session_container(record.container_id)
        with self._lock:
            self._sessions.pop(session_id, None)

    def reap(self) -> list[str]:
        """Remove idle, expired, and orphaned session containers."""

        now = self.clock()
        removed: list[str] = []
        with self._lock:
            expired = [
                record
                for record in self._sessions.values()
                if now - record.last_used > self.config.idle_seconds
                or now - record.created_at > self.config.max_age_seconds
            ]
            for record in expired:
                self._sessions.pop(record.session_id, None)
            known_container_ids = {
                record.container_id for record in self._sessions.values()
            }
        for record in expired:
            self.docker_client.remove_session_container(record.container_id)
            removed.append(record.container_id)

        for container in self.docker_client.list_session_containers():
            if container.container_id in known_container_ids:
                continue
            age = time.time() - container.created_at_epoch
            if age > self.config.idle_seconds:
                self.docker_client.remove_session_container(container.container_id)
                removed.append(container.container_id)
        return removed

    def _adopt_existing_locked(self, task_id: str) -> SessionRecord | None:
        """Adopt a running labeled container for this task after a restart."""

        for container in self.docker_client.list_session_containers():
            if container.task_id != task_id or not container.running:
                continue
            now = self.clock()
            record = SessionRecord(
                session_id=container.session_id
                or _session_id_from_create(container.container_id),
                task_id=task_id,
                container_id=container.container_id,
                profile="workbench",
                created_at=now,
                last_used=now,
            )
            self._sessions[record.session_id] = record
            return record
        return None


def shell_quote(value: str) -> str:
    """Quote one value for safe interpolation into a shell command."""

    return shlex.quote(value)


def _session_info(record: SessionRecord, *, reused: bool) -> SessionInfo:
    return SessionInfo(
        session_id=record.session_id,
        task_id=record.task_id,
        container_id=record.container_id,
        profile=record.profile,
        reused=reused,
    )


def _session_id_from_create(container_id: str) -> str:
    return container_id[:32]


def _require_safe_workspace_path(value: str) -> None:
    if value != WORKSPACE_ROOT and not value.startswith(f"{WORKSPACE_ROOT}/"):
        raise SessionPathError("Path must be /workspace or a child path")
    if ".." in PurePosixPath(value).parts:
        raise SessionPathError("Path must not contain '..'")


def _single_file_tar(filename: str, content: bytes) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        info = tarfile.TarInfo(name=filename)
        info.size = len(content)
        info.mode = 0o644
        tar.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


def _first_file_from_tar(tar_bytes: bytes, max_bytes: int) -> bytes:
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            if member.size > max_bytes:
                raise SessionPathError(f"File exceeds {max_bytes} bytes")
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            return extracted.read()
    raise SessionPathError("Archive did not contain a regular file")
