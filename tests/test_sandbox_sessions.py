import io
import tarfile

import pytest

from kortny.sandbox_runner.docker_api import (
    DockerArchiveResult,
    DockerExecResult,
    DockerSessionContainer,
    DockerSessionCreateResult,
    DockerSessionCreateSpec,
    _demux_docker_stream,
    _session_create_payload,
)
from kortny.sandbox_runner.sessions import (
    SessionConfig,
    SessionDockerError,
    SessionManager,
    SessionNotFoundError,
    SessionPathError,
)


def _config(**overrides: object) -> SessionConfig:
    defaults: dict[str, object] = {"image": "test-image:latest"}
    defaults.update(overrides)
    return SessionConfig(**defaults)  # type: ignore[arg-type]


class FakeDockerSessionClient:
    def __init__(self) -> None:
        self.created: list[DockerSessionCreateSpec] = []
        self.execs: list[tuple[str, tuple[str, ...], str, int]] = []
        self.put_archives: list[tuple[str, str, bytes]] = []
        self.removed: list[str] = []
        self.listed: list[DockerSessionContainer] = []
        self.archive_content: bytes = b""
        self.exec_result = DockerExecResult(
            ok=True, status="succeeded", exit_code=0, stdout="ok"
        )
        self.create_result: DockerSessionCreateResult | None = None
        self._counter = 0

    def create_session_container(
        self, spec: DockerSessionCreateSpec
    ) -> DockerSessionCreateResult:
        self.created.append(spec)
        if self.create_result is not None:
            return self.create_result
        self._counter += 1
        return DockerSessionCreateResult(
            ok=True, status="running", container_id=f"container-{self._counter}"
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
        self.execs.append((container_id, command, workdir, timeout_seconds))
        return self.exec_result

    def get_archive(self, container_id: str, path: str) -> DockerArchiveResult:
        return DockerArchiveResult(
            ok=True, status="succeeded", content=self.archive_content
        )

    def put_archive(
        self, container_id: str, path: str, tar_bytes: bytes
    ) -> DockerArchiveResult:
        self.put_archives.append((container_id, path, tar_bytes))
        return DockerArchiveResult(ok=True, status="succeeded")

    def list_session_containers(self) -> tuple[DockerSessionContainer, ...]:
        return tuple(self.listed)

    def remove_session_container(self, container_id: str) -> str | None:
        self.removed.append(container_id)
        return None


class FakeClock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now


def test_session_create_payload_is_hardened() -> None:
    payload = _session_create_payload(
        DockerSessionCreateSpec(
            image="test-image:latest",
            session_id="abc123",
            task_id="task-1",
            profile="workbench",
            memory_mb=2048,
            workspace_mb=1024,
        )
    )

    assert payload["Cmd"] == ["sleep", "infinity"]
    assert payload["Tty"] is False
    assert "HOME=/workspace/home" in payload["Env"]
    assert payload["Labels"]["kortny.sandbox.kind"] == "session"
    assert payload["Labels"]["kortny.sandbox.session"] == "abc123"
    assert payload["Labels"]["kortny.sandbox.task"] == "task-1"
    host_config = payload["HostConfig"]
    assert host_config["CapDrop"] == ["ALL"]
    assert host_config["NetworkMode"] == "none"
    assert host_config["ReadonlyRootfs"] is True
    assert host_config["SecurityOpt"] == ["no-new-privileges"]
    assert host_config["Init"] is True
    assert host_config["IpcMode"] == "private"
    assert host_config["Memory"] == 2048 * 1024 * 1024
    assert host_config["MemorySwap"] == host_config["Memory"]
    assert host_config["PidsLimit"] == 512
    assert payload["Volumes"] == {"/workspace": {}}
    assert "/workspace" not in host_config["Tmpfs"]
    assert "/tmp" in host_config["Tmpfs"]


def test_demux_docker_stream_splits_stdout_and_stderr() -> None:
    raw = (
        b"\x01\x00\x00\x00\x00\x00\x00\x05hello"
        b"\x02\x00\x00\x00\x00\x00\x00\x04oops"
    )

    stdout, stderr = _demux_docker_stream(raw)

    assert stdout == "hello"
    assert stderr == "oops"


def test_demux_docker_stream_handles_raw_tty_output() -> None:
    stdout, stderr = _demux_docker_stream(b"plain output")

    assert stdout == "plain output"
    assert stderr == ""


def test_create_or_get_reuses_session_for_same_task() -> None:
    docker = FakeDockerSessionClient()
    manager = SessionManager(docker_client=docker, config=_config())

    first = manager.create_or_get("task-1")
    second = manager.create_or_get("task-1")
    other = manager.create_or_get("task-2")

    assert first.reused is False
    assert second.reused is True
    assert second.session_id == first.session_id
    assert other.session_id != first.session_id
    assert len(docker.created) == 2


def test_create_or_get_adopts_running_labeled_container() -> None:
    docker = FakeDockerSessionClient()
    docker.listed = [
        DockerSessionContainer(
            container_id="orphan-1",
            session_id="session-orphan",
            task_id="task-1",
            created_at_epoch=0,
            running=True,
        )
    ]
    manager = SessionManager(docker_client=docker, config=_config())

    info = manager.create_or_get("task-1")

    assert info.reused is True
    assert info.session_id == "session-orphan"
    assert info.container_id == "orphan-1"
    assert docker.created == []


def test_create_or_get_raises_on_docker_failure() -> None:
    docker = FakeDockerSessionClient()
    docker.create_result = DockerSessionCreateResult(
        ok=False, status="session_create_failed", error="boom"
    )
    manager = SessionManager(docker_client=docker, config=_config())

    with pytest.raises(SessionDockerError) as exc_info:
        manager.create_or_get("task-1")

    assert exc_info.value.status == "session_create_failed"


def test_exec_wraps_command_with_timeout_and_caps_it() -> None:
    docker = FakeDockerSessionClient()
    manager = SessionManager(
        docker_client=docker,
        config=_config(exec_max_timeout_seconds=300),
    )
    info = manager.create_or_get("task-1")

    manager.exec(info.session_id, "echo hi", timeout_seconds=9999)

    container_id, command, workdir, timeout = docker.execs[-1]
    assert container_id == info.container_id
    assert command == ("timeout", "300", "/bin/sh", "-lc", "echo hi")
    assert workdir == "/workspace"
    assert timeout == 300


def test_exec_rejects_unknown_session() -> None:
    manager = SessionManager(
        docker_client=FakeDockerSessionClient(), config=_config()
    )

    with pytest.raises(SessionNotFoundError):
        manager.exec("missing", "echo hi")


def test_exec_rejects_workdir_outside_workspace() -> None:
    docker = FakeDockerSessionClient()
    manager = SessionManager(docker_client=docker, config=_config())
    info = manager.create_or_get("task-1")

    with pytest.raises(SessionPathError):
        manager.exec(info.session_id, "echo hi", workdir="/etc")


def test_write_file_creates_parent_and_puts_tar() -> None:
    docker = FakeDockerSessionClient()
    manager = SessionManager(docker_client=docker, config=_config())
    info = manager.create_or_get("task-1")

    written = manager.write_file(
        info.session_id, "/workspace/app/index.html", b"<html></html>"
    )

    assert written == len(b"<html></html>")
    assert docker.execs[-1][1] == ("mkdir", "-p", "/workspace/app")
    container_id, path, tar_bytes = docker.put_archives[-1]
    assert container_id == info.container_id
    assert path == "/workspace/app"
    with tarfile.open(fileobj=io.BytesIO(tar_bytes)) as tar:
        member = tar.getmembers()[0]
        assert member.name == "index.html"
        extracted = tar.extractfile(member)
        assert extracted is not None
        assert extracted.read() == b"<html></html>"


def test_write_file_rejects_paths_outside_workspace() -> None:
    docker = FakeDockerSessionClient()
    manager = SessionManager(docker_client=docker, config=_config())
    info = manager.create_or_get("task-1")

    with pytest.raises(SessionPathError):
        manager.write_file(info.session_id, "/etc/passwd", b"x")
    with pytest.raises(SessionPathError):
        manager.write_file(info.session_id, "/workspace/../etc/passwd", b"x")


def test_read_file_extracts_single_member() -> None:
    docker = FakeDockerSessionClient()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        info_member = tarfile.TarInfo(name="report.csv")
        info_member.size = 4
        tar.addfile(info_member, io.BytesIO(b"a,b\n"))
    docker.archive_content = buffer.getvalue()
    manager = SessionManager(docker_client=docker, config=_config())
    info = manager.create_or_get("task-1")

    content = manager.read_file(info.session_id, "/workspace/report.csv")

    assert content == b"a,b\n"


def test_reap_removes_idle_sessions_and_orphans() -> None:
    docker = FakeDockerSessionClient()
    clock = FakeClock()
    manager = SessionManager(
        docker_client=docker,
        config=_config(idle_seconds=100, max_age_seconds=1000),
        clock=clock,
    )
    info = manager.create_or_get("task-1")
    docker.listed = [
        DockerSessionContainer(
            container_id="orphan-old",
            session_id="s-old",
            task_id="task-x",
            created_at_epoch=0,
            running=True,
        )
    ]

    clock.now += 50
    assert manager.reap() == ["orphan-old"]

    clock.now += 100
    removed = manager.reap()

    assert info.container_id in removed
    with pytest.raises(SessionNotFoundError):
        manager.get(info.session_id)


def test_close_removes_container_and_forgets_session() -> None:
    docker = FakeDockerSessionClient()
    manager = SessionManager(docker_client=docker, config=_config())
    info = manager.create_or_get("task-1")

    manager.close(info.session_id)

    assert docker.removed == [info.container_id]
    with pytest.raises(SessionNotFoundError):
        manager.get(info.session_id)
