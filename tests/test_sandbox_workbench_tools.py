import io
import tarfile
import zipfile
from pathlib import Path

import pytest

from kortny.approvals import (
    SANDBOX_WORKBENCH_APPROVAL_KEY,
    ApprovalScope,
    ToolApprovalPolicy,
    ToolApprovalRequest,
    approval_key_for,
    approval_prompt_text,
)
from kortny.execution.sandbox_sessions import (
    SandboxExecResult,
    SandboxSessionError,
    SandboxSessionInfo,
)
from kortny.tools.sandbox_workbench import (
    SandboxBashTool,
    SandboxExportArtifactTool,
    SandboxPublishPreviewTool,
    SandboxReadFileTool,
    SandboxWriteFileTool,
    WorkbenchSession,
)


class FakeSessionClient:
    def __init__(self) -> None:
        self.opened: list[tuple[str, str]] = []
        self.execs: list[tuple[str, str, str, int]] = []
        self.files: dict[str, bytes] = {}
        self.archive: bytes = b""
        self.exec_result = SandboxExecResult(exit_code=0, stdout="done")
        self.reused = False

    def open_session(
        self, task_id: str, profile: str = "workbench"
    ) -> SandboxSessionInfo:
        self.opened.append((task_id, profile))
        return SandboxSessionInfo(
            session_id="s-1",
            task_id=task_id,
            container_id="c-1",
            profile=profile,
            reused=self.reused,
        )

    def exec(
        self,
        session_id: str,
        command: str,
        *,
        workdir: str = "/workspace",
        timeout_seconds: int = 120,
    ) -> SandboxExecResult:
        self.execs.append((session_id, command, workdir, timeout_seconds))
        return self.exec_result

    def write_file(self, session_id: str, path: str, content: bytes) -> int:
        self.files[path] = content
        return len(content)

    def read_file(self, session_id: str, path: str) -> bytes:
        if path not in self.files:
            raise SandboxSessionError("not found", status_code=404)
        return self.files[path]

    def export_archive(self, session_id: str, path: str) -> bytes:
        return self.archive

    def close_session(self, session_id: str) -> None:
        return None


class FakeTask:
    id = "task-1"


class RecordingEventSink:
    def __init__(self) -> None:
        self.events: list[tuple[object, str, dict]] = []

    def append_event(self, task, event_type, payload=None):  # type: ignore[no-untyped-def]
        self.events.append((task, str(event_type), payload or {}))
        return object()


def _workbench(
    client: FakeSessionClient | None = None,
    sink: RecordingEventSink | None = None,
) -> tuple[WorkbenchSession, FakeSessionClient]:
    resolved = client or FakeSessionClient()
    return (
        WorkbenchSession(client=resolved, task=FakeTask(), task_service=sink),
        resolved,
    )


def _tar_with(entries: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for name, content in entries.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


def test_sandbox_bash_runs_command_and_reports_output() -> None:
    sink = RecordingEventSink()
    workbench, client = _workbench(sink=sink)
    tool = SandboxBashTool(workbench=workbench)

    result = tool.invoke({"command": "echo hi", "timeout_seconds": 60})

    assert result.output["successful"] is True
    assert result.output["stdout"] == "done"
    assert client.execs == [("s-1", "echo hi", "/workspace", 60)]
    assert client.opened == [("task-1", "workbench")]
    assert any(
        payload.get("message") == "sandbox_session" for _, _, payload in sink.events
    )


def test_sandbox_bash_reuses_session_without_logging_again() -> None:
    sink = RecordingEventSink()
    client = FakeSessionClient()
    client.reused = True
    workbench, _ = _workbench(client=client, sink=sink)
    tool = SandboxBashTool(workbench=workbench)

    tool.invoke({"command": "echo hi"})

    assert sink.events == []


def test_sandbox_bash_maps_failed_exit_to_recoverable_error() -> None:
    workbench, client = _workbench()
    client.exec_result = SandboxExecResult(exit_code=2, stderr="boom", timed_out=False)
    tool = SandboxBashTool(workbench=workbench)

    result = tool.invoke({"command": "false"})

    assert result.output["successful"] is False
    assert result.output["error"]["code"] == "sandbox_command_failed"
    assert result.output["error"]["recoverable"] is True


def test_sandbox_bash_validates_arguments() -> None:
    workbench, _ = _workbench()
    tool = SandboxBashTool(workbench=workbench)

    with pytest.raises(ValueError):
        tool.invoke({"command": ""})
    with pytest.raises(ValueError):
        tool.invoke({"command": "echo hi", "timeout_seconds": 9999})


def test_sandbox_write_and_read_file_round_trip() -> None:
    workbench, client = _workbench()
    write_tool = SandboxWriteFileTool(workbench=workbench)
    read_tool = SandboxReadFileTool(workbench=workbench)

    write_result = write_tool.invoke(
        {"path": "/workspace/app.py", "content": "print('hi')"}
    )
    read_result = read_tool.invoke({"path": "/workspace/app.py"})

    assert write_result.output["successful"] is True
    assert client.files["/workspace/app.py"] == b"print('hi')"
    assert read_result.output["content"] == "print('hi')"
    assert read_result.output["truncated"] is False


def test_sandbox_read_file_maps_session_error() -> None:
    workbench, _ = _workbench()
    tool = SandboxReadFileTool(workbench=workbench)

    result = tool.invoke({"path": "/workspace/missing.txt"})

    assert result.output["successful"] is False
    assert result.output["error"]["code"] == "sandbox_session_error"


def test_export_artifact_single_file(tmp_path: Path) -> None:
    workbench, client = _workbench()
    client.archive = _tar_with({"report.csv": b"a,b\n1,2\n"})
    tool = SandboxExportArtifactTool(workbench=workbench, working_dir=tmp_path)

    result = tool.invoke({"path": "/workspace/report.csv"})

    assert result.output["successful"] is True
    assert result.output["filename"] == "report.csv"
    assert result.output["mime_type"] == "text/csv"
    assert result.artifacts[0].filename == "report.csv"
    assert Path(result.output["path"]).read_bytes() == b"a,b\n1,2\n"


def test_export_artifact_directory_becomes_zip(tmp_path: Path) -> None:
    workbench, client = _workbench()
    client.archive = _tar_with(
        {
            "dist/index.html": b"<html></html>",
            "dist/app.js": b"console.log(1)",
        }
    )
    tool = SandboxExportArtifactTool(workbench=workbench, working_dir=tmp_path)

    result = tool.invoke({"path": "/workspace/dist"})

    assert result.output["successful"] is True
    assert result.output["filename"].endswith(".zip")
    with zipfile.ZipFile(Path(result.output["path"])) as archive:
        assert sorted(archive.namelist()) == ["app.js", "index.html"]


def test_export_artifact_rejects_traversal_archive(tmp_path: Path) -> None:
    workbench, client = _workbench()
    client.archive = _tar_with({"../../etc/evil": b"x"})
    tool = SandboxExportArtifactTool(workbench=workbench, working_dir=tmp_path)

    result = tool.invoke({"path": "/workspace/dist"})

    assert result.output["successful"] is False
    assert "Unsafe" in result.output["error"]["message"]


def test_publish_preview_extracts_site_and_returns_url(tmp_path: Path) -> None:
    workbench, client = _workbench()
    client.archive = _tar_with(
        {
            "site/index.html": b"<html>dash</html>",
            "site/data.json": b"{}",
        }
    )
    tool = SandboxPublishPreviewTool(
        workbench=workbench,
        artifacts_dir=tmp_path,
        public_base_url="https://kortny.example.com",
        signing_secret="secret",
    )

    result = tool.invoke({"path": "/workspace/site", "slug": "Sales Dash"})

    assert result.output["successful"] is True
    assert result.output["has_index_html"] is True
    url = result.output["preview_url"]
    assert url.startswith("https://kortny.example.com/preview/")
    assert url.endswith("/task-1/sales-dash/index.html")
    assert (tmp_path / "task-1" / "sales-dash" / "index.html").is_file()


def test_publish_preview_warns_without_index(tmp_path: Path) -> None:
    workbench, client = _workbench()
    client.archive = _tar_with({"site/readme.txt": b"hi"})
    tool = SandboxPublishPreviewTool(
        workbench=workbench,
        artifacts_dir=tmp_path,
        public_base_url="https://kortny.example.com",
        signing_secret="secret",
    )

    result = tool.invoke({"path": "/workspace/site"})

    assert result.output["has_index_html"] is False
    assert "warning" in result.output


def test_workbench_tools_share_one_approval_key() -> None:
    assert (
        approval_key_for("sandbox_bash", "hash-a")
        == approval_key_for("sandbox_write_file", "hash-b")
        == SANDBOX_WORKBENCH_APPROVAL_KEY
    )
    assert approval_key_for("code_exec", "hash-a") == "code_exec:hash-a"


def test_workbench_tools_require_user_approval() -> None:
    policy = ToolApprovalPolicy()
    workbench, _ = _workbench()
    requirement = policy.requirement_for(
        SandboxBashTool(workbench=workbench), {"command": "echo hi"}
    )

    assert requirement.scope is ApprovalScope.user
    assert requirement.risk == "sandboxed_code_execution"


def test_workbench_approval_prompt_mentions_session_scope() -> None:
    request = ToolApprovalRequest(
        approval_key=SANDBOX_WORKBENCH_APPROVAL_KEY,
        tool_name="sandbox_bash",
        tool_call_id="call-1",
        normalized_args_hash="hash",
        argument_keys=("command",),
        scope=ApprovalScope.user,
        reason="reason",
        risk="sandboxed_code_execution",
        arguments={"command": "echo hi"},
    )

    text = approval_prompt_text(request)

    assert "One approval covers all sandbox commands" in text
