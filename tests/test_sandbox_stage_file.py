"""Unit tests for SandboxStageFileTool."""

from __future__ import annotations

import hashlib

import httpx
import pytest
from slack_sdk.errors import SlackApiError

from kortny.execution.sandbox_sessions import (
    SandboxExecResult,
    SandboxSessionError,
    SandboxSessionInfo,
)
from kortny.tools.sandbox_workbench import (
    SANDBOX_STAGE_FILE_MAX_BYTES,
    SandboxStageFileTool,
    WorkbenchSession,
)
from kortny.tools.types import RecoverableToolError


class FakeSessionClient:
    def __init__(self) -> None:
        self.opened: list[tuple[str, str]] = []
        self.files: dict[str, bytes] = {}
        self.exec_result = SandboxExecResult(exit_code=0, stdout="done")
        self.reused = False
        self._raise_write: Exception | None = None

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
        return self.exec_result

    def write_file(self, session_id: str, path: str, content: bytes) -> int:
        if self._raise_write is not None:
            raise self._raise_write
        self.files[path] = content
        return len(content)

    def read_file(self, session_id: str, path: str) -> bytes:
        return self.files.get(path, b"")

    def export_archive(self, session_id: str, path: str) -> bytes:
        return b""

    def close_session(self, session_id: str) -> None:
        return None


class FakeTask:
    id = "task-1"


class FakeTransport(httpx.BaseTransport):
    def __init__(self, content: bytes, status_code: int = 200) -> None:
        self.content = content
        self.status_code = status_code

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(self.status_code, content=self.content)


class FakeSlackFileClient:
    def __init__(
        self,
        *,
        name: str = "doc.docx",
        url: str = "https://files.slack.com/doc.docx",
        raise_error: bool = False,
    ) -> None:
        self.name = name
        self.url = url
        self.calls: list[str] = []
        self.raise_error = raise_error

    def files_info(self, *, file: str) -> dict:
        self.calls.append(file)
        if self.raise_error:
            raise SlackApiError(
                "file_not_found", {"ok": False, "error": "file_not_found"}
            )
        return {
            "ok": True,
            "file": {
                "id": file,
                "name": self.name,
                "url_private_download": self.url,
                "size": 100,
            },
        }


def _make_tool(
    client: FakeSessionClient,
    slack_client: FakeSlackFileClient,
    transport: httpx.BaseTransport,
) -> SandboxStageFileTool:
    workbench = WorkbenchSession(client=client, task=FakeTask(), task_service=None)
    return SandboxStageFileTool(
        workbench=workbench,
        slack_file_client=slack_client,
        bot_token="xoxb-test",
        transport=transport,
    )


def test_happy_path_with_file_id() -> None:
    """Happy path: file_id lookup + download + write to sandbox."""
    content = b"PK fake docx bytes"
    client = FakeSessionClient()
    slack_client = FakeSlackFileClient(
        name="report.docx", url="https://files.slack.com/report.docx"
    )
    transport = FakeTransport(content=content)
    tool = _make_tool(client, slack_client, transport)

    result = tool.invoke({"file_id": "FABC123"})

    assert result.output["filename"] == "report.docx"
    assert result.output["size_bytes"] == len(content)
    assert result.output["sha256"] == hashlib.sha256(content).hexdigest()
    assert result.output["path"] == "/workspace/original.docx"
    assert (
        result.output["mime_type"]
        == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
    assert client.files["/workspace/original.docx"] == content
    assert slack_client.calls == ["FABC123"]


def test_happy_path_with_file_url() -> None:
    """Happy path: file_url bypasses slack lookup."""
    content = b"binary content here"
    client = FakeSessionClient()
    slack_client = FakeSlackFileClient()
    transport = FakeTransport(content=content)
    tool = _make_tool(client, slack_client, transport)

    result = tool.invoke(
        {"file_url": "https://files.slack.com/files-pri/T123/myfile.pdf"}
    )

    assert result.output["filename"] == "myfile.pdf"
    assert result.output["path"] == "/workspace/original.pdf"
    assert client.files["/workspace/original.pdf"] == content
    # slack files_info was NOT called
    assert slack_client.calls == []


def test_explicit_dest_path() -> None:
    """dest_path is used when provided."""
    content = b"bytes"
    client = FakeSessionClient()
    slack_client = FakeSlackFileClient(
        name="report.docx", url="https://files.slack.com/report.docx"
    )
    transport = FakeTransport(content=content)
    tool = _make_tool(client, slack_client, transport)

    result = tool.invoke({"file_id": "F1", "dest_path": "docs/myfile.docx"})

    assert result.output["path"] == "/workspace/docs/myfile.docx"
    assert "/workspace/docs/myfile.docx" in client.files


def test_default_dest_path_uses_ext() -> None:
    """Default dest_path is original.<ext> from filename."""
    content = b"bytes"
    client = FakeSessionClient()
    slack_client = FakeSlackFileClient(
        name="report.docx", url="https://files.slack.com/report.docx"
    )
    transport = FakeTransport(content=content)
    tool = _make_tool(client, slack_client, transport)

    result = tool.invoke({"file_id": "F1"})

    assert result.output["path"] == "/workspace/original.docx"


def test_size_guard_raises() -> None:
    """Files over 5 MB raise RecoverableToolError with file_too_large_for_sandbox."""
    oversized = b"x" * (SANDBOX_STAGE_FILE_MAX_BYTES + 1)
    client = FakeSessionClient()
    slack_client = FakeSlackFileClient(
        name="big.docx", url="https://files.slack.com/big.docx"
    )
    transport = FakeTransport(content=oversized)
    tool = _make_tool(client, slack_client, transport)

    with pytest.raises(RecoverableToolError) as exc_info:
        tool.invoke({"file_id": "F_BIG"})

    assert exc_info.value.code == "file_too_large_for_sandbox"


def test_session_error_raises() -> None:
    """SandboxSessionError in write_file raises RecoverableToolError sandbox_unavailable."""
    content = b"bytes"
    client = FakeSessionClient()
    client._raise_write = SandboxSessionError("fail")
    slack_client = FakeSlackFileClient(
        name="x.docx", url="https://files.slack.com/x.docx"
    )
    transport = FakeTransport(content=content)
    tool = _make_tool(client, slack_client, transport)

    with pytest.raises(RecoverableToolError) as exc_info:
        tool.invoke({"file_id": "F1"})

    assert exc_info.value.code == "sandbox_unavailable"


def test_slack_lookup_error_raises() -> None:
    """SlackApiError in files_info raises RecoverableToolError slack_file_lookup_failed."""
    client = FakeSessionClient()
    slack_client = FakeSlackFileClient(raise_error=True)
    transport = FakeTransport(content=b"bytes")
    tool = _make_tool(client, slack_client, transport)

    with pytest.raises(RecoverableToolError) as exc_info:
        tool.invoke({"file_id": "FXXX"})

    assert exc_info.value.code == "slack_file_lookup_failed"


def test_no_args_raises_value_error() -> None:
    """No file_id or file_url raises ValueError."""
    client = FakeSessionClient()
    slack_client = FakeSlackFileClient()
    transport = FakeTransport(content=b"")
    tool = _make_tool(client, slack_client, transport)

    with pytest.raises(ValueError, match="requires either"):
        tool.invoke({})


def test_both_args_raises_value_error() -> None:
    """Providing both file_id and file_url raises ValueError."""
    client = FakeSessionClient()
    slack_client = FakeSlackFileClient()
    transport = FakeTransport(content=b"")
    tool = _make_tool(client, slack_client, transport)

    with pytest.raises(ValueError, match="not both"):
        tool.invoke({"file_id": "F1", "file_url": "https://example.com/f"})
