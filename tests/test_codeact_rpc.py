"""Unit tests for kortny/execution/codeact_rpc.py — Slice A (engine only).

All tests use a FAKE in-memory session (no real container, no LLM, no network).
The broker security tests are exhaustive: nonce, allowlist, byte caps, call-count,
dispatch exceptions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pytest

from kortny.config import Settings
from kortny.execution.codeact_rpc import (
    CodeActResult,
    CodeActRpcBroker,
    ToolStubSpec,
    generate_stub_module,
    run_codeact,
)
from kortny.execution.sandbox_sessions import SandboxExecResult

# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------


def _settings(**overrides: Any) -> Settings:
    """Build a minimal Settings for tests — no real Slack/LLM/DB required."""
    base: dict[str, Any] = {
        "SLACK_BOT_TOKEN": "xoxb-test",
        "SLACK_APP_TOKEN": "xapp-test",
        "SLACK_SIGNING_SECRET": "test-secret",
        "LLM_PROVIDER": "openai",
        "LLM_API_KEY": "sk-test",
        "LLM_MODEL": "gpt-4o",
        "POSTGRES_URL": "postgresql://test:test@localhost/test",
        "COMPOSIO_API_KEY": "composio-test",
        **{k.upper(): v for k, v in overrides.items()},
    }
    return Settings.model_validate(base)


@dataclass
class FakeSandboxSession:
    """In-memory session: write/read operate on a dict; exec returns a preset result."""

    files: dict[str, bytes] = field(default_factory=dict)
    exec_result: SandboxExecResult = field(
        default_factory=lambda: SandboxExecResult(exit_code=0, stdout="done", stderr="")
    )
    exec_calls: list[str] = field(default_factory=list)

    def write_file(self, session_id: str, path: str, content: bytes) -> int:
        self.files[path] = content
        return len(content)

    def read_file(self, session_id: str, path: str) -> bytes:
        if path not in self.files:
            raise FileNotFoundError(f"No file at {path}")
        return self.files[path]

    def exec(
        self,
        session_id: str,
        command: str,
        *,
        workdir: str = "/workspace",
        timeout_seconds: int = 120,
    ) -> SandboxExecResult:
        self.exec_calls.append(command)
        return self.exec_result

    def open_session(self, task_id: str, profile: str = "workbench") -> Any:
        raise NotImplementedError("FakeSandboxSession.open_session not needed")

    def export_archive(self, session_id: str, path: str) -> bytes:
        raise NotImplementedError("FakeSandboxSession.export_archive not needed")

    def close_session(self, session_id: str) -> None:
        pass


def _broker(
    *,
    allowed: frozenset[str] | None = None,
    nonce: str = "test-nonce",
    dispatch: Any = None,
    max_calls: int = 50,
    max_arg_bytes: int = 65536,
    max_result_bytes: int = 262144,
) -> CodeActRpcBroker:
    if allowed is None:
        allowed = frozenset({"tool_a", "tool_b"})
    if dispatch is None:
        dispatch = lambda name, args: {"echo": args}  # noqa: E731
    settings = _settings(
        KORTNY_CODEACT_MAX_CALLS=max_calls,
        KORTNY_CODEACT_MAX_ARG_BYTES=max_arg_bytes,
        KORTNY_CODEACT_MAX_RESULT_BYTES=max_result_bytes,
    )
    return CodeActRpcBroker(
        allowed_tools=allowed,
        nonce=nonce,
        settings=settings,
        dispatch=dispatch,
    )


# ---------------------------------------------------------------------------
# generate_stub_module tests
# ---------------------------------------------------------------------------


class TestGenerateStubModule:
    def test_produces_valid_python(self) -> None:
        stubs = [
            ToolStubSpec(name="composio_linear_list_issues", description="List issues"),
            ToolStubSpec(name="composio_github_create_pr", description="Create a PR"),
        ]
        src = generate_stub_module(stubs, nonce="abc123", run_id="run-xyz")
        compile(src, "<stub>", "exec")  # raises SyntaxError on invalid Python

    def test_one_function_per_stub(self) -> None:
        stubs = [
            ToolStubSpec(name="tool_one", description="Does one thing"),
            ToolStubSpec(name="tool_two", description="Does two things"),
        ]
        src = generate_stub_module(stubs, nonce="n", run_id="r")
        assert "def tool_one" in src
        assert "def tool_two" in src

    def test_nonce_embedded_in_stub(self) -> None:
        src = generate_stub_module([], nonce="secret-nonce-42", run_id="r")
        assert "secret-nonce-42" in src

    def test_run_id_embedded(self) -> None:
        src = generate_stub_module([], nonce="n", run_id="run-abc-123")
        assert "run-abc-123" in src

    def test_call_helper_present(self) -> None:
        src = generate_stub_module([], nonce="n", run_id="r")
        assert "def _call" in src

    def test_function_docstring_uses_description(self) -> None:
        stubs = [ToolStubSpec(name="my_tool", description="My tool description")]
        src = generate_stub_module(stubs, nonce="n", run_id="r")
        assert "My tool description" in src

    def test_tool_name_with_special_chars_produces_safe_py_name(self) -> None:
        stubs = [ToolStubSpec(name="composio_linear_list-issues", description="desc")]
        src = generate_stub_module(stubs, nonce="n", run_id="r")
        compile(src, "<stub>", "exec")
        # The py_name must be a safe identifier
        stub = stubs[0]
        assert stub.py_name.isidentifier()

    def test_empty_stubs_still_valid_python(self) -> None:
        src = generate_stub_module([], nonce="n", run_id="r")
        compile(src, "<stub>", "exec")

    def test_blank_nonce_raises(self) -> None:
        with pytest.raises(ValueError, match="nonce"):
            generate_stub_module([], nonce="  ", run_id="r")

    def test_blank_run_id_raises(self) -> None:
        with pytest.raises(ValueError, match="run_id"):
            generate_stub_module([], nonce="n", run_id="  ")


# ---------------------------------------------------------------------------
# ToolStubSpec tests
# ---------------------------------------------------------------------------


class TestToolStubSpec:
    def test_auto_derives_py_name(self) -> None:
        s = ToolStubSpec(name="composio_linear_list_issues", description="d")
        assert s.py_name == "composio_linear_list_issues"

    def test_explicit_py_name_kept(self) -> None:
        s = ToolStubSpec(
            name="composio_linear_list_issues", description="d", py_name="custom_fn"
        )
        assert s.py_name == "custom_fn"

    def test_hyphens_replaced_in_py_name(self) -> None:
        s = ToolStubSpec(name="my-tool-name", description="d")
        assert s.py_name == "my_tool_name"
        assert s.py_name.isidentifier()

    def test_blank_name_raises(self) -> None:
        with pytest.raises(ValueError):
            ToolStubSpec(name="  ", description="d")


# ---------------------------------------------------------------------------
# CodeActRpcBroker.handle_request — security-critical tests
# ---------------------------------------------------------------------------


class TestBrokerHandleRequest:
    def test_allowed_tool_dispatched_ok(self) -> None:
        mock_dispatch = MagicMock(return_value={"issues": []})
        b = _broker(allowed=frozenset({"tool_a"}), dispatch=mock_dispatch)
        req = {"seq": 0, "tool": "tool_a", "args": {"q": 1}, "nonce": "test-nonce"}
        resp = b.handle_request(req)
        assert resp["ok"] is True
        assert resp["result"] == {"issues": []}
        mock_dispatch.assert_called_once_with("tool_a", {"q": 1})
        assert b.rpc_call_count == 1
        assert b.rpc_error_count == 0

    def test_non_allowlisted_tool_rejected_and_dispatch_not_called(self) -> None:
        mock_dispatch = MagicMock()
        b = _broker(allowed=frozenset({"tool_a"}), dispatch=mock_dispatch)
        req = {"seq": 1, "tool": "evil_tool", "args": {}, "nonce": "test-nonce"}
        resp = b.handle_request(req)
        assert resp["ok"] is False
        assert "not allowed" in resp["error"]
        mock_dispatch.assert_not_called()
        assert b.rpc_call_count == 0
        assert b.rpc_error_count == 1

    def test_wrong_nonce_rejected_dispatch_not_called(self) -> None:
        mock_dispatch = MagicMock()
        b = _broker(nonce="correct-nonce", dispatch=mock_dispatch)
        req = {"seq": 2, "tool": "tool_a", "args": {}, "nonce": "wrong-nonce"}
        resp = b.handle_request(req)
        assert resp["ok"] is False
        assert "nonce" in resp["error"]
        mock_dispatch.assert_not_called()
        assert b.rpc_error_count == 1

    def test_missing_nonce_rejected(self) -> None:
        mock_dispatch = MagicMock()
        b = _broker(dispatch=mock_dispatch)
        req = {"seq": 3, "tool": "tool_a", "args": {}}
        resp = b.handle_request(req)
        assert resp["ok"] is False
        mock_dispatch.assert_not_called()

    def test_arg_bytes_over_cap_rejected(self) -> None:
        mock_dispatch = MagicMock()
        b = _broker(
            allowed=frozenset({"tool_a"}), dispatch=mock_dispatch, max_arg_bytes=10
        )
        big_args = {"data": "x" * 100}
        req = {"seq": 4, "tool": "tool_a", "args": big_args, "nonce": "test-nonce"}
        resp = b.handle_request(req)
        assert resp["ok"] is False
        assert "max_arg_bytes" in resp["error"]
        mock_dispatch.assert_not_called()
        assert b.rpc_error_count == 1

    def test_result_bytes_over_cap_truncated(self) -> None:
        big_result = {"data": "x" * 1000}
        b = _broker(
            allowed=frozenset({"tool_a"}),
            dispatch=lambda n, a: big_result,
            max_result_bytes=50,
        )
        req = {"seq": 5, "tool": "tool_a", "args": {}, "nonce": "test-nonce"}
        resp = b.handle_request(req)
        assert resp["ok"] is True
        assert resp.get("_result_truncated") is True
        # The result should contain truncation metadata
        assert isinstance(resp["result"], dict)
        assert resp["result"].get("_truncated") is True

    def test_call_count_over_max_calls_rejected(self) -> None:
        b = _broker(allowed=frozenset({"tool_a"}), max_calls=2)

        def req(seq: int) -> dict[str, Any]:
            return {"seq": seq, "tool": "tool_a", "args": {}, "nonce": "test-nonce"}

        # First two calls succeed
        b.handle_request(req(0))
        b.handle_request(req(1))
        assert b.rpc_call_count == 2
        # Third call is over the cap
        resp = b.handle_request(req(2))
        assert resp["ok"] is False
        assert "max_calls" in resp["error"]
        # Dispatch should not have been called a 3rd time
        assert b.rpc_call_count == 2
        assert b.rpc_error_count == 1

    def test_dispatch_exception_returns_error_and_increments_rpc_error(self) -> None:
        def bad_dispatch(name: str, args: dict[str, Any]) -> object:
            raise RuntimeError("external API exploded")

        b = _broker(allowed=frozenset({"tool_a"}), dispatch=bad_dispatch)
        req = {"seq": 6, "tool": "tool_a", "args": {}, "nonce": "test-nonce"}
        resp = b.handle_request(req)
        assert resp["ok"] is False
        assert "dispatch error" in resp["error"]
        assert b.rpc_error_count == 1
        # rpc_call_count is still incremented (the call was attempted)
        assert b.rpc_call_count == 1

    def test_seq_preserved_in_response(self) -> None:
        b = _broker(allowed=frozenset({"tool_a"}))
        req = {"seq": 42, "tool": "tool_a", "args": {}, "nonce": "test-nonce"}
        resp = b.handle_request(req)
        assert resp["seq"] == 42

    def test_multiple_allowed_tools(self) -> None:
        results: list[str] = []

        def dispatch(name: str, args: dict[str, Any]) -> object:
            results.append(name)
            return {"called": name}

        b = _broker(allowed=frozenset({"tool_a", "tool_b"}), dispatch=dispatch)
        b.handle_request(
            {"seq": 0, "tool": "tool_a", "args": {}, "nonce": "test-nonce"}
        )
        b.handle_request(
            {"seq": 1, "tool": "tool_b", "args": {}, "nonce": "test-nonce"}
        )
        assert results == ["tool_a", "tool_b"]
        assert b.rpc_call_count == 2


# ---------------------------------------------------------------------------
# run_codeact with fake session
# ---------------------------------------------------------------------------


class TestRunCodeact:
    def _make_fake_session_with_requests(
        self,
        run_id: str,
        nonce: str,
        requests: list[dict[str, Any]],
        exec_result: SandboxExecResult | None = None,
    ) -> FakeSandboxSession:
        """Build a FakeSandboxSession pre-loaded with request files."""
        session = FakeSandboxSession(
            exec_result=exec_result
            or SandboxExecResult(exit_code=0, stdout="script done", stderr="")
        )
        for req in requests:
            path = f"/workspace/.kortny_rpc/{run_id}/requests/{req['seq']}.json"
            session.files[path] = json.dumps(req).encode()
        return session

    def test_returns_codeact_result_type(self) -> None:
        settings = _settings()
        session = FakeSandboxSession()
        result = run_codeact(
            session,
            session_id="s1",
            code="print('hello')",
            stubs=[],
            allowed_tools=frozenset(),
            dispatch=lambda n, a: {},
            settings=settings,
            nonce="n1",
            run_id="r1",
        )
        assert isinstance(result, CodeActResult)

    def test_stub_and_main_written_to_session(self) -> None:
        settings = _settings()
        session = FakeSandboxSession()
        stubs = [ToolStubSpec(name="tool_x", description="does X")]
        run_codeact(
            session,
            session_id="s1",
            code="import kortny_tools",
            stubs=stubs,
            allowed_tools=frozenset({"tool_x"}),
            dispatch=lambda n, a: {},
            settings=settings,
            nonce="n1",
            run_id="r1",
        )
        assert "/workspace/kortny_tools.py" in session.files
        assert "/workspace/main.py" in session.files
        stub_src = session.files["/workspace/kortny_tools.py"].decode()
        assert "def tool_x" in stub_src

    def test_two_rpc_requests_processed(self) -> None:
        nonce = "test-nonce"
        run_id = "run-test"
        settings = _settings()

        dispatched: list[str] = []

        def dispatch(name: str, args: dict[str, Any]) -> object:
            dispatched.append(name)
            return {"ok": True}

        requests = [
            {"seq": 0, "tool": "tool_a", "args": {}, "nonce": nonce},
            {"seq": 1, "tool": "tool_b", "args": {}, "nonce": nonce},
        ]
        session = self._make_fake_session_with_requests(run_id, nonce, requests)

        result = run_codeact(
            session,
            session_id="s1",
            code="print('done')",
            stubs=[],
            allowed_tools=frozenset({"tool_a", "tool_b"}),
            dispatch=dispatch,
            settings=settings,
            nonce=nonce,
            run_id=run_id,
        )

        assert result.rpc_call_count == 2
        assert result.rpc_error_count == 0
        assert dispatched == ["tool_a", "tool_b"]

    def test_response_files_written_back(self) -> None:
        nonce = "resp-nonce"
        run_id = "run-resp"
        settings = _settings()

        requests = [{"seq": 0, "tool": "tool_a", "args": {"x": 1}, "nonce": nonce}]
        session = self._make_fake_session_with_requests(run_id, nonce, requests)

        run_codeact(
            session,
            session_id="s1",
            code="pass",
            stubs=[],
            allowed_tools=frozenset({"tool_a"}),
            dispatch=lambda n, a: {"data": "result"},
            settings=settings,
            nonce=nonce,
            run_id=run_id,
        )

        resp_path = f"/workspace/.kortny_rpc/{run_id}/responses/0.json"
        assert resp_path in session.files
        resp = json.loads(session.files[resp_path])
        assert resp["ok"] is True
        assert resp["result"] == {"data": "result"}

    def test_exec_called_for_main_py(self) -> None:
        settings = _settings()
        session = FakeSandboxSession()
        run_codeact(
            session,
            session_id="s1",
            code="pass",
            stubs=[],
            allowed_tools=frozenset(),
            dispatch=lambda n, a: {},
            settings=settings,
            nonce="n",
            run_id="r",
        )
        # Should have called exec for main.py (the mkdir call may also be there)
        assert any("main.py" in cmd for cmd in session.exec_calls)

    def test_exit_code_propagated(self) -> None:
        settings = _settings()
        session = FakeSandboxSession(
            exec_result=SandboxExecResult(exit_code=1, stdout="", stderr="error!")
        )
        result = run_codeact(
            session,
            session_id="s1",
            code="raise SystemExit(1)",
            stubs=[],
            allowed_tools=frozenset(),
            dispatch=lambda n, a: {},
            settings=settings,
            nonce="n",
            run_id="r",
        )
        assert result.successful is False
        assert result.exit_code == 1
        assert result.stderr == "error!"

    def test_stdout_propagated(self) -> None:
        settings = _settings()
        session = FakeSandboxSession(
            exec_result=SandboxExecResult(exit_code=0, stdout="42\n", stderr="")
        )
        result = run_codeact(
            session,
            session_id="s1",
            code="print(42)",
            stubs=[],
            allowed_tools=frozenset(),
            dispatch=lambda n, a: {},
            settings=settings,
            nonce="n",
            run_id="r",
        )
        assert result.successful is True
        assert result.stdout == "42\n"
