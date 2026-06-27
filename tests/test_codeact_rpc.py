"""Unit tests for kortny/execution/codeact_rpc.py — Slice B (single-threaded).

All tests use a FAKE in-memory session (no real container, no LLM, no network).
The broker security tests are exhaustive: nonce, allowlist, byte caps, call-count,
dispatch exceptions.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from kortny.config import Settings
from kortny.execution.codeact_rpc import (
    CodeActResult,
    CodeActRpcBroker,
    ToolStubSpec,
    _scrub_rpc_result,
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
        # Short timeout so tests don't hang if something goes wrong
        "KORTNY_CODEACT_TIMEOUT_SECONDS": "2",
        **{k.upper(): v for k, v in overrides.items()},
    }
    return Settings.model_validate(base)


@dataclass
class FakeSandboxSession:
    """In-memory session: write/read operate on a dict; exec returns a preset result.

    When exec() receives a command containing "nohup" (the detached launch
    command), it extracts the run_id and pre-writes the exit_code, stdout.log,
    and stderr.log files so the done-marker is immediately visible to the poll
    loop — making tests fast without hitting the timeout.
    """

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
        # Simulate the detached launch: pre-write done-marker files so the poll
        # loop terminates quickly instead of waiting for the full timeout.
        if "nohup" in command:
            m = re.search(r"\.kortny_rpc/([^/\s&]+)", command)
            if m:
                run_id = m.group(1)
                self.files[f"/workspace/.kortny_rpc/{run_id}/exit_code"] = str(
                    self.exec_result.exit_code
                ).encode()
                self.files[f"/workspace/.kortny_rpc/{run_id}/stdout.log"] = (
                    self.exec_result.stdout.encode()
                )
                self.files[f"/workspace/.kortny_rpc/{run_id}/stderr.log"] = (
                    self.exec_result.stderr.encode()
                )
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

    def test_no_broker_lock_attribute(self) -> None:
        """Slice B: CodeActRpcBroker must NOT have a _broker_lock attribute."""
        b = _broker()
        assert not hasattr(b, "_broker_lock"), (
            "CodeActRpcBroker._broker_lock must be removed in Slice B "
            "(single-threaded model needs no lock)"
        )


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
        assert "/workspace/.kortny_rpc/r1/kortny_tools.py" in session.files
        assert "/workspace/.kortny_rpc/r1/main.py" in session.files
        stub_src = session.files["/workspace/.kortny_rpc/r1/kortny_tools.py"].decode()
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
        # The launch command must contain "nohup", the per-run rpc dir, and "main.py".
        # The launch uses "cd /workspace/.kortny_rpc/{run_id} && nohup sh -c 'python main.py'"
        # so main.py is relative (no full path), but the cd path contains the run_id.
        assert any(
            "nohup" in cmd and ".kortny_rpc/r" in cmd and "main.py" in cmd
            for cmd in session.exec_calls
        )

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


# ---------------------------------------------------------------------------
# F6: Single-threaded poll loop (Slice B)
# ---------------------------------------------------------------------------


class TestF6SingleThreadedLoop:
    """Slice B: the poll loop is single-threaded — no threading.Thread is ever
    spawned inside run_codeact, and pre-seeded requests are dispatched by the
    poll loop before the done-marker terminates it."""

    def test_single_threaded_poll_dispatches_during_run(self) -> None:
        """Pre-seeded requests are dispatched by the poll loop on the main thread."""
        nonce = "poll-nonce"
        run_id = "poll-run"
        settings = _settings()

        dispatched: list[str] = []

        def dispatch(name: str, args: dict[str, Any]) -> object:
            dispatched.append(name)
            return {"ok": True}

        requests = [
            {"seq": 0, "tool": "tool_a", "args": {}, "nonce": nonce},
            {"seq": 1, "tool": "tool_b", "args": {}, "nonce": nonce},
        ]
        session = FakeSandboxSession(
            exec_result=SandboxExecResult(exit_code=0, stdout="done", stderr="")
        )
        for req in requests:
            path = f"/workspace/.kortny_rpc/{run_id}/requests/{req['seq']}.json"
            session.files[path] = json.dumps(req).encode()

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
        assert set(dispatched) == {"tool_a", "tool_b"}

    def test_no_thread_spawned_during_run(self) -> None:
        """Slice B: threading.Thread must NEVER be instantiated inside run_codeact."""
        settings = _settings()
        session = FakeSandboxSession()

        with patch("threading.Thread") as mock_thread_cls:
            run_codeact(
                session,
                session_id="s1",
                code="pass",
                stubs=[],
                allowed_tools=frozenset(),
                dispatch=lambda n, a: {},
                settings=settings,
                nonce="n",
                run_id="thread-test",
            )
            mock_thread_cls.assert_not_called()

    def test_done_marker_stops_poll_loop(self) -> None:
        """The poll loop must stop when the exit_code file is present."""
        nonce = "done-nonce"
        run_id = "done-run"
        settings = _settings()

        dispatched: list[str] = []

        def dispatch(name: str, args: dict[str, Any]) -> object:
            dispatched.append(name)
            return {"ok": True}

        session = FakeSandboxSession(
            exec_result=SandboxExecResult(exit_code=0, stdout="done", stderr="")
        )
        # Pre-seed request at seq 0 and the done marker simultaneously.
        # The poll loop should process the request and then see the done marker.
        req = {"seq": 0, "tool": "tool_a", "args": {}, "nonce": nonce}
        session.files[f"/workspace/.kortny_rpc/{run_id}/requests/0.json"] = json.dumps(
            req
        ).encode()

        result = run_codeact(
            session,
            session_id="s1",
            code="pass",
            stubs=[],
            allowed_tools=frozenset({"tool_a"}),
            dispatch=dispatch,
            settings=settings,
            nonce=nonce,
            run_id=run_id,
        )

        # Requests processed before or at done marker.
        assert result.rpc_call_count >= 0
        # Loop terminated (did not time out the full 2s).
        assert result.timed_out is False

    def test_nonzero_exit_with_no_requests(self) -> None:
        """With no requests seeded, rpc counts are 0 even on non-zero exit."""
        settings = _settings()
        session = FakeSandboxSession(
            exec_result=SandboxExecResult(exit_code=1, stdout="", stderr="error")
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

        assert result.rpc_error_count == 0
        assert result.rpc_call_count == 0
        assert result.successful is False

    def test_partial_read_retry_advances_after_max_retries(self) -> None:
        """A request file with invalid JSON retries up to _MAX_PARTIAL_RETRIES then
        is treated as an error and the seq advances."""
        nonce = "partial-nonce"
        run_id = "partial-run"
        settings = _settings()

        # Pre-seed seq 0 with invalid JSON so parsing always fails.
        session = FakeSandboxSession(
            exec_result=SandboxExecResult(exit_code=0, stdout="done", stderr="")
        )
        session.files[f"/workspace/.kortny_rpc/{run_id}/requests/0.json"] = (
            b"not valid json{"
        )

        dispatched: list[str] = []

        def dispatch(name: str, args: dict[str, Any]) -> object:
            dispatched.append(name)
            return {"ok": True}

        result = run_codeact(
            session,
            session_id="s1",
            code="pass",
            stubs=[],
            allowed_tools=frozenset({"tool_a"}),
            dispatch=dispatch,
            settings=settings,
            nonce=nonce,
            run_id=run_id,
        )

        # No tool should have been dispatched (bad JSON can't be parsed).
        assert len(dispatched) == 0
        # The partial-read path must have incremented rpc_error_count once.
        assert result.rpc_error_count >= 1

    def test_f8_scrub_wired_in_poll_loop(self) -> None:
        """F8: connected_account_id in the dispatch result must be scrubbed in the
        response file written to the sandbox (poll loop applies _scrub_rpc_result)."""
        nonce = "scrub-nonce"
        run_id = "scrub-run"
        settings = _settings()

        def dispatch(name: str, args: dict[str, Any]) -> object:
            return {"connected_account_id": "acct-secret-123", "data": "ok"}

        session = FakeSandboxSession(
            exec_result=SandboxExecResult(exit_code=0, stdout="done", stderr="")
        )
        req = {"seq": 0, "tool": "tool_a", "args": {}, "nonce": nonce}
        session.files[f"/workspace/.kortny_rpc/{run_id}/requests/0.json"] = json.dumps(
            req
        ).encode()

        run_codeact(
            session,
            session_id="s1",
            code="pass",
            stubs=[],
            allowed_tools=frozenset({"tool_a"}),
            dispatch=dispatch,
            settings=settings,
            nonce=nonce,
            run_id=run_id,
        )

        resp_path = f"/workspace/.kortny_rpc/{run_id}/responses/0.json"
        assert resp_path in session.files
        resp = json.loads(session.files[resp_path])
        assert resp["ok"] is True
        result = resp["result"]
        assert isinstance(result, dict)
        # connected_account_id must be scrubbed to "[redacted]"
        assert result.get("connected_account_id") == "[redacted]", (
            f"Expected [redacted] but got {result.get('connected_account_id')!r}"
        )
        # Non-sensitive key must be preserved
        assert result.get("data") == "ok"


# ---------------------------------------------------------------------------
# F5: Per-run file paths
# ---------------------------------------------------------------------------


class TestF5PerRunPaths:
    """Files must be written to per-run paths, not fixed /workspace names."""

    def test_per_run_file_paths(self) -> None:
        """run_codeact must write to /workspace/.kortny_rpc/{run_id}/ paths."""
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
            run_id="testrun",
        )
        assert "/workspace/.kortny_rpc/testrun/kortny_tools.py" in session.files
        assert "/workspace/.kortny_rpc/testrun/main.py" in session.files
        # Must NOT write to old fixed paths
        assert "/workspace/kortny_tools.py" not in session.files
        assert "/workspace/main.py" not in session.files

    def test_exec_uses_per_run_path(self) -> None:
        """The exec launch command must reference the per-run main.py path."""
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
            run_id="testrun",
        )
        # The launch command uses nohup and cds to the per-run rpc dir; main.py is relative.
        # Check that the nohup launch command references both the testrun rpc dir and main.py.
        assert any(
            "nohup" in cmd and ".kortny_rpc/testrun" in cmd and "main.py" in cmd
            for cmd in session.exec_calls
        )

    def test_concurrent_runs_no_collision(self) -> None:
        """Two runs with different run_ids must write to separate directories."""
        settings = _settings()
        session1 = FakeSandboxSession()
        session2 = FakeSandboxSession()

        run_codeact(
            session1,
            session_id="s1",
            code="print('run1')",
            stubs=[],
            allowed_tools=frozenset(),
            dispatch=lambda n, a: {},
            settings=settings,
            nonce="n1",
            run_id="run1",
        )
        run_codeact(
            session2,
            session_id="s2",
            code="print('run2')",
            stubs=[],
            allowed_tools=frozenset(),
            dispatch=lambda n, a: {},
            settings=settings,
            nonce="n2",
            run_id="run2",
        )

        assert "/workspace/.kortny_rpc/run1/main.py" in session1.files
        assert "/workspace/.kortny_rpc/run2/main.py" in session2.files
        assert "/workspace/.kortny_rpc/run2/main.py" not in session1.files
        assert "/workspace/.kortny_rpc/run1/main.py" not in session2.files


# ---------------------------------------------------------------------------
# F8 scrub: standalone scrub helper tests
# ---------------------------------------------------------------------------


class TestF8ScrubHelpers:
    """_scrub_rpc_result must handle snake_case and camelCase secret keys."""

    def test_snake_case_connected_account_id_scrubbed(self) -> None:
        result = _scrub_rpc_result({"connected_account_id": "acct-xyz"})
        assert isinstance(result, dict)
        assert result["connected_account_id"] == "[redacted]"

    def test_camelcase_connectedaccountid_scrubbed(self) -> None:
        result = _scrub_rpc_result({"connectedAccountId": "abc"})
        assert isinstance(result, dict)
        assert result["connectedAccountId"] == "[redacted]"

    def test_non_sensitive_key_preserved(self) -> None:
        result = _scrub_rpc_result({"data": "ok", "name": "test"})
        assert isinstance(result, dict)
        assert result["data"] == "ok"
        assert result["name"] == "test"
