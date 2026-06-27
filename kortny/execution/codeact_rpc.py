"""CodeAct RPC bridge engine — stub generator, file protocol, broker, orchestrator.

This module is INERT in Slice A: it is not imported by any agent, coordinator,
executor, or tool registry. Security gates and reachability land in Slice B.
The flag KORTNY_CODEACT_ENABLED defaults False.
"""

from __future__ import annotations

import contextlib
import json
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from kortny.config import Settings
from kortny.execution.sandbox_sessions import SandboxSessionClient

# ---------------------------------------------------------------------------
# Wire types
# ---------------------------------------------------------------------------

#: Request written by the in-sandbox stub:
#:   {"seq": int, "tool": str, "args": object, "nonce": str}
#:
#: Response written by the host broker:
#:   {"seq": int, "ok": bool, "result": object | null, "error": str | null}

_RPC_REQUEST_DIR = ".kortny_rpc/{run_id}/requests"
_RPC_RESPONSE_DIR = ".kortny_rpc/{run_id}/responses"
_WORKSPACE = "/workspace"


def _rpc_request_path(run_id: str, seq: int) -> str:
    return f"{_WORKSPACE}/.kortny_rpc/{run_id}/requests/{seq}.json"


def _rpc_response_path(run_id: str, seq: int) -> str:
    return f"{_WORKSPACE}/.kortny_rpc/{run_id}/responses/{seq}.json"


# ---------------------------------------------------------------------------
# ToolStubSpec
# ---------------------------------------------------------------------------


def _safe_py_name(name: str) -> str:
    """Derive a safe Python identifier from a tool name.

    Replaces any character that isn't alphanumeric or underscore with '_',
    then ensures the result doesn't start with a digit.
    """
    safe = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    if safe and safe[0].isdigit():
        safe = "_" + safe
    return safe or "_tool"


@dataclass(frozen=True, slots=True)
class ToolStubSpec:
    """Specification for one tool stub to generate inside kortny_tools.py."""

    name: str  # e.g. "composio_linear_list_issues"
    description: str
    py_name: str = field(default="")  # safe Python identifier; auto-derived if blank

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("ToolStubSpec.name must be non-empty")
        if not self.py_name:
            object.__setattr__(self, "py_name", _safe_py_name(self.name))


# ---------------------------------------------------------------------------
# Stub module generator
# ---------------------------------------------------------------------------

_STUB_HEADER_TEMPLATE = '''\
"""kortny_tools — auto-generated stub library for CodeAct.

DO NOT EDIT. Generated per-run by kortny/execution/codeact_rpc.py.
run_id : {run_id}
nonce  : {nonce}
"""

from __future__ import annotations

import json
import os
import time

_NONCE = {nonce_repr}
_RUN_ID = {run_id_repr}
_REQUEST_DIR = "/workspace/.kortny_rpc/{run_id}/requests"
_RESPONSE_DIR = "/workspace/.kortny_rpc/{run_id}/responses"
_POLL_INTERVAL = 0.05   # seconds between response-file polls
_POLL_TIMEOUT  = 30.0   # seconds before a single call is considered hung

_seq_counter = [0]


def _call(tool_name: str, args: dict) -> object:
    """Write a request file and block until the host broker writes the response."""
    seq = _seq_counter[0]
    _seq_counter[0] += 1

    os.makedirs(_REQUEST_DIR, exist_ok=True)
    os.makedirs(_RESPONSE_DIR, exist_ok=True)

    request = {{"seq": seq, "tool": tool_name, "args": args, "nonce": _NONCE}}
    req_path = os.path.join(_REQUEST_DIR, f"{{seq}}.json")
    resp_path = os.path.join(_RESPONSE_DIR, f"{{seq}}.json")

    with open(req_path, "w", encoding="utf-8") as fh:
        json.dump(request, fh)

    deadline = time.monotonic() + _POLL_TIMEOUT
    while time.monotonic() < deadline:
        if os.path.exists(resp_path):
            with open(resp_path, "r", encoding="utf-8") as fh:
                response = json.load(fh)
            if not response.get("ok"):
                raise RuntimeError(
                    f"Tool {{tool_name!r}} failed: {{response.get('error', 'unknown error')}}"
                )
            return response.get("result")
        time.sleep(_POLL_INTERVAL)

    raise TimeoutError(f"No response for tool {{tool_name!r}} seq={{seq}} within {{_POLL_TIMEOUT}}s")

'''

_STUB_FUNC_TEMPLATE = '''\
def {py_name}(**kwargs: object) -> object:
    """{description}"""
    return _call({name_repr}, kwargs)

'''


def generate_stub_module(
    stubs: Sequence[ToolStubSpec],
    *,
    nonce: str,
    run_id: str,
) -> str:
    """Return the source of kortny_tools.py for the given stub set.

    The returned string is valid Python — compile() it to verify.
    Each stub becomes one ``def <py_name>(**kwargs)`` function that
    delegates to the file-RPC ``_call`` helper.
    """
    if not nonce.strip():
        raise ValueError("nonce must be non-empty")
    if not run_id.strip():
        raise ValueError("run_id must be non-empty")

    header = _STUB_HEADER_TEMPLATE.format(
        run_id=run_id,
        nonce=nonce,
        nonce_repr=repr(nonce),
        run_id_repr=repr(run_id),
    )

    parts = [header]
    for stub in stubs:
        parts.append(
            _STUB_FUNC_TEMPLATE.format(
                py_name=stub.py_name,
                description=stub.description.replace('"""', "'''"),
                name_repr=repr(stub.name),
            )
        )

    return "".join(parts)


# ---------------------------------------------------------------------------
# CodeAct result shape
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CodeActResult:
    """Outcome of one codeact_exec run — extends the code_exec result shape."""

    successful: bool
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: int | None
    timed_out: bool
    truncated: bool
    rpc_call_count: int
    rpc_error_count: int


# ---------------------------------------------------------------------------
# Security-critical broker
# ---------------------------------------------------------------------------


class CodeActRpcBroker:
    """Host-side broker: validate + dispatch RPC requests from the sandbox script.

    Security checks (in order):
    1. Nonce match — prevents cross-run forgery.
    2. Tool allowlist — prevents the script from calling un-approved tools.
    3. Arg byte cap — prevents oversized payloads.
    4. Call count cap — prevents floods.
    Then dispatch via the injected ``dispatch`` callable; cap the result bytes.
    """

    def __init__(
        self,
        *,
        allowed_tools: frozenset[str],
        nonce: str,
        settings: Settings,
        dispatch: Callable[[str, dict[str, Any]], object],
    ) -> None:
        if not nonce.strip():
            raise ValueError("nonce must be non-empty")
        self._allowed_tools = allowed_tools
        self._nonce = nonce
        self._settings = settings
        self._dispatch = dispatch
        self.rpc_call_count: int = 0
        self.rpc_error_count: int = 0

    def handle_request(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Validate, dispatch, and return a response dict.

        This method NEVER raises — all errors are returned as
        ``{"ok": False, "error": "<msg>"}`` responses so the script
        receives a structured failure rather than a host-side exception.
        """
        seq = raw.get("seq", 0)

        # 1. Nonce check
        if raw.get("nonce") != self._nonce:
            self.rpc_error_count += 1
            return {"seq": seq, "ok": False, "result": None, "error": "nonce mismatch"}

        tool = raw.get("tool", "")
        args = raw.get("args", {})
        if not isinstance(args, dict):
            args = {}

        # 2. Allowlist check
        if tool not in self._allowed_tools:
            self.rpc_error_count += 1
            return {
                "seq": seq,
                "ok": False,
                "result": None,
                "error": f"tool not allowed: {tool!r}",
            }

        # 3. Arg bytes cap
        try:
            args_json = json.dumps(args)
        except (TypeError, ValueError) as exc:
            self.rpc_error_count += 1
            return {
                "seq": seq,
                "ok": False,
                "result": None,
                "error": f"args serialisation failed: {exc}",
            }
        if len(args_json.encode()) > self._settings.codeact_max_arg_bytes:
            self.rpc_error_count += 1
            return {
                "seq": seq,
                "ok": False,
                "result": None,
                "error": (
                    f"args exceed max_arg_bytes ({self._settings.codeact_max_arg_bytes})"
                ),
            }

        # 4. Call-count cap (checked BEFORE dispatch so an over-limit call is not run)
        if self.rpc_call_count >= self._settings.codeact_max_calls:
            self.rpc_error_count += 1
            return {
                "seq": seq,
                "ok": False,
                "result": None,
                "error": (f"max_calls ({self._settings.codeact_max_calls}) exceeded"),
            }

        # Dispatch
        self.rpc_call_count += 1
        try:
            result = self._dispatch(tool, args)
        except Exception as exc:  # noqa: BLE001
            self.rpc_error_count += 1
            # Sanitise: only expose the exception type + a short message
            return {
                "seq": seq,
                "ok": False,
                "result": None,
                "error": f"dispatch error: {type(exc).__name__}: {str(exc)[:200]}",
            }

        # Cap result bytes
        try:
            result_json = json.dumps(result)
        except (TypeError, ValueError):
            result_json = json.dumps({"_non_serialisable": True})

        max_bytes = self._settings.codeact_max_result_bytes
        result_bytes = result_json.encode()
        truncated_result = False
        if len(result_bytes) > max_bytes:
            truncated_result = True
            result = {
                "_truncated": True,
                "_original_bytes": len(result_bytes),
                "_preview": result_json[: max_bytes // 2],
            }

        return {
            "seq": seq,
            "ok": True,
            "result": result,
            "error": None,
            "_result_truncated": truncated_result,
        }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_codeact(
    session: SandboxSessionClient,
    *,
    session_id: str,
    code: str,
    stubs: Sequence[ToolStubSpec],
    allowed_tools: frozenset[str],
    dispatch: Callable[[str, dict[str, Any]], object],
    settings: Settings,
    nonce: str,
    run_id: str,
) -> CodeActResult:
    """Write files, start the script, run the broker poll loop, return result.

    This is the only function that touches the real SandboxSessionClient.
    Tests inject a fake session client so no real container is needed.

    Steps:
    1. Write kortny_tools.py (stub library) into the workspace.
    2. Write main.py (model's code) into the workspace.
    3. Create the RPC request/response directories (via mkdir -p in sandbox).
    4. Start ``python main.py`` with the configured timeout.
       Because exec() is synchronous and blocks until the process exits, we
       run a broker poll loop CONCURRENTLY using a background thread for the
       exec and a foreground loop for polling.  In practice (tests + sandbox)
       the session is driven synchronously: the fake session pre-loads the
       request files so the poll loop can drain them before exec returns.

    Implementation note: the production path needs concurrent exec + poll.
    For Slice A (engine + unit tests), we use a sequential fake-session model:
    the fake pre-loads requests; run_codeact writes responses then calls exec.
    The real async bridge (thread-per-exec + poll loop) is wired in Slice B
    when the tool is registered and can run against a live container.

    For testability, this function:
    - Writes stub + code files.
    - Drains any pre-loaded request files (for fake sessions in tests).
    - Calls session.exec(...) to run the script.
    - Returns a CodeActResult.
    """
    broker = CodeActRpcBroker(
        allowed_tools=allowed_tools,
        nonce=nonce,
        settings=settings,
        dispatch=dispatch,
    )

    # Write kortny_tools.py
    stub_src = generate_stub_module(stubs, nonce=nonce, run_id=run_id)
    session.write_file(session_id, "/workspace/kortny_tools.py", stub_src.encode())

    # Write main.py
    session.write_file(session_id, "/workspace/main.py", code.encode())

    # Ensure RPC dirs exist (best-effort via exec; ignored in fake sessions)
    req_dir = f"/workspace/.kortny_rpc/{run_id}/requests"
    resp_dir = f"/workspace/.kortny_rpc/{run_id}/responses"
    with contextlib.suppress(Exception):  # noqa: BLE001
        session.exec(
            session_id,
            f"mkdir -p {req_dir} {resp_dir}",
            workdir="/workspace",
            timeout_seconds=5,
        )

    # Pre-execution broker pass: drain any request files already written
    # (supports fake in-memory session tests where requests are pre-seeded).
    _drain_pending_requests(session, session_id, run_id=run_id, broker=broker)

    # Run the script
    start_ms = int(time.monotonic() * 1000)
    exec_result = session.exec(
        session_id,
        "python main.py",
        workdir="/workspace",
        timeout_seconds=settings.codeact_timeout_seconds,
    )
    duration_ms = int(time.monotonic() * 1000) - start_ms

    # Post-execution broker pass: drain any remaining requests the script wrote
    # before exit (handles the case where exec and poll are not concurrent).
    _drain_pending_requests(session, session_id, run_id=run_id, broker=broker)

    return CodeActResult(
        successful=exec_result.exit_code == 0,
        exit_code=exec_result.exit_code,
        stdout=exec_result.stdout,
        stderr=exec_result.stderr,
        duration_ms=exec_result.duration_ms or duration_ms,
        timed_out=exec_result.timed_out,
        truncated=exec_result.truncated,
        rpc_call_count=broker.rpc_call_count,
        rpc_error_count=broker.rpc_error_count,
    )


def _drain_pending_requests(
    session: SandboxSessionClient,
    session_id: str,
    *,
    run_id: str,
    broker: CodeActRpcBroker,
) -> None:
    """Read and handle any pending request files, writing response files back.

    This is a best-effort drain — if a request file cannot be read or parsed,
    we skip it.  Responses are written back via write_file.

    Sequence numbers are discovered by listing files in the request directory;
    we attempt seqs 0..N until a read fails (file not found / session error).
    """
    seq = 0
    while True:
        req_path = _rpc_request_path(run_id, seq)
        try:
            raw_bytes = session.read_file(session_id, req_path)
        except Exception:  # noqa: BLE001
            break  # No more request files

        resp_path = _rpc_response_path(run_id, seq)
        # Skip already-handled requests (idempotent across pre/post drain passes).
        try:
            session.read_file(session_id, resp_path)
            seq += 1
            continue  # Response already written — skip re-dispatch
        except Exception:  # noqa: BLE001
            pass  # No response yet — proceed to dispatch

        try:
            raw = json.loads(raw_bytes.decode())
        except (json.JSONDecodeError, UnicodeDecodeError):
            seq += 1
            continue

        response = broker.handle_request(raw)
        with contextlib.suppress(Exception):  # noqa: BLE001
            session.write_file(session_id, resp_path, json.dumps(response).encode())
        seq += 1
