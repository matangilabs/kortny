"""CodeAct RPC bridge engine — stub generator, file protocol, broker, orchestrator.

Single-threaded poll loop (Slice B): no background broker thread, no
ThreadPoolExecutor inside _rpc_dispatch. The main thread launches the script
detached (nohup & background) and then polls request files + dispatches them
synchronously, interleaved with done-marker checks.
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

# Cap the total requests/errors the poll loop will process before giving up.
_MAX_DRAIN_REQUESTS: int = 200
_MAX_DRAIN_ERRORS: int = 10

# Interval (seconds) between poll iterations when no request file is ready.
_BROKER_POLL_INTERVAL: float = 0.05

# Maximum partial-read retries before treating a seq as errored and advancing.
_MAX_PARTIAL_RETRIES: int = 3

# Keys (lowercased) whose values should be redacted from RPC results before
# sending them back to the sandbox script. Covers both snake_case and the
# camelCase equivalents used by some Composio responses.
_SCRUB_KEYS = frozenset(
    {
        # snake_case (Composio/standard)
        "connected_account_id",
        "auth",
        "authorization",
        "api_key",
        "access_token",
        "refresh_token",
        "token",
        "secret",
        "password",
        "x-api-key",
        "headers",
        # camelCase equivalents (lowercased so `.lower()` comparison works)
        "connectedaccountid",  # connectedAccountId
        "apikey",  # apiKey
        "accesstoken",  # accessToken
        "refreshtoken",  # refreshToken
        "bearertoken",  # bearerToken
        "authtoken",  # authToken
        "clientsecret",  # clientSecret
        "apisecret",  # apiSecret
    }
)


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
# Result scrubbing helpers
# ---------------------------------------------------------------------------


def _scrub_exception_message(msg: str) -> str:
    """Redact secret-looking substrings from an exception message.

    Targets common token/key patterns: Slack xox* tokens, JWT-like strings,
    long base64 blobs, and sk-* API keys.
    """
    return re.sub(
        r"(?i)(sk[-_][a-zA-Z0-9]{10,}|xox[bpar]-[a-zA-Z0-9\-]+|[a-zA-Z0-9+/]{32,}={0,2}|eyJ[a-zA-Z0-9\-_]+\.[a-zA-Z0-9\-_]+)",
        "[redacted]",
        msg,
    )


def _scrub_rpc_result(value: object) -> object:
    """Recursively scrub secret-looking keys/values from an RPC result.

    - Dict keys matching ``_SCRUB_KEYS`` (case-insensitive) have their values
      replaced with ``"[redacted]"``.
    - String values that parse as JSON dicts/lists are scrubbed recursively.
    - Long error-like strings (tracebacks) are truncated.
    """
    if isinstance(value, dict):
        return {
            k: "[redacted]" if k.lower() in _SCRUB_KEYS else _scrub_rpc_result(v)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_scrub_rpc_result(item) for item in value]
    if isinstance(value, str):
        # Try to parse as JSON — if it's a stringified dict/list with secret keys, scrub.
        if len(value) > 10:
            try:
                parsed = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                parsed = None
            if isinstance(parsed, (dict, list)):
                scrubbed = _scrub_rpc_result(parsed)
                try:
                    return json.dumps(scrubbed)
                except (TypeError, ValueError):
                    pass
        if len(value) > 4096 and ("Traceback" in value or "Error:" in value):
            return value[:200] + " [truncated]"
        return value
    return value


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

    This class is ONLY called from the main thread (single-threaded poll loop).
    No locking is needed — counters are simple integer increments.
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
            # Scrub the error string before exposing it: redact secret-looking patterns
            # and don't leak host-internal stack traces.
            raw_error = f"{type(exc).__name__}: {str(exc)[:200]}"
            scrubbed_error = _scrub_exception_message(raw_error)
            return {
                "seq": seq,
                "ok": False,
                "result": None,
                "error": f"dispatch error: {scrubbed_error}",
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
# Done-marker and output-file helpers
# ---------------------------------------------------------------------------


def _check_done_marker(
    session: SandboxSessionClient,
    session_id: str,
    run_id: str,
) -> bool:
    """Return True if the script's exit_code file has been written."""
    try:
        session.read_file(session_id, f"{_WORKSPACE}/.kortny_rpc/{run_id}/exit_code")
        return True
    except Exception:  # noqa: BLE001
        return False


def _read_output_file(
    session: SandboxSessionClient,
    session_id: str,
    path: str,
    *,
    max_bytes: int = 65536,
) -> str:
    """Read a text file from the sandbox, returning "" on any error."""
    try:
        raw = session.read_file(session_id, path)
        return raw[:max_bytes].decode(errors="replace")
    except Exception:  # noqa: BLE001
        return ""


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
    """Write files, launch the script detached, run the single-threaded poll loop.

    Steps:
    1. Write kortny_tools.py (stub library) into the workspace.
    2. Write main.py (model's code) into the workspace.
    3. Create the RPC request/response directories (best-effort exec).
    4. Launch ``python main.py`` DETACHED via nohup so exec() returns immediately
       (the script runs in the background; no thread is spawned).
    5. SINGLE-THREADED POLL LOOP on the main thread: read request files,
       dispatch them synchronously, write response files back, check the
       done-marker (exit_code file) between iterations.
    6. After the loop ends (done marker, deadline, or cap), read
       stdout.log / stderr.log / exit_code from the per-run dir.

    No ``threading`` module is used. No ``ThreadPoolExecutor`` is spawned
    inside this function. The overall ``codeact_timeout_seconds`` deadline
    enforced by the loop replaces per-tool catalog timeouts for RPC calls.

    Test compatibility: FakeSandboxSession.exec() detects the nohup launch
    command and pre-writes exit_code/stdout.log/stderr.log, so the done-marker
    is available immediately after launch and the loop exits fast.
    """
    broker = CodeActRpcBroker(
        allowed_tools=allowed_tools,
        nonce=nonce,
        settings=settings,
        dispatch=dispatch,
    )

    # F5: write to per-run paths under the RPC dir, not fixed /workspace names.
    stub_src = generate_stub_module(stubs, nonce=nonce, run_id=run_id)
    stub_path = f"{_WORKSPACE}/.kortny_rpc/{run_id}/kortny_tools.py"
    main_path = f"{_WORKSPACE}/.kortny_rpc/{run_id}/main.py"
    session.write_file(session_id, stub_path, stub_src.encode())
    session.write_file(session_id, main_path, code.encode())

    # Ensure RPC dirs exist (best-effort via exec; ignored in fake sessions).
    req_dir = f"{_WORKSPACE}/.kortny_rpc/{run_id}/requests"
    resp_dir = f"{_WORKSPACE}/.kortny_rpc/{run_id}/responses"
    with contextlib.suppress(Exception):  # noqa: BLE001
        session.exec(
            session_id,
            f"mkdir -p {req_dir} {resp_dir}",
            workdir=_WORKSPACE,
            timeout_seconds=5,
        )

    # Launch the script detached so this exec() call returns immediately.
    # The script runs under nohup in the background (&); stdout/stderr are
    # redirected to log files; exit code is written to the exit_code file.
    # FakeSandboxSession detects "nohup" in the command and pre-writes
    # exit_code/stdout.log/stderr.log to make tests fast.
    rpc_dir = f"{_WORKSPACE}/.kortny_rpc/{run_id}"
    launch_cmd = (
        f"cd {rpc_dir} && "
        f"nohup sh -c "
        f"'python main.py > stdout.log 2> stderr.log; echo $? > exit_code' "
        f">/dev/null 2>&1 &"
    )
    with contextlib.suppress(Exception):  # noqa: BLE001
        session.exec(session_id, launch_cmd, workdir=_WORKSPACE, timeout_seconds=5)

    # Single-threaded poll loop.
    start_ms = int(time.monotonic() * 1000)
    seq = 0
    partial_read_attempts: dict[int, int] = {}
    deadline = time.monotonic() + settings.codeact_timeout_seconds

    while time.monotonic() < deadline:
        # Check caps.
        if broker.rpc_call_count >= _MAX_DRAIN_REQUESTS:
            break
        if broker.rpc_error_count >= _MAX_DRAIN_ERRORS:
            break

        # Try to read the next request file.
        req_path = _rpc_request_path(run_id, seq)
        try:
            raw_bytes = session.read_file(session_id, req_path)
        except Exception:  # noqa: BLE001
            # No request at this seq yet.
            # Check if the script has finished (done marker) before sleeping —
            # if so, break instead of polling further for a request that will
            # never come. In production the script writes requests BEFORE
            # writing exit_code, so we drain everything first.
            if _check_done_marker(session, session_id, run_id):
                break
            time.sleep(_BROKER_POLL_INTERVAL)
            continue

        # Skip already-responded (idempotent).
        resp_path = _rpc_response_path(run_id, seq)
        try:
            session.read_file(session_id, resp_path)
            seq += 1
            continue
        except Exception:  # noqa: BLE001
            pass

        # Parse — handle partial write with retry.
        try:
            raw = json.loads(raw_bytes.decode())
        except (json.JSONDecodeError, UnicodeDecodeError):
            retries = partial_read_attempts.get(seq, 0) + 1
            partial_read_attempts[seq] = retries
            if retries >= _MAX_PARTIAL_RETRIES:
                # Give up on this seq: write an error response and advance.
                error_resp = {
                    "seq": seq,
                    "ok": False,
                    "result": None,
                    "error": "partial read after retries",
                }
                with contextlib.suppress(Exception):  # noqa: BLE001
                    session.write_file(
                        session_id, resp_path, json.dumps(error_resp).encode()
                    )
                broker.rpc_error_count += 1
                seq += 1
            else:
                time.sleep(_BROKER_POLL_INTERVAL)
            continue

        # Dispatch and write response.
        response = broker.handle_request(raw)
        # F8: scrub the success result BEFORE writing it to the sandbox response file,
        # so connected_account_id and other secrets never reach the script.
        if response.get("ok") and response.get("result") is not None:
            response = {**response, "result": _scrub_rpc_result(response["result"])}

        with contextlib.suppress(Exception):  # noqa: BLE001
            session.write_file(session_id, resp_path, json.dumps(response).encode())
        seq += 1

    duration_ms = int(time.monotonic() * 1000) - start_ms

    # Read results from the per-run output files.
    stdout = _read_output_file(
        session, session_id, f"{rpc_dir}/stdout.log", max_bytes=65536
    )
    stderr = _read_output_file(
        session, session_id, f"{rpc_dir}/stderr.log", max_bytes=16384
    )
    exit_code_raw = _read_output_file(
        session, session_id, f"{rpc_dir}/exit_code", max_bytes=16
    ).strip()

    # Determine exit code and timed_out.
    timed_out = not _check_done_marker(session, session_id, run_id)
    if exit_code_raw and exit_code_raw.isdigit():
        exit_code = int(exit_code_raw)
    elif timed_out:
        exit_code = 124  # standard timeout exit code (same as `timeout` command)
    else:
        exit_code = 0  # done marker exists but exit_code unparseable; assume success

    return CodeActResult(
        successful=exit_code == 0 and not timed_out,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        duration_ms=duration_ms,
        timed_out=timed_out,
        truncated=False,
        rpc_call_count=broker.rpc_call_count,
        rpc_error_count=broker.rpc_error_count,
    )
