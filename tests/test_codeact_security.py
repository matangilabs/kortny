"""Security tests for the CodeAct subsystem (HIG-301 Slice B hardening).

These tests cover the 8 security findings fixed in the hardening pass.  They
use entirely fake in-memory components — no real DB, no LLM, no sandbox.

Test classes:
  TestF1CoordinatorGatePath        — codeact_exec goes through the real approval gate
  TestF2FailClosed                 — per-call re-check blocks escalating calls
  TestF3ArmedTrifectaEscalation    — live trifecta state forces approval for outward tools
  TestF4SlackReadsUntrusted        — slack_channel_history / search_observed_slack_history
                                     are treated as untrusted-origin
  TestF4CanvasUntrusted            — slack_lookup_canvas_sections treated as untrusted-origin
  TestF5EphemeralSession           — session profile is "code_exec" + full SHA-256 key
  TestF8DeeperScrubbing            — camelCase keys, stringified JSON, exception scrubbing
  TestF2CaveatMidScriptTrifectaArm — mid-script untrusted tool arms trifecta for later write
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest

from kortny.agent.coordinator import AgentCoordinator
from kortny.agent.execution import ToolAttemptRecord
from kortny.agent.trifecta import (
    TrifectaGateState,
    is_untrusted_origin_tool,
)
from kortny.approvals import (
    ApprovalScope,
    ToolApprovalRequired,
    ToolApprovalRequirement,
)
from kortny.config import Settings
from kortny.db.models import Task, TaskEventType
from kortny.execution.codeact_rpc import (
    CodeActRpcBroker,
    _scrub_exception_message,
    _scrub_rpc_result,
)
from kortny.llm import Completion, TokenUsage, ToolCall
from kortny.tasks import TaskService
from kortny.tools import ToolRegistry
from kortny.tools.types import JsonObject, JsonSchema, RecoverableToolError, ToolResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _settings(**overrides: Any) -> Settings:
    """Build minimal Settings without real Slack/LLM/DB."""
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


def _make_task(identity_kind: str = "slack_message") -> Task:
    """Return a minimal in-memory Task (no DB).

    Uses MagicMock so that attribute access on the SQLAlchemy model does not
    fail under mypy strict mode (Task attributes are SQLAlchemy Mapped columns
    that are not freely settable at construction time without a real session).
    The MagicMock is cast to Task for the type checker.
    """
    from unittest.mock import MagicMock

    task_id = uuid.uuid4()
    mock_task = MagicMock(spec=Task)
    mock_task.id = task_id
    mock_task.identity_kind = identity_kind
    mock_task.installation_id = uuid.uuid4()
    mock_task.slack_channel_id = "C1234"
    mock_task.slack_user_id = "U1234"
    mock_task.slack_thread_ts = "12345.0"
    mock_task.slack_message_ts = "12345.1"
    mock_task.attempts = 0
    mock_task.lease_expires_at = None
    mock_task.input = "test input"
    mock_task.result_summary = None
    return cast(Task, mock_task)


@dataclass
class FakeLLMSingleTurn:
    """Returns one completion then raises on further calls."""

    completion: Completion

    def complete(
        self,
        *,
        task_id: uuid.UUID,
        messages: Any,
        tools: Any = (),
        response_format: Any = None,
        prompt_name: Any = None,
        prompt_source: str = "code",
    ) -> Completion:
        return self.completion


@dataclass
class FakeTaskService:
    """Records append_event calls without DB."""

    events: list[tuple[Task, TaskEventType, dict[str, Any]]] = field(
        default_factory=list
    )
    approval_granted: bool = False
    waiting_approval_called: bool = False
    status: str = "running"

    def append_event(
        self, task: Task, kind: TaskEventType, payload: dict[str, Any]
    ) -> Any:
        self.events.append((task, kind, payload))
        return MagicMock()

    def mark_waiting_approval(self, task: Task, request: Any) -> None:
        self.waiting_approval_called = True


def _make_coordinator(
    *,
    registry: ToolRegistry | None = None,
    settings: Settings | None = None,
    task_service: FakeTaskService | None = None,
    trifecta_gate_enabled: bool = True,
    approval_granted: bool = False,
) -> AgentCoordinator:
    """Build an AgentCoordinator with a fake DB session and fake LLM."""
    fake_session = MagicMock()
    # Autonomy policy query returns balanced by default.
    fake_ap = MagicMock()
    fake_ap.level = "balanced"
    fake_session.execute.return_value.scalar_one_or_none.return_value = fake_ap

    llm = FakeLLMSingleTurn(
        Completion(
            content="done",
            tool_calls=(),
            usage=TokenUsage(input_tokens=5, output_tokens=2),
            response_id="r1",
            model="openai/gpt-4o",
        )
    )

    coord = AgentCoordinator(
        session=fake_session,
        llm=llm,
        registry=registry or ToolRegistry([]),
        settings=settings,
        trifecta_gate_enabled=trifecta_gate_enabled,
    )

    if task_service is not None:
        coord.task_service = cast(TaskService, task_service)

    if approval_granted:
        # Pre-grant ALL approval keys so the preflight passes.
        def _granted(*args: Any, **kwargs: Any) -> bool:
            return True

        coord._approval_is_granted = _granted  # type: ignore[method-assign]

    return coord


def _make_tool_call(
    name: str = "codeact_exec",
    tool_call_id: str | None = None,
    arguments: dict[str, Any] | None = None,
) -> ToolCall:
    return ToolCall(
        id=tool_call_id or str(uuid.uuid4()),
        name=name,
        arguments=arguments or {},
    )


# ---------------------------------------------------------------------------
# F1: Coordinator approval gate fires for codeact_exec
# ---------------------------------------------------------------------------


class TestF1CoordinatorGatePath:
    """_raise_if_tool_approval_required must fire for codeact_exec before
    _handle_codeact_exec is reached."""

    def test_raise_if_tool_approval_fires_for_codeact_exec(self) -> None:
        """codeact_exec is cataloged user_approval — the gate must raise."""
        from kortny.tools.code_exec import CodeActExecTool

        settings = _settings(
            KORTNY_CODEACT_ENABLED="true",
            KORTNY_SANDBOX_RUNNER_URL="http://localhost:8090",
        )
        registry = ToolRegistry([])
        registry.register_if_absent(CodeActExecTool())

        coord = _make_coordinator(registry=registry, settings=settings)
        task = _make_task()
        attempt = ToolAttemptRecord(
            tool_name="codeact_exec",
            normalized_args_hash="test-hash-000",
            attempt_no=1,
            status="pending",
        )
        tool_call = _make_tool_call(name="codeact_exec")
        arguments: JsonObject = {
            "code": "print('hi')",
            "allowed_tools": ["list_schedules"],
        }

        with pytest.raises(ToolApprovalRequired) as exc_info:
            coord._raise_if_tool_approval_required(
                task_obj=task,
                tool_call=tool_call,
                arguments=arguments,
                attempt=attempt,
                turn=1,
                step_id="step-test",
            )

        req = exc_info.value.request
        assert req.tool_name == "codeact_exec"
        assert req.scope in (ApprovalScope.user, ApprovalScope.admin)

    def test_approval_gate_skips_when_approval_already_granted(self) -> None:
        """If approval is already in the DB, _raise_if_tool_approval_required
        must NOT raise."""
        from kortny.tools.code_exec import CodeActExecTool

        settings = _settings(
            KORTNY_CODEACT_ENABLED="true",
            KORTNY_SANDBOX_RUNNER_URL="http://localhost:8090",
        )
        registry = ToolRegistry([])
        registry.register_if_absent(CodeActExecTool())

        coord = _make_coordinator(
            registry=registry, settings=settings, approval_granted=True
        )
        task = _make_task()
        ts = FakeTaskService()
        coord.task_service = cast(TaskService, ts)

        attempt = ToolAttemptRecord(
            tool_name="codeact_exec",
            normalized_args_hash="already-granted-hash",
            attempt_no=1,
            status="pending",
        )
        tool_call = _make_tool_call(name="codeact_exec")

        # Should not raise — approval pre-granted.
        coord._raise_if_tool_approval_required(
            task_obj=task,
            tool_call=tool_call,
            arguments={"code": "pass", "allowed_tools": ["list_schedules"]},
            attempt=attempt,
            turn=1,
            step_id="step-test",
        )


# ---------------------------------------------------------------------------
# F2: Per-call fail-closed re-check
# ---------------------------------------------------------------------------


class _BlockedTool:
    """A tool that reports as a write/destructive action for any real arg."""

    name = "delete_items"
    description = "Deletes items by id."
    parameters: JsonSchema = {
        "type": "object",
        "properties": {"ids": {"type": "array", "items": {"type": "integer"}}},
        "required": ["ids"],
        "additionalProperties": False,
    }

    def __init__(self) -> None:
        self.invoke_count = 0

    def invoke(self, args: JsonObject) -> ToolResult:
        self.invoke_count += 1
        return ToolResult(output={"deleted": args.get("ids", [])})


class _PassTool:
    """A read-only tool that should never be blocked."""

    name = "list_items"
    description = "Lists items."
    parameters: JsonSchema = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }

    def invoke(self, args: JsonObject) -> ToolResult:
        return ToolResult(output={"items": []})


class TestF2FailClosed:
    """_rpc_dispatch must block a call whose REAL args require higher approval
    than was granted by the preflight (which assessed at empty args {})."""

    def test_rpc_dispatch_blocks_escalating_call(self) -> None:
        """A tool approved at none-scope at empty args {} but requiring user
        scope at real destructive args must be blocked by _rpc_dispatch."""
        from kortny.tools.code_exec import CodeActExecTool

        settings = _settings(
            KORTNY_CODEACT_ENABLED="true",
            KORTNY_SANDBOX_RUNNER_URL="http://localhost:8090",
        )

        delete_tool = _BlockedTool()
        registry = ToolRegistry([])
        registry.register_if_absent(delete_tool)
        registry.register_if_absent(CodeActExecTool())

        task_service = FakeTaskService()
        coord = _make_coordinator(
            registry=registry,
            settings=settings,
            task_service=task_service,
            approval_granted=True,  # bypass the preflight — we test _rpc_dispatch
        )
        task = _make_task()

        # Force granted_scope to ApprovalScope.none to simulate a preflight that
        # saw the tool at {} (no write args) and approved it at none.
        # We do this by monkey-patching approval_policy so that at {} it returns none.
        original_req_for = coord.approval_policy.requirement_for

        def _fake_req_for(
            tool: Any, args: Any, **kwargs: Any
        ) -> ToolApprovalRequirement:
            if not args:
                # Empty args: pretend it's free
                return ToolApprovalRequirement(
                    scope=ApprovalScope.none, risk="read_only", reason="test"
                )
            return original_req_for(tool, args, **kwargs)

        coord.approval_policy.requirement_for = _fake_req_for  # type: ignore[method-assign]

        # Manually call _handle_codeact_exec with allowed_tools=[delete_items].
        # The preflight will assess at {} → scope=none → granted_scope=none.
        # When _rpc_dispatch calls delete_items with real ids=[1,2,3], the
        # real-args scope must be > none, so the call must be blocked.
        tool_call = _make_tool_call(name="codeact_exec")
        arguments: JsonObject = {
            "code": "from kortny_tools import delete_items; delete_items(ids=[1,2,3])",
            "allowed_tools": ["delete_items"],
        }

        # We intercept run_codeact to simulate the script calling the tool.
        dispatched_results: list[Any] = []
        blocked_events: list[dict[str, Any]] = []

        def _fake_run_codeact(
            session: Any,
            *,
            session_id: str,
            code: str,
            stubs: Any,
            allowed_tools: frozenset[str],
            dispatch: Any,
            **kwargs: Any,
        ) -> Any:
            # Simulate the script calling delete_items with real destructive args.
            try:
                result = dispatch("delete_items", {"ids": [1, 2, 3]})
                dispatched_results.append(result)
            except RecoverableToolError as e:
                # The error code is "codeact_rpc_blocked" (check .code attribute,
                # not str(e) which carries the message, not the code).
                if e.code == "codeact_rpc_blocked":
                    blocked_events.append({"blocked": True, "tool": "delete_items"})
                else:
                    raise

            from kortny.execution.codeact_rpc import CodeActResult

            return CodeActResult(
                successful=True,
                exit_code=0,
                stdout="blocked",
                stderr="",
                duration_ms=10,
                timed_out=False,
                truncated=False,
                rpc_call_count=0,
                rpc_error_count=0,
            )

        with (
            patch(
                "kortny.execution.codeact_rpc.run_codeact",
                side_effect=_fake_run_codeact,
            ),
            patch(
                "kortny.execution.sandbox_sessions.HttpSandboxSessionClient"
            ) as mock_session_cls,
        ):
            mock_session = MagicMock()
            mock_session.open_session.return_value = MagicMock(
                session_id="fake-session-id"
            )
            mock_session_cls.return_value = mock_session

            coord._handle_codeact_exec(
                task_obj=task,
                tool_call=tool_call,
                arguments=arguments,
                turn=1,
                step_id="step-1",
            )

        # The real registry.invoke must NOT have been called.
        assert delete_tool.invoke_count == 0
        # A codeact_rpc_blocked event must have been recorded.
        blocked_log_events = [
            payload
            for (_, kind, payload) in task_service.events
            if payload.get("message") == "codeact_rpc_blocked"
        ]
        assert len(blocked_log_events) >= 1
        assert blocked_log_events[0]["tool"] == "delete_items"

    def test_rpc_dispatch_allows_safe_tool(self) -> None:
        """A purely read-only tool must NOT be blocked by the per-call re-check."""
        from kortny.tools.code_exec import CodeActExecTool

        settings = _settings(
            KORTNY_CODEACT_ENABLED="true",
            KORTNY_SANDBOX_RUNNER_URL="http://localhost:8090",
        )

        list_tool = _PassTool()
        registry = ToolRegistry([])
        registry.register_if_absent(list_tool)
        registry.register_if_absent(CodeActExecTool())

        task_service = FakeTaskService()
        coord = _make_coordinator(
            registry=registry,
            settings=settings,
            task_service=task_service,
            approval_granted=True,
        )
        task = _make_task()

        dispatched: list[str] = []

        def _fake_run_codeact(
            session: Any,
            *,
            session_id: str,
            code: str,
            stubs: Any,
            allowed_tools: frozenset[str],
            dispatch: Any,
            **kwargs: Any,
        ) -> Any:
            result = dispatch("list_items", {})
            dispatched.append("list_items")
            from kortny.execution.codeact_rpc import CodeActResult

            return CodeActResult(
                successful=True,
                exit_code=0,
                stdout=str(result),
                stderr="",
                duration_ms=5,
                timed_out=False,
                truncated=False,
                rpc_call_count=1,
                rpc_error_count=0,
            )

        with (
            patch(
                "kortny.execution.codeact_rpc.run_codeact",
                side_effect=_fake_run_codeact,
            ),
            patch(
                "kortny.execution.sandbox_sessions.HttpSandboxSessionClient"
            ) as mock_session_cls,
        ):
            mock_session = MagicMock()
            mock_session.open_session.return_value = MagicMock(
                session_id="fake-session-id"
            )
            mock_session_cls.return_value = mock_session

            coord._handle_codeact_exec(
                task_obj=task,
                tool_call=_make_tool_call(name="codeact_exec"),
                arguments={
                    "code": "from kortny_tools import list_items; list_items()",
                    "allowed_tools": ["list_items"],
                },
                turn=1,
                step_id="step-1",
            )

        assert "list_items" in dispatched


# ---------------------------------------------------------------------------
# F3: Live trifecta armed state forces approval for outward tools
# ---------------------------------------------------------------------------


class _OutwardTool:
    """A write/outward tool (namespace = native.slack so trifecta escalates it)."""

    name = "slack_reply_thread"
    description = "Replies in a Slack thread."
    parameters: JsonSchema = {
        "type": "object",
        "properties": {
            "channel": {"type": "string"},
            "text": {"type": "string"},
        },
        "required": ["channel", "text"],
        "additionalProperties": False,
    }

    def invoke(self, args: JsonObject) -> ToolResult:
        return ToolResult(output={"ok": True})


class TestF3ArmedTrifectaEscalation:
    """When the live trifecta state is already armed (prior untrusted content in
    the task's context) AND the allowlist contains outward tools (but NOT
    untrusted-origin tools, so the combo gate doesn't fire), the preflight
    must still force user approval (F3)."""

    def test_armed_trifecta_forces_approval_for_outward_only_allowlist(self) -> None:
        """Pre-arm trifecta; allowlist has only an outward tool (no untrusted-origin
        tool), so the combo gate does NOT fire, but F3 must still raise."""
        from kortny.tools.code_exec import CodeActExecTool

        settings = _settings(
            KORTNY_CODEACT_ENABLED="true",
            KORTNY_SANDBOX_RUNNER_URL="http://localhost:8090",
        )

        outward_tool = _OutwardTool()
        registry = ToolRegistry([])
        registry.register_if_absent(outward_tool)
        registry.register_if_absent(CodeActExecTool())

        task_service = FakeTaskService()
        coord = _make_coordinator(
            registry=registry,
            settings=settings,
            task_service=task_service,
        )
        task = _make_task()

        # Pre-arm the trifecta gate (simulate a prior turn with untrusted content).
        state = coord._trifecta_state(task)
        state.arm("prior_web_search")
        assert state.armed is True

        tool_call = _make_tool_call(name="codeact_exec")
        arguments: JsonObject = {
            "code": "from kortny_tools import slack_reply_thread; slack_reply_thread(text='hi')",
            "allowed_tools": ["slack_reply_thread"],
        }

        # Without F3, an allowlist with only outward tools (no untrusted-origin)
        # would pass the combo gate and not trigger approval.  With F3 it
        # must raise ToolApprovalRequired because the task context is already armed.
        with pytest.raises(ToolApprovalRequired) as exc_info:
            coord._handle_codeact_exec(
                task_obj=task,
                tool_call=tool_call,
                arguments=arguments,
                turn=1,
                step_id="step-1",
            )

        req = exc_info.value.request
        assert req.tool_name == "codeact_exec"
        assert req.scope in (ApprovalScope.user, ApprovalScope.admin)
        # The risk must identify this as the trifecta-armed path, not the combo gate.
        assert "trifecta_armed_codeact_outward" in req.risk

    def test_unarmed_trifecta_does_not_force_approval_for_outward_only(self) -> None:
        """With trifecta NOT armed, an outward-only allowlist at no-approval
        baseline must NOT raise on its own (no combo, no armed gate)."""
        from kortny.tools.code_exec import CodeActExecTool

        settings = _settings(
            KORTNY_CODEACT_ENABLED="true",
            KORTNY_SANDBOX_RUNNER_URL="http://localhost:8090",
        )

        # Use a tool that is NOT outward (no approval needed at baseline) and is
        # also NOT untrusted-origin — a plain read tool.
        read_tool = _PassTool()
        registry = ToolRegistry([])
        registry.register_if_absent(read_tool)
        registry.register_if_absent(CodeActExecTool())

        task_service = FakeTaskService()
        coord = _make_coordinator(
            registry=registry,
            settings=settings,
            task_service=task_service,
        )
        task = _make_task()

        # Trifecta is NOT armed.
        state = coord._trifecta_state(task)
        assert state.armed is False

        def _fake_run_codeact(session: Any, **kwargs: Any) -> Any:
            from kortny.execution.codeact_rpc import CodeActResult

            return CodeActResult(
                successful=True,
                exit_code=0,
                stdout="ok",
                stderr="",
                duration_ms=5,
                timed_out=False,
                truncated=False,
                rpc_call_count=0,
                rpc_error_count=0,
            )

        with (
            patch(
                "kortny.execution.codeact_rpc.run_codeact",
                side_effect=_fake_run_codeact,
            ),
            patch(
                "kortny.execution.sandbox_sessions.HttpSandboxSessionClient"
            ) as mock_cls,
        ):
            mock_cls.return_value.open_session.return_value = MagicMock(session_id="s1")
            # Should NOT raise.
            coord._handle_codeact_exec(
                task_obj=task,
                tool_call=_make_tool_call(name="codeact_exec"),
                arguments={
                    "code": "from kortny_tools import list_items; list_items()",
                    "allowed_tools": ["list_items"],
                },
                turn=1,
                step_id="step-1",
            )


# ---------------------------------------------------------------------------
# F4: slack_channel_history and search_observed_slack_history are untrusted-origin
# ---------------------------------------------------------------------------


class TestF4SlackReadsUntrusted:
    """slack_channel_history and search_observed_slack_history must be treated
    as untrusted-origin tools (user-authored Slack messages = third-party content)."""

    def test_slack_channel_history_is_untrusted(self) -> None:
        assert is_untrusted_origin_tool("slack_channel_history") is True

    def test_search_observed_slack_history_is_untrusted(self) -> None:
        assert is_untrusted_origin_tool("search_observed_slack_history") is True

    def test_slack_post_message_is_not_untrusted(self) -> None:
        """Write tools must NOT be listed as untrusted-origin — they are outward."""
        assert is_untrusted_origin_tool("slack_post_message") is False

    def test_slack_file_read_is_untrusted(self) -> None:
        """Existing entry must still be present."""
        assert is_untrusted_origin_tool("slack_file_read") is True

    def test_web_search_is_untrusted(self) -> None:
        """Regression: web_search must stay untrusted-origin."""
        assert is_untrusted_origin_tool("web_search") is True

    def test_slack_channel_history_arms_trifecta(self) -> None:
        """TrifectaGateState.note_tool_result must arm when the tool is
        slack_channel_history."""
        state = TrifectaGateState(enabled=True)
        armed = state.note_tool_result("slack_channel_history")
        assert armed is True
        assert state.armed is True
        assert state.armed_by == "slack_channel_history"

    def test_search_observed_slack_history_arms_trifecta(self) -> None:
        state = TrifectaGateState(enabled=True)
        armed = state.note_tool_result("search_observed_slack_history")
        assert armed is True
        assert state.armed is True


# ---------------------------------------------------------------------------
# F4 (canvas): slack_lookup_canvas_sections is untrusted-origin
# ---------------------------------------------------------------------------


class TestF4CanvasUntrusted:
    """slack_lookup_canvas_sections must be treated as untrusted-origin."""

    def test_slack_lookup_canvas_sections_is_untrusted(self) -> None:
        assert is_untrusted_origin_tool("slack_lookup_canvas_sections") is True

    def test_canvas_tool_arms_trifecta(self) -> None:
        state = TrifectaGateState(enabled=True)
        armed = state.note_tool_result("slack_lookup_canvas_sections")
        assert armed is True
        assert state.armed is True


# ---------------------------------------------------------------------------
# F5: Ephemeral container profile + full SHA-256 approval key
# ---------------------------------------------------------------------------


class TestF5EphemeralSession:
    """open_session must be called with profile="code_exec" (not "workbench"),
    and the approval key must use the FULL 64-char sha256 hex digest."""

    def test_session_opened_with_code_exec_profile(self) -> None:
        """open_session must be called with profile='code_exec'."""
        from kortny.tools.code_exec import CodeActExecTool

        settings = _settings(
            KORTNY_CODEACT_ENABLED="true",
            KORTNY_SANDBOX_RUNNER_URL="http://localhost:8090",
        )

        registry = ToolRegistry([])
        registry.register_if_absent(CodeActExecTool())
        registry.register_if_absent(_PassTool())

        task_service = FakeTaskService()
        coord = _make_coordinator(
            registry=registry,
            settings=settings,
            task_service=task_service,
            approval_granted=True,
        )
        task = _make_task()

        def _fake_run_codeact(session: Any, **kwargs: Any) -> Any:
            from kortny.execution.codeact_rpc import CodeActResult

            return CodeActResult(
                successful=True,
                exit_code=0,
                stdout="ok",
                stderr="",
                duration_ms=5,
                timed_out=False,
                truncated=False,
                rpc_call_count=0,
                rpc_error_count=0,
            )

        with (
            patch(
                "kortny.execution.codeact_rpc.run_codeact",
                side_effect=_fake_run_codeact,
            ),
            patch(
                "kortny.execution.sandbox_sessions.HttpSandboxSessionClient"
            ) as mock_cls,
        ):
            mock_session = MagicMock()
            mock_session.open_session.return_value = MagicMock(
                session_id="ephemeral-session-id"
            )
            mock_cls.return_value = mock_session

            coord._handle_codeact_exec(
                task_obj=task,
                tool_call=_make_tool_call(name="codeact_exec"),
                arguments={
                    "code": "print('ok')",
                    "allowed_tools": ["list_items"],
                },
                turn=1,
                step_id="step-1",
            )

            # The session must be opened with the ephemeral profile.
            mock_session.open_session.assert_called_once()
            _, call_kwargs = mock_session.open_session.call_args
            assert call_kwargs.get("profile") == "code_exec", (
                f"Expected profile='code_exec', got {call_kwargs.get('profile')!r}"
            )

    def test_approval_key_uses_full_sha256(self) -> None:
        """The codeact_exec_started log event must contain a full 64-char
        code_sha256 (not the old truncated 16-char version)."""
        from kortny.tools.code_exec import CodeActExecTool

        settings = _settings(
            KORTNY_CODEACT_ENABLED="true",
            KORTNY_SANDBOX_RUNNER_URL="http://localhost:8090",
        )

        registry = ToolRegistry([])
        registry.register_if_absent(CodeActExecTool())
        registry.register_if_absent(_PassTool())

        task_service = FakeTaskService()
        coord = _make_coordinator(
            registry=registry,
            settings=settings,
            task_service=task_service,
            approval_granted=True,
        )
        task = _make_task()

        code = "print('test sha256')"
        expected_full_sha = hashlib.sha256(code.encode()).hexdigest()
        assert len(expected_full_sha) == 64

        def _fake_run_codeact(session: Any, **kwargs: Any) -> Any:
            from kortny.execution.codeact_rpc import CodeActResult

            return CodeActResult(
                successful=True,
                exit_code=0,
                stdout="ok",
                stderr="",
                duration_ms=5,
                timed_out=False,
                truncated=False,
                rpc_call_count=0,
                rpc_error_count=0,
            )

        with (
            patch(
                "kortny.execution.codeact_rpc.run_codeact",
                side_effect=_fake_run_codeact,
            ),
            patch(
                "kortny.execution.sandbox_sessions.HttpSandboxSessionClient"
            ) as mock_cls,
        ):
            mock_cls.return_value.open_session.return_value = MagicMock(session_id="s1")

            coord._handle_codeact_exec(
                task_obj=task,
                tool_call=_make_tool_call(name="codeact_exec"),
                arguments={
                    "code": code,
                    "allowed_tools": ["list_items"],
                },
                turn=1,
                step_id="step-1",
            )

        # Find the codeact_exec_started log event.
        started_events = [
            payload
            for (_, kind, payload) in task_service.events
            if payload.get("message") == "codeact_exec_started"
        ]
        assert started_events, "Expected a codeact_exec_started log event"
        code_sha_in_event = started_events[0].get("code_sha256", "")
        assert len(code_sha_in_event) == 64, (
            f"Expected 64-char sha256, got len={len(code_sha_in_event)}: "
            f"{code_sha_in_event!r}"
        )
        assert code_sha_in_event == expected_full_sha

    def test_session_opened_with_unique_per_run_key(self) -> None:
        """F5: open_session must be called with a per-run key (task_id:codeact:run_id),
        not just the raw task_id, so each run gets its own container."""
        from kortny.tools.code_exec import CodeActExecTool

        settings = _settings(
            KORTNY_CODEACT_ENABLED="true",
            KORTNY_SANDBOX_RUNNER_URL="http://localhost:8090",
        )

        registry = ToolRegistry([])
        registry.register_if_absent(CodeActExecTool())
        registry.register_if_absent(_PassTool())

        task_service = FakeTaskService()
        coord = _make_coordinator(
            registry=registry,
            settings=settings,
            task_service=task_service,
            approval_granted=True,
        )
        task = _make_task()

        def _fake_run_codeact(session: Any, **kwargs: Any) -> Any:
            from kortny.execution.codeact_rpc import CodeActResult

            return CodeActResult(
                successful=True,
                exit_code=0,
                stdout="ok",
                stderr="",
                duration_ms=5,
                timed_out=False,
                truncated=False,
                rpc_call_count=0,
                rpc_error_count=0,
            )

        with (
            patch(
                "kortny.execution.codeact_rpc.run_codeact",
                side_effect=_fake_run_codeact,
            ),
            patch(
                "kortny.execution.sandbox_sessions.HttpSandboxSessionClient"
            ) as mock_cls,
        ):
            mock_session = MagicMock()
            mock_session.open_session.return_value = MagicMock(session_id="s1")
            mock_cls.return_value = mock_session

            coord._handle_codeact_exec(
                task_obj=task,
                tool_call=_make_tool_call(name="codeact_exec"),
                arguments={
                    "code": "print('ok')",
                    "allowed_tools": ["list_items"],
                },
                turn=1,
                step_id="step-1",
            )

            mock_session.open_session.assert_called_once()
            call_args, call_kwargs = mock_session.open_session.call_args
            # The first positional arg is the session key.
            session_key = call_args[0] if call_args else call_kwargs.get("task_id", "")
            task_id_str = str(task.id)
            # Must contain the task_id and "codeact"
            assert task_id_str in session_key, (
                f"Expected task_id {task_id_str!r} in session key {session_key!r}"
            )
            assert "codeact" in session_key, (
                f"Expected 'codeact' in session key {session_key!r}"
            )
            # Must NOT be just the raw task_id
            assert session_key != task_id_str, (
                "Session key must be unique per run, not just the task_id"
            )


# ---------------------------------------------------------------------------
# F8: Deeper scrubbing
# ---------------------------------------------------------------------------


class TestF8DeeperScrubbing:
    """_scrub_rpc_result must handle camelCase keys, exception paths, and
    stringified JSON blobs."""

    def test_camelcase_connectedaccountid_scrubbed(self) -> None:
        result = _scrub_rpc_result({"connectedAccountId": "abc"})
        assert isinstance(result, dict)
        assert result["connectedAccountId"] == "[redacted]"

    def test_camelcase_apikey_scrubbed(self) -> None:
        result = _scrub_rpc_result({"apiKey": "sk-secret"})
        assert isinstance(result, dict)
        assert result["apiKey"] == "[redacted]"

    def test_camelcase_accesstoken_scrubbed(self) -> None:
        result = _scrub_rpc_result({"accessToken": "tok-123"})
        assert isinstance(result, dict)
        assert result["accessToken"] == "[redacted]"

    def test_stringified_json_blob_scrubbed(self) -> None:
        # A string value that is itself a JSON dict with a secret key
        inner = json.dumps({"apiKey": "secret", "name": "ok"})
        result = _scrub_rpc_result({"data": inner})
        assert isinstance(result, dict)
        data_val = result["data"]
        assert isinstance(data_val, str)
        # The scrubbed string should not contain the raw "secret" value
        assert "secret" not in data_val

    def test_exception_path_scrubbed(self) -> None:
        """Broker handle_request dispatch exception message must be scrubbed."""
        token = "sk-abc123definitelyasecretlongtoken"

        def bad_dispatch(name: str, args: dict[str, Any]) -> object:
            raise RuntimeError(f"token={token}")

        settings_obj = _settings()
        b = CodeActRpcBroker(
            allowed_tools=frozenset({"tool_a"}),
            nonce="test-nonce",
            settings=settings_obj,
            dispatch=bad_dispatch,
        )
        req = {"seq": 0, "tool": "tool_a", "args": {}, "nonce": "test-nonce"}
        resp = b.handle_request(req)
        assert resp["ok"] is False
        error_str = resp.get("error", "")
        assert "[redacted]" in error_str
        assert token not in error_str

    def test_scrub_exception_message_redacts_sk_key(self) -> None:
        msg = "Error: sk-abc123secretlongkey caused a failure"
        scrubbed = _scrub_exception_message(msg)
        assert "sk-abc123secretlongkey" not in scrubbed
        assert "[redacted]" in scrubbed

    def test_scrub_exception_message_redacts_xox_token(self) -> None:
        msg = "token=xoxb-abc123-def456 is invalid"
        scrubbed = _scrub_exception_message(msg)
        assert "xoxb-abc123-def456" not in scrubbed
        assert "[redacted]" in scrubbed

    def test_scrub_exception_message_safe_message_unchanged(self) -> None:
        msg = "ValueError: integer expected, got string"
        scrubbed = _scrub_exception_message(msg)
        assert scrubbed == msg

    def test_snake_case_connected_account_id_scrubbed(self) -> None:
        result = _scrub_rpc_result({"connected_account_id": "acct-xyz"})
        assert isinstance(result, dict)
        assert result["connected_account_id"] == "[redacted]"

    def test_nested_dict_scrubbing(self) -> None:
        result = _scrub_rpc_result({"outer": {"apiKey": "secret", "data": "ok"}})
        assert isinstance(result, dict)
        inner = result["outer"]
        assert isinstance(inner, dict)
        assert inner["apiKey"] == "[redacted]"
        assert inner["data"] == "ok"

    def test_list_of_dicts_scrubbed(self) -> None:
        result = _scrub_rpc_result([{"apiKey": "s1"}, {"apiKey": "s2"}])
        assert isinstance(result, list)
        assert result[0]["apiKey"] == "[redacted]"
        assert result[1]["apiKey"] == "[redacted]"


# ---------------------------------------------------------------------------
# F2 caveat: Mid-script trifecta arming
# ---------------------------------------------------------------------------


class TestF2CaveatMidScriptTrifectaArm:
    """After an untrusted-origin RPC call in _rpc_dispatch, the trifecta state
    must be armed so subsequent write calls in the same script are blocked."""

    def test_untrusted_rpc_arms_trifecta_for_later_write(self) -> None:
        """A script that calls an untrusted-origin tool then a write tool must
        have the write tool blocked after the untrusted result is seen.

        Scenario: the allowlist has ONLY slack_reply_thread (outward, no untrusted-
        origin tool listed).  The combo gate does NOT fire (has_untrusted=False) so
        granted_scope=none.  Inside the fake run, the script dispatches
        composio_search_web (registered in the registry but NOT in the allowlist —
        simulating mid-script injection that bypasses the broker's list).  That call
        arms the trifecta.  The subsequent slack_reply_thread dispatch is then
        blocked because the trifecta escalates it from none → user, and user >
        granted_scope (none).
        """
        from kortny.tools.code_exec import CodeActExecTool

        settings = _settings(
            KORTNY_CODEACT_ENABLED="true",
            KORTNY_SANDBOX_RUNNER_URL="http://localhost:8090",
        )

        # Composio-prefixed (untrusted-origin) tool registered but NOT in allowlist.
        class _ComposioTool:
            name = "composio_search_web"
            description = "Search the web."
            parameters: JsonSchema = {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "additionalProperties": False,
            }

            def invoke(self, args: JsonObject) -> ToolResult:
                return ToolResult(output={"results": ["some web content"]})

        composio_tool = _ComposioTool()
        outward_tool = _OutwardTool()  # name="slack_reply_thread"

        registry = ToolRegistry([])
        registry.register_if_absent(composio_tool)
        registry.register_if_absent(outward_tool)
        registry.register_if_absent(CodeActExecTool())

        task_service = FakeTaskService()
        coord = _make_coordinator(
            registry=registry,
            settings=settings,
            task_service=task_service,
            approval_granted=True,
        )
        task = _make_task()

        # Trifecta is not armed initially.
        state = coord._trifecta_state(task)
        assert state.armed is False

        rpc_blocked: list[bool] = []

        def _fake_run_codeact(
            session: Any,
            *,
            session_id: str,
            code: str,
            stubs: Any,
            allowed_tools: frozenset[str],
            dispatch: Any,
            **kwargs: Any,
        ) -> Any:
            from kortny.execution.codeact_rpc import CodeActResult

            # Step 1: call the composio (untrusted-origin) tool — not in allowlist
            # but in registry; the _rpc_dispatch closure is what enforces trifecta,
            # not the allowlist.  This call arms the trifecta.
            dispatch("composio_search_web", {"query": "test"})

            # Step 2: call the outward write tool — trifecta is now armed; the
            # per-call re-check escalates scope to user, which > granted_scope (none).
            try:
                dispatch("slack_reply_thread", {"channel": "C1", "text": "hi"})
            except RecoverableToolError as e:
                if e.code == "codeact_rpc_blocked":
                    rpc_blocked.append(True)
                else:
                    raise

            return CodeActResult(
                successful=True,
                exit_code=0,
                stdout="done",
                stderr="",
                duration_ms=5,
                timed_out=False,
                truncated=False,
                rpc_call_count=1,
                rpc_error_count=0,
            )

        with (
            patch(
                "kortny.execution.codeact_rpc.run_codeact",
                side_effect=_fake_run_codeact,
            ),
            patch(
                "kortny.execution.sandbox_sessions.HttpSandboxSessionClient"
            ) as mock_cls,
        ):
            mock_cls.return_value.open_session.return_value = MagicMock(session_id="s1")
            coord._handle_codeact_exec(
                task_obj=task,
                tool_call=_make_tool_call(name="codeact_exec"),
                arguments={
                    "code": "# test",
                    # Only outward tool in allowlist → no untrusted-origin tool listed
                    # → combo gate does NOT fire → granted_scope = none.
                    "allowed_tools": ["slack_reply_thread"],
                },
                turn=1,
                step_id="step-1",
            )

        # slack_reply_thread must have been blocked by the mid-script trifecta arm.
        assert rpc_blocked == [True], (
            "Expected slack_reply_thread to be blocked after composio_search_web "
            "armed trifecta (trifecta escalates from none to user, greater than "
            f"granted_scope=none). rpc_blocked={rpc_blocked}"
        )

        # The trifecta must now be armed.
        assert state.armed is True


# ---------------------------------------------------------------------------
# F8: scrub is wired in the poll loop (on the success path)
# ---------------------------------------------------------------------------


class TestF8ScrubWiredOnSuccess:
    """F8: _scrub_rpc_result must be applied in the poll loop on the success path
    so that connected_account_id never reaches the sandbox response file."""

    def test_connected_account_id_not_in_sandbox_response(self) -> None:
        """The response file written to the sandbox must not contain the raw
        connected_account_id from a Composio tool result."""
        import re
        from dataclasses import dataclass, field

        from kortny.execution.codeact_rpc import run_codeact
        from kortny.execution.sandbox_sessions import SandboxExecResult

        nonce = "scrub-test-nonce"
        run_id = "scrub-test-run"

        def dispatch(name: str, args: dict[str, Any]) -> object:
            return {"connected_account_id": "acct-secret-123", "data": "ok"}

        settings = _settings()

        @dataclass
        class _FakeSession:
            files: dict[str, bytes] = field(default_factory=dict)
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
                if "nohup" in command:
                    m = re.search(r"\.kortny_rpc/([^/\s]+)/", command)
                    if m:
                        rid = m.group(1)
                        self.files[f"/workspace/.kortny_rpc/{rid}/exit_code"] = b"0"
                        self.files[f"/workspace/.kortny_rpc/{rid}/stdout.log"] = b"done"
                        self.files[f"/workspace/.kortny_rpc/{rid}/stderr.log"] = b""
                return SandboxExecResult(exit_code=0, stdout="done", stderr="")

            def open_session(self, task_id: str, profile: str = "workbench") -> Any:
                raise NotImplementedError

            def export_archive(self, session_id: str, path: str) -> bytes:
                raise NotImplementedError

            def close_session(self, session_id: str) -> None:
                pass

        session = _FakeSession()

        # Pre-seed a request at seq 0.
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

        # Check the response file written to the sandbox.
        resp_path = f"/workspace/.kortny_rpc/{run_id}/responses/0.json"
        assert resp_path in session.files
        resp = json.loads(session.files[resp_path])
        assert resp["ok"] is True
        result = resp["result"]
        assert isinstance(result, dict)
        # connected_account_id must be scrubbed
        assert result.get("connected_account_id") == "[redacted]", (
            f"Expected [redacted] but got {result.get('connected_account_id')!r}"
        )
        assert result.get("data") == "ok"  # non-sensitive key preserved
