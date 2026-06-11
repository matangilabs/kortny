import asyncio
import uuid
from decimal import Decimal
from types import SimpleNamespace
from typing import Any, cast

import pytest

from kortny.agent.adk_tools import KortnyAdkTool, KortnyRegistryToolset
from kortny.approvals import (
    TOOL_APPROVAL_REQUIRED_MESSAGE,
    ToolApprovalRequired,
)
from kortny.db.models import TaskEventType
from kortny.execution import SandboxResult, SandboxSpec
from kortny.tools import ToolRegistry
from kortny.tools.code_exec import CodeExecTool
from kortny.tools.types import JsonObject, ToolResult


def test_adk_tool_declaration_preserves_provider_neutral_schema() -> None:
    adapter = KortnyAdkTool(
        tool=_MarketLookupTool(),
        task=cast(Any, None),
        session=cast(Any, None),
        task_service=cast(Any, None),
    )

    declaration = adapter._get_declaration()

    assert declaration.name == "market_lookup"
    assert declaration.description == "Looks up market data."
    assert declaration.parameters_json_schema == _MarketLookupTool.parameters


def test_adk_tool_enforces_existing_approval_policy() -> None:
    task_service = _RecordingTaskService()
    adapter = KortnyAdkTool(
        tool=_ForgetFactTool(),
        task=cast(Any, SimpleNamespace(id=uuid.uuid4())),
        session=cast(Any, _NoApprovalSession()),
        task_service=cast(Any, task_service),
    )

    with pytest.raises(ToolApprovalRequired) as exc:
        asyncio.run(
            adapter.run_async(
                args={"key": "pdf_branding"},
                tool_context=cast(
                    Any,
                    SimpleNamespace(function_call_id="call-approval"),
                ),
            )
        )

    assert exc.value.request.tool_name == "forget_fact"
    assert task_service.events == [
        (
            TaskEventType.log,
            {
                "message": TOOL_APPROVAL_REQUIRED_MESSAGE,
                "runtime": "adk",
                "turn": 1,
                "step_id": "adk_tool_call",
                "request": exc.value.request.to_payload(),
            },
        )
    ]


def test_adk_code_exec_runs_without_prior_approval() -> None:
    # The sandbox is the safety boundary: code_exec no longer pauses the task
    # for a human gate, even when no approval has ever been recorded.
    task_service = _RecordingTaskService()
    runner = _SuccessfulSandboxRunner(stdout="42\n")
    adapter = KortnyAdkTool(
        tool=CodeExecTool(
            runner=runner,
            image="python:3.11-slim",
        ),
        task=cast(Any, _fake_task()),
        session=cast(Any, _NoApprovalSession()),
        task_service=cast(Any, task_service),
    )

    result = asyncio.run(
        adapter.run_async(
            args={
                "code": "print(6 * 7)",
                "language": "python",
                "timeout_seconds": 5,
            },
            tool_context=cast(
                Any,
                SimpleNamespace(function_call_id="call-code-exec"),
            ),
        )
    )

    assert result["output"]["successful"] is True
    assert [event_type for event_type, _payload in task_service.events] == [
        TaskEventType.tool_call,
        TaskEventType.tool_result,
    ]


def test_adk_code_exec_runs_after_user_approval_is_recorded() -> None:
    task_service = _RecordingTaskService()
    runner = _SuccessfulSandboxRunner(stdout="42\n")
    adapter = KortnyAdkTool(
        tool=CodeExecTool(
            runner=runner,
            image="python:3.11-slim",
        ),
        task=cast(Any, _fake_task()),
        session=cast(Any, _ApprovalGrantedSession()),
        task_service=cast(Any, task_service),
    )

    result = asyncio.run(
        adapter.run_async(
            args={
                "code": "print(6 * 7)",
                "language": "python",
                "timeout_seconds": 5,
            },
            tool_context=cast(
                Any,
                SimpleNamespace(function_call_id="call-code-approved"),
            ),
        )
    )

    assert runner.spec is not None
    assert runner.spec.command == ("python", "-c", "print(6 * 7)")
    assert runner.spec.network == "none"
    assert result["output"]["successful"] is True
    assert result["output"]["stdout"] == "42\n"
    assert [event_type for event_type, _payload in task_service.events] == [
        TaskEventType.tool_call,
        TaskEventType.tool_result,
    ]
    assert task_service.events[0][1]["tool"] == "code_exec"
    assert task_service.events[0][1]["argument_keys"] == [
        "code",
        "language",
        "timeout_seconds",
    ]
    assert task_service.events[1][1]["tool"] == "code_exec"
    assert task_service.events[1][1]["output"]["stdout"] == "42\n"


def test_adk_tool_returns_recoverable_result_for_tool_exception() -> None:
    task_service = _RecordingTaskService()
    adapter = KortnyAdkTool(
        tool=_TimeoutTool(),
        task=cast(Any, _fake_task()),
        session=cast(Any, None),
        task_service=cast(Any, task_service),
    )

    result = asyncio.run(
        adapter.run_async(
            args={"url": "https://example.com"},
            tool_context=cast(Any, SimpleNamespace(function_call_id="call-timeout")),
        )
    )

    assert result["output"]["successful"] is False
    assert result["output"]["error"]["code"] == "tool_execution_timeout"
    assert result["output"]["error"]["recoverable"] is True
    assert result["recoverable_error"] is True
    assert [event_type for event_type, _ in task_service.events] == [
        TaskEventType.tool_call,
        TaskEventType.tool_result,
    ]
    tool_result = task_service.events[1][1]
    assert tool_result["recoverable"] is True
    assert tool_result["output"]["error"]["category"] == "transient_transport"


def test_adk_registry_toolset_loads_registry_lazily_once() -> None:
    task_service = _RecordingTaskService()
    load_count = 0

    def registry_factory() -> ToolRegistry:
        nonlocal load_count
        load_count += 1
        return ToolRegistry([_MarketLookupTool()])

    toolset = KortnyRegistryToolset(
        registry_factory=registry_factory,
        task=cast(Any, SimpleNamespace(id=uuid.uuid4())),
        session=cast(Any, None),
        task_service=cast(Any, task_service),
    )

    first_tools = asyncio.run(toolset.get_tools())
    second_tools = asyncio.run(toolset.get_tools())

    assert load_count == 1
    assert [tool.name for tool in first_tools] == ["market_lookup"]
    assert [tool.name for tool in second_tools] == ["market_lookup"]
    assert task_service.events == [
        (
            TaskEventType.log,
            {
                "message": "adk_lazy_toolset_loaded",
                "runtime": "adk",
                "tool_count": 1,
                "tool_names": ["market_lookup"],
            },
        )
    ]


class _MarketLookupTool:
    name = "market_lookup"
    description = "Looks up market data."
    parameters = {
        "type": "object",
        "properties": {
            "ticker": {"type": "string"},
            "period": {"type": "string", "enum": ["1d", "1mo"]},
        },
        "required": ["ticker"],
    }

    def invoke(self, args: JsonObject) -> ToolResult:
        return ToolResult(output={"args": args}, cost_usd=Decimal("0"))


class _ForgetFactTool:
    name = "forget_fact"
    description = "Deletes a stored fact."
    parameters = {
        "type": "object",
        "properties": {"key": {"type": "string"}},
        "required": ["key"],
    }

    def invoke(self, args: JsonObject) -> ToolResult:
        return ToolResult(output={"deleted": args["key"]})


class _TimeoutTool:
    name = "composio_firecrawl_crawl"
    description = "Crawls a website."
    parameters = {
        "type": "object",
        "properties": {"url": {"type": "string"}},
        "required": ["url"],
    }

    def invoke(self, args: JsonObject) -> ToolResult:
        del args
        raise RuntimeError("The read operation timed out")


class _FailingSandboxRunner:
    def run(self, spec: SandboxSpec) -> SandboxResult:
        del spec
        raise AssertionError("code_exec should pause for approval before running")


class _SuccessfulSandboxRunner:
    def __init__(self, *, stdout: str) -> None:
        self.stdout = stdout
        self.spec: Any | None = None

    def run(self, spec: SandboxSpec) -> SandboxResult:
        self.spec = spec
        return SimpleNamespace(  # type: ignore[return-value]
            exit_code=0,
            stdout=self.stdout,
            stderr="",
            artifacts=(),
            usage={"duration_ms": 1},
            events=(),
        )


def _fake_task() -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid.uuid4(),
        installation_id=uuid.uuid4(),
        slack_channel_id="C123",
        slack_thread_ts="123.456",
        slack_user_id="U123",
    )


class _NoApprovalSession:
    def scalar(self, statement: object) -> None:
        del statement
        return None


class _ApprovalGrantedSession:
    def scalar(self, statement: object) -> SimpleNamespace:
        del statement
        return SimpleNamespace(payload={"decision": "approved"})


class _RecordingTaskService:
    def __init__(self) -> None:
        self.events: list[tuple[TaskEventType, dict[str, Any]]] = []

    def append_event(
        self,
        task: object,
        event_type: TaskEventType,
        payload: dict[str, Any],
    ) -> None:
        del task
        self.events.append((event_type, payload))
