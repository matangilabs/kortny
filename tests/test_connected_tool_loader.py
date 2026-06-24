"""Unit tests for the coordinator's direct callability (resolve-on-call) path.

These tests do not need a real database — they exercise the ConnectedToolLoader
integration in AgentCoordinator using mocks only.
"""

from __future__ import annotations

import uuid
from decimal import Decimal
from unittest.mock import MagicMock, patch

from kortny.agent.coordinator import AgentCoordinator, ConnectedToolLoader
from kortny.llm import ChatMessage, Completion, TokenUsage, ToolCall
from kortny.tools.registry import ToolNotFoundError
from kortny.tools.types import JsonObject, JsonSchema, Tool, ToolResult

# ---------------------------------------------------------------------------
# Minimal stubs
# ---------------------------------------------------------------------------


class _StubTool:
    """Minimal Tool that always returns a fixed result."""

    def __init__(self, name: str, result_output: JsonObject | None = None) -> None:
        self.name = name
        self.description = f"Stub tool {name}"
        self.parameters: JsonSchema = {
            "type": "object",
            "properties": {},
            "required": [],
        }
        self._result_output = result_output or {"ok": True}

    def invoke(self, args: JsonObject) -> ToolResult:
        return ToolResult(output=self._result_output, artifacts=(), cost_usd=Decimal(0))


class _RaisingRegistry:
    """Registry that raises ToolNotFoundError on the first invoke, then succeeds."""

    def __init__(self, fallback_tool: _StubTool) -> None:
        self._fallback_tool = fallback_tool
        self._raise_count = 0
        self._registered: dict[str, Tool] = {}
        self.names_called = False

    def invoke(self, name: str, args: JsonObject) -> ToolResult:
        if name not in self._registered and self._raise_count == 0:
            self._raise_count += 1
            raise ToolNotFoundError(f"Tool not registered: {name}")
        tool = self._registered.get(name, self._fallback_tool)
        return tool.invoke(args)

    def register_if_absent(self, tool: Tool) -> bool:
        if tool.name in self._registered:
            return False
        self._registered[tool.name] = tool
        return True

    def schemas(self) -> tuple[JsonSchema, ...]:
        return ()

    def names(self) -> tuple[str, ...]:
        self.names_called = True
        return tuple(self._registered)

    def get(self, name: str) -> Tool:
        if name not in self._registered:
            raise ToolNotFoundError(f"Tool not registered: {name}")
        return self._registered[name]

    def has(self, name: str) -> bool:
        return name in self._registered


def _make_fake_task(input_text: str = "do something") -> MagicMock:
    """Build a task mock with attributes the coordinator reads as scalars."""
    task = MagicMock()
    task.id = uuid.uuid4()
    task.attempts = 0  # dedup guard: 0 means skip ledger lookup
    task.installation_id = uuid.uuid4()
    task.slack_channel_id = "C123"
    task.slack_thread_ts = None
    task.slack_user_id = "U123"
    task.identity_payload = {}
    task.input = input_text
    # None causes _warn_if_tool_deadline_exceeds_lease to return early.
    task.lease_expires_at = None
    return task


def _make_coordinator(
    registry: _RaisingRegistry,
    completions: list[Completion],
    task: MagicMock,
    connected_tool_loader: ConnectedToolLoader | None = None,
) -> AgentCoordinator:
    """Build a minimal AgentCoordinator.

    ``task_service.get_task`` returns the *same* task mock so attribute
    assignments made before calling ``coordinator.run(task)`` are visible
    inside ``_resolve_task``.
    """
    session = MagicMock()

    task_service = MagicMock()
    task_service.raise_if_cancelled.return_value = None
    task_service.append_event.return_value = None
    # Return the SAME task object so coordinator.run(task) sees correct attrs.
    task_service.get_task.return_value = task

    llm = MagicMock()
    call_iter = iter(completions)

    def _complete(**kwargs: object) -> Completion:
        return next(call_iter)

    llm.complete.side_effect = _complete

    return AgentCoordinator(
        session=session,
        llm=llm,
        registry=registry,  # type: ignore[arg-type]
        task_service=task_service,
        system_prompt=None,
        max_turns=3,
        connected_tool_loader=connected_tool_loader,
    )


def _make_usage() -> TokenUsage:
    return TokenUsage(input_tokens=10, output_tokens=5)


def _make_context_pkg(input_text: str) -> MagicMock:
    context_pkg = MagicMock()
    context_pkg.messages = (ChatMessage(role="user", content=input_text),)
    context_pkg.omissions = ()
    context_pkg.selected_facts = ()
    context_pkg.selected_prior_tasks = ()
    context_pkg.selected_episodes = ()
    context_pkg.selected_artifacts = ()
    context_pkg.selected_graph_entities = ()
    context_pkg.selected_graph_edges = ()
    context_pkg.acknowledgement = None
    context_pkg.budget = MagicMock()
    context_pkg.selected_skills = ()
    context_pkg.context_engine_id = "test"
    context_pkg.context_engine_name = "Test"
    context_pkg.skill_similarities = ()
    context_pkg.execution_hint = None
    context_pkg.matched_skill_slug = None
    return context_pkg


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_direct_callability_lazy_loads_connected_tool() -> None:
    """Loader returns a tool on ToolNotFoundError -> tool is registered + called."""
    stub = _StubTool("composio_linear_list_issues", result_output={"issues": []})
    raising_registry = _RaisingRegistry(fallback_tool=stub)

    loaded_tool: list[Tool] = []

    def loader(name: str) -> Tool | None:
        if name == "composio_linear_list_issues":
            loaded_tool.append(stub)
            return stub
        return None

    tool_call = ToolCall(
        id="tc1",
        name="composio_linear_list_issues",
        arguments={"limit": 10},
    )
    completions = [
        Completion(
            content=None,
            tool_calls=(tool_call,),
            model="test",
            response_id="r1",
            usage=_make_usage(),
        ),
        Completion(
            content="Done.",
            tool_calls=(),
            model="test",
            response_id="r2",
            usage=_make_usage(),
        ),
    ]

    task = _make_fake_task("list my issues")
    coordinator = _make_coordinator(
        raising_registry, completions, task=task, connected_tool_loader=loader
    )
    context_pkg = _make_context_pkg("list my issues")

    with patch.object(coordinator.context_engine, "assemble", return_value=context_pkg):
        result = coordinator.run(task)

    # The loader was called and returned the stub tool.
    assert len(loaded_tool) == 1
    # The tool was registered into the raising registry.
    assert "composio_linear_list_issues" in raising_registry._registered
    # Coordinator completed successfully.
    assert result.result_summary == "Done."


def test_direct_callability_unknown_slug_errors() -> None:
    """Loader returns None -> coordinator falls through to recoverable error."""
    stub = _StubTool("some_other_tool")
    raising_registry = _RaisingRegistry(fallback_tool=stub)

    loader_calls: list[str] = []

    def loader(name: str) -> Tool | None:
        loader_calls.append(name)
        return None  # always returns None

    tool_call = ToolCall(
        id="tc2",
        name="composio_unknown_tool_xyz",
        arguments={},
    )
    completions = [
        Completion(
            content=None,
            tool_calls=(tool_call,),
            model="test",
            response_id="r1",
            usage=_make_usage(),
        ),
        Completion(
            content="I cannot do that.",
            tool_calls=(),
            model="test",
            response_id="r2",
            usage=_make_usage(),
        ),
    ]

    task = _make_fake_task("do the thing")
    coordinator = _make_coordinator(
        raising_registry, completions, task=task, connected_tool_loader=loader
    )
    context_pkg = _make_context_pkg("do the thing")

    with patch.object(coordinator.context_engine, "assemble", return_value=context_pkg):
        result = coordinator.run(task)

    # Loader was consulted for the unknown tool.
    assert "composio_unknown_tool_xyz" in loader_calls
    # Task still completes (recoverable error fed back to model, which replied).
    assert "cannot" in result.result_summary.lower()


def test_direct_callability_no_loader_errors() -> None:
    """No loader set -> ToolNotFoundError yields the standard recoverable error."""
    stub = _StubTool("some_tool")
    raising_registry = _RaisingRegistry(fallback_tool=stub)

    tool_call = ToolCall(
        id="tc3",
        name="composio_linear_get_issues",
        arguments={},
    )
    completions = [
        Completion(
            content=None,
            tool_calls=(tool_call,),
            model="test",
            response_id="r1",
            usage=_make_usage(),
        ),
        Completion(
            content="I could not find that tool.",
            tool_calls=(),
            model="test",
            response_id="r2",
            usage=_make_usage(),
        ),
    ]

    task = _make_fake_task("get issues")
    # No connected_tool_loader passed.
    coordinator = _make_coordinator(raising_registry, completions, task=task)
    context_pkg = _make_context_pkg("get issues")

    with patch.object(coordinator.context_engine, "assemble", return_value=context_pkg):
        result = coordinator.run(task)

    # No loader -> recoverable error surfaced; model responded gracefully.
    assert result.result_summary
