import os
import uuid
from types import SimpleNamespace
from typing import Any, cast

import pytest
from google.adk.models.llm_response import LlmResponse
from google.genai import types as genai_types

from kortny.agent.adk_runtime import AdkAgentRuntime, adk_litellm_model_name
from kortny.config import load_settings
from kortny.db.models import Task
from kortny.tools import ToolRegistry
from kortny.tools.types import JsonObject, ToolResult


def test_adk_model_mapping_prefixes_openrouter_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_required_settings_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("LLM_MODEL", "anthropic/claude-sonnet-test")

    settings = load_settings(env_file=None)

    assert adk_litellm_model_name(settings) == (
        "openrouter/anthropic/claude-sonnet-test"
    )


def test_adk_model_mapping_preserves_existing_openrouter_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_required_settings_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("LLM_MODEL", "openrouter/openai/gpt-test")

    settings = load_settings(env_file=None)

    assert adk_litellm_model_name(settings) == "openrouter/openai/gpt-test"


def test_adk_model_mapping_preserves_direct_provider_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_required_settings_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "openai/gpt-test")

    settings = load_settings(env_file=None)

    assert adk_litellm_model_name(settings) == "openai/gpt-test"


def test_adk_model_mapping_uses_routed_model_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_required_settings_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("LLM_MODEL", "anthropic/sonnet-default")

    settings = load_settings(env_file=None)

    assert (
        adk_litellm_model_name(settings, model="deepseek/deepseek-v4-flash")
        == "openrouter/deepseek/deepseek-v4-flash"
    )


def test_adk_runtime_builds_root_agent_in_chat_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_required_settings_env(monkeypatch)
    settings = load_settings(env_file=None)

    runtime = AdkAgentRuntime(
        settings=settings,
        session=cast(Any, None),
        task_service=cast(Any, None),
    )

    agent = runtime._build_agent()

    assert agent.name == "kortny_root_orchestrator"
    assert agent.mode == "chat"
    assert [getattr(tool, "name", None) for tool in agent.tools] == [
        "intent_triage_agent",
        "quick_response_agent",
        "clarification_agent",
        "eval_agent",
        "humanizer_agent",
    ]


def test_adk_runtime_default_prompt_is_text_only_and_not_tool_claiming(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_required_settings_env(monkeypatch)
    settings = load_settings(env_file=None)

    runtime = AdkAgentRuntime(
        settings=settings,
        session=cast(Any, None),
        task_service=cast(Any, None),
    )

    instruction = runtime._instruction()

    assert "Do not introduce yourself unless the user asks" in instruction
    assert "no tools are connected yet" in instruction
    assert "live integrations" in instruction
    assert "Use the available tools" not in instruction


def test_adk_runtime_prompt_switches_when_tools_are_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_required_settings_env(monkeypatch)
    settings = load_settings(env_file=None)

    runtime = AdkAgentRuntime(
        settings=settings,
        session=cast(Any, None),
        task_service=cast(Any, None),
        registry=ToolRegistry([_EchoTool()]),
    )

    instruction = runtime._instruction()

    assert "ADK agentic orchestration" in instruction
    assert "tool_worker_agent" in instruction
    assert "no tools are connected yet" not in instruction


def test_adk_runtime_prompts_keep_single_kortny_persona(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_required_settings_env(monkeypatch)
    settings = load_settings(env_file=None)
    task = Task(
        id=uuid.UUID("5c53f4e1-9d72-468d-ab18-5021d9e15dad"),
        installation_id=uuid.UUID("1c53f4e1-9d72-468d-ab18-5021d9e15dad"),
        slack_channel_id="C123",
        slack_thread_ts="123.456",
        slack_user_id="U123",
        input="what tools do you have access to?",
    )
    runtime = AdkAgentRuntime(
        settings=settings,
        session=cast(Any, None),
        task_service=cast(Any, None),
        registry=ToolRegistry([_EchoTool()]),
    )

    root_instruction = runtime._instruction()
    agent = runtime._build_agent(task=task)
    agent_by_name = {getattr(tool, "name", None): tool.agent for tool in agent.tools}
    quick_instruction = agent_by_name["quick_response_agent"].instruction
    worker_instruction = agent_by_name["tool_worker_agent"].instruction

    assert "Speak as Kortny, a single Slack-native coworker" in root_instruction
    assert "use tool_worker_agent when it is available" in root_instruction
    assert "Do not call or invent Slack posting/reply tools" in root_instruction
    assert "Speak as Kortny, a single Slack-native coworker" in quick_instruction
    assert "Do not say actual tool access lives in another agent" in quick_instruction
    assert "answer as Kortny" in worker_instruction
    assert "actual tool access lives in the main Kortny agent" not in quick_instruction
    assert "actual tool access lives in the main Kortny agent" not in worker_instruction


def test_adk_runtime_builds_orchestrator_with_registry_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_required_settings_env(monkeypatch)
    settings = load_settings(env_file=None)
    task = Task(
        id=uuid.UUID("2c53f4e1-9d72-468d-ab18-5021d9e15dad"),
        installation_id=uuid.UUID("1c53f4e1-9d72-468d-ab18-5021d9e15dad"),
        slack_channel_id="C123",
        slack_thread_ts="123.456",
        slack_user_id="U123",
        input="echo this",
    )
    runtime = AdkAgentRuntime(
        settings=settings,
        session=cast(Any, None),
        task_service=cast(Any, None),
        registry=ToolRegistry([_EchoTool()]),
    )

    agent = runtime._build_agent(task=task)

    assert [getattr(tool, "name", None) for tool in agent.tools] == [
        "intent_triage_agent",
        "quick_response_agent",
        "clarification_agent",
        "tool_worker_agent",
        "eval_agent",
        "humanizer_agent",
    ]
    worker_tool = next(
        tool
        for tool in agent.tools
        if getattr(tool, "name", None) == "tool_worker_agent"
    )
    worker_agent = worker_tool.agent
    assert [getattr(tool, "name", None) for tool in worker_agent.tools] == ["echo_tool"]


def test_adk_runtime_short_availability_check_uses_direct_quick_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_required_settings_env(monkeypatch)
    monkeypatch.setenv("LLM_MODEL", "anthropic/sonnet-default")
    monkeypatch.setenv("LLM_CHEAP_MODEL", "deepseek/deepseek-v4-flash")
    settings = load_settings(env_file=None)
    task = Task(
        id=uuid.UUID("9c53f4e1-9d72-468d-ab18-5021d9e15dad"),
        installation_id=uuid.UUID("1c53f4e1-9d72-468d-ab18-5021d9e15dad"),
        slack_channel_id="D123",
        slack_thread_ts="D123",
        slack_user_id="U123",
        input="Are you up?",
    )
    task_service = _FakeTaskService()
    factory_called = False

    def registry_factory() -> ToolRegistry:
        nonlocal factory_called
        factory_called = True
        return ToolRegistry([_EchoTool()])

    runtime = AdkAgentRuntime(
        settings=settings,
        session=cast(
            Any,
            _FakeScalarSession(
                [
                    {
                        "message": "runtime_handoff_evaluated",
                        "runtime_class": "quick_response",
                        "selected_backend": "inline",
                    },
                    {
                        "message": "planned_workflow_classified",
                        "reason_codes": ["quick_conversation"],
                    },
                ]
            ),
        ),
        task_service=cast(Any, task_service),
        registry_factory=registry_factory,
    )

    agent = runtime._build_agent(task=task)

    assert agent.name == "quick_response_agent"
    assert agent.model.model == "openrouter/deepseek/deepseek-v4-flash"
    assert not factory_called
    assert task_service.events == [
        (
            task,
            {
                "message": "adk_quick_response_selected",
                "runtime": "adk",
                "agent": "quick_response_agent",
                "reason": "runtime_handoff_quick_conversation",
            },
        )
    ]


def test_adk_runtime_tool_inventory_question_stays_orchestrated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_required_settings_env(monkeypatch)
    settings = load_settings(env_file=None)
    task = Task(
        id=uuid.UUID("0c53f4e1-9d72-468d-ab18-5021d9e15dad"),
        installation_id=uuid.UUID("1c53f4e1-9d72-468d-ab18-5021d9e15dad"),
        slack_channel_id="D123",
        slack_thread_ts="D123",
        slack_user_id="U123",
        input="What tools do you have?",
    )
    runtime = AdkAgentRuntime(
        settings=settings,
        session=cast(Any, _FakeScalarSession([])),
        task_service=cast(Any, _FakeTaskService()),
        registry=ToolRegistry([_EchoTool()]),
    )

    agent = runtime._build_agent(task=task)

    assert agent.name == "kortny_root_orchestrator"


def test_adk_runtime_planned_branch_model_budget_returns_stop_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_required_settings_env(monkeypatch)
    monkeypatch.setenv("KORTNY_PLANNED_WORKFLOW_MAX_BRANCH_MODEL_CALLS", "2")
    settings = load_settings(env_file=None)
    task = Task(
        id=uuid.UUID("ac53f4e1-9d72-468d-ab18-5021d9e15dad"),
        installation_id=uuid.UUID("1c53f4e1-9d72-468d-ab18-5021d9e15dad"),
        slack_channel_id="D123",
        slack_thread_ts="D123",
        slack_user_id="U123",
        input="Research the top James Bond movies.",
    )
    task_service = _FakeTaskService()
    task_service.tasks[task.id] = task
    runtime = AdkAgentRuntime(
        settings=settings,
        session=cast(Any, _FakeScalarSession([])),
        task_service=cast(Any, task_service),
    )
    callback_context = SimpleNamespace(
        agent_name="planned_research_worker",
        invocation_id="inv-model-budget",
        state={"task_id": str(task.id)},
    )

    assert runtime._guard_planned_model_request(cast(Any, callback_context), object()) is None
    assert runtime._guard_planned_model_request(cast(Any, callback_context), object()) is None
    response = runtime._guard_planned_model_request(
        cast(Any, callback_context),
        object(),
    )
    repeated_response = runtime._guard_planned_model_request(
        cast(Any, callback_context),
        object(),
    )

    assert response is not None
    assert repeated_response is not None
    assert response.content is not None
    assert response.content.parts is not None
    assert response.content.parts[0].text is not None
    assert "max_branch_model_calls_exceeded" in response.content.parts[0].text
    budget_events = [
        payload
        for _, payload in task_service.events
        if payload.get("message") == "adk_planned_branch_budget_exceeded"
    ]
    assert len(budget_events) == 1
    assert budget_events[0] == {
        "message": "adk_planned_branch_budget_exceeded",
        "runtime": "adk",
        "adk_agent_name": "planned_research_worker",
        "adk_invocation_id": "inv-model-budget",
        "budget_type": "model_calls",
        "reason": "max_branch_model_calls_exceeded",
        "limit": 2,
        "observed": 3,
    }


def test_adk_runtime_planned_branch_tool_budget_blocks_tool_and_next_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_required_settings_env(monkeypatch)
    monkeypatch.setenv("KORTNY_PLANNED_WORKFLOW_MAX_BRANCH_TOOL_CALLS", "1")
    settings = load_settings(env_file=None)
    task = Task(
        id=uuid.UUID("bc53f4e1-9d72-468d-ab18-5021d9e15dad"),
        installation_id=uuid.UUID("1c53f4e1-9d72-468d-ab18-5021d9e15dad"),
        slack_channel_id="D123",
        slack_thread_ts="D123",
        slack_user_id="U123",
        input="Research the top James Bond movies.",
    )
    task_service = _FakeTaskService()
    task_service.tasks[task.id] = task
    runtime = AdkAgentRuntime(
        settings=settings,
        session=cast(Any, _FakeScalarSession([])),
        task_service=cast(Any, task_service),
    )
    callback_context = SimpleNamespace(
        agent_name="planned_research_worker",
        invocation_id="inv-tool-budget",
        state={"task_id": str(task.id)},
    )
    tool = SimpleNamespace(name="composio_exa_search")

    assert (
        runtime._guard_planned_tool_call(
            tool,
            {"query": "best bond movies"},
            cast(Any, callback_context),
        )
        is None
    )
    blocked_tool_result = runtime._guard_planned_tool_call(
        tool,
        {"query": "more bond rankings"},
        cast(Any, callback_context),
    )
    model_response = runtime._guard_planned_model_request(
        cast(Any, callback_context),
        object(),
    )

    assert blocked_tool_result is not None
    assert blocked_tool_result["budget_exhausted"] is True
    assert blocked_tool_result["budget_type"] == "tool_calls"
    assert blocked_tool_result["tool"] == "composio_exa_search"
    assert model_response is not None
    assert model_response.content is not None
    assert model_response.content.parts is not None
    assert model_response.content.parts[0].text is not None
    assert "max_branch_tool_calls_exceeded" in model_response.content.parts[0].text
    budget_events = [
        payload
        for _, payload in task_service.events
        if payload.get("message") == "adk_planned_branch_budget_exceeded"
    ]
    assert budget_events == [
        {
            "message": "adk_planned_branch_budget_exceeded",
            "runtime": "adk",
            "adk_agent_name": "planned_research_worker",
            "adk_invocation_id": "inv-tool-budget",
            "budget_type": "tool_calls",
            "reason": "max_branch_tool_calls_exceeded",
            "limit": 1,
            "observed": 2,
            "tool": "composio_exa_search",
        }
    ]


def test_adk_runtime_uses_cheaper_models_for_lightweight_specialists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_required_settings_env(monkeypatch)
    monkeypatch.setenv("LLM_MODEL", "anthropic/sonnet-default")
    monkeypatch.setenv("LLM_CHEAP_MODEL", "deepseek/deepseek-v4-flash")
    monkeypatch.setenv("LLM_STANDARD_MODEL", "openai/gpt-5.4-mini")
    monkeypatch.setenv("LLM_HIGH_REASONING_MODEL", "anthropic/opus-review")
    settings = load_settings(env_file=None)
    task = Task(
        id=uuid.UUID("4c53f4e1-9d72-468d-ab18-5021d9e15dad"),
        installation_id=uuid.UUID("1c53f4e1-9d72-468d-ab18-5021d9e15dad"),
        slack_channel_id="C123",
        slack_thread_ts="123.456",
        slack_user_id="U123",
        input="research this",
    )
    runtime = AdkAgentRuntime(
        settings=settings,
        session=cast(Any, None),
        task_service=cast(Any, None),
        registry=ToolRegistry([_EchoTool()]),
        model="anthropic/sonnet-routed",
    )

    agent = runtime._build_agent(task=task)
    agent_by_name = {getattr(tool, "name", None): tool.agent for tool in agent.tools}

    assert agent.model.model == "openrouter/anthropic/sonnet-routed"
    assert (
        agent_by_name["quick_response_agent"].model.model
        == "openrouter/deepseek/deepseek-v4-flash"
    )
    assert (
        agent_by_name["clarification_agent"].model.model
        == "openrouter/deepseek/deepseek-v4-flash"
    )
    assert (
        agent_by_name["tool_worker_agent"].model.model
        == "openrouter/anthropic/sonnet-routed"
    )
    assert (
        agent_by_name["humanizer_agent"].model.model == "openrouter/openai/gpt-5.4-mini"
    )
    assert agent_by_name["eval_agent"].model.model == "openrouter/anthropic/opus-review"


def test_adk_runtime_builds_planned_parallel_pipeline_for_planned_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_required_settings_env(monkeypatch)
    monkeypatch.setenv("LLM_MODEL", "anthropic/sonnet-default")
    monkeypatch.setenv("LLM_CHEAP_MODEL", "deepseek/deepseek-v4-flash")
    monkeypatch.setenv("LLM_STANDARD_MODEL", "anthropic/sonnet-standard")
    monkeypatch.setenv("LLM_HIGH_REASONING_MODEL", "anthropic/opus-planner")
    settings = load_settings(env_file=None)
    task = Task(
        id=uuid.UUID("6c53f4e1-9d72-468d-ab18-5021d9e15dad"),
        installation_id=uuid.UUID("1c53f4e1-9d72-468d-ab18-5021d9e15dad"),
        slack_channel_id="C123",
        slack_thread_ts="123.456",
        slack_user_id="U123",
        input="research AI observability, check Linear, and summarize next steps",
    )
    task_service = _FakeTaskService()
    runtime = AdkAgentRuntime(
        settings=settings,
        session=cast(Any, None),
        task_service=cast(Any, task_service),
        registry=ToolRegistry([_EchoTool()]),
        model="anthropic/sonnet-routed",
    )

    agent = runtime._build_agent(
        task=task,
        planned_workflow_payload={
            "planned_candidate": True,
            "route": "planned_candidate",
            "confidence": 0.84,
            "estimated_subtask_count": 4,
            "reason": "Task looks like broad research plus synthesis.",
        },
    )

    assert agent.name == "kortny_planned_workflow"
    assert [sub_agent.name for sub_agent in agent.sub_agents] == [
        "planned_workflow_planner",
        "planned_parallel_fanout",
        "planned_workflow_merger",
    ]
    planner, parallel, merger = agent.sub_agents
    assert planner.output_key == "planned_workflow_plan"
    assert planner.model.model == "openrouter/anthropic/opus-planner"
    assert merger.model.model == "openrouter/anthropic/sonnet-standard"
    assert [worker.name for worker in parallel.sub_agents] == [
        "planned_research_worker",
        "planned_workspace_worker",
        "planned_integration_worker",
    ]
    assert {
        worker.output_key: worker.model.model for worker in parallel.sub_agents
    } == {
        "planned_research_result": "openrouter/deepseek/deepseek-v4-flash",
        "planned_workspace_result": "openrouter/deepseek/deepseek-v4-flash",
        "planned_integration_result": "openrouter/deepseek/deepseek-v4-flash",
    }
    assert task_service.events[0][1]["message"] == "adk_planned_workflow_selected"
    assert task_service.events[0][1]["mode"] == "planned_parallel"


def test_adk_runtime_respects_planned_workflow_disable_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_required_settings_env(monkeypatch)
    monkeypatch.setenv("KORTNY_PLANNED_WORKFLOWS_ENABLED", "false")
    settings = load_settings(env_file=None)
    task = Task(
        id=uuid.UUID("7c53f4e1-9d72-468d-ab18-5021d9e15dad"),
        installation_id=uuid.UUID("1c53f4e1-9d72-468d-ab18-5021d9e15dad"),
        slack_channel_id="C123",
        slack_thread_ts="123.456",
        slack_user_id="U123",
        input="research AI observability, check Linear, and summarize next steps",
    )
    runtime = AdkAgentRuntime(
        settings=settings,
        session=cast(Any, None),
        task_service=cast(Any, _FakeTaskService()),
        registry=ToolRegistry([_EchoTool()]),
    )

    agent = runtime._build_agent(
        task=task,
        planned_workflow_payload={
            "planned_candidate": True,
            "route": "planned_candidate",
        },
    )

    assert agent.name == "kortny_root_orchestrator"


def test_adk_runtime_registry_factory_is_lazy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_required_settings_env(monkeypatch)
    settings = load_settings(env_file=None)
    task = Task(
        id=uuid.UUID("3c53f4e1-9d72-468d-ab18-5021d9e15dad"),
        installation_id=uuid.UUID("1c53f4e1-9d72-468d-ab18-5021d9e15dad"),
        slack_channel_id="C123",
        slack_thread_ts="123.456",
        slack_user_id="U123",
        input="are you up?",
    )
    factory_called = False

    def registry_factory() -> ToolRegistry:
        nonlocal factory_called
        factory_called = True
        return ToolRegistry([_EchoTool()])

    runtime = AdkAgentRuntime(
        settings=settings,
        session=cast(Any, None),
        task_service=cast(Any, None),
        registry_factory=registry_factory,
    )

    agent = runtime._build_agent(task=task)

    assert not factory_called
    worker_tool = next(
        tool
        for tool in agent.tools
        if getattr(tool, "name", None) == "tool_worker_agent"
    )
    worker_agent = worker_tool.agent
    assert [type(tool).__name__ for tool in worker_agent.tools] == [
        "KortnyRegistryToolset"
    ]


def test_adk_runtime_suppresses_direct_slack_post_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_required_settings_env(monkeypatch)
    settings = load_settings(env_file=None)
    task = Task(
        id=uuid.UUID("8c53f4e1-9d72-468d-ab18-5021d9e15dad"),
        installation_id=uuid.UUID("1c53f4e1-9d72-468d-ab18-5021d9e15dad"),
        slack_channel_id="C123",
        slack_thread_ts="123.456",
        slack_user_id="U123",
        input="what do you know about this channel?",
    )
    task_service = _FakeTaskService()
    task_service.tasks[task.id] = task
    runtime = AdkAgentRuntime(
        settings=settings,
        session=cast(Any, None),
        task_service=cast(Any, task_service),
        registry=ToolRegistry([_EchoTool()]),
    )
    callback_context = SimpleNamespace(
        agent_name="planned_workspace_worker",
        invocation_id="inv-test",
        state={"task_id": str(task.id)},
    )
    response = LlmResponse(
        content=genai_types.Content(
            role="model",
            parts=[
                genai_types.Part(
                    function_call=genai_types.FunctionCall(
                        name="slack_post_message",
                        args={"text": "Channel summary here."},
                    )
                ),
            ],
        )
    )

    guarded = runtime._record_and_guard_adk_model_response(
        cast(Any, callback_context),
        response,
    )

    assert guarded is not None
    assert guarded.content is not None
    assert [part.text for part in guarded.content.parts or []] == [
        "Channel summary here."
    ]
    assert all(part.function_call is None for part in guarded.content.parts or [])
    assert task_service.events[0][1] == {
        "message": "adk_disallowed_tool_call_suppressed",
        "runtime": "adk",
        "adk_agent_name": "planned_workspace_worker",
        "adk_invocation_id": "inv-test",
        "tool_names": ["slack_post_message"],
        "reason": "direct_slack_posting_is_worker_owned",
    }


class _EchoTool:
    name = "echo_tool"
    description = "Echoes provided text."
    parameters = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    def invoke(self, args: JsonObject) -> ToolResult:
        return ToolResult(output={"echo": args})


class _FakeTaskService:
    def __init__(self) -> None:
        self.events: list[tuple[Task, JsonObject]] = []
        self.tasks: dict[uuid.UUID, Task] = {}

    def append_event(
        self,
        task: Task,
        event_type: object,
        payload: JsonObject,
    ) -> None:
        del event_type
        self.events.append((task, payload))

    def get_task(self, task_id: uuid.UUID) -> Task | None:
        return self.tasks.get(task_id)


class _FakeScalarSession:
    def __init__(self, payloads: list[JsonObject]) -> None:
        self.rows = [SimpleNamespace(payload=payload) for payload in payloads]

    def scalar(self, statement: object) -> object | None:
        del statement
        if not self.rows:
            return None
        return self.rows.pop(0)


def set_required_settings_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "SLACK_BOT_TOKEN",
        "SLACK_APP_TOKEN",
        "SLACK_SIGNING_SECRET",
        "LLM_PROVIDER",
        "LLM_API_KEY",
        "LLM_MODEL",
        "LLM_CHEAP_MODEL",
        "LLM_STANDARD_MODEL",
        "LLM_HIGH_REASONING_MODEL",
        "KORTNY_PLANNED_WORKFLOWS_ENABLED",
        "KORTNY_PLANNED_WORKFLOW_MAX_PARALLEL_BRANCHES",
        "KORTNY_PLANNED_WORKFLOW_COST_CEILING_USD",
        "KORTNY_PLANNED_WORKFLOW_MAX_BRANCH_MODEL_CALLS",
        "KORTNY_PLANNED_WORKFLOW_MAX_BRANCH_TOOL_CALLS",
        "POSTGRES_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    monkeypatch.setenv("SLACK_APP_TOKEN", "xapp-test")
    monkeypatch.setenv("SLACK_SIGNING_SECRET", "signing-secret")
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("LLM_API_KEY", "llm-key")
    monkeypatch.setenv("LLM_MODEL", "openai/gpt-4o")
    monkeypatch.setenv("POSTGRES_URL", "postgresql://kortny:kortny@localhost/kortny")
    os.environ.pop("OPENROUTER_API_KEY", None)
