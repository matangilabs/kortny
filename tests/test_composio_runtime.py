import os
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, delete, select
from sqlalchemy.orm import Session

from kortny.composio import (
    ComposioClient,
    ComposioConnectionResolver,
    ComposioTool,
    ComposioToolExecution,
)
from kortny.config.settings import LLMProvider as SettingsLLMProvider
from kortny.config.settings import Settings
from kortny.db.models import (
    ComposioConnection,
    Installation,
    Task,
    TaskEvent,
    TaskEventType,
)
from kortny.db.session import make_engine, make_session_factory, normalize_database_url
from kortny.tasks import TaskService
from kortny.tools import RecoverableToolError, ToolResult
from kortny.tools.composio_execute import (
    ComposioExecuteTool,
    composio_runtime_tool_name,
)
from kortny.tools.types import JsonObject, JsonSchema
from kortny.worker import AgentTaskExecutor

TEST_POSTGRES_URL = os.environ.get("KORTNY_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    TEST_POSTGRES_URL is None,
    reason="KORTNY_TEST_POSTGRES_URL is required for Composio runtime tests",
)


@pytest.fixture(scope="session")
def engine() -> Iterator[Engine]:
    assert TEST_POSTGRES_URL is not None

    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", normalize_database_url(TEST_POSTGRES_URL))
    command.upgrade(config, "head")

    engine = make_engine(TEST_POSTGRES_URL)
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture
def db_session(engine: Engine) -> Iterator[Session]:
    session_factory = make_session_factory(engine=engine)
    with session_factory() as session:
        cleanup_database(session)
        session.commit()
        yield session
        session.rollback()
        cleanup_database(session)
        session.commit()


def test_composio_resolver_prefers_personal_scope(
    db_session: Session,
) -> None:
    task = create_task(db_session, slack_channel_id="CAlpha", slack_user_id="UAneesh")
    add_connection(
        db_session,
        task,
        connected_account_id="ca_workspace",
        scope_type="workspace",
        scope_id=None,
    )
    add_connection(
        db_session,
        task,
        connected_account_id="ca_channel",
        scope_type="channel",
        scope_id="CAlpha",
    )
    add_connection(
        db_session,
        task,
        connected_account_id="ca_personal",
        scope_type="user",
        scope_id="UAneesh",
    )
    db_session.commit()

    resolver = ComposioConnectionResolver(db_session, task)

    allowed = resolver.allowed_connections(toolkit_slug="firecrawl")
    assert [connection.connected_account_id for connection in allowed] == [
        "ca_personal",
        "ca_channel",
        "ca_workspace",
    ]
    assert resolver.best_connection(toolkit_slug="firecrawl") == allowed[0]


def test_composio_resolver_filters_to_matching_visibility(
    db_session: Session,
) -> None:
    task = create_task(
        db_session, slack_channel_id="CAllowed", slack_user_id="UAllowed"
    )
    add_connection(
        db_session,
        task,
        connected_account_id="ca_allowed_channel",
        scope_type="channel",
        scope_id="CAllowed",
    )
    add_connection(
        db_session,
        task,
        connected_account_id="ca_other_channel",
        scope_type="channel",
        scope_id="COther",
    )
    add_connection(
        db_session,
        task,
        connected_account_id="ca_disabled",
        scope_type="workspace",
        scope_id=None,
        status="disabled",
    )
    db_session.commit()

    resolver = ComposioConnectionResolver(db_session, task)

    allowed = resolver.allowed_connections(toolkit_slug="firecrawl")
    assert [connection.connected_account_id for connection in allowed] == [
        "ca_allowed_channel"
    ]


def test_connected_toolkit_slugs_returns_deduped_active_toolkits(
    db_session: Session,
) -> None:
    from kortny.composio.runtime import connected_toolkit_slugs

    task = create_task(db_session, slack_channel_id="CAlpha", slack_user_id="UAneesh")
    add_connection(
        db_session,
        task,
        connected_account_id="ca_linear",
        scope_type="user",
        scope_id="UAneesh",
        toolkit_slug="linear",
    )
    add_connection(
        db_session,
        task,
        connected_account_id="ca_notion",
        scope_type="user",
        scope_id="UAneesh",
        toolkit_slug="notion",
    )
    # Disabled and out-of-scope connections must not appear in the snapshot.
    add_connection(
        db_session,
        task,
        connected_account_id="ca_disabled",
        scope_type="user",
        scope_id="UAneesh",
        toolkit_slug="alpaca",
        status="disabled",
    )
    add_connection(
        db_session,
        task,
        connected_account_id="ca_other_user",
        scope_type="user",
        scope_id="USomeoneElse",
        toolkit_slug="vercel",
    )
    db_session.commit()

    slugs = connected_toolkit_slugs(db_session, task)

    assert set(slugs) == {"linear", "notion"}


def test_connected_toolkit_slugs_for_scope_matches_task_path(
    db_session: Session,
) -> None:
    """The pre-task scope primitive grounds identically to the task path.

    The soft channel-mention surface classifies before a Task row exists, so it
    grounds from an ``IngressConnectionScope`` built from the raw event. That
    must yield the same toolkits the persisted-task path would (HIG-269).
    """

    from kortny.composio.runtime import (
        IngressConnectionScope,
        connected_toolkit_slugs,
        connected_toolkit_slugs_for_scope,
    )

    task = create_task(db_session, slack_channel_id="CAlpha", slack_user_id="UAneesh")
    add_connection(
        db_session,
        task,
        connected_account_id="ca_linear",
        scope_type="user",
        scope_id="UAneesh",
        toolkit_slug="linear",
    )
    add_connection(
        db_session,
        task,
        connected_account_id="ca_channel_notion",
        scope_type="channel",
        scope_id="CAlpha",
        toolkit_slug="notion",
    )
    # A user-scoped connection for someone else must stay invisible to this scope.
    add_connection(
        db_session,
        task,
        connected_account_id="ca_other_user",
        scope_type="user",
        scope_id="USomeoneElse",
        toolkit_slug="vercel",
    )
    db_session.commit()

    scope = IngressConnectionScope(
        installation_id=task.installation_id,
        slack_channel_id="CAlpha",
        slack_user_id="UAneesh",
    )
    scope_slugs = connected_toolkit_slugs_for_scope(db_session, scope)

    assert set(scope_slugs) == {"linear", "notion"}
    assert set(scope_slugs) == set(connected_toolkit_slugs(db_session, task))


def test_composio_execute_tool_uses_scoped_connection(
    db_session: Session,
) -> None:
    task = create_task(
        db_session, slack_channel_id="CResearch", slack_user_id="UAnalyst"
    )
    add_connection(
        db_session,
        task,
        connected_account_id="ca_firecrawl",
        scope_type="channel",
        scope_id="CResearch",
        display_name="Firecrawl research key",
    )
    db_session.commit()
    client = FakeComposioClient()

    tool = ComposioExecuteTool(
        session=db_session,
        task=task,
        client=client,
        tool=firecrawl_tools()[0],
    )

    assert tool.has_available_connection is True
    assert tool.parameters["required"] == ["url"]
    result = tool.invoke({"url": "https://example.com"})

    assert client.calls == [
        {
            "tool_slug": "FIRECRAWL_SCRAPE",
            "user_id": f"slack:{task.installation_id}:UAnalyst",
            "connected_account_id": "ca_firecrawl",
            "arguments": {"url": "https://example.com"},
            "version": None,
        }
    ]
    assert result.output["provider"] == "composio"
    assert result.output["successful"] is True
    assert result.output["data"] == {"markdown": "# Example"}
    assert result.output["scope"] == {"type": "channel", "id": "CResearch"}
    assert result.output["connection"] == {
        "display_name": "Firecrawl research key",
        "connected_account_id": "ca_firecrawl",
    }


def test_composio_execute_tool_strips_blank_optional_arguments(
    db_session: Session,
) -> None:
    task = create_task(
        db_session, slack_channel_id="CResearch", slack_user_id="UAnalyst"
    )
    add_connection(
        db_session,
        task,
        connected_account_id="ca_linear",
        scope_type="user",
        scope_id="UAnalyst",
        toolkit_slug="linear",
    )
    db_session.commit()
    client = FakeComposioClient()
    tool = ComposioExecuteTool(
        session=db_session,
        task=task,
        client=client,
        tool=ComposioTool(
            slug="LINEAR_LIST_LINEAR_PROJECTS",
            name="List Linear projects",
            description="List Linear projects.",
            toolkit_slug="linear",
            input_parameters={
                "type": "object",
                "properties": {
                    "after": {"type": "string"},
                    "first": {"type": "integer"},
                },
            },
            tags=("readOnlyHint",),
            version=None,
        ),
    )

    tool.invoke({"after": "", "first": 250})

    assert client.calls[0]["arguments"] == {"first": 250}


def test_composio_execute_tool_reports_missing_required_args_as_recoverable(
    db_session: Session,
) -> None:
    task = create_task(
        db_session, slack_channel_id="CResearch", slack_user_id="UAnalyst"
    )
    add_connection(
        db_session,
        task,
        connected_account_id="ca_firecrawl",
        scope_type="workspace",
        scope_id=None,
    )
    db_session.commit()
    tool = ComposioExecuteTool(
        session=db_session,
        task=task,
        client=FakeComposioClient(),
        tool=firecrawl_tools()[0],
    )

    with pytest.raises(RecoverableToolError, match="missing required") as exc_info:
        tool.invoke({})
    assert exc_info.value.code == "missing_required_arguments"
    assert exc_info.value.details == {"missing_fields": ["url"]}


def test_composio_runtime_tool_names_are_specific_and_bounded() -> None:
    assert (
        composio_runtime_tool_name("notion", "NOTION_SEARCH_NOTION_PAGE")
        == "composio_notion_search_notion_page"
    )
    long_name = composio_runtime_tool_name(
        "affinda",
        "AFFINDA_CREATE_JOB_DESCRIPTION_SEARCH_EMBED_URL",
    )
    assert len(long_name) <= 64
    assert long_name.startswith("composio_affinda_create_job_description")


def test_worker_registry_exposes_integration_inventory_for_capability_lookup(
    db_session: Session,
    tmp_path: Path,
) -> None:
    task = create_task(db_session, slack_channel_id="CAlpha", slack_user_id="UAneesh")
    task.input = "What integrations do you have?"
    add_connection(
        db_session,
        task,
        connected_account_id="ca_notion",
        scope_type="workspace",
        scope_id=None,
        toolkit_slug="notion",
        display_name="Notion workspace",
    )
    add_connection(
        db_session,
        task,
        connected_account_id="ca_firecrawl",
        scope_type="user",
        scope_id="UAneesh",
        toolkit_slug="firecrawl",
        display_name="Firecrawl personal",
    )
    task_service = TaskService(db_session)
    task_service.append_event(
        task,
        TaskEventType.log,
        {
            "message": "intent_classification_completed",
            "source": "app_mention",
            "decision": {
                "addressed_to_kortny": True,
                "classification": "task_request",
                "confidence": 0.95,
                "should_create_task": True,
                "should_ack_with_reaction": False,
                "needs_channel_context": False,
                "needs_thread_context": False,
                "needs_file_context": False,
                "likely_tools": ["describe_tools", "tool_metadata_lookup"],
                "model_tier": "cheap",
                "reason": "Capability lookup.",
            },
        },
    )
    db_session.commit()
    settings = build_settings(
        composio_api_key="composio-key",
        sandbox_runner_url="http://sandbox-runner:8090",
    )
    composio_client = FakeComposioClient()

    registry = AgentTaskExecutor(
        settings=settings,
        web_search_tool=StaticWebSearchTool(),
        composio_client=composio_client,
    )._build_registry(
        settings=settings,
        session=db_session,
        task=task,
        task_service=task_service,
        working_dir=tmp_path,
    )
    result = registry.invoke("describe_tools", {})

    assert composio_client.list_tool_calls == []
    user_summary = result.output["user_facing_summary"]
    assert user_summary["preferred_opening"] == (
        "Here are the things I can help with right now:"
    )
    assert {group["label"] for group in user_summary["capability_groups"]} >= {
        "Research and current info",
        "Slack context",
        "Slack actions",
        "Scheduled work",
        "Workspace knowledge",
        "Sandboxed coding workbench",
    }
    assert "Runtime" not in {
        group["label"] for group in user_summary["capability_groups"]
    }
    assert {(app["app"], app["scope"]) for app in user_summary["connected_apps"]} == {
        ("Firecrawl", "personal"),
        ("Notion", "workspace"),
    }
    native_tools = result.output["native_tools"]
    assert {tool["name"] for tool in native_tools} >= {
        "web_search",
        "slack_channel_history",
        "search_observed_slack_history",
        "resolve_slack_identity",
        "slack_user_info",
        "slack_channel_info",
        "slack_reply_thread",
        "slack_add_reaction",
        "slack_pin_message",
        "slack_add_bookmark",
        "slack_create_channel_canvas",
        "slack_lookup_canvas_sections",
        "slack_edit_canvas",
        "slack_file_read",
        "code_exec",
        "describe_tools",
    }
    code_exec = next(tool for tool in native_tools if tool["name"] == "code_exec")
    assert code_exec["category"] == "Execution"
    assert code_exec["approval"] == "none"
    assert code_exec["sandbox"]["requires_sandbox"] is True
    assert code_exec["sandbox"]["network"] == "none"
    assert code_exec["required_env_vars"] == ["KORTNY_SANDBOX_RUNNER_URL"]
    web_search = next(tool for tool in native_tools if tool["name"] == "web_search")
    assert web_search["category"] == "Research"
    assert web_search["side_effect"] == "read"
    assert web_search["capabilities"] == ["web_search", "current_research"]
    assert web_search["required_env_vars"] == ["BRAVE_SEARCH_API_KEY"]
    assert {
        connection["toolkit_slug"]
        for connection in result.output["connected_integrations"]
    } == {"firecrawl", "notion"}
    alias_result = registry.invoke("list_integrations", {})
    assert (
        alias_result.output["native_tool_count"] == result.output["native_tool_count"]
    )
    assert not any(
        event.payload.get("message") == "native_tool_unavailable"
        and event.payload.get("tool") == "code_exec"
        for event in task_events(db_session, task)
    )


def test_worker_registry_omits_code_exec_when_sandbox_runner_is_unconfigured(
    db_session: Session,
    tmp_path: Path,
) -> None:
    task = create_task(db_session, slack_channel_id="CAlpha", slack_user_id="UAneesh")
    task.input = "What tools do you have?"
    task_service = TaskService(db_session)
    task_service.append_event(
        task,
        TaskEventType.log,
        {
            "message": "intent_classification_completed",
            "source": "app_mention",
            "decision": {
                "addressed_to_kortny": True,
                "classification": "task_request",
                "confidence": 0.95,
                "should_create_task": True,
                "should_ack_with_reaction": False,
                "needs_channel_context": False,
                "needs_thread_context": False,
                "needs_file_context": False,
                "likely_tools": ["describe_tools"],
                "model_tier": "cheap",
                "reason": "Capability lookup.",
            },
        },
    )
    db_session.commit()
    settings = build_settings(composio_api_key="composio-key")

    registry = AgentTaskExecutor(
        settings=settings,
        web_search_tool=StaticWebSearchTool(),
        composio_client=FakeComposioClient(),
    )._build_registry(
        settings=settings,
        session=db_session,
        task=task,
        task_service=task_service,
        working_dir=tmp_path,
    )

    result = registry.invoke("describe_tools", {})
    native_tools = result.output["native_tools"]

    assert "code_exec" not in registry.names()
    assert "code_exec" not in {tool["name"] for tool in native_tools}
    assert "Sandboxed coding workbench" not in {
        group["label"]
        for group in result.output["user_facing_summary"]["capability_groups"]
    }
    assert any(
        event.payload.get("message") == "native_tool_unavailable"
        and event.payload.get("tool") == "code_exec"
        and event.payload.get("reason") == "missing_sandbox_runner_url"
        for event in task_events(db_session, task)
    )


def cleanup_database(session: Session) -> None:
    for model in (
        ComposioConnection,
        TaskEvent,
        Task,
        Installation,
    ):
        session.execute(delete(model))


def create_task(
    session: Session,
    *,
    slack_channel_id: str,
    slack_user_id: str,
) -> Task:
    installation = Installation(slack_team_id=f"T{uuid.uuid4().hex}")
    session.add(installation)
    session.flush()
    return TaskService(session).create_task(
        installation_id=installation.id,
        slack_event_id=f"Ev{uuid.uuid4().hex}",
        slack_channel_id=slack_channel_id,
        slack_thread_ts="1779660000.000001",
        slack_message_ts="1779660000.000001",
        slack_user_id=slack_user_id,
        input="Use Firecrawl to inspect this website",
    )


def task_events(session: Session, task: Task) -> list[TaskEvent]:
    session.flush()
    return list(
        session.scalars(
            select(TaskEvent)
            .where(TaskEvent.task_id == task.id)
            .order_by(TaskEvent.seq)
        )
    )


def add_connection(
    session: Session,
    task: Task,
    *,
    connected_account_id: str,
    scope_type: str,
    scope_id: str | None,
    toolkit_slug: str = "firecrawl",
    status: str = "active",
    display_name: str | None = None,
) -> None:
    session.add(
        ComposioConnection(
            installation_id=task.installation_id,
            toolkit_slug=toolkit_slug,
            auth_config_id=f"ac_{connected_account_id}",
            connected_account_id=connected_account_id,
            connection_request_id=f"ln_{connected_account_id}",
            composio_user_id=f"slack:{task.installation_id}:{task.slack_user_id}",
            owner_slack_user_id=task.slack_user_id,
            visibility_scope_type=scope_type,
            visibility_scope_id=scope_id,
            status=status,
            display_name=display_name,
        )
    )


def build_settings(
    *,
    composio_api_key: str = "composio-key",
    brave_search_api_key: str | None = "test-brave-key",
    sandbox_runner_url: str | None = None,
    tool_selector_max_external_candidates: int | None = None,
) -> Settings:
    assert TEST_POSTGRES_URL is not None
    kwargs: dict[str, Any] = {
        "SLACK_BOT_TOKEN": "xoxb-test-token",
        "SLACK_APP_TOKEN": "xapp-test-token",
        "SLACK_SIGNING_SECRET": "test-signing-secret",
        "LLM_PROVIDER": SettingsLLMProvider.openrouter,
        "LLM_API_KEY": "test-llm-key",
        "LLM_MODEL": "openai/gpt-5.4-mini",
        "COMPOSIO_API_KEY": composio_api_key,
        "TOOL_SELECTOR_MAX_EXTERNAL_CANDIDATES": (
            tool_selector_max_external_candidates or 24
        ),
        "BRAVE_SEARCH_API_KEY": brave_search_api_key,
        "POSTGRES_URL": TEST_POSTGRES_URL,
        # Tests must never load a real embedding model (HIG-219); the semantic
        # retrieval path is covered with fake backends elsewhere.
        "KORTNY_EMBEDDINGS_BACKEND": "disabled",
    }
    if sandbox_runner_url is not None:
        kwargs["KORTNY_SANDBOX_RUNNER_URL"] = sandbox_runner_url
    return Settings(**kwargs)


class FakeComposioClient(ComposioClient):
    def __init__(
        self,
        *,
        tools_by_toolkit: dict[str, tuple[ComposioTool, ...]] | None = None,
    ) -> None:
        super().__init__(api_key="fake")
        self.calls: list[dict[str, Any]] = []
        self.list_tool_calls: list[dict[str, Any]] = []
        self.tools_by_toolkit = tools_by_toolkit or {"firecrawl": firecrawl_tools()}

    def list_tools(
        self,
        *,
        toolkit_slug: str | None = None,
        tool_slugs: tuple[str, ...] = (),
        query: str | None = None,
        limit: int = 20,
    ) -> tuple[ComposioTool, ...]:
        self.list_tool_calls.append(
            {
                "toolkit_slug": toolkit_slug,
                "tool_slugs": tool_slugs,
                "query": query,
                "limit": limit,
            }
        )
        tools = self.tools_by_toolkit.get(toolkit_slug or "", ())
        if tool_slugs:
            allowed = set(tool_slugs)
            tools = tuple(tool for tool in tools if tool.slug in allowed)
        return tools[:limit]

    def execute_tool(
        self,
        *,
        tool_slug: str,
        user_id: str,
        connected_account_id: str,
        arguments: dict[str, Any],
        version: str | None = None,
    ) -> ComposioToolExecution:
        self.calls.append(
            {
                "tool_slug": tool_slug,
                "user_id": user_id,
                "connected_account_id": connected_account_id,
                "arguments": arguments,
                "version": version,
            }
        )
        return ComposioToolExecution(
            data={"markdown": "# Example"},
            successful=True,
            error=None,
            log_id="log_firecrawl",
            session_info=None,
        )


def firecrawl_tools() -> tuple[ComposioTool, ...]:
    return (
        ComposioTool(
            slug="FIRECRAWL_SCRAPE",
            name="Scrape URL",
            description="Scrape markdown from a URL.",
            toolkit_slug="firecrawl",
            input_parameters={
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
            tags=("readOnlyHint",),
            version=None,
        ),
        ComposioTool(
            slug="FIRECRAWL_SEARCH",
            name="Search Web",
            description="Search the web for sources.",
            toolkit_slug="firecrawl",
            input_parameters={
                "type": "object",
                "properties": {"q": {"type": "string"}},
                "required": ["q"],
            },
            tags=("readOnlyHint",),
            version=None,
        ),
    )


def notion_tools() -> tuple[ComposioTool, ...]:
    return (
        ComposioTool(
            slug="NOTION_SEARCH_NOTION_PAGE",
            name="Search Notion pages and databases",
            description="Search Notion pages and databases by title.",
            toolkit_slug="notion",
            input_parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
            tags=("readOnlyHint",),
            version=None,
        ),
        ComposioTool(
            slug="NOTION_QUERY_DATABASE",
            name="Query database",
            description="Query a Notion database by ID.",
            toolkit_slug="notion",
            input_parameters={
                "type": "object",
                "properties": {
                    "database_id": {"type": "string"},
                    "page_size": {"type": "integer"},
                },
                "required": ["database_id"],
            },
            tags=("readOnlyHint",),
            version=None,
        ),
    )


def alpha_vantage_tools(*, count: int) -> tuple[ComposioTool, ...]:
    return tuple(
        ComposioTool(
            slug=f"ALPHA_VANTAGE_SEARCH_{index}",
            name=f"Search market data {index}",
            description=f"Read-only market data lookup tool {index}.",
            toolkit_slug="alpha_vantage",
            input_parameters={
                "type": "object",
                "properties": {"symbol": {"type": "string"}},
            },
            tags=("readOnlyHint",),
            version=None,
        )
        for index in range(count)
    )


class StaticWebSearchTool:
    name = "web_search"
    description = "Static web search test tool."
    parameters: JsonSchema = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }

    def invoke(self, args: JsonObject) -> ToolResult:
        del args
        return ToolResult(output={"results": []})
