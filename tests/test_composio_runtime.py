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
from kortny.observe.assessment import CHANNEL_ASSESSMENT_REQUESTED_MESSAGE
from kortny.tasks import TaskService
from kortny.tool_selection import ToolCard, ToolSelection, ToolSelectionResult
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
    task = create_task(db_session, slack_channel_id="CAllowed", slack_user_id="UAllowed")
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


def test_composio_execute_tool_uses_scoped_connection(
    db_session: Session,
) -> None:
    task = create_task(db_session, slack_channel_id="CResearch", slack_user_id="UAnalyst")
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


def test_composio_execute_tool_reports_missing_required_args_as_recoverable(
    db_session: Session,
) -> None:
    task = create_task(db_session, slack_channel_id="CResearch", slack_user_id="UAnalyst")
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


def test_worker_registry_adds_composio_tool_for_scoped_connection(
    db_session: Session,
    tmp_path: Path,
) -> None:
    task = create_task(db_session, slack_channel_id="CResearch", slack_user_id="UAnalyst")
    add_connection(
        db_session,
        task,
        connected_account_id="ca_firecrawl",
        scope_type="user",
        scope_id="UAnalyst",
    )
    db_session.commit()
    settings = build_settings(composio_api_key="composio-key")

    registry = AgentTaskExecutor(
        settings=settings,
        web_search_tool=StaticWebSearchTool(),
        composio_client=FakeComposioClient(),
        tool_selector=StaticToolSelector(
            ToolSelectionResult(
                selected_tools=(
                    ToolSelection(
                        registry_name="composio_firecrawl_scrape",
                        confidence=0.9,
                        reason="Firecrawl is scoped and relevant.",
                    ),
                ),
                suppressed_native_tools=("web_search",),
                route_reason="test_selection",
            )
        ),
    )._build_registry(
        settings=settings,
        session=db_session,
        task=task,
        task_service=TaskService(db_session),
        working_dir=tmp_path,
    )

    assert "composio_firecrawl_scrape" in registry.names()
    assert "web_search" not in registry.names()
    event = next(
        event
        for event in task_events(db_session, task)
        if event.payload.get("message") == "tool_selection_completed"
    )
    assert (
        event.payload["selected_tools"][0]["registry_name"]
        == "composio_firecrawl_scrape"
    )
    assert event.payload["suppressed_native_tools"] == ["web_search"]


def test_worker_registry_uses_dynamic_composio_toolkit_catalog(
    db_session: Session,
    tmp_path: Path,
) -> None:
    task = create_task(db_session, slack_channel_id="CResearch", slack_user_id="UAnalyst")
    task.input = "Find the relevant Notion docs for the launch plan"
    add_connection(
        db_session,
        task,
        connected_account_id="ca_notion",
        scope_type="user",
        scope_id="UAnalyst",
        toolkit_slug="notion",
    )
    db_session.commit()
    settings = build_settings(composio_api_key="composio-key")
    composio_client = FakeComposioClient(
        tools_by_toolkit={
            "notion": (
                ComposioTool(
                    slug="NOTION_SEARCH",
                    name="Search Notion",
                    description="Search Notion pages and databases.",
                    toolkit_slug="notion",
                    input_parameters={
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                    },
                    tags=("readOnlyHint",),
                    version=None,
                ),
            )
        }
    )

    registry = AgentTaskExecutor(
        settings=settings,
        web_search_tool=StaticWebSearchTool(),
        composio_client=composio_client,
        tool_selector=StaticToolSelector(
            ToolSelectionResult(
                selected_tools=(
                    ToolSelection(
                        registry_name="composio_notion_search",
                        confidence=0.9,
                        reason="Notion has scoped matching docs.",
                    ),
                ),
                route_reason="test_selection",
            )
        ),
    )._build_registry(
        settings=settings,
        session=db_session,
        task=task,
        task_service=TaskService(db_session),
        working_dir=tmp_path,
    )

    assert "composio_notion_search" in registry.names()
    tool = registry.get("composio_notion_search")
    assert tool.parameters["properties"]["query"]["type"] == "string"
    assert composio_client.list_tool_calls[0]["toolkit_slug"] == "notion"
    assert composio_client.list_tool_calls[0]["query"] == task.input


def test_worker_registry_exposes_required_fields_for_exact_composio_tool(
    db_session: Session,
    tmp_path: Path,
) -> None:
    task = create_task(db_session, slack_channel_id="CResearch", slack_user_id="UAnalyst")
    task.input = "Are there any actionable items in the connected docs?"
    add_connection(
        db_session,
        task,
        connected_account_id="ca_notion",
        scope_type="workspace",
        scope_id=None,
        toolkit_slug="notion",
    )
    db_session.commit()
    settings = build_settings(composio_api_key="composio-key")
    composio_client = FakeComposioClient(
        tools_by_toolkit={"notion": notion_tools()},
    )

    registry = AgentTaskExecutor(
        settings=settings,
        web_search_tool=StaticWebSearchTool(),
        composio_client=composio_client,
        tool_selector=StaticToolSelector(
            ToolSelectionResult(
                selected_tools=(
                    ToolSelection(
                        registry_name="composio_notion_query_database",
                        confidence=0.9,
                        reason="Database query is relevant only after discovery.",
                    ),
                ),
                route_reason="test_selection",
            )
        ),
    )._build_registry(
        settings=settings,
        session=db_session,
        task=task,
        task_service=TaskService(db_session),
        working_dir=tmp_path,
    )

    tool = registry.get("composio_notion_query_database")
    assert tool.parameters["required"] == ["database_id"]
    with pytest.raises(RecoverableToolError, match="database_id") as exc_info:
        tool.invoke({"page_size": 10})
    assert exc_info.value.code == "missing_required_arguments"
    assert composio_client.calls == []


def test_worker_registry_skips_composio_catalog_for_ignored_intent(
    db_session: Session,
    tmp_path: Path,
) -> None:
    task = create_task(db_session, slack_channel_id="CGreeting", slack_user_id="UAneesh")
    task.input = "hey whats up"
    add_connection(
        db_session,
        task,
        connected_account_id="ca_firecrawl",
        scope_type="user",
        scope_id="UAneesh",
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
                "classification": "ignore",
                "confidence": 0.95,
                "should_create_task": False,
                "should_ack_with_reaction": True,
                "needs_channel_context": False,
                "needs_thread_context": False,
                "needs_file_context": False,
                "likely_tools": [],
                "model_tier": "cheap",
                "reason": "Greeting only.",
            },
        },
    )
    db_session.commit()
    settings = build_settings(composio_api_key="composio-key")
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

    assert composio_client.list_tool_calls == []
    assert "web_search" in registry.names()
    assert all(not name.startswith("composio_") for name in registry.names())
    assert any(
        event.payload.get("message") == "external_tool_selection_skipped"
        and event.payload.get("classification") == "ignore"
        for event in task_events(db_session, task)
    )


def test_worker_registry_skips_composio_catalog_for_simple_no_tool_intent(
    db_session: Session,
    tmp_path: Path,
) -> None:
    task = create_task(db_session, slack_channel_id="DUser", slack_user_id="UAneesh")
    task.input = "Are you up?"
    add_connection(
        db_session,
        task,
        connected_account_id="ca_firecrawl",
        scope_type="user",
        scope_id="UAneesh",
    )
    task_service = TaskService(db_session)
    task_service.append_event(
        task,
        TaskEventType.log,
        {
            "message": "intent_classification_completed",
            "source": "dm",
            "decision": {
                "addressed_to_kortny": True,
                "classification": "task_request",
                "confidence": 0.9,
                "should_create_task": True,
                "should_ack_with_reaction": True,
                "needs_channel_context": False,
                "needs_thread_context": False,
                "needs_file_context": False,
                "likely_tools": [],
                "model_tier": "cheap",
                "reason": "Simple availability check.",
            },
        },
    )
    db_session.commit()
    settings = build_settings(composio_api_key="composio-key")
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

    assert composio_client.list_tool_calls == []
    assert "web_search" in registry.names()
    assert all(not name.startswith("composio_") for name in registry.names())
    assert any(
        event.payload.get("message") == "external_tool_selection_skipped"
        and event.payload.get("reason") == "intent_no_external_tools"
        and event.payload.get("classification") == "task_request"
        for event in task_events(db_session, task)
    )


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
                "likely_tools": ["list_integrations", "tool_metadata_lookup"],
                "model_tier": "cheap",
                "reason": "Capability lookup.",
            },
        },
    )
    db_session.commit()
    settings = build_settings(composio_api_key="composio-key")
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
    result = registry.invoke("list_integrations", {})

    assert composio_client.list_tool_calls == []
    assert {tool["name"] for tool in result.output["native_tools"]} >= {
        "web_search",
        "slack_channel_history",
        "slack_file_read",
    }
    assert {
        connection["toolkit_slug"]
        for connection in result.output["connected_integrations"]
    } == {"firecrawl", "notion"}
    assert any(
        event.payload.get("message") == "external_tool_selection_skipped"
        and event.payload.get("reason") == "intent_no_external_tools"
        and event.payload.get("classification") == "task_request"
        for event in task_events(db_session, task)
    )


def test_worker_registry_skips_composio_catalog_for_channel_assessment_task(
    db_session: Session,
    tmp_path: Path,
) -> None:
    task = create_task(db_session, slack_channel_id="CObserve", slack_user_id="UInvite")
    task.input = "Run Kortny's channel onboarding assessment for this Slack channel."
    add_connection(
        db_session,
        task,
        connected_account_id="ca_firecrawl",
        scope_type="workspace",
        scope_id=None,
    )
    task_service = TaskService(db_session)
    task_service.append_event(
        task,
        TaskEventType.log,
        {
            "message": CHANNEL_ASSESSMENT_REQUESTED_MESSAGE,
            "source": "member_joined_channel",
            "channel_id": "CObserve",
            "membership_id": "membership-id",
        },
    )
    db_session.commit()
    settings = build_settings(composio_api_key="composio-key")
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

    assert composio_client.list_tool_calls == []
    assert "slack_channel_history" in registry.names()
    assert all(not name.startswith("composio_") for name in registry.names())
    assert any(
        event.payload.get("message") == "external_tool_selection_skipped"
        and event.payload.get("reason") == "system_observe_channel_assessment"
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
            select(TaskEvent).where(TaskEvent.task_id == task.id).order_by(TaskEvent.seq)
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


def build_settings(*, composio_api_key: str | None = None) -> Settings:
    assert TEST_POSTGRES_URL is not None
    return Settings(
        SLACK_BOT_TOKEN="xoxb-test-token",
        SLACK_APP_TOKEN="xapp-test-token",
        SLACK_SIGNING_SECRET="test-signing-secret",
        LLM_PROVIDER=SettingsLLMProvider.openrouter,
        LLM_API_KEY="test-llm-key",
        LLM_MODEL="openai/gpt-5.4-mini",
        COMPOSIO_API_KEY=composio_api_key,
        BRAVE_SEARCH_API_KEY="test-brave-key",
        POSTGRES_URL=TEST_POSTGRES_URL,
    )


class FakeComposioClient:
    def __init__(
        self,
        *,
        tools_by_toolkit: dict[str, tuple[ComposioTool, ...]] | None = None,
    ) -> None:
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


class StaticToolSelector:
    def __init__(self, result: ToolSelectionResult) -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []

    def select(
        self,
        *,
        task_id: uuid.UUID,
        task_input: str,
        native_cards: tuple[ToolCard, ...],
        external_cards: tuple[ToolCard, ...],
    ) -> ToolSelectionResult:
        self.calls.append(
            {
                "task_id": task_id,
                "task_input": task_input,
                "native_cards": native_cards,
                "external_cards": external_cards,
            }
        )
        return self.result
