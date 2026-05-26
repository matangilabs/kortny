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
    ComposioToolExecution,
)
from kortny.config.settings import LLMProvider as SettingsLLMProvider
from kortny.config.settings import Settings
from kortny.db.models import ComposioConnection, Installation, Task, TaskEvent
from kortny.db.session import make_engine, make_session_factory, normalize_database_url
from kortny.tasks import TaskService
from kortny.tool_selection import ToolCard, ToolSelection, ToolSelectionResult
from kortny.tools import ToolResult
from kortny.tools.composio_execute import ComposioExecuteTool
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

    tool = ComposioExecuteTool(session=db_session, task=task, client=client)

    assert tool.has_available_connections is True
    result = tool.invoke(
        {
            "toolkit_slug": "firecrawl",
            "tool_slug": "FIRECRAWL_SCRAPE",
            "arguments": {"url": "https://example.com"},
        }
    )

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


def test_composio_execute_tool_rejects_unapproved_tool(
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
    )

    with pytest.raises(ValueError, match="not approved"):
        tool.invoke(
            {
                "toolkit_slug": "firecrawl",
                "tool_slug": "FIRECRAWL_MAP",
                "arguments": {"url": "https://example.com"},
            }
        )


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
        tool_selector=StaticToolSelector(
            ToolSelectionResult(
                selected_tools=(
                    ToolSelection(
                        registry_name="composio_execute",
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

    assert "composio_execute" in registry.names()
    assert "web_search" not in registry.names()
    event = next(
        event
        for event in task_events(db_session, task)
        if event.payload.get("message") == "tool_selection_completed"
    )
    assert event.payload["selected_tools"][0]["registry_name"] == "composio_execute"
    assert event.payload["suppressed_native_tools"] == ["web_search"]


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
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

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
