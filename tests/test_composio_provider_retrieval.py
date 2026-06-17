"""HIG-222 provider retrieval: synced-card semantic top-K replaces the search.

Verifies the provider over synced cards: paraphrase retrieval (zero shared
keywords), zero catalog-search HTTP on a task, explicit-name forced include,
top-K cap at 15, on-demand sync of a fresh connection, and the disabled-backend
fallback to the original Composio search path.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from typing import Any

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, delete, select
from sqlalchemy.orm import Session

from kortny.composio.catalog_sync import ComposioCatalogSyncService
from kortny.composio.client import ComposioClient, ComposioTool
from kortny.composio.provider import ComposioExternalToolProvider
from kortny.db.models import (
    ComposioConnection,
    ComposioToolCard,
    Installation,
    Task,
    ToolEmbedding,
)
from kortny.db.session import make_engine, make_session_factory, normalize_database_url
from kortny.embeddings import EmbeddingIndex
from kortny.tasks import TaskService
from tests.fake_embeddings import FakeEmbeddingBackend

TEST_POSTGRES_URL = os.environ.get("KORTNY_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    TEST_POSTGRES_URL is None,
    reason="KORTNY_TEST_POSTGRES_URL is required for Composio provider tests",
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
        _cleanup(session)
        session.commit()
        yield session
        session.rollback()
        _cleanup(session)
        session.commit()


def _cleanup(session: Session) -> None:
    session.execute(delete(ToolEmbedding))
    for model in (ComposioToolCard, ComposioConnection, Task, Installation):
        session.execute(delete(model))


def _task(session: Session, *, text: str) -> Task:
    installation = Installation(slack_team_id=f"T{uuid.uuid4().hex}")
    session.add(installation)
    session.flush()
    return TaskService(session).create_task(
        installation_id=installation.id,
        slack_event_id=f"Ev{uuid.uuid4().hex}",
        slack_channel_id="CRag",
        slack_thread_ts="1779660000.000001",
        slack_message_ts="1779660000.000001",
        slack_user_id="U1",
        input=text,
    )


def _connect(session: Session, task: Task, *, toolkit_slug: str) -> None:
    session.add(
        ComposioConnection(
            installation_id=task.installation_id,
            toolkit_slug=toolkit_slug,
            auth_config_id=f"ac_{toolkit_slug}",
            connected_account_id=f"ca_{toolkit_slug}",
            connection_request_id=f"ln_{toolkit_slug}",
            composio_user_id=f"slack:{task.installation_id}:U1",
            owner_slack_user_id="U1",
            visibility_scope_type="workspace",
            visibility_scope_id=None,
            status="active",
        )
    )
    session.flush()


class RecordingComposioClient(ComposioClient):
    """Records search (query) listing vs lazy schema (tool_slugs) fetches."""

    def __init__(
        self, *, tools_by_toolkit: dict[str, tuple[ComposioTool, ...]]
    ) -> None:
        super().__init__(api_key="fake")
        self.tools_by_toolkit = tools_by_toolkit
        self.search_calls: list[dict[str, Any]] = []
        self.schema_fetch_calls: list[dict[str, Any]] = []

    def list_tools(
        self,
        *,
        toolkit_slug: str | None = None,
        tool_slugs: tuple[str, ...] = (),
        query: str | None = None,
        limit: int = 20,
    ) -> tuple[ComposioTool, ...]:
        tools = self.tools_by_toolkit.get(toolkit_slug or "", ())
        if tool_slugs:
            # Lazy schema fetch for surviving tools — NOT a candidate search.
            self.schema_fetch_calls.append(
                {"toolkit_slug": toolkit_slug, "tool_slugs": tool_slugs}
            )
            allowed = set(tool_slugs)
            return tuple(tool for tool in tools if tool.slug in allowed)
        # Query/candidate listing — the hot-path HTTP we must avoid once synced.
        self.search_calls.append({"toolkit_slug": toolkit_slug, "query": query})
        return tools[:limit]

    def list_tools_page(
        self,
        *,
        toolkit_slug: str,
        limit: int = 20,
        cursor: str | None = None,
    ) -> tuple[tuple[ComposioTool, ...], str | None]:
        tools = self.tools_by_toolkit.get(toolkit_slug, ())
        return tools, None


def _index(session: Session) -> EmbeddingIndex:
    return EmbeddingIndex(session, FakeEmbeddingBackend())


def _sync(session: Session, task: Task, client: ComposioClient) -> None:
    ComposioCatalogSyncService(
        session,
        client=client,
        embedding_index=_index(session),
    ).sync_installation(task.installation_id)


# Issue-tracker tools: names/descriptions deliberately avoid the paraphrase
# wording so retrieval must rely on the fake embedding vocabulary (linear/jira/
# ticket all land on the _ISSUES axis), not lexical overlap.
def _issue_tools() -> tuple[ComposioTool, ...]:
    return (
        ComposioTool(
            slug="LINEAR_LIST_ISSUES",
            name="Linear issue tracker",
            description="List Linear tickets and sprint backlog items.",
            toolkit_slug="linear",
            input_parameters={
                "type": "object",
                "properties": {"team": {"type": "string"}},
            },
            tags=("readOnlyHint",),
            version=None,
        ),
        ComposioTool(
            slug="LINEAR_GET_ISSUE",
            name="Linear ticket lookup",
            description="Get a Linear bug or ticket by id.",
            toolkit_slug="linear",
            input_parameters={
                "type": "object",
                "properties": {"id": {"type": "string"}},
                "required": ["id"],
            },
            tags=("readOnlyHint",),
            version=None,
        ),
    )


def test_paraphrase_retrieval_ranks_right_card_with_zero_keyword_overlap(
    db_session: Session,
) -> None:
    # Query shares no token with the tool name/description, but the fake
    # embedding backend maps "tracker"/"issues" onto the same axis as
    # "linear"/"ticket", so semantic rank still surfaces the issue tools.
    task = _task(db_session, text="check our issue tracker for open bugs")
    _connect(db_session, task, toolkit_slug="linear")
    _connect(db_session, task, toolkit_slug="firecrawl")
    client = RecordingComposioClient(
        tools_by_toolkit={
            "linear": _issue_tools(),
            "firecrawl": (
                ComposioTool(
                    slug="FIRECRAWL_SCRAPE",
                    name="Scrape URL",
                    description="Scrape and crawl a website URL.",
                    toolkit_slug="firecrawl",
                    input_parameters={"type": "object", "properties": {}},
                    tags=("readOnlyHint",),
                    version=None,
                ),
            ),
        }
    )
    _sync(db_session, task, client)

    provider = ComposioExternalToolProvider(
        session=db_session,
        task=task,
        client=client,
        embedding_index=_index(db_session),
    )
    cards = provider.tool_cards()
    toolkits = [card.toolkit_slug for card in cards]

    # The issue-tracker (linear) cards outrank the scrape card for this query.
    assert toolkits[0] == "linear"
    assert "linear" in toolkits


def test_provider_with_synced_cards_makes_zero_search_http(
    db_session: Session,
) -> None:
    task = _task(db_session, text="look up open linear tickets")
    _connect(db_session, task, toolkit_slug="linear")
    client = RecordingComposioClient(tools_by_toolkit={"linear": _issue_tools()})
    _sync(db_session, task, client)
    client.search_calls.clear()
    client.schema_fetch_calls.clear()

    provider = ComposioExternalToolProvider(
        session=db_session,
        task=task,
        client=client,
        embedding_index=_index(db_session),
    )
    cards = provider.tool_cards()
    tools = provider.runtime_tools()

    assert cards  # candidates came from synced table
    assert client.search_calls == []  # NO hot-path candidate-listing HTTP
    # Lazy schema fetch only for survivors, keyed by explicit tool_slugs.
    assert client.schema_fetch_calls
    assert {tool.name for tool in tools} == {
        "composio_linear_list_issues",
        "composio_linear_get_issue",
    }


def test_provider_forced_includes_explicitly_named_toolkit(
    db_session: Session,
) -> None:
    # 16 unrelated alpha_vantage tools (all rank above linear for a market
    # query) would push linear out of the top-15, but naming "linear" forces it.
    task = _task(db_session, text="pull market prices and also check linear")
    _connect(db_session, task, toolkit_slug="alpha_vantage")
    _connect(db_session, task, toolkit_slug="linear")
    alpha = tuple(
        ComposioTool(
            slug=f"ALPHA_SEARCH_{index}",
            name=f"Search market data {index}",
            description=f"Read-only market and web search tool {index}.",
            toolkit_slug="alpha_vantage",
            input_parameters={"type": "object", "properties": {}},
            tags=("readOnlyHint",),
            version=None,
        )
        for index in range(16)
    )
    client = RecordingComposioClient(
        tools_by_toolkit={"alpha_vantage": alpha, "linear": _issue_tools()}
    )
    _sync(db_session, task, client)

    provider = ComposioExternalToolProvider(
        session=db_session,
        task=task,
        client=client,
        embedding_index=_index(db_session),
    )
    cards = provider.tool_cards()
    toolkits = {card.toolkit_slug for card in cards}

    assert "linear" in toolkits  # forced in despite the crowded alpha_vantage set


def test_provider_floors_intent_named_toolkit_not_in_query(
    db_session: Session,
) -> None:
    # HIG-274 / task c65e7b2f: "what's on my plate" never says "linear", so the
    # verbatim force does nothing and a finance-heavy catalog ranks linear out of
    # the top-15. The grounded intent (forced_toolkits=("linear",)) must keep it.
    task = _task(db_session, text="what's on my plate today?")
    _connect(db_session, task, toolkit_slug="alpha_vantage")
    _connect(db_session, task, toolkit_slug="linear")
    alpha = tuple(
        ComposioTool(
            slug=f"ALPHA_SEARCH_{index}",
            name=f"Search market data {index}",
            description=f"Read-only market and web search tool {index}.",
            toolkit_slug="alpha_vantage",
            input_parameters={"type": "object", "properties": {}},
            tags=("readOnlyHint",),
            version=None,
        )
        for index in range(16)
    )
    client = RecordingComposioClient(
        tools_by_toolkit={"alpha_vantage": alpha, "linear": _issue_tools()}
    )
    _sync(db_session, task, client)

    without_floor = ComposioExternalToolProvider(
        session=db_session,
        task=task,
        client=client,
        embedding_index=_index(db_session),
    )
    assert "linear" not in {card.toolkit_slug for card in without_floor.tool_cards()}

    with_floor = ComposioExternalToolProvider(
        session=db_session,
        task=task,
        client=client,
        embedding_index=_index(db_session),
        forced_toolkits=("linear",),
    )
    assert "linear" in {card.toolkit_slug for card in with_floor.tool_cards()}


def test_provider_caps_candidates_at_15(db_session: Session) -> None:
    task = _task(db_session, text="search market data")
    _connect(db_session, task, toolkit_slug="alpha_vantage")
    alpha = tuple(
        ComposioTool(
            slug=f"ALPHA_SEARCH_{index}",
            name=f"Search market data {index}",
            description=f"Read-only market and web search tool {index}.",
            toolkit_slug="alpha_vantage",
            input_parameters={"type": "object", "properties": {}},
            tags=("readOnlyHint",),
            version=None,
        )
        for index in range(40)
    )
    client = RecordingComposioClient(tools_by_toolkit={"alpha_vantage": alpha})
    _sync(db_session, task, client)

    provider = ComposioExternalToolProvider(
        session=db_session,
        task=task,
        client=client,
        embedding_index=_index(db_session),
    )
    cards = provider.tool_cards()

    assert len(cards) == 15


def test_provider_on_demand_syncs_a_fresh_connection(db_session: Session) -> None:
    # No prior sync: a connected toolkit with zero synced cards triggers an
    # inline (one-toolkit) sync so the task sees its tools immediately.
    task = _task(db_session, text="look up linear tickets")
    _connect(db_session, task, toolkit_slug="linear")
    client = RecordingComposioClient(tools_by_toolkit={"linear": _issue_tools()})

    provider = ComposioExternalToolProvider(
        session=db_session,
        task=task,
        client=client,
        embedding_index=_index(db_session),
    )
    cards = provider.tool_cards()

    assert {card.toolkit_slug for card in cards} == {"linear"}
    persisted = db_session.scalars(
        select(ComposioToolCard).where(
            ComposioToolCard.installation_id == task.installation_id
        )
    ).all()
    assert len(persisted) == 2


def test_disabled_embeddings_uses_original_search_path(db_session: Session) -> None:
    task = _task(db_session, text="use firecrawl to scrape a page")
    _connect(db_session, task, toolkit_slug="firecrawl")
    client = RecordingComposioClient(
        tools_by_toolkit={
            "firecrawl": (
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
            )
        }
    )

    # embedding_index=None -> degraded path: original per-task Composio search.
    provider = ComposioExternalToolProvider(
        session=db_session,
        task=task,
        client=client,
        embedding_index=None,
    )
    cards = provider.tool_cards()
    tools = provider.runtime_tools()

    assert client.search_calls  # the original query-based search path ran
    assert {card.registry_name for card in cards} == {"composio_firecrawl_scrape"}
    assert {tool.name for tool in tools} == {"composio_firecrawl_scrape"}
