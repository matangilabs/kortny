"""HIG-222 full-catalog Composio sync against real Postgres + pgvector.

Verifies: full-catalog persistence + embeddings in chunks, sha-gated re-sync,
tombstoning a removed tool, disconnecting a toolkit, and 429 retry. Uses a fake
Composio client (no network) and the deterministic fake embedding backend.
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
from kortny.composio.client import (
    ComposioClient,
    ComposioRateLimitError,
    ComposioTool,
)
from kortny.composio.tool_cards import card_sha
from kortny.db.models import (
    ComposioConnection,
    ComposioToolCard,
    Installation,
    ToolEmbedding,
)
from kortny.db.session import make_engine, make_session_factory, normalize_database_url
from kortny.embeddings import EmbeddingIndex
from kortny.tools.composio_execute import composio_runtime_tool_name
from tests.fake_embeddings import FakeEmbeddingBackend

TEST_POSTGRES_URL = os.environ.get("KORTNY_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    TEST_POSTGRES_URL is None,
    reason="KORTNY_TEST_POSTGRES_URL is required for Composio catalog sync tests",
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
    for model in (ComposioToolCard, ComposioConnection, Installation):
        session.execute(delete(model))


def _installation(session: Session) -> uuid.UUID:
    installation = Installation(slack_team_id=f"T{uuid.uuid4().hex}")
    session.add(installation)
    session.flush()
    return installation.id


def _connect(
    session: Session,
    installation_id: uuid.UUID,
    *,
    toolkit_slug: str,
    status: str = "active",
) -> None:
    session.add(
        ComposioConnection(
            installation_id=installation_id,
            toolkit_slug=toolkit_slug,
            auth_config_id=f"ac_{toolkit_slug}",
            connected_account_id=f"ca_{toolkit_slug}",
            connection_request_id=f"ln_{toolkit_slug}",
            composio_user_id=f"slack:{installation_id}:U1",
            owner_slack_user_id="U1",
            visibility_scope_type="workspace",
            visibility_scope_id=None,
            status=status,
        )
    )
    session.flush()


def _tools(
    toolkit_slug: str, count: int, *, description_suffix: str = ""
) -> tuple[ComposioTool, ...]:
    return tuple(
        ComposioTool(
            slug=f"{toolkit_slug.upper()}_TOOL_{index}",
            name=f"{toolkit_slug} tool {index}",
            description=f"Read-only {toolkit_slug} lookup {index}{description_suffix}.",
            toolkit_slug=toolkit_slug,
            input_parameters={
                "type": "object",
                "properties": {"q": {"type": "string"}},
            },
            tags=("readOnlyHint",),
            version=None,
        )
        for index in range(count)
    )


class FakeComposioClient(ComposioClient):
    """Paginating fake whose pages mirror the real tools endpoint shape."""

    def __init__(
        self,
        *,
        tools_by_toolkit: dict[str, tuple[ComposioTool, ...]],
        page_429_once: bool = False,
    ) -> None:
        super().__init__(api_key="fake")
        self.tools_by_toolkit = tools_by_toolkit
        self.page_calls: list[dict[str, Any]] = []
        self._page_429_pending = page_429_once

    def list_tools_page(
        self,
        *,
        toolkit_slug: str,
        limit: int = 20,
        cursor: str | None = None,
    ) -> tuple[tuple[ComposioTool, ...], str | None]:
        if self._page_429_pending:
            self._page_429_pending = False
            raise ComposioRateLimitError("429 rate limited")
        self.page_calls.append(
            {"toolkit_slug": toolkit_slug, "limit": limit, "cursor": cursor}
        )
        tools = self.tools_by_toolkit.get(toolkit_slug, ())
        start = int(cursor) if cursor else 0
        page = tools[start : start + limit]
        next_cursor = str(start + limit) if start + limit < len(tools) else None
        return page, next_cursor


def _service(
    session: Session,
    client: ComposioClient,
    *,
    page_size: int = 10,
    sleep_calls: list[float] | None = None,
    backend: FakeEmbeddingBackend | None = None,
) -> ComposioCatalogSyncService:
    resolved = backend or FakeEmbeddingBackend()
    return ComposioCatalogSyncService(
        session,
        client=client,
        embedding_index=EmbeddingIndex(session, resolved),
        page_size=page_size,
        sleep=(sleep_calls.append if sleep_calls is not None else (lambda _s: None)),
    )


def test_connected_toolkits_includes_no_auth_connection(
    db_session: Session,
) -> None:
    # A no-auth connection (active, no connected_account_id) must be synced so
    # its tools enter the catalog and become retrievable. Observed bug:
    # hackernews connected no-auth -> find_tools returned [] because the sync
    # skipped it (filtered on connected_account_id IS NOT NULL).
    installation_id = _installation(db_session)
    _connect(db_session, installation_id, toolkit_slug="notion")
    db_session.add(
        ComposioConnection(
            installation_id=installation_id,
            toolkit_slug="hackernews",
            auth_config_id=None,
            connected_account_id=None,
            composio_user_id=f"slack:{installation_id}:U1",
            owner_slack_user_id="U1",
            visibility_scope_type="workspace",
            visibility_scope_id=None,
            status="active",
            no_auth=True,
        )
    )
    db_session.flush()
    client = FakeComposioClient(tools_by_toolkit={})
    slugs = _service(db_session, client).connected_toolkits(installation_id)
    assert "hackernews" in slugs
    assert "notion" in slugs


def test_sync_persists_full_catalog_and_embeddings_in_chunks(
    db_session: Session,
) -> None:
    installation_id = _installation(db_session)
    _connect(db_session, installation_id, toolkit_slug="notion")
    tools = _tools("notion", 30)
    client = FakeComposioClient(tools_by_toolkit={"notion": tools})

    result = _service(db_session, client, page_size=10).sync_toolkit(
        installation_id, "notion"
    )

    # Full catalog pulled paginated (30 tools over 3 pages of 10), not pruned.
    assert result.tool_count == 30
    assert result.upserted == 30
    assert result.embedded == 30
    assert [call["cursor"] for call in client.page_calls] == [None, "10", "20"]

    cards = db_session.scalars(
        select(ComposioToolCard).where(
            ComposioToolCard.installation_id == installation_id
        )
    ).all()
    assert len(cards) == 30
    embeddings = db_session.scalars(
        select(ToolEmbedding).where(ToolEmbedding.kind == "tool_card")
    ).all()
    assert len(embeddings) == 30
    # ref_key uses the same runtime tool name the provider/executor key on.
    expected_refs = {composio_runtime_tool_name("notion", tool.slug) for tool in tools}
    assert {row.ref_key for row in embeddings} == expected_refs


def test_resync_with_one_changed_description_reembeds_only_that_card(
    db_session: Session,
) -> None:
    installation_id = _installation(db_session)
    _connect(db_session, installation_id, toolkit_slug="notion")
    tools = _tools("notion", 5)
    client = FakeComposioClient(tools_by_toolkit={"notion": tools})
    backend = FakeEmbeddingBackend()
    service = _service(db_session, client, backend=backend)
    service.sync_toolkit(installation_id, "notion")

    embedded_before = len(backend.passage_texts)

    # Change exactly one tool's description; re-sync.
    changed = (
        tools[0],
        ComposioTool(
            slug=tools[1].slug,
            name=tools[1].name,
            description="A brand new description for tool 1.",
            toolkit_slug="notion",
            input_parameters=tools[1].input_parameters,
            tags=tools[1].tags,
            version=None,
        ),
    ) + tools[2:]
    client.tools_by_toolkit["notion"] = changed
    result = service.sync_toolkit(installation_id, "notion")

    assert result.upserted == 1
    assert result.embedded == 1
    embedded_after = len(backend.passage_texts)
    assert embedded_after == embedded_before + 1

    changed_ref = composio_runtime_tool_name("notion", tools[1].slug)
    row = db_session.scalar(
        select(ComposioToolCard).where(
            ComposioToolCard.installation_id == installation_id,
            ComposioToolCard.tool_slug == tools[1].slug,
        )
    )
    assert row is not None
    assert row.description == "A brand new description for tool 1."
    assert row.card_sha == card_sha(
        name=tools[1].name,
        description="A brand new description for tool 1.",
        side_effect="read",
    )
    assert (
        db_session.scalar(
            select(ToolEmbedding).where(ToolEmbedding.ref_key == changed_ref)
        )
        is not None
    )


def test_removed_tool_card_and_embedding_are_deleted(db_session: Session) -> None:
    installation_id = _installation(db_session)
    _connect(db_session, installation_id, toolkit_slug="notion")
    tools = _tools("notion", 4)
    client = FakeComposioClient(tools_by_toolkit={"notion": tools})
    service = _service(db_session, client)
    service.sync_toolkit(installation_id, "notion")

    removed = tools[0]
    removed_ref = composio_runtime_tool_name("notion", removed.slug)
    client.tools_by_toolkit["notion"] = tools[1:]
    result = service.sync_toolkit(installation_id, "notion")

    assert result.tombstoned == 1
    assert (
        db_session.scalar(
            select(ComposioToolCard).where(
                ComposioToolCard.installation_id == installation_id,
                ComposioToolCard.tool_slug == removed.slug,
            )
        )
        is None
    )
    assert (
        db_session.scalar(
            select(ToolEmbedding).where(ToolEmbedding.ref_key == removed_ref)
        )
        is None
    )
    # Survivors remain.
    assert (
        db_session.scalar(
            select(ToolEmbedding).where(
                ToolEmbedding.ref_key
                == composio_runtime_tool_name("notion", tools[1].slug)
            )
        )
        is not None
    )


def test_disconnecting_a_toolkit_removes_its_rows(db_session: Session) -> None:
    installation_id = _installation(db_session)
    _connect(db_session, installation_id, toolkit_slug="notion")
    _connect(db_session, installation_id, toolkit_slug="linear")
    client = FakeComposioClient(
        tools_by_toolkit={
            "notion": _tools("notion", 3),
            "linear": _tools("linear", 2),
        }
    )
    service = _service(db_session, client)
    service.sync_installation(installation_id)
    assert (
        db_session.scalar(
            select(ComposioToolCard)
            .where(ComposioToolCard.toolkit_slug == "linear")
            .limit(1)
        )
        is not None
    )

    # Disconnect linear; re-sync the installation -> linear rows tombstoned.
    db_session.execute(
        delete(ComposioConnection).where(ComposioConnection.toolkit_slug == "linear")
    )
    db_session.flush()
    result = service.sync_installation(installation_id)

    assert result.disconnected_tombstoned == 2
    assert (
        db_session.scalars(
            select(ComposioToolCard).where(ComposioToolCard.toolkit_slug == "linear")
        ).all()
        == []
    )
    linear_refs = [
        composio_runtime_tool_name("linear", f"LINEAR_TOOL_{index}")
        for index in range(2)
    ]
    assert (
        db_session.scalars(
            select(ToolEmbedding).where(ToolEmbedding.ref_key.in_(linear_refs))
        ).all()
        == []
    )
    # Notion untouched.
    assert (
        len(
            db_session.scalars(
                select(ComposioToolCard).where(
                    ComposioToolCard.toolkit_slug == "notion"
                )
            ).all()
        )
        == 3
    )


def test_sync_retries_on_rate_limit(db_session: Session) -> None:
    installation_id = _installation(db_session)
    _connect(db_session, installation_id, toolkit_slug="notion")
    client = FakeComposioClient(
        tools_by_toolkit={"notion": _tools("notion", 2)},
        page_429_once=True,
    )
    sleep_calls: list[float] = []
    service = _service(db_session, client, sleep_calls=sleep_calls)

    result = service.sync_toolkit(installation_id, "notion")

    assert result.tool_count == 2
    assert sleep_calls  # backed off at least once before the retry succeeded
