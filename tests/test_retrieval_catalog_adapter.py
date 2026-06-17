"""DB test for the catalog retrieve_fn adapter (Linear HIG-269 increment 1).

Verifies the plumbing — synced rows -> cards -> embedding rank -> tool slugs —
against the FakeEmbeddingBackend (deterministic), not real retrieval quality
(which needs the live fastembed model and is measured offline in the container).
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, delete
from sqlalchemy.orm import Session

from kortny.db.models import ComposioConnection, ComposioToolCard, Installation, Task
from kortny.db.session import make_engine, make_session_factory, normalize_database_url
from kortny.embeddings import EmbeddingIndex
from kortny.evals.retrieval import RetrievalCase
from kortny.evals.retrieval.catalog_retriever import (
    build_catalog_retrieve_fn,
    connected_toolkit_slugs_for_installation,
)
from kortny.evals.retrieval.scoring import score_retrieval
from tests.fake_embeddings import FakeEmbeddingBackend

TEST_POSTGRES_URL = os.environ.get("KORTNY_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    TEST_POSTGRES_URL is None,
    reason="KORTNY_TEST_POSTGRES_URL is required for catalog adapter tests",
)


@pytest.fixture(scope="session")
def engine() -> Iterator[Engine]:
    assert TEST_POSTGRES_URL is not None
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", normalize_database_url(TEST_POSTGRES_URL))
    command.upgrade(config, "head")
    eng = make_engine(TEST_POSTGRES_URL)
    try:
        yield eng
    finally:
        eng.dispose()


@pytest.fixture
def db_session(engine: Engine) -> Iterator[Session]:
    with make_session_factory(engine=engine)() as session:
        _cleanup(session)
        session.commit()
        yield session
        session.rollback()
        _cleanup(session)
        session.commit()


def _cleanup(session: Session) -> None:
    for model in (ComposioToolCard, ComposioConnection, Task, Installation):
        session.execute(delete(model))


def _install(session: Session) -> Installation:
    inst = Installation(slack_team_id=f"T{uuid.uuid4().hex}")
    session.add(inst)
    session.flush()
    return inst


def _card(inst_id: uuid.UUID, toolkit: str, slug: str, desc: str) -> ComposioToolCard:
    return ComposioToolCard(
        installation_id=inst_id,
        toolkit_slug=toolkit,
        tool_slug=slug,
        name=slug.replace("_", " ").title(),
        description=desc,
        side_effect="read",
        card_sha=uuid.uuid4().hex,
    )


def _connect(session: Session, inst_id: uuid.UUID, toolkit: str) -> None:
    session.add(
        ComposioConnection(
            installation_id=inst_id,
            toolkit_slug=toolkit,
            auth_config_id=f"ac_{toolkit}",
            connected_account_id=f"ca_{toolkit}",
            connection_request_id=f"ln_{toolkit}",
            composio_user_id=f"slack:{inst_id}:U1",
            owner_slack_user_id="U1",
            visibility_scope_type="workspace",
            visibility_scope_id=None,
            status="active",
        )
    )


def test_connected_toolkit_slugs_lists_active(db_session: Session) -> None:
    inst = _install(db_session)
    _connect(db_session, inst.id, "linear")
    _connect(db_session, inst.id, "notion")
    db_session.commit()
    assert connected_toolkit_slugs_for_installation(db_session, inst.id) == (
        "linear",
        "notion",
    )


def test_retrieve_fn_maps_ranked_cards_to_tool_slugs(db_session: Session) -> None:
    inst = _install(db_session)
    db_session.add_all(
        [
            _card(inst.id, "linear", "LINEAR_LIST_ISSUES", "List Linear issues."),
            _card(inst.id, "firecrawl", "FIRECRAWL_SCRAPE", "Scrape a website URL."),
        ]
    )
    db_session.commit()

    retrieve_fn = build_catalog_retrieve_fn(
        db_session,
        toolkit_slugs=("linear", "firecrawl"),
        embedding_index=EmbeddingIndex(db_session, FakeEmbeddingBackend()),
    )
    ranked = retrieve_fn("scrape a web page")

    # Returns tool slugs (not registry names), only from the synced catalog.
    assert set(ranked) == {"LINEAR_LIST_ISSUES", "FIRECRAWL_SCRAPE"}

    # Feeds the scorer end to end.
    report = score_retrieval(
        (
            RetrievalCase(
                query="scrape a web page", expected_tool_slugs=("FIRECRAWL_SCRAPE",)
            ),
        ),
        retrieve_fn,
        ks=(1, 2),
    )
    assert report.case_count == 1
    assert report.scores[0].hit is True


def test_retrieve_fn_ranks_mcp_extra_cards_alongside_composio(
    db_session: Session,
) -> None:
    # HIG-269: MCP tools have no synced Composio catalog row; they enter the same
    # ranked index via extra_cards and are returned by their runtime name so the
    # caller's loader dispatches them.
    from kortny.tool_selection import ToolCard

    inst = _install(db_session)
    db_session.add(
        _card(inst.id, "linear", "LINEAR_LIST_ISSUES", "List Linear issues.")
    )
    db_session.commit()

    mcp_card = ToolCard(
        registry_name="mcp__docs__search_docs",
        provider="mcp",
        display_name="search_docs via docs (MCP)",
        description="Search the documentation for a topic or page.",
        capabilities=("external_tool", "mcp_integration"),
        side_effect="read",
        toolkit_slug="docs",
        tool_slugs=("search_docs",),
        tool_count=1,
        required_fields=("query",),
        visibility_scope_type="workspace",
        visibility_scope_id=None,
    )

    retrieve_fn = build_catalog_retrieve_fn(
        db_session,
        toolkit_slugs=("linear",),
        embedding_index=EmbeddingIndex(db_session, FakeEmbeddingBackend()),
        extra_cards=(mcp_card,),
    )
    ranked = retrieve_fn("search the docs")

    # Both providers are in the same ranked set; the MCP slug is its runtime name.
    assert set(ranked) == {"LINEAR_LIST_ISSUES", "mcp__docs__search_docs"}
