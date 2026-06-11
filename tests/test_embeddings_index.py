"""EmbeddingIndex integration tests against real Postgres + pgvector."""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, delete, select
from sqlalchemy.orm import Session

from kortny.db.models import ToolEmbedding
from kortny.db.session import make_engine, make_session_factory, normalize_database_url
from kortny.embeddings import EmbeddingIndex
from tests.fake_embeddings import FakeEmbeddingBackend, RaisingEmbeddingBackend

TEST_POSTGRES_URL = os.environ.get("KORTNY_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    TEST_POSTGRES_URL is None,
    reason="KORTNY_TEST_POSTGRES_URL is required for embedding index tests",
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
        session.execute(delete(ToolEmbedding))
        session.commit()
        yield session
        session.rollback()
        session.execute(delete(ToolEmbedding))
        session.commit()


def test_ensure_skips_unchanged_and_reembeds_changed(db_session: Session) -> None:
    backend = FakeEmbeddingBackend()
    index = EmbeddingIndex(db_session, backend)
    items = [
        ("linear_card", "Linear. Manage issues and tickets."),
        ("firecrawl_card", "Firecrawl. Scrape and crawl websites."),
    ]

    index.ensure("tool_card", items)
    assert sorted(backend.passage_texts) == sorted(text for _, text in items)

    # Re-run with identical content: sha gate skips every row.
    index.ensure("tool_card", items)
    assert len(backend.passage_texts) == 2

    # Change one text: only that row is re-embedded and its sha updates.
    before = {
        row.ref_key: row.content_sha256
        for row in db_session.scalars(select(ToolEmbedding))
    }
    changed = [
        ("linear_card", "Linear. Manage issues, tickets, and sprint backlogs."),
        ("firecrawl_card", "Firecrawl. Scrape and crawl websites."),
    ]
    index.ensure("tool_card", changed)
    assert len(backend.passage_texts) == 3
    assert backend.passage_texts[-1] == changed[0][1]
    after = {
        row.ref_key: row.content_sha256
        for row in db_session.scalars(select(ToolEmbedding))
    }
    assert after["linear_card"] != before["linear_card"]
    assert after["firecrawl_card"] == before["firecrawl_card"]
    assert db_session.scalars(select(ToolEmbedding)).all()  # still two rows
    assert len(after) == 2


def test_rank_orders_by_similarity_within_range(db_session: Session) -> None:
    backend = FakeEmbeddingBackend()
    index = EmbeddingIndex(db_session, backend)
    index.ensure(
        "tool_card",
        [
            ("linear_card", "Linear. Manage issues and tickets in your tracker."),
            ("firecrawl_card", "Firecrawl. Scrape and crawl websites."),
            ("unrelated_card", "Frobnicate the quux."),
        ],
    )

    ranked = index.rank(
        "tool_card",
        "check our issue tracker for open bugs",
        ["linear_card", "firecrawl_card", "unrelated_card"],
        top_k=3,
    )

    assert ranked is not None
    assert ranked[0][0] == "linear_card"
    similarities = [similarity for _, similarity in ranked]
    assert similarities == sorted(similarities, reverse=True)
    for similarity in similarities:
        assert 0.0 <= similarity <= 1.0
    assert ranked[0][1] > ranked[2][1]


def test_rank_respects_top_k_and_candidate_filter(db_session: Session) -> None:
    backend = FakeEmbeddingBackend()
    index = EmbeddingIndex(db_session, backend)
    index.ensure(
        "tool_card",
        [
            ("linear_card", "Linear. Manage issues and tickets."),
            ("jira_card", "Jira. Track issues and bugs."),
            ("firecrawl_card", "Firecrawl. Scrape websites."),
        ],
    )

    ranked = index.rank(
        "tool_card",
        "open issues in the tracker",
        ["linear_card", "jira_card"],
        top_k=1,
    )

    assert ranked is not None
    assert len(ranked) == 1
    assert ranked[0][0] in {"linear_card", "jira_card"}

    assert index.rank("tool_card", "anything", [], top_k=5) == []


def test_mixed_dimension_models_coexist(db_session: Session) -> None:
    backend_8d = FakeEmbeddingBackend(model_name="fake-8d", dim=8)
    backend_4d = FakeEmbeddingBackend(model_name="fake-4d", dim=4)
    EmbeddingIndex(db_session, backend_8d).ensure(
        "tool_card", [("linear_card", "Linear. Manage issues.")]
    )
    EmbeddingIndex(db_session, backend_4d).ensure(
        "tool_card", [("linear_card", "Linear. Manage issues.")]
    )

    rows = db_session.scalars(select(ToolEmbedding)).all()
    assert {(row.model, row.dim) for row in rows} == {("fake-8d", 8), ("fake-4d", 4)}

    # Each backend ranks against its own model's rows only.
    ranked = EmbeddingIndex(db_session, backend_4d).rank(
        "tool_card", "issue tracker", ["linear_card"], top_k=5
    )
    assert ranked is not None
    assert [ref_key for ref_key, _ in ranked] == ["linear_card"]


def test_kinds_are_isolated(db_session: Session) -> None:
    backend = FakeEmbeddingBackend()
    index = EmbeddingIndex(db_session, backend)
    index.ensure("tool_card", [("shared_key", "Linear. Manage issues.")])
    index.ensure("skill", [("shared_key", "Weekly report. Build a weekly report.")])

    ranked = index.rank("skill", "issue tracker", ["shared_key"], top_k=5)
    assert ranked is not None
    assert len(ranked) == 1
    rows = db_session.scalars(select(ToolEmbedding)).all()
    assert {row.kind for row in rows} == {"tool_card", "skill"}


def test_backend_failure_is_isolated(db_session: Session) -> None:
    index = EmbeddingIndex(db_session, RaisingEmbeddingBackend())

    index.ensure("tool_card", [("linear_card", "Linear. Manage issues.")])  # no raise
    assert (index.rank("tool_card", "issue tracker", ["linear_card"], top_k=5)) is None

    # The shared session stays usable after the failure.
    assert db_session.scalars(select(ToolEmbedding)).all() == []


def test_delete_tombstones_only_matching_kind_and_refs(db_session: Session) -> None:
    backend = FakeEmbeddingBackend()
    index = EmbeddingIndex(db_session, backend)
    index.ensure(
        "tool_card",
        [
            ("keep_card", "Linear. Manage issues."),
            ("drop_card", "Firecrawl. Scrape websites."),
        ],
    )
    index.ensure("skill", [("drop_card", "A skill that also uses this ref key.")])

    deleted = index.delete("tool_card", ["drop_card", "missing_card"])

    assert deleted == 1
    rows = {
        (row.kind, row.ref_key) for row in db_session.scalars(select(ToolEmbedding))
    }
    assert rows == {("tool_card", "keep_card"), ("skill", "drop_card")}
    assert index.delete("tool_card", []) == 0


def test_fastembed_passages_use_bounded_batch_size() -> None:
    """A few hundred passages in one ONNX batch peaks at ~2.4GB of attention
    tensors and got the consolidator OOM-killed; the backend must always pass
    its bounded batch size through to fastembed."""

    from kortny.embeddings import backends as backends_module
    from kortny.embeddings.backends import (
        PASSAGE_EMBED_BATCH_SIZE,
        FastembedBackend,
    )

    seen_kwargs: list[dict[str, object]] = []

    class StubModel:
        def passage_embed(
            self, texts: list[str], **kwargs: object
        ) -> Iterator[list[float]]:
            seen_kwargs.append(dict(kwargs))
            yield from ([0.0, 1.0] for _ in texts)

    backends_module._FASTEMBED_MODELS["stub-model"] = StubModel()
    try:
        vectors = FastembedBackend("stub-model").embed_passages(["a", "b", "c"])
    finally:
        del backends_module._FASTEMBED_MODELS["stub-model"]

    assert len(vectors) == 3
    assert seen_kwargs == [{"batch_size": PASSAGE_EMBED_BATCH_SIZE}]
    assert PASSAGE_EMBED_BATCH_SIZE <= 32
