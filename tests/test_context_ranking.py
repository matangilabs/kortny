"""HIG-225: recency x relevance ranked retrieval in the context assembler.

With an embedding index, facts/episodes/graph entities rank by
similarity x recency decay and budgets drop the lowest-relevance rows.
Without one, behavior is exactly the legacy ordering and drop policy.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, delete
from sqlalchemy.orm import Session

from kortny.agent.context import (
    RELEVANCE_BUDGET_OMISSION_REASON,
    ContextAssembler,
    _render_episode_context,
    _render_known_facts,
)
from kortny.db.models import (
    Episode,
    Installation,
    KnowledgeGraphEdge,
    KnowledgeGraphEntity,
    KnowledgeGraphEvidence,
    Task,
    TaskEvent,
    ToolEmbedding,
    WorkspaceState,
)
from kortny.db.session import make_engine, make_session_factory, normalize_database_url
from kortny.embeddings import (
    EPISODE_EMBEDDING_KIND,
    FACT_EMBEDDING_KIND,
    KG_ENTITY_EMBEDDING_KIND,
    EmbeddingIndex,
    episode_embedding_text,
    fact_embedding_text,
    kg_entity_embedding_text,
)
from kortny.knowledge_graph import EvidenceInput, GraphService, VisibilityScope
from kortny.memory import EpisodeService, WorkspaceStateService
from kortny.tasks import TaskService
from tests.fake_embeddings import FakeEmbeddingBackend

TEST_POSTGRES_URL = os.environ.get("KORTNY_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    TEST_POSTGRES_URL is None,
    reason="KORTNY_TEST_POSTGRES_URL is required for context ranking tests",
)

NOW = datetime(2026, 6, 11, 3, 0, 0, tzinfo=UTC)
CHANNEL = "C_MAIN"
USER = "U_USER"


@pytest.fixture(scope="session")
def engine() -> Iterator[Engine]:
    assert TEST_POSTGRES_URL is not None

    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", normalize_database_url(TEST_POSTGRES_URL))
    command.upgrade(config, "heads")

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


def cleanup_database(session: Session) -> None:
    for model in (
        ToolEmbedding,
        KnowledgeGraphEvidence,
        KnowledgeGraphEdge,
        KnowledgeGraphEntity,
        Episode,
        WorkspaceState,
        TaskEvent,
        Task,
        Installation,
    ):
        session.execute(delete(model))


def create_installation(session: Session) -> Installation:
    installation = Installation(slack_team_id=f"T{uuid.uuid4().hex}")
    session.add(installation)
    session.flush()
    return installation


def create_task(
    session: Session,
    installation: Installation,
    *,
    input_text: str,
) -> Task:
    return TaskService(session).create_task(
        installation_id=installation.id,
        slack_event_id=f"Ev{uuid.uuid4().hex}",
        slack_channel_id=CHANNEL,
        slack_thread_ts=None,
        slack_message_ts=f"1780000000.{uuid.uuid4().hex[:6]}",
        slack_user_id=USER,
        input=input_text,
    )


def create_fact(
    session: Session,
    installation: Installation,
    *,
    key: str,
    value_text: str,
    created_at: datetime,
) -> WorkspaceState:
    fact = WorkspaceState(
        installation_id=installation.id,
        scope_type="workspace",
        scope_id=None,
        key=key,
        value_json={"text": value_text},
        value_text=value_text,
        status="active",
        source_kind="user_explicit",
        proposed_by=USER,
        confirmed_by_user_id=USER,
        confirmed_at=created_at,
    )
    session.add(fact)
    session.flush()
    fact.created_at = created_at
    fact.updated_at = NOW - timedelta(days=1)
    session.flush()
    return fact


def create_episode(
    session: Session,
    installation: Installation,
    *,
    summary: str,
    created_at: datetime,
) -> Episode:
    parent = create_task(session, installation, input_text=f"prior: {summary[:30]}")
    episode = Episode(
        installation_id=installation.id,
        task_id=parent.id,
        channel_id=CHANNEL,
        user_id=USER,
        thread_ts=None,
        summary=summary,
        tools_used=[],
        artifacts_created=[],
        source_refs=[],
        outcome="succeeded",
    )
    session.add(episode)
    session.flush()
    episode.created_at = created_at
    session.flush()
    return episode


def fake_index(session: Session) -> EmbeddingIndex:
    return EmbeddingIndex(session, FakeEmbeddingBackend())


def seed_three_facts(
    session: Session, installation: Installation
) -> tuple[WorkspaceState, WorkspaceState, WorkspaceState]:
    """Relevant fact is the OLDEST so legacy and ranked drops diverge."""

    relevant = create_fact(
        session,
        installation,
        key="research_preference",
        value_text="use web search for news and internet research",
        created_at=NOW - timedelta(days=30),
    )
    middle = create_fact(
        session,
        installation,
        key="meeting_cadence",
        value_text="team meeting schedule and reminder cadence",
        created_at=NOW - timedelta(days=20),
    )
    newest = create_fact(
        session,
        installation,
        key="report_format",
        value_text="store report output as pdf files",
        created_at=NOW - timedelta(days=10),
    )
    return relevant, middle, newest


def facts_budget_for_one_drop(session: Session, installation: Installation) -> int:
    facts = WorkspaceStateService(session).list(installation.id)
    assert len(facts) == 3
    return len(_render_known_facts(facts)) - 1


WEB_QUERY = "search the web for current news on the internet"


def test_facts_ranked_drop_keeps_relevant_and_records_relevance_omission(
    db_session: Session,
) -> None:
    installation = create_installation(db_session)
    relevant, middle, newest = seed_three_facts(db_session, installation)
    index = fake_index(db_session)
    index.ensure(
        FACT_EMBEDDING_KIND,
        [
            (str(fact.id), fact_embedding_text(fact))
            for fact in (relevant, middle, newest)
        ],
    )
    task = create_task(db_session, installation, input_text=WEB_QUERY)
    budget = facts_budget_for_one_drop(db_session, installation)

    package = ContextAssembler(
        session=db_session,
        known_facts_max_chars=budget,
        embedding_index=index,
    ).build_for_task(task)

    selected_keys = {fact.key for fact in package.selected_facts}
    # The relevant (and oldest) fact survives; a low-relevance fact dropped.
    assert "research_preference" in selected_keys
    assert len(selected_keys) == 2
    omissions = [
        omission for omission in package.omissions if omission.kind == "known_facts"
    ]
    assert omissions == [
        type(omissions[0])("known_facts", RELEVANCE_BUDGET_OMISSION_REASON, 1)
    ]


def test_facts_without_backend_keep_legacy_drop_oldest(db_session: Session) -> None:
    installation = create_installation(db_session)
    seed_three_facts(db_session, installation)
    task = create_task(db_session, installation, input_text=WEB_QUERY)
    budget = facts_budget_for_one_drop(db_session, installation)

    package = ContextAssembler(
        session=db_session,
        known_facts_max_chars=budget,
        embedding_index=None,
    ).build_for_task(task)

    selected_keys = {fact.key for fact in package.selected_facts}
    # Legacy behavior: the oldest fact drops even though it is the relevant one.
    assert "research_preference" not in selected_keys
    assert selected_keys == {"meeting_cadence", "report_format"}
    reasons = [
        omission.reason
        for omission in package.omissions
        if omission.kind == "known_facts"
    ]
    assert reasons == ["budget_exceeded_drop_oldest"]


def seed_three_episodes(
    session: Session, installation: Installation
) -> tuple[Episode, Episode, Episode]:
    """Relevant episode is the OLDEST in the same channel tier."""

    relevant = create_episode(
        session,
        installation,
        summary="Ran a web search for internet news about the research topic.",
        created_at=NOW - timedelta(days=9),
    )
    middle = create_episode(
        session,
        installation,
        summary="Scheduled the recurring team meeting reminder.",
        created_at=NOW - timedelta(days=6),
    )
    newest = create_episode(
        session,
        installation,
        summary="Generated the quarterly report pdf file upload.",
        created_at=NOW - timedelta(days=3),
    )
    return relevant, middle, newest


def episode_budget_for_one_drop(session: Session, task: Task) -> int:
    episodes = list(EpisodeService(session).relevant_for_task(task, limit=5))
    assert len(episodes) == 3
    return len(_render_episode_context(episodes)) - 1


def test_episodes_ranked_within_tier_drop_lowest_relevance(
    db_session: Session,
) -> None:
    installation = create_installation(db_session)
    relevant, middle, newest = seed_three_episodes(db_session, installation)
    index = fake_index(db_session)
    index.ensure(
        EPISODE_EMBEDDING_KIND,
        [
            (str(episode.id), episode_embedding_text(episode))
            for episode in (relevant, middle, newest)
        ],
    )
    task = create_task(db_session, installation, input_text=WEB_QUERY)
    budget = episode_budget_for_one_drop(db_session, task)

    package = ContextAssembler(
        session=db_session,
        episode_context_max_chars=budget,
        embedding_index=index,
    ).build_for_task(task)

    selected_ids = [episode.episode_id for episode in package.selected_episodes]
    # The relevant episode ranks first within its tier and survives the drop.
    assert selected_ids[0] == relevant.id
    assert len(selected_ids) == 2
    reasons = [
        omission.reason for omission in package.omissions if omission.kind == "episodes"
    ]
    assert reasons == [RELEVANCE_BUDGET_OMISSION_REASON]


def test_episodes_without_backend_keep_legacy_recency_order(
    db_session: Session,
) -> None:
    installation = create_installation(db_session)
    relevant, middle, newest = seed_three_episodes(db_session, installation)
    task = create_task(db_session, installation, input_text=WEB_QUERY)
    budget = episode_budget_for_one_drop(db_session, task)

    package = ContextAssembler(
        session=db_session,
        episode_context_max_chars=budget,
        embedding_index=None,
    ).build_for_task(task)

    selected_ids = [episode.episode_id for episode in package.selected_episodes]
    # Legacy: newest-first within the tier, the oldest (relevant) drops.
    assert selected_ids == [newest.id, middle.id]
    reasons = [
        omission.reason for omission in package.omissions if omission.kind == "episodes"
    ]
    assert reasons == ["budget_exceeded_drop_lowest_relevance"]


def seed_graph(
    session: Session, installation: Installation
) -> tuple[KnowledgeGraphEntity, KnowledgeGraphEntity, KnowledgeGraphEntity]:
    graph = GraphService(session)
    task = create_task(session, installation, input_text="seed graph")
    evidence = EvidenceInput(
        source_type="task_summary",
        extracted_by="test",
        source_task_id=task.id,
    )
    anchor = graph.create_entity(
        installation_id=installation.id,
        entity_type="channel",
        canonical_key=f"slack_channel:{CHANNEL}",
        visibility_scope=VisibilityScope.channel(CHANNEL),
        source_type="slack_authoritative",
        lifecycle_state="active",
        evidence=evidence,
    )
    web_entity = graph.create_entity(
        installation_id=installation.id,
        entity_type="project",
        canonical_key="news_research_project",
        visibility_scope=VisibilityScope.channel(CHANNEL),
        source_type="task_summary",
        attrs_json={"summary": "Tracks web search news and internet research."},
        lifecycle_state="active",
        evidence=evidence,
    )
    files_entity = graph.create_entity(
        installation_id=installation.id,
        entity_type="project",
        canonical_key="report_files_project",
        visibility_scope=VisibilityScope.channel(CHANNEL),
        source_type="task_summary",
        attrs_json={"summary": "Stores report pdf files and uploads."},
        lifecycle_state="active",
        evidence=evidence,
    )
    for target in (web_entity, files_entity):
        graph.create_edge(
            installation_id=installation.id,
            source_entity_id=anchor.id,
            target_entity_id=target.id,
            relationship_type="relates_to",
            visibility_scope=VisibilityScope.channel(CHANNEL),
            source_type="task_summary",
            lifecycle_state="active",
            evidence=evidence,
        )
    return anchor, web_entity, files_entity


def test_graph_context_ranks_relevant_entities_first(db_session: Session) -> None:
    installation = create_installation(db_session)
    anchor, web_entity, files_entity = seed_graph(db_session, installation)
    index = fake_index(db_session)
    index.ensure(
        KG_ENTITY_EMBEDDING_KIND,
        [
            (str(entity.id), kg_entity_embedding_text(entity))
            for entity in (anchor, web_entity, files_entity)
        ],
    )
    task = create_task(db_session, installation, input_text=WEB_QUERY)

    package = ContextAssembler(
        session=db_session,
        embedding_index=index,
    ).build_for_task(task)

    selected_ids = [entity.entity_id for entity in package.selected_graph_entities]
    assert web_entity.id in selected_ids
    assert files_entity.id in selected_ids
    assert selected_ids.index(web_entity.id) < selected_ids.index(files_entity.id)
    graph_message = next(
        message.content
        for message in package.messages
        if message.content and "<workspace_graph_context>" in message.content
    )
    assert graph_message.index(str(web_entity.id)) < graph_message.index(
        str(files_entity.id)
    )


def test_graph_context_excludes_temporally_invalidated_entities(
    db_session: Session,
) -> None:
    installation = create_installation(db_session)
    anchor, web_entity, files_entity = seed_graph(db_session, installation)
    GraphService(db_session).invalidate_entity(web_entity, now=NOW - timedelta(days=1))
    db_session.flush()
    task = create_task(db_session, installation, input_text=WEB_QUERY)

    package = ContextAssembler(
        session=db_session,
        embedding_index=None,
    ).build_for_task(task)

    selected_ids = [entity.entity_id for entity in package.selected_graph_entities]
    assert web_entity.id not in selected_ids
    assert files_entity.id in selected_ids
    # The invalidated row still exists for as-of history queries.
    db_session.refresh(web_entity)
    assert web_entity.invalid_at is not None
    assert web_entity.lifecycle_state == "contradicted"
