"""HIG-225 memory spine: consolidator passes, runs, and leader election."""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, delete, func, select
from sqlalchemy.orm import Session

from kortny.config import Settings
from kortny.consolidator import ConsolidationService
from kortny.consolidator.passes import (
    adjudicate_candidates,
    age_graph,
    backfill_embeddings,
    merge_duplicate_entities,
    project_confirmed_facts,
    run_hygiene,
)
from kortny.consolidator.runner import ConsolidatorRunner, ConsolidatorWorker
from kortny.db.models import (
    ConsolidationRun,
    Episode,
    Installation,
    KnowledgeGraphEdge,
    KnowledgeGraphEntity,
    KnowledgeGraphEvidence,
    LLMUsage,
    ObservationEvent,
    ObserveChannelProfile,
    ObservePolicy,
    SlackChannelMembership,
    Task,
    TaskEvent,
    ToolEmbedding,
    WorkspaceState,
)
from kortny.db.session import make_engine, make_session_factory, normalize_database_url
from kortny.embeddings import (
    KG_ENTITY_EMBEDDING_KIND,
    EmbeddingIndex,
)
from kortny.knowledge_graph import EvidenceInput, GraphService, VisibilityScope
from kortny.llm import ChatMessage, Completion, TokenUsage
from kortny.observe.assessment import (
    CHANNEL_ASSESSMENT_REQUESTED_MESSAGE,
    CHANNEL_ASSESSMENT_SUPPRESS_SLACK_POST_KEY,
)
from kortny.tasks import TaskService
from kortny.tools.types import JsonObject, JsonSchema
from tests.fake_embeddings import FakeEmbeddingBackend

TEST_POSTGRES_URL = os.environ.get("KORTNY_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    TEST_POSTGRES_URL is None,
    reason="KORTNY_TEST_POSTGRES_URL is required for consolidator tests",
)

NOW = datetime(2026, 6, 11, 3, 0, 0, tzinfo=UTC)


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
        ConsolidationRun,
        ToolEmbedding,
        KnowledgeGraphEvidence,
        KnowledgeGraphEdge,
        KnowledgeGraphEntity,
        Episode,
        WorkspaceState,
        ObservationEvent,
        ObserveChannelProfile,
        ObservePolicy,
        SlackChannelMembership,
        LLMUsage,
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
    channel_id: str = "C_MAIN",
    input_text: str = "do the thing",
) -> Task:
    return TaskService(session).create_task(
        installation_id=installation.id,
        slack_event_id=f"Ev{uuid.uuid4().hex}",
        slack_channel_id=channel_id,
        slack_thread_ts="1780000000.000100",
        slack_message_ts=f"1780000000.{uuid.uuid4().hex[:6]}",
        slack_user_id="U_USER",
        input=input_text,
    )


def create_episode(
    session: Session,
    installation: Installation,
    *,
    summary: str,
    channel_id: str = "C_MAIN",
    created_at: datetime | None = None,
) -> Episode:
    task = create_task(session, installation, channel_id=channel_id)
    episode = Episode(
        installation_id=installation.id,
        task_id=task.id,
        channel_id=channel_id,
        user_id="U_USER",
        thread_ts=None,
        summary=summary,
        tools_used=[],
        artifacts_created=[],
        source_refs=[],
        outcome="succeeded",
    )
    session.add(episode)
    session.flush()
    if created_at is not None:
        episode.created_at = created_at
        session.flush()
    return episode


class FakeConsolidatorLLMProvider:
    model = "openai/gpt-4o-mini"

    def __init__(self, completions: list[Completion]) -> None:
        self.completions = completions
        self.calls: list[tuple[tuple[ChatMessage, ...], JsonObject | None]] = []

    def complete(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[JsonSchema] = (),
        *,
        response_format: JsonObject | None = None,
        max_output_tokens: int | None = None,
    ) -> Completion:
        del tools
        self.calls.append((tuple(messages), response_format))
        if not self.completions:
            raise AssertionError("FakeConsolidatorLLMProvider got too many calls")
        return self.completions.pop(0)


class RaisingLLMProvider:
    model = "openai/gpt-4o-mini"

    def complete(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[JsonSchema] = (),
        *,
        response_format: JsonObject | None = None,
        max_output_tokens: int | None = None,
    ) -> Completion:
        raise RuntimeError("cheap tier exploded")


def make_completion(payload: dict) -> Completion:
    return Completion(
        content=json.dumps(payload),
        tool_calls=(),
        usage=TokenUsage(input_tokens=120, output_tokens=40),
        cost_usd=Decimal("0.000100"),
        model="openai/gpt-4o-mini",
    )


def make_service(
    session: Session,
    *,
    completions: list[Completion] | None = None,
    embedding_index: EmbeddingIndex | None = None,
    provider: object | None = None,
) -> ConsolidationService:
    llm_provider = provider
    if llm_provider is None and completions is not None:
        llm_provider = FakeConsolidatorLLMProvider(completions)
    return ConsolidationService(
        session,
        llm_provider=llm_provider,  # type: ignore[arg-type]
        provider_name="openrouter",
        embedding_index=embedding_index,
    )


def fake_index(session: Session) -> EmbeddingIndex:
    return EmbeddingIndex(session, FakeEmbeddingBackend())


# --- pass 1: promotion -------------------------------------------------------


def test_promotion_add_creates_active_entity_with_evidence(
    db_session: Session,
) -> None:
    installation = create_installation(db_session)
    episode = create_episode(
        db_session,
        installation,
        summary="The team tracks all bugs in the Linear issue tracker.",
    )
    index = fake_index(db_session)
    service = make_service(
        db_session,
        completions=[
            make_completion(
                {
                    "decisions": [
                        {
                            "episode_id": str(episode.id),
                            "action": "ADD",
                            "entity_type": "integration",
                            "canonical_key": "issue_tracker_linear",
                            "display_name": "Linear issue tracker",
                            "summary": "Bugs are tracked in Linear.",
                            "confidence": 0.8,
                            "reason": "Episode states the tracker explicitly.",
                        }
                    ]
                }
            )
        ],
        embedding_index=index,
    )

    outcome = service.run_once(installation_id=installation.id, now=NOW)

    assert outcome.status == "succeeded"
    assert outcome.counters["promoted"] == 1
    assert "pass_errors" not in outcome.counters
    entity = db_session.scalar(
        select(KnowledgeGraphEntity).where(
            KnowledgeGraphEntity.canonical_key == "issue_tracker_linear"
        )
    )
    assert entity is not None
    assert entity.lifecycle_state == "active"
    assert entity.source_type == "task_summary"
    assert entity.valid_at is not None
    assert entity.invalid_at is None
    evidence = db_session.scalars(
        select(KnowledgeGraphEvidence).where(
            KnowledgeGraphEvidence.target_id == entity.id
        )
    ).all()
    assert len(evidence) == 1
    assert evidence[0].source_episode_id == episode.id
    # Embed-on-write: the new entity is in the semantic index.
    embedded = db_session.scalar(
        select(ToolEmbedding).where(
            ToolEmbedding.kind == KG_ENTITY_EMBEDDING_KIND,
            ToolEmbedding.ref_key == str(entity.id),
        )
    )
    assert embedded is not None


def test_promotion_update_reinforces_existing_entity(db_session: Session) -> None:
    installation = create_installation(db_session)
    graph = GraphService(db_session)
    task = create_task(db_session, installation)
    entity = graph.create_entity(
        installation_id=installation.id,
        entity_type="project",
        canonical_key="project_apollo",
        visibility_scope=VisibilityScope.channel("C_MAIN"),
        source_type="task_summary",
        attrs_json={"summary": "Apollo launches in July."},
        lifecycle_state="active",
        evidence=EvidenceInput(
            source_type="task_summary",
            extracted_by="test",
            source_task_id=task.id,
        ),
    )
    episode = create_episode(
        db_session,
        installation,
        summary="Apollo launch confirmed for July 20.",
    )
    service = make_service(
        db_session,
        completions=[
            make_completion(
                {
                    "decisions": [
                        {
                            "episode_id": str(episode.id),
                            "action": "UPDATE",
                            "entity_id": str(entity.id),
                            "summary": "Apollo launches July 20.",
                            "confidence": 0.9,
                            "reason": "Episode refreshes the launch date.",
                        }
                    ]
                }
            )
        ],
    )

    outcome = service.run_once(installation_id=installation.id, now=NOW)

    assert outcome.counters["updated"] == 1
    db_session.refresh(entity)
    assert entity.attrs_json["summary"] == "Apollo launches July 20."
    assert entity.reinforcement_count == 1
    assert entity.last_reinforced_at is not None
    evidence_count = db_session.scalar(
        select(func.count())
        .select_from(KnowledgeGraphEvidence)
        .where(KnowledgeGraphEvidence.target_id == entity.id)
    )
    assert evidence_count == 2


def test_promotion_invalidate_keeps_history_and_creates_successor(
    db_session: Session,
) -> None:
    installation = create_installation(db_session)
    graph = GraphService(db_session)
    task = create_task(db_session, installation)
    entity = graph.create_entity(
        installation_id=installation.id,
        entity_type="firm_fact",
        canonical_key="deploy_day",
        visibility_scope=VisibilityScope.channel("C_MAIN"),
        source_type="task_summary",
        attrs_json={"summary": "Deploys happen on Fridays."},
        lifecycle_state="active",
        evidence=EvidenceInput(
            source_type="task_summary",
            extracted_by="test",
            source_task_id=task.id,
        ),
    )
    entity.valid_at = NOW - timedelta(days=10)
    db_session.flush()
    episode = create_episode(
        db_session,
        installation,
        summary="Deploys moved from Friday to Tuesday after the incident.",
    )
    service = make_service(
        db_session,
        completions=[
            make_completion(
                {
                    "decisions": [
                        {
                            "episode_id": str(episode.id),
                            "action": "INVALIDATE",
                            "entity_id": str(entity.id),
                            "replacement_summary": "Deploys happen on Tuesdays.",
                            "confidence": 0.85,
                            "reason": "The deploy day changed.",
                        }
                    ]
                }
            )
        ],
    )

    outcome = service.run_once(installation_id=installation.id, now=NOW)

    assert outcome.counters["invalidated"] == 1
    db_session.refresh(entity)
    # Temporal invalidation, never deletion: history stays queryable as-of.
    assert entity.invalid_at is not None
    assert entity.lifecycle_state == "contradicted"
    assert entity.is_current is False
    rows = db_session.scalars(
        select(KnowledgeGraphEntity).where(
            KnowledgeGraphEntity.canonical_key == "deploy_day"
        )
    ).all()
    assert len(rows) == 2
    successor = next(row for row in rows if row.id != entity.id)
    assert successor.is_current is True
    assert successor.invalid_at is None
    assert successor.valid_at is not None
    assert successor.attrs_json["summary"] == "Deploys happen on Tuesdays."
    # As-of query: the old row's validity interval covers the past.
    assert entity.valid_at is not None
    assert entity.valid_at < entity.invalid_at


def test_promotion_never_invalidates_user_confirmed(db_session: Session) -> None:
    installation = create_installation(db_session)
    graph = GraphService(db_session)
    task = create_task(db_session, installation)
    entity = graph.create_entity(
        installation_id=installation.id,
        entity_type="firm_fact",
        canonical_key="workspace_fact:workspace:workspace:office_city",
        visibility_scope=VisibilityScope.workspace(),
        source_type="user_confirmed",
        attrs_json={"summary": "The office is in Austin."},
        lifecycle_state="confirmed",
        evidence=EvidenceInput(
            source_type="workspace_state",
            extracted_by="test",
            source_task_id=task.id,
        ),
    )
    episode = create_episode(
        db_session,
        installation,
        summary="A doc said the office is in Dallas.",
    )
    service = make_service(
        db_session,
        completions=[
            make_completion(
                {
                    "decisions": [
                        {
                            "episode_id": str(episode.id),
                            "action": "INVALIDATE",
                            "entity_id": str(entity.id),
                            "replacement_summary": "The office is in Dallas.",
                            "confidence": 0.7,
                            "reason": "A task document disagrees.",
                        }
                    ]
                }
            )
        ],
    )

    outcome = service.run_once(installation_id=installation.id, now=NOW)

    assert outcome.counters["invalidated"] == 0
    conflicts = outcome.counters["conflicts"]
    assert isinstance(conflicts, list)
    assert len(conflicts) == 1
    assert conflicts[0]["entity_id"] == str(entity.id)
    db_session.refresh(entity)
    assert entity.invalid_at is None
    assert entity.lifecycle_state == "confirmed"
    assert entity.is_current is True


def test_promotion_noop_changes_nothing(db_session: Session) -> None:
    installation = create_installation(db_session)
    episode = create_episode(db_session, installation, summary="Said hello.")
    service = make_service(
        db_session,
        completions=[
            make_completion(
                {
                    "decisions": [
                        {
                            "episode_id": str(episode.id),
                            "action": "NOOP",
                            "confidence": 0.9,
                            "reason": "Routine greeting.",
                        }
                    ]
                }
            )
        ],
    )

    outcome = service.run_once(installation_id=installation.id, now=NOW)

    promotion = outcome.counters["promotion"]
    assert isinstance(promotion, dict)
    assert promotion["noop"] == 1
    assert (
        db_session.scalar(select(func.count()).select_from(KnowledgeGraphEntity)) == 0
    )


# --- pass 2: adjudication ----------------------------------------------------


def test_adjudication_activates_consensus_and_archives_singletons(
    db_session: Session,
) -> None:
    installation = create_installation(db_session)
    graph = GraphService(db_session)
    task = create_task(db_session, installation)
    evidence = EvidenceInput(
        source_type="task_summary",
        extracted_by="test",
        source_task_id=task.id,
    )

    consensus = graph.create_entity(
        installation_id=installation.id,
        entity_type="project",
        canonical_key="consensus_candidate",
        visibility_scope=VisibilityScope.workspace(),
        source_type="task_summary",
        lifecycle_state="candidate",
        evidence=evidence,
    )
    graph.add_evidence(
        installation_id=installation.id,
        target_kind="entity",
        target_id=consensus.id,
        evidence=evidence,
    )
    consensus.created_at = NOW - timedelta(days=4)
    consensus.valid_at = None

    lonely = graph.create_entity(
        installation_id=installation.id,
        entity_type="project",
        canonical_key="lonely_candidate",
        visibility_scope=VisibilityScope.workspace(),
        source_type="task_summary",
        lifecycle_state="candidate",
        evidence=evidence,
    )
    lonely.created_at = NOW - timedelta(days=8)

    young = graph.create_entity(
        installation_id=installation.id,
        entity_type="project",
        canonical_key="young_candidate",
        visibility_scope=VisibilityScope.workspace(),
        source_type="task_summary",
        lifecycle_state="candidate",
        evidence=evidence,
    )
    young.created_at = NOW - timedelta(days=1)
    db_session.flush()

    counters = adjudicate_candidates(
        db_session, installation_id=installation.id, now=NOW
    )

    assert counters.activated == 1
    assert counters.archived == 1
    db_session.refresh(consensus)
    db_session.refresh(lonely)
    db_session.refresh(young)
    assert consensus.lifecycle_state == "active"
    assert consensus.valid_at is not None
    assert lonely.lifecycle_state == "archived"
    assert lonely.system_expired_at is not None
    assert lonely.is_current is False
    assert young.lifecycle_state == "candidate"


# --- pass 3: duplicate merge ---------------------------------------------------


def test_merge_supersedes_duplicate_and_moves_evidence_and_edges(
    db_session: Session,
) -> None:
    installation = create_installation(db_session)
    index = fake_index(db_session)
    graph = GraphService(db_session, embedding_index=index)
    task = create_task(db_session, installation)
    evidence = EvidenceInput(
        source_type="task_summary",
        extracted_by="test",
        source_task_id=task.id,
    )

    older = graph.create_entity(
        installation_id=installation.id,
        entity_type="integration",
        canonical_key="issue_tracker",
        visibility_scope=VisibilityScope.channel("C_MAIN"),
        source_type="task_summary",
        attrs_json={"summary": "The team tracks issues and bugs in Jira tickets."},
        lifecycle_state="active",
        evidence=evidence,
    )
    older.created_at = NOW - timedelta(days=30)
    newer = graph.create_entity(
        installation_id=installation.id,
        entity_type="integration",
        canonical_key="bug_tracker",
        visibility_scope=VisibilityScope.channel("C_MAIN"),
        source_type="task_summary",
        attrs_json={"summary": "Bugs and tickets live in the Linear issue tracker."},
        lifecycle_state="active",
        evidence=evidence,
    )
    newer.created_at = NOW - timedelta(days=2)
    other = graph.create_entity(
        installation_id=installation.id,
        entity_type="person",
        canonical_key="slack_user:U_OWNER",
        visibility_scope=VisibilityScope.channel("C_MAIN"),
        source_type="task_summary",
        attrs_json={"summary": "Channel owner."},
        lifecycle_state="active",
        evidence=evidence,
    )
    edge = graph.create_edge(
        installation_id=installation.id,
        source_entity_id=newer.id,
        target_entity_id=other.id,
        relationship_type="relates_to",
        visibility_scope=VisibilityScope.channel("C_MAIN"),
        source_type="task_summary",
        lifecycle_state="active",
        evidence=evidence,
    )
    db_session.flush()

    provider = FakeConsolidatorLLMProvider(
        [
            make_completion(
                {
                    "merges": [
                        {
                            "keep_id": str(older.id),
                            "merge_id": str(newer.id),
                            "merge": True,
                        }
                    ]
                }
            )
        ]
    )
    service = make_service(db_session, provider=provider, embedding_index=index)
    llm = service._llm_service(
        task=task,
        task_service=TaskService(db_session),
        pass_errors={},
    )
    counters = merge_duplicate_entities(
        db_session,
        installation_id=installation.id,
        graph=graph,
        embedding_index=index,
        llm=llm,
        task=task,
        now=NOW,
    )

    assert counters.merged == 1
    db_session.refresh(newer)
    db_session.refresh(edge)
    assert newer.is_current is False
    assert newer.lifecycle_state == "superseded"
    # Evidence moved to the surviving older row.
    moved = db_session.scalar(
        select(func.count())
        .select_from(KnowledgeGraphEvidence)
        .where(
            KnowledgeGraphEvidence.target_kind == "entity",
            KnowledgeGraphEvidence.target_id == older.id,
        )
    )
    assert moved == 2
    # The edge now points at the survivor.
    assert edge.source_entity_id == older.id


# --- pass 4: aging -------------------------------------------------------------


def test_aging_marks_stale_and_archives_long_stale(db_session: Session) -> None:
    installation = create_installation(db_session)
    graph = GraphService(db_session)
    task = create_task(db_session, installation)
    evidence = EvidenceInput(
        source_type="task_summary",
        extracted_by="test",
        source_task_id=task.id,
    )

    fading = graph.create_entity(
        installation_id=installation.id,
        entity_type="project",
        canonical_key="fading_project",
        visibility_scope=VisibilityScope.workspace(),
        source_type="task_summary",
        lifecycle_state="active",
        evidence=evidence,
    )
    fading.created_at = NOW - timedelta(days=80)
    fading.recorded_at = NOW - timedelta(days=80)
    fading.last_reinforced_at = NOW - timedelta(days=60)

    fresh = graph.create_entity(
        installation_id=installation.id,
        entity_type="project",
        canonical_key="fresh_project",
        visibility_scope=VisibilityScope.workspace(),
        source_type="task_summary",
        lifecycle_state="active",
        evidence=evidence,
    )
    fresh.last_reinforced_at = NOW - timedelta(days=2)

    ancient = graph.create_entity(
        installation_id=installation.id,
        entity_type="project",
        canonical_key="ancient_project",
        visibility_scope=VisibilityScope.workspace(),
        source_type="task_summary",
        lifecycle_state="active",
        evidence=evidence,
    )
    ancient.lifecycle_state = "stale"
    ancient.updated_at = NOW - timedelta(days=100)
    db_session.flush()

    counters = age_graph(
        db_session,
        installation_id=installation.id,
        graph=graph,
        stale_days=45,
        now=NOW,
    )

    assert counters.staled_entities == 1
    assert counters.archived == 1
    db_session.refresh(fading)
    db_session.refresh(fresh)
    db_session.refresh(ancient)
    assert fading.lifecycle_state == "stale"
    assert fresh.lifecycle_state == "active"
    assert ancient.lifecycle_state == "archived"
    assert ancient.system_expired_at is not None


# --- pass 5: fact reconciliation -------------------------------------------------


def test_fact_projection_is_idempotent_and_refreshes_on_supersede(
    db_session: Session,
) -> None:
    installation = create_installation(db_session)
    graph = GraphService(db_session)
    task = create_task(db_session, installation)
    fact = WorkspaceState(
        installation_id=installation.id,
        scope_type="workspace",
        scope_id=None,
        key="office_city",
        value_json={"text": "Austin"},
        value_text="The office is in Austin",
        status="active",
        source_kind="user_explicit",
        source_task_id=task.id,
        proposed_by="U_USER",
        confirmed_by_user_id="U_USER",
        confirmed_at=NOW,
    )
    db_session.add(fact)
    db_session.flush()

    first = project_confirmed_facts(
        db_session, installation_id=installation.id, graph=graph, task=task, now=NOW
    )
    second = project_confirmed_facts(
        db_session, installation_id=installation.id, graph=graph, task=task, now=NOW
    )

    assert first.projected == 1
    assert second.projected == 0
    assert second.unchanged == 1
    entities = db_session.scalars(
        select(KnowledgeGraphEntity).where(
            KnowledgeGraphEntity.canonical_key
            == "workspace_fact:workspace:workspace:office_city"
        )
    ).all()
    assert len(entities) == 1
    entity = entities[0]
    assert entity.source_type == "user_confirmed"
    assert entity.lifecycle_state == "confirmed"
    assert entity.attrs_json["workspace_state_id"] == str(fact.id)

    # Supersede the fact with a new active row: the same entity refreshes.
    fact.status = "superseded"
    replacement = WorkspaceState(
        installation_id=installation.id,
        scope_type="workspace",
        scope_id=None,
        key="office_city",
        value_json={"text": "Dallas"},
        value_text="The office is in Dallas",
        status="active",
        source_kind="user_explicit",
        source_task_id=task.id,
        proposed_by="U_USER",
        confirmed_by_user_id="U_USER",
        confirmed_at=NOW,
    )
    db_session.add(replacement)
    db_session.flush()

    third = project_confirmed_facts(
        db_session, installation_id=installation.id, graph=graph, task=task, now=NOW
    )
    assert third.refreshed == 1
    db_session.refresh(entity)
    assert entity.attrs_json["workspace_state_id"] == str(replacement.id)
    assert "Dallas" in entity.attrs_json["summary"]


# --- pass 6: hygiene -------------------------------------------------------------


def test_hygiene_purges_observations_per_policy_and_expires_ttl_facts(
    db_session: Session,
) -> None:
    installation = create_installation(db_session)
    db_session.add(
        ObservePolicy(
            installation_id=installation.id,
            scope_type="channel",
            scope_id="C_SHORT",
            observation_status="passive",
            proactivity_status="off",
            retention_days=30,
        )
    )

    def add_event(channel_id: str, age_days: int) -> ObservationEvent:
        event = ObservationEvent(
            installation_id=installation.id,
            slack_team_id="T_TEAM",
            channel_id=channel_id,
            event_type="message",
            raw_payload_checksum=uuid.uuid4().hex,
            observed_at=NOW - timedelta(days=age_days),
        )
        db_session.add(event)
        return event

    purged_short = add_event("C_SHORT", 40)
    kept_short = add_event("C_SHORT", 10)
    purged_default = add_event("C_OTHER", 100)
    kept_default = add_event("C_OTHER", 40)

    task = create_task(db_session, installation)
    expired_fact = WorkspaceState(
        installation_id=installation.id,
        scope_type="workspace",
        scope_id=None,
        key="sprint_focus",
        value_json={"text": "old focus"},
        status="active",
        source_kind="user_explicit",
        source_task_id=task.id,
        proposed_by="U_USER",
        expires_at=NOW - timedelta(days=1),
    )
    durable_fact = WorkspaceState(
        installation_id=installation.id,
        scope_type="workspace",
        scope_id=None,
        key="office_city",
        value_json={"text": "Austin"},
        status="active",
        source_kind="user_explicit",
        source_task_id=task.id,
        proposed_by="U_USER",
    )
    db_session.add_all([expired_fact, durable_fact])
    db_session.flush()

    counters = run_hygiene(db_session, installation_id=installation.id, now=NOW)

    assert counters.purged_observations == 2
    assert counters.expired_facts == 1
    remaining_ids = set(
        db_session.scalars(
            select(ObservationEvent.id).where(
                ObservationEvent.installation_id == installation.id
            )
        )
    )
    assert remaining_ids == {kept_short.id, kept_default.id}
    assert purged_short.id not in remaining_ids
    assert purged_default.id not in remaining_ids
    db_session.refresh(expired_fact)
    db_session.refresh(durable_fact)
    assert expired_fact.status == "superseded"
    assert durable_fact.status == "active"


def test_hygiene_enqueues_profile_refresh_for_active_stale_profiles(
    db_session: Session,
) -> None:
    installation = create_installation(db_session)
    db_session.add(
        SlackChannelMembership(
            installation_id=installation.id,
            channel_id="C_REF",
            membership_status="active",
            discovered_via="member_joined_channel",
            added_by_user_id="U_ADMIN",
        )
    )
    profile = ObserveChannelProfile(
        installation_id=installation.id,
        channel_id="C_REF",
        profile_status="active",
        last_profiled_at=NOW - timedelta(days=8),
    )
    db_session.add(profile)
    db_session.add(
        ObservationEvent(
            installation_id=installation.id,
            slack_team_id="T_TEAM",
            channel_id="C_REF",
            event_type="message",
            raw_payload_checksum=uuid.uuid4().hex,
            observed_at=NOW - timedelta(days=2),
        )
    )
    # Stale profile with no new activity must not be refreshed.
    db_session.add(
        ObserveChannelProfile(
            installation_id=installation.id,
            channel_id="C_QUIET",
            profile_status="active",
            last_profiled_at=NOW - timedelta(days=20),
        )
    )
    db_session.flush()

    counters = run_hygiene(db_session, installation_id=installation.id, now=NOW)

    assert counters.profiles_refreshed == 1
    refresh_task = db_session.scalar(
        select(Task).where(Task.slack_channel_id == "C_REF")
    )
    assert refresh_task is not None
    assert "C_REF" in refresh_task.input
    request_event = db_session.scalar(
        select(TaskEvent).where(
            TaskEvent.task_id == refresh_task.id,
            TaskEvent.payload["message"].as_string()
            == CHANNEL_ASSESSMENT_REQUESTED_MESSAGE,
        )
    )
    assert request_event is not None
    assert request_event.payload[CHANNEL_ASSESSMENT_SUPPRESS_SLACK_POST_KEY] is True


# --- pass 7: backfill --------------------------------------------------------------


def test_backfill_embeds_unembedded_rows_idempotently(db_session: Session) -> None:
    installation = create_installation(db_session)
    graph = GraphService(db_session)  # no index: rows start unembedded
    task = create_task(db_session, installation)
    graph.create_entity(
        installation_id=installation.id,
        entity_type="project",
        canonical_key="backfill_project",
        visibility_scope=VisibilityScope.workspace(),
        source_type="task_summary",
        lifecycle_state="active",
        evidence=EvidenceInput(
            source_type="task_summary",
            extracted_by="test",
            source_task_id=task.id,
        ),
    )
    create_episode(db_session, installation, summary="Backfill episode summary.")
    db_session.add(
        WorkspaceState(
            installation_id=installation.id,
            scope_type="workspace",
            scope_id=None,
            key="backfill_fact",
            value_json={"text": "value"},
            status="active",
            source_kind="user_explicit",
            source_task_id=task.id,
            proposed_by="U_USER",
        )
    )
    db_session.flush()
    index = fake_index(db_session)

    first = backfill_embeddings(
        db_session, installation_id=installation.id, embedding_index=index
    )
    second = backfill_embeddings(
        db_session, installation_id=installation.id, embedding_index=index
    )

    assert first.embedded == 3
    assert second.embedded == 0
    kinds = set(db_session.scalars(select(ToolEmbedding.kind)))
    assert kinds == {"fact", "episode", "kg_entity"}


# --- run orchestration ----------------------------------------------------------


def test_run_records_counters_and_isolates_pass_failures(
    db_session: Session,
) -> None:
    installation = create_installation(db_session)
    create_episode(db_session, installation, summary="Anything at all.")
    service = make_service(db_session, provider=RaisingLLMProvider())

    outcome = service.run_once(installation_id=installation.id, now=NOW)

    # The promotion pass blew up but the run still succeeded with the error
    # noted per-pass in counters_json.
    assert outcome.status == "succeeded"
    pass_errors = outcome.counters["pass_errors"]
    assert isinstance(pass_errors, dict)
    assert "promotion" in pass_errors
    run = db_session.get(ConsolidationRun, outcome.run_id)
    assert run is not None
    assert run.status == "succeeded"
    assert run.finished_at is not None
    assert isinstance(run.counters_json.get("pass_errors"), dict)


def test_run_records_cost_from_llm_usage(db_session: Session) -> None:
    installation = create_installation(db_session)
    episode = create_episode(db_session, installation, summary="Costly episode.")
    service = make_service(
        db_session,
        completions=[
            make_completion(
                {
                    "decisions": [
                        {
                            "episode_id": str(episode.id),
                            "action": "NOOP",
                            "confidence": 0.9,
                            "reason": "Nothing durable.",
                        }
                    ]
                }
            )
        ],
    )

    outcome = service.run_once(installation_id=installation.id, now=NOW)

    run = db_session.get(ConsolidationRun, outcome.run_id)
    assert run is not None
    assert run.cost_usd == Decimal("0.000100")
    assert outcome.task_id is not None
    run_task = db_session.get(Task, outcome.task_id)
    assert run_task is not None
    assert run_task.identity_kind == "synthetic"


def test_consecutive_runs_only_consume_new_episodes(db_session: Session) -> None:
    installation = create_installation(db_session)
    old_episode = create_episode(
        db_session,
        installation,
        summary="Old episode.",
        created_at=NOW - timedelta(hours=2),
    )
    service = make_service(
        db_session,
        completions=[
            make_completion(
                {
                    "decisions": [
                        {
                            "episode_id": str(old_episode.id),
                            "action": "NOOP",
                            "confidence": 0.9,
                            "reason": "Nothing durable.",
                        }
                    ]
                }
            )
        ],
    )
    first = service.run_once(installation_id=installation.id, now=NOW)
    promotion = first.counters["promotion"]
    assert isinstance(promotion, dict)
    assert promotion["episodes_reviewed"] == 1

    # Second run with no new episodes makes no LLM call at all.
    second_service = make_service(db_session, completions=[])
    second = second_service.run_once(
        installation_id=installation.id, now=NOW + timedelta(hours=9)
    )
    promotion = second.counters["promotion"]
    assert isinstance(promotion, dict)
    assert promotion["episodes_reviewed"] == 0
    assert "pass_errors" not in second.counters


def test_backlog_larger_than_cap_drains_across_runs(db_session: Session) -> None:
    installation = create_installation(db_session)
    first_episode = create_episode(
        db_session,
        installation,
        summary="Backlog episode one.",
        created_at=NOW - timedelta(hours=3),
    )
    second_episode = create_episode(
        db_session,
        installation,
        summary="Backlog episode two.",
        created_at=NOW - timedelta(hours=2),
    )

    def noop_completion(episode_id: uuid.UUID) -> Completion:
        return make_completion(
            {
                "decisions": [
                    {
                        "episode_id": str(episode_id),
                        "action": "NOOP",
                        "confidence": 0.9,
                        "reason": "Nothing durable.",
                    }
                ]
            }
        )

    first_service = make_service(
        db_session, completions=[noop_completion(first_episode.id)]
    )
    first_service.promotion_episode_cap = 1
    first = first_service.run_once(installation_id=installation.id, now=NOW)
    promotion = first.counters["promotion"]
    assert isinstance(promotion, dict)
    assert promotion["episodes_reviewed"] == 1

    # Episode two predates the first run but is still picked up next run
    # because the window anchors on the last processed episode, not run time.
    second_service = make_service(
        db_session, completions=[noop_completion(second_episode.id)]
    )
    second_service.promotion_episode_cap = 1
    second = second_service.run_once(
        installation_id=installation.id, now=NOW + timedelta(hours=9)
    )
    promotion = second.counters["promotion"]
    assert isinstance(promotion, dict)
    assert promotion["episodes_reviewed"] == 1


# --- leader election --------------------------------------------------------------


def test_second_runner_skips_when_advisory_lock_held(engine: Engine) -> None:
    session_factory = make_session_factory(engine=engine)
    lock_key = 759340187
    with session_factory() as holder, session_factory() as contender:
        cleanup_database(holder)
        holder.commit()
        acquired = holder.scalar(select(func.pg_try_advisory_lock(lock_key)))
        assert acquired is True
        try:
            result = ConsolidatorRunner(
                contender,
                advisory_lock_key=lock_key,
            ).run_once(use_advisory_lock=True)
            assert result.status == "lock_skipped"
            assert result.leader_acquired is False
            assert result.outcomes == ()
        finally:
            holder.execute(select(func.pg_advisory_unlock(lock_key)))
            holder.commit()
            contender.rollback()
        cleanup_database(holder)
        holder.commit()


def test_fail_stale_runs_marks_interrupted(db_session: Session) -> None:
    installation = create_installation(db_session)
    stale = ConsolidationRun(
        installation_id=installation.id,
        started_at=NOW - timedelta(hours=5),
        status="running",
    )
    fresh = ConsolidationRun(
        installation_id=installation.id,
        started_at=NOW - timedelta(minutes=30),
        status="running",
    )
    db_session.add_all([stale, fresh])
    db_session.flush()
    service = make_service(db_session)

    recovered = service.fail_stale_runs(now=NOW)

    assert recovered == 1
    db_session.refresh(stale)
    db_session.refresh(fresh)
    assert stale.status == "failed"
    assert stale.error == "interrupted"
    assert stale.finished_at is not None
    # A recent running row is presumed live and untouched.
    assert fresh.status == "running"


def test_pass_failure_rolls_back_only_that_pass(db_session: Session) -> None:
    """A failed pass must not poison committed work from earlier passes.

    Promotion (with a raising LLM) fails first; adjudication afterwards
    still promotes a consensus candidate, and the run row keeps per-pass
    progress in counters_json — the crash-safety contract added after the
    first live run lost all pass work to a single-transaction OOM kill.
    """

    installation = create_installation(db_session)
    create_episode(db_session, installation, summary="Anything at all.")
    graph = GraphService(db_session)
    task = create_task(db_session, installation)
    evidence = EvidenceInput(
        source_type="task_summary",
        extracted_by="test",
        source_task_id=task.id,
    )
    entity = graph.create_entity(
        installation_id=installation.id,
        entity_type="project",
        canonical_key="crashsafe_consensus",
        visibility_scope=VisibilityScope.workspace(),
        source_type="task_summary",
        lifecycle_state="candidate",
        evidence=evidence,
    )
    graph.add_evidence(
        installation_id=installation.id,
        target_kind="entity",
        target_id=entity.id,
        evidence=evidence,
    )
    entity.created_at = NOW - timedelta(days=4)
    entity.valid_at = None
    service = make_service(db_session, provider=RaisingLLMProvider())

    outcome = service.run_once(installation_id=installation.id, now=NOW)

    assert outcome.status == "succeeded"
    pass_errors = outcome.counters["pass_errors"]
    assert isinstance(pass_errors, dict)
    assert "promotion" in pass_errors
    db_session.refresh(entity)
    assert entity.lifecycle_state == "active"
    run = db_session.get(ConsolidationRun, outcome.run_id)
    assert run is not None
    counters = run.counters_json
    assert "adjudication" in counters


def test_worker_tick_survives_per_pass_commits_and_releases_lock(
    engine: Engine,
) -> None:
    """Regression: the worker tick must tolerate the service's internal
    commits (the old begin() block raised InvalidRequestError on advisory
    unlock and crash-looped the live service) and must release the advisory
    lock on the same connection so the next tick can acquire it."""

    session_factory = make_session_factory(engine=engine)
    with session_factory() as setup:
        cleanup_database(setup)
        installation = create_installation(setup)
        create_episode(setup, installation, summary="Worker tick episode.")
        setup.commit()
        installation_id = installation.id

    settings = Settings.model_validate(
        {
            "SLACK_BOT_TOKEN": "xoxb-test",
            "SLACK_APP_TOKEN": "xapp-test",
            "SLACK_SIGNING_SECRET": "signing-secret",
            "LLM_PROVIDER": "openrouter",
            "LLM_API_KEY": "test-key",
            "LLM_MODEL": "openai/gpt-test",
            "COMPOSIO_API_KEY": "composio-key",
            "POSTGRES_URL": str(engine.url),
            "KORTNY_EMBEDDINGS_BACKEND": "disabled",
        }
    )
    worker = ConsolidatorWorker(
        session_factory=session_factory,
        settings=settings,
        use_advisory_lock=True,
        poll_interval_seconds=0.01,
    )
    first = worker.run_once(force=True)
    assert first.status == "processed"
    # Lock released: an immediate second forced tick acquires it again.
    second = worker.run_once(force=True)
    assert second.status == "processed"

    with session_factory() as check:
        runs = list(
            check.scalars(
                select(ConsolidationRun).where(
                    ConsolidationRun.installation_id == installation_id,
                    ConsolidationRun.status == "succeeded",
                )
            )
        )
        assert len(runs) == 2
        cleanup_database(check)
        check.commit()
