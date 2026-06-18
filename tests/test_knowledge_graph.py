import os
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, delete, select
from sqlalchemy.orm import Session

from kortny.db.models import (
    Installation,
    KnowledgeGraphEdge,
    KnowledgeGraphEntity,
    KnowledgeGraphEvidence,
    ObservationEvent,
    SlackChannelMembership,
    SlackIdentity,
    Task,
)
from kortny.db.session import make_engine, make_session_factory, normalize_database_url
from kortny.knowledge_graph import (
    DestinationSurface,
    EvidenceInput,
    GraphService,
    KnowledgeGraphExtractionService,
    VisibilityScope,
    is_scope_compatible,
)
from kortny.knowledge_graph.projects import (
    ProjectGraphService,
    project_anchors_and_scopes,
)
from kortny.knowledge_graph.service import GraphContextPack
from kortny.tools import QueryWorkspaceGraphTool
from kortny.tools.workspace_graph import DeclareProjectTool

TEST_POSTGRES_URL = os.environ.get("KORTNY_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    TEST_POSTGRES_URL is None,
    reason="KORTNY_TEST_POSTGRES_URL is required for knowledge graph tests",
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


def test_scope_compatibility_matrix() -> None:
    assert is_scope_compatible(
        VisibilityScope.workspace(), DestinationSurface.channel("C_PUBLIC_A")
    )
    assert is_scope_compatible(
        VisibilityScope.channel("C_PUBLIC_A"),
        DestinationSurface.channel("C_PUBLIC_A"),
    )
    assert not is_scope_compatible(
        VisibilityScope.channel("C_PUBLIC_B"),
        DestinationSurface.channel("C_PUBLIC_A"),
    )
    assert not is_scope_compatible(
        VisibilityScope.private_channel("G_PRIVATE_A"),
        DestinationSurface.channel("C_PUBLIC_A"),
    )
    assert is_scope_compatible(
        VisibilityScope.private_channel("G_PRIVATE_A"),
        DestinationSurface.private_channel("G_PRIVATE_A"),
    )
    assert not is_scope_compatible(
        VisibilityScope.private_channel("G_PRIVATE_B"),
        DestinationSurface.private_channel("G_PRIVATE_A"),
    )
    assert is_scope_compatible(
        VisibilityScope.dm("D_UA"), DestinationSurface.dm("D_UA", user_id="U_A")
    )
    assert is_scope_compatible(
        VisibilityScope.user("U_A"), DestinationSurface.dm("D_UA", user_id="U_A")
    )
    assert not is_scope_compatible(
        VisibilityScope.user("U_B"), DestinationSurface.dm("D_UA", user_id="U_A")
    )


def test_retrieval_enforces_destination_scope_lifecycle_and_evidence(
    db_session: Session,
) -> None:
    installation = create_installation(db_session)
    graph = GraphService(db_session)

    add_fact(graph, installation, "workspace:firm", VisibilityScope.workspace())
    add_fact(graph, installation, "channel:public_a", VisibilityScope.channel("C_A"))
    add_fact(graph, installation, "channel:public_b", VisibilityScope.channel("C_B"))
    add_fact(
        graph,
        installation,
        "private:private_a",
        VisibilityScope.private_channel("G_A"),
    )
    add_fact(
        graph,
        installation,
        "private:private_b",
        VisibilityScope.private_channel("G_B"),
    )
    add_fact(graph, installation, "dm:u_a", VisibilityScope.dm("D_UA"))
    add_fact(graph, installation, "dm:u_b", VisibilityScope.dm("D_UB"))
    add_fact(graph, installation, "user:u_a", VisibilityScope.user("U_A"))
    add_fact(graph, installation, "user:u_b", VisibilityScope.user("U_B"))
    add_fact(
        graph,
        installation,
        "channel:candidate",
        VisibilityScope.channel("C_A"),
        lifecycle_state="candidate",
    )
    add_fact(
        graph,
        installation,
        "channel:stale",
        VisibilityScope.channel("C_A"),
        lifecycle_state="stale",
    )
    graph.create_entity(
        installation_id=installation.id,
        entity_type="firm_fact",
        canonical_key="channel:no_evidence",
        display_name="No evidence",
        visibility_scope=VisibilityScope.channel("C_A"),
        source_type="slack_authoritative",
        lifecycle_state="active",
    )
    db_session.commit()

    public_pack = graph.retrieve_current_context(
        installation_id=installation.id,
        destination=DestinationSurface.channel("C_A"),
        max_items=50,
    )
    assert entity_keys(public_pack) == {"workspace:firm", "channel:public_a"}
    assert (
        graph.scope_guard_violations(public_pack, DestinationSurface.channel("C_A"))
        == ()
    )
    assert all(entity.evidence_ids for entity in public_pack.entities)

    private_pack = graph.retrieve_current_context(
        installation_id=installation.id,
        destination=DestinationSurface.private_channel("G_A"),
        max_items=50,
    )
    assert entity_keys(private_pack) == {"workspace:firm", "private:private_a"}
    assert (
        graph.scope_guard_violations(
            private_pack, DestinationSurface.private_channel("G_A")
        )
        == ()
    )

    dm_pack = graph.retrieve_current_context(
        installation_id=installation.id,
        destination=DestinationSurface.dm("D_UA", user_id="U_A"),
        max_items=50,
    )
    assert entity_keys(dm_pack) == {"workspace:firm", "dm:u_a", "user:u_a"}
    assert (
        graph.scope_guard_violations(
            dm_pack, DestinationSurface.dm("D_UA", user_id="U_A")
        )
        == ()
    )


def test_anchor_traversal_returns_only_scope_safe_evidenced_edges(
    db_session: Session,
) -> None:
    installation = create_installation(db_session)
    graph = GraphService(db_session)

    channel = add_entity(
        graph,
        installation,
        entity_type="channel",
        canonical_key="slack_channel:C_A",
        display_name="#project-a",
        visibility_scope=VisibilityScope.channel("C_A"),
    )
    project = add_entity(
        graph,
        installation,
        entity_type="project",
        canonical_key="project:alpha",
        display_name="Project Alpha",
        visibility_scope=VisibilityScope.channel("C_A"),
    )
    private_project = add_entity(
        graph,
        installation,
        entity_type="project",
        canonical_key="project:private",
        display_name="Private Project",
        visibility_scope=VisibilityScope.private_channel("G_A"),
    )
    graph.create_edge(
        installation_id=installation.id,
        source_entity_id=channel.id,
        target_entity_id=project.id,
        relationship_type="maps_to",
        visibility_scope=VisibilityScope.channel("C_A"),
        source_type="user_explicit",
        lifecycle_state="confirmed",
        confidence_score=Decimal("0.900"),
        evidence=evidence("explicit channel mapping"),
    )
    graph.create_edge(
        installation_id=installation.id,
        source_entity_id=channel.id,
        target_entity_id=private_project.id,
        relationship_type="maps_to",
        visibility_scope=VisibilityScope.private_channel("G_A"),
        source_type="agent_inferred",
        lifecycle_state="confirmed",
        confidence_score=Decimal("0.900"),
        evidence=evidence("private channel mapping"),
    )
    graph.create_edge(
        installation_id=installation.id,
        source_entity_id=channel.id,
        target_entity_id=project.id,
        relationship_type="referenced_in",
        visibility_scope=VisibilityScope.channel("C_A"),
        source_type="agent_inferred",
        lifecycle_state="confirmed",
        confidence_score=Decimal("0.700"),
    )
    db_session.commit()

    pack = graph.retrieve_current_context(
        installation_id=installation.id,
        destination=DestinationSurface.channel("C_A"),
        anchor_keys=("slack_channel:C_A",),
        max_hops=1,
        max_items=20,
    )

    assert entity_keys(pack) == {"slack_channel:C_A", "project:alpha"}
    assert {edge.relationship_type for edge in pack.edges} == {"maps_to"}
    assert all(edge.evidence_ids for edge in pack.edges)
    assert graph.scope_guard_violations(pack, DestinationSurface.channel("C_A")) == ()


def test_mark_stale_current_uses_freshness_windows(db_session: Session) -> None:
    installation = create_installation(db_session)
    graph = GraphService(db_session)
    now = datetime(2026, 6, 2, 12, tzinfo=UTC)
    old_channel = graph.create_entity(
        installation_id=installation.id,
        entity_type="channel",
        canonical_key="slack_channel:C_OLD",
        display_name="#old",
        visibility_scope=VisibilityScope.channel("C_OLD"),
        source_type="slack_authoritative",
        lifecycle_state="active",
        freshness_window_days=7,
        confidence_score=Decimal("0.900"),
        evidence=evidence("old channel"),
    )
    fresh_channel = graph.create_entity(
        installation_id=installation.id,
        entity_type="channel",
        canonical_key="slack_channel:C_FRESH",
        display_name="#fresh",
        visibility_scope=VisibilityScope.channel("C_FRESH"),
        source_type="slack_authoritative",
        lifecycle_state="active",
        freshness_window_days=7,
        confidence_score=Decimal("0.900"),
        evidence=evidence("fresh channel"),
    )
    old_project = graph.create_entity(
        installation_id=installation.id,
        entity_type="project",
        canonical_key="project:old",
        display_name="Old Project",
        visibility_scope=VisibilityScope.channel("C_OLD"),
        source_type="user_explicit",
        lifecycle_state="confirmed",
        confidence_score=Decimal("0.900"),
        evidence=evidence("old project"),
    )
    old_edge = graph.create_edge(
        installation_id=installation.id,
        source_entity_id=old_channel.id,
        target_entity_id=old_project.id,
        relationship_type="maps_to",
        visibility_scope=VisibilityScope.channel("C_OLD"),
        source_type="user_explicit",
        lifecycle_state="confirmed",
        freshness_window_days=7,
        confidence_score=Decimal("0.900"),
        evidence=evidence("old mapping"),
    )
    old_channel.recorded_at = now - timedelta(days=8)
    old_edge.recorded_at = now - timedelta(days=8)
    fresh_channel.recorded_at = now - timedelta(days=2)
    db_session.commit()

    result = graph.mark_stale_current(installation_id=installation.id, now=now)
    db_session.commit()

    assert result.entity_ids == (old_channel.id,)
    assert result.edge_ids == (old_edge.id,)
    db_session.refresh(old_channel)
    db_session.refresh(old_edge)
    db_session.refresh(fresh_channel)
    assert old_channel.lifecycle_state == "stale"
    assert old_edge.lifecycle_state == "stale"
    assert fresh_channel.lifecycle_state == "active"

    pack = graph.retrieve_current_context(
        installation_id=installation.id,
        destination=DestinationSurface.channel("C_OLD"),
        max_items=20,
    )
    assert "slack_channel:C_OLD" not in entity_keys(pack)


def test_query_workspace_graph_tool_returns_scope_safe_provenance_and_evidence(
    db_session: Session,
) -> None:
    installation = create_installation(db_session)
    task = Task(
        installation_id=installation.id,
        slack_channel_id="C_A",
        slack_thread_ts="111.222",
        slack_user_id="U_A",
        input="what do you know about alpha?",
    )
    db_session.add(task)
    db_session.flush()
    graph = GraphService(db_session)
    channel = add_entity(
        graph,
        installation,
        entity_type="channel",
        canonical_key="slack_channel:C_A",
        display_name="#alpha",
        visibility_scope=VisibilityScope.channel("C_A"),
    )
    project = graph.create_entity(
        installation_id=installation.id,
        entity_type="project",
        canonical_key="project:alpha",
        display_name="Project Alpha",
        visibility_scope=VisibilityScope.channel("C_A"),
        source_type="onboarding_scan",
        lifecycle_state="active",
        confidence_score=Decimal("0.820"),
        confidence_reason="Extracted from bounded channel assessment.",
        attrs_json={"review_status": "auto"},
        evidence=evidence("Project Alpha is the main channel workflow."),
    )
    graph.create_entity(
        installation_id=installation.id,
        entity_type="project",
        canonical_key="project:private-alpha",
        display_name="Private Alpha",
        visibility_scope=VisibilityScope.private_channel("G_A"),
        source_type="onboarding_scan",
        lifecycle_state="active",
        confidence_score=Decimal("0.900"),
        evidence=evidence("Private Alpha belongs elsewhere."),
    )
    graph.create_edge(
        installation_id=installation.id,
        source_entity_id=channel.id,
        target_entity_id=project.id,
        relationship_type="relates_to",
        visibility_scope=VisibilityScope.channel("C_A"),
        source_type="onboarding_scan",
        lifecycle_state="active",
        confidence_score=Decimal("0.800"),
        evidence=evidence("The #alpha channel relates to Project Alpha."),
    )
    db_session.commit()

    result = QueryWorkspaceGraphTool(session=db_session, task=task).invoke(
        {"query": "Alpha", "include_evidence": True}
    )

    assert result.output["successful"] is True
    assert result.output["destination"]["surface_type"] == "channel"
    entity_keys_output = {row["canonical_key"] for row in result.output["entities"]}
    assert "project:alpha" in entity_keys_output
    assert "project:private-alpha" not in entity_keys_output
    project_output = next(
        row
        for row in result.output["entities"]
        if row["canonical_key"] == "project:alpha"
    )
    assert project_output["provenance"]["extraction_kind"] == "extracted"
    assert project_output["provenance"]["review_status"] == "auto"
    assert project_output["evidence"][0]["snippet"] == (
        "Project Alpha is the main channel workflow."
    )
    assert result.output["relationships"][0]["provenance"]["label"] == "Extracted"


def test_query_workspace_graph_tool_treats_private_c_channel_as_private_scope(
    db_session: Session,
) -> None:
    installation = create_installation(db_session)
    db_session.add(
        SlackChannelMembership(
            installation_id=installation.id,
            channel_id="C_PRIVATE",
            channel_name="private-project",
            channel_type="group",
            membership_status="active",
            discovered_via="app_mention",
            added_by_user_id="U_A",
            onboarding_status="posted",
            metadata_json={},
        )
    )
    db_session.add(
        SlackIdentity(
            installation_id=installation.id,
            kind="channel",
            slack_id="C_PRIVATE",
            display_name="#private-project",
            raw_name="private-project",
            is_private=True,
            raw_json={"id": "C_PRIVATE", "is_private": True},
        )
    )
    task = Task(
        installation_id=installation.id,
        slack_channel_id="C_PRIVATE",
        slack_thread_ts="111.222",
        slack_user_id="U_A",
        input="what do you know about this channel?",
    )
    db_session.add(task)
    db_session.flush()
    graph = GraphService(db_session)
    channel = add_entity(
        graph,
        installation,
        entity_type="channel",
        canonical_key="slack_channel:C_PRIVATE",
        display_name="#private-project",
        visibility_scope=VisibilityScope.private_channel("C_PRIVATE"),
    )
    profile = graph.create_entity(
        installation_id=installation.id,
        entity_type="firm_fact",
        canonical_key="channel_profile:C_PRIVATE",
        display_name="Private project channel profile",
        visibility_scope=VisibilityScope.private_channel("C_PRIVATE"),
        source_type="onboarding_scan",
        lifecycle_state="active",
        confidence_score=Decimal("0.700"),
        evidence=evidence("Private project channel is used for roadmap work."),
    )
    graph.create_edge(
        installation_id=installation.id,
        source_entity_id=channel.id,
        target_entity_id=profile.id,
        relationship_type="relates_to",
        visibility_scope=VisibilityScope.private_channel("C_PRIVATE"),
        source_type="onboarding_scan",
        lifecycle_state="active",
        confidence_score=Decimal("0.700"),
        evidence=evidence("#private-project relates to roadmap work."),
    )
    db_session.commit()

    result = QueryWorkspaceGraphTool(session=db_session, task=task).invoke(
        {"anchor_keys": ["slack_channel:C_PRIVATE"], "include_evidence": True}
    )

    assert result.output["successful"] is True
    assert result.output["destination"]["surface_type"] == "private_channel"
    assert result.output["entity_count"] >= 2
    entity_keys_output = {row["canonical_key"] for row in result.output["entities"]}
    assert "slack_channel:C_PRIVATE" in entity_keys_output
    assert "channel_profile:C_PRIVATE" in entity_keys_output
    assert result.output["relationships"][0]["visibility_scope"] == {
        "type": "private_channel",
        "id": "C_PRIVATE",
    }


def test_deterministic_projection_builds_scoped_slack_workspace_facts(
    db_session: Session,
) -> None:
    installation = create_installation(db_session)
    membership = SlackChannelMembership(
        installation_id=installation.id,
        channel_id="CGraphDet",
        channel_name="graph-deterministic",
        channel_type="channel",
        membership_status="active",
        discovered_via="message_observation",
        added_by_user_id="UAdded",
        onboarding_status="posted",
        onboarding_message_ts="1780000000.000000",
        metadata_json={},
    )
    db_session.add(membership)
    db_session.add(
        SlackIdentity(
            installation_id=installation.id,
            kind="user",
            slack_id="UDet",
            display_name="Aneesh Melkot",
            raw_name="aneesh",
            raw_json={"id": "UDet", "profile": {"real_name": "Aneesh Melkot"}},
            refreshed_at=datetime(2026, 6, 2, 13, tzinfo=UTC),
            last_seen_at=datetime(2026, 6, 2, 13, tzinfo=UTC),
        )
    )
    db_session.add(
        ObservationEvent(
            installation_id=installation.id,
            slack_team_id=installation.slack_team_id,
            channel_id="CGraphDet",
            user_id="UDet",
            event_type="file_share",
            slack_event_id="EvGraphDet",
            message_ts="1780000010.000000",
            thread_ts="1780000010.000000",
            file_id="FDet",
            raw_payload_checksum="graph-det-checksum",
            text_preview="Uploaded roadmap.csv for review.",
            visibility_metadata={
                "scope_type": "channel",
                "scope_id": "CGraphDet",
                "file_count": 1,
            },
            observed_at=datetime(2026, 6, 2, 13, 5, tzinfo=UTC),
        )
    )
    db_session.commit()

    result = KnowledgeGraphExtractionService(
        db_session
    ).project_deterministic_workspace_facts(installation_id=installation.id)
    db_session.commit()

    assert result.channel_count == 1
    assert result.person_count == 2
    assert result.artifact_count == 1
    assert result.membership_edge_count == 1
    assert result.artifact_edge_count == 1

    rows = {
        entity.canonical_key: entity
        for entity in db_session.scalars(
            select(KnowledgeGraphEntity).where(
                KnowledgeGraphEntity.installation_id == installation.id
            )
        )
    }
    assert rows["slack_channel:CGraphDet"].display_name == "#graph-deterministic"
    assert rows["slack_channel:CGraphDet"].source_type == "slack_authoritative"
    assert rows["slack_channel:CGraphDet"].lifecycle_state == "active"
    assert rows["slack_channel_user:CGraphDet:UDet"].display_name == "Aneesh Melkot"
    assert rows["slack_channel_user:CGraphDet:UDet"].visibility_scope_type == "channel"
    assert rows["slack_channel_user:CGraphDet:UDet"].visibility_scope_id == "CGraphDet"
    assert "slack_channel_file:CGraphDet:FDet" in rows

    relationships = {
        edge.relationship_type
        for edge in db_session.scalars(
            select(KnowledgeGraphEdge).where(
                KnowledgeGraphEdge.installation_id == installation.id
            )
        )
    }
    assert relationships == {"member_of", "referenced_in"}
    assert db_session.scalar(
        select(KnowledgeGraphEvidence.id).where(
            KnowledgeGraphEvidence.source_observation_id.is_not(None),
            KnowledgeGraphEvidence.source_slack_channel_id == "CGraphDet",
            KnowledgeGraphEvidence.source_slack_file_id == "FDet",
        )
    )

    pack = GraphService(db_session).retrieve_current_context(
        installation_id=installation.id,
        destination=DestinationSurface.channel("CGraphDet"),
        anchor_keys=("slack_channel:CGraphDet",),
        max_hops=1,
        max_items=20,
    )
    assert {
        "slack_channel:CGraphDet",
        "slack_channel_user:CGraphDet:UDet",
        "slack_channel_file:CGraphDet:FDet",
    }.issubset(entity_keys(pack))
    assert (
        GraphService(db_session).scope_guard_violations(
            pack,
            DestinationSurface.channel("CGraphDet"),
        )
        == ()
    )


def cleanup_database(session: Session) -> None:
    for model in (
        KnowledgeGraphEvidence,
        KnowledgeGraphEdge,
        KnowledgeGraphEntity,
        ObservationEvent,
        SlackChannelMembership,
        SlackIdentity,
        Task,
        Installation,
    ):
        session.execute(delete(model))


def create_installation(session: Session) -> Installation:
    installation = Installation(slack_team_id=f"T{uuid.uuid4().hex}")
    session.add(installation)
    session.flush()
    return installation


def add_fact(
    graph: GraphService,
    installation: Installation,
    canonical_key: str,
    visibility_scope: VisibilityScope,
    *,
    lifecycle_state: str = "active",
) -> KnowledgeGraphEntity:
    return add_entity(
        graph,
        installation,
        entity_type="firm_fact",
        canonical_key=canonical_key,
        display_name=canonical_key,
        visibility_scope=visibility_scope,
        lifecycle_state=lifecycle_state,
    )


def add_entity(
    graph: GraphService,
    installation: Installation,
    *,
    entity_type: str,
    canonical_key: str,
    display_name: str,
    visibility_scope: VisibilityScope,
    lifecycle_state: str = "active",
) -> KnowledgeGraphEntity:
    return graph.create_entity(
        installation_id=installation.id,
        entity_type=entity_type,
        canonical_key=canonical_key,
        display_name=display_name,
        visibility_scope=visibility_scope,
        source_type="slack_authoritative",
        lifecycle_state=lifecycle_state,
        confidence_score=Decimal("0.900"),
        evidence=evidence(display_name),
    )


def evidence(snippet: str) -> EvidenceInput:
    return EvidenceInput(
        source_type="slack_authoritative",
        extracted_by="test",
        source_slack_channel_id="C_EVIDENCE",
        raw_snippet=snippet,
        confidence_score=Decimal("0.900"),
    )


def entity_keys(pack: GraphContextPack) -> set[str]:
    return {entity.canonical_key for entity in pack.entities}


# --------------------------------------------------------------------------
# Project layer (HIG-276)
# --------------------------------------------------------------------------


def _seed_channel_with_fact(
    graph: GraphService,
    installation: Installation,
    *,
    channel_id: str,
    scope: VisibilityScope,
    fact_key: str,
) -> None:
    """A channel hub + one channel-scoped fact wired by a relates_to edge.

    Mirrors the real graph shape (channel hub --relates_to--> fact) so BFS from
    a project hub can reach the fact through the channel hub.
    """

    channel = add_entity(
        graph,
        installation,
        entity_type="channel",
        canonical_key=f"slack_channel:{channel_id}",
        display_name=f"#{channel_id}",
        visibility_scope=scope,
    )
    fact = add_entity(
        graph,
        installation,
        entity_type="firm_fact",
        canonical_key=fact_key,
        display_name=fact_key,
        visibility_scope=scope,
    )
    graph.create_edge(
        installation_id=installation.id,
        source_entity_id=channel.id,
        target_entity_id=fact.id,
        relationship_type="relates_to",
        visibility_scope=scope,
        source_type="slack_authoritative",
        lifecycle_state="confirmed",
        confidence_score=Decimal("0.900"),
        evidence=evidence(fact_key),
    )


def test_declare_project_links_known_channels_and_is_idempotent(
    db_session: Session,
) -> None:
    installation = create_installation(db_session)
    graph = GraphService(db_session)
    add_entity(
        graph,
        installation,
        entity_type="channel",
        canonical_key="slack_channel:C_ENG",
        display_name="#apollo-eng",
        visibility_scope=VisibilityScope.channel("C_ENG"),
    )
    db_session.commit()

    service = ProjectGraphService(db_session)
    result = service.declare_project(
        installation_id=installation.id,
        name="Apollo",
        channel_ids=["C_ENG", "C_UNKNOWN"],
    )
    db_session.commit()

    assert result.project.canonical_key == "project:apollo"
    assert result.linked_channel_ids == ("C_ENG",)
    assert result.skipped_channel_ids == ("C_UNKNOWN",)

    # Re-declaring is idempotent: same hub, no duplicate edges.
    again = service.declare_project(
        installation_id=installation.id,
        name="Apollo",
        channel_ids=["C_ENG"],
    )
    db_session.commit()
    assert again.project.id == result.project.id
    projects = service.projects_for_channel(
        installation_id=installation.id, channel_id="C_ENG"
    )
    assert [p.canonical_key for p in projects] == ["project:apollo"]


def test_public_member_scopes_exclude_private_channels(db_session: Session) -> None:
    installation = create_installation(db_session)
    graph = GraphService(db_session)
    for channel_id, scope in (
        ("C_ENG", VisibilityScope.channel("C_ENG")),
        ("C_DESIGN", VisibilityScope.channel("C_DESIGN")),
        ("G_SECRET", VisibilityScope.private_channel("G_SECRET")),
    ):
        add_entity(
            graph,
            installation,
            entity_type="channel",
            canonical_key=f"slack_channel:{channel_id}",
            display_name=f"#{channel_id}",
            visibility_scope=scope,
        )
    db_session.commit()
    service = ProjectGraphService(db_session)
    project = service.declare_project(
        installation_id=installation.id,
        name="Apollo",
        channel_ids=["C_ENG", "C_DESIGN", "G_SECRET"],
    ).project
    db_session.commit()

    scopes = service.public_member_channel_scopes(
        installation_id=installation.id, project_ids=[project.id]
    )
    scope_ids = {scope.scope_id for scope in scopes}
    assert scope_ids == {"C_ENG", "C_DESIGN"}  # private G_SECRET excluded
    assert all(scope.scope_type == "channel" for scope in scopes)


def test_project_retrieval_synthesizes_public_channels_and_blocks_private(
    db_session: Session,
) -> None:
    installation = create_installation(db_session)
    graph = GraphService(db_session)
    _seed_channel_with_fact(
        graph,
        installation,
        channel_id="C_ENG",
        scope=VisibilityScope.channel("C_ENG"),
        fact_key="apollo:eng_status",
    )
    _seed_channel_with_fact(
        graph,
        installation,
        channel_id="C_DESIGN",
        scope=VisibilityScope.channel("C_DESIGN"),
        fact_key="apollo:design_status",
    )
    _seed_channel_with_fact(
        graph,
        installation,
        channel_id="G_SECRET",
        scope=VisibilityScope.private_channel("G_SECRET"),
        fact_key="apollo:secret_status",
    )
    service = ProjectGraphService(db_session)
    service.declare_project(
        installation_id=installation.id,
        name="Apollo",
        channel_ids=["C_ENG", "C_DESIGN", "G_SECRET"],
        evidence=evidence("declared Project Apollo"),
    )
    db_session.commit()

    anchor_keys, additional_scopes = project_anchors_and_scopes(
        db_session, installation_id=installation.id, channel_id="C_ENG"
    )
    assert "project:apollo" in anchor_keys
    assert {s.scope_id for s in additional_scopes} == {"C_ENG", "C_DESIGN"}

    destination = DestinationSurface.channel("C_ENG")
    pack = graph.retrieve_current_context(
        installation_id=installation.id,
        destination=destination,
        anchor_keys=("slack_channel:C_ENG", *anchor_keys),
        max_hops=2,
        max_items=50,
        additional_scopes=additional_scopes,
    )
    keys = entity_keys(pack)
    # Magic moment: a sibling PUBLIC channel's fact is synthesized in.
    assert "apollo:eng_status" in keys
    assert "apollo:design_status" in keys
    # Barrier: the PRIVATE channel's fact must never surface in a public reply.
    assert "apollo:secret_status" not in keys
    # Widening to public siblings is not a leak.
    assert graph.scope_guard_violations(pack, destination, additional_scopes) == ()

    # Without the project widening, the sibling fact is unreachable (proves the
    # authorization change is what unlocks cross-channel synthesis).
    plain = graph.retrieve_current_context(
        installation_id=installation.id,
        destination=destination,
        anchor_keys=("slack_channel:C_ENG", *anchor_keys),
        max_hops=2,
        max_items=50,
    )
    assert "apollo:design_status" not in entity_keys(plain)


def test_declare_project_tool_declares_and_reports(db_session: Session) -> None:
    installation = create_installation(db_session)
    graph = GraphService(db_session)
    add_entity(
        graph,
        installation,
        entity_type="channel",
        canonical_key="slack_channel:C_ENG",
        display_name="#apollo-eng",
        visibility_scope=VisibilityScope.channel("C_ENG"),
    )
    task = Task(
        installation_id=installation.id,
        slack_channel_id="C_ENG",
        slack_thread_ts="1.2",
        slack_user_id="U_A",
        input="apollo-eng and apollo-design are Project Apollo",
    )
    db_session.add(task)
    db_session.flush()
    db_session.commit()

    result = DeclareProjectTool(session=db_session, task=task).invoke(
        {"name": "Apollo", "channel_ids": ["C_ENG", "C_MISSING"]}
    )
    assert result.output["successful"] is True
    assert result.output["canonical_key"] == "project:apollo"
    assert result.output["linked_channel_ids"] == ["C_ENG"]
    assert result.output["skipped_channel_ids"] == ["C_MISSING"]
