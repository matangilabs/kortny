"""HIG-219 Capability Fabric: capability card, skill ranking, planner, worker."""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Iterator, Sequence

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, delete, select
from sqlalchemy.orm import Session

from kortny.agent.capabilities import (
    CapabilityOverview,
    build_capability_overview,
    render_capability_overview,
)
from kortny.agent.context import (
    DEFAULT_SKILLS_CONTEXT_MAX_SKILLS,
    ContextAssembler,
    ContextBudget,
    ContextPackage,
    ContextSkill,
)
from kortny.agent.coordinator import DEFAULT_SYSTEM_PROMPT, AgentCoordinator
from kortny.agent.execution import ExecutionGuardrailLimits
from kortny.agent.planner import PLANNER_SYSTEM_PROMPT, ExecutionPlanner
from kortny.db.models import (
    Installation,
    McpServer,
    ProceduralSkill,
    ProceduralSkillVersion,
    SkillEnablement,
    Task,
    TaskEvent,
    TaskEventType,
    ToolEmbedding,
)
from kortny.db.session import make_engine, make_session_factory, normalize_database_url
from kortny.embeddings import EmbeddingIndex
from kortny.execution.sandbox import ToolSandboxPolicy
from kortny.llm import ChatMessage, Completion, TokenUsage
from kortny.tasks import TaskService
from kortny.tool_selection import ToolCard
from kortny.tools import ToolRegistry
from kortny.tools.catalog import ToolDescriptor
from kortny.tools.types import JsonObject, JsonSchema
from tests.fake_embeddings import FakeEmbeddingBackend

TEST_POSTGRES_URL = os.environ.get("KORTNY_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    TEST_POSTGRES_URL is None,
    reason="KORTNY_TEST_POSTGRES_URL is required for capability fabric tests",
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


def cleanup_database(session: Session) -> None:
    for model in (
        ToolEmbedding,
        SkillEnablement,
        ProceduralSkillVersion,
        ProceduralSkill,
        McpServer,
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
    installation: Installation | None = None,
    *,
    input_text: str = "summarize this thread",
) -> Task:
    installation = installation or create_installation(session)
    message_ts = f"{uuid.uuid4().int % 10**6}.{uuid.uuid4().int % 10**6}"
    return TaskService(session).create_task(
        installation_id=installation.id,
        slack_event_id=f"Ev{uuid.uuid4().hex}",
        slack_channel_id="C123",
        slack_thread_ts=message_ts,
        slack_message_ts=message_ts,
        slack_user_id="U123",
        input=input_text,
    )


def create_enabled_skill(
    session: Session,
    installation: Installation,
    *,
    slug: str,
    name: str,
    description: str,
) -> ProceduralSkill:
    skill = ProceduralSkill(
        slug=slug,
        owner_type="system",
        status="active",
        trust_level="trusted",
        visibility="catalog",
        provenance="kortny",
    )
    session.add(skill)
    session.flush()
    session.add(
        ProceduralSkillVersion(
            skill_id=skill.id,
            version="1.0.0",
            status="active",
            name=name,
            description=description,
            instructions_md="## Steps\n1. Do the thing.",
            content_sha256="0" * 64,
            created_by="test",
        )
    )
    session.add(
        SkillEnablement(
            installation_id=installation.id,
            skill_id=skill.id,
            scope_type="workspace",
            scope_id=None,
            added_by="dashboard:test",
        )
    )
    session.flush()
    return skill


def make_descriptor(
    name: str,
    *,
    category: str = "Research",
    enabled: bool = True,
    disabled_reason: str | None = None,
) -> ToolDescriptor:
    return ToolDescriptor(
        name=name,
        namespace=f"native.{category.casefold()}",
        integration="",
        category=category,
        display_name=name,
        description=f"{name} description.",
        parameters={"type": "object", "properties": {}},
        capabilities=("diagnostic",),
        side_effect="read",
        approval="none",
        timeout_seconds=60,
        required_env_vars=(),
        required_slack_scopes=(),
        plan_gates=(),
        result_budget="normal",
        notes=(),
        can_replace_native_tools=(),
        sandbox=ToolSandboxPolicy(),
        enabled=enabled,
        disabled_reason=disabled_reason,
        required_args=(),
        optional_args=(),
    )


def make_overview() -> CapabilityOverview:
    return CapabilityOverview(
        native_categories=("Research", "Documents"),
        disabled_native=(("web_search", "Missing env var BRAVE_SEARCH_API_KEY"),),
        composio_toolkits=("github", "linear"),
        mcp_servers=(("context7", "enabled"), ("legacy", "disabled")),
    )


def make_context_package(
    *,
    skill_similarities: tuple[tuple[str, float], ...] = (),
    selected_skills: tuple[ContextSkill, ...] = (),
) -> ContextPackage:
    return ContextPackage(
        messages=(ChatMessage(role="user", content="do the thing"),),
        selected_facts=(),
        selected_prior_tasks=(),
        selected_episodes=(),
        selected_artifacts=(),
        selected_graph_entities=(),
        selected_graph_edges=(),
        acknowledgement=None,
        budget=ContextBudget(
            system_prompt_chars=0,
            known_facts_max_chars=0,
            known_facts_chars=0,
            thread_context_max_chars=1,
            prior_context_chars=0,
            thread_context_recent_tasks=1,
            thread_transcript_limit=0,
            episode_context_max_chars=0,
            episode_context_chars=0,
            episode_context_limit=0,
            graph_context_max_chars=0,
            graph_context_chars=0,
            graph_context_max_items=0,
            graph_context_max_hops=0,
        ),
        omissions=(),
        skill_similarities=skill_similarities,
        selected_skills=selected_skills,
    )


def make_context_skill(slug: str, description: str) -> ContextSkill:
    return ContextSkill(
        skill_id=uuid.uuid4(),
        version_id=uuid.uuid4(),
        slug=slug,
        name=slug.title(),
        description=description,
        trust_level="trusted",
        scope_type="workspace",
    )


class FakeLLM:
    """Coordinator/planner LLM stub that records prompts."""

    def __init__(self, content: str = "All done.") -> None:
        self.content = content
        self.calls: list[tuple[ChatMessage, ...]] = []

    def complete(
        self,
        *,
        task_id: uuid.UUID,
        messages: Sequence[ChatMessage],
        tools: Sequence[JsonSchema] = (),
        response_format: JsonObject | None = None,
        prompt_name: str | None = None,
        prompt_source: str = "code",
    ) -> Completion:
        del task_id, tools, response_format, prompt_name, prompt_source
        self.calls.append(tuple(messages))
        return Completion(
            content=self.content,
            tool_calls=(),
            usage=TokenUsage(input_tokens=5, output_tokens=5),
            model="test-model",
        )


# ---------------------------------------------------------------------------
# Capability card
# ---------------------------------------------------------------------------


def test_build_capability_overview_from_sources(db_session: Session) -> None:
    descriptors = (
        make_descriptor("web_search", enabled=False, disabled_reason="Missing key"),
        make_descriptor("pdf_generator", category="Documents"),
        make_descriptor("echo", category="Documents"),
    )
    cards = (
        ToolCard(
            registry_name="composio_linear_search",
            provider="composio",
            display_name="Linear search",
            description="Search Linear issues.",
            capabilities=("external_tool",),
            side_effect="read",
            toolkit_slug="linear",
        ),
        ToolCard(
            registry_name="composio_linear_create",
            provider="composio",
            display_name="Linear create",
            description="Create Linear issues.",
            capabilities=("external_tool",),
            side_effect="write",
            toolkit_slug="linear",
        ),
        ToolCard(
            registry_name="mcp__context7__query",
            provider="mcp",
            display_name="context7 query",
            description="Query docs.",
            capabilities=("external_tool",),
            side_effect="read",
            toolkit_slug="context7",
        ),
    )
    installation = create_installation(db_session)
    server = McpServer(
        installation_id=installation.id,
        name="context7",
        transport="streamable_http",
        url="https://example.test/mcp",
        status="enabled",
        created_by="test",
    )

    overview = build_capability_overview(
        native_descriptors=descriptors,
        external_cards=cards,
        mcp_rows=(server,),
    )

    assert overview.native_categories == ("Documents",)
    assert overview.disabled_native == (("web_search", "Missing key"),)
    assert overview.composio_toolkits == ("linear",)
    assert overview.mcp_servers == (("context7", "enabled"),)


def test_capability_overview_uses_connected_toolkits_without_cards() -> None:
    # HIG-274 regression: when intent routes a request away from external tools,
    # external_cards is empty. The connected set must still be authoritative so
    # the <capabilities> block does not go blind and the agent does not fabricate
    # "not connected" (the c65e7b2f failure).
    overview = build_capability_overview(
        native_descriptors=(make_descriptor("echo", category="Documents"),),
        external_cards=(),
        mcp_rows=(),
        connected_composio_toolkits=("notion", "linear"),
    )

    assert overview.composio_toolkits == ("linear", "notion")
    rendered = render_capability_overview(overview)
    assert "linear" in rendered
    assert "notion" in rendered


def test_capability_card_is_second_message_after_system_prompt(
    db_session: Session,
) -> None:
    task = create_task(db_session)
    package = ContextAssembler(
        session=db_session,
        system_prompt="system prompt",
        capability_overview=make_overview(),
    ).build_for_task(task)

    assert package.messages[0].content == "system prompt"
    second = package.messages[1].content
    assert second is not None
    assert second.startswith("<capabilities>")
    assert "Connected: native tool categories: Research, Documents." in second
    assert "Connected: Composio toolkits: github, linear." in second
    assert "Connected: MCP servers: context7." in second
    assert (
        "Unavailable (needs setup): web_search "
        "(Missing env var BRAVE_SEARCH_API_KEY); legacy (MCP server disabled)."
    ) in second


def test_capability_card_overflow_is_budgeted_with_omission(
    db_session: Session,
) -> None:
    task = create_task(db_session)
    # Use enough disabled tools to exceed the 8000-char capability budget.
    overview = CapabilityOverview(
        native_categories=(),
        disabled_native=tuple(
            (f"tool_{index}", "Missing required environment variable EXAMPLE_KEY_WITH_LONG_SUFFIX")
            for index in range(300)
        ),
        composio_toolkits=(),
        mcp_servers=(),
    )
    package = ContextAssembler(
        session=db_session,
        system_prompt="system prompt",
        capability_overview=overview,
    ).build_for_task(task)

    capabilities_message = package.messages[1].content
    assert capabilities_message is not None
    assert len(capabilities_message) <= 8000
    assert "[capabilities truncated at configured budget]" in capabilities_message
    assert any(
        omission.kind == "capabilities" and omission.reason == "budget_compacted"
        for omission in package.omissions
    )


def test_absent_overview_keeps_legacy_message_list(db_session: Session) -> None:
    task = create_task(db_session)

    with_overview = ContextAssembler(
        session=db_session,
        system_prompt="system prompt",
        capability_overview=make_overview(),
    ).build_for_task(task)
    without_overview = ContextAssembler(
        session=db_session,
        system_prompt="system prompt",
    ).build_for_task(task)

    assert not any(
        (message.content or "").startswith("<capabilities>")
        for message in without_overview.messages
    )
    assert len(with_overview.messages) == len(without_overview.messages) + 1
    assert [message.content for message in without_overview.messages] == [
        message.content
        for message in with_overview.messages
        if not (message.content or "").startswith("<capabilities>")
    ]


def test_system_prompt_contains_capability_upsell_rule() -> None:
    assert "Consult the <capabilities> section." in DEFAULT_SYSTEM_PROMPT
    assert "Never respond with a flat refusal." in DEFAULT_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Skill-first planning: ranking, execution hint, events, planner payload
# ---------------------------------------------------------------------------


def setup_two_skills(session: Session, installation: Installation) -> None:
    create_enabled_skill(
        session,
        installation,
        slug="website-scrape-weekly",
        name="Website Scrape Weekly",
        description="Use when the task involves scraping a website for updates.",
    )
    create_enabled_skill(
        session,
        installation,
        slug="issue-tracker-triage",
        name="Issue Tracker Triage",
        description="Use when the task involves triaging issues and bugs.",
    )


def test_skill_ranking_orders_skills_and_sets_skill_direct_hint(
    db_session: Session,
) -> None:
    installation = create_installation(db_session)
    setup_two_skills(db_session, installation)
    task = create_task(
        db_session,
        installation,
        input_text="check our issue tracker for urgent bugs",
    )

    package = ContextAssembler(
        session=db_session,
        embedding_index=EmbeddingIndex(db_session, FakeEmbeddingBackend()),
    ).build_for_task(task)

    assert package.skill_similarities
    assert package.skill_similarities[0][0] == "issue-tracker-triage"
    assert package.execution_hint == "skill_direct"
    assert package.matched_skill_slug == "issue-tracker-triage"
    assert [skill.slug for skill in package.selected_skills] == [
        "issue-tracker-triage",
        "website-scrape-weekly",
    ]
    skills_message = next(
        message.content
        for message in package.messages
        if message.content and "<available_skills>" in message.content
    )
    assert (
        "Highly relevant skill for this task: issue-tracker-triage. "
        "Load it with load_skill and follow it before doing the work yourself."
    ) in skills_message
    triage_index = skills_message.index("issue-tracker-triage")
    scrape_index = skills_message.index("website-scrape-weekly")
    assert triage_index < scrape_index


def test_skill_ranking_below_threshold_keeps_hint_unset(db_session: Session) -> None:
    installation = create_installation(db_session)
    setup_two_skills(db_session, installation)
    task = create_task(
        db_session,
        installation,
        input_text="frobnicate the quux",
    )

    package = ContextAssembler(
        session=db_session,
        embedding_index=EmbeddingIndex(db_session, FakeEmbeddingBackend()),
        skill_direct_threshold=0.60,
    ).build_for_task(task)

    assert package.skill_similarities
    assert package.execution_hint is None
    assert package.matched_skill_slug is None
    skills_message = next(
        message.content
        for message in package.messages
        if message.content and "<available_skills>" in message.content
    )
    assert "Highly relevant skill" not in skills_message


def test_no_embedding_index_falls_back_to_lexical_ranking(
    db_session: Session,
) -> None:
    installation = create_installation(db_session)
    setup_two_skills(db_session, installation)
    task = create_task(
        db_session,
        installation,
        input_text="check our issue tracker for urgent bugs",
    )

    # HIG-239: without an embedding index the ranker falls back to lexical token
    # overlap instead of returning empty / legacy order — the issue-tracker skill
    # (name + description overlap) must surface on top.
    package = ContextAssembler(session=db_session).build_for_task(task)
    ranked_slugs = [skill.slug for skill in package.selected_skills]

    assert package.skill_similarities
    assert ranked_slugs[0] == "issue-tracker-triage"
    assert set(ranked_slugs) == {"issue-tracker-triage", "website-scrape-weekly"}


def test_ranked_slot_budget_is_filled_by_similarity(db_session: Session) -> None:
    installation = create_installation(db_session)
    for index in range(31):
        create_enabled_skill(
            db_session,
            installation,
            slug=f"filler-skill-{index:02d}",
            name=f"Filler Skill {index:02d}",
            description="Use when the task involves nothing in particular.",
        )
    create_enabled_skill(
        db_session,
        installation,
        slug="issue-tracker-triage",
        name="Issue Tracker Triage",
        description="Use when the task involves triaging issues and bugs.",
    )
    task = create_task(
        db_session,
        installation,
        input_text="check our issue tracker for urgent bugs",
    )

    package = ContextAssembler(
        session=db_session,
        embedding_index=EmbeddingIndex(db_session, FakeEmbeddingBackend()),
    ).build_for_task(task)

    # HIG-239: the ranked index is capped at 15 (down from 30); the highest-
    # similarity skill still sorts to the top and overflow is recorded.
    assert len(package.selected_skills) == DEFAULT_SKILLS_CONTEXT_MAX_SKILLS
    assert package.selected_skills[0].slug == "issue-tracker-triage"
    assert any(
        omission.kind == "skills" and omission.reason == "skills_context_max_skills"
        for omission in package.omissions
    )


def test_coordinator_records_skill_ranking_event(db_session: Session) -> None:
    installation = create_installation(db_session)
    setup_two_skills(db_session, installation)
    task = create_task(
        db_session,
        installation,
        input_text="check our issue tracker for urgent bugs",
    )
    assembler = ContextAssembler(
        session=db_session,
        embedding_index=EmbeddingIndex(db_session, FakeEmbeddingBackend()),
    )

    result = AgentCoordinator(
        session=db_session,
        llm=FakeLLM(),
        registry=ToolRegistry([]),
        context_assembler=assembler,
    ).run(task)

    assert result.result_summary == "All done."
    event = next(
        event
        for event in db_session.scalars(
            select(TaskEvent)
            .where(TaskEvent.task_id == task.id, TaskEvent.type == TaskEventType.log)
            .order_by(TaskEvent.seq)
        )
        if event.payload.get("message") == "skill_ranking"
    )
    assert event.payload["execution_hint"] == "skill_direct"
    assert event.payload["matched_skill_slug"] == "issue-tracker-triage"
    ranked = event.payload["ranked"]
    assert ranked[0]["slug"] == "issue-tracker-triage"
    assert 0.0 <= ranked[0]["similarity"] <= 1.0


def test_skill_ranking_event_uses_lexical_fallback_without_index(
    db_session: Session,
) -> None:
    installation = create_installation(db_session)
    setup_two_skills(db_session, installation)
    task = create_task(
        db_session,
        installation,
        input_text="check our issue tracker for urgent bugs",
    )

    AgentCoordinator(
        session=db_session,
        llm=FakeLLM(),
        registry=ToolRegistry([]),
        context_assembler=ContextAssembler(session=db_session),
    ).run(task)

    # HIG-239: even without an embedding index, lexical fallback produces a
    # ranking, so the skill_ranking event is recorded with the top match.
    event = next(
        event
        for event in db_session.scalars(
            select(TaskEvent).where(TaskEvent.task_id == task.id)
        )
        if event.payload.get("message") == "skill_ranking"
    )
    assert event.payload["ranked"][0]["slug"] == "issue-tracker-triage"


def test_planner_payload_includes_matched_skills(db_session: Session) -> None:
    task = create_task(db_session, input_text="triage the tracker backlog")
    package = make_context_package(
        skill_similarities=(
            ("issue-tracker-triage", 0.82),
            ("website-scrape-weekly", 0.5),
            ("low-relevance", 0.2),
        ),
        selected_skills=(
            make_context_skill("issue-tracker-triage", "Triage issues and bugs."),
            make_context_skill("website-scrape-weekly", "Scrape websites weekly."),
            make_context_skill("low-relevance", "Unrelated."),
        ),
    )
    llm = FakeLLM(
        content=json.dumps(
            {
                "objective": "Triage the tracker backlog",
                "steps": [{"description": "Load the matching skill"}],
            }
        )
    )

    ExecutionPlanner().create_plan(
        task=task,
        llm=llm,
        tool_schemas=({"name": "load_skill", "description": "Load a skill"},),
        limits=ExecutionGuardrailLimits(),
        intent_decision=None,
        reason="test",
        context_package=package,
    )

    user_payload = json.loads(llm.calls[0][1].content or "{}")
    assert user_payload["matched_skills"] == [
        {
            "slug": "issue-tracker-triage",
            "description": "Triage issues and bugs.",
            "similarity": 0.82,
        },
        {
            "slug": "website-scrape-weekly",
            "description": "Scrape websites weekly.",
            "similarity": 0.5,
        },
    ]
    assert "make the FIRST step load_skill" in PLANNER_SYSTEM_PROMPT


def test_planner_payload_unchanged_without_matched_skills(db_session: Session) -> None:
    task = create_task(db_session, input_text="triage the tracker backlog")
    llm = FakeLLM(
        content=json.dumps(
            {
                "objective": "Triage the tracker backlog",
                "steps": [{"description": "Look things up"}],
            }
        )
    )

    ExecutionPlanner().create_plan(
        task=task,
        llm=llm,
        tool_schemas=({"name": "web_search", "description": "Search"},),
        limits=ExecutionGuardrailLimits(),
        intent_decision=None,
        reason="test",
        context_package=None,
    )

    user_payload = json.loads(llm.calls[0][1].content or "{}")
    assert "matched_skills" not in user_payload
    assert set(user_payload) == {"task_input", "intent_decision", "available_tools"}
