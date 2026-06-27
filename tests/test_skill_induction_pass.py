"""Tests for the HIG-300 S3 skill induction pass."""

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
from sqlalchemy import Engine, delete, select
from sqlalchemy.orm import Session

from kortny.consolidator.service import ConsolidationService
from kortny.consolidator.skill_induction import (
    SkillInductionPass,
    parse_skill_induction_proposal,
)
from kortny.db.models import (
    ConsolidationRun,
    Episode,
    Installation,
    LLMUsage,
    ProceduralSkill,
    ProceduralSkillInvocation,
    ProceduralSkillVersion,
    SkillEnablement,
    Task,
    TaskEvent,
    TaskEventType,
    TaskStatus,
)
from kortny.db.models import (
    LLMProvider as DbLLMProvider,
)
from kortny.db.session import make_engine, make_session_factory, normalize_database_url
from kortny.embeddings import EmbeddingIndex
from kortny.llm import (
    ChatMessage,
    Completion,
    LLMService,
    ModelRoute,
    ModelRouteTier,
    TokenUsage,
)
from kortny.tasks import TaskService
from kortny.tools.types import JsonObject, JsonSchema
from tests.fake_embeddings import FakeEmbeddingBackend

TEST_POSTGRES_URL = os.environ.get("KORTNY_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    TEST_POSTGRES_URL is None,
    reason="KORTNY_TEST_POSTGRES_URL is required for skill induction tests",
)

NOW = datetime(2026, 6, 26, 3, 0, 0, tzinfo=UTC)


# -- fixtures ------------------------------------------------------------------


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
        ProceduralSkillInvocation,
        SkillEnablement,
        ProceduralSkillVersion,
        ProceduralSkill,
        LLMUsage,
        TaskEvent,
        Episode,
        Task,
        Installation,
    ):
        session.execute(delete(model))


# -- helpers -------------------------------------------------------------------


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
    identity_kind: str | None = None,
    routing_quality: str | None = "clean",
) -> Task:
    task = TaskService(session).create_task(
        installation_id=installation.id,
        slack_event_id=f"Ev{uuid.uuid4().hex}",
        slack_channel_id=channel_id,
        slack_thread_ts="1780000000.000100",
        slack_message_ts=f"1780000000.{uuid.uuid4().hex[:6]}",
        slack_user_id="U_USER",
        input="summarize and send report",
    )
    if identity_kind is not None:
        task.identity_kind = identity_kind
    if routing_quality is not None:
        task.routing_quality = routing_quality
    task.status = TaskStatus.succeeded
    session.flush()
    return task


def add_tool_call_events(
    session: Session,
    task: Task,
    tool_names: list[str],
) -> None:
    """Add tool_call TaskEvents to a task in order."""
    task_service = TaskService(session)
    for tool_name in tool_names:
        task_service.append_event(
            task,
            TaskEventType.tool_call,
            {"tool": tool_name, "args": {}},
        )


def create_episode(
    session: Session,
    installation: Installation,
    task: Task,
    *,
    summary: str = "Searched the web and drafted a report.",
    outcome: str = "succeeded",
    tools_used: list[str] | None = None,
) -> Episode:
    episode = Episode(
        installation_id=installation.id,
        task_id=task.id,
        channel_id=task.slack_channel_id,
        user_id=task.slack_user_id,
        thread_ts=None,
        summary=summary,
        tools_used=tools_used or [],
        artifacts_created=[],
        source_refs=[],
        outcome=outcome,
    )
    session.add(episode)
    session.flush()
    return episode


class FakeInductionLLMProvider:
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
            raise AssertionError("FakeInductionLLMProvider got too many calls")
        return self.completions.pop(0)


def make_completion(payload: dict[str, object]) -> Completion:
    return Completion(
        content=json.dumps(payload),
        tool_calls=(),
        usage=TokenUsage(input_tokens=120, output_tokens=40),
        cost_usd=Decimal("0.000100"),
        model="openai/gpt-4o-mini",
    )


def make_llm_service(
    session: Session,
    task: Task,
    completions: list[Completion],
) -> LLMService:
    """Build a real LLMService wired to a fake provider."""
    provider = FakeInductionLLMProvider(completions)
    model_route = ModelRoute(
        tier=ModelRouteTier.cheap_fast,
        model="openai/gpt-4o-mini",
        reason="test",
    )
    return LLMService(
        session=session,
        provider=provider,
        provider_name=DbLLMProvider.openrouter,
        task_service=TaskService(session),
        model_route=model_route,
    )


def fake_index(session: Session) -> EmbeddingIndex:
    return EmbeddingIndex(session, FakeEmbeddingBackend())


def make_pass(
    session: Session,
    *,
    llm: LLMService | None = None,
    embedding_index: EmbeddingIndex | None = None,
    min_tool_calls: int = 3,
) -> SkillInductionPass:
    return SkillInductionPass(
        session,
        llm=llm,
        embedding_index=embedding_index,
        min_tool_calls=min_tool_calls,
    )


# -- parse_skill_induction_proposal -------------------------------------------


def test_parse_noop_proposal() -> None:
    raw = json.dumps({"action": "NOOP", "confidence": 0.1, "reason": "one-off"})
    result = parse_skill_induction_proposal(raw)
    assert result is not None
    assert result.action == "NOOP"


def test_parse_create_proposal() -> None:
    raw = json.dumps(
        {
            "action": "CREATE",
            "name": "research-and-report",
            "description": "Search the web and draft a summary report.",
            "allowed_tools": ["web_search", "create_document"],
            "instructions_md": "## Steps\n1. Search\n2. Draft",
            "confidence": 0.85,
            "reason": "Repeatable cross-context workflow",
        }
    )
    result = parse_skill_induction_proposal(raw)
    assert result is not None
    assert result.action == "CREATE"
    assert result.name == "research-and-report"
    assert "web_search" in result.allowed_tools
    assert result.confidence == pytest.approx(0.85)


def test_parse_create_proposal_low_confidence_becomes_noop() -> None:
    raw = json.dumps(
        {
            "action": "CREATE",
            "name": "some-skill",
            "description": "Does something.",
            "allowed_tools": [],
            "instructions_md": "## Steps",
            "confidence": 0.50,
            "reason": "Not confident enough",
        }
    )
    result = parse_skill_induction_proposal(raw)
    assert result is not None
    assert result.action == "NOOP"


def test_parse_invalid_json_returns_none() -> None:
    assert parse_skill_induction_proposal("not json") is None
    assert parse_skill_induction_proposal(None) is None
    assert parse_skill_induction_proposal("") is None


# -- scan gate -----------------------------------------------------------------


def test_gate_skips_episode_with_too_few_tool_calls(db_session: Session) -> None:
    installation = create_installation(db_session)
    task = create_task(db_session, installation)
    # Only 1 tool call -- below the min
    add_tool_call_events(db_session, task, ["web_search"])
    episode = create_episode(db_session, installation, task)

    pass_ = make_pass(db_session, min_tool_calls=3)
    eligible = pass_._eligible_episodes(
        installation_id=installation.id, since=None, cap=50
    )
    assert episode.id not in [e.id for e in eligible]


def test_gate_accepts_episode_with_enough_diverse_tool_calls(
    db_session: Session,
) -> None:
    installation = create_installation(db_session)
    task = create_task(db_session, installation)
    add_tool_call_events(
        db_session,
        task,
        ["web_search", "create_document", "send_email"],
    )
    episode = create_episode(db_session, installation, task)

    pass_ = make_pass(db_session, min_tool_calls=3)
    eligible = pass_._eligible_episodes(
        installation_id=installation.id, since=None, cap=50
    )
    assert episode.id in [e.id for e in eligible]


def test_gate_skips_synthetic_tasks(db_session: Session) -> None:
    installation = create_installation(db_session)
    task = create_task(db_session, installation, identity_kind="synthetic")
    add_tool_call_events(
        db_session, task, ["web_search", "create_document", "send_email"]
    )
    episode = create_episode(db_session, installation, task)

    pass_ = make_pass(db_session, min_tool_calls=3)
    eligible = pass_._eligible_episodes(
        installation_id=installation.id, since=None, cap=50
    )
    assert episode.id not in [e.id for e in eligible]


def test_gate_skips_episode_with_no_routing_quality(db_session: Session) -> None:
    installation = create_installation(db_session)
    task = create_task(db_session, installation, routing_quality=None)
    add_tool_call_events(
        db_session, task, ["web_search", "create_document", "send_email"]
    )
    episode = create_episode(db_session, installation, task)

    pass_ = make_pass(db_session, min_tool_calls=3)
    eligible = pass_._eligible_episodes(
        installation_id=installation.id, since=None, cap=50
    )
    assert episode.id not in [e.id for e in eligible]


# -- NOOP proposal -- nothing created -----------------------------------------


def test_noop_proposal_creates_no_skill(db_session: Session) -> None:
    installation = create_installation(db_session)
    task = create_task(db_session, installation)
    add_tool_call_events(
        db_session, task, ["web_search", "create_document", "send_email"]
    )
    create_episode(db_session, installation, task)
    consolidator_task = create_task(db_session, installation)

    noop_completion = make_completion(
        {"action": "NOOP", "confidence": 0.1, "reason": "one-off answer"}
    )
    llm = make_llm_service(db_session, consolidator_task, [noop_completion])
    pass_ = make_pass(db_session, llm=llm, min_tool_calls=3)

    counters = pass_.run(
        installation_id=installation.id,
        task=consolidator_task,
        since=None,
        now=NOW,
    )

    assert counters.noop == 1
    assert counters.created == 0
    skills = list(db_session.scalars(select(ProceduralSkill)))
    assert len(skills) == 0


# -- CREATE proposal -- untrusted skill, no enablement ------------------------


def test_create_proposal_stores_untrusted_skill_with_no_enablement(
    db_session: Session,
) -> None:
    installation = create_installation(db_session)
    task = create_task(db_session, installation)
    add_tool_call_events(
        db_session, task, ["web_search", "create_document", "send_email"]
    )
    episode = create_episode(db_session, installation, task)
    consolidator_task = create_task(db_session, installation)

    create_completion = make_completion(
        {
            "action": "CREATE",
            "name": "research-and-report",
            "description": "Search the web and draft a summary report.",
            "allowed_tools": ["web_search", "create_document"],
            "instructions_md": "## Steps\n1. Search\n2. Draft",
            "confidence": 0.85,
            "reason": "Repeatable cross-context workflow",
        }
    )
    llm = make_llm_service(db_session, consolidator_task, [create_completion])
    pass_ = make_pass(db_session, llm=llm, min_tool_calls=3)

    counters = pass_.run(
        installation_id=installation.id,
        task=consolidator_task,
        since=None,
        now=NOW,
    )

    assert counters.created == 1
    assert counters.noop == 0

    # Verify the skill was created
    skill = db_session.scalar(select(ProceduralSkill))
    assert skill is not None
    assert skill.trust_level == "untrusted"
    assert skill.visibility == "catalog"
    assert skill.status == "active"
    assert "agent_induced" in skill.provenance

    # Verify the version has induction metadata
    version = db_session.scalar(
        select(ProceduralSkillVersion).where(
            ProceduralSkillVersion.skill_id == skill.id,
            ProceduralSkillVersion.status == "active",
        )
    )
    assert version is not None
    assert version.metadata_json.get("induction_state") == "candidate"
    assert version.metadata_json.get("source_episode_id") == str(episode.id)
    assert version.metadata_json.get("source_task_id") == str(task.id)

    # CRITICAL SAFETY ASSERTION: No SkillEnablement must exist
    enablements = list(db_session.scalars(select(SkillEnablement)))
    assert len(enablements) == 0, (
        "Induced skill must NOT have any SkillEnablement -- it must be catalog-only"
    )


def test_induced_skill_not_visible_to_runtime(db_session: Session) -> None:
    """Catalog-only untrusted skill with no enablement is invisible to the runtime."""
    from kortny.skills.service import SkillRegistryService

    installation = create_installation(db_session)
    task = create_task(db_session, installation)
    add_tool_call_events(
        db_session, task, ["web_search", "create_document", "send_email"]
    )
    create_episode(db_session, installation, task)
    consolidator_task = create_task(db_session, installation)

    create_completion = make_completion(
        {
            "action": "CREATE",
            "name": "runtime-invisible-skill",
            "description": "Should never appear at runtime.",
            "allowed_tools": ["web_search"],
            "instructions_md": "## Steps\n1. Never run",
            "confidence": 0.90,
            "reason": "Test",
        }
    )
    llm = make_llm_service(db_session, consolidator_task, [create_completion])
    pass_ = make_pass(db_session, llm=llm, min_tool_calls=3)

    pass_.run(
        installation_id=installation.id,
        task=consolidator_task,
        since=None,
        now=NOW,
    )

    # The runtime queries enabled_skills_for_task which requires a SkillEnablement
    registry = SkillRegistryService(db_session)
    enabled = registry.enabled_skills_for_task(consolidator_task)
    slugs = [s.slug for s in enabled]
    assert "runtime-invisible-skill" not in slugs


# -- watermark / anchor -------------------------------------------------------


def test_anchor_advances_across_runs(db_session: Session) -> None:
    installation = create_installation(db_session)
    task = create_task(db_session, installation)
    add_tool_call_events(
        db_session, task, ["web_search", "create_document", "send_email"]
    )
    create_episode(db_session, installation, task)
    consolidator_task = create_task(db_session, installation)

    noop_completion = make_completion(
        {"action": "NOOP", "confidence": 0.0, "reason": "one-off"}
    )
    llm = make_llm_service(db_session, consolidator_task, [noop_completion])
    pass_ = make_pass(db_session, llm=llm, min_tool_calls=3)

    counters = pass_.run(
        installation_id=installation.id,
        task=consolidator_task,
        since=None,
        now=NOW,
    )

    assert counters.anchor is not None
    # Anchor should be set (the episode's created_at ISO string)
    assert counters.anchor


# -- dedup --------------------------------------------------------------------


def test_dedup_skips_similar_existing_skill(db_session: Session) -> None:
    installation = create_installation(db_session)
    task = create_task(db_session, installation)
    add_tool_call_events(
        db_session, task, ["web_search", "create_document", "send_email"]
    )
    create_episode(db_session, installation, task)
    consolidator_task = create_task(db_session, installation)

    create_completion = make_completion(
        {
            "action": "CREATE",
            "name": "research-and-report",
            "description": "Search the web and draft a summary report.",
            "allowed_tools": ["web_search"],
            "instructions_md": "## Steps\n1. Do the thing",
            "confidence": 0.85,
            "reason": "workflow",
        }
    )
    index = fake_index(db_session)
    llm = make_llm_service(db_session, consolidator_task, [create_completion])
    pass_ = make_pass(db_session, llm=llm, embedding_index=index, min_tool_calls=3)

    # First run: creates the skill
    counters1 = pass_.run(
        installation_id=installation.id,
        task=consolidator_task,
        since=None,
        now=NOW,
    )
    db_session.commit()

    # Second run with a similar episode and similar proposal
    task2 = create_task(db_session, installation)
    add_tool_call_events(
        db_session, task2, ["web_search", "create_document", "send_email"]
    )
    create_episode(db_session, installation, task2)
    consolidator_task2 = create_task(db_session, installation)

    create_completion2 = make_completion(
        {
            "action": "CREATE",
            "name": "research-and-report",
            "description": "Search the web and draft a summary report.",
            "allowed_tools": ["web_search"],
            "instructions_md": "## Steps\n1. Do the thing",
            "confidence": 0.85,
            "reason": "workflow",
        }
    )
    llm2 = make_llm_service(db_session, consolidator_task2, [create_completion2])
    # Use the same fake index which now has the first skill embedded
    pass2 = make_pass(db_session, llm=llm2, embedding_index=index, min_tool_calls=3)
    since_anchor = counters1.anchor
    since_dt = datetime.fromisoformat(since_anchor) if since_anchor else None

    counters2 = pass2.run(
        installation_id=installation.id,
        task=consolidator_task2,
        since=since_dt,
        now=NOW + timedelta(minutes=5),
    )

    # The FakeEmbeddingBackend returns the same vector for any text, so
    # similarity = 1.0 > threshold -- deduped_skipped
    assert counters2.deduped_skipped >= 1


# -- failure isolation --------------------------------------------------------


def test_llm_unavailable_returns_empty_counters(db_session: Session) -> None:
    installation = create_installation(db_session)
    task = create_task(db_session, installation)
    add_tool_call_events(
        db_session, task, ["web_search", "create_document", "send_email"]
    )
    create_episode(db_session, installation, task)
    consolidator_task = create_task(db_session, installation)

    # LLM is None -- pass skips gracefully
    pass_ = make_pass(db_session, llm=None, min_tool_calls=3)
    counters = pass_.run(
        installation_id=installation.id,
        task=consolidator_task,
        since=None,
        now=NOW,
    )

    assert counters.episodes_scanned == 0
    assert counters.created == 0


# -- consolidator service integration ----------------------------------------


def test_consolidator_service_skill_induction_pass_registered_when_flag_on(
    db_session: Session,
) -> None:
    from kortny.config import Settings

    installation = create_installation(db_session)
    task = create_task(db_session, installation)
    add_tool_call_events(
        db_session, task, ["web_search", "create_document", "send_email"]
    )
    create_episode(db_session, installation, task)
    db_session.commit()

    all_completions: list[Completion] = [
        # promotion pass (batch of 1 episode)
        make_completion({"decisions": []}),
        # skill_induction pass (1 episode eligible)
        make_completion({"action": "NOOP", "confidence": 0.0, "reason": "test noop"}),
    ]

    class MultiCompletionProvider:
        model = "openai/gpt-4o-mini"

        def __init__(self, completions: list[Completion]) -> None:
            self._completions = completions

        def complete(
            self,
            messages: Sequence[ChatMessage],
            tools: Sequence[JsonSchema] = (),
            *,
            response_format: JsonObject | None = None,
            max_output_tokens: int | None = None,
        ) -> Completion:
            del tools
            if not self._completions:
                return make_completion({"decisions": []})
            return self._completions.pop(0)

    settings = Settings.model_validate(
        {
            "SLACK_BOT_TOKEN": "xoxb-test",
            "SLACK_APP_TOKEN": "xapp-test",
            "SLACK_SIGNING_SECRET": "signing-secret",
            "LLM_PROVIDER": "openrouter",
            "LLM_API_KEY": "test-key",
            "LLM_MODEL": "openai/gpt-test",
            "COMPOSIO_API_KEY": "composio-key",
            "POSTGRES_URL": "postgresql://test/test",
            "KORTNY_EMBEDDINGS_BACKEND": "disabled",
            "KORTNY_SKILL_INDUCTION_ENABLED": "true",
            "KORTNY_SKILL_INDUCTION_MIN_TOOL_CALLS": "3",
        }
    )

    service = ConsolidationService(
        db_session,
        settings=settings,
        llm_provider=MultiCompletionProvider(all_completions),
        provider_name="openrouter",
    )

    outcome = service.run_once(installation_id=installation.id, now=NOW)

    assert outcome.status == "succeeded"
    # The skill_induction pass counters should be present
    assert "skill_induction" in outcome.counters


def test_consolidator_service_no_skill_induction_when_flag_off(
    db_session: Session,
) -> None:
    from kortny.config import Settings

    installation = create_installation(db_session)
    task = create_task(db_session, installation)
    add_tool_call_events(
        db_session, task, ["web_search", "create_document", "send_email"]
    )
    create_episode(db_session, installation, task)
    db_session.commit()

    settings = Settings.model_validate(
        {
            "SLACK_BOT_TOKEN": "xoxb-test",
            "SLACK_APP_TOKEN": "xapp-test",
            "SLACK_SIGNING_SECRET": "signing-secret",
            "LLM_PROVIDER": "openrouter",
            "LLM_API_KEY": "test-key",
            "LLM_MODEL": "openai/gpt-test",
            "COMPOSIO_API_KEY": "composio-key",
            "POSTGRES_URL": "postgresql://test/test",
            "KORTNY_EMBEDDINGS_BACKEND": "disabled",
            # Flag OFF (default)
        }
    )

    service = ConsolidationService(
        db_session,
        settings=settings,
        llm_provider=None,
        provider_name="openrouter",
    )

    outcome = service.run_once(installation_id=installation.id, now=NOW)

    assert outcome.status == "succeeded"
    # skill_induction pass NOT present when flag off
    assert "skill_induction" not in outcome.counters
