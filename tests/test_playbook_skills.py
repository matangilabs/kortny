"""HIG-229 coworker playbook pack: curated seeding, enablement, ranking."""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, delete, select
from sqlalchemy.orm import Session

from kortny.agent.context import ContextAssembler
from kortny.db.models import (
    Installation,
    ProceduralSkill,
    ProceduralSkillInvocation,
    ProceduralSkillVersion,
    SkillEnablement,
    SkillFile,
    Task,
    TaskEvent,
    ToolEmbedding,
)
from kortny.db.session import make_engine, make_session_factory, normalize_database_url
from kortny.embeddings import EmbeddingIndex
from kortny.skills.ingestion import SkillIngestionService
from kortny.skills.service import (
    CURATED_SKILLS_DIR,
    PLAYBOOK_ENABLEMENT_ADDED_BY,
    PLAYBOOK_SKILL_SLUGS,
    SkillRegistryService,
)
from kortny.tasks import TaskService
from tests.fake_embeddings import FakeEmbeddingBackend

TEST_POSTGRES_URL = os.environ.get("KORTNY_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    TEST_POSTGRES_URL is None,
    reason="KORTNY_TEST_POSTGRES_URL is required for playbook skill tests",
)

# Concept dimensions for the playbook ranking smoke (FakeEmbeddingBackend).
_CHECKIN = 0
_DECISIONS = 1
_AMBIENT = 2
_DATA = 3
_DRAFT = 4

PLAYBOOK_VOCABULARY: dict[str, int] = {
    # Recurring check-in / status posting.
    "check": _CHECKIN,
    "checkin": _CHECKIN,
    "morning": _CHECKIN,
    "standup": _CHECKIN,
    "recurring": _CHECKIN,
    "project": _CHECKIN,
    "status": _CHECKIN,
    # Decision tracking.
    "decision": _DECISIONS,
    "decisions": _DECISIONS,
    "owner": _DECISIONS,
    "deadline": _DECISIONS,
    "tracking": _DECISIONS,
    "unknowns": _DECISIONS,
    # Ambient replies.
    "reply": _AMBIENT,
    "replying": _AMBIENT,
    "addressed": _AMBIENT,
    "expertise": _AMBIENT,
    "unblocking": _AMBIENT,
    "risk": _AMBIENT,
    # Data briefs.
    "spreadsheet": _DATA,
    "csv": _DATA,
    "numbers": _DATA,
    "data": _DATA,
    "figures": _DATA,
    "brief": _DATA,
    # Anticipatory drafting.
    "draft": _DRAFT,
    "announcement": _DRAFT,
    "deliverable": _DRAFT,
    "due": _DRAFT,
    "email": _DRAFT,
    "confirm": _DRAFT,
}


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
        SkillFile,
        ProceduralSkillInvocation,
        ProceduralSkillVersion,
        ProceduralSkill,
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


def playbook_enablements(session: Session) -> list[SkillEnablement]:
    return list(
        session.scalars(
            select(SkillEnablement)
            .join(ProceduralSkill, ProceduralSkill.id == SkillEnablement.skill_id)
            .where(ProceduralSkill.slug.in_(PLAYBOOK_SKILL_SLUGS))
            .order_by(ProceduralSkill.slug)
        )
    )


# ---------------------------------------------------------------------------
# Seeding: five skills present + active + workspace-enabled; idempotent
# ---------------------------------------------------------------------------


def test_seed_fresh_db_five_skills_active_and_workspace_enabled(
    db_session: Session,
) -> None:
    installation = create_installation(db_session)

    SkillRegistryService(db_session).ensure_curated_skills()

    skills = {
        skill.slug: skill
        for skill in db_session.scalars(
            select(ProceduralSkill).where(
                ProceduralSkill.slug.in_(PLAYBOOK_SKILL_SLUGS)
            )
        )
    }
    assert set(skills) == set(PLAYBOOK_SKILL_SLUGS)
    for skill in skills.values():
        assert skill.status == "active"
        assert skill.owner_type == "system"
        assert skill.trust_level == "trusted"
        assert skill.provenance == "kortny"
        version = db_session.scalar(
            select(ProceduralSkillVersion).where(
                ProceduralSkillVersion.skill_id == skill.id,
                ProceduralSkillVersion.status == "active",
            )
        )
        assert version is not None
        assert version.description.strip()

    enablements = playbook_enablements(db_session)
    assert len(enablements) == len(PLAYBOOK_SKILL_SLUGS)
    for enablement in enablements:
        assert enablement.installation_id == installation.id
        assert enablement.scope_type == "workspace"
        assert enablement.scope_id is None
        assert enablement.status == "enabled"
        assert enablement.added_by == PLAYBOOK_ENABLEMENT_ADDED_BY


def test_reseed_is_idempotent(db_session: Session) -> None:
    create_installation(db_session)
    service = SkillRegistryService(db_session)

    service.ensure_curated_skills()
    service.ensure_curated_skills()

    for slug in PLAYBOOK_SKILL_SLUGS:
        skill = db_session.scalar(
            select(ProceduralSkill).where(ProceduralSkill.slug == slug)
        )
        assert skill is not None
        versions = list(
            db_session.scalars(
                select(ProceduralSkillVersion).where(
                    ProceduralSkillVersion.skill_id == skill.id
                )
            )
        )
        assert len(versions) == 1
    assert len(playbook_enablements(db_session)) == len(PLAYBOOK_SKILL_SLUGS)


def test_reseed_does_not_reenable_disabled_playbook(db_session: Session) -> None:
    create_installation(db_session)
    service = SkillRegistryService(db_session)
    service.ensure_curated_skills()

    target = playbook_enablements(db_session)[0]
    service.disable_skill(enablement_id=target.id, by="dashboard:admin")
    service.ensure_curated_skills()

    db_session.refresh(target)
    assert target.status == "disabled"
    assert len(playbook_enablements(db_session)) == len(PLAYBOOK_SKILL_SLUGS)


def test_installation_added_after_first_seed_gets_enablements_on_reseed(
    db_session: Session,
) -> None:
    first = create_installation(db_session)
    service = SkillRegistryService(db_session)
    service.ensure_curated_skills()

    second = create_installation(db_session)
    service.ensure_curated_skills()

    by_installation: dict[uuid.UUID, int] = {}
    for enablement in playbook_enablements(db_session):
        count = by_installation.get(enablement.installation_id, 0)
        by_installation[enablement.installation_id] = count + 1
    assert by_installation == {
        first.id: len(PLAYBOOK_SKILL_SLUGS),
        second.id: len(PLAYBOOK_SKILL_SLUGS),
    }


# ---------------------------------------------------------------------------
# Each SKILL.md parses via the ingestion validators with sane content
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("slug", PLAYBOOK_SKILL_SLUGS)
def test_playbook_skill_md_parses_with_sane_description_and_body(
    db_session: Session, slug: str
) -> None:
    result = SkillIngestionService(db_session).ingest_directory(
        CURATED_SKILLS_DIR / slug,
        owner_type="system",
        owner_id=None,
        provenance="kortny",
        trust_level="trusted",
        created_by="test",
    )

    assert result.skill.slug == slug
    description = result.version.description
    assert description.startswith("Use when")
    assert 50 <= len(description) <= 400
    assert 400 <= len(result.version.instructions_md) <= 4000
    assert result.version.metadata_json.get("display_name")
    assert result.version.metadata_json.get("tags")


def test_project_checkin_ships_example_reference(db_session: Session) -> None:
    result = SkillIngestionService(db_session).ingest_directory(
        CURATED_SKILLS_DIR / "project-checkin",
        owner_type="system",
        owner_id=None,
        provenance="kortny",
        trust_level="trusted",
        created_by="test",
    )

    assert {file.path for file in result.files} == {"references/example.md"}


# ---------------------------------------------------------------------------
# Ranking smoke: check-in-ish input ranks project-checkin on top (HIG-219)
# ---------------------------------------------------------------------------


def test_checkin_query_ranks_project_checkin_top(db_session: Session) -> None:
    installation = create_installation(db_session)
    SkillRegistryService(db_session).ensure_curated_skills()
    task = create_task(
        db_session,
        installation,
        input_text="post the morning project check-in for the team",
    )

    package = ContextAssembler(
        session=db_session,
        embedding_index=EmbeddingIndex(
            db_session, FakeEmbeddingBackend(vocabulary=PLAYBOOK_VOCABULARY)
        ),
    ).build_for_task(task)

    assert package.skill_similarities
    assert package.skill_similarities[0][0] == "project-checkin"
    assert package.execution_hint == "skill_direct"
    assert package.matched_skill_slug == "project-checkin"
    assert [skill.slug for skill in package.selected_skills][0] == "project-checkin"
    assert {skill.slug for skill in package.selected_skills} == set(
        PLAYBOOK_SKILL_SLUGS
    )
    skills_message = next(
        message.content
        for message in package.messages
        if message.content and "<available_skills>" in message.content
    )
    assert "project-checkin" in skills_message
