"""Tests for the skill directory: models, ingestion, scoped enablement."""

from __future__ import annotations

import os
import shutil
import uuid
import zipfile
from collections.abc import Iterator
from io import BytesIO
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from kortny.db.models import (
    Installation,
    ProceduralSkill,
    ProceduralSkillInvocation,
    ProceduralSkillVersion,
    SkillEnablement,
    SkillFile,
    Task,
    TaskEvent,
)
from kortny.db.session import make_engine, make_session_factory, normalize_database_url
from kortny.skills import SkillRegistryService
from kortny.skills.ingestion import SkillIngestionError, SkillIngestionService
from kortny.tasks import TaskService

TEST_POSTGRES_URL = os.environ.get("KORTNY_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    TEST_POSTGRES_URL is None,
    reason="KORTNY_TEST_POSTGRES_URL is required for skill directory tests",
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


def create_installation(session: Session) -> Installation:
    installation = Installation(slack_team_id=f"T{uuid.uuid4().hex}")
    session.add(installation)
    session.flush()
    return installation


def create_task(
    session: Session,
    installation: Installation | None = None,
    *,
    channel_id: str = "C123",
    user_id: str = "U123",
) -> Task:
    installation = installation or create_installation(session)
    thread_ts = f"{uuid.uuid4().int % 10**6}.{uuid.uuid4().int % 10**6}"
    return TaskService(session).create_task(
        installation_id=installation.id,
        slack_event_id=f"Ev{uuid.uuid4().hex}",
        slack_channel_id=channel_id,
        slack_thread_ts=thread_ts,
        slack_message_ts=thread_ts,
        slack_user_id=user_id,
        input="summarize the meeting notes from today",
    )


def create_skill(
    session: Session,
    *,
    slug: str = "demo-skill",
    owner_type: str = "system",
    owner_id: str | None = None,
    trust_level: str = "trusted",
    provenance: str = "kortny",
) -> tuple[ProceduralSkill, ProceduralSkillVersion]:
    skill = ProceduralSkill(
        slug=slug,
        owner_type=owner_type,
        owner_id=owner_id,
        status="active",
        trust_level=trust_level,
        visibility="catalog",
        provenance=provenance,
    )
    session.add(skill)
    session.flush()
    version = ProceduralSkillVersion(
        skill_id=skill.id,
        version="1.0.0",
        status="active",
        name=slug.replace("-", " ").title(),
        description=f"Use when the task involves {slug}.",
        instructions_md="## Steps\n1. Do the thing.",
        content_sha256="0" * 64,
        created_by="test",
    )
    session.add(version)
    session.flush()
    return skill, version


def cleanup_database(session: Session) -> None:
    for model in (
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


class TestSkillDirectoryModels:
    def test_skill_files_and_enablement_round_trip(self, db_session: Session) -> None:
        installation = create_installation(db_session)
        skill, version = create_skill(db_session)

        db_session.add(
            SkillFile(
                skill_version_id=version.id,
                path="references/notes.md",
                kind="reference",
                content_text="# Notes",
                size_bytes=7,
                sha256="a" * 64,
            )
        )
        db_session.add(
            SkillEnablement(
                installation_id=installation.id,
                skill_id=skill.id,
                scope_type="channel",
                scope_id="C42",
                added_by="dashboard:tester",
            )
        )
        db_session.flush()

        stored_file = db_session.scalar(select(SkillFile))
        assert stored_file is not None
        assert stored_file.kind == "reference"
        stored_enablement = db_session.scalar(select(SkillEnablement))
        assert stored_enablement is not None
        assert stored_enablement.status == "enabled"
        assert (
            db_session.scalar(
                select(ProceduralSkill.provenance).where(ProceduralSkill.id == skill.id)
            )
            == "kortny"
        )

    def test_workspace_enablement_rejects_scope_id(self, db_session: Session) -> None:
        installation = create_installation(db_session)
        skill, _ = create_skill(db_session)

        db_session.add(
            SkillEnablement(
                installation_id=installation.id,
                skill_id=skill.id,
                scope_type="workspace",
                scope_id="C42",
                added_by="dashboard:tester",
            )
        )
        with pytest.raises(IntegrityError):
            db_session.flush()
        db_session.rollback()

    def test_duplicate_enablement_rejected(self, db_session: Session) -> None:
        installation = create_installation(db_session)
        skill, _ = create_skill(db_session)

        for _ in range(2):
            db_session.add(
                SkillEnablement(
                    installation_id=installation.id,
                    skill_id=skill.id,
                    scope_type="workspace",
                    scope_id=None,
                    added_by="dashboard:tester",
                )
            )
        with pytest.raises(IntegrityError):
            db_session.flush()
        db_session.rollback()

    def test_legacy_trust_levels_rejected(self, db_session: Session) -> None:
        with pytest.raises(IntegrityError):
            create_skill(db_session, slug="legacy", trust_level="reviewed")
        db_session.rollback()


FIXTURE_SKILL_DIR = Path(__file__).parent / "fixtures" / "skills" / "demo-skill"

INGEST_KWARGS = {
    "owner_type": "workspace",
    "provenance": "user:U123",
    "trust_level": "untrusted",
    "created_by": "dashboard:tester",
}


def make_zip(root: Path, *, prefix: str = "") -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for file_path in sorted(root.rglob("*")):
            if file_path.is_file():
                archive.write(file_path, prefix + str(file_path.relative_to(root)))
    return buffer.getvalue()


class TestSkillIngestion:
    def test_ingest_directory_maps_skill_md_to_registry(
        self, db_session: Session
    ) -> None:
        service = SkillIngestionService(db_session)

        result = service.ingest_directory(
            FIXTURE_SKILL_DIR, owner_id="W1", **INGEST_KWARGS
        )

        assert result.created_new_version
        assert result.skill.slug == "demo-skill"
        assert result.skill.trust_level == "untrusted"
        assert result.skill.provenance == "user:U123"
        assert result.version.version == "1.2.0"
        assert result.version.name == "Demo Skill"
        assert "methodology" in result.version.instructions_md
        assert result.version.description.startswith("Use when the user asks")
        paths = {f.path: f for f in result.files}
        assert paths["references/notes.md"].kind == "reference"
        assert paths["references/notes.md"].content_text is not None
        assert paths["scripts/hello.py"].kind == "script"
        assert paths["references/diagram.png"].content_bytes is not None

    def test_reingest_same_content_is_noop(self, db_session: Session) -> None:
        service = SkillIngestionService(db_session)
        first = service.ingest_directory(
            FIXTURE_SKILL_DIR, owner_id="W1", **INGEST_KWARGS
        )
        second = service.ingest_directory(
            FIXTURE_SKILL_DIR, owner_id="W1", **INGEST_KWARGS
        )

        assert not second.created_new_version
        assert second.version.id == first.version.id

    def test_changed_content_bumps_version_and_deprecates_old(
        self, db_session: Session, tmp_path: Path
    ) -> None:
        service = SkillIngestionService(db_session)
        first = service.ingest_directory(
            FIXTURE_SKILL_DIR, owner_id="W1", **INGEST_KWARGS
        )

        edited = tmp_path / "demo-skill"
        shutil.copytree(FIXTURE_SKILL_DIR, edited)
        skill_md = edited / "SKILL.md"
        skill_md.write_text(skill_md.read_text() + "\n3. Double-check the numbers.\n")
        second = service.ingest_directory(edited, owner_id="W1", **INGEST_KWARGS)

        assert second.created_new_version
        assert second.version.version == "1.2.1"
        db_session.refresh(first.version)
        assert first.version.status == "deprecated"

    def test_ingest_zip_with_nested_root(self, db_session: Session) -> None:
        service = SkillIngestionService(db_session)
        data = make_zip(FIXTURE_SKILL_DIR, prefix="some-upload-name/")

        result = service.ingest_zip(data, owner_id="W1", **INGEST_KWARGS)

        assert result.skill.slug == "demo-skill"
        assert {f.path for f in result.files} >= {
            "references/notes.md",
            "scripts/hello.py",
        }

    def test_ingest_zip_rejects_path_traversal(self, db_session: Session) -> None:
        buffer = BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("../evil.md", "boom")
        service = SkillIngestionService(db_session)

        with pytest.raises(SkillIngestionError, match="Unsafe path"):
            service.ingest_zip(buffer.getvalue(), owner_id="W1", **INGEST_KWARGS)

    def test_ingest_markdown_with_frontmatter(self, db_session: Session) -> None:
        service = SkillIngestionService(db_session)
        content = (
            "---\n"
            "name: release-notes\n"
            "description: Use when drafting release notes from merged PRs.\n"
            "allowed-tools: web_search\n"
            "---\n\n## Steps\nSummarize the changes."
        )

        result = service.ingest_markdown(content, owner_id="W1", **INGEST_KWARGS)

        assert result.skill.slug == "release-notes"
        assert result.version.allowed_tools == ["web_search"]
        assert result.files == []

    def test_ingest_markdown_without_frontmatter_uses_fallbacks(
        self, db_session: Session
    ) -> None:
        service = SkillIngestionService(db_session)

        result = service.ingest_markdown(
            "## How to triage bugs\nAlways reproduce first.",
            owner_id="W1",
            fallback_name="Bug Triage!",
            fallback_description="Use when triaging incoming bug reports.",
            **INGEST_KWARGS,
        )

        assert result.skill.slug == "bug-triage"
        assert result.version.description == "Use when triaging incoming bug reports."

    def test_ingest_markdown_without_frontmatter_or_name_fails(
        self, db_session: Session
    ) -> None:
        service = SkillIngestionService(db_session)

        with pytest.raises(SkillIngestionError, match="name is required"):
            service.ingest_markdown("just some text", owner_id="W1", **INGEST_KWARGS)


class TestCuratedCatalog:
    def test_ensure_curated_skills_seeds_trusted_system_skills(
        self, db_session: Session
    ) -> None:
        service = SkillRegistryService(db_session)

        service.ensure_curated_skills()
        service.ensure_curated_skills()  # idempotent

        skills = {
            skill.slug: skill
            for skill in db_session.scalars(
                select(ProceduralSkill).where(ProceduralSkill.owner_type == "system")
            )
        }
        assert {
            "meeting-notes-summarizer",
            "competitive-analysis",
            "weekly-status-report",
        } <= set(skills)
        curated = skills["competitive-analysis"]
        assert curated.trust_level == "trusted"
        assert curated.provenance == "kortny"
        versions = list(
            db_session.scalars(
                select(ProceduralSkillVersion).where(
                    ProceduralSkillVersion.skill_id == curated.id
                )
            )
        )
        assert len(versions) == 1  # idempotent re-seed created no new version
        reference = db_session.scalar(
            select(SkillFile).where(SkillFile.skill_version_id == versions[0].id)
        )
        assert reference is not None
        assert reference.path == "references/dimensions.md"


class TestSkillEnablement:
    def test_scope_resolution_workspace_channel_user(self, db_session: Session) -> None:
        installation = create_installation(db_session)
        registry = SkillRegistryService(db_session)
        ws_skill, _ = create_skill(db_session, slug="ws-skill", owner_type="system")
        ch_skill, _ = create_skill(db_session, slug="ch-skill", owner_type="system")
        user_skill, _ = create_skill(db_session, slug="user-skill", owner_type="system")

        registry.enable_skill(
            installation_id=installation.id,
            skill_id=ws_skill.id,
            scope_type="workspace",
            scope_id=None,
            added_by="dashboard:tester",
        )
        registry.enable_skill(
            installation_id=installation.id,
            skill_id=ch_skill.id,
            scope_type="channel",
            scope_id="C999",
            added_by="dashboard:tester",
        )
        registry.enable_skill(
            installation_id=installation.id,
            skill_id=user_skill.id,
            scope_type="user",
            scope_id="U999",
            added_by="dashboard:tester",
        )

        task_other = create_task(
            db_session, installation, channel_id="C1", user_id="U1"
        )
        assert [s.slug for s in registry.enabled_skills_for_task(task_other)] == [
            "ws-skill"
        ]

        task_channel = create_task(
            db_session, installation, channel_id="C999", user_id="U1"
        )
        assert {s.slug for s in registry.enabled_skills_for_task(task_channel)} == {
            "ws-skill",
            "ch-skill",
        }

        task_user = create_task(
            db_session, installation, channel_id="C1", user_id="U999"
        )
        assert {s.slug for s in registry.enabled_skills_for_task(task_user)} == {
            "ws-skill",
            "user-skill",
        }

    def test_disabled_enablement_excluded_and_reenable(
        self, db_session: Session
    ) -> None:
        installation = create_installation(db_session)
        registry = SkillRegistryService(db_session)
        skill, _ = create_skill(db_session, slug="toggle-skill", owner_type="system")
        enablement = registry.enable_skill(
            installation_id=installation.id,
            skill_id=skill.id,
            scope_type="workspace",
            scope_id=None,
            added_by="dashboard:tester",
        )
        task = create_task(db_session, installation)
        assert len(registry.enabled_skills_for_task(task)) == 1

        registry.disable_skill(enablement_id=enablement.id, by="dashboard:tester")
        assert registry.enabled_skills_for_task(task) == []

        again = registry.enable_skill(
            installation_id=installation.id,
            skill_id=skill.id,
            scope_type="workspace",
            scope_id=None,
            added_by="dashboard:tester2",
        )
        assert again.id == enablement.id
        assert len(registry.enabled_skills_for_task(task)) == 1

    def test_most_specific_scope_wins_for_attribution(
        self, db_session: Session
    ) -> None:
        installation = create_installation(db_session)
        registry = SkillRegistryService(db_session)
        skill, _ = create_skill(db_session, slug="multi-scope", owner_type="system")
        for scope_type, scope_id in (("workspace", None), ("user", "U999")):
            registry.enable_skill(
                installation_id=installation.id,
                skill_id=skill.id,
                scope_type=scope_type,
                scope_id=scope_id,
                added_by="dashboard:tester",
            )
        task = create_task(db_session, installation, user_id="U999")

        enabled = registry.enabled_skills_for_task(task)

        assert len(enabled) == 1
        assert enabled[0].scope_type == "user"

    def test_invalid_scope_rejected(self, db_session: Session) -> None:
        installation = create_installation(db_session)
        registry = SkillRegistryService(db_session)
        skill, _ = create_skill(db_session, slug="bad-scope", owner_type="system")

        with pytest.raises(ValueError, match="requires a scope_id"):
            registry.enable_skill(
                installation_id=installation.id,
                skill_id=skill.id,
                scope_type="channel",
                scope_id=None,
                added_by="dashboard:tester",
            )
