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
from kortny.execution.sandbox_sessions import SandboxExecResult, SandboxSessionInfo
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


class TestSkillsContextBlock:
    def test_context_includes_l1_block_for_enabled_skills(
        self, db_session: Session
    ) -> None:
        from kortny.agent.context import ContextAssembler

        installation = create_installation(db_session)
        registry = SkillRegistryService(db_session)
        skill, _ = create_skill(db_session, slug="meeting-recap", owner_type="system")
        registry.enable_skill(
            installation_id=installation.id,
            skill_id=skill.id,
            scope_type="workspace",
            scope_id=None,
            added_by="dashboard:tester",
        )
        task = create_task(db_session, installation)

        package = ContextAssembler(session=db_session).build_for_task(task)

        skills_blocks = [
            message.content
            for message in package.messages
            if message.role == "system"
            and message.content
            and "<available_skills>" in message.content
        ]
        assert len(skills_blocks) == 1
        assert "- meeting-recap [workspace]:" in skills_blocks[0]
        assert "load_skill" in skills_blocks[0]
        assert [s.slug for s in package.selected_skills] == ["meeting-recap"]

    def test_context_has_no_skills_block_when_none_enabled(
        self, db_session: Session
    ) -> None:
        from kortny.agent.context import ContextAssembler

        task = create_task(db_session)

        package = ContextAssembler(session=db_session).build_for_task(task)

        assert package.selected_skills == ()
        assert not any(
            message.content and "<available_skills>" in message.content
            for message in package.messages
        )

    def test_skills_block_bounded_by_char_budget(self, db_session: Session) -> None:
        from kortny.agent.context import (
            DEFAULT_SKILLS_CONTEXT_MAX_CHARS,
            ContextAssembler,
        )

        installation = create_installation(db_session)
        registry = SkillRegistryService(db_session)
        for index in range(40):
            skill = ProceduralSkill(
                slug=f"bulk-skill-{index:02d}",
                owner_type="system",
                owner_id=None,
                status="active",
                trust_level="trusted",
                visibility="catalog",
                provenance="kortny",
            )
            db_session.add(skill)
            db_session.flush()
            db_session.add(
                ProceduralSkillVersion(
                    skill_id=skill.id,
                    version="1.0.0",
                    status="active",
                    name=f"Bulk {index}",
                    description="Use when the task involves " + "x" * 200,
                    instructions_md="## Steps",
                    content_sha256="0" * 64,
                    created_by="test",
                )
            )
            db_session.flush()
            registry.enable_skill(
                installation_id=installation.id,
                skill_id=skill.id,
                scope_type="workspace",
                scope_id=None,
                added_by="dashboard:tester",
            )
        task = create_task(db_session, installation)

        package = ContextAssembler(session=db_session).build_for_task(task)

        block = next(
            message.content
            for message in package.messages
            if message.content and "<available_skills>" in message.content
        )
        assert len(block) <= DEFAULT_SKILLS_CONTEXT_MAX_CHARS + 50
        assert any(o.kind == "skills" for o in package.omissions)
        assert 0 < len(package.selected_skills) < 40


class TestSkillTools:
    def _setup(self, db_session: Session) -> tuple[Task, ProceduralSkill]:
        installation = create_installation(db_session)
        ingestion = SkillIngestionService(db_session)
        result = ingestion.ingest_directory(
            FIXTURE_SKILL_DIR,
            owner_type="system",
            owner_id=None,
            provenance="kortny",
            trust_level="trusted",
            created_by="system",
        )
        SkillRegistryService(db_session).enable_skill(
            installation_id=installation.id,
            skill_id=result.skill.id,
            scope_type="workspace",
            scope_id=None,
            added_by="dashboard:tester",
        )
        task = create_task(db_session, installation)
        return task, result.skill

    def test_load_skill_returns_instructions_and_records_invocation(
        self, db_session: Session
    ) -> None:
        from kortny.tasks import TaskService as TS
        from kortny.tools.skills import LoadSkillTool

        task, skill = self._setup(db_session)
        tool = LoadSkillTool(session=db_session, task=task, task_service=TS(db_session))

        result = tool.invoke({"slug": "demo-skill"})

        assert result.output["slug"] == "demo-skill"
        assert "methodology" in result.output["instructions_md"]
        assert "references/notes.md" in result.output["resources"]
        assert "scripts_note" in result.output
        invocation = db_session.scalar(
            select(ProceduralSkillInvocation).where(
                ProceduralSkillInvocation.task_id == task.id
            )
        )
        assert invocation is not None
        assert invocation.invocation_kind == "execution"
        assert invocation.skill_id == skill.id

    def test_load_skill_rejects_unenabled_slug(self, db_session: Session) -> None:
        from kortny.tasks import TaskService as TS
        from kortny.tools.skills import LoadSkillTool
        from kortny.tools.types import RecoverableToolError

        task, _ = self._setup(db_session)
        tool = LoadSkillTool(session=db_session, task=task, task_service=TS(db_session))

        with pytest.raises(RecoverableToolError) as exc_info:
            tool.invoke({"slug": "nonexistent-skill"})
        assert exc_info.value.code == "skill_not_enabled"
        assert "demo-skill" in (exc_info.value.hint or "")

    def test_load_skill_resource_returns_text_and_rejects_binary(
        self, db_session: Session
    ) -> None:
        from kortny.tasks import TaskService as TS
        from kortny.tools.skills import LoadSkillResourceTool
        from kortny.tools.types import RecoverableToolError

        task, _ = self._setup(db_session)
        tool = LoadSkillResourceTool(
            session=db_session, task=task, task_service=TS(db_session)
        )

        result = tool.invoke({"slug": "demo-skill", "path": "references/notes.md"})
        assert "cite the data source" in result.output["content"]
        assert result.output["kind"] == "reference"

        with pytest.raises(RecoverableToolError) as exc_info:
            tool.invoke({"slug": "demo-skill", "path": "references/diagram.png"})
        assert exc_info.value.code == "skill_resource_binary"

        with pytest.raises(RecoverableToolError) as exc_info:
            tool.invoke({"slug": "demo-skill", "path": "references/missing.md"})
        assert exc_info.value.code == "skill_resource_not_found"

    def test_native_factories_gate_on_enabled_skills(self, db_session: Session) -> None:
        from kortny.tools.native_runtime import (
            _build_load_skill_resource_tool,
            _build_load_skill_tool,
        )

        task_with, _ = self._setup(db_session)
        task_without = create_task(db_session)

        from unittest.mock import MagicMock

        from kortny.tools.native_runtime import NativeToolBuildContext

        def make_context(task: Task) -> NativeToolBuildContext:
            return NativeToolBuildContext(
                settings=MagicMock(),
                session=db_session,
                task=task,
                task_service=TaskService(db_session),
                working_dir=Path("/tmp"),
                web_search_tool=None,
                slack_history_client=None,
                slack_file_client=None,
                slack_identity_client=None,
                slack_action_client=None,
                memory_service=MagicMock(),
            )

        assert _build_load_skill_tool(make_context(task_with)) is not None
        assert _build_load_skill_resource_tool(make_context(task_with)) is not None
        assert _build_load_skill_tool(make_context(task_without)) is None
        assert _build_load_skill_resource_tool(make_context(task_without)) is None


class FakeScriptSessionClient:
    """Records write_file/exec calls for run_skill_script tests."""

    def __init__(self) -> None:
        self.writes: list[tuple[str, bytes]] = []
        self.execs: list[tuple[str, str, int]] = []
        self.exec_result = SandboxExecResult(
            exit_code=0, stdout="hello from demo skill\n", stderr=""
        )

    def open_session(
        self, task_id: str, profile: str = "workbench"
    ) -> SandboxSessionInfo:
        return SandboxSessionInfo(
            session_id="s-1",
            task_id=task_id,
            container_id="c-1",
            profile=profile,
            reused=False,
        )

    def exec(
        self,
        session_id: str,
        command: str,
        *,
        workdir: str = "/workspace",
        timeout_seconds: int = 120,
    ) -> SandboxExecResult:
        self.execs.append((command, workdir, timeout_seconds))
        return self.exec_result

    def write_file(self, session_id: str, path: str, content: bytes) -> int:
        self.writes.append((path, content))
        return len(content)

    def read_file(self, session_id: str, path: str) -> bytes:  # pragma: no cover
        raise NotImplementedError

    def export_archive(self, session_id: str, path: str) -> bytes:  # pragma: no cover
        raise NotImplementedError

    def close_session(self, session_id: str) -> None:  # pragma: no cover
        return None


class TestRunSkillScript:
    def _setup(
        self, db_session: Session, *, trust_level: str
    ) -> tuple[Task, ProceduralSkill]:
        installation = create_installation(db_session)
        ingestion = SkillIngestionService(db_session)
        result = ingestion.ingest_directory(
            FIXTURE_SKILL_DIR,
            owner_type="system",
            owner_id=None,
            provenance="kortny",
            trust_level=trust_level,
            created_by="system",
        )
        SkillRegistryService(db_session).enable_skill(
            installation_id=installation.id,
            skill_id=result.skill.id,
            scope_type="workspace",
            scope_id=None,
            added_by="dashboard:tester",
        )
        task = create_task(db_session, installation)
        return task, result.skill

    def _tool(self, db_session: Session, task: Task, client: object):  # type: ignore[no-untyped-def]
        from kortny.tools.sandbox_workbench import WorkbenchSession
        from kortny.tools.skills import RunSkillScriptTool

        task_service = TaskService(db_session)
        workbench = WorkbenchSession(
            client=client,  # type: ignore[arg-type]
            task=task,
            task_service=task_service,
        )
        return RunSkillScriptTool(
            session=db_session,
            task=task,
            task_service=task_service,
            workbench=workbench,
        )

    def test_untrusted_skill_is_blocked_by_trust_gate(
        self, db_session: Session
    ) -> None:
        from kortny.tools.types import RecoverableToolError

        task, _ = self._setup(db_session, trust_level="untrusted")
        client = FakeScriptSessionClient()
        tool = self._tool(db_session, task, client)

        with pytest.raises(RecoverableToolError) as exc_info:
            tool.invoke({"slug": "demo-skill", "path": "scripts/hello.py"})

        assert exc_info.value.code == "skill_scripts_blocked_by_trust"
        assert "trusted" in (exc_info.value.hint or "")
        assert client.execs == []
        assert client.writes == []
        blocked = [
            event
            for event in db_session.scalars(
                select(TaskEvent).where(TaskEvent.task_id == task.id)
            )
            if event.payload.get("message") == "skill_script_blocked"
        ]
        assert len(blocked) == 1
        assert blocked[0].payload["trust_level"] == "untrusted"
        assert blocked[0].payload["path"] == "scripts/hello.py"

    def test_trusted_skill_materializes_and_executes(self, db_session: Session) -> None:
        task, skill = self._setup(db_session, trust_level="trusted")
        client = FakeScriptSessionClient()
        tool = self._tool(db_session, task, client)

        result = tool.invoke(
            {"slug": "demo-skill", "path": "scripts/hello.py", "args": ["--verbose"]}
        )

        assert result.output["successful"] is True
        assert result.output["exit_code"] == 0
        assert result.output["slug"] == "demo-skill"
        assert result.output["path"] == "scripts/hello.py"

        written_paths = {path for path, _ in client.writes}
        assert "/workspace/skills/demo-skill/scripts/hello.py" in written_paths
        assert "/workspace/skills/demo-skill/references/notes.md" in written_paths
        assert "/workspace/skills/demo-skill/references/diagram.png" in written_paths
        assert "/workspace/skills/demo-skill/SKILL.md" in written_paths
        diagram = next(
            content for path, content in client.writes if path.endswith("diagram.png")
        )
        assert isinstance(diagram, bytes)

        assert len(client.execs) == 1
        command, workdir, timeout = client.execs[0]
        assert command == "python scripts/hello.py --verbose"
        assert workdir == "/workspace/skills/demo-skill"
        assert timeout == 300

        invocation = db_session.scalar(
            select(ProceduralSkillInvocation).where(
                ProceduralSkillInvocation.task_id == task.id,
                ProceduralSkillInvocation.invocation_kind == "script_execution",
            )
        )
        assert invocation is not None
        assert invocation.skill_id == skill.id

    def test_bare_script_name_is_normalized(self, db_session: Session) -> None:
        task, _ = self._setup(db_session, trust_level="trusted")
        client = FakeScriptSessionClient()
        tool = self._tool(db_session, task, client)

        result = tool.invoke({"slug": "demo-skill", "path": "hello.py"})

        assert result.output["path"] == "scripts/hello.py"
        assert client.execs[0][0] == "python scripts/hello.py"

    def test_unknown_script_errors_with_available_scripts(
        self, db_session: Session
    ) -> None:
        from kortny.tools.types import RecoverableToolError

        task, _ = self._setup(db_session, trust_level="trusted")
        client = FakeScriptSessionClient()
        tool = self._tool(db_session, task, client)

        with pytest.raises(RecoverableToolError) as exc_info:
            tool.invoke({"slug": "demo-skill", "path": "scripts/missing.py"})

        assert exc_info.value.code == "skill_script_not_found"
        assert "scripts/hello.py" in (exc_info.value.hint or "")
        assert client.execs == []

    def test_unsupported_extension_errors(self, db_session: Session) -> None:
        from kortny.tools.types import RecoverableToolError

        task, _ = self._setup(db_session, trust_level="trusted")
        client = FakeScriptSessionClient()
        tool = self._tool(db_session, task, client)

        with pytest.raises(RecoverableToolError) as exc_info:
            tool.invoke({"slug": "demo-skill", "path": "references/notes.md"})

        assert exc_info.value.code == "skill_script_unsupported"
        assert client.execs == []

    def test_factory_gates_on_scripts_and_workbench(self, db_session: Session) -> None:
        from unittest.mock import MagicMock

        from kortny.tools.native_runtime import (
            NativeToolBuildContext,
            _build_run_skill_script_tool,
        )

        task_with_scripts, _ = self._setup(db_session, trust_level="trusted")

        no_scripts_installation = create_installation(db_session)
        no_scripts_skill, _ = create_skill(
            db_session, slug="no-scripts", owner_type="system"
        )
        SkillRegistryService(db_session).enable_skill(
            installation_id=no_scripts_installation.id,
            skill_id=no_scripts_skill.id,
            scope_type="workspace",
            scope_id=None,
            added_by="dashboard:tester",
        )
        task_no_scripts = create_task(db_session, no_scripts_installation)

        def make_context(
            task: Task, *, runner_url: str | None
        ) -> NativeToolBuildContext:
            settings = MagicMock()
            settings.sandbox_runner_url = runner_url
            settings.sandbox_runner_timeout_seconds = 70.0
            return NativeToolBuildContext(
                settings=settings,
                session=db_session,
                task=task,
                task_service=TaskService(db_session),
                working_dir=Path("/tmp"),
                web_search_tool=None,
                slack_history_client=None,
                slack_file_client=None,
                slack_identity_client=None,
                slack_action_client=None,
                memory_service=MagicMock(),
            )

        assert (
            _build_run_skill_script_tool(
                make_context(task_with_scripts, runner_url="http://runner")
            )
            is not None
        )
        assert (
            _build_run_skill_script_tool(
                make_context(task_no_scripts, runner_url="http://runner")
            )
            is None
        )
        assert (
            _build_run_skill_script_tool(
                make_context(task_with_scripts, runner_url=None)
            )
            is None
        )
