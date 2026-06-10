"""Dashboard integration tests for the skills directory pages."""

from __future__ import annotations

import os
import uuid
import zipfile
from collections.abc import Iterator
from io import BytesIO
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import Engine, delete, select
from sqlalchemy.orm import Session

from kortny.dashboard.app import create_app
from kortny.dashboard.settings import DashboardSettings
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

TEST_POSTGRES_URL = os.environ.get("KORTNY_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    TEST_POSTGRES_URL is None,
    reason="KORTNY_TEST_POSTGRES_URL is required for dashboard skills tests",
)

FIXTURE_SKILL_DIR = Path(__file__).parent / "fixtures" / "skills" / "demo-skill"


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


@pytest.fixture
def client(db_session: Session, engine: Engine) -> Iterator[tuple[TestClient, Session]]:
    assert TEST_POSTGRES_URL is not None
    session_factory = make_session_factory(engine=engine)
    settings = DashboardSettings(
        postgres_url=TEST_POSTGRES_URL,
        username="admin",
        password="secret",
        session_secret="test-dashboard-session-secret",
    )
    with TestClient(
        create_app(settings=settings, session_factory=session_factory)
    ) as test_client:
        login = test_client.post(
            "/login",
            data={"username": "admin", "password": "secret", "next": "/"},
            follow_redirects=False,
        )
        assert login.status_code == 303
        yield test_client, db_session


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


def create_installation(session: Session) -> Installation:
    installation = Installation(slack_team_id=f"T{uuid.uuid4().hex}")
    session.add(installation)
    session.commit()
    return installation


def fixture_zip() -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for file_path in sorted(FIXTURE_SKILL_DIR.rglob("*")):
            if file_path.is_file():
                archive.write(file_path, str(file_path.relative_to(FIXTURE_SKILL_DIR)))
    return buffer.getvalue()


def test_skills_page_renders_curated_catalog(
    client: tuple[TestClient, Session],
) -> None:
    test_client, session = client
    create_installation(session)

    response = test_client.get("/skills")

    assert response.status_code == 200
    assert "Curated skills" in response.text
    assert "Meeting Notes Summarizer" in response.text
    assert "Competitive Analysis" in response.text
    # Builtin humanizer skills are not part of the directory.
    assert "Slack Humanizer" not in response.text


def test_enable_then_disable_skill_scope(
    client: tuple[TestClient, Session],
) -> None:
    test_client, session = client
    create_installation(session)
    test_client.get("/skills")  # seeds curated catalog
    skill_id = session.scalar(
        select(ProceduralSkill.id).where(ProceduralSkill.slug == "weekly-status-report")
    )
    assert skill_id is not None

    enable = test_client.post(
        f"/skills/{skill_id}/enable",
        data={"scope_type": "channel", "scope_id": "C777", "next": "/skills"},
        follow_redirects=False,
    )
    assert enable.status_code == 303
    assert "Skill+enabled" in enable.headers["location"]

    page = test_client.get("/skills")
    assert "channel &middot; C777" in page.text or "channel · C777" in page.text

    enablement_id = session.scalar(
        select(SkillEnablement.id).where(SkillEnablement.skill_id == skill_id)
    )
    disable = test_client.post(
        f"/skills/enablements/{enablement_id}/disable",
        data={"next": "/skills"},
        follow_redirects=False,
    )
    assert disable.status_code == 303

    page = test_client.get("/skills")
    assert "C777" not in page.text


def test_upload_skill_zip_creates_and_enables(
    client: tuple[TestClient, Session],
) -> None:
    test_client, session = client
    create_installation(session)

    response = test_client.post(
        "/skills/upload",
        data={"scope_type": "workspace", "scope_id": "", "next": "/skills"},
        files={"skill_file": ("demo-skill.zip", fixture_zip(), "application/zip")},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert "uploaded+and+enabled" in response.headers["location"]
    skill = session.scalar(
        select(ProceduralSkill).where(ProceduralSkill.slug == "demo-skill")
    )
    assert skill is not None
    assert skill.owner_type == "workspace"
    assert skill.trust_level == "untrusted"
    page = test_client.get("/skills")
    assert "Demo Skill" in page.text


def test_paste_skill_markdown_creates_and_enables(
    client: tuple[TestClient, Session],
) -> None:
    test_client, session = client
    create_installation(session)

    response = test_client.post(
        "/skills/paste",
        data={
            "name": "Bug Triage",
            "description": "Use when triaging incoming bug reports.",
            "content": "## How to triage\nAlways reproduce first.",
            "scope_type": "user",
            "scope_id": "U42",
            "next": "/skills",
        },
        follow_redirects=False,
    )

    assert response.status_code == 303
    skill = session.scalar(
        select(ProceduralSkill).where(ProceduralSkill.slug == "bug-triage")
    )
    assert skill is not None
    enablement = session.scalar(
        select(SkillEnablement).where(SkillEnablement.skill_id == skill.id)
    )
    assert enablement is not None
    assert enablement.scope_type == "user"
    assert enablement.scope_id == "U42"


def test_skill_detail_page_and_trust_change(
    client: tuple[TestClient, Session],
) -> None:
    test_client, session = client
    create_installation(session)
    test_client.post(
        "/skills/upload",
        data={"scope_type": "workspace", "scope_id": "", "next": "/skills"},
        files={"skill_file": ("demo-skill.zip", fixture_zip(), "application/zip")},
        follow_redirects=False,
    )
    skill = session.scalar(
        select(ProceduralSkill).where(ProceduralSkill.slug == "demo-skill")
    )
    assert skill is not None

    detail = test_client.get(f"/skills/{skill.id}")
    assert detail.status_code == 200
    assert "references/notes.md" in detail.text
    assert "Instructions" in detail.text

    promote = test_client.post(
        f"/skills/{skill.id}/trust",
        data={"trust_level": "trusted", "next": f"/skills/{skill.id}"},
        follow_redirects=False,
    )
    assert promote.status_code == 303
    session.expire_all()
    refreshed = session.get(ProceduralSkill, skill.id)
    assert refreshed is not None
    assert refreshed.trust_level == "trusted"


def test_curated_skill_trust_change_rejected(
    client: tuple[TestClient, Session],
) -> None:
    test_client, session = client
    create_installation(session)
    test_client.get("/skills")
    skill_id = session.scalar(
        select(ProceduralSkill.id).where(ProceduralSkill.slug == "competitive-analysis")
    )

    response = test_client.post(
        f"/skills/{skill_id}/trust",
        data={"trust_level": "untrusted", "next": "/skills"},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert "notice_tone=danger" in response.headers["location"]


def test_invalid_upload_surfaces_error_notice(
    client: tuple[TestClient, Session],
) -> None:
    test_client, session = client
    create_installation(session)

    response = test_client.post(
        "/skills/upload",
        data={"scope_type": "workspace", "scope_id": "", "next": "/skills"},
        files={"skill_file": ("broken.zip", b"not a zip", "application/zip")},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert "notice_tone=danger" in response.headers["location"]


def test_script_execution_capability_note(
    client: tuple[TestClient, Session],
) -> None:
    """Detail page shows blocked note for untrusted skill with scripts, enabled note after promotion."""
    test_client, session = client
    create_installation(session)

    # Upload demo-skill which bundles scripts/hello.py; custom uploads default to untrusted.
    test_client.post(
        "/skills/upload",
        data={"scope_type": "workspace", "scope_id": "", "next": "/skills"},
        files={"skill_file": ("demo-skill.zip", fixture_zip(), "application/zip")},
        follow_redirects=False,
    )
    skill = session.scalar(
        select(ProceduralSkill).where(ProceduralSkill.slug == "demo-skill")
    )
    assert skill is not None
    assert skill.trust_level == "untrusted"

    # Untrusted: blocked note should appear.
    detail = test_client.get(f"/skills/{skill.id}")
    assert detail.status_code == 200
    assert "skill-script-note-blocked" in detail.text
    assert "blocked until this skill is promoted to trusted" in detail.text
    assert "skill-script-note-enabled" not in detail.text

    # Promote to trusted.
    promote = test_client.post(
        f"/skills/{skill.id}/trust",
        data={"trust_level": "trusted", "next": f"/skills/{skill.id}"},
        follow_redirects=False,
    )
    assert promote.status_code == 303
    session.expire_all()

    # Trusted: sandbox-enabled note should appear.
    detail_trusted = test_client.get(f"/skills/{skill.id}")
    assert detail_trusted.status_code == 200
    assert "skill-script-note-enabled" in detail_trusted.text
    assert "Scripts run inside the task sandbox" in detail_trusted.text
    assert "skill-script-note-blocked" not in detail_trusted.text
