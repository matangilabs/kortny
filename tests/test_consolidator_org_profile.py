"""DB-backed tests for the org-profile consolidator pass (HIG-271)."""

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

from kortny.consolidator.org_profile import (
    ORG_PROFILE_FACT_KEY,
    OrgProfilePass,
)
from kortny.db.models import (
    Installation,
    LLMUsage,
    ObserveChannelProfile,
    Task,
    TaskEvent,
    TaskEventType,
    WorkspaceState,
)
from kortny.db.session import make_engine, make_session_factory, normalize_database_url
from kortny.llm import ChatMessage, Completion, TokenUsage
from kortny.memory.service import PENDING_PROPOSAL_MESSAGE
from kortny.tasks import TaskService
from kortny.tools.types import JsonObject, JsonSchema

TEST_POSTGRES_URL = os.environ.get("KORTNY_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    TEST_POSTGRES_URL is None,
    reason="KORTNY_TEST_POSTGRES_URL is required for org-profile tests",
)

NOW = datetime(2026, 6, 16, 12, 0, 0, tzinfo=UTC)


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
        _cleanup(session)
        session.commit()
        yield session
        session.rollback()
        _cleanup(session)
        session.commit()


def _cleanup(session: Session) -> None:
    for model in (
        WorkspaceState,
        ObserveChannelProfile,
        LLMUsage,
        TaskEvent,
        Task,
        Installation,
    ):
        session.execute(delete(model))


class FakeLLM:
    """Minimal stand-in for LLMService.complete used by the pass."""

    def __init__(self, completion: Completion) -> None:
        self.completion = completion
        self.calls = 0

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
        del task_id, messages, tools, response_format, prompt_name, prompt_source
        self.calls += 1
        return self.completion


class FakePoster:
    """Records confirmation prompts; mimics ConfirmationPoster.post_message."""

    def __init__(self) -> None:
        self.posts: list[tuple[str, str]] = []

    def post_message(
        self, thread: object, text: str, *, purpose: str = "result"
    ) -> str:
        channel = getattr(thread, "channel_id", "?")
        self.posts.append((channel, text))
        return "1781000000.000001"


def _completion(payload: dict) -> Completion:
    return Completion(
        content=json.dumps(payload),
        tool_calls=(),
        usage=TokenUsage(input_tokens=100, output_tokens=30),
        cost_usd=Decimal("0.0001"),
        model="openai/test",
    )


def _installation(session: Session, *, admin: str | None = "U_ADMIN") -> Installation:
    installation = Installation(
        slack_team_id=f"T{uuid.uuid4().hex}",
        team_name="Acme Robotics",
        primary_admin_user_id=admin,
    )
    session.add(installation)
    session.flush()
    return installation


def _profiles(session: Session, installation: Installation, count: int) -> None:
    for i in range(count):
        session.add(
            ObserveChannelProfile(
                installation_id=installation.id,
                channel_id=f"C_{i}",
                profile_status="active",
                summary=f"Channel {i} discusses warehouse robotics deployments.",
            )
        )
    session.flush()


def _seed_task(session: Session, installation: Installation) -> Task:
    return TaskService(session).create_task(
        installation_id=installation.id,
        slack_event_id=f"Ev{uuid.uuid4().hex}",
        slack_channel_id="consolidator",
        slack_user_id="consolidator",
        input="consolidation run",
    )


def _make_pass(
    session: Session,
    *,
    completion: Completion,
    poster: FakePoster | None,
) -> tuple[OrgProfilePass, FakeLLM]:
    llm = FakeLLM(completion)
    pass_ = OrgProfilePass(
        session,
        llm=llm,  # type: ignore[arg-type]
        poster=poster,
        dm_channel_for_user=(lambda _user: "D_ADMIN"),
    )
    return pass_, llm


def test_proposes_org_profile_when_evidence_sufficient(db_session: Session) -> None:
    installation = _installation(db_session)
    _profiles(db_session, installation, 3)
    task = _seed_task(db_session, installation)
    poster = FakePoster()
    pass_, llm = _make_pass(
        db_session,
        completion=_completion(
            {
                "company_name": "Acme Robotics",
                "what_we_do": "Autonomous warehouse robots.",
                "competitors": ["Locus", "6 River"],
                "confidence": 0.85,
            }
        ),
        poster=poster,
    )

    counters = pass_.run(installation_id=installation.id, task=task, now=NOW)

    assert counters.proposed == 1
    assert llm.calls == 1
    # The confirmation prompt was posted to the admin DM.
    assert poster.posts and poster.posts[0][0] == "D_ADMIN"
    # A pending proposal audit event exists for the workspace org_profile key.
    proposals = [
        event
        for event in db_session.scalars(select(TaskEvent))
        if event.type is TaskEventType.log
        and event.payload.get("message") == PENDING_PROPOSAL_MESSAGE
        and event.payload.get("key") == ORG_PROFILE_FACT_KEY
    ]
    assert len(proposals) == 1
    assert proposals[0].payload.get("scope_type") == "workspace"


def test_skips_when_insufficient_channel_profiles(db_session: Session) -> None:
    installation = _installation(db_session)
    _profiles(db_session, installation, 2)
    task = _seed_task(db_session, installation)
    poster = FakePoster()
    pass_, llm = _make_pass(
        db_session,
        completion=_completion({"company_name": "X", "confidence": 0.9}),
        poster=poster,
    )

    counters = pass_.run(installation_id=installation.id, task=task, now=NOW)

    assert counters.proposed == 0
    assert counters.skipped_reason == "insufficient_evidence"
    assert llm.calls == 0  # never reached the model
    assert poster.posts == []


def test_skips_when_no_primary_admin(db_session: Session) -> None:
    installation = _installation(db_session, admin=None)
    _profiles(db_session, installation, 5)
    task = _seed_task(db_session, installation)
    poster = FakePoster()
    pass_, _ = _make_pass(
        db_session,
        completion=_completion({"company_name": "X", "confidence": 0.9}),
        poster=poster,
    )

    counters = pass_.run(installation_id=installation.id, task=task, now=NOW)

    assert counters.skipped_reason == "no_primary_admin"
    assert poster.posts == []


def test_skips_low_confidence(db_session: Session) -> None:
    installation = _installation(db_session)
    _profiles(db_session, installation, 4)
    task = _seed_task(db_session, installation)
    poster = FakePoster()
    pass_, _ = _make_pass(
        db_session,
        completion=_completion({"company_name": "Maybe", "confidence": 0.2}),
        poster=poster,
    )

    counters = pass_.run(installation_id=installation.id, task=task, now=NOW)

    assert counters.proposed == 0
    assert counters.skipped_reason == "low_confidence"
    assert poster.posts == []


def test_dedup_skips_when_recent_proposal(db_session: Session) -> None:
    installation = _installation(db_session)
    _profiles(db_session, installation, 3)
    task = _seed_task(db_session, installation)
    poster = FakePoster()
    pass_, _ = _make_pass(
        db_session,
        completion=_completion({"company_name": "Acme", "confidence": 0.9}),
        poster=poster,
    )

    first = pass_.run(installation_id=installation.id, task=task, now=NOW)
    # A day later, before the cooldown lapses — must not re-propose.
    second = pass_.run(
        installation_id=installation.id,
        task=task,
        now=NOW + timedelta(days=1),
    )

    assert first.proposed == 1
    assert second.proposed == 0
    assert second.skipped_reason == "recent_proposal"
    assert len(poster.posts) == 1


def test_skips_gracefully_without_poster(db_session: Session) -> None:
    installation = _installation(db_session)
    _profiles(db_session, installation, 3)
    task = _seed_task(db_session, installation)
    pass_, _ = _make_pass(
        db_session,
        completion=_completion({"company_name": "Acme", "confidence": 0.9}),
        poster=None,
    )

    counters = pass_.run(installation_id=installation.id, task=task, now=NOW)

    assert counters.skipped_reason == "slack_or_llm_unavailable"
