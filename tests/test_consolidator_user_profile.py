"""DB-backed tests for the user-profile consolidator pass (HIG-277)."""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from decimal import Decimal

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, delete, select
from sqlalchemy.orm import Session

from kortny.consolidator.user_profile import (
    USER_PROFILE_FACT_KEY,
    UserProfilePass,
)
from kortny.db.models import (
    Installation,
    LLMUsage,
    ObservationEvent,
    Task,
    TaskEvent,
    TaskEventType,
    WorkspaceState,
)
from kortny.db.session import make_engine, make_session_factory, normalize_database_url
from kortny.llm import ChatMessage, Completion, TokenUsage
from kortny.memory.service import PENDING_PROPOSAL_MESSAGE, WorkspaceStateService
from kortny.tasks import TaskService
from kortny.tools.types import JsonObject, JsonSchema

TEST_POSTGRES_URL = os.environ.get("KORTNY_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    TEST_POSTGRES_URL is None,
    reason="KORTNY_TEST_POSTGRES_URL is required for user-profile tests",
)

NOW = datetime(2026, 6, 18, 12, 0, 0, tzinfo=UTC)
USER = "U_DEV"


@pytest.fixture(scope="session")
def engine() -> Iterator[Engine]:
    assert TEST_POSTGRES_URL is not None
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", normalize_database_url(TEST_POSTGRES_URL))
    command.upgrade(config, "heads")
    eng = make_engine(TEST_POSTGRES_URL)
    try:
        yield eng
    finally:
        eng.dispose()


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
        ObservationEvent,
        LLMUsage,
        TaskEvent,
        Task,
        Installation,
    ):
        session.execute(delete(model))


class FakeLLM:
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
    def __init__(self) -> None:
        self.posts: list[tuple[str, str]] = []

    def post_message(
        self, thread: object, text: str, *, purpose: str = "result"
    ) -> str:
        self.posts.append((getattr(thread, "channel_id", "?"), text))
        return "1781000000.000001"


def _completion(payload: dict) -> Completion:
    return Completion(
        content=json.dumps(payload),
        tool_calls=(),
        usage=TokenUsage(input_tokens=100, output_tokens=30),
        cost_usd=Decimal("0.0001"),
        model="openai/test",
    )


def _installation(session: Session) -> Installation:
    installation = Installation(slack_team_id=f"T{uuid.uuid4().hex}", team_name="Acme")
    session.add(installation)
    session.flush()
    return installation


def _observe(session: Session, installation: Installation, *, count: int) -> None:
    for i in range(count):
        session.add(
            ObservationEvent(
                installation_id=installation.id,
                slack_team_id=installation.slack_team_id,
                channel_id=f"C_{i % 3}",
                user_id=USER,
                event_type="message",
                slack_event_id=f"Ev{uuid.uuid4().hex}",
                raw_payload_checksum=uuid.uuid4().hex,
                observed_at=NOW,
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
    session: Session, *, completion: Completion, poster: FakePoster | None
) -> tuple[UserProfilePass, FakeLLM]:
    llm = FakeLLM(completion)
    pass_ = UserProfilePass(
        session,
        llm=llm,  # type: ignore[arg-type]
        poster=poster,
        dm_channel_for_user=(lambda _user: "D_DEV"),
        slack_title_for_user=(lambda _user: "Software Engineer"),
    )
    return pass_, llm


def test_proposes_user_profile_when_evidence_sufficient(db_session: Session) -> None:
    installation = _installation(db_session)
    _observe(db_session, installation, count=6)
    task = _seed_task(db_session, installation)
    poster = FakePoster()
    pass_, llm = _make_pass(
        db_session,
        completion=_completion(
            {
                "role": "Software Engineer",
                "work_surfaces": ["issues", "prs"],
                "confidence": 0.8,
            }
        ),
        poster=poster,
    )

    counters = pass_.run(
        installation_id=installation.id, user_id=USER, task=task, now=NOW
    )

    assert counters.proposed == 1
    assert llm.calls == 1
    # The confirmation prompt was posted to the USER's DM (not an admin).
    assert poster.posts and poster.posts[0][0] == "D_DEV"
    proposals = [
        e
        for e in db_session.scalars(select(TaskEvent))
        if e.type is TaskEventType.log
        and e.payload.get("message") == PENDING_PROPOSAL_MESSAGE
        and e.payload.get("key") == USER_PROFILE_FACT_KEY
    ]
    assert len(proposals) == 1
    assert proposals[0].payload.get("scope_type") == "user"
    assert proposals[0].payload.get("scope_id") == USER


def test_skips_when_insufficient_observation(db_session: Session) -> None:
    installation = _installation(db_session)
    _observe(db_session, installation, count=2)
    task = _seed_task(db_session, installation)
    poster = FakePoster()
    pass_, llm = _make_pass(
        db_session,
        completion=_completion({"role": "X", "confidence": 0.9}),
        poster=poster,
    )

    counters = pass_.run(
        installation_id=installation.id, user_id=USER, task=task, now=NOW
    )

    assert counters.proposed == 0
    assert counters.skipped_reason == "insufficient_evidence"
    assert llm.calls == 0
    assert poster.posts == []


def test_skips_when_profile_exists(db_session: Session) -> None:
    installation = _installation(db_session)
    _observe(db_session, installation, count=6)
    db_session.add(
        WorkspaceState(
            installation_id=installation.id,
            scope_type="user",
            scope_id=USER,
            key=USER_PROFILE_FACT_KEY,
            value_json={"role": "Software Engineer"},
            value_text="Role: Software Engineer",
            status="active",
            source_kind="observer_proposed",
            proposed_by=USER,
        )
    )
    db_session.flush()
    task = _seed_task(db_session, installation)
    poster = FakePoster()
    pass_, llm = _make_pass(
        db_session,
        completion=_completion({"role": "X", "confidence": 0.9}),
        poster=poster,
    )

    counters = pass_.run(
        installation_id=installation.id, user_id=USER, task=task, now=NOW
    )

    assert counters.skipped_reason == "profile_exists"
    assert llm.calls == 0


def test_low_confidence_not_proposed(db_session: Session) -> None:
    installation = _installation(db_session)
    _observe(db_session, installation, count=6)
    task = _seed_task(db_session, installation)
    poster = FakePoster()
    pass_, _ = _make_pass(
        db_session,
        completion=_completion(
            {"role": "Maybe Eng", "work_surfaces": ["issues"], "confidence": 0.4}
        ),
        poster=poster,
    )

    counters = pass_.run(
        installation_id=installation.id, user_id=USER, task=task, now=NOW
    )

    assert counters.proposed == 0
    assert counters.skipped_reason == "low_confidence"
    assert poster.posts == []


def test_work_surfaces_constrained_to_vocab(db_session: Session) -> None:
    installation = _installation(db_session)
    _observe(db_session, installation, count=6)
    task = _seed_task(db_session, installation)
    poster = FakePoster()
    pass_, _ = _make_pass(
        db_session,
        completion=_completion(
            {
                "role": "Software Engineer",
                # 'tickets' + 'slack' are off-vocab and must be dropped.
                "work_surfaces": ["issues", "tickets", "prs", "slack"],
                "confidence": 0.8,
            }
        ),
        poster=poster,
    )

    pass_.run(installation_id=installation.id, user_id=USER, task=task, now=NOW)

    fact = WorkspaceStateService(db_session).get(
        installation.id, "user", USER, USER_PROFILE_FACT_KEY
    )
    # Still pending (propose→confirm), so read the proposed value off the event.
    proposal = next(
        e
        for e in db_session.scalars(select(TaskEvent))
        if e.payload.get("message") == PENDING_PROPOSAL_MESSAGE
        and e.payload.get("key") == USER_PROFILE_FACT_KEY
    )
    assert fact is None  # not auto-active on the propose path
    assert proposal.payload["value_json"]["work_surfaces"] == ["issues", "prs"]
