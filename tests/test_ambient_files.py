"""HIG-231: ambient file analysis — detection, gates, budgets, ingress wiring."""

import os
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, delete, func, select
from sqlalchemy import event as sa_event
from sqlalchemy.orm import Session

from kortny.config import Settings
from kortny.db.models import (
    Installation,
    ObservationEvent,
    ObservePolicy,
    SlackChannelMembership,
    SlackInboundEvent,
    Task,
    TaskEvent,
    WitnessDeliveryLog,
)
from kortny.db.session import make_engine, make_session_factory, normalize_database_url
from kortny.observe.ambient_files import (
    AMBIENT_FILE_BRIEF_DECISION,
    CHANNEL_POST_DECISION,
    AmbientFileAnalysisService,
    ambient_file_identity_key,
    detect_file_candidates,
    maybe_create_ambient_file_brief,
    summarize_event_files,
)
from kortny.observe.service import ObserveService
from kortny.slack.ingress import SlackIngress
from scripts.demo.fixtures import SIM_MARKER_KEY, build_story

TEST_POSTGRES_URL = os.environ.get("KORTNY_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    TEST_POSTGRES_URL is None,
    reason="KORTNY_TEST_POSTGRES_URL is required for ambient file tests",
)

CHANNEL_ID = "C123"
MAX_MB = 15


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
        WitnessDeliveryLog,
        SlackInboundEvent,
        ObservationEvent,
        ObservePolicy,
        SlackChannelMembership,
        TaskEvent,
        Task,
        Installation,
    ):
        session.execute(delete(model))


def make_settings(**overrides: Any) -> Settings:
    payload: dict[str, Any] = {
        "SLACK_BOT_TOKEN": "xoxb-test",
        "SLACK_APP_TOKEN": "xapp-test",
        "SLACK_SIGNING_SECRET": "signing-secret",
        "LLM_PROVIDER": "openrouter",
        "LLM_API_KEY": "test-key",
        "LLM_MODEL": "openai/gpt-test",
        "COMPOSIO_API_KEY": "composio-key",
        "POSTGRES_URL": TEST_POSTGRES_URL,
        "KORTNY_AMBIENT_FILES_ENABLED": True,
        "KORTNY_AMBIENT_FILE_MAX_MB": MAX_MB,
        "KORTNY_AMBIENT_FILE_BRIEFS_PER_DAY": 1,
        "KORTNY_WITNESS_CHANNEL_POSTS_PER_WEEK": 1,
    }
    payload.update(overrides)
    return Settings.model_validate(payload)


def make_installation(session: Session) -> Installation:
    installation = Installation(
        slack_team_id=f"T{uuid.uuid4().hex[:10].upper()}",
        bot_user_id="UBOT",
    )
    session.add(installation)
    session.flush()
    return installation


def make_policy(
    session: Session,
    installation: Installation,
    *,
    proactivity_status: str = "full",
    channel_id: str = CHANNEL_ID,
) -> ObservePolicy:
    policy = ObservePolicy(
        installation_id=installation.id,
        scope_type="channel",
        scope_id=channel_id,
        observation_status="active",
        proactivity_status=proactivity_status,
        retention_days=90,
        cooldown_seconds=86_400,
        enabled_at=datetime.now(UTC),
        metadata_json={},
    )
    session.add(policy)
    session.flush()
    return policy


def make_observation(
    session: Session,
    installation: Installation,
    *,
    channel_id: str = CHANNEL_ID,
    message_ts: str = "1765400000.000100",
    thread_ts: str | None = None,
    user_id: str = "U123",
    file_id: str | None = None,
) -> ObservationEvent:
    observation = ObservationEvent(
        installation_id=installation.id,
        slack_team_id=installation.slack_team_id,
        channel_id=channel_id,
        user_id=user_id,
        event_type="file_share" if file_id else "message",
        slack_event_id=f"Ev{uuid.uuid4().hex}",
        message_ts=message_ts,
        thread_ts=thread_ts,
        file_id=file_id,
        raw_payload_checksum="checksum",
        text_preview="dropping a file",
        visibility_metadata={"scope_type": "channel", "scope_id": channel_id},
    )
    session.add(observation)
    session.flush()
    return observation


def file_entry(
    *,
    file_id: str = "F0001",
    name: str = "q2-pnl.xlsx",
    filetype: str = "xlsx",
    size: int = 48_128,
    **extra: Any,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "id": file_id,
        "name": name,
        "filetype": filetype,
        "size": size,
    }
    entry.update(extra)
    return entry


def file_event(
    *,
    files: list[dict[str, Any]],
    channel_id: str = CHANNEL_ID,
    ts: str = "1765400000.000100",
    thread_ts: str | None = None,
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "type": "message",
        "channel": channel_id,
        "channel_type": "channel",
        "user": "U123",
        "text": "raw numbers attached",
        "ts": ts,
        "files": files,
    }
    if thread_ts is not None:
        event["thread_ts"] = thread_ts
    return event


def make_service(
    session: Session, settings: Settings | None = None
) -> AmbientFileAnalysisService:
    return AmbientFileAnalysisService(
        session=session,
        settings=settings or make_settings(),
    )


def add_delivery_log_row(
    session: Session,
    installation: Installation,
    *,
    decision: str,
    channel_id: str = CHANNEL_ID,
    created_at: datetime | None = None,
) -> None:
    row = WitnessDeliveryLog(
        installation_id=installation.id,
        slack_user_id=f"channel:{channel_id}",
        candidate_id=None,
        decision=decision,
        reason="test row",
    )
    if created_at is not None:
        row.created_at = created_at
    session.add(row)
    session.flush()


# --- detection -------------------------------------------------------------


def test_detect_whitelisted_type_and_size_passes() -> None:
    event = file_event(
        files=[
            file_entry(file_id="F1", filetype="xlsx", size=10),
            file_entry(file_id="F2", name="data.csv", filetype="CSV", size=2_000),
            file_entry(file_id="F3", name="report.pdf", filetype="pdf", size=5_000),
        ]
    )

    candidates = detect_file_candidates(event, max_mb=MAX_MB)

    assert [c.file_id for c in candidates] == ["F1", "F2", "F3"]
    assert [c.filetype for c in candidates] == ["xlsx", "csv", "pdf"]
    assert candidates[0].name == "q2-pnl.xlsx"
    assert candidates[1].size_bytes == 2_000


def test_detect_skips_oversized_wrong_type_no_files_and_sim() -> None:
    oversized = file_entry(file_id="F1", size=(MAX_MB * 1024 * 1024) + 1)
    wrong_type = file_entry(file_id="F2", name="demo.mov", filetype="mov")
    sim_flagged = file_entry(file_id="F3") | {SIM_MARKER_KEY: True}
    missing_id = file_entry(file_id="F4") | {"id": None}
    missing_size = {"id": "F5", "name": "x.xlsx", "filetype": "xlsx"}

    event = file_event(
        files=[oversized, wrong_type, sim_flagged, missing_id, missing_size]
    )
    assert detect_file_candidates(event, max_mb=MAX_MB) == ()

    no_files_event = {"type": "message", "channel": CHANNEL_ID, "text": "hi"}
    assert detect_file_candidates(no_files_event, max_mb=MAX_MB) == ()

    at_limit = file_event(files=[file_entry(file_id="F6", size=MAX_MB * 1024 * 1024)])
    assert len(detect_file_candidates(at_limit, max_mb=MAX_MB)) == 1


def test_summarize_event_files_is_compact_and_keeps_sim_marker() -> None:
    summary = summarize_event_files(
        [
            file_entry(file_id="F1", url_private="https://secret", mode="hosted"),
            file_entry(file_id="F2") | {SIM_MARKER_KEY: True},
        ]
    )

    assert summary[0] == {
        "id": "F1",
        "name": "q2-pnl.xlsx",
        "filetype": "xlsx",
        "size": 48_128,
    }
    assert "url_private" not in summary[0]
    assert summary[1][SIM_MARKER_KEY] is True


# --- policy gates ----------------------------------------------------------


def test_policy_digest_only_and_off_create_no_task(db_session: Session) -> None:
    service = make_service(db_session)
    for status in ("digest_only", "off"):
        installation = make_installation(db_session)
        policy = make_policy(db_session, installation, proactivity_status=status)
        observation = make_observation(db_session, installation)

        decision = service.maybe_create_analysis_task(
            installation=installation,
            policy=policy,
            observation=observation,
            event=file_event(files=[file_entry()]),
        )

        assert decision.created is False
        assert decision.reason == "policy_not_full"

    assert db_session.scalar(select(func.count()).select_from(Task)) == 0
    assert db_session.scalar(select(func.count()).select_from(WitnessDeliveryLog)) == 0


def test_policy_full_creates_task_with_identity_thread_and_input(
    db_session: Session,
) -> None:
    installation = make_installation(db_session)
    policy = make_policy(db_session, installation)
    observation = make_observation(
        db_session,
        installation,
        message_ts="1765400000.000200",
        thread_ts="1765399999.000100",
        file_id="FPNL01",
    )

    decision = make_service(db_session).maybe_create_analysis_task(
        installation=installation,
        policy=policy,
        observation=observation,
        event=file_event(
            files=[file_entry(file_id="FPNL01")],
            ts="1765400000.000200",
            thread_ts="1765399999.000100",
        ),
    )

    assert decision.created is True
    assert decision.reason == "created"
    task = decision.task
    assert task is not None
    assert task.identity_kind == "synthetic"
    assert task.identity_key == f"synthetic:ambient-file:{CHANNEL_ID}:FPNL01"
    assert task.identity_key == ambient_file_identity_key(CHANNEL_ID, "FPNL01")
    assert task.slack_channel_id == CHANNEL_ID
    # The brief must land as a threaded reply on the file message's thread.
    assert task.slack_thread_ts == "1765399999.000100"
    assert task.slack_message_ts == "1765400000.000200"
    assert task.slack_user_id == "U123"
    assert "slack_file_read" in task.input
    assert "sandbox" in task.input
    assert "FPNL01" in task.input
    assert "do not modify anything" in task.input
    assert "offer it, do not build it" in task.input
    assert "low-key" in task.input

    log_rows = db_session.scalars(select(WitnessDeliveryLog)).all()
    assert len(log_rows) == 1
    assert log_rows[0].decision == AMBIENT_FILE_BRIEF_DECISION
    # Same channel-row key convention as HIG-198 channel posts.
    assert log_rows[0].slack_user_id == f"channel:{CHANNEL_ID}"
    assert "FPNL01" in (log_rows[0].reason or "")


def test_thread_falls_back_to_message_ts_for_top_level_posts(
    db_session: Session,
) -> None:
    installation = make_installation(db_session)
    policy = make_policy(db_session, installation)
    observation = make_observation(
        db_session, installation, message_ts="1765400000.000300", thread_ts=None
    )

    decision = make_service(db_session).maybe_create_analysis_task(
        installation=installation,
        policy=policy,
        observation=observation,
        event=file_event(files=[file_entry(file_id="FTOP01")]),
    )

    assert decision.created is True
    assert decision.task is not None
    assert decision.task.slack_thread_ts == "1765400000.000300"


# --- dedup -----------------------------------------------------------------


def test_same_file_id_never_creates_second_task(db_session: Session) -> None:
    installation = make_installation(db_session)
    policy = make_policy(db_session, installation)
    settings = make_settings(
        KORTNY_AMBIENT_FILE_BRIEFS_PER_DAY=10,
        KORTNY_WITNESS_CHANNEL_POSTS_PER_WEEK=10,
    )
    service = make_service(db_session, settings)

    first = service.maybe_create_analysis_task(
        installation=installation,
        policy=policy,
        observation=make_observation(
            db_session, installation, message_ts="1765400000.000400"
        ),
        event=file_event(files=[file_entry(file_id="FDUP01")]),
    )
    assert first.created is True

    # Re-observation of the same file (new Slack event, same file id).
    second = service.maybe_create_analysis_task(
        installation=installation,
        policy=policy,
        observation=make_observation(
            db_session, installation, message_ts="1765400000.000500"
        ),
        event=file_event(files=[file_entry(file_id="FDUP01")]),
    )

    assert second.created is False
    assert second.reason == "duplicate"
    assert db_session.scalar(select(func.count()).select_from(Task)) == 1
    assert db_session.scalar(select(func.count()).select_from(WitnessDeliveryLog)) == 1


# --- budgets ---------------------------------------------------------------


def test_second_brief_same_day_is_deferred(db_session: Session) -> None:
    installation = make_installation(db_session)
    policy = make_policy(db_session, installation)
    settings = make_settings(KORTNY_WITNESS_CHANNEL_POSTS_PER_WEEK=10)
    service = make_service(db_session, settings)

    first = service.maybe_create_analysis_task(
        installation=installation,
        policy=policy,
        observation=make_observation(
            db_session, installation, message_ts="1765400000.000600"
        ),
        event=file_event(files=[file_entry(file_id="FDAY01")]),
    )
    assert first.created is True

    second = service.maybe_create_analysis_task(
        installation=installation,
        policy=policy,
        observation=make_observation(
            db_session, installation, message_ts="1765400000.000700"
        ),
        event=file_event(files=[file_entry(file_id="FDAY02")]),
    )

    assert second.created is False
    assert second.reason == "daily_budget_exhausted"
    assert db_session.scalar(select(func.count()).select_from(Task)) == 1
    # Deferrals are logged, not written: budget rows stay at one.
    assert db_session.scalar(select(func.count()).select_from(WitnessDeliveryLog)) == 1


def test_weekly_budget_shared_with_channel_post_rows(db_session: Session) -> None:
    installation = make_installation(db_session)
    policy = make_policy(db_session, installation)
    # Fake HIG-198 channel post inside the 7-day window, outside today.
    add_delivery_log_row(
        db_session,
        installation,
        decision=CHANNEL_POST_DECISION,
        created_at=datetime.now(UTC) - timedelta(days=1),
    )

    decision = make_service(db_session).maybe_create_analysis_task(
        installation=installation,
        policy=policy,
        observation=make_observation(
            db_session, installation, message_ts="1765400000.000800"
        ),
        event=file_event(files=[file_entry(file_id="FWEEK01")]),
    )

    assert decision.created is False
    assert decision.reason == "weekly_budget_exhausted"
    assert db_session.scalar(select(func.count()).select_from(Task)) == 0


def test_weekly_window_counts_both_decision_kinds(db_session: Session) -> None:
    installation = make_installation(db_session)
    policy = make_policy(db_session, installation)
    settings = make_settings(KORTNY_WITNESS_CHANNEL_POSTS_PER_WEEK=3)
    add_delivery_log_row(
        db_session,
        installation,
        decision=CHANNEL_POST_DECISION,
        created_at=datetime.now(UTC) - timedelta(days=2),
    )
    add_delivery_log_row(
        db_session,
        installation,
        decision=AMBIENT_FILE_BRIEF_DECISION,
        created_at=datetime.now(UTC) - timedelta(days=3),
    )
    add_delivery_log_row(
        db_session,
        installation,
        decision=CHANNEL_POST_DECISION,
        created_at=datetime.now(UTC) - timedelta(days=4),
    )
    # Rows older than the window must not count.
    add_delivery_log_row(
        db_session,
        installation,
        decision=CHANNEL_POST_DECISION,
        created_at=datetime.now(UTC) - timedelta(days=10),
    )

    decision = make_service(db_session, settings).maybe_create_analysis_task(
        installation=installation,
        policy=policy,
        observation=make_observation(
            db_session, installation, message_ts="1765400000.000900"
        ),
        event=file_event(files=[file_entry(file_id="FWEEK02")]),
    )

    assert decision.created is False
    assert decision.reason == "weekly_budget_exhausted"


# --- feature flag ----------------------------------------------------------


def test_flag_off_creates_nothing(db_session: Session) -> None:
    installation = make_installation(db_session)
    policy = make_policy(db_session, installation)
    settings = make_settings(KORTNY_AMBIENT_FILES_ENABLED=False)

    decision = make_service(db_session, settings).maybe_create_analysis_task(
        installation=installation,
        policy=policy,
        observation=make_observation(db_session, installation),
        event=file_event(files=[file_entry(file_id="FOFF01")]),
    )

    assert decision.created is False
    assert decision.reason == "disabled"
    assert db_session.scalar(select(func.count()).select_from(Task)) == 0
    assert db_session.scalar(select(func.count()).select_from(WitnessDeliveryLog)) == 0


# --- observe service file metadata capture ----------------------------------


def test_record_channel_message_persists_files_summary(db_session: Session) -> None:
    installation = make_installation(db_session)
    result = ObserveService(db_session).record_channel_message(
        installation=installation,
        slack_team_id=installation.slack_team_id,
        body={"event_id": f"Ev{uuid.uuid4().hex}"},
        event=file_event(
            files=[file_entry(file_id="FMETA01", url_private="https://secret")]
        ),
    )

    assert result.observed is True
    observation = result.event
    assert observation is not None
    assert observation.event_type == "file_share"
    assert observation.file_id == "FMETA01"
    assert observation.visibility_metadata["file_count"] == 1
    assert observation.visibility_metadata["files"] == [
        {"id": "FMETA01", "name": "q2-pnl.xlsx", "filetype": "xlsx", "size": 48_128}
    ]

    plain = ObserveService(db_session).record_channel_message(
        installation=installation,
        slack_team_id=installation.slack_team_id,
        body={"event_id": f"Ev{uuid.uuid4().hex}"},
        event={
            "type": "message",
            "channel": CHANNEL_ID,
            "channel_type": "channel",
            "user": "U123",
            "text": "no files here",
            "ts": "1765400000.001000",
        },
    )
    assert plain.event is not None
    assert "files" not in plain.event.visibility_metadata


# --- ingress wiring: ack-fast, no LLM ---------------------------------------


class FakeSlackClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def chat_postMessage(
        self,
        *,
        channel: str,
        text: str,
        thread_ts: str | None = None,
        blocks: list[dict[str, Any]] | None = None,
        **_kwargs: object,
    ) -> dict[str, Any]:
        self.calls.append({"channel": channel, "text": text})
        return {"ok": True, "channel": channel, "ts": "1765400001.000001"}

    def auth_test(self) -> dict[str, Any]:
        return {"ok": True, "user_id": "UBOT"}


def _forbid_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("LLM must not be touched on the ambient file gate path")

    monkeypatch.setattr("kortny.llm.service.LLMService.__init__", boom)
    monkeypatch.setattr("kortny.llm.litellm_provider.create_litellm_provider", boom)


def test_ingress_gate_creates_task_without_llm(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    _forbid_llm(monkeypatch)
    monkeypatch.setattr("kortny.observe.ambient_files._default_settings", make_settings)

    ingress = SlackIngress(session=db_session, client=FakeSlackClient())
    body = {"event_id": f"Ev{uuid.uuid4().hex}", "team_id": "T123"}
    event = file_event(files=[file_entry(file_id="FING01")])

    # First observation creates the default policy (digest_only): no task.
    result = ingress.observe_channel_message(body=body, event=event)
    assert result.observed is True
    assert db_session.scalar(select(func.count()).select_from(Task)) == 0

    policy = db_session.scalars(select(ObservePolicy)).one()
    policy.proactivity_status = "full"
    db_session.flush()

    second_event = file_event(
        files=[file_entry(file_id="FING02")], ts="1765400000.001100"
    )
    ingress.observe_channel_message(
        body={"event_id": f"Ev{uuid.uuid4().hex}", "team_id": "T123"},
        event=second_event,
    )

    task = db_session.scalars(select(Task)).one()
    assert task.identity_key == f"synthetic:ambient-file:{CHANNEL_ID}:FING02"
    assert task.slack_thread_ts == "1765400000.001100"
    log_rows = db_session.scalars(select(WitnessDeliveryLog)).all()
    assert [row.decision for row in log_rows] == [AMBIENT_FILE_BRIEF_DECISION]


def test_gate_path_is_cheap_no_files_zero_queries_created_path_bounded(
    db_session: Session, engine: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    _forbid_llm(monkeypatch)
    installation = make_installation(db_session)
    policy = make_policy(db_session, installation)
    observation = make_observation(
        db_session, installation, message_ts="1765400000.001200"
    )
    db_session.flush()

    statements: list[str] = []

    def count_statement(
        conn: Any,
        cursor: Any,
        statement: str,
        parameters: Any,
        context: Any,
        executemany: bool,
    ) -> None:
        statements.append(statement)

    sa_event.listen(engine, "before_cursor_execute", count_statement)
    try:
        no_files = maybe_create_ambient_file_brief(
            session=db_session,
            installation=installation,
            policy=policy,
            observation=observation,
            event={"type": "message", "channel": CHANNEL_ID, "text": "hi"},
            settings=make_settings(),
        )
        assert no_files.reason == "no_files"
        assert statements == []

        created = maybe_create_ambient_file_brief(
            session=db_session,
            installation=installation,
            policy=policy,
            observation=observation,
            event=file_event(files=[file_entry(file_id="FCHEAP01")]),
            settings=make_settings(),
        )
        assert created.created is True
        # Gate + creation stays a handful of statements: dedup read, two
        # budget counts, identity-key reads, savepoint + task insert +
        # release, log insert. No LLM, no network, no scans.
        assert len(statements) <= 12
    finally:
        sa_event.remove(engine, "before_cursor_execute", count_statement)


# --- simulator fixture -------------------------------------------------------


def test_simulator_story_carries_sim_flagged_file_share() -> None:
    story = build_story(now=datetime(2026, 6, 11, 12, 0, tzinfo=UTC), days=21)
    file_messages = [message for message in story if message.files]

    assert len(file_messages) == 1
    fixture = file_messages[0]
    assert fixture.pattern == "file_share"
    entry = fixture.files[0]
    assert entry["filetype"] == "pdf"
    assert entry[SIM_MARKER_KEY] is True
    # The sim marker keeps the gate from analyzing a file id that is not real.
    assert detect_file_candidates({"files": [dict(entry)]}, max_mb=MAX_MB) == ()
