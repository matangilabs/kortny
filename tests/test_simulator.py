"""Workspace simulator tests: seed, witness pickup, idempotency, clean, CLI."""

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

from kortny.db.models import (
    Episode,
    Installation,
    LLMUsage,
    ObservationEvent,
    ObserveChannelProfile,
    ObservePolicy,
    Schedule,
    SlackChannelMembership,
    SlackSideEffect,
    Task,
    TaskEvent,
    TaskStatus,
    WitnessOpportunityCandidate,
)
from kortny.db.models import (
    LLMProvider as DbLLMProvider,
)
from kortny.db.session import make_engine, make_session_factory, normalize_database_url
from kortny.llm import ChatMessage, Completion, TokenUsage
from kortny.simulator import (
    SIM_MARKER_KEY,
    SIM_TASK_IDENTITY_PREFIX,
    SIM_TASK_SPECS,
    SimulatorError,
    clean_simulation,
    seed_simulation,
    simulation_status,
)
from kortny.simulator.__main__ import build_parser
from kortny.tools.types import JsonObject, JsonSchema
from kortny.witness.runner import (
    DEFAULT_WITNESS_SCAN_INTERVAL,
    WitnessRunner,
    _profile_scan_due,
)

TEST_POSTGRES_URL = os.environ.get("KORTNY_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    TEST_POSTGRES_URL is None,
    reason="KORTNY_TEST_POSTGRES_URL is required for simulator tests",
)

SIM_CHANNEL = "CSIMTEST"


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
        WitnessOpportunityCandidate,
        LLMUsage,
        SlackSideEffect,
        Episode,
        ObserveChannelProfile,
        ObservationEvent,
        ObservePolicy,
        SlackChannelMembership,
        TaskEvent,
        Task,
        Schedule,
        Installation,
    ):
        session.execute(delete(model))


def create_installation(session: Session) -> Installation:
    installation = Installation(
        slack_team_id=f"T{uuid.uuid4().hex}",
        bot_user_id="UKORTNY",
    )
    session.add(installation)
    session.flush()
    return installation


class FakeWitnessLLMProvider:
    model = "openai/gpt-4o-mini"

    def __init__(self, completions: list[Completion]) -> None:
        self.completions = completions
        self.calls: list[
            tuple[tuple[ChatMessage, ...], tuple[JsonSchema, ...], JsonObject | None]
        ] = []

    def complete(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[JsonSchema] = (),
        *,
        response_format: JsonObject | None = None,
        max_output_tokens: int | None = None,
    ) -> Completion:
        self.calls.append((tuple(messages), tuple(tools), response_format))
        if not self.completions:
            raise AssertionError("FakeWitnessLLMProvider received too many calls")
        return self.completions.pop(0)


def witness_extraction_completion() -> Completion:
    return Completion(
        content=(
            "{"
            '"candidates":[{'
            '"candidate_type":"recurring_check",'
            '"title":"Automate the EOD trading summary",'
            '"summary":"Post the trading summary the team compiles by hand '
            'every weekday at 5pm.",'
            '"suggested_action":"Offer a weekday 5pm trading summary.",'
            '"suggested_message":"Want me to post the EOD trading summary?",'
            '"automation_kind":"recurring",'
            '"cadence_suggestion":"every weekday at 5pm",'
            '"deliverable":"post the EOD trading summary in this channel",'
            '"evidence":["EOD trading summary posts appear every weekday."],'
            '"confidence_score":0.8,'
            '"confidence_reason":"The profile shows a daily manual pattern."'
            "}],"
            '"skipped_reason":null'
            "}"
        ),
        tool_calls=(),
        usage=TokenUsage(input_tokens=200, output_tokens=80),
        cost_usd=Decimal("0.000100"),
        model="openai/gpt-4o-mini",
    )


def sim_observation_events(session: Session) -> list[ObservationEvent]:
    return list(
        session.scalars(
            select(ObservationEvent).where(
                ObservationEvent.visibility_metadata[SIM_MARKER_KEY]
                .as_boolean()
                .is_(True)
            )
        )
    )


def sim_tasks(session: Session) -> list[Task]:
    return list(
        session.scalars(
            select(Task).where(Task.identity_key.like(f"{SIM_TASK_IDENTITY_PREFIX}%"))
        )
    )


# --- Design-doc scenario: seed writes expected rows, backdated + marked ---


def test_seed_writes_backdated_marked_rows(db_session: Session) -> None:
    create_installation(db_session)
    days = 21
    now = datetime(2026, 6, 10, 12, 0, tzinfo=UTC)
    report = seed_simulation(db_session, channel_id=SIM_CHANNEL, days=days, now=now)

    events = sim_observation_events(db_session)
    assert report.observation_events_created == len(events) > 0
    assert report.observation_events_existing == 0
    assert report.tasks_created == len(SIM_TASK_SPECS) == 3
    assert report.episodes_recorded == 3
    assert report.policy_observable is True
    assert report.membership_created is True
    assert report.profile_created is True
    assert report.profile_version == 1

    window_start = now - timedelta(days=days)
    observed_days = {event.observed_at.date() for event in events}
    assert len(observed_days) >= int(days * 0.6)
    for event in events:
        assert window_start <= event.observed_at <= now
        assert window_start <= event.created_at <= now
        assert event.visibility_metadata[SIM_MARKER_KEY] is True
        assert event.channel_id == SIM_CHANNEL
        assert event.user_id is not None and event.user_id.startswith("USIM")

    # Weekday 17:00 trading-summary pattern is real across the window.
    trading_events = [
        event
        for event in events
        if event.visibility_metadata.get("sim_pattern") == "trading_summary"
    ]
    assert len(trading_events) >= 13  # every weekday in a 21-day window
    for event in trading_events:
        assert event.observed_at.weekday() < 5
        assert event.observed_at.hour == 17

    # Policy ensured through the real observe machinery.
    policy = db_session.scalar(
        select(ObservePolicy).where(
            ObservePolicy.scope_type == "channel",
            ObservePolicy.scope_id == SIM_CHANNEL,
        )
    )
    assert policy is not None
    assert policy.observation_status == "active"

    # Synthetic tasks: marker-recognizable identity, succeeded, backdated.
    tasks = sim_tasks(db_session)
    assert len(tasks) == 3
    for task in tasks:
        assert task.identity_key is not None
        assert task.identity_key.startswith("synthetic:sim:")
        assert task.identity_kind == "synthetic"
        assert task.status is TaskStatus.succeeded
        assert task.result_summary
        assert task.created_at < now - timedelta(days=1)
        assert task.finished_at is not None and task.finished_at < now
        episode = db_session.scalar(select(Episode).where(Episode.task_id == task.id))
        assert episode is not None
        assert episode.summary == task.result_summary
        assert episode.created_at < now

    # Profile row carries the sim marker and the fixture extraction.
    profile = db_session.scalar(
        select(ObserveChannelProfile).where(
            ObserveChannelProfile.channel_id == SIM_CHANNEL
        )
    )
    assert profile is not None
    assert profile.metadata_json[SIM_MARKER_KEY] is True
    assert profile.confidence_score == Decimal("0.750")
    assert profile.message_count == len(events)
    assert profile.observed_range_start_ts is not None
    assert profile.observed_range_end_ts is not None
    extraction = profile.profile_json["semantic_extraction"]
    assert extraction["confidence"] == "high"
    assert any("weekday" in workflow for workflow in extraction["workflows"])


# --- Design-doc scenario: profile is picked up by the witness runner ---


def test_seeded_profile_is_scan_due_for_witness_runner(db_session: Session) -> None:
    installation = create_installation(db_session)
    now = datetime(2026, 6, 10, 12, 0, tzinfo=UTC)
    seed_simulation(db_session, channel_id=SIM_CHANNEL, days=21, now=now)

    runner = WitnessRunner(db_session, runner_id="sim-test")
    rows = runner._candidate_profiles(installation_id=installation.id, limit=10)
    profiles = [profile for profile, _membership in rows]
    assert len(profiles) == 1
    profile = profiles[0]
    assert profile.channel_id == SIM_CHANNEL
    assert _profile_scan_due(
        profile,
        now=now,
        min_scan_interval=DEFAULT_WITNESS_SCAN_INTERVAL,
    )


def test_witness_runner_projects_candidates_from_seeded_profile(
    db_session: Session,
) -> None:
    installation = create_installation(db_session)
    now = datetime(2026, 6, 10, 12, 0, tzinfo=UTC)
    seed_simulation(db_session, channel_id=SIM_CHANNEL, days=21, now=now)

    provider = FakeWitnessLLMProvider([witness_extraction_completion()])
    result = WitnessRunner(
        db_session,
        llm_provider=provider,
        provider_name=DbLLMProvider.openrouter,
        runner_id="sim-test",
    ).run_once(installation_id=installation.id, now=now, profile_limit=5)
    db_session.flush()

    assert result.status == "processed"
    assert result.projected_count == 1
    assert len(provider.calls) == 1
    candidate = db_session.scalar(select(WitnessOpportunityCandidate))
    assert candidate is not None
    assert candidate.channel_id == SIM_CHANNEL
    assert candidate.automation_kind == "recurring"
    profile = db_session.scalar(
        select(ObserveChannelProfile).where(
            ObserveChannelProfile.channel_id == SIM_CHANNEL
        )
    )
    assert profile is not None
    assert candidate.source_profile_id == profile.id
    # The sim marker propagates into candidate evidence via profile refs.
    assert any(
        isinstance(item, dict) and item.get(SIM_MARKER_KEY) is True
        for item in candidate.evidence_json
    )
    # After the scan the profile is no longer due until the interval passes.
    assert not _profile_scan_due(
        profile,
        now=now + timedelta(minutes=5),
        min_scan_interval=DEFAULT_WITNESS_SCAN_INTERVAL,
    )


# --- Design-doc scenario: idempotent re-seed ---


def test_reseed_is_idempotent(db_session: Session) -> None:
    create_installation(db_session)
    now = datetime(2026, 6, 10, 12, 0, tzinfo=UTC)
    first = seed_simulation(db_session, channel_id=SIM_CHANNEL, days=21, now=now)
    second = seed_simulation(db_session, channel_id=SIM_CHANNEL, days=21, now=now)

    assert second.observation_events_created == 0
    assert second.observation_events_existing == first.observation_events_created
    assert second.tasks_created == 0
    assert second.tasks_existing == 3
    assert second.episodes_recorded == 0
    assert second.profile_created is False
    assert second.profile_version == 2  # bumped so witness re-scans

    events = sim_observation_events(db_session)
    assert len(events) == first.observation_events_created
    assert len(sim_tasks(db_session)) == 3
    episodes = list(db_session.scalars(select(Episode)))
    assert len(episodes) == 3
    profiles = list(db_session.scalars(select(ObserveChannelProfile)))
    assert len(profiles) == 1
    memberships = list(db_session.scalars(select(SlackChannelMembership)))
    assert len(memberships) == 1


def test_reseed_makes_scanned_profile_due_again(db_session: Session) -> None:
    installation = create_installation(db_session)
    now = datetime(2026, 6, 10, 12, 0, tzinfo=UTC)
    seed_simulation(db_session, channel_id=SIM_CHANNEL, days=21, now=now)

    provider = FakeWitnessLLMProvider([witness_extraction_completion()])
    WitnessRunner(
        db_session,
        llm_provider=provider,
        provider_name=DbLLMProvider.openrouter,
        runner_id="sim-test",
    ).run_once(installation_id=installation.id, now=now, profile_limit=5)
    profile = db_session.scalar(select(ObserveChannelProfile))
    assert profile is not None
    assert not _profile_scan_due(
        profile,
        now=now + timedelta(minutes=5),
        min_scan_interval=DEFAULT_WITNESS_SCAN_INTERVAL,
    )

    seed_simulation(db_session, channel_id=SIM_CHANNEL, days=21, now=now)
    db_session.refresh(profile)
    assert _profile_scan_due(
        profile,
        now=now + timedelta(minutes=10),
        min_scan_interval=DEFAULT_WITNESS_SCAN_INTERVAL,
    )


# --- Design-doc scenario: clean removes sim + derived rows, nothing else ---


def test_clean_removes_sim_rows_and_derived_candidates_only(
    db_session: Session,
) -> None:
    installation = create_installation(db_session)
    now = datetime(2026, 6, 10, 12, 0, tzinfo=UTC)
    seed_simulation(db_session, channel_id=SIM_CHANNEL, days=21, now=now)

    # Derived candidate produced by the real witness runner from the profile.
    provider = FakeWitnessLLMProvider([witness_extraction_completion()])
    WitnessRunner(
        db_session,
        llm_provider=provider,
        provider_name=DbLLMProvider.openrouter,
        runner_id="sim-test",
    ).run_once(installation_id=installation.id, now=now, profile_limit=5)

    # Fabricated candidate linked directly to a sim task.
    sim_task = sim_tasks(db_session)[0]
    db_session.add(
        WitnessOpportunityCandidate(
            installation_id=installation.id,
            channel_id=SIM_CHANNEL,
            visibility_scope_type="channel",
            visibility_scope_id=SIM_CHANNEL,
            candidate_type="artifact_followup",
            title="Verify the Q2 pipeline numbers doc",
            summary="One-shot verification ask from the sim history.",
            evidence_json=[],
            source_type="task_summary",
            source_id=str(sim_task.id),
            source_task_id=sim_task.id,
            dedupe_key=uuid.uuid4().hex,
            confidence_score=Decimal("0.700"),
            status="candidate",
        )
    )

    # Non-sim rows in the same installation must survive clean.
    real_task = Task(
        installation_id=installation.id,
        slack_channel_id="CREAL",
        slack_user_id="UREAL",
        identity_kind="manual",
        identity_key=f"manual:{uuid.uuid4().hex}",
        input="Real work request.",
        status=TaskStatus.succeeded,
    )
    db_session.add(real_task)
    db_session.flush()
    real_event = ObservationEvent(
        installation_id=installation.id,
        slack_team_id=installation.slack_team_id,
        channel_id="CREAL",
        user_id="UREAL",
        event_type="message",
        slack_event_id=f"Ev{uuid.uuid4().hex}",
        raw_payload_checksum="realchecksum",
        text_preview="real channel message",
        visibility_metadata={"scope_type": "channel", "scope_id": "CREAL"},
    )
    db_session.add(real_event)
    real_candidate = WitnessOpportunityCandidate(
        installation_id=installation.id,
        channel_id="CREAL",
        visibility_scope_type="channel",
        visibility_scope_id="CREAL",
        candidate_type="general_help",
        title="Real candidate",
        summary="Derived from real activity.",
        evidence_json=[{"type": "task_response", "snippet": "real"}],
        source_type="task_summary",
        source_id=str(real_task.id),
        source_task_id=real_task.id,
        dedupe_key=uuid.uuid4().hex,
        confidence_score=Decimal("0.600"),
        status="candidate",
    )
    db_session.add(real_candidate)
    db_session.flush()

    report = clean_simulation(db_session)

    assert report.candidates_deleted == 2
    assert report.tasks_deleted == 3
    assert report.episodes_deleted == 3
    assert report.profiles_deleted == 1
    assert report.task_events_deleted > 0
    assert report.observation_events_deleted > 0
    assert report.memberships_deleted == 1
    assert report.automated_candidate_notes == ()

    # No sim residue.
    assert sim_observation_events(db_session) == []
    assert sim_tasks(db_session) == []
    assert (
        db_session.scalar(
            select(ObserveChannelProfile).where(
                ObserveChannelProfile.channel_id == SIM_CHANNEL
            )
        )
        is None
    )
    status = simulation_status(db_session)
    assert status.observation_events == 0
    assert status.tasks == 0
    assert status.profiles == 0
    assert status.episodes == 0
    assert status.candidates_by_status == {}
    assert status.memberships == 0

    # Non-sim rows untouched.
    assert db_session.get(Task, real_task.id) is not None
    assert db_session.get(ObservationEvent, real_event.id) is not None
    assert db_session.get(WitnessOpportunityCandidate, real_candidate.id) is not None


def test_clean_leaves_schedule_for_automated_candidate(db_session: Session) -> None:
    installation = create_installation(db_session)
    now = datetime(2026, 6, 10, 12, 0, tzinfo=UTC)
    seed_simulation(db_session, channel_id=SIM_CHANNEL, days=21, now=now)

    provider = FakeWitnessLLMProvider([witness_extraction_completion()])
    WitnessRunner(
        db_session,
        llm_provider=provider,
        provider_name=DbLLMProvider.openrouter,
        runner_id="sim-test",
    ).run_once(installation_id=installation.id, now=now, profile_limit=5)
    candidate = db_session.scalar(select(WitnessOpportunityCandidate))
    assert candidate is not None

    schedule = Schedule(
        installation_id=installation.id,
        owner_type="system",
        title="EOD trading summary",
        spec_kind="cron",
        cron_expr="0 17 * * 1-5",
        status="active",
    )
    db_session.add(schedule)
    db_session.flush()
    candidate.status = "automated"
    candidate.automated_schedule_id = schedule.id
    db_session.flush()

    report = clean_simulation(db_session)

    assert report.candidates_deleted == 1
    assert len(report.automated_candidate_notes) == 1
    assert str(schedule.id) in report.automated_candidate_notes[0]
    assert db_session.get(Schedule, schedule.id) is not None
    assert db_session.get(WitnessOpportunityCandidate, candidate.id) is None


# --- Design-doc scenario: CLI arg validation + empty install refusal ---


def test_cli_seed_requires_channel() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["seed"])
    assert excinfo.value.code == 2


def test_cli_requires_subcommand() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args([])
    assert excinfo.value.code == 2


def test_seed_refuses_empty_installation(db_session: Session) -> None:
    with pytest.raises(SimulatorError, match="No installation"):
        seed_simulation(db_session, channel_id=SIM_CHANNEL, days=21)


def test_seed_refuses_blank_channel(db_session: Session) -> None:
    create_installation(db_session)
    with pytest.raises(SimulatorError, match="--channel"):
        seed_simulation(db_session, channel_id="   ", days=21)
