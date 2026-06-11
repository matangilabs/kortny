"""HIG-227: proactivity quality loop — reinforcement, receptivity, digests."""

import os
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, delete, func, select, text
from sqlalchemy.orm import Session

from kortny.config import Settings
from kortny.dashboard.data import get_witness_kpis
from kortny.db.models import (
    Installation,
    LLMUsage,
    ObserveChannelProfile,
    SlackChannelMembership,
    SlackSideEffect,
    Task,
    TaskEvent,
    WitnessDeliveryLog,
    WitnessOpportunityCandidate,
)
from kortny.db.session import make_engine, make_session_factory, normalize_database_url
from kortny.tasks import TaskService
from kortny.witness import (
    WITNESS_DIGEST_PURPOSE,
    WitnessOpportunityCandidateInput,
    WitnessOpportunityService,
    WitnessRunner,
    effective_confidence,
    receptivity,
    recurrence_evidence_line,
    recurrence_is_proven,
)
from kortny.witness.receptivity import UserFeedbackEvent
from kortny.witness.runner import _in_quiet_hours

TEST_POSTGRES_URL = os.environ.get("KORTNY_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    TEST_POSTGRES_URL is None,
    reason="KORTNY_TEST_POSTGRES_URL is required for witness quality tests",
)

NOW = datetime(2026, 6, 10, 15, 0, tzinfo=UTC)


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
        WitnessOpportunityCandidate,
        LLMUsage,
        SlackSideEffect,
        ObserveChannelProfile,
        SlackChannelMembership,
        TaskEvent,
        Task,
        Installation,
    ):
        session.execute(delete(model))


class FakeWitnessSlackClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def chat_postMessage(
        self,
        *,
        channel: str,
        text: str,
        thread_ts: str | None = None,
        blocks: list[dict[str, Any]] | None = None,
    ) -> dict[str, str | bool]:
        self.calls.append({"channel": channel, "text": text, "thread_ts": thread_ts})
        return {"ok": True, "ts": f"1780300{len(self.calls):03d}.000001"}


def make_installation(session: Session) -> Installation:
    installation = Installation(slack_team_id=f"T{uuid.uuid4().hex}")
    session.add(installation)
    session.flush()
    return installation


def make_dm_candidate(
    session: Session,
    installation_id: uuid.UUID,
    *,
    title: str,
    confidence: str = "0.900",
    candidate_type: str = "recurring_check",
    automation_kind: str | None = None,
    cadence_suggestion: str | None = None,
    channel_id: str = "DQualityUser",
    user_id: str = "UQuality",
    reinforcement_count: int = 1,
    first_observed_at: datetime | None = None,
    evidence_count: int = 1,
    feedback_json: dict[str, Any] | None = None,
) -> WitnessOpportunityCandidate:
    candidate = WitnessOpportunityCandidate(
        installation_id=installation_id,
        channel_id=channel_id,
        target_slack_user_id=user_id,
        visibility_scope_type="dm",
        visibility_scope_id=channel_id,
        candidate_type=candidate_type,
        title=title,
        summary=f"{title} summary.",
        suggested_action=f"Help with {title}.",
        suggested_message=f"I can help with {title}.",
        evidence_json=[
            {"type": "llm_evidence", "snippet": f"{title} evidence {index}"}
            for index in range(evidence_count)
        ],
        source_type="task_summary",
        source_id=None,
        source_task_id=None,
        source_profile_id=None,
        dedupe_key=uuid.uuid4().hex[:32],
        confidence_score=Decimal(confidence),
        confidence_reason="test fixture",
        status="candidate",
        automation_kind=automation_kind,
        cadence_suggestion=cadence_suggestion,
        deliverable=None,
        reinforcement_count=reinforcement_count,
        first_observed_at=first_observed_at or NOW - timedelta(days=1),
        feedback_json=feedback_json or {},
        metadata_json={},
        created_at=NOW - timedelta(days=1),
        updated_at=NOW - timedelta(days=1),
    )
    session.add(candidate)
    session.flush()
    return candidate


def run_delivery(
    session: Session,
    installation_id: uuid.UUID,
    client: FakeWitnessSlackClient,
    *,
    now: datetime = NOW,
    digest_max_items: int = 5,
    digest_interval: timedelta = timedelta(hours=24),
    quiet_hours_start: int | None = None,
    quiet_hours_end: int | None = None,
    delivery_threshold: Decimal = Decimal("0.55"),
) -> Any:
    return WitnessRunner(
        session,
        slack_client=client,
        runner_id="witness-quality-test",
    ).run_once(
        installation_id=installation_id,
        now=now,
        profile_limit=0,
        deliver_private=True,
        delivery_threshold=delivery_threshold,
        digest_interval=digest_interval,
        digest_max_items=digest_max_items,
        quiet_hours_start=quiet_hours_start,
        quiet_hours_end=quiet_hours_end,
    )


def delivery_log_rows(session: Session) -> tuple[WitnessDeliveryLog, ...]:
    return tuple(
        session.scalars(
            select(WitnessDeliveryLog).order_by(WitnessDeliveryLog.created_at.asc())
        )
    )


# --- Design-doc test: migration + backfill first_observed_at ---


def _migration_0032_down_revision() -> str:
    """Read 0032's parent so the test survives orchestrator re-parenting."""

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "kortny_migration_0032",
        "kortny/db/migrations/versions/0032_witness_quality.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    down_revision = module.down_revision
    assert isinstance(down_revision, str)
    return down_revision


def test_migration_0032_backfills_first_observed_at(engine: Engine) -> None:
    assert TEST_POSTGRES_URL is not None
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", normalize_database_url(TEST_POSTGRES_URL))

    # Step down to 0032's parent, insert a legacy row, then re-apply 0032 to
    # exercise the backfill. Keep this aligned with 0032's down_revision.
    down_revision = _migration_0032_down_revision()
    command.downgrade(config, down_revision)
    installation_id = uuid.uuid4()
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO installations (id, slack_team_id) VALUES (:id, :team_id)"
            ),
            {"id": installation_id, "team_id": f"T{uuid.uuid4().hex}"},
        )
        conn.execute(
            text(
                "INSERT INTO witness_opportunity_candidates "
                "(installation_id, channel_id, visibility_scope_type, "
                "visibility_scope_id, candidate_type, title, summary, "
                "source_type, dedupe_key) "
                "VALUES (:installation_id, 'CBackfill', 'channel', 'CBackfill', "
                "'recurring_check', 'Backfill check', 'Backfill summary.', "
                "'channel_profile', 'backfill-test-key')"
            ),
            {"installation_id": installation_id},
        )
    command.upgrade(config, "head")

    with engine.begin() as conn:
        row = conn.execute(
            text(
                "SELECT reinforcement_count, first_observed_at, created_at, "
                "last_decision, receptivity_score "
                "FROM witness_opportunity_candidates "
                "WHERE dedupe_key = 'backfill-test-key'"
            )
        ).one()
        assert row.reinforcement_count == 1
        assert row.first_observed_at is not None
        assert row.first_observed_at == row.created_at
        assert row.last_decision is None
        assert row.receptivity_score is None
        conn.execute(
            text("DELETE FROM installations WHERE id = :id"),
            {"id": installation_id},
        )


# --- Design-doc test: reinforcement increments on dedupe re-observation ---


def test_reinforcement_increments_on_dedupe_reobservation(
    db_session: Session,
) -> None:
    installation = make_installation(db_session)
    dm_task = TaskService(db_session).create_task(
        installation_id=installation.id,
        slack_event_id=f"Ev{uuid.uuid4().hex}",
        slack_channel_id="DReinforce",
        slack_thread_ts="DReinforce",
        slack_message_ts="1780200000.000001",
        slack_user_id="UReinforce",
        input="Watch my daily report request.",
    )
    candidate_input = WitnessOpportunityCandidateInput(
        candidate_type="recurring_check",
        title="Daily report follow-up",
        summary="Offer to check the daily report.",
        suggested_action="Check the daily report.",
        suggested_message="I can keep an eye on the daily report.",
        evidence=("The user asked about the daily report.",),
        confidence_score=Decimal("0.700"),
        confidence_reason="Recurring request.",
        metadata_json={"extractor": "test"},
        automation_kind="recurring",
        cadence_suggestion="every weekday at 9am",
    )
    service = WitnessOpportunityService(db_session)

    first = service.project_from_task_candidates(
        task=dm_task,
        candidates=(candidate_input,),
        response_text="I can watch this.",
    )
    assert first.created_count == 1
    candidate = db_session.scalar(select(WitnessOpportunityCandidate))
    assert candidate is not None
    assert candidate.reinforcement_count == 1
    assert candidate.first_observed_at is not None
    first_observed = candidate.first_observed_at

    second = service.project_from_task_candidates(
        task=dm_task,
        candidates=(candidate_input,),
        response_text="I can still watch this.",
    )
    assert second.updated_count == 1
    assert candidate.reinforcement_count == 2
    assert candidate.first_observed_at == first_observed

    third = service.project_from_task_candidates(
        task=dm_task,
        candidates=(candidate_input,),
        response_text="Still the same opportunity.",
    )
    assert third.updated_count == 1
    assert candidate.reinforcement_count == 3


# --- Design-doc test: recurrence framing gate (both failure modes) ---


def test_recurrence_evidence_line_requires_count_and_span(
    db_session: Session,
) -> None:
    installation = make_installation(db_session)
    proven = make_dm_candidate(
        db_session,
        installation.id,
        title="Proven recurrence",
        automation_kind="recurring",
        cadence_suggestion="every weekday",
        reinforcement_count=3,
        first_observed_at=NOW - timedelta(days=8),
    )
    line = recurrence_evidence_line(proven, now=NOW)
    assert line is not None
    assert "I've seen this 3 times since" in line
    assert (NOW - timedelta(days=8)).date().isoformat() in line
    assert recurrence_is_proven(proven, now=NOW) is True

    # Failure 1: enough span, too few observations.
    low_count = make_dm_candidate(
        db_session,
        installation.id,
        title="Low count recurrence",
        automation_kind="recurring",
        reinforcement_count=2,
        first_observed_at=NOW - timedelta(days=8),
    )
    assert recurrence_evidence_line(low_count, now=NOW) is None
    assert recurrence_is_proven(low_count, now=NOW) is False

    # Failure 2: enough observations, span too short.
    short_span = make_dm_candidate(
        db_session,
        installation.id,
        title="Short span recurrence",
        automation_kind="recurring",
        reinforcement_count=5,
        first_observed_at=NOW - timedelta(days=5),
    )
    assert recurrence_evidence_line(short_span, now=NOW) is None
    assert recurrence_is_proven(short_span, now=NOW) is False

    # Non-recurring candidates never claim recurrence.
    one_shot = make_dm_candidate(
        db_session,
        installation.id,
        title="One shot",
        automation_kind="one_shot",
        reinforcement_count=5,
        first_observed_at=NOW - timedelta(days=30),
    )
    assert recurrence_evidence_line(one_shot, now=NOW) is None


# --- Design-doc test: effective_confidence composition math ---


def test_effective_confidence_composition() -> None:
    # 0.8 * min(1, 0.6 + 0.1*1 + 0.05*2) = 0.8 * 0.8 = 0.64
    assert effective_confidence(
        Decimal("0.800"), reinforcement_count=1, evidence_count=2
    ) == Decimal("0.640")
    # multiplier caps at 1.0: 0.6 + 0.5 + 0.2 = 1.3 -> 1.0
    assert effective_confidence(
        Decimal("0.900"), reinforcement_count=5, evidence_count=4
    ) == Decimal("0.900")
    # zero counts: 0.7 * 0.6 = 0.42
    assert effective_confidence(
        Decimal("0.700"), reinforcement_count=0, evidence_count=0
    ) == Decimal("0.420")
    # never exceeds 1.0
    assert effective_confidence(
        Decimal("1.000"), reinforcement_count=10, evidence_count=10
    ) == Decimal("1.000")


# --- Design-doc test: receptivity features ---


def test_receptivity_dismissal_penalty_and_recovery() -> None:
    now = NOW
    fresh_dismissal = (
        UserFeedbackEvent("dismissed", "recurring_check", now - timedelta(hours=1)),
    )
    assert receptivity(fresh_dismissal, "recurring_check", now) == pytest.approx(
        0.6, abs=0.01
    )

    # Linear recovery: 7 of 14 days elapsed -> 0.6 + 0.4 * 0.5 = 0.8
    half_recovered = (
        UserFeedbackEvent("dismissed", "recurring_check", now - timedelta(days=7)),
    )
    assert receptivity(half_recovered, "recurring_check", now) == pytest.approx(
        0.8, abs=0.01
    )

    # Fully recovered after 14+ days (still inside the 30d window).
    recovered = (
        UserFeedbackEvent("dismissed", "recurring_check", now - timedelta(days=20)),
    )
    assert receptivity(recovered, "recurring_check", now) == pytest.approx(1.0)

    # Dismissals in other categories do not hit this category's penalty.
    other_category = (
        UserFeedbackEvent("dismissed", "general_help", now - timedelta(hours=1)),
    )
    assert receptivity(other_category, "recurring_check", now) == pytest.approx(1.0)


def test_receptivity_global_cooldown_and_acceptance_boost() -> None:
    now = NOW
    # >=3 dismissals across categories in last 7d -> x0.5 even for a clean
    # category.
    spread_dismissals = (
        UserFeedbackEvent("dismissed", "general_help", now - timedelta(days=1)),
        UserFeedbackEvent("dismissed", "workflow_gap", now - timedelta(days=2)),
        UserFeedbackEvent("dismissed", "data_quality_issue", now - timedelta(days=3)),
    )
    assert receptivity(spread_dismissals, "recurring_check", now) == pytest.approx(0.5)

    # Acceptance boost: dismissal penalty 0.6 * 1.15 boost = 0.69.
    mixed = (
        UserFeedbackEvent("dismissed", "recurring_check", now - timedelta(hours=2)),
        UserFeedbackEvent("accepted", "recurring_check", now - timedelta(days=2)),
    )
    assert receptivity(mixed, "recurring_check", now) == pytest.approx(0.69, abs=0.01)

    # Boost alone caps at 1.0.
    boosts = (
        UserFeedbackEvent("accepted", "recurring_check", now - timedelta(days=1)),
        UserFeedbackEvent("accepted", "recurring_check", now - timedelta(days=2)),
    )
    assert receptivity(boosts, "recurring_check", now) == pytest.approx(1.0)


def test_receptivity_bounds() -> None:
    now = NOW
    pile = tuple(
        UserFeedbackEvent("dismissed", "recurring_check", now - timedelta(hours=index))
        for index in range(12)
    )
    score = receptivity(pile, "recurring_check", now)
    assert 0.0 <= score <= 1.0
    assert score < 0.1
    assert receptivity((), "recurring_check", now) == pytest.approx(1.0)


# --- Design-doc test: decision assignment incl. silent; silent never posts ---


def test_silent_decision_is_logged_and_never_posts(db_session: Session) -> None:
    installation = make_installation(db_session)
    candidate = make_dm_candidate(
        db_session,
        installation.id,
        title="Weak hunch",
        confidence="0.300",
    )
    client = FakeWitnessSlackClient()

    result = run_delivery(db_session, installation.id, client)
    db_session.flush()

    assert client.calls == []
    assert result.delivered_count == 0
    outcome = result.deliveries[0]
    assert outcome.status == "silent"
    assert outcome.decision == "silent"
    assert candidate.status == "candidate"
    assert candidate.last_decision == "silent"
    assert candidate.receptivity_score == Decimal("1.000")
    rows = delivery_log_rows(db_session)
    assert len(rows) == 1
    assert rows[0].decision == "silent"
    assert rows[0].candidate_id == candidate.id
    assert rows[0].slack_user_id == "UQuality"
    assert rows[0].reason is not None
    assert "below threshold" in rows[0].reason
    assert "score=" in rows[0].reason
    assert db_session.scalar(select(func.count()).select_from(SlackSideEffect)) == 0


def test_decision_assignment_notify_question_draft(db_session: Session) -> None:
    installation = make_installation(db_session)
    notify = make_dm_candidate(
        db_session,
        installation.id,
        title="Notify item",
        automation_kind="recurring",
        cadence_suggestion="every weekday at 5pm",
        reinforcement_count=3,
        first_observed_at=NOW - timedelta(days=10),
        evidence_count=2,
    )
    question = make_dm_candidate(
        db_session,
        installation.id,
        title="Question item",
        automation_kind="recurring",
        cadence_suggestion=None,
        reinforcement_count=3,
        first_observed_at=NOW - timedelta(days=10),
        evidence_count=2,
    )
    draft = make_dm_candidate(
        db_session,
        installation.id,
        title="Draft item",
        automation_kind="one_shot",
        evidence_count=2,
    )
    client = FakeWitnessSlackClient()

    result = run_delivery(db_session, installation.id, client)
    db_session.flush()

    decisions = {
        outcome.candidate_id: outcome.decision for outcome in result.deliveries
    }
    assert decisions[notify.id] == "notify"
    assert decisions[question.id] == "question"
    assert decisions[draft.id] == "draft"
    assert notify.last_decision == "notify"
    assert question.last_decision == "question"
    assert draft.last_decision == "draft"

    assert len(client.calls) == 1
    digest_text = client.calls[0]["text"]
    assert "Approve once and I'll run it every weekday at 5pm." in digest_text
    assert "What cadence should I use?" in digest_text
    assert "Say go and I'll do it." in digest_text
    # Proven recurrence earns the evidence line in copy.
    assert "I've seen this 3 times since" in digest_text


def test_unproven_recurrence_lowers_receptivity_no_claim_in_copy(
    db_session: Session,
) -> None:
    installation = make_installation(db_session)
    # score = 0.9 * min(1, 0.6+0.1+0.05) * (1.0 * 0.8) = 0.675*0.8 = 0.54 < 0.55
    candidate = make_dm_candidate(
        db_session,
        installation.id,
        title="Unproven recurrence",
        confidence="0.900",
        automation_kind="recurring",
        cadence_suggestion="every weekday",
        reinforcement_count=1,
        first_observed_at=NOW - timedelta(days=1),
        evidence_count=1,
    )
    client = FakeWitnessSlackClient()

    result = run_delivery(db_session, installation.id, client)
    db_session.flush()

    assert client.calls == []
    assert result.deliveries[0].status == "silent"
    assert candidate.last_decision == "silent"


# --- Design-doc test: digest batching, budget, idempotency, interval, quiet ---


def test_digest_batches_items_into_one_outbox_dm(db_session: Session) -> None:
    installation = make_installation(db_session)
    candidates = [
        make_dm_candidate(
            db_session,
            installation.id,
            title=f"Suggestion {index}",
            evidence_count=2,
        )
        for index in range(3)
    ]
    client = FakeWitnessSlackClient()

    result = run_delivery(db_session, installation.id, client)
    db_session.flush()

    assert result.delivered_count == 3
    assert len(client.calls) == 1
    digest_text = client.calls[0]["text"]
    assert "3 suggestions worth a look" in digest_text
    for index in range(3):
        assert f"Suggestion {index}" in digest_text
    side_effects = tuple(
        db_session.scalars(
            select(SlackSideEffect).where(
                SlackSideEffect.purpose == WITNESS_DIGEST_PURPOSE
            )
        )
    )
    assert len(side_effects) == 1
    assert side_effects[0].status == "succeeded"
    for candidate in candidates:
        assert candidate.status == "sent"
        assert candidate.feedback_json["last_action"]["delivery_policy"] == "digest_dm"
    rows = delivery_log_rows(db_session)
    digest_rows = [row for row in rows if row.decision == "digest"]
    assert len(digest_rows) == 1
    assert digest_rows[0].reason == "sent:3"
    assert digest_rows[0].candidate_id is None
    sent_rows = [row for row in rows if row.reason == "sent"]
    assert len(sent_rows) == 3


def test_digest_hard_cap_defers_overflow_with_budget_reason(
    db_session: Session,
) -> None:
    installation = make_installation(db_session)
    for index in range(7):
        make_dm_candidate(
            db_session,
            installation.id,
            title=f"Budget item {index}",
            confidence=f"0.{90 - index}0",
            evidence_count=2,
        )
    client = FakeWitnessSlackClient()

    result = run_delivery(db_session, installation.id, client, digest_max_items=5)
    db_session.flush()

    assert result.delivered_count == 5
    assert len(client.calls) == 1
    deferred = [
        outcome for outcome in result.deliveries if outcome.status == "budget_deferred"
    ]
    assert len(deferred) == 2
    deferred_rows = [
        row for row in delivery_log_rows(db_session) if row.reason == "budget_deferred"
    ]
    assert len(deferred_rows) == 2
    pending = tuple(
        db_session.scalars(
            select(WitnessOpportunityCandidate).where(
                WitnessOpportunityCandidate.status == "candidate"
            )
        )
    )
    assert len(pending) == 2
    # The two lowest-scored candidates are the deferred ones.
    assert {candidate.title for candidate in pending} == {
        "Budget item 5",
        "Budget item 6",
    }


def test_digest_idempotent_within_window(db_session: Session) -> None:
    installation = make_installation(db_session)
    make_dm_candidate(
        db_session,
        installation.id,
        title="First wave",
        evidence_count=2,
    )
    client = FakeWitnessSlackClient()
    run_delivery(db_session, installation.id, client)
    db_session.flush()
    assert len(client.calls) == 1

    # Simulate a lost interval marker: even then, the outbox window key blocks
    # a second digest DM inside the same window.
    db_session.execute(delete(WitnessDeliveryLog))
    late_candidate = make_dm_candidate(
        db_session,
        installation.id,
        title="Second wave",
        evidence_count=2,
    )
    result = run_delivery(
        db_session,
        installation.id,
        client,
        now=NOW + timedelta(minutes=10),
    )
    db_session.flush()

    assert len(client.calls) == 1
    deduped = [
        outcome for outcome in result.deliveries if outcome.status == "window_deduped"
    ]
    assert len(deduped) == 1
    assert deduped[0].candidate_id == late_candidate.id
    assert late_candidate.status == "candidate"


def test_digest_interval_respected_across_runner_ticks(db_session: Session) -> None:
    installation = make_installation(db_session)
    make_dm_candidate(
        db_session,
        installation.id,
        title="Tick one",
        evidence_count=2,
    )
    client = FakeWitnessSlackClient()
    first = run_delivery(db_session, installation.id, client, now=NOW)
    db_session.flush()
    assert first.delivered_count == 1

    second_candidate = make_dm_candidate(
        db_session,
        installation.id,
        title="Tick two",
        evidence_count=2,
    )
    second = run_delivery(
        db_session,
        installation.id,
        client,
        now=NOW + timedelta(hours=1),
    )
    db_session.flush()
    assert second.delivered_count == 0
    assert len(client.calls) == 1
    assert second.deliveries[0].status == "interval_deferred"
    assert second_candidate.status == "candidate"

    third = run_delivery(
        db_session,
        installation.id,
        client,
        now=NOW + timedelta(hours=25),
    )
    db_session.flush()
    assert third.delivered_count == 1
    assert len(client.calls) == 2
    assert second_candidate.status == "sent"


def test_quiet_hours_defer_digest_without_dropping(db_session: Session) -> None:
    installation = make_installation(db_session)
    candidate = make_dm_candidate(
        db_session,
        installation.id,
        title="Quiet hours item",
        evidence_count=2,
    )
    client = FakeWitnessSlackClient()
    quiet_now = NOW.replace(hour=22)

    result = run_delivery(
        db_session,
        installation.id,
        client,
        now=quiet_now,
        quiet_hours_start=21,
        quiet_hours_end=23,
    )
    db_session.flush()

    assert client.calls == []
    assert result.delivered_count == 0
    assert result.deliveries[0].status == "quiet_hours_deferred"
    assert candidate.status == "candidate"
    assert delivery_log_rows(db_session) == ()

    # Outside quiet hours the same candidate delivers — deferred, not dropped.
    after = run_delivery(
        db_session,
        installation.id,
        client,
        now=quiet_now + timedelta(hours=2),
        quiet_hours_start=21,
        quiet_hours_end=23,
    )
    db_session.flush()
    assert after.delivered_count == 1
    assert candidate.status == "sent"


def test_in_quiet_hours_wraps_midnight() -> None:
    assert _in_quiet_hours(NOW.replace(hour=23), 22, 6) is True
    assert _in_quiet_hours(NOW.replace(hour=3), 22, 6) is True
    assert _in_quiet_hours(NOW.replace(hour=12), 22, 6) is False
    assert _in_quiet_hours(NOW.replace(hour=12), None, None) is False
    assert _in_quiet_hours(NOW.replace(hour=12), 9, None) is False
    assert _in_quiet_hours(NOW.replace(hour=12), 12, 12) is False


# --- Design-doc test: KPIs computed from fixture log rows ---


def test_witness_kpis_from_fixture_rows(db_session: Session) -> None:
    installation = make_installation(db_session)
    delivered_at = NOW - timedelta(days=2)
    accepted_at = delivered_at + timedelta(hours=2)
    accepted_candidate = make_dm_candidate(
        db_session,
        installation.id,
        title="Accepted suggestion",
        feedback_json={
            "history": [
                {
                    "action": "sent",
                    "by_user_id": "witness_runner",
                    "at": delivered_at.isoformat(),
                },
                {
                    "action": "accepted",
                    "by_user_id": "UQuality",
                    "at": accepted_at.isoformat(),
                },
            ]
        },
    )
    accepted_candidate.status = "accepted"
    silent_candidate = make_dm_candidate(
        db_session,
        installation.id,
        title="Silent suggestion",
    )
    db_session.add_all(
        [
            WitnessDeliveryLog(
                installation_id=installation.id,
                slack_user_id="UQuality",
                candidate_id=accepted_candidate.id,
                decision="notify",
                reason="sent",
                created_at=delivered_at,
            ),
            WitnessDeliveryLog(
                installation_id=installation.id,
                slack_user_id="UQuality",
                candidate_id=None,
                decision="digest",
                reason="sent:1",
                created_at=delivered_at,
            ),
            WitnessDeliveryLog(
                installation_id=installation.id,
                slack_user_id="UQuality",
                candidate_id=silent_candidate.id,
                decision="silent",
                reason="score=0.300 below threshold=0.55",
                created_at=delivered_at,
            ),
        ]
    )
    db_session.flush()

    kpis = get_witness_kpis(
        db_session,
        installation_id=installation.id,
        now=NOW,
        window_days=30,
    )

    assert kpis.window_days == 30
    assert kpis.candidates_created == 2
    assert kpis.trigger_rate_per_day == pytest.approx(2 / 30)
    assert kpis.delivered_count == 1
    assert kpis.silent_count == 1
    assert kpis.silent_rate == pytest.approx(0.5)
    assert kpis.acceptance_rate == pytest.approx(1.0)
    assert kpis.dismissal_rate == pytest.approx(0.0)
    assert kpis.time_to_action_median_hours == pytest.approx(2.0, abs=0.01)
    # 1 accepted, 0 automated.
    assert kpis.conversion_to_automation == pytest.approx(0.0)
    counts = {row.decision: row.count for row in kpis.decision_counts}
    # HIG-198/HIG-230 added channel_sent / channel_deferred / draft_executed
    # to the decision breakdown; none occur in this fixture.
    assert counts == {
        "notify": 1,
        "question": 0,
        "draft": 0,
        "silent": 1,
        "digest": 1,
        "channel_sent": 0,
        "channel_deferred": 0,
        "draft_executed": 0,
    }
    assert kpis.silent_rate_label == "50.0%"
    assert kpis.time_to_action_label == "2.0h"


# --- Settings: new witness-block fields parse with UPPERCASE aliases ---


def test_settings_parse_witness_quality_fields() -> None:
    assert TEST_POSTGRES_URL is not None
    settings = Settings.model_validate(
        {
            "SLACK_BOT_TOKEN": "xoxb-test",
            "SLACK_APP_TOKEN": "xapp-test",
            "SLACK_SIGNING_SECRET": "signing-secret",
            "LLM_PROVIDER": "openrouter",
            "LLM_API_KEY": "test-key",
            "LLM_MODEL": "openai/gpt-test",
            "COMPOSIO_API_KEY": "composio-key",
            "POSTGRES_URL": TEST_POSTGRES_URL,
            "KORTNY_EMBEDDINGS_BACKEND": "disabled",
            "KORTNY_WITNESS_DELIVERY_THRESHOLD": "0.60",
            "KORTNY_WITNESS_DIGEST_INTERVAL_HOURS": 12,
            "KORTNY_WITNESS_DIGEST_MAX_ITEMS": 3,
            "KORTNY_WITNESS_QUIET_HOURS_START": 21,
            "KORTNY_WITNESS_QUIET_HOURS_END": 6,
        }
    )
    assert settings.witness_delivery_threshold == Decimal("0.60")
    assert settings.witness_digest_interval_hours == 12
    assert settings.witness_digest_max_items == 3
    assert settings.witness_quiet_hours_start == 21
    assert settings.witness_quiet_hours_end == 6

    with pytest.raises(ValueError, match="DELIVERY_THRESHOLD"):
        Settings.model_validate(
            {
                "SLACK_BOT_TOKEN": "xoxb-test",
                "SLACK_APP_TOKEN": "xapp-test",
                "SLACK_SIGNING_SECRET": "signing-secret",
                "LLM_PROVIDER": "openrouter",
                "LLM_API_KEY": "test-key",
                "LLM_MODEL": "openai/gpt-test",
                "COMPOSIO_API_KEY": "composio-key",
                "POSTGRES_URL": TEST_POSTGRES_URL,
                "KORTNY_EMBEDDINGS_BACKEND": "disabled",
                "KORTNY_WITNESS_DELIVERY_THRESHOLD": "1.5",
            }
        )
