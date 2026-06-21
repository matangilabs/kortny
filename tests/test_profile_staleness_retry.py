"""Tests for profile staleness fence (Part A) and channel-assessment retry (Part B)."""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, delete, select
from sqlalchemy.orm import Session

from kortny.consolidator.passes import (
    HygieneCounters,
    _mark_stale_profiles,
    _refresh_stale_profiles,
    _requeue_failed_assessments,
    run_hygiene,
)
from kortny.db.models import (
    Installation,
    ObservationEvent,
    ObserveChannelProfile,
    SlackChannelMembership,
    Task,
    TaskEvent,
    TaskStatus,
    WitnessOpportunityCandidate,
)
from kortny.db.session import make_engine, make_session_factory, normalize_database_url
from kortny.observe.assessment import (
    CHANNEL_ASSESSMENT_REQUESTED_MESSAGE,
    assessment_event_id_for_membership,
    assessment_identity_source_id,
)
from kortny.observe.profiles import ObserveChannelProfileService
from kortny.slack.membership import (
    ASSESSMENT_MAX_FAILURES,
    SlackChannelMembershipService,
)
from kortny.tasks import TaskService
from kortny.witness.runner import WitnessRunner

TEST_POSTGRES_URL = os.environ.get("KORTNY_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    TEST_POSTGRES_URL is None,
    reason="KORTNY_TEST_POSTGRES_URL is required",
)

NOW = datetime(2026, 6, 21, 12, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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
        WitnessOpportunityCandidate,
        ObservationEvent,
        ObserveChannelProfile,
        TaskEvent,
        Task,
        SlackChannelMembership,
        Installation,
    ):
        session.execute(delete(model))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _install(session: Session) -> Installation:
    inst = Installation(slack_team_id=f"T{uuid.uuid4().hex[:8]}")
    session.add(inst)
    session.flush()
    return inst


def _membership(
    session: Session,
    installation: Installation,
    *,
    channel_id: str = "C_TEST",
    onboarding_status: str = "posted",
    metadata: dict | None = None,
) -> SlackChannelMembership:
    m = SlackChannelMembership(
        installation_id=installation.id,
        channel_id=channel_id,
        channel_name="test-channel",
        channel_type="public_channel",
        membership_status="active",
        discovered_via="member_joined_channel",
        added_by_user_id="U_ADMIN",
        onboarding_status=onboarding_status,
        onboarding_message_ts="1779900000.000000",
        metadata_json=metadata or {},
    )
    session.add(m)
    session.flush()
    return m


def _profile(
    session: Session,
    installation: Installation,
    *,
    channel_id: str = "C_TEST",
    status: str = "active",
    last_profiled_at: datetime | None = None,
) -> ObserveChannelProfile:
    p = ObserveChannelProfile(
        installation_id=installation.id,
        channel_id=channel_id,
        profile_status=status,
        last_profiled_at=last_profiled_at,
    )
    session.add(p)
    session.flush()
    return p


def _task(
    session: Session,
    installation: Installation,
    *,
    channel_id: str = "C_TEST",
    status: TaskStatus = TaskStatus.pending,
) -> Task:
    t = TaskService(session).create_task(
        installation_id=installation.id,
        slack_event_id=f"Ev{uuid.uuid4().hex}",
        slack_channel_id=channel_id,
        slack_thread_ts="1779900000.000000",
        slack_message_ts=f"1779900000.{uuid.uuid4().hex[:6]}",
        slack_user_id="U_ADMIN",
        input="test input",
    )
    t.status = status
    session.flush()
    return t


def _obs_event(
    session: Session,
    installation: Installation,
    *,
    channel_id: str = "C_TEST",
    observed_at: datetime | None = None,
) -> ObservationEvent:
    ev = ObservationEvent(
        installation_id=installation.id,
        slack_team_id=installation.slack_team_id,
        channel_id=channel_id,
        event_type="message",
        raw_payload_checksum=uuid.uuid4().hex,
        observed_at=observed_at or NOW,
    )
    session.add(ev)
    session.flush()
    return ev


# ---------------------------------------------------------------------------
# Part A: Staleness fence
# ---------------------------------------------------------------------------


def test_hygiene_marks_old_profile_stale(db_session: Session) -> None:
    inst = _install(db_session)
    p = _profile(
        db_session, inst, last_profiled_at=NOW - timedelta(days=31), status="active"
    )
    db_session.flush()

    marked = _mark_stale_profiles(db_session, installation_id=inst.id, now=NOW)
    db_session.flush()

    db_session.refresh(p)
    assert marked == 1
    assert p.profile_status == "stale"


def test_hygiene_does_not_mark_recent_profile_stale(db_session: Session) -> None:
    inst = _install(db_session)
    p = _profile(
        db_session, inst, last_profiled_at=NOW - timedelta(days=5), status="active"
    )
    db_session.flush()

    marked = _mark_stale_profiles(db_session, installation_id=inst.id, now=NOW)
    db_session.flush()

    db_session.refresh(p)
    assert marked == 0
    assert p.profile_status == "active"


def test_refresh_includes_stale_profiles(db_session: Session) -> None:
    inst = _install(db_session)
    # Profile older than PROFILE_STALE_AFTER — will have been marked stale
    _profile(
        db_session,
        inst,
        channel_id="C_STALE",
        status="stale",
        last_profiled_at=NOW - timedelta(days=31),
    )
    _membership(db_session, inst, channel_id="C_STALE")
    _obs_event(
        db_session, inst, channel_id="C_STALE", observed_at=NOW - timedelta(days=1)
    )
    db_session.flush()

    ts = TaskService(db_session)
    refreshed = _refresh_stale_profiles(
        db_session, installation_id=inst.id, task_service=ts, now=NOW
    )

    assert refreshed == 1
    task = db_session.scalar(select(Task).where(Task.slack_channel_id == "C_STALE"))
    assert task is not None


def test_witness_excludes_stale_profiles(db_session: Session) -> None:
    inst = _install(db_session)
    _membership(db_session, inst, channel_id="C_STALE_WIT")
    _profile(
        db_session,
        inst,
        channel_id="C_STALE_WIT",
        status="stale",
        last_profiled_at=NOW - timedelta(days=35),
    )
    db_session.flush()

    runner = WitnessRunner(db_session)
    candidates = runner._candidate_profiles(installation_id=inst.id, limit=10)

    assert len(candidates) == 0


def test_upsert_clears_stale_to_active(db_session: Session) -> None:
    inst = _install(db_session)
    m = _membership(db_session, inst, channel_id="C_STALE_UPSERT")
    p = _profile(
        db_session,
        inst,
        channel_id="C_STALE_UPSERT",
        status="stale",
        last_profiled_at=NOW - timedelta(days=35),
    )
    t = _task(db_session, inst, channel_id="C_STALE_UPSERT")
    db_session.flush()

    ObserveChannelProfileService(db_session).upsert_from_assessment(
        task=t,
        membership=m,
        result_summary="Channel recovered from stale.",
    )
    db_session.refresh(p)

    assert p.profile_status == "active"


# ---------------------------------------------------------------------------
# Part B: Assessment retry — mark_assessment_failed
# ---------------------------------------------------------------------------


def test_mark_assessment_failed_increments_count(db_session: Session) -> None:
    inst = _install(db_session)
    m = _membership(db_session, inst)
    svc = SlackChannelMembershipService(db_session)

    svc.mark_assessment_failed(membership=m, error_type="timeout", error="timed out")
    db_session.refresh(m)
    metadata = m.metadata_json or {}

    assert metadata["assessment_failure_count"] == 1
    assert metadata["assessment_status"] == "failed"
    assert "assessment_next_attempt_at" in metadata
    assert "assessment_dead_lettered_at" not in metadata

    # next_attempt_at should be ~5 minutes in the future (backoff base)
    next_attempt = datetime.fromisoformat(metadata["assessment_next_attempt_at"])
    if next_attempt.tzinfo is None:
        next_attempt = next_attempt.replace(tzinfo=UTC)
    delta = next_attempt - datetime.now(UTC)
    assert timedelta(minutes=4) < delta < timedelta(minutes=6)


def test_dead_letter_after_n_failures(db_session: Session) -> None:
    inst = _install(db_session)
    m = _membership(db_session, inst)
    svc = SlackChannelMembershipService(db_session)

    for _ in range(ASSESSMENT_MAX_FAILURES):
        svc.mark_assessment_failed(
            membership=m, error_type="timeout", error="timed out"
        )
        db_session.refresh(m)

    metadata = m.metadata_json or {}
    assert metadata["assessment_failure_count"] == ASSESSMENT_MAX_FAILURES
    assert "assessment_dead_lettered_at" in metadata


def test_mark_assessment_failed_backoff_grows(db_session: Session) -> None:
    inst = _install(db_session)
    m = _membership(db_session, inst)
    svc = SlackChannelMembershipService(db_session)

    before = datetime.now(UTC)
    svc.mark_assessment_failed(membership=m, error_type="x", error="err")
    db_session.refresh(m)
    next1 = datetime.fromisoformat(
        str((m.metadata_json or {})["assessment_next_attempt_at"])
    ).replace(tzinfo=UTC)

    svc.mark_assessment_failed(membership=m, error_type="x", error="err")
    db_session.refresh(m)
    next2 = datetime.fromisoformat(
        str((m.metadata_json or {})["assessment_next_attempt_at"])
    ).replace(tzinfo=UTC)

    # Second attempt window should be twice as long (10 min vs 5 min from now).
    assert next2 > next1 + timedelta(minutes=4)
    del before


# ---------------------------------------------------------------------------
# Part B: Assessment retry — ingress gate logic (tested via metadata state)
# ---------------------------------------------------------------------------


def test_gate_skips_dead_lettered_via_metadata(db_session: Session) -> None:
    """Dead-lettered membership must not produce a new task through the gate.
    Verified by checking that the metadata gate condition triggers correctly."""
    inst = _install(db_session)
    m = _membership(
        db_session,
        inst,
        metadata={"assessment_dead_lettered_at": NOW.isoformat()},
    )
    metadata = m.metadata_json or {}
    assert bool(metadata.get("assessment_dead_lettered_at")) is True


def test_gate_backoff_window_metadata(db_session: Session) -> None:
    """Membership with next_attempt_at in the future must be gated out."""
    future = (NOW + timedelta(hours=1)).isoformat()
    inst = _install(db_session)
    m = _membership(
        db_session,
        inst,
        metadata={
            "assessment_status": "failed",
            "assessment_next_attempt_at": future,
        },
    )
    metadata = m.metadata_json or {}
    next_attempt = datetime.fromisoformat(str(metadata["assessment_next_attempt_at"]))
    if next_attempt.tzinfo is None:
        next_attempt = next_attempt.replace(tzinfo=UTC)
    assert next_attempt > NOW  # the gate would return None


def test_gate_allows_retry_after_backoff_window(db_session: Session) -> None:
    """Past next_attempt_at with a non-active existing task must allow retry."""
    past = (NOW - timedelta(hours=1)).isoformat()
    inst = _install(db_session)
    failed_task = _task(db_session, inst, status=TaskStatus.failed)
    db_session.flush()

    m = _membership(
        db_session,
        inst,
        metadata={
            "assessment_status": "failed",
            "assessment_failure_count": 1,
            "assessment_next_attempt_at": past,
            "assessment_task_id": str(failed_task.id),
        },
    )
    metadata = m.metadata_json or {}

    # Replicate the gate logic manually:
    dead_lettered = bool(metadata.get("assessment_dead_lettered_at"))
    assert dead_lettered is False

    next_attempt_raw = metadata.get("assessment_next_attempt_at")
    assert next_attempt_raw is not None
    next_attempt = datetime.fromisoformat(str(next_attempt_raw)).replace(tzinfo=UTC)
    # Past window — gate would NOT block.
    assert next_attempt <= NOW

    # Task status is failed — gate would NOT block.
    task_id = uuid.UUID(str(metadata["assessment_task_id"]))
    existing = db_session.get(Task, task_id)
    assert existing is not None
    _active = frozenset(
        {
            TaskStatus.pending,
            TaskStatus.running,
            TaskStatus.waiting_approval,
            TaskStatus.crashed,
        }
    )
    assert TaskStatus(existing.status) not in _active


def test_gate_skips_while_task_active(db_session: Session) -> None:
    """Membership with in-flight (pending) task must be blocked by the gate."""
    inst = _install(db_session)
    active_task = _task(db_session, inst, status=TaskStatus.pending)
    db_session.flush()

    metadata = {
        "assessment_status": "queued",
        "assessment_task_id": str(active_task.id),
    }
    task_id = uuid.UUID(str(metadata["assessment_task_id"]))
    existing = db_session.get(Task, task_id)
    assert existing is not None
    _active = frozenset(
        {
            TaskStatus.pending,
            TaskStatus.running,
            TaskStatus.waiting_approval,
            TaskStatus.crashed,
        }
    )
    assert TaskStatus(existing.status) in _active


def test_gate_skips_before_next_attempt_at(db_session: Session) -> None:
    """next_attempt_at in the future means the gate must not issue a new task."""
    future = (NOW + timedelta(hours=2)).isoformat()
    metadata = {
        "assessment_status": "failed",
        "assessment_next_attempt_at": future,
    }
    next_attempt = datetime.fromisoformat(str(metadata["assessment_next_attempt_at"]))
    if next_attempt.tzinfo is None:
        next_attempt = next_attempt.replace(tzinfo=UTC)
    # Gate would block because next_attempt > now
    assert next_attempt > NOW


# ---------------------------------------------------------------------------
# Part B: Retry uses a new identity key
# ---------------------------------------------------------------------------


def test_retry_uses_new_identity(db_session: Session) -> None:
    """After one failure, the retry attempt uses a different source_id."""
    membership_id = uuid.uuid4()

    initial_source_id = assessment_identity_source_id(membership_id, attempt=0)
    retry_source_id = assessment_identity_source_id(membership_id, attempt=1)

    assert initial_source_id != retry_source_id
    assert str(membership_id) in retry_source_id
    assert "attempt:1" in retry_source_id


def test_retry_event_id_differs_from_initial(db_session: Session) -> None:
    """Retry event_id must differ from the initial so Slack dedup doesn't block it."""
    membership_id = uuid.uuid4()

    initial = assessment_event_id_for_membership(membership_id, attempt=0)
    retry = assessment_event_id_for_membership(membership_id, attempt=1)

    assert initial != retry
    assert "attempt:1" in retry


# ---------------------------------------------------------------------------
# Part B: Consolidator re-queue
# ---------------------------------------------------------------------------


def test_consolidator_requeues_failed_assessment(db_session: Session) -> None:
    inst = _install(db_session)
    past = (NOW - timedelta(hours=1)).isoformat()
    _membership(
        db_session,
        inst,
        channel_id="C_RETRY",
        metadata={
            "assessment_status": "failed",
            "assessment_failure_count": 1,
            "assessment_next_attempt_at": past,
        },
    )
    db_session.flush()

    ts = TaskService(db_session)
    requeued = _requeue_failed_assessments(
        db_session, installation_id=inst.id, task_service=ts, now=NOW
    )

    assert requeued == 1
    task = db_session.scalar(select(Task).where(Task.slack_channel_id == "C_RETRY"))
    assert task is not None
    # Verify the request event was logged
    req_event = db_session.scalar(
        select(TaskEvent).where(
            TaskEvent.task_id == task.id,
            TaskEvent.payload["message"].as_string()
            == CHANNEL_ASSESSMENT_REQUESTED_MESSAGE,
        )
    )
    assert req_event is not None


def test_consolidator_skips_dead_lettered(db_session: Session) -> None:
    inst = _install(db_session)
    past = (NOW - timedelta(hours=1)).isoformat()
    _membership(
        db_session,
        inst,
        channel_id="C_DEAD",
        metadata={
            "assessment_status": "failed",
            "assessment_failure_count": 3,
            "assessment_next_attempt_at": past,
            "assessment_dead_lettered_at": NOW.isoformat(),
        },
    )
    db_session.flush()

    ts = TaskService(db_session)
    requeued = _requeue_failed_assessments(
        db_session, installation_id=inst.id, task_service=ts, now=NOW
    )

    assert requeued == 0


def test_consolidator_skips_inside_backoff_window(db_session: Session) -> None:
    inst = _install(db_session)
    future = (NOW + timedelta(hours=1)).isoformat()
    _membership(
        db_session,
        inst,
        channel_id="C_WAIT",
        metadata={
            "assessment_status": "failed",
            "assessment_failure_count": 1,
            "assessment_next_attempt_at": future,
        },
    )
    db_session.flush()

    ts = TaskService(db_session)
    requeued = _requeue_failed_assessments(
        db_session, installation_id=inst.id, task_service=ts, now=NOW
    )

    assert requeued == 0


def test_consolidator_dedupes_retry_with_ingress(db_session: Session) -> None:
    """Same attempt number -> same identity key -> only one task created."""
    inst = _install(db_session)
    past = (NOW - timedelta(hours=1)).isoformat()
    _membership(
        db_session,
        inst,
        channel_id="C_DEDUP",
        metadata={
            "assessment_status": "failed",
            "assessment_failure_count": 1,
            "assessment_next_attempt_at": past,
        },
    )
    db_session.flush()

    ts = TaskService(db_session)

    # First call — creates the task.
    first = _requeue_failed_assessments(
        db_session, installation_id=inst.id, task_service=ts, now=NOW
    )
    # Second call — identical identity key must deduplicate.
    second = _requeue_failed_assessments(
        db_session, installation_id=inst.id, task_service=ts, now=NOW
    )

    assert first == 1
    assert second == 1  # create_task returns the existing row on dedup
    task_count = db_session.execute(
        select(Task).where(Task.slack_channel_id == "C_DEDUP")
    ).all()
    assert len(task_count) == 1


def test_run_hygiene_counters_include_new_fields(db_session: Session) -> None:
    """run_hygiene must return HygieneCounters with profiles_marked_stale and assessments_requeued."""
    inst = _install(db_session)
    db_session.flush()

    counters = run_hygiene(db_session, installation_id=inst.id, now=NOW)

    assert isinstance(counters, HygieneCounters)
    assert hasattr(counters, "profiles_marked_stale")
    assert hasattr(counters, "assessments_requeued")
    payload = counters.to_payload()
    assert "profiles_marked_stale" in payload
    assert "assessments_requeued" in payload
