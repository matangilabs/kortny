"""Tests for ProactiveActionEvent dual-write (Chunk 2 of the Proactive Action Ledger)."""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import patch

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, delete, select
from sqlalchemy.orm import Session

from kortny.db.models import (
    Installation,
    ProactiveActionEvent,
    Schedule,
    SlackSideEffect,
    Task,
    TaskEvent,
    WitnessDeliveryLog,
    WitnessOpportunityCandidate,
)
from kortny.db.session import make_engine, make_session_factory, normalize_database_url
from kortny.witness import (
    WitnessAutopilot,
    accept_candidate,
    dismiss_candidate,
    materialize_acceptance,
    reactivate_candidate,
    snooze_candidate,
    sync_candidate_for_schedule_action,
)
from kortny.witness.lifecycle import archive_candidate

TEST_POSTGRES_URL = os.environ.get("KORTNY_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    TEST_POSTGRES_URL is None,
    reason="KORTNY_TEST_POSTGRES_URL is required for proactive ledger event tests",
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
        _cleanup(session)
        session.commit()
        yield session
        session.rollback()
        _cleanup(session)
        session.commit()


def _cleanup(session: Session) -> None:
    for model in (
        ProactiveActionEvent,
        WitnessDeliveryLog,
        WitnessOpportunityCandidate,
        SlackSideEffect,
        TaskEvent,
        Task,
        Installation,
    ):
        session.execute(delete(model))


def _make_installation(session: Session) -> Installation:
    installation = Installation(slack_team_id=f"T{uuid.uuid4().hex}")
    session.add(installation)
    session.flush()
    return installation


def _make_candidate(
    session: Session,
    installation: Installation,
    *,
    status: str = "candidate",
) -> WitnessOpportunityCandidate:
    now = datetime.now(UTC)
    candidate = WitnessOpportunityCandidate(
        installation_id=installation.id,
        channel_id="DDMCHAN1",
        target_slack_user_id="UDMTARGET",
        visibility_scope_type="dm",
        visibility_scope_id="UDMTARGET",
        candidate_type="recurring_check",
        title="Daily summary",
        summary="Post a daily trading summary.",
        suggested_action="Post in channel.",
        suggested_message="I can post a summary here.",
        evidence_json=[],
        source_type="channel_profile",
        source_id="profile-1",
        dedupe_key=f"test:{uuid.uuid4()}",
        confidence_score=Decimal("0.750"),
        confidence_reason="Strong recurring signal.",
        status=status,
        metadata_json={},
        feedback_json={},
        created_at=now,
        updated_at=now,
    )
    session.add(candidate)
    session.flush()
    return candidate


def _events_for(
    session: Session, candidate: WitnessOpportunityCandidate
) -> list[ProactiveActionEvent]:
    return list(
        session.scalars(
            select(ProactiveActionEvent)
            .where(ProactiveActionEvent.candidate_id == candidate.id)
            .order_by(ProactiveActionEvent.created_at)
        )
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_dismiss_candidate_writes_event(db_session: Session) -> None:
    installation = _make_installation(db_session)
    candidate = _make_candidate(db_session, installation)
    initial_status = candidate.status  # "candidate"

    dismiss_candidate(
        db_session,
        candidate.id,
        installation_id=installation.id,
        by_user_id="U_TESTER",
        reason="Not relevant",
    )
    db_session.commit()

    events = _events_for(db_session, candidate)
    assert len(events) == 1
    ev = events[0]
    assert ev.from_state == initial_status
    assert ev.to_state == "dismissed"
    assert ev.event_type == "dismissed"
    assert ev.actor_id == "U_TESTER"
    assert ev.candidate_id == candidate.id
    assert ev.installation_id == installation.id


def test_snooze_candidate_writes_event(db_session: Session) -> None:
    installation = _make_installation(db_session)
    candidate = _make_candidate(db_session, installation)

    snooze_candidate(
        db_session,
        candidate.id,
        installation_id=installation.id,
        by_user_id="U_TESTER",
        duration=timedelta(days=3),
    )
    db_session.commit()

    events = _events_for(db_session, candidate)
    assert len(events) == 1
    ev = events[0]
    assert ev.to_state == "cooldown"
    assert ev.event_type == "snoozed"
    assert ev.reason_code == "snoozed"
    assert ev.actor_id == "U_TESTER"


def test_accept_candidate_writes_event(db_session: Session) -> None:
    installation = _make_installation(db_session)
    candidate = _make_candidate(db_session, installation)

    accept_candidate(
        db_session,
        candidate.id,
        installation_id=installation.id,
        by_user_id="U_ACCEPTOR",
    )
    db_session.commit()

    events = _events_for(db_session, candidate)
    assert len(events) == 1
    ev = events[0]
    assert ev.from_state == "candidate"
    assert ev.to_state == "accepted"
    assert ev.event_type == "accepted"
    assert ev.actor_id == "U_ACCEPTOR"


def test_reactivate_candidate_writes_event(db_session: Session) -> None:
    installation = _make_installation(db_session)
    candidate = _make_candidate(db_session, installation, status="dismissed")

    reactivate_candidate(
        db_session,
        candidate.id,
        installation_id=installation.id,
        by_user_id="U_REACTIVATOR",
    )
    db_session.commit()

    events = _events_for(db_session, candidate)
    assert len(events) == 1
    ev = events[0]
    assert ev.from_state == "dismissed"
    assert ev.to_state == "candidate"
    assert ev.event_type == "reactivated"
    assert ev.actor_id == "U_REACTIVATOR"


def test_archive_candidate_writes_event(db_session: Session) -> None:
    installation = _make_installation(db_session)
    candidate = _make_candidate(db_session, installation)

    archive_candidate(
        db_session,
        candidate.id,
        installation_id=installation.id,
        by_user_id="U_ARCHIVER",
    )
    db_session.commit()

    events = _events_for(db_session, candidate)
    assert len(events) == 1
    ev = events[0]
    assert ev.to_state == "archived"
    assert ev.event_type == "archived"
    assert ev.actor_id == "U_ARCHIVER"


def test_events_disabled_flag_suppresses_writes(db_session: Session) -> None:
    """When KORTNY_PROACTIVE_LEDGER_EVENTS_ENABLED=False, no events are written."""
    installation = _make_installation(db_session)
    candidate = _make_candidate(db_session, installation)

    with patch.dict("os.environ", {"KORTNY_PROACTIVE_LEDGER_EVENTS_ENABLED": "false"}):
        dismiss_candidate(
            db_session,
            candidate.id,
            installation_id=installation.id,
            by_user_id="U_TESTER",
        )
    db_session.commit()

    events = _events_for(db_session, candidate)
    assert len(events) == 0
    # The candidate status transition still happened
    db_session.refresh(candidate)
    assert candidate.status == "dismissed"


def test_multiple_transitions_produce_ordered_events(db_session: Session) -> None:
    """Snooze then reactivate produces two events in order."""
    installation = _make_installation(db_session)
    candidate = _make_candidate(db_session, installation)

    snooze_candidate(
        db_session,
        candidate.id,
        installation_id=installation.id,
        by_user_id="U_ONE",
        duration=timedelta(days=1),
    )
    reactivate_candidate(
        db_session,
        candidate.id,
        installation_id=installation.id,
        by_user_id="U_TWO",
    )
    db_session.commit()

    events = _events_for(db_session, candidate)
    assert len(events) == 2
    assert events[0].event_type == "snoozed"
    assert events[0].to_state == "cooldown"
    assert events[1].event_type == "reactivated"
    assert events[1].to_state == "candidate"
    assert events[1].from_state == "cooldown"


# ---------------------------------------------------------------------------
# Autopilot transition tests
# ---------------------------------------------------------------------------


def test_autopilot_defer_without_review_writes_event(db_session: Session) -> None:
    """WitnessAutopilot._defer_without_review writes an autopilot_deferred event."""
    installation = _make_installation(db_session)
    # Candidate with no source_task_id so _source_task returns None
    candidate = _make_candidate(db_session, installation)

    now = datetime.now(UTC)
    autopilot = WitnessAutopilot(db_session, actor_id="witness_autopilot")
    initial_status = candidate.status  # "candidate"

    autopilot._defer_without_review(
        candidate,
        now=now,
        reason="No source task for auditable LLM review.",
    )
    db_session.commit()

    assert candidate.status == "cooldown"

    events = _events_for(db_session, candidate)
    assert len(events) == 1
    ev = events[0]
    assert ev.from_state == initial_status
    assert ev.to_state == "cooldown"
    assert ev.event_type == "autopilot_deferred"
    assert ev.reason_code == "no_review_deferred"
    assert ev.actor_id == "witness_autopilot"
    assert ev.candidate_id == candidate.id


# ---------------------------------------------------------------------------
# Automation transition tests
# ---------------------------------------------------------------------------


def _make_schedule(
    session: Session,
    installation: Installation,
    *,
    candidate: WitnessOpportunityCandidate,
    status: str = "active",
) -> Schedule:
    now = datetime.now(UTC)
    from kortny.witness.automation import (
        SCHEDULE_WITNESS_CANDIDATE_KEY,  # noqa: PLC0415
    )

    schedule = Schedule(
        installation_id=installation.id,
        owner_type="system",
        title="Daily summary schedule",
        spec_kind="cron",
        cron_expr="0 9 * * 1-5",
        timezone="UTC",
        status=status,
        task_template={},
        metadata_json={SCHEDULE_WITNESS_CANDIDATE_KEY: str(candidate.id)},
        created_at=now,
        updated_at=now,
    )
    session.add(schedule)
    session.flush()
    return schedule


def test_automation_one_shot_materialize_writes_event(db_session: Session) -> None:
    """materialize_acceptance for a one_shot candidate writes an automated_one_shot event."""
    installation = _make_installation(db_session)
    now = datetime.now(UTC)
    candidate = WitnessOpportunityCandidate(
        installation_id=installation.id,
        channel_id="DDMCHAN1",
        target_slack_user_id="UDMTARGET",
        visibility_scope_type="dm",
        visibility_scope_id="UDMTARGET",
        candidate_type="recurring_check",
        automation_kind="one_shot",
        title="Run monthly report",
        summary="Generate and send the monthly trading summary.",
        suggested_action="Generate and post the report.",
        suggested_message="I can generate the monthly summary now.",
        evidence_json=[],
        source_type="channel_profile",
        source_id="profile-2",
        dedupe_key=f"test-oneshot:{uuid.uuid4()}",
        confidence_score=Decimal("0.850"),
        confidence_reason="Strong one-shot signal.",
        status="accepted",
        metadata_json={},
        feedback_json={},
        created_at=now,
        updated_at=now,
    )
    db_session.add(candidate)
    db_session.flush()
    initial_status = candidate.status  # "accepted"

    outcome = materialize_acceptance(
        db_session,
        None,  # settings=None skips the witness_automation_enabled check
        candidate,
        accepted_by="U_AUTOMATOR",
    )
    # Flush so event rows are visible to the same session query; no commit needed.
    db_session.flush()

    assert outcome.kind == "one_shot"
    assert outcome.task_id is not None
    assert candidate.status == "automated"

    events = _events_for(db_session, candidate)
    assert len(events) == 1
    ev = events[0]
    assert ev.from_state == initial_status
    assert ev.to_state == "automated"
    assert ev.event_type == "automated_one_shot"
    assert ev.actor_id == "U_AUTOMATOR"
    assert ev.task_id == outcome.task_id


def test_automation_sync_schedule_activate_writes_event(db_session: Session) -> None:
    """sync_candidate_for_schedule_action with activate writes an automated_recurring event."""
    installation = _make_installation(db_session)
    candidate = _make_candidate(db_session, installation, status="accepted")
    schedule = _make_schedule(
        db_session, installation, candidate=candidate, status="active"
    )
    initial_status = candidate.status  # "accepted"

    result = sync_candidate_for_schedule_action(
        db_session,
        schedule,
        action="activate",
        by_user_id="U_SCHEDULER",
    )
    # Flush so event rows are visible to the same session query; no commit needed.
    db_session.flush()

    assert result is not None
    assert result.status == "automated"

    events = _events_for(db_session, candidate)
    assert len(events) == 1
    ev = events[0]
    assert ev.from_state == initial_status
    assert ev.to_state == "automated"
    assert ev.event_type == "automated_recurring"
    assert ev.actor_id == "U_SCHEDULER"
    assert ev.candidate_id == candidate.id
