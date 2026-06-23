"""Integration tests for Proactive Action Ledger Chunk 2: outcome reconciliation.

Tests the _reconcile_proactive_outcomes hygiene step that back-fills
task_status / task_finished_at on WitnessOpportunityCandidate rows whose
linked task has reached a terminal state.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from decimal import Decimal

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, delete, select, text
from sqlalchemy.orm import Session

from kortny.consolidator.passes import _reconcile_proactive_outcomes, run_hygiene
from kortny.db.models import (
    Installation,
    ProactiveActionEvent,
    Task,
    TaskEvent,
    TaskStatus,
    WitnessOpportunityCandidate,
)
from kortny.db.session import make_engine, make_session_factory, normalize_database_url
from kortny.tasks import TaskService

TEST_POSTGRES_URL = os.environ.get("KORTNY_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    TEST_POSTGRES_URL is None,
    reason="KORTNY_TEST_POSTGRES_URL is required for proactive ledger outcome tests",
)

NOW = datetime(2026, 6, 21, 12, 0, 0, tzinfo=UTC)


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
        WitnessOpportunityCandidate,
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


def _make_task(
    session: Session,
    installation: Installation,
    *,
    status: TaskStatus,
) -> Task:
    task = TaskService(session).create_task(
        installation_id=installation.id,
        slack_event_id=f"Ev{uuid.uuid4().hex}",
        slack_channel_id="CCHAN1",
        slack_thread_ts="1780000000.000100",
        slack_message_ts=f"1780000000.{uuid.uuid4().hex[:6]}",
        slack_user_id="U_USER",
        input="Proactive action task",
    )
    # Force the desired status directly; TaskService.create_task sets pending.
    task.status = status
    task.updated_at = NOW
    session.flush()
    return task


def _make_candidate(
    session: Session,
    installation: Installation,
    *,
    automated_task_id: uuid.UUID | None = None,
    task_status: str | None = None,
    status: str = "automated",
) -> WitnessOpportunityCandidate:
    now = datetime.now(UTC)
    candidate = WitnessOpportunityCandidate(
        installation_id=installation.id,
        channel_id="CCHAN1",
        target_slack_user_id="U_TARGET",
        visibility_scope_type="channel",
        visibility_scope_id="CCHAN1",
        candidate_type="recurring_check",
        title="Daily summary",
        summary="Post a daily trading summary.",
        suggested_action="Post in channel.",
        suggested_message="I can post a summary here.",
        evidence_json=[],
        source_type="channel_profile",
        source_id="profile-1",
        dedupe_key=f"test-outcomes:{uuid.uuid4()}",
        confidence_score=Decimal("0.750"),
        confidence_reason="Strong recurring signal.",
        status=status,
        automated_task_id=automated_task_id,
        task_status=task_status,
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
# Core reconciliation tests
# ---------------------------------------------------------------------------


def test_reconcile_succeeded(db_session: Session) -> None:
    """Candidate linked to a succeeded task gets task_status=succeeded and an event."""
    installation = _make_installation(db_session)
    task = _make_task(db_session, installation, status=TaskStatus.succeeded)
    candidate = _make_candidate(db_session, installation, automated_task_id=task.id)

    count = _reconcile_proactive_outcomes(
        db_session, installation_id=installation.id, now=NOW
    )
    db_session.flush()

    assert count == 1
    db_session.refresh(candidate)
    assert candidate.task_status == "succeeded"
    assert candidate.task_finished_at == task.updated_at

    events = _events_for(db_session, candidate)
    assert len(events) == 1
    ev = events[0]
    assert ev.event_type == "task_terminal"
    assert ev.to_state == "task_succeeded"
    assert ev.reason_code == "succeeded"
    assert ev.task_id == task.id
    assert ev.candidate_id == candidate.id
    assert ev.installation_id == installation.id


def test_reconcile_failed(db_session: Session) -> None:
    """Candidate linked to a failed task gets task_status=failed and to_state=task_failed."""
    installation = _make_installation(db_session)
    task = _make_task(db_session, installation, status=TaskStatus.failed)
    candidate = _make_candidate(db_session, installation, automated_task_id=task.id)

    count = _reconcile_proactive_outcomes(
        db_session, installation_id=installation.id, now=NOW
    )
    db_session.flush()

    assert count == 1
    db_session.refresh(candidate)
    assert candidate.task_status == "failed"
    assert candidate.task_finished_at == task.updated_at

    events = _events_for(db_session, candidate)
    assert len(events) == 1
    ev = events[0]
    assert ev.event_type == "task_terminal"
    assert ev.to_state == "task_failed"
    assert ev.reason_code == "failed"
    assert ev.task_id == task.id


def test_reconcile_crashed(db_session: Session) -> None:
    """Candidate linked to a crashed task gets task_status=crashed and to_state=task_failed."""
    installation = _make_installation(db_session)
    task = _make_task(db_session, installation, status=TaskStatus.crashed)
    candidate = _make_candidate(db_session, installation, automated_task_id=task.id)

    count = _reconcile_proactive_outcomes(
        db_session, installation_id=installation.id, now=NOW
    )
    db_session.flush()

    assert count == 1
    db_session.refresh(candidate)
    assert candidate.task_status == "crashed"

    events = _events_for(db_session, candidate)
    assert len(events) == 1
    ev = events[0]
    assert ev.to_state == "task_failed"
    assert ev.reason_code == "crashed"


def test_reconcile_running_skipped(db_session: Session) -> None:
    """Candidate linked to a running task is left untouched (task not terminal)."""
    installation = _make_installation(db_session)
    task = _make_task(db_session, installation, status=TaskStatus.running)
    candidate = _make_candidate(db_session, installation, automated_task_id=task.id)

    count = _reconcile_proactive_outcomes(
        db_session, installation_id=installation.id, now=NOW
    )
    db_session.flush()

    assert count == 0
    db_session.refresh(candidate)
    assert candidate.task_status is None
    assert candidate.task_finished_at is None
    assert len(_events_for(db_session, candidate)) == 0


def test_reconcile_already_reconciled_skipped(db_session: Session) -> None:
    """Candidate that already has task_status set is skipped — idempotent."""
    installation = _make_installation(db_session)
    task = _make_task(db_session, installation, status=TaskStatus.succeeded)
    candidate = _make_candidate(
        db_session,
        installation,
        automated_task_id=task.id,
        task_status="succeeded",  # already reconciled
    )

    count = _reconcile_proactive_outcomes(
        db_session, installation_id=installation.id, now=NOW
    )
    db_session.flush()

    assert count == 0
    # No duplicate events written
    assert len(_events_for(db_session, candidate)) == 0


def test_reconcile_no_task_id_ignored(db_session: Session) -> None:
    """Candidate with automated_task_id=None is not touched by the reconciler."""
    installation = _make_installation(db_session)
    candidate = _make_candidate(
        db_session,
        installation,
        automated_task_id=None,
        status="candidate",
    )

    count = _reconcile_proactive_outcomes(
        db_session, installation_id=installation.id, now=NOW
    )
    db_session.flush()

    assert count == 0
    db_session.refresh(candidate)
    assert candidate.task_status is None
    assert len(_events_for(db_session, candidate)) == 0


# ---------------------------------------------------------------------------
# run_hygiene integration: outcomes_reconciled counter
# ---------------------------------------------------------------------------


def test_run_hygiene_includes_outcomes_reconciled(db_session: Session) -> None:
    """run_hygiene returns HygieneCounters.outcomes_reconciled for reconciled rows."""
    installation = _make_installation(db_session)
    task = _make_task(db_session, installation, status=TaskStatus.succeeded)
    _make_candidate(db_session, installation, automated_task_id=task.id)

    counters = run_hygiene(db_session, installation_id=installation.id, now=NOW)

    assert counters.outcomes_reconciled == 1
    payload = counters.to_payload()
    assert "outcomes_reconciled" in payload
    assert payload["outcomes_reconciled"] == 1


# ---------------------------------------------------------------------------
# Migration smoke test
# ---------------------------------------------------------------------------


def test_migration_columns_present(db_session: Session) -> None:
    """SELECT the two new columns to verify migration 0053 applied correctly."""
    result = db_session.execute(
        text(
            "SELECT task_status, task_finished_at "
            "FROM witness_opportunity_candidates LIMIT 0"
        )
    )
    # If columns are missing the above query raises; getting here confirms they exist.
    assert result is not None
