import os
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, delete, select
from sqlalchemy.orm import Session

from kortny.db.models import (
    Installation,
    LLMUsage,
    Schedule,
    Task,
    TaskEvent,
    TaskEventType,
)
from kortny.db.session import make_engine, make_session_factory, normalize_database_url
from kortny.scheduler import ScheduleMaterializer
from kortny.tasks import TaskService

TEST_POSTGRES_URL = os.environ.get("KORTNY_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    TEST_POSTGRES_URL is None,
    reason="KORTNY_TEST_POSTGRES_URL is required for scheduler integration tests",
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


def test_materializer_turns_due_oneoff_schedule_into_pending_task(
    db_session: Session,
) -> None:
    now = datetime(2026, 6, 4, 9, 30, tzinfo=UTC)
    schedule = create_schedule(
        db_session,
        spec_kind="oneoff",
        run_at=now - timedelta(seconds=30),
        next_run_at=now - timedelta(seconds=30),
        planned_cost_ceiling_usd=Decimal("0.2500"),
    )
    db_session.commit()

    results = ScheduleMaterializer(db_session).materialize_due_schedules(
        now=now,
        use_advisory_lock=False,
    )
    db_session.commit()

    assert len(results) == 1
    assert results[0].action == "materialized"
    assert results[0].task_id is not None

    db_session.refresh(schedule)
    assert schedule.status == "completed"
    assert schedule.next_run_at is None
    assert schedule.last_run_at == now - timedelta(seconds=30)

    task = db_session.get(Task, results[0].task_id)
    assert task is not None
    assert task.status == "pending"
    assert task.available_at == now
    assert task.identity_kind == "scheduled"
    assert task.identity_payload["schedule_id"] == str(schedule.id)
    assert (
        task.identity_payload["fire_time"] == (now - timedelta(seconds=30)).isoformat()
    )
    assert task.identity_payload["planned_cost_ceiling_usd"] == "0.2500"
    assert task.identity_payload["delivery_kind"] == "slack_dm"
    assert task.identity_payload["artifact_delivery_policy"] == "message_only"
    assert task.slack_channel_id == "DTestUser"
    assert task.slack_user_id == "UTestUser"
    assert task.slack_thread_ts == "DTestUser"

    materialized_event = db_session.scalar(
        select(TaskEvent).where(
            TaskEvent.task_id == task.id,
            TaskEvent.type == TaskEventType.log,
            TaskEvent.payload["message"].as_string() == "scheduled_task_materialized",
        )
    )
    assert materialized_event is not None
    assert materialized_event.payload["schedule_id"] == str(schedule.id)
    assert materialized_event.payload["delivery_kind"] == "slack_dm"
    admitted_event = db_session.scalar(
        select(TaskEvent).where(
            TaskEvent.task_id == task.id,
            TaskEvent.type == TaskEventType.log,
            TaskEvent.payload["message"].as_string()
            == "scheduled_task_budget_admitted",
        )
    )
    assert admitted_event is not None
    assert admitted_event.payload["cost_ceiling_usd"] == "0.2500"


def test_materializer_pauses_schedule_without_run_budget(
    db_session: Session,
) -> None:
    now = datetime(2026, 6, 4, 9, 30, tzinfo=UTC)
    schedule = create_schedule(
        db_session,
        spec_kind="oneoff",
        run_at=now - timedelta(seconds=30),
        next_run_at=now - timedelta(seconds=30),
        planned_cost_ceiling_usd=None,
    )
    db_session.commit()

    results = ScheduleMaterializer(db_session).materialize_due_schedules(
        now=now,
        use_advisory_lock=False,
    )
    db_session.commit()

    assert len(results) == 1
    assert results[0].action == "paused"
    assert results[0].task_id is None
    assert results[0].reason == "missing_planned_cost_ceiling"
    db_session.refresh(schedule)
    assert schedule.status == "paused"
    assert schedule.metadata_json["last_budget_status"] == "admission_failed"
    assert (
        db_session.scalar(select(Task).where(Task.identity_kind == "scheduled")) is None
    )


def test_materializer_uses_channel_root_delivery_contract(
    db_session: Session,
) -> None:
    now = datetime(2026, 6, 4, 9, 30, tzinfo=UTC)
    schedule = create_schedule(
        db_session,
        spec_kind="oneoff",
        run_at=now - timedelta(seconds=30),
        next_run_at=now - timedelta(seconds=30),
    )
    schedule.delivery_kind = "slack_channel"
    schedule.delivery_slack_channel_id = "CMarket"
    schedule.delivery_slack_thread_ts = None
    schedule.task_template = {
        **dict(schedule.task_template),
        "slack_channel_id": "CMarket",
        "slack_thread_ts": None,
        "delivery_surface": "channel",
    }
    db_session.commit()

    results = ScheduleMaterializer(db_session).materialize_due_schedules(
        now=now,
        use_advisory_lock=False,
    )
    db_session.commit()

    task = db_session.get(Task, results[0].task_id)
    assert task is not None
    assert task.slack_channel_id == "CMarket"
    assert task.slack_thread_ts is None
    assert task.identity_payload["delivery_kind"] == "slack_channel"


def test_scheduled_task_budget_breach_pauses_future_runs(
    db_session: Session,
) -> None:
    now = datetime(2026, 6, 4, 9, 30, tzinfo=UTC)
    schedule = create_schedule(
        db_session,
        spec_kind="interval",
        interval_seconds=60,
        next_run_at=now - timedelta(seconds=30),
        planned_cost_ceiling_usd=Decimal("0.0001"),
    )
    db_session.commit()
    results = ScheduleMaterializer(db_session).materialize_due_schedules(
        now=now,
        use_advisory_lock=False,
    )
    db_session.commit()
    task = db_session.get(Task, results[0].task_id)
    assert task is not None

    service = TaskService(db_session)
    service.record_llm_usage(
        task,
        provider="openrouter",
        model="test/model",
        model_tier="standard",
        input_tokens=10,
        output_tokens=5,
        cost_usd=Decimal("0.0002"),
    )
    db_session.commit()

    db_session.refresh(schedule)
    assert schedule.status == "paused"
    assert schedule.metadata_json["last_budget_status"] == "exceeded"
    events = tuple(
        db_session.scalars(
            select(TaskEvent).where(
                TaskEvent.task_id == task.id,
                TaskEvent.payload["message"].as_string()
                == "scheduled_task_cost_ceiling_exceeded",
            )
        )
    )
    assert len(events) == 1
    assert events[0].payload["cost_ceiling_usd"] == "0.0001"
    assert events[0].payload["cumulative_cost_usd"] == "0.000200"
    assert events[0].payload["schedule_paused"] is True


def test_materializer_skips_stale_oneoff_when_catchup_window_elapsed(
    db_session: Session,
) -> None:
    now = datetime(2026, 6, 4, 9, 30, tzinfo=UTC)
    schedule = create_schedule(
        db_session,
        spec_kind="oneoff",
        run_at=now - timedelta(hours=1),
        next_run_at=now - timedelta(hours=1),
        catchup_window_seconds=60,
    )
    db_session.commit()

    results = ScheduleMaterializer(db_session).materialize_due_schedules(
        now=now,
        use_advisory_lock=False,
    )
    db_session.commit()

    assert len(results) == 1
    assert results[0].action == "skipped"
    assert results[0].reason == "missed_catchup_window"
    assert results[0].task_id is None

    db_session.refresh(schedule)
    assert schedule.status == "completed"
    assert schedule.next_run_at is None
    assert schedule.last_run_at == now - timedelta(hours=1)
    assert db_session.scalar(select(Task.id)) is None


def test_materializer_advances_interval_schedule_to_next_future_fire(
    db_session: Session,
) -> None:
    now = datetime(2026, 6, 4, 9, 30, tzinfo=UTC)
    due_at = now - timedelta(seconds=10)
    schedule = create_schedule(
        db_session,
        spec_kind="interval",
        interval_seconds=60,
        next_run_at=due_at,
    )
    db_session.commit()

    results = ScheduleMaterializer(db_session).materialize_due_schedules(
        now=now,
        use_advisory_lock=False,
    )
    db_session.commit()

    assert len(results) == 1
    assert results[0].action == "materialized"

    db_session.refresh(schedule)
    assert schedule.status == "active"
    assert schedule.last_run_at == due_at
    assert schedule.next_run_at == due_at + timedelta(seconds=60)


def test_materializer_runs_simple_weekly_cron_schedule(
    db_session: Session,
) -> None:
    now = datetime(2026, 6, 8, 9, 5, tzinfo=UTC)
    due_at = datetime(2026, 6, 8, 9, 0, tzinfo=UTC)
    schedule = create_schedule(
        db_session,
        spec_kind="cron",
        cron_expr="0 9 * * 1",
        next_run_at=due_at,
    )
    db_session.commit()

    results = ScheduleMaterializer(db_session).materialize_due_schedules(
        now=now,
        use_advisory_lock=False,
    )
    db_session.commit()

    assert len(results) == 1
    assert results[0].action == "materialized"

    db_session.refresh(schedule)
    assert schedule.status == "active"
    assert schedule.last_run_at == due_at
    assert schedule.next_run_at == datetime(2026, 6, 15, 9, 0, tzinfo=UTC)


def test_materializer_runs_weekday_range_cron_schedule(
    db_session: Session,
) -> None:
    now = datetime(2026, 6, 5, 13, 5, tzinfo=UTC)
    due_at = datetime(2026, 6, 5, 13, 0, tzinfo=UTC)
    schedule = create_schedule(
        db_session,
        spec_kind="cron",
        cron_expr="0 8 * * 1-5",
        timezone="America/Chicago",
        next_run_at=due_at,
    )
    db_session.commit()

    results = ScheduleMaterializer(db_session).materialize_due_schedules(
        now=now,
        use_advisory_lock=False,
    )
    db_session.commit()

    assert len(results) == 1
    assert results[0].action == "materialized"

    db_session.refresh(schedule)
    assert schedule.status == "active"
    assert schedule.last_run_at == due_at
    assert schedule.next_run_at == datetime(2026, 6, 8, 13, 0, tzinfo=UTC)


def cleanup_database(session: Session) -> None:
    for model in (LLMUsage, TaskEvent, Task, Schedule, Installation):
        session.execute(delete(model))


def create_installation(session: Session) -> Installation:
    installation = Installation(slack_team_id=f"T{uuid.uuid4().hex}")
    session.add(installation)
    session.flush()
    return installation


def create_schedule(
    session: Session,
    *,
    spec_kind: str,
    run_at: datetime | None = None,
    interval_seconds: int | None = None,
    cron_expr: str | None = None,
    next_run_at: datetime,
    timezone: str = "UTC",
    catchup_window_seconds: int | None = None,
    planned_cost_ceiling_usd: Decimal | None = Decimal("0.2500"),
) -> Schedule:
    installation = create_installation(session)
    schedule = Schedule(
        installation_id=installation.id,
        owner_type="user",
        owner_slack_user_id="UTestUser",
        title="Weekly unresolved decision check",
        spec_kind=spec_kind,
        cron_expr=cron_expr,
        run_at=run_at,
        interval_seconds=interval_seconds,
        timezone=timezone,
        next_run_at=next_run_at,
        catchup_policy="skip",
        catchup_window_seconds=catchup_window_seconds,
        overlap_policy="skip",
        status="active",
        delivery_kind="slack_dm",
        delivery_slack_user_id="UTestUser",
        delivery_slack_channel_id="DTestUser",
        delivery_slack_thread_ts="DTestUser",
        artifact_delivery_policy="message_only",
        task_template={
            "input": "check unresolved decisions I was involved in",
            "slack_channel_id": "DTestUser",
            "slack_user_id": "UTestUser",
            "slack_thread_ts": "DTestUser",
            "delivery_surface": "dm",
            "artifact_delivery_policy": "message_only",
        },
        planned_cost_ceiling_usd=planned_cost_ceiling_usd,
        created_by_slack_user_id="UTestUser",
        metadata_json={},
    )
    session.add(schedule)
    session.flush()
    return schedule
