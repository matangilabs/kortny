import os
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from decimal import Decimal

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, delete, select
from sqlalchemy.orm import Session

from kortny.db.models import Installation, Schedule, Task, TaskEvent, TaskEventType
from kortny.db.session import make_engine, make_session_factory, normalize_database_url
from kortny.tasks import TaskService
from kortny.tools.schedules import (
    CancelScheduleTool,
    CreateScheduleTool,
    ListSchedulesTool,
    PauseScheduleTool,
    ResumeScheduleTool,
    UpdateScheduleTool,
)
from kortny.tools.types import RecoverableToolError

TEST_POSTGRES_URL = os.environ.get("KORTNY_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    TEST_POSTGRES_URL is None,
    reason="KORTNY_TEST_POSTGRES_URL is required for schedule tool tests",
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


def test_list_schedules_reads_scheduler_truth_for_current_user(
    db_session: Session,
) -> None:
    task = create_task(db_session)
    schedule = create_schedule(
        db_session,
        installation_id=task.installation_id,
        owner_user_id=task.slack_user_id,
        delivery_channel_id=task.slack_channel_id,
        delivery_user_id=task.slack_user_id,
        title="Daily market update",
        task_input="send a stock market update",
        next_run_at=datetime(2026, 6, 8, 13, 0, tzinfo=UTC),
        timezone="America/Chicago",
    )
    create_schedule(
        db_session,
        installation_id=task.installation_id,
        owner_user_id="UOther",
        delivery_channel_id="DOther",
        delivery_user_id="UOther",
        title="Private other user schedule",
        task_input="private",
        next_run_at=datetime(2026, 6, 8, 13, 0, tzinfo=UTC),
    )
    db_session.commit()

    result = ListSchedulesTool(session=db_session, task=task).invoke(
        {"query": "market", "status": "active"}
    )

    assert result.output["successful"] is True
    assert result.output["count"] == 1
    assert result.output["schedules"][0]["id"] == str(schedule.id)
    assert result.output["schedules"][0]["status"] == "active"
    assert result.output["schedules"][0]["delivery"]["label"] == "this DM"
    assert "I found 1 schedule" in result.output["assistant_summary"]
    assert "Next run" not in result.output["assistant_summary"]


def test_create_schedule_tool_creates_active_humanized_schedule(
    db_session: Session,
) -> None:
    task = create_task(db_session)

    result = CreateScheduleTool(session=db_session, task=task).invoke(
        {
            "request": (
                "Every morning at 8AM central time send me a stock market update"
            ),
            "timezone": "America/Chicago",
        }
    )
    db_session.commit()

    schedule = db_session.scalar(select(Schedule))
    assert schedule is not None
    assert schedule.status == "active"
    assert schedule.cron_expr == "0 8 * * *"
    assert schedule.timezone == "America/Chicago"
    assert schedule.task_template["input"] == "send me a stock market update"
    assert result.output["action"] == "created"
    assert "Done, I'll take care of that" in result.output["assistant_summary"]
    assert "Schedule id" not in result.output["assistant_summary"]
    events = tuple(
        db_session.scalars(
            select(TaskEvent).where(
                TaskEvent.task_id == task.id,
                TaskEvent.type == TaskEventType.log,
            )
        )
    )
    assert any(event.payload.get("message") == "schedule_created" for event in events)


def test_schedule_mutation_tools_require_current_user_ownership(
    db_session: Session,
) -> None:
    task = create_task(db_session)
    schedule = create_schedule(
        db_session,
        installation_id=task.installation_id,
        owner_user_id="UOther",
        delivery_channel_id=task.slack_channel_id,
        delivery_user_id=task.slack_user_id,
        title="Channel-visible schedule",
        task_input="post a channel update",
        next_run_at=datetime(2026, 6, 8, 13, 0, tzinfo=UTC),
    )
    db_session.commit()

    with pytest.raises(RecoverableToolError) as exc_info:
        PauseScheduleTool(session=db_session, task=task).invoke(
            {"schedule_id": str(schedule.id)}
        )

    assert exc_info.value.code == "schedule_not_owned"


def test_pause_resume_and_cancel_schedule_updates_status_and_events(
    db_session: Session,
) -> None:
    task = create_task(db_session)
    schedule = create_schedule(
        db_session,
        installation_id=task.installation_id,
        owner_user_id=task.slack_user_id,
        delivery_channel_id=task.slack_channel_id,
        delivery_user_id=task.slack_user_id,
        title="Daily market update",
        task_input="send a stock market update",
        next_run_at=datetime(2026, 6, 8, 13, 0, tzinfo=UTC),
    )
    db_session.commit()

    pause = PauseScheduleTool(session=db_session, task=task).invoke(
        {"schedule_id": str(schedule.id)}
    )
    assert pause.output["action"] == "paused"
    assert schedule.status == "paused"
    assert "paused" in pause.output["assistant_summary"]

    resume = ResumeScheduleTool(session=db_session, task=task).invoke(
        {"schedule_id": str(schedule.id)}
    )
    assert resume.output["action"] == "resumed"
    assert schedule.status == "active"
    assert "resumed" in resume.output["assistant_summary"]

    cancel = CancelScheduleTool(session=db_session, task=task).invoke(
        {"schedule_id": str(schedule.id)}
    )
    db_session.commit()

    assert cancel.output["action"] == "cancelled"
    assert schedule.status == "cancelled"
    assert schedule.next_run_at is None
    event_messages = {
        event.payload.get("message")
        for event in db_session.scalars(
            select(TaskEvent).where(TaskEvent.task_id == task.id)
        )
    }
    assert {
        "schedule_paused",
        "schedule_resumed",
        "schedule_cancelled",
    } <= event_messages


def test_update_schedule_tool_changes_cadence_without_losing_task_body(
    db_session: Session,
) -> None:
    task = create_task(db_session)
    schedule = create_schedule(
        db_session,
        installation_id=task.installation_id,
        owner_user_id=task.slack_user_id,
        delivery_channel_id=task.slack_channel_id,
        delivery_user_id=task.slack_user_id,
        title="Daily market update",
        task_input="send a stock market update",
        next_run_at=datetime(2026, 6, 8, 13, 0, tzinfo=UTC),
        timezone="America/Chicago",
    )
    db_session.commit()

    result = UpdateScheduleTool(session=db_session, task=task).invoke(
        {
            "schedule_id": str(schedule.id),
            "update_request": "Change this schedule to every Friday morning",
        }
    )
    db_session.commit()

    assert result.output["action"] == "updated"
    assert schedule.cron_expr == "0 9 * * 5"
    assert schedule.task_template["input"] == "send a stock market update"
    assert schedule.metadata_json["cadence_label"] == "Every Friday morning"
    assert "updated" in result.output["assistant_summary"]


def test_cancelled_schedule_cannot_be_mutated_by_exact_id(
    db_session: Session,
) -> None:
    task = create_task(db_session)
    schedule = create_schedule(
        db_session,
        installation_id=task.installation_id,
        owner_user_id=task.slack_user_id,
        delivery_channel_id=task.slack_channel_id,
        delivery_user_id=task.slack_user_id,
        title="Cancelled market update",
        task_input="send a stock market update",
        next_run_at=datetime(2026, 6, 8, 13, 0, tzinfo=UTC),
        status="cancelled",
    )
    db_session.commit()

    with pytest.raises(RecoverableToolError) as exc_info:
        UpdateScheduleTool(session=db_session, task=task).invoke(
            {
                "schedule_id": str(schedule.id),
                "update_request": "Change it to Fridays",
            }
        )

    assert exc_info.value.code == "schedule_not_mutable"


def cleanup_database(session: Session) -> None:
    for model in (TaskEvent, Task, Schedule, Installation):
        session.execute(delete(model))


def create_task(session: Session) -> Task:
    installation = Installation(slack_team_id=f"T{uuid.uuid4().hex}")
    session.add(installation)
    session.flush()
    return TaskService(session).create_task(
        installation_id=installation.id,
        slack_event_id=f"Ev{uuid.uuid4().hex}",
        slack_channel_id="DUser",
        slack_thread_ts="DUser",
        slack_message_ts="1780200000.000001",
        slack_user_id="UUser",
        input="Check my schedules.",
    )


def create_schedule(
    session: Session,
    *,
    installation_id: uuid.UUID,
    owner_user_id: str,
    delivery_channel_id: str,
    delivery_user_id: str,
    title: str,
    task_input: str,
    next_run_at: datetime,
    timezone: str = "UTC",
    status: str = "active",
) -> Schedule:
    schedule = Schedule(
        installation_id=installation_id,
        owner_type="user",
        owner_slack_user_id=owner_user_id,
        title=title,
        spec_kind="cron",
        cron_expr="0 8 * * *",
        timezone=timezone,
        next_run_at=next_run_at,
        catchup_policy="skip",
        catchup_window_seconds=300,
        overlap_policy="skip",
        status=status,
        delivery_kind="slack_dm",
        delivery_slack_user_id=delivery_user_id,
        delivery_slack_channel_id=delivery_channel_id,
        delivery_slack_thread_ts=delivery_channel_id,
        artifact_delivery_policy="message_only",
        task_template={
            "input": task_input,
            "slack_channel_id": delivery_channel_id,
            "slack_user_id": delivery_user_id,
            "slack_thread_ts": delivery_channel_id,
            "delivery_surface": "dm",
            "artifact_delivery_policy": "message_only",
        },
        planned_cost_ceiling_usd=Decimal("0.2500"),
        created_by_slack_user_id=owner_user_id,
        metadata_json={"cadence_label": "Every morning at 8:00 AM Central"},
    )
    session.add(schedule)
    session.flush()
    return schedule
