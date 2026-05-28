import os
import uuid
from collections.abc import Iterator

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, delete, func, select
from sqlalchemy.orm import Session

from kortny.db.models import (
    Installation,
    ObserveChannelProfile,
    SlackChannelMembership,
    Task,
    TaskEvent,
    TaskEventType,
)
from kortny.db.session import make_engine, make_session_factory, normalize_database_url
from kortny.observe.profiles import ObserveChannelProfileService
from kortny.tasks import TaskService

TEST_POSTGRES_URL = os.environ.get("KORTNY_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    TEST_POSTGRES_URL is None,
    reason="KORTNY_TEST_POSTGRES_URL is required for observe profile tests",
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


def test_channel_profile_upserts_from_assessment_tool_result(
    db_session: Session,
) -> None:
    task, membership = create_assessment_task(db_session)
    task_service = TaskService(db_session)
    task_service.append_event(
        task,
        TaskEventType.tool_result,
        {
            "tool": "slack_channel_history",
            "tool_call_id": "call-history",
            "output": {
                "channel_id": "CObserve",
                "message_count": 3,
                "messages": [
                    {
                        "ts": "1779900000.000001",
                        "user": "U1",
                        "text": "Morning blotter uploaded.",
                        "files": [{"id": "F1", "name": "blotter.csv"}],
                    },
                    {
                        "ts": "1779900100.000002",
                        "user": "U2",
                        "text": "Need a review on ticker changes.",
                    },
                    {
                        "ts": "1779900200.000003",
                        "user": "U1",
                        "text": "Second file.",
                        "files": [{"id": "F2", "name": "notes.pdf"}],
                    },
                ],
            },
            "cost_usd": "0",
            "artifacts": [],
        },
    )
    db_session.commit()

    profile = ObserveChannelProfileService(db_session).upsert_from_assessment(
        task=task,
        membership=membership,
        result_summary="This channel is used for trade blotter review.",
    )
    db_session.commit()

    assert profile.channel_id == "CObserve"
    assert profile.profile_status == "active"
    assert profile.profile_version == 1
    assert profile.summary == "This channel is used for trade blotter review."
    assert profile.message_count == 3
    assert profile.file_count == 2
    assert profile.observed_range_start_ts == "1779900000.000001"
    assert profile.observed_range_end_ts == "1779900200.000003"
    assert profile.last_scanned_message_ts == "1779900200.000003"
    assert profile.source_task_id == task.id
    assert profile.fresh_window_days == 30
    assert profile.archive_window_days == 365
    assert profile.profile_json["archive_context"]["window_days"] == 365
    assert profile.assumptions_json[0]["confidence"] == "low"
    assert profile.evidence_refs_json[-1]["tool"] == "slack_channel_history"
    assert profile.metadata_json["membership_id"] == str(membership.id)

    updated = ObserveChannelProfileService(db_session).upsert_from_assessment(
        task=task,
        membership=membership,
        result_summary="Updated channel profile.",
    )
    db_session.commit()

    profile_count = db_session.scalar(
        select(func.count()).select_from(ObserveChannelProfile)
    )
    assert updated.id == profile.id
    assert updated.profile_version == 2
    assert updated.summary == "Updated channel profile."
    assert profile_count == 1


def test_channel_profile_handles_missing_history_result(
    db_session: Session,
) -> None:
    task, membership = create_assessment_task(db_session, channel_id="CEmpty")
    db_session.commit()

    profile = ObserveChannelProfileService(db_session).upsert_from_assessment(
        task=task,
        membership=membership,
        result_summary="Not enough history yet.",
    )
    db_session.commit()

    assert profile.channel_id == "CEmpty"
    assert profile.message_count == 0
    assert profile.file_count == 0
    assert profile.observed_range_start_ts is None
    assert profile.observed_range_end_ts == task.slack_message_ts
    assert profile.last_scanned_message_ts == task.slack_message_ts
    assert profile.confidence_reason is not None
    assert "No channel-history tool result" in profile.confidence_reason


def cleanup_database(session: Session) -> None:
    for model in (
        ObserveChannelProfile,
        SlackChannelMembership,
        TaskEvent,
        Task,
        Installation,
    ):
        session.execute(delete(model))


def create_assessment_task(
    session: Session,
    *,
    channel_id: str = "CObserve",
) -> tuple[Task, SlackChannelMembership]:
    installation = Installation(slack_team_id=f"T{uuid.uuid4().hex}")
    session.add(installation)
    session.flush()
    membership = SlackChannelMembership(
        installation_id=installation.id,
        channel_id=channel_id,
        channel_name="observe",
        channel_type="public_channel",
        membership_status="active",
        discovered_via="member_joined_channel",
        added_by_user_id="UInvite",
        onboarding_status="posted",
        onboarding_message_ts="1779900000.000000",
        metadata_json={},
    )
    session.add(membership)
    session.flush()
    task = TaskService(session).create_task(
        installation_id=installation.id,
        slack_event_id=f"Ev{uuid.uuid4().hex}",
        slack_channel_id=channel_id,
        slack_thread_ts="1779900000.000000",
        slack_message_ts="1779900000.000000",
        slack_user_id="UInvite",
        input="Run Kortny's channel onboarding assessment.",
    )
    return task, membership
