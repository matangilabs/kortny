import os
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, delete, select
from sqlalchemy.orm import Session

from kortny.db.models import (
    Artifact,
    EncryptedSecret,
    Installation,
    LLMUsage,
    ModelPricing,
    SlackSideEffect,
    Task,
    TaskEvent,
    TaskEventType,
)
from kortny.db.session import make_engine, make_session_factory, normalize_database_url
from kortny.slack import SlackPoster, SlackThread
from kortny.slack.outbox import (
    SLACK_EFFECT_FAILED,
    SLACK_EFFECT_IN_PROGRESS,
    SlackSideEffectOutbox,
    slack_message_key,
)
from kortny.tasks import TaskIdentity, TaskService

TEST_POSTGRES_URL = os.environ.get("KORTNY_TEST_POSTGRES_URL")


class FakeSlackClient:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []
        self.uploads: list[dict[str, Any]] = []
        self.unfurl_flags: list[tuple[bool, bool]] = []

    def chat_postMessage(
        self,
        *,
        channel: str,
        text: str,
        thread_ts: str | None = None,
        blocks: list[dict[str, Any]] | None = None,
        unfurl_links: bool = True,
        unfurl_media: bool = True,
    ) -> dict[str, Any]:
        message: dict[str, Any] = {
            "channel": channel,
            "text": text,
            "thread_ts": thread_ts,
        }
        if blocks is not None:
            message["blocks"] = blocks
        self.messages.append(message)
        self.unfurl_flags.append((unfurl_links, unfurl_media))
        return {"ok": True, "ts": f"1716400001.{len(self.messages):06d}"}

    def files_upload_v2(
        self,
        *,
        file: str,
        filename: str | None = None,
        title: str | None = None,
        channel: str | None = None,
        initial_comment: str | None = None,
        thread_ts: str | None = None,
    ) -> dict[str, Any]:
        self.uploads.append(
            {
                "file": file,
                "filename": filename,
                "title": title,
                "channel": channel,
                "initial_comment": initial_comment,
                "thread_ts": thread_ts,
            }
        )
        return {"ok": True, "files": [{"id": f"F{len(self.uploads):06d}"}]}


class FakeSlackSdkResponse:
    """Slack SDK responses expose data but are not plain dict mappings."""

    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data


class FakeSlackSdkResponseClient(FakeSlackClient):
    def chat_postMessage(  # type: ignore[override]
        self,
        *,
        channel: str,
        text: str,
        thread_ts: str | None = None,
        blocks: list[dict[str, Any]] | None = None,
        unfurl_links: bool = True,
        unfurl_media: bool = True,
    ) -> FakeSlackSdkResponse:
        message: dict[str, Any] = {
            "channel": channel,
            "text": text,
            "thread_ts": thread_ts,
        }
        if blocks is not None:
            message["blocks"] = blocks
        self.messages.append(message)
        self.unfurl_flags.append((unfurl_links, unfurl_media))
        return FakeSlackSdkResponse(
            {"ok": True, "ts": f"1716400001.{len(self.messages):06d}"}
        )


@pytest.fixture(scope="session")
def engine() -> Iterator[Engine]:
    if TEST_POSTGRES_URL is None:
        pytest.skip("KORTNY_TEST_POSTGRES_URL is required for Slack posting tests")

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


def test_post_message_posts_to_thread_and_logs_event(db_session: Session) -> None:
    task = create_task(db_session)
    client = FakeSlackClient()
    message_ts = SlackPoster(session=db_session, client=client).post_message(
        SlackThread.from_task(task),
        "Done.",
    )

    event = db_session.scalar(
        select(TaskEvent).where(
            TaskEvent.task_id == task.id,
            TaskEvent.type == TaskEventType.message_posted,
        )
    )

    assert message_ts == "1716400001.000001"
    assert client.messages == [
        {
            "channel": "C123",
            "text": "Done.",
            "thread_ts": "1716400000.000001",
        }
    ]
    assert event is not None
    assert event.payload == {
        "channel": "C123",
        "thread_ts": "1716400000.000001",
        "message_ts": "1716400001.000001",
        "text": "Done.",
        "purpose": "result",
        "slack_side_effect_id": event.payload["slack_side_effect_id"],
        "idempotency_key": f"slack:message:{task.id}:result",
    }
    side_effect = db_session.scalar(
        select(SlackSideEffect).where(SlackSideEffect.task_id == task.id)
    )
    assert side_effect is not None
    assert side_effect.status == "succeeded"
    assert side_effect.operation == "chat_postMessage"
    assert side_effect.idempotency_key == f"slack:message:{task.id}:result"


def test_post_message_allows_scheduled_channel_root_delivery(
    db_session: Session,
) -> None:
    task = create_task(
        db_session,
        channel_id="C123",
        thread_ts=None,
    )
    task.identity_kind = "scheduled"
    db_session.flush()
    client = FakeSlackClient()

    message_ts = SlackPoster(session=db_session, client=client).post_message(
        SlackThread.from_task(task),
        "Scheduled update.",
    )

    assert message_ts == "1716400001.000001"
    assert client.messages == [
        {
            "channel": "C123",
            "text": "Scheduled update.",
            "thread_ts": None,
        }
    ]


def test_post_message_allows_witness_autopilot_root_delivery(
    db_session: Session,
) -> None:
    task = create_task(
        db_session,
        channel_id="C123",
        thread_ts=None,
    )
    task.identity_kind = "synthetic"
    task.identity_payload = {"source": "witness_autopilot"}
    db_session.flush()
    client = FakeSlackClient()

    message_ts = SlackPoster(session=db_session, client=client).post_message(
        SlackThread.from_task(task),
        "Proactive update.",
    )

    assert message_ts == "1716400001.000001"
    assert client.messages == [
        {
            "channel": "C123",
            "text": "Proactive update.",
            "thread_ts": None,
        }
    ]


def test_post_message_records_blocks_in_outbox_and_event(db_session: Session) -> None:
    task = create_task(db_session)
    client = FakeSlackClient()
    blocks = [
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Pause"},
                    "action_id": "kortny_schedule_pause",
                    "value": "schedule-id",
                }
            ],
        }
    ]

    SlackPoster(session=db_session, client=client).post_message(
        SlackThread.from_task(task),
        "Done.",
        purpose="schedule_created",
        blocks=blocks,
    )

    event = db_session.scalar(
        select(TaskEvent).where(
            TaskEvent.task_id == task.id,
            TaskEvent.type == TaskEventType.message_posted,
        )
    )
    side_effect = db_session.scalar(
        select(SlackSideEffect).where(SlackSideEffect.task_id == task.id)
    )

    assert client.messages[0]["blocks"] == blocks
    assert event is not None
    assert event.payload["blocks"] == blocks
    assert side_effect is not None
    assert side_effect.request_json["blocks"] == blocks


def test_post_message_in_dm_posts_without_thread_ts(db_session: Session) -> None:
    task = create_task(db_session, channel_id="D123")
    client = FakeSlackClient()

    SlackPoster(session=db_session, client=client).post_message(
        SlackThread.from_task(task),
        "Done.",
    )

    event = db_session.scalar(
        select(TaskEvent)
        .where(
            TaskEvent.task_id == task.id,
            TaskEvent.type == TaskEventType.message_posted,
        )
        .order_by(TaskEvent.seq.desc())
        .limit(1)
    )

    assert client.messages == [
        {
            "channel": "D123",
            "text": "Done.",
            "thread_ts": None,
        }
    ]
    assert event is not None
    assert event.payload["thread_ts"] is None


def test_post_message_normalizes_slack_mrkdwn(db_session: Session) -> None:
    task = create_task(db_session)
    client = FakeSlackClient()

    SlackPoster(session=db_session, client=client).post_message(
        SlackThread.from_task(task),
        "### Capabilities\n1. **Web Searches:** Read [docs](https://docs.slack.dev).",
    )

    event = db_session.scalar(
        select(TaskEvent)
        .where(
            TaskEvent.task_id == task.id,
            TaskEvent.type == TaskEventType.message_posted,
        )
        .order_by(TaskEvent.seq.desc())
        .limit(1)
    )

    expected_text = (
        "*Capabilities*\n1. *Web Searches:* Read <https://docs.slack.dev|docs>."
    )
    assert client.messages[0]["text"] == expected_text
    assert event is not None
    assert event.payload["text"] == expected_text


def test_post_message_normalizes_em_dash_at_slack_boundary(
    db_session: Session,
) -> None:
    task = create_task(db_session)
    client = FakeSlackClient()
    em_dash = chr(0x2014)

    SlackPoster(session=db_session, client=client).post_message(
        SlackThread.from_task(task),
        f"Use this {em_dash} not that.\n```txt\nkeep {em_dash} in code\n```",
    )

    event = db_session.scalar(
        select(TaskEvent)
        .where(
            TaskEvent.task_id == task.id,
            TaskEvent.type == TaskEventType.message_posted,
        )
        .order_by(TaskEvent.seq.desc())
        .limit(1)
    )

    expected_text = "Use this - not that.\n```txt\nkeep \u2014 in code\n```"
    assert client.messages[0]["text"] == expected_text
    assert event is not None
    assert event.payload["text"] == expected_text


def test_post_message_normalizes_block_text_at_slack_boundary(
    db_session: Session,
) -> None:
    task = create_task(db_session)
    client = FakeSlackClient()
    em_dash = chr(0x2014)
    blocks = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"**Decision** {em_dash} ship it.",
            },
            "accessory": {
                "type": "button",
                "text": {"type": "plain_text", "text": f"Approve {em_dash} now"},
                "value": f"internal{em_dash}value",
            },
        }
    ]

    SlackPoster(session=db_session, client=client).post_message(
        SlackThread.from_task(task),
        "Decision ready.",
        blocks=blocks,
    )

    event = db_session.scalar(
        select(TaskEvent)
        .where(
            TaskEvent.task_id == task.id,
            TaskEvent.type == TaskEventType.message_posted,
        )
        .order_by(TaskEvent.seq.desc())
        .limit(1)
    )

    expected_blocks = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*Decision* - ship it.",
            },
            "accessory": {
                "type": "button",
                "text": {"type": "plain_text", "text": "Approve - now"},
                "value": f"internal{em_dash}value",
            },
        }
    ]
    assert client.messages[0]["blocks"] == expected_blocks
    assert event is not None
    assert event.payload["blocks"] == expected_blocks


def test_post_message_reuses_successful_side_effect(
    db_session: Session,
) -> None:
    task = create_task(db_session)
    client = FakeSlackClient()
    poster = SlackPoster(session=db_session, client=client)
    thread = SlackThread.from_task(task)

    first_ts = poster.post_message(thread, "Done.")
    second_ts = poster.post_message(thread, "Done.")

    events = list(
        db_session.scalars(
            select(TaskEvent).where(
                TaskEvent.task_id == task.id,
                TaskEvent.type == TaskEventType.message_posted,
            )
        )
    )
    side_effects = list(
        db_session.scalars(
            select(SlackSideEffect).where(SlackSideEffect.task_id == task.id)
        )
    )

    assert first_ts == second_ts == "1716400001.000001"
    assert len(client.messages) == 1
    assert len(events) == 1
    assert len(side_effects) == 1
    assert side_effects[0].attempts == 1


def test_post_message_accepts_slack_sdk_response_shape(
    db_session: Session,
) -> None:
    task = create_task(db_session)
    client = FakeSlackSdkResponseClient()

    message_ts = SlackPoster(session=db_session, client=client).post_message(
        SlackThread.from_task(task),
        "Done.",
    )

    side_effect = db_session.scalar(
        select(SlackSideEffect).where(SlackSideEffect.task_id == task.id)
    )

    assert message_ts == "1716400001.000001"
    assert side_effect is not None
    assert side_effect.status == "succeeded"
    assert side_effect.response_json == {"ok": True, "ts": "1716400001.000001"}


def test_outbox_marks_stale_in_progress_rows_failed(db_session: Session) -> None:
    task = create_task(db_session)
    now = datetime(2026, 5, 31, 12, 0, tzinfo=UTC)
    stale = SlackSideEffect(
        installation_id=task.installation_id,
        task_id=task.id,
        idempotency_key=f"stale:{task.id}",
        operation="chat_postMessage",
        purpose="result",
        request_json={"channel": "C123"},
        status=SLACK_EFFECT_IN_PROGRESS,
        attempts=1,
        started_at=now - timedelta(minutes=10),
        available_at=now - timedelta(minutes=10),
    )
    fresh = SlackSideEffect(
        installation_id=task.installation_id,
        task_id=task.id,
        idempotency_key=f"fresh:{task.id}",
        operation="chat_postMessage",
        purpose="result",
        request_json={"channel": "C123"},
        status=SLACK_EFFECT_IN_PROGRESS,
        attempts=1,
        started_at=now - timedelta(seconds=30),
        available_at=now - timedelta(seconds=30),
    )
    db_session.add_all([stale, fresh])
    db_session.flush()

    result = SlackSideEffectOutbox(db_session).recover_stale_in_progress(
        now=now,
        stale_after=timedelta(minutes=5),
    )

    assert result.recovered_ids == (stale.id,)
    assert result.recovered_count == 1
    assert stale.status == SLACK_EFFECT_FAILED
    assert stale.available_at == now
    assert stale.last_error is not None
    assert stale.last_error["type"] == "StaleSideEffectLease"
    assert stale.last_error["delivery_state"] == "unknown"
    assert fresh.status == SLACK_EFFECT_IN_PROGRESS
    assert fresh.last_error is None


def test_post_message_retries_failed_side_effect(db_session: Session) -> None:
    task = create_task(db_session)
    idempotency_key = slack_message_key(task.id, "result")
    failed = SlackSideEffect(
        installation_id=task.installation_id,
        task_id=task.id,
        idempotency_key=idempotency_key,
        operation="chat_postMessage",
        purpose="result",
        target_channel_id="C123",
        target_thread_ts="1716400000.000001",
        request_json={"channel": "C123", "text": "Done."},
        status=SLACK_EFFECT_FAILED,
        attempts=1,
        last_error={"type": "StaleSideEffectLease"},
        available_at=datetime(2026, 5, 31, 12, 0, tzinfo=UTC),
    )
    db_session.add(failed)
    db_session.flush()
    client = FakeSlackClient()

    message_ts = SlackPoster(session=db_session, client=client).post_message(
        SlackThread.from_task(task),
        "Done.",
    )

    assert message_ts == "1716400001.000001"
    assert len(client.messages) == 1
    assert failed.status == "succeeded"
    assert failed.attempts == 2
    assert failed.last_error is None
    assert failed.response_json == {"ok": True, "ts": "1716400001.000001"}


def test_upload_file_updates_artifact_and_logs_event(
    db_session: Session,
    tmp_path: Path,
) -> None:
    task = create_task(db_session)
    report_path = tmp_path / "report.pdf"
    report_path.write_bytes(b"%PDF-1.4 test")
    artifact = Artifact(
        task_id=task.id,
        filename="report.pdf",
        mime_type="application/pdf",
        size_bytes=report_path.stat().st_size,
        storage_path=str(report_path),
    )
    db_session.add(artifact)
    db_session.flush()
    client = FakeSlackClient()

    slack_file_id = SlackPoster(session=db_session, client=client).upload_file(
        SlackThread.from_task(task),
        report_path,
        initial_comment="Here is the report.",
        now=datetime(2026, 5, 23, 3, 45, tzinfo=UTC),
    )

    event = db_session.scalar(
        select(TaskEvent)
        .where(
            TaskEvent.task_id == task.id,
            TaskEvent.type == TaskEventType.message_posted,
        )
        .order_by(TaskEvent.seq.desc())
        .limit(1)
    )

    assert slack_file_id == "F000001"
    assert artifact.slack_file_id == "F000001"
    assert artifact.posted_at == datetime(2026, 5, 23, 3, 45, tzinfo=UTC)
    assert client.uploads == [
        {
            "file": str(report_path),
            "filename": "report.pdf",
            "title": "report.pdf",
            "channel": "C123",
            "initial_comment": "Here is the report.",
            "thread_ts": "1716400000.000001",
        }
    ]
    assert event is not None
    assert event.payload["slack_file_id"] == "F000001"
    assert event.payload["artifact_id"] == str(artifact.id)
    assert event.payload["purpose"] == "file_upload"
    assert event.payload["idempotency_key"] == f"slack:file_upload:{artifact.id}"


def test_upload_file_normalizes_initial_comment_at_slack_boundary(
    db_session: Session,
    tmp_path: Path,
) -> None:
    task = create_task(db_session)
    report_path = tmp_path / "report.pdf"
    report_path.write_bytes(b"%PDF-1.4 test")
    db_session.add(
        Artifact(
            task_id=task.id,
            filename="report.pdf",
            mime_type="application/pdf",
            size_bytes=report_path.stat().st_size,
            storage_path=str(report_path),
        )
    )
    db_session.flush()
    client = FakeSlackClient()
    em_dash = chr(0x2014)

    SlackPoster(session=db_session, client=client).upload_file(
        SlackThread.from_task(task),
        report_path,
        initial_comment=f"**Report** {em_dash} ready.",
    )

    assert client.uploads[0]["initial_comment"] == "*Report* - ready."


def test_upload_file_in_dm_posts_without_thread_ts(
    db_session: Session,
    tmp_path: Path,
) -> None:
    task = create_task(db_session, channel_id="D123")
    report_path = tmp_path / "report.pdf"
    report_path.write_bytes(b"%PDF-1.4 test")
    artifact = Artifact(
        task_id=task.id,
        filename="report.pdf",
        mime_type="application/pdf",
        size_bytes=report_path.stat().st_size,
        storage_path=str(report_path),
    )
    db_session.add(artifact)
    db_session.flush()
    client = FakeSlackClient()

    SlackPoster(session=db_session, client=client).upload_file(
        SlackThread.from_task(task),
        report_path,
        initial_comment="Here is the report.",
    )

    event = db_session.scalar(
        select(TaskEvent)
        .where(
            TaskEvent.task_id == task.id,
            TaskEvent.type == TaskEventType.message_posted,
        )
        .order_by(TaskEvent.seq.desc())
        .limit(1)
    )

    assert client.uploads == [
        {
            "file": str(report_path),
            "filename": "report.pdf",
            "title": "report.pdf",
            "channel": "D123",
            "initial_comment": "Here is the report.",
            "thread_ts": None,
        }
    ]
    assert event is not None
    assert event.payload["thread_ts"] is None


def test_upload_file_uses_posted_at_as_dedup_guard(
    db_session: Session,
    tmp_path: Path,
) -> None:
    task = create_task(db_session)
    report_path = tmp_path / "report.pdf"
    report_path.write_bytes(b"%PDF-1.4 test")
    artifact = Artifact(
        task_id=task.id,
        filename="report.pdf",
        mime_type="application/pdf",
        size_bytes=report_path.stat().st_size,
        storage_path=str(report_path),
        slack_file_id="FALREADY",
        posted_at=datetime(2026, 5, 23, 3, 50, tzinfo=UTC),
    )
    db_session.add(artifact)
    db_session.flush()
    client = FakeSlackClient()

    slack_file_id = SlackPoster(session=db_session, client=client).upload_file(
        SlackThread.from_task(task),
        report_path,
    )

    assert slack_file_id == "FALREADY"
    assert client.uploads == []


def test_upload_file_creates_artifact_if_needed(
    db_session: Session,
    tmp_path: Path,
) -> None:
    task = create_task(db_session)
    report_path = tmp_path / "report.pdf"
    report_path.write_bytes(b"%PDF-1.4 test")
    client = FakeSlackClient()

    slack_file_id = SlackPoster(session=db_session, client=client).upload_file(
        SlackThread.from_task(task),
        report_path,
    )

    artifact = db_session.scalar(select(Artifact).where(Artifact.task_id == task.id))
    artifact_event = db_session.scalar(
        select(TaskEvent).where(
            TaskEvent.task_id == task.id,
            TaskEvent.type == TaskEventType.artifact_created,
        )
    )

    assert slack_file_id == "F000001"
    assert artifact is not None
    assert artifact.filename == "report.pdf"
    assert artifact.mime_type == "application/pdf"
    assert artifact.storage_path == str(report_path)
    assert artifact.slack_file_id == "F000001"
    assert artifact.posted_at is not None
    assert artifact_event is not None
    assert artifact_event.payload["artifact_id"] == str(artifact.id)


def cleanup_database(session: Session) -> None:
    for model in (
        Artifact,
        LLMUsage,
        TaskEvent,
        SlackSideEffect,
        Task,
        ModelPricing,
        EncryptedSecret,
        Installation,
    ):
        session.execute(delete(model))


# --- HIG-169 P0.1: egress unfurl-off + URL flagging --------------------------


def test_post_message_disables_link_unfurling(db_session: Session) -> None:
    # Unfurl-off is the exfiltration fix; it must be set on every outbound post.
    task = create_task(db_session)
    client = FakeSlackClient()
    SlackPoster(session=db_session, client=client).post_message(
        SlackThread.from_task(task),
        "Here is the result.",
    )
    assert client.unfurl_flags == [(False, False)]


def test_post_message_without_task_disables_unfurling(db_session: Session) -> None:
    client = FakeSlackClient()
    SlackPoster(session=db_session, client=client).post_message(
        SlackThread(channel_id="C123", thread_ts="1716400000.000001"),
        "No task here.",
    )
    assert client.unfurl_flags == [(False, False)]


def test_post_message_flags_suspicious_egress_url(db_session: Session) -> None:
    from kortny.slack.posting import EGRESS_URL_FLAGGED_MESSAGE

    task = create_task(db_session)
    client = FakeSlackClient()
    payload = "x" * 100
    SlackPoster(session=db_session, client=client).post_message(
        SlackThread.from_task(task),
        f"See https://evil.example.com/collect?data={payload}",
    )
    event = db_session.scalar(
        select(TaskEvent).where(
            TaskEvent.task_id == task.id,
            TaskEvent.type == TaskEventType.log,
            TaskEvent.payload["message"].as_string() == EGRESS_URL_FLAGGED_MESSAGE,
        )
    )
    assert event is not None
    flagged = event.payload["flagged"]
    assert flagged[0]["host"] == "evil.example.com"


def test_post_message_does_not_flag_plain_url(db_session: Session) -> None:
    from kortny.slack.posting import EGRESS_URL_FLAGGED_MESSAGE

    task = create_task(db_session)
    client = FakeSlackClient()
    SlackPoster(session=db_session, client=client).post_message(
        SlackThread.from_task(task),
        "Docs at https://example.com/page?id=42",
    )
    event = db_session.scalar(
        select(TaskEvent).where(
            TaskEvent.task_id == task.id,
            TaskEvent.type == TaskEventType.log,
            TaskEvent.payload["message"].as_string() == EGRESS_URL_FLAGGED_MESSAGE,
        )
    )
    assert event is None


def test_post_message_allowlisted_host_not_flagged(db_session: Session) -> None:
    from kortny.slack.posting import EGRESS_URL_FLAGGED_MESSAGE

    task = create_task(db_session)
    client = FakeSlackClient()
    payload = "y" * 100
    SlackPoster(
        session=db_session,
        client=client,
        egress_url_allowlist=frozenset({"trusted.example.com"}),
    ).post_message(
        SlackThread.from_task(task),
        f"Internal https://trusted.example.com/x?d={payload}",
    )
    event = db_session.scalar(
        select(TaskEvent).where(
            TaskEvent.task_id == task.id,
            TaskEvent.type == TaskEventType.log,
            TaskEvent.payload["message"].as_string() == EGRESS_URL_FLAGGED_MESSAGE,
        )
    )
    assert event is None


def create_task(
    session: Session,
    *,
    channel_id: str = "C123",
    thread_ts: str | None = "1716400000.000001",
) -> Task:
    installation = Installation(slack_team_id=f"T{uuid.uuid4().hex}")
    session.add(installation)
    session.flush()
    return TaskService(session).create_task(
        installation_id=installation.id,
        slack_event_id=f"Ev{uuid.uuid4().hex}",
        slack_channel_id=channel_id,
        slack_thread_ts=thread_ts,
        slack_message_ts=thread_ts,
        slack_user_id="U123",
        input="make a report",
    )


def create_assistant_task(
    session: Session,
    *,
    channel_id: str = "D0AU8HZT285",
    thread_ts: str = "1716400000.000777",
) -> Task:
    """A task on an assistant ("Agents & AI Apps") thread — a 'D' DM channel
    whose reply MUST stay threaded under the assistant thread root."""

    installation = Installation(slack_team_id=f"T{uuid.uuid4().hex}")
    session.add(installation)
    session.flush()
    event_id = f"Ev{uuid.uuid4().hex}"
    identity = TaskIdentity.slack_message(
        channel_id=channel_id,
        message_ts=thread_ts,
        thread_ts=thread_ts,
        user_id="U123",
        input_text="what do you know about this workspace?",
        slack_event_id=event_id,
        source_surface="assistant",
    )
    return TaskService(session).create_task(
        installation_id=installation.id,
        slack_event_id=event_id,
        slack_channel_id=channel_id,
        slack_thread_ts=thread_ts,
        slack_message_ts=thread_ts,
        slack_user_id="U123",
        input="what do you know about this workspace?",
        identity=identity,
        source_surface="assistant",
    )


class FakeStreamingSlackClient(FakeSlackClient):
    """FakeSlackClient + the assistant streaming/status surface (HIG-246)."""

    def __init__(self) -> None:
        super().__init__()
        self.stream_starts: list[dict[str, Any]] = []
        self.stream_appends: list[dict[str, Any]] = []
        self.stream_stops: list[dict[str, Any]] = []
        self.status_calls: list[dict[str, Any]] = []

    def chat_startStream(
        self,
        *,
        channel: str,
        thread_ts: str | None = None,
    ) -> dict[str, Any]:
        self.stream_starts.append({"channel": channel, "thread_ts": thread_ts})
        return {"ok": True, "ts": f"1716400002.{len(self.stream_starts):06d}"}

    def chat_appendStream(
        self,
        *,
        channel: str,
        ts: str,
        markdown_text: str,
    ) -> dict[str, Any]:
        self.stream_appends.append(
            {"channel": channel, "ts": ts, "markdown_text": markdown_text}
        )
        return {"ok": True, "ts": ts}

    def chat_stopStream(
        self,
        *,
        channel: str,
        ts: str,
        blocks: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        self.stream_stops.append({"channel": channel, "ts": ts, "blocks": blocks})
        return {"ok": True, "ts": ts}

    def assistant_threads_setStatus(
        self,
        *,
        channel_id: str,
        thread_ts: str,
        status: str,
        loading_messages: list[str] | None = None,
    ) -> dict[str, Any]:
        self.status_calls.append(
            {
                "channel_id": channel_id,
                "thread_ts": thread_ts,
                "status": status,
                "loading_messages": loading_messages,
            }
        )
        return {"ok": True}


def test_assistant_thread_keeps_thread_ts_on_dm_channel(db_session: Session) -> None:
    # Regression: assistant threads are 'D' channels but must stay threaded so
    # the live agent pane renders the reply (HIG-246).
    task = create_assistant_task(db_session)
    thread = SlackThread.from_task(task)
    assert thread.is_assistant is True
    client = FakeSlackClient()
    SlackPoster(session=db_session, client=client).post_message(thread, "Done.")
    assert client.messages[0]["thread_ts"] == "1716400000.000777"


def test_stream_message_streams_and_records_event(db_session: Session) -> None:
    task = create_assistant_task(db_session)
    client = FakeStreamingSlackClient()
    thread = SlackThread.from_task(task)

    ts = SlackPoster(session=db_session, client=client).stream_message(
        thread, "Here's what I know."
    )

    assert client.stream_starts == [
        {"channel": "D0AU8HZT285", "thread_ts": "1716400000.000777"}
    ]
    assert client.stream_appends[0]["markdown_text"] == "Here's what I know."
    assert client.stream_stops[0]["ts"] == ts
    # No plain chat.postMessage when streaming.
    assert client.messages == []

    event = db_session.scalar(
        select(TaskEvent)
        .where(
            TaskEvent.task_id == task.id,
            TaskEvent.type == TaskEventType.message_posted,
        )
        .order_by(TaskEvent.seq.desc())
        .limit(1)
    )
    assert event is not None
    assert event.payload["delivery"] == "stream"
    assert event.payload["thread_ts"] == "1716400000.000777"
    assert event.payload["message_ts"] == ts


def test_stream_message_idempotent_on_retry(db_session: Session) -> None:
    task = create_assistant_task(db_session)
    client = FakeStreamingSlackClient()
    poster = SlackPoster(session=db_session, client=client)
    thread = SlackThread.from_task(task)

    first = poster.stream_message(thread, "Answer.")
    second = poster.stream_message(thread, "Answer.")

    assert first == second
    # Only one stream sequence — the retry short-circuits on the prior event.
    assert len(client.stream_starts) == 1


def test_stream_message_falls_back_without_streaming_support(
    db_session: Session,
) -> None:
    task = create_assistant_task(db_session)
    client = FakeSlackClient()  # no chat_startStream
    poster = SlackPoster(session=db_session, client=client)

    poster.stream_message(SlackThread.from_task(task), "Answer.")

    # Fell back to a plain (but still threaded) post.
    assert client.messages[0]["thread_ts"] == "1716400000.000777"
    assert client.messages[0]["text"] == "Answer."


def test_clear_assistant_status_clears_for_assistant_thread(
    db_session: Session,
) -> None:
    task = create_assistant_task(db_session)
    client = FakeStreamingSlackClient()
    SlackPoster(session=db_session, client=client).clear_assistant_status(
        SlackThread.from_task(task)
    )
    assert client.status_calls == [
        {
            "channel_id": "D0AU8HZT285",
            "thread_ts": "1716400000.000777",
            "status": "",
            "loading_messages": [],
        }
    ]


def test_clear_assistant_status_noop_for_non_assistant(db_session: Session) -> None:
    task = create_task(db_session, channel_id="D123")
    client = FakeStreamingSlackClient()
    SlackPoster(session=db_session, client=client).clear_assistant_status(
        SlackThread.from_task(task)
    )
    assert client.status_calls == []


def test_post_message_distinct_idempotency_purpose_posts_both(
    db_session: Session,
) -> None:
    # Two approvals in one task must each get their own prompt — the outbox must
    # not dedup the second against the first (HIG-248).
    task = create_task(db_session, channel_id="C123")
    client = FakeSlackClient()
    poster = SlackPoster(session=db_session, client=client)
    thread = SlackThread.from_task(task)

    poster.post_message(
        thread,
        "Approve A?",
        purpose="tool_approval_request",
        idempotency_purpose="tool_approval_request:keyA",
    )
    poster.post_message(
        thread,
        "Approve B?",
        purpose="tool_approval_request",
        idempotency_purpose="tool_approval_request:keyB",
    )

    assert [m["text"] for m in client.messages] == ["Approve A?", "Approve B?"]
    events = db_session.scalars(
        select(TaskEvent)
        .where(
            TaskEvent.task_id == task.id,
            TaskEvent.type == TaskEventType.message_posted,
        )
        .order_by(TaskEvent.seq)
    ).all()
    # Both recorded under the same purpose so reaction lookup finds the latest.
    assert [e.payload["purpose"] for e in events] == [
        "tool_approval_request",
        "tool_approval_request",
    ]


def test_post_message_same_idempotency_purpose_dedups(db_session: Session) -> None:
    task = create_task(db_session, channel_id="C123")
    client = FakeSlackClient()
    poster = SlackPoster(session=db_session, client=client)
    thread = SlackThread.from_task(task)

    first = poster.post_message(
        thread,
        "Approve?",
        purpose="tool_approval_request",
        idempotency_purpose="tool_approval_request:sameKey",
    )
    second = poster.post_message(
        thread,
        "Approve?",
        purpose="tool_approval_request",
        idempotency_purpose="tool_approval_request:sameKey",
    )

    # Same key → outbox dedups → posted once, same ts returned.
    assert first == second
    assert len(client.messages) == 1
