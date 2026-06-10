import os
import uuid
from collections.abc import Iterator
from typing import Any

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, delete, select
from sqlalchemy.orm import Session

from kortny.db.models import (
    Installation,
    SlackSideEffect,
    Task,
    TaskEvent,
    TaskEventType,
    TaskStatus,
)
from kortny.db.session import make_engine, make_session_factory, normalize_database_url
from kortny.tools.slack_actions import (
    SlackAddBookmarkTool,
    SlackAddReactionTool,
    SlackCreateChannelCanvasTool,
    SlackEditCanvasTool,
    SlackLookupCanvasSectionsTool,
    SlackPinMessageTool,
    SlackReplyThreadTool,
)

TEST_POSTGRES_URL = os.environ.get("KORTNY_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    TEST_POSTGRES_URL is None,
    reason="KORTNY_TEST_POSTGRES_URL is required for Slack action tool tests",
)


class FakeSlackActionClient:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []
        self.reactions: list[dict[str, Any]] = []
        self.pins: list[dict[str, Any]] = []
        self.bookmarks: list[dict[str, Any]] = []
        self.channel_canvases: list[dict[str, Any]] = []
        self.canvas_section_lookups: list[dict[str, Any]] = []
        self.canvas_edits: list[dict[str, Any]] = []

    def chat_postMessage(
        self,
        *,
        channel: str,
        text: str,
        thread_ts: str | None = None,
    ) -> dict[str, Any]:
        self.messages.append(
            {
                "channel": channel,
                "text": text,
                "thread_ts": thread_ts,
            }
        )
        return {"ok": True, "ts": f"1716400001.{len(self.messages):06d}"}

    def reactions_add(
        self,
        *,
        channel: str,
        timestamp: str,
        name: str,
    ) -> dict[str, Any]:
        self.reactions.append(
            {
                "channel": channel,
                "timestamp": timestamp,
                "name": name,
            }
        )
        return {"ok": True}

    def pins_add(
        self,
        *,
        channel: str,
        timestamp: str,
    ) -> dict[str, Any]:
        self.pins.append(
            {
                "channel": channel,
                "timestamp": timestamp,
            }
        )
        return {"ok": True}

    def bookmarks_add(
        self,
        *,
        channel_id: str,
        title: str,
        type: str,
        link: str,
        emoji: str | None = None,
    ) -> dict[str, Any]:
        self.bookmarks.append(
            {
                "channel_id": channel_id,
                "title": title,
                "type": type,
                "link": link,
                "emoji": emoji,
            }
        )
        return {
            "ok": True,
            "bookmark": {
                "id": f"Bk{len(self.bookmarks):06d}",
                "channel_id": channel_id,
                "title": title,
                "link": link,
                "type": type,
            },
        }

    def conversations_canvases_create(
        self,
        *,
        channel_id: str,
        document_content: dict[str, str],
        title: str | None = None,
    ) -> dict[str, Any]:
        self.channel_canvases.append(
            {
                "channel_id": channel_id,
                "document_content": document_content,
                "title": title,
            }
        )
        return {
            "ok": True,
            "canvas_id": f"Fcanvas{len(self.channel_canvases):06d}",
        }

    def canvases_edit(
        self,
        *,
        canvas_id: str,
        changes: list[dict[str, Any]],
    ) -> dict[str, Any]:
        self.canvas_edits.append(
            {
                "canvas_id": canvas_id,
                "changes": changes,
            }
        )
        return {"ok": True}

    def canvases_sections_lookup(
        self,
        *,
        canvas_id: str,
        criteria: dict[str, Any],
    ) -> dict[str, Any]:
        self.canvas_section_lookups.append(
            {
                "canvas_id": canvas_id,
                "criteria": criteria,
            }
        )
        return {
            "ok": True,
            "sections": [
                {
                    "id": "section_open_items",
                    "type": "h2",
                    "text": "Open Items",
                },
                {
                    "section_id": "section_risks",
                    "section_type": "h2",
                    "plain_text": "Risks",
                },
            ],
        }


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


def test_slack_reply_thread_posts_current_thread_and_records_side_effect(
    db_session: Session,
) -> None:
    task = create_task(db_session)
    client = FakeSlackActionClient()

    result = SlackReplyThreadTool(
        client=client,
        session=db_session,
        task=task,
    ).invoke({"text": "I left a note here.", "purpose": "status_update"})

    event = db_session.scalar(
        select(TaskEvent).where(
            TaskEvent.task_id == task.id,
            TaskEvent.type == TaskEventType.message_posted,
        )
    )
    side_effect = db_session.scalar(
        select(SlackSideEffect).where(SlackSideEffect.task_id == task.id)
    )

    assert result.output["successful"] is True
    assert result.output["message_ts"] == "1716400001.000001"
    assert client.messages == [
        {
            "channel": "C123",
            "text": "I left a note here.",
            "thread_ts": "1716400000.000001",
        }
    ]
    assert event is not None
    assert event.payload["tool"] == "slack_reply_thread"
    assert event.payload["purpose"] == "tool_status_update"
    assert event.payload["message_ts"] == "1716400001.000001"
    assert side_effect is not None
    assert side_effect.operation == "chat_postMessage"
    assert side_effect.status == "succeeded"
    assert side_effect.request_json["thread_ts"] == "1716400000.000001"


def test_slack_reply_thread_does_not_thread_dm_replies(
    db_session: Session,
) -> None:
    task = create_task(
        db_session,
        channel_id="D123",
        thread_ts="D123",
        message_ts="1716400000.000001",
    )
    client = FakeSlackActionClient()

    result = SlackReplyThreadTool(
        client=client,
        session=db_session,
        task=task,
    ).invoke({"text": "Yep, I see it."})

    assert result.output["thread_ts"] is None
    assert client.messages == [
        {
            "channel": "D123",
            "text": "Yep, I see it.",
            "thread_ts": None,
        }
    ]


def test_slack_reply_thread_rejects_other_channel(
    db_session: Session,
) -> None:
    task = create_task(db_session)
    client = FakeSlackActionClient()

    with pytest.raises(ValueError, match="current Slack channel"):
        SlackReplyThreadTool(
            client=client,
            session=db_session,
            task=task,
        ).invoke({"text": "Nope.", "channel_id": "C999"})

    assert client.messages == []


def test_slack_add_reaction_uses_current_message_and_records_side_effect(
    db_session: Session,
) -> None:
    task = create_task(db_session)
    client = FakeSlackActionClient()

    result = SlackAddReactionTool(
        client=client,
        session=db_session,
        task=task,
    ).invoke({"name": ":eyes:"})

    event = db_session.scalar(
        select(TaskEvent).where(
            TaskEvent.task_id == task.id,
            TaskEvent.type == TaskEventType.log,
            TaskEvent.payload["message"].as_string() == "slack_reaction_added",
        )
    )
    side_effect = db_session.scalar(
        select(SlackSideEffect).where(SlackSideEffect.task_id == task.id)
    )

    assert result.output["successful"] is True
    assert result.output["reaction"] == "eyes"
    assert client.reactions == [
        {
            "channel": "C123",
            "timestamp": "1716400000.000001",
            "name": "eyes",
        }
    ]
    assert event is not None
    assert event.payload["tool"] == "slack_add_reaction"
    assert event.payload["reaction"] == "eyes"
    assert side_effect is not None
    assert side_effect.operation == "reactions_add"
    assert side_effect.target_message_ts == "1716400000.000001"


def test_slack_add_reaction_rejects_other_message(
    db_session: Session,
) -> None:
    task = create_task(db_session)
    client = FakeSlackActionClient()

    with pytest.raises(ValueError, match="current Slack message"):
        SlackAddReactionTool(
            client=client,
            session=db_session,
            task=task,
        ).invoke({"name": "eyes", "message_ts": "1716400099.000001"})

    assert client.reactions == []


def test_slack_pin_message_uses_current_message_and_records_side_effect(
    db_session: Session,
) -> None:
    task = create_task(db_session)
    client = FakeSlackActionClient()

    result = SlackPinMessageTool(
        client=client,
        session=db_session,
        task=task,
    ).invoke({})

    event = db_session.scalar(
        select(TaskEvent).where(
            TaskEvent.task_id == task.id,
            TaskEvent.type == TaskEventType.log,
            TaskEvent.payload["message"].as_string() == "slack_message_pinned",
        )
    )
    side_effect = db_session.scalar(
        select(SlackSideEffect).where(SlackSideEffect.task_id == task.id)
    )

    assert result.output["successful"] is True
    assert client.pins == [
        {
            "channel": "C123",
            "timestamp": "1716400000.000001",
        }
    ]
    assert event is not None
    assert event.payload["tool"] == "slack_pin_message"
    assert event.payload["message_ts"] == "1716400000.000001"
    assert side_effect is not None
    assert side_effect.operation == "pins_add"
    assert side_effect.target_message_ts == "1716400000.000001"


def test_slack_pin_message_rejects_other_message(
    db_session: Session,
) -> None:
    task = create_task(db_session)
    client = FakeSlackActionClient()

    with pytest.raises(ValueError, match="current Slack message"):
        SlackPinMessageTool(
            client=client,
            session=db_session,
            task=task,
        ).invoke({"message_ts": "1716400099.000001"})

    assert client.pins == []


def test_slack_add_bookmark_adds_link_to_current_channel(
    db_session: Session,
) -> None:
    task = create_task(db_session)
    client = FakeSlackActionClient()

    result = SlackAddBookmarkTool(
        client=client,
        session=db_session,
        task=task,
    ).invoke(
        {
            "title": "Slack API docs",
            "link": "https://docs.slack.dev/reference/methods/bookmarks.add",
            "emoji": ":bookmark:",
        }
    )

    event = db_session.scalar(
        select(TaskEvent).where(
            TaskEvent.task_id == task.id,
            TaskEvent.type == TaskEventType.log,
            TaskEvent.payload["message"].as_string() == "slack_bookmark_added",
        )
    )
    side_effect = db_session.scalar(
        select(SlackSideEffect).where(SlackSideEffect.task_id == task.id)
    )

    assert result.output["successful"] is True
    assert result.output["bookmark_id"] == "Bk000001"
    assert client.bookmarks == [
        {
            "channel_id": "C123",
            "title": "Slack API docs",
            "type": "link",
            "link": "https://docs.slack.dev/reference/methods/bookmarks.add",
            "emoji": "bookmark",
        }
    ]
    assert event is not None
    assert event.payload["tool"] == "slack_add_bookmark"
    assert event.payload["bookmark_id"] == "Bk000001"
    assert side_effect is not None
    assert side_effect.operation == "bookmarks_add"
    assert side_effect.target_channel_id == "C123"
    assert side_effect.request_json["type"] == "link"


def test_slack_add_bookmark_rejects_dm_and_invalid_link(
    db_session: Session,
) -> None:
    task = create_task(db_session, channel_id="D123", thread_ts="D123")
    client = FakeSlackActionClient()

    with pytest.raises(ValueError, match="only available in Slack channels"):
        SlackAddBookmarkTool(
            client=client,
            session=db_session,
            task=task,
        ).invoke({"title": "Bad", "link": "https://example.com"})

    task.slack_channel_id = "C123"
    db_session.flush()
    with pytest.raises(ValueError, match="must start with http:// or https://"):
        SlackAddBookmarkTool(
            client=client,
            session=db_session,
            task=task,
        ).invoke({"title": "Bad", "link": "ftp://example.com"})

    assert client.bookmarks == []


def test_slack_create_channel_canvas_records_side_effect_and_event(
    db_session: Session,
) -> None:
    task = create_task(db_session)
    client = FakeSlackActionClient()

    result = SlackCreateChannelCanvasTool(
        client=client,
        session=db_session,
        task=task,
    ).invoke(
        {
            "title": "Channel Brief",
            "markdown": "# Channel Brief\n\n- Owner: Aneesh\n- Status: active",
        }
    )

    event = db_session.scalar(
        select(TaskEvent).where(
            TaskEvent.task_id == task.id,
            TaskEvent.type == TaskEventType.log,
            TaskEvent.payload["message"].as_string() == "slack_channel_canvas_created",
        )
    )
    side_effect = db_session.scalar(
        select(SlackSideEffect).where(SlackSideEffect.task_id == task.id)
    )

    assert result.output["successful"] is True
    assert result.output["canvas_id"] == "Fcanvas000001"
    assert client.channel_canvases == [
        {
            "channel_id": "C123",
            "title": "Channel Brief",
            "document_content": {
                "type": "markdown",
                "markdown": "# Channel Brief\n\n- Owner: Aneesh\n- Status: active",
            },
        }
    ]
    assert event is not None
    assert event.payload["tool"] == "slack_create_channel_canvas"
    assert event.payload["canvas_id"] == "Fcanvas000001"
    assert event.payload["title"] == "Channel Brief"
    assert side_effect is not None
    assert side_effect.operation == "conversations_canvases_create"
    assert side_effect.purpose == "tool_create_channel_canvas"
    assert side_effect.target_channel_id == "C123"


def test_slack_create_channel_canvas_rejects_dm(
    db_session: Session,
) -> None:
    task = create_task(db_session, channel_id="D123", thread_ts="D123")
    client = FakeSlackActionClient()

    with pytest.raises(ValueError, match="only available in Slack channels"):
        SlackCreateChannelCanvasTool(
            client=client,
            session=db_session,
            task=task,
        ).invoke({"title": "DM Canvas", "markdown": "Nope"})

    assert client.channel_canvases == []


def test_slack_edit_canvas_appends_markdown_and_records_event(
    db_session: Session,
) -> None:
    task = create_task(db_session)
    client = FakeSlackActionClient()

    result = SlackEditCanvasTool(
        client=client,
        session=db_session,
        task=task,
    ).invoke(
        {
            "canvas_id": "Fcanvas123",
            "operation": "insert_at_end",
            "markdown": "## Follow-up\n\n- Check pricing",
        }
    )

    event = db_session.scalar(
        select(TaskEvent).where(
            TaskEvent.task_id == task.id,
            TaskEvent.type == TaskEventType.log,
            TaskEvent.payload["message"].as_string() == "slack_canvas_edited",
        )
    )
    side_effect = db_session.scalar(
        select(SlackSideEffect).where(SlackSideEffect.task_id == task.id)
    )

    assert result.output["successful"] is True
    assert result.output["operation"] == "insert_at_end"
    assert client.canvas_edits == [
        {
            "canvas_id": "Fcanvas123",
            "changes": [
                {
                    "operation": "insert_at_end",
                    "document_content": {
                        "type": "markdown",
                        "markdown": "## Follow-up\n\n- Check pricing",
                    },
                }
            ],
        }
    ]
    assert event is not None
    assert event.payload["tool"] == "slack_edit_canvas"
    assert event.payload["canvas_id"] == "Fcanvas123"
    assert event.payload["operation"] == "insert_at_end"
    assert side_effect is not None
    assert side_effect.operation == "canvases_edit"
    assert side_effect.purpose == "tool_edit_canvas"


def test_slack_lookup_canvas_sections_returns_normalized_sections(
    db_session: Session,
) -> None:
    create_task(db_session)
    client = FakeSlackActionClient()

    result = SlackLookupCanvasSectionsTool(client=client).invoke(
        {
            "canvas_id": "Fcanvas123",
            "contains_text": "  Open   Items  ",
            "section_types": ["h2", "h2"],
        }
    )

    assert client.canvas_section_lookups == [
        {
            "canvas_id": "Fcanvas123",
            "criteria": {
                "contains_text": "Open Items",
                "section_types": ["h2"],
            },
        }
    ]
    assert result.output == {
        "successful": True,
        "canvas_id": "Fcanvas123",
        "criteria": {
            "contains_text": "Open Items",
            "section_types": ["h2"],
        },
        "section_count": 2,
        "section_ids": ["section_open_items", "section_risks"],
        "sections": [
            {
                "section_id": "section_open_items",
                "section_type": "h2",
                "text": "Open Items",
            },
            {
                "section_id": "section_risks",
                "section_type": "h2",
                "text": "Risks",
            },
        ],
    }


def test_slack_lookup_canvas_sections_requires_criteria(
    db_session: Session,
) -> None:
    create_task(db_session)
    client = FakeSlackActionClient()

    with pytest.raises(ValueError, match="criteria"):
        SlackLookupCanvasSectionsTool(client=client).invoke(
            {
                "canvas_id": "Fcanvas123",
            }
        )

    assert client.canvas_section_lookups == []


def test_slack_edit_canvas_requires_section_for_insert_after(
    db_session: Session,
) -> None:
    task = create_task(db_session)
    client = FakeSlackActionClient()

    with pytest.raises(ValueError, match="section_id"):
        SlackEditCanvasTool(
            client=client,
            session=db_session,
            task=task,
        ).invoke(
            {
                "canvas_id": "Fcanvas123",
                "operation": "insert_after",
                "markdown": "Needs a section target.",
            }
        )

    assert client.canvas_edits == []


def cleanup_database(session: Session) -> None:
    for model in (SlackSideEffect, TaskEvent, Task, Installation):
        session.execute(delete(model))


def create_task(
    session: Session,
    *,
    channel_id: str = "C123",
    thread_ts: str | None = "1716400000.000001",
    message_ts: str | None = "1716400000.000001",
) -> Task:
    installation = Installation(slack_team_id=f"T{uuid.uuid4().hex}")
    session.add(installation)
    session.flush()
    task = Task(
        installation_id=installation.id,
        slack_event_id=f"Ev{uuid.uuid4().hex}",
        slack_channel_id=channel_id,
        slack_thread_ts=thread_ts,
        slack_message_ts=message_ts,
        slack_user_id="U123",
        input="perform Slack action",
        status=TaskStatus.pending,
    )
    session.add(task)
    session.flush()
    return task
