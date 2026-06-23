"""Tests for HIG-285: channel isolation gate on cross-channel tool reads.

Covers ChannelAccessGate, SlackChannelHistoryTool, and SlackFileReadTool
membership enforcement.  DB-backed (real Postgres) for task creation so
identity_kind is correctly persisted.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import Engine, delete
from sqlalchemy.orm import Session

from kortny.db.models import Installation, Task, TaskEvent
from kortny.db.session import make_engine, make_session_factory
from kortny.tasks import TaskService
from kortny.tasks.identity import TaskIdentity
from kortny.tools.channel_access import ChannelAccessGate
from kortny.tools.slack_channel_history import SlackChannelHistoryTool
from kortny.tools.slack_file_read import SlackFileReadTool
from kortny.tools.types import RecoverableToolError

TEST_POSTGRES_URL = os.environ.get("KORTNY_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    TEST_POSTGRES_URL is None,
    reason="KORTNY_TEST_POSTGRES_URL is required for channel isolation tests",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def engine() -> Iterator[Engine]:
    assert TEST_POSTGRES_URL is not None
    # Do not call alembic upgrade here — the test DB schema is managed by the
    # shared test infrastructure (conftest.py xdist setup or make test-serial
    # which runs migrations before the suite). Calling upgrade from inside a
    # worktree whose migration files may lag behind the live DB would fail with
    # "No such revision" if the DB already has newer migrations applied.
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
    for model in (TaskEvent, Task, Installation):
        session.execute(delete(model))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_installation(session: Session) -> Installation:
    inst = Installation(slack_team_id=f"T{uuid.uuid4().hex}")
    session.add(inst)
    session.flush()
    return inst


def _make_user_task(
    session: Session,
    installation: Installation,
    *,
    channel_id: str,
    user_id: str,
    message_ts: str | None = None,
) -> Task:
    """Create a slack_message task (user-initiated, subject to the gate)."""
    ts = message_ts or f"17000000{uuid.uuid4().hex[:6]}.000001"
    return TaskService(session).create_task(
        installation_id=installation.id,
        slack_channel_id=channel_id,
        slack_user_id=user_id,
        input="test",
        slack_event_id=f"Ev{uuid.uuid4().hex}",
        slack_thread_ts=ts,
        slack_message_ts=ts,
        identity=TaskIdentity.slack_message(
            channel_id=channel_id,
            message_ts=ts,
            thread_ts=ts,
            user_id=user_id,
            input_text="test",
        ),
    )


def _make_synthetic_task(
    session: Session,
    installation: Installation,
    *,
    channel_id: str,
) -> Task:
    """Create a synthetic task (ambient pipeline, bypasses gate)."""
    source_id = uuid.uuid4().hex
    return TaskService(session).create_task(
        installation_id=installation.id,
        slack_channel_id=channel_id,
        slack_user_id="UBOT",
        input="observe: assess channel",
        identity=TaskIdentity.synthetic(
            source="test_observe",
            source_id=source_id,
            input_text="observe: assess channel",
        ),
    )


class FakeMembershipClient:
    """Fake Slack client whose conversations_members response is controllable."""

    def __init__(self, members_by_channel: dict[str, list[str]]) -> None:
        self._members = members_by_channel
        self.calls: list[dict[str, Any]] = []

    def conversations_members(
        self,
        *,
        channel: str,
        cursor: str | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        self.calls.append({"channel": channel, "cursor": cursor, "limit": limit})
        members = self._members.get(channel, [])
        return {
            "ok": True,
            "members": members,
            "response_metadata": {"next_cursor": ""},
        }

    # Needed by SlackChannelHistoryTool (SlackChannelHistoryClient protocol)
    def conversations_history(self, **_: Any) -> dict[str, Any]:
        return {
            "ok": True,
            "messages": [],
            "response_metadata": {"next_cursor": ""},
        }

    def conversations_replies(self, **_: Any) -> dict[str, Any]:
        return {
            "ok": True,
            "messages": [],
            "response_metadata": {"next_cursor": ""},
        }

    # Needed by SlackFileReadTool (SlackFileReadClient protocol)
    def files_info(self, *, file: str) -> dict[str, Any]:
        return {
            "ok": True,
            "file": {
                "id": file,
                "name": "report.txt",
                "mimetype": "text/plain",
                "size": 10,
                "url_private_download": "https://files.slack.com/files-pri/fake",
                "channels": list(self._members.keys()),
            },
        }


# ---------------------------------------------------------------------------
# ChannelAccessGate unit tests
# ---------------------------------------------------------------------------


def test_gate_denies_non_member(db_session: Session) -> None:
    """User-initiated task: asker not in target channel -> denied."""
    inst = _make_installation(db_session)
    task = _make_user_task(db_session, inst, channel_id="CDM001", user_id="UASKER")
    db_session.commit()

    client = FakeMembershipClient(members_by_channel={"CPRIVATE": ["UOTHER"]})
    gate = ChannelAccessGate(task=task, client=client)

    with pytest.raises(RecoverableToolError) as exc_info:
        gate.check("CPRIVATE")

    err = exc_info.value
    assert err.code == "channel_access_denied"
    assert "CPRIVATE" in err.message


def test_gate_allows_member(db_session: Session) -> None:
    """User-initiated task: asker IS a member of target -> allowed."""
    inst = _make_installation(db_session)
    task = _make_user_task(db_session, inst, channel_id="CDM001", user_id="UASKER")
    db_session.commit()

    client = FakeMembershipClient(members_by_channel={"CPUBLIC": ["UASKER", "UOTHER"]})
    gate = ChannelAccessGate(task=task, client=client)

    gate.check("CPUBLIC")  # must not raise


def test_gate_allows_own_channel(db_session: Session) -> None:
    """User-initiated task: reading own task channel -> always allowed, no API call."""
    inst = _make_installation(db_session)
    task = _make_user_task(db_session, inst, channel_id="CTASK", user_id="UASKER")
    db_session.commit()

    client = FakeMembershipClient(members_by_channel={})
    gate = ChannelAccessGate(task=task, client=client)

    gate.check("CTASK")  # must not raise

    # No API calls should be made for the current task channel
    assert client.calls == []


def test_gate_allows_synthetic_task(db_session: Session) -> None:
    """Synthetic task (ambient pipeline) -> gate is bypassed entirely."""
    inst = _make_installation(db_session)
    task = _make_synthetic_task(db_session, inst, channel_id="CCHANNEL")
    db_session.commit()

    assert task.identity_kind == "synthetic"

    # Client would deny if called, but gate should never call it
    client = FakeMembershipClient(members_by_channel={})
    gate = ChannelAccessGate(task=task, client=client)

    gate.check("CSECRET")  # must not raise
    assert client.calls == []


def test_gate_caches_membership_per_channel(db_session: Session) -> None:
    """Second check for the same channel uses the cache, not the API again."""
    inst = _make_installation(db_session)
    task = _make_user_task(db_session, inst, channel_id="CDM", user_id="UASKER")
    db_session.commit()

    client = FakeMembershipClient(members_by_channel={"CPUBLIC": ["UASKER"]})
    gate = ChannelAccessGate(task=task, client=client)

    gate.check("CPUBLIC")
    gate.check("CPUBLIC")

    # Only one conversations.members call despite two checks
    assert sum(1 for c in client.calls if c["channel"] == "CPUBLIC") == 1


# ---------------------------------------------------------------------------
# SlackChannelHistoryTool integration
# ---------------------------------------------------------------------------


def test_history_tool_denies_non_member_on_cache_path(db_session: Session) -> None:
    """history tool: non-member DM task targeting private channel -> denied (cache path)."""
    inst = _make_installation(db_session)
    task = _make_user_task(db_session, inst, channel_id="CDM", user_id="UASKER")
    db_session.commit()

    client = FakeMembershipClient(members_by_channel={"CPRIVATE": []})
    gate = ChannelAccessGate(task=task, client=client)

    tool = SlackChannelHistoryTool(
        client,
        default_channel_id="CDM",
        access_gate=gate,
    )

    with pytest.raises(RecoverableToolError) as exc_info:
        tool.invoke({"channel_id": "CPRIVATE"})

    assert exc_info.value.code == "channel_access_denied"


def test_history_tool_allows_member_on_live_api_path(db_session: Session) -> None:
    """history tool: member task targeting other channel -> allowed (live API)."""
    inst = _make_installation(db_session)
    task = _make_user_task(db_session, inst, channel_id="CDM", user_id="UASKER")
    db_session.commit()

    client = FakeMembershipClient(members_by_channel={"CPUBLIC": ["UASKER"]})
    gate = ChannelAccessGate(task=task, client=client)

    tool = SlackChannelHistoryTool(
        client,
        default_channel_id="CDM",
        access_gate=gate,
    )

    result = tool.invoke({"channel_id": "CPUBLIC", "source": "slack_api"})
    assert "messages" in result.output
    assert result.output.get("channel_id") == "CPUBLIC"


def test_history_tool_allows_own_channel(db_session: Session) -> None:
    """history tool: task channel omitted -> always allowed."""
    inst = _make_installation(db_session)
    task = _make_user_task(db_session, inst, channel_id="CTASK", user_id="UASKER")
    db_session.commit()

    client = FakeMembershipClient(members_by_channel={})
    gate = ChannelAccessGate(task=task, client=client)

    tool = SlackChannelHistoryTool(
        client,
        default_channel_id="CTASK",
        access_gate=gate,
    )

    # channel_id omitted -> uses default_channel_id = task channel
    result = tool.invoke({"source": "slack_api"})
    assert result.output.get("channel_id") == "CTASK"
    assert client.calls == []  # no membership check needed


def test_history_tool_synthetic_task_allowed(db_session: Session) -> None:
    """history tool: synthetic task reading any channel -> allowed (ambient pipeline)."""
    inst = _make_installation(db_session)
    task = _make_synthetic_task(db_session, inst, channel_id="CCHANNEL")
    db_session.commit()

    client = FakeMembershipClient(members_by_channel={})
    gate = ChannelAccessGate(task=task, client=client)

    tool = SlackChannelHistoryTool(
        client,
        default_channel_id="CCHANNEL",
        access_gate=gate,
    )

    result = tool.invoke({"channel_id": "CSECRET", "source": "slack_api"})
    assert "messages" in result.output
    assert client.calls == []


# ---------------------------------------------------------------------------
# SlackFileReadTool integration
# ---------------------------------------------------------------------------


def test_file_read_denies_file_from_inaccessible_channel(
    tmp_path: Path,
    db_session: Session,
) -> None:
    """file read: file shared only in a channel the asker is not in -> denied."""
    inst = _make_installation(db_session)
    task = _make_user_task(db_session, inst, channel_id="CDM", user_id="UASKER")
    db_session.commit()

    # Channel CPRIVATE has the file but UASKER is not a member
    client = FakeMembershipClient(members_by_channel={"CPRIVATE": ["UOTHER"]})
    gate = ChannelAccessGate(task=task, client=client)

    tool = SlackFileReadTool(
        client=client,
        bot_token="xoxb-fake",
        working_dir=tmp_path,
        access_gate=gate,
    )

    with pytest.raises(RecoverableToolError) as exc_info:
        tool.invoke({"file_id": "F123PRIVATE"})

    assert exc_info.value.code == "channel_access_denied"


def test_file_read_allows_file_when_asker_in_at_least_one_channel(
    tmp_path: Path,
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """file read: file in multiple channels, asker in one -> allowed."""
    inst = _make_installation(db_session)
    task = _make_user_task(db_session, inst, channel_id="CDM", user_id="UASKER")
    db_session.commit()

    # Two channels: asker is only in CPUBLIC
    class MultiChannelClient(FakeMembershipClient):
        def files_info(self, *, file: str) -> dict[str, Any]:
            return {
                "ok": True,
                "file": {
                    "id": file,
                    "name": "report.txt",
                    "mimetype": "text/plain",
                    "size": 10,
                    "url_private_download": "https://files.slack.com/fake",
                    "channels": ["CPRIVATE", "CPUBLIC"],
                },
            }

        def conversations_members(
            self,
            *,
            channel: str,
            cursor: str | None = None,
            limit: int | None = None,
        ) -> dict[str, Any]:
            self.calls.append({"channel": channel, "cursor": cursor, "limit": limit})
            members = ["UASKER", "UOTHER"] if channel == "CPUBLIC" else ["UOTHER"]
            return {
                "ok": True,
                "members": members,
                "response_metadata": {"next_cursor": ""},
            }

    client = MultiChannelClient(members_by_channel={})
    gate = ChannelAccessGate(task=task, client=client)

    # Patch the download to avoid real HTTP
    import httpx  # noqa: PLC0415

    fake_content = b"hello world"

    class FakeTransport(httpx.BaseTransport):
        def handle_request(self, request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=fake_content,
                headers={"content-type": "text/plain"},
            )

    tool = SlackFileReadTool(
        client=client,
        bot_token="xoxb-fake",
        working_dir=tmp_path,
        transport=FakeTransport(),
        access_gate=gate,
    )

    result = tool.invoke({"file_id": "F123MULTI"})
    # Should succeed (not raise), file content extracted
    assert "filename" in result.output


def test_file_read_synthetic_task_allowed(
    tmp_path: Path,
    db_session: Session,
) -> None:
    """file read: synthetic task -> gate bypassed, file accessible regardless."""
    inst = _make_installation(db_session)
    task = _make_synthetic_task(db_session, inst, channel_id="CCHANNEL")
    db_session.commit()

    client = FakeMembershipClient(members_by_channel={"CPRIVATE": []})
    gate = ChannelAccessGate(task=task, client=client)

    import httpx  # noqa: PLC0415

    fake_content = b"secret data"

    class FakeTransport(httpx.BaseTransport):
        def handle_request(self, request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=fake_content,
                headers={"content-type": "text/plain"},
            )

    tool = SlackFileReadTool(
        client=client,
        bot_token="xoxb-fake",
        working_dir=tmp_path,
        transport=FakeTransport(),
        access_gate=gate,
    )

    # Synthetic task should be able to read any file without restriction
    result = tool.invoke({"file_id": "F123SECRET"})
    assert "filename" in result.output
    # Gate was not consulted
    assert client.calls == []
