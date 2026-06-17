"""Assistant thread surface tests (HIG-236).

Real Postgres, following tests/test_slack_ingress.py conventions. The Bolt
assistant utility callables (set_status / set_title / set_suggested_prompts /
save_thread_context / get_thread_context) are faked as recorders so the
task-creation contract is exercised directly without a live Bolt dispatch.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from typing import Any

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, delete, select
from sqlalchemy.orm import Session, sessionmaker

from kortny.config import Settings
from kortny.db.models import (
    AssistantThreadContext,
    Installation,
    SlackInboundEvent,
    Task,
    TaskEvent,
)
from kortny.db.session import (
    make_engine,
    make_session_factory,
    normalize_database_url,
)
from kortny.slack.assistant import (
    PostgresAssistantThreadContextStore,
    _handle_user_message_core,
    context_store_has_thread,
)
from kortny.slack.ingress import SlackIngress

TEST_POSTGRES_URL = os.environ.get("KORTNY_TEST_POSTGRES_URL")


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class FakeSlackClient:
    """Slack client recorder sufficient for the DM ingress path."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.reactions: list[dict[str, Any]] = []

    def chat_postMessage(
        self,
        *,
        channel: str,
        text: str,
        thread_ts: str | None = None,
        blocks: list[dict[str, Any]] | None = None,
        **_kwargs: object,
    ) -> dict[str, Any]:
        self.calls.append({"channel": channel, "text": text, "thread_ts": thread_ts})
        return {
            "ok": True,
            "channel": channel,
            "ts": f"1716800000.{len(self.calls):06d}",
        }

    def reactions_add(
        self, *, channel: str, name: str, timestamp: str
    ) -> dict[str, Any]:
        self.reactions.append(
            {"channel": channel, "name": name, "timestamp": timestamp}
        )
        return {"ok": True}

    def users_info(self, *, user: str) -> dict[str, Any]:
        raise RuntimeError("users_info not configured")

    def conversations_info(self, *, channel: str) -> dict[str, Any]:
        raise RuntimeError("conversations_info not configured")

    def auth_test(self) -> dict[str, Any]:
        return {"ok": True, "user_id": "UBOT"}


class FakeStatus:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, status: str, loading_messages: list[str] | None = None) -> None:
        self.calls.append({"status": status, "loading_messages": loading_messages})


class FakeTitle:
    def __init__(self) -> None:
        self.titles: list[str] = []

    def __call__(self, title: str) -> None:
        self.titles.append(title)


class FakeSaveThreadContext:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> None:
        self.calls += 1


class FakeGetThreadContext:
    def __init__(self, context: dict[str, str] | None = None) -> None:
        self._context = context
        self.calls = 0

    def __call__(self) -> dict[str, str] | None:
        self.calls += 1
        return self._context


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="session")
def engine() -> Iterator[Engine]:
    if TEST_POSTGRES_URL is None:
        pytest.skip("KORTNY_TEST_POSTGRES_URL is required for assistant tests")

    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", normalize_database_url(TEST_POSTGRES_URL))
    command.upgrade(config, "head")

    engine = make_engine(TEST_POSTGRES_URL)
    try:
        yield engine
    finally:
        engine.dispose()


def _cleanup(session: Session) -> None:
    for model in (
        SlackInboundEvent,
        TaskEvent,
        Task,
        AssistantThreadContext,
        Installation,
    ):
        session.execute(delete(model))


@pytest.fixture
def session_factory(engine: Engine) -> Iterator[sessionmaker[Session]]:
    factory = make_session_factory(engine=engine)
    with factory() as session:
        _cleanup(session)
        session.commit()
    yield factory
    with factory() as session:
        _cleanup(session)
        session.commit()


@pytest.fixture
def db_session(engine: Engine) -> Iterator[Session]:
    factory = make_session_factory(engine=engine)
    with factory() as session:
        _cleanup(session)
        session.commit()
        yield session
        session.rollback()
        _cleanup(session)
        session.commit()


def make_settings(**overrides: Any) -> Settings:
    payload: dict[str, Any] = {
        "SLACK_BOT_TOKEN": "xoxb-test",
        "SLACK_APP_TOKEN": "xapp-test",
        "SLACK_SIGNING_SECRET": "signing-secret",
        "LLM_PROVIDER": "openrouter",
        "LLM_API_KEY": "test-key",
        "LLM_MODEL": "openai/gpt-test",
        "COMPOSIO_API_KEY": "composio-key",
        "POSTGRES_URL": TEST_POSTGRES_URL,
        "KORTNY_ASSISTANT_ENABLED": True,
    }
    payload.update(overrides)
    return Settings.model_validate(payload)


def dm_message_event(
    *,
    channel: str = "D123",
    user: str = "U123",
    text: str = "what's on my plate?",
    ts: str = "1716700000.000002",
    thread_ts: str | None = "1716700000.000001",
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "type": "message",
        "channel": channel,
        "channel_type": "im",
        "user": user,
        "text": text,
        "ts": ts,
    }
    if thread_ts is not None:
        event["thread_ts"] = thread_ts
    return event


def message_body(*, event_id: str | None = None) -> dict[str, Any]:
    return {
        "event_id": event_id or f"Ev{uuid.uuid4().hex}",
        "team_id": "T123",
    }


# --------------------------------------------------------------------------- #
# Context store
# --------------------------------------------------------------------------- #


def test_context_store_save_find_round_trip(
    session_factory: sessionmaker[Session],
) -> None:
    store = PostgresAssistantThreadContextStore(session_factory)

    assert store.find(channel_id="D1", thread_ts="t1") is None

    store.save(
        channel_id="D1",
        thread_ts="t1",
        context={"channel_id": "C999", "team_id": "T1"},
    )

    found = store.find(channel_id="D1", thread_ts="t1")
    assert found is not None
    assert found["channel_id"] == "C999"
    assert found.channel_id == "C999"


def test_context_store_upserts_on_same_channel_thread(
    session_factory: sessionmaker[Session],
) -> None:
    store = PostgresAssistantThreadContextStore(session_factory)

    store.save(channel_id="D1", thread_ts="t1", context={"channel_id": "C1"})
    store.save(channel_id="D1", thread_ts="t1", context={"channel_id": "C2"})

    with session_factory() as session:
        rows = list(
            session.scalars(
                select(AssistantThreadContext).where(
                    AssistantThreadContext.channel_id == "D1",
                    AssistantThreadContext.thread_ts == "t1",
                )
            )
        )
    assert len(rows) == 1
    assert rows[0].context_json["channel_id"] == "C2"

    found = store.find(channel_id="D1", thread_ts="t1")
    assert found is not None
    assert found["channel_id"] == "C2"


def test_context_store_find_returns_none_without_channel_id(
    session_factory: sessionmaker[Session],
) -> None:
    store = PostgresAssistantThreadContextStore(session_factory)
    # Empty/contextless rows are not wrappable into Bolt's AssistantThreadContext.
    store.save(channel_id="D1", thread_ts="t1", context={})
    assert store.find(channel_id="D1", thread_ts="t1") is None


def test_context_store_has_thread_helper(
    session_factory: sessionmaker[Session],
) -> None:
    store = PostgresAssistantThreadContextStore(session_factory)
    assert (
        context_store_has_thread(session_factory, channel_id="D1", thread_ts="t1")
        is False
    )
    store.save(channel_id="D1", thread_ts="t1", context={"channel_id": "C1"})
    assert (
        context_store_has_thread(session_factory, channel_id="D1", thread_ts="t1")
        is True
    )


# --------------------------------------------------------------------------- #
# user_message core: status + title + exactly one task with DM identity shape
# --------------------------------------------------------------------------- #


def test_user_message_creates_single_task_with_assistant_identity(
    session_factory: sessionmaker[Session],
) -> None:
    set_status = FakeStatus()
    set_title = FakeTitle()
    get_thread_context = FakeGetThreadContext(
        context={"channel_id": "C777", "team_id": "T123"}
    )
    payload = dm_message_event(text="What's on my plate today?")
    body = message_body(event_id="EvAssist1")

    _handle_user_message_core(
        session_factory=session_factory,
        payload=payload,
        body=body,
        set_status=set_status,
        set_title=set_title,
        get_thread_context=get_thread_context,
        logger=_silent_logger(),
    )

    # Status set once with loading messages; title from first ~40 chars.
    assert len(set_status.calls) == 1
    assert set_status.calls[0]["status"] == "is thinking..."
    assert set_status.calls[0]["loading_messages"]
    assert len(set_status.calls[0]["loading_messages"]) <= 10
    assert set_title.titles == ["What's on my plate today?"]

    with session_factory() as session:
        tasks = list(session.scalars(select(Task)))
        assert len(tasks) == 1
        task = tasks[0]
        assert task.slack_channel_id == "D123"
        assert task.slack_thread_ts == "1716700000.000001"
        assert task.slack_message_ts == "1716700000.000002"
        assert task.slack_user_id == "U123"
        assert task.identity_kind == "slack_message"
        # DM identity shape: slack-message:{channel}:{thread_ts}:{message_ts}
        assert (
            task.identity_key
            == "slack-message:D123:1716700000.000001:1716700000.000002"
        )
        assert task.identity_payload["assistant_context_channel_id"] == "C777"


def test_user_message_is_idempotent_for_same_message(
    session_factory: sessionmaker[Session],
) -> None:
    payload = dm_message_event()
    body = message_body(event_id="EvAssistDup")

    for _ in range(2):
        _handle_user_message_core(
            session_factory=session_factory,
            payload=payload,
            body=body,
            set_status=FakeStatus(),
            set_title=FakeTitle(),
            get_thread_context=FakeGetThreadContext(),
            logger=_silent_logger(),
        )

    with session_factory() as session:
        assert len(list(session.scalars(select(Task)))) == 1


def test_user_message_without_context_omits_hint(
    session_factory: sessionmaker[Session],
) -> None:
    _handle_user_message_core(
        session_factory=session_factory,
        payload=dm_message_event(),
        body=message_body(),
        set_status=FakeStatus(),
        set_title=FakeTitle(),
        get_thread_context=FakeGetThreadContext(context=None),
        logger=_silent_logger(),
    )
    with session_factory() as session:
        task = session.scalar(select(Task))
        assert task is not None
        assert "assistant_context_channel_id" not in task.identity_payload


# --------------------------------------------------------------------------- #
# thread_started suggested prompts (≤4) + save context
# --------------------------------------------------------------------------- #


def test_thread_started_saves_context_without_prompts() -> None:
    # Exercise the registered handler closures directly via a recording App.
    from slack_bolt import Assistant

    captured: dict[str, Any] = {}

    def _recorder(name: str) -> Any:
        # Mirrors slack_bolt's bare-decorator usage: @assistant.thread_started
        # calls the method with the handler function as the sole positional arg.
        def method(self: Any, *args: Any, **kwargs: Any) -> Any:
            if args and callable(args[0]):
                captured[name] = args[0]
                return args[0]

            def deco(func: Any) -> Any:
                captured[name] = func
                return func

            return deco

        return method

    class RecordingAssistant(Assistant):
        thread_started = _recorder("thread_started")
        thread_context_changed = _recorder("thread_context_changed")
        user_message = _recorder("user_message")

    class RecordingApp:
        def __init__(self) -> None:
            self.assistants: list[Any] = []

        def assistant(self, assistant: Any) -> None:
            self.assistants.append(assistant)

    # Patch the Assistant class used by register_assistant.
    import kortny.slack.assistant as assistant_module

    original = assistant_module.Assistant
    assistant_module.Assistant = RecordingAssistant  # type: ignore[misc]
    try:
        factory = _DummyFactory()
        assistant_module.register_assistant(
            RecordingApp(),  # type: ignore[arg-type]
            settings=make_settings(),
            session_factory=factory,  # type: ignore[arg-type]
        )
    finally:
        assistant_module.Assistant = original  # type: ignore[misc]

    save = FakeSaveThreadContext()
    # The handler no longer accepts/sets suggested prompts ("Try these prompts"
    # was removed); it still saves thread context.
    captured["thread_started"](
        say=lambda **_: None,
        save_thread_context=save,
        payload={"assistant_thread": {"channel_id": "D1", "thread_ts": "t1"}},
        logger=_silent_logger(),
    )

    assert save.calls == 1

    # thread_context_changed saves context too.
    save2 = FakeSaveThreadContext()
    captured["thread_context_changed"](
        save_thread_context=save2,
        payload={},
        logger=_silent_logger(),
    )
    assert save2.calls == 1


# --------------------------------------------------------------------------- #
# Dedup guard in SlackIngress.handle_dm
# --------------------------------------------------------------------------- #


def test_handle_dm_skips_known_assistant_thread(db_session: Session) -> None:
    installation = Installation(slack_team_id="T123")
    db_session.add(installation)
    # Mark the thread as a known assistant thread.
    db_session.add(
        AssistantThreadContext(
            channel_id="D123",
            thread_ts="1716700000.000001",
            context_json={"channel_id": "C777"},
        )
    )
    db_session.flush()

    ingress = SlackIngress(
        session=db_session,
        client=FakeSlackClient(),
        settings=make_settings(),
    )
    result = ingress.handle_dm(
        body=message_body(event_id="EvDmAssist"),
        event=dm_message_event(),
    )
    db_session.commit()

    assert result is None
    assert db_session.scalar(select(Task)) is None


def test_handle_dm_without_context_row_still_creates_task(
    db_session: Session,
) -> None:
    ingress = SlackIngress(
        session=db_session,
        client=FakeSlackClient(),
        settings=make_settings(),
    )
    # Plain (unthreaded) DM with no assistant context row -> normal task.
    result = ingress.handle_dm(
        body=message_body(event_id="EvDmNormal"),
        event=dm_message_event(thread_ts=None, ts="1716700000.000009"),
    )
    db_session.commit()

    assert result is not None
    assert result.created is True
    task = db_session.scalar(select(Task))
    assert task is not None
    assert task.slack_channel_id == "D123"


def test_handle_dm_guard_inactive_when_assistant_disabled(
    db_session: Session,
) -> None:
    installation = Installation(slack_team_id="T123")
    db_session.add(installation)
    db_session.add(
        AssistantThreadContext(
            channel_id="D123",
            thread_ts="1716700000.000001",
            context_json={"channel_id": "C777"},
        )
    )
    db_session.flush()

    ingress = SlackIngress(
        session=db_session,
        client=FakeSlackClient(),
        settings=make_settings(KORTNY_ASSISTANT_ENABLED=False),
    )
    result = ingress.handle_dm(
        body=message_body(event_id="EvDmDisabled"),
        event=dm_message_event(),
    )
    db_session.commit()

    # Guard off -> the threaded DM falls through to normal task creation.
    assert result is not None
    assert db_session.scalar(select(Task)) is not None


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


class _DummyFactory:
    """Stand-in session_factory for register_assistant wiring tests."""

    def __call__(self) -> Any:  # pragma: no cover - never invoked here
        raise AssertionError("session not expected in wiring test")

    def begin(self) -> Any:  # pragma: no cover - never invoked here
        raise AssertionError("session not expected in wiring test")


def _silent_logger() -> Any:
    import logging

    logger = logging.getLogger("test.assistant")
    logger.disabled = True
    return logger
