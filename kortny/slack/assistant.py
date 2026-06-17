"""Assistant thread surface registration (HIG-236).

Slack's "Agents & AI Apps" assistant container gives Kortny a dedicated DM-side
thread with suggested prompts, a status indicator, and a title. This module
wires slack_bolt's ``Assistant`` middleware so those threads behave like the
rest of Kortny: a user message becomes a durable ``Task`` through the exact
same ``TaskService`` path ``SlackIngress.handle_dm`` uses, and the worker's
normal ``SlackPoster`` reply lands back in the thread (assistant threads are
ordinary DM threads — zero worker changes needed).

Key platform facts that shape this code:

- ``message.im`` events inside an assistant thread DO NOT carry the channel the
  user was viewing when they opened the assistant. We persist that context per
  thread in ``assistant_thread_context`` (the Postgres-backed
  ``AssistantThreadContextStore`` below) and read it back at task creation as
  the ``assistant_context_channel_id`` identity hint.
- ``assistant.threads.setStatus`` auto-clears after ~2 minutes per call and
  also clears when the app replies. KNOWN V1 LIMITATION: for tasks that run
  longer than ~2 minutes the "is thinking..." status fades before the worker
  posts its reply (no mid-task refresh). Worker-side status refresh and
  streaming responses are HIG-236 phase 2 (see the design doc).

Handler exceptions are logged and never re-raised into Bolt so a single bad
event cannot wedge the assistant listener.
"""

from __future__ import annotations

import logging
from typing import Any

from slack_bolt import App, Assistant
from slack_bolt.context.assistant.thread_context import (
    AssistantThreadContext as BoltAssistantThreadContext,
)
from slack_bolt.context.assistant.thread_context_store.store import (
    AssistantThreadContextStore,
)
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session, sessionmaker

from kortny.config import Settings
from kortny.db.models import AssistantThreadContext as AssistantThreadContextRow
from kortny.db.models import Installation
from kortny.tasks import TaskIdentity, TaskService

logger = logging.getLogger(__name__)

# ≤10 loading messages (Slack caps setStatus loading_messages at 10).
LOADING_MESSAGES: tuple[str, ...] = (
    "Reading the thread...",
    "Checking what I remember...",
    "Lining up tools...",
    "Looking at the channel...",
    "Pulling the relevant context...",
    "Working through it...",
)

ASSISTANT_STATUS = "is thinking..."

# Title is derived from the first message; Slack titles are short.
_TITLE_MAX_CHARS = 40


class PostgresAssistantThreadContextStore(AssistantThreadContextStore):
    """Postgres-backed ``slack_bolt.AssistantThreadContextStore``.

    Implements the installed base class exactly (``save`` / ``find`` with
    keyword-only ``channel_id`` / ``thread_ts`` / ``context``). Each call opens
    its own short-lived session from ``session_factory`` and commits on save —
    the store is shared across events and must not hold a session open.
    """

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def save(
        self,
        *,
        channel_id: str,
        thread_ts: str,
        context: dict[str, str],
    ) -> None:
        payload: dict[str, Any] = dict(context or {})
        with self._session_factory.begin() as session:
            stmt = (
                pg_insert(AssistantThreadContextRow)
                .values(
                    channel_id=channel_id,
                    thread_ts=thread_ts,
                    context_json=payload,
                )
                .on_conflict_do_update(
                    constraint="uq_assistant_thread_context_channel_thread",
                    set_={"context_json": payload},
                )
            )
            session.execute(stmt)

    def find(
        self,
        *,
        channel_id: str,
        thread_ts: str,
    ) -> BoltAssistantThreadContext | None:
        with self._session_factory() as session:
            row = session.scalar(
                select(AssistantThreadContextRow).where(
                    AssistantThreadContextRow.channel_id == channel_id,
                    AssistantThreadContextRow.thread_ts == thread_ts,
                )
            )
            if row is None:
                return None
            payload = dict(row.context_json or {})
        if not payload:
            return None
        # Bolt's AssistantThreadContext requires channel_id; only wrap when the
        # stored context actually carries one (the documented shape).
        if payload.get("channel_id") is None:
            return None
        return BoltAssistantThreadContext(payload)


def context_store_has_thread(
    session_factory: sessionmaker[Session],
    *,
    channel_id: str,
    thread_ts: str,
) -> bool:
    """Return True when (channel_id, thread_ts) is a known assistant thread.

    Used by the ``SlackIngress.handle_dm`` dedup guard so an assistant-thread
    message the Assistant middleware already owns never creates a second task.
    """

    with session_factory() as session:
        return (
            session.scalar(
                select(AssistantThreadContextRow.id).where(
                    AssistantThreadContextRow.channel_id == channel_id,
                    AssistantThreadContextRow.thread_ts == thread_ts,
                )
            )
            is not None
        )


def _resolve_installation(session: Session, team_id: str) -> Installation:
    installation = session.scalar(
        select(Installation).where(Installation.slack_team_id == team_id)
    )
    if installation is None:
        installation = Installation(slack_team_id=team_id)
        session.add(installation)
        session.flush()
    return installation


def _assistant_team_id(body: dict[str, Any], event: dict[str, Any]) -> str | None:
    team = event.get("team") or body.get("team_id") or body.get("team")
    if isinstance(team, str) and team:
        return team
    return None


def _title_from_message(text: str) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= _TITLE_MAX_CHARS:
        return cleaned or "New conversation"
    return cleaned[:_TITLE_MAX_CHARS].rstrip() + "…"


def register_assistant(
    app: App,
    *,
    settings: Settings,
    session_factory: sessionmaker[Session] | None,
) -> None:
    """Register the Slack assistant thread surface (HIG-236).

    Constructs an ``Assistant`` backed by the Postgres context store, registers
    the thread lifecycle handlers, and mounts it via ``app.assistant(...)``.
    """

    if session_factory is None:
        logger.warning(
            "assistant surface disabled: no session_factory provided to register_assistant"
        )
        return

    store = PostgresAssistantThreadContextStore(session_factory)
    assistant = Assistant(thread_context_store=store)

    @assistant.thread_started
    def handle_thread_started(
        say: Any,
        save_thread_context: Any,
        payload: dict[str, Any],
        logger: Any = logger,
    ) -> None:
        try:
            # No suggested prompts: the "Try these prompts" block was noise.
            # Persist whatever context Slack handed us when the thread opened so
            # later message events (which lack it) can resolve the viewing
            # channel. Mirrors Assistant.default_thread_context_changed.
            save_thread_context()
        except Exception:
            logger.exception("assistant thread_started handler failed")

    @assistant.thread_context_changed
    def handle_thread_context_changed(
        save_thread_context: Any,
        payload: dict[str, Any],
        logger: Any = logger,
    ) -> None:
        try:
            save_thread_context()
        except Exception:
            logger.exception("assistant thread_context_changed handler failed")

    @assistant.user_message
    def handle_user_message(
        payload: dict[str, Any],
        body: dict[str, Any],
        set_status: Any,
        set_title: Any,
        get_thread_context: Any,
        logger: Any = logger,
    ) -> None:
        try:
            _handle_user_message_core(
                session_factory=session_factory,
                payload=payload,
                body=body,
                set_status=set_status,
                set_title=set_title,
                get_thread_context=get_thread_context,
                logger=logger,
            )
        except Exception:
            logger.exception("assistant user_message handler failed")

    app.assistant(assistant)


def _handle_user_message_core(
    *,
    session_factory: sessionmaker[Session],
    payload: dict[str, Any],
    body: dict[str, Any],
    set_status: Any,
    set_title: Any,
    get_thread_context: Any,
    logger: Any,
) -> None:
    """Status/title + durable task creation for one assistant user message.

    Extracted from the Bolt listener so the task-creation contract is unit
    testable with fake assistant-utility recorders. Creates the task through
    the SAME ``TaskService`` identity ``handle_dm`` uses (assistant thread ==
    im channel with thread_ts) so dedup keys match exactly, then augments the
    identity payload with the stored context channel hint.
    """

    set_status(ASSISTANT_STATUS, loading_messages=list(LOADING_MESSAGES))

    input_text = str(payload.get("text") or "").strip()
    set_title(_title_from_message(input_text))

    channel_id = payload.get("channel")
    message_ts = payload.get("ts")
    thread_ts = payload.get("thread_ts")
    user_id = payload.get("user")
    if not (
        isinstance(channel_id, str)
        and isinstance(message_ts, str)
        and isinstance(thread_ts, str)
        and isinstance(user_id, str)
    ):
        logger.warning(
            "assistant user_message missing identity fields; skipping task creation"
        )
        return

    team_id = _assistant_team_id(body, payload)
    if team_id is None:
        logger.warning("assistant user_message missing team id; skipping task creation")
        return

    event_id = body.get("event_id")
    slack_event_id = event_id if isinstance(event_id, str) else None

    # Stored thread context carries the channel the user was viewing when they
    # opened the assistant — surface it as an identity-payload hint.
    context_channel_id: str | None = None
    try:
        thread_context = get_thread_context()
    except Exception:
        thread_context = None
        logger.exception("assistant user_message failed to load thread context")
    if thread_context is not None:
        candidate = getattr(thread_context, "channel_id", None) or (
            thread_context.get("channel_id")
            if isinstance(thread_context, dict)
            else None
        )
        if isinstance(candidate, str) and candidate:
            context_channel_id = candidate

    # Build the identity exactly the way handle_dm does (slack_message factory),
    # then attach the assistant context hint without disturbing key/fingerprint.
    identity = TaskIdentity.slack_message(
        channel_id=channel_id,
        message_ts=message_ts,
        thread_ts=thread_ts,
        user_id=user_id,
        input_text=input_text,
        slack_event_id=slack_event_id,
        source_surface="assistant",
    )
    if context_channel_id is not None:
        identity.payload["assistant_context_channel_id"] = context_channel_id

    with session_factory.begin() as session:
        installation = _resolve_installation(session, team_id)
        task_service = TaskService(session)
        task = task_service.create_task(
            installation_id=installation.id,
            slack_event_id=slack_event_id,
            slack_channel_id=channel_id,
            slack_thread_ts=thread_ts,
            slack_message_ts=message_ts,
            slack_user_id=user_id,
            input=input_text,
            identity=identity,
            source_surface="assistant",
        )
        logger.info(
            "assistant user_message created task_id=%s channel=%s thread_ts=%s context_channel=%s",
            task.id,
            channel_id,
            thread_ts,
            context_channel_id,
        )
