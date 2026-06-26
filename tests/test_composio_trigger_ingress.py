"""Real-Postgres DB integration tests for the Composio trigger ingress pipeline.

Tests: dedup, subscription matching, scorer decisions for all 3 launch triggers.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, delete
from sqlalchemy.orm import Session

from kortny.composio.client import ParsedTriggerEvent
from kortny.composio.trigger_ingress import ingest_trigger_event
from kortny.db.models import (
    ComposioConnection,
    ComposioTriggerEvent,
    ComposioTriggerSubscription,
    Installation,
)
from kortny.db.session import make_engine, make_session_factory, normalize_database_url

TEST_POSTGRES_URL = os.environ.get("KORTNY_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    TEST_POSTGRES_URL is None,
    reason="KORTNY_TEST_POSTGRES_URL is required for trigger ingress tests",
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
    session.execute(delete(ComposioTriggerEvent))
    session.execute(delete(ComposioTriggerSubscription))
    for model in (ComposioConnection, Installation):
        session.execute(delete(model))


def _installation(session: Session) -> uuid.UUID:
    inst = Installation(slack_team_id=f"T{uuid.uuid4().hex[:8]}")
    session.add(inst)
    session.flush()
    return inst.id


def _subscription(
    session: Session,
    installation_id: uuid.UUID,
    *,
    connected_account_id: str = "ca_test",
    trigger_slug: str = "GITHUB_PULL_REQUEST_EVENT",
    toolkit_slug: str = "github",
    status: str = "active",
) -> ComposioTriggerSubscription:
    sub = ComposioTriggerSubscription(
        installation_id=installation_id,
        connected_account_id=connected_account_id,
        composio_user_id=f"user_{installation_id.hex[:8]}",
        toolkit_slug=toolkit_slug,
        trigger_slug=trigger_slug,
        status=status,
    )
    session.add(sub)
    session.flush()
    return sub


def _parsed(
    *,
    trigger_slug: str = "GITHUB_PULL_REQUEST_EVENT",
    event_id: str = "evt_001",
    connected_account_id: str | None = "ca_test",
    user_id: str | None = "user_1",
    trigger_id: str | None = None,
    data: dict | None = None,
) -> ParsedTriggerEvent:
    return ParsedTriggerEvent(
        id=event_id,
        type="composio.trigger.message",
        trigger_slug=trigger_slug,
        trigger_id=trigger_id,
        connected_account_id=connected_account_id,
        user_id=user_id,
        data=data or {},
        timestamp="2026-06-26T00:00:00Z",
    )


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def test_ingest_dedup_same_event_returns_same_row(db_session: Session) -> None:
    iid = _installation(db_session)
    _subscription(db_session, iid)
    db_session.commit()

    p = _parsed(event_id="evt_dedup_01")
    first = ingest_trigger_event(db_session, installation_id=iid, parsed=p)
    db_session.flush()
    second = ingest_trigger_event(db_session, installation_id=iid, parsed=p)
    db_session.flush()

    assert first.id == second.id


def test_ingest_different_event_ids_create_separate_rows(db_session: Session) -> None:
    iid = _installation(db_session)
    _subscription(db_session, iid)
    db_session.commit()

    first = ingest_trigger_event(
        db_session, installation_id=iid, parsed=_parsed(event_id="evt_a")
    )
    db_session.flush()
    second = ingest_trigger_event(
        db_session, installation_id=iid, parsed=_parsed(event_id="evt_b")
    )
    db_session.flush()

    assert first.id != second.id


# ---------------------------------------------------------------------------
# Subscription matching
# ---------------------------------------------------------------------------


def test_ingest_with_matching_subscription_links_subscription(
    db_session: Session,
) -> None:
    iid = _installation(db_session)
    sub = _subscription(db_session, iid, connected_account_id="ca_match")
    db_session.commit()

    event = ingest_trigger_event(
        db_session,
        installation_id=iid,
        parsed=_parsed(connected_account_id="ca_match"),
    )
    db_session.flush()

    assert event.subscription_id == sub.id
    assert event.decision != "unmatched"


def test_ingest_without_subscription_marks_unmatched(db_session: Session) -> None:
    iid = _installation(db_session)
    db_session.commit()

    event = ingest_trigger_event(
        db_session,
        installation_id=iid,
        parsed=_parsed(connected_account_id="ca_no_sub"),
    )
    db_session.flush()

    assert event.subscription_id is None
    assert event.decision == "unmatched"


def test_ingest_no_connected_account_id_marks_unmatched(db_session: Session) -> None:
    iid = _installation(db_session)
    _subscription(db_session, iid)
    db_session.commit()

    event = ingest_trigger_event(
        db_session,
        installation_id=iid,
        parsed=_parsed(connected_account_id=None),
    )
    db_session.flush()

    assert event.decision == "unmatched"


# ---------------------------------------------------------------------------
# GitHub scorer
# ---------------------------------------------------------------------------


def test_github_direct_review_request_asks(db_session: Session) -> None:
    iid = _installation(db_session)
    _subscription(db_session, iid, trigger_slug="GITHUB_PULL_REQUEST_EVENT")
    db_session.commit()

    data = {
        "action": "review_requested",
        "pull_request": {"draft": False, "title": "Add feature X"},
        "requested_reviewer": {"login": "alice"},
    }
    event = ingest_trigger_event(
        db_session,
        installation_id=iid,
        parsed=_parsed(trigger_slug="GITHUB_PULL_REQUEST_EVENT", data=data),
    )
    db_session.flush()

    assert event.decision == "ask"
    assert event.importance_score is not None
    assert float(event.importance_score) > 0.5


def test_github_draft_pr_is_silent(db_session: Session) -> None:
    iid = _installation(db_session)
    _subscription(db_session, iid, trigger_slug="GITHUB_PULL_REQUEST_EVENT")
    db_session.commit()

    data = {
        "action": "review_requested",
        "pull_request": {"draft": True},
        "requested_reviewer": {"login": "alice"},
    }
    event = ingest_trigger_event(
        db_session,
        installation_id=iid,
        parsed=_parsed(trigger_slug="GITHUB_PULL_REQUEST_EVENT", data=data),
    )
    db_session.flush()

    assert event.decision == "silent"


# ---------------------------------------------------------------------------
# Email scorer
# ---------------------------------------------------------------------------


def test_email_important_label_asks(db_session: Session) -> None:
    iid = _installation(db_session)
    _subscription(
        db_session,
        iid,
        trigger_slug="GMAIL_NEW_GMAIL_MESSAGE",
        toolkit_slug="gmail",
    )
    db_session.commit()

    data = {
        "labelIds": ["INBOX", "IMPORTANT"],
        "payload": {
            "headers": [
                {"name": "From", "value": "colleague@example.com"},
                {"name": "Subject", "value": "Q3 planning"},
            ]
        },
    }
    event = ingest_trigger_event(
        db_session,
        installation_id=iid,
        parsed=_parsed(
            trigger_slug="GMAIL_NEW_GMAIL_MESSAGE",
            connected_account_id="ca_test",
            data=data,
        ),
    )
    db_session.flush()

    assert event.decision == "ask"


def test_email_noreply_sender_is_silent(db_session: Session) -> None:
    iid = _installation(db_session)
    _subscription(
        db_session,
        iid,
        trigger_slug="GMAIL_NEW_GMAIL_MESSAGE",
        toolkit_slug="gmail",
    )
    db_session.commit()

    data = {
        "labelIds": ["INBOX"],
        "payload": {
            "headers": [
                {"name": "From", "value": "noreply@github.com"},
                {"name": "Subject", "value": "New activity on your PR"},
            ]
        },
    }
    event = ingest_trigger_event(
        db_session,
        installation_id=iid,
        parsed=_parsed(
            trigger_slug="GMAIL_NEW_GMAIL_MESSAGE",
            connected_account_id="ca_test",
            data=data,
            event_id="evt_email_silent",
        ),
    )
    db_session.flush()

    assert event.decision == "silent"


# ---------------------------------------------------------------------------
# Calendar scorer
# ---------------------------------------------------------------------------


def test_calendar_near_term_accepted_event_asks(db_session: Session) -> None:
    import datetime as dt

    iid = _installation(db_session)
    _subscription(
        db_session,
        iid,
        trigger_slug="GOOGLECALENDAR_EVENT_TRIGGERED",
        toolkit_slug="googlecalendar",
    )
    db_session.commit()

    soon = dt.datetime.now(tz=dt.UTC) + dt.timedelta(hours=2)
    end = soon + dt.timedelta(hours=1)
    data = {
        "event": {
            "start": {"dateTime": soon.isoformat()},
            "end": {"dateTime": end.isoformat()},
            "attendees": [
                {"email": "alice@example.com", "responseStatus": "accepted"},
                {"email": "bob@example.com", "responseStatus": "accepted"},
            ],
        }
    }
    event = ingest_trigger_event(
        db_session,
        installation_id=iid,
        parsed=_parsed(
            trigger_slug="GOOGLECALENDAR_EVENT_TRIGGERED",
            connected_account_id="ca_test",
            data=data,
        ),
    )
    db_session.flush()

    assert event.decision == "ask"


def test_calendar_far_future_event_is_silent(db_session: Session) -> None:
    import datetime as dt

    iid = _installation(db_session)
    _subscription(
        db_session,
        iid,
        trigger_slug="GOOGLECALENDAR_EVENT_TRIGGERED",
        toolkit_slug="googlecalendar",
    )
    db_session.commit()

    far = dt.datetime.now(tz=dt.UTC) + dt.timedelta(hours=48)
    end = far + dt.timedelta(hours=1)
    data = {
        "event": {
            "start": {"dateTime": far.isoformat()},
            "end": {"dateTime": end.isoformat()},
            "attendees": [
                {"email": "alice@example.com", "responseStatus": "accepted"},
                {"email": "bob@example.com", "responseStatus": "accepted"},
            ],
        }
    }
    event = ingest_trigger_event(
        db_session,
        installation_id=iid,
        parsed=_parsed(
            trigger_slug="GOOGLECALENDAR_EVENT_TRIGGERED",
            connected_account_id="ca_test",
            data=data,
            event_id="evt_cal_far",
        ),
    )
    db_session.flush()

    assert event.decision == "silent"


def test_calendar_all_day_event_is_silent(db_session: Session) -> None:
    iid = _installation(db_session)
    _subscription(
        db_session,
        iid,
        trigger_slug="GOOGLECALENDAR_EVENT_TRIGGERED",
        toolkit_slug="googlecalendar",
    )
    db_session.commit()

    data = {
        "event": {
            "start": {"date": "2026-06-27"},  # all-day: no dateTime
            "attendees": [
                {"email": "alice@example.com", "responseStatus": "accepted"},
                {"email": "bob@example.com", "responseStatus": "accepted"},
            ],
        }
    }
    event = ingest_trigger_event(
        db_session,
        installation_id=iid,
        parsed=_parsed(
            trigger_slug="GOOGLECALENDAR_EVENT_TRIGGERED",
            connected_account_id="ca_test",
            data=data,
            event_id="evt_cal_allday",
        ),
    )
    db_session.flush()

    assert event.decision == "silent"
