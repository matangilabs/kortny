"""Real-Postgres tests for slice C2: Composio trigger events → Witness DM candidates.

Covers WitnessOpportunityService.project_from_trigger_event, the
ingest_trigger_event(shadow=False) call site, the mode='ask' end-to-end flow
(and the shadow-mode default staying inert), and the ask-only clamp that keeps
composio_trigger-sourced candidates out of autopilot at both the query layer
(autopilot._eligible_candidates) and the policy layer
(ProactiveActionPolicy._decide_autopilot).
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from decimal import Decimal

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
    WitnessOpportunityCandidate,
)
from kortny.db.session import make_engine, make_session_factory, normalize_database_url
from kortny.witness.autopilot import WitnessAutopilot
from kortny.witness.ledger.policy import (
    CandidateInputs,
    DeliveryContext,
    ProactiveActionPolicy,
)
from kortny.witness.opportunities import WitnessOpportunityService
from kortny.witness.runner import WitnessRunner

TEST_POSTGRES_URL = os.environ.get("KORTNY_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    TEST_POSTGRES_URL is None,
    reason="KORTNY_TEST_POSTGRES_URL is required for witness projection tests",
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
    session.execute(delete(WitnessOpportunityCandidate))
    session.execute(delete(ComposioTriggerEvent))
    session.execute(delete(ComposioTriggerSubscription))
    session.execute(delete(ComposioConnection))
    session.execute(delete(Installation))


def _installation(session: Session) -> uuid.UUID:
    inst = Installation(slack_team_id=f"T{uuid.uuid4().hex[:8]}")
    session.add(inst)
    session.flush()
    return inst.id


def _subscription(
    session: Session,
    installation_id: uuid.UUID,
    *,
    connected_account_id: str = "ca_synthetic",
    trigger_slug: str = "GITHUB_PULL_REQUEST_EVENT",
    toolkit_slug: str = "github",
    owner_slack_user_id: str | None = "U_SYNTHETIC_OWNER",
) -> ComposioTriggerSubscription:
    sub = ComposioTriggerSubscription(
        installation_id=installation_id,
        connected_account_id=connected_account_id,
        composio_user_id=f"user_{installation_id.hex[:8]}",
        owner_slack_user_id=owner_slack_user_id,
        toolkit_slug=toolkit_slug,
        trigger_slug=trigger_slug,
        status="active",
    )
    session.add(sub)
    session.flush()
    return sub


def _event(
    session: Session,
    installation_id: uuid.UUID,
    subscription: ComposioTriggerSubscription | None,
    *,
    event_id: str = "evt_synthetic_001",
    trigger_slug: str = "GITHUB_PULL_REQUEST_EVENT",
    decision: str = "ask",
    decision_reason: str = "direct review request on a non-draft pull request",
    importance_score: Decimal | None = Decimal("0.850"),
) -> ComposioTriggerEvent:
    event = ComposioTriggerEvent(
        installation_id=installation_id,
        subscription_id=subscription.id if subscription is not None else None,
        event_id=event_id,
        trigger_slug=trigger_slug,
        connected_account_id=(
            subscription.connected_account_id if subscription is not None else None
        ),
        raw_payload_json={"synthetic": True},
        importance_score=importance_score,
        decision=decision,
        decision_reason=decision_reason,
    )
    session.add(event)
    session.flush()
    return event


def _always_channel(channel_id: str) -> Callable[[str], str | None]:
    def _resolve(user_id: str) -> str | None:
        return channel_id

    return _resolve


def _always_none(user_id: str) -> str | None:
    return None


# ---------------------------------------------------------------------------
# project_from_trigger_event — field contract + guard clauses
# ---------------------------------------------------------------------------


def test_ask_event_with_owner_creates_dm_candidate(db_session: Session) -> None:
    iid = _installation(db_session)
    sub = _subscription(db_session, iid, owner_slack_user_id="U_OWNER_1")
    event = _event(db_session, iid, sub)
    db_session.commit()

    service = WitnessOpportunityService(db_session)
    candidate = service.project_from_trigger_event(
        event=event,
        subscription=sub,
        dm_channel_resolver=_always_channel("D_SYNTHETIC_1"),
    )
    db_session.commit()

    assert candidate is not None
    assert candidate.visibility_scope_type == "dm"
    assert candidate.visibility_scope_id == "D_SYNTHETIC_1"
    assert candidate.channel_id == "D_SYNTHETIC_1"
    assert candidate.target_slack_user_id == "U_OWNER_1"
    assert candidate.source_type == "composio_trigger"
    assert candidate.source_id == str(event.id)
    assert candidate.source_task_id is None
    assert candidate.source_profile_id is None
    assert candidate.candidate_type == "general_help"
    assert candidate.title == "Review requested on GitHub"
    assert candidate.summary == event.decision_reason
    assert float(candidate.confidence_score) == pytest.approx(0.85, abs=0.001)
    assert candidate.status == "candidate"
    assert candidate.evidence_json == [
        {
            "kind": "composio_trigger_event",
            "event_id": str(event.id),
            "trigger_slug": "GITHUB_PULL_REQUEST_EVENT",
        }
    ]


def test_silent_decision_creates_no_candidate(db_session: Session) -> None:
    iid = _installation(db_session)
    sub = _subscription(db_session, iid)
    event = _event(db_session, iid, sub, decision="silent", event_id="evt_silent")
    db_session.commit()

    service = WitnessOpportunityService(db_session)
    candidate = service.project_from_trigger_event(
        event=event,
        subscription=sub,
        dm_channel_resolver=_always_channel("D_SYNTHETIC_2"),
    )
    db_session.commit()

    assert candidate is None
    assert db_session.query(WitnessOpportunityCandidate).count() == 0


def test_digest_decision_creates_no_candidate(db_session: Session) -> None:
    iid = _installation(db_session)
    sub = _subscription(db_session, iid)
    event = _event(db_session, iid, sub, decision="digest", event_id="evt_digest")
    db_session.commit()

    service = WitnessOpportunityService(db_session)
    candidate = service.project_from_trigger_event(
        event=event,
        subscription=sub,
        dm_channel_resolver=_always_channel("D_SYNTHETIC_3"),
    )
    db_session.commit()

    assert candidate is None


def test_no_owner_creates_no_candidate(db_session: Session) -> None:
    iid = _installation(db_session)
    sub = _subscription(db_session, iid, owner_slack_user_id=None)
    event = _event(db_session, iid, sub, event_id="evt_no_owner")
    db_session.commit()

    service = WitnessOpportunityService(db_session)
    candidate = service.project_from_trigger_event(
        event=event,
        subscription=sub,
        dm_channel_resolver=_always_channel("D_SYNTHETIC_4"),
    )
    db_session.commit()

    assert candidate is None


def test_unresolvable_dm_channel_creates_no_candidate_and_does_not_raise(
    db_session: Session,
) -> None:
    iid = _installation(db_session)
    sub = _subscription(db_session, iid, owner_slack_user_id="U_OWNER_2")
    event = _event(db_session, iid, sub, event_id="evt_no_dm")
    db_session.commit()

    service = WitnessOpportunityService(db_session)
    candidate = service.project_from_trigger_event(
        event=event,
        subscription=sub,
        dm_channel_resolver=_always_none,
    )
    db_session.commit()

    assert candidate is None


def test_double_projection_is_idempotent(db_session: Session) -> None:
    iid = _installation(db_session)
    sub = _subscription(db_session, iid, owner_slack_user_id="U_OWNER_3")
    event = _event(db_session, iid, sub, event_id="evt_double")
    db_session.commit()

    service = WitnessOpportunityService(db_session)
    first = service.project_from_trigger_event(
        event=event,
        subscription=sub,
        dm_channel_resolver=_always_channel("D_SYNTHETIC_5"),
    )
    db_session.commit()
    second = service.project_from_trigger_event(
        event=event,
        subscription=sub,
        dm_channel_resolver=_always_channel("D_SYNTHETIC_5"),
    )
    db_session.commit()

    assert first is not None and second is not None
    assert first.id == second.id
    assert db_session.query(WitnessOpportunityCandidate).count() == 1


# ---------------------------------------------------------------------------
# ingest_trigger_event call site — shadow vs ask mode
# ---------------------------------------------------------------------------


def _parsed_ask_github(
    *, event_id: str = "evt_ingest_ask", connected_account_id: str = "ca_synthetic"
) -> ParsedTriggerEvent:
    return ParsedTriggerEvent(
        id=event_id,
        type="composio.trigger.message",
        trigger_slug="GITHUB_PULL_REQUEST_EVENT",
        trigger_id=None,
        connected_account_id=connected_account_id,
        user_id="composio_user_synthetic",
        data={
            "action": "review_requested",
            "pull_request": {"draft": False, "title": "Synthetic PR"},
            "requested_reviewer": {"login": "synthetic-reviewer"},
        },
        timestamp="2026-07-01T00:00:00Z",
    )


def test_shadow_mode_default_creates_no_candidate_even_for_ask_event(
    db_session: Session,
) -> None:
    iid = _installation(db_session)
    _subscription(db_session, iid, owner_slack_user_id="U_OWNER_SHADOW")
    db_session.commit()

    event = ingest_trigger_event(
        db_session,
        installation_id=iid,
        parsed=_parsed_ask_github(event_id="evt_shadow"),
        # shadow defaults to True — mirrors KORTNY_COMPOSIO_TRIGGERS_MODE=shadow.
        dm_channel_resolver=_always_channel("D_SYNTHETIC_SHADOW"),
    )
    db_session.commit()

    assert event is not None
    assert event.decision == "ask"
    assert db_session.query(WitnessOpportunityCandidate).count() == 0


def test_ask_mode_end_to_end_is_eligible_for_dm_digest(db_session: Session) -> None:
    """Mirrors the listener's real call shape: shadow=False + a resolver."""

    iid = _installation(db_session)
    _subscription(db_session, iid, owner_slack_user_id="U_OWNER_E2E")
    db_session.commit()

    event = ingest_trigger_event(
        db_session,
        installation_id=iid,
        parsed=_parsed_ask_github(event_id="evt_e2e"),
        shadow=False,
        dm_channel_resolver=_always_channel("D_SYNTHETIC_E2E"),
    )
    db_session.commit()

    assert event is not None
    assert event.decision == "ask"

    candidate = db_session.query(WitnessOpportunityCandidate).one()
    assert candidate.target_slack_user_id == "U_OWNER_E2E"
    assert candidate.source_type == "composio_trigger"

    eligible = WitnessRunner(db_session)._eligible_dm_candidates(
        installation_id=iid, now=datetime.now(UTC), limit=10
    )
    assert candidate.id in {c.id for c in eligible}


def test_ask_mode_without_resolver_skips_projection(db_session: Session) -> None:
    """shadow=False but no dm_channel_resolver supplied: skip, don't raise."""

    iid = _installation(db_session)
    _subscription(db_session, iid, owner_slack_user_id="U_OWNER_NORESOLVER")
    db_session.commit()

    event = ingest_trigger_event(
        db_session,
        installation_id=iid,
        parsed=_parsed_ask_github(event_id="evt_noresolver"),
        shadow=False,
    )
    db_session.commit()

    assert event is not None
    assert event.decision == "ask"
    assert db_session.query(WitnessOpportunityCandidate).count() == 0


# ---------------------------------------------------------------------------
# Autopilot ask-only clamp (query layer + policy layer)
# ---------------------------------------------------------------------------


def test_autopilot_excludes_composio_trigger_candidates(db_session: Session) -> None:
    iid = _installation(db_session)
    sub = _subscription(db_session, iid, owner_slack_user_id="U_OWNER_AUTOPILOT")
    event = _event(db_session, iid, sub, event_id="evt_autopilot")
    db_session.commit()

    service = WitnessOpportunityService(db_session)
    candidate = service.project_from_trigger_event(
        event=event,
        subscription=sub,
        dm_channel_resolver=_always_channel("D_SYNTHETIC_AUTOPILOT"),
    )
    db_session.commit()
    assert candidate is not None
    # High confidence + candidate status: would otherwise be eligible-looking.
    assert candidate.status == "candidate"
    assert candidate.confidence_score > Decimal("0.5")

    autopilot = WitnessAutopilot(db_session)
    eligible = autopilot._eligible_candidates(
        installation_id=iid,
        now=datetime.now(UTC),
        limit=10,
        min_confidence=Decimal("0.0"),
    )
    assert candidate.id not in {c.id for c in eligible}


def test_policy_decides_ask_for_composio_trigger_source_type() -> None:
    policy = ProactiveActionPolicy()
    candidate_inputs = CandidateInputs(
        confidence_score=Decimal("0.900"),
        reinforcement_count=1,
        evidence_count=1,
        span_days=0,
        candidate_type="general_help",
        created_at=datetime.now(UTC),
        source_type="composio_trigger",
    )
    ctx = DeliveryContext(now=datetime.now(UTC), delivery_threshold=Decimal("0.5"))

    decision = policy.decide("autopilot", candidate_inputs, ctx)

    assert decision.decision == "ask"
    assert decision.reason_code == "composio_trigger_ask_only"


# ---------------------------------------------------------------------------
# Migration 0061 — constraint accepts 'composio_trigger'
# ---------------------------------------------------------------------------


def test_migration_allows_composio_trigger_source_type_on_insert(
    db_session: Session,
) -> None:
    iid = _installation(db_session)
    db_session.commit()

    candidate = WitnessOpportunityCandidate(
        installation_id=iid,
        channel_id="D_SYNTHETIC_MIGRATION",
        target_slack_user_id="U_OWNER_MIGRATION",
        visibility_scope_type="dm",
        visibility_scope_id="D_SYNTHETIC_MIGRATION",
        candidate_type="general_help",
        title="Migration constraint smoke test",
        summary="Directly inserted row to prove the DB check constraint accepts "
        "source_type='composio_trigger'.",
        evidence_json=[],
        source_type="composio_trigger",
        source_id="evt_migration_smoke",
        dedupe_key="migration_smoke_dedupe_key",
        confidence_score=Decimal("0.500"),
        status="candidate",
    )
    db_session.add(candidate)
    db_session.commit()

    reloaded = db_session.get(WitnessOpportunityCandidate, candidate.id)
    assert reloaded is not None
    assert reloaded.source_type == "composio_trigger"
