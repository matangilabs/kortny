"""Tests for the Witness autopilot re-fire bug fixes.

Three regression tests:
1. dedup — same title, different summary -> one candidate row (not two)
2. link  — autopilot execution sets automated_task_id (not just feedback_json)
3. preflight — recent equivalent execution defers; old execution or no execution fires
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Iterator, Sequence
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
    ObserveChannelProfile,
    SlackChannelMembership,
    SlackSideEffect,
    Task,
    TaskEvent,
    TaskStatus,
    WitnessOpportunityCandidate,
)
from kortny.db.models import (
    LLMProvider as DbLLMProvider,
)
from kortny.db.session import make_engine, make_session_factory, normalize_database_url
from kortny.llm import ChatMessage, Completion, TokenUsage
from kortny.tasks import TaskService
from kortny.tools.types import JsonObject, JsonSchema
from kortny.witness import (
    WitnessAutopilot,
    WitnessOpportunityCandidateInput,
    WitnessOpportunityService,
)
from kortny.witness.autopilot import (
    _autopilot_preflight_defer_reason,
)

TEST_POSTGRES_URL = os.environ.get("KORTNY_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    TEST_POSTGRES_URL is None,
    reason="KORTNY_TEST_POSTGRES_URL is required for autopilot refire tests",
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
    for model in (
        WitnessOpportunityCandidate,
        LLMUsage,
        SlackSideEffect,
        ObserveChannelProfile,
        SlackChannelMembership,
        TaskEvent,
        Task,
        Installation,
    ):
        session.execute(delete(model))


def _make_installation(session: Session) -> Installation:
    inst = Installation(slack_team_id=f"T{uuid.uuid4().hex[:8]}")
    session.add(inst)
    session.flush()
    return inst


def _make_membership(
    session: Session,
    installation: Installation,
    channel_id: str = "CRefire01",
) -> SlackChannelMembership:
    m = SlackChannelMembership(
        installation_id=installation.id,
        channel_id=channel_id,
        channel_name="test-channel",
        channel_type="public_channel",
        membership_status="active",
        discovered_via="member_joined_channel",
        added_by_user_id="UTestUser",
        onboarding_status="posted",
        onboarding_message_ts="1780000000.000000",
        metadata_json={},
    )
    session.add(m)
    session.flush()
    return m


def _make_task(
    session: Session,
    installation: Installation,
    channel_id: str,
    *,
    message_ts: str | None = None,
) -> Task:
    return TaskService(session).create_task(
        installation_id=installation.id,
        slack_event_id=f"Ev{uuid.uuid4().hex}",
        slack_channel_id=channel_id,
        slack_thread_ts=None,
        slack_message_ts=message_ts
        or f"178{uuid.uuid4().int % 10000000:07d}.{uuid.uuid4().int % 1000000:06d}",
        slack_user_id="UTestUser",
        input="Channel observation task.",
    )


def _make_profile(
    session: Session,
    installation: Installation,
    membership: SlackChannelMembership,
    task: Task,
) -> ObserveChannelProfile:
    profile = ObserveChannelProfile(
        installation_id=installation.id,
        channel_id=membership.channel_id,
        profile_status="active",
        profile_version=1,
        summary="Test profile summary.",
        profile_json={},
        assumptions_json=[],
        evidence_refs_json=[],
        confidence_score=Decimal("0.700"),
        confidence_reason="Enough messages.",
        fresh_window_days=30,
        archive_window_days=365,
        observed_range_start_ts="1779900000.000001",
        observed_range_end_ts="1779900200.000003",
        message_count=10,
        file_count=0,
        last_scanned_message_ts="1779900200.000003",
        last_profiled_at=datetime.now(UTC),
        source_task_id=task.id,
        metadata_json={},
    )
    session.add(profile)
    session.flush()
    return profile


def _candidate_input(title: str, summary: str) -> WitnessOpportunityCandidateInput:
    return WitnessOpportunityCandidateInput(
        candidate_type="recurring_check",
        title=title,
        summary=summary,
        suggested_action=f"Check: {title}",
        suggested_message=f"I noticed: {title}",
        evidence=("Evidence snippet.",),
        confidence_score=Decimal("0.750"),
        confidence_reason="Clear signal.",
        metadata_json={},
        automation_kind="one_shot",
    )


class FakeWitnessLLMProvider:
    """Minimal fake that pops pre-baked Completions in order."""

    model = "openai/gpt-4o-mini"

    def __init__(self, completions: list[Completion]) -> None:
        self.completions = completions

    def complete(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[JsonSchema] = (),
        *,
        response_format: JsonObject | None = None,
        max_output_tokens: int | None = None,
    ) -> Completion:
        if not self.completions:
            raise AssertionError("FakeWitnessLLMProvider received too many calls")
        return self.completions.pop(0)


def _execute_completion(task_input: str = "Check the Q2 pipeline doc.") -> Completion:
    payload = {
        "decision": "execute_task",
        "risk": "low",
        "action_kind": "read_only_analysis",
        "delivery_target": "channel",
        "requires_user_reply": False,
        "allowed_without_confirmation": True,
        "reason": "Safe read-only check.",
        "task_input": task_input,
        "confidence_score": 0.85,
    }
    return Completion(
        content=json.dumps(payload, sort_keys=True),
        tool_calls=(),
        usage=TokenUsage(input_tokens=10, output_tokens=20),
        cost_usd=Decimal("0.000080"),
        model="openai/gpt-4o-mini",
    )


# ---------------------------------------------------------------------------
# Test 1: dedup collapses paraphrased summaries to one candidate
# ---------------------------------------------------------------------------


def test_dedup_same_title_different_summary_creates_one_candidate(
    db_session: Session,
) -> None:
    """Two inputs with the same title but different summary -> ONE candidate row (reinforced)."""
    inst = _make_installation(db_session)
    membership = _make_membership(db_session, inst)
    task = _make_task(db_session, inst, membership.channel_id)
    profile = _make_profile(db_session, inst, membership, task)

    svc = WitnessOpportunityService(db_session)

    title = "Verify Q2 pipeline numbers doc"
    input_a = _candidate_input(
        title=title,
        summary="The Q2 numbers doc may be stale and worth verifying.",
    )
    input_b = _candidate_input(
        title=title,
        summary="Please double-check the Q2 pipeline document for accuracy.",
    )

    result_a = svc.project_from_channel_profile(
        task=task,
        membership=membership,
        profile=profile,
        candidates=(input_a,),
    )
    db_session.commit()

    result_b = svc.project_from_channel_profile(
        task=task,
        membership=membership,
        profile=profile,
        candidates=(input_b,),
    )
    db_session.commit()

    candidates = list(db_session.scalars(select(WitnessOpportunityCandidate)))
    assert len(candidates) == 1, (
        f"Expected 1 candidate (deduped by title), got {len(candidates)}"
    )
    assert result_a.created_count == 1
    assert result_b.created_count == 0
    assert result_b.updated_count == 1
    candidate = candidates[0]
    assert (candidate.reinforcement_count or 1) >= 2


# ---------------------------------------------------------------------------
# Test 2: automated_task_id is set when autopilot executes
# ---------------------------------------------------------------------------


def test_autopilot_sets_automated_task_id_on_execute(
    db_session: Session,
) -> None:
    """When autopilot executes a candidate, candidate.automated_task_id equals the created task id."""
    inst = _make_installation(db_session)
    membership = _make_membership(db_session, inst, channel_id="CRefire02")
    source_task = _make_task(db_session, inst, membership.channel_id)
    profile = _make_profile(db_session, inst, membership, source_task)

    svc = WitnessOpportunityService(db_session)
    ci = _candidate_input(
        title="Verify Q2 pipeline numbers doc",
        summary="The Q2 pipeline doc may be stale.",
    )
    svc.project_from_channel_profile(
        task=source_task,
        membership=membership,
        profile=profile,
        candidates=(ci,),
    )
    db_session.commit()

    candidate = db_session.scalar(select(WitnessOpportunityCandidate))
    assert candidate is not None
    # Raise confidence to ensure autopilot won't skip it
    candidate.confidence_score = Decimal("0.900")
    db_session.flush()

    provider = FakeWitnessLLMProvider([_execute_completion()])
    autopilot = WitnessAutopilot(
        db_session,
        llm_provider=provider,
        provider_name=DbLLMProvider.openrouter,
    )
    result = autopilot.run_once(installation_id=inst.id)
    db_session.commit()

    db_session.expire(candidate)
    assert candidate.automated_task_id is not None, (
        "automated_task_id must be set after autopilot execution"
    )
    assert result.executed_count == 1
    executed_outcome = next(o for o in result.outcomes if o.status == "executed")
    assert executed_outcome.task_id is not None
    assert candidate.automated_task_id == executed_outcome.task_id, (
        "candidate.automated_task_id must equal the created task id"
    )
    # Verify feedback_json still carries generated_task_id for audit trail
    feedback_task_id = (
        (candidate.feedback_json or {}).get("last_action", {}).get("generated_task_id")
    )
    assert feedback_task_id == str(candidate.automated_task_id), (
        "feedback_json should still carry generated_task_id for audit trail"
    )


# ---------------------------------------------------------------------------
# Test 3: preflight defers on recent equivalent execution; fires on new/old
# ---------------------------------------------------------------------------


def _make_auto_task(
    session: Session,
    installation: Installation,
    channel_id: str,
    *,
    task_status: TaskStatus = TaskStatus.succeeded,
    task_created_at: datetime | None = None,
) -> Task:
    """Create an automated task with the specified status and timestamp."""
    auto_task = TaskService(session).create_task(
        installation_id=installation.id,
        slack_event_id=f"Ev{uuid.uuid4().hex}",
        slack_channel_id=channel_id,
        slack_thread_ts=None,
        slack_message_ts=f"178{uuid.uuid4().int % 10000000:07d}.{uuid.uuid4().int % 1000000:06d}",
        slack_user_id="UTestUser",
        input="Autopilot-generated task.",
    )
    if task_created_at is not None:
        auto_task.created_at = task_created_at
    auto_task.status = task_status
    session.flush()
    return auto_task


def test_preflight_defers_when_equivalent_executed_within_cooldown(
    db_session: Session,
) -> None:
    """Candidate with automated_task_id pointing to a succeeded task within 7 days -> deferred.

    The preflight guard fires when a candidate has already been run (automated_task_id
    set to a succeeded task) and has been re-activated (e.g. via reactivate_candidate
    in the dashboard) before the 7-day cooldown expires.
    """
    inst = _make_installation(db_session)
    membership = _make_membership(db_session, inst, channel_id="CRefire03")
    source_task = _make_task(
        db_session, inst, membership.channel_id, message_ts="1780000001.000001"
    )
    profile = _make_profile(db_session, inst, membership, source_task)

    # Create the candidate via the service
    svc = WitnessOpportunityService(db_session)
    ci = _candidate_input(
        title="Verify Q2 pipeline numbers doc",
        summary="The Q2 pipeline doc may be stale.",
    )
    svc.project_from_channel_profile(
        task=source_task,
        membership=membership,
        profile=profile,
        candidates=(ci,),
    )
    db_session.flush()
    candidate = db_session.scalar(select(WitnessOpportunityCandidate))
    assert candidate is not None

    # Simulate a previous autopilot execution 2 days ago (within 7-day cooldown)
    auto_task = _make_auto_task(
        db_session,
        inst,
        membership.channel_id,
        task_status=TaskStatus.succeeded,
        task_created_at=datetime.now(UTC) - timedelta(days=2),
    )
    # The bug-fix sets automated_task_id; the preflight guard checks this field
    candidate.automated_task_id = auto_task.id
    # Simulating a reactivation: status is back to "candidate" but automated_task_id is set
    candidate.status = "candidate"
    db_session.commit()

    reason = _autopilot_preflight_defer_reason(
        db_session,
        candidate,
        source_task=source_task,
        witness_deliver_private=False,
    )
    assert reason is not None, (
        "Preflight should defer when automated task succeeded within 7 days"
    )
    assert "7 days" in reason


def test_preflight_fires_when_no_prior_execution(
    db_session: Session,
) -> None:
    """A brand-new candidate with no prior automated execution -> preflight returns None (fires)."""
    inst = _make_installation(db_session)
    membership = _make_membership(db_session, inst, channel_id="CRefire04")
    source_task = _make_task(db_session, inst, membership.channel_id)
    profile = _make_profile(db_session, inst, membership, source_task)

    svc = WitnessOpportunityService(db_session)
    ci = _candidate_input(
        title="Brand new opportunity",
        summary="Never been done before.",
    )
    svc.project_from_channel_profile(
        task=source_task,
        membership=membership,
        profile=profile,
        candidates=(ci,),
    )
    db_session.flush()
    candidate = db_session.scalar(select(WitnessOpportunityCandidate))
    assert candidate is not None
    assert candidate.automated_task_id is None
    db_session.commit()

    reason = _autopilot_preflight_defer_reason(
        db_session,
        candidate,
        source_task=source_task,
        witness_deliver_private=False,
    )
    assert reason is None, (
        f"Preflight should not defer a brand-new opportunity, got: {reason}"
    )


def test_preflight_fires_when_prior_execution_outside_cooldown(
    db_session: Session,
) -> None:
    """A candidate whose automated task ran 10 days ago (outside 7-day cooldown) -> fires."""
    inst = _make_installation(db_session)
    membership = _make_membership(db_session, inst, channel_id="CRefire05")
    source_task = _make_task(
        db_session, inst, membership.channel_id, message_ts="1780000002.000001"
    )
    profile = _make_profile(db_session, inst, membership, source_task)

    svc = WitnessOpportunityService(db_session)
    ci = _candidate_input(
        title="Old opportunity already executed",
        summary="Executed once before but the cooldown has expired.",
    )
    svc.project_from_channel_profile(
        task=source_task,
        membership=membership,
        profile=profile,
        candidates=(ci,),
    )
    db_session.flush()
    candidate = db_session.scalar(select(WitnessOpportunityCandidate))
    assert candidate is not None

    # Auto task ran 10 days ago — OUTSIDE the 7-day cooldown
    auto_task = _make_auto_task(
        db_session,
        inst,
        membership.channel_id,
        task_status=TaskStatus.succeeded,
        task_created_at=datetime.now(UTC) - timedelta(days=10),
    )
    candidate.automated_task_id = auto_task.id
    candidate.status = "candidate"
    db_session.commit()

    reason = _autopilot_preflight_defer_reason(
        db_session,
        candidate,
        source_task=source_task,
        witness_deliver_private=False,
    )
    assert reason is None, (
        f"Preflight should not defer when automated task was outside cooldown, got: {reason}"
    )
