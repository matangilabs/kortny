import os
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, delete, func, select
from sqlalchemy.orm import Session

from kortny.db.models import (
    Installation,
    ObserveChannelProfile,
    SlackChannelMembership,
    SlackSideEffect,
    Task,
    TaskEvent,
    WitnessOpportunityCandidate,
)
from kortny.db.session import make_engine, make_session_factory, normalize_database_url
from kortny.tasks import TaskService
from kortny.witness import (
    WITNESS_SUGGESTION_PURPOSE,
    WitnessOpportunityCandidateInput,
    WitnessOpportunityService,
    accept_candidate,
    dismiss_candidate,
    reactivate_candidate,
    send_private_suggestion,
    snooze_candidate,
)

TEST_POSTGRES_URL = os.environ.get("KORTNY_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    TEST_POSTGRES_URL is None,
    reason="KORTNY_TEST_POSTGRES_URL is required for witness opportunity tests",
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


def test_project_from_channel_profile_creates_and_dedupes_candidates(
    db_session: Session,
) -> None:
    task, membership, profile = create_profile_fixture(db_session)
    service = WitnessOpportunityService(db_session)
    candidate_inputs = channel_profile_candidate_inputs()

    result = service.project_from_channel_profile(
        task=task,
        membership=membership,
        profile=profile,
        candidates=candidate_inputs,
        extraction_metadata={"raw_candidate_count": 2},
    )
    db_session.commit()

    candidates = tuple(
        db_session.scalars(
            select(WitnessOpportunityCandidate).order_by(
                WitnessOpportunityCandidate.candidate_type,
                WitnessOpportunityCandidate.title,
            )
        )
    )
    assert result.created_count == 2
    assert result.updated_count == 0
    assert result.skipped_count == 0
    assert len(result.candidate_ids) == 2
    assert {candidate.candidate_type for candidate in candidates} == {
        "data_quality_issue",
        "recurring_check",
    }
    assert all(candidate.status == "candidate" for candidate in candidates)
    assert all(candidate.visibility_scope_type == "channel" for candidate in candidates)
    assert all(candidate.visibility_scope_id == "CWitness" for candidate in candidates)
    assert all(candidate.source_profile_id == profile.id for candidate in candidates)
    assert all(candidate.source_type == "channel_profile" for candidate in candidates)
    assert all(candidate.source_task_id == task.id for candidate in candidates)
    assert all(candidate.metadata_json["raw_candidate_count"] == 2 for candidate in candidates)
    assert all(
        candidate.metadata_json["source"] == "llm_channel_profile_extractor"
        for candidate in candidates
    )
    assert not any(
        candidate.metadata_json.get("source") == "channel_profile_help_opportunity"
        for candidate in candidates
    )
    assert any(
        item.get("type") == "llm_evidence"
        for candidate in candidates
        for item in candidate.evidence_json
    )

    profile.profile_version = 2
    profile.summary = "Updated profile still sees the same daily blotter workflow."
    db_session.flush()

    second = service.project_from_channel_profile(
        task=task,
        membership=membership,
        profile=profile,
        candidates=candidate_inputs,
        extraction_metadata={"raw_candidate_count": 2},
    )
    db_session.commit()

    assert second.created_count == 0
    assert second.updated_count == 2
    assert (
        db_session.scalar(select(func.count()).select_from(WitnessOpportunityCandidate))
        == 2
    )
    refreshed = tuple(db_session.scalars(select(WitnessOpportunityCandidate)))
    assert all(
        candidate.metadata_json["profile_version"] == 2 for candidate in refreshed
    )
    assert all("last_reinforced_at" in candidate.metadata_json for candidate in refreshed)


def test_project_from_task_candidates_persists_llm_proposals(
    db_session: Session,
) -> None:
    task, membership, _profile = create_profile_fixture(db_session)
    membership.channel_name = "rag"
    membership.channel_type = "private_channel"
    task.input = "what do you know about how this channel is used?"
    response_text = "Kortny identified a few useful future watch areas."
    candidate_inputs = (
        WitnessOpportunityCandidateInput(
            candidate_type="unresolved_decision",
            title="Linear decision follow-ups",
            summary="Surface unresolved Linear decisions and blockers in this channel.",
            suggested_action="Track unresolved Linear decisions.",
            suggested_message="I can keep an eye on unresolved Linear decisions here.",
            evidence=("The answer called out Linear summaries and blockers.",),
            confidence_score=Decimal("0.720"),
            confidence_reason="The completed answer directly named this watch area.",
            metadata_json={"extractor": "test"},
        ),
        WitnessOpportunityCandidateInput(
            candidate_type="data_quality_issue",
            title="Integration output quality",
            summary="Flag missing CSV files or broken integration output.",
            suggested_action="Watch for integration output quality issues.",
            suggested_message="I can flag broken tool output when I see it.",
            evidence=("The answer mentioned missing CSV files and broken output.",),
            confidence_score=Decimal("0.680"),
            confidence_reason="The completed answer provided specific evidence.",
            metadata_json={"extractor": "test"},
        ),
    )
    service = WitnessOpportunityService(db_session)

    result = service.project_from_task_candidates(
        task=task,
        candidates=candidate_inputs,
        response_text=response_text,
        extraction_metadata={"raw_candidate_count": 2},
    )
    db_session.commit()

    candidates = tuple(
        db_session.scalars(
            select(WitnessOpportunityCandidate).order_by(
                WitnessOpportunityCandidate.candidate_type
            )
        )
    )

    assert result.created_count == 2
    assert result.updated_count == 0
    assert result.skipped_count == 0
    assert {candidate.candidate_type for candidate in candidates} == {
        "data_quality_issue",
        "unresolved_decision",
    }
    assert all(
        candidate.visibility_scope_type == "private_channel"
        for candidate in candidates
    )
    assert all(candidate.visibility_scope_id == membership.channel_id for candidate in candidates)
    assert all(candidate.source_type == "task_summary" for candidate in candidates)
    assert all(candidate.source_task_id == task.id for candidate in candidates)
    assert all(candidate.source_profile_id is None for candidate in candidates)
    assert all(
        candidate.metadata_json["source"] == "llm_task_response_extractor"
        for candidate in candidates
    )
    assert all(candidate.metadata_json["raw_candidate_count"] == 2 for candidate in candidates)
    assert any(
        item.get("type") == "llm_evidence"
        for candidate in candidates
        for item in candidate.evidence_json
    )

    second = service.project_from_task_candidates(
        task=task,
        candidates=candidate_inputs,
        response_text=response_text,
        extraction_metadata={"raw_candidate_count": 2},
    )
    db_session.commit()

    assert second.created_count == 0
    assert second.updated_count == 2
    assert (
        db_session.scalar(select(func.count()).select_from(WitnessOpportunityCandidate))
        == 2
    )


def test_eligible_private_suggestions_respects_status_and_cooldown(
    db_session: Session,
) -> None:
    task, membership, profile = create_profile_fixture(db_session)
    service = WitnessOpportunityService(db_session)
    service.project_from_channel_profile(
        task=task,
        membership=membership,
        profile=profile,
        candidates=channel_profile_candidate_inputs(),
        extraction_metadata={"raw_candidate_count": 2},
    )
    candidates = tuple(db_session.scalars(select(WitnessOpportunityCandidate)))
    assert len(candidates) == 2
    now = datetime.now(UTC)
    candidates[0].cooldown_until = now + timedelta(hours=2)
    candidates[1].status = "dismissed"
    db_session.commit()

    eligible = service.eligible_private_suggestions(
        installation_id=task.installation_id,
        now=now,
    )

    assert eligible == ()

    candidates[0].cooldown_until = now - timedelta(minutes=5)
    candidates[1].status = "candidate"
    db_session.commit()

    eligible_after_cooldown = service.eligible_private_suggestions(
        installation_id=task.installation_id,
        now=now,
    )
    assert {candidate.id for candidate in eligible_after_cooldown} == {
        candidate.id for candidate in candidates
    }


def test_witness_lifecycle_actions_record_feedback(
    db_session: Session,
) -> None:
    task, membership, profile = create_profile_fixture(db_session)
    service = WitnessOpportunityService(db_session)
    service.project_from_channel_profile(
        task=task,
        membership=membership,
        profile=profile,
        candidates=channel_profile_candidate_inputs(),
        extraction_metadata={"raw_candidate_count": 2},
    )
    candidate = db_session.scalar(
        select(WitnessOpportunityCandidate).order_by(
            WitnessOpportunityCandidate.created_at.asc()
        )
    )
    assert candidate is not None

    snooze_candidate(
        db_session,
        candidate.id,
        installation_id=task.installation_id,
        by_user_id="UAdmin",
        duration=timedelta(days=3),
    )
    db_session.flush()
    assert candidate.status == "cooldown"
    assert candidate.cooldown_until is not None
    assert candidate.feedback_json["last_action"]["action"] == "snoozed"

    reactivate_candidate(
        db_session,
        candidate.id,
        installation_id=task.installation_id,
        by_user_id="UAdmin",
    )
    assert candidate.status == "candidate"
    assert candidate.cooldown_until is None
    assert candidate.feedback_json["last_action"]["action"] == "reactivated"

    accept_candidate(
        db_session,
        candidate.id,
        installation_id=task.installation_id,
        by_user_id="UAdmin",
    )
    assert candidate.status == "accepted"
    assert candidate.feedback_json["last_action"]["action"] == "accepted"

    dismiss_candidate(
        db_session,
        candidate.id,
        installation_id=task.installation_id,
        by_user_id="UAdmin",
        reason="not useful right now",
    )
    assert candidate.status == "dismissed"
    assert candidate.feedback_json["last_action"]["action"] == "dismissed"
    assert candidate.feedback_json["last_action"]["reason"] == "not useful right now"


def test_send_private_suggestion_requires_dm_scope_and_records_outbox(
    db_session: Session,
) -> None:
    task, membership, profile = create_profile_fixture(db_session)
    service = WitnessOpportunityService(db_session)
    service.project_from_channel_profile(
        task=task,
        membership=membership,
        profile=profile,
        candidates=channel_profile_candidate_inputs(),
        extraction_metadata={"raw_candidate_count": 2},
    )
    channel_candidate = db_session.scalar(select(WitnessOpportunityCandidate))
    assert channel_candidate is not None
    with pytest.raises(ValueError, match="Only DM-scoped"):
        send_private_suggestion(
            db_session,
            channel_candidate.id,
            installation_id=task.installation_id,
            by_user_id="UAdmin",
            client=FakeWitnessSlackClient(),
        )

    dm_task = TaskService(db_session).create_task(
        installation_id=task.installation_id,
        slack_event_id=f"Ev{uuid.uuid4().hex}",
        slack_channel_id="DUser123",
        slack_thread_ts="DUser123",
        slack_message_ts="1780200000.000001",
        slack_user_id="UUser123",
        input="Keep an eye on my recurring vendor checklist.",
    )
    service.project_from_task_candidates(
        task=dm_task,
        candidates=(
            WitnessOpportunityCandidateInput(
                candidate_type="recurring_check",
                title="Vendor checklist follow-up",
                summary="Offer to check recurring vendor checklist gaps.",
                suggested_action="Watch recurring vendor checklist gaps.",
                suggested_message=(
                    "I can keep an eye on recurring vendor checklist gaps for you."
                ),
                evidence=("The answer identified recurring checklist gaps.",),
                confidence_score=Decimal("0.800"),
                confidence_reason="The source task named a recurring check.",
                metadata_json={"extractor": "test"},
            ),
        ),
        response_text="I can watch recurring vendor checklist gaps.",
        extraction_metadata={"raw_candidate_count": 1},
    )
    dm_candidate = db_session.scalar(
        select(WitnessOpportunityCandidate).where(
            WitnessOpportunityCandidate.visibility_scope_type == "dm"
        )
    )
    assert dm_candidate is not None
    client = FakeWitnessSlackClient()

    result = send_private_suggestion(
        db_session,
        dm_candidate.id,
        installation_id=task.installation_id,
        by_user_id="UAdmin",
        client=client,
        now=datetime(2026, 6, 5, 12, 30, tzinfo=UTC),
    )
    db_session.flush()

    assert result.channel_id == "DUser123"
    assert result.message_ts == "1780200100.000002"
    assert client.calls == [
        {
            "channel": "DUser123",
            "text": "I can keep an eye on recurring vendor checklist gaps for you.",
            "thread_ts": None,
        }
    ]
    assert dm_candidate.status == "sent"
    assert dm_candidate.last_suggested_at == datetime(2026, 6, 5, 12, 30, tzinfo=UTC)
    assert dm_candidate.feedback_json["last_action"]["action"] == "sent"
    assert (
        dm_candidate.feedback_json["last_action"]["delivery_policy"]
        == "explicit_dm_only"
    )
    side_effect = db_session.scalar(
        select(SlackSideEffect).where(
            SlackSideEffect.id == result.side_effect_id,
        )
    )
    assert side_effect is not None
    assert side_effect.status == "succeeded"
    assert side_effect.purpose == WITNESS_SUGGESTION_PURPOSE


def cleanup_database(session: Session) -> None:
    for model in (
        WitnessOpportunityCandidate,
        SlackSideEffect,
        ObserveChannelProfile,
        SlackChannelMembership,
        TaskEvent,
        Task,
        Installation,
    ):
        session.execute(delete(model))


def create_profile_fixture(
    session: Session,
) -> tuple[Task, SlackChannelMembership, ObserveChannelProfile]:
    installation = Installation(slack_team_id=f"T{uuid.uuid4().hex}")
    session.add(installation)
    session.flush()
    membership = SlackChannelMembership(
        installation_id=installation.id,
        channel_id="CWitness",
        channel_name="daily-blotter",
        channel_type="public_channel",
        membership_status="active",
        discovered_via="member_joined_channel",
        added_by_user_id="UInvite",
        onboarding_status="posted",
        onboarding_message_ts="1780000000.000000",
        metadata_json={},
    )
    session.add(membership)
    session.flush()
    task = TaskService(session).create_task(
        installation_id=installation.id,
        slack_event_id=f"Ev{uuid.uuid4().hex}",
        slack_channel_id=membership.channel_id,
        slack_thread_ts="1780000000.000000",
        slack_message_ts="1780000000.000000",
        slack_user_id="UInvite",
        input="Run channel assessment.",
    )
    profile = ObserveChannelProfile(
        installation_id=installation.id,
        channel_id=membership.channel_id,
        profile_status="active",
        profile_version=1,
        summary="Daily blotter review and exception checks.",
        profile_json={
            "semantic_extraction": {
                "likely_purpose": "Daily trade blotter review.",
                "recurring_topics": ["daily blotter"],
                "workflows": ["Review daily blotter files before PM meeting"],
                "important_entities": ["n8n", "blotter.csv"],
                "assumptions": ["The channel is operational and report-driven."],
                "help_opportunities": [
                    "Summarize daily blotter changes",
                    "Flag missing CSV placeholders and failed file formatting",
                ],
                "evidence": [
                    "Morning blotter uploaded.",
                    "Need a review on ticker changes.",
                ],
                "confidence": "medium",
            }
        },
        assumptions_json=[],
        evidence_refs_json=[
            {
                "type": "tool_result",
                "tool": "slack_channel_history",
                "message_count": 12,
            }
        ],
        confidence_score=Decimal("0.650"),
        confidence_reason="Assessment had enough recent messages.",
        fresh_window_days=30,
        archive_window_days=365,
        observed_range_start_ts="1779900000.000001",
        observed_range_end_ts="1779900200.000003",
        message_count=12,
        file_count=2,
        last_scanned_message_ts="1779900200.000003",
        last_profiled_at=datetime.now(UTC),
        source_task_id=task.id,
        metadata_json={"synthesis": "semantic_llm"},
    )
    session.add(profile)
    session.flush()
    return task, membership, profile


def channel_profile_candidate_inputs() -> tuple[WitnessOpportunityCandidateInput, ...]:
    return (
        WitnessOpportunityCandidateInput(
            candidate_type="recurring_check",
            title="Daily blotter review",
            summary="Offer to summarize daily blotter changes before review.",
            suggested_action="Watch for daily blotter report posts.",
            suggested_message="I can summarize daily blotter changes when they land.",
            evidence=("The channel profile names recurring daily blotter review.",),
            confidence_score=Decimal("0.700"),
            confidence_reason="The profile has repeated report evidence.",
            metadata_json={"extractor": "test_channel_profile_extractor"},
        ),
        WitnessOpportunityCandidateInput(
            candidate_type="data_quality_issue",
            title="CSV placeholder checks",
            summary="Flag missing CSV placeholders and failed file formatting.",
            suggested_action="Watch for broken report output.",
            suggested_message="I can flag broken placeholders in report files.",
            evidence=("The profile names missing CSV placeholders.",),
            confidence_score=Decimal("0.760"),
            confidence_reason="The profile includes direct evidence.",
            metadata_json={"extractor": "test_channel_profile_extractor"},
        ),
    )


class FakeWitnessSlackClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, str | None]] = []

    def chat_postMessage(
        self,
        *,
        channel: str,
        text: str,
        thread_ts: str | None = None,
        blocks: list[dict] | None = None,
    ) -> dict[str, str | bool]:
        self.calls.append(
            {
                "channel": channel,
                "text": text,
                "thread_ts": thread_ts,
            }
        )
        return {"ok": True, "ts": "1780200100.000002"}
