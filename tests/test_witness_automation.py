"""HIG-224: accepted Witness suggestions become standing automations."""

import json
import os
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, delete, func, select
from sqlalchemy.orm import Session

from kortny.config import Settings
from kortny.db.models import (
    Installation,
    LLMUsage,
    ObserveChannelProfile,
    Schedule,
    SlackChannelMembership,
    SlackSideEffect,
    Task,
    TaskEvent,
    TaskStatus,
    WitnessOpportunityCandidate,
)
from kortny.db.session import make_engine, make_session_factory, normalize_database_url
from kortny.scheduler.commands import ScheduleCommandService
from kortny.scheduler.creation import ScheduleCreationContext, ScheduleDraft
from kortny.tasks import TaskService
from kortny.tasks.identity import TaskIdentity
from kortny.witness import (
    WITNESS_AUTOMATION_CONFIRMATION_PURPOSE,
    WitnessOpportunityCandidateInput,
    WitnessOpportunityService,
    accept_candidate,
    materialize_acceptance,
    reactivate_candidate,
    snooze_candidate,
)
from kortny.witness.automation import (
    WITNESS_AUTOMATION_CLARIFICATION_PURPOSE,
    WITNESS_AUTOMATION_DRAFTED_MESSAGE,
)
from kortny.witness.extractor import parse_witness_task_response_extraction

TEST_POSTGRES_URL = os.environ.get("KORTNY_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    TEST_POSTGRES_URL is None,
    reason="KORTNY_TEST_POSTGRES_URL is required for witness automation tests",
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


def cleanup_database(session: Session) -> None:
    for model in (
        WitnessOpportunityCandidate,
        Schedule,
        LLMUsage,
        SlackSideEffect,
        ObserveChannelProfile,
        SlackChannelMembership,
        TaskEvent,
        Task,
        Installation,
    ):
        session.execute(delete(model))


def make_settings(*, automation_enabled: bool = True) -> Settings:
    assert TEST_POSTGRES_URL is not None
    return Settings.model_validate(
        {
            "SLACK_BOT_TOKEN": "xoxb-test",
            "SLACK_APP_TOKEN": "xapp-test",
            "SLACK_SIGNING_SECRET": "signing-secret",
            "LLM_PROVIDER": "openrouter",
            "LLM_API_KEY": "test-key",
            "LLM_MODEL": "openai/gpt-test",
            "COMPOSIO_API_KEY": "composio-key",
            "POSTGRES_URL": TEST_POSTGRES_URL,
            "KORTNY_EMBEDDINGS_BACKEND": "disabled",
            "KORTNY_WITNESS_AUTOMATION_ENABLED": automation_enabled,
        }
    )


class FakeWitnessSlackClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def chat_postMessage(
        self,
        *,
        channel: str,
        text: str,
        thread_ts: str | None = None,
        blocks: list[dict[str, Any]] | None = None,
    ) -> dict[str, str | bool]:
        self.calls.append(
            {
                "channel": channel,
                "text": text,
                "thread_ts": thread_ts,
                "blocks": blocks,
            }
        )
        return {"ok": True, "ts": "1780200100.000002"}


class FakeScheduleParser:
    """ScheduleFallbackParser fake mirroring LLMScheduleParser's surface."""

    def __init__(
        self,
        draft: ScheduleDraft | None,
        *,
        clarifying_question: str | None = None,
    ) -> None:
        self.draft = draft
        self.last_clarifying_question = clarifying_question
        self.calls: list[str] = []

    def parse(
        self,
        *,
        task: Task,
        context: ScheduleCreationContext,
        text: str,
        now: datetime,
    ) -> ScheduleDraft | None:
        self.calls.append(text)
        return self.draft


def create_profile_fixture(
    session: Session,
) -> tuple[Task, SlackChannelMembership, ObserveChannelProfile]:
    installation = Installation(slack_team_id=f"T{uuid.uuid4().hex}")
    session.add(installation)
    session.flush()
    membership = SlackChannelMembership(
        installation_id=installation.id,
        channel_id="CWitnessAuto",
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
        slack_message_ts=f"178000000{uuid.uuid4().int % 10}.{uuid.uuid4().int % 1000000:06d}",
        slack_user_id="UInvite",
        input="Run channel assessment.",
    )
    profile = ObserveChannelProfile(
        installation_id=installation.id,
        channel_id=membership.channel_id,
        profile_status="active",
        profile_version=1,
        summary="Daily blotter review and exception checks.",
        profile_json={},
        assumptions_json=[],
        evidence_refs_json=[],
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
        metadata_json={},
    )
    session.add(profile)
    session.flush()
    return task, membership, profile


def automation_candidate_input(
    *,
    automation_kind: str | None,
    cadence_suggestion: str | None = None,
    deliverable: str | None = None,
    title: str = "Daily trading summary",
) -> WitnessOpportunityCandidateInput:
    return WitnessOpportunityCandidateInput(
        candidate_type="recurring_check",
        title=title,
        summary="Post a trading summary so the team starts with fresh numbers.",
        suggested_action="Post a trading summary in this channel.",
        suggested_message="I can post a trading summary here on a cadence.",
        evidence=("The channel reviews the blotter every morning.",),
        confidence_score=Decimal("0.810"),
        confidence_reason="The profile shows a clear recurring workflow.",
        metadata_json={"extractor": "test"},
        automation_kind=automation_kind,
        cadence_suggestion=cadence_suggestion,
        deliverable=deliverable,
    )


def project_channel_candidate(
    session: Session,
    *,
    automation_kind: str | None,
    cadence_suggestion: str | None = None,
    deliverable: str | None = None,
) -> tuple[Task, WitnessOpportunityCandidate]:
    task, membership, profile = create_profile_fixture(session)
    result = WitnessOpportunityService(session).project_from_channel_profile(
        task=task,
        membership=membership,
        profile=profile,
        candidates=(
            automation_candidate_input(
                automation_kind=automation_kind,
                cadence_suggestion=cadence_suggestion,
                deliverable=deliverable,
            ),
        ),
        extraction_metadata={"raw_candidate_count": 1},
    )
    session.flush()
    candidate = session.get(
        WitnessOpportunityCandidate, uuid.UUID(result.candidate_ids[0])
    )
    assert candidate is not None
    return task, candidate


def task_events(session: Session, task: Task) -> tuple[TaskEvent, ...]:
    return tuple(
        session.scalars(
            select(TaskEvent)
            .where(TaskEvent.task_id == task.id)
            .order_by(TaskEvent.seq.asc())
        )
    )


def recurring_draft() -> ScheduleDraft:
    return ScheduleDraft(
        title="Post a trading summary",
        spec_kind="cron",
        cron_expr="0 8 * * 1-5",
        timezone="America/Chicago",
        next_run_at=datetime.now(UTC) + timedelta(days=1),
        cadence_label="Every weekday at 8:00 AM Central time",
        task_input="post a trading summary in this channel",
        needs_confirmation=False,
        parse_strategy="llm_schedule_parser",
    )


# --- Design-doc test: migration columns + ORM round-trip ---


def test_migration_columns_and_orm_round_trip(db_session: Session) -> None:
    task, candidate = project_channel_candidate(
        db_session,
        automation_kind="recurring",
        cadence_suggestion="every weekday at 8am central time",
        deliverable="post a trading summary in this channel",
    )
    schedule = Schedule(
        installation_id=task.installation_id,
        owner_type="user",
        owner_slack_user_id="UInvite",
        title="Trading summary",
        spec_kind="cron",
        cron_expr="0 8 * * 1-5",
        timezone="UTC",
        next_run_at=datetime.now(UTC) + timedelta(days=1),
        status="proposed",
        task_template={"input": "post a trading summary"},
        metadata_json={},
    )
    db_session.add(schedule)
    db_session.flush()

    candidate.status = "automated"
    candidate.automated_schedule_id = schedule.id
    candidate.automated_task_id = task.id
    db_session.commit()
    db_session.expire_all()

    refreshed = db_session.get(WitnessOpportunityCandidate, candidate.id)
    assert refreshed is not None
    assert refreshed.automation_kind == "recurring"
    assert refreshed.cadence_suggestion == "every weekday at 8am central time"
    assert refreshed.deliverable == "post a trading summary in this channel"
    assert refreshed.status == "automated"
    assert refreshed.automated_schedule_id == schedule.id
    assert refreshed.automated_task_id == task.id


def test_opportunity_service_persists_and_refreshes_automation_fields(
    db_session: Session,
) -> None:
    task, membership, profile = create_profile_fixture(db_session)
    service = WitnessOpportunityService(db_session)
    service.project_from_channel_profile(
        task=task,
        membership=membership,
        profile=profile,
        candidates=(
            automation_candidate_input(
                automation_kind="recurring",
                cadence_suggestion="every morning",
                deliverable="post a trading summary in this channel",
            ),
        ),
        extraction_metadata={},
    )
    candidate = db_session.scalar(select(WitnessOpportunityCandidate))
    assert candidate is not None
    assert candidate.automation_kind == "recurring"
    assert candidate.cadence_suggestion == "every morning"

    # Dedupe update path refreshes provided values, keeps known ones otherwise.
    service.project_from_channel_profile(
        task=task,
        membership=membership,
        profile=profile,
        candidates=(
            automation_candidate_input(
                automation_kind=None,
                cadence_suggestion="every weekday morning",
                deliverable=None,
            ),
        ),
        extraction_metadata={},
    )
    db_session.flush()
    assert (
        db_session.scalar(select(func.count()).select_from(WitnessOpportunityCandidate))
        == 1
    )
    assert candidate.automation_kind == "recurring"
    assert candidate.cadence_suggestion == "every weekday morning"
    assert candidate.deliverable == "post a trading summary in this channel"


# --- Design-doc test: extractor parses new fields; legacy payload still parses ---


def test_extractor_parses_automation_fields() -> None:
    payload = {
        "candidates": [
            {
                "candidate_type": "recurring_check",
                "title": "Daily trading summary",
                "summary": "Post a trading summary every morning.",
                "suggested_action": "Post a trading summary in this channel.",
                "suggested_message": "I can post a trading summary here.",
                "automation_kind": "Recurring",
                "cadence_suggestion": "every weekday at 8am central",
                "deliverable": "post a trading summary in this channel",
                "evidence": ["The channel reviews the blotter daily."],
                "confidence_score": 0.8,
                "confidence_reason": "Clear recurring workflow.",
            },
            {
                "candidate_type": "general_help",
                "title": "Invalid kind falls back",
                "summary": "Watch for one-off questions.",
                "automation_kind": "definitely_not_a_kind",
                "confidence_score": 0.6,
                "confidence_reason": "Some evidence.",
            },
        ]
    }
    extraction = parse_witness_task_response_extraction(json.dumps(payload))

    assert len(extraction.candidates) == 2
    first, second = extraction.candidates
    assert first.automation_kind == "recurring"
    assert first.cadence_suggestion == "every weekday at 8am central"
    assert first.deliverable == "post a trading summary in this channel"
    assert second.automation_kind is None


def test_extractor_legacy_payload_parses_with_none_automation_fields() -> None:
    payload = {
        "candidates": [
            {
                "candidate_type": "recurring_check",
                "title": "Legacy candidate",
                "summary": "Old payload without automation fields.",
                "evidence": ["evidence"],
                "confidence_score": 0.7,
                "confidence_reason": "Legacy.",
            }
        ]
    }
    extraction = parse_witness_task_response_extraction(json.dumps(payload))

    assert len(extraction.candidates) == 1
    candidate = extraction.candidates[0]
    assert candidate.automation_kind is None
    assert candidate.cadence_suggestion is None
    assert candidate.deliverable is None


# --- Design-doc test: accept one_shot -> task with provenance, candidate automated ---


def test_accept_one_shot_creates_task_with_provenance_and_automates_candidate(
    db_session: Session,
) -> None:
    source_task, candidate = project_channel_candidate(
        db_session,
        automation_kind="one_shot",
        deliverable="summarize this week's trading anomalies",
    )
    accept_candidate(
        db_session,
        candidate.id,
        installation_id=source_task.installation_id,
        by_user_id="UAdmin",
    )

    outcome = materialize_acceptance(
        db_session,
        make_settings(),
        candidate,
        accepted_by="UAdmin",
    )
    db_session.commit()

    assert outcome.kind == "one_shot"
    assert outcome.task_id is not None
    assert outcome.failure_reason is None

    generated = db_session.get(Task, outcome.task_id)
    assert generated is not None
    assert generated.status == TaskStatus.pending
    assert generated.slack_channel_id == "CWitnessAuto"
    assert generated.identity_kind == "synthetic"
    assert generated.identity_key == f"synthetic:witness_automation:{candidate.id}"
    assert generated.identity_payload["candidate_id"] == str(candidate.id)
    assert generated.identity_payload["automation_kind"] == "one_shot"
    assert "trading anomalies" in generated.input

    db_session.expire_all()
    refreshed = db_session.get(WitnessOpportunityCandidate, candidate.id)
    assert refreshed is not None
    assert refreshed.status == "automated"
    assert refreshed.automated_task_id == outcome.task_id
    assert refreshed.feedback_json["last_action"]["action"] == "automated"
    assert refreshed.feedback_json["last_action"]["task_id"] == str(outcome.task_id)

    source_events = task_events(db_session, source_task)
    assert any(
        event.payload.get("message") == "witness_candidate_automated"
        and event.payload.get("candidate_id") == str(candidate.id)
        for event in source_events
    )


# --- Design-doc test: accept recurring high-confidence -> draft + confirmation ---


def test_accept_recurring_high_confidence_drafts_schedule_and_posts_confirmation(
    db_session: Session,
) -> None:
    source_task, candidate = project_channel_candidate(
        db_session,
        automation_kind="recurring",
        cadence_suggestion="every weekday at 8am central time",
        deliverable="post a trading summary in this channel",
    )
    accept_candidate(
        db_session,
        candidate.id,
        installation_id=source_task.installation_id,
        by_user_id="UAdmin",
    )
    client = FakeWitnessSlackClient()
    parser = FakeScheduleParser(recurring_draft())

    outcome = materialize_acceptance(
        db_session,
        make_settings(),
        candidate,
        accepted_by="UAdmin",
        slack_client=client,
        schedule_parser=parser,
    )
    db_session.commit()

    assert outcome.kind == "recurring"
    assert outcome.schedule_id is not None
    assert outcome.confirmation_posted is True
    assert outcome.failure_reason is None
    assert parser.calls and "every weekday at 8am central time" in parser.calls[0]

    schedule = db_session.get(Schedule, outcome.schedule_id)
    assert schedule is not None
    assert schedule.status == "proposed"
    assert schedule.metadata_json["witness_candidate_id"] == str(candidate.id)
    assert schedule.task_template["input"] == "post a trading summary in this channel"
    assert schedule.metadata_json["confirmation_required"] is True

    db_session.expire_all()
    refreshed = db_session.get(WitnessOpportunityCandidate, candidate.id)
    assert refreshed is not None
    assert refreshed.status == "accepted"
    assert refreshed.automated_schedule_id is None
    assert refreshed.feedback_json["last_action"]["action"] == "automation_drafted"
    assert refreshed.feedback_json["last_action"]["schedule_id"] == str(schedule.id)

    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["channel"] == "CWitnessAuto"
    assert "confirm" in call["text"].casefold()
    blocks = call["blocks"]
    assert blocks is not None and blocks[0]["type"] == "actions"
    activate = blocks[0]["elements"][0]
    assert activate["text"]["text"] == "Activate"
    assert activate["value"] == str(schedule.id)

    side_effect = db_session.scalar(
        select(SlackSideEffect).where(
            SlackSideEffect.purpose == WITNESS_AUTOMATION_CONFIRMATION_PURPOSE
        )
    )
    assert side_effect is not None
    assert side_effect.status == "succeeded"
    assert side_effect.idempotency_key == (
        f"{WITNESS_AUTOMATION_CONFIRMATION_PURPOSE}:{candidate.id}"
    )

    source_events = task_events(db_session, source_task)
    messages = [event.payload.get("message") for event in source_events]
    assert "schedule_created" in messages
    assert WITNESS_AUTOMATION_DRAFTED_MESSAGE in messages


def _accepted_recurring_with_draft(
    db_session: Session,
) -> tuple[Task, WitnessOpportunityCandidate, Schedule]:
    source_task, candidate = project_channel_candidate(
        db_session,
        automation_kind="recurring",
        cadence_suggestion="every weekday at 8am central time",
        deliverable="post a trading summary in this channel",
    )
    accept_candidate(
        db_session,
        candidate.id,
        installation_id=source_task.installation_id,
        by_user_id="UAdmin",
    )
    outcome = materialize_acceptance(
        db_session,
        make_settings(),
        candidate,
        accepted_by="UAdmin",
        slack_client=FakeWitnessSlackClient(),
        schedule_parser=FakeScheduleParser(recurring_draft()),
    )
    assert outcome.schedule_id is not None
    schedule = db_session.get(Schedule, outcome.schedule_id)
    assert schedule is not None
    return source_task, candidate, schedule


def _confirm_context_and_task(
    db_session: Session,
    schedule: Schedule,
) -> tuple[Task, ScheduleCreationContext]:
    owner = schedule.owner_slack_user_id
    assert owner is not None
    action_task = TaskService(db_session).create_task(
        installation_id=schedule.installation_id,
        slack_channel_id="CWitnessAuto",
        slack_user_id=owner,
        slack_thread_ts="1780000300.000001",
        slack_message_ts=None,
        input=f"schedule button action {schedule.id}",
        identity=TaskIdentity.synthetic(
            source="slack_schedule_action",
            source_id=f"test:{schedule.id}:{uuid.uuid4().hex}",
            input_text="schedule button action",
        ),
        source_surface="slack_action",
    )
    context = ScheduleCreationContext(
        installation_id=schedule.installation_id,
        slack_channel_id="CWitnessAuto",
        slack_user_id=owner,
        slack_thread_ts="1780000300.000001",
        source_surface="slack_action",
        source_task_id=action_task.id,
    )
    return action_task, context


def test_schedule_confirm_action_automates_candidate(db_session: Session) -> None:
    _source_task, candidate, schedule = _accepted_recurring_with_draft(db_session)
    action_task, context = _confirm_context_and_task(db_session, schedule)

    result = ScheduleCommandService(db_session).handle_text(
        task=action_task,
        context=context,
        text=f"activate schedule {schedule.id}",
    )
    db_session.commit()

    assert result is not None
    assert result.action == "activate"
    assert schedule.status == "active"
    db_session.expire_all()
    refreshed = db_session.get(WitnessOpportunityCandidate, candidate.id)
    assert refreshed is not None
    assert refreshed.status == "automated"
    assert refreshed.automated_schedule_id == schedule.id
    assert refreshed.feedback_json["last_action"]["action"] == "automated"
    assert refreshed.feedback_json["last_action"]["schedule_id"] == str(schedule.id)


def test_schedule_cancel_action_records_automation_declined(
    db_session: Session,
) -> None:
    _source_task, candidate, schedule = _accepted_recurring_with_draft(db_session)
    action_task, context = _confirm_context_and_task(db_session, schedule)

    result = ScheduleCommandService(db_session).handle_text(
        task=action_task,
        context=context,
        text=f"cancel schedule {schedule.id}",
    )
    db_session.commit()

    assert result is not None
    assert result.action == "cancel"
    assert schedule.status == "cancelled"
    db_session.expire_all()
    refreshed = db_session.get(WitnessOpportunityCandidate, candidate.id)
    assert refreshed is not None
    assert refreshed.status == "accepted"
    assert refreshed.automated_schedule_id is None
    history_actions = [entry["action"] for entry in refreshed.feedback_json["history"]]
    assert history_actions[-1] == "automation_declined"


# --- Design-doc test: accept recurring low-confidence -> clarifying question ---


def test_accept_recurring_low_confidence_posts_clarifying_question(
    db_session: Session,
) -> None:
    source_task, candidate = project_channel_candidate(
        db_session,
        automation_kind="recurring",
        cadence_suggestion="whenever it makes sense",
        deliverable="post a trading summary in this channel",
    )
    accept_candidate(
        db_session,
        candidate.id,
        installation_id=source_task.installation_id,
        by_user_id="UAdmin",
    )
    client = FakeWitnessSlackClient()
    question = "What time should I post the trading summary?"
    parser = FakeScheduleParser(None, clarifying_question=question)

    outcome = materialize_acceptance(
        db_session,
        make_settings(),
        candidate,
        accepted_by="UAdmin",
        slack_client=client,
        schedule_parser=parser,
    )
    db_session.commit()

    assert outcome.kind == "recurring"
    assert outcome.schedule_id is None
    assert outcome.failure_reason == "schedule_parse_low_confidence"
    assert db_session.scalar(select(func.count()).select_from(Schedule)) == 0

    db_session.expire_all()
    refreshed = db_session.get(WitnessOpportunityCandidate, candidate.id)
    assert refreshed is not None
    assert refreshed.status == "accepted"
    assert refreshed.feedback_json["last_action"]["action"] == "automation_failed"
    assert (
        refreshed.feedback_json["last_action"]["failure_reason"]
        == "schedule_parse_low_confidence"
    )

    assert len(client.calls) == 1
    assert client.calls[0]["text"] == question
    assert client.calls[0]["blocks"] is None
    side_effect = db_session.scalar(
        select(SlackSideEffect).where(
            SlackSideEffect.purpose == WITNESS_AUTOMATION_CLARIFICATION_PURPOSE
        )
    )
    assert side_effect is not None
    assert side_effect.status == "succeeded"


# --- Design-doc test: flag off -> status-only acceptance ---


def test_automation_flag_off_keeps_status_only_acceptance(
    db_session: Session,
) -> None:
    source_task, candidate = project_channel_candidate(
        db_session,
        automation_kind="one_shot",
        deliverable="summarize this week's trading anomalies",
    )
    accept_candidate(
        db_session,
        candidate.id,
        installation_id=source_task.installation_id,
        by_user_id="UAdmin",
    )
    feedback_before = json.dumps(candidate.feedback_json, sort_keys=True)
    client = FakeWitnessSlackClient()

    outcome = materialize_acceptance(
        db_session,
        make_settings(automation_enabled=False),
        candidate,
        accepted_by="UAdmin",
        slack_client=client,
        schedule_parser=FakeScheduleParser(recurring_draft()),
    )
    db_session.commit()

    assert outcome.kind == "disabled"
    db_session.expire_all()
    refreshed = db_session.get(WitnessOpportunityCandidate, candidate.id)
    assert refreshed is not None
    assert refreshed.status == "accepted"
    assert refreshed.automated_task_id is None
    assert refreshed.automated_schedule_id is None
    assert json.dumps(refreshed.feedback_json, sort_keys=True) == feedback_before
    assert client.calls == []
    assert db_session.scalar(select(func.count()).select_from(Schedule)) == 0
    assert (
        db_session.scalar(
            select(func.count())
            .select_from(Task)
            .where(Task.identity_key.like("synthetic:witness_automation:%"))
        )
        == 0
    )


# --- Design-doc test: watch/None keeps today's behavior, noted in feedback ---


def test_watch_candidate_acceptance_keeps_todays_behavior(
    db_session: Session,
) -> None:
    source_task, candidate = project_channel_candidate(
        db_session,
        automation_kind=None,
    )
    accept_candidate(
        db_session,
        candidate.id,
        installation_id=source_task.installation_id,
        by_user_id="UAdmin",
    )

    outcome = materialize_acceptance(
        db_session,
        make_settings(),
        candidate,
        accepted_by="UAdmin",
    )
    db_session.commit()

    assert outcome.kind == "watch"
    db_session.expire_all()
    refreshed = db_session.get(WitnessOpportunityCandidate, candidate.id)
    assert refreshed is not None
    assert refreshed.status == "accepted"
    assert refreshed.feedback_json["last_action"]["action"] == "automation_watch"
    assert refreshed.feedback_json["last_action"]["automation_kind"] == "watch"
    assert db_session.scalar(select(func.count()).select_from(Schedule)) == 0


# --- Design-doc test: automated is terminal ---


def test_automated_candidate_is_terminal(db_session: Session) -> None:
    source_task, candidate = project_channel_candidate(
        db_session,
        automation_kind="one_shot",
    )
    candidate.status = "automated"
    db_session.flush()

    with pytest.raises(ValueError, match="standing automation"):
        snooze_candidate(
            db_session,
            candidate.id,
            installation_id=source_task.installation_id,
            by_user_id="UAdmin",
        )
    with pytest.raises(ValueError, match="standing automation"):
        reactivate_candidate(
            db_session,
            candidate.id,
            installation_id=source_task.installation_id,
            by_user_id="UAdmin",
        )
    with pytest.raises(ValueError, match="standing automation"):
        accept_candidate(
            db_session,
            candidate.id,
            installation_id=source_task.installation_id,
            by_user_id="UAdmin",
        )
