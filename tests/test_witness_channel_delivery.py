"""HIG-198 + HIG-230: witness channel delivery + autopilot draft tier."""

import json
import os
import uuid
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, delete, func, select
from sqlalchemy.orm import Session

from kortny.config import Settings
from kortny.dashboard.data import get_witness_kpis
from kortny.db.models import (
    Installation,
    LLMUsage,
    ObserveChannelProfile,
    ObservePolicy,
    Schedule,
    SlackChannelMembership,
    SlackInboundEvent,
    SlackSideEffect,
    Task,
    TaskEvent,
    WitnessDeliveryLog,
    WitnessOpportunityCandidate,
)
from kortny.db.models import (
    LLMProvider as DbLLMProvider,
)
from kortny.db.session import make_engine, make_session_factory, normalize_database_url
from kortny.llm import ChatMessage, Completion, TokenUsage
from kortny.scheduler.creation import ScheduleCreationContext, ScheduleDraft
from kortny.slack.ingress import SlackIngress
from kortny.tasks import TaskService
from kortny.tools.types import JsonObject, JsonSchema
from kortny.witness import (
    WITNESS_CHANNEL_SUGGESTION_PURPOSE,
    WitnessAutopilot,
    WitnessRunner,
)
from kortny.witness.autopilot import WITNESS_AUTOPILOT_DRAFT_POSTED_MESSAGE
from kortny.witness.lifecycle import accept_candidate

TEST_POSTGRES_URL = os.environ.get("KORTNY_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    TEST_POSTGRES_URL is None,
    reason="KORTNY_TEST_POSTGRES_URL is required for witness channel delivery tests",
)

NOW = datetime(2026, 6, 11, 15, 0, tzinfo=UTC)
CHANNEL_ID = "CChanDeliv"
SOURCE_THREAD_TS = "1780100000.000100"


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
        WitnessDeliveryLog,
        WitnessOpportunityCandidate,
        Schedule,
        LLMUsage,
        SlackSideEffect,
        SlackInboundEvent,
        ObserveChannelProfile,
        ObservePolicy,
        SlackChannelMembership,
        TaskEvent,
        Task,
        Installation,
    ):
        session.execute(delete(model))


def make_settings(**overrides: Any) -> Settings:
    assert TEST_POSTGRES_URL is not None
    payload: dict[str, Any] = {
        "SLACK_BOT_TOKEN": "xoxb-test",
        "SLACK_APP_TOKEN": "xapp-test",
        "SLACK_SIGNING_SECRET": "signing-secret",
        "LLM_PROVIDER": "openrouter",
        "LLM_API_KEY": "test-key",
        "LLM_MODEL": "openai/gpt-test",
        "COMPOSIO_API_KEY": "composio-key",
        "POSTGRES_URL": TEST_POSTGRES_URL,
        "KORTNY_EMBEDDINGS_BACKEND": "disabled",
    }
    payload.update(overrides)
    return Settings.model_validate(payload)


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
        return {"ok": True, "ts": f"1780300{len(self.calls):03d}.000001"}


class FakeWitnessLLMProvider:
    """LLMProvider fake returning canned completions in order."""

    model = "openai/gpt-test"

    def __init__(self, completions: list[Completion]) -> None:
        self.completions = completions
        self.calls: list[
            tuple[tuple[ChatMessage, ...], tuple[JsonSchema, ...], JsonObject | None]
        ] = []

    def complete(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[JsonSchema] = (),
        *,
        response_format: JsonObject | None = None,
    ) -> Completion:
        self.calls.append((tuple(messages), tuple(tools), response_format))
        if not self.completions:
            raise AssertionError("FakeWitnessLLMProvider received too many calls")
        return self.completions.pop(0)


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


def make_installation(session: Session) -> Installation:
    installation = Installation(slack_team_id=f"T{uuid.uuid4().hex}")
    session.add(installation)
    session.flush()
    return installation


def make_channel_fixture(
    session: Session,
    *,
    proactivity_status: str = "full",
    channel_id: str = CHANNEL_ID,
) -> tuple[Installation, Task]:
    """Installation + active membership + channel policy + threaded source task."""

    installation = make_installation(session)
    membership = SlackChannelMembership(
        installation_id=installation.id,
        channel_id=channel_id,
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
    policy = ObservePolicy(
        installation_id=installation.id,
        scope_type="channel",
        scope_id=channel_id,
        observation_status="active",
        proactivity_status=proactivity_status,
        retention_days=90,
        cooldown_seconds=900,
        enabled_by_user_id="UInvite",
        enabled_at=NOW - timedelta(days=30),
        metadata_json={},
    )
    session.add(policy)
    source_task = TaskService(session).create_task(
        installation_id=installation.id,
        slack_event_id=f"Ev{uuid.uuid4().hex}",
        slack_channel_id=channel_id,
        slack_thread_ts=SOURCE_THREAD_TS,
        slack_message_ts=SOURCE_THREAD_TS,
        slack_user_id="UAsker",
        input="Watch the daily blotter workflow.",
    )
    session.flush()
    return installation, source_task


def make_channel_candidate(
    session: Session,
    installation_id: uuid.UUID,
    *,
    title: str,
    source_task_id: uuid.UUID | None = None,
    channel_id: str = CHANNEL_ID,
    confidence: str = "0.900",
    candidate_type: str = "workflow_gap",
    automation_kind: str | None = "one_shot",
    cadence_suggestion: str | None = None,
    deliverable: str | None = "summarize this week's blotter exceptions",
    evidence_count: int = 2,
    reinforcement_count: int = 2,
    feedback_json: dict[str, Any] | None = None,
) -> WitnessOpportunityCandidate:
    candidate = WitnessOpportunityCandidate(
        installation_id=installation_id,
        channel_id=channel_id,
        target_slack_user_id=None,
        visibility_scope_type="channel",
        visibility_scope_id=channel_id,
        candidate_type=candidate_type,
        title=title,
        summary=f"{title} summary.",
        suggested_action=f"Help with {title}.",
        suggested_message=f"I noticed {title} keeps slipping.",
        evidence_json=[
            {"type": "llm_evidence", "snippet": f"{title} evidence {index}"}
            for index in range(evidence_count)
        ],
        source_type="channel_profile",
        source_id=None,
        source_task_id=source_task_id,
        source_profile_id=None,
        dedupe_key=uuid.uuid4().hex[:32],
        confidence_score=Decimal(confidence),
        confidence_reason="test fixture",
        status="candidate",
        automation_kind=automation_kind,
        cadence_suggestion=cadence_suggestion,
        deliverable=deliverable,
        reinforcement_count=reinforcement_count,
        first_observed_at=NOW - timedelta(days=10),
        feedback_json=feedback_json or {},
        metadata_json={},
        created_at=NOW - timedelta(days=1),
        updated_at=NOW - timedelta(days=1),
    )
    session.add(candidate)
    session.flush()
    return candidate


def run_channel_delivery(
    session: Session,
    installation_id: uuid.UUID,
    client: FakeWitnessSlackClient,
    *,
    now: datetime = NOW,
    channel_posts_per_week: int = 1,
    quiet_hours_start: int | None = None,
    quiet_hours_end: int | None = None,
    delivery_threshold: Decimal = Decimal("0.55"),
) -> Any:
    return WitnessRunner(
        session,
        slack_client=client,
        runner_id="witness-channel-test",
    ).run_once(
        installation_id=installation_id,
        now=now,
        profile_limit=0,
        deliver_private=False,
        autopilot_enabled=False,
        delivery_threshold=delivery_threshold,
        channel_posts_per_week=channel_posts_per_week,
        quiet_hours_start=quiet_hours_start,
        quiet_hours_end=quiet_hours_end,
    )


def delivery_log_rows(session: Session) -> tuple[WitnessDeliveryLog, ...]:
    return tuple(
        session.scalars(
            select(WitnessDeliveryLog).order_by(WitnessDeliveryLog.created_at.asc())
        )
    )


def witness_autopilot_completion(
    *,
    decision: str = "execute_task",
    risk: str = "low",
    action_kind: str = "draft_artifact",
    delivery_target: str = "channel",
    task_input: str | None = "Prepare a draft summary of this week's exceptions.",
    confidence: float = 0.9,
) -> Completion:
    payload = {
        "decision": decision,
        "risk": risk,
        "action_kind": action_kind,
        "delivery_target": delivery_target,
        "requires_user_reply": False,
        "allowed_without_confirmation": True,
        "reason": "Useful, low-risk draft.",
        "task_input": task_input,
        "confidence_score": confidence,
    }
    return Completion(
        content=json.dumps(payload),
        tool_calls=(),
        usage=TokenUsage(input_tokens=180, output_tokens=70),
        cost_usd=Decimal("0.000080"),
        response_id="witness-draft-review",
        model="openai/gpt-test",
    )


# --- Design-doc test: policy full + budget free -> threaded post with copy ---


def test_channel_post_policy_full_budget_free_threaded_with_affordances(
    db_session: Session,
) -> None:
    installation, source_task = make_channel_fixture(db_session)
    candidate = make_channel_candidate(
        db_session,
        installation.id,
        title="Blotter exception follow-up",
        source_task_id=source_task.id,
    )
    client = FakeWitnessSlackClient()

    result = run_channel_delivery(db_session, installation.id, client)
    db_session.flush()

    assert result.delivered_count == 1
    outcome = next(o for o in result.deliveries if o.status == "sent")
    assert outcome.candidate_id == candidate.id
    assert outcome.channel_id == CHANNEL_ID
    assert outcome.decision == "draft"

    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["channel"] == CHANNEL_ID
    # Threaded reply onto the source thread.
    assert call["thread_ts"] == SOURCE_THREAD_TS
    text = call["text"]
    assert "I noticed Blotter exception follow-up keeps slipping." in text
    assert "Evidence:" in text
    assert "Proposed: summarize this week's blotter exceptions" in text
    assert "React :white_check_mark: to set it up" in text
    assert ":no_entry_sign: to drop it" in text

    side_effect = db_session.scalar(
        select(SlackSideEffect).where(
            SlackSideEffect.purpose == WITNESS_CHANNEL_SUGGESTION_PURPOSE
        )
    )
    assert side_effect is not None
    assert side_effect.status == "succeeded"
    assert side_effect.idempotency_key == (
        f"{WITNESS_CHANNEL_SUGGESTION_PURPOSE}:{candidate.id}"
    )
    assert side_effect.target_channel_id == CHANNEL_ID
    assert side_effect.target_thread_ts == SOURCE_THREAD_TS

    assert candidate.status == "sent"
    assert candidate.last_decision == "draft"
    assert candidate.feedback_json["last_action"]["delivery_policy"] == "channel_post"

    rows = delivery_log_rows(db_session)
    assert len(rows) == 1
    assert rows[0].decision == "channel_sent"
    assert rows[0].reason == "sent"
    assert rows[0].candidate_id == candidate.id
    assert rows[0].slack_user_id == f"channel:{CHANNEL_ID}"


def test_channel_post_without_source_thread_goes_top_level(
    db_session: Session,
) -> None:
    installation, _source_task = make_channel_fixture(db_session)
    # No source task and no ts-shaped evidence: top-level post.
    make_channel_candidate(
        db_session,
        installation.id,
        title="Top level item",
        source_task_id=None,
    )
    client = FakeWitnessSlackClient()

    result = run_channel_delivery(db_session, installation.id, client)
    db_session.flush()

    assert result.delivered_count == 1
    assert client.calls[0]["thread_ts"] is None


# --- Design-doc test: policy digest_only -> no channel post (deferred) ---


def test_policy_digest_only_defers_with_policy_reason(db_session: Session) -> None:
    installation, source_task = make_channel_fixture(
        db_session,
        proactivity_status="digest_only",
    )
    candidate = make_channel_candidate(
        db_session,
        installation.id,
        title="Digest only channel",
        source_task_id=source_task.id,
    )
    client = FakeWitnessSlackClient()

    result = run_channel_delivery(db_session, installation.id, client)
    db_session.flush()

    assert client.calls == []
    assert result.delivered_count == 0
    outcome = result.deliveries[0]
    assert outcome.status == "policy_deferred"
    assert outcome.reason == "policy"
    # Deferred, never dropped: the candidate stays pending.
    assert candidate.status == "candidate"
    rows = delivery_log_rows(db_session)
    assert len(rows) == 1
    assert rows[0].decision == "channel_deferred"
    assert rows[0].reason == "policy"
    assert rows[0].slack_user_id == f"channel:{CHANNEL_ID}"

    # Per-tick deferral rows are deduped per (candidate, reason) per day.
    run_channel_delivery(
        db_session,
        installation.id,
        client,
        now=NOW + timedelta(minutes=5),
    )
    db_session.flush()
    assert len(delivery_log_rows(db_session)) == 1

    # Flipping the policy to full delivers on the next tick.
    policy = db_session.scalar(
        select(ObservePolicy).where(ObservePolicy.scope_id == CHANNEL_ID)
    )
    assert policy is not None
    policy.proactivity_status = "full"
    db_session.flush()
    after = run_channel_delivery(
        db_session,
        installation.id,
        client,
        now=NOW + timedelta(hours=1),
    )
    db_session.flush()
    assert after.delivered_count == 1
    assert candidate.status == "sent"


# --- Design-doc test: weekly budget defers, then delivers next window ---


def test_channel_budget_window_defers_then_delivers_next_window(
    db_session: Session,
) -> None:
    installation, source_task = make_channel_fixture(db_session)
    first = make_channel_candidate(
        db_session,
        installation.id,
        title="First channel post",
        source_task_id=source_task.id,
        confidence="0.950",
    )
    second = make_channel_candidate(
        db_session,
        installation.id,
        title="Second channel post",
        source_task_id=source_task.id,
        confidence="0.900",
    )
    client = FakeWitnessSlackClient()

    result = run_channel_delivery(
        db_session,
        installation.id,
        client,
        channel_posts_per_week=1,
    )
    db_session.flush()

    # Budget 1/week: highest-scored candidate posts, the other defers.
    assert result.delivered_count == 1
    assert len(client.calls) == 1
    sent = next(o for o in result.deliveries if o.status == "sent")
    deferred = next(o for o in result.deliveries if o.status == "budget_deferred")
    assert sent.candidate_id == first.id
    assert deferred.candidate_id == second.id
    assert deferred.reason == "budget"
    assert first.status == "sent"
    assert second.status == "candidate"
    deferred_rows = [
        row
        for row in delivery_log_rows(db_session)
        if row.decision == "channel_deferred"
    ]
    assert len(deferred_rows) == 1
    assert deferred_rows[0].reason == "budget"
    assert deferred_rows[0].candidate_id == second.id

    # Inside the window the budget still blocks.
    within = run_channel_delivery(
        db_session,
        installation.id,
        client,
        now=NOW + timedelta(days=2),
        channel_posts_per_week=1,
    )
    db_session.flush()
    assert within.delivered_count == 0
    assert len(client.calls) == 1
    assert second.status == "candidate"

    # Next window: delivered, not dropped.
    after = run_channel_delivery(
        db_session,
        installation.id,
        client,
        now=NOW + timedelta(days=8),
        channel_posts_per_week=1,
    )
    db_session.flush()
    assert after.delivered_count == 1
    assert len(client.calls) == 2
    assert second.status == "sent"


def test_ambient_file_brief_rows_consume_channel_post_budget(
    db_session: Session,
) -> None:
    """HIG-231 integration: the weekly channel window is shared both ways.

    Ambient file briefs log decision='ambient_file_brief' keyed
    'channel:{channel_id}' in witness_delivery_log; the witness channel-post
    budget query must count those rows alongside 'channel_sent'.
    """

    installation, source_task = make_channel_fixture(db_session)
    candidate = make_channel_candidate(
        db_session,
        installation.id,
        title="Post after ambient brief",
        source_task_id=source_task.id,
    )
    db_session.add(
        WitnessDeliveryLog(
            installation_id=installation.id,
            slack_user_id=f"channel:{CHANNEL_ID}",
            candidate_id=None,
            decision="ambient_file_brief",
            reason="sent",
            created_at=NOW - timedelta(days=1),
        )
    )
    db_session.flush()
    client = FakeWitnessSlackClient()

    result = run_channel_delivery(
        db_session,
        installation.id,
        client,
        channel_posts_per_week=1,
    )
    db_session.flush()

    # The brief used this week's only slot: deferred (budget), never dropped.
    assert client.calls == []
    assert result.delivered_count == 0
    deferred = next(o for o in result.deliveries if o.status == "budget_deferred")
    assert deferred.candidate_id == candidate.id
    assert deferred.reason == "budget"
    assert candidate.status == "candidate"

    # Once the brief ages out of the 7-day window, the post delivers.
    after = run_channel_delivery(
        db_session,
        installation.id,
        client,
        now=NOW + timedelta(days=7),
        channel_posts_per_week=1,
    )
    db_session.flush()
    assert after.delivered_count == 1
    assert candidate.status == "sent"


# --- Design-doc test: quiet hours defer channel posts ---


def test_quiet_hours_defer_channel_posts(db_session: Session) -> None:
    installation, source_task = make_channel_fixture(db_session)
    candidate = make_channel_candidate(
        db_session,
        installation.id,
        title="Quiet hours channel item",
        source_task_id=source_task.id,
    )
    client = FakeWitnessSlackClient()
    quiet_now = NOW.replace(hour=22)

    result = run_channel_delivery(
        db_session,
        installation.id,
        client,
        now=quiet_now,
        quiet_hours_start=21,
        quiet_hours_end=23,
    )
    db_session.flush()

    assert client.calls == []
    assert result.delivered_count == 0
    assert result.deliveries[0].status == "quiet_hours_deferred"
    assert result.deliveries[0].reason == "quiet_hours"
    assert candidate.status == "candidate"
    rows = delivery_log_rows(db_session)
    assert len(rows) == 1
    assert rows[0].decision == "channel_deferred"
    assert rows[0].reason == "quiet_hours"

    # Outside quiet hours the same candidate delivers — deferred, not dropped.
    after = run_channel_delivery(
        db_session,
        installation.id,
        client,
        now=quiet_now + timedelta(hours=2),
        quiet_hours_start=21,
        quiet_hours_end=23,
    )
    db_session.flush()
    assert after.delivered_count == 1
    assert candidate.status == "sent"


# --- Decision gate: below-threshold channel candidates stay silent ---


def test_channel_below_threshold_logs_silent_and_never_posts(
    db_session: Session,
) -> None:
    installation, source_task = make_channel_fixture(db_session)
    candidate = make_channel_candidate(
        db_session,
        installation.id,
        title="Weak channel hunch",
        source_task_id=source_task.id,
        confidence="0.300",
    )
    client = FakeWitnessSlackClient()

    result = run_channel_delivery(db_session, installation.id, client)
    db_session.flush()

    assert client.calls == []
    assert result.deliveries[0].status == "silent"
    assert candidate.status == "candidate"
    assert candidate.last_decision == "silent"
    rows = delivery_log_rows(db_session)
    assert len(rows) == 1
    assert rows[0].decision == "silent"
    assert rows[0].slack_user_id == f"channel:{CHANNEL_ID}"


# --- Design-doc test: reaction accept -> materialize_acceptance ---


def _post_suggestion_and_get_ts(
    db_session: Session,
    installation: Installation,
    candidate: WitnessOpportunityCandidate,
) -> str:
    client = FakeWitnessSlackClient()
    result = run_channel_delivery(db_session, installation.id, client)
    db_session.flush()
    sent = next(o for o in result.deliveries if o.candidate_id == candidate.id)
    assert sent.status == "sent"
    message_ts = sent.message_ts
    assert isinstance(message_ts, str)
    return message_ts


def _reaction_event(
    *,
    reaction: str,
    message_ts: str,
    user_id: str = "UReactor",
    channel_id: str = CHANNEL_ID,
) -> tuple[dict[str, Any], dict[str, Any]]:
    body = {"team_id": "", "event_id": f"Ev{uuid.uuid4().hex}"}
    event = {
        "type": "reaction_added",
        "reaction": reaction,
        "user": user_id,
        "item": {"type": "message", "channel": channel_id, "ts": message_ts},
        "event_ts": "1780300999.000001",
    }
    return body, event


def test_reaction_accept_one_shot_materializes_acceptance(
    db_session: Session,
) -> None:
    installation, source_task = make_channel_fixture(db_session)
    candidate = make_channel_candidate(
        db_session,
        installation.id,
        title="Accept via reaction",
        source_task_id=source_task.id,
        automation_kind="one_shot",
        deliverable="summarize this week's blotter exceptions",
    )
    message_ts = _post_suggestion_and_get_ts(db_session, installation, candidate)

    ingress_client = FakeWitnessSlackClient()
    ingress = SlackIngress(
        session=db_session,
        client=ingress_client,
        settings=make_settings(),
    )
    body, event = _reaction_event(reaction="white_check_mark", message_ts=message_ts)
    body["team_id"] = installation.slack_team_id

    result = ingress.handle_reaction_added(body=body, event=event)
    db_session.flush()

    assert result.handled is True
    assert result.action == "accept_witness_suggestion"

    db_session.expire_all()
    refreshed = db_session.get(WitnessOpportunityCandidate, candidate.id)
    assert refreshed is not None
    # one_shot acceptance materializes immediately: candidate automated, task
    # created through the existing HIG-224 path.
    assert refreshed.status == "automated"
    assert refreshed.automated_task_id is not None
    history_actions = [entry["action"] for entry in refreshed.feedback_json["history"]]
    assert "accepted" in history_actions
    assert history_actions[-1] == "automated"
    accepted_entry = next(
        entry
        for entry in refreshed.feedback_json["history"]
        if entry["action"] == "accepted"
    )
    assert accepted_entry["by_user_id"] == "UReactor"

    generated = db_session.get(Task, refreshed.automated_task_id)
    assert generated is not None
    assert generated.identity_key == f"synthetic:witness_automation:{candidate.id}"


def test_reaction_accept_recurring_drafts_schedule_confirmation(
    db_session: Session,
) -> None:
    installation, source_task = make_channel_fixture(db_session)
    candidate = make_channel_candidate(
        db_session,
        installation.id,
        title="Recurring via reaction",
        source_task_id=source_task.id,
        candidate_type="recurring_check",
        automation_kind="recurring",
        cadence_suggestion="every weekday at 8am central time",
        deliverable="post a blotter summary in this channel",
        reinforcement_count=4,
    )
    message_ts = _post_suggestion_and_get_ts(db_session, installation, candidate)

    draft = ScheduleDraft(
        title="Post a blotter summary",
        spec_kind="cron",
        cron_expr="0 8 * * 1-5",
        timezone="America/Chicago",
        next_run_at=NOW + timedelta(days=1),
        cadence_label="Every weekday at 8:00 AM Central time",
        task_input="post a blotter summary in this channel",
        needs_confirmation=False,
        parse_strategy="llm_schedule_parser",
    )
    parser = FakeScheduleParser(draft)
    ingress_client = FakeWitnessSlackClient()
    ingress = SlackIngress(
        session=db_session,
        client=ingress_client,
        settings=make_settings(),
        schedule_fallback_parser=parser,
    )
    body, event = _reaction_event(reaction="white_check_mark", message_ts=message_ts)
    body["team_id"] = installation.slack_team_id

    result = ingress.handle_reaction_added(body=body, event=event)
    db_session.flush()

    assert result.handled is True
    assert result.action == "accept_witness_suggestion"
    assert parser.calls  # the schedule confirmation flow ran

    schedule = db_session.scalar(select(Schedule))
    assert schedule is not None
    assert schedule.status == "proposed"
    assert schedule.metadata_json["witness_candidate_id"] == str(candidate.id)

    # Confirmation blocks posted into the channel through the ingress client.
    confirmation = next(
        call for call in ingress_client.calls if call["blocks"] is not None
    )
    assert confirmation["channel"] == CHANNEL_ID

    db_session.expire_all()
    refreshed = db_session.get(WitnessOpportunityCandidate, candidate.id)
    assert refreshed is not None
    assert refreshed.status == "accepted"
    assert refreshed.feedback_json["last_action"]["action"] == "automation_drafted"


# --- Design-doc test: reaction dismiss -> dismissed + receptivity history ---


def test_reaction_dismiss_records_feedback_for_receptivity(
    db_session: Session,
) -> None:
    installation, source_task = make_channel_fixture(db_session)
    candidate = make_channel_candidate(
        db_session,
        installation.id,
        title="Dismiss via reaction",
        source_task_id=source_task.id,
    )
    message_ts = _post_suggestion_and_get_ts(db_session, installation, candidate)

    ingress = SlackIngress(
        session=db_session,
        client=FakeWitnessSlackClient(),
        settings=make_settings(),
    )
    body, event = _reaction_event(reaction="no_entry_sign", message_ts=message_ts)
    body["team_id"] = installation.slack_team_id

    result = ingress.handle_reaction_added(body=body, event=event)
    db_session.flush()

    assert result.handled is True
    assert result.action == "dismiss_witness_suggestion"

    db_session.expire_all()
    refreshed = db_session.get(WitnessOpportunityCandidate, candidate.id)
    assert refreshed is not None
    assert refreshed.status == "dismissed"
    last = refreshed.feedback_json["last_action"]
    assert last["action"] == "dismissed"
    assert last["by_user_id"] == "UReactor"
    assert last["reason"] == "slack_reaction"

    # The dismissal now feeds channel receptivity: the next candidate of the
    # same type scores lower and stays silent. Without the fresh dismissal
    # this candidate scores 0.9 * 0.75 = 0.675 >= 0.55; with the dismissal
    # penalty (x~0.6) it lands below the threshold. The dismissal feedback
    # entry is stamped with wall-clock time, so score at wall-clock now.
    follow_up = make_channel_candidate(
        db_session,
        installation.id,
        title="Follow up same type",
        source_task_id=source_task.id,
        confidence="0.900",
        evidence_count=1,
        reinforcement_count=1,
    )
    client = FakeWitnessSlackClient()
    after = run_channel_delivery(
        db_session,
        installation.id,
        client,
        now=datetime.now(UTC),
    )
    db_session.flush()
    outcome = next(o for o in after.deliveries if o.candidate_id == follow_up.id)
    assert outcome.status == "silent"
    assert client.calls == []


def test_reaction_on_unrelated_message_is_ignored(db_session: Session) -> None:
    installation, _source_task = make_channel_fixture(db_session)
    ingress = SlackIngress(
        session=db_session,
        client=FakeWitnessSlackClient(),
        settings=make_settings(),
    )
    body, event = _reaction_event(
        reaction="white_check_mark",
        message_ts="1780999999.000001",
    )
    body["team_id"] = installation.slack_team_id

    result = ingress.handle_reaction_added(body=body, event=event)

    assert result.handled is False


# --- Design-doc tests: autopilot draft tier (HIG-230) ---


def make_draft_review_fixture(
    db_session: Session,
    *,
    proactivity_status: str = "full",
    completion: Completion | None = None,
) -> tuple[Installation, WitnessOpportunityCandidate, FakeWitnessLLMProvider]:
    installation, source_task = make_channel_fixture(
        db_session,
        proactivity_status=proactivity_status,
    )
    candidate = make_channel_candidate(
        db_session,
        installation.id,
        title="Weekly exceptions draft",
        source_task_id=source_task.id,
        automation_kind="one_shot",
        deliverable="draft the weekly exceptions summary",
    )
    provider = FakeWitnessLLMProvider([completion or witness_autopilot_completion()])
    return installation, candidate, provider


def run_autopilot(
    db_session: Session,
    installation_id: uuid.UUID,
    provider: FakeWitnessLLMProvider,
    *,
    now: datetime = NOW,
    drafts_per_channel_per_day: int | None = None,
) -> Any:
    return WitnessAutopilot(
        db_session,
        llm_provider=provider,
        provider_name=DbLLMProvider.openrouter,
        drafts_per_channel_per_day=drafts_per_channel_per_day,
    ).run_once(
        installation_id=installation_id,
        now=now,
        limit=1,
        min_confidence=Decimal("0.600"),
    )


def test_draft_tier_creates_threaded_draft_task_without_consuming_candidate(
    db_session: Session,
) -> None:
    installation, candidate, provider = make_draft_review_fixture(db_session)

    result = run_autopilot(db_session, installation.id, provider)
    db_session.flush()

    assert result.reviewed_count == 1
    assert result.drafted_count == 1
    assert result.executed_count == 0
    outcome = result.outcomes[0]
    assert outcome.status == "draft_executed"
    assert outcome.task_id is not None

    draft_task = db_session.get(Task, outcome.task_id)
    assert draft_task is not None
    assert draft_task.identity_key == f"synthetic:witness_draft:{candidate.id}"
    assert draft_task.slack_channel_id == CHANNEL_ID
    # Threaded into the candidate's source thread.
    assert draft_task.slack_thread_ts == SOURCE_THREAD_TS
    # The task input instructs the visible-draft response shape.
    assert "Draft (not sent) - " in draft_task.input
    assert "Tell me changes or say 'go' to finalize." in draft_task.input
    assert "Do not send, publish, or execute anything external." in draft_task.input

    events = tuple(
        db_session.scalars(select(TaskEvent).where(TaskEvent.task_id == draft_task.id))
    )
    assert any(
        event.payload.get("message") == WITNESS_AUTOPILOT_DRAFT_POSTED_MESSAGE
        for event in events
    )

    db_session.expire_all()
    refreshed = db_session.get(WitnessOpportunityCandidate, candidate.id)
    assert refreshed is not None
    # A draft does not consume the candidate.
    assert refreshed.status == "candidate"
    assert refreshed.last_decision == "draft"
    assert refreshed.feedback_json["last_action"]["action"] == "draft_posted"
    assert refreshed.feedback_json["last_action"]["generated_task_id"] == str(
        outcome.task_id
    )

    rows = delivery_log_rows(db_session)
    draft_rows = [row for row in rows if row.decision == "draft_executed"]
    assert len(draft_rows) == 1
    assert draft_rows[0].candidate_id == candidate.id
    assert draft_rows[0].slack_user_id == f"channel:{CHANNEL_ID}"

    # Acceptance later still flows through materialize_acceptance.
    accepted = accept_candidate(
        db_session,
        candidate.id,
        installation_id=installation.id,
        by_user_id="UAdmin",
    )
    assert accepted.status == "accepted"


def test_draft_tier_budget_cap_enforced_per_channel_per_day(
    db_session: Session,
) -> None:
    installation, candidate, provider = make_draft_review_fixture(db_session)
    first = run_autopilot(db_session, installation.id, provider)
    db_session.flush()
    assert first.drafted_count == 1

    # Second one-shot candidate in the same channel, same day: budget blocks.
    second_candidate = make_channel_candidate(
        db_session,
        installation.id,
        title="Second draft attempt",
        source_task_id=candidate.source_task_id,
        automation_kind="one_shot",
        deliverable="draft another summary",
    )
    second_provider = FakeWitnessLLMProvider([witness_autopilot_completion()])
    second = run_autopilot(
        db_session,
        installation.id,
        second_provider,
        now=NOW + timedelta(hours=1),
    )
    db_session.flush()

    assert second.drafted_count == 0
    assert second.deferred_count == 1
    deferred = second.outcomes[0]
    assert deferred.candidate_id == second_candidate.id
    assert deferred.reason is not None
    assert "Draft budget" in deferred.reason
    assert second_candidate.status == "cooldown"

    # Next day the budget window has passed. Park the already-drafted first
    # candidate so the retry deterministically reviews the second one.
    db_session.expire_all()
    drafted = db_session.get(WitnessOpportunityCandidate, candidate.id)
    assert drafted is not None
    drafted.status = "dismissed"
    refreshed = db_session.get(WitnessOpportunityCandidate, second_candidate.id)
    assert refreshed is not None
    refreshed.status = "candidate"
    refreshed.cooldown_until = None
    db_session.flush()
    third_provider = FakeWitnessLLMProvider([witness_autopilot_completion()])
    third = run_autopilot(
        db_session,
        installation.id,
        third_provider,
        now=NOW + timedelta(days=1, hours=1),
    )
    db_session.flush()
    assert third.drafted_count == 1


def test_draft_tier_requires_channel_policy_full(db_session: Session) -> None:
    installation, candidate, provider = make_draft_review_fixture(
        db_session,
        proactivity_status="digest_only",
    )

    result = run_autopilot(db_session, installation.id, provider)
    db_session.flush()

    assert result.drafted_count == 0
    assert result.deferred_count == 1
    assert result.outcomes[0].reason is not None
    assert "proactivity_status" in result.outcomes[0].reason
    assert candidate.status == "cooldown"
    assert delivery_log_rows(db_session) == ()


def test_draft_tier_never_runs_for_ineligible_kinds(db_session: Session) -> None:
    # schedule_management / external_write style work never drafts: the gate
    # only opens for action_kind == draft_artifact, and even then only for
    # one-shot (scorer decision 'draft') candidates.
    installation, source_task = make_channel_fixture(db_session)
    recurring = make_channel_candidate(
        db_session,
        installation.id,
        title="Recurring never drafts",
        source_task_id=source_task.id,
        candidate_type="recurring_check",
        automation_kind="recurring",
        cadence_suggestion="every weekday",
    )
    provider = FakeWitnessLLMProvider([witness_autopilot_completion()])

    result = run_autopilot(db_session, installation.id, provider)
    db_session.flush()

    assert result.drafted_count == 0
    assert result.deferred_count == 1
    assert result.outcomes[0].candidate_id == recurring.id
    assert result.outcomes[0].reason is not None
    assert "scorer decision is draft" in result.outcomes[0].reason
    assert (
        db_session.scalar(
            select(func.count())
            .select_from(Task)
            .where(Task.identity_key.like("synthetic:witness_draft:%"))
        )
        == 0
    )

    # And a schedule_management review never reaches the draft tier at all.
    one_shot = make_channel_candidate(
        db_session,
        installation.id,
        title="Schedule kind never drafts",
        source_task_id=source_task.id,
        automation_kind="one_shot",
    )
    recurring.status = "dismissed"
    db_session.flush()
    schedule_provider = FakeWitnessLLMProvider(
        [witness_autopilot_completion(action_kind="schedule_management")]
    )
    second = run_autopilot(
        db_session,
        installation.id,
        schedule_provider,
        now=NOW + timedelta(hours=1),
    )
    db_session.flush()
    assert second.drafted_count == 0
    assert second.deferred_count == 1
    assert second.outcomes[0].candidate_id == one_shot.id
    assert (
        db_session.scalar(
            select(func.count())
            .select_from(Task)
            .where(Task.identity_key.like("synthetic:witness_draft:%"))
        )
        == 0
    )


def test_draft_tier_does_not_redraft_same_candidate(db_session: Session) -> None:
    installation, candidate, provider = make_draft_review_fixture(db_session)
    first = run_autopilot(db_session, installation.id, provider)
    db_session.flush()
    assert first.drafted_count == 1

    # Even after the cooldown and budget window pass, the same candidate is
    # not re-drafted while its draft awaits feedback.
    candidate.cooldown_until = None
    db_session.flush()
    second_provider = FakeWitnessLLMProvider([witness_autopilot_completion()])
    second = run_autopilot(
        db_session,
        installation.id,
        second_provider,
        now=NOW + timedelta(days=2),
    )
    db_session.flush()
    assert second.drafted_count == 0
    assert second.deferred_count == 1
    assert second.outcomes[0].reason is not None
    assert "already has a posted draft" in second.outcomes[0].reason


# --- Design-doc test: KPI table includes the new decision values ---


def test_witness_kpis_include_new_decision_values(db_session: Session) -> None:
    installation = make_installation(db_session)
    candidate = make_channel_candidate(
        db_session,
        installation.id,
        title="KPI channel candidate",
    )
    log_user = f"channel:{CHANNEL_ID}"
    db_session.add_all(
        [
            WitnessDeliveryLog(
                installation_id=installation.id,
                slack_user_id=log_user,
                candidate_id=candidate.id,
                decision="channel_sent",
                reason="sent",
                created_at=NOW - timedelta(days=1),
            ),
            WitnessDeliveryLog(
                installation_id=installation.id,
                slack_user_id=log_user,
                candidate_id=candidate.id,
                decision="channel_deferred",
                reason="budget",
                created_at=NOW - timedelta(days=2),
            ),
            WitnessDeliveryLog(
                installation_id=installation.id,
                slack_user_id=log_user,
                candidate_id=candidate.id,
                decision="draft_executed",
                reason="sent",
                created_at=NOW - timedelta(days=3),
            ),
        ]
    )
    db_session.flush()

    kpis = get_witness_kpis(
        db_session,
        installation_id=installation.id,
        now=NOW,
        window_days=30,
    )

    counts = {row.decision: row.count for row in kpis.decision_counts}
    assert counts == {
        "notify": 0,
        "question": 0,
        "draft": 0,
        "silent": 0,
        "digest": 0,
        "channel_sent": 1,
        "channel_deferred": 1,
        "draft_executed": 1,
    }
    labels = {row.decision: row.label for row in kpis.decision_counts}
    assert labels["channel_sent"] == "Channel post sent"
    assert labels["channel_deferred"] == "Channel deferred (policy/quiet/budget)"
    assert labels["draft_executed"] == "Draft executed (autopilot)"
    # channel_sent counts as a delivered suggestion.
    assert kpis.delivered_count == 1


# --- Settings: new witness-block fields parse with UPPERCASE aliases ---


def test_settings_parse_channel_and_draft_budgets() -> None:
    settings = make_settings(
        **{
            "KORTNY_WITNESS_CHANNEL_POSTS_PER_WEEK": 3,
            "KORTNY_WITNESS_DRAFTS_PER_CHANNEL_PER_DAY": 2,
        }
    )
    assert settings.witness_channel_posts_per_week == 3
    assert settings.witness_drafts_per_channel_per_day == 2

    defaults = make_settings()
    assert defaults.witness_channel_posts_per_week == 1
    assert defaults.witness_drafts_per_channel_per_day == 1

    with pytest.raises(ValueError, match="CHANNEL_POSTS_PER_WEEK"):
        make_settings(**{"KORTNY_WITNESS_CHANNEL_POSTS_PER_WEEK": -1})
    with pytest.raises(ValueError, match="DRAFTS_PER_CHANNEL_PER_DAY"):
        make_settings(**{"KORTNY_WITNESS_DRAFTS_PER_CHANNEL_PER_DAY": 26})


# --- channel_posts_per_week=0 disables channel delivery entirely ---


def test_channel_delivery_disabled_with_zero_budget(db_session: Session) -> None:
    installation, source_task = make_channel_fixture(db_session)
    candidate = make_channel_candidate(
        db_session,
        installation.id,
        title="Disabled channel delivery",
        source_task_id=source_task.id,
    )
    client = FakeWitnessSlackClient()

    result = run_channel_delivery(
        db_session,
        installation.id,
        client,
        channel_posts_per_week=0,
    )
    db_session.flush()

    assert client.calls == []
    assert result.deliveries == ()
    assert candidate.status == "candidate"
    assert delivery_log_rows(db_session) == ()
