import os
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, delete, func, select
from sqlalchemy.orm import Session

from kortny.approvals import (
    TOOL_APPROVAL_PROMPT_PURPOSE,
    TOOL_APPROVAL_REJECTED_PURPOSE,
    TOOL_APPROVAL_REQUIRED_MESSAGE,
)
from kortny.db.models import (
    Artifact,
    ComposioConnection,
    EncryptedSecret,
    Installation,
    LLMUsage,
    ModelPricing,
    ObservationEvent,
    ObservePolicy,
    Schedule,
    SlackChannelMembership,
    SlackIdentity,
    SlackInboundEvent,
    SlackSideEffect,
    Task,
    TaskEvent,
    TaskEventType,
    TaskStatus,
)
from kortny.db.session import make_engine, make_session_factory, normalize_database_url
from kortny.intent import (
    IntentClassification,
    IntentDecision,
    IntentRequest,
    IntentSurface,
    ModelTier,
)
from kortny.observe.assessment import CHANNEL_ASSESSMENT_REQUESTED_MESSAGE
from kortny.scheduler import ScheduleDraft
from kortny.slack import SlackIngress, acknowledge_then_handle
from kortny.slack.ingress import INTENT_CLASSIFIED_MESSAGE, is_bare_app_mention
from kortny.slack.outbox import slack_install_intro_key
from kortny.slack.reactions import ACK_REACTION_ADDED_MESSAGE, ReactionChoice
from kortny.tasks import TaskService

TEST_POSTGRES_URL = os.environ.get("KORTNY_TEST_POSTGRES_URL")


class FakeSlackClient:
    def __init__(
        self,
        *,
        reaction_error: Exception | None = None,
        user_info: dict[str, Any] | None = None,
        channel_info: dict[str, Any] | None = None,
        identity_error: Exception | None = None,
        auth_test: dict[str, Any] | None = None,
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self.reactions: list[dict[str, Any]] = []
        self.identity_calls: list[dict[str, str]] = []
        self.reaction_error = reaction_error
        self.user_info = user_info
        self.channel_info = channel_info
        self.identity_error = identity_error
        self.auth_test_response = auth_test or {"ok": True, "user_id": "UBOT"}

    def chat_postMessage(
        self,
        *,
        channel: str,
        text: str,
        thread_ts: str | None = None,
        blocks: list[dict[str, Any]] | None = None,
        **_kwargs: object,
    ) -> dict[str, Any]:
        call: dict[str, Any] = {
            "channel": channel,
            "text": text,
            "thread_ts": thread_ts,
        }
        if blocks is not None:
            call["blocks"] = blocks
        self.calls.append(call)
        return {
            "ok": True,
            "channel": channel,
            "ts": f"1716400000.{len(self.calls):06d}",
        }

    def reactions_add(
        self,
        *,
        channel: str,
        name: str,
        timestamp: str,
    ) -> dict[str, Any]:
        if self.reaction_error is not None:
            raise self.reaction_error
        self.reactions.append(
            {
                "channel": channel,
                "name": name,
                "timestamp": timestamp,
            }
        )
        return {"ok": True}

    def users_info(self, *, user: str) -> dict[str, Any]:
        self.identity_calls.append({"method": "users_info", "id": user})
        if self.identity_error is not None:
            raise self.identity_error
        if self.user_info is None:
            raise RuntimeError("users_info not configured")
        return self.user_info

    def conversations_info(self, *, channel: str) -> dict[str, Any]:
        self.identity_calls.append({"method": "conversations_info", "id": channel})
        if self.identity_error is not None:
            raise self.identity_error
        if self.channel_info is None:
            raise RuntimeError("conversations_info not configured")
        return self.channel_info

    def auth_test(self) -> dict[str, Any]:
        self.identity_calls.append({"method": "auth_test", "id": "self"})
        return self.auth_test_response


class FakeAcknowledgementGenerator:
    def __init__(
        self,
        text: str = "I'll pull that together and post it here.",
        *,
        error: Exception | None = None,
    ) -> None:
        self.text = text
        self.error = error
        self.calls: list[str] = []

    def generate(
        self,
        *,
        session: Session,
        task: Task,
        task_service: TaskService,
    ) -> str:
        self.calls.append(task.input)
        if self.error is not None:
            raise self.error
        return self.text


class FakeReactionProvider:
    def __init__(self, name: str = "eyes", intent: str = "working") -> None:
        self.choice = ReactionChoice(name=name, intent=intent)
        self.calls: list[dict[str, str]] = []
        self.intent_decisions: list[IntentDecision | None] = []

    def acknowledgement_reaction(
        self,
        *,
        input_text: str,
        source: str,
        intent_decision: IntentDecision | None = None,
    ) -> ReactionChoice:
        self.calls.append({"input_text": input_text, "source": source})
        self.intent_decisions.append(intent_decision)
        return self.choice

    def completion_reaction(
        self, *, input_text: str, source: str, succeeded: bool
    ) -> ReactionChoice:
        del input_text, source, succeeded
        return ReactionChoice(name="heavy_check_mark", intent="completed")


class FakeIntentClassifier:
    def __init__(
        self,
        decision: IntentDecision | None = None,
        *,
        error: Exception | None = None,
        require_task_id: bool = True,
    ) -> None:
        self.decision = decision or intent_decision()
        self.error = error
        self.require_task_id = require_task_id
        self.calls: list[tuple[uuid.UUID | None, IntentRequest]] = []

    def classify(
        self,
        *,
        request: IntentRequest,
        task_id: uuid.UUID | None = None,
    ) -> IntentDecision:
        if self.require_task_id and task_id is None:
            raise AssertionError("FakeIntentClassifier expected a task_id")
        self.calls.append((task_id, request))
        if self.error is not None:
            raise self.error
        return self.decision


class FakeScheduleFallbackParser:
    def __init__(self, draft: ScheduleDraft) -> None:
        self.draft = draft
        self.calls: list[dict[str, object]] = []

    def parse(
        self,
        *,
        task: Task,
        context: object,
        text: str,
        now: object,
    ) -> ScheduleDraft | None:
        self.calls.append(
            {
                "task_id": task.id,
                "context": context,
                "text": text,
                "now": now,
            }
        )
        return self.draft


def test_acknowledge_then_handle_acks_before_work() -> None:
    calls: list[str] = []

    def ack() -> None:
        calls.append("ack")

    def handle() -> str:
        calls.append("handle")
        return "done"

    assert acknowledge_then_handle(ack, handle) == "done"
    assert calls == ["ack", "handle"]


def test_bare_app_mention_detection_only_matches_empty_invite_mentions() -> None:
    assert is_bare_app_mention(app_mention_event(text="<@UBOT>")) is True
    assert is_bare_app_mention(app_mention_event(text="   <@UBOT>   ")) is True
    assert (
        is_bare_app_mention(app_mention_event(text="<@UBOT> summarize this channel"))
        is False
    )
    assert (
        is_bare_app_mention(
            app_mention_event(
                text="<@UBOT>",
                files=[{"id": "F123", "name": "report.pdf"}],
            )
        )
        is False
    )


@pytest.fixture(scope="session")
def engine() -> Iterator[Engine]:
    if TEST_POSTGRES_URL is None:
        pytest.skip("KORTNY_TEST_POSTGRES_URL is required for Slack ingress tests")

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


@pytest.fixture(autouse=True)
def _existing_default_installation(db_session: Session) -> None:
    """Pre-seed the default T123 workspace so the HIG-209 first-run intro DM
    (which fires only for genuinely new installations) does not perturb the
    pre-existing ingress tests. Tests that exercise the intro DM use a fresh
    team id so they still observe a new-installation creation."""

    if (
        db_session.scalar(
            select(Installation).where(Installation.slack_team_id == "T123")
        )
        is None
    ):
        db_session.add(Installation(slack_team_id="T123"))
        db_session.commit()


def test_app_mention_creates_task_and_adds_reaction_ack(
    db_session: Session,
) -> None:
    client = FakeSlackClient()
    acknowledgements = FakeAcknowledgementGenerator(
        "I'll pull together the pandas report and post it here."
    )
    reactions = FakeReactionProvider(name="page_facing_up", intent="document")
    result = SlackIngress(
        session=db_session,
        client=client,
        acknowledgement_generator=acknowledgements,
        reaction_provider=reactions,
    ).handle_app_mention(
        body=app_mention_body(event_id="EvMention1"),
        event=app_mention_event(text="<@UBOT> research pandas and make a PDF"),
    )
    db_session.commit()

    task = db_session.scalar(select(Task).where(Task.id == result.task.id))
    installation = db_session.scalar(select(Installation))
    message_event = db_session.scalar(
        select(TaskEvent).where(
            TaskEvent.task_id == result.task.id,
            TaskEvent.type == TaskEventType.message_posted,
        )
    )
    events = task_events(db_session, result.task)

    assert result.created is True
    assert result.thread_ts == "1716400000.000001"
    assert result.acknowledgement_ts is None
    assert installation is not None
    assert installation.slack_team_id == "T123"
    assert task is not None
    assert task.slack_event_id == "EvMention1"
    assert task.slack_channel_id == "C123"
    assert task.slack_thread_ts == "1716400000.000001"
    assert task.slack_message_ts == "1716400000.000001"
    assert task.slack_user_id == "U123"
    assert task.input == "research pandas and make a PDF"
    assert client.calls == []
    assert client.reactions == [
        {
            "channel": "C123",
            "name": "page_facing_up",
            "timestamp": "1716400000.000001",
        }
    ]
    assert acknowledgements.calls == []
    assert reactions.calls == [
        {"input_text": "research pandas and make a PDF", "source": "app_mention"}
    ]
    assert message_event is None
    assert any(
        event.payload.get("message") == ACK_REACTION_ADDED_MESSAGE
        and event.payload.get("reaction") == "page_facing_up"
        and event.payload.get("reaction_intent") == "document"
        for event in events
    )

    inbound = db_session.scalar(
        select(SlackInboundEvent).where(
            SlackInboundEvent.slack_event_id == "EvMention1"
        )
    )
    assert inbound is not None
    assert inbound.installation_id == result.task.installation_id
    assert inbound.event_type == "app_mention"
    assert inbound.surface == "app_mention"
    assert inbound.channel_id == "C123"
    assert inbound.user_id == "U123"
    assert inbound.message_ts == "1716400000.000001"
    assert inbound.processing_status == "task_created"
    assert inbound.task_id == result.task.id
    assert inbound.raw_event["text"] == "<@UBOT> research pandas and make a PDF"
    assert inbound.metadata_json["source"] == "app_mention"
    assert inbound.processed_at is not None


def test_app_mention_strips_bot_mention_outside_leading_position(
    db_session: Session,
) -> None:
    client = FakeSlackClient(auth_test={"ok": True, "user_id": "UBOT"})

    result = SlackIngress(session=db_session, client=client).handle_app_mention(
        body=app_mention_body(event_id="EvMentionMidSentence"),
        event=app_mention_event(text="Yo <@UBOT> are you up?"),
    )
    db_session.commit()

    task = db_session.get(Task, result.task.id)
    installation = db_session.scalar(select(Installation))

    assert task is not None
    assert task.input == "Yo are you up?"
    assert installation is not None
    assert installation.bot_user_id == "UBOT"


def test_app_mention_refreshes_slack_identity_cache(
    db_session: Session,
) -> None:
    client = FakeSlackClient(
        user_info={
            "ok": True,
            "user": {
                "id": "U123",
                "name": "aneesh",
                "real_name": "Aneesh Melkot",
                "profile": {"display_name": "Aneesh"},
            },
        },
        channel_info={
            "ok": True,
            "channel": {
                "id": "C123",
                "name": "general",
                "is_private": False,
            },
        },
    )

    result = SlackIngress(session=db_session, client=client).handle_app_mention(
        body=app_mention_body(event_id="EvMentionIdentity"),
        event=app_mention_event(text="<@UBOT> summarize this channel"),
    )
    db_session.commit()

    user_identity = db_session.scalar(
        select(SlackIdentity).where(
            SlackIdentity.installation_id == result.task.installation_id,
            SlackIdentity.kind == "user",
            SlackIdentity.slack_id == "U123",
        )
    )
    channel_identity = db_session.scalar(
        select(SlackIdentity).where(
            SlackIdentity.installation_id == result.task.installation_id,
            SlackIdentity.kind == "channel",
            SlackIdentity.slack_id == "C123",
        )
    )
    events = task_events(db_session, result.task)

    assert user_identity is not None
    assert user_identity.display_name == "Aneesh Melkot"
    assert channel_identity is not None
    assert channel_identity.display_name == "#general"
    assert client.identity_calls == [
        {"method": "auth_test", "id": "self"},
        {"method": "users_info", "id": "U123"},
        {"method": "conversations_info", "id": "C123"},
    ]
    assert any(
        event.payload.get("message") == "slack_identity_cache_checked"
        and event.payload.get("user_cached") is True
        and event.payload.get("channel_cached") is True
        and event.payload.get("user_refreshed") is True
        and event.payload.get("channel_refreshed") is True
        for event in events
    )


def test_app_mention_uses_existing_thread_ts(db_session: Session) -> None:
    client = FakeSlackClient()
    result = SlackIngress(session=db_session, client=client).handle_app_mention(
        body=app_mention_body(event_id="EvMentionThread"),
        event=app_mention_event(thread_ts="1716300000.000999"),
    )

    assert result.thread_ts == "1716300000.000999"
    assert result.task.slack_thread_ts == "1716300000.000999"
    assert client.calls == []


def test_thread_follow_up_creates_task_without_visible_ack(
    db_session: Session,
) -> None:
    client = FakeSlackClient()
    acknowledgements = FakeAcknowledgementGenerator()
    reactions = FakeReactionProvider(name="mag", intent="research")

    result = SlackIngress(
        session=db_session,
        client=client,
        acknowledgement_generator=acknowledgements,
        reaction_provider=reactions,
    ).handle_app_mention(
        body=app_mention_body(event_id="EvMentionFollowUp"),
        event=app_mention_event(
            text="<@UBOT> what was your source for this?",
            ts="1716400100.000001",
            thread_ts="1716400000.000001",
        ),
    )
    db_session.commit()

    message_event_count = db_session.scalar(
        select(func.count())
        .select_from(TaskEvent)
        .where(TaskEvent.type == TaskEventType.message_posted)
    )

    assert result.created is True
    assert result.acknowledgement_ts is None
    assert result.task.input == "what was your source for this?"
    assert result.task.slack_thread_ts == "1716400000.000001"
    assert client.calls == []
    assert client.reactions == [
        {
            "channel": "C123",
            "name": "mag",
            "timestamp": "1716400100.000001",
        }
    ]
    assert acknowledgements.calls == []
    assert message_event_count == 0


def test_app_mention_records_intent_decision_when_classifier_configured(
    db_session: Session,
) -> None:
    client = FakeSlackClient()
    decision = intent_decision(
        classification=IntentClassification.memory_candidate,
        suggested_reaction="memo",
    )
    classifier = FakeIntentClassifier(decision)
    reactions = FakeReactionProvider(name="memo", intent="memory")

    result = SlackIngress(
        session=db_session,
        client=client,
        intent_classifier=classifier,
        reaction_provider=reactions,
    ).handle_app_mention(
        body=app_mention_body(event_id="EvIntentClassified"),
        event=app_mention_event(text="<@UBOT> remember that briefs stay concise"),
    )
    db_session.commit()

    events = task_events(db_session, result.task)

    assert classifier.calls[0][0] == result.task.id
    assert classifier.calls[0][1].surface is IntentSurface.app_mention
    assert classifier.calls[0][1].text == "remember that briefs stay concise"
    assert reactions.intent_decisions == [decision]
    assert any(
        event.payload.get("message") == INTENT_CLASSIFIED_MESSAGE
        and event.payload.get("decision", {}).get("classification")
        == "memory_candidate"
        for event in events
    )


def test_ack_reaction_failure_does_not_block_task_creation(
    db_session: Session,
) -> None:
    client = FakeSlackClient(reaction_error=RuntimeError("reaction denied"))
    reactions = FakeReactionProvider(name="memo", intent="memory")

    result = SlackIngress(
        session=db_session,
        client=client,
        reaction_provider=reactions,
    ).handle_app_mention(
        body=app_mention_body(event_id="EvReactionFailOpen"),
        event=app_mention_event(
            text="<@UBOT> remember that weekly recaps go on Fridays"
        ),
    )
    db_session.commit()

    events = task_events(db_session, result.task)

    assert result.created is True
    assert result.task.input == "remember that weekly recaps go on Fridays"
    assert client.reactions == []
    assert client.calls == []
    assert result.acknowledgement_ts is None
    assert any(
        event.payload.get("message") == "slack_ack_reaction_failed"
        and event.payload.get("reaction") == "memo"
        and event.payload.get("reaction_intent") == "memory"
        for event in events
    )


def test_app_mention_includes_attached_slack_file_ids_in_task_input(
    db_session: Session,
) -> None:
    client = FakeSlackClient()

    result = SlackIngress(session=db_session, client=client).handle_app_mention(
        body=app_mention_body(event_id="EvMentionFile"),
        event=app_mention_event(
            text="<@UBOT> summarize this file",
            files=[
                {
                    "id": "F123",
                    "name": "report.pdf",
                    "mimetype": "application/pdf",
                    "size": 2048,
                }
            ],
        ),
    )

    assert result.task.input == (
        "summarize this file\n\n"
        "<slack_files>\n"
        "- id: F123\n"
        "  name: report.pdf\n"
        "  mimetype: application/pdf\n"
        "  size_bytes: 2048\n"
        "</slack_files>"
    )


def test_app_mention_skips_visible_ack_generation_by_default(
    db_session: Session,
) -> None:
    client = FakeSlackClient()
    acknowledgements = FakeAcknowledgementGenerator(error=RuntimeError("ack failed"))
    result = SlackIngress(
        session=db_session,
        client=client,
        acknowledgement_generator=acknowledgements,
        reaction_provider=FakeReactionProvider(name="eyes", intent="working"),
    ).handle_app_mention(
        body=app_mention_body(event_id="EvMentionAckFailure"),
        event=app_mention_event(text="<@UBOT> research ack failure"),
    )
    db_session.commit()

    events = list(
        db_session.scalars(
            select(TaskEvent)
            .where(TaskEvent.task_id == result.task.id)
            .order_by(TaskEvent.seq)
        )
    )

    assert result.created is True
    assert result.acknowledgement_ts is None
    assert acknowledgements.calls == []
    assert client.calls == []
    assert client.reactions == [
        {
            "channel": "C123",
            "name": "eyes",
            "timestamp": "1716400000.000001",
        }
    ]
    assert not any(
        event.payload.get("message") == "acknowledgement_generation_failed"
        for event in events
    )


def test_redelivered_app_mention_is_idempotent(db_session: Session) -> None:
    client = FakeSlackClient()
    ingress = SlackIngress(session=db_session, client=client)
    body = app_mention_body(event_id="EvMentionDuplicate")
    redelivery_body = {
        **body,
        "headers": {
            "X-Slack-Retry-Num": "1",
            "X-Slack-Retry-Reason": "http_timeout",
        },
    }
    event = app_mention_event(text="<@UBOT> search duplicate delivery")

    first = ingress.handle_app_mention(body=body, event=event)
    second = ingress.handle_app_mention(body=redelivery_body, event=event)
    db_session.commit()

    task_count = db_session.scalar(select(func.count()).select_from(Task))
    message_event_count = db_session.scalar(
        select(func.count())
        .select_from(TaskEvent)
        .where(TaskEvent.type == TaskEventType.message_posted)
    )
    inbound = db_session.scalar(
        select(SlackInboundEvent).where(
            SlackInboundEvent.slack_event_id == "EvMentionDuplicate"
        )
    )

    assert first.created is True
    assert second.created is False
    assert second.task.id == first.task.id
    assert task_count == 1
    assert message_event_count == 0
    assert client.calls == []
    assert len(client.reactions) == 1
    assert inbound is not None
    assert inbound.processing_status == "task_created"
    assert inbound.task_id == first.task.id
    assert inbound.retry_num == 1
    assert inbound.retry_reason == "http_timeout"
    assert inbound.metadata_json["delivery_count"] == 2


def test_app_mention_dedupes_by_slack_message_timestamp(db_session: Session) -> None:
    client = FakeSlackClient()
    ingress = SlackIngress(session=db_session, client=client)

    first = ingress.handle_app_mention(
        body=app_mention_body(event_id="EvMessageFirst"),
        event=app_mention_event(text="<@UBOT> search duplicate event shapes"),
    )
    second = ingress.handle_app_mention(
        body=app_mention_body(event_id="EvAppMentionSecond"),
        event=app_mention_event(text="<@UBOT> search duplicate event shapes"),
    )
    db_session.commit()

    task_count = db_session.scalar(select(func.count()).select_from(Task))

    assert first.created is True
    assert second.created is False
    assert second.task.id == first.task.id
    assert task_count == 1
    assert client.calls == []
    assert len(client.reactions) == 1


def test_dm_creates_task_without_visible_ack(db_session: Session) -> None:
    client = FakeSlackClient()
    acknowledgements = FakeAcknowledgementGenerator(
        "I'll take a look and send the answer here."
    )
    reactions = FakeReactionProvider(name="mag", intent="research")

    result = SlackIngress(
        session=db_session,
        client=client,
        acknowledgement_generator=acknowledgements,
        reaction_provider=reactions,
    ).handle_dm(
        body=message_body(event_id="EvDm1"),
        event=dm_event(text="<@UBOT> research private context"),
    )
    db_session.commit()

    assert result is not None
    task = db_session.scalar(select(Task).where(Task.id == result.task.id))
    installation = db_session.scalar(select(Installation))
    message_event = db_session.scalar(
        select(TaskEvent).where(
            TaskEvent.task_id == result.task.id,
            TaskEvent.type == TaskEventType.message_posted,
        )
    )

    assert result.created is True
    assert result.thread_ts == "D123"
    assert result.acknowledgement_ts is None
    assert installation is not None
    assert installation.slack_team_id == "T123"
    assert task is not None
    assert task.slack_event_id == "EvDm1"
    assert task.slack_channel_id == "D123"
    assert task.slack_thread_ts == "D123"
    assert task.slack_message_ts == "1716500000.000001"
    assert task.slack_user_id == "U123"
    assert task.input == "<@UBOT> research private context"
    assert client.calls == []
    assert client.reactions == [
        {
            "channel": "D123",
            "name": "mag",
            "timestamp": "1716500000.000001",
        }
    ]
    assert acknowledgements.calls == []
    assert message_event is None


def test_dm_scheduling_request_creates_active_schedule(
    db_session: Session,
) -> None:
    client = FakeSlackClient()
    result = SlackIngress(
        session=db_session,
        client=client,
        intent_classifier=FakeIntentClassifier(intent_decision()),
        reaction_provider=FakeReactionProvider(name="calendar", intent="scheduled"),
    ).handle_dm(
        body=message_body(event_id="EvScheduleDm"),
        event=dm_event(
            text=(
                "Every morning can you check on PYPL ticker and give me a "
                "market summary"
            ),
            ts="1716500100.000001",
        ),
    )
    db_session.commit()

    assert result is not None
    task = db_session.get(Task, result.task.id)
    schedule = db_session.scalar(select(Schedule))
    assert task is not None
    assert schedule is not None

    assert task.status is TaskStatus.succeeded
    assert schedule.status == "active"
    assert schedule.owner_type == "user"
    assert schedule.owner_slack_user_id == "U123"
    assert schedule.spec_kind == "cron"
    assert schedule.cron_expr == "0 9 * * *"
    assert schedule.delivery_kind == "slack_dm"
    assert schedule.delivery_slack_user_id == "U123"
    assert schedule.delivery_slack_channel_id == "D123"
    assert schedule.delivery_slack_thread_ts == "D123"
    assert schedule.artifact_delivery_policy == "message_only"
    assert schedule.task_template["delivery_surface"] == "dm"
    assert schedule.task_template["slack_channel_id"] == "D123"
    assert schedule.task_template["slack_thread_ts"] == "D123"
    assert schedule.task_template["input"] == (
        "can you check on PYPL ticker and give me a market summary"
    )
    assert schedule.metadata_json["source_task_id"] == str(task.id)
    assert schedule.metadata_json["confirmation_required"] is False

    assert len(client.calls) == 1
    assert client.calls[0]["channel"] == "D123"
    assert client.calls[0]["thread_ts"] is None
    assert "Done" in client.calls[0]["text"]
    assert "PYPL" in client.calls[0]["text"]
    assert "every morning" in client.calls[0]["text"].casefold()
    assert "this DM" in client.calls[0]["text"]
    assert "pause, change, or cancel" in client.calls[0]["text"]
    assert "Schedule id" not in client.calls[0]["text"]
    assert "Budget cap" not in client.calls[0]["text"]
    assert "proposed schedule" not in client.calls[0]["text"]
    blocks = client.calls[0]["blocks"]
    assert blocks[0]["type"] == "actions"
    assert [item["text"]["text"] for item in blocks[0]["elements"]] == [
        "Pause",
        "Change",
        "Cancel",
    ]

    events = task_events(db_session, task)
    assert any(
        event.payload.get("message") == "schedule_created"
        and event.payload.get("schedule_id") == str(schedule.id)
        and event.payload.get("schedule_status") == "active"
        and event.payload.get("delivery_kind") == "slack_dm"
        for event in events
    )
    message_event = next(
        event
        for event in events
        if event.type == TaskEventType.message_posted
        and event.payload.get("purpose") == "schedule_created"
    )
    assert message_event.payload["text"] == client.calls[0]["text"]


def test_dm_scheduling_request_uses_llm_fallback_when_rules_do_not_parse(
    db_session: Session,
) -> None:
    client = FakeSlackClient()
    fallback = FakeScheduleFallbackParser(
        ScheduleDraft(
            title="Send a stock market update",
            spec_kind="cron",
            cron_expr="0 8 * * 1-5",
            timezone="America/Chicago",
            next_run_at=datetime(2026, 6, 5, 13, 0, tzinfo=UTC),
            cadence_label="Every weekday at 8:00 AM Central time",
            task_input="send a stock market update",
            parse_strategy="llm_schedule_parser",
        )
    )

    result = SlackIngress(
        session=db_session,
        client=client,
        intent_classifier=FakeIntentClassifier(intent_decision()),
        reaction_provider=FakeReactionProvider(name="calendar", intent="scheduled"),
        schedule_fallback_parser=fallback,
    ).handle_dm(
        body=message_body(event_id="EvScheduleDmFallback"),
        event=dm_event(
            text="Every weekday at 8AM central time I want a stock market update",
            ts="1716500110.000001",
        ),
    )
    db_session.commit()

    assert result is not None
    task = db_session.get(Task, result.task.id)
    schedule = db_session.scalar(select(Schedule))
    assert task is not None
    assert schedule is not None
    assert len(fallback.calls) == 1
    assert schedule.status == "active"
    assert schedule.spec_kind == "cron"
    assert schedule.cron_expr == "0 8 * * 1-5"
    assert schedule.timezone == "America/Chicago"
    assert schedule.metadata_json["parse_strategy"] == "llm_schedule_parser"
    assert schedule.task_template["input"] == "send a stock market update"
    assert "every weekday at 8:00 AM Central time" in client.calls[0]["text"]


def test_schedule_pause_button_pauses_active_schedule(
    db_session: Session,
) -> None:
    client = FakeSlackClient()
    ingress = SlackIngress(
        session=db_session,
        client=client,
        intent_classifier=FakeIntentClassifier(intent_decision()),
        reaction_provider=FakeReactionProvider(name="calendar", intent="scheduled"),
    )
    created = ingress.handle_dm(
        body=message_body(event_id="EvScheduleButtonCreate"),
        event=dm_event(
            text="Every morning send me a market update",
            ts="1716500200.000001",
        ),
    )
    db_session.commit()
    schedule = db_session.scalar(select(Schedule))
    assert created is not None
    assert schedule is not None
    assert schedule.status == "active"

    result = ingress.handle_schedule_action(
        body=schedule_action_body(
            channel_id="D123",
            user_id="U123",
            message_ts="1716500200.000001",
        ),
        action={
            "action_id": "kortny_schedule_pause",
            "value": str(schedule.id),
        },
    )
    db_session.commit()

    db_session.refresh(schedule)
    assert result.handled is True
    assert schedule.status == "paused"
    assert len(client.calls) == 2
    assert "Paused that scheduled task" in client.calls[1]["text"]
    task = result.task
    assert task is not None
    events = task_events(db_session, task)
    assert any(
        event.payload.get("message") == "schedule_paused"
        and event.payload.get("schedule_id") == str(schedule.id)
        for event in events
    )


def test_channel_scheduling_request_defaults_to_thread_delivery(
    db_session: Session,
) -> None:
    client = FakeSlackClient()
    result = SlackIngress(
        session=db_session,
        client=client,
        intent_classifier=FakeIntentClassifier(intent_decision()),
        reaction_provider=FakeReactionProvider(name="calendar", intent="scheduled"),
    ).handle_app_mention(
        body=app_mention_body(event_id="EvScheduleChannel"),
        event=app_mention_event(
            text="<@UBOT> Every morning check the market and summarize it",
            ts="1716400200.000001",
        ),
    )
    db_session.commit()

    assert result is not None
    schedule = db_session.scalar(select(Schedule))
    assert schedule is not None
    assert schedule.status == "active"
    assert schedule.delivery_kind == "slack_thread"
    assert schedule.delivery_slack_user_id == "U123"
    assert schedule.delivery_slack_channel_id == "C123"
    assert schedule.delivery_slack_thread_ts == "1716400200.000001"
    assert schedule.task_template["delivery_surface"] == "thread"

    assert len(client.calls) == 1
    assert client.calls[0]["channel"] == "C123"
    assert client.calls[0]["thread_ts"] == "1716400200.000001"
    assert "this thread" in client.calls[0]["text"]


def test_channel_scheduling_request_can_target_channel_root(
    db_session: Session,
) -> None:
    client = FakeSlackClient()
    result = SlackIngress(
        session=db_session,
        client=client,
        intent_classifier=FakeIntentClassifier(intent_decision()),
        reaction_provider=FakeReactionProvider(name="calendar", intent="scheduled"),
    ).handle_app_mention(
        body=app_mention_body(event_id="EvScheduleChannelRoot"),
        event=app_mention_event(
            text="<@UBOT> Every morning post a market update in this channel",
            ts="1716400300.000001",
        ),
    )
    db_session.commit()

    assert result is not None
    schedule = db_session.scalar(select(Schedule))
    assert schedule is not None
    assert schedule.status == "active"
    assert schedule.delivery_kind == "slack_channel"
    assert schedule.delivery_slack_channel_id == "C123"
    assert schedule.delivery_slack_thread_ts is None
    assert schedule.task_template["delivery_surface"] == "channel"

    assert len(client.calls) == 1
    assert client.calls[0]["channel"] == "C123"
    assert client.calls[0]["thread_ts"] == "1716400300.000001"
    assert "this channel" in client.calls[0]["text"]


def test_dm_confirmation_activates_latest_proposed_schedule(
    db_session: Session,
) -> None:
    client = FakeSlackClient()
    ingress = SlackIngress(
        session=db_session,
        client=client,
        intent_classifier=FakeIntentClassifier(intent_decision()),
        reaction_provider=FakeReactionProvider(name="calendar", intent="scheduled"),
    )
    proposed = ingress.handle_dm(
        body=message_body(event_id="EvSchedulePropose"),
        event=dm_event(
            text=(
                "Draft a schedule for every Monday morning to check unresolved "
                "decisions, but wait for me to confirm before running it."
            ),
            ts="1716500100.000001",
        ),
    )
    confirmed = ingress.handle_dm(
        body=message_body(event_id="EvScheduleConfirm"),
        event=dm_event(
            text="yes set it up",
            ts="1716500110.000001",
        ),
    )
    db_session.commit()

    assert proposed is not None
    assert confirmed is not None
    schedule = db_session.scalar(select(Schedule))
    assert schedule is not None
    assert schedule.status == "active"
    assert schedule.metadata_json["activated_by"] == "U123"
    assert schedule.metadata_json["activated_from_task_id"] == str(confirmed.task.id)

    confirmed_task = db_session.get(Task, confirmed.task.id)
    assert confirmed_task is not None
    assert confirmed_task.status is TaskStatus.succeeded
    assert len(client.calls) == 2
    assert "activated that scheduled task" in client.calls[1]["text"]

    events = task_events(db_session, confirmed_task)
    assert any(
        event.payload.get("message") == "schedule_activated"
        and event.payload.get("schedule_id") == str(schedule.id)
        for event in events
    )
    assert any(
        event.type == TaskEventType.message_posted
        and event.payload.get("purpose") == "schedule_activate"
        for event in events
    )


def test_dm_from_bot_is_ignored(db_session: Session) -> None:
    client = FakeSlackClient()

    result = SlackIngress(session=db_session, client=client).handle_dm(
        body=message_body(event_id="EvDmBot"),
        event=dm_event(bot_id="B123", text="bot reply"),
    )
    db_session.commit()

    task_count = db_session.scalar(select(func.count()).select_from(Task))
    inbound = db_session.scalar(
        select(SlackInboundEvent).where(SlackInboundEvent.slack_event_id == "EvDmBot")
    )

    assert result is None
    assert task_count == 0
    assert client.calls == []
    assert inbound is not None
    assert inbound.processing_status == "ignored"
    assert inbound.metadata_json["reason"] == "bot_id"
    assert inbound.task_id is None


def test_dm_edit_event_is_ignored(db_session: Session) -> None:
    client = FakeSlackClient()

    result = SlackIngress(session=db_session, client=client).handle_dm(
        body=message_body(event_id="EvDmEdit"),
        event=dm_event(subtype="message_changed", text="edited"),
    )
    db_session.commit()

    task_count = db_session.scalar(select(func.count()).select_from(Task))

    assert result is None
    assert task_count == 0
    assert client.calls == []


def test_non_dm_message_event_is_ignored_by_dm_ingress(db_session: Session) -> None:
    client = FakeSlackClient()

    result = SlackIngress(session=db_session, client=client).handle_dm(
        body=message_body(event_id="EvChannelMessage"),
        event=dm_event(channel="C123", channel_type="channel"),
    )
    db_session.commit()

    task_count = db_session.scalar(select(func.count()).select_from(Task))

    assert result is None
    assert task_count == 0
    assert client.calls == []


def test_channel_message_observation_records_policy_gated_event(
    db_session: Session,
) -> None:
    result = SlackIngress(
        session=db_session,
        client=FakeSlackClient(),
    ).observe_channel_message(
        body=message_body(event_id="EvObserveMessage"),
        event=channel_event(
            text="Aneesh is discussing a weekly reporting workflow with <@U999>."
        ),
    )
    db_session.commit()

    policy = db_session.scalar(select(ObservePolicy))
    observation = db_session.scalar(select(ObservationEvent))
    membership = db_session.scalar(select(SlackChannelMembership))
    task_count = db_session.scalar(select(func.count()).select_from(Task))
    inbound = db_session.scalar(
        select(SlackInboundEvent).where(
            SlackInboundEvent.slack_event_id == "EvObserveMessage"
        )
    )

    assert result.observed is True
    assert result.reason == "observed"
    assert policy is not None
    assert policy.scope_type == "channel"
    assert policy.scope_id == "C123"
    assert policy.observation_status == "active"
    assert policy.proactivity_status == "digest_only"
    assert observation is not None
    assert observation.event_type == "message"
    assert observation.slack_event_id == "EvObserveMessage"
    assert observation.channel_id == "C123"
    assert observation.user_id == "U123"
    assert observation.text_preview == (
        "Aneesh is discussing a weekly reporting workflow with <@user>."
    )
    assert observation.raw_payload_checksum
    assert membership is not None
    assert membership.channel_id == "C123"
    assert membership.membership_status == "active"
    assert membership.discovered_via == "message_observation"
    assert membership.onboarding_status == "pending"
    assert membership.last_event_id == "EvObserveMessage"
    assert task_count == 0
    assert inbound is not None
    assert inbound.processing_status == "observed"
    assert inbound.observation_event_id == observation.id
    assert inbound.task_id is None
    assert inbound.metadata_json["reason"] == "observed"


def test_channel_observation_skips_dm_events(db_session: Session) -> None:
    result = SlackIngress(
        session=db_session,
        client=FakeSlackClient(),
    ).observe_channel_message(
        body=message_body(event_id="EvObserveDm"),
        event=dm_event(text="private DM with Kortny"),
    )
    db_session.commit()

    observation_count = db_session.scalar(
        select(func.count()).select_from(ObservationEvent)
    )
    policy_count = db_session.scalar(select(func.count()).select_from(ObservePolicy))

    assert result.observed is False
    assert result.reason == "dm_excluded"
    assert observation_count == 0
    assert policy_count == 0


def test_channel_observation_skips_bot_authored_messages(db_session: Session) -> None:
    result = SlackIngress(
        session=db_session,
        client=FakeSlackClient(),
    ).observe_channel_message(
        body=message_body(event_id="EvObserveBot"),
        event=channel_event(text="bot output", bot_id="B123"),
    )
    db_session.commit()

    observation_count = db_session.scalar(
        select(func.count()).select_from(ObservationEvent)
    )

    assert result.observed is False
    assert result.reason == "bot_message"
    assert observation_count == 0


def test_channel_observation_respects_paused_policy(db_session: Session) -> None:
    installation = create_installation(db_session, slack_team_id="T123")
    db_session.add(
        ObservePolicy(
            installation_id=installation.id,
            scope_type="channel",
            scope_id="C123",
            observation_status="off",
            proactivity_status="off",
            retention_days=30,
            metadata_json={"test": True},
        )
    )
    db_session.commit()

    result = SlackIngress(
        session=db_session,
        client=FakeSlackClient(),
    ).observe_channel_message(
        body=message_body(event_id="EvObservePaused"),
        event=channel_event(text="should not be observed"),
    )
    db_session.commit()

    observation_count = db_session.scalar(
        select(func.count()).select_from(ObservationEvent)
    )

    assert result.observed is False
    assert result.reason == "policy_disabled"
    assert observation_count == 0


def test_member_joined_channel_records_onboarding_and_posts_intro(
    db_session: Session,
) -> None:
    client = FakeSlackClient(auth_test={"ok": True, "user_id": "UBOT"})
    ingress = SlackIngress(session=db_session, client=client)

    result = ingress.handle_member_joined_channel(
        body=member_joined_body(event_id="EvObserveJoin"),
        event=member_joined_event(user="UBOT", channel="C123", inviter="U123"),
    )
    duplicate = ingress.handle_member_joined_channel(
        body=member_joined_body(event_id="EvObserveJoin"),
        event=member_joined_event(user="UBOT", channel="C123", inviter="U123"),
    )
    db_session.commit()

    installation = db_session.scalar(select(Installation))
    policy = db_session.scalar(select(ObservePolicy))
    membership = db_session.scalar(select(SlackChannelMembership))
    assessment_task = db_session.scalar(
        select(Task)
        .join(TaskEvent, TaskEvent.task_id == Task.id)
        .where(
            TaskEvent.payload["message"].as_string()
            == CHANNEL_ASSESSMENT_REQUESTED_MESSAGE
        )
    )
    observations = list(
        db_session.scalars(
            select(ObservationEvent).order_by(ObservationEvent.event_type)
        )
    )
    inbound = db_session.scalar(
        select(SlackInboundEvent).where(
            SlackInboundEvent.slack_event_id == "EvObserveJoin"
        )
    )

    assert result.observed is True
    assert result.intro_text is not None
    assert duplicate.observed is False
    assert duplicate.reason == "duplicate"
    assert installation is not None
    assert installation.bot_user_id == "UBOT"
    assert policy is not None
    assert policy.scope_type == "channel"
    assert policy.scope_id == "C123"
    assert policy.metadata_json.get("onboarding_intro_posted_at")
    assert membership is not None
    assert membership.channel_id == "C123"
    assert membership.membership_status == "active"
    assert membership.discovered_via == "member_joined_channel"
    assert membership.added_by_user_id == "U123"
    assert membership.onboarding_status == "posted"
    assert membership.onboarding_message_ts == "1716400000.000001"
    assert membership.last_event_id == "EvObserveJoin"
    assert membership.metadata_json["assessment_status"] == "queued"
    assert assessment_task is not None
    assert membership.metadata_json["assessment_task_id"] == str(assessment_task.id)
    assert assessment_task.slack_thread_ts == "1716400000.000001"
    assert assessment_task.slack_message_ts == "1716400000.000001"
    assert assessment_task.slack_user_id == "U123"
    assert "channel onboarding assessment" in assessment_task.input
    assert [event.event_type for event in observations] == [
        "channel_join",
        "channel_onboarding_intro",
    ]
    assert client.calls == [
        {
            "channel": "C123",
            "text": result.intro_text,
            "thread_ts": None,
        }
    ]
    assert inbound is not None
    assert inbound.processing_status == "task_created"
    assert inbound.task_id == assessment_task.id
    assert inbound.metadata_json["task_kind"] == "channel_assessment"
    assert (
        db_session.scalar(
            select(func.count())
            .select_from(TaskEvent)
            .where(
                TaskEvent.payload["message"].as_string()
                == CHANNEL_ASSESSMENT_REQUESTED_MESSAGE
            )
        )
        == 1
    )


def test_member_joined_channel_resolves_bot_from_authorization_when_auth_test_lacks_user_id(
    db_session: Session,
) -> None:
    client = FakeSlackClient(auth_test={"ok": True})

    result = SlackIngress(
        session=db_session,
        client=client,
    ).handle_member_joined_channel(
        body=member_joined_body(
            event_id="EvObserveJoinAuthFallback",
            authorizations=[{"user_id": "UBOT_FROM_EVENT", "is_bot": True}],
        ),
        event=member_joined_event(user="UBOT_FROM_EVENT", channel="C123"),
    )
    db_session.commit()

    installation = db_session.scalar(select(Installation))
    policy = db_session.scalar(select(ObservePolicy))

    assert result.observed is True
    assert result.intro_text is not None
    assert installation is not None
    assert installation.bot_user_id == "UBOT_FROM_EVENT"
    assert policy is not None
    assert policy.scope_id == "C123"
    assert client.calls == [
        {
            "channel": "C123",
            "text": result.intro_text,
            "thread_ts": None,
        }
    ]


def test_member_joined_channel_resolves_bot_from_users_info_when_auth_test_lacks_user_id(
    db_session: Session,
) -> None:
    client = FakeSlackClient(
        auth_test={"ok": True},
        user_info={
            "ok": True,
            "user": {
                "id": "UBOT_FROM_USERS_INFO",
                "name": "kortny",
                "is_bot": True,
            },
        },
    )

    result = SlackIngress(
        session=db_session,
        client=client,
    ).handle_member_joined_channel(
        body=member_joined_body(event_id="EvObserveJoinUsersInfoFallback"),
        event=member_joined_event(user="UBOT_FROM_USERS_INFO", channel="C123"),
    )
    db_session.commit()

    installation = db_session.scalar(select(Installation))
    membership = db_session.scalar(select(SlackChannelMembership))

    assert result.observed is True
    assert result.intro_text is not None
    assert installation is not None
    assert installation.bot_user_id == "UBOT_FROM_USERS_INFO"
    assert membership is not None
    assert membership.discovered_via == "member_joined_channel"
    assert membership.onboarding_status == "posted"
    assert client.identity_calls == [
        {"method": "auth_test", "id": "self"},
        {"method": "users_info", "id": "UBOT_FROM_USERS_INFO"},
    ]


def test_member_joined_channel_skips_unverified_joined_user_when_auth_test_lacks_user_id(
    db_session: Session,
) -> None:
    client = FakeSlackClient(auth_test={"ok": True})

    result = SlackIngress(
        session=db_session,
        client=client,
    ).handle_member_joined_channel(
        body=member_joined_body(event_id="EvObserveJoinUnverified"),
        event=member_joined_event(user="UBOT_FROM_EVENT", channel="C123"),
    )
    db_session.commit()

    installation = db_session.scalar(select(Installation))
    policy_count = db_session.scalar(select(func.count()).select_from(ObservePolicy))
    membership_count = db_session.scalar(
        select(func.count()).select_from(SlackChannelMembership)
    )

    assert result.observed is False
    assert result.reason == "bot_user_unresolved"
    assert installation is not None
    assert installation.bot_user_id is None
    assert policy_count == 0
    assert membership_count == 0
    assert client.calls == []


def test_app_mention_can_trigger_channel_onboarding_when_join_event_missing(
    db_session: Session,
) -> None:
    client = FakeSlackClient(auth_test={"ok": True, "user_id": "UBOT"})
    ingress = SlackIngress(session=db_session, client=client)
    body = app_mention_body(event_id="EvImplicitJoin")
    event = app_mention_event(text="<@UBOT> hi")

    result = ingress.ensure_channel_onboarding_from_mention(body=body, event=event)
    duplicate = ingress.ensure_channel_onboarding_from_mention(body=body, event=event)
    db_session.commit()

    installation = db_session.scalar(select(Installation))
    policy = db_session.scalar(select(ObservePolicy))
    membership = db_session.scalar(select(SlackChannelMembership))
    assessment_task = db_session.scalar(
        select(Task)
        .join(TaskEvent, TaskEvent.task_id == Task.id)
        .where(
            TaskEvent.payload["message"].as_string()
            == CHANNEL_ASSESSMENT_REQUESTED_MESSAGE
        )
    )
    observations = list(
        db_session.scalars(
            select(ObservationEvent).order_by(ObservationEvent.event_type)
        )
    )

    assert result.observed is True
    assert result.intro_text is not None
    assert duplicate.observed is False
    assert duplicate.reason == "intro_already_posted"
    assert installation is not None
    assert installation.bot_user_id == "UBOT"
    assert policy is not None
    assert policy.scope_type == "channel"
    assert policy.scope_id == "C123"
    assert policy.enabled_by_user_id == "U123"
    assert policy.metadata_json.get("onboarding_intro_posted_at")
    assert membership is not None
    assert membership.channel_id == "C123"
    assert membership.membership_status == "active"
    assert membership.discovered_via == "app_mention"
    assert membership.added_by_user_id == "U123"
    assert membership.onboarding_status == "posted"
    assert membership.onboarding_message_ts == "1716400000.000001"
    assert membership.metadata_json["assessment_status"] == "queued"
    assert assessment_task is not None
    assert membership.metadata_json["assessment_task_id"] == str(assessment_task.id)
    assert assessment_task.slack_thread_ts == "1716400000.000001"
    assert [event.event_type for event in observations] == [
        "channel_join",
        "channel_onboarding_intro",
    ]
    assert observations[0].slack_event_id == (
        "EvImplicitJoin:implicit_channel_activation"
    )
    assert observations[0].visibility_metadata["activation_source"] == "app_mention"
    assert client.calls == [
        {
            "channel": "C123",
            "text": result.intro_text,
            "thread_ts": None,
        }
    ]


def test_app_mention_after_onboarding_does_not_keep_onboarding_skip_reason(
    db_session: Session,
) -> None:
    client = FakeSlackClient(auth_test={"ok": True, "user_id": "UBOT"})
    ingress = SlackIngress(session=db_session, client=client)

    initial = ingress.ensure_channel_onboarding_from_mention(
        body=app_mention_body(event_id="EvInitialOnboarding"),
        event=app_mention_event(text="<@UBOT> hi"),
    )
    follow_up_body = app_mention_body(event_id="EvAlreadyOnboardedTask")
    follow_up_event = app_mention_event(
        text="<@UBOT> summarize the unresolved items here",
        ts="1716400200.000001",
    )
    onboarding_preflight = ingress.ensure_channel_onboarding_from_mention(
        body=follow_up_body,
        event=follow_up_event,
    )
    task_result = ingress.handle_app_mention(
        body=follow_up_body,
        event=follow_up_event,
    )
    db_session.commit()

    inbound = db_session.scalar(
        select(SlackInboundEvent).where(
            SlackInboundEvent.slack_event_id == "EvAlreadyOnboardedTask"
        )
    )

    assert initial.observed is True
    assert onboarding_preflight.observed is False
    assert onboarding_preflight.reason == "intro_already_posted"
    assert task_result.created is True
    assert inbound is not None
    assert inbound.processing_status == "task_created"
    assert inbound.task_id == task_result.task.id
    assert inbound.metadata_json["source"] == "app_mention"
    assert "reason" not in inbound.metadata_json


def test_member_joined_channel_ignores_non_bot_joins(db_session: Session) -> None:
    client = FakeSlackClient(auth_test={"ok": True, "user_id": "UBOT"})

    result = SlackIngress(
        session=db_session,
        client=client,
    ).handle_member_joined_channel(
        body=member_joined_body(event_id="EvObserveJoinUser"),
        event=member_joined_event(user="U999", channel="C123", inviter="U123"),
    )
    db_session.commit()

    observation_count = db_session.scalar(
        select(func.count()).select_from(ObservationEvent)
    )
    policy_count = db_session.scalar(select(func.count()).select_from(ObservePolicy))
    membership_count = db_session.scalar(
        select(func.count()).select_from(SlackChannelMembership)
    )

    assert result.observed is False
    assert result.reason == "not_bot_join"
    assert observation_count == 0
    assert policy_count == 0
    assert membership_count == 0


def test_soft_channel_message_creates_task_after_high_confidence_intent(
    db_session: Session,
) -> None:
    client = FakeSlackClient()
    decision = intent_decision(suggested_reaction="thinking_face")
    classifier = FakeIntentClassifier(decision, require_task_id=False)
    reactions = FakeReactionProvider(name="thinking_face", intent="thinking")

    result = SlackIngress(
        session=db_session,
        client=client,
        intent_classifier=classifier,
        reaction_provider=reactions,
    ).handle_channel_message(
        body=message_body(event_id="EvSoftMention1"),
        event=channel_event(
            text=(
                "Kortny can you compare the tradeoffs between reaction ACKs "
                "and verbal ACKs for this app?"
            )
        ),
        app_name="kortny",
    )
    db_session.commit()

    assert result is not None
    assert result.created is True
    assert result.thread_ts == "1716600000.000001"
    assert result.acknowledgement_ts is None
    assert result.task.input == (
        "Kortny can you compare the tradeoffs between reaction ACKs "
        "and verbal ACKs for this app?"
    )
    assert result.task.slack_thread_ts == "1716600000.000001"
    assert result.task.slack_message_ts == "1716600000.000001"
    assert classifier.calls[0][0] is None
    assert classifier.calls[0][1].surface is IntentSurface.channel_message
    assert classifier.calls[0][1].app_name == "kortny"
    assert client.calls == []
    assert client.reactions == [
        {
            "channel": "C123",
            "name": "thinking_face",
            "timestamp": "1716600000.000001",
        }
    ]
    assert reactions.calls == [
        {
            "input_text": result.task.input,
            "source": "channel_message",
        }
    ]
    assert any(
        event.payload.get("message") == INTENT_CLASSIFIED_MESSAGE
        and event.payload.get("source") == "channel_message"
        for event in task_events(db_session, result.task)
    )


def test_soft_channel_message_grounds_classifier_with_connected_integrations(
    db_session: Session,
) -> None:
    """The soft-mention path must classify with connected integrations.

    It classifies before a Task row exists, so it long routed blind to what was
    connected — the one ingress surface missing capability grounding (HIG-269).
    Seed a connection and assert the classifier sees it.
    """

    installation = create_installation(db_session, slack_team_id="T123")
    db_session.add(
        ComposioConnection(
            installation_id=installation.id,
            toolkit_slug="notion",
            auth_config_id="ac_notion",
            connected_account_id="ca_notion",
            connection_request_id="ln_notion",
            composio_user_id="slack:soft:U123",
            owner_slack_user_id="U123",
            visibility_scope_type="workspace",
            visibility_scope_id=None,
            status="active",
        )
    )
    db_session.commit()

    client = FakeSlackClient()
    classifier = FakeIntentClassifier(intent_decision(), require_task_id=False)

    SlackIngress(
        session=db_session,
        client=client,
        intent_classifier=classifier,
    ).handle_channel_message(
        body=message_body(event_id="EvSoftMentionGrounded"),
        event=channel_event(text="Kortny what notes can you see on Notion?"),
        app_name="kortny",
    )
    db_session.commit()

    assert classifier.calls
    request = classifier.calls[0][1]
    assert request.surface is IntentSurface.channel_message
    assert "notion" in request.connected_integrations


def test_soft_channel_message_ignores_third_person_reference(
    db_session: Session,
) -> None:
    client = FakeSlackClient()
    decision = intent_decision(
        classification=IntentClassification.third_person_reference,
    ).model_copy(
        update={
            "addressed_to_kortny": False,
            "should_create_task": False,
            "should_ack_with_reaction": False,
            "confidence": 0.98,
        }
    )
    classifier = FakeIntentClassifier(decision, require_task_id=False)

    result = SlackIngress(
        session=db_session,
        client=client,
        intent_classifier=classifier,
    ).handle_channel_message(
        body=message_body(event_id="EvSoftMentionThirdPerson"),
        event=channel_event(text="I think Kortny might be too eager here."),
        app_name="kortny",
    )
    db_session.commit()

    task_count = db_session.scalar(select(func.count()).select_from(Task))

    assert result is None
    assert task_count == 0
    assert len(classifier.calls) == 1
    assert client.calls == []
    assert client.reactions == []


def test_soft_channel_message_can_react_to_social_third_person_reference(
    db_session: Session,
) -> None:
    client = FakeSlackClient()
    decision = intent_decision(
        classification=IntentClassification.third_person_reference,
        suggested_reaction="wave",
    ).model_copy(
        update={
            "addressed_to_kortny": False,
            "should_create_task": False,
            "should_ack_with_reaction": True,
            "confidence": 0.92,
        }
    )
    classifier = FakeIntentClassifier(decision, require_task_id=False)
    reactions = FakeReactionProvider(name="wave", intent="third_person_reference")

    result = SlackIngress(
        session=db_session,
        client=client,
        intent_classifier=classifier,
        reaction_provider=reactions,
    ).handle_channel_message(
        body=message_body(event_id="EvSoftMentionIntro"),
        event=channel_event(
            text=(
                "Hey guys guess what? Kortny is our new coworker who will "
                "help us with tasks."
            )
        ),
        app_name="kortny",
    )
    db_session.commit()

    task_count = db_session.scalar(select(func.count()).select_from(Task))

    assert result is None
    assert task_count == 0
    assert len(classifier.calls) == 1
    assert client.calls == []
    assert client.reactions == [
        {
            "channel": "C123",
            "name": "wave",
            "timestamp": "1716600000.000001",
        }
    ]
    assert reactions.calls == [
        {
            "input_text": (
                "Hey guys guess what? Kortny is our new coworker who will "
                "help us with tasks."
            ),
            "source": "channel_message",
        }
    ]


def test_soft_channel_message_without_app_name_is_ignored_before_llm(
    db_session: Session,
) -> None:
    client = FakeSlackClient()
    classifier = FakeIntentClassifier(require_task_id=False)

    result = SlackIngress(
        session=db_session,
        client=client,
        intent_classifier=classifier,
    ).handle_channel_message(
        body=message_body(event_id="EvSoftMentionNoName"),
        event=channel_event(text="Can someone compare the two options?"),
        app_name="kortny",
    )
    db_session.commit()

    task_count = db_session.scalar(select(func.count()).select_from(Task))

    assert result is None
    assert task_count == 0
    assert classifier.calls == []
    assert client.calls == []
    assert client.reactions == []


def test_soft_channel_message_from_bot_is_ignored_before_llm(
    db_session: Session,
) -> None:
    client = FakeSlackClient()
    classifier = FakeIntentClassifier(require_task_id=False)

    result = SlackIngress(
        session=db_session,
        client=client,
        intent_classifier=classifier,
    ).handle_channel_message(
        body=message_body(event_id="EvSoftMentionBot"),
        event=channel_event(
            text="Kortny can you summarize this?",
            bot_id="B123",
        ),
        app_name="kortny",
    )
    db_session.commit()

    task_count = db_session.scalar(select(func.count()).select_from(Task))

    assert result is None
    assert task_count == 0
    assert classifier.calls == []
    assert client.calls == []
    assert client.reactions == []


def test_soft_channel_message_with_explicit_app_mention_is_ignored_before_llm(
    db_session: Session,
) -> None:
    client = FakeSlackClient()
    classifier = FakeIntentClassifier(require_task_id=False)

    result = SlackIngress(
        session=db_session,
        client=client,
        intent_classifier=classifier,
    ).handle_channel_message(
        body=message_body(event_id="EvSoftMentionExplicitMention"),
        event=channel_event(
            text="<@UBOT> Compare current AI observability tools for Kortny.",
        ),
        app_name="kortny",
    )
    db_session.commit()

    task_count = db_session.scalar(select(func.count()).select_from(Task))

    assert result is None
    assert task_count == 0
    assert classifier.calls == []
    assert client.calls == []
    assert client.reactions == []


def test_soft_channel_message_preserves_thread_context(
    db_session: Session,
) -> None:
    client = FakeSlackClient()
    classifier = FakeIntentClassifier(require_task_id=False)

    result = SlackIngress(
        session=db_session,
        client=client,
        intent_classifier=classifier,
    ).handle_channel_message(
        body=message_body(event_id="EvSoftMentionThread"),
        event=channel_event(
            text="Kortny can you summarize the decision above?",
            ts="1716600100.000001",
            thread_ts="1716600000.000001",
        ),
        app_name="kortny",
    )

    assert result is not None
    assert result.thread_ts == "1716600000.000001"
    assert result.task.slack_thread_ts == "1716600000.000001"
    assert result.task.slack_message_ts == "1716600100.000001"
    assert classifier.calls[0][1].is_thread_follow_up is True


def test_redelivered_dm_is_idempotent(db_session: Session) -> None:
    client = FakeSlackClient()
    ingress = SlackIngress(session=db_session, client=client)
    body = message_body(event_id="EvDmDuplicate")
    event = dm_event(text="research duplicate delivery")

    first = ingress.handle_dm(body=body, event=event)
    second = ingress.handle_dm(body=body, event=event)
    db_session.commit()

    task_count = db_session.scalar(select(func.count()).select_from(Task))
    message_event_count = db_session.scalar(
        select(func.count())
        .select_from(TaskEvent)
        .where(TaskEvent.type == TaskEventType.message_posted)
    )

    assert first is not None
    assert second is not None
    assert first.created is True
    assert second.created is False
    assert second.task.id == first.task.id
    assert task_count == 1
    assert message_event_count == 0
    assert client.calls == []


def test_dm_messages_share_conversation_context_key(db_session: Session) -> None:
    client = FakeSlackClient()
    ingress = SlackIngress(session=db_session, client=client)

    first = ingress.handle_dm(
        body=message_body(event_id="EvDmFirst"),
        event=dm_event(text="summarize this report", ts="1716500000.000001"),
    )
    second = ingress.handle_dm(
        body=message_body(event_id="EvDmSecond"),
        event=dm_event(text="extend that report", ts="1716500010.000001"),
    )

    assert first is not None
    assert second is not None
    assert first.task.slack_thread_ts == "D123"
    assert second.task.slack_thread_ts == "D123"
    assert first.task.slack_message_ts == "1716500000.000001"
    assert second.task.slack_message_ts == "1716500010.000001"


def test_cancel_reaction_on_source_message_cancels_pending_task(
    db_session: Session,
) -> None:
    client = FakeSlackClient()
    ingress = SlackIngress(session=db_session, client=client)
    created = ingress.handle_app_mention(
        body=app_mention_body(event_id="EvCancelAck"),
        event=app_mention_event(text="<@UBOT> do a cancellable task"),
    )

    result = ingress.handle_reaction_added(
        body=app_mention_body(event_id="EvReactionCancel"),
        event=reaction_event(
            reaction="x",
            user="U123",
            channel="C123",
            ts="1716400000.000001",
        ),
    )
    db_session.commit()

    db_session.refresh(created.task)
    events = task_events(db_session, created.task)

    assert result.handled is True
    assert result.action == "cancel"
    assert created.task.status is TaskStatus.cancelled
    assert events[-1].type is TaskEventType.status_changed
    assert events[-1].payload["to"] == "cancelled"
    assert events[-1].payload["by_user_id"] == "U123"


def test_cancel_reaction_cancels_running_task(db_session: Session) -> None:
    installation = create_installation(db_session)
    service = TaskService(db_session)
    task = service.create_task(
        installation_id=installation.id,
        slack_event_id="EvRunningCancel",
        slack_channel_id="C123",
        slack_thread_ts="1716401000.000001",
        slack_message_ts="1716401000.000001",
        slack_user_id="U123",
        input="long running work",
    )
    service.transition(task, TaskStatus.running)

    result = SlackIngress(
        session=db_session,
        client=FakeSlackClient(),
    ).handle_reaction_added(
        body=app_mention_body(event_id="EvReactionRunningCancel"),
        event=reaction_event(
            reaction="x",
            user="U123",
            channel="C123",
            ts="1716401000.000001",
        ),
    )

    assert result.handled is True
    assert result.action == "cancel"
    assert task.status is TaskStatus.cancelled
    assert task.locked_by is None
    assert task.lease_expires_at is None


def test_retry_reaction_requeues_failed_task_from_failure_notice(
    db_session: Session,
) -> None:
    installation = create_installation(db_session)
    service = TaskService(db_session)
    task = service.create_task(
        installation_id=installation.id,
        slack_event_id="EvRetryFailed",
        slack_channel_id="C123",
        slack_thread_ts="1716402000.000001",
        slack_message_ts="1716402000.000001",
        slack_user_id="U123",
        input="retry me",
    )
    service.transition(task, TaskStatus.running)
    service.transition(task, TaskStatus.failed)
    task.attempts = 2
    task.error = {"type": "RuntimeError", "message": "boom"}
    service.append_event(
        task,
        TaskEventType.message_posted,
        {
            "channel": "C123",
            "thread_ts": "1716402000.000001",
            "message_ts": "1716402001.000001",
            "purpose": "failure",
        },
    )

    result = SlackIngress(
        session=db_session,
        client=FakeSlackClient(),
    ).handle_reaction_added(
        body=app_mention_body(event_id="EvReactionRetry"),
        event=reaction_event(
            reaction="arrows_counterclockwise",
            user="U123",
            channel="C123",
            ts="1716402001.000001",
        ),
    )

    assert result.handled is True
    assert result.action == "retry"
    assert task.status is TaskStatus.pending
    assert task.attempts == 0
    assert task.error is None
    assert task.finished_at is None


def test_approval_reaction_requeues_waiting_task(db_session: Session) -> None:
    installation = create_installation(db_session)
    service = TaskService(db_session)
    task = service.create_task(
        installation_id=installation.id,
        slack_event_id="EvApprovalApprove",
        slack_channel_id="C123",
        slack_thread_ts="1716404000.000001",
        slack_message_ts="1716404000.000001",
        slack_user_id="U123",
        input="create a Linear issue",
    )
    approval_key = "composio_linear_create_issue:abc123"
    service.append_event(
        task,
        TaskEventType.log,
        {
            "message": TOOL_APPROVAL_REQUIRED_MESSAGE,
            "request": {
                "approval_key": approval_key,
                "tool": "composio_linear_create_issue",
                "tool_call_id": "call-create",
                "normalized_args_hash": "abc123",
                "argument_keys": ["title"],
                "scope": "user",
                "reason": "create action",
                "risk": "external_side_effect",
                "arguments": {"title": "Follow up"},
            },
        },
    )
    service.append_event(
        task,
        TaskEventType.message_posted,
        {
            "channel": "C123",
            "thread_ts": "1716404000.000001",
            "message_ts": "1716404001.000001",
            "purpose": TOOL_APPROVAL_PROMPT_PURPOSE,
        },
    )
    task.status = TaskStatus.waiting_approval
    db_session.commit()

    result = SlackIngress(
        session=db_session,
        client=FakeSlackClient(),
    ).handle_reaction_added(
        body=app_mention_body(event_id="EvReactionApprovalApprove"),
        event=reaction_event(
            reaction="white_check_mark",
            user="U123",
            channel="C123",
            ts="1716404001.000001",
        ),
    )

    assert result.handled is True
    assert result.action == "approve_tool"
    assert task.status is TaskStatus.pending
    events = task_events(db_session, task)
    assert any(
        event.payload.get("message") == "tool_approval_decision"
        and event.payload.get("decision") == "approved"
        and event.payload.get("approval_key") == approval_key
        for event in events
    )


def test_reject_approval_reaction_cancels_task_and_posts_note(
    db_session: Session,
) -> None:
    installation = create_installation(db_session)
    service = TaskService(db_session)
    task = service.create_task(
        installation_id=installation.id,
        slack_event_id="EvApprovalReject",
        slack_channel_id="C123",
        slack_thread_ts="1716405000.000001",
        slack_message_ts="1716405000.000001",
        slack_user_id="U123",
        input="create a Linear issue",
    )
    approval_key = "composio_linear_create_issue:def456"
    service.append_event(
        task,
        TaskEventType.log,
        {
            "message": TOOL_APPROVAL_REQUIRED_MESSAGE,
            "request": {
                "approval_key": approval_key,
                "tool": "composio_linear_create_issue",
                "tool_call_id": "call-create",
                "normalized_args_hash": "def456",
                "argument_keys": ["title"],
                "scope": "user",
                "reason": "create action",
                "risk": "external_side_effect",
                "arguments": {"title": "Follow up"},
            },
        },
    )
    service.append_event(
        task,
        TaskEventType.message_posted,
        {
            "channel": "C123",
            "thread_ts": "1716405000.000001",
            "message_ts": "1716405001.000001",
            "purpose": TOOL_APPROVAL_PROMPT_PURPOSE,
        },
    )
    task.status = TaskStatus.waiting_approval
    db_session.commit()
    client = FakeSlackClient()

    result = SlackIngress(
        session=db_session,
        client=client,
    ).handle_reaction_added(
        body=app_mention_body(event_id="EvReactionApprovalReject"),
        event=reaction_event(
            reaction="no_entry_sign",
            user="U123",
            channel="C123",
            ts="1716405001.000001",
        ),
    )

    assert result.handled is True
    assert result.action == "reject_tool"
    assert task.status is TaskStatus.cancelled
    assert client.calls == [
        {
            "channel": "C123",
            "text": "Okay, I won't run *composio_linear_create_issue*.",
            "thread_ts": "1716405000.000001",
        }
    ]
    events = task_events(db_session, task)
    assert any(
        event.type is TaskEventType.message_posted
        and event.payload.get("purpose") == TOOL_APPROVAL_REJECTED_PURPOSE
        for event in events
    )


def test_reaction_from_non_owner_is_ignored(db_session: Session) -> None:
    installation = create_installation(db_session)
    task = TaskService(db_session).create_task(
        installation_id=installation.id,
        slack_event_id="EvCancelNonOwner",
        slack_channel_id="C123",
        slack_thread_ts="1716403000.000001",
        slack_message_ts="1716403000.000001",
        slack_user_id="U123",
        input="owned work",
    )

    result = SlackIngress(
        session=db_session,
        client=FakeSlackClient(),
    ).handle_reaction_added(
        body=app_mention_body(event_id="EvReactionNonOwner"),
        event=reaction_event(
            reaction="x",
            user="U999",
            channel="C123",
            ts="1716403000.000001",
        ),
    )

    assert result.handled is False
    assert result.reason == "non_owner"
    assert task.status is TaskStatus.pending


def _install_intro_dms(client: FakeSlackClient) -> list[dict[str, Any]]:
    # The intro DM is the only chat_postMessage addressed to the user (U123),
    # not the channel (C123). Ack/result messages target the channel/thread.
    return [call for call in client.calls if call["channel"] == "U123"]


def _fresh_team_body(*, event_id: str, team_id: str) -> dict[str, Any]:
    return {"event_id": event_id, "team_id": team_id}


def test_first_installation_posts_intro_dm_once(db_session: Session) -> None:
    client = FakeSlackClient()
    team_id = "TIntroFresh1"
    SlackIngress(session=db_session, client=client).handle_app_mention(
        body=_fresh_team_body(event_id="EvIntro1", team_id=team_id),
        event=app_mention_event(text="<@UBOT> hello"),
    )
    db_session.commit()

    intro_dms = _install_intro_dms(client)
    assert len(intro_dms) == 1
    assert intro_dms[0]["channel"] == "U123"
    assert "AI coworker" in intro_dms[0]["text"]

    installation = db_session.scalar(
        select(Installation).where(Installation.slack_team_id == team_id)
    )
    assert installation is not None
    side_effect = db_session.scalar(
        select(SlackSideEffect).where(
            SlackSideEffect.idempotency_key
            == slack_install_intro_key(installation_id=installation.id)
        )
    )
    assert side_effect is not None
    assert side_effect.purpose == "install_intro"


def test_second_event_does_not_repost_intro_dm(db_session: Session) -> None:
    client = FakeSlackClient()
    team_id = "TIntroFresh2"
    ingress = SlackIngress(session=db_session, client=client)
    ingress.handle_app_mention(
        body=_fresh_team_body(event_id="EvIntro2a", team_id=team_id),
        event=app_mention_event(text="<@UBOT> first", ts="1716400000.000010"),
    )
    db_session.commit()
    assert len(_install_intro_dms(client)) == 1

    # A second event for the same (already-created) installation: no second DM.
    ingress.handle_app_mention(
        body=_fresh_team_body(event_id="EvIntro2b", team_id=team_id),
        event=app_mention_event(text="<@UBOT> second", ts="1716400000.000020"),
    )
    db_session.commit()
    assert len(_install_intro_dms(client)) == 1


def cleanup_database(session: Session) -> None:
    for model in (
        Artifact,
        LLMUsage,
        SlackInboundEvent,
        TaskEvent,
        SlackSideEffect,
        Task,
        Schedule,
        ModelPricing,
        ObservationEvent,
        ObservePolicy,
        SlackChannelMembership,
        SlackIdentity,
        EncryptedSecret,
        Installation,
    ):
        session.execute(delete(model))


def create_installation(
    session: Session,
    *,
    slack_team_id: str | None = None,
) -> Installation:
    team_id = slack_team_id or f"T{uuid.uuid4().hex}"
    existing = session.scalar(
        select(Installation).where(Installation.slack_team_id == team_id)
    )
    if existing is not None:
        return existing
    installation = Installation(slack_team_id=team_id)
    session.add(installation)
    session.flush()
    return installation


def intent_decision(
    *,
    classification: IntentClassification = IntentClassification.task_request,
    suggested_reaction: str | None = "eyes",
) -> IntentDecision:
    return IntentDecision(
        addressed_to_kortny=True,
        classification=classification,
        confidence=0.9,
        should_create_task=True,
        should_ack_with_reaction=True,
        suggested_reaction=suggested_reaction,
        needs_channel_context=False,
        needs_thread_context=False,
        needs_file_context=False,
        likely_tools=[],
        model_tier=ModelTier.cheap,
        reason="Direct request to Kortny.",
    )


def task_events(session: Session, task: Task) -> list[TaskEvent]:
    return list(
        session.scalars(
            select(TaskEvent)
            .where(TaskEvent.task_id == task.id)
            .order_by(TaskEvent.seq)
        )
    )


def app_mention_body(*, event_id: str | None = None) -> dict[str, Any]:
    return {
        "event_id": event_id or f"Ev{uuid.uuid4().hex}",
        "team_id": "T123",
    }


def message_body(*, event_id: str | None = None) -> dict[str, Any]:
    return {
        "event_id": event_id or f"Ev{uuid.uuid4().hex}",
        "team_id": "T123",
    }


def schedule_action_body(
    *,
    channel_id: str,
    user_id: str,
    message_ts: str,
    team_id: str = "T123",
) -> dict[str, Any]:
    return {
        "team": {"id": team_id},
        "user": {"id": user_id},
        "channel": {"id": channel_id},
        "message": {"ts": message_ts},
    }


def member_joined_body(
    *,
    event_id: str | None = None,
    authorizations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "event_id": event_id or f"Ev{uuid.uuid4().hex}",
        "team_id": "T123",
    }
    if authorizations is not None:
        body["authorizations"] = authorizations
    return body


def app_mention_event(
    *,
    text: str = "<@UBOT> research a topic",
    ts: str = "1716400000.000001",
    thread_ts: str | None = None,
    files: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "type": "app_mention",
        "channel": "C123",
        "user": "U123",
        "text": text,
        "ts": ts,
    }
    if thread_ts is not None:
        event["thread_ts"] = thread_ts
    if files is not None:
        event["files"] = files
    return event


def dm_event(
    *,
    text: str = "research a private topic",
    channel: str = "D123",
    channel_type: str = "im",
    ts: str = "1716500000.000001",
    thread_ts: str | None = None,
    subtype: str | None = None,
    bot_id: str | None = None,
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "type": "message",
        "channel": channel,
        "channel_type": channel_type,
        "user": "U123",
        "text": text,
        "ts": ts,
    }
    if thread_ts is not None:
        event["thread_ts"] = thread_ts
    if subtype is not None:
        event["subtype"] = subtype
    if bot_id is not None:
        event["bot_id"] = bot_id
    return event


def channel_event(
    *,
    text: str = "Kortny can you research this?",
    channel: str = "C123",
    channel_type: str = "channel",
    ts: str = "1716600000.000001",
    thread_ts: str | None = None,
    subtype: str | None = None,
    bot_id: str | None = None,
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "type": "message",
        "channel": channel,
        "channel_type": channel_type,
        "user": "U123",
        "text": text,
        "ts": ts,
    }
    if thread_ts is not None:
        event["thread_ts"] = thread_ts
    if subtype is not None:
        event["subtype"] = subtype
    if bot_id is not None:
        event["bot_id"] = bot_id
    return event


def member_joined_event(
    *,
    user: str = "UBOT",
    channel: str = "C123",
    inviter: str | None = "U123",
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "type": "member_joined_channel",
        "channel": channel,
        "user": user,
        "team": "T123",
    }
    if inviter is not None:
        event["inviter"] = inviter
    return event


def reaction_event(
    *,
    reaction: str,
    user: str,
    channel: str,
    ts: str,
) -> dict[str, Any]:
    return {
        "type": "reaction_added",
        "reaction": reaction,
        "user": user,
        "item": {
            "type": "message",
            "channel": channel,
            "ts": ts,
        },
    }
