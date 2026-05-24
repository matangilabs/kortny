import os
import uuid
from collections.abc import Iterator
from typing import Any

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, delete, func, select
from sqlalchemy.orm import Session

from kortny.db.models import (
    Artifact,
    EncryptedSecret,
    Installation,
    LLMUsage,
    ModelPricing,
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
from kortny.slack import SlackIngress, acknowledge_then_handle
from kortny.slack.ingress import INTENT_CLASSIFIED_MESSAGE
from kortny.slack.reactions import ACK_REACTION_ADDED_MESSAGE, ReactionChoice
from kortny.tasks import TaskService

TEST_POSTGRES_URL = os.environ.get("KORTNY_TEST_POSTGRES_URL")


class FakeSlackClient:
    def __init__(self, *, reaction_error: Exception | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.reactions: list[dict[str, Any]] = []
        self.reaction_error = reaction_error

    def chat_postMessage(
        self,
        *,
        channel: str,
        text: str,
        thread_ts: str | None = None,
    ) -> dict[str, Any]:
        call = {
            "channel": channel,
            "text": text,
            "thread_ts": thread_ts,
        }
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


def test_acknowledge_then_handle_acks_before_work() -> None:
    calls: list[str] = []

    def ack() -> None:
        calls.append("ack")

    def handle() -> str:
        calls.append("handle")
        return "done"

    assert acknowledge_then_handle(ack, handle) == "done"
    assert calls == ["ack", "handle"]


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
    event = app_mention_event(text="<@UBOT> search duplicate delivery")

    first = ingress.handle_app_mention(body=body, event=event)
    second = ingress.handle_app_mention(body=body, event=event)
    db_session.commit()

    task_count = db_session.scalar(select(func.count()).select_from(Task))
    message_event_count = db_session.scalar(
        select(func.count())
        .select_from(TaskEvent)
        .where(TaskEvent.type == TaskEventType.message_posted)
    )

    assert first.created is True
    assert second.created is False
    assert second.task.id == first.task.id
    assert task_count == 1
    assert message_event_count == 0
    assert client.calls == []
    assert len(client.reactions) == 1


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


def test_dm_from_bot_is_ignored(db_session: Session) -> None:
    client = FakeSlackClient()

    result = SlackIngress(session=db_session, client=client).handle_dm(
        body=message_body(event_id="EvDmBot"),
        event=dm_event(bot_id="B123", text="bot reply"),
    )
    db_session.commit()

    task_count = db_session.scalar(select(func.count()).select_from(Task))

    assert result is None
    assert task_count == 0
    assert client.calls == []


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


def cleanup_database(session: Session) -> None:
    for model in (
        Artifact,
        LLMUsage,
        TaskEvent,
        Task,
        ModelPricing,
        EncryptedSecret,
        Installation,
    ):
        session.execute(delete(model))


def create_installation(session: Session) -> Installation:
    installation = Installation(slack_team_id=f"T{uuid.uuid4().hex}")
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
