import os
import uuid
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from decimal import Decimal

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, delete, select
from sqlalchemy.orm import Session

from kortny.db.models import (
    Artifact,
    EncryptedSecret,
    Installation,
    LLMProvider,
    LLMUsage,
    ModelPricing,
    Task,
    TaskEvent,
    TaskEventType,
)
from kortny.db.session import make_engine, make_session_factory, normalize_database_url
from kortny.llm import (
    ChatMessage,
    Completion,
    LLMService,
    ModelRoute,
    ModelRouteTier,
    TokenUsage,
)
from kortny.llm.service import ModelPricingNotFoundError, calculate_cost_usd
from kortny.tasks import TaskService
from kortny.tools.types import JsonObject, JsonSchema

TEST_POSTGRES_URL = os.environ.get("KORTNY_TEST_POSTGRES_URL")


class FakeProvider:
    model = "openai/gpt-4o-mini"

    def __init__(self, completion: Completion) -> None:
        self.completion = completion
        self.calls: list[tuple[Sequence[ChatMessage], Sequence[JsonSchema]]] = []

    def complete(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[JsonSchema] = (),
        *,
        response_format: JsonObject | None = None,
    ) -> Completion:
        del response_format
        self.calls.append((messages, tools))
        return self.completion


def test_calculate_cost_usd_uses_per_million_token_pricing() -> None:
    pricing = ModelPricing(
        provider=LLMProvider.openrouter,
        model="openai/gpt-4o-mini",
        input_price_per_mtok=Decimal("10.000000"),
        output_price_per_mtok=Decimal("30.000000"),
    )

    assert calculate_cost_usd(
        TokenUsage(input_tokens=1000, output_tokens=2000),
        pricing,
    ) == Decimal("0.070000")


@pytest.fixture(scope="session")
def engine() -> Iterator[Engine]:
    if TEST_POSTGRES_URL is None:
        pytest.skip("KORTNY_TEST_POSTGRES_URL is required for LLM service tests")

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


def test_llm_service_records_usage_and_rolls_up_cost(db_session: Session) -> None:
    task = create_task(db_session)
    pricing = ModelPricing(
        provider=LLMProvider.openrouter,
        model="openai/gpt-4o-mini",
        input_price_per_mtok=Decimal("10.000000"),
        output_price_per_mtok=Decimal("30.000000"),
        effective_from=datetime(2026, 1, 1, tzinfo=UTC),
    )
    db_session.add(pricing)
    db_session.flush()
    provider = FakeProvider(
        Completion(
            content="Use web_search.",
            tool_calls=(),
            usage=TokenUsage(input_tokens=1000, output_tokens=2000),
            response_id="gen-123",
            model="openai/gpt-4o-mini",
        )
    )

    completion = LLMService(
        session=db_session,
        provider=provider,
        provider_name=LLMProvider.openrouter,
    ).complete(
        task_id=task.id,
        messages=[ChatMessage(role="user", content="hello")],
        tools=[{"name": "web_search", "description": "Search.", "parameters": {}}],
    )

    usage = db_session.scalar(select(LLMUsage).where(LLMUsage.task_id == task.id))
    events = list(
        db_session.scalars(
            select(TaskEvent)
            .where(TaskEvent.task_id == task.id)
            .order_by(TaskEvent.seq)
        )
    )

    assert completion.response_id == "gen-123"
    assert provider.calls[0][0] == [ChatMessage(role="user", content="hello")]
    assert usage is not None
    assert usage.provider is LLMProvider.openrouter
    assert usage.model == "openai/gpt-4o-mini"
    assert usage.input_tokens == 1000
    assert usage.output_tokens == 2000
    assert usage.cost_usd == Decimal("0.070000")
    assert task.total_input_tokens == 1000
    assert task.total_output_tokens == 2000
    assert task.total_cost_usd == Decimal("0.070000")
    assert events[-2].type is TaskEventType.log
    assert events[-2].payload["message"] == "llm_call_started"
    assert events[-1].type is TaskEventType.llm_call
    assert events[-1].payload["message"] == "llm_call_completed"
    assert events[-1].payload["response_id"] == "gen-123"
    assert events[-1].payload["latency_ms"] >= 0
    assert events[-1].payload["message_count"] == 1
    assert events[-1].payload["tool_count"] == 1
    assert events[-1].payload["total_tokens"] == 3000
    assert events[-1].payload["prompt_name"] == "kortny.agent_coordinator.system"
    assert events[-1].payload["prompt_source"] == "code"


def test_llm_service_records_model_tier(db_session: Session) -> None:
    task = create_task(db_session)
    provider = FakeProvider(
        Completion(
            content="done",
            tool_calls=(),
            usage=TokenUsage(input_tokens=10, output_tokens=5),
            cost_usd=Decimal("0.000123"),
            model="anthropic/sonnet",
        )
    )

    LLMService(
        session=db_session,
        provider=provider,
        provider_name=LLMProvider.openrouter,
        model_route=ModelRoute(
            tier=ModelRouteTier.analysis,
            model="anthropic/sonnet",
            reason="test",
        ),
    ).complete(
        task_id=task.id,
        messages=[ChatMessage(role="user", content="hello")],
    )

    usage = db_session.scalar(select(LLMUsage).where(LLMUsage.task_id == task.id))
    event = db_session.scalar(
        select(TaskEvent).where(
            TaskEvent.task_id == task.id,
            TaskEvent.type == TaskEventType.llm_call,
        )
    )

    assert usage is not None
    assert usage.model_tier == "analysis"
    assert event is not None
    assert event.payload["model_tier"] == "analysis"
    assert event.payload["route_reason"] == "test"


def test_llm_service_uses_latest_effective_pricing(db_session: Session) -> None:
    task = create_task(db_session)
    db_session.add_all(
        [
            ModelPricing(
                provider=LLMProvider.openrouter,
                model="openai/gpt-4o-mini",
                input_price_per_mtok=Decimal("100.000000"),
                output_price_per_mtok=Decimal("100.000000"),
                effective_from=datetime(2026, 1, 1, tzinfo=UTC),
            ),
            ModelPricing(
                provider=LLMProvider.openrouter,
                model="openai/gpt-4o-mini",
                input_price_per_mtok=Decimal("1.000000"),
                output_price_per_mtok=Decimal("1.000000"),
                effective_from=datetime(2026, 5, 1, tzinfo=UTC),
            ),
        ]
    )
    db_session.flush()
    provider = FakeProvider(
        Completion(
            content="done",
            tool_calls=(),
            usage=TokenUsage(input_tokens=1000, output_tokens=1000),
            model="openai/gpt-4o-mini",
        )
    )

    LLMService(
        session=db_session,
        provider=provider,
        provider_name=LLMProvider.openrouter,
    ).complete(
        task_id=task.id,
        messages=[ChatMessage(role="user", content="hello")],
    )

    assert task.total_cost_usd == Decimal("0.002000")


def test_llm_service_requires_model_pricing(db_session: Session) -> None:
    task = create_task(db_session)
    provider = FakeProvider(
        Completion(
            content="done",
            tool_calls=(),
            usage=TokenUsage(input_tokens=1, output_tokens=1),
            model="openai/gpt-4o-mini",
        )
    )

    with pytest.raises(ModelPricingNotFoundError):
        LLMService(
            session=db_session,
            provider=provider,
            provider_name=LLMProvider.openrouter,
        ).complete(
            task_id=task.id,
            messages=[ChatMessage(role="user", content="hello")],
        )


def test_llm_service_uses_provider_cost_without_pricing(db_session: Session) -> None:
    task = create_task(db_session)
    provider = FakeProvider(
        Completion(
            content="done",
            tool_calls=(),
            usage=TokenUsage(input_tokens=1000, output_tokens=1000),
            cost_usd=Decimal("0.004200"),
            model="openai/gpt-4o-mini",
        )
    )

    LLMService(
        session=db_session,
        provider=provider,
        provider_name=LLMProvider.openrouter,
    ).complete(
        task_id=task.id,
        messages=[ChatMessage(role="user", content="hello")],
    )

    usage = db_session.scalar(select(LLMUsage).where(LLMUsage.task_id == task.id))

    assert usage is not None
    assert usage.cost_usd == Decimal("0.004200")
    assert task.total_cost_usd == Decimal("0.004200")


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


def create_task(session: Session) -> Task:
    installation = Installation(slack_team_id=f"T{uuid.uuid4().hex}")
    session.add(installation)
    session.flush()
    return TaskService(session).create_task(
        installation_id=installation.id,
        slack_event_id=f"Ev{uuid.uuid4().hex}",
        slack_channel_id="C123",
        slack_thread_ts="1716400000.000001",
        slack_message_ts="1716400000.000001",
        slack_user_id="U123",
        input="hello",
    )
