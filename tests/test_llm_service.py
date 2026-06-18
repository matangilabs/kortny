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

import kortny.llm.service as llm_service_module
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
        self.max_output_tokens: list[int | None] = []

    def complete(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[JsonSchema] = (),
        *,
        response_format: JsonObject | None = None,
        max_output_tokens: int | None = None,
    ) -> Completion:
        del response_format
        self.calls.append((messages, tools))
        self.max_output_tokens.append(max_output_tokens)
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


def test_calculate_cost_usd_zero_cache_passthrough() -> None:
    pricing = ModelPricing(
        provider=LLMProvider.openrouter,
        model="m",
        input_price_per_mtok=Decimal("10.000000"),
        output_price_per_mtok=Decimal("30.000000"),
    )
    # No cache split → identical to the plain per-token cost.
    assert calculate_cost_usd(
        TokenUsage(
            input_tokens=1000,
            output_tokens=2000,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
        pricing,
    ) == Decimal("0.070000")


def test_calculate_cost_usd_applies_cache_multipliers() -> None:
    pricing = ModelPricing(
        provider=LLMProvider.openrouter,
        model="m",
        input_price_per_mtok=Decimal("10.000000"),
        output_price_per_mtok=Decimal("0.000000"),
        cache_write_multiplier=Decimal("1.25"),
        cache_read_multiplier=Decimal("0.10"),
    )
    # 1000 total prompt = 200 uncached + 300 creation + 500 read.
    # cost = (200*1 + 300*1.25 + 500*0.10) * 10 / 1e6
    #      = (200 + 375 + 50) * 10 / 1e6 = 6250 / 1e6 = 0.006250
    cost = calculate_cost_usd(
        TokenUsage(
            input_tokens=1000,
            output_tokens=0,
            cache_creation_input_tokens=300,
            cache_read_input_tokens=500,
        ),
        pricing,
    )
    assert cost == Decimal("0.006250")


def test_calculate_cost_usd_clamps_uncached_remainder() -> None:
    pricing = ModelPricing(
        provider=LLMProvider.openrouter,
        model="m",
        input_price_per_mtok=Decimal("10.000000"),
        output_price_per_mtok=Decimal("0.000000"),
        cache_write_multiplier=Decimal("1.25"),
        cache_read_multiplier=Decimal("0.10"),
    )
    # cache tokens exceed the reported total → uncached must clamp to 0,
    # never produce a negative charge.
    cost = calculate_cost_usd(
        TokenUsage(
            input_tokens=100,
            output_tokens=0,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=500,
        ),
        pricing,
    )
    # uncached = max(0, 100 - 500) = 0; read = 500*0.10*10/1e6 = 0.000500
    assert cost == Decimal("0.000500")


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


def test_llm_service_records_usage_and_rolls_up_cost(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    span_attributes: list[JsonObject] = []
    monkeypatch.setattr(
        llm_service_module, "set_span_attributes", span_attributes.append
    )
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
    assert span_attributes[-1]["openinference.span.kind"] == "LLM"
    assert span_attributes[-1]["llm.model_name"] == "openai/gpt-4o-mini"
    assert span_attributes[-1]["llm.token_count.prompt"] == 1000
    assert span_attributes[-1]["llm.token_count.completion"] == 2000
    assert span_attributes[-1]["llm.token_count.total"] == 3000


def _usage_pricing() -> ModelPricing:
    return ModelPricing(
        provider=LLMProvider.openrouter,
        model="openai/gpt-4o-mini",
        input_price_per_mtok=Decimal("10.000000"),
        output_price_per_mtok=Decimal("30.000000"),
        effective_from=datetime(2026, 1, 1, tzinfo=UTC),
    )


def test_llm_service_clamps_utility_prompt_output(db_session: Session) -> None:
    # HIG-220: a utility prompt is clamped by name; a normal prompt is not.
    task = create_task(db_session)
    db_session.add(_usage_pricing())
    db_session.flush()

    def _provider() -> FakeProvider:
        return FakeProvider(
            Completion(
                content="{}",
                tool_calls=(),
                usage=TokenUsage(input_tokens=10, output_tokens=10),
                model="openai/gpt-4o-mini",
            )
        )

    clamped = _provider()
    LLMService(
        session=db_session,
        provider=clamped,
        provider_name=LLMProvider.openrouter,
    ).complete(
        task_id=task.id,
        messages=[ChatMessage(role="user", content="hi")],
        prompt_name="kortny.intent_classifier",
    )
    assert clamped.max_output_tokens[0] == 1024

    unclamped = _provider()
    LLMService(
        session=db_session,
        provider=unclamped,
        provider_name=LLMProvider.openrouter,
    ).complete(
        task_id=task.id,
        messages=[ChatMessage(role="user", content="hi")],
        prompt_name="kortny.some_long_synthesis",
    )
    assert unclamped.max_output_tokens[0] is None


def test_llm_service_records_cache_token_split(
    db_session: Session,
) -> None:
    task = create_task(db_session)
    pricing = ModelPricing(
        provider=LLMProvider.openrouter,
        model="openai/gpt-4o-mini",
        input_price_per_mtok=Decimal("10.000000"),
        output_price_per_mtok=Decimal("30.000000"),
        cache_write_multiplier=Decimal("1.25"),
        cache_read_multiplier=Decimal("0.10"),
        effective_from=datetime(2026, 1, 1, tzinfo=UTC),
    )
    db_session.add(pricing)
    db_session.flush()
    provider = FakeProvider(
        Completion(
            content="ok",
            tool_calls=(),
            usage=TokenUsage(
                input_tokens=1000,
                output_tokens=100,
                cache_creation_input_tokens=200,
                cache_read_input_tokens=600,
            ),
            response_id="gen-cache",
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
    assert usage.input_tokens == 1000
    assert usage.cache_creation_input_tokens == 200
    assert usage.cache_read_input_tokens == 600
    # Cost reflects the cache split (D5):
    # uncached = 200, creation = 200*1.25, read = 600*0.10
    # input = (200 + 250 + 60) * 10 / 1e6 = 5100/1e6 = 0.005100
    # output = 100 * 30 / 1e6 = 0.003000 ; total = 0.008100
    assert usage.cost_usd == Decimal("0.008100")
    event = db_session.scalar(
        select(TaskEvent).where(
            TaskEvent.task_id == task.id, TaskEvent.type == TaskEventType.llm_call
        )
    )
    assert event is not None
    assert event.payload["cache_creation_input_tokens"] == 200
    assert event.payload["cache_read_input_tokens"] == 600


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


def test_llm_service_stamps_registered_prompt_version(db_session: Session) -> None:
    # HIG-203: a registered prompt's version lands in the usage event payload.
    task = create_task(db_session)
    db_session.add(
        ModelPricing(
            provider=LLMProvider.openrouter,
            model="openai/gpt-4o-mini",
            input_price_per_mtok=Decimal("10.000000"),
            output_price_per_mtok=Decimal("30.000000"),
            effective_from=datetime(2026, 1, 1, tzinfo=UTC),
        )
    )
    db_session.flush()
    provider = FakeProvider(
        Completion(
            content="{}",
            tool_calls=(),
            usage=TokenUsage(input_tokens=5, output_tokens=5),
            model="openai/gpt-4o-mini",
        )
    )
    LLMService(
        session=db_session,
        provider=provider,
        provider_name=LLMProvider.openrouter,
    ).complete(
        task_id=task.id,
        messages=[ChatMessage(role="user", content="hi")],
        prompt_name="kortny.intent_classifier",
    )
    events = list(
        db_session.scalars(
            select(TaskEvent)
            .where(TaskEvent.task_id == task.id)
            .order_by(TaskEvent.seq)
        )
    )
    completed = [e for e in events if e.payload.get("message") == "llm_call_completed"]
    assert completed
    assert completed[-1].payload["prompt_name"] == "kortny.intent_classifier"
    assert completed[-1].payload["prompt_version"] == "1"
