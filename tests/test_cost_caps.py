"""Tests for the runtime cost cap feature.

Covers:
- LLMService.complete() raises TaskCostBudgetExceeded pre-call when ceiling is met.
- LLMService.complete() proceeds normally when under ceiling or no ceiling is set.
- Settings defaults for ambient and consolidator cost ceilings.
- AgentCoordinator terminates gracefully (no extra LLM call) when budget exceeded.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator, Sequence
from decimal import Decimal

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, delete
from sqlalchemy.orm import Session

from kortny.db.models import (
    Artifact,
    AutonomyPolicy,
    EncryptedSecret,
    Episode,
    Installation,
    KnowledgeGraphEdge,
    KnowledgeGraphEntity,
    KnowledgeGraphEvidence,
    LLMUsage,
    ModelPricing,
    SlackChannelMembership,
    Task,
    TaskEvent,
    WorkspaceState,
)
from kortny.db.session import make_engine, make_session_factory, normalize_database_url
from kortny.llm import ChatMessage, Completion, LLMService, TokenUsage
from kortny.llm.service import TaskCostBudgetExceeded
from kortny.tasks import TaskService
from kortny.tools.types import JsonObject, JsonSchema

TEST_POSTGRES_URL = os.environ.get("KORTNY_TEST_POSTGRES_URL")

needs_db = pytest.mark.skipif(
    TEST_POSTGRES_URL is None,
    reason="KORTNY_TEST_POSTGRES_URL is required for cost-cap DB tests",
)


# ---------------------------------------------------------------------------
# DB fixtures (mirrors test_llm_service.py / test_agent_coordinator.py)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def _engine() -> Iterator[Engine]:
    if TEST_POSTGRES_URL is None:
        pytest.skip("KORTNY_TEST_POSTGRES_URL is required")

    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", normalize_database_url(TEST_POSTGRES_URL))
    command.upgrade(cfg, "head")

    engine = make_engine(TEST_POSTGRES_URL)
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture
def db_session(_engine: Engine) -> Iterator[Session]:
    factory = make_session_factory(engine=_engine)
    with factory() as session:
        _cleanup(session)
        session.commit()
        yield session
        session.rollback()
        _cleanup(session)
        session.commit()


def _cleanup(session: Session) -> None:
    from sqlalchemy import update

    session.execute(update(WorkspaceState).values(superseded_by_id=None))
    for model in (
        KnowledgeGraphEvidence,
        KnowledgeGraphEdge,
        KnowledgeGraphEntity,
        SlackChannelMembership,
        WorkspaceState,
        Episode,
        Artifact,
        LLMUsage,
        TaskEvent,
        Task,
        ModelPricing,
        EncryptedSecret,
        AutonomyPolicy,
        Installation,
    ):
        session.execute(delete(model))


def _make_installation(session: Session) -> Installation:
    inst = Installation(slack_team_id=f"T{uuid.uuid4().hex}")
    session.add(inst)
    session.flush()
    return inst


def _make_task(
    session: Session,
    *,
    input_text: str = "hello",
    identity_payload: dict | None = None,
    total_cost_usd: Decimal = Decimal("0"),
) -> Task:
    inst = _make_installation(session)
    task = TaskService(session).create_task(
        installation_id=inst.id,
        slack_event_id=f"Ev{uuid.uuid4().hex}",
        slack_channel_id="C123",
        slack_thread_ts="1716400000.000001",
        slack_message_ts=f"{uuid.uuid4().int % 10**16:016d}.000001",
        slack_user_id="U123",
        input=input_text,
    )
    if identity_payload is not None:
        task.identity_payload = identity_payload
    if total_cost_usd:
        task.total_cost_usd = total_cost_usd
    session.flush()
    return task


# ---------------------------------------------------------------------------
# Minimal fake provider (no actual LLM calls)
# ---------------------------------------------------------------------------


class _FakeProvider:
    """Fake LLM provider that records calls and returns a canned Completion."""

    model = "openai/gpt-4o-mini"

    def __init__(self, completion: Completion) -> None:
        self._completion = completion
        self.call_count = 0

    def complete(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[JsonSchema] = (),
        *,
        response_format: JsonObject | None = None,
        max_output_tokens: int | None = None,
    ) -> Completion:
        del messages, tools, response_format, max_output_tokens
        self.call_count += 1
        return self._completion


def _fake_completion() -> Completion:
    return Completion(
        content="done",
        tool_calls=(),
        usage=TokenUsage(input_tokens=10, output_tokens=5),
        cost_usd=Decimal("0.000050"),
        model="openai/gpt-4o-mini",
    )


# ---------------------------------------------------------------------------
# LLMService cost ceiling tests (require DB)
# ---------------------------------------------------------------------------


class TestLLMServiceCostCeiling:
    @needs_db
    def test_raises_when_ceiling_exceeded(self, db_session: Session) -> None:
        """Pre-call guard fires when total_cost_usd >= ceiling; provider not called."""
        task = _make_task(
            db_session,
            identity_payload={"runtime_cost_ceiling_usd": "0.10"},
            total_cost_usd=Decimal("0.15"),
        )
        provider = _FakeProvider(_fake_completion())

        with pytest.raises(TaskCostBudgetExceeded) as exc_info:
            LLMService(
                session=db_session,
                provider=provider,
                provider_name="openrouter",
            ).complete(
                task_id=task.id,
                messages=[ChatMessage(role="user", content="hello")],
            )

        assert exc_info.value.task_id == task.id
        assert exc_info.value.ceiling == Decimal("0.10")
        assert exc_info.value.current == Decimal("0.15")
        # The provider must not have been called — no provider spend occurred.
        assert provider.call_count == 0

    @needs_db
    def test_raises_when_ceiling_exactly_met(self, db_session: Session) -> None:
        """Ceiling is inclusive: cost == ceiling also raises before the provider."""
        task = _make_task(
            db_session,
            identity_payload={"runtime_cost_ceiling_usd": "0.10"},
            total_cost_usd=Decimal("0.10"),
        )
        provider = _FakeProvider(_fake_completion())

        with pytest.raises(TaskCostBudgetExceeded):
            LLMService(
                session=db_session,
                provider=provider,
                provider_name="openrouter",
            ).complete(
                task_id=task.id,
                messages=[ChatMessage(role="user", content="hello")],
            )

        assert provider.call_count == 0

    @needs_db
    def test_proceeds_when_under_ceiling(self, db_session: Session) -> None:
        """When cost < ceiling the provider call goes through normally."""
        task = _make_task(
            db_session,
            identity_payload={"runtime_cost_ceiling_usd": "0.10"},
            total_cost_usd=Decimal("0.05"),
        )
        provider = _FakeProvider(_fake_completion())

        completion = LLMService(
            session=db_session,
            provider=provider,
            provider_name="openrouter",
        ).complete(
            task_id=task.id,
            messages=[ChatMessage(role="user", content="hello")],
        )

        assert provider.call_count == 1
        assert completion.content == "done"

    @needs_db
    def test_proceeds_with_no_ceiling_set(self, db_session: Session) -> None:
        """Interactive tasks have no ceiling key; even huge cumulative cost is allowed."""
        task = _make_task(
            db_session,
            identity_payload={},  # no runtime_cost_ceiling_usd
            total_cost_usd=Decimal("999.99"),
        )
        provider = _FakeProvider(_fake_completion())

        completion = LLMService(
            session=db_session,
            provider=provider,
            provider_name="openrouter",
        ).complete(
            task_id=task.id,
            messages=[ChatMessage(role="user", content="hello")],
        )

        assert provider.call_count == 1
        assert completion.content == "done"

    @needs_db
    def test_error_message_contains_task_and_amounts(self, db_session: Session) -> None:
        """TaskCostBudgetExceeded string representation is auditable."""
        task = _make_task(
            db_session,
            identity_payload={"runtime_cost_ceiling_usd": "0.25"},
            total_cost_usd=Decimal("0.30"),
        )

        with pytest.raises(TaskCostBudgetExceeded) as exc_info:
            LLMService(
                session=db_session,
                provider=_FakeProvider(_fake_completion()),
                provider_name="openrouter",
            ).complete(
                task_id=task.id,
                messages=[ChatMessage(role="user", content="hello")],
            )

        msg = str(exc_info.value)
        assert str(task.id) in msg
        assert "0.25" in msg
        assert "0.30" in msg


# ---------------------------------------------------------------------------
# Settings defaults (pure unit test — no DB required)
# ---------------------------------------------------------------------------


class TestSettingsCostCeilingDefaults:
    def test_ambient_and_consolidator_ceiling_defaults(self) -> None:
        """Ambient task ceiling = 0.25 USD, consolidator ceiling = 2.00 USD by default."""
        from kortny.config.settings import Settings

        s = Settings.model_validate(
            {
                "SLACK_BOT_TOKEN": "xoxb-test",
                "SLACK_APP_TOKEN": "xapp-test",
                "SLACK_SIGNING_SECRET": "secret",
                "LLM_PROVIDER": "openai",
                "LLM_API_KEY": "test-key",
                "LLM_MODEL": "gpt-4",
                "POSTGRES_URL": "postgresql://localhost/test",
                "COMPOSIO_API_KEY": "test",
                "ENCRYPTION_KEY": "test-key",
            }
        )

        assert s.ambient_task_cost_ceiling_usd == 0.25
        assert s.consolidator_run_cost_ceiling_usd == 2.00

    def test_ceiling_values_round_trip_as_string(self) -> None:
        """str(setting) produces the canonical value stamped in identity_payload."""
        from kortny.config.settings import Settings

        s = Settings.model_validate(
            {
                "SLACK_BOT_TOKEN": "xoxb-test",
                "SLACK_APP_TOKEN": "xapp-test",
                "SLACK_SIGNING_SECRET": "secret",
                "LLM_PROVIDER": "openai",
                "LLM_API_KEY": "test-key",
                "LLM_MODEL": "gpt-4",
                "POSTGRES_URL": "postgresql://localhost/test",
                "COMPOSIO_API_KEY": "test",
                "ENCRYPTION_KEY": "test-key",
            }
        )

        # These are the strings that ambient task creators stamp into identity_payload.
        # Decimal("0.25") must parse back identically so the ceiling guard works.
        ambient_ceiling_str = str(s.ambient_task_cost_ceiling_usd)
        consolidator_ceiling_str = str(s.consolidator_run_cost_ceiling_usd)

        assert Decimal(ambient_ceiling_str) == Decimal("0.25")
        assert Decimal(consolidator_ceiling_str) == Decimal("2.0")


# ---------------------------------------------------------------------------
# Coordinator cost cap handling (requires DB)
# ---------------------------------------------------------------------------


class TestCoordinatorCostCapHandling:
    @needs_db
    def test_coordinator_terminates_without_partial_synthesis_call(
        self, db_session: Session
    ) -> None:
        """When TaskCostBudgetExceeded is raised, coordinator returns the graceful
        stop summary and does NOT fire an extra LLM call for partial synthesis.

        The guard fires on the FIRST turn; _finish_with_partial would spend
        another LLM call. This test verifies only the initial turn call happens
        before the exception, and the result is 'stopped: cost ceiling reached'.
        """
        from kortny.agent import AgentCoordinator
        from kortny.llm.service import TaskCostBudgetExceeded

        task = _make_task(
            db_session,
            input_text="summarize market trends",
            identity_payload={"runtime_cost_ceiling_usd": "0.10"},
            total_cost_usd=Decimal("0.12"),  # already over ceiling
        )

        # FakeLLM that raises TaskCostBudgetExceeded on the first call,
        # mimicking LLMService.complete() behavior.
        class BudgetExhaustedLLM:
            def __init__(self) -> None:
                self.call_count = 0
                self.partial_synthesis_calls = 0

            def complete(
                self,
                *,
                task_id: uuid.UUID,
                messages: Sequence[ChatMessage],
                tools: Sequence[JsonSchema] = (),
                response_format: JsonObject | None = None,
                prompt_name: str | None = None,
                prompt_source: str = "code",
            ) -> Completion:
                self.call_count += 1
                raise TaskCostBudgetExceeded(
                    task_id=task_id,
                    ceiling=Decimal("0.10"),
                    current=Decimal("0.12"),
                )

        fake_llm = BudgetExhaustedLLM()

        from kortny.tools import ToolRegistry

        result = AgentCoordinator(
            session=db_session,
            llm=fake_llm,
            registry=ToolRegistry(),
        ).run(task)

        # Coordinator must return the cost-ceiling stop summary.
        assert result.result_summary == "stopped: cost ceiling reached"
        assert result.partial is True

        # Exactly one LLM call attempted (the initial turn that raised); no
        # additional call for partial synthesis.
        assert fake_llm.call_count == 1

        # The task event log must contain the budget-exceeded entry.
        from sqlalchemy import select

        events = list(
            db_session.scalars(
                select(TaskEvent)
                .where(TaskEvent.task_id == task.id)
                .order_by(TaskEvent.seq)
            )
        )
        budget_events = [
            e
            for e in events
            if e.payload.get("message") == "agent_cost_budget_exceeded"
        ]
        assert len(budget_events) == 1
        assert budget_events[0].payload["ceiling_usd"] == "0.10"
        assert budget_events[0].payload["current_usd"] == "0.12"

    @needs_db
    def test_coordinator_completion_event_reason_is_cost_ceiling_exceeded(
        self, db_session: Session
    ) -> None:
        """The agent_completed event carries reason='cost_ceiling_exceeded'."""
        from sqlalchemy import select

        from kortny.agent import AgentCoordinator
        from kortny.llm.service import TaskCostBudgetExceeded
        from kortny.tools import ToolRegistry

        task = _make_task(
            db_session,
            input_text="write a report",
            identity_payload={"runtime_cost_ceiling_usd": "0.05"},
            total_cost_usd=Decimal("0.06"),
        )

        class BudgetExhaustedLLM:
            def complete(
                self,
                *,
                task_id: uuid.UUID,
                messages: Sequence[ChatMessage],
                tools: Sequence[JsonSchema] = (),
                response_format: JsonObject | None = None,
                prompt_name: str | None = None,
                prompt_source: str = "code",
            ) -> Completion:
                raise TaskCostBudgetExceeded(
                    task_id=task_id,
                    ceiling=Decimal("0.05"),
                    current=Decimal("0.06"),
                )

        AgentCoordinator(
            session=db_session,
            llm=BudgetExhaustedLLM(),
            registry=ToolRegistry(),
        ).run(task)

        events = list(
            db_session.scalars(
                select(TaskEvent)
                .where(TaskEvent.task_id == task.id)
                .order_by(TaskEvent.seq)
            )
        )
        completed_events = [
            e for e in events if e.payload.get("message") == "agent_completed"
        ]
        assert len(completed_events) == 1
        assert completed_events[0].payload["reason"] == "cost_ceiling_exceeded"
        assert completed_events[0].payload["partial"] is True
