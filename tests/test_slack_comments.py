import os
import uuid
from collections.abc import Iterator, Sequence
from decimal import Decimal

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, delete, select
from sqlalchemy.orm import Session

from kortny.config.settings import Settings
from kortny.db.models import (
    Artifact,
    EncryptedSecret,
    Installation,
    LLMProvider,
    LLMUsage,
    ModelPricing,
    Task,
    TaskEvent,
)
from kortny.db.session import make_engine, make_session_factory, normalize_database_url
from kortny.llm import ChatMessage, Completion, TokenUsage
from kortny.slack.comments import (
    LLMArtifactCommentGenerator,
    sanitize_artifact_comment,
)
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


@pytest.fixture(scope="session")
def engine() -> Iterator[Engine]:
    if TEST_POSTGRES_URL is None:
        pytest.skip("KORTNY_TEST_POSTGRES_URL is required for Slack comment tests")

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


def test_llm_artifact_comment_generator_records_usage(db_session: Session) -> None:
    task, artifact = create_task_and_artifact(db_session)
    provider = FakeProvider(
        Completion(
            content="Here's the PYPL report.",
            tool_calls=(),
            usage=TokenUsage(input_tokens=20, output_tokens=6),
            cost_usd=Decimal("0.000003"),
            model="openai/gpt-4o-mini",
        )
    )

    text = LLMArtifactCommentGenerator(
        settings=make_settings(),
        provider=provider,
        provider_name=LLMProvider.openrouter,
    ).generate(
        session=db_session,
        task=task,
        artifact=artifact,
        task_service=TaskService(db_session),
    )

    usage = db_session.scalar(select(LLMUsage).where(LLMUsage.task_id == task.id))

    assert text == "Here's the PYPL report."
    assert usage is not None
    assert usage.cost_usd == Decimal("0.000003")
    assert provider.calls[0][0][1].content is not None
    assert "generate a report about PYPL ticker" in provider.calls[0][0][1].content
    assert "pypl_report.pdf" in provider.calls[0][0][1].content


def test_sanitize_artifact_comment_normalizes_model_output() -> None:
    assert sanitize_artifact_comment('  "Here is the AAPL report."  ') == (
        "Here is the AAPL report."
    )


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


def create_task_and_artifact(session: Session) -> tuple[Task, Artifact]:
    installation = Installation(slack_team_id=f"T{uuid.uuid4().hex}")
    session.add(installation)
    session.flush()
    task = TaskService(session).create_task(
        installation_id=installation.id,
        slack_event_id=f"Ev{uuid.uuid4().hex}",
        slack_channel_id="C123",
        slack_thread_ts="1716400000.000001",
        slack_message_ts="1716400000.000001",
        slack_user_id="U123",
        input="generate a report about PYPL ticker",
    )
    task.result_summary = "Generated 1 artifact."
    artifact = Artifact(
        task_id=task.id,
        filename="pypl_report.pdf",
        mime_type="application/pdf",
        size_bytes=123,
        storage_path="/tmp/pypl_report.pdf",
    )
    session.add(artifact)
    session.flush()
    return task, artifact


def make_settings() -> Settings:
    from kortny.config.settings import LLMProvider as SettingsLLMProvider

    return Settings.model_validate(
        {
            "SLACK_BOT_TOKEN": "xoxb-test",
            "SLACK_APP_TOKEN": "xapp-test",
            "SLACK_SIGNING_SECRET": "signing-secret",
            "LLM_PROVIDER": SettingsLLMProvider.openrouter,
            "LLM_API_KEY": "openrouter-key",
            "LLM_MODEL": "openai/gpt-4o-mini",
            "POSTGRES_URL": "postgresql://kortny:kortny@localhost/kortny",
        }
    )
