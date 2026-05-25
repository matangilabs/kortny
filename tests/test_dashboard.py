import os
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from decimal import Decimal

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from httpx import Response
from sqlalchemy import Engine, delete
from sqlalchemy.orm import Session

from kortny.dashboard.app import create_app
from kortny.dashboard.settings import DashboardSettings
from kortny.db.models import (
    Artifact,
    EncryptedSecret,
    Episode,
    Installation,
    LLMProvider,
    LLMUsage,
    ModelPricing,
    Task,
    TaskEvent,
    TaskEventType,
    TaskStatus,
    WorkspaceState,
)
from kortny.db.session import make_engine, make_session_factory, normalize_database_url

TEST_POSTGRES_URL = os.environ.get("KORTNY_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    TEST_POSTGRES_URL is None,
    reason="KORTNY_TEST_POSTGRES_URL is required for dashboard integration tests",
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


@pytest.fixture
def client(db_session: Session, engine: Engine) -> Iterator[tuple[TestClient, Session]]:
    assert TEST_POSTGRES_URL is not None
    session_factory = make_session_factory(engine=engine)
    settings = DashboardSettings(
        postgres_url=TEST_POSTGRES_URL,
        username="admin",
        password="secret",
        session_secret="test-dashboard-session-secret",
    )
    with TestClient(
        create_app(settings=settings, session_factory=session_factory)
    ) as test_client:
        yield test_client, db_session


def test_dashboard_redirects_unauthenticated_users_to_login(
    client: tuple[TestClient, Session],
) -> None:
    test_client, _session = client

    response = test_client.get("/", follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"] == "/login?next=%2F"


def test_dashboard_login_rejects_invalid_credentials(
    client: tuple[TestClient, Session],
) -> None:
    test_client, _session = client

    response = test_client.post(
        "/login",
        data={"username": "admin", "password": "wrong"},
        follow_redirects=False,
    )

    assert response.status_code == 401
    assert "The username or password is incorrect." in response.text


def test_dashboard_login_and_logout_flow(
    client: tuple[TestClient, Session],
) -> None:
    test_client, _session = client

    login_response = login(test_client)
    assert login_response.status_code == 303
    assert login_response.headers["location"] == "/"

    page_response = test_client.get("/")
    assert page_response.status_code == 200
    assert "Signed in as" in page_response.text
    assert "admin" in page_response.text

    logout_response = test_client.post("/logout", follow_redirects=False)
    assert logout_response.status_code == 303
    assert logout_response.headers["location"] == "/login"

    locked_response = test_client.get("/", follow_redirects=False)
    assert locked_response.status_code == 303


def test_dashboard_task_list_shows_cost_models_and_turns(
    client: tuple[TestClient, Session],
) -> None:
    test_client, session = client
    task = create_dashboard_task(session)
    login(test_client)

    response = test_client.get("/")

    assert response.status_code == 200
    assert "CCost" in response.text
    assert "UCost" in response.text
    assert "openai/gpt-5.4-mini" in response.text
    assert "$0.004200" in response.text
    assert f"/tasks/{task.id}" in response.text
    assert 'class="card"' in response.text
    assert 'class="table"' in response.text
    assert 'class="badge status-succeeded"' in response.text
    assert 'class="sidebar"' in response.text


def test_dashboard_task_detail_shows_events_usage_and_artifacts(
    client: tuple[TestClient, Session],
) -> None:
    test_client, session = client
    task = create_dashboard_task(session)
    login(test_client)

    response = test_client.get(f"/tasks/{task.id}")

    assert response.status_code == 200
    assert "Create a usage dashboard" in response.text
    assert "Done with cost summary" in response.text
    assert "status_changed" in response.text
    assert "Task created" in response.text
    assert "LLM call started" in response.text
    assert "LLM call completed" in response.text
    assert "Tool result recorded" in response.text
    assert "Raw payload" in response.text
    assert "&#34;source&#34;: &#34;test&#34;" in response.text
    assert "{&#x27;source&#x27;: &#x27;test&#x27;}" not in response.text
    assert "dashboard_report.pdf" in response.text
    assert "analysis" in response.text
    assert 'class="card metric-card"' in response.text
    assert 'class="timeline"' in response.text


def test_dashboard_usage_rollups_by_model_user_and_day(
    client: tuple[TestClient, Session],
) -> None:
    test_client, session = client
    create_dashboard_task(session)
    login(test_client)

    response = test_client.get(
        "/usage?from=2026-05-24&to=2026-05-24",
    )

    assert response.status_code == 200
    assert "openai/gpt-5.4-mini" in response.text
    assert "UCost" in response.text
    assert "2026-05-24" in response.text
    assert "$0.004200" in response.text
    assert 'class="input" type="date"' in response.text


def login(test_client: TestClient) -> Response:
    return test_client.post(
        "/login",
        data={"username": "admin", "password": "secret", "next": "/"},
        follow_redirects=False,
    )


def create_dashboard_task(session: Session) -> Task:
    installation = Installation(slack_team_id=f"T{uuid.uuid4().hex}")
    session.add(installation)
    session.flush()

    task = Task(
        installation_id=installation.id,
        slack_event_id=f"Ev{uuid.uuid4().hex}",
        slack_channel_id="CCost",
        slack_thread_ts="1779660000.000001",
        slack_message_ts="1779660000.000001",
        slack_user_id="UCost",
        input="Create a usage dashboard",
        status=TaskStatus.succeeded,
        result_summary="Done with cost summary",
        total_input_tokens=1200,
        total_output_tokens=300,
        total_cost_usd=Decimal("0.004200"),
        created_at=datetime(2026, 5, 24, 12, 0, tzinfo=UTC),
        finished_at=datetime(2026, 5, 24, 12, 1, tzinfo=UTC),
    )
    session.add(task)
    session.flush()

    session.add_all(
        [
            TaskEvent(
                task_id=task.id,
                seq=1,
                type=TaskEventType.task_created,
                payload={
                    "source": "test",
                    "slack_channel_id": task.slack_channel_id,
                    "slack_user_id": task.slack_user_id,
                    "slack_thread_ts": task.slack_thread_ts,
                    "slack_event_id": task.slack_event_id,
                },
                created_at=datetime(2026, 5, 24, 12, 0, tzinfo=UTC),
            ),
            TaskEvent(
                task_id=task.id,
                seq=2,
                type=TaskEventType.log,
                payload={
                    "message": "llm_call_started",
                    "provider": "openrouter",
                    "model": "openai/gpt-5.4-mini",
                    "model_tier": "analysis",
                    "prompt_name": "kortny.agent_coordinator.system",
                    "route_reason": "intent_classifier",
                },
                created_at=datetime(2026, 5, 24, 12, 0, 20, tzinfo=UTC),
            ),
            TaskEvent(
                task_id=task.id,
                seq=3,
                type=TaskEventType.llm_call,
                payload={
                    "message": "llm_call_completed",
                    "provider": "openrouter",
                    "model": "openai/gpt-5.4-mini",
                    "model_tier": "analysis",
                    "input_tokens": 1200,
                    "output_tokens": 300,
                    "total_tokens": 1500,
                    "cost_usd": "0.004200",
                    "latency_ms": 890,
                    "prompt_name": "kortny.agent_coordinator.system",
                },
                created_at=datetime(2026, 5, 24, 12, 0, 30, tzinfo=UTC),
            ),
            TaskEvent(
                task_id=task.id,
                seq=4,
                type=TaskEventType.tool_result,
                payload={
                    "turn": 1,
                    "tool_call_id": "call_dashboard",
                    "tool": "web_search",
                    "latency_ms": 240,
                    "artifact_count": 0,
                    "recoverable": False,
                },
                created_at=datetime(2026, 5, 24, 12, 0, 45, tzinfo=UTC),
            ),
            TaskEvent(
                task_id=task.id,
                seq=5,
                type=TaskEventType.status_changed,
                payload={"from": "running", "to": "succeeded"},
                created_at=datetime(2026, 5, 24, 12, 1, tzinfo=UTC),
            ),
            LLMUsage(
                task_id=task.id,
                provider=LLMProvider.openrouter,
                model="openai/gpt-5.4-mini",
                model_tier="analysis",
                input_tokens=1200,
                output_tokens=300,
                cost_usd=Decimal("0.004200"),
                created_at=datetime(2026, 5, 24, 12, 0, 30, tzinfo=UTC),
            ),
            Artifact(
                task_id=task.id,
                filename="dashboard_report.pdf",
                mime_type="application/pdf",
                size_bytes=12345,
                slack_file_id="Fdashboard",
                posted_at=datetime(2026, 5, 24, 12, 1, tzinfo=UTC),
                created_at=datetime(2026, 5, 24, 12, 1, tzinfo=UTC),
            ),
        ]
    )
    session.commit()
    return task


def cleanup_database(session: Session) -> None:
    for model in (
        Artifact,
        LLMUsage,
        WorkspaceState,
        Episode,
        TaskEvent,
        Task,
        ModelPricing,
        EncryptedSecret,
        Installation,
    ):
        session.execute(delete(model))
