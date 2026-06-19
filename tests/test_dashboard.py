import os
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from typing import cast
from urllib.parse import parse_qs as parse_url_qs
from urllib.parse import urlsplit

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from httpx import Response
from sqlalchemy import Engine, delete, func, select
from sqlalchemy.orm import Session

from kortny.composio import (
    ComposioAuthConfig,
    ComposioCatalog,
    ComposioConnectionRequest,
    ComposioTool,
    ComposioToolkit,
)
from kortny.dashboard.app import create_app
from kortny.dashboard.auth import SlackOpenIDProfile
from kortny.dashboard.settings import DashboardAuthMode, DashboardSettings
from kortny.db.models import (
    Artifact,
    ComposioConnection,
    DashboardOAuthState,
    DashboardUser,
    EncryptedSecret,
    Episode,
    Installation,
    KnowledgeGraphEdge,
    KnowledgeGraphEntity,
    KnowledgeGraphEvidence,
    LLMConfigAudit,
    LLMModelCatalog,
    LLMModelPricing,
    LLMProvider,
    LLMProviderAccount,
    LLMTierAssignment,
    LLMUsage,
    ModelPricing,
    ObserveChannelProfile,
    Schedule,
    SlackChannelMembership,
    SlackIdentity,
    Task,
    TaskEvent,
    TaskEventType,
    TaskStatus,
    WitnessOpportunityCandidate,
    WorkspaceState,
)
from kortny.db.session import make_engine, make_session_factory, normalize_database_url
from kortny.knowledge_graph.refresh import KG_CHANNEL_REFRESH_REQUESTED_MESSAGE
from kortny.llm.litellm_catalog import LiteLLMModelCandidate
from kortny.observe.assessment import (
    CHANNEL_ASSESSMENT_REQUESTED_MESSAGE,
    CHANNEL_ASSESSMENT_SUPPRESS_SLACK_POST_KEY,
)
from kortny.secrets import decrypt_secret_value, encrypt_secret_value
from tests.db_safety import assert_safe_test_database

TEST_POSTGRES_URL = os.environ.get("KORTNY_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    TEST_POSTGRES_URL is None,
    reason="KORTNY_TEST_POSTGRES_URL is required for dashboard integration tests",
)


@pytest.fixture(scope="session")
def engine() -> Iterator[Engine]:
    assert TEST_POSTGRES_URL is not None
    assert_safe_test_database(TEST_POSTGRES_URL)

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
    assert 'class="user-pill"' in page_response.text
    assert "admin" in page_response.text
    assert "Log out" in page_response.text

    logout_response = test_client.post("/logout", follow_redirects=False)
    assert logout_response.status_code == 303
    assert logout_response.headers["location"] == "/login"

    locked_response = test_client.get("/", follow_redirects=False)
    assert locked_response.status_code == 303


def test_dashboard_slack_login_start_redirects_to_slack_and_stores_state(
    db_session: Session,
    engine: Engine,
) -> None:
    assert TEST_POSTGRES_URL is not None
    session_factory = make_session_factory(engine=engine)
    settings = DashboardSettings(
        postgres_url=TEST_POSTGRES_URL,
        username="admin",
        password="secret",
        session_secret="test-dashboard-session-secret",
        auth_mode=DashboardAuthMode.hybrid,
        slack_client_id="slack-client",
        slack_client_secret="slack-secret",
        slack_redirect_uri="http://testserver/auth/slack/callback",
    )
    with TestClient(
        create_app(settings=settings, session_factory=session_factory)
    ) as test_client:
        response = test_client.get(
            "/auth/slack/start?next=/composio",
            follow_redirects=False,
        )

    assert response.status_code == 303
    redirect = urlsplit(response.headers["location"])
    assert redirect.scheme == "https"
    assert redirect.netloc == "slack.com"
    assert redirect.path == "/openid/connect/authorize"
    query = parse_url_qs(redirect.query)
    assert query["client_id"] == ["slack-client"]
    assert query["scope"] == ["openid profile email"]
    assert query["redirect_uri"] == ["http://testserver/auth/slack/callback"]
    state_value = query["state"][0]
    oauth_state = db_session.scalar(
        select(DashboardOAuthState).where(DashboardOAuthState.state == state_value)
    )
    assert oauth_state is not None
    assert oauth_state.provider == "slack"
    assert oauth_state.redirect_path == "/composio"
    assert oauth_state.used_at is None


def test_dashboard_slack_login_callback_creates_dashboard_user_and_session(
    db_session: Session,
    engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert TEST_POSTGRES_URL is not None
    installation = Installation(slack_team_id="T123", team_name="Test Team")
    oauth_state = DashboardOAuthState(
        provider="slack",
        state="state-123",
        redirect_path="/usage",
        expires_at=datetime.now(UTC) + timedelta(minutes=10),
    )
    db_session.add_all([installation, oauth_state])
    db_session.commit()
    session_factory = make_session_factory(engine=engine)
    settings = DashboardSettings(
        postgres_url=TEST_POSTGRES_URL,
        username="admin",
        password="secret",
        session_secret="test-dashboard-session-secret",
        auth_mode=DashboardAuthMode.hybrid,
        slack_client_id="slack-client",
        slack_client_secret="slack-secret",
        slack_redirect_uri="http://testserver/auth/slack/callback",
    )

    class FakeSlackOpenIDClient:
        def __init__(self, **kwargs: object) -> None:
            assert kwargs["client_id"] == "slack-client"
            assert kwargs["client_secret"] == "slack-secret"
            assert kwargs["redirect_uri"] == "http://testserver/auth/slack/callback"

        def exchange_code(self, *, code: str) -> str:
            assert code == "code-123"
            return "openid-token"

        def user_info(self, *, access_token: str) -> SlackOpenIDProfile:
            assert access_token == "openid-token"
            return SlackOpenIDProfile(
                team_id="T123",
                user_id="U123",
                display_name="Aneesh Melkot",
                email="aneesh@example.com",
                avatar_url="https://example.com/avatar.png",
                raw_json={
                    "name": "Aneesh Melkot",
                    "email": "aneesh@example.com",
                    "https://slack.com/team_id": "T123",
                    "https://slack.com/user_id": "U123",
                },
            )

    monkeypatch.setattr(
        "kortny.dashboard.app.SlackOpenIDClient",
        FakeSlackOpenIDClient,
    )
    with TestClient(
        create_app(settings=settings, session_factory=session_factory)
    ) as test_client:
        response = test_client.get(
            "/auth/slack/callback?code=code-123&state=state-123",
            follow_redirects=False,
        )
        page_response = test_client.get("/", follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"] == "/usage"
    assert page_response.status_code == 200
    assert "Aneesh Melkot" in page_response.text
    db_session.expire_all()
    dashboard_user = db_session.scalar(
        select(DashboardUser).where(DashboardUser.slack_user_id == "U123")
    )
    assert dashboard_user is not None
    assert dashboard_user.installation_id == installation.id
    assert dashboard_user.display_name == "Aneesh Melkot"
    assert dashboard_user.email == "aneesh@example.com"
    assert dashboard_user.role == "admin"
    assert dashboard_user.status == "active"
    assert dashboard_user.last_login_at is not None
    identity = db_session.scalar(
        select(SlackIdentity).where(
            SlackIdentity.kind == "user",
            SlackIdentity.slack_id == "U123",
        )
    )
    assert identity is not None
    assert identity.display_name == "Aneesh Melkot"
    db_session.refresh(oauth_state)
    assert oauth_state.used_at is not None


def test_dashboard_slack_login_callback_rejects_expired_state(
    db_session: Session,
    engine: Engine,
) -> None:
    assert TEST_POSTGRES_URL is not None
    db_session.add(
        DashboardOAuthState(
            provider="slack",
            state="expired-state",
            redirect_path="/usage",
            expires_at=datetime.now(UTC) - timedelta(minutes=1),
        )
    )
    db_session.commit()
    session_factory = make_session_factory(engine=engine)
    settings = DashboardSettings(
        postgres_url=TEST_POSTGRES_URL,
        username="admin",
        password="secret",
        session_secret="test-dashboard-session-secret",
        auth_mode=DashboardAuthMode.hybrid,
        slack_client_id="slack-client",
        slack_client_secret="slack-secret",
        slack_redirect_uri="http://testserver/auth/slack/callback",
    )
    with TestClient(
        create_app(settings=settings, session_factory=session_factory)
    ) as test_client:
        response = test_client.get(
            "/auth/slack/callback?code=code-123&state=expired-state",
            follow_redirects=False,
        )

    assert response.status_code == 303
    assert response.headers["location"].startswith("/login?")
    assert "Slack+login+state+expired." in response.headers["location"]


def test_dashboard_member_is_filtered_to_own_tasks(
    db_session: Session,
    engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert TEST_POSTGRES_URL is not None
    installation = Installation(slack_team_id="TMember", team_name="Member Team")
    db_session.add(installation)
    db_session.flush()
    own_task = create_dashboard_task(
        db_session,
        installation=installation,
        slack_channel_id="CMember",
        slack_user_id="UMember",
        slack_user_name="Member User",
        input_text="Member private task",
    )
    other_task = create_dashboard_task(
        db_session,
        installation=installation,
        slack_channel_id="COther",
        slack_user_id="UOther",
        slack_user_name="Other User",
        input_text="Other user task",
    )
    db_session.add(
        DashboardUser(
            installation_id=installation.id,
            slack_user_id="UMember",
            email="member@example.com",
            display_name="Member User",
            role="member",
            status="active",
        )
    )
    oauth_state = DashboardOAuthState(
        provider="slack",
        state="member-state",
        redirect_path="/",
        expires_at=datetime.now(UTC) + timedelta(minutes=10),
    )
    db_session.add(oauth_state)
    db_session.commit()
    session_factory = make_session_factory(engine=engine)
    settings = slack_dashboard_settings()

    class FakeSlackOpenIDClient:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def exchange_code(self, *, code: str) -> str:
            assert code == "member-code"
            return "member-token"

        def user_info(self, *, access_token: str) -> SlackOpenIDProfile:
            assert access_token == "member-token"
            return SlackOpenIDProfile(
                team_id="TMember",
                user_id="UMember",
                display_name="Member User",
                email="member@example.com",
                avatar_url=None,
                raw_json={
                    "name": "Member User",
                    "https://slack.com/team_id": "TMember",
                    "https://slack.com/user_id": "UMember",
                },
            )

    monkeypatch.setattr(
        "kortny.dashboard.app.SlackOpenIDClient",
        FakeSlackOpenIDClient,
    )
    with TestClient(
        create_app(settings=settings, session_factory=session_factory)
    ) as test_client:
        login_response = test_client.get(
            "/auth/slack/callback?code=member-code&state=member-state",
            follow_redirects=False,
        )
        member_root_response = test_client.get("/", follow_redirects=False)
        admin_tasks_response = test_client.get("/tasks", follow_redirects=False)
        member_home_response = test_client.get("/me")
        member_tasks_response = test_client.get("/me/tasks")
        own_detail_response = test_client.get(f"/me/tasks/{own_task.id}")
        other_detail_response = test_client.get(f"/me/tasks/{other_task.id}")

    assert login_response.status_code == 303
    assert login_response.headers["location"] == "/me"
    assert member_root_response.status_code == 303
    assert member_root_response.headers["location"] == "/me"
    assert admin_tasks_response.status_code == 403
    assert member_home_response.status_code == 200
    assert "Recent Work" in member_home_response.text
    assert "Usage Footprint" in member_home_response.text
    assert "Recent Artifacts" in member_home_response.text
    assert "Member private task" in member_home_response.text
    assert "Other user task" not in member_home_response.text
    assert f'href="/me/tasks/{own_task.id}"' in member_home_response.text
    assert 'href="/me/composio"' in member_home_response.text
    assert "<table" not in member_home_response.text
    assert member_tasks_response.status_code == 200
    assert "Member User" in member_tasks_response.text
    assert "Other User" not in member_tasks_response.text
    assert own_detail_response.status_code == 200
    assert "Member private task" in own_detail_response.text
    assert other_detail_response.status_code == 404


def test_dashboard_admin_can_view_and_pause_schedules(
    client: tuple[TestClient, Session],
) -> None:
    test_client, session = client
    installation = Installation(slack_team_id="TSchedules", team_name="Schedules Team")
    session.add(installation)
    session.flush()
    active = create_dashboard_schedule(
        session,
        installation=installation,
        title="Send stock market update",
        owner_slack_user_id="UScheduleOwner",
        status="active",
    )
    system = create_dashboard_schedule(
        session,
        installation=installation,
        title="Workflow Discovery",
        owner_type="system",
        owner_slack_user_id=None,
        status="paused",
    )
    session.commit()
    login(test_client)

    page_response = test_client.get("/schedules")
    system_response = test_client.get("/schedules?view=system")
    pause_response = test_client.post(
        f"/schedules/{active.id}/pause",
        data={"next": "/schedules"},
        follow_redirects=False,
    )

    assert page_response.status_code == 200
    assert "Scheduled Tasks" in page_response.text
    assert "Send stock market update" in page_response.text
    assert "Workflow Discovery" in page_response.text
    assert "Every morning at 8:00 AM Central time" in page_response.text
    assert system_response.status_code == 200
    assert "Workflow Discovery" in system_response.text
    assert "Send stock market update" not in system_response.text
    assert pause_response.status_code == 303
    assert "notice=Scheduled+task+paused." in pause_response.headers["location"]
    session.refresh(active)
    session.refresh(system)
    assert active.status == "paused"
    assert system.status == "paused"


def test_dashboard_admin_can_edit_schedule_detail(
    client: tuple[TestClient, Session],
) -> None:
    test_client, session = client
    installation = Installation(
        slack_team_id="TScheduleEdit",
        team_name="Schedule Edit Team",
    )
    session.add(installation)
    session.flush()
    schedule = create_dashboard_schedule(
        session,
        installation=installation,
        title="Send stock market update",
        owner_slack_user_id="UScheduleOwner",
        status="active",
    )
    run = create_dashboard_schedule_run(
        session,
        installation=installation,
        schedule=schedule,
        input_text="send a stock market update",
        cost_usd=Decimal("0.018500"),
        input_tokens=3200,
        output_tokens=450,
    )
    session.commit()
    login(test_client)

    detail_response = test_client.get(f"/schedules/{schedule.id}")
    edit_response = test_client.post(
        f"/schedules/{schedule.id}/edit",
        data={
            "next": f"/schedules/{schedule.id}",
            "title": "PYPL market brief",
            "schedule_text": "Every afternoon at 1:30 PM central time",
            "task_input": "send a PYPL market update",
            "planned_cost_ceiling_usd": "0.5000",
            "delivery_kind": "slack_channel",
            "delivery_slack_user_id": "UScheduleOwner",
            "delivery_slack_channel_id": "CMarketBriefs",
            "delivery_slack_thread_ts": "1780420000.000100",
            "artifact_delivery_policy": "link_artifacts",
        },
        follow_redirects=False,
    )

    assert detail_response.status_code == 200
    assert "Edit Schedule" in detail_response.text
    assert "Recent Runs" in detail_response.text
    assert "send a stock market update" in detail_response.text
    assert f'href="/tasks/{run.id}"' in detail_response.text
    assert "DM to Schedule Owner" in detail_response.text
    assert "$0.0185" in detail_response.text
    assert "3,650" in detail_response.text
    assert "Every morning at 8:00 AM Central time" in detail_response.text
    assert edit_response.status_code == 303
    assert "notice=Scheduled+task+updated." in edit_response.headers["location"]
    session.refresh(schedule)
    assert schedule.title == "PYPL market brief"
    assert schedule.task_template["input"] == "send a PYPL market update"
    assert schedule.cron_expr == "30 13 * * *"
    assert schedule.timezone == "America/Chicago"
    assert schedule.delivery_kind == "slack_channel"
    assert schedule.delivery_slack_channel_id == "CMarketBriefs"
    assert schedule.delivery_slack_thread_ts is None
    assert schedule.artifact_delivery_policy == "link_artifacts"
    assert schedule.planned_cost_ceiling_usd == Decimal("0.5000")
    assert schedule.metadata_json["dashboard_edited_by"] == "admin"
    assert schedule.metadata_json["cadence_label"] == (
        "Every afternoon at 1:30 PM Central time"
    )
    assert schedule.metadata_json["dashboard_edit_history"]


def test_dashboard_edit_preserves_existing_timing_when_cadence_is_unchanged(
    client: tuple[TestClient, Session],
) -> None:
    test_client, session = client
    installation = Installation(
        slack_team_id="TScheduleEditLegacy",
        team_name="Schedule Legacy Team",
    )
    session.add(installation)
    session.flush()
    schedule = create_dashboard_schedule(
        session,
        installation=installation,
        title="Legacy cron schedule",
        owner_slack_user_id="ULegacySchedule",
        status="active",
    )
    schedule.metadata_json = {}
    session.commit()
    login(test_client)

    edit_response = test_client.post(
        f"/schedules/{schedule.id}/edit",
        data={
            "next": f"/schedules/{schedule.id}",
            "title": "Legacy cron schedule",
            "schedule_text": "0 8 * * *",
            "task_input": "send the legacy market update",
            "planned_cost_ceiling_usd": "0.3000",
            "delivery_kind": "slack_dm",
            "delivery_slack_user_id": "ULegacySchedule",
            "delivery_slack_channel_id": "DMemberSchedule",
            "delivery_slack_thread_ts": "DMemberSchedule",
            "artifact_delivery_policy": "message_only",
        },
        follow_redirects=False,
    )

    assert edit_response.status_code == 303
    session.refresh(schedule)
    assert schedule.cron_expr == "0 8 * * *"
    assert schedule.task_template["input"] == "send the legacy market update"
    assert schedule.planned_cost_ceiling_usd == Decimal("0.3000")


def test_dashboard_member_schedules_are_scoped_to_their_user(
    db_session: Session,
    engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert TEST_POSTGRES_URL is not None
    installation = Installation(
        slack_team_id="TMemberSchedules", team_name="Member Team"
    )
    db_session.add(installation)
    db_session.flush()
    own_schedule = create_dashboard_schedule(
        db_session,
        installation=installation,
        title="Member market update",
        owner_slack_user_id="UMemberSchedule",
        status="active",
    )
    own_run = create_dashboard_schedule_run(
        db_session,
        installation=installation,
        schedule=own_schedule,
        input_text="send my morning market update",
        slack_user_name="Member Schedule",
        cost_usd=Decimal("0.009900"),
    )
    other_schedule = create_dashboard_schedule(
        db_session,
        installation=installation,
        title="Other user's schedule",
        owner_slack_user_id="UOtherSchedule",
        status="active",
    )
    system_schedule = create_dashboard_schedule(
        db_session,
        installation=installation,
        title="System heartbeat",
        owner_type="system",
        owner_slack_user_id=None,
        status="active",
    )
    dashboard_user = DashboardUser(
        installation_id=installation.id,
        slack_user_id="UMemberSchedule",
        email="member-schedule@example.com",
        display_name="Member Schedule",
        role="member",
        status="active",
    )
    oauth_state = DashboardOAuthState(
        provider="slack",
        state="member-schedule-state",
        redirect_path="/me/schedules",
        expires_at=datetime.now(UTC) + timedelta(minutes=10),
    )
    db_session.add_all([dashboard_user, oauth_state])
    db_session.commit()
    session_factory = make_session_factory(engine=engine)
    settings = slack_dashboard_settings()

    class FakeSlackOpenIDClient:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def exchange_code(self, *, code: str) -> str:
            assert code == "member-schedule-code"
            return "member-schedule-token"

        def user_info(self, *, access_token: str) -> SlackOpenIDProfile:
            assert access_token == "member-schedule-token"
            return SlackOpenIDProfile(
                team_id="TMemberSchedules",
                user_id="UMemberSchedule",
                display_name="Member Schedule",
                email="member-schedule@example.com",
                avatar_url=None,
                raw_json={
                    "name": "Member Schedule",
                    "https://slack.com/team_id": "TMemberSchedules",
                    "https://slack.com/user_id": "UMemberSchedule",
                },
            )

    monkeypatch.setattr(
        "kortny.dashboard.app.SlackOpenIDClient",
        FakeSlackOpenIDClient,
    )
    with TestClient(
        create_app(settings=settings, session_factory=session_factory)
    ) as test_client:
        login_response = test_client.get(
            "/auth/slack/callback?code=member-schedule-code&state=member-schedule-state",
            follow_redirects=False,
        )
        page_response = test_client.get("/me/schedules")
        detail_response = test_client.get(f"/me/schedules/{own_schedule.id}")
        blocked_detail = test_client.get(f"/me/schedules/{other_schedule.id}")
        admin_response = test_client.get("/schedules", follow_redirects=False)
        blocked_cancel = test_client.post(
            f"/schedules/{other_schedule.id}/cancel",
            data={"next": "/me/schedules"},
            follow_redirects=False,
        )
        edit_response = test_client.post(
            f"/me/schedules/{own_schedule.id}/edit",
            data={
                "next": f"/me/schedules/{own_schedule.id}",
                "title": "Member PM update",
                "schedule_text": "Every afternoon at 1:00 PM central time",
                "task_input": "send my market update",
                "planned_cost_ceiling_usd": "0.4000",
                "delivery_kind": "slack_dm",
                "delivery_slack_user_id": "UMemberSchedule",
                "delivery_slack_channel_id": "DMemberSchedule",
                "delivery_slack_thread_ts": "DMemberSchedule",
                "artifact_delivery_policy": "message_only",
            },
            follow_redirects=False,
        )
        cancel_response = test_client.post(
            f"/schedules/{own_schedule.id}/cancel",
            data={"next": "/me/schedules"},
            follow_redirects=False,
        )

    assert login_response.status_code == 303
    assert login_response.headers["location"] == "/me/schedules"
    assert page_response.status_code == 200
    assert "Member market update" in page_response.text
    assert "Other user's schedule" not in page_response.text
    assert "System heartbeat" not in page_response.text
    assert detail_response.status_code == 200
    assert "Recent Runs" in detail_response.text
    assert "send my morning market update" in detail_response.text
    assert f'href="/me/tasks/{own_run.id}"' in detail_response.text
    assert "$0.0099" in detail_response.text
    assert blocked_detail.status_code == 404
    assert admin_response.status_code == 403
    assert blocked_cancel.status_code == 303
    assert "notice_tone=danger" in blocked_cancel.headers["location"]
    assert edit_response.status_code == 303
    assert "notice=Scheduled+task+updated." in edit_response.headers["location"]
    assert cancel_response.status_code == 303
    assert "notice=Scheduled+task+cancelled." in cancel_response.headers["location"]
    db_session.refresh(own_schedule)
    db_session.refresh(other_schedule)
    db_session.refresh(system_schedule)
    assert own_schedule.status == "cancelled"
    assert own_schedule.next_run_at is None
    assert own_schedule.title == "Member PM update"
    assert own_schedule.task_template["input"] == "send my market update"
    assert own_schedule.planned_cost_ceiling_usd == Decimal("0.4000")
    assert other_schedule.status == "active"
    assert system_schedule.status == "active"


def test_dashboard_admin_can_manage_dashboard_user_role_and_status(
    client: tuple[TestClient, Session],
) -> None:
    test_client, session = client
    installation = Installation(slack_team_id="TAccess", team_name="Access Team")
    session.add(installation)
    session.flush()
    member = DashboardUser(
        installation_id=installation.id,
        slack_user_id="UAccessMember",
        email="member@example.com",
        display_name="Access Member",
        role="member",
        status="active",
    )
    admin = DashboardUser(
        installation_id=installation.id,
        slack_user_id="UAccessAdmin",
        email="admin@example.com",
        display_name="Access Admin",
        role="admin",
        status="active",
    )
    session.add_all([member, admin])
    session.commit()
    login(test_client)

    page_response = test_client.get("/admin/users")
    role_response = test_client.post(
        f"/admin/users/{member.id}/role",
        data={"role": "admin"},
        follow_redirects=False,
    )
    status_response = test_client.post(
        f"/admin/users/{member.id}/status",
        data={"status": "disabled"},
        follow_redirects=False,
    )

    assert page_response.status_code == 200
    assert "Access Member" in page_response.text
    assert "Access Admin" in page_response.text
    assert role_response.status_code == 303
    assert status_response.status_code == 303
    session.refresh(member)
    assert member.role == "admin"
    assert member.status == "disabled"


def test_dashboard_admin_model_config_page_shows_provider_state(
    client: tuple[TestClient, Session],
) -> None:
    test_client, session = client
    installation = Installation(slack_team_id="TModels", team_name="Models Team")
    session.add(installation)
    session.flush()
    provider = LLMProviderAccount(
        installation_id=installation.id,
        provider_kind="openrouter",
        display_name="OpenRouter env provider",
        status="active",
        health_status="unknown",
        metadata_json={"credential_source": "env", "source": "env_bootstrap"},
    )
    session.add(provider)
    session.flush()
    model = LLMModelCatalog(
        provider_account_id=provider.id,
        model_identifier="deepseek/deepseek-v4-flash",
        display_name="DeepSeek Flash",
        is_enabled=True,
        source="env_bootstrap",
    )
    session.add(model)
    session.flush()
    session.add(
        LLMTierAssignment(
            installation_id=installation.id,
            tier="cheap_fast",
            model_catalog_id=model.id,
            priority=1,
            is_active=True,
        )
    )
    session.commit()
    login(test_client)

    response = test_client.get("/admin/models")

    assert response.status_code == 200
    assert "LLM Providers" in response.text
    assert "Models Team" in response.text
    assert "OpenRouter env provider" in response.text
    assert "DeepSeek Flash" in response.text
    assert "cheap_fast" in response.text
    assert "Env managed" in response.text
    assert "Select a provider..." in response.text
    assert "Other LiteLLM provider" in response.text
    assert "Advanced connection settings" in response.text
    assert "How Routing Works" in response.text
    assert "Connect &amp; Sync Models" in response.text
    assert "Import Models" not in response.text
    assert "Model Catalog" not in response.text


def test_dashboard_admin_bootstrap_backfills_env_model_pricing(
    client: tuple[TestClient, Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_client, session = client
    set_runtime_settings_env(monkeypatch)
    monkeypatch.setenv("LLM_MODEL", "fallback/model")
    monkeypatch.setenv("LLM_CHEAP_MODEL", "deepseek/deepseek-v4-flash")
    monkeypatch.setenv("LLM_STANDARD_MODEL", "deepseek/deepseek-v4-pro")
    monkeypatch.setenv("LLM_ANALYSIS_MODEL", "anthropic/claude-sonnet-4.6")
    monkeypatch.setenv("LLM_DOCUMENT_MODEL", "openai/gpt-5.1")
    monkeypatch.setenv("LLM_HIGH_REASONING_MODEL", "anthropic/claude-opus-4.8")
    installation = Installation(
        slack_team_id="TProviderBootstrap", team_name="Bootstrap Team"
    )
    session.add(installation)
    session.commit()

    prices = {
        "deepseek/deepseek-v4-flash": ("0.010000", "0.020000"),
        "deepseek/deepseek-v4-pro": ("0.030000", "0.040000"),
        "anthropic/claude-sonnet-4.6": ("3.000000", "15.000000"),
        "openai/gpt-5.1": ("1.250000", "10.000000"),
        "anthropic/claude-opus-4.8": ("15.000000", "75.000000"),
    }

    def candidate_for_identifier(
        provider_kind: str,
        model_identifier: str,
        *,
        include_provider_catalog: bool = False,
    ) -> LiteLLMModelCandidate | None:
        assert provider_kind == "openrouter"
        assert include_provider_catalog is True
        input_price, output_price = prices[model_identifier]
        return LiteLLMModelCandidate(
            model_identifier=model_identifier,
            display_name=f"{model_identifier} display",
            provider_kind=provider_kind,
            source="provider_api",
            capabilities={"max_input_tokens": 128000},
            metadata={"litellm_provider": provider_kind},
            input_price_per_mtok=Decimal(input_price),
            output_price_per_mtok=Decimal(output_price),
        )

    monkeypatch.setattr(
        "kortny.dashboard.app.model_candidate_for_identifier",
        candidate_for_identifier,
    )
    login(test_client)

    response = test_client.post(
        "/admin/models/bootstrap",
        data={"next": "/admin/models"},
        follow_redirects=False,
    )

    assert response.status_code == 303
    session.expire_all()
    provider = session.scalar(select(LLMProviderAccount))
    assert provider is not None
    models = session.scalars(
        select(LLMModelCatalog).where(
            LLMModelCatalog.provider_account_id == provider.id
        )
    ).all()
    pricing_rows = session.scalars(
        select(LLMModelPricing).where(
            LLMModelPricing.provider_account_id == provider.id
        )
    ).all()
    assert {model.model_identifier for model in models} == set(prices)
    assert len(pricing_rows) == len(prices)
    assert {
        row.model_identifier: (
            row.input_price_per_mtok,
            row.output_price_per_mtok,
        )
        for row in pricing_rows
    }["deepseek/deepseek-v4-pro"] == (
        Decimal("0.030000"),
        Decimal("0.040000"),
    )


def test_dashboard_admin_can_update_primary_model_tier_and_audit(
    client: tuple[TestClient, Session],
) -> None:
    test_client, session = client
    installation = Installation(slack_team_id="TModelEdit", team_name="Model Edit Team")
    session.add(installation)
    session.flush()
    provider = LLMProviderAccount(
        installation_id=installation.id,
        provider_kind="openrouter",
        display_name="OpenRouter env provider",
        status="active",
        health_status="ok",
        metadata_json={"credential_source": "env", "source": "env_bootstrap"},
    )
    session.add(provider)
    session.flush()
    old_model = LLMModelCatalog(
        provider_account_id=provider.id,
        model_identifier="deepseek/deepseek-v4-flash",
        display_name="DeepSeek Flash",
        is_enabled=True,
        source="env_bootstrap",
    )
    new_model = LLMModelCatalog(
        provider_account_id=provider.id,
        model_identifier="deepseek/deepseek-v4-pro",
        display_name="DeepSeek Pro",
        is_enabled=True,
        source="manual",
    )
    session.add_all([old_model, new_model])
    session.flush()
    assignment = LLMTierAssignment(
        installation_id=installation.id,
        tier="cheap_fast",
        model_catalog_id=old_model.id,
        priority=1,
        is_active=True,
    )
    session.add(assignment)
    session.commit()
    login(test_client)

    response = test_client.post(
        "/admin/models/tiers/cheap_fast",
        data={"model_catalog_id": str(new_model.id), "next": "/admin/models"},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert "notice=" in response.headers["location"]
    session.expire_all()
    updated_assignment = session.scalar(
        select(LLMTierAssignment).where(
            LLMTierAssignment.id == assignment.id,
        )
    )
    assert updated_assignment is not None
    assert updated_assignment.model_catalog_id == new_model.id
    audit = session.scalar(
        select(LLMConfigAudit).where(
            LLMConfigAudit.installation_id == installation.id,
            LLMConfigAudit.entity_type == "llm_tier_assignment",
        )
    )
    assert audit is not None
    assert audit.action == "update"
    assert audit.previous_value is not None
    assert audit.new_value is not None
    assert audit.previous_value["model_identifier"] == "deepseek/deepseek-v4-flash"
    assert audit.new_value["model_identifier"] == "deepseek/deepseek-v4-pro"


def test_dashboard_admin_can_create_secret_backed_model_provider(
    client: tuple[TestClient, Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_client, session = client
    set_runtime_settings_env(monkeypatch)
    monkeypatch.setenv("ENCRYPTION_KEY", "dashboard-encryption-key")
    installation = Installation(
        slack_team_id="TProviderCreate", team_name="Provider Team"
    )
    session.add(installation)
    session.commit()
    candidate = LiteLLMModelCandidate(
        model_identifier="openai/test-model",
        display_name="OpenAI Test Model",
        provider_kind="openai",
        source="litellm_catalog",
        capabilities={"max_input_tokens": 128000},
        metadata={"litellm_provider": "openai"},
        input_price_per_mtok=Decimal("0.150000"),
        output_price_per_mtok=Decimal("0.600000"),
    )
    monkeypatch.setattr(
        "kortny.dashboard.app.litellm_model_candidates",
        lambda _provider_kind, *, limit=24: (candidate,),
    )
    monkeypatch.setattr(
        "kortny.dashboard.app.litellm_endpoint_model_candidates",
        lambda _provider_kind, *, api_key, api_base=None, limit=24: (),
    )
    login(test_client)

    response = test_client.post(
        "/admin/models/providers",
        data={
            "provider_kind": "openai",
            "display_name": "OpenAI team key",
            "api_key": "sk-dashboard-provider-secret",
            "base_url": "https://api.openai.com/v1",
            "api_version": "",
            "next": "/admin/models",
        },
        follow_redirects=False,
    )

    assert response.status_code == 303
    session.expire_all()
    provider = session.scalar(
        select(LLMProviderAccount).where(
            LLMProviderAccount.installation_id == installation.id,
            LLMProviderAccount.provider_kind == "openai",
        )
    )
    assert provider is not None
    assert provider.display_name == "OpenAI team key"
    assert provider.metadata_json["credential_source"] == "encrypted_secret"
    assert provider.base_url == "https://api.openai.com/v1"
    assert provider.encrypted_secret_id is not None
    model = session.scalar(
        select(LLMModelCatalog).where(
            LLMModelCatalog.provider_account_id == provider.id,
            LLMModelCatalog.model_identifier == "openai/test-model",
        )
    )
    assert model is not None
    assert model.display_name == "OpenAI Test Model"
    secret = session.get(EncryptedSecret, provider.encrypted_secret_id)
    assert secret is not None
    assert b"sk-dashboard-provider-secret" not in secret.ciphertext
    assert (
        decrypt_secret_value(
            bytes(secret.ciphertext),
            encryption_key="dashboard-encryption-key",
        )
        == "sk-dashboard-provider-secret"
    )
    audit = session.scalar(
        select(LLMConfigAudit).where(
            LLMConfigAudit.entity_type == "llm_provider_account",
            LLMConfigAudit.entity_id == str(provider.id),
        )
    )
    assert audit is not None
    assert audit.action == "create"
    assert audit.new_value is not None
    assert audit.new_value["operation"] == "create_provider"
    assert audit.new_value["imported_count"] == 1
    assert "sk-dashboard-provider-secret" not in str(audit.new_value)


def test_dashboard_admin_can_test_provider_and_import_models(
    client: tuple[TestClient, Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_client, session = client
    set_runtime_settings_env(monkeypatch)
    monkeypatch.setenv("ENCRYPTION_KEY", "dashboard-encryption-key")
    installation = Installation(
        slack_team_id="TProviderImport", team_name="Import Team"
    )
    session.add(installation)
    session.flush()
    secret = EncryptedSecret(
        installation_id=installation.id,
        secret_type="llm_provider:openrouter:test",
        ciphertext=encrypt_secret_value(
            "provider-test-key",
            encryption_key="dashboard-encryption-key",
        ),
    )
    session.add(secret)
    session.flush()
    provider = LLMProviderAccount(
        installation_id=installation.id,
        provider_kind="openrouter",
        display_name="OpenRouter dashboard provider",
        status="active",
        health_status="unknown",
        encrypted_secret_id=secret.id,
        metadata_json={"credential_source": "encrypted_secret", "source": "dashboard"},
    )
    session.add(provider)
    session.commit()

    monkeypatch.setattr(
        "kortny.dashboard.app.check_litellm_provider_key",
        lambda **_kwargs: True,
    )
    candidate = LiteLLMModelCandidate(
        model_identifier="openrouter/test-model",
        display_name="OpenRouter Test Model",
        provider_kind="openrouter",
        source="litellm_catalog",
        capabilities={"max_input_tokens": 128000},
        metadata={"litellm_provider": "openrouter"},
        input_price_per_mtok=Decimal("0.250000"),
        output_price_per_mtok=Decimal("1.000000"),
    )
    monkeypatch.setattr(
        "kortny.dashboard.app.litellm_model_candidates",
        lambda _provider_kind, *, limit=24: (candidate,),
    )
    monkeypatch.setattr(
        "kortny.dashboard.app.litellm_endpoint_model_candidates",
        lambda _provider_kind, *, api_key, api_base=None, limit=24: (),
    )
    login(test_client)

    test_response = test_client.post(
        f"/admin/models/providers/{provider.id}/test",
        data={"next": "/admin/models"},
        follow_redirects=False,
    )
    import_response = test_client.post(
        f"/admin/models/providers/{provider.id}/import-models",
        data={"limit": "24", "next": "/admin/models"},
        follow_redirects=False,
    )

    assert test_response.status_code == 303
    assert import_response.status_code == 303
    session.expire_all()
    updated_provider = session.get(LLMProviderAccount, provider.id)
    assert updated_provider is not None
    assert updated_provider.health_status == "ok"
    model = session.scalar(
        select(LLMModelCatalog).where(
            LLMModelCatalog.provider_account_id == provider.id,
            LLMModelCatalog.model_identifier == "openrouter/test-model",
        )
    )
    assert model is not None
    assert model.display_name == "OpenRouter Test Model"
    pricing = session.scalar(
        select(LLMModelPricing).where(
            LLMModelPricing.provider_account_id == provider.id,
            LLMModelPricing.model_identifier == "openrouter/test-model",
        )
    )
    assert pricing is not None
    assert pricing.input_price_per_mtok == Decimal("0.250000")


def test_dashboard_admin_imports_full_openrouter_catalog(
    client: tuple[TestClient, Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_client, session = client
    set_runtime_settings_env(monkeypatch)
    installation = Installation(
        slack_team_id="TOpenRouterFull", team_name="OpenRouter Full Team"
    )
    session.add(installation)
    session.flush()
    provider = LLMProviderAccount(
        installation_id=installation.id,
        provider_kind="openrouter",
        display_name="OpenRouter provider",
        status="active",
        health_status="ok",
        metadata_json={"credential_source": "env", "source": "env_bootstrap"},
    )
    session.add(provider)
    session.commit()
    candidates = tuple(
        LiteLLMModelCandidate(
            model_identifier=f"vendor/model-{index:03d}",
            display_name=f"Vendor Model {index:03d}",
            provider_kind="openrouter",
            source="provider_api",
            capabilities={"runtime_routable": True},
            metadata={"litellm_provider": "openrouter"},
            input_price_per_mtok=Decimal("0.010000"),
            output_price_per_mtok=Decimal("0.020000"),
        )
        for index in range(105)
    )

    def endpoint_candidates(
        _provider_kind: str,
        *,
        api_key: str,
        api_base: str | None = None,
        limit: int | None = 24,
    ) -> tuple[LiteLLMModelCandidate, ...]:
        assert _provider_kind == "openrouter"
        assert api_key
        assert api_base is None
        assert limit is None
        return candidates

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setattr(
        "kortny.dashboard.app.litellm_model_candidates",
        lambda _provider_kind, *, limit=24: (),
    )
    monkeypatch.setattr(
        "kortny.dashboard.app.litellm_endpoint_model_candidates",
        endpoint_candidates,
    )
    monkeypatch.setattr(
        "kortny.dashboard.app.model_candidate_for_identifier",
        lambda *_args, **_kwargs: None,
    )
    login(test_client)

    response = test_client.post(
        f"/admin/models/providers/{provider.id}/import-models",
        data={"next": f"/admin/models/providers/{provider.id}", "limit": "100"},
        follow_redirects=False,
    )

    assert response.status_code == 303
    session.expire_all()
    model_count = session.scalar(
        select(func.count())
        .select_from(LLMModelCatalog)
        .where(LLMModelCatalog.provider_account_id == provider.id)
    )
    pricing_count = session.scalar(
        select(func.count())
        .select_from(LLMModelPricing)
        .where(LLMModelPricing.provider_account_id == provider.id)
    )
    audit = session.scalar(
        select(LLMConfigAudit).where(
            LLMConfigAudit.installation_id == installation.id,
            LLMConfigAudit.entity_type == "llm_provider_account",
            LLMConfigAudit.entity_id == str(provider.id),
        )
    )
    assert model_count == 105
    assert pricing_count == 105
    assert audit is not None
    assert audit.new_value is not None
    assert audit.new_value["operation"] == "import_models"
    assert audit.new_value["candidate_count"] == 105


def test_dashboard_admin_import_models_skips_invalid_pricing(
    client: tuple[TestClient, Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_client, session = client
    set_runtime_settings_env(monkeypatch)
    installation = Installation(
        slack_team_id="TOpenRouterInvalidPricing",
        team_name="OpenRouter Invalid Pricing Team",
    )
    session.add(installation)
    session.flush()
    provider = LLMProviderAccount(
        installation_id=installation.id,
        provider_kind="openrouter",
        display_name="OpenRouter provider",
        status="active",
        health_status="ok",
        metadata_json={"credential_source": "env", "source": "env_bootstrap"},
    )
    session.add(provider)
    session.commit()
    invalid_candidate = LiteLLMModelCandidate(
        model_identifier="auto",
        display_name="Auto",
        provider_kind="openrouter",
        source="provider_api",
        capabilities={"runtime_routable": True},
        metadata={"litellm_provider": "openrouter"},
        input_price_per_mtok=Decimal("-1000000.000000"),
        output_price_per_mtok=Decimal("-1000000.000000"),
    )

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setattr(
        "kortny.dashboard.app.litellm_model_candidates",
        lambda _provider_kind, *, limit=24: (),
    )
    monkeypatch.setattr(
        "kortny.dashboard.app.litellm_endpoint_model_candidates",
        lambda _provider_kind, *, api_key, api_base=None, limit=24: (
            invalid_candidate,
        ),
    )
    login(test_client)

    response = test_client.post(
        f"/admin/models/providers/{provider.id}/import-models",
        data={"next": f"/admin/models/providers/{provider.id}", "limit": "100"},
        follow_redirects=False,
    )

    assert response.status_code == 303
    session.expire_all()
    model = session.scalar(
        select(LLMModelCatalog).where(
            LLMModelCatalog.provider_account_id == provider.id,
            LLMModelCatalog.model_identifier == "auto",
        )
    )
    pricing_count = session.scalar(
        select(func.count())
        .select_from(LLMModelPricing)
        .where(LLMModelPricing.provider_account_id == provider.id)
    )
    assert model is not None
    assert pricing_count == 0


def test_dashboard_admin_can_inspect_provider_detail_page(
    client: tuple[TestClient, Session],
) -> None:
    test_client, session = client
    installation = Installation(
        slack_team_id="TProviderDetail", team_name="Detail Team"
    )
    session.add(installation)
    session.flush()
    provider = LLMProviderAccount(
        installation_id=installation.id,
        provider_kind="openrouter",
        display_name="OpenRouter detail provider",
        status="active",
        health_status="ok",
        base_url="https://openrouter.ai/api/v1",
        metadata_json={"credential_source": "env", "source": "env_bootstrap"},
    )
    session.add(provider)
    session.flush()
    model = LLMModelCatalog(
        provider_account_id=provider.id,
        model_identifier="deepseek/deepseek-v4-flash",
        display_name="DeepSeek Flash",
        is_enabled=True,
        capabilities_json={
            "max_input_tokens": 128000,
            "supports_function_calling": True,
        },
        metadata_json={"litellm_metadata": {"max_output_tokens": 8192, "mode": "chat"}},
        source="env_bootstrap",
    )
    session.add(model)
    session.flush()
    assignment = LLMTierAssignment(
        installation_id=installation.id,
        tier="cheap_fast",
        model_catalog_id=model.id,
        priority=1,
        is_active=True,
    )
    pricing = LLMModelPricing(
        provider_account_id=provider.id,
        model_identifier=model.model_identifier,
        input_price_per_mtok=Decimal("0.100000"),
        output_price_per_mtok=Decimal("0.200000"),
        currency="USD",
        pricing_source="litellm_catalog",
    )
    audit = LLMConfigAudit(
        installation_id=installation.id,
        action="update",
        entity_type="llm_provider_account",
        entity_id=str(provider.id),
        new_value={"operation": "test_provider"},
    )
    session.add_all([assignment, pricing, audit])
    session.commit()
    login(test_client)

    list_response = test_client.get("/admin/models")
    detail_response = test_client.get(f"/admin/models/providers/{provider.id}")

    assert list_response.status_code == 200
    assert f"/admin/models/providers/{provider.id}" in list_response.text
    assert detail_response.status_code == 200
    assert "OpenRouter detail provider" in detail_response.text
    assert "Test Connection" in detail_response.text
    assert "Refresh Models" in detail_response.text
    assert "Update Missing Pricing" in detail_response.text
    assert (
        "All synced model rows already have pricing metadata." in detail_response.text
    )
    assert "All Providers" in detail_response.text
    assert "Routed Models" in detail_response.text
    # Catalog search is now always rendered (client-side filtering)
    assert "Search models" in detail_response.text
    assert "Needs Attention" in detail_response.text
    assert "Model Catalog" in detail_response.text
    assert "Add Manual Model" in detail_response.text
    assert "DeepSeek Flash" in detail_response.text
    assert "Cheap Fast P1" in detail_response.text
    assert "$0.10 in / $0.20 out" in detail_response.text
    assert "per 1M tokens · USD" in detail_response.text
    assert "128,000 tokens" in detail_response.text
    assert "8,192 tokens" in detail_response.text
    assert "Chat" in detail_response.text
    assert "Tools" in detail_response.text
    assert "assign-tier" in detail_response.text
    assert "https://openrouter.ai/api/v1" in detail_response.text


def test_dashboard_admin_can_update_missing_provider_model_pricing(
    client: tuple[TestClient, Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_client, session = client
    installation = Installation(
        slack_team_id="TProviderPricing", team_name="Pricing Team"
    )
    session.add(installation)
    session.flush()
    provider = LLMProviderAccount(
        installation_id=installation.id,
        provider_kind="openrouter",
        display_name="OpenRouter pricing provider",
        status="active",
        health_status="ok",
        metadata_json={"credential_source": "env", "source": "env_bootstrap"},
    )
    session.add(provider)
    session.flush()
    priced_model = LLMModelCatalog(
        provider_account_id=provider.id,
        model_identifier="openrouter/already-priced",
        display_name="Already Priced",
        is_enabled=True,
        source="provider_api",
    )
    missing_model = LLMModelCatalog(
        provider_account_id=provider.id,
        model_identifier="openrouter/missing-price",
        display_name="Missing Price",
        is_enabled=True,
        source="provider_api",
    )
    session.add_all([priced_model, missing_model])
    session.flush()
    session.add(
        LLMModelPricing(
            provider_account_id=provider.id,
            model_identifier=priced_model.model_identifier,
            input_price_per_mtok=Decimal("0.100000"),
            output_price_per_mtok=Decimal("0.200000"),
            currency="USD",
            pricing_source="litellm_catalog",
        )
    )
    session.commit()

    looked_up_models: list[str] = []

    def candidate_for_identifier(
        provider_kind: str,
        model_identifier: str,
        *,
        include_provider_catalog: bool = False,
    ) -> LiteLLMModelCandidate | None:
        looked_up_models.append(model_identifier)
        assert provider_kind == "openrouter"
        assert include_provider_catalog is True
        assert model_identifier == "openrouter/missing-price"
        return LiteLLMModelCandidate(
            model_identifier=model_identifier,
            display_name="Missing Price",
            provider_kind=provider_kind,
            source="provider_api",
            capabilities={"max_input_tokens": 128000},
            metadata={"litellm_provider": provider_kind},
            input_price_per_mtok=Decimal("0.300000"),
            output_price_per_mtok=Decimal("0.400000"),
        )

    monkeypatch.setattr(
        "kortny.dashboard.app.model_candidate_for_identifier",
        candidate_for_identifier,
    )
    login(test_client)

    detail_response = test_client.get(f"/admin/models/providers/{provider.id}")
    update_response = test_client.post(
        f"/admin/models/providers/{provider.id}/update-pricing",
        data={"next": f"/admin/models/providers/{provider.id}"},
        follow_redirects=False,
    )

    assert detail_response.status_code == 200
    assert "Looks up pricing for 1 model missing cost metadata." in detail_response.text
    assert update_response.status_code == 303
    session.expire_all()
    pricing_rows = session.scalars(
        select(LLMModelPricing).where(
            LLMModelPricing.provider_account_id == provider.id
        )
    ).all()
    assert looked_up_models == ["openrouter/missing-price"]
    assert len(pricing_rows) == 2
    pricing_by_model = {row.model_identifier: row for row in pricing_rows}
    assert pricing_by_model[
        "openrouter/already-priced"
    ].input_price_per_mtok == Decimal("0.100000")
    assert pricing_by_model["openrouter/missing-price"].input_price_per_mtok == Decimal(
        "0.300000"
    )
    audit = session.scalar(
        select(LLMConfigAudit).where(
            LLMConfigAudit.installation_id == installation.id,
            LLMConfigAudit.entity_type == "llm_provider_account",
        )
    )
    assert audit is not None
    assert audit.new_value is not None
    assert audit.new_value["operation"] == "update_missing_pricing"
    assert audit.new_value["pricing_count"] == 1


def test_dashboard_provider_detail_shows_model_catalog_search_after_ten_rows(
    client: tuple[TestClient, Session],
) -> None:
    test_client, session = client
    installation = Installation(
        slack_team_id="TProviderSearch", team_name="Search Team"
    )
    session.add(installation)
    session.flush()
    provider = LLMProviderAccount(
        installation_id=installation.id,
        provider_kind="openrouter",
        display_name="OpenRouter search provider",
        status="active",
        health_status="ok",
        metadata_json={"credential_source": "env", "source": "env_bootstrap"},
    )
    session.add(provider)
    session.flush()
    for index in range(11):
        session.add(
            LLMModelCatalog(
                provider_account_id=provider.id,
                model_identifier=f"openrouter/search-model-{index}",
                display_name=f"Search Model {index}",
                is_enabled=True,
                source="provider_api",
            )
        )
    session.commit()
    login(test_client)

    detail_response = test_client.get(f"/admin/models/providers/{provider.id}")

    assert detail_response.status_code == 200
    assert "Search models" in detail_response.text
    assert "Search models by name, id, or source" in detail_response.text
    assert "11 shown of 11 models" in detail_response.text
    assert "data-model-catalog-row" in detail_response.text


def test_dashboard_provider_detail_lazy_loads_model_catalog_page(
    client: tuple[TestClient, Session],
) -> None:
    test_client, session = client
    installation = Installation(slack_team_id="TProviderLazy", team_name="Lazy Team")
    session.add(installation)
    session.flush()
    provider = LLMProviderAccount(
        installation_id=installation.id,
        provider_kind="openrouter",
        display_name="OpenRouter lazy provider",
        status="active",
        health_status="ok",
        metadata_json={"credential_source": "env", "source": "env_bootstrap"},
    )
    session.add(provider)
    session.flush()
    for index in range(30):
        session.add(
            LLMModelCatalog(
                provider_account_id=provider.id,
                model_identifier=f"openrouter/paged-model-{index:02d}",
                display_name=f"Paged Model {index:02d}",
                is_enabled=True,
                source="provider_api",
            )
        )
    session.commit()
    login(test_client)

    detail_response = test_client.get(f"/admin/models/providers/{provider.id}")
    next_response = test_client.get(
        f"/admin/models/providers/{provider.id}/models?offset=25&limit=25"
    )
    search_response = test_client.get(
        f"/admin/models/providers/{provider.id}/models?q=Paged+Model+29&offset=0&limit=25"
    )

    assert detail_response.status_code == 200
    assert "Paged Model 00" in detail_response.text
    assert "Paged Model 29" not in detail_response.text
    assert "25 shown of 30 models" in detail_response.text
    # Pagination is infinite-scroll now; the sentinel carries the loader copy
    assert "data-model-catalog-load-more" in detail_response.text
    assert "Loading more models" in detail_response.text
    assert next_response.status_code == 200
    next_payload = next_response.json()
    assert next_payload["total_count"] == 30
    assert next_payload["shown_count"] == 30
    assert next_payload["has_more"] is False
    assert "Paged Model 29" in next_payload["html"]
    assert "assign-tier" in next_payload["html"]
    assert search_response.status_code == 200
    search_payload = search_response.json()
    assert search_payload["total_count"] == 1
    assert "Paged Model 29" in search_payload["html"]


def test_dashboard_admin_can_assign_fallback_model_tier(
    client: tuple[TestClient, Session],
) -> None:
    test_client, session = client
    installation = Installation(
        slack_team_id="TModelFallback", team_name="Fallback Team"
    )
    session.add(installation)
    session.flush()
    provider = LLMProviderAccount(
        installation_id=installation.id,
        provider_kind="openrouter",
        display_name="OpenRouter env provider",
        status="active",
        health_status="ok",
        metadata_json={"credential_source": "env", "source": "env_bootstrap"},
    )
    session.add(provider)
    session.flush()
    primary_model = LLMModelCatalog(
        provider_account_id=provider.id,
        model_identifier="deepseek/deepseek-v4-pro",
        display_name="DeepSeek Pro",
        is_enabled=True,
        source="env_bootstrap",
    )
    fallback_model = LLMModelCatalog(
        provider_account_id=provider.id,
        model_identifier="deepseek/deepseek-v4-flash",
        display_name="DeepSeek Flash",
        is_enabled=True,
        source="manual",
    )
    session.add_all([primary_model, fallback_model])
    session.flush()
    session.add(
        LLMTierAssignment(
            installation_id=installation.id,
            tier="standard",
            model_catalog_id=primary_model.id,
            priority=1,
            is_active=True,
        )
    )
    session.commit()
    login(test_client)

    response = test_client.post(
        "/admin/models/tiers/standard",
        data={
            "model_catalog_id": str(fallback_model.id),
            "priority": "2",
            "next": "/admin/models",
        },
        follow_redirects=False,
    )

    assert response.status_code == 303
    fallback_assignment = session.scalar(
        select(LLMTierAssignment).where(
            LLMTierAssignment.installation_id == installation.id,
            LLMTierAssignment.tier == "standard",
            LLMTierAssignment.priority == 2,
        )
    )
    assert fallback_assignment is not None
    assert fallback_assignment.model_catalog_id == fallback_model.id
    audit = session.scalar(
        select(LLMConfigAudit).where(
            LLMConfigAudit.installation_id == installation.id,
            LLMConfigAudit.entity_type == "llm_tier_assignment",
            LLMConfigAudit.entity_id == str(fallback_assignment.id),
        )
    )
    assert audit is not None
    assert audit.action == "create"
    assert audit.new_value is not None
    assert audit.new_value["priority"] == 2


def test_dashboard_admin_can_assign_model_tier_from_provider_catalog(
    client: tuple[TestClient, Session],
) -> None:
    test_client, session = client
    installation = Installation(
        slack_team_id="TModelCatalogAssign", team_name="Catalog Assign Team"
    )
    session.add(installation)
    session.flush()
    provider = LLMProviderAccount(
        installation_id=installation.id,
        provider_kind="openrouter",
        display_name="OpenRouter catalog provider",
        status="active",
        health_status="ok",
        metadata_json={"credential_source": "env", "source": "env_bootstrap"},
    )
    session.add(provider)
    session.flush()
    model = LLMModelCatalog(
        provider_account_id=provider.id,
        model_identifier="anthropic/claude-sonnet-4.6",
        display_name="Claude Sonnet 4.6",
        is_enabled=True,
        source="provider_api",
    )
    session.add(model)
    session.commit()
    login(test_client)

    response = test_client.post(
        f"/admin/models/catalog/{model.id}/assign-tier",
        data={
            "tier": "humanizer",
            "priority": "1",
            "next": f"/admin/models/providers/{provider.id}",
        },
        follow_redirects=False,
    )

    assert response.status_code == 303
    assignment = session.scalar(
        select(LLMTierAssignment).where(
            LLMTierAssignment.installation_id == installation.id,
            LLMTierAssignment.tier == "humanizer",
            LLMTierAssignment.priority == 1,
        )
    )
    assert assignment is not None
    assert assignment.model_catalog_id == model.id
    audit = session.scalar(
        select(LLMConfigAudit).where(
            LLMConfigAudit.installation_id == installation.id,
            LLMConfigAudit.entity_type == "llm_tier_assignment",
            LLMConfigAudit.entity_id == str(assignment.id),
        )
    )
    assert audit is not None
    assert audit.action == "create"
    assert audit.new_value is not None
    assert audit.new_value["tier"] == "humanizer"
    assert audit.new_value["model_identifier"] == model.model_identifier


def test_dashboard_renders_theme_toggle(
    client: tuple[TestClient, Session],
) -> None:
    test_client, _session = client

    login_page = test_client.get("/login")

    assert login_page.status_code == 200
    assert "data-theme-toggle" in login_page.text
    assert "kortny.theme" in login_page.text
    assert "theme.js" in login_page.text

    login(test_client)
    dashboard_page = test_client.get("/")

    assert dashboard_page.status_code == 200
    assert "data-theme-toggle" in dashboard_page.text
    assert "data-theme-toggle-value" in dashboard_page.text
    assert "theme.js" in dashboard_page.text


def test_dashboard_system_page_shows_health_and_redacted_config(
    client: tuple[TestClient, Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_client, session = client
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-system-secret")
    monkeypatch.setenv("SLACK_APP_TOKEN", "xapp-system-secret")
    monkeypatch.setenv("SLACK_SIGNING_SECRET", "system-signing-secret")
    monkeypatch.setenv("SLACK_APP_NAME", "kortny")
    monkeypatch.setenv("COMPOSIO_CATALOG_ENABLED", "false")
    monkeypatch.setenv("COMPOSIO_REQUEST_TIMEOUT_SECONDS", "0.2")
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("LLM_API_KEY", "llm-system-secret")
    monkeypatch.setenv("LLM_MODEL", "openai/gpt-5.4-mini")
    monkeypatch.setenv("LLM_ANALYSIS_MODEL", "anthropic/claude-sonnet-4.6")
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "brave-system-secret")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://phoenix:4317")
    monkeypatch.setenv(
        "POSTGRES_URL",
        "postgresql://kortny:db-system-secret@localhost/kortny",
    )
    create_dashboard_task(session)
    login(test_client)

    response = test_client.get("/system")

    assert response.status_code == 200
    assert "System" in response.text
    assert "Readiness Checks" in response.text
    assert "Runtime configuration" in response.text
    assert "Database" in response.text
    assert "Slack app" in response.text
    assert "LLM provider" in response.text
    assert "Model routing" in response.text
    assert "anthropic/claude-sonnet-4.6" in response.text
    assert "postgresql://kortny:***@localhost/kortny" in response.text
    assert "xoxb-system-secret" not in response.text
    assert "xapp-system-secret" not in response.text
    assert "system-signing-secret" not in response.text
    assert "llm-system-secret" not in response.text
    assert "brave-system-secret" not in response.text
    assert "db-system-secret" not in response.text


def test_dashboard_integrations_page_shows_providers_tools_and_redacts_secrets(
    client: tuple[TestClient, Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_client, _session = client
    set_runtime_settings_env(monkeypatch)
    monkeypatch.setenv("COMPOSIO_API_KEY", "composio-dashboard-secret")
    login(test_client)

    response = test_client.get("/integrations")

    assert response.status_code == 200
    assert "Integrations" in response.text
    assert "Providers" in response.text
    assert "Native Tool Registry" in response.text
    assert "Slack workspace" in response.text
    assert "LLM provider" in response.text
    assert "Brave Search" in response.text
    assert "PDF generation" in response.text
    assert "Workspace memory" in response.text
    assert "Composio" in response.text
    assert "Key present, catalog available" in response.text
    assert "Composio Catalog" in response.text
    assert "web_search" in response.text
    assert "pdf_generator" in response.text
    assert "slack_channel_history" in response.text
    assert "remember_fact" in response.text
    assert "BRAVE_SEARCH_API_KEY" in response.text
    assert "xoxb-dashboard-secret" not in response.text
    assert "llm-dashboard-secret" not in response.text
    assert "brave-dashboard-secret" not in response.text
    assert "composio-dashboard-secret" not in response.text
    assert 'href="/integrations" aria-current="page"' in response.text
    assert 'href="/composio"' in response.text


def test_dashboard_composio_page_renders_catalog_shell(
    client: tuple[TestClient, Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_client, _session = client
    set_runtime_settings_env(monkeypatch)
    monkeypatch.setenv("COMPOSIO_API_KEY", "composio-dashboard-secret")
    login(test_client)

    response = test_client.get("/composio")

    assert response.status_code == 200
    assert "Composio" in response.text
    assert "Integration Catalog" in response.text
    assert "Catalog not available" in response.text
    assert "Composio catalog results" in response.text
    assert 'href="/composio" aria-current="page"' in response.text
    assert "composio-dashboard-secret" not in response.text


def test_dashboard_composio_page_sorts_connected_toolkits_first(
    client: tuple[TestClient, Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_client, session = client
    task = create_dashboard_task(session)
    session.add(
        ComposioConnection(
            installation_id=task.installation_id,
            toolkit_slug="notion",
            auth_config_id="ac_notion",
            connection_request_id="ln_notion",
            connected_account_id="ca_notion",
            composio_user_id=f"slack:{task.installation_id}:UCost",
            owner_slack_user_id="UCost",
            visibility_scope_type="user",
            visibility_scope_id="UCost",
            status="active",
            display_name="Notion personal",
            metadata_json={},
        )
    )
    session.commit()
    set_runtime_settings_env(monkeypatch)
    monkeypatch.setenv("COMPOSIO_API_KEY", "composio-dashboard-secret")
    monkeypatch.setenv("COMPOSIO_CATALOG_ENABLED", "true")

    class FakeComposioClient:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def list_toolkits(
            self,
            *,
            search: str | None = None,
            limit: int = 60,
            cursor: str | None = None,
        ) -> ComposioCatalog:
            assert search is None
            assert limit == 60
            assert cursor is None
            return ComposioCatalog(
                items=(
                    _composio_toolkit(slug="github", name="GitHub"),
                    _composio_toolkit(slug="notion", name="Notion"),
                ),
                total_items=2,
                next_cursor=None,
            )

    monkeypatch.setattr("kortny.dashboard.data.ComposioClient", FakeComposioClient)
    login(test_client)

    response = test_client.get("/composio")

    assert response.status_code == 200
    assert response.text.index("<h3>Notion</h3>") < response.text.index(
        "<h3>GitHub</h3>"
    )


def test_dashboard_composio_page_pins_connected_toolkits_missing_from_page(
    client: tuple[TestClient, Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_client, session = client
    task = create_dashboard_task(session)
    session.add(
        ComposioConnection(
            installation_id=task.installation_id,
            toolkit_slug="alpha_vantage",
            auth_config_id="ac_alpha",
            connection_request_id="ln_alpha",
            connected_account_id="ca_alpha",
            composio_user_id=f"slack:{task.installation_id}:UCost",
            owner_slack_user_id="UCost",
            visibility_scope_type="user",
            visibility_scope_id="UCost",
            status="active",
            display_name="Alpha Vantage connection",
            metadata_json={},
        )
    )
    session.commit()
    set_runtime_settings_env(monkeypatch)
    monkeypatch.setenv("COMPOSIO_API_KEY", "composio-dashboard-secret")
    monkeypatch.setenv("COMPOSIO_CATALOG_ENABLED", "true")

    class FakeComposioClient:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def list_toolkits(
            self,
            *,
            search: str | None = None,
            limit: int = 60,
            cursor: str | None = None,
        ) -> ComposioCatalog:
            assert search is None
            assert limit == 60
            assert cursor is None
            return ComposioCatalog(
                items=(_composio_toolkit(slug="github", name="GitHub"),),
                total_items=1043,
                next_cursor="cursor_2",
            )

        def get_toolkit(self, slug: str) -> ComposioToolkit:
            assert slug == "alpha_vantage"
            return _composio_toolkit(slug="alpha_vantage", name="Alpha Vantage")

    monkeypatch.setattr("kortny.dashboard.data.ComposioClient", FakeComposioClient)
    login(test_client)

    response = test_client.get("/composio")

    assert response.status_code == 200
    assert response.text.index("<h3>Alpha Vantage</h3>") < response.text.index(
        "<h3>GitHub</h3>"
    )
    assert "Connected Apps" in response.text
    assert "Includes 1 connected app" in response.text


def test_dashboard_composio_page_exposes_cursor_pagination(
    client: tuple[TestClient, Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_client, _session = client
    set_runtime_settings_env(monkeypatch)
    monkeypatch.setenv("COMPOSIO_API_KEY", "composio-dashboard-secret")
    monkeypatch.setenv("COMPOSIO_CATALOG_ENABLED", "true")

    class FakeComposioClient:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def list_toolkits(
            self,
            *,
            search: str | None = None,
            limit: int = 60,
            cursor: str | None = None,
        ) -> ComposioCatalog:
            assert search == "git"
            assert limit == 24
            assert cursor == "cursor_1"
            return ComposioCatalog(
                items=(
                    _composio_toolkit(
                        slug="github",
                        name="GitHub",
                        logo_url="https://assets.composio.dev/logos/github.png",
                    ),
                ),
                total_items=120,
                next_cursor="cursor_2",
            )

    monkeypatch.setattr("kortny.dashboard.data.ComposioClient", FakeComposioClient)
    login(test_client)

    response = test_client.get("/composio?q=git&page_size=24&cursor=cursor_1")

    assert response.status_code == 200
    assert "Load more" in response.text
    assert "cursor_2" in response.text
    assert "page_size=24" in response.text
    assert "view=card" in response.text
    assert "Cards" in response.text
    assert "List" in response.text
    assert 'src="https://assets.composio.dev/logos/github.png"' in response.text


def test_dashboard_composio_detail_renders_scope_preview(
    client: tuple[TestClient, Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_client, _session = client
    set_runtime_settings_env(monkeypatch)
    monkeypatch.setenv("COMPOSIO_API_KEY", "composio-dashboard-secret")
    login(test_client)

    response = test_client.get("/composio/github")

    assert response.status_code == 200
    assert "github" in response.text
    assert "Visibility Scope" in response.text
    assert "Personal" in response.text
    assert "Channel" in response.text
    assert "Workspace" in response.text
    assert "Connect Account" in response.text
    assert "Start connection" in response.text
    assert "Connect github" in response.text
    assert "composio-dashboard-secret" not in response.text


def test_dashboard_composio_detail_lists_tool_capabilities(
    client: tuple[TestClient, Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_client, _session = client
    set_runtime_settings_env(monkeypatch)
    monkeypatch.setenv("COMPOSIO_API_KEY", "composio-dashboard-secret")
    monkeypatch.setenv("COMPOSIO_CATALOG_ENABLED", "true")

    class FakeComposioClient:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def get_toolkit(self, slug: str) -> ComposioToolkit:
            assert slug == "github"
            return _composio_toolkit(
                slug="github",
                name="GitHub",
                logo_url="https://assets.composio.dev/logos/github.png",
            )

        def list_auth_configs(
            self,
            *,
            toolkit_slug: str,
            limit: int = 20,
        ) -> tuple[ComposioAuthConfig, ...]:
            assert toolkit_slug == "github"
            assert limit == 20
            return ()

        def list_tools(
            self,
            *,
            toolkit_slug: str | None = None,
            tool_slugs: tuple[str, ...] = (),
            query: str | None = None,
            limit: int = 20,
        ) -> tuple[ComposioTool, ...]:
            assert toolkit_slug == "github"
            assert tool_slugs == ()
            assert query is None
            assert limit == 12
            return (
                ComposioTool(
                    slug="GITHUB_SEARCH_REPOSITORIES",
                    name="Search repositories",
                    description="Find repositories by keyword.",
                    toolkit_slug="github",
                    input_parameters={},
                    tags=("read", "repository"),
                    version="latest",
                ),
            )

    monkeypatch.setattr("kortny.dashboard.data.ComposioClient", FakeComposioClient)
    login(test_client)

    response = test_client.get("/composio/github")

    assert response.status_code == 200
    assert "Capabilities" in response.text
    assert "Search repositories" in response.text
    assert "Find repositories by keyword." in response.text
    assert 'src="https://assets.composio.dev/logos/github.png"' in response.text
    assert "Setup State" not in response.text
    assert "Waiting on scope gate" not in response.text


def test_dashboard_composio_detail_shows_disconnect_for_active_connection(
    client: tuple[TestClient, Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_client, session = client
    task = create_dashboard_task(session)
    connection = ComposioConnection(
        installation_id=task.installation_id,
        toolkit_slug="notion",
        auth_config_id="ac_notion",
        connection_request_id="ln_notion",
        connected_account_id="ca_notion",
        composio_user_id=f"slack:{task.installation_id}:UCost",
        owner_slack_user_id="UCost",
        visibility_scope_type="user",
        visibility_scope_id="UCost",
        status="active",
        display_name="Notion personal",
        metadata_json={},
    )
    session.add(connection)
    session.commit()
    set_runtime_settings_env(monkeypatch)
    monkeypatch.setenv("COMPOSIO_API_KEY", "composio-dashboard-secret")
    login(test_client)

    response = test_client.get("/composio/notion")

    assert response.status_code == 200
    assert "Connected Account" in response.text
    assert "Notion personal" in response.text
    assert f'action="/composio/connections/{connection.id}/scope"' in response.text
    assert "Visibility Scope" in response.text
    assert response.text.count("Visibility Scope") == 1
    assert "Change visibility to Workspace for this connected account?" in response.text
    assert 'name="visibility_scope_type" value="workspace"' in response.text
    assert "Current" in response.text
    assert 'type="radio"' not in response.text
    assert "Save Visibility" not in response.text
    assert f'action="/composio/connections/{connection.id}/disconnect"' in response.text
    assert "Connect notion" not in response.text
    assert "Scoped Connections" not in response.text


def test_dashboard_composio_scope_update_changes_visibility(
    client: tuple[TestClient, Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_client, session = client
    task = create_dashboard_task(session)
    connection = ComposioConnection(
        installation_id=task.installation_id,
        toolkit_slug="notion",
        auth_config_id="ac_notion",
        connection_request_id="ln_notion",
        connected_account_id="ca_notion",
        composio_user_id=f"slack:{task.installation_id}:UCost",
        owner_slack_user_id="UCost",
        visibility_scope_type="user",
        visibility_scope_id="UCost",
        status="active",
        display_name="Notion personal",
        metadata_json={},
    )
    session.add(connection)
    session.commit()
    connection_id = connection.id
    set_runtime_settings_env(monkeypatch)
    login(test_client)

    response = test_client.post(
        f"/composio/connections/{connection_id}/scope",
        data={
            "next": "/composio/notion",
            "visibility_scope_type": "channel",
            "channel_scope_id": "CShared",
        },
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"].startswith("/composio/notion?")
    session.expire_all()
    updated = session.get(ComposioConnection, connection_id)
    assert updated is not None
    assert updated.visibility_scope_type == "channel"
    assert updated.visibility_scope_id == "CShared"
    assert updated.metadata_json["visibility_updated_by"] == "admin"
    assert updated.metadata_json["previous_visibility_scope"] == {
        "type": "user",
        "id": "UCost",
    }


def test_dashboard_composio_disconnect_disables_connection(
    client: tuple[TestClient, Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_client, session = client
    task = create_dashboard_task(session)
    connection = ComposioConnection(
        installation_id=task.installation_id,
        toolkit_slug="notion",
        auth_config_id="ac_notion",
        connection_request_id="ln_notion",
        connected_account_id="ca_notion",
        composio_user_id=f"slack:{task.installation_id}:UCost",
        owner_slack_user_id="UCost",
        visibility_scope_type="user",
        visibility_scope_id="UCost",
        status="active",
        display_name="Notion personal",
        metadata_json={},
    )
    session.add(connection)
    session.commit()
    connection_id = connection.id
    set_runtime_settings_env(monkeypatch)
    monkeypatch.setenv("COMPOSIO_API_KEY", "composio-dashboard-secret")
    calls: list[tuple[str, bool]] = []

    class FakeComposioClient:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def set_connected_account_enabled(
            self,
            connected_account_id: str,
            *,
            enabled: bool,
        ) -> bool:
            calls.append((connected_account_id, enabled))
            return True

    monkeypatch.setattr("kortny.dashboard.app.ComposioClient", FakeComposioClient)
    login(test_client)

    response = test_client.post(
        f"/composio/connections/{connection_id}/disconnect",
        data={"next": "/composio/notion"},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"].startswith("/composio/notion?")
    assert calls == [("ca_notion", False)]
    session.expire_all()
    updated = session.get(ComposioConnection, connection_id)
    assert updated is not None
    assert updated.status == "disabled"
    assert updated.connected_account_id == "ca_notion"
    assert updated.metadata_json["disconnected_by"] == "admin"
    assert updated.metadata_json["disconnected_at"]


def test_dashboard_composio_connect_creates_pending_connection(
    client: tuple[TestClient, Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_client, session = client
    task = create_dashboard_task(session)
    set_runtime_settings_env(monkeypatch)
    monkeypatch.setenv("COMPOSIO_API_KEY", "composio-dashboard-secret")

    class FakeComposioClient:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def list_auth_configs(
            self,
            *,
            toolkit_slug: str,
            limit: int = 20,
        ) -> tuple[ComposioAuthConfig, ...]:
            assert toolkit_slug == "github"
            assert limit == 20
            return (
                ComposioAuthConfig(
                    id="ac_123",
                    name="GitHub OAuth",
                    toolkit_slug="github",
                    auth_scheme="OAUTH2",
                    is_composio_managed=True,
                    enabled=True,
                ),
            )

        def create_connect_link(
            self,
            *,
            user_id: str,
            auth_config_id: str,
            callback_url: str,
        ) -> ComposioConnectionRequest:
            assert user_id == f"slack:{task.installation_id}:UCost"
            assert auth_config_id == "ac_123"
            assert callback_url.startswith("http://testserver/composio/callback")
            return ComposioConnectionRequest(
                id="conn_req_123",
                redirect_url="https://connect.composio.dev/auth",
                status="pending",
            )

    monkeypatch.setattr("kortny.dashboard.app.ComposioClient", FakeComposioClient)
    login(test_client)

    response = test_client.post(
        "/composio/github/connect",
        data={
            "visibility_scope_type": "user",
            "display_name": "GitHub personal",
        },
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"] == "https://connect.composio.dev/auth"
    connection = session.scalar(
        select(ComposioConnection).where(
            ComposioConnection.toolkit_slug == "github",
            ComposioConnection.auth_config_id == "ac_123",
        )
    )
    assert connection is not None
    assert connection.installation_id == task.installation_id
    assert connection.owner_slack_user_id == "UCost"
    assert connection.visibility_scope_type == "user"
    assert connection.visibility_scope_id == "UCost"
    assert connection.status == "pending"
    assert connection.connection_request_id == "conn_req_123"
    assert connection.composio_user_id == f"slack:{task.installation_id}:UCost"
    assert connection.metadata_json["auth_config_source"] == "existing"
    assert connection.metadata_json["connect_link_status"] == "pending"


def test_dashboard_composio_connect_creates_custom_auth_config_for_api_key_toolkit(
    client: tuple[TestClient, Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_client, session = client
    task = create_dashboard_task(session)
    set_runtime_settings_env(monkeypatch)
    monkeypatch.setenv("COMPOSIO_API_KEY", "composio-dashboard-secret")

    class FakeComposioClient:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def list_auth_configs(
            self,
            *,
            toolkit_slug: str,
            limit: int = 20,
        ) -> tuple[ComposioAuthConfig, ...]:
            assert toolkit_slug == "firecrawl"
            assert limit == 20
            return ()

        def get_toolkit(self, slug: str) -> ComposioToolkit:
            assert slug == "firecrawl"
            return _composio_toolkit(
                slug="firecrawl",
                name="Firecrawl",
                auth_schemes=("API_KEY",),
                managed_auth_schemes=(),
            )

        def create_custom_auth_config(
            self,
            *,
            toolkit_slug: str,
            auth_scheme: str,
        ) -> ComposioAuthConfig:
            assert toolkit_slug == "firecrawl"
            assert auth_scheme == "API_KEY"
            return ComposioAuthConfig(
                id="ac_firecrawl",
                name="Firecrawl API key",
                toolkit_slug="firecrawl",
                auth_scheme="API_KEY",
                is_composio_managed=False,
                enabled=True,
            )

        def create_connect_link(
            self,
            *,
            user_id: str,
            auth_config_id: str,
            callback_url: str,
        ) -> ComposioConnectionRequest:
            assert user_id == f"slack:{task.installation_id}:UCost"
            assert auth_config_id == "ac_firecrawl"
            assert callback_url.startswith("http://testserver/composio/callback")
            return ComposioConnectionRequest(
                id="ln_firecrawl",
                redirect_url="https://connect.composio.dev/link/firecrawl",
                status="pending",
            )

    monkeypatch.setattr("kortny.dashboard.app.ComposioClient", FakeComposioClient)
    login(test_client)

    response = test_client.post(
        "/composio/firecrawl/connect",
        data={
            "visibility_scope_type": "user",
            "display_name": "Firecrawl personal",
        },
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"] == "https://connect.composio.dev/link/firecrawl"
    connection = session.scalar(
        select(ComposioConnection).where(
            ComposioConnection.toolkit_slug == "firecrawl",
            ComposioConnection.auth_config_id == "ac_firecrawl",
        )
    )
    assert connection is not None
    assert connection.status == "pending"
    assert connection.metadata_json["auth_config_source"] == "created_api_key"
    assert connection.metadata_json["connect_link_status"] == "pending"


def test_dashboard_composio_callback_marks_connection_active(
    client: tuple[TestClient, Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_client, session = client
    task = create_dashboard_task(session)
    set_runtime_settings_env(monkeypatch)
    monkeypatch.setenv("COMPOSIO_API_KEY", "composio-dashboard-secret")
    connection = ComposioConnection(
        installation_id=task.installation_id,
        toolkit_slug="github",
        auth_config_id="ac_123",
        connection_request_id="conn_req_123",
        composio_user_id=f"slack:{task.installation_id}:UCost",
        owner_slack_user_id="UCost",
        visibility_scope_type="user",
        visibility_scope_id="UCost",
        status="pending",
        display_name="GitHub personal",
        metadata_json={},
    )
    session.add(connection)
    session.commit()
    connection_id = connection.id
    login(test_client)

    response = test_client.get(
        "/composio/callback"
        f"?connection_id={connection_id}"
        "&status=success"
        "&connected_account_id=ca_123",
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"].startswith("/composio/github?")
    session.expire_all()
    updated = session.get(ComposioConnection, connection_id)
    assert updated is not None
    assert updated.status == "active"
    assert updated.connected_account_id == "ca_123"
    assert updated.metadata_json["callback"]["status"] == "success"


def test_dashboard_member_composio_connect_uses_logged_in_user_scope(
    db_session: Session,
    engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert TEST_POSTGRES_URL is not None
    installation = Installation(
        slack_team_id="TComposioMember", team_name="Member Team"
    )
    db_session.add(installation)
    db_session.flush()
    db_session.add(
        SlackIdentity(
            installation_id=installation.id,
            kind="user",
            slack_id="UMember",
            display_name="Member User",
            raw_name="Member User",
            raw_json={"id": "UMember", "profile": {"real_name": "Member User"}},
            refreshed_at=datetime(2026, 5, 24, 11, 59, tzinfo=UTC),
            last_seen_at=datetime(2026, 5, 24, 11, 59, tzinfo=UTC),
        )
    )
    db_session.add(
        DashboardUser(
            installation_id=installation.id,
            slack_user_id="UMember",
            email="member@example.com",
            display_name="Member User",
            role="member",
            status="active",
        )
    )
    db_session.add(
        DashboardOAuthState(
            provider="slack",
            state="member-composio-state",
            redirect_path="/me/composio/notion",
            expires_at=datetime.now(UTC) + timedelta(minutes=10),
        )
    )
    db_session.commit()
    session_factory = make_session_factory(engine=engine)
    settings = slack_dashboard_settings()
    set_runtime_settings_env(monkeypatch)
    monkeypatch.setenv("COMPOSIO_API_KEY", "composio-dashboard-secret")
    monkeypatch.setenv("COMPOSIO_CATALOG_ENABLED", "true")

    class FakeSlackOpenIDClient:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def exchange_code(self, *, code: str) -> str:
            assert code == "member-code"
            return "member-token"

        def user_info(self, *, access_token: str) -> SlackOpenIDProfile:
            assert access_token == "member-token"
            return SlackOpenIDProfile(
                team_id="TComposioMember",
                user_id="UMember",
                display_name="Member User",
                email="member@example.com",
                avatar_url=None,
                raw_json={
                    "name": "Member User",
                    "https://slack.com/team_id": "TComposioMember",
                    "https://slack.com/user_id": "UMember",
                },
            )

    class FakeComposioClient:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def list_toolkits(
            self,
            *,
            search: str | None = None,
            limit: int = 60,
            cursor: str | None = None,
        ) -> ComposioCatalog:
            assert search is None
            assert limit == 60
            assert cursor is None
            return ComposioCatalog(
                items=(_composio_toolkit(slug="notion", name="Notion"),),
                total_items=1,
                next_cursor=None,
            )

        def list_auth_configs(
            self,
            *,
            toolkit_slug: str,
            limit: int = 20,
        ) -> tuple[ComposioAuthConfig, ...]:
            assert toolkit_slug == "notion"
            assert limit == 20
            return (
                ComposioAuthConfig(
                    id="ac_discord",
                    name="Discord OAuth",
                    toolkit_slug="discord",
                    auth_scheme="OAUTH2",
                    is_composio_managed=True,
                    enabled=True,
                ),
            )

        def get_toolkit(self, slug: str) -> ComposioToolkit:
            assert slug == "notion"
            return _composio_toolkit(slug="notion", name="Notion")

        def create_managed_auth_config(
            self,
            *,
            toolkit_slug: str,
        ) -> ComposioAuthConfig:
            assert toolkit_slug == "notion"
            return ComposioAuthConfig(
                id="ac_member",
                name="Notion OAuth",
                toolkit_slug="notion",
                auth_scheme="OAUTH2",
                is_composio_managed=True,
                enabled=True,
            )

        def create_connect_link(
            self,
            *,
            user_id: str,
            auth_config_id: str,
            callback_url: str,
        ) -> ComposioConnectionRequest:
            assert user_id == f"slack:{installation.id}:UMember"
            assert auth_config_id == "ac_member"
            callback = urlsplit(callback_url)
            callback_query = parse_url_qs(callback.query)
            assert callback.path == "/composio/callback"
            assert callback_query["connection_id"]
            assert callback_query["connection_token"]
            return ComposioConnectionRequest(
                id="ln_member",
                redirect_url="https://connect.composio.dev/link/ln_member",
                status="pending",
                connected_account_id="ca_pending_member",
            )

    monkeypatch.setattr(
        "kortny.dashboard.app.SlackOpenIDClient",
        FakeSlackOpenIDClient,
    )
    monkeypatch.setattr("kortny.dashboard.data.ComposioClient", FakeComposioClient)
    monkeypatch.setattr("kortny.dashboard.app.ComposioClient", FakeComposioClient)

    with TestClient(
        create_app(settings=settings, session_factory=session_factory)
    ) as test_client:
        login_response = test_client.get(
            "/auth/slack/callback?code=member-code&state=member-composio-state",
            follow_redirects=False,
        )
        catalog_response = test_client.get("/me/composio")
        response = test_client.post(
            "/composio/notion/connect",
            data={
                "owner_slack_user_id": "UOther",
                "visibility_scope_type": "workspace",
                "auth_config_id": "ac_override",
                "display_name": "Notion personal",
            },
            follow_redirects=False,
        )

    assert login_response.status_code == 303
    assert login_response.headers["location"] == "/me/composio/notion"
    assert catalog_response.status_code == 200
    assert 'href="/me/composio" aria-current="page"' in catalog_response.text
    assert 'href="/me/composio/notion"' in catalog_response.text
    assert response.status_code == 303
    assert response.headers["location"] == "https://connect.composio.dev/link/ln_member"
    connection = db_session.scalar(
        select(ComposioConnection).where(ComposioConnection.toolkit_slug == "notion")
    )
    assert connection is not None
    assert connection.installation_id == installation.id
    assert connection.owner_slack_user_id == "UMember"
    assert connection.visibility_scope_type == "user"
    assert connection.visibility_scope_id == "UMember"
    assert connection.auth_config_id == "ac_member"
    assert connection.connected_account_id == "ca_pending_member"
    assert connection.connection_request_id == "ln_member"
    assert connection.composio_user_id == f"slack:{installation.id}:UMember"
    assert connection.metadata_json["dashboard_source"] == "member"
    assert connection.metadata_json["callback_token"]


def test_dashboard_member_composio_callback_returns_to_me_scope(
    db_session: Session,
    engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert TEST_POSTGRES_URL is not None
    installation = Installation(
        slack_team_id="TComposioCallback", team_name="Member Team"
    )
    db_session.add(installation)
    db_session.flush()
    db_session.add(
        DashboardUser(
            installation_id=installation.id,
            slack_user_id="UMember",
            email="member@example.com",
            display_name="Member User",
            role="member",
            status="active",
        )
    )
    db_session.add(
        DashboardOAuthState(
            provider="slack",
            state="member-callback-state",
            redirect_path="/me",
            expires_at=datetime.now(UTC) + timedelta(minutes=10),
        )
    )
    connection = ComposioConnection(
        installation_id=installation.id,
        toolkit_slug="gmail",
        auth_config_id="ac_gmail",
        connection_request_id="ln_gmail",
        composio_user_id=f"slack:{installation.id}:UMember",
        owner_slack_user_id="UMember",
        visibility_scope_type="user",
        visibility_scope_id="UMember",
        status="pending",
        display_name="Gmail personal",
        metadata_json={"callback_token": "callback-secret"},
    )
    db_session.add(connection)
    db_session.commit()
    connection_id = connection.id
    session_factory = make_session_factory(engine=engine)
    settings = slack_dashboard_settings()

    class FakeSlackOpenIDClient:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def exchange_code(self, *, code: str) -> str:
            assert code == "member-code"
            return "member-token"

        def user_info(self, *, access_token: str) -> SlackOpenIDProfile:
            assert access_token == "member-token"
            return SlackOpenIDProfile(
                team_id="TComposioCallback",
                user_id="UMember",
                display_name="Member User",
                email="member@example.com",
                avatar_url=None,
                raw_json={
                    "name": "Member User",
                    "https://slack.com/team_id": "TComposioCallback",
                    "https://slack.com/user_id": "UMember",
                },
            )

    monkeypatch.setattr(
        "kortny.dashboard.app.SlackOpenIDClient",
        FakeSlackOpenIDClient,
    )
    with TestClient(
        create_app(settings=settings, session_factory=session_factory)
    ) as test_client:
        login_response = test_client.get(
            "/auth/slack/callback?code=member-code&state=member-callback-state",
            follow_redirects=False,
        )
        response = test_client.get(
            "/composio/callback"
            f"?connection_id={connection_id}"
            "&connection_token=callback-secret"
            "&status=success"
            "&connected_account_id=ca_gmail",
            follow_redirects=False,
        )

    assert login_response.status_code == 303
    assert response.status_code == 303
    assert response.headers["location"].startswith("/me/composio/gmail?")
    db_session.expire_all()
    updated = db_session.get(ComposioConnection, connection_id)
    assert updated is not None
    assert updated.status == "active"
    assert updated.connected_account_id == "ca_gmail"
    assert updated.metadata_json["callback"]["connection_token"] == "callback-secret"


def test_dashboard_integrations_page_marks_missing_optional_search(
    client: tuple[TestClient, Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_client, _session = client
    set_runtime_settings_env(monkeypatch)
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "")
    login(test_client)

    response = test_client.get("/integrations")

    assert response.status_code == 200
    assert (
        "The web_search tool needs BRAVE_SEARCH_API_KEY before it can run."
        in response.text
    )
    assert "Add BRAVE_SEARCH_API_KEY to enable web research." in response.text
    assert "Needs setup" in response.text


def test_dashboard_homepage_renders_operator_overview(
    client: tuple[TestClient, Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_client, session = client
    set_runtime_settings_env(monkeypatch)
    create_dashboard_task(
        session,
        created_at=datetime.now(UTC) - timedelta(minutes=12),
    )
    failed_task = create_dashboard_task(
        session,
        input_text="Investigate failed overview task",
        status=TaskStatus.failed,
        cost_usd=Decimal("0.008400"),
        input_tokens=2400,
        output_tokens=600,
        created_at=datetime.now(UTC) - timedelta(minutes=5),
    )
    login(test_client)

    response = test_client.get("/")

    assert response.status_code == 200
    assert "Overview" in response.text
    assert "Operator snapshot" in response.text
    assert "Needs Review" in response.text
    assert "Top Models" in response.text
    assert "Top Users" in response.text
    assert "Daily Cost" in response.text
    assert "Task Volume" in response.text
    assert "Recent Work" in response.text
    assert "View all tasks" in response.text
    assert "Today Cost" in response.text
    assert "$0.012600" in response.text
    assert "50.0%" in response.text
    assert "Investigate failed overview task" in response.text
    assert f"/tasks/{failed_task.id}" in response.text
    assert 'href="/tasks"' in response.text
    assert 'class="usage-charts-grid"' in response.text


def test_dashboard_task_list_shows_cost_models_and_turns(
    client: tuple[TestClient, Session],
) -> None:
    test_client, session = client
    task = create_dashboard_task(session)
    session.add(
        TaskEvent(
            task_id=task.id,
            seq=7,
            type=TaskEventType.log,
            payload={
                "message": "planned_task_budget_reached",
                "runtime": "adk",
                "phase": "budget_reached",
                "branch": "research",
                "budget_type": "tool_calls",
                "limit": 8,
                "observed": 9,
            },
        )
    )
    session.commit()
    login(test_client)

    response = test_client.get("/tasks")

    assert response.status_code == 200
    assert "Tasks" in response.text
    assert "#ops-desk" in response.text
    assert "CCost" in response.text
    assert "Aneesh Melkot" in response.text
    assert "UCost" in response.text
    assert "openai/gpt-5.4-mini" in response.text
    assert "$0.004200" in response.text
    assert f"/tasks/{task.id}" in response.text
    assert 'class="card"' in response.text
    assert 'class="task-list"' in response.text
    assert 'class="task-list-item task-status-succeeded"' in response.text
    assert 'class="badge status-succeeded"' in response.text
    assert 'class="sidebar"' in response.text


def test_dashboard_task_detail_shows_events_usage_and_artifacts(
    client: tuple[TestClient, Session],
) -> None:
    test_client, session = client
    task = create_dashboard_task(session)
    session.add_all(
        [
            TaskEvent(
                task_id=task.id,
                seq=7,
                type=TaskEventType.log,
                payload={
                    "message": "adk_planned_workflow_selected",
                    "runtime": "adk",
                    "mode": "planned_parallel",
                    "planner_agent": "planned_workflow_planner",
                    "merger_agent": "planned_workflow_merger",
                    "branch_agents": [
                        "planned_research_worker",
                        "planned_workspace_worker",
                        "planned_integration_worker",
                    ],
                    "max_parallel_branches": 3,
                    "max_branch_model_calls": 3,
                    "max_branch_tool_calls": 4,
                    "max_total_tool_calls": 12,
                    "cost_ceiling_usd": "0.075",
                    "classifier_payload": {"route": "research_analysis"},
                },
                created_at=task.created_at + timedelta(seconds=31),
            ),
            TaskEvent(
                task_id=task.id,
                seq=8,
                type=TaskEventType.log,
                payload={"message": "planned_task_started", "runtime": "adk"},
                created_at=task.created_at + timedelta(seconds=32),
            ),
            TaskEvent(
                task_id=task.id,
                seq=9,
                type=TaskEventType.log,
                payload={
                    "message": "planned_task_branch_started",
                    "runtime": "adk",
                    "phase": "branch_started",
                    "branch": "research",
                    "adk_agent_name": "planned_research_worker",
                },
                created_at=task.created_at + timedelta(seconds=33),
            ),
            TaskEvent(
                task_id=task.id,
                seq=10,
                type=TaskEventType.llm_call,
                payload={
                    "message": "llm_call_completed",
                    "runtime": "adk",
                    "provider": "openrouter",
                    "model": "deepseek/deepseek-v4-flash",
                    "model_tier": "cheap_fast",
                    "prompt_name": "kortny.adk.planned_research_worker",
                    "adk_agent_name": "planned_research_worker",
                    "input_tokens": 800,
                    "output_tokens": 120,
                    "total_tokens": 920,
                    "cost_usd": "0.000050",
                },
                created_at=task.created_at + timedelta(seconds=34),
            ),
            TaskEvent(
                task_id=task.id,
                seq=11,
                type=TaskEventType.tool_call,
                payload={
                    "turn": 1,
                    "tool_call_id": "call_planned_search",
                    "tool": "composio_exa_search",
                    "runtime": "adk",
                    "adk_agent_name": "planned_research_worker",
                    "argument_keys": ["query", "numResults"],
                },
                created_at=task.created_at + timedelta(seconds=35),
            ),
            TaskEvent(
                task_id=task.id,
                seq=12,
                type=TaskEventType.tool_result,
                payload={
                    "turn": 1,
                    "tool_call_id": "call_planned_search",
                    "tool": "composio_exa_search",
                    "runtime": "adk",
                    "adk_agent_name": "planned_research_worker",
                    "latency_ms": 420,
                    "artifact_count": 0,
                    "cost_usd": "0",
                },
                created_at=task.created_at + timedelta(seconds=36),
            ),
            TaskEvent(
                task_id=task.id,
                seq=13,
                type=TaskEventType.log,
                payload={
                    "message": "planned_task_budget_reached",
                    "runtime": "adk",
                    "phase": "budget_reached",
                    "branch": "research",
                    "adk_agent_name": "planned_research_worker",
                    "budget_type": "total_tool_calls",
                    "reason": "limit_reached",
                    "limit": 12,
                    "observed": 13,
                },
                created_at=task.created_at + timedelta(seconds=37),
            ),
            TaskEvent(
                task_id=task.id,
                seq=14,
                type=TaskEventType.log,
                payload={
                    "message": "planned_task_branch_completed",
                    "runtime": "adk",
                    "phase": "branch_completed",
                    "branch": "research",
                    "adk_agent_name": "planned_research_worker",
                },
                created_at=task.created_at + timedelta(seconds=38),
            ),
            TaskEvent(
                task_id=task.id,
                seq=15,
                type=TaskEventType.log,
                payload={
                    "message": "final_response_sanitized",
                    "reason": "internal_preamble_removed",
                    "raw_chars": 1200,
                    "output_chars": 620,
                },
                created_at=task.created_at + timedelta(seconds=39),
            ),
        ]
    )
    session.commit()
    login(test_client)

    response = test_client.get(f"/tasks/{task.id}")

    assert response.status_code == 200
    assert "Create a usage dashboard" in response.text
    assert "#ops-desk" in response.text
    assert "Aneesh Melkot" in response.text
    assert "Done with cost summary" in response.text
    assert "Posted Slack Response" in response.text
    assert "Posted Slack response after humanizer" in response.text
    assert "Raw Agent Result" in response.text
    assert "status_changed" in response.text
    assert "Task created" in response.text
    assert "LLM call started" in response.text
    assert "LLM call completed" in response.text
    assert "Tool result recorded" in response.text
    assert "1,200" in response.text
    assert "1,500 tokens" in response.text
    assert "12,345" in response.text
    assert "Raw payload" in response.text
    assert "&#34;source&#34;: &#34;test&#34;" in response.text
    assert "{&#x27;source&#x27;: &#x27;test&#x27;}" not in response.text
    assert "dashboard_report.pdf" in response.text
    assert "analysis" in response.text
    assert "Planned Trace" in response.text
    assert "Legacy planned workflow" in response.text
    assert "planned_parallel" in response.text
    assert "research_analysis" in response.text
    assert "Budget Hits" in response.text
    assert "Max branches" in response.text
    assert "Branch model calls" in response.text
    assert "Total tool calls" in response.text
    assert "Raw vs Posted" in response.text
    assert "Sanitized" in response.text
    assert "Research" in response.text
    assert "composio_exa_search" in response.text
    assert "Branch not recorded" not in response.text
    assert "Planned budget reached" in response.text
    assert "A planned branch reached its budget" in response.text
    assert "tool calls" in response.text
    assert 'class="card metric-card"' in response.text
    assert 'class="timeline"' in response.text


def test_dashboard_usage_rollups_by_model_user_and_day(
    client: tuple[TestClient, Session],
) -> None:
    test_client, session = client
    create_dashboard_task(session)
    create_dashboard_task(
        session,
        slack_channel_id="DUsage",
        slack_user_id="UDmCost",
        input_text="Create a private usage dashboard",
    )
    login(test_client)

    response = test_client.get(
        "/usage?from=2026-05-24&to=2026-05-24",
    )

    assert response.status_code == 200
    assert "openai/gpt-5.4-mini" in response.text
    assert "Aneesh Melkot" in response.text
    assert "UCost" in response.text
    assert "Cost by Channel" in response.text
    assert "#ops-desk" in response.text
    assert "CCost" in response.text
    assert "UDmCost" in response.text
    assert "DUsage" not in response.text
    assert "2026-05-24" in response.text
    assert "2,400" in response.text
    assert "$0.004200" in response.text
    assert 'type="date"' in response.text
    assert 'name="from"' in response.text
    assert 'name="to"' in response.text


def test_dashboard_usage_renders_visual_analytics(
    client: tuple[TestClient, Session],
) -> None:
    test_client, session = client
    create_dashboard_task(session)
    create_dashboard_task(
        session,
        slack_user_id="UCost",
        input_text="Investigate failed dashboard task",
        status=TaskStatus.failed,
        cost_usd=Decimal("0.008400"),
        input_tokens=2400,
        output_tokens=600,
        created_at=datetime(2026, 5, 25, 12, 0, tzinfo=UTC),
    )
    login(test_client)

    response = test_client.get("/usage?from=2026-05-24&to=2026-05-25")

    assert response.status_code == 200
    assert "Daily Cost" in response.text
    assert "Task Volume" in response.text
    assert "Cost by Model" in response.text
    assert "Cost by User" in response.text
    assert "Failure Rate" in response.text
    assert "50.0%" in response.text
    assert "$0.012600" in response.text
    assert "4,500" in response.text
    assert "1 failed" in response.text
    assert 'class="usage-charts-grid"' in response.text
    # Charts are JS-rendered into canvas divs now, not CSS bar partials
    assert 'id="chart-daily-cost"' in response.text
    assert "usage-chart-canvas" in response.text


def test_dashboard_usage_shows_prompt_cache_stat(
    client: tuple[TestClient, Session],
) -> None:
    test_client, session = client
    session.add(
        ModelPricing(
            provider=LLMProvider.openrouter,
            model="anthropic/claude-opus-4-8",
            input_price_per_mtok=Decimal("5.000000"),
            output_price_per_mtok=Decimal("25.000000"),
            cache_write_multiplier=Decimal("1.25"),
            cache_read_multiplier=Decimal("0.10"),
            effective_from=datetime(2026, 1, 1, tzinfo=UTC),
        )
    )
    session.flush()
    create_dashboard_task(
        session,
        model="anthropic/claude-opus-4-8",
        input_tokens=10000,
        output_tokens=500,
        cache_creation_input_tokens=2000,
        cache_read_input_tokens=6000,
    )
    login(test_client)

    response = test_client.get("/usage?from=2026-05-24&to=2026-05-24")

    assert response.status_code == 200
    assert "Prompt Cache" in response.text
    # hit rate = 6000 / 10000 = 60.0%
    assert "60.0%" in response.text
    # savings = 6000 * 5 * (1 - 0.10) / 1e6 = 0.027 → "$0.03"
    assert "$0.03 saved" in response.text


def test_dashboard_users_list_shows_rollups_and_detail_links(
    client: tuple[TestClient, Session],
) -> None:
    test_client, session = client
    create_dashboard_task(session)
    login(test_client)

    response = test_client.get("/users?from=2026-05-24&to=2026-05-24")

    assert response.status_code == 200
    assert "User Activity" in response.text
    assert "Aneesh Melkot" in response.text
    assert "UCost" in response.text
    assert "/users/UCost?from=2026-05-24&to=2026-05-24" in response.text
    assert "1,200" in response.text
    assert "300" in response.text
    assert "$0.004200" in response.text
    assert "dashboard_report.pdf" not in response.text


def test_dashboard_user_detail_shows_tasks_usage_artifacts_and_trace_links(
    client: tuple[TestClient, Session],
) -> None:
    test_client, session = client
    task = create_dashboard_task(session)
    login(test_client)

    response = test_client.get("/users/UCost?from=2026-05-24&to=2026-05-24")

    assert response.status_code == 200
    assert "Aneesh Melkot" in response.text
    assert "Recent Tasks" in response.text
    assert "LLM Usage" in response.text
    assert "Artifacts" in response.text
    assert "Create a usage dashboard" in response.text
    assert "#ops-desk" in response.text
    assert "1,200" in response.text
    assert "$0.004200" in response.text
    assert "dashboard_report.pdf" in response.text
    assert f"/tasks/{task.id}" in response.text


def test_dashboard_users_list_shows_empty_state_for_date_filter(
    client: tuple[TestClient, Session],
) -> None:
    test_client, session = client
    create_dashboard_task(session)
    login(test_client)

    response = test_client.get("/users?from=2026-05-25&to=2026-05-25")

    assert response.status_code == 200
    assert "No user activity in this range." in response.text
    assert "Aneesh Melkot" not in response.text


def test_dashboard_user_detail_returns_404_when_date_filter_has_no_tasks(
    client: tuple[TestClient, Session],
) -> None:
    test_client, session = client
    create_dashboard_task(session)
    login(test_client)

    response = test_client.get("/users/UCost?from=2026-05-25&to=2026-05-25")

    assert response.status_code == 404


def test_dashboard_memory_page_shows_facts_and_episodes(
    client: tuple[TestClient, Session],
) -> None:
    test_client, session = client
    task = create_dashboard_task(session)
    session.add(
        WorkspaceState(
            installation_id=task.installation_id,
            scope_type="user",
            scope_id="UCost",
            key="no_auto_pdfs",
            value_json={
                "preference": "Do not generate PDFs unless explicitly requested"
            },
            value_text="Do not generate PDFs unless explicitly requested",
            status="active",
            source_kind="user_explicit",
            source_task_id=task.id,
            proposed_by="UCost",
            confirmed_by_user_id="UCost",
            confirmed_at=datetime(2026, 5, 24, 12, 1, tzinfo=UTC),
        )
    )
    session.add(
        Episode(
            installation_id=task.installation_id,
            task_id=task.id,
            channel_id="CCost",
            user_id="UCost",
            thread_ts=task.slack_thread_ts,
            summary="Remembered the user's PDF preference after confirmation.",
            tools_used=["remember_fact"],
            artifacts_created=[],
            source_refs=[{"url": "https://example.com/source"}],
            outcome="succeeded",
            created_at=datetime(2026, 5, 24, 12, 2, tzinfo=UTC),
        )
    )
    session.commit()
    login(test_client)

    response = test_client.get("/memory")

    assert response.status_code == 200
    assert "Memory" in response.text
    assert "Active Facts" in response.text
    assert "Workspace State" in response.text
    assert "Showing 1-1" in response.text
    assert "of 1 facts." in response.text
    assert "All scopes" in response.text
    assert "Recently updated" in response.text
    assert "no_auto_pdfs" in response.text
    assert "Do not generate PDFs unless explicitly requested" in response.text
    assert "Aneesh Melkot" in response.text
    assert "Audit" in response.text
    assert "Forget" in response.text
    assert "Supersede" in response.text
    assert f"/tasks/{task.id}" in response.text
    assert (
        "Remembered the user&#39;s PDF preference after confirmation."
        not in response.text
    )

    filtered_facts = test_client.get(
        "/memory?q=Aneesh&scope=user&status=active&sort=key_asc"
    )

    assert filtered_facts.status_code == 200
    assert "Showing 1-1" in filtered_facts.text
    assert "of 1 facts." in filtered_facts.text
    assert "no_auto_pdfs" in filtered_facts.text

    episodes_response = test_client.get(
        "/memory?view=episodes&q=ops-desk&outcome=succeeded&sort=created_asc"
    )

    assert episodes_response.status_code == 200
    assert "Episodes" in episodes_response.text
    assert "Task memories retained for follow-up context" in episodes_response.text
    assert "Showing 1-1" in episodes_response.text
    assert "of 1 episodes." in episodes_response.text
    assert (
        "Remembered the user&#39;s PDF preference after confirmation."
        in episodes_response.text
    )
    assert "1 tool" in episodes_response.text
    assert "1 source" in episodes_response.text
    assert "#ops-desk" in episodes_response.text
    assert "Aneesh Melkot" in episodes_response.text
    assert f"/tasks/{task.id}" in episodes_response.text


def test_dashboard_knowledge_graph_page_shows_entities_relationships_and_evidence(
    client: tuple[TestClient, Session],
) -> None:
    test_client, session = client
    task = create_dashboard_task(session)
    channel = KnowledgeGraphEntity(
        installation_id=task.installation_id,
        entity_type="channel",
        canonical_key="slack_channel:CCost",
        display_name="#ops-desk",
        visibility_scope_type="channel",
        visibility_scope_id="CCost",
        source_type="slack_authoritative",
        lifecycle_state="active",
        confidence_score=Decimal("0.950"),
    )
    project = KnowledgeGraphEntity(
        installation_id=task.installation_id,
        entity_type="project",
        canonical_key="project:operator-console",
        display_name="Operator Console",
        visibility_scope_type="channel",
        visibility_scope_id="CCost",
        source_type="onboarding_scan",
        lifecycle_state="candidate",
        confidence_score=Decimal("0.720"),
        attrs_json={"theme": "admin dashboard"},
    )
    session.add_all([channel, project])
    session.flush()
    edge = KnowledgeGraphEdge(
        installation_id=task.installation_id,
        source_entity_id=channel.id,
        target_entity_id=project.id,
        relationship_type="relates_to",
        visibility_scope_type="channel",
        visibility_scope_id="CCost",
        source_type="onboarding_scan",
        lifecycle_state="active",
        confidence_score=Decimal("0.800"),
    )
    session.add(edge)
    session.flush()
    session.add_all(
        [
            KnowledgeGraphEvidence(
                installation_id=task.installation_id,
                target_kind="entity",
                target_id=channel.id,
                source_type="slack_authoritative",
                source_task_id=task.id,
                source_slack_channel_id="CCost",
                extracted_by="test",
                raw_snippet="Channel membership recorded for ops-desk.",
                confidence_score=Decimal("0.950"),
            ),
            KnowledgeGraphEvidence(
                installation_id=task.installation_id,
                target_kind="entity",
                target_id=project.id,
                source_type="onboarding_scan",
                source_task_id=task.id,
                source_slack_channel_id="CCost",
                extracted_by="test",
                raw_snippet="Operators discussed the dashboard knowledge graph.",
                confidence_score=Decimal("0.720"),
            ),
            KnowledgeGraphEvidence(
                installation_id=task.installation_id,
                target_kind="edge",
                target_id=edge.id,
                source_type="onboarding_scan",
                source_task_id=task.id,
                source_slack_channel_id="CCost",
                extracted_by="test",
                raw_snippet="The channel is related to the operator console project.",
                confidence_score=Decimal("0.800"),
            ),
        ]
    )
    session.commit()
    login(test_client)

    response = test_client.get("/knowledge-graph")

    assert response.status_code == 200
    assert "Knowledge Graph" in response.text
    assert "Graph Map" in response.text
    assert "2 nodes / 1 links" in response.text
    assert 'class="kg-data-node"' in response.text
    assert 'class="kg-data-edge"' in response.text
    assert 'data-source="' in response.text
    assert 'data-target="' in response.text
    assert "Entities" in response.text
    assert "slack_channel:CCost" in response.text
    assert "#ops-desk" in response.text
    assert "project:operator-console" in response.text
    assert "candidate" in response.text
    assert "Extracted" in response.text
    assert "auto" in response.text
    assert "Channel membership recorded for ops-desk." in response.text
    assert "runtime eligible" in response.text

    relationship_response = test_client.get(
        "/knowledge-graph?view=relationships&q=operator-console"
    )

    assert relationship_response.status_code == 200
    assert "Relationships" in relationship_response.text
    assert "Graph Map" in relationship_response.text
    assert "relates_to" in relationship_response.text
    assert "slack_channel:CCost" in relationship_response.text
    assert "project:operator-console" in relationship_response.text
    assert "The channel is related to the operator console project." in (
        relationship_response.text
    )

    candidate_response = test_client.get("/knowledge-graph?state=candidate")

    assert candidate_response.status_code == 200
    assert "project:operator-console" in candidate_response.text
    assert "slack_channel:CCost" not in candidate_response.text


def test_dashboard_witness_page_shows_candidates_and_filters(
    client: tuple[TestClient, Session],
) -> None:
    test_client, session = client
    task = create_dashboard_task(session)
    candidate = WitnessOpportunityCandidate(
        installation_id=task.installation_id,
        channel_id=task.slack_channel_id,
        target_slack_user_id=task.slack_user_id,
        visibility_scope_type="channel",
        visibility_scope_id=task.slack_channel_id,
        candidate_type="data_quality_issue",
        title="Data quality watch: flag unresolved placeholders",
        summary="Kortny should watch this channel for unresolved CSV placeholders.",
        suggested_action="Check recent files for unresolved placeholders.",
        suggested_message=(
            "I noticed this channel has recurring CSV placeholder issues. "
            "Want me to monitor the next report?"
        ),
        evidence_json=[
            {
                "type": "semantic_evidence",
                "snippet": "Report had {TICKER} placeholders in a recent upload.",
            },
            {
                "type": "channel_profile",
                "summary": "Daily blotter channel with generated reports.",
            },
        ],
        source_type="channel_profile",
        source_id="profile-123",
        source_task_id=task.id,
        dedupe_key="witness-test-dedupe-key",
        confidence_score=Decimal("0.770"),
        confidence_reason="Channel assessment found recurring report quality issues.",
        status="candidate",
        reinforcement_count=3,
        first_observed_at=datetime(2026, 5, 28, tzinfo=UTC),
        metadata_json={"channel_name": "ops-desk"},
    )
    other_candidate = WitnessOpportunityCandidate(
        installation_id=task.installation_id,
        channel_id=task.slack_channel_id,
        target_slack_user_id=None,
        visibility_scope_type="channel",
        visibility_scope_id=task.slack_channel_id,
        candidate_type="recurring_check",
        title="Recurring check: morning project scan",
        summary="A recurring project scan may help this channel.",
        suggested_action="Create a recurring project scan.",
        suggested_message="Want me to check this every morning?",
        evidence_json=[],
        source_type="manual",
        source_id="manual-1",
        dedupe_key="witness-test-recurring-key",
        confidence_score=Decimal("0.600"),
        status="dismissed",
        metadata_json={},
    )
    session.add_all([candidate, other_candidate])
    session.commit()
    login(test_client)

    response = test_client.get("/witness")

    assert response.status_code == 200
    assert "Witness" in response.text
    assert "Opportunity Candidates" in response.text
    assert "Data quality watch: flag unresolved placeholders" in response.text
    assert "77% confidence" in response.text
    # HIG-197: reinforcement count + first-observed date are visible on the row.
    assert "Observed 3x" in response.text
    assert "First seen" in response.text
    assert "#ops-desk" in response.text
    assert "Aneesh Melkot" in response.text
    assert "Report had {TICKER} placeholders" in response.text
    assert "Daily blotter channel with generated reports." in response.text
    assert f"/tasks/{task.id}" in response.text
    assert "witness-test-dedupe-key" in response.text
    assert "Run scan" in response.text
    assert "Review safe actions" in response.text
    assert "Mark useful" in response.text
    assert "Snooze" in response.text
    assert "Dismiss" in response.text
    assert "Send DM" not in response.text
    assert "Recurring check: morning project scan" not in response.text
    assert 'href="/witness"' in response.text
    assert "LLM Providers" in response.text

    filtered_response = test_client.get("/witness?status=all&type=recurring_check")

    assert filtered_response.status_code == 200
    assert "Recurring check: morning project scan" in filtered_response.text
    assert (
        "Data quality watch: flag unresolved placeholders" not in filtered_response.text
    )


def test_dashboard_witness_candidate_lifecycle_action_updates_row(
    client: tuple[TestClient, Session],
) -> None:
    test_client, session = client
    task = create_dashboard_task(session)
    candidate = WitnessOpportunityCandidate(
        installation_id=task.installation_id,
        channel_id=task.slack_channel_id,
        target_slack_user_id=task.slack_user_id,
        visibility_scope_type="channel",
        visibility_scope_id=task.slack_channel_id,
        candidate_type="project_status_gap",
        title="Project status follow-up",
        summary="Kortny should watch for missing project status updates.",
        suggested_action="Watch project status gaps.",
        suggested_message="I can watch for missing project updates here.",
        evidence_json=[],
        source_type="manual",
        source_id="manual-status-gap",
        source_task_id=task.id,
        dedupe_key="witness-dashboard-action-key",
        confidence_score=Decimal("0.700"),
        confidence_reason="Manual test candidate.",
        status="candidate",
        metadata_json={},
    )
    session.add(candidate)
    session.commit()
    login(test_client)

    response = test_client.post(
        f"/witness/candidates/{candidate.id}/snooze",
        data={"next": "/witness?status=all&type=project_status_gap"},
        follow_redirects=False,
    )

    assert response.status_code == 303
    location = response.headers["location"]
    assert location.startswith("/witness?status=all&type=project_status_gap")
    assert "notice=" in location
    session.expire_all()
    refreshed = session.get(WitnessOpportunityCandidate, candidate.id)
    assert refreshed is not None
    assert refreshed.status == "cooldown"
    assert refreshed.cooldown_until is not None
    assert refreshed.feedback_json["last_action"]["action"] == "snoozed"


def test_dashboard_witness_run_scan_uses_selected_workspace(
    client: tuple[TestClient, Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_client, session = client
    task = create_dashboard_task(session)
    session.commit()
    set_runtime_settings_env(monkeypatch)
    calls: list[dict[str, object]] = []

    class FakeWitnessRunner:
        def __init__(
            self,
            runner_session: Session,
            *,
            settings: object,
            runner_id: str,
        ) -> None:
            calls.append(
                {
                    "session": runner_session,
                    "settings": settings,
                    "runner_id": runner_id,
                }
            )

        def run_once(self, **kwargs: object) -> object:
            calls.append(kwargs)
            return SimpleNamespace(
                projections=(
                    SimpleNamespace(
                        created_count=2,
                        updated_count=1,
                        skipped_count=0,
                    ),
                ),
                deliveries=(),
            )

    monkeypatch.setattr("kortny.dashboard.app.WitnessRunner", FakeWitnessRunner)
    login(test_client)

    response = test_client.post(
        "/witness/run",
        data={"next": "/witness?status=all"},
        follow_redirects=False,
    )

    assert response.status_code == 303
    location = response.headers["location"]
    assert location.startswith("/witness?status=all")
    query = parse_url_qs(urlsplit(location).query)
    assert query["notice_tone"] == ["success"]
    assert "created 2 candidates" in query["notice"][0]
    assert len(calls) == 2
    init_call, run_call = calls
    assert isinstance(init_call["session"], Session)
    assert str(init_call["runner_id"]).startswith("dashboard:")
    assert run_call["installation_id"] == task.installation_id
    assert run_call["profile_limit"] == 10
    assert run_call["delivery_limit"] == 0
    assert run_call["deliver_private"] is False
    assert run_call["use_advisory_lock"] is False
    assert run_call["min_scan_interval"] == timedelta(seconds=0)

    notice_response = test_client.get(location)
    assert notice_response.status_code == 200
    assert "Witness scan complete" in notice_response.text


def test_dashboard_witness_run_autopilot_uses_selected_workspace(
    client: tuple[TestClient, Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_client, session = client
    task = create_dashboard_task(session)
    session.commit()
    set_runtime_settings_env(monkeypatch)
    calls: list[dict[str, object]] = []

    class FakeWitnessAutopilot:
        def __init__(
            self,
            autopilot_session: Session,
            *,
            settings: object,
            actor_id: str,
        ) -> None:
            calls.append(
                {
                    "session": autopilot_session,
                    "settings": settings,
                    "actor_id": actor_id,
                }
            )

        def run_once(self, **kwargs: object) -> object:
            calls.append(kwargs)
            return SimpleNamespace(
                reviewed_count=3,
                executed_count=2,
                deferred_count=1,
                dismissed_count=0,
            )

    monkeypatch.setattr("kortny.dashboard.app.WitnessAutopilot", FakeWitnessAutopilot)
    login(test_client)

    response = test_client.post(
        "/witness/autopilot",
        data={"next": "/witness?status=candidate"},
        follow_redirects=False,
    )

    assert response.status_code == 303
    location = response.headers["location"]
    assert location.startswith("/witness?status=candidate")
    query = parse_url_qs(urlsplit(location).query)
    assert query["notice_tone"] == ["success"]
    assert "reviewed 3" in query["notice"][0]
    assert "started 2 proactive tasks" in query["notice"][0]
    assert len(calls) == 2
    init_call, run_call = calls
    assert isinstance(init_call["session"], Session)
    assert str(init_call["actor_id"]).startswith("dashboard:")
    assert run_call["installation_id"] == task.installation_id
    assert run_call["limit"] == 1
    assert run_call["min_confidence"] == Decimal("0.600")


def test_dashboard_knowledge_graph_refresh_queues_channel_assessment_tasks(
    client: tuple[TestClient, Session],
) -> None:
    test_client, session = client
    task = create_dashboard_task(session)
    membership = SlackChannelMembership(
        installation_id=task.installation_id,
        channel_id="CRefresh",
        channel_name="graph-refresh",
        channel_type="channel",
        membership_status="active",
        discovered_via="message_observation",
        added_by_user_id="UCost",
        onboarding_status="posted",
        onboarding_message_ts="1779900000.000000",
        metadata_json={},
    )
    session.add(membership)
    session.commit()
    login(test_client)

    page_response = test_client.get("/knowledge-graph")

    assert page_response.status_code == 200
    assert "<span>Channels</span>" in page_response.text
    assert "profiled" in page_response.text
    assert "Refresh Channel Graph" in page_response.text

    response = test_client.post(
        "/knowledge-graph/refresh",
        data={"next": "/knowledge-graph"},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert "Queued+1+graph+refresh+assessment" in response.headers["location"]
    assert "trusted+Slack+graph+facts" in response.headers["location"]
    session.refresh(membership)
    queued_task_id = uuid.UUID(membership.metadata_json["assessment_task_id"])
    queued_task = session.get(Task, queued_task_id)
    assert queued_task is not None
    assert queued_task.status == TaskStatus.pending
    assert queued_task.slack_channel_id == "CRefresh"
    assert queued_task.slack_thread_ts == "1779900000.000000"
    assert queued_task.slack_user_id == "dashboard:admin"
    assert queued_task.identity_kind == "synthetic"
    assert queued_task.identity_key is not None
    assert queued_task.identity_key.startswith(
        "synthetic:dashboard_knowledge_graph_refresh:"
    )
    assert membership.metadata_json["assessment_source"] == (
        "dashboard_knowledge_graph_refresh"
    )
    assert session.scalar(
        select(TaskEvent).where(
            TaskEvent.task_id == queued_task_id,
            TaskEvent.payload["message"].as_string()
            == CHANNEL_ASSESSMENT_REQUESTED_MESSAGE,
            TaskEvent.payload[CHANNEL_ASSESSMENT_SUPPRESS_SLACK_POST_KEY]
            .as_boolean()
            .is_(True),
        )
    )
    assert session.scalar(
        select(TaskEvent).where(
            TaskEvent.task_id == queued_task_id,
            TaskEvent.payload["message"].as_string()
            == KG_CHANNEL_REFRESH_REQUESTED_MESSAGE,
        )
    )
    channel_entity = session.scalar(
        select(KnowledgeGraphEntity).where(
            KnowledgeGraphEntity.installation_id == task.installation_id,
            KnowledgeGraphEntity.canonical_key == "slack_channel:CRefresh",
        )
    )
    assert channel_entity is not None
    assert channel_entity.display_name == "#graph-refresh"
    assert channel_entity.source_type == "slack_authoritative"
    assert channel_entity.lifecycle_state == "active"

    duplicate_response = test_client.post(
        "/knowledge-graph/refresh",
        data={"next": "/knowledge-graph"},
        follow_redirects=False,
    )

    assert duplicate_response.status_code == 303
    assert (
        "Queued+0+graph+refresh+assessments" in duplicate_response.headers["location"]
    )
    assert session.scalar(select(func.count()).select_from(Task)) == 2


def test_dashboard_knowledge_graph_review_actions_preserve_evidence(
    client: tuple[TestClient, Session],
) -> None:
    test_client, session = client
    task = create_dashboard_task(session)
    channel = KnowledgeGraphEntity(
        installation_id=task.installation_id,
        entity_type="channel",
        canonical_key="slack_channel:CCost",
        display_name="#ops-desk",
        visibility_scope_type="channel",
        visibility_scope_id="CCost",
        source_type="slack_authoritative",
        lifecycle_state="active",
        confidence_score=Decimal("0.950"),
    )
    project = KnowledgeGraphEntity(
        installation_id=task.installation_id,
        entity_type="project",
        canonical_key="project:operator-console",
        display_name="Operator Console",
        visibility_scope_type="channel",
        visibility_scope_id="CCost",
        source_type="onboarding_scan",
        lifecycle_state="candidate",
        confidence_score=Decimal("0.720"),
    )
    session.add_all([channel, project])
    session.flush()
    edge = KnowledgeGraphEdge(
        installation_id=task.installation_id,
        source_entity_id=channel.id,
        target_entity_id=project.id,
        relationship_type="relates_to",
        visibility_scope_type="channel",
        visibility_scope_id="CCost",
        source_type="onboarding_scan",
        lifecycle_state="candidate",
        confidence_score=Decimal("0.800"),
    )
    session.add(edge)
    session.flush()
    session.add_all(
        [
            KnowledgeGraphEvidence(
                installation_id=task.installation_id,
                target_kind="entity",
                target_id=project.id,
                source_type="onboarding_scan",
                source_task_id=task.id,
                source_slack_channel_id="CCost",
                extracted_by="test",
                raw_snippet="Operators discussed the dashboard knowledge graph.",
                confidence_score=Decimal("0.720"),
            ),
            KnowledgeGraphEvidence(
                installation_id=task.installation_id,
                target_kind="edge",
                target_id=edge.id,
                source_type="onboarding_scan",
                source_task_id=task.id,
                source_slack_channel_id="CCost",
                extracted_by="test",
                raw_snippet="The channel is related to the operator console project.",
                confidence_score=Decimal("0.800"),
            ),
        ]
    )
    session.commit()
    project_id = project.id
    edge_id = edge.id
    login(test_client)

    confirm_entity_response = test_client.post(
        f"/knowledge-graph/entities/{project_id}/confirm",
        data={"next": "/knowledge-graph?state=candidate"},
        follow_redirects=False,
    )

    assert confirm_entity_response.status_code == 303
    assert "Graph+entity+confirmed" in confirm_entity_response.headers["location"]
    session.refresh(project)
    assert project.lifecycle_state == "confirmed"
    assert project.last_reinforced_at is not None
    assert project.reinforcement_count == 1
    assert session.scalar(
        select(KnowledgeGraphEvidence).where(
            KnowledgeGraphEvidence.target_kind == "entity",
            KnowledgeGraphEvidence.target_id == project_id,
            KnowledgeGraphEvidence.source_type == "admin_import",
        )
    )

    confirm_edge_response = test_client.post(
        f"/knowledge-graph/edges/{edge_id}/confirm",
        data={"next": "/knowledge-graph?view=relationships"},
        follow_redirects=False,
    )

    assert confirm_edge_response.status_code == 303
    assert "Graph+relationship+confirmed" in confirm_edge_response.headers["location"]
    session.refresh(edge)
    assert edge.lifecycle_state == "confirmed"
    assert edge.last_reinforced_at is not None
    assert edge.reinforcement_count == 1

    archive_response = test_client.post(
        f"/knowledge-graph/entities/{project_id}/archive",
        data={"next": "/knowledge-graph?state=confirmed"},
        follow_redirects=False,
    )

    assert archive_response.status_code == 303
    assert "Graph+entity+archived" in archive_response.headers["location"]
    session.refresh(project)
    session.refresh(edge)
    assert project.lifecycle_state == "archived"
    assert project.is_current is False
    assert edge.lifecycle_state == "archived"
    assert edge.is_current is False


def test_dashboard_memory_forget_preserves_audit_and_hides_from_active_view(
    client: tuple[TestClient, Session],
) -> None:
    test_client, session = client
    task = create_dashboard_task(session)
    fact = create_dashboard_memory_fact(
        session,
        task,
        key="pdf_style",
        value_text="Use concise PDF summaries",
    )
    login(test_client)

    response = test_client.post(
        f"/memory/facts/{fact.id}/forget",
        data={"next": "/memory?q=pdf_style"},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"].startswith("/memory?q=pdf_style")
    assert "notice=Memory+fact+forgotten." in response.headers["location"]

    session.refresh(fact)
    assert fact.status == "forgotten"
    assert fact.forgotten_by_user_id == "dashboard:admin"
    assert fact.forgotten_at is not None
    events = list(
        session.scalars(
            select(TaskEvent)
            .where(TaskEvent.task_id == task.id)
            .order_by(TaskEvent.seq)
        )
    )
    assert any(
        event.payload.get("message") == "workspace_state_fact_forgotten"
        and event.payload.get("workspace_state_id") == str(fact.id)
        for event in events
    )

    active_view = test_client.get("/memory?q=pdf_style")
    forgotten_view = test_client.get("/memory?q=pdf_style&status=forgotten")

    assert "No facts match these filters." in active_view.text
    assert "Use concise PDF summaries" in forgotten_view.text
    assert "dashboard:admin" in forgotten_view.text


def test_dashboard_memory_supersede_replaces_active_fact_and_links_history(
    client: tuple[TestClient, Session],
) -> None:
    test_client, session = client
    task = create_dashboard_task(session)
    fact = create_dashboard_memory_fact(
        session,
        task,
        key="briefing_style",
        value_text="Use short briefing notes",
    )
    login(test_client)

    response = test_client.post(
        f"/memory/facts/{fact.id}/supersede",
        data={
            "next": "/memory?q=briefing_style",
            "value_text": "Use executive-ready briefing notes with open questions.",
        },
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert "notice=Memory+fact+superseded." in response.headers["location"]

    session.refresh(fact)
    replacement = session.scalar(
        select(WorkspaceState).where(
            WorkspaceState.installation_id == task.installation_id,
            WorkspaceState.scope_type == "user",
            WorkspaceState.scope_id == "UCost",
            WorkspaceState.key == "briefing_style",
            WorkspaceState.status == "active",
        )
    )
    assert replacement is not None
    assert replacement.id != fact.id
    assert replacement.value_json == {
        "text": "Use executive-ready briefing notes with open questions."
    }
    assert (
        replacement.value_text
        == "Use executive-ready briefing notes with open questions."
    )
    assert replacement.confirmed_by_user_id == "dashboard:admin"
    assert replacement.proposed_by == "dashboard:admin"
    assert fact.status == "superseded"
    assert fact.superseded_by_id == replacement.id
    assert fact.superseded_at is not None

    events = list(
        session.scalars(
            select(TaskEvent)
            .where(TaskEvent.task_id == task.id)
            .order_by(TaskEvent.seq)
        )
    )
    assert any(
        event.payload.get("message") == "workspace_state_dashboard_fact_superseded"
        and event.payload.get("workspace_state_id") == str(fact.id)
        and event.payload.get("replacement_workspace_state_id") == str(replacement.id)
        for event in events
    )
    assert replacement.source_event_id in {event.id for event in events}

    active_view = test_client.get("/memory?q=briefing_style")
    superseded_view = test_client.get("/memory?q=briefing_style&status=superseded")

    assert "Use executive-ready briefing notes with open questions." in active_view.text
    assert "Use short briefing notes" not in active_view.text
    assert "Use short briefing notes" in superseded_view.text
    assert str(replacement.id) in superseded_view.text


def create_dashboard_channel_profile(
    session: Session,
    *,
    channel_id: str = "CStyle",
    profile_json: dict[str, object] | None = None,
) -> ObserveChannelProfile:
    installation = Installation(slack_team_id=f"T{uuid.uuid4().hex}")
    session.add(installation)
    session.flush()
    profile = ObserveChannelProfile(
        installation_id=installation.id,
        channel_id=channel_id,
        profile_status="active",
        profile_version=1,
        summary="Launch coordination channel.",
        profile_json=profile_json or {},
    )
    session.add(profile)
    session.flush()
    return profile


DASHBOARD_STYLE_CARD = {
    "formality": "casual",
    "brevity": "terse",
    "emoji_culture": "heavy",
    "punctuation": "relaxed",
    "common_phrases": ["ship it"],
    "threading_norm": "threads_heavy",
    "notes": "Quick informal replies.",
}


def test_dashboard_consolidation_page_renders_style_cards(
    client: tuple[TestClient, Session],
) -> None:
    test_client, session = client
    create_dashboard_channel_profile(
        session,
        profile_json={
            "style_card": dict(DASHBOARD_STYLE_CARD),
            "style_card_updated_at": "2026-06-10T03:00:00+00:00",
            "pinned_style": "Keep it short.",
        },
    )
    session.commit()
    login(test_client)

    response = test_client.get("/consolidation")

    assert response.status_code == 200
    assert "Channel Voice" in response.text
    assert "casual" in response.text
    assert "terse" in response.text
    assert "Quick informal replies." in response.text
    assert "Keep it short." in response.text


def test_dashboard_style_card_reset_action(
    client: tuple[TestClient, Session],
) -> None:
    test_client, session = client
    profile = create_dashboard_channel_profile(
        session,
        profile_json={
            "style_card": dict(DASHBOARD_STYLE_CARD),
            "style_card_updated_at": "2026-06-10T03:00:00+00:00",
            "style_card_input_sha": "abc123",
        },
    )
    session.commit()
    login(test_client)

    response = test_client.post(
        f"/consolidation/style-cards/{profile.id}/reset",
        data={"next": "/consolidation"},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"].startswith("/consolidation")
    assert "Style+card+reset" in response.headers["location"]
    session.refresh(profile)
    assert "style_card" not in profile.profile_json
    assert "style_card_updated_at" not in profile.profile_json
    assert "style_card_input_sha" not in profile.profile_json
    assert profile.profile_json["style_card_reset_by"] == "dashboard:admin"


def test_dashboard_style_card_pin_and_clear_actions(
    client: tuple[TestClient, Session],
) -> None:
    test_client, session = client
    profile = create_dashboard_channel_profile(
        session,
        profile_json={"style_card": dict(DASHBOARD_STYLE_CARD)},
    )
    session.commit()
    login(test_client)

    pin_response = test_client.post(
        f"/consolidation/style-cards/{profile.id}/pin",
        data={"next": "/consolidation", "pinned_style": "Always boardroom formal."},
        follow_redirects=False,
    )

    assert pin_response.status_code == 303
    assert "Pinned+style+saved" in pin_response.headers["location"]
    session.refresh(profile)
    assert profile.profile_json["pinned_style"] == "Always boardroom formal."
    assert profile.profile_json["pinned_style_set_by"] == "dashboard:admin"
    # The derived card survives a pin; the pin only overrides the voice line.
    assert profile.profile_json["style_card"]["formality"] == "casual"

    clear_response = test_client.post(
        f"/consolidation/style-cards/{profile.id}/pin",
        data={"next": "/consolidation", "pinned_style": ""},
        follow_redirects=False,
    )

    assert clear_response.status_code == 303
    assert "Pinned+style+cleared" in clear_response.headers["location"]
    session.refresh(profile)
    assert "pinned_style" not in profile.profile_json


def login(test_client: TestClient) -> Response:
    return cast(
        Response,
        test_client.post(
            "/login",
            data={"username": "admin", "password": "secret", "next": "/"},
            follow_redirects=False,
        ),
    )


def slack_dashboard_settings() -> DashboardSettings:
    assert TEST_POSTGRES_URL is not None
    return DashboardSettings(
        postgres_url=TEST_POSTGRES_URL,
        username="admin",
        password="secret",
        session_secret="test-dashboard-session-secret",
        auth_mode=DashboardAuthMode.hybrid,
        slack_client_id="slack-client",
        slack_client_secret="slack-secret",
        slack_redirect_uri="http://testserver/auth/slack/callback",
    )


def set_runtime_settings_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-dashboard-secret")
    monkeypatch.setenv("SLACK_APP_TOKEN", "xapp-dashboard-secret")
    monkeypatch.setenv("SLACK_SIGNING_SECRET", "dashboard-signing-secret")
    monkeypatch.setenv("SLACK_APP_NAME", "kortny")
    monkeypatch.setenv("COMPOSIO_API_KEY", "composio-dashboard-secret")
    monkeypatch.setenv("COMPOSIO_CATALOG_ENABLED", "false")
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("LLM_API_KEY", "llm-dashboard-secret")
    monkeypatch.setenv("LLM_MODEL", "openai/gpt-5.4-mini")
    monkeypatch.setenv("LLM_ANALYSIS_MODEL", "anthropic/claude-sonnet-4.6")
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "brave-dashboard-secret")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://phoenix:4317")
    monkeypatch.setenv(
        "POSTGRES_URL",
        "postgresql://kortny:db-dashboard-secret@localhost/kortny",
    )


def _composio_toolkit(
    *,
    slug: str,
    name: str,
    auth_schemes: tuple[str, ...] = ("oauth2",),
    managed_auth_schemes: tuple[str, ...] = ("oauth2",),
    logo_url: str | None = None,
) -> ComposioToolkit:
    return ComposioToolkit(
        slug=slug,
        name=name,
        description=f"{name} toolkit",
        categories=(),
        auth_schemes=auth_schemes,
        managed_auth_schemes=managed_auth_schemes,
        tools_count=4,
        triggers_count=1,
        logo_url=logo_url,
        app_url=None,
        auth_guide_url=None,
        base_url=None,
        enabled=True,
        no_auth=False,
        is_local_toolkit=False,
    )


def create_dashboard_task(
    session: Session,
    *,
    installation: Installation | None = None,
    slack_channel_id: str = "CCost",
    slack_user_id: str = "UCost",
    slack_user_name: str = "Aneesh Melkot",
    input_text: str = "Create a usage dashboard",
    created_at: datetime | None = None,
    status: TaskStatus = TaskStatus.succeeded,
    cost_usd: Decimal | None = None,
    input_tokens: int = 1200,
    output_tokens: int = 300,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
    model: str = "openai/gpt-5.4-mini",
) -> Task:
    task_created_at = created_at or datetime(2026, 5, 24, 12, 0, tzinfo=UTC)
    task_finished_at = task_created_at + timedelta(minutes=1)
    task_cost_usd = cost_usd if cost_usd is not None else Decimal("0.004200")
    if installation is None:
        installation = Installation(slack_team_id=f"T{uuid.uuid4().hex}")
        session.add(installation)
        session.flush()

    task = Task(
        installation_id=installation.id,
        slack_event_id=f"Ev{uuid.uuid4().hex}",
        slack_channel_id=slack_channel_id,
        slack_thread_ts="1779660000.000001",
        slack_message_ts="1779660000.000001",
        slack_user_id=slack_user_id,
        input=input_text,
        status=status,
        result_summary="Done with cost summary",
        total_input_tokens=input_tokens,
        total_output_tokens=output_tokens,
        total_cost_usd=task_cost_usd,
        created_at=task_created_at,
        finished_at=task_finished_at,
    )
    session.add(task)
    session.flush()

    session.add_all(
        [
            SlackIdentity(
                installation_id=installation.id,
                kind="channel",
                slack_id=slack_channel_id,
                display_name="#ops-desk",
                raw_name="ops-desk",
                raw_json={"id": slack_channel_id, "name": "ops-desk"},
                refreshed_at=datetime(2026, 5, 24, 11, 59, tzinfo=UTC),
                last_seen_at=datetime(2026, 5, 24, 11, 59, tzinfo=UTC),
            ),
            SlackIdentity(
                installation_id=installation.id,
                kind="user",
                slack_id=slack_user_id,
                display_name=slack_user_name,
                raw_name=slack_user_name,
                raw_json={
                    "id": slack_user_id,
                    "profile": {"real_name": slack_user_name},
                },
                refreshed_at=datetime(2026, 5, 24, 11, 59, tzinfo=UTC),
                last_seen_at=datetime(2026, 5, 24, 11, 59, tzinfo=UTC),
            ),
        ]
    )

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
                created_at=task_created_at,
            ),
            TaskEvent(
                task_id=task.id,
                seq=2,
                type=TaskEventType.log,
                payload={
                    "message": "llm_call_started",
                    "provider": "openrouter",
                    "model": model,
                    "model_tier": "analysis",
                    "prompt_name": "kortny.agent_coordinator.system",
                    "route_reason": "intent_classifier",
                },
                created_at=task_created_at + timedelta(seconds=20),
            ),
            TaskEvent(
                task_id=task.id,
                seq=3,
                type=TaskEventType.llm_call,
                payload={
                    "message": "llm_call_completed",
                    "provider": "openrouter",
                    "model": model,
                    "model_tier": "analysis",
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                    "cost_usd": str(task_cost_usd),
                    "latency_ms": 890,
                    "prompt_name": "kortny.agent_coordinator.system",
                },
                created_at=task_created_at + timedelta(seconds=30),
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
                created_at=task_created_at + timedelta(seconds=45),
            ),
            TaskEvent(
                task_id=task.id,
                seq=5,
                type=TaskEventType.message_posted,
                payload={
                    "channel": task.slack_channel_id,
                    "thread_ts": task.slack_thread_ts,
                    "message_ts": "1779660060.000001",
                    "purpose": "result",
                    "text": "Posted Slack response after humanizer",
                },
                created_at=task_finished_at - timedelta(seconds=5),
            ),
            TaskEvent(
                task_id=task.id,
                seq=6,
                type=TaskEventType.status_changed,
                payload={"from": "running", "to": status.value},
                created_at=task_finished_at,
            ),
            LLMUsage(
                task_id=task.id,
                provider=LLMProvider.openrouter,
                model=model,
                model_tier="analysis",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_creation_input_tokens=cache_creation_input_tokens,
                cache_read_input_tokens=cache_read_input_tokens,
                cost_usd=task_cost_usd,
                created_at=task_created_at + timedelta(seconds=30),
            ),
            Artifact(
                task_id=task.id,
                filename="dashboard_report.pdf",
                mime_type="application/pdf",
                size_bytes=12345,
                slack_file_id="Fdashboard",
                posted_at=task_finished_at,
                created_at=task_finished_at,
            ),
        ]
    )
    session.commit()
    return task


def create_dashboard_schedule(
    session: Session,
    *,
    installation: Installation,
    title: str,
    owner_type: str = "user",
    owner_slack_user_id: str | None = "UScheduleOwner",
    status: str = "active",
) -> Schedule:
    schedule = Schedule(
        installation_id=installation.id,
        owner_type=owner_type,
        owner_slack_user_id=owner_slack_user_id,
        title=title,
        spec_kind="cron",
        cron_expr="0 8 * * *",
        timezone="America/Chicago",
        next_run_at=datetime(2026, 6, 5, 13, 0, tzinfo=UTC)
        if status != "cancelled"
        else None,
        last_run_at=None,
        catchup_policy="skip",
        catchup_window_seconds=300,
        overlap_policy="skip",
        status=status,
        delivery_kind="slack_dm",
        delivery_slack_user_id=owner_slack_user_id or "USystem",
        delivery_slack_channel_id="DMemberSchedule",
        delivery_slack_thread_ts="DMemberSchedule",
        artifact_delivery_policy="message_only",
        task_template={
            "input": "send a stock market update",
            "slack_channel_id": "DMemberSchedule",
            "slack_user_id": owner_slack_user_id or "USystem",
            "slack_thread_ts": "DMemberSchedule",
            "delivery_surface": "dm",
            "artifact_delivery_policy": "message_only",
        },
        planned_cost_ceiling_usd=Decimal("0.2500"),
        created_by_slack_user_id=owner_slack_user_id,
        metadata_json={
            "cadence_label": "Every morning at 8:00 AM Central time",
            "delivery_surface": "dm",
            "confirmation_required": False,
        },
    )
    session.add(schedule)
    session.flush()
    return schedule


def create_dashboard_schedule_run(
    session: Session,
    *,
    installation: Installation,
    schedule: Schedule,
    input_text: str,
    slack_user_name: str = "Schedule Owner",
    created_at: datetime | None = None,
    status: TaskStatus = TaskStatus.succeeded,
    cost_usd: Decimal = Decimal("0.012300"),
    input_tokens: int = 900,
    output_tokens: int = 120,
) -> Task:
    run_created_at = created_at or datetime(2026, 6, 5, 13, 0, tzinfo=UTC)
    run_finished_at = (
        run_created_at + timedelta(minutes=2)
        if status in {TaskStatus.succeeded, TaskStatus.failed, TaskStatus.cancelled}
        else None
    )
    user_id = (
        schedule.delivery_slack_user_id or schedule.owner_slack_user_id or "USchedule"
    )
    channel_id = schedule.delivery_slack_channel_id or "DSchedule"
    thread_ts = schedule.delivery_slack_thread_ts or channel_id
    task = Task(
        installation_id=installation.id,
        slack_event_id=f"EvScheduleRun{uuid.uuid4().hex}",
        slack_channel_id=channel_id,
        slack_thread_ts=thread_ts,
        slack_message_ts=None,
        slack_user_id=user_id,
        input=input_text,
        status=status,
        result_summary="Scheduled run completed",
        total_input_tokens=input_tokens,
        total_output_tokens=output_tokens,
        total_cost_usd=cost_usd,
        identity_kind="scheduled",
        identity_key=f"scheduled:{schedule.id}:{run_created_at.isoformat()}",
        identity_payload={
            "schedule_id": str(schedule.id),
            "fire_time": run_created_at.isoformat(),
            "schedule_title": schedule.title,
            "owner_type": schedule.owner_type,
            "owner_slack_user_id": schedule.owner_slack_user_id,
            "spec_kind": schedule.spec_kind,
            "delivery_kind": schedule.delivery_kind,
            "delivery_slack_user_id": user_id,
            "delivery_slack_channel_id": channel_id,
            "delivery_slack_thread_ts": thread_ts,
            "artifact_delivery_policy": schedule.artifact_delivery_policy,
            "planned_cost_ceiling_usd": str(schedule.planned_cost_ceiling_usd),
        },
        identity_fingerprint=f"scheduled-run-{uuid.uuid4().hex}",
        created_at=run_created_at,
        finished_at=run_finished_at,
    )
    session.add(task)
    session.flush()
    session.add_all(
        [
            SlackIdentity(
                installation_id=installation.id,
                kind="user",
                slack_id=user_id,
                display_name=slack_user_name,
                raw_name=slack_user_name,
                raw_json={"id": user_id, "profile": {"real_name": slack_user_name}},
                refreshed_at=run_created_at,
                last_seen_at=run_created_at,
            ),
            SlackIdentity(
                installation_id=installation.id,
                kind="channel",
                slack_id=channel_id,
                display_name="#scheduled-dm",
                raw_name="scheduled-dm",
                raw_json={"id": channel_id, "name": "scheduled-dm"},
                refreshed_at=run_created_at,
                last_seen_at=run_created_at,
            ),
        ]
    )
    session.flush()
    return task


def create_dashboard_memory_fact(
    session: Session,
    task: Task,
    *,
    key: str,
    value_text: str,
    scope_type: str = "user",
    scope_id: str | None = "UCost",
) -> WorkspaceState:
    fact = WorkspaceState(
        installation_id=task.installation_id,
        scope_type=scope_type,
        scope_id=scope_id,
        key=key,
        value_json={"text": value_text},
        value_text=value_text,
        status="active",
        source_kind="user_explicit",
        source_task_id=task.id,
        source_slack_channel_id=task.slack_channel_id,
        source_slack_message_ts=task.slack_message_ts,
        proposed_by=task.slack_user_id,
        confirmed_by_user_id=task.slack_user_id,
        confirmed_at=datetime(2026, 5, 24, 12, 1, tzinfo=UTC),
    )
    session.add(fact)
    session.commit()
    return fact


def test_dashboard_playground_routes(
    client: tuple[TestClient, Session],
) -> None:
    test_client, session = client
    login(test_client)

    # 1. GET /playground should render successfully
    response = test_client.get("/playground")
    assert response.status_code == 200
    assert "Agent Playground" in response.text
    assert "Sandbox Prompt" in response.text

    # Seed an installation so task creation succeeds
    installation = Installation(
        slack_team_id="TPlayground", team_name="Playground Team"
    )
    session.add(installation)
    session.commit()

    # 2. POST /playground/run should create a task
    run_response = test_client.post(
        "/playground/run",
        data={"prompt": "Hello Sandbox!"},
    )
    assert run_response.status_code == 200
    data = run_response.json()
    assert "task_id" in data
    task_id = data["task_id"]

    # Verify task was created in DB
    task = session.get(Task, uuid.UUID(task_id))
    assert task is not None
    assert task.input == "Hello Sandbox!"
    assert task.slack_channel_id == "playground"
    assert task.identity_kind == "manual"
    assert task.status == TaskStatus.pending

    # 3. GET /playground/{task_id}/stream should return SSE stream.
    # The stream only terminates once the task reaches a terminal status —
    # TestClient.get() reads the full body, so a pending task would poll forever.
    task.status = TaskStatus.succeeded
    session.commit()

    stream_response = test_client.get(f"/playground/{task_id}/stream")
    assert stream_response.status_code == 200
    assert "text/event-stream" in stream_response.headers["content-type"]

    lines = stream_response.text.split("\n\n")
    assert len(lines) > 0
    # First message should be the connection handshake, last the finish marker
    assert "connected" in lines[0]
    assert '"finished": true' in stream_response.text


def cleanup_database(session: Session) -> None:
    for model in (
        KnowledgeGraphEvidence,
        KnowledgeGraphEdge,
        KnowledgeGraphEntity,
        WitnessOpportunityCandidate,
        ObserveChannelProfile,
        Artifact,
        LLMUsage,
        WorkspaceState,
        Episode,
        ComposioConnection,
        DashboardOAuthState,
        DashboardUser,
        Schedule,
        SlackChannelMembership,
        SlackIdentity,
        TaskEvent,
        Task,
        LLMConfigAudit,
        LLMTierAssignment,
        LLMModelPricing,
        LLMModelCatalog,
        LLMProviderAccount,
        ModelPricing,
        EncryptedSecret,
        Installation,
    ):
        session.execute(delete(model))
