import os
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from urllib.parse import parse_qs as parse_url_qs
from urllib.parse import urlsplit

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from httpx import Response
from sqlalchemy import Engine, delete, select
from sqlalchemy.orm import Session

from kortny.composio import (
    ComposioAuthConfig,
    ComposioCatalog,
    ComposioConnectionRequest,
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
    LLMProvider,
    LLMUsage,
    ModelPricing,
    SlackIdentity,
    Task,
    TaskEvent,
    TaskEventType,
    TaskStatus,
    WorkspaceState,
)
from kortny.db.session import make_engine, make_session_factory, normalize_database_url

TEST_POSTGRES_URL = os.environ.get("KORTNY_TEST_POSTGRES_URL")
ALLOW_DESTRUCTIVE_TEST_DB = os.environ.get("KORTNY_ALLOW_DESTRUCTIVE_TEST_DB") == "1"

pytestmark = pytest.mark.skipif(
    TEST_POSTGRES_URL is None,
    reason="KORTNY_TEST_POSTGRES_URL is required for dashboard integration tests",
)


@pytest.fixture(scope="session")
def engine() -> Iterator[Engine]:
    assert TEST_POSTGRES_URL is not None
    assert_safe_dashboard_test_database(TEST_POSTGRES_URL)

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
        admin_tasks_response = test_client.get("/tasks", follow_redirects=False)
        member_tasks_response = test_client.get("/me/tasks")
        own_detail_response = test_client.get(f"/me/tasks/{own_task.id}")
        other_detail_response = test_client.get(f"/me/tasks/{other_task.id}")

    assert login_response.status_code == 303
    assert login_response.headers["location"] == "/me"
    assert admin_tasks_response.status_code == 403
    assert member_tasks_response.status_code == 200
    assert "Member User" in member_tasks_response.text
    assert "Other User" not in member_tasks_response.text
    assert own_detail_response.status_code == 200
    assert "Member private task" in own_detail_response.text
    assert other_detail_response.status_code == 404


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
        ) -> ComposioCatalog:
            assert search is None
            assert limit == 60
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
    assert response.text.index("<h3>Notion</h3>") < response.text.index("<h3>GitHub</h3>")


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
    assert "Hosted Composio authorization" in response.text
    assert "Connect github" in response.text
    assert "composio-dashboard-secret" not in response.text


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
    assert 'name="visibility_scope_type" value="workspace"' in response.text
    assert "Save Visibility" in response.text
    assert f'action="/composio/connections/{connection.id}/disconnect"' in response.text
    assert "Connect notion" not in response.text


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
    installation = Installation(slack_team_id="TComposioMember", team_name="Member Team")
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
            redirect_path="/me/integrations/notion",
            expires_at=datetime.now(UTC) + timedelta(minutes=10),
        )
    )
    db_session.commit()
    session_factory = make_session_factory(engine=engine)
    settings = slack_dashboard_settings()
    set_runtime_settings_env(monkeypatch)
    monkeypatch.setenv("COMPOSIO_API_KEY", "composio-dashboard-secret")

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
    monkeypatch.setattr("kortny.dashboard.app.ComposioClient", FakeComposioClient)

    with TestClient(
        create_app(settings=settings, session_factory=session_factory)
    ) as test_client:
        login_response = test_client.get(
            "/auth/slack/callback?code=member-code&state=member-composio-state",
            follow_redirects=False,
        )
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
    assert login_response.headers["location"] == "/me/integrations/notion"
    assert response.status_code == 303
    assert response.headers["location"] == "https://connect.composio.dev/link/ln_member"
    connection = db_session.scalar(
        select(ComposioConnection).where(
            ComposioConnection.toolkit_slug == "notion"
        )
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
    installation = Installation(slack_team_id="TComposioCallback", team_name="Member Team")
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
    assert response.headers["location"].startswith("/me/integrations/gmail?")
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
    assert "Requires BRAVE_SEARCH_API_KEY." in response.text
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
    assert "Top Usage Drivers" in response.text
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
    assert 'class="overview-main-grid"' in response.text


def test_dashboard_task_list_shows_cost_models_and_turns(
    client: tuple[TestClient, Session],
) -> None:
    test_client, session = client
    task = create_dashboard_task(session)
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
    assert "#ops-desk" in response.text
    assert "Aneesh Melkot" in response.text
    assert "Done with cost summary" in response.text
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
    assert 'class="input" type="date"' in response.text


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
    assert "Model Spend" in response.text
    assert "User Spend" in response.text
    assert "Task Failure Rate" in response.text
    assert "50.0%" in response.text
    assert "$0.012600" in response.text
    assert "4,500" in response.text
    assert "1 failed" in response.text
    assert 'class="charts-grid"' in response.text
    assert "--bar-value:" in response.text


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


def login(test_client: TestClient) -> Response:
    return test_client.post(
        "/login",
        data={"username": "admin", "password": "secret", "next": "/"},
        follow_redirects=False,
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


def assert_safe_dashboard_test_database(database_url: str) -> None:
    parsed = urlsplit(database_url)
    hostname = parsed.hostname or ""
    port = parsed.port or 5432
    database_name = parsed.path.lstrip("/")
    is_default_local_dev_db = (
        hostname in {"localhost", "127.0.0.1", "::1"}
        and port == 5432
        and database_name == "kortny"
    )
    if is_default_local_dev_db and not ALLOW_DESTRUCTIVE_TEST_DB:
        pytest.fail(
            "Refusing to run destructive dashboard integration tests against "
            "the default local dev database. Use a dedicated test database, "
            "for example postgresql://kortny:kortny@localhost:5432/kortny_test, "
            "or set KORTNY_ALLOW_DESTRUCTIVE_TEST_DB=1 if this is intentional.",
            pytrace=False,
        )


def set_runtime_settings_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-dashboard-secret")
    monkeypatch.setenv("SLACK_APP_TOKEN", "xapp-dashboard-secret")
    monkeypatch.setenv("SLACK_SIGNING_SECRET", "dashboard-signing-secret")
    monkeypatch.setenv("SLACK_APP_NAME", "kortny")
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


def _composio_toolkit(*, slug: str, name: str) -> ComposioToolkit:
    return ComposioToolkit(
        slug=slug,
        name=name,
        description=f"{name} toolkit",
        categories=(),
        auth_schemes=("oauth2",),
        managed_auth_schemes=("oauth2",),
        tools_count=4,
        triggers_count=1,
        logo_url=None,
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


def cleanup_database(session: Session) -> None:
    for model in (
        Artifact,
        LLMUsage,
        WorkspaceState,
        Episode,
        ComposioConnection,
        DashboardOAuthState,
        DashboardUser,
        SlackIdentity,
        TaskEvent,
        Task,
        ModelPricing,
        EncryptedSecret,
        Installation,
    ):
        session.execute(delete(model))
