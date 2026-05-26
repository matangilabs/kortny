import pytest

from kortny.dashboard.settings import DashboardAuthMode, DashboardSettings


def test_dashboard_settings_defaults_to_bootstrap_login() -> None:
    settings = DashboardSettings(
        _env_file=None,
        postgres_url="postgresql://kortny:kortny@localhost/kortny",
        session_secret="test-dashboard-session-secret",
    )

    assert settings.auth_mode is DashboardAuthMode.bootstrap
    assert settings.bootstrap_login_enabled is True
    assert settings.slack_login_configured is False
    assert settings.slack_login_enabled is False


def test_dashboard_settings_enables_slack_login_when_configured() -> None:
    settings = DashboardSettings(
        _env_file=None,
        postgres_url="postgresql://kortny:kortny@localhost/kortny",
        session_secret="test-dashboard-session-secret",
        auth_mode=DashboardAuthMode.hybrid,
        slack_client_id="client-id",
        slack_client_secret="client-secret",
        slack_redirect_uri="http://localhost:8080/auth/slack/callback",
    )

    assert settings.bootstrap_login_enabled is True
    assert settings.slack_login_configured is True
    assert settings.slack_login_enabled is True


def test_dashboard_settings_rejects_invalid_state_ttl() -> None:
    with pytest.raises(ValueError):
        DashboardSettings(
            _env_file=None,
            postgres_url="postgresql://kortny:kortny@localhost/kortny",
            session_secret="test-dashboard-session-secret",
            slack_oauth_state_ttl_minutes=0,
        )
