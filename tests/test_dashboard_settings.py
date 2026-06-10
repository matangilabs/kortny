import pytest

from kortny.dashboard.settings import DashboardAuthMode, DashboardSettings


def test_dashboard_settings_defaults_to_bootstrap_login(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "DASHBOARD_AUTH_MODE",
        "DASHBOARD_SLACK_CLIENT_ID",
        "DASHBOARD_SLACK_CLIENT_SECRET",
        "DASHBOARD_SLACK_REDIRECT_URI",
    ):
        monkeypatch.delenv(name, raising=False)
    settings = DashboardSettings(  # type: ignore[call-arg]
        _env_file=None,
        postgres_url="postgresql://kortny:kortny@localhost/kortny",
        session_secret="test-dashboard-session-secret",
    )

    assert settings.auth_mode is DashboardAuthMode.bootstrap
    assert settings.bootstrap_login_enabled is True
    assert settings.slack_login_configured is False
    assert settings.slack_login_enabled is False


def test_dashboard_settings_enables_slack_login_when_configured() -> None:
    settings = DashboardSettings(  # type: ignore[call-arg]
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


def test_dashboard_settings_requires_slack_config_for_hybrid_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "DASHBOARD_SLACK_CLIENT_ID",
        "DASHBOARD_SLACK_CLIENT_SECRET",
        "DASHBOARD_SLACK_REDIRECT_URI",
    ):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match="DASHBOARD_SLACK_CLIENT_ID"):
        DashboardSettings(  # type: ignore[call-arg]
            _env_file=None,
            postgres_url="postgresql://kortny:kortny@localhost/kortny",
            session_secret="test-dashboard-session-secret",
            auth_mode=DashboardAuthMode.hybrid,
        )


def test_dashboard_settings_rejects_invalid_state_ttl() -> None:
    with pytest.raises(ValueError):
        DashboardSettings(  # type: ignore[call-arg]
            _env_file=None,
            postgres_url="postgresql://kortny:kortny@localhost/kortny",
            session_secret="test-dashboard-session-secret",
            slack_oauth_state_ttl_minutes=0,
        )
