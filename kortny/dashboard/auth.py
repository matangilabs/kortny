"""Dashboard authentication helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from urllib.parse import urlencode
from uuid import UUID

import httpx
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from kortny.db.models import DashboardUser, Installation, SlackIdentity

SLACK_AUTHORIZE_ENDPOINT = "https://slack.com/openid/connect/authorize"
SLACK_TOKEN_ENDPOINT = "https://slack.com/api/openid.connect.token"
SLACK_USERINFO_ENDPOINT = "https://slack.com/api/openid.connect.userInfo"
SLACK_USER_ID_CLAIM = "https://slack.com/user_id"
SLACK_TEAM_ID_CLAIM = "https://slack.com/team_id"


@dataclass(frozen=True)
class DashboardPrincipal:
    """Authenticated dashboard identity stored in the session."""

    display_name: str
    role: str
    source: str
    dashboard_user_id: UUID | None = None
    installation_id: UUID | None = None
    slack_user_id: str | None = None


@dataclass(frozen=True)
class SlackOpenIDProfile:
    """User profile returned by Slack's OpenID userInfo method."""

    team_id: str
    user_id: str
    display_name: str
    email: str | None
    avatar_url: str | None
    raw_json: dict[str, Any]


class DashboardAuthError(RuntimeError):
    """Raised when dashboard authentication cannot proceed safely."""


class SlackOpenIDClient:
    """Small Sign in with Slack client for the dashboard login flow."""

    def __init__(
        self,
        *,
        client_id: str,
        client_secret: str,
        redirect_uri: str,
        timeout_seconds: float = 10,
    ) -> None:
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.timeout_seconds = timeout_seconds

    def authorize_url(self, *, state: str) -> str:
        """Build the Slack OpenID authorization URL."""

        query = urlencode(
            {
                "response_type": "code",
                "client_id": self.client_id,
                "scope": "openid profile email",
                "redirect_uri": self.redirect_uri,
                "state": state,
            }
        )
        return f"{SLACK_AUTHORIZE_ENDPOINT}?{query}"

    def exchange_code(self, *, code: str) -> str:
        """Exchange an authorization code for a Slack OpenID access token."""

        response = httpx.post(
            SLACK_TOKEN_ENDPOINT,
            data={
                "grant_type": "authorization_code",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "code": code,
                "redirect_uri": self.redirect_uri,
            },
            timeout=self.timeout_seconds,
        )
        payload = self._payload(response)
        access_token = payload.get("access_token")
        if not isinstance(access_token, str) or not access_token:
            raise DashboardAuthError("Slack did not return an OpenID access token.")
        return access_token

    def user_info(self, *, access_token: str) -> SlackOpenIDProfile:
        """Fetch the Slack profile for a Sign in with Slack access token."""

        response = httpx.post(
            SLACK_USERINFO_ENDPOINT,
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=self.timeout_seconds,
        )
        payload = self._payload(response)
        return _profile_from_slack_user_info(payload)

    def _payload(self, response: httpx.Response) -> dict[str, Any]:
        try:
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise DashboardAuthError("Slack OpenID request failed.") from exc

        if not isinstance(payload, dict):
            raise DashboardAuthError("Slack OpenID returned an invalid response.")
        if payload.get("ok") is False:
            error = payload.get("error")
            if isinstance(error, str) and error:
                raise DashboardAuthError(f"Slack OpenID error: {error}")
            raise DashboardAuthError("Slack OpenID returned an error.")
        return payload


def upsert_dashboard_user(
    session: Session,
    *,
    profile: SlackOpenIDProfile,
    now: datetime,
) -> DashboardUser:
    """Create or update a dashboard user from a Slack OpenID profile."""

    installation = _dashboard_installation_for_profile(session, profile)
    user = session.scalar(
        select(DashboardUser).where(
            DashboardUser.installation_id == installation.id,
            DashboardUser.slack_user_id == profile.user_id,
        )
    )
    if user is None:
        user_count = int(
            session.scalar(
                select(func.count()).select_from(DashboardUser).where(
                    DashboardUser.installation_id == installation.id
                )
            )
            or 0
        )
        user = DashboardUser(
            installation_id=installation.id,
            slack_user_id=profile.user_id,
            email=profile.email,
            display_name=profile.display_name,
            avatar_url=profile.avatar_url,
            role="admin" if user_count == 0 else "member",
            status="active",
            last_login_at=now,
        )
        session.add(user)
    else:
        if user.status == "disabled":
            raise DashboardAuthError("This dashboard user has been disabled.")
        user.email = profile.email
        user.display_name = profile.display_name
        user.avatar_url = profile.avatar_url
        user.last_login_at = now

    _upsert_slack_identity(session, installation=installation, profile=profile, now=now)
    session.flush()
    return user


def _dashboard_installation_for_profile(
    session: Session,
    profile: SlackOpenIDProfile,
) -> Installation:
    installation = session.scalar(
        select(Installation).where(Installation.slack_team_id == profile.team_id)
    )
    if installation is not None:
        return installation

    installation_count = int(session.scalar(select(func.count()).select_from(Installation)) or 0)
    if installation_count > 0:
        raise DashboardAuthError(
            "This Slack workspace is not connected to this Kortny instance."
        )

    installation = Installation(slack_team_id=profile.team_id)
    session.add(installation)
    session.flush()
    return installation


def _upsert_slack_identity(
    session: Session,
    *,
    installation: Installation,
    profile: SlackOpenIDProfile,
    now: datetime,
) -> None:
    identity = session.scalar(
        select(SlackIdentity).where(
            SlackIdentity.installation_id == installation.id,
            SlackIdentity.kind == "user",
            SlackIdentity.slack_id == profile.user_id,
        )
    )
    raw_json = {
        **profile.raw_json,
        "id": profile.user_id,
        "team_id": profile.team_id,
    }
    if identity is None:
        identity = SlackIdentity(
            installation_id=installation.id,
            kind="user",
            slack_id=profile.user_id,
            display_name=profile.display_name,
            raw_name=profile.display_name,
            raw_json=raw_json,
            refreshed_at=now,
            last_seen_at=now,
        )
        session.add(identity)
        return

    identity.display_name = profile.display_name
    identity.raw_name = profile.display_name
    identity.raw_json = raw_json
    identity.is_deleted = False
    identity.refreshed_at = now
    identity.last_seen_at = now


def _profile_from_slack_user_info(payload: dict[str, Any]) -> SlackOpenIDProfile:
    user_id = _required_string(payload, SLACK_USER_ID_CLAIM, "Slack user ID")
    team_id = _required_string(payload, SLACK_TEAM_ID_CLAIM, "Slack team ID")
    display_name = (
        _optional_string(payload.get("name"))
        or _optional_string(payload.get("preferred_username"))
        or user_id
    )
    return SlackOpenIDProfile(
        team_id=team_id,
        user_id=user_id,
        display_name=display_name,
        email=_optional_string(payload.get("email")),
        avatar_url=_optional_string(payload.get("picture")),
        raw_json=payload,
    )


def _required_string(
    payload: dict[str, Any],
    key: str,
    label: str,
) -> str:
    value = _optional_string(payload.get(key))
    if value is None:
        raise DashboardAuthError(f"Slack OpenID response is missing {label}.")
    return value


def _optional_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None
