"""Slack user/channel identity cache.

Dashboard pages should resolve names from Postgres, not from live Slack API
calls. This service refreshes identities opportunistically while processing
Slack events and keeps failures non-fatal.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol, cast

from sqlalchemy import select
from sqlalchemy.orm import Session

from kortny.db.models import SlackIdentity

logger = logging.getLogger(__name__)
DEFAULT_REFRESH_AFTER = timedelta(hours=24)


class SlackIdentityClient(Protocol):
    """Subset of Slack WebClient methods used for identity refresh."""

    def users_info(self, *, user: str) -> object:
        """Return Slack user metadata."""

    def conversations_info(self, *, channel: str) -> object:
        """Return Slack conversation metadata."""


@dataclass(frozen=True, slots=True)
class IdentityRefreshResult:
    """Outcome of an identity refresh attempt."""

    identity: SlackIdentity | None
    refreshed: bool
    reason: str | None = None


class SlackIdentityService:
    """Read/write service for cached Slack display identities."""

    def __init__(
        self,
        session: Session,
        *,
        refresh_after: timedelta = DEFAULT_REFRESH_AFTER,
    ) -> None:
        self.session = session
        self.refresh_after = refresh_after

    def ensure_user(
        self,
        *,
        installation_id: uuid.UUID,
        user_id: str,
        client: object,
        now: datetime | None = None,
    ) -> IdentityRefreshResult:
        """Refresh a user identity if missing or stale."""

        return self._ensure(
            installation_id=installation_id,
            kind="user",
            slack_id=user_id,
            client=client,
            now=now,
        )

    def ensure_channel(
        self,
        *,
        installation_id: uuid.UUID,
        channel_id: str,
        client: object,
        now: datetime | None = None,
    ) -> IdentityRefreshResult:
        """Refresh a channel identity if missing or stale."""

        return self._ensure(
            installation_id=installation_id,
            kind="channel",
            slack_id=channel_id,
            client=client,
            now=now,
        )

    def _ensure(
        self,
        *,
        installation_id: uuid.UUID,
        kind: str,
        slack_id: str,
        client: object,
        now: datetime | None,
    ) -> IdentityRefreshResult:
        current_time = now or datetime.now(UTC)
        existing = self.get(
            installation_id=installation_id,
            kind=kind,
            slack_id=slack_id,
        )
        if existing is not None and not self._is_stale(existing, current_time):
            self._normalize_cached_identity(existing, now=current_time)
            existing.last_seen_at = current_time
            self.session.flush()
            return IdentityRefreshResult(identity=existing, refreshed=False)

        try:
            payload = self._fetch(kind=kind, slack_id=slack_id, client=client)
        except Exception as exc:
            if existing is not None:
                existing.last_seen_at = current_time
                self.session.flush()
            logger.info(
                "slack identity refresh failed kind=%s slack_id=%s error_type=%s error=%s",
                kind,
                slack_id,
                type(exc).__name__,
                exc,
            )
            return IdentityRefreshResult(
                identity=existing,
                refreshed=False,
                reason=type(exc).__name__,
            )

        fields = (
            _user_fields(slack_id, payload)
            if kind == "user"
            else _channel_fields(slack_id, payload)
        )
        identity = self.upsert(
            installation_id=installation_id,
            kind=kind,
            slack_id=slack_id,
            now=current_time,
            **fields,
        )
        return IdentityRefreshResult(identity=identity, refreshed=True)

    def get(
        self,
        *,
        installation_id: uuid.UUID,
        kind: str,
        slack_id: str,
    ) -> SlackIdentity | None:
        """Return one cached identity row."""

        return self.session.scalar(
            select(SlackIdentity).where(
                SlackIdentity.installation_id == installation_id,
                SlackIdentity.kind == kind,
                SlackIdentity.slack_id == slack_id,
            )
        )

    def upsert(
        self,
        *,
        installation_id: uuid.UUID,
        kind: str,
        slack_id: str,
        display_name: str,
        raw_name: str | None,
        is_deleted: bool = False,
        is_bot: bool = False,
        is_private: bool = False,
        raw_json: dict[str, Any] | None = None,
        now: datetime | None = None,
    ) -> SlackIdentity:
        """Insert or update an identity row."""

        current_time = now or datetime.now(UTC)
        identity = self.get(
            installation_id=installation_id,
            kind=kind,
            slack_id=slack_id,
        )
        if identity is None:
            identity = SlackIdentity(
                installation_id=installation_id,
                kind=kind,
                slack_id=slack_id,
                display_name=display_name,
            )
            self.session.add(identity)

        identity.display_name = display_name
        identity.raw_name = raw_name
        identity.is_deleted = is_deleted
        identity.is_bot = is_bot
        identity.is_private = is_private
        identity.raw_json = raw_json or {}
        identity.last_seen_at = current_time
        identity.refreshed_at = current_time
        self.session.flush()
        return identity

    def _is_stale(self, identity: SlackIdentity, now: datetime) -> bool:
        if identity.refreshed_at is None:
            return True
        return identity.refreshed_at <= now - self.refresh_after

    def _normalize_cached_identity(
        self,
        identity: SlackIdentity,
        *,
        now: datetime,
    ) -> None:
        if not identity.raw_json:
            return
        fields = (
            _user_fields(identity.slack_id, {"user": identity.raw_json})
            if identity.kind == "user"
            else _channel_fields(identity.slack_id, {"channel": identity.raw_json})
        )
        if fields["display_name"] != identity.display_name:
            identity.display_name = fields["display_name"]
            identity.raw_name = fields["raw_name"]
            identity.refreshed_at = now

    def _fetch(
        self,
        *,
        kind: str,
        slack_id: str,
        client: object,
    ) -> Mapping[str, Any]:
        if kind == "user":
            users_info = getattr(client, "users_info", None)
            if not callable(users_info):
                raise RuntimeError("Slack client does not support users_info")
            return _api_response_mapping(users_info(user=slack_id))

        conversations_info = getattr(client, "conversations_info", None)
        if not callable(conversations_info):
            raise RuntimeError("Slack client does not support conversations_info")
        return _api_response_mapping(conversations_info(channel=slack_id))


def _user_fields(slack_id: str, response: Mapping[str, Any]) -> dict[str, Any]:
    user = _mapping(response.get("user"))
    profile = _mapping(user.get("profile"))
    display_name = _first_present(
        profile.get("real_name_normalized"),
        profile.get("real_name"),
        user.get("real_name_normalized"),
        user.get("real_name"),
        profile.get("display_name_normalized"),
        profile.get("display_name"),
        user.get("name"),
        slack_id,
    )
    raw_name = _first_present(
        profile.get("real_name_normalized"),
        profile.get("real_name"),
        user.get("real_name_normalized"),
        user.get("real_name"),
        user.get("name"),
    )
    return {
        "display_name": str(display_name),
        "raw_name": str(raw_name) if raw_name else None,
        "is_deleted": bool(user.get("deleted")),
        "is_bot": bool(user.get("is_bot")),
        "is_private": False,
        "raw_json": dict(user),
    }


def _channel_fields(slack_id: str, response: Mapping[str, Any]) -> dict[str, Any]:
    channel = _mapping(response.get("channel"))
    raw_name = _first_present(channel.get("name"), channel.get("user"))
    is_im = bool(channel.get("is_im"))
    display_name = _channel_display_name(slack_id, channel, raw_name, is_im=is_im)
    return {
        "display_name": display_name,
        "raw_name": str(raw_name) if raw_name else None,
        "is_deleted": bool(channel.get("is_archived")),
        "is_bot": False,
        "is_private": bool(channel.get("is_private") or channel.get("is_group")),
        "raw_json": dict(channel),
    }


def _channel_display_name(
    slack_id: str,
    channel: Mapping[str, Any],
    raw_name: object | None,
    *,
    is_im: bool,
) -> str:
    if is_im:
        user_id = _first_present(channel.get("user"), raw_name)
        return f"DM {user_id}" if user_id else "Direct message"
    if raw_name:
        return f"#{raw_name}"
    return slack_id


def _mapping(value: object) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _api_response_mapping(response: object) -> Mapping[str, Any]:
    payload = _mapping(response)
    if not payload:
        data = getattr(response, "data", None)
        payload = _mapping(data)
    if not payload and hasattr(response, "get"):
        payload = cast(Mapping[str, Any], response)
    if payload.get("ok") is False:
        error = payload.get("error") or "unknown_error"
        raise RuntimeError(f"Slack API returned {error}")
    return payload


def _first_present(*values: object | None) -> object | None:
    for value in values:
        if isinstance(value, str) and value.strip() == "":
            continue
        if value is not None:
            return value
    return None
