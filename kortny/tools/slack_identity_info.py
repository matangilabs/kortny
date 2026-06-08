"""Live Slack identity refresh tools."""

from __future__ import annotations

from datetime import timedelta
from typing import Literal

from sqlalchemy.orm import Session

from kortny.db.models import SlackIdentity, Task
from kortny.slack.identity import SlackIdentityService
from kortny.tools.types import JsonObject, JsonSchema, ToolResult

IdentityKind = Literal["user", "channel"]


class SlackUserInfoTool:
    """Refresh and return a Slack user identity."""

    name = "slack_user_info"
    description = (
        "Refreshes and returns Slack user identity information using Slack "
        "users.info, backed by Kortny's identity cache. Use this when a Slack "
        "user ID needs a natural name and resolve_slack_identity returned a "
        "miss or stale-looking result. Omit user_id for the current requester."
    )
    parameters: JsonSchema = {
        "type": "object",
        "properties": {
            "user_id": {
                "type": "string",
                "description": (
                    "Optional Slack user ID such as U123 or W123. Omit to use "
                    "the current task user."
                ),
            },
            "force_refresh": {
                "type": "boolean",
                "description": "Force a Slack API refresh even when the cache is fresh.",
                "default": False,
            },
        },
        "additionalProperties": False,
    }

    def __init__(
        self,
        *,
        client: object,
        session: Session,
        task: Task,
    ) -> None:
        self.client = client
        self.session = session
        self.task = task

    def invoke(self, args: JsonObject) -> ToolResult:
        user_id = _coerce_user_id(args.get("user_id"), default=self.task.slack_user_id)
        force_refresh = _coerce_bool(args.get("force_refresh"), name="force_refresh")
        service = _identity_service(self.session, force_refresh=force_refresh)
        result = service.ensure_user(
            installation_id=self.task.installation_id,
            user_id=user_id,
            client=self.client,
        )
        return ToolResult(
            output=_identity_result_payload(
                kind="user",
                slack_id=user_id,
                identity=result.identity,
                refreshed=result.refreshed,
                reason=result.reason,
            )
        )


class SlackChannelInfoTool:
    """Refresh and return the current Slack channel identity."""

    name = "slack_channel_info"
    description = (
        "Refreshes and returns the current Slack channel identity using Slack "
        "conversations.info, backed by Kortny's identity cache. Use this when "
        "the current channel ID needs a natural name and resolve_slack_identity "
        "returned a miss or stale-looking result. This tool is scoped to the "
        "current task channel."
    )
    parameters: JsonSchema = {
        "type": "object",
        "properties": {
            "channel_id": {
                "type": "string",
                "description": (
                    "Optional current Slack channel ID. Omit unless copying the "
                    "current task channel exactly."
                ),
            },
            "force_refresh": {
                "type": "boolean",
                "description": "Force a Slack API refresh even when the cache is fresh.",
                "default": False,
            },
        },
        "additionalProperties": False,
    }

    def __init__(
        self,
        *,
        client: object,
        session: Session,
        task: Task,
    ) -> None:
        self.client = client
        self.session = session
        self.task = task

    def invoke(self, args: JsonObject) -> ToolResult:
        channel_id = _coerce_current_channel_id(
            args.get("channel_id"),
            task=self.task,
        )
        force_refresh = _coerce_bool(args.get("force_refresh"), name="force_refresh")
        service = _identity_service(self.session, force_refresh=force_refresh)
        result = service.ensure_channel(
            installation_id=self.task.installation_id,
            channel_id=channel_id,
            client=self.client,
        )
        return ToolResult(
            output=_identity_result_payload(
                kind="channel",
                slack_id=channel_id,
                identity=result.identity,
                refreshed=result.refreshed,
                reason=result.reason,
            )
        )


def _identity_service(session: Session, *, force_refresh: bool) -> SlackIdentityService:
    refresh_after = timedelta(seconds=0) if force_refresh else None
    if refresh_after is None:
        return SlackIdentityService(session)
    return SlackIdentityService(session, refresh_after=refresh_after)


def _identity_result_payload(
    *,
    kind: IdentityKind,
    slack_id: str,
    identity: SlackIdentity | None,
    refreshed: bool,
    reason: str | None,
) -> JsonObject:
    if identity is None:
        return {
            "successful": False,
            "kind": kind,
            "slack_id": slack_id,
            "resolved": False,
            "refreshed": refreshed,
            "reason": reason,
            "source": "slack_api",
        }
    return {
        "successful": True,
        "kind": identity.kind,
        "slack_id": identity.slack_id,
        "display_name": identity.display_name,
        "raw_name": identity.raw_name,
        "resolved": True,
        "refreshed": refreshed,
        "reason": reason,
        "source": "slack_identity_cache",
        "is_deleted": identity.is_deleted,
        "is_bot": identity.is_bot,
        "is_private": identity.is_private,
        "last_seen_at": identity.last_seen_at.isoformat(),
        "refreshed_at": (
            identity.refreshed_at.isoformat()
            if identity.refreshed_at is not None
            else None
        ),
    }


def _coerce_user_id(value: object, *, default: str) -> str:
    if value is None:
        return default
    if not isinstance(value, str) or not value.strip():
        raise ValueError("slack_user_info 'user_id' must be a Slack user ID")
    user_id = value.strip()
    if not user_id.startswith(("U", "W")):
        raise ValueError("slack_user_info 'user_id' must start with U or W")
    return user_id


def _coerce_current_channel_id(value: object, *, task: Task) -> str:
    if value is None:
        return task.slack_channel_id
    if not isinstance(value, str) or not value.strip():
        raise ValueError("slack_channel_info 'channel_id' must be a Slack channel ID")
    channel_id = value.strip()
    if channel_id != task.slack_channel_id:
        raise ValueError("slack_channel_info can only target the current Slack channel")
    return channel_id


def _coerce_bool(value: object, *, name: str) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    raise ValueError(f"Slack identity info '{name}' must be a boolean")
