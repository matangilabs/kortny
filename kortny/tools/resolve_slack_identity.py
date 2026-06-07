"""Resolve Slack user and channel IDs from Kortny's local identity cache."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from kortny.db.models import SlackChannelMembership, SlackIdentity, Task
from kortny.tools.types import JsonObject, JsonSchema, ToolResult

IdentityKind = Literal["auto", "user", "channel"]
MAX_IDENTITY_LOOKUP_IDS = 25


@dataclass(frozen=True, slots=True)
class ResolvedSlackIdentity:
    """User-facing resolved Slack identity payload."""

    slack_id: str
    kind: str
    display_name: str
    resolved: bool
    source: str
    raw_name: str | None = None
    is_deleted: bool = False
    is_bot: bool = False
    is_private: bool = False
    last_seen_at: str | None = None
    refreshed_at: str | None = None

    def to_payload(self) -> JsonObject:
        return {
            "slack_id": self.slack_id,
            "kind": self.kind,
            "display_name": self.display_name,
            "raw_name": self.raw_name,
            "resolved": self.resolved,
            "source": self.source,
            "is_deleted": self.is_deleted,
            "is_bot": self.is_bot,
            "is_private": self.is_private,
            "last_seen_at": self.last_seen_at,
            "refreshed_at": self.refreshed_at,
        }


class ResolveSlackIdentityTool:
    """Resolve Slack IDs to cached user/channel display names."""

    name = "resolve_slack_identity"
    description = (
        "Resolves Slack user IDs and channel IDs to cached display names from "
        "Kortny's local identity cache. Use this when Slack evidence or tool "
        "results contain IDs like U123, C123, G123, or D123 and the answer "
        "should name the person or channel naturally. This tool is read-only "
        "and does not call Slack APIs."
    )
    parameters: JsonSchema = {
        "type": "object",
        "properties": {
            "slack_ids": {
                "type": "array",
                "description": (
                    "Slack user or channel IDs to resolve. Omit this to resolve "
                    "the current task user and channel."
                ),
                "items": {"type": "string"},
                "maxItems": MAX_IDENTITY_LOOKUP_IDS,
            },
            "kind": {
                "type": "string",
                "description": (
                    "Optional expected identity kind. Use auto unless the user "
                    "or prior tool result clearly identifies user or channel."
                ),
                "enum": ["auto", "user", "channel"],
                "default": "auto",
            },
            "include_current_context": {
                "type": "boolean",
                "description": (
                    "Also resolve the current task Slack user and channel when "
                    "explicit IDs are provided."
                ),
                "default": False,
            },
            "include_deleted": {
                "type": "boolean",
                "description": "Whether to return deleted or archived cached identities.",
                "default": True,
            },
        },
        "additionalProperties": False,
    }

    def __init__(self, *, session: Session, task: Task) -> None:
        self.session = session
        self.task = task

    def invoke(self, args: JsonObject) -> ToolResult:
        kind = _coerce_kind(args.get("kind"))
        include_current_context = _coerce_bool(
            args.get("include_current_context"),
            default=False,
            name="include_current_context",
        )
        include_deleted = _coerce_bool(
            args.get("include_deleted"),
            default=True,
            name="include_deleted",
        )
        slack_ids = _coerce_slack_ids(args.get("slack_ids"))
        if include_current_context or not slack_ids:
            slack_ids = _with_current_context(
                slack_ids,
                task=self.task,
                kind=kind,
            )

        identities = [
            self._resolve_one(
                slack_id=slack_id,
                kind=kind,
                include_deleted=include_deleted,
            )
            for slack_id in slack_ids
        ]
        resolved_count = sum(1 for identity in identities if identity.resolved)
        return ToolResult(
            output={
                "requested_ids": list(slack_ids),
                "identity_count": len(identities),
                "resolved_count": resolved_count,
                "unresolved_count": len(identities) - resolved_count,
                "identities": [identity.to_payload() for identity in identities],
                "source": "slack_identities",
                "source_note": (
                    "Resolved from Kortny's local Slack identity cache and "
                    "channel membership records only. No Slack API call was made."
                ),
            }
        )

    def _resolve_one(
        self,
        *,
        slack_id: str,
        kind: IdentityKind,
        include_deleted: bool,
    ) -> ResolvedSlackIdentity:
        resolved_kind = _resolve_kind(slack_id, kind)
        identity = self._identity(
            slack_id=slack_id,
            kind=resolved_kind,
            include_deleted=include_deleted,
        )
        if identity is not None:
            return _from_identity(identity)
        if resolved_kind == "channel":
            membership = self._channel_membership(slack_id=slack_id)
            if membership is not None and membership.channel_name:
                return ResolvedSlackIdentity(
                    slack_id=slack_id,
                    kind="channel",
                    display_name=f"#{membership.channel_name}",
                    raw_name=membership.channel_name,
                    resolved=True,
                    source="slack_channel_memberships",
                    is_private=membership.channel_type in {"private_channel", "mpim"},
                    last_seen_at=membership.last_seen_at.isoformat(),
                )
        return ResolvedSlackIdentity(
            slack_id=slack_id,
            kind=resolved_kind,
            display_name=slack_id,
            resolved=False,
            source="unresolved",
        )

    def _identity(
        self,
        *,
        slack_id: str,
        kind: str,
        include_deleted: bool,
    ) -> SlackIdentity | None:
        query = select(SlackIdentity).where(
            SlackIdentity.installation_id == self.task.installation_id,
            SlackIdentity.kind == kind,
            SlackIdentity.slack_id == slack_id,
        )
        if not include_deleted:
            query = query.where(SlackIdentity.is_deleted.is_(False))
        return self.session.scalar(query)

    def _channel_membership(self, *, slack_id: str) -> SlackChannelMembership | None:
        return self.session.scalar(
            select(SlackChannelMembership).where(
                SlackChannelMembership.installation_id == self.task.installation_id,
                SlackChannelMembership.channel_id == slack_id,
                SlackChannelMembership.membership_status == "active",
            )
        )


def _from_identity(identity: SlackIdentity) -> ResolvedSlackIdentity:
    return ResolvedSlackIdentity(
        slack_id=identity.slack_id,
        kind=identity.kind,
        display_name=identity.display_name,
        raw_name=identity.raw_name,
        resolved=True,
        source="slack_identities",
        is_deleted=identity.is_deleted,
        is_bot=identity.is_bot,
        is_private=identity.is_private,
        last_seen_at=identity.last_seen_at.isoformat(),
        refreshed_at=(
            identity.refreshed_at.isoformat()
            if identity.refreshed_at is not None
            else None
        ),
    )


def _with_current_context(
    slack_ids: tuple[str, ...],
    *,
    task: Task,
    kind: IdentityKind,
) -> tuple[str, ...]:
    values = list(slack_ids)
    if kind in {"auto", "user"} and task.slack_user_id:
        values.append(task.slack_user_id)
    if kind in {"auto", "channel"} and task.slack_channel_id:
        values.append(task.slack_channel_id)
    return _dedupe(values)


def _coerce_slack_ids(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError("resolve_slack_identity 'slack_ids' must be an array")
    values: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError("resolve_slack_identity 'slack_ids' must contain strings")
        cleaned = item.strip()
        if cleaned:
            values.append(cleaned)
    return _dedupe(values)[:MAX_IDENTITY_LOOKUP_IDS]


def _dedupe(values: list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _coerce_kind(value: object) -> IdentityKind:
    if value is None:
        return "auto"
    if value == "auto":
        return "auto"
    if value == "user":
        return "user"
    if value == "channel":
        return "channel"
    raise ValueError("resolve_slack_identity 'kind' must be auto, user, or channel")


def _coerce_bool(value: object, *, default: bool, name: str) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raise ValueError(f"resolve_slack_identity '{name}' must be a boolean")


def _resolve_kind(slack_id: str, kind: IdentityKind) -> str:
    if kind != "auto":
        return kind
    if slack_id.startswith(("U", "W")):
        return "user"
    if slack_id.startswith(("C", "G", "D")):
        return "channel"
    return "user"
