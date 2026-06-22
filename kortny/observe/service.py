"""Policy-gated Slack observation capture for Kortny Observe."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from kortny.db.models import Installation, ObservationEvent, ObservePolicy
from kortny.observe.ambient_files import summarize_event_files

logger = logging.getLogger(__name__)

DEFAULT_CHANNEL_OBSERVATION_STATUS = "active"
DEFAULT_CHANNEL_PROACTIVITY_STATUS = "digest_only"
DEFAULT_RETENTION_DAYS = 90
DEFAULT_COOLDOWN_SECONDS = 86_400
TEXT_PREVIEW_MAX_CHARS = 500
MENTION_RE = re.compile(r"<@([A-Z0-9]+)>")


@dataclass(frozen=True, slots=True)
class ObservationResult:
    """Result of attempting to record an observation."""

    observed: bool
    reason: str
    event: ObservationEvent | None = None
    policy: ObservePolicy | None = None


@dataclass(frozen=True, slots=True)
class ChannelJoinObservationResult:
    """Result of handling Kortny being added to a channel."""

    observed: bool
    reason: str
    event: ObservationEvent | None = None
    policy: ObservePolicy | None = None
    intro_text: str | None = None


class ObserveService:
    """Capture bounded observations from Slack events after policy checks."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def record_channel_message(
        self,
        *,
        installation: Installation,
        slack_team_id: str,
        body: dict[str, Any],
        event: dict[str, Any],
    ) -> ObservationResult:
        """Persist a policy-approved channel message observation."""

        ignore_reason = _channel_message_ignore_reason(
            event,
            bot_user_id=installation.bot_user_id,
        )
        if ignore_reason is not None:
            return ObservationResult(observed=False, reason=ignore_reason)

        channel_id = _optional_str(event.get("channel"))
        if channel_id is None:
            return ObservationResult(observed=False, reason="missing_channel")

        policy = self.ensure_channel_policy(
            installation=installation,
            channel_id=channel_id,
            enabled_by_user_id=None,
        )
        if not self.is_observable(policy):
            return ObservationResult(
                observed=False,
                reason="policy_disabled",
                policy=policy,
            )

        existing = self._existing_event(installation.id, body.get("event_id"))
        if existing is not None:
            return ObservationResult(
                observed=False,
                reason="duplicate",
                event=existing,
                policy=policy,
            )

        files = _event_files(event)
        visibility_metadata: dict[str, Any] = {
            "scope_type": "channel",
            "scope_id": channel_id,
            "channel_type": event.get("channel_type"),
            "subtype": event.get("subtype"),
            "file_count": len(files),
            "policy_id": str(policy.id) if policy.id is not None else None,
        }
        if files:
            visibility_metadata["files"] = summarize_event_files(files)
        observation = ObservationEvent(
            installation_id=installation.id,
            slack_team_id=slack_team_id,
            channel_id=channel_id,
            user_id=_optional_str(event.get("user")),
            event_type="file_share" if files else "message",
            slack_event_id=_optional_str(body.get("event_id")),
            message_ts=_optional_str(event.get("ts")),
            thread_ts=_optional_str(event.get("thread_ts")),
            file_id=_first_file_id(files),
            raw_payload_checksum=_payload_checksum({"body": body, "event": event}),
            text_preview=_text_preview(event.get("text")),
            visibility_metadata=visibility_metadata,
        )
        try:
            with self.session.begin_nested():
                self.session.add(observation)
                self.session.flush()
        except IntegrityError:
            existing = self._existing_event(installation.id, body.get("event_id"))
            if existing is None:
                raise
            return ObservationResult(
                observed=False,
                reason="duplicate",
                event=existing,
                policy=policy,
            )

        logger.info(
            "observe channel message recorded event_id=%s channel=%s user=%s observation_id=%s",
            observation.slack_event_id,
            channel_id,
            observation.user_id,
            observation.id,
        )
        return ObservationResult(
            observed=True,
            reason="observed",
            event=observation,
            policy=policy,
        )

    def record_channel_join(
        self,
        *,
        installation: Installation,
        slack_team_id: str,
        body: dict[str, Any],
        event: dict[str, Any],
        bot_user_id: str,
    ) -> ChannelJoinObservationResult:
        """Record Kortny being added to a channel and prepare intro text."""

        joined_user_id = _optional_str(event.get("user"))
        if joined_user_id != bot_user_id:
            return ChannelJoinObservationResult(
                observed=False,
                reason="not_bot_join",
            )

        channel_id = _optional_str(event.get("channel"))
        if channel_id is None:
            return ChannelJoinObservationResult(
                observed=False,
                reason="missing_channel",
            )

        policy = self.ensure_channel_policy(
            installation=installation,
            channel_id=channel_id,
            enabled_by_user_id=_optional_str(event.get("inviter")),
        )
        existing = self._existing_event(installation.id, body.get("event_id"))
        if existing is not None:
            return ChannelJoinObservationResult(
                observed=False,
                reason="duplicate",
                event=existing,
                policy=policy,
            )

        observation = ObservationEvent(
            installation_id=installation.id,
            slack_team_id=slack_team_id,
            channel_id=channel_id,
            user_id=joined_user_id,
            event_type="channel_join",
            slack_event_id=_optional_str(body.get("event_id")),
            message_ts=None,
            thread_ts=None,
            file_id=None,
            raw_payload_checksum=_payload_checksum({"body": body, "event": event}),
            text_preview=None,
            visibility_metadata={
                "scope_type": "channel",
                "scope_id": channel_id,
                "inviter": event.get("inviter"),
                "policy_id": str(policy.id) if policy.id is not None else None,
            },
        )
        try:
            with self.session.begin_nested():
                self.session.add(observation)
                self.session.flush()
        except IntegrityError:
            existing = self._existing_event(installation.id, body.get("event_id"))
            if existing is None:
                raise
            return ChannelJoinObservationResult(
                observed=False,
                reason="duplicate",
                event=existing,
                policy=policy,
            )

        intro_text = None
        if self.is_observable(policy) and _should_post_channel_intro(policy):
            intro_text = (
                "Thanks for adding me here. I can help summarize this channel, "
                "research questions, read shared files, and spot workflows I might "
                "be able to help with. I will keep it low-key."
            )

        logger.info(
            "observe channel join recorded event_id=%s channel=%s bot_user=%s observation_id=%s intro=%s",
            observation.slack_event_id,
            channel_id,
            bot_user_id,
            observation.id,
            bool(intro_text),
        )
        return ChannelJoinObservationResult(
            observed=True,
            reason="observed",
            event=observation,
            policy=policy,
            intro_text=intro_text,
        )

    def record_channel_activation(
        self,
        *,
        installation: Installation,
        slack_team_id: str,
        body: dict[str, Any],
        event: dict[str, Any],
    ) -> ChannelJoinObservationResult:
        """Record first observed channel activation when Slack omits a join event."""

        channel_id = _optional_str(event.get("channel"))
        if channel_id is None:
            return ChannelJoinObservationResult(
                observed=False,
                reason="missing_channel",
            )

        policy = self.ensure_channel_policy(
            installation=installation,
            channel_id=channel_id,
            enabled_by_user_id=_optional_str(event.get("user")),
        )
        if not self.is_observable(policy):
            return ChannelJoinObservationResult(
                observed=False,
                reason="policy_disabled",
                policy=policy,
            )
        if not _should_post_channel_intro(policy):
            return ChannelJoinObservationResult(
                observed=False,
                reason="intro_already_posted",
                policy=policy,
            )

        synthetic_event_id = _synthetic_activation_event_id(body.get("event_id"))
        existing = self._existing_event(installation.id, synthetic_event_id)
        if existing is not None:
            return ChannelJoinObservationResult(
                observed=False,
                reason="duplicate",
                event=existing,
                policy=policy,
            )

        observation = ObservationEvent(
            installation_id=installation.id,
            slack_team_id=slack_team_id,
            channel_id=channel_id,
            user_id=installation.bot_user_id,
            event_type="channel_join",
            slack_event_id=synthetic_event_id,
            message_ts=_optional_str(event.get("ts")),
            thread_ts=_optional_str(event.get("thread_ts")),
            file_id=None,
            raw_payload_checksum=_payload_checksum(
                {
                    "event_type": "implicit_channel_activation",
                    "body": body,
                    "event": event,
                }
            ),
            text_preview=None,
            visibility_metadata={
                "scope_type": "channel",
                "scope_id": channel_id,
                "activated_by_user_id": event.get("user"),
                "activation_source": "app_mention",
                "policy_id": str(policy.id) if policy.id is not None else None,
            },
        )
        try:
            with self.session.begin_nested():
                self.session.add(observation)
                self.session.flush()
        except IntegrityError:
            existing = self._existing_event(installation.id, synthetic_event_id)
            if existing is None:
                raise
            return ChannelJoinObservationResult(
                observed=False,
                reason="duplicate",
                event=existing,
                policy=policy,
            )

        intro_text = (
            "Thanks for adding me here. I can help summarize this channel, "
            "research questions, read shared files, and spot workflows I might "
            "be able to help with. I will keep it low-key."
        )
        logger.info(
            "observe channel activation recorded event_id=%s channel=%s observation_id=%s intro=%s",
            synthetic_event_id,
            channel_id,
            observation.id,
            bool(intro_text),
        )
        return ChannelJoinObservationResult(
            observed=True,
            reason="observed",
            event=observation,
            policy=policy,
            intro_text=intro_text,
        )

    def mark_channel_intro_posted(
        self,
        *,
        policy: ObservePolicy,
        slack_team_id: str,
        channel_id: str,
        message_ts: str | None,
    ) -> ObservationEvent:
        """Mark onboarding as posted and record the intro as an observation event."""

        now = datetime.now(UTC)
        metadata = dict(policy.metadata_json or {})
        metadata["onboarding_intro_posted_at"] = now.isoformat()
        metadata["onboarding_intro_channel_id"] = channel_id
        if message_ts is not None:
            metadata["onboarding_intro_message_ts"] = message_ts
        policy.metadata_json = metadata
        policy.updated_at = now

        observation = ObservationEvent(
            installation_id=policy.installation_id,
            slack_team_id=slack_team_id,
            channel_id=channel_id,
            user_id=None,
            event_type="channel_onboarding_intro",
            slack_event_id=None,
            message_ts=message_ts,
            thread_ts=None,
            file_id=None,
            raw_payload_checksum=_payload_checksum(
                {
                    "event_type": "channel_onboarding_intro",
                    "channel_id": channel_id,
                    "message_ts": message_ts,
                    "posted_at": now.isoformat(),
                }
            ),
            text_preview="Kortny channel onboarding intro posted.",
            visibility_metadata={
                "scope_type": "channel",
                "scope_id": channel_id,
                "policy_id": str(policy.id) if policy.id is not None else None,
            },
        )
        self.session.add(observation)
        self.session.flush()
        return observation

    def ensure_channel_policy(
        self,
        *,
        installation: Installation,
        channel_id: str,
        enabled_by_user_id: str | None,
    ) -> ObservePolicy:
        """Create the default channel policy if it does not already exist."""

        existing = self.session.scalar(
            select(ObservePolicy).where(
                ObservePolicy.installation_id == installation.id,
                ObservePolicy.scope_type == "channel",
                ObservePolicy.scope_id == channel_id,
            )
        )
        if existing is not None:
            return existing

        policy = ObservePolicy(
            installation_id=installation.id,
            scope_type="channel",
            scope_id=channel_id,
            observation_status=DEFAULT_CHANNEL_OBSERVATION_STATUS,
            proactivity_status=DEFAULT_CHANNEL_PROACTIVITY_STATUS,
            retention_days=DEFAULT_RETENTION_DAYS,
            cooldown_seconds=DEFAULT_COOLDOWN_SECONDS,
            enabled_by_user_id=enabled_by_user_id,
            enabled_at=datetime.now(UTC),
            metadata_json={"created_from": "channel_membership"},
        )
        try:
            with self.session.begin_nested():
                self.session.add(policy)
                self.session.flush()
        except IntegrityError:
            existing = self.session.scalar(
                select(ObservePolicy).where(
                    ObservePolicy.installation_id == installation.id,
                    ObservePolicy.scope_type == "channel",
                    ObservePolicy.scope_id == channel_id,
                )
            )
            if existing is None:
                raise
            return existing
        return policy

    def set_channel_proactivity_status(
        self,
        installation_id: UUID,
        channel_id: str,
        status: str,
        *,
        by_user_id: str | None = None,
        now: datetime | None = None,
    ) -> ObservePolicy:
        """Set proactivity_status for a channel policy.

        When transitioning TO 'full' and full_enabled_at is NULL, stamps the
        activation epoch. Leaving 'full' does NOT clear full_enabled_at — the
        original epoch is preserved so re-enabling keeps the original backlog exclusion.
        Valid statuses: 'off', 'digest_only', 'full'.
        """
        valid = {"off", "digest_only", "full"}
        if status not in valid:
            raise ValueError(
                f"Invalid proactivity_status {status!r}; must be one of {valid}"
            )
        observed_now = now or datetime.now(UTC)
        installation = self.session.get(Installation, installation_id)
        if installation is None:
            raise ValueError(f"Installation {installation_id} not found")
        policy = self.ensure_channel_policy(
            installation=installation,
            channel_id=channel_id,
            enabled_by_user_id=by_user_id,
        )
        policy.proactivity_status = status
        if status == "full" and policy.full_enabled_at is None:
            policy.full_enabled_at = observed_now
        policy.updated_at = observed_now
        if by_user_id:
            policy.enabled_by_user_id = by_user_id
        return policy

    def set_digest_delivery(
        self,
        installation_id: UUID,
        *,
        enabled: bool,
        now: datetime | None = None,
    ) -> None:
        """Enable or disable DM digest delivery at workspace level.

        Enabling stamps digest_enabled_at (the epoch) if not already set.
        Disabling clears digest_enabled_at so future re-enables get a fresh epoch
        (unlike full_enabled_at which is preserved — digests are workspace-wide
        so operators expect a clean slate on re-enable).
        """
        observed_now = now or datetime.now(UTC)
        installation = self.session.get(Installation, installation_id)
        if installation is None:
            raise ValueError(f"Installation {installation_id} not found")
        if enabled:
            if installation.digest_enabled_at is None:
                installation.digest_enabled_at = observed_now
        else:
            installation.digest_enabled_at = None
        installation.updated_at = observed_now

    def set_autopilot_enabled(
        self,
        installation_id: UUID,
        *,
        enabled: bool,
        now: datetime | None = None,
    ) -> None:
        """Set DB-level autopilot override for a workspace.

        DB value overrides env when not None. Set to None to fall back to env.
        This method sets True or False (not None — use direct model update to clear).
        """
        observed_now = now or datetime.now(UTC)
        installation = self.session.get(Installation, installation_id)
        if installation is None:
            raise ValueError(f"Installation {installation_id} not found")
        installation.autopilot_enabled = enabled
        installation.updated_at = observed_now

    @staticmethod
    def is_observable(policy: ObservePolicy) -> bool:
        """Return whether a policy allows passive event observation."""

        return policy.observation_status != "off" and policy.paused_at is None

    def _existing_event(
        self,
        installation_id: UUID,
        slack_event_id: object,
    ) -> ObservationEvent | None:
        event_id = _optional_str(slack_event_id)
        if event_id is None:
            return None
        return self.session.scalar(
            select(ObservationEvent).where(
                ObservationEvent.installation_id == installation_id,
                ObservationEvent.slack_event_id == event_id,
            )
        )


def _channel_message_ignore_reason(
    event: dict[str, Any],
    *,
    bot_user_id: str | None,
) -> str | None:
    if event.get("channel_type") == "im":
        return "dm_excluded"
    if _optional_str(event.get("channel")) is None:
        return "missing_channel"
    subtype = event.get("subtype")
    if subtype in {"message_changed", "message_deleted", "channel_leave"}:
        return f"subtype:{subtype}"
    if event.get("bot_id"):
        return "bot_message"
    if subtype == "bot_message":
        return "bot_message"
    if bot_user_id and event.get("user") == bot_user_id:
        return "self_authored"
    return None


def _should_post_channel_intro(policy: ObservePolicy) -> bool:
    if policy.proactivity_status == "off":
        return False
    metadata = policy.metadata_json or {}
    return not bool(metadata.get("onboarding_intro_posted_at"))


def _payload_checksum(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _synthetic_activation_event_id(slack_event_id: object) -> str | None:
    event_id = _optional_str(slack_event_id)
    if event_id is None:
        return None
    return f"{event_id}:implicit_channel_activation"


def _text_preview(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped:
        return None
    redacted = MENTION_RE.sub("<@user>", stripped)
    if len(redacted) <= TEXT_PREVIEW_MAX_CHARS:
        return redacted
    return redacted[: TEXT_PREVIEW_MAX_CHARS - 1].rstrip() + "..."


def _event_files(event: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    raw_files = event.get("files")
    if not isinstance(raw_files, list):
        return ()
    return tuple(file for file in raw_files if isinstance(file, dict))


def _first_file_id(files: tuple[dict[str, Any], ...]) -> str | None:
    for file in files:
        file_id = _optional_str(file.get("id"))
        if file_id is not None:
            return file_id
    return None


def _optional_str(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None
