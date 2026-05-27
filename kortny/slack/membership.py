"""Slack channel membership projection for Kortny."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from kortny.db.models import Installation, SlackChannelMembership

ONBOARDING_TRIGGER_SOURCES = frozenset({"member_joined_channel", "app_mention"})
SOURCE_PRIORITY = {
    "message_observation": 0,
    "channel_history": 1,
    "manual_backfill": 1,
    "app_mention": 2,
    "member_joined_channel": 3,
}


@dataclass(frozen=True, slots=True)
class ChannelMembershipResult:
    """Outcome of recording Kortny's channel presence."""

    membership: SlackChannelMembership
    created: bool
    onboarding_due: bool
    reason: str


class SlackChannelMembershipService:
    """Maintain a durable projection of Slack channels Kortny can see."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def record_seen_channel(
        self,
        *,
        installation: Installation,
        channel_id: str,
        discovered_via: str,
        channel_type: str | None = None,
        channel_name: str | None = None,
        added_by_user_id: str | None = None,
        event_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ChannelMembershipResult:
        """Insert or update a known channel membership row."""

        now = datetime.now(UTC)
        current = self.get(installation=installation, channel_id=channel_id)
        if current is None:
            membership = SlackChannelMembership(
                installation_id=installation.id,
                channel_id=channel_id,
                channel_name=channel_name,
                channel_type=channel_type,
                membership_status="active",
                discovered_via=discovered_via,
                added_by_user_id=added_by_user_id,
                first_seen_at=now,
                last_seen_at=now,
                onboarding_status="pending",
                last_event_id=event_id,
                metadata_json=metadata or {},
            )
            try:
                with self.session.begin_nested():
                    self.session.add(membership)
                    self.session.flush()
            except IntegrityError:
                current = self.get(installation=installation, channel_id=channel_id)
                if current is None:
                    raise
                self._update_membership(
                    current,
                    now=now,
                    discovered_via=discovered_via,
                    channel_type=channel_type,
                    channel_name=channel_name,
                    added_by_user_id=added_by_user_id,
                    event_id=event_id,
                    metadata=metadata,
                )
                return self._result(current, created=False, discovered_via=discovered_via)

            return self._result(
                membership,
                created=True,
                discovered_via=discovered_via,
            )

        self._update_membership(
            current,
            now=now,
            discovered_via=discovered_via,
            channel_type=channel_type,
            channel_name=channel_name,
            added_by_user_id=added_by_user_id,
            event_id=event_id,
            metadata=metadata,
        )
        return self._result(current, created=False, discovered_via=discovered_via)

    def mark_onboarding_posted(
        self,
        *,
        membership: SlackChannelMembership,
        message_ts: str | None,
    ) -> None:
        """Record that the channel intro has already been posted."""

        now = datetime.now(UTC)
        membership.onboarding_status = "posted"
        membership.onboarding_posted_at = now
        membership.onboarding_message_ts = message_ts
        membership.updated_at = now
        self.session.flush()

    def mark_assessment_queued(
        self,
        *,
        membership: SlackChannelMembership,
        task_id: UUID,
    ) -> None:
        """Record that the channel assessment follow-up has been queued."""

        now = datetime.now(UTC)
        metadata = dict(membership.metadata_json or {})
        metadata["assessment_task_id"] = str(task_id)
        metadata["assessment_status"] = "queued"
        metadata["assessment_requested_at"] = now.isoformat()
        membership.metadata_json = metadata
        membership.updated_at = now
        self.session.flush()

    def mark_assessment_completed(
        self,
        *,
        membership: SlackChannelMembership,
        result_summary: str,
    ) -> None:
        """Record that the channel assessment task completed successfully."""

        now = datetime.now(UTC)
        metadata = dict(membership.metadata_json or {})
        metadata["assessment_status"] = "posted"
        metadata["assessment_completed_at"] = now.isoformat()
        metadata["assessment_summary"] = result_summary[:2000]
        membership.metadata_json = metadata
        membership.updated_at = now
        self.session.flush()

    def mark_assessment_failed(
        self,
        *,
        membership: SlackChannelMembership,
        error_type: str,
        error: str,
    ) -> None:
        """Record that the channel assessment task failed."""

        now = datetime.now(UTC)
        metadata = dict(membership.metadata_json or {})
        metadata["assessment_status"] = "failed"
        metadata["assessment_failed_at"] = now.isoformat()
        metadata["assessment_error_type"] = error_type
        metadata["assessment_error"] = error[:1000]
        membership.metadata_json = metadata
        membership.updated_at = now
        self.session.flush()

    def find_by_assessment_task_id(
        self,
        *,
        task_id: UUID,
    ) -> SlackChannelMembership | None:
        """Return the channel membership tied to an assessment task."""

        return self.session.scalar(
            select(SlackChannelMembership).where(
                SlackChannelMembership.metadata_json["assessment_task_id"].as_string()
                == str(task_id)
            )
        )

    def get(
        self,
        *,
        installation: Installation,
        channel_id: str,
    ) -> SlackChannelMembership | None:
        return self.session.scalar(
            select(SlackChannelMembership).where(
                SlackChannelMembership.installation_id == installation.id,
                SlackChannelMembership.channel_id == channel_id,
            )
        )

    def _update_membership(
        self,
        membership: SlackChannelMembership,
        *,
        now: datetime,
        discovered_via: str,
        channel_type: str | None,
        channel_name: str | None,
        added_by_user_id: str | None,
        event_id: str | None,
        metadata: dict[str, Any] | None,
    ) -> None:
        membership.membership_status = "active"
        membership.last_seen_at = now
        membership.updated_at = now
        if self._should_replace_source(membership.discovered_via, discovered_via):
            membership.discovered_via = discovered_via
        if channel_type is not None:
            membership.channel_type = channel_type
        if channel_name is not None:
            membership.channel_name = channel_name
        if added_by_user_id is not None and membership.added_by_user_id is None:
            membership.added_by_user_id = added_by_user_id
        if event_id is not None:
            membership.last_event_id = event_id
        if metadata:
            merged_metadata = dict(membership.metadata_json or {})
            merged_metadata.update(metadata)
            membership.metadata_json = merged_metadata
        self.session.flush()

    def _result(
        self,
        membership: SlackChannelMembership,
        *,
        created: bool,
        discovered_via: str,
    ) -> ChannelMembershipResult:
        if membership.onboarding_status != "pending":
            return ChannelMembershipResult(
                membership=membership,
                created=created,
                onboarding_due=False,
                reason=f"onboarding_{membership.onboarding_status}",
            )
        if discovered_via not in ONBOARDING_TRIGGER_SOURCES:
            return ChannelMembershipResult(
                membership=membership,
                created=created,
                onboarding_due=False,
                reason="seen_without_onboarding",
            )
        return ChannelMembershipResult(
            membership=membership,
            created=created,
            onboarding_due=True,
            reason="onboarding_due",
        )

    @staticmethod
    def _should_replace_source(current: str, candidate: str) -> bool:
        return SOURCE_PRIORITY.get(candidate, 0) > SOURCE_PRIORITY.get(current, 0)
