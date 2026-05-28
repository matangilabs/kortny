"""Durable channel profile projection for Kortny Observe."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from kortny.db.models import (
    ObserveChannelProfile,
    SlackChannelMembership,
    Task,
    TaskEvent,
    TaskEventType,
)

DEFAULT_FRESH_WINDOW_DAYS = 30
DEFAULT_ARCHIVE_WINDOW_DAYS = 365


@dataclass(frozen=True, slots=True)
class ChannelAssessmentStats:
    """Context stats derived from a channel-history tool result."""

    message_count: int
    file_count: int
    observed_range_start_ts: str | None
    observed_range_end_ts: str | None
    last_scanned_message_ts: str | None
    history_event_ids: tuple[int, ...]


class ObserveChannelProfileService:
    """Create or update staleness-aware channel profiles from assessments."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def upsert_from_assessment(
        self,
        *,
        task: Task,
        membership: SlackChannelMembership,
        result_summary: str,
    ) -> ObserveChannelProfile:
        """Persist the latest channel assessment as a durable profile."""

        now = datetime.now(UTC)
        stats = self._assessment_stats(task)
        summary = _bounded_text(result_summary, 8000)
        confidence_score = _confidence_score(stats.message_count)
        confidence_reason = _confidence_reason(stats.message_count)
        profile = self.get(
            installation_id=task.installation_id,
            channel_id=membership.channel_id,
        )
        created = False
        if profile is None:
            profile = ObserveChannelProfile(
                installation_id=task.installation_id,
                channel_id=membership.channel_id,
                profile_status="active",
                profile_version=1,
                created_at=now,
            )
            try:
                with self.session.begin_nested():
                    self.session.add(profile)
                    self.session.flush()
                    created = True
            except IntegrityError:
                profile = self.get(
                    installation_id=task.installation_id,
                    channel_id=membership.channel_id,
                )
                if profile is None:
                    raise

        if not created:
            profile.profile_version += 1

        profile.profile_status = "active"
        profile.summary = summary
        profile.profile_json = _profile_payload(
            summary=summary,
            stats=stats,
            task=task,
        )
        profile.assumptions_json = _assumptions_payload(
            summary=summary,
            message_count=stats.message_count,
        )
        profile.evidence_refs_json = _evidence_refs_payload(
            task=task,
            membership=membership,
            stats=stats,
        )
        profile.confidence_score = confidence_score
        profile.confidence_reason = confidence_reason
        profile.fresh_window_days = DEFAULT_FRESH_WINDOW_DAYS
        profile.archive_window_days = DEFAULT_ARCHIVE_WINDOW_DAYS
        profile.observed_range_start_ts = stats.observed_range_start_ts
        profile.observed_range_end_ts = (
            stats.observed_range_end_ts or task.slack_message_ts
        )
        profile.message_count = stats.message_count
        profile.file_count = stats.file_count
        profile.last_scanned_message_ts = (
            stats.last_scanned_message_ts or task.slack_message_ts
        )
        profile.last_profiled_at = now
        profile.source_task_id = task.id
        profile.metadata_json = {
            "source": "channel_assessment",
            "membership_id": str(membership.id),
            "source_task_id": str(task.id),
            "source_slack_event_id": task.slack_event_id,
            "source_slack_channel_id": task.slack_channel_id,
            "source_slack_thread_ts": task.slack_thread_ts,
            "source_slack_message_ts": task.slack_message_ts,
            "fresh_window_days": DEFAULT_FRESH_WINDOW_DAYS,
            "archive_window_days": DEFAULT_ARCHIVE_WINDOW_DAYS,
            "history_event_ids": list(stats.history_event_ids),
        }
        profile.updated_at = now
        self.session.flush()
        return profile

    def get(
        self,
        *,
        installation_id: object,
        channel_id: str,
    ) -> ObserveChannelProfile | None:
        """Return the current profile for a channel."""

        return self.session.scalar(
            select(ObserveChannelProfile).where(
                ObserveChannelProfile.installation_id == installation_id,
                ObserveChannelProfile.channel_id == channel_id,
            )
        )

    def _assessment_stats(self, task: Task) -> ChannelAssessmentStats:
        history_events = list(
            self.session.scalars(
                select(TaskEvent)
                .where(
                    TaskEvent.task_id == task.id,
                    TaskEvent.type == TaskEventType.tool_result,
                )
                .order_by(TaskEvent.seq)
            )
        )
        messages_by_ts: dict[str, dict[str, Any]] = {}
        history_event_ids: list[int] = []
        for event in history_events:
            if event.payload.get("tool") != "slack_channel_history":
                continue
            output = event.payload.get("output")
            if not isinstance(output, dict):
                continue
            history_event_ids.append(event.id)
            for message in _history_messages(output):
                ts = message.get("ts")
                if not isinstance(ts, str) or not ts:
                    continue
                messages_by_ts[ts] = message

        ts_values = sorted(messages_by_ts, key=_slack_ts_sort_key)
        file_ids: set[str] = set()
        fallback_file_count = 0
        for message in messages_by_ts.values():
            files = message.get("files")
            if not isinstance(files, list):
                continue
            for file in files:
                if not isinstance(file, dict):
                    continue
                file_id = file.get("id")
                if isinstance(file_id, str) and file_id:
                    file_ids.add(file_id)
                else:
                    fallback_file_count += 1

        return ChannelAssessmentStats(
            message_count=len(messages_by_ts),
            file_count=len(file_ids) + fallback_file_count,
            observed_range_start_ts=ts_values[0] if ts_values else None,
            observed_range_end_ts=ts_values[-1] if ts_values else None,
            last_scanned_message_ts=ts_values[-1] if ts_values else None,
            history_event_ids=tuple(history_event_ids),
        )


def _history_messages(output: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    messages = output.get("messages")
    if not isinstance(messages, list):
        return ()
    return tuple(message for message in messages if isinstance(message, dict))


def _profile_payload(
    *,
    summary: str,
    stats: ChannelAssessmentStats,
    task: Task,
) -> dict[str, Any]:
    return {
        "kind": "slack_channel_profile",
        "source": "channel_assessment",
        "summary": summary,
        "fresh_context": {
            "window_days": DEFAULT_FRESH_WINDOW_DAYS,
            "use_for": "current working context and recent channel patterns",
        },
        "archive_context": {
            "window_days": DEFAULT_ARCHIVE_WINDOW_DAYS,
            "use_for": "older files, recurring workflows, and resurfacing lost context",
            "staleness_note": (
                "Older evidence is lower confidence unless reinforced by recent activity."
            ),
        },
        "observed": {
            "channel_id": task.slack_channel_id,
            "message_count": stats.message_count,
            "file_count": stats.file_count,
            "range_start_ts": stats.observed_range_start_ts,
            "range_end_ts": stats.observed_range_end_ts,
            "last_scanned_message_ts": stats.last_scanned_message_ts,
        },
    }


def _assumptions_payload(
    *,
    summary: str,
    message_count: int,
) -> list[dict[str, Any]]:
    if not summary:
        return []
    confidence = "medium" if message_count >= 10 else "low"
    return [
        {
            "type": "channel_purpose",
            "text": _bounded_text(summary, 1000),
            "confidence": confidence,
            "source": "assessment_summary",
            "staleness": "fresh_profile_seed",
        }
    ]


def _evidence_refs_payload(
    *,
    task: Task,
    membership: SlackChannelMembership,
    stats: ChannelAssessmentStats,
) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = [
        {
            "type": "task",
            "task_id": str(task.id),
            "slack_channel_id": task.slack_channel_id,
            "slack_thread_ts": task.slack_thread_ts,
            "slack_message_ts": task.slack_message_ts,
        },
        {
            "type": "membership",
            "membership_id": str(membership.id),
            "channel_id": membership.channel_id,
            "discovered_via": membership.discovered_via,
        },
    ]
    if stats.history_event_ids:
        refs.append(
            {
                "type": "tool_result",
                "tool": "slack_channel_history",
                "task_event_ids": list(stats.history_event_ids),
                "message_count": stats.message_count,
                "file_count": stats.file_count,
                "observed_range_start_ts": stats.observed_range_start_ts,
                "observed_range_end_ts": stats.observed_range_end_ts,
            }
        )
    return refs


def _confidence_score(message_count: int) -> Decimal:
    if message_count >= 30:
        return Decimal("0.750")
    if message_count >= 10:
        return Decimal("0.650")
    if message_count > 0:
        return Decimal("0.450")
    return Decimal("0.250")


def _confidence_reason(message_count: int) -> str:
    if message_count >= 30:
        return "Assessment had a broad recent channel sample."
    if message_count >= 10:
        return "Assessment had enough recent messages for a first-pass profile."
    if message_count > 0:
        return "Assessment had limited channel history; treat assumptions as tentative."
    return "No channel-history tool result was available; profile is only a shell."


def _slack_ts_sort_key(ts: str) -> tuple[Decimal, str]:
    try:
        return (Decimal(ts), ts)
    except InvalidOperation:
        return (Decimal(0), ts)


def _bounded_text(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 1] + "..."
