"""Lifecycle and delivery actions for Witness opportunity candidates."""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session

from kortny.db.models import WitnessOpportunityCandidate
from kortny.slack.formatting import normalize_user_facing_text
from kortny.slack.outbox import SlackSideEffectOutbox

WITNESS_SUGGESTION_PURPOSE = "witness_suggestion"
DEFAULT_WITNESS_SNOOZE = timedelta(days=7)
MAX_WITNESS_AUDIT_HISTORY = 25


class WitnessSlackClient(Protocol):
    """Subset of Slack client needed for Witness delivery."""

    def chat_postMessage(
        self,
        *,
        channel: str,
        text: str,
        thread_ts: str | None = None,
        blocks: list[dict[str, Any]] | None = None,
    ) -> Mapping[str, Any]:
        """Post a Slack message."""


@dataclass(frozen=True, slots=True)
class WitnessDeliveryResult:
    """Result from one Witness delivery attempt."""

    candidate_id: uuid.UUID
    channel_id: str
    message_ts: str
    side_effect_id: uuid.UUID
    deduped: bool


def dismiss_candidate(
    session: Session,
    candidate_id: uuid.UUID,
    *,
    installation_id: uuid.UUID,
    by_user_id: str,
    reason: str | None = None,
) -> WitnessOpportunityCandidate:
    candidate = _candidate_for_update(session, candidate_id, installation_id)
    _ensure_not_archived(candidate)
    _ensure_not_automated(candidate)
    now = datetime.now(UTC)
    candidate.status = "dismissed"
    candidate.cooldown_until = None
    candidate.updated_at = now
    _record_feedback(
        candidate,
        action="dismissed",
        by_user_id=by_user_id,
        now=now,
        details={"reason": _bounded(reason, 240) if reason else None},
    )
    session.flush()
    return candidate


def snooze_candidate(
    session: Session,
    candidate_id: uuid.UUID,
    *,
    installation_id: uuid.UUID,
    by_user_id: str,
    duration: timedelta = DEFAULT_WITNESS_SNOOZE,
) -> WitnessOpportunityCandidate:
    if duration.total_seconds() <= 0:
        raise ValueError("Snooze duration must be positive.")
    candidate = _candidate_for_update(session, candidate_id, installation_id)
    _ensure_not_archived(candidate)
    _ensure_not_automated(candidate)
    now = datetime.now(UTC)
    cooldown_until = now + duration
    candidate.status = "cooldown"
    candidate.cooldown_until = cooldown_until
    candidate.updated_at = now
    _record_feedback(
        candidate,
        action="snoozed",
        by_user_id=by_user_id,
        now=now,
        details={
            "cooldown_until": cooldown_until.isoformat(),
            "duration_seconds": int(duration.total_seconds()),
        },
    )
    session.flush()
    return candidate


def accept_candidate(
    session: Session,
    candidate_id: uuid.UUID,
    *,
    installation_id: uuid.UUID,
    by_user_id: str,
) -> WitnessOpportunityCandidate:
    candidate = _candidate_for_update(session, candidate_id, installation_id)
    _ensure_not_archived(candidate)
    _ensure_not_automated(candidate)
    now = datetime.now(UTC)
    candidate.status = "accepted"
    candidate.cooldown_until = None
    candidate.updated_at = now
    _record_feedback(
        candidate,
        action="accepted",
        by_user_id=by_user_id,
        now=now,
        details={},
    )
    session.flush()
    return candidate


def reactivate_candidate(
    session: Session,
    candidate_id: uuid.UUID,
    *,
    installation_id: uuid.UUID,
    by_user_id: str,
) -> WitnessOpportunityCandidate:
    candidate = _candidate_for_update(session, candidate_id, installation_id)
    if candidate.status == "archived":
        raise ValueError("Archived Witness candidates cannot be reactivated.")
    _ensure_not_automated(candidate)
    now = datetime.now(UTC)
    candidate.status = "candidate"
    candidate.cooldown_until = None
    candidate.updated_at = now
    _record_feedback(
        candidate,
        action="reactivated",
        by_user_id=by_user_id,
        now=now,
        details={},
    )
    session.flush()
    return candidate


def archive_candidate(
    session: Session,
    candidate_id: uuid.UUID,
    *,
    installation_id: uuid.UUID,
    by_user_id: str,
) -> WitnessOpportunityCandidate:
    candidate = _candidate_for_update(session, candidate_id, installation_id)
    now = datetime.now(UTC)
    candidate.status = "archived"
    candidate.cooldown_until = None
    candidate.updated_at = now
    _record_feedback(
        candidate,
        action="archived",
        by_user_id=by_user_id,
        now=now,
        details={},
    )
    session.flush()
    return candidate


def send_private_suggestion(
    session: Session,
    candidate_id: uuid.UUID,
    *,
    installation_id: uuid.UUID,
    by_user_id: str,
    client: WitnessSlackClient,
    now: datetime | None = None,
) -> WitnessDeliveryResult:
    """Send one explicit, DM-scoped Witness suggestion.

    This intentionally refuses channel-scoped candidates. Channel delivery needs
    a stronger interruption policy and should be a later slice.
    """

    candidate = _candidate_for_update(session, candidate_id, installation_id)
    delivered_at = now or datetime.now(UTC)
    _ensure_private_deliverable(candidate, now=delivered_at)
    channel_id = candidate.channel_id
    if channel_id is None:
        raise ValueError("Witness candidate has no Slack DM channel.")

    text = _suggestion_text(candidate)
    request: dict[str, Any] = {
        "channel": channel_id,
        "text": text,
        "thread_ts": None,
    }
    side_effect = SlackSideEffectOutbox(session).deliver(
        installation_id=candidate.installation_id,
        task_id=candidate.source_task_id,
        idempotency_key=f"witness_suggestion:{candidate.id}",
        operation="chat_postMessage",
        purpose=WITNESS_SUGGESTION_PURPOSE,
        target_channel_id=channel_id,
        request=request,
        call=lambda: client.chat_postMessage(
            channel=channel_id,
            text=text,
            thread_ts=None,
        ),
    )
    message_ts = _response_ts(side_effect.response)
    if message_ts is None:
        side_effect.side_effect.status = "failed"
        side_effect.side_effect.last_error = {
            "type": "WitnessDeliveryError",
            "message": "Slack response did not include a message timestamp.",
        }
        side_effect.side_effect.updated_at = datetime.now(UTC)
        session.flush()
        raise ValueError("Slack response did not include a message timestamp.")

    candidate.status = "sent"
    candidate.cooldown_until = None
    candidate.last_suggested_at = delivered_at
    candidate.updated_at = delivered_at
    _record_feedback(
        candidate,
        action="sent",
        by_user_id=by_user_id,
        now=delivered_at,
        details={
            "channel_id": channel_id,
            "message_ts": message_ts,
            "side_effect_id": str(side_effect.side_effect.id),
            "deduped": side_effect.deduped,
            "delivery_policy": "explicit_dm_only",
        },
    )
    session.flush()
    return WitnessDeliveryResult(
        candidate_id=candidate.id,
        channel_id=channel_id,
        message_ts=message_ts,
        side_effect_id=side_effect.side_effect.id,
        deduped=side_effect.deduped,
    )


def _candidate_for_update(
    session: Session,
    candidate_id: uuid.UUID,
    installation_id: uuid.UUID,
) -> WitnessOpportunityCandidate:
    candidate = session.scalar(
        select(WitnessOpportunityCandidate)
        .where(
            WitnessOpportunityCandidate.id == candidate_id,
            WitnessOpportunityCandidate.installation_id == installation_id,
        )
        .limit(1)
        .with_for_update()
    )
    if candidate is None:
        raise LookupError(f"Witness candidate not found: {candidate_id}")
    return candidate


def _ensure_not_archived(candidate: WitnessOpportunityCandidate) -> None:
    if candidate.status == "archived":
        raise ValueError("This Witness candidate is archived.")


def _ensure_not_automated(candidate: WitnessOpportunityCandidate) -> None:
    if candidate.status == "automated":
        raise ValueError("This Witness candidate already became a standing automation.")


def _ensure_private_deliverable(
    candidate: WitnessOpportunityCandidate,
    *,
    now: datetime,
) -> None:
    if candidate.status != "candidate":
        raise ValueError("Only active Witness candidates can be sent.")
    if candidate.cooldown_until is not None and candidate.cooldown_until > now:
        raise ValueError("This Witness candidate is still cooling down.")
    if candidate.visibility_scope_type != "dm":
        raise ValueError("Only DM-scoped Witness candidates can be sent right now.")
    if not candidate.channel_id or not candidate.channel_id.startswith("D"):
        raise ValueError("Witness candidate does not have a Slack DM channel.")
    if not candidate.target_slack_user_id:
        raise ValueError("Witness candidate does not have a target Slack user.")


def _suggestion_text(candidate: WitnessOpportunityCandidate) -> str:
    text = candidate.suggested_message or (
        "I noticed something that might be worth keeping an eye on: "
        f"{candidate.summary}"
    )
    return normalize_user_facing_text(_bounded(text, 1800))


def _record_feedback(
    candidate: WitnessOpportunityCandidate,
    *,
    action: str,
    by_user_id: str,
    now: datetime,
    details: Mapping[str, Any],
) -> None:
    feedback = dict(candidate.feedback_json or {})
    history_value = feedback.get("history")
    history = list(history_value) if isinstance(history_value, list) else []
    entry = {
        "action": action,
        "by_user_id": by_user_id,
        "at": now.isoformat(),
        **{key: value for key, value in details.items() if value is not None},
    }
    history.append(entry)
    feedback["history"] = history[-MAX_WITNESS_AUDIT_HISTORY:]
    feedback["last_action"] = entry
    candidate.feedback_json = feedback


def _response_ts(response: Mapping[str, Any]) -> str | None:
    value = response.get("ts")
    if isinstance(value, str) and value:
        return value
    value = response.get("message")
    if isinstance(value, Mapping):
        message_ts = value.get("ts")
        if isinstance(message_ts, str) and message_ts:
            return message_ts
    return None


def _bounded(value: str, max_chars: int) -> str:
    return " ".join(value.split()).strip()[:max_chars].strip()
