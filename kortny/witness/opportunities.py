"""Evidence-backed Witness opportunity candidates.

This module intentionally stops at candidate persistence. Delivery, feedback UI,
and public proactive posting are separate policy slices.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from kortny.db.models import (
    ObserveChannelProfile,
    SlackChannelMembership,
    Task,
    WitnessOpportunityCandidate,
)

WITNESS_OPPORTUNITY_CANDIDATES_PROJECTED_MESSAGE = (
    "witness_opportunity_candidates_projected"
)
MAX_PROFILE_OPPORTUNITIES = 5
ELIGIBLE_STATUSES = ("candidate",)
ALLOWED_CANDIDATE_TYPES = frozenset(
    (
        "workflow_gap",
        "artifact_followup",
        "unresolved_decision",
        "data_quality_issue",
        "recurring_check",
        "project_status_gap",
        "general_help",
    )
)

_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class WitnessOpportunityCandidateResult:
    """Result from projecting candidate rows from one source."""

    created_count: int
    updated_count: int
    skipped_count: int
    candidate_ids: tuple[str, ...]

    @property
    def total_count(self) -> int:
        return self.created_count + self.updated_count


@dataclass(frozen=True, slots=True)
class WitnessOpportunityCandidateInput:
    """LLM-proposed candidate that has not yet been persisted."""

    candidate_type: str
    title: str
    summary: str
    suggested_action: str | None
    suggested_message: str | None
    evidence: tuple[str, ...]
    confidence_score: Decimal
    confidence_reason: str
    metadata_json: dict[str, Any]


class WitnessOpportunityService:
    """Create and query proactive opportunity candidates."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def project_from_channel_profile(
        self,
        *,
        task: Task,
        membership: SlackChannelMembership,
        profile: ObserveChannelProfile,
        candidates: tuple[WitnessOpportunityCandidateInput, ...],
        extraction_metadata: dict[str, Any] | None = None,
    ) -> WitnessOpportunityCandidateResult:
        """Create/update candidates proposed from a channel profile."""

        valid_candidates = tuple(
            candidate
            for candidate in candidates
            if candidate.candidate_type in ALLOWED_CANDIDATE_TYPES
            and candidate.title.strip()
            and candidate.summary.strip()
        )
        if not valid_candidates:
            return WitnessOpportunityCandidateResult(
                created_count=0,
                updated_count=0,
                skipped_count=1,
                candidate_ids=(),
            )

        now = datetime.now(UTC)
        scope_type, scope_id = _scope_for_membership(membership)
        channel_label = (
            f"#{membership.channel_name}"
            if membership.channel_name
            else membership.channel_id
        )

        created_count = 0
        updated_count = 0
        skipped_count = 0
        candidate_ids: list[str] = []

        for candidate_input in valid_candidates[:MAX_PROFILE_OPPORTUNITIES]:
            candidate_type = candidate_input.candidate_type
            dedupe_key = _dedupe_key(
                channel_id=membership.channel_id,
                candidate_type=candidate_type,
                opportunity=f"{candidate_input.title}:{candidate_input.summary}",
            )
            existing = self._find_existing(
                installation_id=task.installation_id,
                scope_type=scope_type,
                scope_id=scope_id,
                candidate_type=candidate_type,
                dedupe_key=dedupe_key,
            )
            title = _bounded_text(candidate_input.title, 140)
            summary = _bounded_text(candidate_input.summary, 1000)
            suggested_action = (
                _bounded_text(candidate_input.suggested_action, 500)
                if candidate_input.suggested_action
                else _suggested_action(summary)
            )
            suggested_message = (
                _bounded_text(candidate_input.suggested_message, 500)
                if candidate_input.suggested_message
                else _suggested_message_for_label(summary, channel_label=channel_label)
            )
            metadata = {
                "source": "llm_channel_profile_extractor",
                "profile_version": profile.profile_version,
                "channel_name": membership.channel_name,
                "message_count": profile.message_count,
                "file_count": profile.file_count,
                "observed_range_start_ts": profile.observed_range_start_ts,
                "observed_range_end_ts": profile.observed_range_end_ts,
                **(extraction_metadata or {}),
                **candidate_input.metadata_json,
            }
            evidence_items = _channel_profile_candidate_evidence_items(
                task=task,
                membership=membership,
                profile=profile,
                candidate=candidate_input,
            )
            if existing is None:
                candidate = WitnessOpportunityCandidate(
                    installation_id=task.installation_id,
                    channel_id=membership.channel_id,
                    target_slack_user_id=None,
                    visibility_scope_type=scope_type,
                    visibility_scope_id=scope_id,
                    candidate_type=candidate_type,
                    title=title,
                    summary=summary,
                    suggested_action=suggested_action,
                    suggested_message=suggested_message,
                    evidence_json=evidence_items,
                    source_type="channel_profile",
                    source_id=str(profile.id),
                    source_task_id=task.id,
                    source_profile_id=profile.id,
                    dedupe_key=dedupe_key,
                    confidence_score=_bounded_confidence(
                        candidate_input.confidence_score
                    ),
                    confidence_reason=_bounded_text(
                        candidate_input.confidence_reason,
                        500,
                    ),
                    status="candidate",
                    feedback_json={},
                    metadata_json=metadata,
                    created_at=now,
                    updated_at=now,
                )
                self.session.add(candidate)
                self.session.flush()
                created_count += 1
            else:
                candidate = existing
                candidate.title = title
                candidate.summary = summary
                candidate.suggested_action = suggested_action
                candidate.suggested_message = suggested_message
                candidate.evidence_json = evidence_items
                candidate.source_id = str(profile.id)
                candidate.source_task_id = task.id
                candidate.source_profile_id = profile.id
                candidate.confidence_score = max(
                    candidate.confidence_score or Decimal("0.000"),
                    _bounded_confidence(candidate_input.confidence_score),
                )
                candidate.confidence_reason = (
                    _bounded_text(candidate_input.confidence_reason, 500)
                    or "Reinforced by the Witness channel profile extractor."
                )
                candidate.metadata_json = {
                    **(candidate.metadata_json or {}),
                    **metadata,
                    "last_reinforced_at": now.isoformat(),
                }
                candidate.updated_at = now
                self.session.flush()
                updated_count += 1

            candidate_ids.append(str(candidate.id))

        self.session.flush()
        return WitnessOpportunityCandidateResult(
            created_count=created_count,
            updated_count=updated_count,
            skipped_count=skipped_count,
            candidate_ids=tuple(candidate_ids),
        )

    def project_from_task_candidates(
        self,
        *,
        task: Task,
        candidates: tuple[WitnessOpportunityCandidateInput, ...],
        response_text: str,
        extraction_metadata: dict[str, Any] | None = None,
    ) -> WitnessOpportunityCandidateResult:
        """Create/update candidates proposed by the Witness extractor."""

        valid_candidates = tuple(
            candidate
            for candidate in candidates
            if candidate.candidate_type in ALLOWED_CANDIDATE_TYPES
            and candidate.title.strip()
            and candidate.summary.strip()
        )
        if not valid_candidates:
            return WitnessOpportunityCandidateResult(
                created_count=0,
                updated_count=0,
                skipped_count=1,
                candidate_ids=(),
            )

        membership = _membership_for_task(self.session, task)
        scope_type, scope_id = _scope_for_task(task, membership)
        if scope_id is None:
            return WitnessOpportunityCandidateResult(
                created_count=0,
                updated_count=0,
                skipped_count=1,
                candidate_ids=(),
            )

        now = datetime.now(UTC)
        channel_label = _channel_label(task, membership)
        channel_id = task.slack_channel_id or scope_id
        created_count = 0
        updated_count = 0
        candidate_ids: list[str] = []

        for candidate_input in valid_candidates:
            candidate_type = candidate_input.candidate_type
            dedupe_key = _dedupe_key(
                channel_id=channel_id,
                candidate_type=candidate_type,
                opportunity=f"{candidate_input.title}:{candidate_input.summary}",
            )
            existing = self._find_existing(
                installation_id=task.installation_id,
                scope_type=scope_type,
                scope_id=scope_id,
                candidate_type=candidate_type,
                dedupe_key=dedupe_key,
            )
            title = _bounded_text(candidate_input.title, 140)
            summary = _bounded_text(candidate_input.summary, 1000)
            suggested_action = (
                _bounded_text(candidate_input.suggested_action, 500)
                if candidate_input.suggested_action
                else _suggested_action(summary)
            )
            suggested_message = (
                _bounded_text(candidate_input.suggested_message, 500)
                if candidate_input.suggested_message
                else _suggested_message_for_label(summary, channel_label=channel_label)
            )
            metadata = {
                "source": "llm_task_response_extractor",
                "channel_name": membership.channel_name if membership else None,
                "input": _bounded_text(task.input, 280),
                "response_chars": len(response_text),
                **(extraction_metadata or {}),
                **candidate_input.metadata_json,
            }
            evidence_items = _candidate_evidence_items(
                task=task,
                candidate=candidate_input,
                response_text=response_text,
                channel_id=channel_id,
            )
            if existing is None:
                candidate = WitnessOpportunityCandidate(
                    installation_id=task.installation_id,
                    channel_id=task.slack_channel_id,
                    target_slack_user_id=(
                        task.slack_user_id if scope_type == "dm" else None
                    ),
                    visibility_scope_type=scope_type,
                    visibility_scope_id=scope_id,
                    candidate_type=candidate_type,
                    title=title,
                    summary=summary,
                    suggested_action=suggested_action,
                    suggested_message=suggested_message,
                    evidence_json=evidence_items,
                    source_type="task_summary",
                    source_id=str(task.id),
                    source_task_id=task.id,
                    source_profile_id=None,
                    dedupe_key=dedupe_key,
                    confidence_score=_bounded_confidence(
                        candidate_input.confidence_score
                    ),
                    confidence_reason=_bounded_text(
                        candidate_input.confidence_reason,
                        500,
                    ),
                    status="candidate",
                    feedback_json={},
                    metadata_json=metadata,
                    created_at=now,
                    updated_at=now,
                )
                self.session.add(candidate)
                self.session.flush()
                created_count += 1
            else:
                candidate = existing
                candidate.title = title
                candidate.summary = summary
                candidate.suggested_action = suggested_action
                candidate.suggested_message = suggested_message
                candidate.evidence_json = evidence_items
                if candidate.source_type == "task_summary":
                    candidate.source_id = str(task.id)
                candidate.source_task_id = task.id
                candidate.confidence_score = max(
                    candidate.confidence_score or Decimal("0.000"),
                    _bounded_confidence(candidate_input.confidence_score),
                )
                candidate.confidence_reason = (
                    _bounded_text(candidate_input.confidence_reason, 500)
                    or "Reinforced by the Witness extractor."
                )
                candidate.metadata_json = {
                    **(candidate.metadata_json or {}),
                    **metadata,
                    "last_reinforced_at": now.isoformat(),
                }
                candidate.updated_at = now
                self.session.flush()
                updated_count += 1

            candidate_ids.append(str(candidate.id))

        self.session.flush()
        return WitnessOpportunityCandidateResult(
            created_count=created_count,
            updated_count=updated_count,
            skipped_count=0,
            candidate_ids=tuple(candidate_ids),
        )

    def eligible_private_suggestions(
        self,
        *,
        installation_id: uuid.UUID,
        limit: int = 20,
        now: datetime | None = None,
    ) -> tuple[WitnessOpportunityCandidate, ...]:
        """Return currently eligible candidates for a future private DM sender."""

        observed_now = now or datetime.now(UTC)
        rows = self.session.scalars(
            select(WitnessOpportunityCandidate)
            .where(
                WitnessOpportunityCandidate.installation_id == installation_id,
                WitnessOpportunityCandidate.status.in_(ELIGIBLE_STATUSES),
                or_(
                    WitnessOpportunityCandidate.cooldown_until.is_(None),
                    WitnessOpportunityCandidate.cooldown_until <= observed_now,
                ),
            )
            .order_by(
                WitnessOpportunityCandidate.confidence_score.desc(),
                WitnessOpportunityCandidate.created_at.asc(),
            )
            .limit(limit)
        )
        return tuple(rows)

    def _find_existing(
        self,
        *,
        installation_id: uuid.UUID,
        scope_type: str,
        scope_id: str | None,
        candidate_type: str,
        dedupe_key: str,
    ) -> WitnessOpportunityCandidate | None:
        return self.session.scalar(
            select(WitnessOpportunityCandidate).where(
                WitnessOpportunityCandidate.installation_id == installation_id,
                WitnessOpportunityCandidate.visibility_scope_type == scope_type,
                WitnessOpportunityCandidate.visibility_scope_id == scope_id,
                WitnessOpportunityCandidate.candidate_type == candidate_type,
                WitnessOpportunityCandidate.dedupe_key == dedupe_key,
            )
        )


def _channel_profile_candidate_evidence_items(
    *,
    task: Task,
    membership: SlackChannelMembership,
    profile: ObserveChannelProfile,
    candidate: WitnessOpportunityCandidateInput,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = [
        {
            "type": "channel_profile",
            "profile_id": str(profile.id),
            "profile_version": profile.profile_version,
            "source_task_id": str(task.id),
            "channel_id": membership.channel_id,
            "snippet": _bounded_text(candidate.summary, 300),
            "profile_summary": _bounded_text(profile.summary or "", 500),
        }
    ]
    for snippet in candidate.evidence[:5]:
        bounded = _bounded_text(snippet, 300)
        if not bounded:
            continue
        items.append(
            {
                "type": "llm_evidence",
                "snippet": bounded,
                "profile_id": str(profile.id),
                "channel_id": membership.channel_id,
            }
        )
    if isinstance(profile.evidence_refs_json, list):
        for ref in profile.evidence_refs_json[:5]:
            if isinstance(ref, dict):
                items.append({"type": "profile_ref", **ref})
    return items[:10]


def _scope_for_membership(membership: SlackChannelMembership) -> tuple[str, str]:
    channel_type = (membership.channel_type or "").lower()
    if channel_type in {"private_channel", "group", "mpim"}:
        return "private_channel", membership.channel_id
    return "channel", membership.channel_id


def _membership_for_task(
    session: Session,
    task: Task,
) -> SlackChannelMembership | None:
    if not task.slack_channel_id or task.slack_channel_id.startswith("D"):
        return None
    return session.scalar(
        select(SlackChannelMembership).where(
            SlackChannelMembership.installation_id == task.installation_id,
            SlackChannelMembership.channel_id == task.slack_channel_id,
        )
    )


def _scope_for_task(
    task: Task,
    membership: SlackChannelMembership | None,
) -> tuple[str, str | None]:
    if membership is not None:
        return _scope_for_membership(membership)
    if task.slack_channel_id and task.slack_channel_id.startswith("D"):
        return "dm", task.slack_channel_id
    if task.slack_channel_id:
        return "channel", task.slack_channel_id
    if task.slack_user_id:
        return "user", task.slack_user_id
    return "workspace", None


def _channel_label(
    task: Task,
    membership: SlackChannelMembership | None,
) -> str:
    if membership is not None and membership.channel_name:
        return f"#{membership.channel_name}"
    if task.slack_channel_id and task.slack_channel_id.startswith("D"):
        return "this DM"
    if task.slack_channel_id:
        return task.slack_channel_id
    return "this workspace"


def _dedupe_key(
    *,
    channel_id: str,
    candidate_type: str,
    opportunity: str,
) -> str:
    normalized = _normalize_for_key(f"{channel_id}:{candidate_type}:{opportunity}")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:32]


def _normalize_for_key(value: str) -> str:
    return _WHITESPACE_RE.sub(" ", value.strip().lower())


def _suggested_action(opportunity: str) -> str:
    return _bounded_text(f"Offer help with: {opportunity}", 500)


def _suggested_message_for_label(opportunity: str, *, channel_label: str) -> str:
    return _bounded_text(
        f"I noticed {channel_label} may benefit from help with {opportunity}. "
        "Want me to take a pass?",
        500,
    )


def _candidate_evidence_items(
    *,
    task: Task,
    candidate: WitnessOpportunityCandidateInput,
    response_text: str,
    channel_id: str,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = [
        {
            "type": "task_response",
            "source_task_id": str(task.id),
            "channel_id": channel_id,
            "snippet": _bounded_text(candidate.summary, 300),
        },
    ]
    for snippet in candidate.evidence[:5]:
        bounded = _bounded_text(snippet, 300)
        if bounded:
            items.append(
                {
                    "type": "llm_evidence",
                    "source_task_id": str(task.id),
                    "channel_id": channel_id,
                    "snippet": bounded,
                }
            )
    items.append(
        {
            "type": "task_response_context",
            "source_task_id": str(task.id),
            "channel_id": channel_id,
            "summary": _bounded_text(response_text, 700),
        }
    )
    return items[:10]


def _bounded_text(value: str, max_chars: int) -> str:
    return _WHITESPACE_RE.sub(" ", value).strip()[:max_chars].strip()


def _bounded_confidence(value: Decimal) -> Decimal:
    if value < 0:
        return Decimal("0.000")
    if value > 1:
        return Decimal("1.000")
    return value.quantize(Decimal("0.001"))
