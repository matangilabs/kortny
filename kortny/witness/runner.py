"""Operational Witness runner.

The runner coordinates existing Witness primitives:

- active channel profiles are re-read by the LLM extractor;
- candidate rows are persisted through the opportunity service;
- optional delivery is restricted to DM-scoped candidates by lifecycle policy.

Delivery (HIG-227) is digest-based: each due DM-scoped candidate gets one of
four decisions — notify / question / draft / silent — scored by
``effective_confidence * receptivity`` against a threshold. Deliverable
candidates batch into ONE digest DM per user per digest window (hard budget),
sent through the Slack outbox with a per-window idempotency key. Every
decision, including silence, lands in ``witness_delivery_log``.
"""

from __future__ import annotations

import argparse
import logging
import os
import socket
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any, cast

from slack_sdk import WebClient
from sqlalchemy import and_, desc, func, select
from sqlalchemy.orm import Session, sessionmaker

from kortny.config import Settings, load_settings
from kortny.db.models import (
    LLMProvider as DbLLMProvider,
)
from kortny.db.models import (
    ObserveChannelProfile,
    ObservePolicy,
    SlackChannelMembership,
    Task,
    TaskEventType,
    TaskStatus,
    WitnessDeliveryLog,
    WitnessOpportunityCandidate,
)
from kortny.db.session import make_session_factory
from kortny.llm import LLMProvider, LLMService, ModelRoute, ModelRouter, ModelRouteTier
from kortny.llm.runtime_config import (
    create_provider_for_selection,
    select_runtime_model,
)
from kortny.logging_config import configure_logging
from kortny.observability import configure_tracing, start_span
from kortny.slack.formatting import normalize_user_facing_text
from kortny.slack.outbox import SlackSideEffectOutbox
from kortny.tasks import TaskService
from kortny.witness.autopilot import (
    DEFAULT_WITNESS_AUTOPILOT_LIMIT,
    DEFAULT_WITNESS_AUTOPILOT_MIN_CONFIDENCE,
    WitnessAutopilot,
    WitnessAutopilotOutcome,
)
from kortny.witness.extractor import WitnessChannelProfileExtractor
from kortny.witness.lifecycle import (
    WITNESS_CHANNEL_SUGGESTION_PURPOSE,
    WitnessSlackClient,
    _record_feedback,
)
from kortny.witness.opportunities import (
    WITNESS_OPPORTUNITY_CANDIDATES_PROJECTED_MESSAGE,
    WitnessOpportunityService,
    candidate_delivery_decision,
    candidate_thread_ts,
    recurrence_evidence_line,
    recurrence_is_proven,
)
from kortny.witness.receptivity import (
    collect_channel_feedback_events,
    collect_user_feedback_events,
    effective_confidence,
    receptivity,
)

logger = logging.getLogger(__name__)

DEFAULT_WITNESS_PROFILE_SCAN_LIMIT = 10
DEFAULT_WITNESS_DELIVERY_LIMIT = 5
DEFAULT_WITNESS_SCAN_INTERVAL = timedelta(hours=6)
DEFAULT_WITNESS_POLL_INTERVAL_SECONDS = 300.0
DEFAULT_WITNESS_ADVISORY_LOCK_KEY = 759340186

DEFAULT_WITNESS_DELIVERY_THRESHOLD = Decimal("0.55")
DEFAULT_WITNESS_DIGEST_INTERVAL = timedelta(hours=24)
DEFAULT_WITNESS_DIGEST_MAX_ITEMS = 5
# Unproven recurrence (HIG-197 framing gate failed) lowers receptivity at the
# delivery decision instead of claiming recurrence in copy.
UNPROVEN_RECURRENCE_RECEPTIVITY_FACTOR = 0.8
WITNESS_DIGEST_CANDIDATE_SCAN_LIMIT = 50
WITNESS_DIGEST_PURPOSE = "witness_digest"
WITNESS_DIGEST_MAX_CHARS = 3500

# Channel delivery (HIG-198). Channel posting is the most abusable surface in
# the product: every channel delivery passes per-channel policy opt-in
# (ObservePolicy proactivity_status == "full") AND the receptivity threshold
# AND the channel budget AND quiet hours. Failed gates defer, never drop.
DEFAULT_WITNESS_CHANNEL_POSTS_PER_WEEK = 1
WITNESS_CHANNEL_POST_WINDOW = timedelta(days=7)
WITNESS_CHANNEL_CANDIDATE_SCAN_LIMIT = 50
WITNESS_CHANNEL_SUGGESTION_MAX_CHARS = 1800
# Per-channel rows in witness_delivery_log key slack_user_id as
# "channel:{channel_id}" so sliding budget windows stay queryable.
WITNESS_CHANNEL_LOG_USER_PREFIX = "channel:"

WITNESS_RUNNER_PROFILE_SCAN_STARTED_MESSAGE = "witness_runner_profile_scan_started"
WITNESS_RUNNER_PROFILE_SCAN_COMPLETED_MESSAGE = "witness_runner_profile_scan_completed"
WITNESS_RUNNER_PROFILE_SCAN_FAILED_MESSAGE = "witness_runner_profile_scan_failed"
WITNESS_RUNNER_DELIVERY_SKIPPED_MESSAGE = "witness_runner_delivery_skipped"
WITNESS_RUNNER_DELIVERY_SENT_MESSAGE = "witness_runner_delivery_sent"
WITNESS_RUNNER_DIGEST_SENT_MESSAGE = "witness_runner_digest_sent"
WITNESS_RUNNER_CHANNEL_POST_SENT_MESSAGE = "witness_runner_channel_post_sent"


@dataclass(frozen=True, slots=True)
class WitnessProjectionOutcome:
    """Projection result for one channel profile."""

    task_id: uuid.UUID
    channel_id: str
    profile_id: uuid.UUID
    created_count: int
    updated_count: int
    skipped_count: int
    candidate_ids: tuple[str, ...]
    raw_candidate_count: int
    skipped_reason: str | None

    @property
    def total_count(self) -> int:
        return self.created_count + self.updated_count


@dataclass(frozen=True, slots=True)
class WitnessDeliveryOutcome:
    """Delivery decision result for one candidate.

    ``status`` is one of: sent, silent, budget_deferred, interval_deferred,
    quiet_hours_deferred, policy_deferred, window_deduped, skipped.
    """

    candidate_id: uuid.UUID
    status: str
    reason: str | None = None
    channel_id: str | None = None
    message_ts: str | None = None
    decision: str | None = None
    score: float | None = None


@dataclass(frozen=True, slots=True)
class WitnessRunResult:
    """Outcome from one Witness runner tick."""

    runner_id: str
    status: str
    projections: tuple[WitnessProjectionOutcome, ...] = ()
    deliveries: tuple[WitnessDeliveryOutcome, ...] = ()
    autopilot_outcomes: tuple[WitnessAutopilotOutcome, ...] = ()
    leader_acquired: bool = True

    @property
    def projected_count(self) -> int:
        return sum(outcome.total_count for outcome in self.projections)

    @property
    def delivered_count(self) -> int:
        return sum(1 for outcome in self.deliveries if outcome.status == "sent")

    @property
    def autopilot_reviewed_count(self) -> int:
        return len(self.autopilot_outcomes)

    @property
    def autopilot_executed_count(self) -> int:
        return sum(
            1 for outcome in self.autopilot_outcomes if outcome.status == "executed"
        )


class WitnessRunner:
    """Project and optionally deliver conservative Witness suggestions."""

    def __init__(
        self,
        session: Session,
        *,
        settings: Settings | None = None,
        llm_provider: LLMProvider | None = None,
        provider_name: DbLLMProvider | str | None = None,
        slack_client: WitnessSlackClient | None = None,
        runner_id: str | None = None,
        advisory_lock_key: int = DEFAULT_WITNESS_ADVISORY_LOCK_KEY,
    ) -> None:
        self.session = session
        self.settings = settings
        self.llm_provider = llm_provider
        self.provider_name = provider_name
        self.slack_client = slack_client
        self.runner_id = runner_id or default_witness_runner_id()
        self.advisory_lock_key = advisory_lock_key

    def run_once(
        self,
        *,
        installation_id: uuid.UUID | None = None,
        now: datetime | None = None,
        profile_limit: int = DEFAULT_WITNESS_PROFILE_SCAN_LIMIT,
        delivery_limit: int = DEFAULT_WITNESS_DELIVERY_LIMIT,
        deliver_private: bool = False,
        autopilot_enabled: bool | None = None,
        autopilot_limit: int = DEFAULT_WITNESS_AUTOPILOT_LIMIT,
        autopilot_min_confidence: Decimal = DEFAULT_WITNESS_AUTOPILOT_MIN_CONFIDENCE,
        min_scan_interval: timedelta = DEFAULT_WITNESS_SCAN_INTERVAL,
        use_advisory_lock: bool = False,
        delivery_threshold: Decimal = DEFAULT_WITNESS_DELIVERY_THRESHOLD,
        digest_interval: timedelta = DEFAULT_WITNESS_DIGEST_INTERVAL,
        digest_max_items: int = DEFAULT_WITNESS_DIGEST_MAX_ITEMS,
        quiet_hours_start: int | None = None,
        quiet_hours_end: int | None = None,
        channel_posts_per_week: int = DEFAULT_WITNESS_CHANNEL_POSTS_PER_WEEK,
        drafts_per_channel_per_day: int | None = None,
    ) -> WitnessRunResult:
        """Run one Witness tick.

        Private delivery only attempts candidates that already have DM scope,
        batched into one digest DM per user per digest window.
        ``delivery_limit`` is deprecated: the digest budget
        (``digest_max_items``) is the volume control now.

        Channel delivery (HIG-198) attempts channel-scoped candidates whenever
        a Slack client is configured; each channel must opt in via its
        ObservePolicy (``proactivity_status == "full"``) and respects the
        weekly per-channel budget, the decision gate, and quiet hours.
        """

        if profile_limit < 0:
            raise ValueError("profile_limit must be non-negative")
        if delivery_limit < 0:
            raise ValueError("delivery_limit must be non-negative")
        if digest_max_items < 1:
            raise ValueError("digest_max_items must be positive")
        if channel_posts_per_week < 0:
            raise ValueError("channel_posts_per_week must be non-negative")
        run_at = _coerce_utc(now)

        lock_acquired = True
        if use_advisory_lock:
            lock_acquired = self._try_advisory_lock()
        if not lock_acquired:
            return WitnessRunResult(
                runner_id=self.runner_id,
                status="lock_skipped",
                leader_acquired=False,
            )

        try:
            with start_span(
                "witness.run",
                attributes={
                    "openinference.span.kind": "CHAIN",
                    "witness.runner.id": self.runner_id,
                    "witness.deliver_private": deliver_private,
                },
            ):
                projections = self._project_due_profiles(
                    installation_id=installation_id,
                    now=run_at,
                    limit=profile_limit,
                    min_scan_interval=min_scan_interval,
                )
                deliveries = (
                    self._deliver_digests(
                        installation_id=installation_id,
                        now=run_at,
                        delivery_threshold=delivery_threshold,
                        digest_interval=digest_interval,
                        digest_max_items=digest_max_items,
                        quiet_hours_start=quiet_hours_start,
                        quiet_hours_end=quiet_hours_end,
                    )
                    if deliver_private
                    else ()
                )
                deliveries = deliveries + self._deliver_channel_suggestions(
                    installation_id=installation_id,
                    now=run_at,
                    delivery_threshold=delivery_threshold,
                    channel_posts_per_week=channel_posts_per_week,
                    quiet_hours_start=quiet_hours_start,
                    quiet_hours_end=quiet_hours_end,
                )
                should_run_autopilot = (
                    autopilot_enabled
                    if autopilot_enabled is not None
                    else bool(
                        self.settings is not None
                        and self.settings.witness_autopilot_enabled
                    )
                )
                autopilot_outcomes = (
                    WitnessAutopilot(
                        self.session,
                        settings=self.settings,
                        llm_provider=self.llm_provider,
                        provider_name=self.provider_name,
                        actor_id=f"witness_runner:{self.runner_id}",
                        drafts_per_channel_per_day=drafts_per_channel_per_day,
                    )
                    .run_once(
                        installation_id=installation_id,
                        now=run_at,
                        limit=autopilot_limit,
                        min_confidence=autopilot_min_confidence,
                    )
                    .outcomes
                    if should_run_autopilot
                    else ()
                )
            status = (
                "processed"
                if projections or deliveries or autopilot_outcomes
                else "idle"
            )
            return WitnessRunResult(
                runner_id=self.runner_id,
                status=status,
                projections=projections,
                deliveries=deliveries,
                autopilot_outcomes=autopilot_outcomes,
            )
        finally:
            if use_advisory_lock:
                self._release_advisory_lock()

    def _project_due_profiles(
        self,
        *,
        installation_id: uuid.UUID | None,
        now: datetime,
        limit: int,
        min_scan_interval: timedelta,
    ) -> tuple[WitnessProjectionOutcome, ...]:
        if limit == 0:
            return ()
        rows = self._candidate_profiles(
            installation_id=installation_id,
            limit=limit * 3,
        )
        outcomes: list[WitnessProjectionOutcome] = []
        for profile, membership in rows:
            if len(outcomes) >= limit:
                break
            if not _profile_scan_due(
                profile,
                now=now,
                min_scan_interval=min_scan_interval,
            ):
                continue
            try:
                outcomes.append(
                    self._project_profile(
                        profile=profile,
                        membership=membership,
                        now=now,
                    )
                )
            except Exception:
                logger.exception(
                    "witness profile scan failed runner_id=%s profile_id=%s channel_id=%s",
                    self.runner_id,
                    profile.id,
                    membership.channel_id,
                )
        return tuple(outcomes)

    def _project_profile(
        self,
        *,
        profile: ObserveChannelProfile,
        membership: SlackChannelMembership,
        now: datetime,
    ) -> WitnessProjectionOutcome:
        task_service = TaskService(self.session)
        task = self._create_scan_task(
            task_service=task_service,
            profile=profile,
            membership=membership,
            now=now,
        )
        task_service.append_event(
            task,
            TaskEventType.log,
            {
                "message": WITNESS_RUNNER_PROFILE_SCAN_STARTED_MESSAGE,
                "runner_id": self.runner_id,
                "channel_id": membership.channel_id,
                "profile_id": str(profile.id),
                "profile_version": profile.profile_version,
            },
        )
        task_service.transition(task, TaskStatus.running)
        try:
            extraction = self._profile_extractor(
                task=task,
                task_service=task_service,
            ).extract(
                task=task,
                membership=membership,
                profile=profile,
            )
            result = WitnessOpportunityService(
                self.session
            ).project_from_channel_profile(
                task=task,
                membership=membership,
                profile=profile,
                candidates=extraction.candidates,
                extraction_metadata={
                    "runner_id": self.runner_id,
                    "runner_source": "witness_runner",
                    "raw_candidate_count": extraction.raw_candidate_count,
                    "skipped_reason": extraction.skipped_reason,
                },
            )
            _mark_profile_scanned(
                profile,
                now=now,
                task_id=task.id,
                runner_id=self.runner_id,
            )
            task.result_summary = (
                f"Witness projected {result.total_count} candidate(s) from "
                f"{membership.channel_id}."
            )
            task_service.append_event(
                task,
                TaskEventType.log,
                {
                    "message": WITNESS_OPPORTUNITY_CANDIDATES_PROJECTED_MESSAGE,
                    "source_type": "channel_profile",
                    "extractor": "llm",
                    "runner_id": self.runner_id,
                    "channel_id": membership.channel_id,
                    "membership_id": str(membership.id),
                    "profile_id": str(profile.id),
                    "raw_candidate_count": extraction.raw_candidate_count,
                    "skipped_reason": extraction.skipped_reason,
                    "created_count": result.created_count,
                    "updated_count": result.updated_count,
                    "skipped_count": result.skipped_count,
                    "candidate_ids": list(result.candidate_ids),
                },
            )
            task_service.append_event(
                task,
                TaskEventType.log,
                {
                    "message": WITNESS_RUNNER_PROFILE_SCAN_COMPLETED_MESSAGE,
                    "runner_id": self.runner_id,
                    "channel_id": membership.channel_id,
                    "profile_id": str(profile.id),
                    "created_count": result.created_count,
                    "updated_count": result.updated_count,
                    "skipped_count": result.skipped_count,
                },
            )
            task_service.transition(task, TaskStatus.succeeded)
            self.session.flush()
            return WitnessProjectionOutcome(
                task_id=task.id,
                channel_id=membership.channel_id,
                profile_id=profile.id,
                created_count=result.created_count,
                updated_count=result.updated_count,
                skipped_count=result.skipped_count,
                candidate_ids=result.candidate_ids,
                raw_candidate_count=extraction.raw_candidate_count,
                skipped_reason=extraction.skipped_reason,
            )
        except Exception as exc:
            task.error = {
                "type": type(exc).__name__,
                "message": str(exc),
                "runner_id": self.runner_id,
            }
            task_service.append_event(
                task,
                TaskEventType.error,
                {
                    "message": WITNESS_RUNNER_PROFILE_SCAN_FAILED_MESSAGE,
                    "runner_id": self.runner_id,
                    "channel_id": membership.channel_id,
                    "profile_id": str(profile.id),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            task_service.transition(task, TaskStatus.failed)
            self.session.flush()
            raise

    def _deliver_digests(
        self,
        *,
        installation_id: uuid.UUID | None,
        now: datetime,
        delivery_threshold: Decimal,
        digest_interval: timedelta,
        digest_max_items: int,
        quiet_hours_start: int | None,
        quiet_hours_end: int | None,
    ) -> tuple[WitnessDeliveryOutcome, ...]:
        if self.slack_client is None:
            return ()

        candidates = self._eligible_dm_candidates(
            installation_id=installation_id,
            now=now,
            limit=WITNESS_DIGEST_CANDIDATE_SCAN_LIMIT,
        )
        if not candidates:
            return ()

        if _in_quiet_hours(now, quiet_hours_start, quiet_hours_end):
            # Deferred, never dropped: candidates stay pending for the next
            # window outside quiet hours.
            return tuple(
                WitnessDeliveryOutcome(
                    candidate_id=candidate.id,
                    status="quiet_hours_deferred",
                    reason="quiet_hours",
                )
                for candidate in candidates
            )

        grouped: dict[
            tuple[uuid.UUID, str, str], list[WitnessOpportunityCandidate]
        ] = {}
        for candidate in candidates:
            user_id = candidate.target_slack_user_id
            channel_id = candidate.channel_id
            if user_id is None or channel_id is None:
                continue
            grouped.setdefault(
                (candidate.installation_id, user_id, channel_id), []
            ).append(candidate)

        outcomes: list[WitnessDeliveryOutcome] = []
        for (group_installation_id, user_id, channel_id), group in grouped.items():
            outcomes.extend(
                self._deliver_user_digest(
                    installation_id=group_installation_id,
                    user_id=user_id,
                    channel_id=channel_id,
                    candidates=group,
                    now=now,
                    delivery_threshold=delivery_threshold,
                    digest_interval=digest_interval,
                    digest_max_items=digest_max_items,
                )
            )
        self.session.flush()
        return tuple(outcomes)

    def _deliver_user_digest(
        self,
        *,
        installation_id: uuid.UUID,
        user_id: str,
        channel_id: str,
        candidates: list[WitnessOpportunityCandidate],
        now: datetime,
        delivery_threshold: Decimal,
        digest_interval: timedelta,
        digest_max_items: int,
    ) -> tuple[WitnessDeliveryOutcome, ...]:
        if self._digest_sent_in_window(
            installation_id=installation_id,
            user_id=user_id,
            now=now,
            digest_interval=digest_interval,
        ):
            return tuple(
                WitnessDeliveryOutcome(
                    candidate_id=candidate.id,
                    status="interval_deferred",
                    reason="digest_interval",
                )
                for candidate in candidates
            )

        events = collect_user_feedback_events(
            self.session,
            installation_id=installation_id,
            slack_user_id=user_id,
            now=now,
        )
        outcomes: list[WitnessDeliveryOutcome] = []
        deliverable: list[tuple[float, str, WitnessOpportunityCandidate]] = []
        for candidate in candidates:
            evidence_count = len(candidate.evidence_json or [])
            confidence = effective_confidence(
                candidate.confidence_score or Decimal("0.500"),
                reinforcement_count=candidate.reinforcement_count or 1,
                evidence_count=evidence_count,
            )
            receptivity_value = receptivity(events, candidate.candidate_type, now)
            if candidate.automation_kind == "recurring" and not recurrence_is_proven(
                candidate, now=now
            ):
                receptivity_value *= UNPROVEN_RECURRENCE_RECEPTIVITY_FACTOR
            score = float(confidence) * receptivity_value
            candidate.receptivity_score = _quantized_score(receptivity_value)
            if Decimal(str(score)) < delivery_threshold:
                candidate.last_decision = "silent"
                candidate.updated_at = now
                self._log_delivery_decision(
                    installation_id=installation_id,
                    user_id=user_id,
                    candidate_id=candidate.id,
                    decision="silent",
                    reason=(
                        f"score={score:.3f} below threshold={delivery_threshold} "
                        f"(confidence={confidence} receptivity="
                        f"{receptivity_value:.3f})"
                    ),
                    now=now,
                )
                outcomes.append(
                    WitnessDeliveryOutcome(
                        candidate_id=candidate.id,
                        status="silent",
                        reason="below_threshold",
                        decision="silent",
                        score=score,
                    )
                )
                continue
            decision = _candidate_decision(candidate)
            candidate.last_decision = decision
            candidate.updated_at = now
            deliverable.append((score, decision, candidate))

        deliverable.sort(key=lambda item: item[0], reverse=True)
        included = deliverable[:digest_max_items]
        overflow = deliverable[digest_max_items:]
        for score, decision, candidate in overflow:
            # HARD budget stop: overflow stays pending for the next window.
            self._log_delivery_decision(
                installation_id=installation_id,
                user_id=user_id,
                candidate_id=candidate.id,
                decision=decision,
                reason="budget_deferred",
                now=now,
            )
            outcomes.append(
                WitnessDeliveryOutcome(
                    candidate_id=candidate.id,
                    status="budget_deferred",
                    reason="budget_deferred",
                    decision=decision,
                    score=score,
                )
            )

        if not included:
            return tuple(outcomes)

        text = _digest_text(
            [(decision, candidate) for _score, decision, candidate in included],
            now=now,
        )
        window_index = int(now.timestamp()) // max(
            int(digest_interval.total_seconds()), 1
        )
        idempotency_key = (
            f"{WITNESS_DIGEST_PURPOSE}:{installation_id}:{user_id}:{window_index}"
        )
        request: dict[str, object] = {
            "channel": channel_id,
            "text": text,
            "thread_ts": None,
        }
        client = self.slack_client
        assert client is not None  # checked by caller
        side_effect = SlackSideEffectOutbox(self.session).deliver(
            installation_id=installation_id,
            task_id=None,
            idempotency_key=idempotency_key,
            operation="chat_postMessage",
            purpose=WITNESS_DIGEST_PURPOSE,
            target_channel_id=channel_id,
            request=request,
            call=lambda: client.chat_postMessage(
                channel=channel_id,
                text=text,
                thread_ts=None,
            ),
        )
        if side_effect.deduped:
            # A digest already went out in this window; do not mark anything
            # sent, the candidates stay pending for the next window.
            outcomes.extend(
                WitnessDeliveryOutcome(
                    candidate_id=candidate.id,
                    status="window_deduped",
                    reason="digest_window_deduped",
                    decision=decision,
                    score=score,
                )
                for score, decision, candidate in included
            )
            return tuple(outcomes)

        message_ts = _response_ts(side_effect.response)
        for score, decision, candidate in included:
            candidate.status = "sent"
            candidate.cooldown_until = None
            candidate.last_suggested_at = now
            candidate.updated_at = now
            _record_feedback(
                candidate,
                action="sent",
                by_user_id="witness_runner",
                now=now,
                details={
                    "channel_id": channel_id,
                    "message_ts": message_ts,
                    "side_effect_id": str(side_effect.side_effect.id),
                    "deduped": side_effect.deduped,
                    "delivery_policy": "digest_dm",
                    "decision": decision,
                },
            )
            self._log_delivery_decision(
                installation_id=installation_id,
                user_id=user_id,
                candidate_id=candidate.id,
                decision=decision,
                reason="sent",
                now=now,
            )
            _append_candidate_event(
                self.session,
                candidate,
                message=WITNESS_RUNNER_DELIVERY_SENT_MESSAGE,
                payload={
                    "runner_id": self.runner_id,
                    "channel_id": channel_id,
                    "message_ts": message_ts,
                    "decision": decision,
                    "delivery_policy": "digest_dm",
                },
            )
            outcomes.append(
                WitnessDeliveryOutcome(
                    candidate_id=candidate.id,
                    status="sent",
                    channel_id=channel_id,
                    message_ts=message_ts,
                    decision=decision,
                    score=score,
                )
            )
        self._log_delivery_decision(
            installation_id=installation_id,
            user_id=user_id,
            candidate_id=None,
            decision="digest",
            reason=f"sent:{len(included)}",
            now=now,
        )
        logger.info(
            "%s runner_id=%s user_id=%s items=%s",
            WITNESS_RUNNER_DIGEST_SENT_MESSAGE,
            self.runner_id,
            user_id,
            len(included),
        )
        return tuple(outcomes)

    def _deliver_channel_suggestions(
        self,
        *,
        installation_id: uuid.UUID | None,
        now: datetime,
        delivery_threshold: Decimal,
        channel_posts_per_week: int,
        quiet_hours_start: int | None,
        quiet_hours_end: int | None,
    ) -> tuple[WitnessDeliveryOutcome, ...]:
        """Deliver channel-scoped suggestions through every HIG-198 gate."""

        if self.slack_client is None or channel_posts_per_week < 1:
            return ()
        candidates = self._eligible_channel_candidates(
            installation_id=installation_id,
            now=now,
            limit=WITNESS_CHANNEL_CANDIDATE_SCAN_LIMIT,
        )
        if not candidates:
            return ()

        grouped: dict[tuple[uuid.UUID, str], list[WitnessOpportunityCandidate]] = {}
        for candidate in candidates:
            channel_id = candidate.channel_id
            if channel_id is None:
                continue
            grouped.setdefault((candidate.installation_id, channel_id), []).append(
                candidate
            )

        outcomes: list[WitnessDeliveryOutcome] = []
        for (group_installation_id, channel_id), group in grouped.items():
            outcomes.extend(
                self._deliver_channel_group(
                    installation_id=group_installation_id,
                    channel_id=channel_id,
                    candidates=group,
                    now=now,
                    delivery_threshold=delivery_threshold,
                    channel_posts_per_week=channel_posts_per_week,
                    quiet_hours_start=quiet_hours_start,
                    quiet_hours_end=quiet_hours_end,
                )
            )
        self.session.flush()
        return tuple(outcomes)

    def _deliver_channel_group(
        self,
        *,
        installation_id: uuid.UUID,
        channel_id: str,
        candidates: list[WitnessOpportunityCandidate],
        now: datetime,
        delivery_threshold: Decimal,
        channel_posts_per_week: int,
        quiet_hours_start: int | None,
        quiet_hours_end: int | None,
    ) -> tuple[WitnessDeliveryOutcome, ...]:
        log_user = f"{WITNESS_CHANNEL_LOG_USER_PREFIX}{channel_id}"

        # Gate 1: per-channel policy opt-in. "full" finally means something;
        # digest_only keeps today's DM-digest-only behavior.
        if not self._channel_policy_allows_posting(
            installation_id=installation_id,
            channel_id=channel_id,
        ):
            return self._defer_channel_candidates(
                candidates,
                installation_id=installation_id,
                log_user=log_user,
                reason="policy",
                now=now,
            )

        # Gate 2: quiet hours — deferred, never dropped.
        if _in_quiet_hours(now, quiet_hours_start, quiet_hours_end):
            return self._defer_channel_candidates(
                candidates,
                installation_id=installation_id,
                log_user=log_user,
                reason="quiet_hours",
                now=now,
                status="quiet_hours_deferred",
            )

        # Gate 3: the HIG-227 decision gate, scored on channel-level
        # receptivity (everyone's reactions to suggestions in this channel).
        events = collect_channel_feedback_events(
            self.session,
            installation_id=installation_id,
            channel_id=channel_id,
            now=now,
        )
        outcomes: list[WitnessDeliveryOutcome] = []
        deliverable: list[tuple[float, str, WitnessOpportunityCandidate]] = []
        for candidate in candidates:
            evidence_count = len(candidate.evidence_json or [])
            confidence = effective_confidence(
                candidate.confidence_score or Decimal("0.500"),
                reinforcement_count=candidate.reinforcement_count or 1,
                evidence_count=evidence_count,
            )
            receptivity_value = receptivity(events, candidate.candidate_type, now)
            if candidate.automation_kind == "recurring" and not recurrence_is_proven(
                candidate, now=now
            ):
                receptivity_value *= UNPROVEN_RECURRENCE_RECEPTIVITY_FACTOR
            score = float(confidence) * receptivity_value
            candidate.receptivity_score = _quantized_score(receptivity_value)
            if Decimal(str(score)) < delivery_threshold:
                candidate.last_decision = "silent"
                candidate.updated_at = now
                self._log_delivery_decision(
                    installation_id=installation_id,
                    user_id=log_user,
                    candidate_id=candidate.id,
                    decision="silent",
                    reason=(
                        f"score={score:.3f} below threshold={delivery_threshold} "
                        f"(confidence={confidence} receptivity="
                        f"{receptivity_value:.3f})"
                    ),
                    now=now,
                )
                outcomes.append(
                    WitnessDeliveryOutcome(
                        candidate_id=candidate.id,
                        status="silent",
                        reason="below_threshold",
                        channel_id=channel_id,
                        decision="silent",
                        score=score,
                    )
                )
                continue
            decision = candidate_delivery_decision(candidate)
            candidate.last_decision = decision
            candidate.updated_at = now
            deliverable.append((score, decision, candidate))

        if not deliverable:
            return tuple(outcomes)

        # Gate 4: sliding weekly budget per channel over witness_delivery_log.
        deliverable.sort(key=lambda item: item[0], reverse=True)
        budget_left = channel_posts_per_week - self._channel_posts_in_window(
            installation_id=installation_id,
            log_user=log_user,
            now=now,
        )
        included = deliverable[: max(budget_left, 0)]
        overflow = deliverable[max(budget_left, 0) :]
        for score, decision, candidate in overflow:
            # Budget stop: stays pending, delivers next window.
            self._log_channel_deferral(
                installation_id=installation_id,
                log_user=log_user,
                candidate_id=candidate.id,
                reason="budget",
                now=now,
            )
            outcomes.append(
                WitnessDeliveryOutcome(
                    candidate_id=candidate.id,
                    status="budget_deferred",
                    reason="budget",
                    channel_id=channel_id,
                    decision=decision,
                    score=score,
                )
            )

        for score, decision, candidate in included:
            outcomes.append(
                self._post_channel_suggestion(
                    installation_id=installation_id,
                    channel_id=channel_id,
                    log_user=log_user,
                    candidate=candidate,
                    decision=decision,
                    score=score,
                    now=now,
                )
            )
        return tuple(outcomes)

    def _post_channel_suggestion(
        self,
        *,
        installation_id: uuid.UUID,
        channel_id: str,
        log_user: str,
        candidate: WitnessOpportunityCandidate,
        decision: str,
        score: float,
        now: datetime,
    ) -> WitnessDeliveryOutcome:
        client = self.slack_client
        assert client is not None  # checked by caller
        source_task = (
            self.session.get(Task, candidate.source_task_id)
            if candidate.source_task_id is not None
            else None
        )
        thread_ts = candidate_thread_ts(candidate, source_task=source_task)
        text = _channel_suggestion_text(candidate, decision=decision, now=now)
        request: dict[str, object] = {
            "channel": channel_id,
            "text": text,
            "thread_ts": thread_ts,
        }

        def _post(
            client: WitnessSlackClient = client,
            channel_id: str = channel_id,
            text: str = text,
            thread_ts: str | None = thread_ts,
        ) -> Mapping[str, Any]:
            return client.chat_postMessage(
                channel=channel_id,
                text=text,
                thread_ts=thread_ts,
            )

        side_effect = SlackSideEffectOutbox(self.session).deliver(
            installation_id=installation_id,
            task_id=candidate.source_task_id,
            idempotency_key=f"{WITNESS_CHANNEL_SUGGESTION_PURPOSE}:{candidate.id}",
            operation="chat_postMessage",
            purpose=WITNESS_CHANNEL_SUGGESTION_PURPOSE,
            target_channel_id=channel_id,
            target_thread_ts=thread_ts,
            request=request,
            call=_post,
        )
        if side_effect.deduped:
            # This candidate already has a visible suggestion post; keep it
            # pending without double-posting.
            return WitnessDeliveryOutcome(
                candidate_id=candidate.id,
                status="window_deduped",
                reason="channel_suggestion_deduped",
                channel_id=channel_id,
                decision=decision,
                score=score,
            )

        message_ts = _response_ts(side_effect.response)
        candidate.status = "sent"
        candidate.cooldown_until = None
        candidate.last_suggested_at = now
        candidate.updated_at = now
        _record_feedback(
            candidate,
            action="sent",
            by_user_id="witness_runner",
            now=now,
            details={
                "channel_id": channel_id,
                "message_ts": message_ts,
                "thread_ts": thread_ts,
                "side_effect_id": str(side_effect.side_effect.id),
                "deduped": side_effect.deduped,
                "delivery_policy": "channel_post",
                "decision": decision,
            },
        )
        self._log_delivery_decision(
            installation_id=installation_id,
            user_id=log_user,
            candidate_id=candidate.id,
            decision="channel_sent",
            reason="sent",
            now=now,
        )
        _append_candidate_event(
            self.session,
            candidate,
            message=WITNESS_RUNNER_CHANNEL_POST_SENT_MESSAGE,
            payload={
                "runner_id": self.runner_id,
                "channel_id": channel_id,
                "message_ts": message_ts,
                "thread_ts": thread_ts,
                "decision": decision,
                "delivery_policy": "channel_post",
            },
        )
        logger.info(
            "%s runner_id=%s channel_id=%s candidate_id=%s decision=%s",
            WITNESS_RUNNER_CHANNEL_POST_SENT_MESSAGE,
            self.runner_id,
            channel_id,
            candidate.id,
            decision,
        )
        return WitnessDeliveryOutcome(
            candidate_id=candidate.id,
            status="sent",
            channel_id=channel_id,
            message_ts=message_ts,
            decision=decision,
            score=score,
        )

    def _defer_channel_candidates(
        self,
        candidates: list[WitnessOpportunityCandidate],
        *,
        installation_id: uuid.UUID,
        log_user: str,
        reason: str,
        now: datetime,
        status: str = "policy_deferred",
    ) -> tuple[WitnessDeliveryOutcome, ...]:
        outcomes: list[WitnessDeliveryOutcome] = []
        for candidate in candidates:
            self._log_channel_deferral(
                installation_id=installation_id,
                log_user=log_user,
                candidate_id=candidate.id,
                reason=reason,
                now=now,
            )
            outcomes.append(
                WitnessDeliveryOutcome(
                    candidate_id=candidate.id,
                    status=status,
                    reason=reason,
                    channel_id=candidate.channel_id,
                )
            )
        return tuple(outcomes)

    def _log_channel_deferral(
        self,
        *,
        installation_id: uuid.UUID,
        log_user: str,
        candidate_id: uuid.UUID,
        reason: str,
        now: datetime,
    ) -> None:
        """Log a channel deferral, deduped per (candidate, reason) per day.

        The runner ticks every few minutes; without dedupe a channel that
        never opted in would emit a deferral row per candidate per tick.
        Deferred is still never dropped — the candidate stays pending.
        """

        cutoff = now - timedelta(hours=24)
        already_logged = (
            self.session.scalar(
                select(func.count())
                .select_from(WitnessDeliveryLog)
                .where(
                    WitnessDeliveryLog.installation_id == installation_id,
                    WitnessDeliveryLog.candidate_id == candidate_id,
                    WitnessDeliveryLog.decision == "channel_deferred",
                    WitnessDeliveryLog.reason == reason,
                    WitnessDeliveryLog.created_at > cutoff,
                )
            )
            or 0
        ) > 0
        if already_logged:
            return
        self._log_delivery_decision(
            installation_id=installation_id,
            user_id=log_user,
            candidate_id=candidate_id,
            decision="channel_deferred",
            reason=reason,
            now=now,
        )

    def _channel_policy_allows_posting(
        self,
        *,
        installation_id: uuid.UUID,
        channel_id: str,
    ) -> bool:
        policy = self.session.scalar(
            select(ObservePolicy).where(
                ObservePolicy.installation_id == installation_id,
                ObservePolicy.scope_type == "channel",
                ObservePolicy.scope_id == channel_id,
            )
        )
        return (
            policy is not None
            and policy.proactivity_status == "full"
            and policy.paused_at is None
        )

    def _channel_posts_in_window(
        self,
        *,
        installation_id: uuid.UUID,
        log_user: str,
        now: datetime,
    ) -> int:
        # HIG-231 ambient file briefs share this weekly per-channel window:
        # both account through rows keyed 'channel:{channel_id}', so a brief
        # consumes a channel post slot and vice versa (bidirectional budget).
        cutoff = now - WITNESS_CHANNEL_POST_WINDOW
        return int(
            self.session.scalar(
                select(func.count())
                .select_from(WitnessDeliveryLog)
                .where(
                    WitnessDeliveryLog.installation_id == installation_id,
                    WitnessDeliveryLog.slack_user_id == log_user,
                    WitnessDeliveryLog.decision.in_(
                        ("channel_sent", "ambient_file_brief")
                    ),
                    WitnessDeliveryLog.created_at > cutoff,
                )
            )
            or 0
        )

    def _eligible_channel_candidates(
        self,
        *,
        installation_id: uuid.UUID | None,
        now: datetime,
        limit: int,
    ) -> tuple[WitnessOpportunityCandidate, ...]:
        filters = [
            WitnessOpportunityCandidate.status == "candidate",
            WitnessOpportunityCandidate.visibility_scope_type == "channel",
            WitnessOpportunityCandidate.channel_id.is_not(None),
            WitnessOpportunityCandidate.channel_id.not_like("D%"),
            (
                (WitnessOpportunityCandidate.cooldown_until.is_(None))
                | (WitnessOpportunityCandidate.cooldown_until <= now)
            ),
        ]
        if installation_id is not None:
            filters.append(
                WitnessOpportunityCandidate.installation_id == installation_id
            )
        return tuple(
            self.session.scalars(
                select(WitnessOpportunityCandidate)
                .where(*filters)
                .order_by(
                    WitnessOpportunityCandidate.confidence_score.desc(),
                    WitnessOpportunityCandidate.created_at.asc(),
                )
                .with_for_update(skip_locked=True)
                .limit(limit)
            )
        )

    def _digest_sent_in_window(
        self,
        *,
        installation_id: uuid.UUID,
        user_id: str,
        now: datetime,
        digest_interval: timedelta,
    ) -> bool:
        cutoff = now - digest_interval
        return (
            self.session.scalar(
                select(func.count())
                .select_from(WitnessDeliveryLog)
                .where(
                    WitnessDeliveryLog.installation_id == installation_id,
                    WitnessDeliveryLog.slack_user_id == user_id,
                    WitnessDeliveryLog.decision == "digest",
                    WitnessDeliveryLog.reason.like("sent%"),
                    WitnessDeliveryLog.created_at > cutoff,
                )
            )
            or 0
        ) > 0

    def _log_delivery_decision(
        self,
        *,
        installation_id: uuid.UUID,
        user_id: str,
        candidate_id: uuid.UUID | None,
        decision: str,
        reason: str | None,
        now: datetime,
    ) -> None:
        self.session.add(
            WitnessDeliveryLog(
                installation_id=installation_id,
                slack_user_id=user_id,
                candidate_id=candidate_id,
                decision=decision,
                reason=reason,
                created_at=now,
            )
        )

    def _candidate_profiles(
        self,
        *,
        installation_id: uuid.UUID | None,
        limit: int,
    ) -> tuple[tuple[ObserveChannelProfile, SlackChannelMembership], ...]:
        filters = [
            ObserveChannelProfile.profile_status == "active",
            SlackChannelMembership.membership_status == "active",
        ]
        if installation_id is not None:
            filters.append(ObserveChannelProfile.installation_id == installation_id)
        rows = self.session.execute(
            select(ObserveChannelProfile, SlackChannelMembership)
            .join(
                SlackChannelMembership,
                and_(
                    SlackChannelMembership.installation_id
                    == ObserveChannelProfile.installation_id,
                    SlackChannelMembership.channel_id
                    == ObserveChannelProfile.channel_id,
                ),
            )
            .where(*filters)
            .order_by(
                desc(ObserveChannelProfile.last_profiled_at),
                desc(ObserveChannelProfile.updated_at),
            )
            .limit(limit)
        ).all()
        return tuple((profile, membership) for profile, membership in rows)

    def _eligible_dm_candidates(
        self,
        *,
        installation_id: uuid.UUID | None,
        now: datetime,
        limit: int,
    ) -> tuple[WitnessOpportunityCandidate, ...]:
        filters = [
            WitnessOpportunityCandidate.status == "candidate",
            WitnessOpportunityCandidate.visibility_scope_type == "dm",
            WitnessOpportunityCandidate.channel_id.like("D%"),
            WitnessOpportunityCandidate.target_slack_user_id.is_not(None),
            (
                (WitnessOpportunityCandidate.cooldown_until.is_(None))
                | (WitnessOpportunityCandidate.cooldown_until <= now)
            ),
        ]
        if installation_id is not None:
            filters.append(
                WitnessOpportunityCandidate.installation_id == installation_id
            )
        return tuple(
            self.session.scalars(
                select(WitnessOpportunityCandidate)
                .where(*filters)
                .order_by(
                    WitnessOpportunityCandidate.confidence_score.desc(),
                    WitnessOpportunityCandidate.created_at.asc(),
                )
                .with_for_update(skip_locked=True)
                .limit(limit)
            )
        )

    def _create_scan_task(
        self,
        *,
        task_service: TaskService,
        profile: ObserveChannelProfile,
        membership: SlackChannelMembership,
        now: datetime,
    ) -> Task:
        channel_label = (
            f"#{membership.channel_name}"
            if membership.channel_name
            else membership.channel_id
        )
        return task_service.create_task(
            installation_id=profile.installation_id,
            slack_event_id=(
                f"witness:{profile.id}:{profile.profile_version}:{int(now.timestamp())}"
            ),
            slack_channel_id=membership.channel_id,
            slack_thread_ts=membership.channel_id,
            slack_message_ts=None,
            slack_user_id=membership.added_by_user_id or "witness_runner",
            input=f"Run Witness opportunity scan for {channel_label}.",
            source_surface="witness_runner",
        )

    def _profile_extractor(
        self,
        *,
        task: Task,
        task_service: TaskService,
    ) -> WitnessChannelProfileExtractor:
        if self.llm_provider is not None:
            model_route = ModelRoute(
                tier=ModelRouteTier.cheap_fast,
                model=self.llm_provider.model,
                reason="witness_runner_channel_profile_extraction",
            )
            provider = self.llm_provider
            provider_name = self.provider_name or DbLLMProvider.openrouter
        else:
            model_route = ModelRouter(self._settings).route_for_tier(
                ModelRouteTier.cheap_fast,
                reason="witness_runner_channel_profile_extraction",
            )
            selection = select_runtime_model(
                session=self.session,
                settings=self._settings,
                installation_id=task.installation_id,
                model_route=model_route,
            )
            model_route = selection.model_route
            provider = create_provider_for_selection(
                settings=self._settings,
                selection=selection,
            )
            provider_name = selection.provider_name
        return WitnessChannelProfileExtractor(
            LLMService(
                session=self.session,
                provider=provider,
                provider_name=provider_name,
                task_service=task_service,
                model_route=model_route,
            )
        )

    @property
    def _settings(self) -> Settings:
        if self.settings is None:
            self.settings = load_settings()
        return self.settings

    def _try_advisory_lock(self) -> bool:
        return bool(
            self.session.scalar(
                select(func.pg_try_advisory_lock(self.advisory_lock_key))
            )
        )

    def _release_advisory_lock(self) -> None:
        self.session.execute(select(func.pg_advisory_unlock(self.advisory_lock_key)))


class WitnessWorker:
    """Poll the Witness runner forever."""

    def __init__(
        self,
        *,
        session_factory: sessionmaker[Session] | None = None,
        settings: Settings | None = None,
        runner_id: str | None = None,
        poll_interval_seconds: float = DEFAULT_WITNESS_POLL_INTERVAL_SECONDS,
        profile_limit: int = DEFAULT_WITNESS_PROFILE_SCAN_LIMIT,
        delivery_limit: int = DEFAULT_WITNESS_DELIVERY_LIMIT,
        scan_interval: timedelta = DEFAULT_WITNESS_SCAN_INTERVAL,
        deliver_private: bool = False,
        use_advisory_lock: bool = True,
    ) -> None:
        self.session_factory = session_factory or make_session_factory()
        self.settings = settings
        self.runner_id = runner_id or default_witness_runner_id()
        self.poll_interval_seconds = poll_interval_seconds
        self.profile_limit = profile_limit
        self.delivery_limit = delivery_limit
        self.scan_interval = scan_interval
        self.deliver_private = deliver_private
        self.use_advisory_lock = use_advisory_lock

    def run_once(self, *, now: datetime | None = None) -> WitnessRunResult:
        with self.session_factory.begin() as session:
            # The client is always available: channel delivery (HIG-198) is
            # gated per channel by ObservePolicy proactivity_status == "full",
            # while deliver_private still gates the DM digest path.
            slack_client = cast(
                WitnessSlackClient,
                WebClient(token=self._settings.slack_bot_token),
            )
            result = WitnessRunner(
                session,
                settings=self._settings,
                slack_client=slack_client,
                runner_id=self.runner_id,
            ).run_once(
                now=now,
                profile_limit=self.profile_limit,
                delivery_limit=self.delivery_limit,
                deliver_private=self.deliver_private,
                autopilot_enabled=self._settings.witness_autopilot_enabled,
                autopilot_limit=self._settings.witness_autopilot_limit,
                autopilot_min_confidence=(
                    self._settings.witness_autopilot_min_confidence
                ),
                min_scan_interval=self.scan_interval,
                use_advisory_lock=self.use_advisory_lock,
                delivery_threshold=self._settings.witness_delivery_threshold,
                digest_interval=timedelta(
                    hours=self._settings.witness_digest_interval_hours
                ),
                digest_max_items=self._settings.witness_digest_max_items,
                quiet_hours_start=self._settings.witness_quiet_hours_start,
                quiet_hours_end=self._settings.witness_quiet_hours_end,
                channel_posts_per_week=(self._settings.witness_channel_posts_per_week),
                drafts_per_channel_per_day=(
                    self._settings.witness_drafts_per_channel_per_day
                ),
            )
            logger.info(
                "witness runner tick runner_id=%s status=%s projected=%s delivered=%s autopilot_reviewed=%s autopilot_executed=%s",
                result.runner_id,
                result.status,
                result.projected_count,
                result.delivered_count,
                result.autopilot_reviewed_count,
                result.autopilot_executed_count,
            )
            return result

    def run_forever(self) -> None:
        while True:
            self.run_once()
            time.sleep(self.poll_interval_seconds)

    @property
    def _settings(self) -> Settings:
        if self.settings is None:
            self.settings = load_settings()
        return self.settings


def default_witness_runner_id() -> str:
    return f"{socket.gethostname()}-{os.getpid()}"


def _profile_scan_due(
    profile: ObserveChannelProfile,
    *,
    now: datetime,
    min_scan_interval: timedelta,
) -> bool:
    metadata = profile.metadata_json if isinstance(profile.metadata_json, dict) else {}
    runner = metadata.get("witness_runner")
    if not isinstance(runner, dict):
        return True
    scanned_version = runner.get("profile_version")
    if scanned_version != profile.profile_version:
        return True
    scanned_at = _parse_datetime(runner.get("last_scanned_at"))
    if scanned_at is None:
        return True
    return scanned_at <= now - min_scan_interval


def _mark_profile_scanned(
    profile: ObserveChannelProfile,
    *,
    now: datetime,
    task_id: uuid.UUID,
    runner_id: str,
) -> None:
    metadata = (
        dict(profile.metadata_json) if isinstance(profile.metadata_json, dict) else {}
    )
    metadata["witness_runner"] = {
        "last_scanned_at": now.isoformat(),
        "profile_version": profile.profile_version,
        "task_id": str(task_id),
        "runner_id": runner_id,
    }
    profile.metadata_json = metadata
    profile.updated_at = now


def _append_candidate_event(
    session: Session,
    candidate: WitnessOpportunityCandidate,
    *,
    message: str,
    payload: dict[str, object],
) -> None:
    if candidate.source_task_id is None:
        return
    task = session.get(Task, candidate.source_task_id)
    if task is None:
        return
    TaskService(session).append_event(
        task,
        TaskEventType.log,
        {
            "message": message,
            "candidate_id": str(candidate.id),
            **payload,
        },
    )


def _in_quiet_hours(
    now: datetime,
    start_hour: int | None,
    end_hour: int | None,
) -> bool:
    """True when ``now`` (UTC hour) falls inside the configured quiet window."""

    if start_hour is None or end_hour is None or start_hour == end_hour:
        return False
    hour = now.astimezone(UTC).hour
    if start_hour < end_hour:
        return start_hour <= hour < end_hour
    return hour >= start_hour or hour < end_hour


def _candidate_decision(candidate: WitnessOpportunityCandidate) -> str:
    """Assign the above-threshold action — see candidate_delivery_decision."""

    return candidate_delivery_decision(candidate)


def _digest_text(
    items: list[tuple[str, WitnessOpportunityCandidate]],
    *,
    now: datetime,
) -> str:
    count = len(items)
    plural = "s" if count != 1 else ""
    lines = [f"Here's your Kortny digest - {count} suggestion{plural} worth a look:"]
    for index, (decision, candidate) in enumerate(items, start=1):
        body = candidate.suggested_message or candidate.summary
        lines.append(f"{index}. {candidate.title} - {body}")
        evidence = _digest_evidence_line(candidate, now=now)
        if evidence:
            lines.append(f"   Evidence: {evidence}")
        lines.append(f"   {_digest_action_line(decision, candidate)}")
    text = "\n".join(lines)
    return normalize_user_facing_text(text[:WITNESS_DIGEST_MAX_CHARS])


def _channel_suggestion_text(
    candidate: WitnessOpportunityCandidate,
    *,
    decision: str,
    now: datetime,
) -> str:
    """Threaded, low-key, evidence-first channel suggestion copy (HIG-198)."""

    noticed = (
        candidate.suggested_message
        or f"I noticed something that might be worth a look: {candidate.summary}"
    )
    lines = [noticed]
    evidence = _digest_evidence_line(candidate, now=now)
    if evidence:
        lines.append(f"Evidence: {evidence}")
    proposed = candidate.deliverable or candidate.suggested_action
    if proposed:
        lines.append(f"Proposed: {proposed}")
    if decision == "question":
        lines.append(
            "What cadence should I use? Reply with one (like 'every weekday "
            "5pm') and I'll set it up."
        )
    lines.append("React :white_check_mark: to set it up or :no_entry_sign: to drop it.")
    text = "\n".join(lines)
    return normalize_user_facing_text(text[:WITNESS_CHANNEL_SUGGESTION_MAX_CHARS])


def _digest_evidence_line(
    candidate: WitnessOpportunityCandidate,
    *,
    now: datetime,
) -> str | None:
    recurrence = recurrence_evidence_line(candidate, now=now)
    if recurrence is not None:
        return recurrence
    evidence = candidate.evidence_json or []
    for item in evidence:
        if not isinstance(item, dict):
            continue
        snippet = item.get("snippet")
        if isinstance(snippet, str) and snippet.strip():
            return snippet.strip()[:240]
    return None


def _digest_action_line(
    decision: str,
    candidate: WitnessOpportunityCandidate,
) -> str:
    if decision == "draft":
        return "Say go and I'll do it."
    if decision == "question":
        return (
            "What cadence should I use? Reply with one (like 'every weekday "
            "5pm') and I'll set it up."
        )
    cadence = (candidate.cadence_suggestion or "").strip()
    if candidate.automation_kind == "recurring" and cadence:
        return f"Approve once and I'll run it {cadence}."
    return "Reply with the number to act on it."


def _quantized_score(value: float) -> Decimal:
    bounded = min(1.0, max(0.0, value))
    return Decimal(str(bounded)).quantize(Decimal("0.001"))


def _response_ts(response: Mapping[str, Any]) -> str | None:
    value = response.get("ts")
    if isinstance(value, str) and value:
        return value
    message = response.get("message")
    if isinstance(message, Mapping):
        message_ts = message.get("ts")
        if isinstance(message_ts, str) and message_ts:
            return message_ts
    return None


def _parse_datetime(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return _coerce_utc(parsed)


def _coerce_utc(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(UTC)
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entrypoint for the Witness runner."""

    configure_logging()
    parser = argparse.ArgumentParser(description="Run the Kortny Witness runner")
    parser.add_argument("--once", action="store_true", help="Run one Witness tick")
    parser.add_argument(
        "--runner-id",
        default=None,
        help="Override runner id used in logs",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=None,
        help="Seconds to sleep between ticks",
    )
    parser.add_argument(
        "--profile-limit",
        type=int,
        default=None,
        help="Maximum channel profiles to scan per tick",
    )
    parser.add_argument(
        "--delivery-limit",
        type=int,
        default=None,
        help="Maximum private suggestions to deliver per tick",
    )
    parser.add_argument(
        "--deliver-private",
        action="store_true",
        default=None,
        help="Send eligible DM-scoped suggestions",
    )
    args = parser.parse_args(argv)

    settings = load_settings()
    configure_tracing(settings)
    if not settings.witness_enabled:
        logger.info("witness runner disabled by KORTNY_WITNESS_ENABLED=false")
        print("witness runner disabled")
        return

    worker = WitnessWorker(
        settings=settings,
        runner_id=args.runner_id,
        poll_interval_seconds=(
            args.poll_interval
            if args.poll_interval is not None
            else settings.witness_poll_interval_seconds
        ),
        profile_limit=(
            args.profile_limit
            if args.profile_limit is not None
            else settings.witness_profile_scan_limit
        ),
        delivery_limit=(
            args.delivery_limit
            if args.delivery_limit is not None
            else settings.witness_delivery_limit
        ),
        scan_interval=timedelta(seconds=settings.witness_scan_interval_seconds),
        deliver_private=(
            args.deliver_private
            if args.deliver_private is not None
            else settings.witness_deliver_private
        ),
    )
    logger.info(
        "witness runner started runner_id=%s once=%s deliver_private=%s",
        worker.runner_id,
        args.once,
        args.deliver_private,
    )
    if args.once:
        result = worker.run_once()
        print(
            f"runner_id={result.runner_id} status={result.status} "
            f"projected_count={result.projected_count} "
            f"delivered_count={result.delivered_count} "
            f"autopilot_reviewed_count={result.autopilot_reviewed_count} "
            f"autopilot_executed_count={result.autopilot_executed_count}"
        )
        return

    worker.run_forever()
