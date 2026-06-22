"""Proactive Action Ledger policy — Step 1 shadow (observe-only).

Pure stateless decision function that reproduces the existing Witness
delivery gate decisions without accessing the database or Slack. In
Step 1 this is exercised in shadow mode only and never routes real
delivery or autopilot execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Literal

from kortny.witness.autopilot import (
    _AUTOPILOT_EXECUTABLE_ACTION_KINDS,
    _AUTOPILOT_EXECUTABLE_DELIVERY_TARGETS,
    WitnessAutopilotDecision,
)
from kortny.witness.receptivity import (
    UserFeedbackEvent,
    effective_confidence,
    receptivity,
)
from kortny.witness.runner import DEFAULT_WITNESS_DIGEST_MAX_ITEMS

LedgerSurface = Literal["dm_digest", "channel_post", "autopilot"]
LedgerOutcome = Literal["act", "ask", "silent", "defer"]


@dataclass(frozen=True, slots=True)
class LedgerDecision:
    """Typed outcome from ProactiveActionPolicy.decide()."""

    decision: LedgerOutcome
    reason_code: str


@dataclass(frozen=True, slots=True)
class CandidateInputs:
    """Scoring inputs drawn from a WitnessOpportunityCandidate row."""

    confidence_score: Decimal
    reinforcement_count: int
    evidence_count: int
    span_days: int
    candidate_type: str
    created_at: datetime


@dataclass(frozen=True, slots=True)
class DeliveryContext:
    """Gate inputs that differ per surface and per call site."""

    now: datetime
    delivery_threshold: Decimal
    # ── DM digest ──────────────────────────────────────────────────────────
    # digest_epoch: None means no epoch filter (env deliver_private=True back-compat)
    digest_epoch: datetime | None = None
    # digest_interval_exceeded: True when the current window has NOT yet had a digest
    digest_interval_exceeded: bool = True
    digest_max_items: int = DEFAULT_WITNESS_DIGEST_MAX_ITEMS
    # items_before_this: candidates ranked above this one (for budget check)
    items_before_this: int = 0
    # ── Channel post ───────────────────────────────────────────────────────
    # channel_full_enabled: policy proactivity_status == "full" AND not paused
    channel_full_enabled: bool = True
    # channel_epoch: full_enabled_at from ObservePolicy; None = no epoch set
    channel_epoch: datetime | None = None
    # channel_posts_budget_left: (channel_posts_per_week - posts in last 7 days)
    channel_posts_budget_left: int = 1
    # items_before_this_channel: candidates for this channel ranked above this one
    items_before_this_channel: int = 0
    # ── Shared ─────────────────────────────────────────────────────────────
    in_quiet_hours: bool = False
    # ── Autopilot ──────────────────────────────────────────────────────────
    autopilot_preflight_defer_reason: str | None = None
    autopilot_decision: WitnessAutopilotDecision | None = None
    autopilot_min_confidence: Decimal = Decimal("0.600")
    # ── Receptivity ────────────────────────────────────────────────────────
    user_feedback_events: tuple[UserFeedbackEvent, ...] = field(default_factory=tuple)


class ProactiveActionPolicy:
    """Mirror of the existing Witness delivery gates as a pure function.

    Reproduces the gate logic from runner._deliver_user_digest,
    runner._deliver_channel_group, and autopilot._review_candidate
    without any side effects. Reuses the canonical effective_confidence()
    and receptivity() functions from kortny.witness.receptivity; never
    duplicates constants or thresholds.

    Decision mapping (surface -> gate outcome -> LedgerOutcome):

    dm_digest:
        sent              -> act
        below_threshold   -> silent
        quiet_hours       -> defer (reason: quiet_hours)
        digest_interval   -> defer (reason: digest_interval)
        budget_deferred   -> defer (reason: budget_deferred)
        pre_epoch         -> defer (reason: pre_epoch)

    channel_post:
        channel_sent      -> act
        below_threshold   -> silent
        policy / paused   -> defer (reason: policy)
        no_epoch          -> defer (reason: no_epoch)
        pre_epoch         -> defer (reason: pre_epoch)
        quiet_hours       -> defer (reason: quiet_hours)
        budget            -> defer (reason: budget)

    autopilot:
        execute_task (safe) -> act
        draft_artifact      -> ask
        dismiss             -> silent
        preflight fail      -> defer
        safety fail         -> defer
        defer/monitor_only  -> defer
    """

    def decide(
        self,
        surface: LedgerSurface,
        candidate: CandidateInputs,
        ctx: DeliveryContext,
    ) -> LedgerDecision:
        if surface == "dm_digest":
            return self._decide_dm_digest(candidate, ctx)
        if surface == "channel_post":
            return self._decide_channel_post(candidate, ctx)
        if surface == "autopilot":
            return self._decide_autopilot(candidate, ctx)
        return LedgerDecision(decision="defer", reason_code="unknown_surface")

    # ── DM DIGEST ─────────────────────────────────────────────────────────
    # Gate order matches runner._deliver_digests + _deliver_user_digest:
    #   1. (caller gate: deliver_private OR digest_epoch set -- pre-filtered)
    #   2. Epoch filter: created_at < digest_epoch -> defer:pre_epoch
    #   3. Quiet hours -> defer:quiet_hours
    #   4. Digest interval already sent -> defer:digest_interval
    #   5. Score < threshold -> silent:below_threshold
    #   6. Budget (ranked position >= max_items) -> defer:budget_deferred
    #   7. -> act:sent
    def _decide_dm_digest(
        self, candidate: CandidateInputs, ctx: DeliveryContext
    ) -> LedgerDecision:
        if ctx.digest_epoch is not None and candidate.created_at < ctx.digest_epoch:
            return LedgerDecision(decision="defer", reason_code="pre_epoch")
        if ctx.in_quiet_hours:
            return LedgerDecision(decision="defer", reason_code="quiet_hours")
        if not ctx.digest_interval_exceeded:
            return LedgerDecision(decision="defer", reason_code="digest_interval")
        score = self._score(candidate, ctx)
        if score < ctx.delivery_threshold:
            return LedgerDecision(decision="silent", reason_code="below_threshold")
        if ctx.items_before_this >= ctx.digest_max_items:
            return LedgerDecision(decision="defer", reason_code="budget_deferred")
        return LedgerDecision(decision="act", reason_code="sent")

    # ── CHANNEL POST ───────────────────────────────────────────────────────
    # Gate order matches runner._deliver_channel_group:
    #   1. Policy gate: channel_full_enabled -> else defer:policy
    #   2. Epoch: channel_epoch is None -> defer:no_epoch
    #             created_at < epoch -> defer:pre_epoch
    #   3. Quiet hours -> defer:quiet_hours
    #   4. Score < threshold -> silent:below_threshold
    #   5. Budget: items_before_this_channel >= budget_left (budget exhausted) -> defer:budget
    #   6. -> act:channel_sent
    def _decide_channel_post(
        self, candidate: CandidateInputs, ctx: DeliveryContext
    ) -> LedgerDecision:
        if not ctx.channel_full_enabled:
            return LedgerDecision(decision="defer", reason_code="policy")
        if ctx.channel_epoch is None:
            return LedgerDecision(decision="defer", reason_code="no_epoch")
        if candidate.created_at < ctx.channel_epoch:
            return LedgerDecision(decision="defer", reason_code="pre_epoch")
        if ctx.in_quiet_hours:
            return LedgerDecision(decision="defer", reason_code="quiet_hours")
        score = self._score(candidate, ctx)
        if score < ctx.delivery_threshold:
            return LedgerDecision(decision="silent", reason_code="below_threshold")
        if ctx.channel_posts_budget_left <= ctx.items_before_this_channel:
            return LedgerDecision(decision="defer", reason_code="budget")
        return LedgerDecision(decision="act", reason_code="channel_sent")

    # ── AUTOPILOT ──────────────────────────────────────────────────────────
    # Gate order matches autopilot._review_candidate:
    #   1. Preflight fail -> defer:preflight_defer
    #   2. (LLM review already done; result in ctx.autopilot_decision)
    #   3. execute_task + draft_artifact -> ask:draft_artifact
    #   4. execute_task safety gates (risk, action_kind, delivery_target,
    #      requires_user_reply, allowed_without_confirmation, confidence,
    #      task_input) -> defer:safety_*
    #   5. should_execute -> act:execute_task
    #   6. dismiss -> silent:dismissed
    #   7. else -> defer:deferred
    def _decide_autopilot(
        self, candidate: CandidateInputs, ctx: DeliveryContext
    ) -> LedgerDecision:
        if ctx.autopilot_preflight_defer_reason is not None:
            return LedgerDecision(decision="defer", reason_code="preflight_defer")
        decision = ctx.autopilot_decision
        if decision is None:
            return LedgerDecision(decision="defer", reason_code="no_review")
        # Draft tier (HIG-230)
        if (
            decision.decision == "execute_task"
            and decision.action_kind == "draft_artifact"
        ):
            return LedgerDecision(decision="ask", reason_code="draft_artifact")
        if decision.decision == "execute_task":
            if decision.risk != "low":
                return LedgerDecision(decision="defer", reason_code="safety_high_risk")
            if decision.action_kind not in _AUTOPILOT_EXECUTABLE_ACTION_KINDS:
                return LedgerDecision(
                    decision="defer", reason_code="safety_action_kind"
                )
            if decision.delivery_target not in _AUTOPILOT_EXECUTABLE_DELIVERY_TARGETS:
                return LedgerDecision(
                    decision="defer", reason_code="safety_delivery_target"
                )
            if decision.requires_user_reply:
                return LedgerDecision(
                    decision="defer", reason_code="safety_requires_reply"
                )
            if not decision.allowed_without_confirmation:
                return LedgerDecision(
                    decision="defer", reason_code="safety_no_confirmation"
                )
            if decision.confidence_score < ctx.autopilot_min_confidence:
                return LedgerDecision(
                    decision="defer", reason_code="safety_low_confidence"
                )
            if not decision.task_input:
                return LedgerDecision(
                    decision="defer", reason_code="safety_no_task_input"
                )
            return LedgerDecision(decision="act", reason_code="execute_task")
        if decision.decision == "dismiss":
            return LedgerDecision(decision="silent", reason_code="dismissed")
        return LedgerDecision(decision="defer", reason_code="deferred")

    def _score(self, candidate: CandidateInputs, ctx: DeliveryContext) -> Decimal:
        """effective_confidence * receptivity -- the delivery gate score."""
        conf = effective_confidence(
            candidate.confidence_score,
            reinforcement_count=candidate.reinforcement_count,
            evidence_count=candidate.evidence_count,
            span_days=candidate.span_days,
        )
        rec = receptivity(
            list(ctx.user_feedback_events),
            candidate.candidate_type,
            ctx.now,
        )
        return Decimal(str(float(conf) * rec))
