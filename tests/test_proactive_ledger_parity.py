"""Parity tests for the Proactive Action Ledger (Step 1).

Each test fixture constructs the exact inputs that mirror a real Witness
gate outcome and asserts that ProactiveActionPolicy.decide() produces the
same decision. Tests are pure (no DB, no Slack, no LLM).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from kortny.witness.autopilot import (
    DEFAULT_WITNESS_AUTOPILOT_MIN_CONFIDENCE,
    WitnessAutopilotDecision,
)
from kortny.witness.ledger.policy import (
    CandidateInputs,
    DeliveryContext,
    ProactiveActionPolicy,
)
from kortny.witness.ledger.service import _normalise_real
from kortny.witness.receptivity import UserFeedbackEvent

NOW = datetime(2026, 6, 21, 14, 0, 0, tzinfo=UTC)
THRESHOLD = Decimal("0.55")


def _candidate(
    confidence: str = "0.80",
    reinforcement: int = 1,
    evidence: int = 0,
    span_days: int = 0,
    candidate_type: str = "suggestion",
    created_at: datetime | None = None,
) -> CandidateInputs:
    return CandidateInputs(
        confidence_score=Decimal(confidence),
        reinforcement_count=reinforcement,
        evidence_count=evidence,
        span_days=span_days,
        candidate_type=candidate_type,
        created_at=created_at or NOW - timedelta(hours=1),
    )


def _ctx_dm(**kwargs: object) -> DeliveryContext:
    defaults: dict[str, object] = dict(
        now=NOW,
        delivery_threshold=THRESHOLD,
        digest_epoch=None,
        digest_interval_exceeded=True,
        digest_max_items=5,
        items_before_this=0,
        in_quiet_hours=False,
        user_feedback_events=(),
    )
    defaults.update(kwargs)
    return DeliveryContext(**defaults)  # type: ignore[arg-type]


def _ctx_channel(**kwargs: object) -> DeliveryContext:
    epoch = NOW - timedelta(days=30)
    defaults: dict[str, object] = dict(
        now=NOW,
        delivery_threshold=THRESHOLD,
        channel_full_enabled=True,
        channel_epoch=epoch,
        channel_posts_budget_left=1,
        items_before_this_channel=0,
        in_quiet_hours=False,
        user_feedback_events=(),
    )
    defaults.update(kwargs)
    return DeliveryContext(**defaults)  # type: ignore[arg-type]


def _autopilot_decision(
    decision: str = "execute_task",
    risk: str = "low",
    action_kind: str = "read_only_analysis",
    delivery_target: str = "channel",
    requires_user_reply: bool = False,
    allowed_without_confirmation: bool = True,
    confidence: str = "0.75",
    task_input: str | None = "Check the project status.",
) -> WitnessAutopilotDecision:
    return WitnessAutopilotDecision(
        decision=decision,
        risk=risk,
        action_kind=action_kind,
        delivery_target=delivery_target,
        requires_user_reply=requires_user_reply,
        allowed_without_confirmation=allowed_without_confirmation,
        reason="test reason",
        task_input=task_input,
        confidence_score=Decimal(confidence),
    )


def _ctx_autopilot(**kwargs: object) -> DeliveryContext:
    defaults: dict[str, object] = dict(
        now=NOW,
        delivery_threshold=THRESHOLD,
        autopilot_preflight_defer_reason=None,
        autopilot_decision=_autopilot_decision(),
        autopilot_min_confidence=DEFAULT_WITNESS_AUTOPILOT_MIN_CONFIDENCE,
    )
    defaults.update(kwargs)
    return DeliveryContext(**defaults)  # type: ignore[arg-type]


policy = ProactiveActionPolicy()


# ── DM DIGEST ──────────────────────────────────────────────────────────────────


class TestDmDigest:
    def test_above_threshold_all_gates_pass(self) -> None:
        result = policy.decide("dm_digest", _candidate(confidence="0.80"), _ctx_dm())
        assert result.decision == "act"
        assert result.reason_code == "sent"

    def test_below_threshold_silent(self) -> None:
        result = policy.decide("dm_digest", _candidate(confidence="0.30"), _ctx_dm())
        assert result.decision == "silent"
        assert result.reason_code == "below_threshold"

    def test_quiet_hours_defer(self) -> None:
        result = policy.decide("dm_digest", _candidate(), _ctx_dm(in_quiet_hours=True))
        assert result.decision == "defer"
        assert result.reason_code == "quiet_hours"

    def test_digest_interval_not_exceeded_defer(self) -> None:
        result = policy.decide(
            "dm_digest", _candidate(), _ctx_dm(digest_interval_exceeded=False)
        )
        assert result.decision == "defer"
        assert result.reason_code == "digest_interval"

    def test_budget_exhausted_defer(self) -> None:
        result = policy.decide(
            "dm_digest",
            _candidate(),
            _ctx_dm(items_before_this=5, digest_max_items=5),
        )
        assert result.decision == "defer"
        assert result.reason_code == "budget_deferred"

    def test_pre_epoch_defer(self) -> None:
        # epoch is NOW; candidate created 2h ago => pre-epoch
        epoch = NOW
        old_candidate = _candidate(created_at=NOW - timedelta(hours=2))
        result = policy.decide("dm_digest", old_candidate, _ctx_dm(digest_epoch=epoch))
        assert result.decision == "defer"
        assert result.reason_code == "pre_epoch"

    def test_post_epoch_act(self) -> None:
        # epoch was 3h ago; candidate created 1h ago => post-epoch
        epoch = NOW - timedelta(hours=3)
        fresh_candidate = _candidate(created_at=NOW - timedelta(hours=1))
        result = policy.decide(
            "dm_digest", fresh_candidate, _ctx_dm(digest_epoch=epoch)
        )
        assert result.decision == "act"

    def test_no_epoch_filter_when_epoch_is_none(self) -> None:
        # epoch=None disables epoch filter; old candidate still passes
        old_candidate = _candidate(created_at=NOW - timedelta(days=30))
        result = policy.decide("dm_digest", old_candidate, _ctx_dm(digest_epoch=None))
        assert result.decision == "act"

    def test_receptivity_penalized_by_dismissals_goes_silent(self) -> None:
        # Three recent dismissals of the same category tank receptivity
        events = tuple(
            UserFeedbackEvent(
                action="dismissed",
                category="suggestion",
                at=NOW - timedelta(days=1),
            )
            for _ in range(3)
        )
        # confidence=0.70 alone would pass, but 3 dismissals each * 0.6 penalty
        result = policy.decide(
            "dm_digest",
            _candidate(confidence="0.70"),
            _ctx_dm(user_feedback_events=events),
        )
        assert result.decision == "silent"

    def test_quiet_hours_checked_before_score(self) -> None:
        # Even 0.99 confidence yields defer when quiet hours apply
        result = policy.decide(
            "dm_digest",
            _candidate(confidence="0.99"),
            _ctx_dm(in_quiet_hours=True),
        )
        assert result.decision == "defer"
        assert result.reason_code == "quiet_hours"

    def test_budget_one_slot_last_candidate_acts(self) -> None:
        # items_before_this=4, max_items=5 => 4 < 5 => should act
        result = policy.decide(
            "dm_digest",
            _candidate(),
            _ctx_dm(items_before_this=4, digest_max_items=5),
        )
        assert result.decision == "act"

    def test_interval_check_comes_after_quiet_hours(self) -> None:
        # Gate order: quiet_hours (gate 3) before digest_interval (gate 4)
        result = policy.decide(
            "dm_digest",
            _candidate(),
            _ctx_dm(in_quiet_hours=True, digest_interval_exceeded=False),
        )
        assert result.reason_code == "quiet_hours"


# ── CHANNEL POST ───────────────────────────────────────────────────────────────


class TestChannelPost:
    def test_all_gates_pass(self) -> None:
        result = policy.decide("channel_post", _candidate(), _ctx_channel())
        assert result.decision == "act"
        assert result.reason_code == "channel_sent"

    def test_policy_not_full_defer(self) -> None:
        result = policy.decide(
            "channel_post",
            _candidate(),
            _ctx_channel(channel_full_enabled=False),
        )
        assert result.decision == "defer"
        assert result.reason_code == "policy"

    def test_no_epoch_defer(self) -> None:
        result = policy.decide(
            "channel_post",
            _candidate(),
            _ctx_channel(channel_epoch=None),
        )
        assert result.decision == "defer"
        assert result.reason_code == "no_epoch"

    def test_pre_epoch_defer(self) -> None:
        # epoch is NOW; candidate created yesterday => pre-epoch
        epoch = NOW
        old_candidate = _candidate(created_at=NOW - timedelta(days=1))
        result = policy.decide(
            "channel_post", old_candidate, _ctx_channel(channel_epoch=epoch)
        )
        assert result.decision == "defer"
        assert result.reason_code == "pre_epoch"

    def test_quiet_hours_defer(self) -> None:
        result = policy.decide(
            "channel_post",
            _candidate(),
            _ctx_channel(in_quiet_hours=True),
        )
        assert result.decision == "defer"
        assert result.reason_code == "quiet_hours"

    def test_below_threshold_silent(self) -> None:
        result = policy.decide(
            "channel_post",
            _candidate(confidence="0.20"),
            _ctx_channel(),
        )
        assert result.decision == "silent"
        assert result.reason_code == "below_threshold"

    def test_budget_exhausted_defer(self) -> None:
        # budget_left=1, items_before=1 => 1 <= 1 => defer
        result = policy.decide(
            "channel_post",
            _candidate(),
            _ctx_channel(channel_posts_budget_left=1, items_before_this_channel=1),
        )
        assert result.decision == "defer"
        assert result.reason_code == "budget"

    def test_budget_zero_defer(self) -> None:
        # budget_left=0 <= items_before=0 => defer
        result = policy.decide(
            "channel_post",
            _candidate(),
            _ctx_channel(channel_posts_budget_left=0, items_before_this_channel=0),
        )
        assert result.decision == "defer"
        assert result.reason_code == "budget"

    def test_gate_order_policy_before_epoch(self) -> None:
        # Policy fails before epoch check; should report policy not no_epoch
        result = policy.decide(
            "channel_post",
            _candidate(created_at=NOW - timedelta(days=1)),
            _ctx_channel(channel_full_enabled=False, channel_epoch=None),
        )
        assert result.reason_code == "policy"

    def test_gate_order_epoch_before_quiet_hours(self) -> None:
        # pre_epoch should be returned before quiet_hours check
        result = policy.decide(
            "channel_post",
            _candidate(created_at=NOW - timedelta(days=1)),
            _ctx_channel(channel_epoch=NOW, in_quiet_hours=True),
        )
        assert result.reason_code == "pre_epoch"

    def test_gate_order_quiet_hours_before_score(self) -> None:
        # quiet_hours should fire before the score threshold check
        result = policy.decide(
            "channel_post",
            _candidate(confidence="0.10"),
            _ctx_channel(in_quiet_hours=True),
        )
        assert result.reason_code == "quiet_hours"


# ── AUTOPILOT ──────────────────────────────────────────────────────────────────


class TestAutopilot:
    def test_execute_low_risk_read_only_act(self) -> None:
        ctx = _ctx_autopilot(
            autopilot_decision=_autopilot_decision(
                decision="execute_task",
                risk="low",
                action_kind="read_only_analysis",
                delivery_target="channel",
            )
        )
        result = policy.decide("autopilot", _candidate(), ctx)
        assert result.decision == "act"
        assert result.reason_code == "execute_task"

    def test_execute_status_check_dm_act(self) -> None:
        ctx = _ctx_autopilot(
            autopilot_decision=_autopilot_decision(
                decision="execute_task",
                risk="low",
                action_kind="status_check",
                delivery_target="dm",
            )
        )
        result = policy.decide("autopilot", _candidate(), ctx)
        assert result.decision == "act"

    def test_draft_artifact_ask(self) -> None:
        ctx = _ctx_autopilot(
            autopilot_decision=_autopilot_decision(
                decision="execute_task",
                action_kind="draft_artifact",
            )
        )
        result = policy.decide("autopilot", _candidate(), ctx)
        assert result.decision == "ask"
        assert result.reason_code == "draft_artifact"

    def test_preflight_defer_reason_set(self) -> None:
        ctx = _ctx_autopilot(
            autopilot_preflight_defer_reason="Candidate has no Slack delivery target."
        )
        result = policy.decide("autopilot", _candidate(), ctx)
        assert result.decision == "defer"
        assert result.reason_code == "preflight_defer"

    def test_high_risk_defer(self) -> None:
        ctx = _ctx_autopilot(
            autopilot_decision=_autopilot_decision(
                decision="execute_task",
                risk="high",
                action_kind="read_only_analysis",
            )
        )
        result = policy.decide("autopilot", _candidate(), ctx)
        assert result.decision == "defer"
        assert result.reason_code == "safety_high_risk"

    def test_non_executable_action_kind_defer(self) -> None:
        ctx = _ctx_autopilot(
            autopilot_decision=_autopilot_decision(
                decision="execute_task",
                risk="low",
                action_kind="external_write",
            )
        )
        result = policy.decide("autopilot", _candidate(), ctx)
        assert result.decision == "defer"
        assert result.reason_code == "safety_action_kind"

    def test_requires_user_reply_defer(self) -> None:
        ctx = _ctx_autopilot(
            autopilot_decision=_autopilot_decision(
                decision="execute_task",
                risk="low",
                requires_user_reply=True,
            )
        )
        result = policy.decide("autopilot", _candidate(), ctx)
        assert result.decision == "defer"
        assert result.reason_code == "safety_requires_reply"

    def test_not_allowed_without_confirmation_defer(self) -> None:
        ctx = _ctx_autopilot(
            autopilot_decision=_autopilot_decision(
                decision="execute_task",
                risk="low",
                allowed_without_confirmation=False,
            )
        )
        result = policy.decide("autopilot", _candidate(), ctx)
        assert result.decision == "defer"
        assert result.reason_code == "safety_no_confirmation"

    def test_low_confidence_defer(self) -> None:
        ctx = _ctx_autopilot(
            autopilot_decision=_autopilot_decision(
                decision="execute_task",
                risk="low",
                confidence="0.55",
            ),
            autopilot_min_confidence=Decimal("0.600"),
        )
        result = policy.decide("autopilot", _candidate(), ctx)
        assert result.decision == "defer"
        assert result.reason_code == "safety_low_confidence"

    def test_dismiss_decision_silent(self) -> None:
        ctx = _ctx_autopilot(autopilot_decision=_autopilot_decision(decision="dismiss"))
        result = policy.decide("autopilot", _candidate(), ctx)
        assert result.decision == "silent"
        assert result.reason_code == "dismissed"

    def test_defer_decision_defer(self) -> None:
        ctx = _ctx_autopilot(autopilot_decision=_autopilot_decision(decision="defer"))
        result = policy.decide("autopilot", _candidate(), ctx)
        assert result.decision == "defer"
        assert result.reason_code == "deferred"

    def test_monitor_only_decision_defer(self) -> None:
        ctx = _ctx_autopilot(
            autopilot_decision=_autopilot_decision(decision="monitor_only")
        )
        result = policy.decide("autopilot", _candidate(), ctx)
        assert result.decision == "defer"
        assert result.reason_code == "deferred"

    def test_unknown_delivery_target_defer(self) -> None:
        ctx = _ctx_autopilot(
            autopilot_decision=_autopilot_decision(
                decision="execute_task",
                risk="low",
                delivery_target="unknown",
            )
        )
        result = policy.decide("autopilot", _candidate(), ctx)
        assert result.decision == "defer"
        assert result.reason_code == "safety_delivery_target"

    def test_no_task_input_defer(self) -> None:
        ctx = _ctx_autopilot(
            autopilot_decision=_autopilot_decision(
                decision="execute_task",
                risk="low",
                task_input=None,
            )
        )
        result = policy.decide("autopilot", _candidate(), ctx)
        assert result.decision == "defer"
        assert result.reason_code == "safety_no_task_input"

    def test_no_review_no_autopilot_decision(self) -> None:
        ctx = _ctx_autopilot(autopilot_decision=None)
        result = policy.decide("autopilot", _candidate(), ctx)
        assert result.decision == "defer"
        assert result.reason_code == "no_review"

    def test_preflight_checked_before_decision(self) -> None:
        # Even with a valid execute_task decision, preflight takes priority
        ctx = _ctx_autopilot(
            autopilot_preflight_defer_reason="re-execute cooldown active",
            autopilot_decision=_autopilot_decision(decision="execute_task"),
        )
        result = policy.decide("autopilot", _candidate(), ctx)
        assert result.reason_code == "preflight_defer"

    def test_ask_user_decision_defer(self) -> None:
        ctx = _ctx_autopilot(
            autopilot_decision=_autopilot_decision(decision="ask_user")
        )
        result = policy.decide("autopilot", _candidate(), ctx)
        assert result.decision == "defer"
        assert result.reason_code == "deferred"

    def test_medium_risk_defer(self) -> None:
        ctx = _ctx_autopilot(
            autopilot_decision=_autopilot_decision(
                decision="execute_task",
                risk="medium",
                action_kind="read_only_analysis",
            )
        )
        result = policy.decide("autopilot", _candidate(), ctx)
        assert result.decision == "defer"
        assert result.reason_code == "safety_high_risk"


# ── NORMALISE_REAL MAPPING ─────────────────────────────────────────────────────


class TestNormaliseReal:
    def test_sent_maps_to_act(self) -> None:
        assert _normalise_real("sent", "dm_digest") == "act"

    def test_channel_sent_maps_to_act(self) -> None:
        assert _normalise_real("channel_sent", "channel_post") == "act"

    def test_execute_task_maps_to_act(self) -> None:
        assert _normalise_real("execute_task", "autopilot") == "act"

    def test_silent_maps_to_silent(self) -> None:
        assert _normalise_real("silent", "dm_digest") == "silent"

    def test_dismissed_maps_to_silent(self) -> None:
        assert _normalise_real("dismissed", "autopilot") == "silent"

    def test_dismiss_maps_to_silent(self) -> None:
        assert _normalise_real("dismiss", "autopilot") == "silent"

    def test_below_threshold_maps_to_silent(self) -> None:
        assert _normalise_real("below_threshold", "dm_digest") == "silent"

    def test_quiet_hours_deferred_maps_to_defer(self) -> None:
        assert _normalise_real("quiet_hours_deferred", "dm_digest") == "defer"

    def test_budget_deferred_maps_to_defer(self) -> None:
        assert _normalise_real("budget_deferred", "dm_digest") == "defer"

    def test_interval_deferred_maps_to_defer(self) -> None:
        assert _normalise_real("interval_deferred", "dm_digest") == "defer"

    def test_draft_artifact_maps_to_ask(self) -> None:
        assert _normalise_real("draft_artifact", "autopilot") == "ask"

    def test_ask_user_maps_to_ask(self) -> None:
        assert _normalise_real("ask_user", "autopilot") == "ask"

    def test_unknown_string_maps_to_defer(self) -> None:
        assert _normalise_real("something_unexpected", "dm_digest") == "defer"
