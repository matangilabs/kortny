"""Tests for Proactive Action Ledger Chunk 3: autopilot cutover + faithful shadow.

Covers:
- Cutover ON: execute-eligible candidate → policy says act → same outcome as OFF.
- Cutover ON: high-risk / bad action-kind / low-confidence → defer/dismiss → same as OFF.
- on==off parity matrix: for a range of decision inputs, flag-on and flag-off produce
  the same autopilot outcome (executed / deferred / dismissed).
- Faithful in-place shadow: divergence is logged when policy disagrees with a synthetic
  real decision; errors inside the shadow are always swallowed.
- autopilot_decision ledger event is recorded with the right policy_decision field.

The decision-logic tests are pure (no DB, no Slack, no LLM).
The ledger-event tests need Postgres and use the same session fixture pattern as the
other proactive ledger test files.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import Engine, delete, select
from sqlalchemy.orm import Session

from kortny.db.models import (
    Installation,
    ProactiveActionEvent,
    Task,
    TaskEvent,
    WitnessOpportunityCandidate,
)
from kortny.db.session import make_engine, make_session_factory, normalize_database_url
from kortny.witness.autopilot import (
    DEFAULT_WITNESS_AUTOPILOT_MIN_CONFIDENCE,
    WitnessAutopilotDecision,
    _autopilot_cutover_enabled,
    _autopilot_shadow_evaluate,
    _inline_record_autopilot_decision_event,
)
from kortny.witness.ledger.policy import (
    CandidateInputs,
    DeliveryContext,
    LedgerDecision,
    ProactiveActionPolicy,
)

TEST_POSTGRES_URL = os.environ.get("KORTNY_TEST_POSTGRES_URL")

NOW = datetime(2026, 6, 22, 12, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _decision(
    decision: str = "execute_task",
    risk: str = "low",
    action_kind: str = "read_only_analysis",
    delivery_target: str = "channel",
    requires_user_reply: bool = False,
    allowed_without_confirmation: bool = True,
    confidence: str = "0.750",
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


def _candidate_inputs(
    confidence: str = "0.750",
    reinforcement: int = 1,
    evidence: int = 0,
) -> CandidateInputs:
    return CandidateInputs(
        confidence_score=Decimal(confidence),
        reinforcement_count=reinforcement,
        evidence_count=evidence,
        span_days=0,
        candidate_type="recurring_check",
        created_at=NOW - timedelta(hours=1),
    )


def _ctx(
    *,
    preflight_defer_reason: str | None = None,
    autopilot_decision: WitnessAutopilotDecision | None = None,
    min_confidence: Decimal = DEFAULT_WITNESS_AUTOPILOT_MIN_CONFIDENCE,
) -> DeliveryContext:
    return DeliveryContext(
        now=NOW,
        delivery_threshold=Decimal("0.550"),
        autopilot_preflight_defer_reason=preflight_defer_reason,
        autopilot_decision=autopilot_decision,
        autopilot_min_confidence=min_confidence,
    )


def _policy_outcome(
    *,
    preflight_defer_reason: str | None = None,
    autopilot_decision: WitnessAutopilotDecision | None = None,
) -> LedgerDecision:
    policy = ProactiveActionPolicy()
    return policy.decide(
        "autopilot",
        _candidate_inputs(),
        _ctx(
            preflight_defer_reason=preflight_defer_reason,
            autopilot_decision=autopilot_decision,
        ),
    )


# ---------------------------------------------------------------------------
# Policy parity (pure, no DB) — validate that the policy outcomes match the
# inline gate outcomes for a range of inputs.  This is the foundation for
# the cutover claim: flag-on and flag-off produce the same outcome.
# ---------------------------------------------------------------------------


class TestPolicyParityMatrix:
    """Matrix of (decision-inputs, expected-inline-outcome, expected-policy-outcome).

    For each case we verify:
    1. The expected inline outcome is what the autopilot inline gate would
       produce (via should_execute / decision.decision checks).
    2. The policy produces the same LedgerOutcome.

    This proves that flag-on == flag-off for the same inputs.
    """

    # ── Execute-eligible (low-risk read-only with sufficient confidence) ──

    def test_execute_eligible_read_only_analysis(self) -> None:
        d = _decision(
            decision="execute_task",
            risk="low",
            action_kind="read_only_analysis",
            delivery_target="channel",
            requires_user_reply=False,
            allowed_without_confirmation=True,
            confidence="0.750",
            task_input="Check status.",
        )
        assert d.should_execute  # inline gate would execute
        ledger = _policy_outcome(autopilot_decision=d)
        assert ledger.decision == "act"
        assert ledger.reason_code == "execute_task"

    def test_execute_eligible_status_check(self) -> None:
        d = _decision(action_kind="status_check")
        assert d.should_execute
        ledger = _policy_outcome(autopilot_decision=d)
        assert ledger.decision == "act"

    def test_execute_eligible_dm_delivery(self) -> None:
        d = _decision(delivery_target="dm")
        assert d.should_execute
        ledger = _policy_outcome(autopilot_decision=d)
        assert ledger.decision == "act"

    # ── Safety gates: high-risk ──

    def test_high_risk_deferred(self) -> None:
        d = _decision(risk="high")
        assert not d.should_execute
        ledger = _policy_outcome(autopilot_decision=d)
        assert ledger.decision == "defer"
        assert ledger.reason_code == "safety_high_risk"

    def test_medium_risk_deferred(self) -> None:
        d = _decision(risk="medium")
        assert not d.should_execute
        ledger = _policy_outcome(autopilot_decision=d)
        assert ledger.decision == "defer"

    # ── Safety gates: bad action_kind ──

    def test_schedule_management_deferred(self) -> None:
        d = _decision(action_kind="schedule_management")
        assert not d.should_execute
        ledger = _policy_outcome(autopilot_decision=d)
        assert ledger.decision == "defer"
        assert ledger.reason_code == "safety_action_kind"

    def test_memory_write_deferred(self) -> None:
        d = _decision(action_kind="memory_write")
        assert not d.should_execute
        ledger = _policy_outcome(autopilot_decision=d)
        assert ledger.decision == "defer"

    def test_external_write_deferred(self) -> None:
        d = _decision(action_kind="external_write")
        ledger = _policy_outcome(autopilot_decision=d)
        assert ledger.decision == "defer"

    # ── Safety gates: delivery target ──

    def test_unknown_delivery_target_deferred(self) -> None:
        d = _decision(delivery_target="unknown")
        assert not d.should_execute
        ledger = _policy_outcome(autopilot_decision=d)
        assert ledger.decision == "defer"
        assert ledger.reason_code == "safety_delivery_target"

    def test_none_delivery_target_deferred(self) -> None:
        d = _decision(delivery_target="none")
        ledger = _policy_outcome(autopilot_decision=d)
        assert ledger.decision == "defer"

    # ── Safety gates: interaction flags ──

    def test_requires_user_reply_deferred(self) -> None:
        d = _decision(requires_user_reply=True)
        assert not d.should_execute
        ledger = _policy_outcome(autopilot_decision=d)
        assert ledger.decision == "defer"
        assert ledger.reason_code == "safety_requires_reply"

    def test_not_allowed_without_confirmation_deferred(self) -> None:
        d = _decision(allowed_without_confirmation=False)
        assert not d.should_execute
        ledger = _policy_outcome(autopilot_decision=d)
        assert ledger.decision == "defer"
        assert ledger.reason_code == "safety_no_confirmation"

    # ── Safety gates: confidence ──

    def test_low_confidence_deferred(self) -> None:
        d = _decision(confidence="0.599")
        assert not d.should_execute
        ledger = _policy_outcome(autopilot_decision=d)
        assert ledger.decision == "defer"
        assert ledger.reason_code == "safety_low_confidence"

    def test_boundary_confidence_executes(self) -> None:
        d = _decision(confidence="0.600")
        assert d.should_execute
        ledger = _policy_outcome(autopilot_decision=d)
        assert ledger.decision == "act"

    # ── Safety gates: no task_input ──

    def test_no_task_input_deferred(self) -> None:
        d = _decision(task_input=None)
        assert not d.should_execute
        ledger = _policy_outcome(autopilot_decision=d)
        assert ledger.decision == "defer"
        assert ledger.reason_code == "safety_no_task_input"

    # ── Dismiss ──

    def test_dismiss_maps_to_silent(self) -> None:
        d = _decision(decision="dismiss")
        assert not d.should_execute
        assert d.decision == "dismiss"
        ledger = _policy_outcome(autopilot_decision=d)
        assert ledger.decision == "silent"
        assert ledger.reason_code == "dismissed"

    # ── Draft artifact ──

    def test_draft_artifact_maps_to_ask(self) -> None:
        d = _decision(decision="execute_task", action_kind="draft_artifact")
        ledger = _policy_outcome(autopilot_decision=d)
        assert ledger.decision == "ask"
        assert ledger.reason_code == "draft_artifact"

    # ── Non-execute decisions: defer/monitor_only/ask_user ──

    def test_defer_decision_maps_to_defer(self) -> None:
        d = _decision(decision="defer")
        ledger = _policy_outcome(autopilot_decision=d)
        assert ledger.decision == "defer"
        assert ledger.reason_code == "deferred"

    def test_monitor_only_maps_to_defer(self) -> None:
        d = _decision(decision="monitor_only")
        ledger = _policy_outcome(autopilot_decision=d)
        assert ledger.decision == "defer"

    def test_ask_user_maps_to_defer(self) -> None:
        d = _decision(decision="ask_user")
        ledger = _policy_outcome(autopilot_decision=d)
        assert ledger.decision == "defer"

    # ── Preflight failure ──

    def test_preflight_failure_maps_to_defer(self) -> None:
        ledger = _policy_outcome(
            preflight_defer_reason="Not in active channel.",
            autopilot_decision=_decision(),
        )
        assert ledger.decision == "defer"
        assert ledger.reason_code == "preflight_defer"

    # ── No review (decision is None) ──

    def test_no_review_maps_to_defer(self) -> None:
        ledger = _policy_outcome(autopilot_decision=None)
        assert ledger.decision == "defer"
        assert ledger.reason_code == "no_review"


# ---------------------------------------------------------------------------
# _autopilot_cutover_enabled helper
# ---------------------------------------------------------------------------


class TestAutopilotCutoverEnabled:
    def test_default_is_false(self) -> None:
        with patch("kortny.witness.autopilot.load_settings") as mock_settings_loader:
            mock_settings = MagicMock()
            mock_settings.kortny_proactive_ledger_autopilot_cutover = False
            mock_settings_loader.return_value = mock_settings
            assert _autopilot_cutover_enabled() is False

    def test_returns_true_when_flag_set(self) -> None:
        with patch("kortny.witness.autopilot.load_settings") as mock_settings_loader:
            mock_settings = MagicMock()
            mock_settings.kortny_proactive_ledger_autopilot_cutover = True
            mock_settings_loader.return_value = mock_settings
            assert _autopilot_cutover_enabled() is True

    def test_returns_false_on_load_error(self) -> None:
        with patch(
            "kortny.witness.autopilot.load_settings", side_effect=RuntimeError("boom")
        ):
            assert _autopilot_cutover_enabled() is False


# ---------------------------------------------------------------------------
# Faithful in-place shadow: divergence logging + error swallowing
# ---------------------------------------------------------------------------


class TestAutopilotShadowEvaluate:
    """Tests for _autopilot_shadow_evaluate.

    The shadow must:
    - Be silenced when KORTNY_PROACTIVE_LEDGER_SHADOW_ENABLED=false.
    - Log a warning on divergence.
    - Swallow any exception without re-raising.
    """

    def _make_orm_candidate(self) -> MagicMock:
        """Build a minimal mock WitnessOpportunityCandidate."""
        candidate = MagicMock()
        candidate.id = uuid.uuid4()
        candidate.confidence_score = Decimal("0.750")
        candidate.reinforcement_count = 1
        candidate.evidence_json = []
        candidate.candidate_type = "recurring_check"
        candidate.created_at = NOW - timedelta(hours=1)
        return candidate

    def test_no_divergence_no_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """When policy agrees with the real decision, no divergence log."""
        candidate = self._make_orm_candidate()
        d = _decision(
            decision="execute_task",
            risk="low",
            action_kind="read_only_analysis",
            delivery_target="channel",
            confidence="0.750",
        )
        import logging

        with caplog.at_level(logging.WARNING, logger="kortny.witness.autopilot"):
            _autopilot_shadow_evaluate(
                candidate=candidate,
                decision=d,
                preflight_defer_reason=None,
                now=NOW,
            )
        divergence_logs = [
            r for r in caplog.records if "shadow_divergence" in r.message
        ]
        assert len(divergence_logs) == 0

    def test_divergence_logs_warning(self) -> None:
        """Force a divergence by injecting a policy that disagrees.

        Uses patch on the logger.warning call to avoid caplog ordering issues
        when run after tests that import witness modules into the logging tree.
        """
        candidate = self._make_orm_candidate()
        d = _decision(
            decision="execute_task",
            risk="low",
            action_kind="read_only_analysis",
            delivery_target="channel",
            confidence="0.750",
        )

        # Patch the policy (at source) to always return "silent" to force divergence
        # with a real decision of "act" (execute-eligible).
        # Also patch logger.warning at the point of use so we don't rely on
        # caplog's test-order-dependent log level propagation.
        from kortny.witness.ledger.policy import LedgerDecision

        logged_messages: list[str] = []

        def capture_warning(msg: str, *args: object, **kwargs: object) -> None:
            logged_messages.append(msg % args if args else msg)

        with patch(
            "kortny.witness.ledger.policy.ProactiveActionPolicy.decide",
            return_value=LedgerDecision(decision="silent", reason_code="test_diverge"),
        ):
            with patch(
                "kortny.witness.autopilot.logger.warning", side_effect=capture_warning
            ):
                _autopilot_shadow_evaluate(
                    candidate=candidate,
                    decision=d,
                    preflight_defer_reason=None,
                    now=NOW,
                )

        divergence_msgs = [m for m in logged_messages if "shadow_divergence" in m]
        assert len(divergence_msgs) == 1
        assert "surface=autopilot" in divergence_msgs[0]

    def test_exception_inside_shadow_is_swallowed(self) -> None:
        """An exception inside the shadow must never propagate."""
        candidate = self._make_orm_candidate()
        d = _decision()

        with patch(
            "kortny.witness.autopilot._candidate_inputs_from_orm",
            side_effect=RuntimeError("internal shadow failure"),
        ):
            # Must not raise.
            _autopilot_shadow_evaluate(
                candidate=candidate,
                decision=d,
                preflight_defer_reason=None,
                now=NOW,
            )

    def test_shadow_disabled_by_env(self, caplog: pytest.LogCaptureFixture) -> None:
        """When KORTNY_PROACTIVE_LEDGER_SHADOW_ENABLED=false, nothing runs."""
        candidate = self._make_orm_candidate()
        d = _decision()

        with patch.dict(
            os.environ, {"KORTNY_PROACTIVE_LEDGER_SHADOW_ENABLED": "false"}
        ):
            # Patch _candidate_inputs_from_orm: if the shadow runs past the early
            # return, it will call this function. Assert it is NOT called.
            with patch(
                "kortny.witness.autopilot._candidate_inputs_from_orm"
            ) as mock_inputs:
                import logging

                with caplog.at_level(
                    logging.WARNING, logger="kortny.witness.autopilot"
                ):
                    _autopilot_shadow_evaluate(
                        candidate=candidate,
                        decision=d,
                        preflight_defer_reason=None,
                        now=NOW,
                    )
                # Shadow must exit before building policy inputs.
                mock_inputs.assert_not_called()

        divergence_logs = [
            r for r in caplog.records if "shadow_divergence" in r.message
        ]
        assert len(divergence_logs) == 0

    def test_shadow_dismiss_no_divergence(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Dismiss decision: real=silent, policy=silent — no divergence."""
        candidate = self._make_orm_candidate()
        d = _decision(decision="dismiss")

        import logging

        with caplog.at_level(logging.WARNING, logger="kortny.witness.autopilot"):
            _autopilot_shadow_evaluate(
                candidate=candidate,
                decision=d,
                preflight_defer_reason=None,
                now=NOW,
            )
        divergence_logs = [
            r for r in caplog.records if "shadow_divergence" in r.message
        ]
        assert len(divergence_logs) == 0

    def test_shadow_defer_no_divergence(self, caplog: pytest.LogCaptureFixture) -> None:
        """Defer decision: real=defer, policy=defer — no divergence."""
        candidate = self._make_orm_candidate()
        d = _decision(decision="defer")

        import logging

        with caplog.at_level(logging.WARNING, logger="kortny.witness.autopilot"):
            _autopilot_shadow_evaluate(
                candidate=candidate,
                decision=d,
                preflight_defer_reason=None,
                now=NOW,
            )
        divergence_logs = [
            r for r in caplog.records if "shadow_divergence" in r.message
        ]
        assert len(divergence_logs) == 0

    def test_shadow_draft_artifact_no_divergence(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Draft artifact: real=ask, policy=ask — no divergence."""
        candidate = self._make_orm_candidate()
        d = _decision(decision="execute_task", action_kind="draft_artifact")

        import logging

        with caplog.at_level(logging.WARNING, logger="kortny.witness.autopilot"):
            _autopilot_shadow_evaluate(
                candidate=candidate,
                decision=d,
                preflight_defer_reason=None,
                now=NOW,
            )
        divergence_logs = [
            r for r in caplog.records if "shadow_divergence" in r.message
        ]
        assert len(divergence_logs) == 0


# ---------------------------------------------------------------------------
# _inline_record_autopilot_decision_event: policy_decision mapping
# ---------------------------------------------------------------------------


class TestInlineRecordAutopilotDecisionEvent:
    """Verify that the inline event recorder uses the correct policy_decision value."""

    def _call(self, d: WitnessAutopilotDecision) -> str | None:
        """Call _inline_record_autopilot_decision_event and return policy_decision."""
        recorded: list[str | None] = []

        def fake_record_transition(
            session: object,
            candidate: object,
            *,
            to_state: str,
            event_type: str,
            policy_decision: str | None = None,
            reason_code: str | None = None,
            actor_id: str | None = None,
            now: datetime,
        ) -> None:
            recorded.append(policy_decision)

        with patch("kortny.witness.autopilot._get_ledger") as mock_get_ledger:
            mock_ledger = MagicMock()
            mock_ledger.record_transition.side_effect = fake_record_transition
            mock_get_ledger.return_value = mock_ledger

            candidate = MagicMock()
            candidate.status = "candidate"

            _inline_record_autopilot_decision_event(
                MagicMock(),  # session
                candidate,
                decision=d,
                actor_id="test_actor",
                now=NOW,
            )

        return recorded[0] if recorded else None

    def test_execute_eligible_maps_to_act(self) -> None:
        d = _decision(decision="execute_task", action_kind="read_only_analysis")
        assert self._call(d) == "act"

    def test_dismiss_maps_to_silent(self) -> None:
        d = _decision(decision="dismiss")
        assert self._call(d) == "silent"

    def test_draft_artifact_maps_to_ask(self) -> None:
        d = _decision(decision="execute_task", action_kind="draft_artifact")
        assert self._call(d) == "ask"

    def test_defer_decision_maps_to_defer(self) -> None:
        d = _decision(decision="defer")
        assert self._call(d) == "defer"

    def test_safety_blocked_execute_task_maps_to_defer(self) -> None:
        # execute_task but high-risk → inline gate would defer → policy_decision=defer
        d = _decision(decision="execute_task", risk="high")
        assert self._call(d) == "defer"

    def test_monitor_only_maps_to_defer(self) -> None:
        d = _decision(decision="monitor_only")
        assert self._call(d) == "defer"


# ---------------------------------------------------------------------------
# Ledger event integration tests (require Postgres)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Ledger event integration tests (require Postgres)
# ---------------------------------------------------------------------------

_pg_skipif = pytest.mark.skipif(
    TEST_POSTGRES_URL is None,
    reason="KORTNY_TEST_POSTGRES_URL is required for ledger event tests",
)


@pytest.fixture(scope="module")
def _pg_engine() -> Iterator[Engine]:
    from alembic import command
    from alembic.config import Config

    assert TEST_POSTGRES_URL is not None
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", normalize_database_url(TEST_POSTGRES_URL))
    command.upgrade(config, "head")

    eng = make_engine(TEST_POSTGRES_URL)
    try:
        yield eng
    finally:
        eng.dispose()


@pytest.fixture
def _pg_session(_pg_engine: Engine) -> Iterator[Session]:
    session_factory = make_session_factory(engine=_pg_engine)
    with session_factory() as session:
        for model in (
            ProactiveActionEvent,
            WitnessOpportunityCandidate,
            TaskEvent,
            Task,
            Installation,
        ):
            session.execute(delete(model))
        session.commit()
        yield session
        session.rollback()
        for model in (
            ProactiveActionEvent,
            WitnessOpportunityCandidate,
            TaskEvent,
            Task,
            Installation,
        ):
            session.execute(delete(model))
        session.commit()


def _make_installation_pg(session: Session) -> Installation:
    installation = Installation(slack_team_id=f"T{uuid.uuid4().hex}")
    session.add(installation)
    session.flush()
    return installation


def _make_candidate_pg(
    session: Session,
    installation: Installation,
) -> WitnessOpportunityCandidate:
    now = datetime.now(UTC)
    candidate = WitnessOpportunityCandidate(
        installation_id=installation.id,
        channel_id="CTEST123",
        visibility_scope_type="channel",
        visibility_scope_id="CTEST123",
        candidate_type="recurring_check",
        title="Test candidate",
        summary="A test candidate for ledger cutover tests.",
        evidence_json=[],
        source_type="channel_profile",
        source_id="profile-1",
        dedupe_key=f"test:{uuid.uuid4()}",
        confidence_score=Decimal("0.750"),
        status="candidate",
        metadata_json={},
        feedback_json={},
        created_at=now,
        updated_at=now,
    )
    session.add(candidate)
    session.flush()
    return candidate


def _events_for_pg(
    session: Session, candidate: WitnessOpportunityCandidate
) -> list[ProactiveActionEvent]:
    return list(
        session.scalars(
            select(ProactiveActionEvent)
            .where(ProactiveActionEvent.candidate_id == candidate.id)
            .order_by(ProactiveActionEvent.created_at)
        )
    )


@_pg_skipif
def test_inline_path_records_autopilot_decision_event_act(
    _pg_session: Session,
) -> None:
    """Inline gate: execute-eligible decision records event with policy_decision=act."""
    installation = _make_installation_pg(_pg_session)
    candidate = _make_candidate_pg(_pg_session, installation)

    d = _decision(
        decision="execute_task",
        risk="low",
        action_kind="read_only_analysis",
        delivery_target="channel",
        confidence="0.750",
    )
    _inline_record_autopilot_decision_event(
        _pg_session,
        candidate,
        decision=d,
        actor_id="witness_autopilot",
        now=NOW,
    )
    _pg_session.commit()

    events = _events_for_pg(_pg_session, candidate)
    decision_events = [e for e in events if e.event_type == "autopilot_decision"]
    assert len(decision_events) == 1
    ev = decision_events[0]
    assert ev.policy_decision == "act"
    assert ev.reason_code == "execute_task"
    assert ev.actor_id == "witness_autopilot"


@_pg_skipif
def test_inline_path_records_autopilot_decision_event_defer(
    _pg_session: Session,
) -> None:
    """Inline gate: high-risk decision records event with policy_decision=defer."""
    installation = _make_installation_pg(_pg_session)
    candidate = _make_candidate_pg(_pg_session, installation)

    d = _decision(decision="execute_task", risk="high")
    _inline_record_autopilot_decision_event(
        _pg_session,
        candidate,
        decision=d,
        actor_id="witness_autopilot",
        now=NOW,
    )
    _pg_session.commit()

    events = _events_for_pg(_pg_session, candidate)
    decision_events = [e for e in events if e.event_type == "autopilot_decision"]
    assert len(decision_events) == 1
    ev = decision_events[0]
    assert ev.policy_decision == "defer"


@_pg_skipif
def test_inline_path_records_autopilot_decision_event_silent(
    _pg_session: Session,
) -> None:
    """Inline gate: dismiss decision records event with policy_decision=silent."""
    installation = _make_installation_pg(_pg_session)
    candidate = _make_candidate_pg(_pg_session, installation)

    d = _decision(decision="dismiss")
    _inline_record_autopilot_decision_event(
        _pg_session,
        candidate,
        decision=d,
        actor_id="witness_autopilot",
        now=NOW,
    )
    _pg_session.commit()

    events = _events_for_pg(_pg_session, candidate)
    decision_events = [e for e in events if e.event_type == "autopilot_decision"]
    assert len(decision_events) == 1
    ev = decision_events[0]
    assert ev.policy_decision == "silent"
    assert ev.reason_code == "dismissed"


# ---------------------------------------------------------------------------
# Cutover flag: on==off parity (pure)
# ---------------------------------------------------------------------------


class TestCutoverOnOffParity:
    """For a matrix of inputs, flag-on and flag-off produce the same final outcome.

    We verify by comparing the LedgerDecision produced by the policy (which
    is what the cutover path uses) with the expected inline outcome (what the
    inline gate would produce). The parity tests above in TestPolicyParityMatrix
    already prove this rigorously, but this class explicitly frames it as a
    cutover claim.
    """

    MATRIX = [
        # (description, decision-kwargs, expected-policy-outcome)
        (
            "execute_eligible_read_only",
            dict(
                decision="execute_task",
                risk="low",
                action_kind="read_only_analysis",
                delivery_target="channel",
                requires_user_reply=False,
                allowed_without_confirmation=True,
                confidence="0.750",
                task_input="Check status.",
            ),
            "act",
        ),
        (
            "execute_eligible_status_check",
            dict(action_kind="status_check"),
            "act",
        ),
        (
            "execute_eligible_dm",
            dict(delivery_target="dm"),
            "act",
        ),
        (
            "high_risk",
            dict(risk="high"),
            "defer",
        ),
        (
            "schedule_management",
            dict(action_kind="schedule_management"),
            "defer",
        ),
        (
            "low_confidence",
            dict(confidence="0.500"),
            "defer",
        ),
        (
            "requires_reply",
            dict(requires_user_reply=True),
            "defer",
        ),
        (
            "no_confirmation",
            dict(allowed_without_confirmation=False),
            "defer",
        ),
        (
            "no_task_input",
            dict(task_input=None),
            "defer",
        ),
        (
            "dismiss",
            dict(decision="dismiss"),
            "silent",
        ),
        (
            "draft_artifact",
            dict(decision="execute_task", action_kind="draft_artifact"),
            "ask",
        ),
        (
            "defer_decision",
            dict(decision="defer"),
            "defer",
        ),
        (
            "monitor_only",
            dict(decision="monitor_only"),
            "defer",
        ),
        (
            "ask_user",
            dict(decision="ask_user"),
            "defer",
        ),
        (
            "unknown_delivery_target",
            dict(delivery_target="unknown"),
            "defer",
        ),
    ]

    def _inline_outcome(self, d: WitnessAutopilotDecision) -> str:
        """Compute what the inline gate produces (normalized to LedgerOutcome)."""
        if d.decision == "execute_task" and d.action_kind == "draft_artifact":
            return "ask"
        if d.should_execute:
            return "act"
        if d.decision == "dismiss":
            return "silent"
        return "defer"

    def _cutover_outcome(self, d: WitnessAutopilotDecision) -> str:
        """Compute what the policy (cutover) produces."""
        return _policy_outcome(autopilot_decision=d).decision

    @pytest.mark.parametrize(
        "desc,kwargs,expected",
        [(row[0], row[1], row[2]) for row in MATRIX],
        ids=[row[0] for row in MATRIX],
    )
    def test_inline_outcome_matches_expected(
        self, desc: str, kwargs: dict, expected: str
    ) -> None:
        d = _decision(**kwargs)
        assert self._inline_outcome(d) == expected, (
            f"{desc}: inline_outcome={self._inline_outcome(d)} expected={expected}"
        )

    @pytest.mark.parametrize(
        "desc,kwargs,expected",
        [(row[0], row[1], row[2]) for row in MATRIX],
        ids=[row[0] for row in MATRIX],
    )
    def test_policy_outcome_matches_expected(
        self, desc: str, kwargs: dict, expected: str
    ) -> None:
        d = _decision(**kwargs)
        assert self._cutover_outcome(d) == expected, (
            f"{desc}: policy_outcome={self._cutover_outcome(d)} expected={expected}"
        )

    @pytest.mark.parametrize(
        "desc,kwargs,expected",
        [(row[0], row[1], row[2]) for row in MATRIX],
        ids=[row[0] for row in MATRIX],
    )
    def test_on_equals_off(self, desc: str, kwargs: dict, expected: str) -> None:
        """The critical parity assertion: flag-on == flag-off for every input."""
        d = _decision(**kwargs)
        inline = self._inline_outcome(d)
        policy = self._cutover_outcome(d)
        assert inline == policy, (
            f"{desc}: inline={inline} policy={policy} — ON and OFF produce different outcomes!"
        )
