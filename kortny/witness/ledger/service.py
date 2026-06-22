"""Proactive Action Ledger service -- Step 1 (shadow-only).

shadow_evaluate() is called by the runner and autopilot AFTER their real
decision to verify parity. Exceptions are always swallowed so a bug here
can never affect real delivery or autopilot execution.

Step 2+ will add the full lifecycle API and the cutover switch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from kortny.witness.ledger.policy import (
    CandidateInputs,
    DeliveryContext,
    LedgerDecision,
    LedgerOutcome,
    LedgerSurface,
    ProactiveActionPolicy,
)

logger = logging.getLogger(__name__)

LEDGER_DIVERGENCE_MESSAGE = "proactive_ledger_shadow_divergence"


@dataclass(frozen=True, slots=True)
class ShadowEvaluationResult:
    """Outcome of one shadow evaluation."""

    surface: LedgerSurface
    ledger_decision: LedgerDecision
    real_decision: str
    diverged: bool


class ProactiveActionService:
    """Shadow-only Proactive Action Ledger (Step 1).

    Holds a ProactiveActionPolicy and exposes shadow_evaluate() which the
    runner and autopilot call after their real decision. Any exception
    inside this method is caught and logged so a bug here can never affect
    real delivery or autopilot execution.
    """

    def __init__(self) -> None:
        self._policy = ProactiveActionPolicy()

    def shadow_evaluate(
        self,
        surface: LedgerSurface,
        candidate: CandidateInputs,
        ctx: DeliveryContext,
        *,
        real_decision: str,
        candidate_id: object = None,
    ) -> ShadowEvaluationResult | None:
        """Compute ledger decision; log if it diverges from the real decision.

        Always returns None on any internal exception.
        """
        try:
            ledger = self._policy.decide(surface, candidate, ctx)
            normalised_real = _normalise_real(real_decision, surface)
            diverged = ledger.decision != normalised_real
            if diverged:
                logger.warning(
                    "%s surface=%s candidate_id=%s real=%s normalised_real=%s "
                    "ledger=%s reason=%s",
                    LEDGER_DIVERGENCE_MESSAGE,
                    surface,
                    candidate_id,
                    real_decision,
                    normalised_real,
                    ledger.decision,
                    ledger.reason_code,
                )
            return ShadowEvaluationResult(
                surface=surface,
                ledger_decision=ledger,
                real_decision=real_decision,
                diverged=diverged,
            )
        except Exception:
            logger.exception(
                "proactive_ledger_shadow_evaluate failed surface=%s candidate_id=%s",
                surface,
                candidate_id,
            )
            return None


def _normalise_real(real_decision: str, surface: LedgerSurface) -> LedgerOutcome:
    """Map the existing gate outcome strings to LedgerOutcome literals."""
    _ACT = frozenset({"sent", "execute_task", "channel_sent"})
    _SILENT = frozenset({"silent", "dismiss", "dismissed", "below_threshold"})
    _ASK = frozenset({"draft_artifact", "ask_user"})
    if real_decision in _ACT:
        return "act"
    if real_decision in _SILENT:
        return "silent"
    if real_decision in _ASK:
        return "ask"
    return "defer"
