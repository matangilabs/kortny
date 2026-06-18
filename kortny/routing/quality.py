"""Per-task routing-quality signal (HIG-221 learning loop).

Every routing decision is traced, but nothing scored whether the chosen route
actually worked. This computes a deterministic outcome label for each terminal
task from its status + TaskEvent stream — the feedback signal a later trace→eval
pipeline, semantic-router promotion gate, and episodic priors all consume.

Pure and deterministic (no DB/LLM): given the terminal status and the task's
event payloads, classify the outcome. The worker stores it on the Task after
transition so it is queryable for aggregation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum


class RoutingQuality(StrEnum):
    """Outcome label for a terminal task."""

    clean = "clean"  # succeeded first-pass, no recovery or truncation
    recovered = "recovered"  # succeeded but needed a retry / tool recovery
    partial = "partial"  # succeeded but ran out of budget (partial answer)
    failed = "failed"  # failed or crashed
    cancelled = "cancelled"  # user/worker cancelled — excluded from scoring

    @property
    def score(self) -> float | None:
        """A 0..1 quality score; ``None`` for cancelled (not a routing outcome)."""

        return {
            RoutingQuality.clean: 1.0,
            RoutingQuality.recovered: 0.75,
            RoutingQuality.partial: 0.5,
            RoutingQuality.failed: 0.0,
            RoutingQuality.cancelled: None,
        }[self]


@dataclass(frozen=True, slots=True)
class RoutingQualityResult:
    quality: RoutingQuality
    score: float | None
    reason_codes: tuple[str, ...]


# TaskEvent payload "message" values that signal degraded-but-succeeded routing.
_RECOVERY_MESSAGES = frozenset(
    {
        "tool_call_recoverable_failed",
        "execution_recovery_planner_failed",
        "agent_empty_completion_retry",
        "agent_empty_response_retry",
    }
)
_PARTIAL_MESSAGES = frozenset({"execution_budget_exceeded"})
_COMPLETION_MESSAGES = frozenset({"agent_completed", "adk_runtime_completed"})


def compute_routing_quality(
    *,
    status: str,
    event_payloads: Sequence[Mapping[str, object]],
    attempts: int = 1,
) -> RoutingQualityResult:
    """Classify a terminal task's routing outcome from its status + events.

    - failed/crashed -> failed
    - cancelled -> cancelled (excluded from scoring)
    - succeeded + a partial marker (budget exhaustion) -> partial
    - succeeded + a retry/recovery signal (or >1 attempt) -> recovered
    - succeeded clean -> clean
    """

    if status in ("failed", "crashed"):
        return RoutingQualityResult(
            RoutingQuality.failed, RoutingQuality.failed.score, ("terminal_failure",)
        )
    if status == "cancelled":
        return RoutingQualityResult(
            RoutingQuality.cancelled, RoutingQuality.cancelled.score, ("cancelled",)
        )
    if status != "succeeded":
        # Non-terminal (pending/running/waiting_approval) — caller should only
        # score terminal tasks, but be defensive rather than mislabel.
        return RoutingQualityResult(RoutingQuality.cancelled, None, ("not_terminal",))

    reason_codes: list[str] = []
    partial = False
    recovered = attempts > 1
    if recovered:
        reason_codes.append("retried")
    for payload in event_payloads:
        message = payload.get("message")
        if not isinstance(message, str):
            continue
        if (
            message in _PARTIAL_MESSAGES
            or message in _COMPLETION_MESSAGES
            and payload.get("partial") is True
        ):
            partial = True
        elif message in _RECOVERY_MESSAGES:
            recovered = True
            if message not in reason_codes:
                reason_codes.append(message)

    if partial:
        reason_codes.insert(0, "budget_exhausted")
        return RoutingQualityResult(
            RoutingQuality.partial, RoutingQuality.partial.score, tuple(reason_codes)
        )
    if recovered:
        return RoutingQualityResult(
            RoutingQuality.recovered,
            RoutingQuality.recovered.score,
            tuple(reason_codes),
        )
    return RoutingQualityResult(RoutingQuality.clean, RoutingQuality.clean.score, ())
