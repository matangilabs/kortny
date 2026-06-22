"""Witness autopilot for default-on proactive help."""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Any

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

if TYPE_CHECKING:
    from kortny.witness.ledger.service import ProactiveActionService

from kortny.config import Settings, load_settings
from kortny.db.models import LLMProvider as DbLLMProvider
from kortny.db.models import (
    ObservePolicy,
    SlackChannelMembership,
    Task,
    TaskEventType,
    TaskStatus,
    WitnessDeliveryLog,
    WitnessOpportunityCandidate,
)
from kortny.llm import (
    ChatMessage,
    LLMProvider,
    LLMService,
    ModelRoute,
    ModelRouter,
    ModelRouteTier,
)
from kortny.llm.runtime_config import (
    create_provider_for_selection,
    select_runtime_model,
)
from kortny.observability import start_span
from kortny.tasks import TaskService
from kortny.tasks.identity import TaskIdentity
from kortny.tools.types import JsonObject
from kortny.witness.opportunities import (
    candidate_delivery_decision,
    candidate_thread_ts,
)

logger = logging.getLogger(__name__)


def _get_ledger() -> ProactiveActionService:
    # Deferred import to break the ledger/policy → autopilot → ledger/service cycle.
    from kortny.witness.ledger.service import ProactiveActionService  # noqa: PLC0415

    return ProactiveActionService()


WITNESS_AUTOPILOT_REVIEW_PROMPT_NAME = "kortny.witness_autopilot_reviewer"
WITNESS_AUTOPILOT_REVIEW_RESPONSE_FORMAT: JsonObject = {"type": "json_object"}
WITNESS_AUTOPILOT_TASK_CREATED_MESSAGE = "witness_autopilot_task_created"
WITNESS_AUTOPILOT_CANDIDATE_DEFERRED_MESSAGE = "witness_autopilot_candidate_deferred"
WITNESS_AUTOPILOT_CANDIDATE_DISMISSED_MESSAGE = "witness_autopilot_candidate_dismissed"
WITNESS_AUTOPILOT_DRAFT_POSTED_MESSAGE = "witness_autopilot_draft_posted"
WITNESS_DRAFT_TASK_SOURCE = "witness_draft"

DEFAULT_WITNESS_AUTOPILOT_LIMIT = 1
DEFAULT_WITNESS_AUTOPILOT_MIN_CONFIDENCE = Decimal("0.600")
DEFAULT_WITNESS_AUTOPILOT_COOLDOWN = timedelta(hours=24)
AUTOPILOT_REEXECUTE_COOLDOWN = timedelta(days=7)
MAX_AUTOPILOT_TASK_INPUT_CHARS = 1800

# Draft tier (HIG-230): sliding 24h budget per channel; drafts count against
# witness_delivery_log via decision='draft_executed'.
DEFAULT_WITNESS_DRAFTS_PER_CHANNEL_PER_DAY = 1
WITNESS_DRAFT_WINDOW = timedelta(days=1)
WITNESS_DRAFT_LOG_USER_PREFIX = "channel:"

_VALID_DECISIONS = frozenset(
    ("execute_task", "defer", "dismiss", "monitor_only", "ask_user")
)
_VALID_RISKS = frozenset(("low", "medium", "high"))
_VALID_ACTION_KINDS = frozenset(
    (
        "read_only_analysis",
        "status_check",
        "schedule_management",
        "memory_write",
        "external_write",
        "approval_request",
        "reminder",
        "monitoring_setup",
        "draft_artifact",
        "other",
    )
)
_VALID_DELIVERY_TARGETS = frozenset(("channel", "dm", "none", "unknown"))
_AUTOPILOT_EXECUTABLE_ACTION_KINDS = frozenset(("read_only_analysis", "status_check"))
_AUTOPILOT_EXECUTABLE_DELIVERY_TARGETS = frozenset(("channel", "dm"))


@dataclass(frozen=True, slots=True)
class WitnessAutopilotDecision:
    """Structured LLM review for one Witness candidate."""

    decision: str
    risk: str
    action_kind: str
    delivery_target: str
    requires_user_reply: bool
    allowed_without_confirmation: bool
    reason: str
    task_input: str | None
    confidence_score: Decimal

    @property
    def should_execute(self) -> bool:
        return (
            self.decision == "execute_task"
            and self.risk == "low"
            and self.action_kind in _AUTOPILOT_EXECUTABLE_ACTION_KINDS
            and self.delivery_target in _AUTOPILOT_EXECUTABLE_DELIVERY_TARGETS
            and not self.requires_user_reply
            and self.allowed_without_confirmation
            and self.confidence_score >= DEFAULT_WITNESS_AUTOPILOT_MIN_CONFIDENCE
            and bool(self.task_input)
        )


@dataclass(frozen=True, slots=True)
class WitnessAutopilotOutcome:
    """Outcome from reviewing one candidate."""

    candidate_id: uuid.UUID
    status: str
    decision: str | None = None
    risk: str | None = None
    reason: str | None = None
    task_id: uuid.UUID | None = None


@dataclass(frozen=True, slots=True)
class WitnessAutopilotRunResult:
    """Outcome from one autopilot run."""

    outcomes: tuple[WitnessAutopilotOutcome, ...]

    @property
    def reviewed_count(self) -> int:
        return len(self.outcomes)

    @property
    def executed_count(self) -> int:
        return sum(1 for outcome in self.outcomes if outcome.status == "executed")

    @property
    def deferred_count(self) -> int:
        return sum(1 for outcome in self.outcomes if outcome.status == "deferred")

    @property
    def dismissed_count(self) -> int:
        return sum(1 for outcome in self.outcomes if outcome.status == "dismissed")

    @property
    def drafted_count(self) -> int:
        return sum(1 for outcome in self.outcomes if outcome.status == "draft_executed")


class WitnessAutopilot:
    """Review due Witness candidates and turn useful ones into normal tasks."""

    def __init__(
        self,
        session: Session,
        *,
        settings: Settings | None = None,
        llm_provider: LLMProvider | None = None,
        provider_name: DbLLMProvider | str | None = None,
        actor_id: str = "witness_autopilot",
        cooldown: timedelta = DEFAULT_WITNESS_AUTOPILOT_COOLDOWN,
        drafts_per_channel_per_day: int | None = None,
    ) -> None:
        self.session = session
        self.settings = settings
        self.llm_provider = llm_provider
        self.provider_name = provider_name
        self.actor_id = actor_id
        self.cooldown = cooldown
        self.drafts_per_channel_per_day = drafts_per_channel_per_day

    def run_once(
        self,
        *,
        installation_id: uuid.UUID | None = None,
        now: datetime | None = None,
        limit: int = DEFAULT_WITNESS_AUTOPILOT_LIMIT,
        min_confidence: Decimal = DEFAULT_WITNESS_AUTOPILOT_MIN_CONFIDENCE,
    ) -> WitnessAutopilotRunResult:
        """Review due candidates and create low-risk proactive tasks."""

        if limit < 0:
            raise ValueError("limit must be non-negative")
        if limit == 0:
            return WitnessAutopilotRunResult(outcomes=())

        run_at = _coerce_utc(now)
        candidates = self._eligible_candidates(
            installation_id=installation_id,
            now=run_at,
            limit=limit,
            min_confidence=min_confidence,
        )
        outcomes: list[WitnessAutopilotOutcome] = []
        with start_span(
            "witness.autopilot",
            attributes={
                "openinference.span.kind": "CHAIN",
                "witness.autopilot.limit": limit,
                "witness.autopilot.min_confidence": str(min_confidence),
            },
        ):
            for candidate in candidates:
                try:
                    outcome = self._review_candidate(candidate, now=run_at)
                    outcomes.append(outcome)
                    # ponytail: live autopilot shadow deferred to ledger Step 2 — the real WitnessAutopilotDecision (action_kind/risk) isn't available here; a faithful shadow must hook inside _review_candidate.
                except Exception as exc:
                    logger.exception(
                        "witness autopilot candidate review failed candidate_id=%s",
                        candidate.id,
                    )
                    _record_feedback(
                        candidate,
                        action="autopilot_failed",
                        by_user_id=self.actor_id,
                        now=run_at,
                        details={
                            "error_type": type(exc).__name__,
                            "error": _bounded(str(exc), 280),
                        },
                    )
                    outcomes.append(
                        WitnessAutopilotOutcome(
                            candidate_id=candidate.id,
                            status="failed",
                            reason=str(exc),
                        )
                    )
            self.session.flush()
        return WitnessAutopilotRunResult(outcomes=tuple(outcomes))

    def _eligible_candidates(
        self,
        *,
        installation_id: uuid.UUID | None,
        now: datetime,
        limit: int,
        min_confidence: Decimal,
    ) -> tuple[WitnessOpportunityCandidate, ...]:
        filters = [
            WitnessOpportunityCandidate.status == "candidate",
            WitnessOpportunityCandidate.confidence_score >= min_confidence,
            or_(
                WitnessOpportunityCandidate.source_task_id.is_(None),
                WitnessOpportunityCandidate.source_task_id.not_in(
                    select(Task.id).where(
                        or_(
                            Task.identity_key.like("synthetic:witness_autopilot:%"),
                            Task.identity_key.like("synthetic:witness_draft:%"),
                        )
                    )
                ),
            ),
            or_(
                WitnessOpportunityCandidate.cooldown_until.is_(None),
                WitnessOpportunityCandidate.cooldown_until <= now,
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
                    WitnessOpportunityCandidate.updated_at.asc(),
                    WitnessOpportunityCandidate.created_at.asc(),
                )
                .with_for_update(skip_locked=True)
                .limit(limit)
            )
        )

    def _review_candidate(
        self,
        candidate: WitnessOpportunityCandidate,
        *,
        now: datetime,
    ) -> WitnessAutopilotOutcome:
        source_task = _source_task(self.session, candidate)
        if source_task is None:
            return self._defer_without_review(
                candidate,
                now=now,
                reason="Candidate has no source task for auditable LLM review.",
            )

        preflight_reason = _autopilot_preflight_defer_reason(
            self.session,
            candidate,
            source_task=source_task,
            witness_deliver_private=(
                self.settings.witness_deliver_private
                if self.settings is not None
                else False
            ),
        )
        if preflight_reason is not None:
            return self._defer_without_review(
                candidate,
                now=now,
                reason=preflight_reason,
            )

        decision = self._review_with_llm(candidate, source_task=source_task)
        if (
            decision.decision == "execute_task"
            and decision.action_kind == "draft_artifact"
        ):
            # HIG-230 draft tier: drafts are visibly drafts, never sent or
            # published anywhere external, and count against draft budgets.
            return self._review_draft_candidate(
                candidate,
                decision=decision,
                source_task=source_task,
                now=now,
            )
        decision_safety_reason = _decision_safety_defer_reason(decision)
        if decision_safety_reason is not None:
            return self._defer_reviewed_decision(
                candidate,
                decision=decision,
                now=now,
                reason=decision_safety_reason,
            )

        if decision.should_execute:
            task = self._create_proactive_task(
                candidate,
                decision=decision,
                source_task=source_task,
                now=now,
            )
            prev_status = candidate.status
            candidate.automated_task_id = task.id
            candidate.status = "accepted"
            candidate.cooldown_until = None
            candidate.last_suggested_at = now
            candidate.updated_at = now
            _record_feedback(
                candidate,
                action="autopilot_executed",
                by_user_id=self.actor_id,
                now=now,
                details={
                    "decision": decision.decision,
                    "risk": decision.risk,
                    "action_kind": decision.action_kind,
                    "delivery_target": decision.delivery_target,
                    "requires_user_reply": decision.requires_user_reply,
                    "allowed_without_confirmation": (
                        decision.allowed_without_confirmation
                    ),
                    "reason": decision.reason,
                    "generated_task_id": str(task.id),
                    "execution_policy": "default_on_low_risk_task",
                },
            )
            _get_ledger().record_transition(
                self.session,
                candidate,
                to_state="accepted",
                event_type="autopilot_executed",
                from_state=prev_status,
                actor_id=self.actor_id,
                task_id=task.id,
                now=now,
            )
            TaskService(self.session).append_event(
                task,
                TaskEventType.log,
                {
                    "message": WITNESS_AUTOPILOT_TASK_CREATED_MESSAGE,
                    "candidate_id": str(candidate.id),
                    "source_task_id": str(source_task.id),
                    "decision": decision.decision,
                    "risk": decision.risk,
                    "action_kind": decision.action_kind,
                    "delivery_target": decision.delivery_target,
                    "reason": decision.reason,
                },
            )
            self.session.flush()
            return WitnessAutopilotOutcome(
                candidate_id=candidate.id,
                status="executed",
                decision=decision.decision,
                risk=decision.risk,
                reason=decision.reason,
                task_id=task.id,
            )

        if decision.decision == "dismiss":
            prev_status = candidate.status
            candidate.status = "dismissed"
            candidate.cooldown_until = None
            candidate.updated_at = now
            _record_feedback(
                candidate,
                action="autopilot_dismissed",
                by_user_id=self.actor_id,
                now=now,
                details={
                    "decision": decision.decision,
                    "risk": decision.risk,
                    "action_kind": decision.action_kind,
                    "delivery_target": decision.delivery_target,
                    "reason": decision.reason,
                },
            )
            _get_ledger().record_transition(
                self.session,
                candidate,
                to_state="dismissed",
                event_type="autopilot_dismissed",
                from_state=prev_status,
                actor_id=self.actor_id,
                now=now,
            )
            _append_candidate_event(
                self.session,
                candidate,
                message=WITNESS_AUTOPILOT_CANDIDATE_DISMISSED_MESSAGE,
                payload={
                    "decision": decision.decision,
                    "risk": decision.risk,
                    "reason": decision.reason,
                },
            )
            self.session.flush()
            return WitnessAutopilotOutcome(
                candidate_id=candidate.id,
                status="dismissed",
                decision=decision.decision,
                risk=decision.risk,
                reason=decision.reason,
            )

        prev_status = candidate.status
        candidate.status = "cooldown"
        candidate.cooldown_until = now + self.cooldown
        candidate.updated_at = now
        _record_feedback(
            candidate,
            action="autopilot_deferred",
            by_user_id=self.actor_id,
            now=now,
            details={
                "decision": decision.decision,
                "risk": decision.risk,
                "action_kind": decision.action_kind,
                "delivery_target": decision.delivery_target,
                "reason": decision.reason,
                "cooldown_until": candidate.cooldown_until.isoformat(),
            },
        )
        _get_ledger().record_transition(
            self.session,
            candidate,
            to_state="cooldown",
            event_type="autopilot_deferred",
            from_state=prev_status,
            reason_code="decision_deferred",
            actor_id=self.actor_id,
            now=now,
        )
        _append_candidate_event(
            self.session,
            candidate,
            message=WITNESS_AUTOPILOT_CANDIDATE_DEFERRED_MESSAGE,
            payload={
                "decision": decision.decision,
                "risk": decision.risk,
                "action_kind": decision.action_kind,
                "delivery_target": decision.delivery_target,
                "reason": decision.reason,
                "cooldown_until": candidate.cooldown_until.isoformat(),
            },
        )
        self.session.flush()
        return WitnessAutopilotOutcome(
            candidate_id=candidate.id,
            status="deferred",
            decision=decision.decision,
            risk=decision.risk,
            reason=decision.reason,
        )

    def _defer_reviewed_decision(
        self,
        candidate: WitnessOpportunityCandidate,
        *,
        decision: WitnessAutopilotDecision,
        now: datetime,
        reason: str,
    ) -> WitnessAutopilotOutcome:
        prev_status = candidate.status
        candidate.status = "cooldown"
        candidate.cooldown_until = now + self.cooldown
        candidate.updated_at = now
        _record_feedback(
            candidate,
            action="autopilot_deferred",
            by_user_id=self.actor_id,
            now=now,
            details={
                "decision": decision.decision,
                "risk": decision.risk,
                "action_kind": decision.action_kind,
                "delivery_target": decision.delivery_target,
                "requires_user_reply": decision.requires_user_reply,
                "allowed_without_confirmation": decision.allowed_without_confirmation,
                "reason": reason,
                "review_reason": decision.reason,
                "cooldown_until": candidate.cooldown_until.isoformat(),
            },
        )
        _get_ledger().record_transition(
            self.session,
            candidate,
            to_state="cooldown",
            event_type="autopilot_deferred",
            from_state=prev_status,
            reason_code="reviewed_decision_deferred",
            actor_id=self.actor_id,
            now=now,
        )
        _append_candidate_event(
            self.session,
            candidate,
            message=WITNESS_AUTOPILOT_CANDIDATE_DEFERRED_MESSAGE,
            payload={
                "decision": decision.decision,
                "risk": decision.risk,
                "action_kind": decision.action_kind,
                "delivery_target": decision.delivery_target,
                "reason": reason,
                "review_reason": decision.reason,
                "cooldown_until": candidate.cooldown_until.isoformat(),
            },
        )
        self.session.flush()
        return WitnessAutopilotOutcome(
            candidate_id=candidate.id,
            status="deferred",
            decision=decision.decision,
            risk=decision.risk,
            reason=reason,
        )

    def _review_draft_candidate(
        self,
        candidate: WitnessOpportunityCandidate,
        *,
        decision: WitnessAutopilotDecision,
        source_task: Task,
        now: datetime,
    ) -> WitnessAutopilotOutcome:
        """Auto-execute a draft_artifact candidate when every gate passes."""

        channel_id = _candidate_channel_id(candidate, source_task=source_task)
        defer_reason = self._draft_defer_reason(
            candidate,
            decision=decision,
            channel_id=channel_id,
            now=now,
        )
        if defer_reason is not None:
            return self._defer_reviewed_decision(
                candidate,
                decision=decision,
                now=now,
                reason=defer_reason,
            )
        assert channel_id is not None  # _draft_defer_reason checked it

        task = self._create_draft_task(
            candidate,
            decision=decision,
            source_task=source_task,
            channel_id=channel_id,
            now=now,
        )
        # A draft does not consume the candidate: it stays accept-able and
        # acceptance later still flows through materialize_acceptance. The
        # cooldown only stops the autopilot re-reviewing it every tick.
        candidate.cooldown_until = now + self.cooldown
        candidate.last_decision = "draft"
        candidate.last_suggested_at = now
        candidate.updated_at = now
        _record_feedback(
            candidate,
            action="draft_posted",
            by_user_id=self.actor_id,
            now=now,
            details={
                "decision": decision.decision,
                "risk": decision.risk,
                "action_kind": decision.action_kind,
                "delivery_target": decision.delivery_target,
                "reason": decision.reason,
                "generated_task_id": str(task.id),
                "channel_id": channel_id,
                "execution_policy": "draft_tier",
            },
        )
        self.session.add(
            WitnessDeliveryLog(
                installation_id=candidate.installation_id,
                slack_user_id=f"{WITNESS_DRAFT_LOG_USER_PREFIX}{channel_id}",
                candidate_id=candidate.id,
                decision="draft_executed",
                reason="sent",
                created_at=now,
            )
        )
        TaskService(self.session).append_event(
            task,
            TaskEventType.log,
            {
                "message": WITNESS_AUTOPILOT_DRAFT_POSTED_MESSAGE,
                "candidate_id": str(candidate.id),
                "source_task_id": str(source_task.id),
                "decision": decision.decision,
                "risk": decision.risk,
                "action_kind": decision.action_kind,
                "delivery_target": decision.delivery_target,
                "reason": decision.reason,
            },
        )
        self.session.flush()
        return WitnessAutopilotOutcome(
            candidate_id=candidate.id,
            status="draft_executed",
            decision=decision.decision,
            risk=decision.risk,
            reason=decision.reason,
            task_id=task.id,
        )

    def _draft_defer_reason(
        self,
        candidate: WitnessOpportunityCandidate,
        *,
        decision: WitnessAutopilotDecision,
        channel_id: str | None,
        now: datetime,
    ) -> str | None:
        if channel_id is None:
            return "Draft tier needs a concrete Slack channel or DM target."
        if decision.risk != "low":
            return "Draft tier only executes low-risk candidates."
        if candidate_delivery_decision(candidate) != "draft":
            return (
                "Draft tier only runs when the delivery scorer decision is "
                "draft (one-shot candidates)."
            )
        if not decision.allowed_without_confirmation:
            return "Autopilot reviewer did not mark this safe without confirmation."
        if decision.confidence_score < DEFAULT_WITNESS_AUTOPILOT_MIN_CONFIDENCE:
            return "Autopilot reviewer confidence was below the execution threshold."
        if not decision.task_input:
            return "Autopilot reviewer did not provide a draft task to execute."
        if _candidate_has_posted_draft(candidate):
            return "Candidate already has a posted draft awaiting feedback."
        is_dm = channel_id.startswith("D") or candidate.visibility_scope_type == "dm"
        if not is_dm and not self._channel_policy_is_full(
            installation_id=candidate.installation_id,
            channel_id=channel_id,
        ):
            return (
                "Draft tier needs the channel policy at proactivity_status="
                "'full' (or DM scope)."
            )
        limit = self._draft_budget_limit()
        if limit < 1:
            return "Draft tier is disabled (draft budget is zero)."
        if (
            self._drafts_in_window(
                installation_id=candidate.installation_id,
                channel_id=channel_id,
                now=now,
            )
            >= limit
        ):
            return "Draft budget for this channel is used up for today."
        return None

    def _draft_budget_limit(self) -> int:
        if self.drafts_per_channel_per_day is not None:
            return self.drafts_per_channel_per_day
        if self.settings is not None:
            return self.settings.witness_drafts_per_channel_per_day
        return DEFAULT_WITNESS_DRAFTS_PER_CHANNEL_PER_DAY

    def _drafts_in_window(
        self,
        *,
        installation_id: uuid.UUID,
        channel_id: str,
        now: datetime,
    ) -> int:
        cutoff = now - WITNESS_DRAFT_WINDOW
        return int(
            self.session.scalar(
                select(func.count())
                .select_from(WitnessDeliveryLog)
                .where(
                    WitnessDeliveryLog.installation_id == installation_id,
                    WitnessDeliveryLog.slack_user_id
                    == f"{WITNESS_DRAFT_LOG_USER_PREFIX}{channel_id}",
                    WitnessDeliveryLog.decision == "draft_executed",
                    WitnessDeliveryLog.created_at > cutoff,
                )
            )
            or 0
        )

    def _channel_policy_is_full(
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

    def _create_draft_task(
        self,
        candidate: WitnessOpportunityCandidate,
        *,
        decision: WitnessAutopilotDecision,
        source_task: Task,
        channel_id: str,
        now: datetime,
    ) -> Task:
        task_input = _draft_task_input(candidate, decision=decision)
        thread_ts = candidate_thread_ts(candidate, source_task=source_task)
        user_id = (
            candidate.target_slack_user_id
            or source_task.slack_user_id
            or _membership_added_by(self.session, candidate)
            or self.actor_id
        )
        identity = TaskIdentity.synthetic(
            source=WITNESS_DRAFT_TASK_SOURCE,
            source_id=str(candidate.id),
            input_text=task_input,
            payload={
                "candidate_id": str(candidate.id),
                "candidate_type": candidate.candidate_type,
                "visibility_scope_type": candidate.visibility_scope_type,
                "visibility_scope_id": candidate.visibility_scope_id,
                "source_task_id": str(source_task.id),
                "autopilot_decision": decision.decision,
                "autopilot_risk": decision.risk,
                "autopilot_action_kind": decision.action_kind,
                "autopilot_delivery_target": decision.delivery_target,
                "response_contract": (
                    "Produce a visible draft only. Never send, publish, or "
                    "execute anything external. The reply must read as a "
                    "draft offered for feedback."
                ),
                "created_at": now.isoformat(),
            },
        )
        return TaskService(self.session).create_task(
            installation_id=candidate.installation_id,
            slack_event_id=None,
            slack_channel_id=channel_id,
            slack_thread_ts=thread_ts,
            slack_message_ts=None,
            slack_user_id=user_id,
            input=task_input,
            identity=identity,
            source_surface=WITNESS_DRAFT_TASK_SOURCE,
        )

    def _review_with_llm(
        self,
        candidate: WitnessOpportunityCandidate,
        *,
        source_task: Task,
    ) -> WitnessAutopilotDecision:
        completion = self._llm(source_task=source_task).complete(
            task_id=source_task.id,
            messages=_review_messages(candidate=candidate, source_task=source_task),
            response_format=WITNESS_AUTOPILOT_REVIEW_RESPONSE_FORMAT,
            prompt_name=WITNESS_AUTOPILOT_REVIEW_PROMPT_NAME,
        )
        return parse_witness_autopilot_decision(completion.content)

    def _llm(self, *, source_task: Task) -> LLMService:
        task_service = TaskService(self.session)
        if self.llm_provider is not None:
            route = ModelRoute(
                tier=ModelRouteTier.cheap_fast,
                model=self.llm_provider.model,
                reason="witness_autopilot_review",
            )
            return LLMService(
                session=self.session,
                provider=self.llm_provider,
                provider_name=self.provider_name or DbLLMProvider.openrouter,
                task_service=task_service,
                model_route=route,
            )

        route = ModelRouter(self._settings).route_for_tier(
            ModelRouteTier.cheap_fast,
            reason="witness_autopilot_review",
        )
        selection = select_runtime_model(
            session=self.session,
            settings=self._settings,
            installation_id=source_task.installation_id,
            model_route=route,
        )
        provider = create_provider_for_selection(
            settings=self._settings,
            selection=selection,
        )
        return LLMService(
            session=self.session,
            provider=provider,
            provider_name=selection.provider_name,
            task_service=task_service,
            model_route=selection.model_route,
        )

    def _create_proactive_task(
        self,
        candidate: WitnessOpportunityCandidate,
        *,
        decision: WitnessAutopilotDecision,
        source_task: Task,
        now: datetime,
    ) -> Task:
        channel_id = _candidate_channel_id(candidate, source_task=source_task)
        if channel_id is None:
            raise ValueError("Witness candidate has no channel for proactive task.")
        task_input = _task_input(candidate, decision=decision)
        user_id = (
            candidate.target_slack_user_id
            or source_task.slack_user_id
            or _membership_added_by(self.session, candidate)
            or self.actor_id
        )
        identity = TaskIdentity.synthetic(
            source="witness_autopilot",
            source_id=str(candidate.id),
            input_text=task_input,
            payload={
                "candidate_id": str(candidate.id),
                "candidate_type": candidate.candidate_type,
                "visibility_scope_type": candidate.visibility_scope_type,
                "visibility_scope_id": candidate.visibility_scope_id,
                "source_task_id": str(source_task.id),
                "autopilot_decision": decision.decision,
                "autopilot_risk": decision.risk,
                "autopilot_action_kind": decision.action_kind,
                "autopilot_delivery_target": decision.delivery_target,
                "response_contract": (
                    "Act as Kortny noticing a useful gap and delivering the "
                    "finished output. Do not expose internal planning, ask for "
                    "review, or say you are drafting/checking unless the check "
                    "result itself is the answer."
                ),
                "created_at": now.isoformat(),
                "runtime_cost_ceiling_usd": str(
                    self.settings.ambient_task_cost_ceiling_usd
                    if self.settings is not None
                    else 0.25
                ),
            },
        )
        return TaskService(self.session).create_task(
            installation_id=candidate.installation_id,
            slack_event_id=None,
            slack_channel_id=channel_id,
            slack_thread_ts=None,
            slack_message_ts=None,
            slack_user_id=user_id,
            input=task_input,
            identity=identity,
            source_surface="witness_autopilot",
        )

    def _defer_without_review(
        self,
        candidate: WitnessOpportunityCandidate,
        *,
        now: datetime,
        reason: str,
    ) -> WitnessAutopilotOutcome:
        prev_status = candidate.status
        candidate.status = "cooldown"
        candidate.cooldown_until = now + self.cooldown
        candidate.updated_at = now
        _record_feedback(
            candidate,
            action="autopilot_deferred",
            by_user_id=self.actor_id,
            now=now,
            details={
                "decision": "defer",
                "risk": "medium",
                "reason": reason,
                "cooldown_until": candidate.cooldown_until.isoformat(),
            },
        )
        _get_ledger().record_transition(
            self.session,
            candidate,
            to_state="cooldown",
            event_type="autopilot_deferred",
            from_state=prev_status,
            reason_code="no_review_deferred",
            actor_id=self.actor_id,
            now=now,
        )
        self.session.flush()
        return WitnessAutopilotOutcome(
            candidate_id=candidate.id,
            status="deferred",
            decision="defer",
            risk="medium",
            reason=reason,
        )

    @property
    def _settings(self) -> Settings:
        if self.settings is None:
            self.settings = load_settings()
        return self.settings


def parse_witness_autopilot_decision(
    content: str | None,
) -> WitnessAutopilotDecision:
    """Parse and validate the autopilot review payload."""

    if not content:
        return _fallback_decision("empty_model_output")
    try:
        payload = json.loads(_extract_json_object(content))
    except (json.JSONDecodeError, ValueError):
        return _fallback_decision("invalid_json")
    if not isinstance(payload, dict):
        return _fallback_decision("invalid_payload")

    decision = _choice(payload.get("decision"), valid=_VALID_DECISIONS, default="defer")
    risk = _choice(payload.get("risk"), valid=_VALID_RISKS, default="medium")
    action_kind = _choice(
        payload.get("action_kind"), valid=_VALID_ACTION_KINDS, default="other"
    )
    delivery_target = _choice(
        payload.get("delivery_target"),
        valid=_VALID_DELIVERY_TARGETS,
        default="unknown",
    )
    requires_user_reply = _bool(payload.get("requires_user_reply"), default=True)
    allowed_without_confirmation = _bool(
        payload.get("allowed_without_confirmation"), default=False
    )
    reason = _bounded(
        _optional_text(payload.get("reason")) or "No reason provided.", 500
    )
    task_input = _optional_text(payload.get("task_input"))
    confidence = _decimal(payload.get("confidence_score"), default=Decimal("0.500"))
    return WitnessAutopilotDecision(
        decision=decision,
        risk=risk,
        action_kind=action_kind,
        delivery_target=delivery_target,
        requires_user_reply=requires_user_reply,
        allowed_without_confirmation=allowed_without_confirmation,
        reason=reason,
        task_input=_bounded(task_input, MAX_AUTOPILOT_TASK_INPUT_CHARS)
        if task_input
        else None,
        confidence_score=max(Decimal("0.000"), min(confidence, Decimal("1.000"))),
    )


def _review_messages(
    *,
    candidate: WitnessOpportunityCandidate,
    source_task: Task,
) -> tuple[ChatMessage, ...]:
    return (
        ChatMessage(
            role="system",
            content=(
                "You are Kortny's Witness autopilot reviewer. Kortny is an AI "
                "coworker in Slack, and proactive help is ON by default unless "
                "users opt out. Use semantic judgment. Decide whether Kortny "
                "should act now on this opportunity candidate. Prefer "
                "execute_task only for low-risk, non-interruptive read-only "
                "analysis or status checks with clear evidence. One exception: "
                "when the most useful next step is preparing a concrete "
                "deliverable text (a summary, checklist, doc outline, or "
                "message body) that will only be shown as a visible draft and "
                "never sent or published, classify it as action_kind "
                "draft_artifact with decision execute_task. Do not execute "
                "anything that asks the user to confirm, approve, review, or "
                "reply; creates, edits, cancels, or audits a schedule; writes "
                "memory or policy; posts a reminder; retries an external tool "
                "that needs approval; performs external writes (including "
                "anything email-like); exposes private data; or is "
                "stale/speculative. For those, return defer, "
                "monitor_only, ask_user, or dismiss. Return JSON only with schema: "
                '{"decision":"execute_task|defer|dismiss|monitor_only|ask_user",'
                '"risk":"low|medium|high",'
                '"action_kind":"read_only_analysis|status_check|'
                "schedule_management|memory_write|external_write|approval_request|"
                'reminder|monitoring_setup|draft_artifact|other",'
                '"delivery_target":"channel|dm|none|unknown",'
                '"requires_user_reply":false,'
                '"allowed_without_confirmation":true,'
                '"reason":"brief reason",'
                '"task_input":"normal Slack-style task request for Kortny to run '
                'when decision is execute_task","confidence_score":0.0}. '
                "The task_input must be self-contained and humanlike. It should "
                "make Kortny act like a smart coworker who noticed a concrete "
                "gap and is delivering the finished useful output, not drafting "
                "for review or exposing internal planning. A good shape is: "
                '"I noticed [specific gap], so check/prepare [useful output] '
                'and respond with what you found." The task_input must not '
                "mention internal candidate IDs, autopilot, this review, chain "
                "of thought, or backend infrastructure."
            ),
        ),
        ChatMessage(
            role="user",
            content=json.dumps(
                {
                    "candidate": {
                        "id": str(candidate.id),
                        "type": candidate.candidate_type,
                        "title": candidate.title,
                        "summary": candidate.summary,
                        "suggested_action": candidate.suggested_action,
                        "suggested_message": candidate.suggested_message,
                        "confidence_score": str(candidate.confidence_score),
                        "confidence_reason": candidate.confidence_reason,
                        "scope_type": candidate.visibility_scope_type,
                        "scope_id": candidate.visibility_scope_id,
                        "channel_id": candidate.channel_id,
                        "target_slack_user_id": candidate.target_slack_user_id,
                        "evidence": candidate.evidence_json,
                        "metadata": candidate.metadata_json,
                    },
                    "source_task": {
                        "id": str(source_task.id),
                        "identity_kind": source_task.identity_kind,
                        "identity_payload": source_task.identity_payload,
                        "input": source_task.input,
                        "result_summary": source_task.result_summary,
                        "slack_channel_id": source_task.slack_channel_id,
                        "slack_user_id": source_task.slack_user_id,
                    },
                    "execution_policy": {
                        "default_on": True,
                        "allowed_now": (
                            "low-risk, non-interruptive read-only analysis or "
                            "status checks that can be completed without asking "
                            "the user for anything"
                        ),
                        "blocked_now": (
                            "schedule management, memory writes, approval prompts, "
                            "confirmation requests, reminders, external writes, "
                            "private-data exposure, or noisy speculative posts"
                        ),
                    },
                },
                default=str,
                separators=(",", ":"),
                sort_keys=True,
            ),
        ),
    )


def _source_task(
    session: Session,
    candidate: WitnessOpportunityCandidate,
) -> Task | None:
    if candidate.source_task_id is None:
        return None
    return session.get(Task, candidate.source_task_id)


def _candidate_channel_id(
    candidate: WitnessOpportunityCandidate,
    *,
    source_task: Task,
) -> str | None:
    if candidate.channel_id:
        return candidate.channel_id
    if candidate.visibility_scope_id and candidate.visibility_scope_type in {
        "channel",
        "private_channel",
        "dm",
    }:
        return candidate.visibility_scope_id
    return source_task.slack_channel_id


def _autopilot_preflight_defer_reason(
    session: Session,
    candidate: WitnessOpportunityCandidate,
    *,
    source_task: Task,
    witness_deliver_private: bool,
) -> str | None:
    channel_id = _candidate_channel_id(candidate, source_task=source_task)
    if channel_id is None:
        return "Candidate has no Slack delivery target."
    if _source_task_is_scheduled(source_task):
        return (
            "Candidate came from a scheduled task output; keep scheduled-task "
            "follow-ups monitor-only until the scheduler inspection tool is "
            "available."
        )
    if channel_id.startswith("D") and not witness_deliver_private:
        return (
            "Candidate would deliver a proactive DM, but private Witness delivery "
            "is disabled."
        )
    if not channel_id.startswith("D") and not _channel_membership_is_active(
        session,
        installation_id=candidate.installation_id,
        channel_id=channel_id,
    ):
        return "Candidate channel is not recorded as an active Kortny membership."
    if candidate.automated_task_id is not None:
        recently_executed_task = session.scalar(
            select(Task).where(
                Task.id == candidate.automated_task_id,
                Task.status == TaskStatus.succeeded,
                Task.created_at >= _coerce_utc(None) - AUTOPILOT_REEXECUTE_COOLDOWN,
            )
        )
        if recently_executed_task is not None:
            return "Equivalent opportunity already executed by autopilot in the last 7 days."
    return None


def _decision_safety_defer_reason(
    decision: WitnessAutopilotDecision,
) -> str | None:
    if decision.decision != "execute_task":
        return None
    if decision.risk != "low":
        return "Autopilot only executes low-risk candidates."
    if decision.action_kind not in _AUTOPILOT_EXECUTABLE_ACTION_KINDS:
        return (
            "Autopilot only executes read-only analysis or status-check actions; "
            f"review classified this as {decision.action_kind}."
        )
    if decision.delivery_target not in _AUTOPILOT_EXECUTABLE_DELIVERY_TARGETS:
        return "Autopilot needs a concrete Slack channel or DM delivery target."
    if decision.requires_user_reply:
        return "Autopilot will not execute actions that require a user reply."
    if not decision.allowed_without_confirmation:
        return "Autopilot reviewer did not mark this safe without confirmation."
    if decision.confidence_score < DEFAULT_WITNESS_AUTOPILOT_MIN_CONFIDENCE:
        return "Autopilot reviewer confidence was below the execution threshold."
    if not decision.task_input:
        return "Autopilot reviewer did not provide a task to execute."
    return None


def _source_task_is_scheduled(task: Task) -> bool:
    if task.identity_kind == "scheduled":
        return True
    payload = task.identity_payload
    if isinstance(payload, dict):
        return payload.get("source") == "scheduler" or bool(payload.get("schedule_id"))
    return False


def _channel_membership_is_active(
    session: Session,
    *,
    installation_id: uuid.UUID,
    channel_id: str,
) -> bool:
    membership = session.scalar(
        select(SlackChannelMembership)
        .where(
            SlackChannelMembership.installation_id == installation_id,
            SlackChannelMembership.channel_id == channel_id,
        )
        .limit(1)
    )
    return membership is not None and membership.membership_status == "active"


def _membership_added_by(
    session: Session,
    candidate: WitnessOpportunityCandidate,
) -> str | None:
    if not candidate.channel_id:
        return None
    membership = session.scalar(
        select(SlackChannelMembership)
        .where(
            SlackChannelMembership.installation_id == candidate.installation_id,
            SlackChannelMembership.channel_id == candidate.channel_id,
        )
        .limit(1)
    )
    return membership.added_by_user_id if membership is not None else None


def _candidate_has_posted_draft(candidate: WitnessOpportunityCandidate) -> bool:
    feedback = candidate.feedback_json or {}
    history = feedback.get("history")
    if not isinstance(history, list):
        return False
    return any(
        isinstance(entry, dict) and entry.get("action") == "draft_posted"
        for entry in history
    )


def _draft_task_input(
    candidate: WitnessOpportunityCandidate,
    *,
    decision: WitnessAutopilotDecision,
) -> str:
    """Task input that produces a visibly-draft deliverable (HIG-230)."""

    base = decision.task_input or (
        candidate.deliverable or candidate.suggested_action or candidate.summary
    )
    return _bounded(
        (
            f"{base}\n\n"
            "Produce the deliverable as a draft only. Do not send, publish, "
            "or execute anything external. Start your reply with "
            '"Draft (not sent) - " plus one line on what this is and why, '
            "then the draft itself, and end with: Tell me changes or say "
            "'go' to finalize."
        ),
        MAX_AUTOPILOT_TASK_INPUT_CHARS,
    )


def _task_input(
    candidate: WitnessOpportunityCandidate,
    *,
    decision: WitnessAutopilotDecision,
) -> str:
    if decision.task_input:
        return _bounded(
            (
                f"{decision.task_input}\n\n"
                "Respond as Kortny in a finished, human Slack note. Start from "
                "what you noticed and what you checked or handled. Do not expose "
                "internal planning, do not ask for review, and do not say this is "
                "a draft."
            ),
            MAX_AUTOPILOT_TASK_INPUT_CHARS,
        )
    fallback = candidate.suggested_action or candidate.summary
    return _bounded(
        f"{fallback}\n\nUse this Witness context: {candidate.title} - {candidate.summary}",
        MAX_AUTOPILOT_TASK_INPUT_CHARS,
    )


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


def _record_feedback(
    candidate: WitnessOpportunityCandidate,
    *,
    action: str,
    by_user_id: str,
    now: datetime,
    details: dict[str, Any],
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
    feedback["history"] = history[-25:]
    feedback["last_action"] = entry
    candidate.feedback_json = feedback


def _fallback_decision(reason: str) -> WitnessAutopilotDecision:
    return WitnessAutopilotDecision(
        decision="defer",
        risk="medium",
        action_kind="other",
        delivery_target="unknown",
        requires_user_reply=True,
        allowed_without_confirmation=False,
        reason=reason,
        task_input=None,
        confidence_score=Decimal("0.500"),
    )


def _choice(value: object, *, valid: frozenset[str], default: str) -> str:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in valid:
            return normalized
    return default


def _decimal(value: object, *, default: Decimal) -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return default


def _bool(value: object, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    return default


def _optional_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = " ".join(value.split()).strip()
    return normalized or None


def _bounded(value: str, max_chars: int) -> str:
    return " ".join(value.split()).strip()[:max_chars].strip()


def _extract_json_object(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found.")
    return stripped[start : end + 1]


def _coerce_utc(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(UTC)
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
