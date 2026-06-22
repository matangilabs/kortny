"""Accepted Witness suggestions become standing automations (HIG-224).

Acceptance is the last human gate for one-shots: accepting a ``one_shot``
candidate creates its task immediately. Recurring candidates get exactly one
confirmation surface — the existing schedule confirmation blocks — drafted
through the existing schedule creation machinery. Failures never undo the
user's acceptance; they are recorded on the candidate and a clarifying
message is posted where the suggestion lives.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from kortny.config import Settings

if TYPE_CHECKING:
    from kortny.witness.ledger.service import ProactiveActionService
from kortny.db.models import LLMProvider as DbLLMProvider
from kortny.db.models import (
    Schedule,
    Task,
    TaskEventType,
    WitnessOpportunityCandidate,
)
from kortny.llm import LLMProvider, LLMService, ModelRoute, ModelRouter, ModelRouteTier
from kortny.llm.runtime_config import (
    create_provider_for_selection,
    select_runtime_model,
)
from kortny.observability import log_observation
from kortny.scheduler.creation import (
    ScheduleCreationContext,
    ScheduleCreationService,
    ScheduleFallbackParser,
)
from kortny.scheduler.llm_parser import LLMScheduleParser
from kortny.slack.formatting import normalize_user_facing_text
from kortny.slack.outbox import SlackSideEffectOutbox
from kortny.slack.schedule_blocks import schedule_action_blocks
from kortny.tasks import TaskService
from kortny.tasks.identity import TaskIdentity
from kortny.witness.lifecycle import WitnessSlackClient
from kortny.witness.opportunities import RecurrenceGate, recurrence_evidence_line

logger = logging.getLogger(__name__)


def _get_ledger() -> ProactiveActionService:
    # Deferred import to break the ledger/policy → runner → lifecycle → ledger/service cycle.
    from kortny.witness.ledger.service import ProactiveActionService  # noqa: PLC0415

    return ProactiveActionService()


WITNESS_AUTOMATION_DRAFTED_MESSAGE = "witness_candidate_automation_drafted"
WITNESS_AUTOMATED_MESSAGE = "witness_candidate_automated"
WITNESS_AUTOMATION_FAILED_MESSAGE = "witness_candidate_automation_failed"
WITNESS_AUTOMATION_CONFIRMATION_PURPOSE = "witness_automation_confirmation"
WITNESS_AUTOMATION_CLARIFICATION_PURPOSE = "witness_automation_clarification"
WITNESS_AUTOMATION_SOURCE = "witness_automation"
SCHEDULE_WITNESS_CANDIDATE_KEY = "witness_candidate_id"
MAX_AUTOMATION_TASK_INPUT_CHARS = 1800

DEFAULT_CLARIFYING_QUESTION = (
    "I want to set this up as a standing automation, but I could not pin down "
    "the cadence. Reply with the schedule you want, like 'every weekday at 9am'."
)


@dataclass(frozen=True, slots=True)
class AutomationOutcome:
    """Result from materializing one accepted Witness candidate."""

    kind: str
    schedule_id: uuid.UUID | None = None
    task_id: uuid.UUID | None = None
    confirmation_posted: bool = False
    failure_reason: str | None = None


def materialize_acceptance(
    session: Session,
    settings: Settings | None,
    candidate: WitnessOpportunityCandidate,
    *,
    accepted_by: str,
    slack_client: WitnessSlackClient | None = None,
    schedule_parser: ScheduleFallbackParser | None = None,
    llm_provider: LLMProvider | None = None,
    provider_name: DbLLMProvider | str | None = None,
    now: datetime | None = None,
) -> AutomationOutcome:
    """Turn an accepted candidate into a task or a proposed schedule.

    The candidate must already be ``accepted`` (this runs after the status
    flip, in the same transaction). This function never raises: drafting
    failures are recorded in ``feedback_json`` and the acceptance stands.
    """

    if settings is not None and not settings.witness_automation_enabled:
        return AutomationOutcome(kind="disabled")

    run_at = _coerce_utc(now)
    kind = (
        candidate.automation_kind
        if candidate.automation_kind in {"recurring", "one_shot"}
        else "watch"
    )
    if kind == "watch":
        _record_feedback(
            candidate,
            action="automation_watch",
            by_user_id=accepted_by,
            now=run_at,
            details={"automation_kind": "watch"},
        )
        session.flush()
        return AutomationOutcome(kind="watch")

    if kind == "one_shot":
        return _materialize_one_shot(
            session,
            candidate,
            accepted_by=accepted_by,
            now=run_at,
        )

    return _materialize_recurring(
        session,
        settings,
        candidate,
        accepted_by=accepted_by,
        slack_client=slack_client,
        schedule_parser=schedule_parser,
        llm_provider=llm_provider,
        provider_name=provider_name,
        now=run_at,
    )


def sync_candidate_for_schedule_action(
    session: Session,
    schedule: Schedule,
    *,
    action: str,
    by_user_id: str,
    now: datetime | None = None,
) -> WitnessOpportunityCandidate | None:
    """Link a confirmed/declined Witness schedule draft back to its candidate.

    ``activate`` on a witness-drafted schedule moves the accepted candidate to
    the terminal ``automated`` status; ``cancel`` records that the automation
    was declined while the candidate stays accepted.
    """

    candidate_id = _witness_candidate_id(schedule)
    if candidate_id is None:
        return None
    candidate = session.scalar(
        select(WitnessOpportunityCandidate)
        .where(
            WitnessOpportunityCandidate.id == candidate_id,
            WitnessOpportunityCandidate.installation_id == schedule.installation_id,
        )
        .limit(1)
        .with_for_update()
    )
    if candidate is None or candidate.status != "accepted":
        return None

    run_at = _coerce_utc(now)
    if action == "activate" and schedule.status == "active":
        prev_status = candidate.status
        candidate.status = "automated"
        candidate.automated_schedule_id = schedule.id
        candidate.cooldown_until = None
        candidate.updated_at = run_at
        _record_feedback(
            candidate,
            action="automated",
            by_user_id=by_user_id,
            now=run_at,
            details={
                "automation_kind": candidate.automation_kind or "recurring",
                "schedule_id": str(schedule.id),
            },
        )
        _get_ledger().record_transition(
            session,
            candidate,
            to_state="automated",
            event_type="automated_recurring",
            from_state=prev_status,
            actor_id=by_user_id,
            now=run_at,
        )
        log_observation(
            logger,
            WITNESS_AUTOMATED_MESSAGE,
            candidate_id=candidate.id,
            schedule_id=schedule.id,
            automation_kind=candidate.automation_kind or "recurring",
        )
        _append_source_task_event(
            session,
            candidate,
            message=WITNESS_AUTOMATED_MESSAGE,
            payload={
                "automation_kind": candidate.automation_kind or "recurring",
                "schedule_id": str(schedule.id),
            },
        )
        session.flush()
        return candidate

    if action == "cancel":
        _record_feedback(
            candidate,
            action="automation_declined",
            by_user_id=by_user_id,
            now=run_at,
            details={"schedule_id": str(schedule.id)},
        )
        session.flush()
        return candidate

    return None


def _materialize_one_shot(
    session: Session,
    candidate: WitnessOpportunityCandidate,
    *,
    accepted_by: str,
    now: datetime,
) -> AutomationOutcome:
    try:
        channel_id = _origin_channel_id(session, candidate)
        if channel_id is None:
            raise ValueError("Witness candidate has no Slack channel for the task.")
        task_input = _one_shot_task_input(candidate)
        user_id = (
            candidate.target_slack_user_id
            or _source_task_user_id(session, candidate)
            or accepted_by
        )
        task = TaskService(session).create_task(
            installation_id=candidate.installation_id,
            slack_event_id=None,
            slack_channel_id=channel_id,
            slack_thread_ts=None,
            slack_message_ts=None,
            slack_user_id=user_id,
            input=task_input,
            identity=TaskIdentity.synthetic(
                source=WITNESS_AUTOMATION_SOURCE,
                source_id=str(candidate.id),
                input_text=task_input,
                payload={
                    "candidate_id": str(candidate.id),
                    "candidate_type": candidate.candidate_type,
                    "automation_kind": "one_shot",
                    "accepted_by": accepted_by,
                    "created_at": now.isoformat(),
                },
            ),
            source_surface=WITNESS_AUTOMATION_SOURCE,
        )
    except Exception as exc:
        return _record_automation_failure(
            session,
            candidate,
            kind="one_shot",
            accepted_by=accepted_by,
            now=now,
            failure_reason=f"{type(exc).__name__}: {exc}",
        )

    prev_status = candidate.status
    candidate.status = "automated"
    candidate.automated_task_id = task.id
    candidate.cooldown_until = None
    candidate.updated_at = now
    _record_feedback(
        candidate,
        action="automated",
        by_user_id=accepted_by,
        now=now,
        details={
            "automation_kind": "one_shot",
            "task_id": str(task.id),
        },
    )
    _get_ledger().record_transition(
        session,
        candidate,
        to_state="automated",
        event_type="automated_one_shot",
        from_state=prev_status,
        actor_id=accepted_by,
        task_id=task.id,
        now=now,
    )
    log_observation(
        logger,
        WITNESS_AUTOMATED_MESSAGE,
        candidate_id=candidate.id,
        task_id=task.id,
        automation_kind="one_shot",
    )
    _append_source_task_event(
        session,
        candidate,
        message=WITNESS_AUTOMATED_MESSAGE,
        payload={
            "automation_kind": "one_shot",
            "generated_task_id": str(task.id),
        },
    )
    session.flush()
    return AutomationOutcome(kind="one_shot", task_id=task.id)


def _materialize_recurring(
    session: Session,
    settings: Settings | None,
    candidate: WitnessOpportunityCandidate,
    *,
    accepted_by: str,
    slack_client: WitnessSlackClient | None,
    schedule_parser: ScheduleFallbackParser | None,
    llm_provider: LLMProvider | None,
    provider_name: DbLLMProvider | str | None,
    now: datetime,
) -> AutomationOutcome:
    source_task = _source_task(session, candidate)
    if source_task is None:
        return _record_automation_failure(
            session,
            candidate,
            kind="recurring",
            accepted_by=accepted_by,
            now=now,
            failure_reason="missing_source_task",
            slack_client=slack_client,
            clarifying_question=DEFAULT_CLARIFYING_QUESTION,
        )
    channel_id = _origin_channel_id(session, candidate)
    if channel_id is None:
        return _record_automation_failure(
            session,
            candidate,
            kind="recurring",
            accepted_by=accepted_by,
            now=now,
            failure_reason="missing_origin_channel",
        )

    context = ScheduleCreationContext(
        installation_id=candidate.installation_id,
        slack_channel_id=channel_id,
        slack_user_id=(
            candidate.target_slack_user_id or source_task.slack_user_id or accepted_by
        ),
        slack_thread_ts=(
            channel_id
            if channel_id.startswith("D")
            else source_task.slack_thread_ts or channel_id
        ),
        source_surface=WITNESS_AUTOMATION_SOURCE,
        source_task_id=source_task.id,
    )
    try:
        parser = schedule_parser or _build_schedule_parser(
            session,
            settings,
            source_task=source_task,
            llm_provider=llm_provider,
            provider_name=provider_name,
        )
    except Exception as exc:
        return _record_automation_failure(
            session,
            candidate,
            kind="recurring",
            accepted_by=accepted_by,
            now=now,
            failure_reason=f"{type(exc).__name__}: {exc}",
            slack_client=slack_client,
            clarifying_question=DEFAULT_CLARIFYING_QUESTION,
        )

    try:
        proposal = ScheduleCreationService(
            session,
            fallback_parser=parser,
        ).propose_from_text(
            task=source_task,
            context=context,
            text=_schedule_request_text(candidate),
            now=now,
            force_confirmation=True,
        )
    except Exception as exc:
        return _record_automation_failure(
            session,
            candidate,
            kind="recurring",
            accepted_by=accepted_by,
            now=now,
            failure_reason=f"{type(exc).__name__}: {exc}",
            slack_client=slack_client,
            clarifying_question=DEFAULT_CLARIFYING_QUESTION,
        )

    if proposal is None:
        question = (
            getattr(parser, "last_clarifying_question", None)
            or DEFAULT_CLARIFYING_QUESTION
        )
        return _record_automation_failure(
            session,
            candidate,
            kind="recurring",
            accepted_by=accepted_by,
            now=now,
            failure_reason="schedule_parse_low_confidence",
            slack_client=slack_client,
            clarifying_question=question,
        )

    schedule = proposal.schedule
    schedule.metadata_json = {
        **dict(schedule.metadata_json or {}),
        SCHEDULE_WITNESS_CANDIDATE_KEY: str(candidate.id),
    }
    confirmation_text = _schedule_confirmation_text(
        proposal.response_text,
        candidate,
        settings=settings,
        now=now,
    )
    confirmation_posted = _post_candidate_message(
        session,
        candidate,
        client=slack_client,
        channel_id=channel_id,
        text=confirmation_text,
        blocks=schedule_action_blocks(schedule),
        purpose=WITNESS_AUTOMATION_CONFIRMATION_PURPOSE,
        idempotency_key=f"{WITNESS_AUTOMATION_CONFIRMATION_PURPOSE}:{candidate.id}",
    )
    candidate.updated_at = now
    _record_feedback(
        candidate,
        action="automation_drafted",
        by_user_id=accepted_by,
        now=now,
        details={
            "automation_kind": "recurring",
            "schedule_id": str(schedule.id),
            "cadence_label": proposal.draft.cadence_label,
            "confirmation_posted": confirmation_posted,
        },
    )
    log_observation(
        logger,
        WITNESS_AUTOMATION_DRAFTED_MESSAGE,
        candidate_id=candidate.id,
        schedule_id=schedule.id,
        confirmation_posted=confirmation_posted,
    )
    _append_source_task_event(
        session,
        candidate,
        message=WITNESS_AUTOMATION_DRAFTED_MESSAGE,
        payload={
            "automation_kind": "recurring",
            "schedule_id": str(schedule.id),
            "confirmation_posted": confirmation_posted,
        },
    )
    session.flush()
    return AutomationOutcome(
        kind="recurring",
        schedule_id=schedule.id,
        confirmation_posted=confirmation_posted,
    )


def _record_automation_failure(
    session: Session,
    candidate: WitnessOpportunityCandidate,
    *,
    kind: str,
    accepted_by: str,
    now: datetime,
    failure_reason: str,
    slack_client: WitnessSlackClient | None = None,
    clarifying_question: str | None = None,
) -> AutomationOutcome:
    """Record a drafting failure without undoing the acceptance."""

    clarification_posted = False
    if clarifying_question is not None:
        channel_id = _origin_channel_id(session, candidate)
        if channel_id is not None:
            try:
                clarification_posted = _post_candidate_message(
                    session,
                    candidate,
                    client=slack_client,
                    channel_id=channel_id,
                    text=clarifying_question,
                    blocks=None,
                    purpose=WITNESS_AUTOMATION_CLARIFICATION_PURPOSE,
                    idempotency_key=(
                        f"{WITNESS_AUTOMATION_CLARIFICATION_PURPOSE}:{candidate.id}"
                    ),
                )
            except Exception:
                logger.exception(
                    "witness automation clarifying message failed candidate_id=%s",
                    candidate.id,
                )
    candidate.updated_at = now
    _record_feedback(
        candidate,
        action="automation_failed",
        by_user_id=accepted_by,
        now=now,
        details={
            "automation_kind": kind,
            "failure_reason": _bounded(failure_reason, 280),
            "clarification_posted": clarification_posted,
        },
    )
    log_observation(
        logger,
        WITNESS_AUTOMATION_FAILED_MESSAGE,
        candidate_id=candidate.id,
        automation_kind=kind,
        reason=_bounded(failure_reason, 280),
    )
    _append_source_task_event(
        session,
        candidate,
        message=WITNESS_AUTOMATION_FAILED_MESSAGE,
        payload={
            "automation_kind": kind,
            "reason": _bounded(failure_reason, 280),
        },
    )
    session.flush()
    return AutomationOutcome(kind=kind, failure_reason=failure_reason)


def _post_candidate_message(
    session: Session,
    candidate: WitnessOpportunityCandidate,
    *,
    client: WitnessSlackClient | None,
    channel_id: str,
    text: str,
    blocks: list[dict[str, Any]] | None,
    purpose: str,
    idempotency_key: str,
) -> bool:
    if client is None:
        return False
    normalized = normalize_user_facing_text(text.strip()[:1800])
    request: dict[str, Any] = {
        "channel": channel_id,
        "text": normalized,
        "thread_ts": None,
    }
    if blocks is not None:
        request["blocks"] = blocks
    result = SlackSideEffectOutbox(session).deliver(
        installation_id=candidate.installation_id,
        task_id=candidate.source_task_id,
        idempotency_key=idempotency_key,
        operation="chat_postMessage",
        purpose=purpose,
        target_channel_id=channel_id,
        request=request,
        call=lambda: client.chat_postMessage(
            channel=channel_id,
            text=normalized,
            thread_ts=None,
            blocks=blocks,
        ),
    )
    return result.delivered or result.deduped


def _build_schedule_parser(
    session: Session,
    settings: Settings | None,
    *,
    source_task: Task,
    llm_provider: LLMProvider | None,
    provider_name: DbLLMProvider | str | None,
) -> LLMScheduleParser:
    """Build the LLM schedule parser on the cheap tier (runner-style wiring)."""

    task_service = TaskService(session)
    if llm_provider is not None:
        route = ModelRoute(
            tier=ModelRouteTier.cheap_fast,
            model=llm_provider.model,
            reason="witness_automation_schedule_parse",
        )
        llm = LLMService(
            session=session,
            provider=llm_provider,
            provider_name=provider_name or DbLLMProvider.openrouter,
            task_service=task_service,
            model_route=route,
        )
        return LLMScheduleParser(llm=llm)

    if settings is None:
        raise ValueError("Witness automation needs settings or an LLM provider.")
    route = ModelRouter(settings).route_for_tier(
        ModelRouteTier.cheap_fast,
        reason="witness_automation_schedule_parse",
    )
    selection = select_runtime_model(
        session=session,
        settings=settings,
        installation_id=source_task.installation_id,
        model_route=route,
    )
    provider = create_provider_for_selection(settings=settings, selection=selection)
    llm = LLMService(
        session=session,
        provider=provider,
        provider_name=selection.provider_name,
        task_service=task_service,
        model_route=selection.model_route,
    )
    return LLMScheduleParser(llm=llm)


def _schedule_request_text(candidate: WitnessOpportunityCandidate) -> str:
    deliverable = (
        candidate.deliverable or candidate.suggested_action or candidate.summary
    )
    cadence = (candidate.cadence_suggestion or "").strip()
    if cadence:
        return _bounded(f"{cadence}, {deliverable}", 1200)
    return _bounded(deliverable, 1200)


def _schedule_confirmation_text(
    response_text: str,
    candidate: WitnessOpportunityCandidate,
    *,
    settings: Settings | None,
    now: datetime,
) -> str:
    """Land the recurrence evidence into the schedule-confirmation copy (HIG-224).

    When the candidate has crossed the recurrence gate, the proven frequency
    evidence ("I've noticed this N times since ...") leads the confirmation so
    the standing-automation proposal carries the same track record the digest
    cited. Below the gate the schedule copy is unchanged.
    """

    gate = RecurrenceGate.from_values(
        min_reinforcements=(
            settings.witness_recurring_min_reinforcements if settings else None
        ),
        min_span_days=(settings.witness_recurring_min_span_days if settings else None),
    )
    evidence = recurrence_evidence_line(
        candidate,
        now=now,
        min_reinforcements=gate.min_reinforcements,
        min_span=gate.min_span,
    )
    if evidence is None:
        return response_text
    return normalize_user_facing_text(f"{evidence}. {response_text}".strip())


def _one_shot_task_input(candidate: WitnessOpportunityCandidate) -> str:
    action = candidate.deliverable or candidate.suggested_action or candidate.summary
    return _bounded(
        f"{action}\n\nUse this Witness context: "
        f"{candidate.title} - {candidate.summary}",
        MAX_AUTOMATION_TASK_INPUT_CHARS,
    )


def _origin_channel_id(
    session: Session,
    candidate: WitnessOpportunityCandidate,
) -> str | None:
    """Return the surface where the suggestion lives: its DM, else its channel."""

    if candidate.channel_id:
        return candidate.channel_id
    if candidate.visibility_scope_id and candidate.visibility_scope_type in {
        "channel",
        "private_channel",
        "dm",
    }:
        return candidate.visibility_scope_id
    source_task = _source_task(session, candidate)
    if source_task is not None:
        return source_task.slack_channel_id
    return None


def _source_task(
    session: Session,
    candidate: WitnessOpportunityCandidate,
) -> Task | None:
    if candidate.source_task_id is None:
        return None
    return session.get(Task, candidate.source_task_id)


def _source_task_user_id(
    session: Session,
    candidate: WitnessOpportunityCandidate,
) -> str | None:
    source_task = _source_task(session, candidate)
    return source_task.slack_user_id if source_task is not None else None


def _witness_candidate_id(schedule: Schedule) -> uuid.UUID | None:
    metadata = (
        schedule.metadata_json if isinstance(schedule.metadata_json, dict) else {}
    )
    value = metadata.get(SCHEDULE_WITNESS_CANDIDATE_KEY)
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return uuid.UUID(value.strip())
    except ValueError:
        return None


def _append_source_task_event(
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


def _bounded(value: str, max_chars: int) -> str:
    return " ".join(value.split()).strip()[:max_chars].strip()


def _coerce_utc(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(UTC)
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
