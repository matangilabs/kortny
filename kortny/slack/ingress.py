"""Slack event ingress into durable Kortny tasks."""

from __future__ import annotations

import logging
import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from kortny.db.models import Installation, Task, TaskEventType
from kortny.intent import (
    IntentClassifier,
    IntentDecision,
    IntentRequest,
    IntentSurface,
    should_classify_channel_message,
    should_create_task_from_soft_mention,
    should_react_to_rejected_soft_mention,
)
from kortny.memory import Fact, PendingFact, WorkspaceStateService
from kortny.slack.acknowledgement import (
    AcknowledgementGenerator,
    StaticAcknowledgementGenerator,
    generate_acknowledgement,
)
from kortny.slack.reactions import (
    ACK_REACTION_ADD_FAILED_MESSAGE,
    ACK_REACTION_ADDED_MESSAGE,
    ACK_REACTION_UNAVAILABLE_MESSAGE,
    LibraryReactionProvider,
    ReactionProvider,
)
from kortny.tasks import TaskService

LEADING_MENTION_RE = re.compile(r"^\s*<@[^>]+>\s*")
IGNORED_DM_SUBTYPES = frozenset(
    {
        "bot_message",
        "channel_join",
        "group_join",
        "message_changed",
        "message_deleted",
    }
)
logger = logging.getLogger(__name__)
VERBAL_ACKS_ENABLED = False
REACTION_CANCEL = "x"
REACTION_RETRY = "arrows_counterclockwise"
REACTION_CONFIRM = "white_check_mark"
REACTION_REJECT = "no_entry_sign"
CONFIRMATION_REACTIONS = frozenset({REACTION_CONFIRM, REACTION_REJECT})
INTENT_CLASSIFIED_MESSAGE = "intent_classification_completed"
INTENT_CLASSIFICATION_FAILED_MESSAGE = "intent_classification_failed"


class SlackPostMessageClient(Protocol):
    """Subset of the Slack WebClient used by ingress."""

    def chat_postMessage(
        self,
        *,
        channel: str,
        text: str,
        thread_ts: str | None = None,
    ) -> Mapping[str, Any]:
        """Post a Slack message and return the API response."""


@dataclass(frozen=True, slots=True)
class AppMentionResult:
    """Result of processing a Slack event that creates or finds a task."""

    task: Task
    created: bool
    thread_ts: str
    acknowledgement_ts: str | None = None


@dataclass(frozen=True, slots=True)
class ReactionResult:
    """Result of processing a Slack reaction event."""

    handled: bool
    action: str
    task: Task | None = None
    reason: str | None = None


class SlackIngress:
    """Turns Slack trigger events into queued tasks."""

    def __init__(
        self,
        *,
        session: Session,
        client: SlackPostMessageClient,
        task_service: TaskService | None = None,
        acknowledgement_generator: AcknowledgementGenerator | None = None,
        reaction_provider: ReactionProvider | None = None,
        intent_classifier: IntentClassifier | None = None,
    ) -> None:
        self.session = session
        self.client = client
        self.task_service = task_service or TaskService(session)
        self.acknowledgement_generator = (
            acknowledgement_generator or StaticAcknowledgementGenerator()
        )
        self.reaction_provider = reaction_provider or LibraryReactionProvider()
        self.intent_classifier = intent_classifier

    def handle_app_mention(
        self,
        *,
        body: Mapping[str, Any],
        event: Mapping[str, Any],
    ) -> AppMentionResult:
        """Create a task for a Slack app_mention and acknowledge it visually."""

        return self._handle_addressed_message(
            body=body,
            event=event,
            input_text=_task_input(event, strip_leading_mention=True),
            source="app_mention",
        )

    def handle_dm(
        self,
        *,
        body: Mapping[str, Any],
        event: Mapping[str, Any],
    ) -> AppMentionResult | None:
        """Create a task for a direct message user event."""

        ignore_reason = _dm_ignore_reason(event)
        if ignore_reason is not None:
            logger.info(
                "slack dm ignored reason=%s event_id=%s channel=%s",
                ignore_reason,
                body.get("event_id"),
                event.get("channel"),
            )
            return None

        return self._handle_addressed_message(
            body=body,
            event=event,
            input_text=_task_input(event, strip_leading_mention=False),
            source="dm",
        )

    def handle_channel_message(
        self,
        *,
        body: Mapping[str, Any],
        event: Mapping[str, Any],
        app_name: str = "kortny",
    ) -> AppMentionResult | None:
        """Create a task for a direct soft app-name mention in a channel."""

        if not should_classify_channel_message(event, app_name=app_name):
            logger.info(
                "slack channel_message ignored reason=not_soft_mention event_id=%s channel=%s",
                body.get("event_id"),
                event.get("channel"),
            )
            return None

        event_id = _required_str(body, "event_id")
        channel_id = _required_str(event, "channel")
        message_ts = _required_str(event, "ts")
        existing = self._find_existing_task(event_id, channel_id, message_ts)
        if existing is not None:
            logger.info(
                "slack channel_message duplicate task_id=%s event_id=%s channel=%s thread_ts=%s",
                existing.id,
                event_id,
                channel_id,
                existing.slack_thread_ts or _event_thread_ts(event),
            )
            return AppMentionResult(
                task=existing,
                created=False,
                thread_ts=existing.slack_thread_ts
                or _context_thread_ts(
                    event,
                    source="channel_message",
                    channel_id=channel_id,
                ),
            )

        input_text = _task_input(event, strip_leading_mention=False)
        intent_decision = self._classify_soft_channel_message(
            event=event,
            input_text=input_text,
            app_name=app_name,
        )
        if intent_decision is None:
            return None
        if not should_create_task_from_soft_mention(intent_decision):
            self._post_rejected_soft_mention_reaction(
                source="channel_message",
                channel_id=channel_id,
                message_ts=message_ts,
                input_text=input_text,
                intent_decision=intent_decision,
            )
            logger.info(
                "slack channel_message ignored reason=intent_rejected event_id=%s channel=%s classification=%s confidence=%.3f addressed=%s",
                event_id,
                channel_id,
                intent_decision.classification.value,
                intent_decision.confidence,
                intent_decision.addressed_to_kortny,
            )
            return None

        return self._handle_addressed_message(
            body=body,
            event=event,
            input_text=input_text,
            source="channel_message",
            preclassified_intent_decision=intent_decision,
        )

    def handle_reaction_added(
        self,
        *,
        body: Mapping[str, Any],
        event: Mapping[str, Any],
    ) -> ReactionResult:
        """Dispatch a Slack reaction to cancel/retry/confirmation handlers."""

        del body
        reaction = _required_str(event, "reaction")
        user_id = _required_str(event, "user")
        item = event.get("item")
        if not isinstance(item, Mapping) or item.get("type") != "message":
            return ReactionResult(
                handled=False,
                action="ignored",
                reason="unsupported_item",
            )

        channel_id = _required_str(item, "channel")
        message_ts = _required_str(item, "ts")
        if reaction in CONFIRMATION_REACTIONS:
            return self._handle_confirmation_reaction(
                reaction=reaction,
                channel_id=channel_id,
                message_ts=message_ts,
                user_id=user_id,
            )

        task = self.task_service.get_by_slack_reaction_target(channel_id, message_ts)
        if task is None:
            logger.info(
                "slack reaction ignored reason=no_task channel=%s message_ts=%s reaction=%s user=%s",
                channel_id,
                message_ts,
                reaction,
                user_id,
            )
            return ReactionResult(
                handled=False,
                action="ignored",
                reason="no_task",
            )

        if reaction == REACTION_CANCEL:
            return self._handle_cancel_reaction(task, user_id=user_id)
        if reaction == REACTION_RETRY:
            return self._handle_retry_reaction(task, user_id=user_id)

        return ReactionResult(
            handled=False,
            action="ignored",
            task=task,
            reason="unsupported_reaction",
        )

    def _handle_addressed_message(
        self,
        *,
        body: Mapping[str, Any],
        event: Mapping[str, Any],
        input_text: str,
        source: str,
        preclassified_intent_decision: IntentDecision | None = None,
    ) -> AppMentionResult:
        event_id = _required_str(body, "event_id")
        channel_id = _required_str(event, "channel")
        message_ts = _required_str(event, "ts")
        existing = self._find_existing_task(event_id, channel_id, message_ts)
        if existing is not None:
            logger.info(
                "slack %s duplicate task_id=%s event_id=%s channel=%s thread_ts=%s",
                source,
                existing.id,
                event_id,
                channel_id,
                existing.slack_thread_ts or _event_thread_ts(event),
            )
            return AppMentionResult(
                task=existing,
                created=False,
                thread_ts=existing.slack_thread_ts
                or _context_thread_ts(event, source=source, channel_id=channel_id),
            )

        team_id = _team_id(body, event)
        user_id = _required_str(event, "user")
        thread_ts = _context_thread_ts(event, source=source, channel_id=channel_id)
        installation = self._get_or_create_installation(team_id)

        task = self.task_service.create_task(
            installation_id=installation.id,
            slack_event_id=event_id,
            slack_channel_id=channel_id,
            slack_thread_ts=thread_ts,
            slack_message_ts=message_ts,
            slack_user_id=user_id,
            input=input_text,
        )
        logger.info(
            "slack %s created task_id=%s event_id=%s channel=%s thread_ts=%s user=%s input_len=%s",
            source,
            task.id,
            event_id,
            channel_id,
            thread_ts,
            user_id,
            len(task.input),
        )
        if preclassified_intent_decision is None:
            intent_decision = self._classify_intent(
                task=task,
                source=source,
                event=event,
            )
        else:
            intent_decision = preclassified_intent_decision
            self._record_intent_decision(
                task=task,
                source=source,
                decision=intent_decision,
            )
        self._post_ack_reaction(
            task=task,
            source=source,
            channel_id=channel_id,
            message_ts=message_ts,
            intent_decision=intent_decision,
        )

        if _should_skip_visible_ack(event, source=source):
            logger.info(
                "slack %s acknowledgement skipped task_id=%s channel=%s thread_ts=%s",
                source,
                task.id,
                channel_id,
                thread_ts,
            )
            return AppMentionResult(
                task=task,
                created=True,
                thread_ts=thread_ts,
            )

        acknowledgement_text = generate_acknowledgement(
            self.acknowledgement_generator,
            session=self.session,
            task=task,
            task_service=self.task_service,
        )
        acknowledgement = self.client.chat_postMessage(
            channel=channel_id,
            text=acknowledgement_text,
            thread_ts=thread_ts,
        )
        acknowledgement_ts = _optional_response_ts(acknowledgement)
        self.task_service.append_event(
            task,
            TaskEventType.message_posted,
            {
                "channel": channel_id,
                "thread_ts": thread_ts,
                "message_ts": acknowledgement_ts,
                "text": acknowledgement_text,
                "purpose": "acknowledgement",
            },
        )
        logger.info(
            "slack %s acknowledgement posted task_id=%s channel=%s thread_ts=%s message_ts=%s",
            source,
            task.id,
            channel_id,
            thread_ts,
            acknowledgement_ts,
        )

        return AppMentionResult(
            task=task,
            created=True,
            thread_ts=thread_ts,
            acknowledgement_ts=acknowledgement_ts,
        )

    def _find_existing_task(
        self,
        event_id: str,
        channel_id: str,
        message_ts: str,
    ) -> Task | None:
        existing = self.task_service.get_by_slack_event_id(event_id)
        if existing is None:
            existing = self.task_service.get_by_slack_message(channel_id, message_ts)
        return existing

    def _post_ack_reaction(
        self,
        *,
        task: Task,
        source: str,
        channel_id: str,
        message_ts: str,
        intent_decision: IntentDecision | None,
    ) -> None:
        choice = self.reaction_provider.acknowledgement_reaction(
            input_text=task.input,
            source=source,
            intent_decision=intent_decision,
        )
        reactions_add = getattr(self.client, "reactions_add", None)
        if not callable(reactions_add):
            self.task_service.append_event(
                task,
                TaskEventType.log,
                {
                    "message": ACK_REACTION_UNAVAILABLE_MESSAGE,
                    "source": source,
                    "channel": channel_id,
                    "message_ts": message_ts,
                    "reaction": choice.name,
                    "reaction_intent": choice.intent,
                },
            )
            return
        try:
            reactions_add(
                channel=channel_id,
                name=choice.name,
                timestamp=message_ts,
            )
        except Exception as exc:
            logger.info(
                "slack %s ack reaction failed task_id=%s channel=%s message_ts=%s reaction=%s error_type=%s error=%s",
                source,
                task.id,
                channel_id,
                message_ts,
                choice.name,
                type(exc).__name__,
                exc,
            )
            self.task_service.append_event(
                task,
                TaskEventType.log,
                {
                    "message": ACK_REACTION_ADD_FAILED_MESSAGE,
                    "source": source,
                    "channel": channel_id,
                    "message_ts": message_ts,
                    "reaction": choice.name,
                    "reaction_intent": choice.intent,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            return

        self.task_service.append_event(
            task,
            TaskEventType.log,
            {
                "message": ACK_REACTION_ADDED_MESSAGE,
                "source": source,
                "channel": channel_id,
                "message_ts": message_ts,
                "reaction": choice.name,
                "reaction_intent": choice.intent,
            },
        )
        logger.info(
            "slack %s ack reaction added task_id=%s channel=%s message_ts=%s reaction=%s",
            source,
            task.id,
            channel_id,
            message_ts,
            choice.name,
        )

    def _post_rejected_soft_mention_reaction(
        self,
        *,
        source: str,
        channel_id: str,
        message_ts: str,
        input_text: str,
        intent_decision: IntentDecision,
    ) -> None:
        if not should_react_to_rejected_soft_mention(intent_decision):
            return

        choice = self.reaction_provider.acknowledgement_reaction(
            input_text=input_text,
            source=source,
            intent_decision=intent_decision,
        )
        reactions_add = getattr(self.client, "reactions_add", None)
        if not callable(reactions_add):
            logger.info(
                "slack %s rejected soft mention reaction unavailable channel=%s message_ts=%s reaction=%s",
                source,
                channel_id,
                message_ts,
                choice.name,
            )
            return
        try:
            reactions_add(
                channel=channel_id,
                name=choice.name,
                timestamp=message_ts,
            )
        except Exception as exc:
            logger.info(
                "slack %s rejected soft mention reaction failed channel=%s message_ts=%s reaction=%s error_type=%s error=%s",
                source,
                channel_id,
                message_ts,
                choice.name,
                type(exc).__name__,
                exc,
            )
            return

        logger.info(
            "slack %s rejected soft mention reaction added channel=%s message_ts=%s reaction=%s classification=%s",
            source,
            channel_id,
            message_ts,
            choice.name,
            intent_decision.classification.value,
        )

    def _classify_intent(
        self,
        *,
        task: Task,
        source: str,
        event: Mapping[str, Any],
    ) -> IntentDecision | None:
        if self.intent_classifier is None:
            return None

        try:
            decision = self.intent_classifier.classify(
                task_id=task.id,
                request=IntentRequest(
                    text=task.input,
                    surface=_intent_surface(source),
                    is_thread_follow_up=_is_thread_follow_up(event),
                    has_files=bool(_event_files(event)),
                ),
            )
        except Exception as exc:
            logger.info(
                "slack intent classification failed task_id=%s source=%s error_type=%s error=%s",
                task.id,
                source,
                type(exc).__name__,
                exc,
            )
            self.task_service.append_event(
                task,
                TaskEventType.log,
                {
                    "message": INTENT_CLASSIFICATION_FAILED_MESSAGE,
                    "source": source,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            return None

        self._record_intent_decision(task=task, source=source, decision=decision)
        return decision

    def _classify_soft_channel_message(
        self,
        *,
        event: Mapping[str, Any],
        input_text: str,
        app_name: str,
    ) -> IntentDecision | None:
        if self.intent_classifier is None:
            logger.info(
                "slack channel_message ignored reason=no_intent_classifier channel=%s",
                event.get("channel"),
            )
            return None
        try:
            decision = self.intent_classifier.classify(
                request=IntentRequest(
                    text=input_text,
                    surface=IntentSurface.channel_message,
                    app_name=app_name,
                    is_thread_follow_up=_is_thread_follow_up(event),
                    has_files=bool(_event_files(event)),
                ),
            )
        except Exception as exc:
            logger.info(
                "slack channel_message intent classification failed channel=%s error_type=%s error=%s",
                event.get("channel"),
                type(exc).__name__,
                exc,
            )
            return None

        logger.info(
            "slack channel_message intent classified channel=%s classification=%s confidence=%.3f addressed=%s",
            event.get("channel"),
            decision.classification.value,
            decision.confidence,
            decision.addressed_to_kortny,
        )
        return decision

    def _record_intent_decision(
        self,
        *,
        task: Task,
        source: str,
        decision: IntentDecision,
    ) -> None:
        self.task_service.append_event(
            task,
            TaskEventType.log,
            {
                "message": INTENT_CLASSIFIED_MESSAGE,
                "source": source,
                "decision": decision.model_dump(mode="json"),
            },
        )
        logger.info(
            "slack intent classified task_id=%s source=%s classification=%s confidence=%.3f",
            task.id,
            source,
            decision.classification.value,
            decision.confidence,
        )

    def _get_or_create_installation(self, slack_team_id: str) -> Installation:
        existing = self.session.scalar(
            select(Installation).where(Installation.slack_team_id == slack_team_id)
        )
        if existing is not None:
            return existing

        installation = Installation(slack_team_id=slack_team_id)
        try:
            with self.session.begin_nested():
                self.session.add(installation)
                self.session.flush()
        except IntegrityError:
            existing = self.session.scalar(
                select(Installation).where(Installation.slack_team_id == slack_team_id)
            )
            if existing is None:
                raise
            return existing

        return installation

    def _handle_cancel_reaction(
        self,
        task: Task,
        *,
        user_id: str,
    ) -> ReactionResult:
        if task.slack_user_id != user_id:
            logger.info(
                "slack cancel reaction ignored reason=non_owner task_id=%s owner=%s user=%s",
                task.id,
                task.slack_user_id,
                user_id,
            )
            return ReactionResult(
                handled=False,
                action="cancel",
                task=task,
                reason="non_owner",
            )

        cancelled = self.task_service.cancel_task(task, by_user_id=user_id)
        if cancelled is None:
            return ReactionResult(
                handled=False,
                action="cancel",
                task=task,
                reason="not_cancellable",
            )

        logger.info(
            "slack cancel reaction handled task_id=%s user=%s", task.id, user_id
        )
        return ReactionResult(handled=True, action="cancel", task=cancelled)

    def _handle_retry_reaction(
        self,
        task: Task,
        *,
        user_id: str,
    ) -> ReactionResult:
        if task.slack_user_id != user_id:
            logger.info(
                "slack retry reaction ignored reason=non_owner task_id=%s owner=%s user=%s",
                task.id,
                task.slack_user_id,
                user_id,
            )
            return ReactionResult(
                handled=False,
                action="retry",
                task=task,
                reason="non_owner",
            )

        retried = self.task_service.retry_failed_task(task, by_user_id=user_id)
        if retried is None:
            return ReactionResult(
                handled=False,
                action="retry",
                task=task,
                reason="not_failed",
            )

        logger.info("slack retry reaction handled task_id=%s user=%s", task.id, user_id)
        return ReactionResult(handled=True, action="retry", task=retried)

    def _handle_confirmation_reaction(
        self,
        *,
        reaction: str,
        channel_id: str,
        message_ts: str,
        user_id: str,
    ) -> ReactionResult:
        memory_service = WorkspaceStateService(
            self.session,
            task_service=self.task_service,
        )
        try:
            if reaction == REACTION_CONFIRM:
                fact = memory_service.confirm(
                    message_ts,
                    user_id,
                    channel_id=channel_id,
                )
                logger.info(
                    "slack memory confirmation handled fact_id=%s key=%s user=%s",
                    fact.id,
                    fact.key,
                    user_id,
                )
                self._post_memory_reaction_result(
                    task_id=fact.source_task_id,
                    channel_id=channel_id,
                    text=_memory_confirmed_text(fact),
                    purpose="memory_confirmed",
                )
                return ReactionResult(handled=True, action="confirm_memory")

            pending = memory_service.reject(
                message_ts,
                user_id,
                channel_id=channel_id,
            )
            logger.info(
                "slack memory rejection handled key=%s user=%s",
                pending.key,
                user_id,
            )
            self._post_memory_reaction_result(
                task_id=pending.task_id,
                channel_id=channel_id,
                text=_memory_rejected_text(pending),
                purpose="memory_rejected",
            )
            return ReactionResult(handled=True, action="reject_memory")
        except LookupError:
            logger.info(
                "slack confirmation reaction ignored reason=no_pending_memory_proposal channel=%s message_ts=%s reaction=%s user=%s",
                channel_id,
                message_ts,
                reaction,
                user_id,
            )
            return ReactionResult(
                handled=False,
                action="confirmation",
                reason="no_pending_memory_proposal",
            )

    def _post_memory_reaction_result(
        self,
        *,
        task_id: uuid.UUID | None,
        channel_id: str,
        text: str,
        purpose: str,
    ) -> None:
        if task_id is None:
            return
        task = self.task_service.get_task(task_id)
        if task is None:
            return

        thread_ts = _result_thread_ts(task)
        response = self.client.chat_postMessage(
            channel=channel_id,
            text=text,
            thread_ts=thread_ts,
        )
        self.task_service.append_event(
            task,
            TaskEventType.message_posted,
            {
                "channel": channel_id,
                "thread_ts": thread_ts,
                "message_ts": _optional_response_ts(response),
                "text": text,
                "purpose": purpose,
            },
        )


def _required_str(values: Mapping[str, Any], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Slack event is missing {key!r}")
    return value


def _team_id(body: Mapping[str, Any], event: Mapping[str, Any]) -> str:
    team_id = body.get("team_id") or event.get("team")
    if not isinstance(team_id, str) or not team_id:
        raise ValueError("Slack event is missing team_id")
    return team_id


def _result_thread_ts(task: Task) -> str | None:
    if task.slack_channel_id.startswith("D"):
        return None
    return task.slack_thread_ts or task.slack_message_ts


def _memory_confirmed_text(fact: Fact) -> str:
    detail = (fact.value_text or "").strip()
    if detail:
        return f"Saved. I'll use this going forward: {detail}"
    return "Saved. I'll use this going forward."


def _memory_rejected_text(pending: PendingFact) -> str:
    del pending
    return "No problem, I won't save that."


def _event_thread_ts(event: Mapping[str, Any]) -> str:
    thread_ts = event.get("thread_ts") or event.get("ts")
    if not isinstance(thread_ts, str) or not thread_ts:
        raise ValueError("Slack event is missing ts")
    return thread_ts


def _context_thread_ts(
    event: Mapping[str, Any],
    *,
    source: str,
    channel_id: str,
) -> str:
    # DMs are linear conversations in the product. We still post replies as
    # normal unthreaded DM messages, but group task context by DM channel so
    # follow-ups can resolve "this report" and prior attached files.
    if source == "dm":
        return channel_id
    return _event_thread_ts(event)


def _is_thread_follow_up(event: Mapping[str, Any]) -> bool:
    thread_ts = event.get("thread_ts")
    message_ts = event.get("ts")
    return (
        isinstance(thread_ts, str)
        and bool(thread_ts)
        and isinstance(message_ts, str)
        and thread_ts != message_ts
    )


def _intent_surface(source: str) -> IntentSurface:
    if source == "dm":
        return IntentSurface.dm
    if source == "channel_message":
        return IntentSurface.channel_message
    return IntentSurface.app_mention


def _should_skip_visible_ack(event: Mapping[str, Any], *, source: str) -> bool:
    if not VERBAL_ACKS_ENABLED:
        return True
    return source == "dm" or _is_thread_follow_up(event)


def _dm_ignore_reason(event: Mapping[str, Any]) -> str | None:
    channel_type = event.get("channel_type")
    if channel_type != "im":
        return "non_dm"
    subtype = event.get("subtype")
    if isinstance(subtype, str) and subtype in IGNORED_DM_SUBTYPES:
        return f"subtype:{subtype}"
    bot_id = event.get("bot_id")
    if isinstance(bot_id, str) and bot_id:
        return "bot_id"
    return None


def _task_input(
    event: Mapping[str, Any],
    *,
    strip_leading_mention: bool,
) -> str:
    text = event.get("text")
    if not isinstance(text, str):
        return ""
    stripped = text.strip()
    if strip_leading_mention:
        stripped = LEADING_MENTION_RE.sub("", text, count=1).strip()
    return _append_file_context(stripped or text.strip(), event)


def _append_file_context(input_text: str, event: Mapping[str, Any]) -> str:
    files = _event_files(event)
    if not files:
        return input_text

    file_lines: list[str] = []
    for file in files:
        file_id = _optional_file_string(file.get("id"))
        if file_id is None:
            continue
        file_lines.append(f"- id: {file_id}")
        for key, label in (
            ("name", "name"),
            ("title", "title"),
            ("mimetype", "mimetype"),
            ("size", "size_bytes"),
        ):
            value = file.get(key)
            if isinstance(value, str) and value.strip():
                file_lines.append(f"  {label}: {value.strip()}")
            elif (
                key == "size" and isinstance(value, int) and not isinstance(value, bool)
            ):
                file_lines.append(f"  {label}: {value}")
    if not file_lines:
        return input_text

    lines = [input_text, "", "<slack_files>", *file_lines]
    lines.append("</slack_files>")
    return "\n".join(lines)


def _event_files(event: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    raw_files = event.get("files")
    if not isinstance(raw_files, list):
        return ()
    return tuple(file for file in raw_files if isinstance(file, Mapping))


def _optional_file_string(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _optional_response_ts(response: Mapping[str, Any]) -> str | None:
    ts = response.get("ts")
    if isinstance(ts, str) and ts:
        return ts
    message = response.get("message")
    if isinstance(message, Mapping):
        message_ts = message.get("ts")
        if isinstance(message_ts, str) and message_ts:
            return message_ts
    return None
