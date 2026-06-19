"""Confirmation-gated workspace memory service."""

from __future__ import annotations

import builtins
import json
import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Protocol

from sqlalchemy import ColumnElement, Select, or_, select
from sqlalchemy.orm import Session

from kortny.db.models import Task, TaskEvent, TaskEventType, WorkspaceState
from kortny.embeddings import (
    FACT_EMBEDDING_KIND,
    EmbeddingIndex,
    fact_embedding_text,
)
from kortny.tasks import TaskService

SCOPE_TYPES = frozenset({"workspace", "channel", "user"})
STATE_STATUSES = frozenset(
    {"proposed", "active", "rejected", "superseded", "forgotten"}
)
SOURCE_KINDS = frozenset(
    {
        "user_explicit",
        "agent_proposed",
        "summarizer_proposed",
        "observer_proposed",
        "import",
    }
)
PENDING_PROPOSAL_MESSAGE = "workspace_state_proposal_created"
CONFIRMED_PROPOSAL_MESSAGE = "workspace_state_proposal_confirmed"
AUTO_ACTIVATED_MESSAGE = "workspace_state_fact_auto_activated"
REJECTED_PROPOSAL_MESSAGE = "workspace_state_proposal_rejected"
FORGOTTEN_FACT_MESSAGE = "workspace_state_fact_forgotten"
FORGET_REQUEST_MESSAGE = "workspace_state_forget_requested"
INSPECTED_MEMORY_MESSAGE = "workspace_state_inspected"
BLOCKED_SECRET_MESSAGE = "workspace_state_secret_blocked"
DEFAULT_SOURCE_KIND = "agent_proposed"
GENERIC_MEMORY_DETAIL_KEYS = frozenset(
    {
        "details",
        "preference",
        "preferences",
        "summary",
        "text",
        "value",
    }
)
MAX_APPENDED_MEMORY_DETAILS = 6
SECRET_KEY_RE = re.compile(
    r"(api[_-]?key|access[_-]?token|refresh[_-]?token|bearer[_-]?token|"
    r"bot[_-]?token|client[_-]?secret|signing[_-]?secret|private[_-]?key|"
    r"password|passwd|secret|token)",
    re.I,
)
SECRET_VALUE_RE = re.compile(
    r"(xox[baprs]-[A-Za-z0-9-]+|xapp-[A-Za-z0-9-]+|"
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----|"
    r"bearer\s+[A-Za-z0-9._~+/\-]+=*|"
    r"\bsk-[A-Za-z0-9][A-Za-z0-9_-]{16,}\b|"
    r"\b[A-Za-z0-9._%+-]+:[A-Za-z0-9._~+/\-]{12,}\b)",
    re.I,
)
SECRET_ASSIGNMENT_RE = re.compile(
    r"(api[_ -]?key|token|secret|password)\s*[:=]\s*\S{8,}",
    re.I,
)


class WorkspaceStateServiceError(RuntimeError):
    """Raised when the workspace memory service cannot complete an operation."""


class WorkspaceStateSecretError(WorkspaceStateServiceError):
    """Raised when a memory proposal appears to contain a secret."""

    def __init__(self, reason: str) -> None:
        super().__init__("Memory proposals cannot store secrets")
        self.reason = reason


@dataclass(frozen=True, slots=True)
class MemoryPromptThread:
    """Slack channel/thread target for a memory confirmation prompt."""

    channel_id: str
    thread_ts: str
    task_id: uuid.UUID | None = None
    # Matches SlackThread so the posting boundary threads assistant replies
    # correctly even for memory-confirmation prompts (HIG-247).
    is_assistant: bool = False


class ConfirmationPoster(Protocol):
    """Subset of SlackPoster needed for memory confirmation prompts."""

    def post_message(
        self,
        thread: Any,
        text: str,
        *,
        purpose: str = "result",
    ) -> str:
        """Post a prompt and return the Slack message timestamp."""


@dataclass(frozen=True, slots=True)
class Fact:
    """Public service view of a workspace_state row."""

    id: uuid.UUID
    installation_id: uuid.UUID
    scope_type: str
    scope_id: str | None
    key: str
    value: dict[str, Any]
    value_text: str | None
    status: str
    source_kind: str
    source_task_id: uuid.UUID | None
    source_event_id: int | None
    confirmed_by_user_id: str | None
    confirmed_at: datetime | None
    source_slack_channel_id: str | None
    source_slack_message_ts: str | None
    proposed_by: str
    proposed_reason: str | None
    confidence_reason: str | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class PendingFact:
    """A proposed memory write awaiting Slack reaction confirmation."""

    event_id: int
    task_id: uuid.UUID
    installation_id: uuid.UUID
    scope_type: str
    scope_id: str | None
    key: str
    value: dict[str, Any]
    value_text: str | None
    prompt_channel_id: str
    prompt_message_ts: str
    proposed_by: str
    status: str


class WorkspaceStateService:
    """Service API over workspace_state and pending memory proposals."""

    def __init__(
        self,
        session: Session,
        *,
        task_service: TaskService | None = None,
        poster: ConfirmationPoster | None = None,
        embedding_index: EmbeddingIndex | None = None,
    ) -> None:
        self.session = session
        self.task_service = task_service or TaskService(session)
        self.poster = poster
        self.embedding_index = embedding_index

    def get(
        self,
        installation_id: uuid.UUID,
        scope_type: str,
        scope_id: str | None,
        key: str,
    ) -> Fact | None:
        """Return the current active fact for a key/scope, if one exists."""

        _validate_scope(scope_type, scope_id)
        state = self.session.scalar(
            self._current_statement(installation_id)
            .where(
                WorkspaceState.scope_type == scope_type,
                _scope_id_clause(scope_id),
                WorkspaceState.key == _normalize_key(key),
            )
            .order_by(WorkspaceState.created_at.desc(), WorkspaceState.id.desc())
            .limit(1)
        )
        if state is None:
            return None
        return _fact_from_state(state)

    def list(
        self,
        installation_id: uuid.UUID,
        *,
        scope_type: str | None = None,
        scope_id: str | None = None,
    ) -> builtins.list[Fact]:
        """Return current active facts, ignoring superseded/forgotten rows."""

        statement = self._current_statement(installation_id)
        if scope_type is not None:
            _validate_scope(scope_type, scope_id)
            statement = statement.where(
                WorkspaceState.scope_type == scope_type,
                _scope_id_clause(scope_id),
            )
        elif scope_id is not None:
            raise ValueError("scope_id requires scope_type")

        return [
            _fact_from_state(state)
            for state in self.session.scalars(
                statement.order_by(
                    WorkspaceState.scope_type,
                    WorkspaceState.scope_id,
                    WorkspaceState.key,
                    WorkspaceState.created_at,
                )
            )
        ]

    def list_history(
        self,
        installation_id: uuid.UUID,
        *,
        key: str,
        scope_type: str | None = None,
        scope_id: str | None = None,
    ) -> builtins.list[Fact]:
        """Return all state rows for a key, including superseded/forgotten rows."""

        statement = select(WorkspaceState).where(
            WorkspaceState.installation_id == installation_id,
            WorkspaceState.key == _normalize_key(key),
        )
        if scope_type is not None:
            _validate_scope(scope_type, scope_id)
            statement = statement.where(
                WorkspaceState.scope_type == scope_type,
                _scope_id_clause(scope_id),
            )
        elif scope_id is not None:
            raise ValueError("scope_id requires scope_type")

        return [
            _fact_from_state(state)
            for state in self.session.scalars(
                statement.order_by(
                    WorkspaceState.confirmed_at,
                    WorkspaceState.created_at,
                    WorkspaceState.id,
                )
            )
        ]

    def propose(
        self,
        installation_id: uuid.UUID,
        scope_type: str,
        scope_id: str | None,
        key: str,
        value: Mapping[str, Any],
        source_task_id: uuid.UUID,
        *,
        value_text: str | None = None,
        proposed_reason: str | None = None,
        confidence_score: Decimal | float | str | None = None,
        confidence_reason: str | None = None,
        source_kind: str = DEFAULT_SOURCE_KIND,
    ) -> PendingFact:
        """Post a confirmation prompt and record a pending write in task_events."""

        if self.poster is None:
            raise WorkspaceStateServiceError("A poster is required to propose a fact")
        task = self.task_service.get_task(source_task_id)
        if task is None:
            raise LookupError(f"Task not found: {source_task_id}")
        if task.installation_id != installation_id:
            raise ValueError("source_task_id does not belong to installation_id")

        _validate_scope(scope_type, scope_id)
        source_kind = _validate_source_kind(source_kind)
        normalized_key = _normalize_key(key)
        value_json = _json_object(value)
        readable_value = value_text or _value_text(value_json)
        readable_value = _faithful_value_text(readable_value, value_json)
        if secret_reason := _secret_candidate_reason(
            key=normalized_key,
            value=value_json,
            value_text=readable_value,
        ):
            self.task_service.append_event(
                task,
                TaskEventType.log,
                {
                    "message": BLOCKED_SECRET_MESSAGE,
                    "installation_id": str(installation_id),
                    "scope_type": scope_type,
                    "scope_id": scope_id,
                    "key": normalized_key,
                    "reason": secret_reason,
                },
            )
            raise WorkspaceStateSecretError(secret_reason)
        confidence = _optional_confidence_score(confidence_score)

        prompt_ts = self.poster.post_message(
            _thread_from_task(task),
            _confirmation_prompt_text(
                scope_type=scope_type,
                scope_id=scope_id,
                key=normalized_key,
                value_text=readable_value,
            ),
            purpose="memory_confirmation",
        )
        event = self.task_service.append_event(
            task,
            TaskEventType.log,
            {
                "message": PENDING_PROPOSAL_MESSAGE,
                "status": "pending",
                "installation_id": str(installation_id),
                "scope_type": scope_type,
                "scope_id": scope_id,
                "key": normalized_key,
                "value_json": value_json,
                "value_text": readable_value,
                "source_kind": source_kind,
                "source_task_id": str(source_task_id),
                "source_slack_channel_id": task.slack_channel_id,
                "source_slack_message_ts": task.slack_message_ts,
                "prompt_channel_id": task.slack_channel_id,
                "prompt_message_ts": prompt_ts,
                "proposed_by": task.slack_user_id,
                "proposed_reason": proposed_reason,
                "confidence_score": str(confidence) if confidence is not None else None,
                "confidence_reason": confidence_reason,
            },
        )
        return _pending_from_event(event)

    def confirm(
        self,
        prompt_message_ts: str,
        confirming_user_id: str,
        *,
        channel_id: str | None = None,
    ) -> Fact:
        """Materialize the pending fact tied to a confirmation prompt."""

        event = self._pending_proposal_event(
            prompt_message_ts,
            channel_id=channel_id,
            for_update=True,
        )
        pending = _pending_from_event(event)
        now = datetime.now(UTC)

        existing = list(
            self.session.scalars(
                select(WorkspaceState)
                .where(
                    WorkspaceState.installation_id == pending.installation_id,
                    WorkspaceState.scope_type == pending.scope_type,
                    _scope_id_clause(pending.scope_id),
                    WorkspaceState.key == pending.key,
                    WorkspaceState.status == "active",
                )
                .order_by(WorkspaceState.created_at)
                .with_for_update()
            )
        )

        for previous in existing:
            previous.status = "superseded"
            previous.superseded_at = now
            previous.updated_at = now
        if existing:
            self.session.flush()

        state = WorkspaceState(
            installation_id=pending.installation_id,
            scope_type=pending.scope_type,
            scope_id=pending.scope_id,
            key=pending.key,
            value_json=pending.value,
            value_text=pending.value_text,
            status="active",
            source_kind=_payload_str(event.payload, "source_kind"),
            source_task_id=pending.task_id,
            source_event_id=event.id,
            source_slack_channel_id=_payload_optional_str(
                event.payload, "source_slack_channel_id"
            ),
            source_slack_message_ts=_payload_optional_str(
                event.payload, "source_slack_message_ts"
            ),
            proposed_by=pending.proposed_by,
            proposed_reason=_payload_optional_str(event.payload, "proposed_reason"),
            confidence_score=_optional_confidence_score(
                event.payload.get("confidence_score")
            ),
            confidence_reason=_payload_optional_str(event.payload, "confidence_reason"),
            confirmed_by_user_id=confirming_user_id,
            confirmed_at=now,
        )
        self.session.add(state)
        self.session.flush()

        for previous in existing:
            previous.superseded_by_id = state.id

        # Embed-on-write (failure-isolated inside ensure; no-op without index).
        if self.embedding_index is not None:
            self.embedding_index.ensure(
                FACT_EMBEDDING_KIND,
                [(str(state.id), fact_embedding_text(state))],
            )

        event.payload = {
            **event.payload,
            "status": "confirmed",
            "confirmed_by_user_id": confirming_user_id,
            "confirmed_at": now.isoformat(),
            "workspace_state_id": str(state.id),
        }
        event.task_id = pending.task_id
        self.session.flush()
        self.task_service.append_event(
            pending.task_id,
            TaskEventType.log,
            {
                "message": CONFIRMED_PROPOSAL_MESSAGE,
                "prompt_message_ts": prompt_message_ts,
                "workspace_state_id": str(state.id),
                "confirmed_by_user_id": confirming_user_id,
            },
        )
        return _fact_from_state(state)

    def set_active(
        self,
        installation_id: uuid.UUID,
        scope_type: str,
        scope_id: str | None,
        key: str,
        value: Mapping[str, Any],
        source_task_id: uuid.UUID,
        *,
        value_text: str | None = None,
        confidence_score: Decimal | float | str | None = None,
        confidence_reason: str | None = None,
        source_kind: str = DEFAULT_SOURCE_KIND,
        activated_by: str = "system",
    ) -> Fact:
        """Write an active fact directly, no Slack prompt (trusted auto-activation).

        For high-confidence inferred facts that the locked policy auto-activates
        (e.g. a persona at confidence >= 0.85, HIG-277). Mirrors ``confirm``'s
        activation write — supersede prior actives, insert active, embed — but
        skips the propose→confirm round-trip. Graph projection still happens via
        the consolidator's confirmed-facts pass, which scans active rows.
        """

        _validate_scope(scope_type, scope_id)
        source_kind = _validate_source_kind(source_kind)
        normalized_key = _normalize_key(key)
        value_json = _json_object(value)
        readable_value = _faithful_value_text(
            value_text or _value_text(value_json), value_json
        )
        now = datetime.now(UTC)

        existing = list(
            self.session.scalars(
                select(WorkspaceState)
                .where(
                    WorkspaceState.installation_id == installation_id,
                    WorkspaceState.scope_type == scope_type,
                    _scope_id_clause(scope_id),
                    WorkspaceState.key == normalized_key,
                    WorkspaceState.status == "active",
                )
                .order_by(WorkspaceState.created_at)
                .with_for_update()
            )
        )
        for previous in existing:
            previous.status = "superseded"
            previous.superseded_at = now
            previous.updated_at = now
        if existing:
            self.session.flush()

        state = WorkspaceState(
            installation_id=installation_id,
            scope_type=scope_type,
            scope_id=scope_id,
            key=normalized_key,
            value_json=value_json,
            value_text=readable_value,
            status="active",
            source_kind=source_kind,
            source_task_id=source_task_id,
            proposed_by=activated_by,
            confidence_score=_optional_confidence_score(confidence_score),
            confidence_reason=confidence_reason,
            confirmed_by_user_id=activated_by,
            confirmed_at=now,
        )
        self.session.add(state)
        self.session.flush()
        for previous in existing:
            previous.superseded_by_id = state.id

        if self.embedding_index is not None:
            self.embedding_index.ensure(
                FACT_EMBEDDING_KIND,
                [(str(state.id), fact_embedding_text(state))],
            )

        self.task_service.append_event(
            source_task_id,
            TaskEventType.log,
            {
                "message": AUTO_ACTIVATED_MESSAGE,
                "installation_id": str(installation_id),
                "scope_type": scope_type,
                "scope_id": scope_id,
                "key": normalized_key,
                "workspace_state_id": str(state.id),
                "source_kind": source_kind,
            },
        )
        return _fact_from_state(state)

    def reject(
        self,
        prompt_message_ts: str,
        rejecting_user_id: str,
        *,
        channel_id: str | None = None,
    ) -> PendingFact:
        """Reject a pending memory proposal without writing workspace_state."""

        event = self._pending_proposal_event(
            prompt_message_ts,
            channel_id=channel_id,
            for_update=True,
        )
        pending = _pending_from_event(event)
        now = datetime.now(UTC)
        event.payload = {
            **event.payload,
            "status": "rejected",
            "rejected_by_user_id": rejecting_user_id,
            "rejected_at": now.isoformat(),
        }
        self.session.flush()
        self.task_service.append_event(
            pending.task_id,
            TaskEventType.log,
            {
                "message": REJECTED_PROPOSAL_MESSAGE,
                "prompt_message_ts": prompt_message_ts,
                "rejected_by_user_id": rejecting_user_id,
            },
        )
        return _pending_from_event(event)

    def forget(
        self,
        installation_id: uuid.UUID,
        scope_type: str,
        scope_id: str | None,
        key: str,
        by_user_id: str,
        *,
        audit_task_id: uuid.UUID | None = None,
    ) -> int:
        """Forget current active facts for a key/scope with no replacement."""

        _validate_scope(scope_type, scope_id)
        now = datetime.now(UTC)
        rows = list(
            self.session.scalars(
                select(WorkspaceState)
                .where(
                    WorkspaceState.installation_id == installation_id,
                    WorkspaceState.scope_type == scope_type,
                    _scope_id_clause(scope_id),
                    WorkspaceState.key == _normalize_key(key),
                    WorkspaceState.status == "active",
                )
                .with_for_update()
            )
        )
        for row in rows:
            row.status = "forgotten"
            row.forgotten_by_user_id = by_user_id
            row.forgotten_at = now
            row.updated_at = now
        self.session.flush()
        for row in rows:
            if row.source_task_id is not None:
                self.task_service.append_event(
                    row.source_task_id,
                    TaskEventType.log,
                    {
                        "message": FORGOTTEN_FACT_MESSAGE,
                        "workspace_state_id": str(row.id),
                        "forgotten_by_user_id": by_user_id,
                    },
                )
        if audit_task_id is not None:
            self.task_service.append_event(
                audit_task_id,
                TaskEventType.log,
                {
                    "message": FORGET_REQUEST_MESSAGE,
                    "scope_type": scope_type,
                    "scope_id": scope_id,
                    "key": _normalize_key(key),
                    "forgotten_by_user_id": by_user_id,
                    "forgotten_count": len(rows),
                    "workspace_state_ids": [str(row.id) for row in rows],
                },
            )
        return len(rows)

    def record_inspection(
        self,
        source_task_id: uuid.UUID,
        *,
        scope_type: str,
        scope_id: str | None,
        key: str | None,
        include_history: bool,
        count: int,
    ) -> None:
        """Append an audit event for a memory inspection request."""

        _validate_scope(scope_type, scope_id)
        self.task_service.append_event(
            source_task_id,
            TaskEventType.log,
            {
                "message": INSPECTED_MEMORY_MESSAGE,
                "scope_type": scope_type,
                "scope_id": scope_id,
                "key": _normalize_key(key) if key is not None else None,
                "include_history": include_history,
                "count": count,
            },
        )

    def _current_statement(
        self, installation_id: uuid.UUID
    ) -> Select[tuple[WorkspaceState]]:
        now = datetime.now(UTC)
        return select(WorkspaceState).where(
            WorkspaceState.installation_id == installation_id,
            WorkspaceState.status == "active",
            or_(WorkspaceState.expires_at.is_(None), WorkspaceState.expires_at > now),
        )

    def _pending_proposal_event(
        self,
        prompt_message_ts: str,
        *,
        channel_id: str | None,
        for_update: bool = False,
    ) -> TaskEvent:
        statement = (
            select(TaskEvent)
            .where(
                TaskEvent.type == TaskEventType.log,
                TaskEvent.payload["message"].as_string() == PENDING_PROPOSAL_MESSAGE,
                TaskEvent.payload["status"].as_string() == "pending",
                TaskEvent.payload["prompt_message_ts"].as_string() == prompt_message_ts,
            )
            .order_by(TaskEvent.created_at.desc(), TaskEvent.id.desc())
            .limit(1)
        )
        if channel_id is not None:
            statement = statement.where(
                TaskEvent.payload["prompt_channel_id"].as_string() == channel_id
            )
        if for_update:
            statement = statement.with_for_update()

        event = self.session.scalar(statement)
        if event is None:
            raise LookupError(
                f"No pending workspace_state proposal for prompt {prompt_message_ts}"
            )
        return event


def _fact_from_state(state: WorkspaceState) -> Fact:
    return Fact(
        id=state.id,
        installation_id=state.installation_id,
        scope_type=state.scope_type,
        scope_id=state.scope_id,
        key=state.key,
        value=dict(state.value_json),
        value_text=state.value_text,
        status=state.status,
        source_kind=state.source_kind,
        source_task_id=state.source_task_id,
        source_event_id=state.source_event_id,
        confirmed_by_user_id=state.confirmed_by_user_id,
        confirmed_at=state.confirmed_at,
        source_slack_channel_id=state.source_slack_channel_id,
        source_slack_message_ts=state.source_slack_message_ts,
        proposed_by=state.proposed_by,
        proposed_reason=state.proposed_reason,
        confidence_reason=state.confidence_reason,
        created_at=state.created_at,
        updated_at=state.updated_at,
    )


def _pending_from_event(event: TaskEvent) -> PendingFact:
    payload = event.payload
    return PendingFact(
        event_id=event.id,
        task_id=uuid.UUID(_payload_str(payload, "source_task_id")),
        installation_id=uuid.UUID(_payload_str(payload, "installation_id")),
        scope_type=_payload_str(payload, "scope_type"),
        scope_id=_payload_optional_str(payload, "scope_id"),
        key=_payload_str(payload, "key"),
        value=_json_object(_payload_mapping(payload, "value_json")),
        value_text=_payload_optional_str(payload, "value_text"),
        prompt_channel_id=_payload_str(payload, "prompt_channel_id"),
        prompt_message_ts=_payload_str(payload, "prompt_message_ts"),
        proposed_by=_payload_str(payload, "proposed_by"),
        status=_payload_str(payload, "status"),
    )


def _validate_scope(scope_type: str, scope_id: str | None) -> None:
    if scope_type not in SCOPE_TYPES:
        raise ValueError(f"Unsupported scope_type: {scope_type}")
    if scope_type == "workspace" and scope_id is not None:
        raise ValueError("workspace scope requires scope_id=None")
    if scope_type in {"channel", "user"} and not scope_id:
        raise ValueError(f"{scope_type} scope requires scope_id")


def _validate_source_kind(source_kind: str) -> str:
    if source_kind not in SOURCE_KINDS:
        raise ValueError(f"Unsupported source_kind: {source_kind}")
    return source_kind


def _normalize_key(key: str) -> str:
    normalized = key.strip().lower().replace(" ", "_")
    if not normalized:
        raise ValueError("key is required")
    return normalized


def _scope_id_clause(scope_id: str | None) -> ColumnElement[bool]:
    if scope_id is None:
        return WorkspaceState.scope_id.is_(None)
    return WorkspaceState.scope_id == scope_id


def _json_object(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("value must be a JSON object")
    result = dict(value)
    json.dumps(result)
    return result


def _value_text(value: Mapping[str, Any]) -> str:
    if "text" in value and isinstance(value["text"], str):
        return value["text"]
    if "value" in value and isinstance(value["value"], str):
        return value["value"]
    return json.dumps(value, sort_keys=True)


def _faithful_value_text(value_text: str, value: Mapping[str, Any]) -> str:
    text = value_text.strip()
    if not text:
        text = _value_text(value)

    missing_phrases: list[str] = []
    seen_values: set[str] = set()
    for phrase, search_value in _memory_detail_phrases(value):
        normalized_search = search_value.casefold()
        if normalized_search in seen_values:
            continue
        seen_values.add(normalized_search)
        if normalized_search in text.casefold():
            continue
        missing_phrases.append(phrase)
        if len(missing_phrases) >= MAX_APPENDED_MEMORY_DETAILS:
            break

    if not missing_phrases:
        return text
    return f"{text}; {'; '.join(missing_phrases)}"


def _secret_candidate_reason(
    *,
    key: str,
    value: Mapping[str, Any],
    value_text: str | None,
) -> str | None:
    if SECRET_KEY_RE.search(key):
        return f"memory key {key!r} looks like a secret field"

    for path, text in _string_values(value):
        path_text = ".".join(path)
        if path_text and SECRET_KEY_RE.search(path_text):
            return f"value field {path_text!r} looks like a secret field"
        if SECRET_VALUE_RE.search(text) or SECRET_ASSIGNMENT_RE.search(text):
            label = path_text or "value"
            return f"{label!r} looks like it contains a secret"

    if value_text and (
        SECRET_VALUE_RE.search(value_text) or SECRET_ASSIGNMENT_RE.search(value_text)
    ):
        return "value_text looks like it contains a secret"

    return None


def _string_values(
    value: Any,
    path: tuple[str, ...] = (),
) -> list[tuple[tuple[str, ...], str]]:
    if isinstance(value, Mapping):
        values: list[tuple[tuple[str, ...], str]] = []
        for key, child in value.items():
            values.extend(_string_values(child, path + (str(key),)))
        return values
    if isinstance(value, list):
        values = []
        for index, child in enumerate(value):
            values.extend(_string_values(child, path + (str(index),)))
        return values
    if isinstance(value, str):
        text = value.strip()
        return [(path, text)] if text else []
    return []


def _memory_detail_phrases(
    value: Any,
    path: tuple[str, ...] = (),
) -> list[tuple[str, str]]:
    if isinstance(value, Mapping):
        phrases: list[tuple[str, str]] = []
        for key, child in value.items():
            phrases.extend(_memory_detail_phrases(child, path + (str(key),)))
        return phrases
    if isinstance(value, list):
        phrases = []
        for child in value:
            phrases.extend(_memory_detail_phrases(child, path))
        return phrases
    if isinstance(value, str):
        search_value = value.strip()
        if len(search_value) < 3:
            return []
        label = _memory_detail_label(path)
        phrase = search_value if label is None else f"{label}: {search_value}"
        return [(phrase, search_value)]
    return []


def _memory_detail_label(path: tuple[str, ...]) -> str | None:
    meaningful = [
        _humanize_memory_key(part)
        for part in path
        if part and part.casefold() not in GENERIC_MEMORY_DETAIL_KEYS
    ]
    if not meaningful:
        return None
    return " ".join(meaningful)


def _humanize_memory_key(value: str) -> str:
    return value.replace("_", " ").replace("-", " ").strip()


def _confirmation_prompt_text(
    *,
    scope_type: str,
    scope_id: str | None,
    key: str,
    value_text: str,
) -> str:
    del key, scope_id
    scope_label = _scope_label(scope_type)
    return (
        f"Should I remember this {scope_label}?\n"
        f"{value_text}\n\n"
        "React with :white_check_mark: to save it or :no_entry_sign: to skip."
    )


def _scope_label(scope_type: str) -> str:
    if scope_type == "workspace":
        return "for this workspace"
    if scope_type == "channel":
        return "for this channel"
    if scope_type == "user":
        return "for you"
    return "here"


def _thread_from_task(task: Task) -> MemoryPromptThread:
    thread_ts = task.slack_thread_ts or task.slack_message_ts
    if not thread_ts:
        raise ValueError("Task has no Slack thread timestamp")
    payload = task.identity_payload if isinstance(task.identity_payload, dict) else {}
    return MemoryPromptThread(
        channel_id=task.slack_channel_id,
        thread_ts=thread_ts,
        task_id=task.id,
        is_assistant=payload.get("source_surface") == "assistant",
    )


def _optional_confidence_score(
    value: Decimal | float | str | None,
) -> Decimal | None:
    if value is None:
        return None
    score = value if isinstance(value, Decimal) else Decimal(str(value))
    if score < 0 or score > 1:
        raise ValueError("confidence_score must be between 0 and 1")
    return score


def _payload_str(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise WorkspaceStateServiceError(f"Pending proposal missing {key!r}")
    return value


def _payload_optional_str(payload: Mapping[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise WorkspaceStateServiceError(f"Pending proposal has invalid {key!r}")
    return value


def _payload_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise WorkspaceStateServiceError(f"Pending proposal missing {key!r}")
    return value
