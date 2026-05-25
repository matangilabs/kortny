"""Write actions for dashboard memory governance."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from kortny.db.models import TaskEventType, WorkspaceState
from kortny.memory import WorkspaceStateService
from kortny.tasks import TaskService

DASHBOARD_USER_PREFIX = "dashboard:"
SUPERSEDED_FACT_MESSAGE = "workspace_state_dashboard_fact_superseded"


def dashboard_actor(username: str) -> str:
    """Return the stable audit actor string for a dashboard operator."""

    return f"{DASHBOARD_USER_PREFIX}{username}"


def forget_fact(
    session: Session,
    fact_id: uuid.UUID,
    *,
    by_user_id: str,
) -> WorkspaceState:
    """Mark one active fact as forgotten while preserving audit history."""

    fact = _active_fact_for_update(session, fact_id)
    service = WorkspaceStateService(session)
    service.forget(
        fact.installation_id,
        fact.scope_type,
        fact.scope_id,
        fact.key,
        by_user_id,
    )
    session.refresh(fact)
    return fact


def supersede_fact(
    session: Session,
    fact_id: uuid.UUID,
    *,
    value_text: str,
    by_user_id: str,
) -> WorkspaceState:
    """Replace one active fact with a dashboard-confirmed successor."""

    replacement_text = " ".join(value_text.split())
    if not replacement_text:
        raise ValueError("Replacement memory value is required.")

    current = _active_fact_for_update(session, fact_id)
    now = datetime.now(UTC)
    current.status = "superseded"
    current.superseded_at = now
    current.updated_at = now
    session.flush()

    replacement = WorkspaceState(
        installation_id=current.installation_id,
        scope_type=current.scope_type,
        scope_id=current.scope_id,
        key=current.key,
        value_json={"text": replacement_text},
        value_text=replacement_text,
        status="active",
        source_kind=current.source_kind,
        source_task_id=current.source_task_id,
        source_event_id=None,
        source_slack_channel_id=current.source_slack_channel_id,
        source_slack_message_ts=current.source_slack_message_ts,
        source_slack_file_id=current.source_slack_file_id,
        source_url=current.source_url,
        proposed_by=by_user_id,
        proposed_reason="Dashboard memory supersede action.",
        confidence_score=current.confidence_score,
        confidence_reason=current.confidence_reason,
        confirmed_by_user_id=by_user_id,
        confirmed_at=now,
    )
    session.add(replacement)
    session.flush()

    current.superseded_by_id = replacement.id
    session.flush()

    if current.source_task_id is not None:
        event = TaskService(session).append_event(
            current.source_task_id,
            TaskEventType.log,
            {
                "message": SUPERSEDED_FACT_MESSAGE,
                "workspace_state_id": str(current.id),
                "replacement_workspace_state_id": str(replacement.id),
                "superseded_by_user_id": by_user_id,
            },
        )
        replacement.source_event_id = event.id
        session.flush()
    return replacement


def _active_fact_for_update(session: Session, fact_id: uuid.UUID) -> WorkspaceState:
    fact = session.scalar(
        select(WorkspaceState)
        .where(WorkspaceState.id == fact_id)
        .limit(1)
        .with_for_update()
    )
    if fact is None:
        raise LookupError(f"Memory fact not found: {fact_id}")
    if fact.status != "active":
        raise ValueError("Only active memory facts can be changed.")
    return fact
