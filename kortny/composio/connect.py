"""In-thread Composio OAuth connect + auto-resume (HIG-209 Part 3).

When a Composio tool runs and no connected account exists in scope, the
coordinator surfaces an ``auth_connection`` / ``wait_auth`` failure. Instead of
letting the task fail, the worker parks it on the EXISTING ``waiting_approval``
status (the task-status enum is test-locked, so we never add a value) and posts
a threaded connect prompt with a Composio OAuth initiate link for the toolkit.

A ``wait_auth`` ``TaskEvent`` marker distinguishes a connect-parked task from a
tool-approval-parked one. The ambient ``composio_catalog_sync`` tick scans for
these parked tasks and, once the toolkit shows a connected account in the task's
scope, requeues the task through the existing retry/requeue path and syncs that
toolkit's catalog so the tool resolves on re-run.
"""

from __future__ import annotations

import logging
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from kortny.composio.client import (
    ComposioClient,
)
from kortny.composio.identity import capture_external_identity
from kortny.composio.runtime import ComposioConnectionResolver
from kortny.db.models import ComposioConnection, Task, TaskEventType
from kortny.db.models import TaskStatus as DbTaskStatus
from kortny.tasks import TaskService

logger = logging.getLogger(__name__)

# TaskEvent message + log markers so the resume scan can find connect-parked
# tasks without a new status enum value.
COMPOSIO_CONNECT_WAITING_MESSAGE = "composio_connect_wait_auth"
COMPOSIO_CONNECT_RESUMED_MESSAGE = "composio_connect_resumed"
WAIT_AUTH_RECOVERY_ACTION = "wait_auth"

# Default scope for an in-thread connect: the requesting Slack user.
DEFAULT_CONNECT_SCOPE_TYPE = "user"


class ComposioConnectionRequired(RuntimeError):
    """Raised when a Composio tool needs an account connected first.

    Carries the toolkit slug so the worker can post a connect prompt and park
    the task. Analogous to ``ToolApprovalRequired`` but for the OAuth-connect
    flow rather than a human yes/no approval.
    """

    def __init__(self, *, toolkit_slug: str, tool_name: str) -> None:
        super().__init__(
            f"Composio toolkit {toolkit_slug!r} needs a connected account "
            f"for {tool_name!r}."
        )
        self.toolkit_slug = toolkit_slug
        self.tool_name = tool_name


@dataclass(frozen=True, slots=True)
class ConnectPrompt:
    """A prepared connect prompt: the OAuth redirect link + the pending row."""

    toolkit_slug: str
    redirect_url: str
    connection_id: object
    scope_type: str
    scope_id: str | None


def composio_user_id_for_task(task: Task) -> str:
    """Stable Composio ``user_id`` for a task's connecting Slack user."""

    return f"slack:{task.installation_id}:{task.slack_user_id or 'unknown'}"


def initiate_connect_for_task(
    session: Session,
    *,
    task: Task,
    toolkit_slug: str,
    client: ComposioClient,
    callback_url: str,
    auth_config_id: str | None = None,
    scope_type: str = DEFAULT_CONNECT_SCOPE_TYPE,
) -> ConnectPrompt:
    """Create a managed auth config + connect link and a pending connection row.

    Mirrors the dashboard's connect flow but is driven by a task context: the
    connection is scoped to the task's Slack user by default so the very user
    who asked for the integration is the one whose account resolves it.
    """

    normalized_slug = toolkit_slug.strip().lower()
    composio_user_id = composio_user_id_for_task(task)
    scope_id = _scope_id_for(task, scope_type)

    resolved_auth_config_id = auth_config_id
    if resolved_auth_config_id is None:
        auth_config = client.create_managed_auth_config(toolkit_slug=normalized_slug)
        resolved_auth_config_id = auth_config.id

    callback_token = secrets.token_urlsafe(24)
    connection = ComposioConnection(
        installation_id=task.installation_id,
        toolkit_slug=normalized_slug,
        auth_config_id=resolved_auth_config_id,
        composio_user_id=composio_user_id,
        owner_slack_user_id=task.slack_user_id or "unknown",
        visibility_scope_type=scope_type,
        visibility_scope_id=scope_id,
        status="pending",
        metadata_json={
            "source": "in_thread_connect",
            "task_id": str(task.id),
            "callback_token": callback_token,
        },
    )
    session.add(connection)
    session.flush()

    full_callback_url = (
        f"{callback_url}?connection_id={connection.id}"
        f"&connection_token={callback_token}"
    )
    connect_request = client.create_connect_link(
        user_id=composio_user_id,
        auth_config_id=resolved_auth_config_id,
        callback_url=full_callback_url,
    )
    connection.connection_request_id = connect_request.id
    if connect_request.connected_account_id:
        connection.connected_account_id = connect_request.connected_account_id
        # Best-effort, non-blocking: only fires here for auth schemes that
        # hand back an immediate connected_account_id (e.g. no_auth-shaped
        # flows). The common OAuth case only knows the account id once the
        # provider redirects back through the connect callback.
        capture_external_identity(client, connection)
    connection.metadata_json = {
        **dict(connection.metadata_json or {}),
        "connect_link_status": connect_request.status,
        "redirect_url": connect_request.redirect_url,
    }
    session.flush()
    return ConnectPrompt(
        toolkit_slug=normalized_slug,
        redirect_url=connect_request.redirect_url,
        connection_id=connection.id,
        scope_type=scope_type,
        scope_id=scope_id,
    )


def _scope_id_for(task: Task, scope_type: str) -> str | None:
    if scope_type == "user":
        return task.slack_user_id
    if scope_type == "channel":
        return task.slack_channel_id
    return None


def connect_prompt_text(*, toolkit_slug: str, redirect_url: str) -> str:
    """Coworker-voiced connect prompt mirroring the approval-prompt UX."""

    return (
        f"I need access to your *{toolkit_slug}* account before I can run that. "
        f"Connect it here and I'll pick the task back up automatically:\n"
        f"{redirect_url}"
    )


def park_payload(
    *,
    toolkit_slug: str,
    tool_name: str,
    connection_id: object,
    scope_type: str,
    scope_id: str | None,
    prompt_message_ts: str | None,
) -> dict[str, object]:
    """Build the ``wait_auth`` marker payload stored on the parked task.

    Reused by ``mark_waiting_for_tool_approval``'s request payload so the parked
    task carries everything the resume scan needs (toolkit + scope) without a
    new status enum value.
    """

    return {
        "recovery_action": WAIT_AUTH_RECOVERY_ACTION,
        "toolkit_slug": toolkit_slug,
        "tool": tool_name,
        "approval_key": f"composio-connect:{toolkit_slug}",
        "connection_id": str(connection_id),
        "scope_type": scope_type,
        "scope_id": scope_id,
        "prompt_message_ts": prompt_message_ts,
    }


@dataclass(frozen=True, slots=True)
class ConnectResumeResult:
    """Summary of one connect-resume scan."""

    requeued_task_ids: tuple[object, ...]
    resumed_toolkits: tuple[str, ...]


def resume_parked_connect_tasks(
    session: Session,
    *,
    installation_id: object,
    task_service: TaskService | None = None,
    now: datetime | None = None,
) -> ConnectResumeResult:
    """Requeue connect-parked tasks whose toolkit is now connected in scope.

    A cheap query for ``waiting_approval`` tasks carrying a ``wait_auth`` marker;
    for each, if the toolkit now resolves a connected account in the task's
    scope, the task is requeued through the existing approval-resume path. The
    caller (the composio_catalog_sync tick) then syncs that toolkit's catalog.
    """

    service = task_service or TaskService(session)
    parked = _parked_connect_tasks(session, installation_id=installation_id)
    requeued: list[object] = []
    toolkits: set[str] = set()
    for task, marker in parked:
        toolkit_slug = str(marker.get("toolkit_slug") or "").strip().lower()
        if not toolkit_slug:
            continue
        resolver = ComposioConnectionResolver(session, task)
        if resolver.best_connection(toolkit_slug=toolkit_slug) is None:
            continue
        resumed = service.approve_tool_approval(
            task,
            approval_key=str(marker.get("approval_key") or ""),
            by_user_id="composio-connect-resume",
            available_at=now or datetime.now(UTC),
        )
        if resumed is None:
            continue
        service.append_event(
            resumed,
            TaskEventType.log,
            {
                "message": COMPOSIO_CONNECT_RESUMED_MESSAGE,
                "toolkit_slug": toolkit_slug,
            },
        )
        requeued.append(resumed.id)
        toolkits.add(toolkit_slug)
        logger.info(
            "composio connect resume requeued task_id=%s toolkit=%s",
            resumed.id,
            toolkit_slug,
        )
    return ConnectResumeResult(
        requeued_task_ids=tuple(requeued),
        resumed_toolkits=tuple(sorted(toolkits)),
    )


def _parked_connect_tasks(
    session: Session,
    *,
    installation_id: object,
) -> list[tuple[Task, dict[str, object]]]:
    """Find waiting_approval tasks carrying a composio connect wait_auth marker."""

    tasks = session.scalars(
        select(Task).where(
            Task.installation_id == installation_id,
            Task.status == DbTaskStatus.waiting_approval,
        )
    )
    parked: list[tuple[Task, dict[str, object]]] = []
    for task in tasks:
        marker = _latest_connect_marker(session, task)
        if marker is not None:
            parked.append((task, marker))
    return parked


def _latest_connect_marker(
    session: Session,
    task: Task,
) -> dict[str, object] | None:
    """Return the most recent connect wait_auth request payload for a task."""

    pending = TaskService(session).latest_pending_tool_approval(task)
    if not isinstance(pending, dict):
        return None
    if pending.get("recovery_action") != WAIT_AUTH_RECOVERY_ACTION:
        return None
    if not pending.get("toolkit_slug"):
        return None
    return pending
