"""Runtime selection of scoped Composio connections."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from kortny.db.models import ComposioConnection, Task

SCOPE_PRIORITY = {
    "user": 0,
    "channel": 1,
    "workspace": 2,
}


@dataclass(frozen=True, slots=True)
class RuntimeComposioConnection:
    """A Composio connected account that is allowed for one Slack task."""

    toolkit_slug: str
    connected_account_id: str
    composio_user_id: str
    visibility_scope_type: str
    visibility_scope_id: str | None
    display_name: str | None


class ComposioConnectionResolver:
    """Resolve active Composio accounts for a Slack task context."""

    def __init__(self, session: Session, task: Task) -> None:
        self.session = session
        self.task = task

    def allowed_connections(
        self,
        *,
        toolkit_slug: str | None = None,
    ) -> tuple[RuntimeComposioConnection, ...]:
        clauses = [
            ComposioConnection.installation_id == self.task.installation_id,
            ComposioConnection.status == "active",
            ComposioConnection.connected_account_id.is_not(None),
        ]
        if toolkit_slug:
            clauses.append(ComposioConnection.toolkit_slug == toolkit_slug.lower())

        rows = tuple(
            self.session.scalars(
                select(ComposioConnection)
                .where(*clauses)
                .order_by(
                    ComposioConnection.updated_at.desc(),
                    ComposioConnection.id.desc(),
                )
            )
        )
        allowed = [row for row in rows if self._allows_task(row)]
        allowed.sort(
            key=lambda row: (
                SCOPE_PRIORITY.get(row.visibility_scope_type, 99),
                row.toolkit_slug,
            )
        )
        return tuple(_runtime_connection(row) for row in allowed)

    def best_connection(
        self,
        *,
        toolkit_slug: str,
    ) -> RuntimeComposioConnection | None:
        return next(
            iter(self.allowed_connections(toolkit_slug=toolkit_slug.lower())),
            None,
        )

    def has_allowed_connection(
        self,
        *,
        toolkit_slugs: tuple[str, ...],
    ) -> bool:
        normalized = {slug.lower() for slug in toolkit_slugs}
        return any(
            connection.toolkit_slug in normalized
            for connection in self.allowed_connections()
        )

    def _allows_task(self, row: ComposioConnection) -> bool:
        if row.visibility_scope_type == "workspace":
            return row.visibility_scope_id is None
        if row.visibility_scope_type == "channel":
            return row.visibility_scope_id == self.task.slack_channel_id
        if row.visibility_scope_type == "user":
            return row.visibility_scope_id == self.task.slack_user_id
        return False


def connected_toolkit_slugs(session: Session, task: Task) -> tuple[str, ...]:
    """Deterministic set of active Composio toolkit slugs allowed for this task.

    This is the capability-grounding primitive (HIG-274): a DB-derived fact that
    does not depend on tool selection or the external-tool skip path, so routing,
    selection, and agent context can all be told what is actually connected
    regardless of how the request was classified.
    """

    connections = ComposioConnectionResolver(session, task).allowed_connections()
    return tuple(dict.fromkeys(connection.toolkit_slug for connection in connections))


def _runtime_connection(row: ComposioConnection) -> RuntimeComposioConnection:
    if row.connected_account_id is None:
        raise ValueError("Composio connection is missing connected_account_id")
    return RuntimeComposioConnection(
        toolkit_slug=row.toolkit_slug,
        connected_account_id=row.connected_account_id,
        composio_user_id=row.composio_user_id,
        visibility_scope_type=row.visibility_scope_type,
        visibility_scope_id=row.visibility_scope_id,
        display_name=row.display_name,
    )
