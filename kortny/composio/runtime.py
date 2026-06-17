"""Runtime selection of scoped Composio connections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session

from kortny.db.models import ComposioConnection, Task

SCOPE_PRIORITY = {
    "user": 0,
    "channel": 1,
    "workspace": 2,
}


class ConnectionScope(Protocol):
    """The minimal Slack context needed to resolve scoped connections.

    A ``Task`` satisfies this, but so does a pre-task ingress context: the soft
    channel-mention path classifies intent *before* a Task row exists, yet still
    needs the same capability grounding. Depending only on these three fields
    keeps grounding available at every surface, persisted task or not.
    """

    @property
    def installation_id(self) -> object: ...

    @property
    def slack_channel_id(self) -> str | None: ...

    @property
    def slack_user_id(self) -> str | None: ...


@dataclass(frozen=True, slots=True)
class IngressConnectionScope:
    """A pre-task connection scope built straight from a Slack event."""

    installation_id: object
    slack_channel_id: str | None
    slack_user_id: str | None


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

    def __init__(self, session: Session, scope: ConnectionScope) -> None:
        self.session = session
        self.scope = scope

    def allowed_connections(
        self,
        *,
        toolkit_slug: str | None = None,
    ) -> tuple[RuntimeComposioConnection, ...]:
        clauses = [
            ComposioConnection.installation_id == self.scope.installation_id,
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
            return row.visibility_scope_id == self.scope.slack_channel_id
        if row.visibility_scope_type == "user":
            return row.visibility_scope_id == self.scope.slack_user_id
        return False


def connected_toolkit_slugs_for_scope(
    session: Session, scope: ConnectionScope
) -> tuple[str, ...]:
    """Deterministic active Composio toolkit slugs allowed for a Slack scope.

    This is the capability-grounding primitive (HIG-274): a DB-derived fact that
    does not depend on tool selection or the external-tool skip path, so routing,
    selection, and agent context can all be told what is actually connected
    regardless of how the request was classified. Taking a ``ConnectionScope``
    rather than a ``Task`` lets every surface ground identically — including the
    soft channel-mention path, which classifies before any Task row exists.
    """

    connections = ComposioConnectionResolver(session, scope).allowed_connections()
    return tuple(dict.fromkeys(connection.toolkit_slug for connection in connections))


def connected_toolkit_slugs(session: Session, task: Task) -> tuple[str, ...]:
    """Connected toolkit slugs for a persisted task (see scope variant)."""

    return connected_toolkit_slugs_for_scope(session, task)


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
