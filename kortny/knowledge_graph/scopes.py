"""Deterministic visibility rules for workspace graph retrieval."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy import and_, false, or_
from sqlalchemy.sql.elements import ColumnElement

SCOPE_WORKSPACE = "workspace"
SCOPE_CHANNEL = "channel"
SCOPE_PRIVATE_CHANNEL = "private_channel"
SCOPE_DM = "dm"
SCOPE_USER = "user"

VALID_SCOPE_TYPES = frozenset(
    {
        SCOPE_WORKSPACE,
        SCOPE_CHANNEL,
        SCOPE_PRIVATE_CHANNEL,
        SCOPE_DM,
        SCOPE_USER,
    }
)

SURFACE_CHANNEL = "channel"
SURFACE_PRIVATE_CHANNEL = "private_channel"
SURFACE_DM = "dm"
SURFACE_USER = "user"

VALID_DESTINATION_SURFACES = frozenset(
    {
        SURFACE_CHANNEL,
        SURFACE_PRIVATE_CHANNEL,
        SURFACE_DM,
        SURFACE_USER,
    }
)


@dataclass(frozen=True)
class VisibilityScope:
    scope_type: str
    scope_id: str | None = None

    def __post_init__(self) -> None:
        if self.scope_type not in VALID_SCOPE_TYPES:
            raise ValueError(f"Unsupported graph visibility scope: {self.scope_type}")
        if self.scope_type == SCOPE_WORKSPACE and self.scope_id is not None:
            raise ValueError("Workspace graph scope must not have a scope_id")
        if self.scope_type != SCOPE_WORKSPACE and not self.scope_id:
            raise ValueError(f"{self.scope_type} graph scope requires a scope_id")

    @classmethod
    def workspace(cls) -> VisibilityScope:
        return cls(SCOPE_WORKSPACE)

    @classmethod
    def channel(cls, channel_id: str) -> VisibilityScope:
        return cls(SCOPE_CHANNEL, channel_id)

    @classmethod
    def private_channel(cls, channel_id: str) -> VisibilityScope:
        return cls(SCOPE_PRIVATE_CHANNEL, channel_id)

    @classmethod
    def dm(cls, dm_key: str) -> VisibilityScope:
        return cls(SCOPE_DM, dm_key)

    @classmethod
    def user(cls, slack_user_id: str) -> VisibilityScope:
        return cls(SCOPE_USER, slack_user_id)


@dataclass(frozen=True)
class DestinationSurface:
    surface_type: str
    surface_id: str | None = None
    user_id: str | None = None

    def __post_init__(self) -> None:
        if self.surface_type not in VALID_DESTINATION_SURFACES:
            raise ValueError(
                f"Unsupported graph destination surface: {self.surface_type}"
            )
        if self.surface_type != SURFACE_USER and not self.surface_id:
            raise ValueError(f"{self.surface_type} destination requires a surface_id")
        if self.surface_type in {SURFACE_DM, SURFACE_USER} and not self.user_id:
            raise ValueError(f"{self.surface_type} destination requires a user_id")

    @classmethod
    def channel(cls, channel_id: str) -> DestinationSurface:
        return cls(SURFACE_CHANNEL, channel_id)

    @classmethod
    def private_channel(cls, channel_id: str) -> DestinationSurface:
        return cls(SURFACE_PRIVATE_CHANNEL, channel_id)

    @classmethod
    def dm(cls, dm_key: str, *, user_id: str) -> DestinationSurface:
        return cls(SURFACE_DM, dm_key, user_id)

    @classmethod
    def user(cls, user_id: str) -> DestinationSurface:
        return cls(SURFACE_USER, user_id, user_id)


def is_scope_compatible(
    source_scope: VisibilityScope,
    destination: DestinationSurface,
) -> bool:
    """Return whether a graph row can be used in the destination surface."""

    if source_scope.scope_type == SCOPE_WORKSPACE:
        return True
    if source_scope.scope_type == SCOPE_CHANNEL:
        return (
            destination.surface_type == SURFACE_CHANNEL
            and source_scope.scope_id == destination.surface_id
        )
    if source_scope.scope_type == SCOPE_PRIVATE_CHANNEL:
        return (
            destination.surface_type == SURFACE_PRIVATE_CHANNEL
            and source_scope.scope_id == destination.surface_id
        )
    if source_scope.scope_type == SCOPE_DM:
        return (
            destination.surface_type == SURFACE_DM
            and source_scope.scope_id == destination.surface_id
        )
    if source_scope.scope_type == SCOPE_USER:
        return destination.surface_type in {SURFACE_DM, SURFACE_USER} and (
            source_scope.scope_id == destination.user_id
        )
    return False


def compatible_scope_predicate(
    model: Any, destination: DestinationSurface
) -> ColumnElement[bool]:
    """Build the SQL predicate matching `is_scope_compatible`.

    The model must expose `visibility_scope_type` and `visibility_scope_id`.
    """

    clauses = [
        and_(
            model.visibility_scope_type == SCOPE_WORKSPACE,
            model.visibility_scope_id.is_(None),
        )
    ]

    if destination.surface_type == SURFACE_CHANNEL:
        clauses.append(
            and_(
                model.visibility_scope_type == SCOPE_CHANNEL,
                model.visibility_scope_id == destination.surface_id,
            )
        )
    elif destination.surface_type == SURFACE_PRIVATE_CHANNEL:
        clauses.append(
            and_(
                model.visibility_scope_type == SCOPE_PRIVATE_CHANNEL,
                model.visibility_scope_id == destination.surface_id,
            )
        )
    elif destination.surface_type == SURFACE_DM:
        clauses.extend(
            [
                and_(
                    model.visibility_scope_type == SCOPE_DM,
                    model.visibility_scope_id == destination.surface_id,
                ),
                and_(
                    model.visibility_scope_type == SCOPE_USER,
                    model.visibility_scope_id == destination.user_id,
                ),
            ]
        )
    elif destination.surface_type == SURFACE_USER:
        clauses.append(
            and_(
                model.visibility_scope_type == SCOPE_USER,
                model.visibility_scope_id == destination.user_id,
            )
        )
    else:
        return false()

    return or_(*clauses)


def project_scope_predicate(
    model: Any,
    destination: DestinationSurface,
    additional_scopes: Sequence[VisibilityScope],
) -> ColumnElement[bool]:
    """Authorization predicate for project-aware retrieval (HIG-276).

    A project answer must draw from more than one channel, but the delivery
    AUDIENCE still bounds it: a public-channel reply may include the project's
    other PUBLIC member channels, never private siblings — regardless of who is
    asking. ``additional_scopes`` is that audience-safe extra set (the project's
    public member-channel scopes), OR-ed onto the normal destination predicate.
    Private/DM/user scopes in ``additional_scopes`` are ignored here: surfacing
    them needs per-user membership + governance (deferred to a later increment),
    so this function refuses to widen the audience to them.
    """

    clauses: list[ColumnElement[bool]] = [
        compatible_scope_predicate(model, destination)
    ]
    for scope in additional_scopes:
        if scope.scope_type not in (SCOPE_CHANNEL, SCOPE_WORKSPACE):
            # Audience safety: only public channels (and workspace) may widen a
            # project answer. Never private_channel / dm / user.
            continue
        clauses.append(
            and_(
                model.visibility_scope_type == scope.scope_type,
                model.visibility_scope_id == scope.scope_id,
            )
        )
    return or_(*clauses)
