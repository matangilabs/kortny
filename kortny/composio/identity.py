"""External-identity capture for Composio connections.

Captures a best-effort external handle (the connecting user's identity on the
connected app — e.g. a GitHub login) at connect time, so "assigned to
me"-style queries can resolve to a concrete server-side filter (e.g.
``assignee:octocat-x``) instead of guessing. The label lands on
``ComposioConnection.external_account_label`` (nullable; present since
migration 0006).

Non-blocking by design: any fetch or parse failure is logged and leaves the
label unset. Connecting an account must never fail because identity capture
failed.

VERIFIED FINDING (see kortny/composio/client.py:``get_connected_account``):
the Composio connected-account details endpoint
(``GET /api/v3.1/connected_accounts/{id}``) does not expose a dedicated
username/login/email field for the connected app's end-user identity. Its
``state.val`` payload is toolkit/auth-scheme-specific connection-parameter
bookkeeping (OAuth tokens, instance subdomains, region, shop domain, ...),
not profile data. This module scans that payload for a narrow, explicit set
of identity-shaped keys (see ``_USERNAME_KEYS`` / ``_EMAIL_KEYS``) and
returns ``None`` when none are present — it never fabricates a handle. In
practice this means most OAuth-based connections will resolve to ``None``
today; the seam (storage column, capture call, read-side resolver, context
exposure) is what this slice delivers, so a handle appears automatically the
moment a toolkit's connection payload does carry one.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Mapping

from sqlalchemy import select
from sqlalchemy.orm import Session

from kortny.composio.client import (
    ComposioCatalogError,
    ComposioClient,
    ComposioConnectionError,
)
from kortny.db.models import ComposioConnection

logger = logging.getLogger(__name__)

MAX_LABEL_LENGTH = 120

# Best-effort key names that, if present in a connected account's dynamic
# `state.val` payload, plausibly carry the user's identity on the connected
# app. Checked in priority order: explicit username/login first, then
# account email. None of these are guaranteed by the Composio API — see the
# module docstring.
_USERNAME_KEYS: tuple[str, ...] = (
    "username",
    "login",
    "handle",
    "screen_name",
    "user_name",
    "account_name",
)
_EMAIL_KEYS: tuple[str, ...] = ("email", "account_email", "user_email")


def extract_external_handle(state_val: Mapping[str, object]) -> str | None:
    """Best-effort external handle from a connected account's ``state.val``.

    Prefers an explicit username/login-shaped key; falls back to an
    email-shaped key. Returns ``None`` when neither is present rather than
    guessing from unrelated fields.
    """

    for keys in (_USERNAME_KEYS, _EMAIL_KEYS):
        for key in keys:
            value = state_val.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()[:MAX_LABEL_LENGTH]
    return None


def capture_external_identity(
    client: ComposioClient,
    connection: ComposioConnection,
) -> None:
    """Fetch + store a best-effort external handle on ``connection``.

    Non-blocking: any client error, or a payload that lacks an identity-shaped
    field, logs (for client errors) and leaves
    ``connection.external_account_label`` at its current value (``None`` by
    default). Never raises — the connect flow must never fail because of this.
    """

    if not connection.connected_account_id:
        return
    try:
        details = client.get_connected_account(connection.connected_account_id)
    except (ComposioConnectionError, ComposioCatalogError):
        logger.warning(
            "composio external identity fetch failed connected_account_id=%s toolkit=%s",
            connection.connected_account_id,
            connection.toolkit_slug,
            exc_info=True,
        )
        return
    except Exception:  # noqa: BLE001 - identity capture must never fail connect
        logger.warning(
            "composio external identity fetch failed unexpectedly connected_account_id=%s",
            connection.connected_account_id,
            exc_info=True,
        )
        return
    handle = extract_external_handle(details.state_val)
    if handle:
        connection.external_account_label = handle


def resolve_external_identities(
    session: Session,
    installation_id: uuid.UUID,
    owner_slack_user_id: str,
) -> dict[str, str]:
    """Toolkit slug -> captured external handle for one user's connections.

    Cheap, no-network read: only surfaces labels already captured at connect
    time (via ``capture_external_identity``). Scoped to the caller's own
    active connections so one user's connected-app identity never leaks into
    another user's context.
    """

    rows = session.scalars(
        select(ComposioConnection).where(
            ComposioConnection.installation_id == installation_id,
            ComposioConnection.owner_slack_user_id == owner_slack_user_id,
            ComposioConnection.status == "active",
            ComposioConnection.external_account_label.is_not(None),
        )
    )
    identities: dict[str, str] = {}
    for row in rows:
        label = row.external_account_label
        if label:
            identities[row.toolkit_slug] = label
    return identities
