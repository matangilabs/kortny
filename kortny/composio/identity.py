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
returns ``None`` when none are present — it never fabricates a handle.

Because of that finding, capture has a second stage for the toolkits that
matter most: when the details payload yields no handle, execute the
toolkit's cheap read-only "who am I" tool through the same REST execute path
the runtime uses (``ComposioClient.execute_tool``, with the connection's own
``composio_user_id`` / ``connected_account_id``, exactly like
``kortny/tools/composio_execute.py``) and pluck the handle from the
response. The registry (``WHOAMI_REGISTRY``) lists only toolkits whose tool
slug was verified against Composio's published toolkit catalog
(docs.composio.dev/toolkits/<slug>); toolkits without a verified whoami tool
(e.g. googlecalendar, which has no tool that returns the user's identity)
are omitted and skip the fallback entirely rather than guessing a slug.
Same non-blocking discipline throughout: any failure logs and leaves the
label ``None``.
"""

from __future__ import annotations

import logging
import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass

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

# Modest cap for capture-time request timeouts: identity capture rides inside
# connect completion, so ``capture_external_identity`` swaps a real client for
# a short-timeout sibling (see ``_capture_client``) rather than inherit a
# scrape-tolerant 30s runtime timeout.
CAPTURE_TIMEOUT_SECONDS = 10.0

# Characters stripped from labels outright: quoting/markup characters that
# could break out of the `account "<label>"` framing the context renderers
# use, or smuggle tag-like structure into the prompt.
_LABEL_STRIPPED_CHARS = "\"'`<>"

# Literal escape-sequence remnants (a backslash followed by n/r) surviving
# sanitization indicate an attacker-shaped label; reject those outright.
_LABEL_ESCAPE_REMNANT = re.compile(r"\\[nr]")

_WHITESPACE_RUN = re.compile(r"\s+")

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


@dataclass(frozen=True, slots=True)
class WhoamiSpec:
    """One toolkit's verified "who am I" tool + the fields carrying a handle.

    ``field_keys`` are checked in priority order against the execution's
    ``data`` payload (top level, then one level under common response
    wrappers) — explicit username/login-shaped fields first, email last.
    """

    tool_slug: str
    field_keys: tuple[str, ...]


# Toolkit slug -> verified whoami tool. Slugs verified against Composio's
# published toolkit catalog before hardcoding:
# - github: GITHUB_GET_THE_AUTHENTICATED_USER (GitHub GET /user -> `login`)
# - linear: LINEAR_GET_CURRENT_USER (viewer profile -> `email`, `name`)
# - gmail:  GMAIL_GET_PROFILE (users.getProfile -> `emailAddress`)
# googlecalendar is deliberately absent: Composio exposes no calendar tool
# that returns the authenticated user's identity (settings/list tools return
# timezone/preferences only), so we skip it rather than infer from
# side-channel fields.
#
# Field ordering favors stable identifiers over display names: Linear prefers
# `email` because the viewer `name` is attacker-controllable free text, while
# the email is a verified account identifier (and the value Linear's own
# assignee filters accept). GitHub `login` and Gmail `emailAddress` are
# already the stable identifiers for their toolkits.
WHOAMI_REGISTRY: dict[str, WhoamiSpec] = {
    "github": WhoamiSpec(
        tool_slug="GITHUB_GET_THE_AUTHENTICATED_USER",
        field_keys=("login",),
    ),
    "linear": WhoamiSpec(
        tool_slug="LINEAR_GET_CURRENT_USER",
        field_keys=("email", "name"),
    ),
    "gmail": WhoamiSpec(
        tool_slug="GMAIL_GET_PROFILE",
        field_keys=("emailAddress", "email_address", "email"),
    ),
}

# Wrapper keys under which Composio tool responses commonly nest the actual
# API payload (varies per toolkit; e.g. GraphQL viewer objects, REST detail
# envelopes). Checked one level deep only.
_RESPONSE_WRAPPER_KEYS: tuple[str, ...] = (
    "data",
    "details",
    "response_data",
    "viewer",
    "user",
    "profile",
)


def sanitize_external_label(value: str | None) -> str | None:
    """Sanitize an external handle as untrusted input.

    External labels originate from third-party API payloads (display names
    especially are attacker-controllable free text), and they are rendered
    into the model's prompt. Keep only printable non-control characters,
    collapse whitespace runs to single spaces, strip quote/backtick/angle
    characters, and reject the label entirely when nothing survives or when
    literal escape-sequence remnants (``\\n`` / ``\\r``) remain. Applied both
    before storage and again at every render site (defense in depth for rows
    written before this sanitizer existed).
    """

    if value is None:
        return None
    # Collapse whitespace runs (including newlines/tabs) to single spaces
    # BEFORE the printable filter, so a newline becomes a word boundary
    # rather than silently gluing two fragments together.
    cleaned = _WHITESPACE_RUN.sub(" ", value)
    cleaned = "".join(ch for ch in cleaned if ch.isprintable())
    cleaned = cleaned.translate(str.maketrans("", "", _LABEL_STRIPPED_CHARS))
    cleaned = _WHITESPACE_RUN.sub(" ", cleaned).strip()
    if not cleaned:
        return None
    if _LABEL_ESCAPE_REMNANT.search(cleaned):
        return None
    return cleaned[:MAX_LABEL_LENGTH]


def extract_external_handle(state_val: Mapping[str, object]) -> str | None:
    """Best-effort external handle from a connected account's ``state.val``.

    Prefers an explicit username/login-shaped key; falls back to an
    email-shaped key. Returns ``None`` when neither is present rather than
    guessing from unrelated fields.
    """

    for keys in (_USERNAME_KEYS, _EMAIL_KEYS):
        handle = _pluck(state_val, keys)
        if handle:
            return handle
    return None


def extract_whoami_handle(toolkit_slug: str, data: Mapping[str, object]) -> str | None:
    """Extract a handle from a whoami tool execution's ``data`` payload.

    Uses the toolkit's registered ``field_keys``, checking the payload's top
    level first and then one level under common response wrappers. Returns
    ``None`` for toolkits without a registry entry or payloads lacking the
    registered fields.
    """

    spec = WHOAMI_REGISTRY.get(toolkit_slug)
    if spec is None:
        return None
    return _pluck(data, spec.field_keys)


def _pluck(payload: Mapping[str, object], keys: tuple[str, ...]) -> str | None:
    """First sanitized non-empty string under ``keys`` (top level, then one
    wrapper deep)."""

    for key in keys:
        value = payload.get(key)
        if isinstance(value, str):
            sanitized = sanitize_external_label(value)
            if sanitized:
                return sanitized
    for wrapper in _RESPONSE_WRAPPER_KEYS:
        inner = payload.get(wrapper)
        if not isinstance(inner, Mapping):
            continue
        for key in keys:
            value = inner.get(key)
            if isinstance(value, str):
                sanitized = sanitize_external_label(value)
                if sanitized:
                    return sanitized
    return None


def _capture_client(client: ComposioClient) -> ComposioClient:
    """A short-timeout sibling of ``client`` for capture-time calls.

    The single latency bound for every capture call site (connect.py's
    immediate branch, both dashboard connect paths): a plain ``ComposioClient``
    configured with a longer runtime timeout is swapped for a sibling capped
    at ``CAPTURE_TIMEOUT_SECONDS``, so capture can never stall a connect flow
    past ~10s per call. Subclasses (test fakes) and clients with an injected
    ``http_client`` (whose transport carries its own timeout) pass through
    unchanged.
    """

    if type(client) is not ComposioClient:
        return client
    if client.http_client is not None:
        return client
    if client.timeout_seconds <= CAPTURE_TIMEOUT_SECONDS:
        return client
    return ComposioClient(
        api_key=client.api_key,
        base_url=client.base_url,
        timeout_seconds=CAPTURE_TIMEOUT_SECONDS,
    )


def capture_external_identity(
    client: ComposioClient,
    connection: ComposioConnection,
) -> None:
    """Fetch + store a best-effort external handle on ``connection``.

    Two stages: the connected-account details payload first, then — when
    that yields nothing and the toolkit has a verified whoami tool — one
    read-only whoami execution through the normal REST execute path.

    Non-blocking: any client error, or payloads that lack an identity-shaped
    field, log (for client errors) and leave
    ``connection.external_account_label`` at its current value (``None`` by
    default). Never raises — the connect flow must never fail because of
    this. Latency is bounded to ``CAPTURE_TIMEOUT_SECONDS`` per call via
    ``_capture_client``. Stored labels pass ``sanitize_external_label``.
    """

    if not connection.connected_account_id:
        return
    client = _capture_client(client)
    handle: str | None = None
    try:
        details = client.get_connected_account(connection.connected_account_id)
        handle = extract_external_handle(details.state_val)
    except (ComposioConnectionError, ComposioCatalogError):
        logger.warning(
            "composio external identity fetch failed connected_account_id=%s toolkit=%s",
            connection.connected_account_id,
            connection.toolkit_slug,
            exc_info=True,
        )
    except Exception:  # noqa: BLE001 - identity capture must never fail connect
        logger.warning(
            "composio external identity fetch failed unexpectedly connected_account_id=%s",
            connection.connected_account_id,
            exc_info=True,
        )
    if handle is None:
        handle = _whoami_fallback(client, connection)
    # Extraction already sanitizes, but storage re-applies it so no future
    # extraction path can write an unsanitized label.
    handle = sanitize_external_label(handle)
    if handle:
        connection.external_account_label = handle


def _whoami_fallback(
    client: ComposioClient,
    connection: ComposioConnection,
) -> str | None:
    """Execute the toolkit's verified whoami tool and pluck the handle.

    Skips toolkits without a ``WHOAMI_REGISTRY`` entry. Mirrors the runtime
    execute path (``kortny/tools/composio_execute.py``): same
    ``composio_user_id`` / ``connected_account_id`` pairing. Never raises.
    """

    spec = WHOAMI_REGISTRY.get(connection.toolkit_slug)
    if spec is None:
        return None
    try:
        execution = client.execute_tool(
            tool_slug=spec.tool_slug,
            user_id=connection.composio_user_id,
            connected_account_id=connection.connected_account_id,
            arguments={},
        )
    except (ComposioConnectionError, ComposioCatalogError):
        logger.warning(
            "composio whoami execution failed toolkit=%s tool=%s connected_account_id=%s",
            connection.toolkit_slug,
            spec.tool_slug,
            connection.connected_account_id,
            exc_info=True,
        )
        return None
    except Exception:  # noqa: BLE001 - identity capture must never fail connect
        logger.warning(
            "composio whoami execution failed unexpectedly toolkit=%s tool=%s",
            connection.toolkit_slug,
            spec.tool_slug,
            exc_info=True,
        )
        return None
    if not execution.successful:
        logger.info(
            "composio whoami execution unsuccessful toolkit=%s tool=%s error=%s",
            connection.toolkit_slug,
            spec.tool_slug,
            execution.error,
        )
        return None
    data = execution.data
    if not isinstance(data, Mapping):
        return None
    return extract_whoami_handle(connection.toolkit_slug, data)


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
        # Re-sanitize at read: rows written before the sanitizer existed (or
        # by any other writer) must not reach the prompt unsanitized.
        label = sanitize_external_label(row.external_account_label)
        if label:
            identities[row.toolkit_slug] = label
    return identities
