"""Composio trigger subscription opt-in (dashboard, admin-only).

Plain functions over a ``Session`` and the REST ``ComposioClient`` — no SDK
involvement here. The realtime SDK client lives entirely inside the listener
thread (``kortny/composio/listener.py``); this module only ever talks to
Composio over the REST ``/api/v3.1/triggers`` endpoints already wrapped by
``ComposioClient.create_trigger`` / ``disable_trigger``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from kortny.composio.client import ComposioClient, ComposioConnectionError
from kortny.db.models import ComposioConnection, ComposioTriggerSubscription

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TriggerSubscriptionResult:
    ok: bool
    subscription: ComposioTriggerSubscription | None
    error: str | None


def enable_trigger_subscription(
    session: Session,
    *,
    connection: ComposioConnection,
    trigger_slug: str,
    client: ComposioClient,
) -> TriggerSubscriptionResult:
    """Turn a launch trigger on for a connection's connected account.

    REST ``create_trigger`` runs first; only on success is the local
    subscription row upserted to ``status="active"`` carrying the returned
    ``ti_*`` id. A REST failure leaves no active local row (an existing
    disabled row, if any, is left untouched) and the error is returned for
    the caller to surface.
    """

    if not connection.connected_account_id:
        return TriggerSubscriptionResult(
            ok=False,
            subscription=None,
            error="This connection has no connected account yet — finish connecting first.",
        )

    try:
        response = client.create_trigger(
            trigger_slug,
            connection.composio_user_id,
            {},
            connected_account_id=connection.connected_account_id,
        )
    except ComposioConnectionError as exc:
        logger.warning(
            "composio trigger enable failed toolkit=%s trigger_slug=%s connected_account_id=%s "
            "error=%s",
            connection.toolkit_slug,
            trigger_slug,
            connection.connected_account_id,
            exc,
        )
        return TriggerSubscriptionResult(ok=False, subscription=None, error=str(exc))

    composio_trigger_id = _extract_trigger_id(response)

    existing = session.execute(
        select(ComposioTriggerSubscription).where(
            ComposioTriggerSubscription.installation_id == connection.installation_id,
            ComposioTriggerSubscription.connected_account_id
            == connection.connected_account_id,
            ComposioTriggerSubscription.trigger_slug == trigger_slug,
        )
    ).scalar_one_or_none()

    if existing is not None:
        existing.status = "active"
        existing.composio_trigger_id = composio_trigger_id
        existing.composio_connection_id = connection.id
        existing.composio_user_id = connection.composio_user_id
        existing.owner_slack_user_id = connection.owner_slack_user_id
        existing.toolkit_slug = connection.toolkit_slug
        session.flush()
        return TriggerSubscriptionResult(ok=True, subscription=existing, error=None)

    subscription = ComposioTriggerSubscription(
        installation_id=connection.installation_id,
        composio_connection_id=connection.id,
        connected_account_id=connection.connected_account_id,
        composio_user_id=connection.composio_user_id,
        owner_slack_user_id=connection.owner_slack_user_id,
        toolkit_slug=connection.toolkit_slug,
        trigger_slug=trigger_slug,
        composio_trigger_id=composio_trigger_id,
        status="active",
    )
    session.add(subscription)
    session.flush()
    return TriggerSubscriptionResult(ok=True, subscription=subscription, error=None)


def disable_trigger_subscription(
    session: Session,
    *,
    subscription: ComposioTriggerSubscription,
    client: ComposioClient,
) -> TriggerSubscriptionResult:
    """Turn a trigger off.

    Tolerates a remote REST failure: log it and disable the local
    subscription anyway, so a Composio-side outage or an already-deleted
    remote trigger can never strand the local row as "active" — that would
    otherwise keep routing realtime events to a subscription the operator
    believed was off.
    """

    if subscription.composio_trigger_id:
        try:
            client.disable_trigger(subscription.composio_trigger_id)
        except ComposioConnectionError as exc:
            logger.warning(
                "composio trigger disable REST call failed trigger_id=%s error=%s; "
                "disabling local subscription anyway",
                subscription.composio_trigger_id,
                exc,
            )

    subscription.status = "disabled"
    session.flush()
    return TriggerSubscriptionResult(ok=True, subscription=subscription, error=None)


def _extract_trigger_id(response: dict[str, Any]) -> str | None:
    """Best-effort ``ti_*`` id extraction — Composio's create response shape varies."""

    for key in ("id", "trigger_id", "triggerId", "nano_id", "nanoId"):
        value = response.get(key)
        if isinstance(value, str) and value.strip():
            return value
    nested = response.get("trigger_instance") or response.get("data")
    if isinstance(nested, dict):
        return _extract_trigger_id(nested)
    return None
