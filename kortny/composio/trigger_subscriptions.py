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
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from kortny.composio.client import ComposioClient, ComposioConnectionError
from kortny.db.models import ComposioConnection, ComposioTriggerSubscription

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TriggerSubscriptionResult:
    ok: bool
    subscription: ComposioTriggerSubscription | None
    error: str | None
    already_enabled: bool = False


def enable_trigger_subscription(
    session: Session,
    *,
    connection: ComposioConnection,
    trigger_slug: str,
    client: ComposioClient,
) -> TriggerSubscriptionResult:
    """Turn a launch trigger on for a connection's connected account.

    Concurrency: the local subscription row is claimed and row-locked FIRST
    (inserting an inert ``status="disabled"`` placeholder when absent, then
    ``SELECT ... FOR UPDATE``), and only then does the remote REST
    ``create_trigger`` run. Two concurrent enables serialize on the row lock:
    the second waits, finds ``status="active"``, and returns an
    already-enabled no-op instead of firing a second remote create.

    Activation requires a trigger instance id: if the REST create response
    yields no ``ti_*`` id, the enable is treated as a failure and the row is
    NOT activated — an active row with ``composio_trigger_id=None`` would
    strand the upstream trigger (disable would skip the REST call) and
    silently degrade trigger-id-first event routing.

    A REST failure leaves no active local row (the caller rolls back, which
    also discards a just-inserted placeholder) and the error is returned for
    the caller to surface.
    """

    if not connection.connected_account_id:
        return TriggerSubscriptionResult(
            ok=False,
            subscription=None,
            error="This connection has no connected account yet — finish connecting first.",
        )

    # Claim + lock the local row BEFORE the remote create so concurrent
    # enables serialize here rather than both firing REST create_trigger.
    session.execute(
        pg_insert(ComposioTriggerSubscription)
        .values(
            installation_id=connection.installation_id,
            composio_connection_id=connection.id,
            connected_account_id=connection.connected_account_id,
            composio_user_id=connection.composio_user_id,
            owner_slack_user_id=connection.owner_slack_user_id,
            toolkit_slug=connection.toolkit_slug,
            trigger_slug=trigger_slug,
            status="disabled",
        )
        .on_conflict_do_nothing(
            index_elements=["installation_id", "connected_account_id", "trigger_slug"]
        )
    )
    subscription = session.execute(
        select(ComposioTriggerSubscription)
        .where(
            ComposioTriggerSubscription.installation_id == connection.installation_id,
            ComposioTriggerSubscription.connected_account_id
            == connection.connected_account_id,
            ComposioTriggerSubscription.trigger_slug == trigger_slug,
        )
        .with_for_update()
    ).scalar_one()

    if subscription.status == "active":
        return TriggerSubscriptionResult(
            ok=True, subscription=subscription, error=None, already_enabled=True
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
    if composio_trigger_id is None:
        logger.warning(
            "composio trigger enable returned no trigger instance id "
            "toolkit=%s trigger_slug=%s connected_account_id=%s",
            connection.toolkit_slug,
            trigger_slug,
            connection.connected_account_id,
        )
        return TriggerSubscriptionResult(
            ok=False,
            subscription=None,
            error=(
                "Composio did not return a trigger instance id; "
                "the trigger was not enabled."
            ),
        )

    subscription.status = "active"
    subscription.composio_trigger_id = composio_trigger_id
    subscription.composio_connection_id = connection.id
    subscription.composio_user_id = connection.composio_user_id
    subscription.owner_slack_user_id = connection.owner_slack_user_id
    subscription.toolkit_slug = connection.toolkit_slug
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
