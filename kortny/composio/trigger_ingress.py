"""Composio trigger event ingestion pipeline.

Receives a parsed trigger event (already verified by
``verify_and_parse_trigger_webhook`` for the REST webhook path, or by
``kortny.composio.listener.adapt_sdk_event`` for the realtime SDK listener
path), deduplicates it race-safely, resolves the owning subscription with
routing precision (trigger-id first, connected-account fallback), scores it
with the deterministic launch-trigger scorer, and persists the result as a
``ComposioTriggerEvent`` row.

The caller owns the transaction; this function flushes but does not commit.
In shadow mode (the default) the function stops after recording the event and
never creates witness candidates or posts to Slack.
"""

from __future__ import annotations

import logging
import uuid
from decimal import Decimal
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from kortny.composio.client import ParsedTriggerEvent
from kortny.composio.triggers import LAUNCH_TRIGGERS
from kortny.db.models import ComposioTriggerEvent, ComposioTriggerSubscription

logger = logging.getLogger(__name__)


def _resolve_subscription(
    session: Session,
    installation_id: uuid.UUID,
    parsed: ParsedTriggerEvent,
) -> tuple[ComposioTriggerSubscription | None, bool]:
    """Resolve the owning active subscription for a trigger event.

    Routing precision: resolve by ``composio_trigger_id`` (the ``ti_*`` id)
    FIRST when the event carries one; fall back to
    ``(connected_account_id, trigger_slug)`` only when the trigger-id lookup
    finds nothing. Both lookups are scoped to the installation and to active
    subscriptions.

    Returns ``(subscription, ambiguous)``. ``ambiguous`` is ``True`` when a
    lookup step finds more than one candidate row — the caller must drop the
    event rather than guess which subscription owns it.
    """

    if parsed.trigger_id:
        by_trigger_id = list(
            session.execute(
                select(ComposioTriggerSubscription).where(
                    ComposioTriggerSubscription.installation_id == installation_id,
                    ComposioTriggerSubscription.composio_trigger_id
                    == parsed.trigger_id,
                    ComposioTriggerSubscription.status == "active",
                )
            ).scalars()
        )
        if len(by_trigger_id) > 1:
            return None, True
        if len(by_trigger_id) == 1:
            return by_trigger_id[0], False

    if parsed.connected_account_id:
        by_account = list(
            session.execute(
                select(ComposioTriggerSubscription).where(
                    ComposioTriggerSubscription.installation_id == installation_id,
                    ComposioTriggerSubscription.connected_account_id
                    == parsed.connected_account_id,
                    ComposioTriggerSubscription.trigger_slug == parsed.trigger_slug,
                    ComposioTriggerSubscription.status == "active",
                )
            ).scalars()
        )
        if len(by_account) > 1:
            return None, True
        if len(by_account) == 1:
            return by_account[0], False

    return None, False


def ingest_trigger_event(
    session: Session,
    *,
    installation_id: uuid.UUID,
    parsed: ParsedTriggerEvent,
    shadow: bool = True,
) -> ComposioTriggerEvent | None:
    """Ingest a verified Composio trigger event.

    Steps:
    1. RESOLVE: find an active ComposioTriggerSubscription by
       ``composio_trigger_id`` first, falling back to
       ``(connected_account_id, trigger_slug)``. Multiple candidate matches at
       either step are ambiguous — the event is dropped (logged, not
       persisted) rather than guessed at, and this function returns ``None``.
    2. SCORE: if a known launch trigger, run its deterministic scorer on
       ``parsed.data``. Unmatched events are recorded with metadata only —
       the full ``data`` payload is never persisted for an event that could
       not be attributed to a subscription.
    3. RECORD (race-safe): ``INSERT ... ON CONFLICT DO NOTHING`` against the
       ``(installation_id, trigger_slug, event_id)`` unique constraint. A
       concurrent duplicate delivery loses the race harmlessly; the caller
       gets the existing row back either way.
    4. In shadow mode: stop here — no witness candidates, no Slack delivery.

    The caller owns the transaction. This function flushes but does not
    commit. Returns ``None`` when the event was dropped (ambiguous routing).
    """

    subscription, ambiguous = _resolve_subscription(session, installation_id, parsed)
    if ambiguous:
        logger.warning(
            "composio trigger event dropped: ambiguous subscription match "
            "installation_id=%s trigger_slug=%s trigger_id=%s "
            "connected_account_id=%s",
            installation_id,
            parsed.trigger_slug,
            parsed.trigger_id,
            parsed.connected_account_id,
        )
        return None

    importance_score: Decimal | None = None
    decision: str | None = None
    decision_reason: str | None = None
    raw_payload_json: dict[str, Any]

    if subscription is None:
        decision = "unmatched"
        decision_reason = "no active subscription found for this trigger and account"
        # Metadata only — never persist the full untrusted payload for an
        # event we could not attribute to a subscription.
        raw_payload_json = {
            "id": parsed.id,
            "trigger_slug": parsed.trigger_slug,
            "trigger_id": parsed.trigger_id,
            "connected_account_id": parsed.connected_account_id,
            "user_id": parsed.user_id,
            "timestamp": parsed.timestamp,
        }
    else:
        launch_trigger = LAUNCH_TRIGGERS.get(parsed.trigger_slug)
        if launch_trigger is None:
            decision = "silent"
            importance_score = Decimal("0.0")
            decision_reason = "unrecognized trigger — no scorer registered"
        else:
            result = launch_trigger.scorer(parsed.data)
            importance_score = Decimal(str(result.importance))
            decision = result.decision
            decision_reason = result.reason
        raw_payload_json = {
            "id": parsed.id,
            "type": parsed.type,
            "trigger_slug": parsed.trigger_slug,
            "trigger_id": parsed.trigger_id,
            "connected_account_id": parsed.connected_account_id,
            "user_id": parsed.user_id,
            "data": parsed.data,
            "timestamp": parsed.timestamp,
        }

    insert_stmt = (
        pg_insert(ComposioTriggerEvent)
        .values(
            installation_id=installation_id,
            subscription_id=subscription.id if subscription is not None else None,
            composio_trigger_id=parsed.trigger_id,
            event_id=parsed.id,
            trigger_slug=parsed.trigger_slug,
            connected_account_id=parsed.connected_account_id,
            composio_user_id=parsed.user_id,
            raw_payload_json=raw_payload_json,
            importance_score=importance_score,
            decision=decision,
            decision_reason=decision_reason,
        )
        .on_conflict_do_nothing(
            index_elements=["installation_id", "trigger_slug", "event_id"]
        )
        .returning(ComposioTriggerEvent.id)
    )
    inserted_id = session.execute(insert_stmt).scalar_one_or_none()
    session.flush()

    if inserted_id is None:
        # Lost the race to a concurrent insert (or this exact event was
        # already recorded) — return the existing row.
        existing = session.execute(
            select(ComposioTriggerEvent).where(
                ComposioTriggerEvent.installation_id == installation_id,
                ComposioTriggerEvent.trigger_slug == parsed.trigger_slug,
                ComposioTriggerEvent.event_id == parsed.id,
            )
        ).scalar_one()
        return existing

    event = session.get(ComposioTriggerEvent, inserted_id)
    assert event is not None, "just-inserted row must be loadable in this transaction"

    # Shadow mode: stop here.
    # When shadow=False, future slices will project into the Witness ledger.
    _ = shadow  # acknowledged; future use

    return event
