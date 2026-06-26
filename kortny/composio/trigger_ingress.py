"""Composio trigger event ingestion pipeline.

Receives a parsed trigger event (already verified by
``verify_and_parse_trigger_webhook``), deduplicates it, resolves the owning
subscription, scores it with the deterministic launch-trigger scorer, and
persists the result as a ``ComposioTriggerEvent`` row.

The caller owns the transaction; this function flushes but does not commit.
In shadow mode (the default) the function stops after recording the event and
never creates witness candidates or posts to Slack.
"""

from __future__ import annotations

import uuid
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.orm import Session

from kortny.composio.client import ParsedTriggerEvent
from kortny.composio.triggers import LAUNCH_TRIGGERS
from kortny.db.models import ComposioTriggerEvent, ComposioTriggerSubscription


def ingest_trigger_event(
    session: Session,
    *,
    installation_id: uuid.UUID,
    parsed: ParsedTriggerEvent,
    shadow: bool = True,
) -> ComposioTriggerEvent:
    """Ingest a verified Composio trigger event.

    Steps:
    1. DEDUPE: if (installation_id, trigger_slug, event_id) already exists,
       return the existing row immediately.
    2. RESOLVE: find an active ComposioTriggerSubscription matching the
       installation, connected_account_id, and trigger_slug.
    3. SCORE: if a known launch trigger, run its deterministic scorer on
       ``parsed.data``.
    4. RECORD: persist a ComposioTriggerEvent row.
    5. In shadow mode: stop here — no witness candidates, no Slack delivery.

    The caller owns the transaction. This function flushes but does not commit.
    """
    # 1. Deduplicate
    existing = session.execute(
        select(ComposioTriggerEvent).where(
            ComposioTriggerEvent.installation_id == installation_id,
            ComposioTriggerEvent.trigger_slug == parsed.trigger_slug,
            ComposioTriggerEvent.event_id == parsed.id,
        )
    ).scalar_one_or_none()
    if existing is not None:
        return existing

    # 2. Resolve subscription
    subscription: ComposioTriggerSubscription | None = None
    if parsed.connected_account_id:
        subscription = session.execute(
            select(ComposioTriggerSubscription).where(
                ComposioTriggerSubscription.installation_id == installation_id,
                ComposioTriggerSubscription.connected_account_id
                == parsed.connected_account_id,
                ComposioTriggerSubscription.trigger_slug == parsed.trigger_slug,
                ComposioTriggerSubscription.status == "active",
            )
        ).scalar_one_or_none()

    # 3. Score
    importance_score: Decimal | None = None
    decision: str | None = None
    decision_reason: str | None = None

    if subscription is None:
        decision = "unmatched"
        decision_reason = "no active subscription found for this trigger and account"
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

    # 4. Record
    event = ComposioTriggerEvent(
        installation_id=installation_id,
        subscription_id=subscription.id if subscription is not None else None,
        composio_trigger_id=parsed.trigger_id,
        event_id=parsed.id,
        trigger_slug=parsed.trigger_slug,
        connected_account_id=parsed.connected_account_id,
        composio_user_id=parsed.user_id,
        raw_payload_json={
            "id": parsed.id,
            "type": parsed.type,
            "trigger_slug": parsed.trigger_slug,
            "trigger_id": parsed.trigger_id,
            "connected_account_id": parsed.connected_account_id,
            "user_id": parsed.user_id,
            "data": parsed.data,
            "timestamp": parsed.timestamp,
        },
        importance_score=importance_score,
        decision=decision,
        decision_reason=decision_reason,
    )
    session.add(event)
    session.flush()

    # 5. Shadow mode: stop here
    # When shadow=False, future slices will project into the Witness ledger.
    _ = shadow  # acknowledged; future use

    return event
