"""Witness Block Kit action handler registration (HIG-235).

Accept/Dismiss buttons on Witness suggestion + digest posts route here. They
drive the *same* lifecycle the reaction path uses (``accept_candidate`` ->
``materialize_acceptance`` on accept, ``dismiss_candidate`` on dismiss), so a
button click and a :white_check_mark: reaction are equivalent. The reaction
copy stays in the message text as a fallback for clients that can't render the
buttons.

Handlers ack immediately, log-and-swallow every error (never raise into Bolt),
and are idempotent: a double-click on an already accepted/dismissed candidate
is a quiet no-op.
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import Any, cast

from slack_bolt import App
from sqlalchemy.orm import Session, sessionmaker

from kortny.config import Settings, SettingsError, load_settings
from kortny.db.models import WitnessOpportunityCandidate
from kortny.db.session import session_scope
from kortny.slack.blockkit import WITNESS_ACTION_PREFIX
from kortny.witness.automation import materialize_acceptance
from kortny.witness.lifecycle import (
    WitnessSlackClient,
    accept_candidate,
    dismiss_candidate,
)

logger = logging.getLogger(__name__)

_ACCEPT_CONFIRMATION = "Got it - I'll set this up. :white_check_mark:"
_DISMISS_CONFIRMATION = "Okay, I'll drop this one. :no_entry_sign:"


def register_witness_actions(
    app: App,
    *,
    settings: Settings,
    session_factory: sessionmaker[Session] | None,
) -> None:
    """Register the ``kortny_witness_*`` Block Kit action handler (HIG-235)."""

    @app.action(re.compile(f"^{re.escape(WITNESS_ACTION_PREFIX)}"))
    def handle_witness_action(
        ack: Any,
        body: dict[str, Any],
        action: dict[str, Any],
        client: Any,
        logger: Any = logger,
    ) -> None:
        ack()
        if session_factory is None:
            logger.warning("witness action received without a session factory")
            return
        try:
            _process_witness_action(
                body=body,
                action=action,
                client=client,
                settings=settings,
                session_factory=session_factory,
            )
        except Exception:  # noqa: BLE001 - never raise into Bolt
            logger.exception("Failed to process Witness Block Kit action")


def _process_witness_action(
    *,
    body: dict[str, Any],
    action: dict[str, Any],
    client: Any,
    settings: Settings,
    session_factory: sessionmaker[Session],
) -> None:
    action_id = str(action.get("action_id") or "")
    suffix = action_id[len(WITNESS_ACTION_PREFIX) :]
    if suffix not in {"accept", "dismiss"}:
        logger.info("ignoring unknown witness action_id=%s", action_id)
        return

    candidate_id = _parse_candidate_id(action.get("value"))
    if candidate_id is None:
        logger.info(
            "witness action missing a valid candidate id action_id=%s", action_id
        )
        return

    user_id = _actor_user_id(body)
    channel_id, message_ts = _post_target(body)

    with session_scope(session_factory) as session:
        candidate = session.get(WitnessOpportunityCandidate, candidate_id)
        if candidate is None:
            logger.info("witness action for unknown candidate_id=%s", candidate_id)
            return
        installation_id = candidate.installation_id

        if suffix == "accept":
            confirmation = _do_accept(
                session=session,
                candidate_id=candidate_id,
                installation_id=installation_id,
                user_id=user_id,
                client=client,
                settings=settings,
            )
        else:
            confirmation = _do_dismiss(
                session=session,
                candidate_id=candidate_id,
                installation_id=installation_id,
                user_id=user_id,
            )

    if confirmation is not None and channel_id is not None:
        _post_confirmation(
            client=client,
            channel_id=channel_id,
            thread_ts=message_ts,
            text=confirmation,
        )


def _do_accept(
    *,
    session: Session,
    candidate_id: uuid.UUID,
    installation_id: uuid.UUID,
    user_id: str,
    client: Any,
    settings: Settings,
) -> str | None:
    try:
        candidate = accept_candidate(
            session,
            candidate_id,
            installation_id=installation_id,
            by_user_id=user_id,
        )
    except (LookupError, ValueError) as exc:
        # Double-click idempotency: an already accepted/dismissed/archived
        # candidate is not actionable — quiet no-op.
        logger.info(
            "witness accept ignored candidate_id=%s user=%s reason=%s",
            candidate_id,
            user_id,
            exc,
        )
        return None

    resolved_settings: Settings | None = settings
    if resolved_settings is None:
        try:
            resolved_settings = load_settings()
        except SettingsError:
            resolved_settings = None
    outcome = materialize_acceptance(
        session,
        resolved_settings,
        candidate,
        accepted_by=user_id,
        slack_client=cast(WitnessSlackClient, client),
    )
    logger.info(
        "witness suggestion accepted via button candidate_id=%s user=%s kind=%s",
        candidate_id,
        user_id,
        outcome.kind,
    )
    return _ACCEPT_CONFIRMATION


def _do_dismiss(
    *,
    session: Session,
    candidate_id: uuid.UUID,
    installation_id: uuid.UUID,
    user_id: str,
) -> str | None:
    try:
        dismiss_candidate(
            session,
            candidate_id,
            installation_id=installation_id,
            by_user_id=user_id,
            reason="slack_button",
        )
    except (LookupError, ValueError) as exc:
        logger.info(
            "witness dismiss ignored candidate_id=%s user=%s reason=%s",
            candidate_id,
            user_id,
            exc,
        )
        return None
    logger.info(
        "witness suggestion dismissed via button candidate_id=%s user=%s",
        candidate_id,
        user_id,
    )
    return _DISMISS_CONFIRMATION


def _parse_candidate_id(value: Any) -> uuid.UUID | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return uuid.UUID(value)
    except ValueError:
        return None


def _actor_user_id(body: dict[str, Any]) -> str:
    user = body.get("user")
    if isinstance(user, dict):
        user_id = user.get("id")
        if isinstance(user_id, str) and user_id:
            return user_id
    return "unknown"


def _post_target(body: dict[str, Any]) -> tuple[str | None, str | None]:
    """Resolve the channel id and root message ts for a threaded confirmation."""

    channel_id: str | None = None
    channel = body.get("channel")
    if isinstance(channel, dict):
        candidate = channel.get("id")
        if isinstance(candidate, str) and candidate:
            channel_id = candidate

    message_ts: str | None = None
    message = body.get("message")
    if isinstance(message, dict):
        # Reply in the suggestion's own thread when possible.
        thread_ts = message.get("thread_ts") or message.get("ts")
        if isinstance(thread_ts, str) and thread_ts:
            message_ts = thread_ts

    return channel_id, message_ts


def _post_confirmation(
    *,
    client: Any,
    channel_id: str,
    thread_ts: str | None,
    text: str,
) -> None:
    try:
        client.chat_postMessage(
            channel=channel_id,
            text=text,
            thread_ts=thread_ts,
        )
    except Exception:  # noqa: BLE001 - confirmation is best-effort
        logger.exception("Failed to post Witness action confirmation")
