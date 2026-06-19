"""Server-owned lifecycle for Block Kit interactive actions (HIG-255 slice 2).

The trust model (codex): the opaque key in a Slack button is only the *lookup*
key, never the security model. A click is honored only when the key resolves AND
the Slack-authenticated actor, workspace, and container match the bound action
AND the target is still actionable — all under a row lock, with the state
transition made idempotent. Wrong-actor / forged / expired clicks are recorded
as denials and leave the action usable for the legitimate actor.

This module owns minting (bind an action into a message), claiming (authorize +
lock a click), completion, and superseding siblings. It deliberately does NOT
perform the task-level transition (approve/retry/…) — that stays in TaskService,
which keeps its own idempotency (waiting_approval + latest approval key).
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum

from sqlalchemy import select
from sqlalchemy.orm import Session

from kortny.db.models import InteractiveAction

# Action lifecycle.
STATUS_PENDING_SEND = "pending_send"
STATUS_SENT = "sent"
STATUS_CONSUMING = "consuming"
STATUS_CONSUMED = "consumed"
STATUS_SUPERSEDED = "superseded"
STATUS_EXPIRED = "expired"
STATUS_SEND_FAILED = "send_failed"

_KEY_PREFIX = "iact_v1_"
DEFAULT_TTL = timedelta(hours=24)


class ClaimStatus(StrEnum):
    """Outcome of trying to claim a clicked action."""

    ok = "ok"  # authorized + locked, caller may apply the transition
    not_found = "not_found"  # unknown/forged key
    expired = "expired"
    denied = "denied"  # wrong actor / workspace / route / channel
    already_handled = "already_handled"  # consumed or superseded
    wrong_status = "wrong_status"  # e.g. never sent


@dataclass(frozen=True, slots=True)
class ClaimResult:
    status: ClaimStatus
    action: InteractiveAction | None = None


@dataclass(frozen=True, slots=True)
class MintedAction:
    raw_key: str  # goes in the Slack button value; never persisted
    action: InteractiveAction


class InteractiveActionService:
    """Mint, claim, and retire interactive actions."""

    def __init__(self, session: Session, *, signing_key: str | None) -> None:
        self.session = session
        # The key only needs to be a stable secret; fall back to a constant so a
        # dev/test env without ENCRYPTION_KEY still functions (the actor +
        # workspace + target checks carry the real security either way).
        self._signing_key = (signing_key or "kortny-interaction-dev-key").encode()

    # -- minting ------------------------------------------------------------

    def mint(
        self,
        *,
        installation_id: uuid.UUID,
        action_kind: str,
        route: str,
        target_type: str,
        target_id: str,
        task_id: uuid.UUID | None = None,
        payload: dict | None = None,
        created_for_user_id: str | None = None,
        allowed_user_id: str | None = None,
        required_role: str | None = None,
        allowed_channel_id: str | None = None,
        slack_team_id: str | None = None,
        ttl: timedelta = DEFAULT_TTL,
        now: datetime | None = None,
    ) -> MintedAction:
        """Create a pending action and return its one-time raw key.

        The raw key is returned to put in the Slack button value; only its HMAC
        hash is stored, so the DB never holds a usable bearer token.
        """

        moment = now or datetime.now(UTC)
        raw_key = _KEY_PREFIX + secrets.token_urlsafe(32)
        action = InteractiveAction(
            installation_id=installation_id,
            task_id=task_id,
            action_key_hash=self._hash(raw_key),
            action_kind=action_kind,
            route=route,
            status=STATUS_PENDING_SEND,
            target_type=target_type,
            target_id=target_id,
            payload_json=payload or {},
            created_for_user_id=created_for_user_id,
            allowed_user_id=allowed_user_id,
            required_role=required_role,
            allowed_channel_id=allowed_channel_id,
            slack_team_id=slack_team_id,
            expires_at=moment + ttl,
        )
        self.session.add(action)
        self.session.flush()
        return MintedAction(raw_key=raw_key, action=action)

    def mark_sent(
        self,
        action: InteractiveAction,
        *,
        channel_id: str,
        message_ts: str | None,
        block_id: str | None,
        slack_action_id: str | None,
        now: datetime | None = None,
    ) -> None:
        action.status = STATUS_SENT
        action.slack_channel_id = channel_id
        action.slack_message_ts = message_ts
        action.slack_block_id = block_id
        action.slack_action_id = slack_action_id
        action.sent_at = now or datetime.now(UTC)
        self.session.flush()

    def mark_send_failed(self, action: InteractiveAction) -> None:
        action.status = STATUS_SEND_FAILED
        self.session.flush()

    # -- claiming -----------------------------------------------------------

    def claim(
        self,
        raw_key: str,
        *,
        actor_user_id: str,
        team_id: str | None,
        channel_id: str | None,
        route: str | None = None,
        actor_role: str | None = None,
        now: datetime | None = None,
    ) -> ClaimResult:
        """Authorize + lock a clicked action. Does NOT apply the transition.

        On success the row is locked (FOR UPDATE) and moved to ``consuming``;
        the caller applies the task transition then calls :meth:`complete`. A
        denial (wrong actor/workspace/route) is audited and the row stays usable.
        """

        moment = now or datetime.now(UTC)
        action = self.session.scalar(
            select(InteractiveAction)
            .where(InteractiveAction.action_key_hash == self._hash(raw_key))
            .with_for_update()
        )
        if action is None:
            return ClaimResult(ClaimStatus.not_found)
        if action.status in (STATUS_CONSUMED, STATUS_SUPERSEDED):
            return ClaimResult(ClaimStatus.already_handled, action)
        if action.status not in (STATUS_SENT, STATUS_CONSUMING):
            return ClaimResult(ClaimStatus.wrong_status, action)
        if action.expires_at <= moment:
            action.status = STATUS_EXPIRED
            self.session.flush()
            return ClaimResult(ClaimStatus.expired, action)
        if not self._actor_allowed(
            action,
            actor_user_id=actor_user_id,
            team_id=team_id,
            channel_id=channel_id,
            actor_role=actor_role,
        ) or (route is not None and route != action.route):
            self._record_denial(action, moment)
            return ClaimResult(ClaimStatus.denied, action)

        action.status = STATUS_CONSUMING
        self.session.flush()
        return ClaimResult(ClaimStatus.ok, action)

    def complete(
        self,
        action: InteractiveAction,
        *,
        consumed_by_user_id: str,
        result_task_id: uuid.UUID | None = None,
        now: datetime | None = None,
    ) -> None:
        action.status = STATUS_CONSUMED
        action.consumed_at = now or datetime.now(UTC)
        action.consumed_by_user_id = consumed_by_user_id
        if result_task_id is not None:
            action.result_task_id = result_task_id
        self.session.flush()

    def supersede_siblings(
        self,
        *,
        installation_id: uuid.UUID,
        target_type: str,
        target_id: str,
        keep_id: uuid.UUID | None = None,
    ) -> int:
        """Retire the other live actions for the same target (e.g. the reject
        button once approve was clicked, or both once a reaction approved)."""

        rows = self.session.scalars(
            select(InteractiveAction)
            .where(
                InteractiveAction.installation_id == installation_id,
                InteractiveAction.target_type == target_type,
                InteractiveAction.target_id == target_id,
                InteractiveAction.status.in_(
                    (STATUS_PENDING_SEND, STATUS_SENT, STATUS_CONSUMING)
                ),
            )
            .with_for_update()
        ).all()
        count = 0
        for row in rows:
            if keep_id is not None and row.id == keep_id:
                continue
            row.status = STATUS_SUPERSEDED
            count += 1
        self.session.flush()
        return count

    # -- internals ----------------------------------------------------------

    def _hash(self, raw_key: str) -> str:
        return hmac.new(self._signing_key, raw_key.encode(), hashlib.sha256).hexdigest()

    def _actor_allowed(
        self,
        action: InteractiveAction,
        *,
        actor_user_id: str,
        team_id: str | None,
        channel_id: str | None,
        actor_role: str | None,
    ) -> bool:
        if action.slack_team_id and team_id and action.slack_team_id != team_id:
            return False
        if (
            action.allowed_channel_id
            and channel_id
            and action.allowed_channel_id != channel_id
        ):
            return False
        if action.allowed_user_id is not None and action.allowed_user_id != actor_user_id:
            return False
        # required_role is fail-closed: if a role is required, the caller must
        # supply the actor's role and it must match. An action that asks for a
        # role but gets no role denies, so a role gate can never silently pass.
        return not (action.required_role is not None and action.required_role != actor_role)

    def _record_denial(self, action: InteractiveAction, moment: datetime) -> None:
        action.denied_count = (action.denied_count or 0) + 1
        action.last_denied_at = moment
        self.session.flush()


__all__ = [
    "ClaimResult",
    "ClaimStatus",
    "InteractiveActionService",
    "MintedAction",
]
