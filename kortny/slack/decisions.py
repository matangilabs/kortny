"""Decision surfaces: deterministic Block Kit for human decisions (HIG-255 s2).

A "decision" is a server-owned message that asks the user to choose — approve a
tool, confirm a fact, pick an option. The layout is fixed (a statement + bound
option buttons + a prose/reaction fallback); the LLM never authors it. Each
option is minted as an interactive_actions row (kortny/slack/interactions.py),
so the button carries only an opaque key and a click is authorized server-side.

Slice 2a ships the approval decision (approve/reject) — buttons primary, the
existing emoji reactions kept as a fallback that calls the same TaskService
methods. The Bolt handler claims the action, dispatches the task transition,
retires the sibling button, and updates the message.
"""

from __future__ import annotations

import logging
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from kortny.config import Settings
from kortny.slack import blockkit
from kortny.slack.interactions import (
    ClaimStatus,
    InteractiveActionService,
    MintedAction,
)
from kortny.tasks import TaskService

logger = logging.getLogger(__name__)

DECISION_ACTION_PREFIX = "kortny:v1:"
ROUTE_APPROVAL_APPROVE = "kortny:v1:approval.approve"
ROUTE_APPROVAL_REJECT = "kortny:v1:approval.reject"

TARGET_APPROVAL = "approval"


@dataclass(frozen=True, slots=True)
class DecisionOptionSpec:
    """One bound option. Routes/targets/payload are server-owned; only ``label``
    is display copy."""

    label: str
    route: str
    action_kind: str
    style: str | None = None


@dataclass(frozen=True, slots=True)
class DecisionSpec:
    """A decision to render: a statement + options sharing one target."""

    statement: str
    fallback_text: str
    target_type: str
    target_id: str
    options: list[DecisionOptionSpec] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)


def approval_decision(
    *, approval_key: str, tool_name: str, statement: str, fallback_text: str
) -> DecisionSpec:
    """The tool-approval decision: Approve (primary) / Reject (danger)."""

    return DecisionSpec(
        statement=statement,
        fallback_text=fallback_text,
        target_type=TARGET_APPROVAL,
        target_id=approval_key,
        # ``summary`` preserves what was being decided so the resolved message
        # can show it (not just "Approved by @user").
        payload={
            "approval_key": approval_key,
            "tool_name": tool_name,
            "summary": statement,
        },
        options=[
            DecisionOptionSpec(
                label="Approve",
                route=ROUTE_APPROVAL_APPROVE,
                action_kind="approve",
                style="primary",
            ),
            DecisionOptionSpec(
                label="Reject",
                route=ROUTE_APPROVAL_REJECT,
                action_kind="reject",
                style="danger",
            ),
        ],
    )


def render_decision(
    spec: DecisionSpec,
    service: InteractiveActionService,
    *,
    installation_id: uuid.UUID,
    task_id: uuid.UUID | None,
    allowed_user_id: str | None,
    allowed_channel_id: str | None,
    slack_team_id: str | None,
) -> tuple[list[dict], list[MintedAction]]:
    """Mint each option and build the decision blocks.

    Returns (blocks, minted_actions). The caller posts the blocks, then calls
    ``service.mark_sent`` on each minted action with the resulting message ts.
    """

    block_id = f"kortny:decision:{spec.target_type}:{spec.target_id}"[:255]
    buttons: list[dict] = []
    minted: list[MintedAction] = []
    for option in spec.options:
        action = service.mint(
            installation_id=installation_id,
            action_kind=option.action_kind,
            route=option.route,
            target_type=spec.target_type,
            target_id=spec.target_id,
            task_id=task_id,
            payload=spec.payload,
            created_for_user_id=allowed_user_id,
            allowed_user_id=allowed_user_id,
            allowed_channel_id=allowed_channel_id,
            slack_team_id=slack_team_id,
        )
        minted.append(action)
        buttons.append(
            blockkit.button(
                option.label[: blockkit.MAX_BUTTON_TEXT_CHARS],
                action_id=option.route,
                value=action.raw_key,
                style=option.style,
            )
        )
    blocks = [
        blockkit.section(spec.statement[: blockkit.MAX_SECTION_TEXT_CHARS]),
        blockkit.actions(*buttons, block_id=block_id),
    ]
    return blocks, minted


class DecisionOutcome(StrEnum):
    applied = "applied"
    already_handled = "already_handled"
    denied = "denied"
    expired = "expired"
    not_found = "not_found"
    error = "error"


# route -> (TaskService method name, resolved-message line)
_APPROVAL_ROUTES = {
    ROUTE_APPROVAL_APPROVE: ("approve_tool_approval", ":white_check_mark: *Approved*"),
    ROUTE_APPROVAL_REJECT: ("reject_tool_approval", ":no_entry_sign: *Rejected*"),
}

# Keep the resolved message readable; the summary can be a longish prompt.
_RESOLVED_SUMMARY_CHARS = 1500


def process_decision_action(
    session: Session,
    settings: Settings,
    *,
    action_id: str,
    raw_key: str,
    actor_user_id: str,
    team_id: str | None,
    channel_id: str | None,
    update_message: Callable[[str], None] | None = None,
) -> DecisionOutcome:
    """Authorize + apply a clicked decision. Testable core of the Bolt handler.

    Claims the action (actor/workspace/route validated under a row lock),
    dispatches the task transition via TaskService, retires sibling buttons, and
    optionally updates the source message. Idempotent: a second click or a
    reaction that already resolved the task lands as ``already_handled``.
    """

    service = InteractiveActionService(session, signing_key=settings.encryption_key)
    result = service.claim(
        raw_key,
        actor_user_id=actor_user_id,
        team_id=team_id,
        channel_id=channel_id,
        route=action_id,
    )
    if result.status is ClaimStatus.not_found:
        return DecisionOutcome.not_found
    if result.status is ClaimStatus.expired:
        return DecisionOutcome.expired
    if result.status is ClaimStatus.denied:
        return DecisionOutcome.denied
    if result.status is ClaimStatus.already_handled:
        return DecisionOutcome.already_handled
    action = result.action
    if action is None or result.status is not ClaimStatus.ok:
        return DecisionOutcome.error

    method_name, verb = _APPROVAL_ROUTES.get(action_id, ("", ""))
    if not method_name:
        logger.info("decision action: unknown route %s", action_id)
        return DecisionOutcome.error

    task_service = TaskService(session)
    approval_key = str(action.payload_json.get("approval_key") or "")
    method = getattr(task_service, method_name)
    transitioned = (
        method(action.task_id, approval_key=approval_key, by_user_id=actor_user_id)
        if action.task_id is not None
        else None
    )

    # Either way the button is spent: complete it and retire its sibling. A
    # None transition means a reaction (or earlier click) already resolved the
    # task — still a clean "already handled" from the user's view.
    service.complete(action, consumed_by_user_id=actor_user_id)
    service.supersede_siblings(
        installation_id=action.installation_id,
        target_type=action.target_type,
        target_id=action.target_id,
        keep_id=action.id,
    )
    if update_message is not None:
        try:
            who = f"<@{actor_user_id}>"
            summary = str(action.payload_json.get("summary") or "").strip()
            resolution = f"{verb} by {who}."
            # Keep what was decided visible above the resolution line.
            text = (
                f"{summary[:_RESOLVED_SUMMARY_CHARS]}\n\n{resolution}"
                if summary
                else (resolution)
            )
            update_message(text)
        except Exception:  # noqa: BLE001 — message update must never fail the action
            logger.info("decision action: message update failed", exc_info=True)
    return (
        DecisionOutcome.applied
        if transitioned is not None
        else (DecisionOutcome.already_handled)
    )


def register_decision_actions(
    app: Any,
    *,
    settings: Settings,
    session_factory: sessionmaker[Session],
) -> None:
    """Register the Bolt handler for ``kortny:v1:*`` decision buttons."""

    from kortny.db.session import session_scope

    @app.action(re.compile(f"^{re.escape(DECISION_ACTION_PREFIX)}"))
    def handle_decision_action(
        ack: Callable[[], None],
        body: dict[str, Any],
        action: dict[str, Any],
        client: Any,
        logger: Any,
    ) -> None:
        ack()
        try:
            action_id = str(action.get("action_id") or "")
            raw_key = str(action.get("value") or "")
            user = body.get("user") or {}
            actor = str(user.get("id") or "")
            team = (body.get("team") or {}).get("id")
            container = body.get("container") or {}
            channel = (body.get("channel") or {}).get("id") or container.get(
                "channel_id"
            )
            message_ts = container.get("message_ts") or (body.get("message") or {}).get(
                "ts"
            )
            if not action_id or not raw_key or not actor:
                return

            def _update(text: str, _ch: Any = channel, _ts: Any = message_ts) -> None:
                if _ch and _ts:
                    client.chat_update(channel=_ch, ts=_ts, text=text, blocks=[])

            with session_scope(session_factory) as session:
                process_decision_action(
                    session,
                    settings,
                    action_id=action_id,
                    raw_key=raw_key,
                    actor_user_id=actor,
                    team_id=team,
                    channel_id=channel,
                    update_message=_update,
                )
        except Exception:  # noqa: BLE001 — never raise into Bolt
            logger.exception("Failed to process Slack decision action")


__all__ = [
    "DecisionOptionSpec",
    "DecisionOutcome",
    "DecisionSpec",
    "ROUTE_APPROVAL_APPROVE",
    "ROUTE_APPROVAL_REJECT",
    "approval_decision",
    "process_decision_action",
    "register_decision_actions",
    "render_decision",
]
