"""The living-document control deck (HIG-244 close-out).

A Block Kit action panel posted under every Document Studio doc. Tapping a
button re-renders or edits the *stored* spec (kept on the Artifact) into a new
version, all in-thread — the user never leaves Slack and never retypes. This is
what Viktor doesn't do: a posted doc stays live.

Two route kinds (kept to two so the dispatch table stays small; the button's
intent rides in the minted action's payload, not the route):

* ``document.rerender`` — deterministic, no LLM: format / theme / revert. Re-render
  the stored spec with a different param.
* ``document.edit`` — LLM spec-diff: shorten / lengthen / regenerate.

This module builds the deck and mints one interactive_actions row per button.
The click handler + worker re-render land in the next slice; until then nothing
posts a deck, so no button is ever dead.
"""

from __future__ import annotations

import json
import uuid
from datetime import timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from kortny.db.models import Artifact, InteractiveAction, Task
from kortny.slack import blockkit
from kortny.slack.interactions import InteractiveActionService, MintedAction
from kortny.tasks import TaskService
from kortny.tasks.identity import TaskIdentity

ROUTE_DOC_RERENDER = "kortny:v1:document.rerender"
ROUTE_DOC_EDIT = "kortny:v1:document.edit"
TARGET_DOCUMENT = "document"

# Docs live longer than approvals — give the deck a week before its keys expire.
DECK_TTL = timedelta(days=7)

RERENDER_FORMATS = ("pdf", "pptx", "docx", "xlsx")
# (label, edit_kind)
EDIT_KINDS = (
    ("✂️ Shorter", "shorten"),
    ("➕ Longer", "lengthen"),
    ("🔄 Regenerate", "regenerate"),
)
_EDIT_INSTRUCTIONS = {
    "shorten": "make it noticeably more concise without dropping key points.",
    "lengthen": "expand it with more depth, detail, and supporting context.",
    "regenerate": "regenerate it with a fresh take, keeping the same topic and title.",
}


def render_control_deck(
    service: InteractiveActionService,
    *,
    installation_id: uuid.UUID,
    task_id: uuid.UUID | None,
    doc_group_id: uuid.UUID,
    doc_version: int,
    current_format: str,
    themes: list[str],
    allowed_user_id: str | None,
    allowed_channel_id: str | None,
    slack_team_id: str | None,
    include_edit: bool = True,
) -> tuple[list[dict], list[MintedAction]]:
    """Build the control deck and mint an interactive_action per button.

    Returns (blocks, minted). The caller posts the blocks, then ``mark_sent``\\ s
    each minted action with the resulting message ts. ``include_edit`` gates the
    LLM edit buttons (shorten/lengthen/regenerate); the deterministic
    format/theme/revert buttons are always present.
    """

    minted: list[MintedAction] = []

    def mint_button(
        label: str, route: str, mode: str, value: str, *, style: str | None = None
    ) -> dict:
        action = service.mint(
            installation_id=installation_id,
            action_kind=mode,
            route=route,
            target_type=TARGET_DOCUMENT,
            target_id=str(doc_group_id),
            task_id=task_id,
            payload={
                "doc_group_id": str(doc_group_id),
                "base_version": doc_version,
                "mode": mode,
                "value": value,
            },
            created_for_user_id=allowed_user_id,
            allowed_user_id=allowed_user_id,
            allowed_channel_id=allowed_channel_id,
            slack_team_id=slack_team_id,
            ttl=DECK_TTL,
        )
        minted.append(action)
        # action_id only needs to match the "^kortny:v1:" handler regex and be
        # unique within the message; the authoritative route/params live on the
        # claimed DB row.
        action_id = f"{route}.{mode}.{value}"[:255]
        return blockkit.button(
            label[: blockkit.MAX_BUTTON_TEXT_CHARS],
            action_id=action_id,
            value=action.raw_key,
            style=style,
        )

    format_buttons = [
        mint_button(
            fmt.upper(),
            ROUTE_DOC_RERENDER,
            "format",
            fmt,
            style="primary" if fmt == current_format else None,
        )
        for fmt in RERENDER_FORMATS
    ]
    theme_buttons = [
        mint_button(theme.title(), ROUTE_DOC_RERENDER, "theme", theme)
        for theme in themes[: blockkit.MAX_ACTIONS_ELEMENTS]
    ]
    # Revert is a deterministic rerender (load an older version's spec); the
    # edit buttons are LLM and gated by ``include_edit``.
    extra_buttons: list[dict] = []
    if doc_version > 1:
        extra_buttons.append(
            mint_button("↩ Revert", ROUTE_DOC_RERENDER, "revert", str(doc_version - 1))
        )
    if include_edit:
        extra_buttons.extend(
            mint_button(label, ROUTE_DOC_EDIT, "edit", kind)
            for label, kind in EDIT_KINDS
        )

    base = f"kortny:doc:{doc_group_id}"
    blocks = [
        blockkit.section(
            f"*Refine this document* (v{doc_version}) — tap to re-render or edit:"
        ),
        blockkit.actions(*format_buttons, block_id=f"{base}:format"[:255]),
        blockkit.actions(*theme_buttons, block_id=f"{base}:theme"[:255]),
    ]
    if extra_buttons:
        blocks.append(blockkit.actions(*extra_buttons, block_id=f"{base}:edit"[:255]))
    return blocks, minted


def process_document_action(
    session: Session,
    action: InteractiveAction,
    *,
    actor_user_id: str,
    task_service: TaskService,
) -> uuid.UUID | None:
    """Spawn the child task that re-renders/edits the document for a clicked button.

    The deterministic modes (format/theme/revert) become a ``document_rerender``
    task the worker fast-path handles with no LLM; edit modes become a
    ``document_edit`` task that runs the agent loop. Returns the child task id.
    """

    payload = action.payload_json or {}
    group = str(payload.get("doc_group_id") or action.target_id or "")
    base_version = payload.get("base_version")
    if not group:
        return None

    # Thread the child's output under the original document's conversation.
    parent = session.get(Task, action.task_id) if action.task_id else None
    channel = (parent.slack_channel_id if parent else None) or action.slack_channel_id
    if not channel:
        return None
    thread_ts = None
    if parent is not None:
        thread_ts = parent.slack_thread_ts or parent.slack_message_ts

    if action.route == ROUTE_DOC_EDIT:
        edit_kind = str(payload.get("value") or "regenerate")
        latest = session.scalar(
            select(Artifact)
            .where(Artifact.doc_group_id == uuid.UUID(group))
            .order_by(Artifact.doc_version.desc())
        )
        if latest is None or latest.spec_json is None:
            return None
        instruction = _EDIT_INSTRUCTIONS.get(
            edit_kind, _EDIT_INSTRUCTIONS["regenerate"]
        )
        input_text = (
            f"Revise the document below: {instruction} Keep everything else "
            "intact. Then re-render it with the document_studio tool, passing "
            f'doc_group_id="{group}" and base_version={latest.doc_version} so it '
            "stays this document's next version (do not start a new document).\n\n"
            f"Current document spec (JSON):\n{json.dumps(latest.spec_json)}"
        )
        identity_payload = {
            "kind": "document_edit",
            "doc_group_id": group,
            "edit_kind": edit_kind,
            "base_version": latest.doc_version,
        }
        source = "document_edit"
    else:
        mode = str(payload.get("mode") or "")
        value = str(payload.get("value") or "")
        input_text = f"Re-render document ({mode}={value})."
        identity_payload = {
            "kind": "document_rerender",
            "doc_group_id": group,
            "mode": mode,
            "value": value,
            "base_version": base_version,
        }
        source = "document_rerender"

    child = task_service.create_task(
        installation_id=action.installation_id,
        slack_channel_id=channel,
        slack_user_id=actor_user_id,
        slack_thread_ts=thread_ts,
        input=input_text,
        parent_task_id=action.task_id,
        identity=TaskIdentity.synthetic(
            source=source,
            # action.id keeps each click's identity unique so a retried handler
            # doesn't collide with the first attempt.
            source_id=f"{group}:{action.id}",
            input_text=input_text,
            payload=identity_payload,
        ),
    )
    return child.id


__all__ = [
    "DECK_TTL",
    "EDIT_KINDS",
    "RERENDER_FORMATS",
    "ROUTE_DOC_EDIT",
    "ROUTE_DOC_RERENDER",
    "TARGET_DOCUMENT",
    "process_document_action",
    "render_control_deck",
]
