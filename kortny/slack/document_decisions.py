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

import uuid
from datetime import timedelta

from kortny.slack import blockkit
from kortny.slack.interactions import InteractiveActionService, MintedAction

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
) -> tuple[list[dict], list[MintedAction]]:
    """Build the control deck and mint an interactive_action per button.

    Returns (blocks, minted). The caller posts the blocks, then ``mark_sent``\\ s
    each minted action with the resulting message ts.
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
    edit_buttons = [
        mint_button(label, ROUTE_DOC_EDIT, "edit", kind) for label, kind in EDIT_KINDS
    ]
    if doc_version > 1:
        edit_buttons.append(
            mint_button("↩ Revert", ROUTE_DOC_RERENDER, "revert", str(doc_version - 1))
        )

    base = f"kortny:doc:{doc_group_id}"
    blocks = [
        blockkit.section(
            f"*Refine this document* (v{doc_version}) — tap to re-render or edit:"
        ),
        blockkit.actions(*format_buttons, block_id=f"{base}:format"[:255]),
        blockkit.actions(*theme_buttons, block_id=f"{base}:theme"[:255]),
        blockkit.actions(*edit_buttons, block_id=f"{base}:edit"[:255]),
    ]
    return blocks, minted


__all__ = [
    "DECK_TTL",
    "EDIT_KINDS",
    "RERENDER_FORMATS",
    "ROUTE_DOC_EDIT",
    "ROUTE_DOC_RERENDER",
    "TARGET_DOCUMENT",
    "render_control_deck",
]
