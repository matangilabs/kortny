"""Typed, limit-enforcing Slack Block Kit builders.

Voice stays prose; data and controls go through these builders. The LLM
never authors Block Kit JSON — feature code calls these helpers, which raise
``ValueError`` on any Slack platform limit violation so malformed payloads
never reach the API.
"""

from __future__ import annotations

from collections.abc import Sequence

HOME_ACTION_PREFIX = "kortny_home_"
WITNESS_ACTION_PREFIX = "kortny_witness_"

MAX_MESSAGE_BLOCKS = 50
MAX_VIEW_BLOCKS = 100
MAX_MARKDOWN_BLOCK_CHARS = 12_000

MAX_HEADER_CHARS = 150
MAX_SECTION_TEXT_CHARS = 3_000
MAX_SECTION_FIELDS = 10
MAX_SECTION_FIELD_CHARS = 2_000
MAX_CONTEXT_ELEMENTS = 10
MAX_ACTIONS_ELEMENTS = 25
MAX_BUTTON_TEXT_CHARS = 75
MAX_BUTTON_VALUE_CHARS = 2_000
MAX_TABLE_ROWS = 100
MAX_TABLE_COLUMNS = 20
MAX_MODAL_TITLE_CHARS = 24
MAX_MODAL_SUBMIT_CHARS = 24
MAX_MODAL_CLOSE_CHARS = 24


def header(text: str) -> dict:
    """Header block (plain_text, ≤150 chars)."""

    if len(text) > MAX_HEADER_CHARS:
        raise ValueError(
            f"header text exceeds {MAX_HEADER_CHARS} chars (got {len(text)})"
        )
    return {"type": "header", "text": {"type": "plain_text", "text": text}}


def section(
    text: str | None = None,
    *,
    fields: Sequence[str] = (),
    accessory: dict | None = None,
) -> dict:
    """Section block (mrkdwn, ≤3000 chars text, ≤10 fields × 2000 chars)."""

    if text is None and not fields:
        raise ValueError("section requires text or fields")
    if text is not None and len(text) > MAX_SECTION_TEXT_CHARS:
        raise ValueError(
            f"section text exceeds {MAX_SECTION_TEXT_CHARS} chars (got {len(text)})"
        )
    if len(fields) > MAX_SECTION_FIELDS:
        raise ValueError(
            f"section allows at most {MAX_SECTION_FIELDS} fields (got {len(fields)})"
        )
    for field in fields:
        if len(field) > MAX_SECTION_FIELD_CHARS:
            raise ValueError(
                f"section field exceeds {MAX_SECTION_FIELD_CHARS} chars "
                f"(got {len(field)})"
            )

    block: dict = {"type": "section"}
    if text is not None:
        block["text"] = {"type": "mrkdwn", "text": text}
    if fields:
        block["fields"] = [{"type": "mrkdwn", "text": field} for field in fields]
    if accessory is not None:
        block["accessory"] = accessory
    return block


def context(*elements: str) -> dict:
    """Context block (mrkdwn elements, ≤10)."""

    if not elements:
        raise ValueError("context requires at least one element")
    if len(elements) > MAX_CONTEXT_ELEMENTS:
        raise ValueError(
            f"context allows at most {MAX_CONTEXT_ELEMENTS} elements "
            f"(got {len(elements)})"
        )
    return {
        "type": "context",
        "elements": [{"type": "mrkdwn", "text": element} for element in elements],
    }


def divider() -> dict:
    """Divider block."""

    return {"type": "divider"}


def markdown_block(text: str) -> dict:
    """Markdown block (Slack-side markdown→blocks, ≤12k chars)."""

    if len(text) > MAX_MARKDOWN_BLOCK_CHARS:
        raise ValueError(
            f"markdown block exceeds {MAX_MARKDOWN_BLOCK_CHARS} chars (got {len(text)})"
        )
    return {"type": "markdown", "text": text}


def button(
    text: str,
    action_id: str,
    *,
    value: str = "",
    style: str | None = None,
    url: str | None = None,
    confirm_title: str | None = None,
    confirm_text: str | None = None,
) -> dict:
    """Button element (≤75 chars text, ≤2000 chars value)."""

    if len(text) > MAX_BUTTON_TEXT_CHARS:
        raise ValueError(
            f"button text exceeds {MAX_BUTTON_TEXT_CHARS} chars (got {len(text)})"
        )
    if len(value) > MAX_BUTTON_VALUE_CHARS:
        raise ValueError(
            f"button value exceeds {MAX_BUTTON_VALUE_CHARS} chars (got {len(value)})"
        )

    element: dict = {
        "type": "button",
        "text": {"type": "plain_text", "text": text},
        "action_id": action_id,
    }
    if value:
        element["value"] = value
    if style is not None:
        element["style"] = style
    if url is not None:
        element["url"] = url
    if confirm_title is not None or confirm_text is not None:
        element["confirm"] = {
            "title": {"type": "plain_text", "text": confirm_title or "Are you sure?"},
            "text": {"type": "mrkdwn", "text": confirm_text or ""},
            "confirm": {"type": "plain_text", "text": "Confirm"},
            "deny": {"type": "plain_text", "text": "Cancel"},
        }
    return element


def actions(*elements: dict, block_id: str | None = None) -> dict:
    """Actions block (≤25 elements)."""

    if not elements:
        raise ValueError("actions requires at least one element")
    if len(elements) > MAX_ACTIONS_ELEMENTS:
        raise ValueError(
            f"actions allows at most {MAX_ACTIONS_ELEMENTS} elements "
            f"(got {len(elements)})"
        )
    block: dict = {"type": "actions", "elements": list(elements)}
    if block_id is not None:
        block["block_id"] = block_id
    return block


def table(
    rows: Sequence[Sequence[str]],
    *,
    column_alignments: Sequence[str] | None = None,
) -> dict:
    """Table block (raw_text cells, ≤100 rows × 20 cols; MESSAGES ONLY)."""

    if not rows:
        raise ValueError("table requires at least one row")
    if len(rows) > MAX_TABLE_ROWS:
        raise ValueError(
            f"table allows at most {MAX_TABLE_ROWS} rows (got {len(rows)})"
        )
    column_count = len(rows[0])
    if column_count > MAX_TABLE_COLUMNS:
        raise ValueError(
            f"table allows at most {MAX_TABLE_COLUMNS} columns (got {column_count})"
        )
    if column_alignments is not None and len(column_alignments) != column_count:
        raise ValueError(
            "column_alignments must match the number of columns "
            f"({column_count}, got {len(column_alignments)})"
        )

    block: dict = {
        "type": "table",
        "rows": [[{"type": "raw_text", "text": cell} for cell in row] for row in rows],
    }
    if column_alignments is not None:
        block["column_settings"] = [{"align": align} for align in column_alignments]
    return block


def static_select(
    action_id: str,
    options: Sequence[tuple[str, str]],
    *,
    placeholder: str = "",
    initial: str | None = None,
) -> dict:
    """Static select element. ``options`` are ``(text, value)`` pairs."""

    if not options:
        raise ValueError("static_select requires at least one option")
    option_objects = [
        {"text": {"type": "plain_text", "text": label}, "value": value}
        for label, value in options
    ]
    element: dict = {
        "type": "static_select",
        "action_id": action_id,
        "options": option_objects,
    }
    if placeholder:
        element["placeholder"] = {"type": "plain_text", "text": placeholder}
    if initial is not None:
        for option in option_objects:
            if option["value"] == initial:
                element["initial_option"] = option
                break
    return element


def plain_text_input(
    action_id: str,
    *,
    multiline: bool = False,
    placeholder: str = "",
    initial: str = "",
) -> dict:
    """Plain text input element."""

    element: dict = {"type": "plain_text_input", "action_id": action_id}
    if multiline:
        element["multiline"] = True
    if placeholder:
        element["placeholder"] = {"type": "plain_text", "text": placeholder}
    if initial:
        element["initial_value"] = initial
    return element


def input_block(
    label: str,
    element: dict,
    *,
    block_id: str,
    optional: bool = False,
    hint: str | None = None,
) -> dict:
    """Input block wrapping an interactive element."""

    block: dict = {
        "type": "input",
        "block_id": block_id,
        "label": {"type": "plain_text", "text": label},
        "element": element,
        "optional": optional,
    }
    if hint is not None:
        block["hint"] = {"type": "plain_text", "text": hint}
    return block


def home_view(blocks: Sequence[dict]) -> dict:
    """Home tab view envelope (≤100 blocks)."""

    if len(blocks) > MAX_VIEW_BLOCKS:
        raise ValueError(
            f"home view allows at most {MAX_VIEW_BLOCKS} blocks (got {len(blocks)})"
        )
    return {"type": "home", "blocks": list(blocks)}


def modal(
    title: str,
    blocks: Sequence[dict],
    *,
    callback_id: str,
    submit: str = "Save",
    close: str = "Cancel",
    private_metadata: str = "",
) -> dict:
    """Modal view envelope (≤100 blocks; ≤24 chars title/submit/close)."""

    if len(title) > MAX_MODAL_TITLE_CHARS:
        raise ValueError(
            f"modal title exceeds {MAX_MODAL_TITLE_CHARS} chars (got {len(title)})"
        )
    if len(submit) > MAX_MODAL_SUBMIT_CHARS:
        raise ValueError(
            f"modal submit exceeds {MAX_MODAL_SUBMIT_CHARS} chars (got {len(submit)})"
        )
    if len(close) > MAX_MODAL_CLOSE_CHARS:
        raise ValueError(
            f"modal close exceeds {MAX_MODAL_CLOSE_CHARS} chars (got {len(close)})"
        )
    if len(blocks) > MAX_VIEW_BLOCKS:
        raise ValueError(
            f"modal allows at most {MAX_VIEW_BLOCKS} blocks (got {len(blocks)})"
        )
    return {
        "type": "modal",
        "callback_id": callback_id,
        "title": {"type": "plain_text", "text": title},
        "submit": {"type": "plain_text", "text": submit},
        "close": {"type": "plain_text", "text": close},
        "private_metadata": private_metadata,
        "blocks": list(blocks),
    }
