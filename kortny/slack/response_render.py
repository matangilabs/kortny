"""Render a humanized response + optional presentation hint to Block Kit (HIG-255).

This is the deterministic renderer half of the hybrid architecture: the LLM
authors prose + an optional :class:`PresentationHint`; this module turns that
into validated Block Kit using :mod:`kortny.slack.blockkit` builders. The LLM
never authors Slack JSON.

Safety contract:

* Voice (the humanized prose) renders as a single ``markdown`` block — Slack's
  own markdown→blocks translation, so links/lists/bold render natively.
* Each hint element is built independently; any element that violates a Slack
  limit is dropped (the builders raise ``ValueError``), never fatal.
* The final block list is validated against message limits; if it can't be made
  valid it degrades to prose-only (``None``), so a bad layout never drops the
  answer.
* With no hint and no markdown table, returns ``None`` so plain prose posts
  byte-identical to before (existing exact-text tests stay stable).

Surface-awareness: native ``table`` blocks are message-only, so on modal/Home
surfaces a table degrades to a markdown table inside the voice/markdown block.
"""

from __future__ import annotations

import logging

from kortny.slack import blockkit
from kortny.slack.presentation import (
    CardItem,
    CardsElement,
    ContextElement,
    FieldsElement,
    ItemsElement,
    ListItem,
    PresentationElement,
    PresentationHint,
    SourcesElement,
    TableElement,
)
from kortny.slack.response_blocks import _contains_markdown_table
from kortny.slack.source_index import SourceIndex

logger = logging.getLogger(__name__)

MESSAGE_SURFACE = "message"
_TABLE_CAPABLE_SURFACES = frozenset({MESSAGE_SURFACE})


def render_blocks(
    message: str,
    hint: PresentationHint | None = None,
    *,
    surface: str = MESSAGE_SURFACE,
    source_index: SourceIndex | None = None,
) -> list[dict] | None:
    """Render ``message`` + optional ``hint`` to Block Kit, or ``None`` for prose.

    ``None`` means "post the prose as plain text" — the caller always carries the
    full prose in the message ``text`` fallback. ``source_index`` resolves a
    ``sources`` element's refs to server-owned URLs (the LLM never authors them).
    """

    text = message.strip()
    if not text:
        return None

    has_table_in_prose = _contains_markdown_table(text)
    if hint is None:
        # Preserve the legacy path exactly: a markdown table → one markdown block,
        # everything else → plain text (None).
        if has_table_in_prose and len(text) <= blockkit.MAX_MARKDOWN_BLOCK_CHARS:
            return [blockkit.markdown_block(text)]
        return None

    voice = _voice_block(text)
    if voice is None:
        # Prose too large to wrap; fall back to plain text rather than risk a
        # rejected payload.
        return None

    element_blocks: list[dict] = []
    for element in hint.elements:
        element_blocks.extend(
            _render_element(element, surface=surface, source_index=source_index)
        )

    if not element_blocks:
        # Hint produced nothing usable: keep legacy behavior (table→block,
        # else prose) so we never post a lone markdown block for plain prose.
        if has_table_in_prose:
            return [voice]
        return None

    blocks = [voice, *element_blocks]
    return _enforce_message_limits(blocks, voice=voice)


def _voice_block(text: str) -> dict | None:
    try:
        return blockkit.markdown_block(text)
    except ValueError:
        return None


def _render_element(
    element: PresentationElement,
    *,
    surface: str,
    source_index: SourceIndex | None,
) -> list[dict]:
    """Build blocks for one hint element; drop (return []) on any limit error."""

    try:
        if isinstance(element, FieldsElement):
            return _render_fields(element)
        if isinstance(element, TableElement):
            return _render_table(element, surface=surface)
        if isinstance(element, CardsElement):
            return _render_cards(element)
        if isinstance(element, ItemsElement):
            return _render_items(element)
        if isinstance(element, SourcesElement):
            return _render_sources(element, source_index=source_index)
        if isinstance(element, ContextElement):
            return _render_context(element)
    except ValueError as exc:
        logger.info(
            "dropping presentation element type=%s reason=%s",
            getattr(element, "type", "?"),
            exc,
        )
        return []
    return []


def _render_fields(element: FieldsElement) -> list[dict]:
    field_strings = [f"*{item.label}*\n{item.value}" for item in element.items]
    field_strings = field_strings[: blockkit.MAX_SECTION_FIELDS]
    title_text = f"*{element.title}*" if element.title else None
    return [blockkit.section(title_text, fields=field_strings)]


def _render_table(element: TableElement, *, surface: str) -> list[dict]:
    blocks: list[dict] = []
    if element.title:
        blocks.append(blockkit.section(f"*{element.title}*"))
    if surface in _TABLE_CAPABLE_SURFACES:
        rows = [list(element.columns), *[list(row) for row in element.rows]]
        blocks.append(blockkit.table(rows))
    else:
        # Native table blocks are message-only; degrade to a markdown table.
        blocks.append(blockkit.markdown_block(_markdown_table(element)))
    return blocks


def _render_cards(element: CardsElement) -> list[dict]:
    blocks: list[dict] = []
    if element.title:
        blocks.append(blockkit.section(f"*{element.title}*"))
    for item in element.items:
        blocks.extend(_render_card(item))
    return blocks


def _render_card(item: CardItem) -> list[dict]:
    """Render one card: a real card block when it fits, else section+fields."""

    body_lines: list[str] = []
    if item.body:
        body_lines.append(item.body)
    body_lines.extend(f"*{f.label}*: {f.value}" for f in item.fields)
    body = "\n".join(body_lines) or None
    try:
        if body is not None and len(body) <= blockkit.MAX_CARD_BODY_CHARS:
            return [blockkit.card(title=item.title, subtitle=item.subtitle, body=body)]
    except ValueError:
        pass
    # Fallback: a section header + fields (always renders, no length cliff).
    blocks: list[dict] = [blockkit.section(f"*{item.title}*")]
    if item.subtitle:
        blocks.append(blockkit.context(item.subtitle))
    if item.body:
        blocks.append(blockkit.section(item.body[: blockkit.MAX_SECTION_TEXT_CHARS]))
    if item.fields:
        field_strings = [f"*{f.label}*\n{f.value}" for f in item.fields][
            : blockkit.MAX_SECTION_FIELDS
        ]
        blocks.append(blockkit.section(fields=field_strings))
    return blocks


def _render_context(element: ContextElement) -> list[dict]:
    items = element.items[: blockkit.MAX_CONTEXT_ELEMENTS]
    return [blockkit.context(*items)]


def _render_items(element: ItemsElement) -> list[dict]:
    """Render an entity list the Slack-native way: per item a section (title +
    facts/body) and a context (meta), divider-separated. The default for lists."""

    blocks: list[dict] = []
    if element.title:
        blocks.append(blockkit.section(f"*{element.title}*"))
    for index, item in enumerate(element.items):
        if element.dividers and (blocks or index > 0):
            blocks.append(blockkit.divider())
        blocks.extend(_render_list_item(item))
    return blocks


def _render_list_item(item: ListItem) -> list[dict]:
    blocks: list[dict] = []
    lines = [f"*{item.title}*"]
    if item.body:
        lines.append(item.body)
    section_text = "\n".join(lines)[: blockkit.MAX_SECTION_TEXT_CHARS]
    field_strings = [f"*{f.label}*\n{f.value}" for f in item.facts][
        : blockkit.MAX_SECTION_FIELDS
    ]
    blocks.append(blockkit.section(section_text, fields=field_strings))
    if item.context:
        blocks.append(blockkit.context(*item.context[: blockkit.MAX_CONTEXT_ELEMENTS]))
    return blocks


def _render_sources(
    element: SourcesElement, *, source_index: SourceIndex | None
) -> list[dict]:
    """Render citations as linked source cards. The link target always comes from
    the server-built index — an unresolved ref is dropped, never guessed."""

    if source_index is None:
        return []
    resolved: list[tuple[dict, str]] = []  # (hint item display, resolved url+meta)
    cards: list[dict] = []
    for item in element.items:
        source = source_index.resolve(item.source_ref)
        if source is None:
            logger.info("dropping source: unresolved ref %s", item.source_ref)
            continue
        title = (item.title or source.domain).strip()
        subtitle = (item.subtitle or source.domain).strip()
        body_text = (item.body or source.snippet or "").strip()
        # Link lives in the card body as mrkdwn (no button) so it stays a pure
        # display link — no interaction payload to ack (slice 2 owns buttons).
        link = f"<{source.url}|Read →>"
        body = f"{body_text}\n{link}" if body_text else link
        body = body[: blockkit.MAX_CARD_BODY_CHARS]
        try:
            cards.append(blockkit.card(title=title, subtitle=subtitle, body=body))
        except ValueError:
            resolved.append(({"title": title, "body": body}, source.url))
    if not cards and not resolved:
        return []
    # Carousel needs real card blocks; fall back to stacked sections for any that
    # overflowed the card limits, and for the "stacked" display preference.
    if element.display == "carousel" and cards and not resolved:
        return [blockkit.carousel(*cards)]
    blocks: list[dict] = list(cards)
    for display, _url in resolved:
        blocks.append(blockkit.section(f"*{display['title']}*\n{display['body']}"))
    return blocks


def _markdown_table(element: TableElement) -> str:
    header = "| " + " | ".join(element.columns) + " |"
    sep = "| " + " | ".join("---" for _ in element.columns) + " |"
    body = "\n".join("| " + " | ".join(row) + " |" for row in element.rows)
    return f"{header}\n{sep}\n{body}"


def _enforce_message_limits(blocks: list[dict], *, voice: dict) -> list[dict] | None:
    """Trim to the 50-block message limit, always keeping the voice block."""

    if len(blocks) <= blockkit.MAX_MESSAGE_BLOCKS:
        return blocks
    # Keep the voice block + as many leading element blocks as fit.
    trimmed = blocks[: blockkit.MAX_MESSAGE_BLOCKS]
    if voice not in trimmed:
        trimmed = [voice, *blocks[1 : blockkit.MAX_MESSAGE_BLOCKS]]
    logger.info(
        "presentation trimmed to message block limit kept=%s dropped=%s",
        len(trimmed),
        len(blocks) - len(trimmed),
    )
    return trimmed


__all__ = ["MESSAGE_SURFACE", "render_blocks"]
