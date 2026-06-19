"""Presentation hint schema for the humanizer → Block Kit renderer (HIG-255).

The product principle (HIG-235) is "voice = prose, data = Block Kit; the LLM
never authors Block Kit JSON." This module is the narrow seam that lets the LLM
express *presentation intent* without authoring Slack JSON: the humanizer may
emit a small, constrained ``presentation`` hint alongside its prose, and
deterministic code (``response_render``) turns that hint into validated blocks.

Slice 1 covers display-only elements — fields, table, context, cards — where the
hint's content is display text the LLM is already trusted to author (the same
trust as the prose). Interactive elements (buttons/selects/modals) and
server-resolved refs (approval keys, IDs, URLs that carry authority) are NOT in
this schema yet: they arrive with the interactivity slice, which binds them to
server-owned records rather than LLM-authored values.

Parsing is lenient by design: an unknown element type or an element that fails
validation is dropped, never fatal — a bad hint must degrade to prose, never
drop the answer.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

logger = logging.getLogger(__name__)

# Slice-1 element-count guard: a hint is a presentation aid, not a second UI
# framework. Excess elements past this are dropped (over-formatting guard).
MAX_PRESENTATION_ELEMENTS = 8


class FieldItem(BaseModel):
    """A single label/value pair for a fields or card element."""

    model_config = ConfigDict(extra="forbid")

    label: str
    value: str


class FieldsElement(BaseModel):
    """Key-value facts/metrics/status rendered as a section with fields."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["fields"] = "fields"
    title: str | None = None
    items: list[FieldItem] = Field(min_length=1)


class TableElement(BaseModel):
    """Tabular data rendered as a native Slack table block (message-only)."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["table"] = "table"
    title: str | None = None
    columns: list[str] = Field(min_length=1)
    rows: list[list[str]] = Field(min_length=1)


class CardItem(BaseModel):
    """A discrete entity (issue/PR/schedule/etc.) shown as a card."""

    model_config = ConfigDict(extra="forbid")

    title: str
    subtitle: str | None = None
    body: str | None = None
    fields: list[FieldItem] = Field(default_factory=list)


class CardsElement(BaseModel):
    """One or more discrete entities, rendered as stacked cards."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["cards"] = "cards"
    title: str | None = None
    items: list[CardItem] = Field(min_length=1)


class ContextElement(BaseModel):
    """Provenance / freshness / source footnotes rendered as a context block."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["context"] = "context"
    items: list[str] = Field(min_length=1)


class ListItem(BaseModel):
    """One entity in an ``items`` list (a title + a few facts + a meta line)."""

    model_config = ConfigDict(extra="forbid")

    title: str
    facts: list[FieldItem] = Field(default_factory=list, max_length=6)
    body: str | None = None
    context: list[str] = Field(default_factory=list, max_length=3)


class ItemsElement(BaseModel):
    """A list of entities rendered the Slack-native way: section + context per
    item, divider-separated. The default for "list my X" answers — readable in
    narrow threads and budget-friendly, unlike a wall of cards."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["items"] = "items"
    title: str | None = None
    items: list[ListItem] = Field(min_length=1, max_length=12)
    dividers: bool = True


class SourceCardItem(BaseModel):
    """A citation. The LLM gives a ``source_ref`` (and optional display copy);
    the real URL is resolved server-side from evidence — never LLM-authored, so
    a hallucinated/phishing link can't reach the user. ``extra="forbid"`` means a
    stray ``url`` key invalidates (and drops) the item."""

    model_config = ConfigDict(extra="forbid")

    source_ref: str
    title: str | None = None
    subtitle: str | None = None
    body: str | None = None


class SourcesElement(BaseModel):
    """Citations/references the answer draws on, rendered as linked source cards
    (links-only). Trust rendering, not decoration — only when the answer cites
    real sources."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["sources"] = "sources"
    title: str | None = None
    items: list[SourceCardItem] = Field(min_length=1, max_length=10)
    display: Literal["carousel", "stacked"] = "carousel"


PresentationElement = Annotated[
    FieldsElement
    | TableElement
    | CardsElement
    | ContextElement
    | ItemsElement
    | SourcesElement,
    Field(discriminator="type"),
]

# The element types this schema understands today. Unknown types in a hint are
# dropped (forward-compatible with hints authored against a newer vocabulary).
KNOWN_ELEMENT_TYPES = frozenset(
    {"fields", "table", "cards", "context", "items", "sources"}
)


class PresentationHint(BaseModel):
    """The humanizer's optional presentation intent for one response."""

    model_config = ConfigDict(extra="forbid")

    version: int = 1
    elements: list[PresentationElement] = Field(default_factory=list)


def parse_presentation(data: Any) -> PresentationHint | None:
    """Parse a presentation hint leniently, dropping unknown/invalid elements.

    Returns ``None`` when there is nothing usable. Never raises — a malformed
    hint must degrade to prose-only, never break the response. Each element is
    validated independently so one bad element doesn't discard the good ones.
    """

    if not isinstance(data, dict):
        return None
    raw_elements = data.get("elements")
    if not isinstance(raw_elements, list):
        return None

    kept: list[PresentationElement] = []
    for raw in raw_elements:
        if not isinstance(raw, dict):
            continue
        element_type = raw.get("type")
        if element_type not in KNOWN_ELEMENT_TYPES:
            # Forward-compatible: a hint may reference an element we don't render
            # yet (e.g. a future "chart"); drop it rather than fail the hint.
            continue
        try:
            kept.append(_validate_element(raw))
        except ValidationError as exc:
            logger.info(
                "dropping invalid presentation element type=%s error=%s",
                element_type,
                exc.error_count(),
            )
        if len(kept) >= MAX_PRESENTATION_ELEMENTS:
            break

    if not kept:
        return None
    version = data.get("version")
    return PresentationHint(
        version=version if isinstance(version, int) else 1,
        elements=kept,
    )


_ELEMENT_MODELS: dict[str, type[BaseModel]] = {
    "fields": FieldsElement,
    "table": TableElement,
    "cards": CardsElement,
    "context": ContextElement,
    "items": ItemsElement,
    "sources": SourcesElement,
}


def _validate_element(raw: dict[str, Any]) -> PresentationElement:
    model = _ELEMENT_MODELS[raw["type"]]
    return model.model_validate(raw)  # type: ignore[return-value]


__all__ = [
    "CardItem",
    "CardsElement",
    "ContextElement",
    "FieldItem",
    "FieldsElement",
    "ItemsElement",
    "ListItem",
    "PresentationElement",
    "PresentationHint",
    "SourceCardItem",
    "SourcesElement",
    "TableElement",
    "parse_presentation",
]
