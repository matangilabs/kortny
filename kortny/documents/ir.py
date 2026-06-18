"""Document Studio intermediate representation (HIG-244).

A single structured, format-agnostic document spec — modelled on the
Pandoc-AST pattern (metadata + a typed list of blocks) — that a backend
writer fans out to a concrete format. Phase 1 targets editorial-grade PDF via
the Typst writer; the same IR is the seam PPTX/DOCX/XLSX writers attach to
later.

The agent emits this IR as JSON (never Typst), so authoring stays in a stable,
validated schema and the engine underneath can change without touching the
model-facing contract. ``doc_kind`` carries intent (pitch vs report vs memo);
density and the default theme key off it rather than off the output format.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class DocKind(StrEnum):
    """The document archetype — drives default theme and information density.

    Beauty-vs-information is intent-driven, not format-driven: a ``pitch`` and a
    ``report`` can both render to PDF at very different densities. Keeping the
    archetype explicit lets one engine serve both.
    """

    pitch = "pitch"
    report = "report"
    brief = "brief"
    memo = "memo"


class _Block(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CoverHeader(_Block):
    """Opening title block — eyebrow, large title, subtitle, meta line."""

    type: Literal["cover_header"] = "cover_header"
    eyebrow: str | None = None
    title: str
    subtitle: str | None = None
    # One word of the title can be accent-coloured (Viktor's "Investors." move).
    accent_tail: str | None = None
    meta: list[str] = Field(default_factory=list)


class SectionDivider(_Block):
    """Full-section break — numbered index, label, title, optional subtitle.

    Rendered as its own page; on dark themes it becomes a full-bleed dark page,
    the alternating-rhythm device editorial decks use.
    """

    type: Literal["section_divider"] = "section_divider"
    index: str | None = None
    label: str | None = None
    title: str
    subtitle: str | None = None


class Heading(_Block):
    type: Literal["heading"] = "heading"
    text: str


class Prose(_Block):
    """Body copy. Double newlines split paragraphs."""

    type: Literal["prose"] = "prose"
    text: str


class StatCard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str
    label: str
    note: str | None = None


class StatCards(_Block):
    type: Literal["stat_cards"] = "stat_cards"
    cards: list[StatCard] = Field(min_length=1, max_length=4)


class Table(_Block):
    type: Literal["table"] = "table"
    caption: str | None = None
    columns: list[str] = Field(min_length=1)
    rows: list[list[str]] = Field(default_factory=list)


class Callout(_Block):
    """Highlighted aside — label + body, left accent border on a tinted panel."""

    type: Literal["callout"] = "callout"
    label: str | None = None
    text: str


class PullQuote(_Block):
    type: Literal["pull_quote"] = "pull_quote"
    text: str
    attribution: str | None = None


class CTA(_Block):
    """Call-to-action — a solid pill with mono label, optional supporting line."""

    type: Literal["cta"] = "cta"
    label: str
    text: str | None = None


class ChartPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Category label or numeric position on the x axis.
    x: str | float
    y: float


class ChartSeries(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    points: list[ChartPoint] = Field(min_length=1)


class Chart(_Block):
    """A data visualisation.

    The agent emits this compact, constrained shape — never raw Vega-Lite — and
    the writer compiles it to a themed Vega-Lite spec rendered to an image
    (vector for PDF, raster for Office). This keeps the agent off the
    hallucination-prone Vega-Lite surface and lets theming / colour-blind-safe
    palettes / chart-type curation be applied deterministically.
    """

    type: Literal["chart"] = "chart"
    chart_type: Literal["bar", "line", "area", "pie", "scatter"] = "bar"
    title: str | None = None
    x_label: str | None = None
    y_label: str | None = None
    series: list[ChartSeries] = Field(min_length=1, max_length=8)
    caption: str | None = None


Block = Annotated[
    CoverHeader
    | SectionDivider
    | Heading
    | Prose
    | StatCards
    | Table
    | Callout
    | PullQuote
    | CTA
    | Chart,
    Field(discriminator="type"),
]


class DocumentSpec(BaseModel):
    """The full document: intent, metadata, optional theme override, blocks."""

    model_config = ConfigDict(extra="forbid")

    doc_kind: DocKind = DocKind.report
    title: str
    # Theme is resolved from doc_kind when omitted; named override otherwise.
    theme: str | None = None
    blocks: list[Block] = Field(min_length=1)
