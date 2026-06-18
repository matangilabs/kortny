"""Render a Document Studio IR to Typst source (HIG-244 Phase 1).

Typst is the beauty engine: a no-browser compiler purpose-built for paged,
editorial PDF. Two structural choices here directly fix the failure modes seen
in the WeasyPrint baseline:

* Panels (callouts, cards) are content-sized Typst ``block``/``box`` — they
  shrink to their content, so a short panel can never leave the giant empty
  tinted void the HTML-background path produced.
* Section dividers and the cover are full-bleed *pages* (``#page(fill: ...)``),
  not background ``<div>``s, so the dark rhythm is real pagination rather than a
  fixed-height box.

The agent never sees this output — it emits the JSON IR and this writer turns
it into Typst deterministically.
"""

from __future__ import annotations

from kortny.documents.charts import render_chart_svg
from kortny.documents.ir import (
    CTA,
    Callout,
    Chart,
    CoverHeader,
    DocumentSpec,
    Heading,
    Prose,
    PullQuote,
    SectionDivider,
    StatCards,
    Table,
)
from kortny.documents.themes import Theme, resolve_theme

# Typst markup sigils that must be escaped in text placed inside [ ... ].
_TYPST_SIGILS = ("$", "#", "[", "]", "*", "_", "`", "@", "<", ">")


def esc(text: str) -> str:
    """Escape Typst markup-significant characters in free text.

    Backslash first (so we don't double-escape the escapes we add), then the
    markup sigils. ``$`` is the big one — it opens math mode and silently breaks
    compilation on values like ``$15B``.
    """

    text = text.replace("\\", "\\\\")
    for sigil in _TYPST_SIGILS:
        text = text.replace(sigil, "\\" + sigil)
    return text


def build_typst(spec: DocumentSpec) -> tuple[str, dict[str, bytes]]:
    """Render ``spec`` to Typst source plus any side-car assets (chart SVGs).

    Charts compile to a themed Vega-Lite spec rendered to an SVG that the source
    references by filename; the assets dict maps filename -> bytes so the render
    layer can drop them next to the source in a compile dir.
    """

    theme = resolve_theme(doc_kind=spec.doc_kind, name=spec.theme)
    assets: dict[str, bytes] = {}
    parts = [_preamble(spec, theme)]
    for index, block in enumerate(spec.blocks):
        parts.append(_render_block(block, theme, index, assets))
    return "\n".join(parts) + "\n", assets


def render_document(spec: DocumentSpec) -> str:
    """Render ``spec`` to Typst source (without assets — used by tests)."""

    return build_typst(spec)[0]


def _preamble(spec: DocumentSpec, theme: Theme) -> str:
    c = theme.colors
    margin = f"{theme.page_margin_mm}mm"
    justify = "true" if theme.justify_body else "false"
    brand = esc(spec.title)
    return f"""\
#let ink = rgb("{c.ink}")
#let paper = rgb("{c.paper}")
#let accent = rgb("{c.accent}")
#let muted = rgb("{c.muted}")
#let card = rgb("{c.card}")
#let rule = rgb("{c.rule}")
#let onDark = rgb("{c.on_dark}")
#let onDarkMuted = rgb("{c.on_dark_muted}")
#let dividerBg = rgb("{c.divider_bg}")
#let panelTint = rgb("{theme.panel_tint}")
#let display = "{theme.display_font}"
#let bodyFont = "{theme.body_font}"
#let mono = "{theme.mono_font}"
#let dispWeight = {theme.display_weight}

#let eyebrow(t, c: accent) = text(font: mono, size: 8.5pt, fill: c, tracking: 2.5pt)[#upper(t)]

#set document(title: "{brand}")
#set page(
  width: 210mm, height: 297mm, margin: {margin}, fill: paper,
  footer: context [
    #set text(font: mono, size: 7pt, fill: muted)
    #upper("{brand}") #h(1fr) #counter(page).display()
  ],
)
#set par(justify: {justify}, leading: 0.68em)
#set text(font: bodyFont, size: 11pt, fill: ink)
"""


def _render_block(
    block: object, theme: Theme, index: int, assets: dict[str, bytes]
) -> str:
    if isinstance(block, CoverHeader):
        return _cover(block, theme)
    if isinstance(block, SectionDivider):
        return _divider(block, theme)
    if isinstance(block, Heading):
        return _heading(block)
    if isinstance(block, Prose):
        return _prose(block, theme)
    if isinstance(block, StatCards):
        return _stat_cards(block)
    if isinstance(block, Table):
        return _table(block)
    if isinstance(block, Callout):
        return _callout(block)
    if isinstance(block, PullQuote):
        return _pull_quote(block)
    if isinstance(block, CTA):
        return _cta(block)
    if isinstance(block, Chart):
        return _chart(block, theme, index, assets)
    return ""


def _chart(block: Chart, theme: Theme, index: int, assets: dict[str, bytes]) -> str:
    filename = f"chart_{index}.svg"
    assets[filename] = render_chart_svg(block, theme).encode("utf-8")
    lines = [f'#image("{filename}", width: 100%)']
    if block.caption:
        lines.append(
            f"#v(4pt)\n#text(font: mono, size: 8pt, fill: muted, "
            f'tracking: 1pt)[#upper("{esc(block.caption)}")]'
        )
    lines.append("#v(14pt)")
    return "\n".join(lines)


def _title_markup(title: str, accent_tail: str | None) -> str:
    """Title with an optional accent-coloured trailing fragment."""

    if accent_tail and title.endswith(accent_tail):
        head = title[: -len(accent_tail)]
        return f"{esc(head)}#text(fill: accent)[{esc(accent_tail)}]"
    return esc(title)


def _cover(block: CoverHeader, theme: Theme) -> str:
    fill = "dividerBg" if theme.dark_dividers else "paper"
    fg = "onDark" if theme.dark_dividers else "ink"
    fg_muted = "onDarkMuted" if theme.dark_dividers else "muted"
    margin = f"{theme.page_margin_mm + 4}mm"
    lines = [
        f"#page(fill: {fill}, header: none, footer: none, margin: {margin})[",
        # Display titles must not hyphenate ("EV Mar-ket Report") or justify
        # (which spreads a 2-line title); this page is all short display text.
        "  #set text(hyphenate: false)",
        "  #set par(justify: false)",
        "  #v(1fr)",
    ]
    if block.eyebrow:
        lines.append(f'  #eyebrow("{esc(block.eyebrow)}")')
        lines.append("  #v(12pt)")
    lines.append(
        f"  #text(font: display, size: 50pt, weight: dispWeight, fill: {fg})"
        f"[{_title_markup(block.title, block.accent_tail)}]"
    )
    if block.subtitle:
        lines.append("  #v(12pt)")
        lines.append(
            f"  #block(width: 64%)[#text(font: bodyFont, size: 13pt, "
            f"fill: {fg_muted})[{esc(block.subtitle)}]]"
        )
    lines.append("  #v(22pt)")
    lines.append("  #line(length: 36pt, stroke: 1pt + accent)")
    if block.meta:
        meta = "  ·  ".join(esc(m) for m in block.meta)
        lines.append("  #v(10pt)")
        lines.append(f'  #eyebrow("{meta}", c: {fg_muted})')
    lines.append("  #v(2fr)")
    lines.append("]")
    return "\n".join(lines)


def _divider(block: SectionDivider, theme: Theme) -> str:
    if theme.dark_dividers:
        fill, fg, fg_muted = "dividerBg", "onDark", "onDarkMuted"
    else:
        fill, fg, fg_muted = "paper", "ink", "muted"
    margin = f"{theme.page_margin_mm + 4}mm"
    lines = [
        f"#page(fill: {fill}, header: none, footer: none, margin: {margin})[",
        "  #set text(hyphenate: false)",
        "  #set par(justify: false)",
        "  #v(1fr)",
    ]
    if block.index:
        lines.append(
            f"  #text(font: mono, size: 13pt, fill: accent, "
            f"tracking: 2pt)[{esc(block.index)}]"
        )
        lines.append("  #v(6pt)")
    if block.label:
        lines.append(f'  #eyebrow("{esc(block.label)}", c: {fg_muted})')
        lines.append("  #v(8pt)")
    lines.append(
        f"  #text(font: display, size: 34pt, weight: dispWeight, "
        f"fill: {fg})[{esc(block.title)}]"
    )
    if block.subtitle:
        lines.append("  #v(12pt)")
        lines.append(
            f"  #block(width: 60%)[#text(font: bodyFont, size: 13pt, "
            f"fill: {fg_muted})[{esc(block.subtitle)}]]"
        )
    lines.append("  #v(2fr)")
    lines.append("]")
    return "\n".join(lines)


def _heading(block: Heading) -> str:
    return (
        f"#block(above: 18pt, below: 8pt)[\n"
        f"  #text(font: mono, size: 10pt, weight: 700, fill: ink, "
        f'tracking: 1.5pt)[#upper("{esc(block.text)}")]\n'
        f"  #v(3pt)\n"
        f"  #line(length: 34pt, stroke: 2pt + accent)\n"
        f"]"
    )


def _prose(block: Prose, theme: Theme) -> str:
    paragraphs = [p.strip() for p in block.text.split("\n\n") if p.strip()]
    out = []
    for i, para in enumerate(paragraphs):
        out.append(f"#text(font: bodyFont, size: 11pt, fill: ink)[{esc(para)}]")
        if i < len(paragraphs) - 1:
            out.append("#v(8pt)")
    out.append("#v(14pt)")
    return "\n".join(out)


def _stat_cards(block: StatCards) -> str:
    n = len(block.cards)
    cols = ", ".join(["1fr"] * n)
    cells = []
    for card in block.cards:
        note = (
            f"\n    #v(2pt)\n    #text(font: mono, size: 7.5pt, "
            f"fill: muted)[{esc(card.note)}]"
            if card.note
            else ""
        )
        cells.append(
            f"box(fill: card, inset: 13pt, radius: 3pt, width: 100%)[\n"
            f"    #text(font: display, size: 28pt, weight: dispWeight, "
            f"fill: accent)[{esc(card.value)}]\n"
            f"    #v(3pt)\n"
            f"    #text(font: bodyFont, size: 11pt, weight: 700, "
            f"fill: ink)[{esc(card.label)}]{note}\n"
            f"  ]"
        )
    grid_cells = ",\n  ".join(cells)
    return f"#grid(columns: ({cols}), gutter: 12pt,\n  {grid_cells}\n)\n#v(16pt)"


def _table(block: Table) -> str:
    ncol = len(block.columns)
    cols = ", ".join(["1fr"] * ncol)
    header = ", ".join(
        f"table.cell(fill: ink)[#text(fill: white, font: mono, size: 8pt, "
        f'weight: 700, tracking: 1pt)[#upper("{esc(col)}")]]'
        for col in block.columns
    )
    body_cells = []
    for row in block.rows:
        # Pad/truncate ragged rows to the column count so the grid stays valid.
        padded = (list(row) + [""] * ncol)[:ncol]
        body_cells.append(
            ", ".join(
                f"[#text(font: bodyFont, size: 10pt)[{esc(str(v))}]]" for v in padded
            )
        )
    body = ",\n  ".join(body_cells)
    parts = []
    if block.caption:
        parts.append(
            f"#text(font: mono, size: 8pt, fill: muted, "
            f'tracking: 1pt)[#upper("{esc(block.caption)}")]'
        )
        parts.append("#v(4pt)")
    table_body = f",\n  {body}" if body else ""
    parts.append(
        f"#table(columns: ({cols}), stroke: 0.5pt + rule, inset: 8pt,\n"
        f"  {header}{table_body}\n)"
    )
    parts.append("#v(14pt)")
    return "\n".join(parts)


def _callout(block: Callout) -> str:
    lines = [
        "#block(width: 100%, fill: panelTint, inset: 18pt, radius: 2pt,",
        "  stroke: (left: 3pt + accent))[",
    ]
    if block.label:
        lines.append(f'  #eyebrow("{esc(block.label)}")\n  #v(7pt)')
    lines.append(f"  #text(font: bodyFont, size: 12pt, fill: ink)[{esc(block.text)}]")
    lines.append("]")
    lines.append("#v(14pt)")
    return "\n".join(lines)


def _pull_quote(block: PullQuote) -> str:
    lines = [
        "#block(inset: (left: 16pt), stroke: (left: 3pt + accent))[",
        f'  #text(font: display, size: 17pt, style: "italic", '
        f"fill: ink)[{esc(block.text)}]",
    ]
    if block.attribution:
        lines.append("  #v(5pt)")
        lines.append(
            f"  #text(font: mono, size: 8pt, fill: muted, "
            f'tracking: 1pt)[#upper("— {esc(block.attribution)}")]'
        )
    lines.append("]")
    lines.append("#v(14pt)")
    return "\n".join(lines)


def _cta(block: CTA) -> str:
    lines = []
    if block.text:
        lines.append(
            f"#text(font: bodyFont, size: 12pt, fill: muted)[{esc(block.text)}]"
        )
        lines.append("#v(8pt)")
    lines.append(
        f"#box(fill: ink, inset: (x: 18pt, y: 11pt), radius: 2pt)["
        f"#text(font: mono, size: 9pt, fill: white, "
        f'tracking: 1pt)[#upper("{esc(block.label)}")]]'
    )
    lines.append("#v(14pt)")
    return "\n".join(lines)
