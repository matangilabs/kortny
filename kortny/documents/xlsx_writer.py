"""Render a Document Studio IR to an XLSX workbook (HIG-244 close-out).

XLSX is a *data export* of the document, not a paginated reproduction: the
data-bearing blocks (tables, stat cards, chart data) become sheets, and the
narrative blocks (cover, headings, prose, callouts) collapse onto a Summary
sheet so nothing is silently dropped. A spreadsheet is the wrong surface for a
prose-heavy doc — ``xlsx_is_poor_fit`` lets the tool warn the agent, but the
writer always produces a valid workbook.
"""

from __future__ import annotations

import io
import re

import xlsxwriter

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

_MAX_SHEET_NAME = 31
_INVALID_SHEET_CHARS = re.compile(r"[\[\]:*?/\\]")
# xlsxwriter chart_type per IR chart_type.
_CHART_TYPES = {
    "bar": "column",
    "line": "line",
    "area": "area",
    "pie": "pie",
    "scatter": "scatter",
}


def render_xlsx(spec: DocumentSpec) -> bytes:
    """Render ``spec`` to an .xlsx workbook as bytes."""

    theme = resolve_theme(doc_kind=spec.doc_kind, name=spec.theme)
    buffer = io.BytesIO()
    workbook = xlsxwriter.Workbook(buffer, {"in_memory": True})
    fmts = _formats(workbook, theme)
    used_names: set[str] = set()

    summary = workbook.add_worksheet(_sheet_name("Summary", used_names))
    summary.set_column(0, 0, 24)
    summary.set_column(1, 3, 40)
    row = _write_summary_title(summary, spec, fmts)

    metrics_cards: list[StatCards] = []
    table_index = 0
    chart_index = 0
    for block in spec.blocks:
        if isinstance(block, CoverHeader):
            row = _write_cover(summary, block, fmts, row)
        elif isinstance(block, SectionDivider):
            row = _write_summary_line(
                summary, _divider_label(block), fmts["section"], row
            )
        elif isinstance(block, Heading):
            row = _write_summary_line(summary, block.text, fmts["section"], row)
        elif isinstance(block, Prose):
            row = _write_prose(summary, block, fmts, row)
        elif isinstance(block, Callout):
            label = f"{block.label}: " if block.label else ""
            row = _write_summary_line(
                summary, f"{label}{block.text}", fmts["note"], row
            )
        elif isinstance(block, PullQuote):
            attr = f" — {block.attribution}" if block.attribution else ""
            row = _write_summary_line(
                summary, f"“{block.text}”{attr}", fmts["note"], row
            )
        elif isinstance(block, CTA):
            line = block.label + (f": {block.text}" if block.text else "")
            row = _write_summary_line(summary, line, fmts["note"], row)
        elif isinstance(block, StatCards):
            metrics_cards.append(block)
        elif isinstance(block, Table):
            table_index += 1
            _write_table_sheet(workbook, block, table_index, fmts, used_names)
        elif isinstance(block, Chart):
            chart_index += 1
            _write_chart_sheet(workbook, block, chart_index, fmts, used_names)

    if metrics_cards:
        _write_metrics_sheet(workbook, metrics_cards, fmts, used_names)

    workbook.close()
    return buffer.getvalue()


def xlsx_is_poor_fit(spec: DocumentSpec) -> bool:
    """True when the doc is narrative-heavy and a spreadsheet adds little.

    Heuristic (codex/GLM): prose-dominant AND fewer than two data blocks. The
    tool surfaces this as a warning; it never auto-switches the format.
    """

    prose_like = sum(
        1
        for b in spec.blocks
        if isinstance(b, Prose | Heading | Callout | PullQuote | CTA | CoverHeader)
    )
    data_blocks = sum(
        1 for b in spec.blocks if isinstance(b, Table | Chart | StatCards)
    )
    return data_blocks < 2 and prose_like > len(spec.blocks) / 2


# -- sheet writers ----------------------------------------------------------


def _write_summary_title(summary, spec: DocumentSpec, fmts) -> int:  # type: ignore[no-untyped-def]
    summary.write(0, 0, spec.title, fmts["title"])
    return 2


def _write_cover(summary, block: CoverHeader, fmts, row: int) -> int:  # type: ignore[no-untyped-def]
    if block.eyebrow:
        summary.write(row, 0, block.eyebrow, fmts["eyebrow"])
        row += 1
    if block.subtitle:
        row = _write_summary_line(summary, block.subtitle, fmts["wrap"], row)
    for item in block.meta:
        row = _write_summary_line(summary, item, fmts["eyebrow"], row)
    return row + 1


def _write_prose(summary, block: Prose, fmts, row: int) -> int:  # type: ignore[no-untyped-def]
    for para in (p.strip() for p in block.text.split("\n\n") if p.strip()):
        row = _write_summary_line(summary, para[:2000], fmts["wrap"], row)
    return row


def _write_summary_line(summary, text: str, fmt, row: int) -> int:  # type: ignore[no-untyped-def]
    summary.write(row, 0, text, fmt)
    return row + 1


def _write_table_sheet(workbook, block: Table, index: int, fmts, used) -> None:  # type: ignore[no-untyped-def]
    sheet = workbook.add_worksheet(_sheet_name(f"Table {index}", used))
    sheet.set_column(0, max(len(block.columns) - 1, 0), 22)
    row = 0
    if block.caption:
        sheet.write(0, 0, block.caption, fmts["caption"])
        row = 2
    for col, header in enumerate(block.columns):
        sheet.write(row, col, header, fmts["header"])
    ncol = len(block.columns)
    for data_row in block.rows:
        row += 1
        padded = (list(data_row) + [""] * ncol)[:ncol]
        for col, cell in enumerate(padded):
            sheet.write(row, col, cell)
    sheet.freeze_panes(row - len(block.rows) + 1 if block.rows else 1, 0)


def _write_metrics_sheet(workbook, groups: list[StatCards], fmts, used) -> None:  # type: ignore[no-untyped-def]
    sheet = workbook.add_worksheet(_sheet_name("Metrics", used))
    sheet.set_column(0, 0, 30)
    sheet.set_column(1, 1, 20)
    sheet.set_column(2, 2, 40)
    for col, header in enumerate(("Label", "Value", "Note")):
        sheet.write(0, col, header, fmts["header"])
    row = 0
    for group in groups:
        for card in group.cards:
            row += 1
            sheet.write(row, 0, card.label)
            sheet.write(row, 1, card.value)
            sheet.write(row, 2, card.note or "")


def _write_chart_sheet(workbook, block: Chart, index: int, fmts, used) -> None:  # type: ignore[no-untyped-def]
    sheet = workbook.add_worksheet(_sheet_name(f"Chart {index} Data", used))
    sheet.set_column(0, len(block.series), 18)
    # Header: x + one column per series.
    sheet.write(0, 0, block.x_label or "x", fmts["header"])
    for col, series in enumerate(block.series, start=1):
        sheet.write(0, col, series.name, fmts["header"])
    # Use the first series' x values as the shared x column (long-form enough
    # for a spreadsheet; the data is the point, not a pixel-perfect chart).
    x_values = [p.x for p in block.series[0].points]
    for i, x in enumerate(x_values, start=1):
        sheet.write(i, 0, x)
    for col, series in enumerate(block.series, start=1):
        for i, point in enumerate(series.points, start=1):
            sheet.write(i, col, point.y)
    _add_native_chart(workbook, sheet, block, index, len(x_values))


def _add_native_chart(workbook, sheet, block: Chart, index: int, rows: int) -> None:  # type: ignore[no-untyped-def]
    if rows < 1:
        return
    try:
        chart = workbook.add_chart(
            {"type": _CHART_TYPES.get(block.chart_type, "column")}
        )
        sheet_name = sheet.get_name()
        for col, series in enumerate(block.series, start=1):
            chart.add_series(
                {
                    "name": series.name,
                    "categories": [sheet_name, 1, 0, rows, 0],
                    "values": [sheet_name, 1, col, rows, col],
                }
            )
        if block.title:
            chart.set_title({"name": block.title})
        sheet.insert_chart(1, len(block.series) + 2, chart)
    except Exception:  # noqa: BLE001 — a chart object is best-effort; data sheet stands alone
        return


# -- helpers ----------------------------------------------------------------


def _formats(workbook, theme: Theme):  # type: ignore[no-untyped-def]
    return {
        "title": workbook.add_format(
            {"bold": True, "font_size": 16, "font_color": theme.colors.ink}
        ),
        "eyebrow": workbook.add_format(
            {"italic": True, "font_color": theme.colors.accent}
        ),
        "section": workbook.add_format({"bold": True, "font_size": 12}),
        "caption": workbook.add_format({"bold": True, "italic": True}),
        "note": workbook.add_format({"italic": True, "text_wrap": True}),
        "wrap": workbook.add_format({"text_wrap": True, "valign": "top"}),
        "header": workbook.add_format(
            {
                "bold": True,
                "bg_color": theme.colors.accent,
                "font_color": theme.colors.paper,
                "border": 1,
            }
        ),
    }


def _divider_label(block: SectionDivider) -> str:
    prefix = f"{block.index}  " if block.index else ""
    return f"{prefix}{block.title}"


def _sheet_name(base: str, used: set[str]) -> str:
    name = _INVALID_SHEET_CHARS.sub("", base).strip()[:_MAX_SHEET_NAME] or "Sheet"
    if name not in used:
        used.add(name)
        return name
    counter = 2
    while True:
        suffix = f" ({counter})"
        candidate = name[: _MAX_SHEET_NAME - len(suffix)] + suffix
        if candidate not in used:
            used.add(candidate)
            return candidate
        counter += 1


__all__ = ["render_xlsx", "xlsx_is_poor_fit"]
