"""Deterministic document critique: lint + semantics-preserving auto-fix (HIG-244).

The vision-free half of the render→critique→revise "moat". It does NOT call an
LLM (that revise half is deferred until the deterministic issue reports prove
which defects are common, and would need its own LLMService instrumentation).

Two layers:

* :func:`critique_and_fix` — IR-level: detect defects and apply only the fixes
  that lose no information (pad ragged rows, drop empty blocks/cards, dedupe
  blank/duplicate columns, drop an invalid accent tail). Everything else is
  reported as a warning, never silently changed.
* :func:`validate_render` — post-render: the bytes are a real, non-empty file of
  the expected kind (PDF magic + sane page count; OOXML is a valid zip).

The tool runs lint+autofix before rendering and validates after, surfacing the
issue report in its output. A doc that lints down to nothing, or renders to an
invalid file, is a recoverable error the agent can correct.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel

from kortny.documents.ir import (
    Block,
    Chart,
    CoverHeader,
    DocumentSpec,
    Heading,
    Prose,
    SectionDivider,
    StatCard,
    StatCards,
    Table,
)

Severity = Literal["error", "warning", "info"]
AutofixState = Literal["applied", "available", "none"]

# Soft caps — beyond these we warn (truncation is lossy, so it's not auto-applied).
_MAX_CARD_VALUE_CHARS = 40
_MAX_TABLE_CELL_CHARS = 500
_MAX_PDF_PAGES = 100


class DocumentIssue(BaseModel):
    code: str
    severity: Severity
    message: str
    block_index: int | None = None
    autofix: AutofixState = "none"


@dataclass(frozen=True, slots=True)
class CritiqueResult:
    spec: DocumentSpec
    issues: list[DocumentIssue]

    @property
    def has_errors(self) -> bool:
        return any(i.severity == "error" for i in self.issues)

    def summary(self) -> dict[str, object]:
        return {
            "issues": len(self.issues),
            "autofixes": sum(1 for i in self.issues if i.autofix == "applied"),
            "errors": sum(1 for i in self.issues if i.severity == "error"),
            "warnings": sum(1 for i in self.issues if i.severity == "warning"),
            "codes": sorted({i.code for i in self.issues}),
        }


def critique_and_fix(spec: DocumentSpec) -> CritiqueResult:
    """Lint the spec and apply semantics-preserving auto-fixes."""

    issues: list[DocumentIssue] = []
    new_blocks: list[Block] = []
    for index, block in enumerate(spec.blocks):
        fixed, block_issues = _fix_block(block, index)
        issues.extend(block_issues)
        if fixed is not None:
            new_blocks.append(fixed)
    issues.extend(_structural_lint(new_blocks))
    if not new_blocks:
        issues.append(
            DocumentIssue(
                code="empty_document",
                severity="error",
                message="No renderable blocks remain after dropping empty ones.",
            )
        )
    fixed_spec = spec.model_copy(update={"blocks": new_blocks})
    return CritiqueResult(spec=fixed_spec, issues=issues)


def validate_render(data: bytes, fmt: str) -> list[DocumentIssue]:
    """Post-render sanity: the bytes are a real, non-empty file of the right kind."""

    if not data:
        return [
            DocumentIssue(
                code="empty_render",
                severity="error",
                message="Render produced no bytes.",
            )
        ]
    if fmt == "pdf":
        return _validate_pdf(data)
    return _validate_ooxml(data, fmt)


# -- block-level fix/lint ---------------------------------------------------


def _fix_block(block: Block, index: int) -> tuple[Block | None, list[DocumentIssue]]:
    if isinstance(block, Table):
        return _fix_table(block, index)
    if isinstance(block, StatCards):
        return _fix_stat_cards(block, index)
    if isinstance(block, CoverHeader):
        return _fix_cover(block, index)
    if isinstance(block, Heading) and not block.text.strip():
        return None, [
            DocumentIssue(
                code="empty_heading",
                severity="warning",
                message="Dropped a heading with no text.",
                block_index=index,
                autofix="applied",
            )
        ]
    if isinstance(block, Prose) and not block.text.strip():
        return None, [
            DocumentIssue(
                code="empty_prose",
                severity="warning",
                message="Dropped an empty prose block.",
                block_index=index,
                autofix="applied",
            )
        ]
    if isinstance(block, Chart):
        return block, _lint_chart(block, index)
    return block, []


def _fix_table(block: Table, index: int) -> tuple[Block | None, list[DocumentIssue]]:
    issues: list[DocumentIssue] = []
    ncol = len(block.columns)

    # Dedupe blank/duplicate column names → Column N.
    seen: set[str] = set()
    columns: list[str] = []
    renamed = False
    for position, name in enumerate(block.columns, start=1):
        candidate = name.strip()
        if not candidate or candidate in seen:
            candidate = f"Column {position}"
            renamed = True
        seen.add(candidate)
        columns.append(candidate)
    if renamed:
        issues.append(
            DocumentIssue(
                code="table_columns_renamed",
                severity="info",
                message="Renamed blank/duplicate table columns.",
                block_index=index,
                autofix="applied",
            )
        )

    if not block.rows:
        issues.append(
            DocumentIssue(
                code="empty_table",
                severity="warning",
                message="Dropped a table with columns but no rows.",
                block_index=index,
                autofix="applied",
            )
        )
        return None, issues

    # Pad/truncate ragged rows to the column count.
    rows: list[list[str]] = []
    ragged = False
    for row in block.rows:
        if len(row) != ncol:
            ragged = True
        rows.append((list(row) + [""] * ncol)[:ncol])
    if ragged:
        issues.append(
            DocumentIssue(
                code="ragged_table_rows",
                severity="warning",
                message="Padded ragged table rows to the column count.",
                block_index=index,
                autofix="applied",
            )
        )
    if any(len(cell) > _MAX_TABLE_CELL_CHARS for row in rows for cell in row):
        issues.append(
            DocumentIssue(
                code="long_table_cell",
                severity="warning",
                message=f"A table cell exceeds {_MAX_TABLE_CELL_CHARS} chars.",
                block_index=index,
                autofix="none",
            )
        )
    return block.model_copy(update={"columns": columns, "rows": rows}), issues


def _fix_stat_cards(
    block: StatCards, index: int
) -> tuple[Block | None, list[DocumentIssue]]:
    issues: list[DocumentIssue] = []
    kept: list[StatCard] = []
    for card in block.cards:
        if not card.value.strip() or not card.label.strip():
            issues.append(
                DocumentIssue(
                    code="empty_stat_card",
                    severity="warning",
                    message="Dropped a stat card with no value/label.",
                    block_index=index,
                    autofix="applied",
                )
            )
            continue
        if len(card.value) > _MAX_CARD_VALUE_CHARS:
            issues.append(
                DocumentIssue(
                    code="long_stat_value",
                    severity="warning",
                    message="A stat-card value is too long for a metric tile.",
                    block_index=index,
                    autofix="none",
                )
            )
        kept.append(card)
    if not kept:
        issues.append(
            DocumentIssue(
                code="empty_stat_cards",
                severity="warning",
                message="Dropped a stat-cards block with no usable cards.",
                block_index=index,
                autofix="applied",
            )
        )
        return None, issues
    return block.model_copy(update={"cards": kept}), issues


def _fix_cover(block: CoverHeader, index: int) -> tuple[Block, list[DocumentIssue]]:
    if block.accent_tail and not block.title.endswith(block.accent_tail):
        return block.model_copy(update={"accent_tail": None}), [
            DocumentIssue(
                code="invalid_accent_tail",
                severity="info",
                message="Dropped an accent_tail that isn't a suffix of the title.",
                block_index=index,
                autofix="applied",
            )
        ]
    return block, []


def _lint_chart(block: Chart, index: int) -> list[DocumentIssue]:
    issues: list[DocumentIssue] = []
    if block.chart_type == "pie" and len(block.series) > 1:
        issues.append(
            DocumentIssue(
                code="pie_multi_series",
                severity="warning",
                message="A pie chart has multiple series; only the first is meaningful.",
                block_index=index,
            )
        )
    if block.chart_type in ("bar", "line", "area"):
        lengths = {len(series.points) for series in block.series}
        if len(lengths) > 1:
            issues.append(
                DocumentIssue(
                    code="chart_series_length_mismatch",
                    severity="warning",
                    message="Chart series have differing point counts.",
                    block_index=index,
                )
            )
    return issues


def _structural_lint(blocks: list[Block]) -> list[DocumentIssue]:
    issues: list[DocumentIssue] = []
    for i, block in enumerate(blocks):
        if isinstance(block, Heading):
            nxt = blocks[i + 1] if i + 1 < len(blocks) else None
            if nxt is None or isinstance(nxt, Heading | SectionDivider):
                issues.append(
                    DocumentIssue(
                        code="orphan_heading",
                        severity="warning",
                        message="A heading is not followed by content.",
                        block_index=i,
                    )
                )
        if isinstance(block, SectionDivider) and i == 0:
            issues.append(
                DocumentIssue(
                    code="leading_divider",
                    severity="info",
                    message="A section divider is the first block.",
                    block_index=i,
                )
            )
    return issues


# -- post-render ------------------------------------------------------------


def _validate_pdf(data: bytes) -> list[DocumentIssue]:
    if not data.startswith(b"%PDF"):
        return [
            DocumentIssue(
                code="invalid_pdf", severity="error", message="Output is not a PDF."
            )
        ]
    try:
        from pypdf import PdfReader

        pages = len(PdfReader(io.BytesIO(data)).pages)
    except Exception:  # noqa: BLE001 — page count is best-effort
        return []
    if pages == 0:
        return [
            DocumentIssue(
                code="empty_pdf", severity="error", message="PDF has no pages."
            )
        ]
    if pages > _MAX_PDF_PAGES:
        return [
            DocumentIssue(
                code="pdf_too_long",
                severity="warning",
                message=f"PDF has {pages} pages (> {_MAX_PDF_PAGES}).",
            )
        ]
    return []


def _validate_ooxml(data: bytes, fmt: str) -> list[DocumentIssue]:
    if data[:2] != b"PK":
        return [
            DocumentIssue(
                code="invalid_ooxml",
                severity="error",
                message=f"Output is not a valid {fmt} (OOXML zip).",
            )
        ]
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            if "[Content_Types].xml" not in zf.namelist():
                return [
                    DocumentIssue(
                        code="invalid_ooxml",
                        severity="error",
                        message=f"{fmt} is missing the OOXML content-types part.",
                    )
                ]
    except zipfile.BadZipFile:
        return [
            DocumentIssue(
                code="invalid_ooxml",
                severity="error",
                message=f"{fmt} is a corrupt zip.",
            )
        ]
    return []


__all__ = [
    "CritiqueResult",
    "DocumentIssue",
    "critique_and_fix",
    "validate_render",
]
