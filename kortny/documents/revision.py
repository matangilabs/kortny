"""Document revision primitives — content fingerprinting, patch schema, applier (HIG-244).

Everything here is pure: no I/O except pypdf reading passed-in bytes inside
``map_pages_to_blocks``, no DB, no LLM, no Slack.

The module exposes:
- Content fingerprinting (``ContentFingerprint``, ``content_fingerprint``)
- A conservative content-preservation gate (``content_preserved``)
- A page→block mapper using pypdf text extraction (``map_pages_to_blocks``)
- A category-aware candidate selector (``candidate_blocks_for_issue``)
- A patch schema and applier (``VisualRevisionPatch``, ``apply_patch``)
- A heuristic overflow-patch proposer (``propose_overflow_patch``)
- An LLM-driven patch proposer (``propose_llm_patch``)
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from kortny.documents.critique import (
    VisualCritique,
    VisualIssue,
    critique_and_fix,
    validate_render,
)
from kortny.documents.ir import (
    Block,
    Callout,
    Chart,
    ChartSeries,
    CoverHeader,
    DocumentSpec,
    Heading,
    Prose,
    PullQuote,
    SectionDivider,
    StatCard,
    StatCards,
    Table,
)
from kortny.documents.themes import theme_names

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize(s: str) -> str:
    """Collapse runs of whitespace to a single space; strip leading/trailing."""
    return re.sub(r"\s+", " ", s).strip()


# ---------------------------------------------------------------------------
# 1. ContentFingerprint
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContentFingerprint:
    """Ordered tuple of (category, normalized_value) semantic content atoms."""

    atoms: tuple[tuple[str, str], ...]  # (category, normalized_value)


def _block_atoms(block: Block) -> list[tuple[str, str]]:
    """Extract (category, value) atoms from a single block."""
    atoms: list[tuple[str, str]] = []

    if isinstance(block, CoverHeader):
        atoms.append(("cover_title", _normalize(block.title)))
        if block.eyebrow is not None:
            atoms.append(("cover_eyebrow", _normalize(block.eyebrow)))
        if block.subtitle is not None:
            atoms.append(("cover_subtitle", _normalize(block.subtitle)))
        # accent_tail is intentionally excluded (presentation overlay, not a distinct atom)
        for m in block.meta:
            atoms.append(("cover_meta", _normalize(m)))

    elif isinstance(block, SectionDivider):
        if block.index is not None:
            atoms.append(("section_index", _normalize(block.index)))
        if block.label is not None:
            atoms.append(("section_label", _normalize(block.label)))
        atoms.append(("section_title", _normalize(block.title)))
        if block.subtitle is not None:
            atoms.append(("section_subtitle", _normalize(block.subtitle)))

    elif isinstance(block, Heading):
        atoms.append(("heading", _normalize(block.text)))

    elif isinstance(block, Prose):
        atoms.append(("prose", _normalize(block.text)))

    elif isinstance(block, StatCards):
        for card in block.cards:
            atoms.append(("stat_value", _normalize(card.value)))
            atoms.append(("stat_label", _normalize(card.label)))
            if card.note is not None:
                atoms.append(("stat_note", _normalize(card.note)))

    elif isinstance(block, Table):
        if block.caption is not None:
            atoms.append(("table_caption", _normalize(block.caption)))
        for col in block.columns:
            atoms.append(("table_col", _normalize(col)))
        for row in block.rows:
            for cell in row:
                atoms.append(("table_cell", _normalize(cell)))

    elif isinstance(block, Callout):
        if block.label is not None:
            atoms.append(("callout_label", _normalize(block.label)))
        atoms.append(("callout_text", _normalize(block.text)))

    elif isinstance(block, PullQuote):
        atoms.append(("pullquote_text", _normalize(block.text)))
        if block.attribution is not None:
            atoms.append(("pullquote_attribution", _normalize(block.attribution)))

    elif isinstance(block, Chart):
        if block.title is not None:
            atoms.append(("chart_title", _normalize(block.title)))
        if block.caption is not None:
            atoms.append(("chart_caption", _normalize(block.caption)))
        if block.x_label is not None:
            atoms.append(("chart_x_label", _normalize(block.x_label)))
        if block.y_label is not None:
            atoms.append(("chart_y_label", _normalize(block.y_label)))
        for series in block.series:
            atoms.append(("chart_series_name", _normalize(series.name)))
            for point in series.points:
                atoms.append(("chart_point_x", _normalize(str(point.x))))
                atoms.append(("chart_point_y", _normalize(str(point.y))))

    # CTA block (not caught by any isinstance above if we reach here)
    else:
        from kortny.documents.ir import CTA  # noqa: PLC0415

        if isinstance(block, CTA):
            atoms.append(("cta_label", _normalize(block.label)))
            if block.text is not None:
                atoms.append(("cta_text", _normalize(block.text)))

    return atoms


def content_fingerprint(spec: DocumentSpec) -> ContentFingerprint:
    """Build a ``ContentFingerprint`` from a ``DocumentSpec``."""
    atoms: list[tuple[str, str]] = [("doc_title", _normalize(spec.title))]
    for block in spec.blocks:
        atoms.extend(_block_atoms(block))
    return ContentFingerprint(atoms=tuple(atoms))


# ---------------------------------------------------------------------------
# 2. content_preserved
# ---------------------------------------------------------------------------

# Categories where new atoms in revised are not considered injections (renames
# are allowed by critique_and_fix, and "(cont.)" captions are a known extension).
# Chart label categories are added here because SetChartLabels only fills
# previously-None fields; new label atoms that were absent (None) in original
# are safe additions, while Rule 2 still catches any removal of existing labels.
_SAFE_ADDED_CATEGORIES: frozenset[str] = frozenset(
    {
        "doc_title",  # always present
        "table_col",  # critique_and_fix may rename blank/duplicate cols
        "chart_title",  # SetChartLabels fills previously-None title
        "chart_x_label",  # SetChartLabels fills previously-None x_label
        "chart_y_label",  # SetChartLabels fills previously-None y_label
        "chart_series_name",  # SetChartLabels fills previously-empty series names
    }
)


def _prose_values_from_atoms(atoms: tuple[tuple[str, str], ...]) -> list[str]:
    """Extract all 'prose' atom values from an atom tuple."""
    return [v for c, v in atoms if c == "prose"]


def _covers_original_prose(orig_value: str, rev_prose_values: list[str]) -> bool:
    """Return True if *orig_value* is covered by a consecutive subsequence of revised prose.

    A ``SplitProse`` op turns one ``("prose", "A B")`` atom into
    ``("prose", "A")`` + ``("prose", "B")``.  We check that the joined text
    of some window of *rev_prose_values* normalises to *orig_value*.
    """
    n = len(rev_prose_values)
    for start in range(n):
        for end in range(start + 1, n + 1):
            joined = _normalize(" ".join(rev_prose_values[start:end]))
            if joined == orig_value:
                return True
    return False


def content_preserved(
    original: DocumentSpec, revised: DocumentSpec
) -> tuple[bool, list[str]]:
    """Conservative content-preservation check.

    Returns ``(True, [])`` if all original atoms appear in revised (as a
    subsequence) and revised introduces no new text atoms.

    Prose splits are explicitly allowed: a single prose atom in *original*
    may be covered by a window of consecutive prose atoms in *revised* whose
    joined-and-normalised text equals the original prose value.

    ``CompactStatCards`` moves ``stat_note`` text into a ``Prose`` block; a
    ``stat_note`` atom missing from revised is allowed when its value appears
    as a ``prose`` atom in revised.

    Parameters
    ----------
    original:
        The spec before the revision patch was applied.
    revised:
        The spec produced by ``apply_patch``.

    Returns
    -------
    tuple of (preserved: bool, reasons: list[str])
    """
    orig_fp = content_fingerprint(original)
    rev_fp = content_fingerprint(revised)

    orig_atoms = orig_fp.atoms
    rev_atoms = rev_fp.atoms

    reasons: list[str] = []

    orig_set = set(orig_atoms)
    rev_set = set(rev_atoms)

    rev_prose_values = _prose_values_from_atoms(rev_atoms)
    orig_prose_values_set = {v for c, v in orig_atoms if c == "prose"}

    # --- Rule 2: every original atom must appear in revised (exact or prose-split) ---
    for atom in orig_atoms:
        category, value = atom
        if atom in rev_set:
            continue
        # Allow prose atoms that have been split across multiple consecutive prose blocks
        if category == "prose" and _covers_original_prose(value, rev_prose_values):
            continue
        # Allow stat_note atoms that have been moved to prose blocks (CompactStatCards)
        if category == "stat_note" and any(
            v == value for c, v in rev_atoms if c == "prose"
        ):
            continue
        reasons.append(f"missing atom: {atom!r}")

    if reasons:
        return False, reasons

    # --- Rule 3: new atoms in revised that are not from safe categories or allowed suffixes ---
    new_atoms = [a for a in rev_atoms if a not in orig_set]
    for atom in new_atoms:
        category, value = atom
        if category in _SAFE_ADDED_CATEGORIES:
            continue
        # Allow table_caption that is original caption + " (cont.)" suffix
        if category == "table_caption":
            orig_captions = {v for c, v in orig_atoms if c == "table_caption"}
            is_continuation = (
                any(
                    value == f"{cap} (cont.)" or value == "(cont.)"
                    for cap in orig_captions
                )
                or value == "(cont.)"
            )
            if is_continuation:
                continue
        # Allow prose atoms that are sub-spans of an original prose value.
        # A split produces shorter fragments; injection produces atoms longer than or
        # not contained in any original prose value.
        if category == "prose":
            norm_value = _normalize(value)
            if any(
                norm_value in orig_prose_val for orig_prose_val in orig_prose_values_set
            ):
                continue
            # Also allow prose that matches a moved stat_note value (CompactStatCards)
            orig_stat_note_values = {
                _normalize(v) for c, v in orig_atoms if c == "stat_note"
            }
            if norm_value in orig_stat_note_values:
                continue
        reasons.append(f"injected atom: {atom!r}")
        return False, reasons

    return True, []


# ---------------------------------------------------------------------------
# 3. Page→block mapper
# ---------------------------------------------------------------------------


def _block_text_tokens(block: Block) -> set[str]:
    """Return the set of whitespace-split tokens from all atom values of a block."""
    atoms = _block_atoms(block)
    combined = " ".join(v for _, v in atoms)
    return set(combined.split())


def map_pages_to_blocks(pdf_bytes: bytes, spec: DocumentSpec) -> dict[int, list[int]]:
    """Map 1-based page numbers to block indices via pypdf text token overlap.

    Parameters
    ----------
    pdf_bytes:
        Raw PDF bytes.
    spec:
        The ``DocumentSpec`` whose blocks are to be located.

    Returns
    -------
    dict mapping 1-based page number → sorted list of block indices whose
    best-match page is that page number.
    """
    from pypdf import PdfReader  # noqa: PLC0415

    reader = PdfReader(io.BytesIO(pdf_bytes))

    # Per-page text token sets (1-based)
    page_tokens: dict[int, set[str]] = {}
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        page_tokens[page_num] = set(text.split())

    # Per-block token sets
    block_token_sets: list[set[str]] = [
        _block_text_tokens(block) for block in spec.blocks
    ]

    # For each block, find the page with the highest token overlap.
    # block_best_page[i] = page number with max overlap for block i
    block_best_page: dict[int, int] = {}
    for block_idx, block_tokens in enumerate(block_token_sets):
        if not block_tokens:
            continue
        best_page = 1
        best_overlap = -1
        for page_num, ptokens in page_tokens.items():
            overlap = len(block_tokens & ptokens)
            if overlap > best_overlap:
                best_overlap = overlap
                best_page = page_num
        block_best_page[block_idx] = best_page

    # Invert: page → list of block indices
    result: dict[int, list[int]] = {}
    for block_idx, page_num in block_best_page.items():
        result.setdefault(page_num, []).append(block_idx)

    # Sort indices within each page entry
    for page_num in result:
        result[page_num].sort()

    return result


def candidate_blocks_for_issue(
    issue: VisualIssue,
    page_map: dict[int, list[int]],
    spec: DocumentSpec,
) -> list[int]:
    """Return candidate block indices for a ``VisualIssue``.

    First tries the page_map lookup; falls back to category-aware heuristics
    when the page has no mapped blocks.
    """
    mapped = page_map.get(issue.page, [])
    if mapped:
        return mapped

    # Category-aware heuristic fallback

    indices: list[int] = []
    for idx, block in enumerate(spec.blocks):
        if issue.category == "labels":
            if isinstance(block, Chart):
                indices.append(idx)
        elif issue.category == "overflow":
            if (
                isinstance(block, Table)
                or isinstance(block, Prose)
                and len(block.text) > 500
            ):
                indices.append(idx)
        elif issue.category == "whitespace":
            if (
                isinstance(block, Prose)
                and len(block.text) > 200
                or isinstance(block, SectionDivider)
            ):
                indices.append(idx)
        elif issue.category == "hierarchy" and isinstance(
            block, (Heading, SectionDivider, CoverHeader)
        ):
            indices.append(idx)

    return indices


# ---------------------------------------------------------------------------
# 4. Patch schema + applier
# ---------------------------------------------------------------------------


class SplitTable(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op: Literal["split_table"] = "split_table"
    block_index: int
    max_rows_per_table: int


class SplitProse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op: Literal["split_prose"] = "split_prose"
    block_index: int


class SetChartLabels(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op: Literal["set_chart_labels"] = "set_chart_labels"
    block_index: int
    title: str | None = None
    x_axis_label: str | None = None
    y_axis_label: str | None = None
    series_names: list[str] | None = None


class ChangeChartType(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op: Literal["change_chart_type"] = "change_chart_type"
    block_index: int
    chart_type: str


class CompactStatCards(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op: Literal["compact_stat_cards"] = "compact_stat_cards"
    block_index: int


class SetTheme(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op: Literal["set_theme"] = "set_theme"
    theme: str


RevisionOp = Annotated[
    SplitTable
    | SplitProse
    | SetChartLabels
    | ChangeChartType
    | CompactStatCards
    | SetTheme,
    Field(discriminator="op"),
]


class VisualRevisionPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_spec_hash: str
    operations: list[RevisionOp]
    rationale: str


def spec_hash(spec: DocumentSpec) -> str:
    """Stable SHA-256 hex digest of the spec's JSON representation."""
    data = json.dumps(spec.model_dump(mode="json"), sort_keys=True)
    return hashlib.sha256(data.encode()).hexdigest()


def _split_table_block(block: Table, max_rows: int) -> list[Block]:
    """Split a ``Table`` into chunks of at most *max_rows* rows."""
    if not block.rows:
        return [block]
    chunks: list[Block] = []
    for chunk_start in range(0, len(block.rows), max_rows):
        chunk_rows = block.rows[chunk_start : chunk_start + max_rows]
        if chunk_start == 0:
            caption = block.caption
        else:
            caption = f"{block.caption} (cont.)" if block.caption else "(cont.)"
        chunks.append(
            Table(
                type="table",
                caption=caption,
                columns=list(block.columns),
                rows=chunk_rows,
            )
        )
    return chunks


def _split_prose_block(block: Prose) -> list[Block]:
    """Split a ``Prose`` block at paragraph / sentence boundaries."""
    text = block.text

    # 1. Try paragraph boundaries (\n\n)
    parts = [p for p in text.split("\n\n") if p.strip()]
    if len(parts) > 1:
        return [Prose(type="prose", text=p) for p in parts]

    # 2. Try line boundaries (\n)
    parts = [p for p in text.split("\n") if p.strip()]
    if len(parts) > 1:
        return [Prose(type="prose", text=p) for p in parts]

    # 3. Sentence boundary fallback
    parts = [p for p in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text) if p.strip()]
    if len(parts) > 1:
        return [Prose(type="prose", text=p) for p in parts]

    # 4. No boundary found — keep as-is
    return [block]


def apply_patch(spec: DocumentSpec, patch: VisualRevisionPatch) -> DocumentSpec:
    """Apply a ``VisualRevisionPatch`` to *spec*, returning a new ``DocumentSpec``.

    Raises ``ValueError`` when the patch's base hash doesn't match or an
    unknown op is encountered. Never mutates *spec*.
    """
    if patch.base_spec_hash != spec_hash(spec):
        raise ValueError("patch base_spec_hash mismatch")

    # Work on a mutable list copy of the blocks
    blocks: list[Block] = list(spec.blocks)
    spec_dict = spec.model_dump(mode="json")

    # Separate spec-level ops (no block_index) from block-level ops
    spec_level_ops: list[SetTheme] = []
    block_level_ops: list[
        SplitTable | SplitProse | SetChartLabels | ChangeChartType | CompactStatCards
    ] = []
    for op in patch.operations:
        if isinstance(op, SetTheme):
            spec_level_ops.append(op)
        else:
            block_level_ops.append(op)

    # Apply spec-level ops first
    for op in spec_level_ops:
        known = theme_names()
        if op.theme not in known:
            raise ValueError(f"SetTheme: unknown theme {op.theme!r}; known: {known}")
        spec_dict["theme"] = op.theme

    # Sort block ops descending by block_index so earlier splits don't shift later indices
    sorted_ops = sorted(block_level_ops, key=lambda o: o.block_index, reverse=True)

    for op in sorted_ops:
        idx = op.block_index
        if idx < 0 or idx >= len(blocks):
            raise ValueError(
                f"block_index {idx} is out of range (spec has {len(blocks)} blocks)"
            )
        target = blocks[idx]

        if isinstance(op, SplitTable):
            if not isinstance(target, Table):
                raise ValueError(
                    f"SplitTable op at index {idx} targets a {type(target).__name__}, not a Table"
                )
            split_tables = _split_table_block(target, op.max_rows_per_table)
            blocks = blocks[:idx] + split_tables + blocks[idx + 1 :]

        elif isinstance(op, SplitProse):
            if not isinstance(target, Prose):
                raise ValueError(
                    f"SplitProse op at index {idx} targets a {type(target).__name__}, not a Prose"
                )
            split_prose = _split_prose_block(target)
            blocks = blocks[:idx] + split_prose + blocks[idx + 1 :]

        elif isinstance(op, SetChartLabels):
            if not isinstance(target, Chart):
                raise ValueError(
                    f"SetChartLabels op at index {idx} targets a {type(target).__name__}, not a Chart"
                )
            # ONLY fill missing (None or empty string) fields — never overwrite existing non-empty labels
            new_title = (
                target.title
                if (target.title is not None and target.title != "")
                else op.title
            )
            new_x_label = (
                target.x_label
                if (target.x_label is not None and target.x_label != "")
                else op.x_axis_label
            )
            new_y_label = (
                target.y_label
                if (target.y_label is not None and target.y_label != "")
                else op.y_axis_label
            )
            # For series_names, only fill series whose name is empty/None positionally
            new_series: list[ChartSeries] = []
            for i, series in enumerate(target.series):
                if (
                    op.series_names is not None
                    and i < len(op.series_names)
                    and (series.name is None or series.name == "")
                ):
                    new_series.append(
                        ChartSeries(name=op.series_names[i], points=list(series.points))
                    )
                else:
                    new_series.append(series)
            updated_chart_dict = target.model_dump(mode="json")
            updated_chart_dict["title"] = new_title
            updated_chart_dict["x_label"] = new_x_label
            updated_chart_dict["y_label"] = new_y_label
            updated_chart_dict["series"] = [
                s.model_dump(mode="json") for s in new_series
            ]
            updated_chart = Chart.model_validate(updated_chart_dict)
            blocks = blocks[:idx] + [updated_chart] + blocks[idx + 1 :]

        elif isinstance(op, ChangeChartType):
            if not isinstance(target, Chart):
                raise ValueError(
                    f"ChangeChartType op at index {idx} targets a {type(target).__name__}, not a Chart"
                )
            # Compatibility: {bar, line, area} are mutually compatible; pie is NOT
            _BAR_LINE_AREA = {"bar", "line", "area"}
            if target.chart_type in _BAR_LINE_AREA and op.chart_type == "pie":
                raise ValueError(
                    f"ChangeChartType: cannot change chart from {target.chart_type!r} to 'pie' (incompatible)"
                )
            if target.chart_type == "pie" and op.chart_type in _BAR_LINE_AREA:
                raise ValueError(
                    f"ChangeChartType: cannot change chart from 'pie' to {op.chart_type!r} (incompatible)"
                )
            updated_dict = target.model_dump(mode="json")
            updated_dict["chart_type"] = op.chart_type
            try:
                updated_chart = Chart.model_validate(updated_dict)
            except Exception as e:
                raise ValueError(
                    f"ChangeChartType: invalid chart_type {op.chart_type!r}: {e}"
                ) from e
            blocks = blocks[:idx] + [updated_chart] + blocks[idx + 1 :]

        elif isinstance(op, CompactStatCards):
            if not isinstance(target, StatCards):
                raise ValueError(
                    f"CompactStatCards op at index {idx} targets a {type(target).__name__}, not a StatCards"
                )
            # Collect all note texts and clear notes on cards
            note_parts: list[str] = []
            new_cards: list[StatCard] = []
            for card in target.cards:
                if card.note is not None and card.note.strip():
                    note_parts.append(card.note)
                new_cards.append(
                    StatCard(value=card.value, label=card.label, note=None)
                )
            updated_stat_cards = StatCards(type="stat_cards", cards=new_cards)
            # Insert Prose block AFTER the StatCards block with the combined notes (if any)
            if note_parts:
                notes_prose = Prose(type="prose", text="\n\n".join(note_parts))
                blocks = (
                    blocks[:idx] + [updated_stat_cards, notes_prose] + blocks[idx + 1 :]
                )
            else:
                blocks = blocks[:idx] + [updated_stat_cards] + blocks[idx + 1 :]

        else:
            # This branch is unreachable with the current discriminated union but
            # provides a runtime safety net for ops loaded from untrusted JSON.
            op_name = getattr(op, "op", repr(op))
            raise ValueError(f"unknown op: {op_name}")

    # Rebuild the spec without mutating the original
    spec_dict["blocks"] = [
        b.model_dump(mode="json") if hasattr(b, "model_dump") else b for b in blocks
    ]
    return DocumentSpec.model_validate(spec_dict)


# ---------------------------------------------------------------------------
# 5. propose_overflow_patch
# ---------------------------------------------------------------------------


def propose_overflow_patch(
    spec: DocumentSpec,
    critique: VisualCritique,
    page_map: dict[int, list[int]],
) -> VisualRevisionPatch | None:
    """Propose a patch that splits overflowing tables and prose blocks.

    Returns ``None`` when no actionable overflow issues are found.
    """
    overflow_issues = [i for i in critique.issues if i.category == "overflow"]
    if not overflow_issues:
        return None

    ops: list[RevisionOp] = []
    seen_indices: set[int] = set()

    for issue in overflow_issues:
        candidates = candidate_blocks_for_issue(issue, page_map, spec)
        for idx in candidates:
            if idx in seen_indices:
                continue
            block = spec.blocks[idx]
            if isinstance(block, Table):
                ops.append(SplitTable(block_index=idx, max_rows_per_table=12))
                seen_indices.add(idx)
            elif isinstance(block, Prose) and len(block.text) > 500:
                ops.append(SplitProse(block_index=idx))
                seen_indices.add(idx)

    if not ops:
        return None

    return VisualRevisionPatch(
        base_spec_hash=spec_hash(spec),
        operations=ops,
        rationale="Overflow fix: split tables/prose to prevent page overflow.",
    )


# ---------------------------------------------------------------------------
# 5b. propose_llm_patch
# ---------------------------------------------------------------------------


@dataclass
class LlmPatchContext:
    """Context passed to an LLM proposer for non-overflow revision issues."""

    base_spec_hash: str
    issues: list[VisualIssue]  # non-overflow issues only
    candidates: dict[int, dict]  # block_index -> block JSON for each candidate block


def propose_llm_patch(
    spec: DocumentSpec,
    critique: VisualCritique,
    page_map: dict[int, list[int]],
    *,
    propose_fn: Callable[[LlmPatchContext], VisualRevisionPatch | None],
) -> VisualRevisionPatch | None:
    """Propose a patch for non-overflow issues using an LLM proposer callable.

    Filters critique issues to non-overflow, builds a context with candidate
    block JSON, calls *propose_fn*, validates the returned patch's hash, and
    does a light whitelist check on op types.

    Returns ``None`` when there are no non-overflow issues, when *propose_fn*
    returns ``None``, or when the returned patch fails validation.
    """
    non_overflow = [i for i in critique.issues if i.category != "overflow"]
    if not non_overflow:
        return None

    # Collect candidate block indices across all non-overflow issues
    candidate_indices: set[int] = set()
    for issue in non_overflow:
        for idx in candidate_blocks_for_issue(issue, page_map, spec):
            candidate_indices.add(idx)

    candidates: dict[int, dict] = {}
    for idx in candidate_indices:
        if 0 <= idx < len(spec.blocks):
            block = spec.blocks[idx]
            candidates[idx] = (
                block.model_dump(mode="json") if hasattr(block, "model_dump") else {}
            )

    context = LlmPatchContext(
        base_spec_hash=spec_hash(spec),
        issues=non_overflow,
        candidates=candidates,
    )

    result = propose_fn(context)
    if result is None:
        return None

    # Sanity check: base_spec_hash must match
    if result.base_spec_hash != spec_hash(spec):
        return None

    # Light whitelist check on op types (apply_patch enforces fully)
    _WHITELISTED_OP_TYPES = (
        SplitTable,
        SplitProse,
        SetChartLabels,
        ChangeChartType,
        CompactStatCards,
        SetTheme,
    )
    for op in result.operations:
        if not isinstance(op, _WHITELISTED_OP_TYPES):
            return None

    return result


# ---------------------------------------------------------------------------
# 6. Revision outcome types
# ---------------------------------------------------------------------------

RevisionStatus = Literal["accepted", "rejected", "noop"]


class RevisionEvent(BaseModel):
    kind: Literal[
        "visual_revision_started",
        "visual_revision_candidate_rejected",
        "visual_revision_accepted",
        "visual_revision_noop",
    ]
    detail: str
    old_score: int | None = None
    new_score: int | None = None


class RevisionOutcome(BaseModel):
    status: RevisionStatus
    revised_spec: DocumentSpec | None = None
    revised_pdf: bytes | None = Field(default=None, repr=False, exclude=True)
    new_critique: VisualCritique | None = None
    reason: str
    events: list[RevisionEvent] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 7. attempt_visual_revision
# ---------------------------------------------------------------------------


def attempt_visual_revision(
    spec: DocumentSpec,
    original_critique: VisualCritique,
    *,
    render: Callable[[DocumentSpec], bytes],
    critique_fn: Callable[[bytes], VisualCritique | None],
    trigger_below: int = 7,
    min_improvement: int = 1,
    original_pdf: bytes | None = None,
    llm_propose_fn: Callable[[LlmPatchContext], VisualRevisionPatch | None]
    | None = None,
) -> RevisionOutcome:
    """Attempt a single visual-revision cycle (deterministic or LLM-driven).

    First proposes a deterministic overflow patch; if none is found, falls back
    to *llm_propose_fn* when provided.  Applies the patch, re-renders,
    re-critiques, and accepts or rejects the candidate based on quality gates.

    Never raises — all exceptions are caught and returned as a rejected
    ``RevisionOutcome``.

    Parameters
    ----------
    spec:
        The ``DocumentSpec`` to revise.
    original_critique:
        The ``VisualCritique`` that motivated the revision attempt.
    render:
        Callable that renders a ``DocumentSpec`` to PDF bytes.  May raise.
    critique_fn:
        Callable that critiques PDF bytes, returning a ``VisualCritique`` or
        ``None`` on failure.
    trigger_below:
        Only attempt revision when ``original_critique.overall_score`` is
        strictly below this threshold.  Default 7.
    min_improvement:
        Minimum score improvement required to accept the candidate.  Default 1.
    original_pdf:
        Optional original PDF bytes for page→block mapping.  When omitted,
        an empty page map is used.
    llm_propose_fn:
        Optional callable that proposes a patch for non-overflow issues via LLM.
        Called only when the deterministic overflow proposer returns ``None``.
    """
    started_event = RevisionEvent(
        kind="visual_revision_started",
        detail=f"Starting visual revision (score={original_critique.overall_score})",
        old_score=original_critique.overall_score,
    )
    events: list[RevisionEvent] = [started_event]

    try:
        # Gate 1: only revise if score is below threshold
        if original_critique.overall_score >= trigger_below:
            noop_event = RevisionEvent(
                kind="visual_revision_noop",
                detail="score already acceptable",
                old_score=original_critique.overall_score,
            )
            events.append(noop_event)
            return RevisionOutcome(
                status="noop",
                reason="score already acceptable",
                events=events,
            )

        # Build page map
        page_map: dict[int, list[int]]
        if original_pdf is not None:
            page_map = map_pages_to_blocks(original_pdf, spec)
        else:
            page_map = {}

        # Propose patch — deterministic overflow first, then LLM fallback
        patch = propose_overflow_patch(spec, original_critique, page_map)
        if patch is None:
            if llm_propose_fn is not None:
                patch = propose_llm_patch(
                    spec, original_critique, page_map, propose_fn=llm_propose_fn
                )
            if patch is None:
                noop_event = RevisionEvent(
                    kind="visual_revision_noop",
                    detail="no actionable deterministic fix",
                    old_score=original_critique.overall_score,
                )
                events.append(noop_event)
                return RevisionOutcome(
                    status="noop",
                    reason="no actionable deterministic fix",
                    events=events,
                )

        # Apply patch
        try:
            candidate = apply_patch(spec, patch)
        except Exception as e:
            reject_event = RevisionEvent(
                kind="visual_revision_candidate_rejected",
                detail=f"patch apply failed: {e}",
                old_score=original_critique.overall_score,
            )
            events.append(reject_event)
            return RevisionOutcome(
                status="rejected",
                reason=f"patch apply failed: {e}",
                events=events,
            )

        # Gate a: structural errors
        fix_result = critique_and_fix(candidate)
        if fix_result.has_errors:
            reject_event = RevisionEvent(
                kind="visual_revision_candidate_rejected",
                detail="candidate has structural errors",
                old_score=original_critique.overall_score,
            )
            events.append(reject_event)
            return RevisionOutcome(
                status="rejected",
                reason="candidate has structural errors",
                events=events,
            )

        # Gate b: render succeeds
        try:
            candidate_pdf = render(candidate)
        except Exception as e:
            reject_event = RevisionEvent(
                kind="visual_revision_candidate_rejected",
                detail=f"candidate render failed: {e}",
                old_score=original_critique.overall_score,
            )
            events.append(reject_event)
            return RevisionOutcome(
                status="rejected",
                reason=f"candidate render failed: {e}",
                events=events,
            )

        # Gate c: render validation — reject if any error-severity issue
        render_issues = validate_render(candidate_pdf, "pdf")
        error_render_issues = [i for i in render_issues if i.severity == "error"]
        if error_render_issues:
            reject_event = RevisionEvent(
                kind="visual_revision_candidate_rejected",
                detail="candidate failed render validation",
                old_score=original_critique.overall_score,
            )
            events.append(reject_event)
            return RevisionOutcome(
                status="rejected",
                reason="candidate failed render validation",
                events=events,
            )

        # Gate d: content preservation
        ok, reasons = content_preserved(spec, candidate)
        if not ok:
            reject_event = RevisionEvent(
                kind="visual_revision_candidate_rejected",
                detail=f"content not preserved: {'; '.join(reasons)}",
                old_score=original_critique.overall_score,
            )
            events.append(reject_event)
            return RevisionOutcome(
                status="rejected",
                reason=f"content not preserved: {'; '.join(reasons)}",
                events=events,
            )

        # Gate e: critique candidate
        new_critique = critique_fn(candidate_pdf)
        if new_critique is None:
            reject_event = RevisionEvent(
                kind="visual_revision_candidate_rejected",
                detail="could not critique candidate",
                old_score=original_critique.overall_score,
            )
            events.append(reject_event)
            return RevisionOutcome(
                status="rejected",
                reason="could not critique candidate",
                events=events,
            )

        # Gate f: improvement check
        # Count error-severity issues in both critiques (VisualIssue uses severity str)
        # Note: VisualIssue.severity is Severity (Literal["error","warning","info"])
        orig_error_count = len(
            [i for i in original_critique.issues if i.severity == "error"]
        )
        new_error_count = len([i for i in new_critique.issues if i.severity == "error"])
        score_improved = (
            new_critique.overall_score
            >= original_critique.overall_score + min_improvement
        )
        tiebreak_ok = (
            new_critique.overall_score == original_critique.overall_score
            and new_error_count < orig_error_count
        )
        if not (score_improved or tiebreak_ok):
            reject_event = RevisionEvent(
                kind="visual_revision_candidate_rejected",
                detail=f"no improvement: old={original_critique.overall_score} new={new_critique.overall_score}",
                old_score=original_critique.overall_score,
                new_score=new_critique.overall_score,
            )
            events.append(reject_event)
            return RevisionOutcome(
                status="rejected",
                reason=f"no improvement: old={original_critique.overall_score} new={new_critique.overall_score}",
                events=events,
            )

        # All gates passed — accept
        accept_event = RevisionEvent(
            kind="visual_revision_accepted",
            detail=f"accepted: old={original_critique.overall_score} new={new_critique.overall_score}",
            old_score=original_critique.overall_score,
            new_score=new_critique.overall_score,
        )
        events.append(accept_event)
        return RevisionOutcome(
            status="accepted",
            revised_spec=candidate,
            revised_pdf=candidate_pdf,
            new_critique=new_critique,
            reason=f"accepted: old={original_critique.overall_score} new={new_critique.overall_score}",
            events=events,
        )

    except Exception as e:
        return RevisionOutcome(
            status="rejected",
            reason=f"unexpected error: {e}",
            events=events,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "ChangeChartType",
    "CompactStatCards",
    "ContentFingerprint",
    "LlmPatchContext",
    "RevisionEvent",
    "RevisionOutcome",
    "RevisionOp",
    "RevisionStatus",
    "SetChartLabels",
    "SetTheme",
    "SplitProse",
    "SplitTable",
    "VisualRevisionPatch",
    "apply_patch",
    "attempt_visual_revision",
    "candidate_blocks_for_issue",
    "content_fingerprint",
    "content_preserved",
    "map_pages_to_blocks",
    "propose_llm_patch",
    "propose_overflow_patch",
    "spec_hash",
]
