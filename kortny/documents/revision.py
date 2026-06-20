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
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from kortny.documents.critique import VisualCritique, VisualIssue
from kortny.documents.ir import (
    Block,
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
_SAFE_ADDED_CATEGORIES: frozenset[str] = frozenset(
    {
        "doc_title",  # always present
        "table_col",  # critique_and_fix may rename blank/duplicate cols
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


RevisionOp = Annotated[SplitTable | SplitProse, Field(discriminator="op")]


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

    # Work on a mutable list copy of the blocks (as dicts for safe manipulation)
    blocks: list[Block] = list(spec.blocks)

    # Sort ops descending by block_index so earlier splits don't shift later indices
    sorted_ops = sorted(patch.operations, key=lambda op: op.block_index, reverse=True)

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

        else:
            # This branch is unreachable with the current discriminated union but
            # provides a runtime safety net for ops loaded from untrusted JSON.
            op_name = getattr(op, "op", repr(op))
            raise ValueError(f"unknown op: {op_name}")

    # Rebuild the spec without mutating the original
    spec_dict = spec.model_dump(mode="json")
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
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "ContentFingerprint",
    "RevisionOp",
    "SplitProse",
    "SplitTable",
    "VisualRevisionPatch",
    "apply_patch",
    "candidate_blocks_for_issue",
    "content_fingerprint",
    "content_preserved",
    "map_pages_to_blocks",
    "propose_overflow_patch",
    "spec_hash",
]
