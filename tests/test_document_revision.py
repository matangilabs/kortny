"""Tests for kortny/documents/revision.py (HIG-244 critique slice 2).

Pure module — no DB, no async, no Postgres required.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from kortny.documents.critique import VisualCritique, VisualIssue
from kortny.documents.ir import (
    CTA,
    Callout,
    Chart,
    ChartPoint,
    ChartSeries,
    CoverHeader,
    DocKind,
    DocumentSpec,
    Heading,
    Prose,
    PullQuote,
    SectionDivider,
    StatCard,
    StatCards,
    Table,
)
from kortny.documents.revision import (
    ChangeChartType,
    CompactStatCards,
    ContentFingerprint,
    LlmPatchContext,
    SetChartLabels,
    SetTheme,
    SplitProse,
    SplitTable,
    VisualRevisionPatch,
    apply_patch,
    attempt_visual_revision,
    candidate_blocks_for_issue,
    content_fingerprint,
    content_preserved,
    map_pages_to_blocks,
    propose_llm_patch,
    propose_overflow_patch,
    spec_hash,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def full_spec() -> DocumentSpec:
    """A DocumentSpec with one of every block type."""
    return DocumentSpec(
        doc_kind=DocKind.report,
        title="Full Test Document",
        blocks=[
            CoverHeader(
                type="cover_header",
                eyebrow="Q4 Brief",
                title="The Big Picture",
                subtitle="An overview",
                accent_tail="Picture",
                meta=["Kortny", "2026"],
            ),
            SectionDivider(
                type="section_divider",
                index="01",
                label="Part One",
                title="Introduction",
                subtitle="Why this matters",
            ),
            Heading(type="heading", text="Background"),
            Prose(type="prose", text="First paragraph.\n\nSecond paragraph."),
            StatCards(
                type="stat_cards",
                cards=[
                    StatCard(value="$1B", label="Revenue", note="YoY"),
                    StatCard(value="42%", label="Growth"),
                ],
            ),
            Table(
                type="table",
                caption="Comparables",
                columns=["Company", "Raise"],
                rows=[["Arm", "$4.9B"], ["Reddit", "$748M"]],
            ),
            Callout(type="callout", label="Note", text="Important context here."),
            PullQuote(
                type="pull_quote",
                text="The market shifted.",
                attribution="Analyst",
            ),
            CTA(type="cta", label="Learn more", text="Visit the dashboard."),
            Chart(
                type="chart",
                chart_type="bar",
                title="Revenue Growth",
                x_label="Quarter",
                y_label="USD M",
                series=[
                    ChartSeries(
                        name="Series A",
                        points=[
                            ChartPoint(x="Q1", y=100.0),
                            ChartPoint(x="Q2", y=150.0),
                        ],
                    )
                ],
                caption="Annual revenue",
            ),
        ],
    )


@pytest.fixture()
def simple_spec() -> DocumentSpec:
    """Minimal spec with one Table and one Prose."""
    return DocumentSpec(
        doc_kind=DocKind.report,
        title="Simple Doc",
        blocks=[
            Table(
                type="table",
                caption="Data",
                columns=["Name", "Value"],
                rows=[["Alpha", "1"], ["Beta", "2"], ["Gamma", "3"]],
            ),
            Prose(type="prose", text="This is the prose section of the document."),
        ],
    )


# ---------------------------------------------------------------------------
# content_fingerprint tests
# ---------------------------------------------------------------------------


class TestContentFingerprint:
    def test_atoms_tuple_type(self, full_spec: DocumentSpec) -> None:
        fp = content_fingerprint(full_spec)
        assert isinstance(fp, ContentFingerprint)
        assert isinstance(fp.atoms, tuple)
        assert all(isinstance(a, tuple) and len(a) == 2 for a in fp.atoms)

    def test_doc_title_first(self, full_spec: DocumentSpec) -> None:
        fp = content_fingerprint(full_spec)
        assert fp.atoms[0] == ("doc_title", "Full Test Document")

    def test_cover_atoms(self, full_spec: DocumentSpec) -> None:
        fp = content_fingerprint(full_spec)
        categories = [c for c, _ in fp.atoms]
        assert "cover_title" in categories
        assert "cover_eyebrow" in categories
        assert "cover_subtitle" in categories
        assert "cover_meta" in categories

    def test_cover_title_value(self, full_spec: DocumentSpec) -> None:
        fp = content_fingerprint(full_spec)
        cover_title_atoms = [(c, v) for c, v in fp.atoms if c == "cover_title"]
        assert cover_title_atoms == [("cover_title", "The Big Picture")]

    def test_accent_tail_not_included(self, full_spec: DocumentSpec) -> None:
        """accent_tail should NOT appear as its own atom."""
        fp = content_fingerprint(full_spec)
        categories = [c for c, _ in fp.atoms]
        assert "cover_accent_tail" not in categories

    def test_section_atoms(self, full_spec: DocumentSpec) -> None:
        fp = content_fingerprint(full_spec)
        categories = [c for c, _ in fp.atoms]
        assert "section_label" in categories
        assert "section_title" in categories
        assert "section_index" in categories

    def test_section_title_value(self, full_spec: DocumentSpec) -> None:
        fp = content_fingerprint(full_spec)
        section_titles = [(c, v) for c, v in fp.atoms if c == "section_title"]
        assert section_titles == [("section_title", "Introduction")]

    def test_heading_atom(self, full_spec: DocumentSpec) -> None:
        fp = content_fingerprint(full_spec)
        headings = [(c, v) for c, v in fp.atoms if c == "heading"]
        assert ("heading", "Background") in headings

    def test_prose_atom(self, full_spec: DocumentSpec) -> None:
        fp = content_fingerprint(full_spec)
        # Prose is stored as a single atom (full text, normalized)
        prose_atoms = [(c, v) for c, v in fp.atoms if c == "prose"]
        assert len(prose_atoms) == 1
        # Normalized: \n\n collapses to a space
        assert "First paragraph." in prose_atoms[0][1]
        assert "Second paragraph." in prose_atoms[0][1]

    def test_stat_value_and_label_atoms(self, full_spec: DocumentSpec) -> None:
        fp = content_fingerprint(full_spec)
        stat_values = [(c, v) for c, v in fp.atoms if c == "stat_value"]
        stat_labels = [(c, v) for c, v in fp.atoms if c == "stat_label"]
        assert ("stat_value", "$1B") in stat_values
        assert ("stat_label", "Revenue") in stat_labels

    def test_stat_note_atom(self, full_spec: DocumentSpec) -> None:
        fp = content_fingerprint(full_spec)
        stat_notes = [(c, v) for c, v in fp.atoms if c == "stat_note"]
        assert ("stat_note", "YoY") in stat_notes

    def test_table_atoms(self, full_spec: DocumentSpec) -> None:
        fp = content_fingerprint(full_spec)
        assert ("table_caption", "Comparables") in fp.atoms
        assert ("table_col", "Company") in fp.atoms
        assert ("table_col", "Raise") in fp.atoms
        assert ("table_cell", "Arm") in fp.atoms
        assert ("table_cell", "$4.9B") in fp.atoms

    def test_callout_atoms(self, full_spec: DocumentSpec) -> None:
        fp = content_fingerprint(full_spec)
        assert ("callout_label", "Note") in fp.atoms
        assert ("callout_text", "Important context here.") in fp.atoms

    def test_pullquote_atoms(self, full_spec: DocumentSpec) -> None:
        fp = content_fingerprint(full_spec)
        assert ("pullquote_text", "The market shifted.") in fp.atoms
        assert ("pullquote_attribution", "Analyst") in fp.atoms

    def test_cta_atoms(self, full_spec: DocumentSpec) -> None:
        fp = content_fingerprint(full_spec)
        assert ("cta_label", "Learn more") in fp.atoms
        assert ("cta_text", "Visit the dashboard.") in fp.atoms

    def test_chart_atoms(self, full_spec: DocumentSpec) -> None:
        fp = content_fingerprint(full_spec)
        assert ("chart_title", "Revenue Growth") in fp.atoms
        assert ("chart_caption", "Annual revenue") in fp.atoms
        assert ("chart_x_label", "Quarter") in fp.atoms
        assert ("chart_y_label", "USD M") in fp.atoms
        assert ("chart_series_name", "Series A") in fp.atoms
        assert ("chart_point_x", "Q1") in fp.atoms
        assert ("chart_point_y", "100.0") in fp.atoms

    def test_normalization(self) -> None:
        """Extra whitespace is collapsed to a single space."""
        spec = DocumentSpec(
            doc_kind=DocKind.report,
            title="  Test   Doc  ",
            blocks=[Prose(type="prose", text="Hello   world  ")],
        )
        fp = content_fingerprint(spec)
        assert fp.atoms[0] == ("doc_title", "Test Doc")
        assert ("prose", "Hello world") in fp.atoms

    def test_frozen(self, full_spec: DocumentSpec) -> None:
        fp = content_fingerprint(full_spec)
        with pytest.raises((AttributeError, TypeError)):
            fp.atoms = ()  # type: ignore[misc]


# ---------------------------------------------------------------------------
# content_preserved tests
# ---------------------------------------------------------------------------


class TestContentPreserved:
    def test_identical_specs(self, full_spec: DocumentSpec) -> None:
        ok, reasons = content_preserved(full_spec, full_spec)
        assert ok is True
        assert reasons == []

    def test_missing_table_row(self, simple_spec: DocumentSpec) -> None:
        """Dropping a table row should be detected."""
        truncated = DocumentSpec(
            doc_kind=DocKind.report,
            title="Simple Doc",
            blocks=[
                Table(
                    type="table",
                    caption="Data",
                    columns=["Name", "Value"],
                    rows=[["Alpha", "1"], ["Beta", "2"]],  # Gamma row dropped
                ),
                Prose(type="prose", text="This is the prose section of the document."),
            ],
        )
        ok, reasons = content_preserved(simple_spec, truncated)
        assert ok is False
        assert any("missing atom" in r for r in reasons)

    def test_truncated_prose(self) -> None:
        """Dropping content from a prose block should be detected."""
        original = DocumentSpec(
            doc_kind=DocKind.report,
            title="Doc",
            blocks=[Prose(type="prose", text="Full sentence one. Full sentence two.")],
        )
        truncated = DocumentSpec(
            doc_kind=DocKind.report,
            title="Doc",
            blocks=[Prose(type="prose", text="Full sentence one.")],
        )
        ok, reasons = content_preserved(original, truncated)
        assert ok is False
        assert any("missing atom" in r for r in reasons)

    def test_dropped_chart_point(self) -> None:
        """Dropping a chart data point should be detected."""
        original = DocumentSpec(
            doc_kind=DocKind.report,
            title="Chart Doc",
            blocks=[
                Chart(
                    type="chart",
                    chart_type="bar",
                    title="Sales",
                    series=[
                        ChartSeries(
                            name="S1",
                            points=[
                                ChartPoint(x="Q1", y=10.0),
                                ChartPoint(x="Q2", y=20.0),
                            ],
                        )
                    ],
                )
            ],
        )
        fewer_points = DocumentSpec(
            doc_kind=DocKind.report,
            title="Chart Doc",
            blocks=[
                Chart(
                    type="chart",
                    chart_type="bar",
                    title="Sales",
                    series=[
                        ChartSeries(
                            name="S1",
                            points=[ChartPoint(x="Q1", y=10.0)],
                        )
                    ],
                )
            ],
        )
        ok, reasons = content_preserved(original, fewer_points)
        assert ok is False
        assert any("missing atom" in r for r in reasons)

    def test_injected_prose_sentence(self) -> None:
        """Replacing prose text is detected as a content change (missing original atom)."""
        original = DocumentSpec(
            doc_kind=DocKind.report,
            title="Doc",
            blocks=[Prose(type="prose", text="Original text only.")],
        )
        injected = DocumentSpec(
            doc_kind=DocKind.report,
            title="Doc",
            blocks=[
                Prose(
                    type="prose",
                    text="Original text only. Plus this injected sentence.",
                )
            ],
        )
        ok, reasons = content_preserved(original, injected)
        assert ok is False
        # The original prose atom is absent (replaced with longer text), so the
        # gate fires as "missing atom" (rule 2 fires before rule 3).
        assert len(reasons) > 0

    def test_split_table_passes(self) -> None:
        """A legitimately split table (same cells, cont. caption) should pass."""
        original = DocumentSpec(
            doc_kind=DocKind.report,
            title="Doc",
            blocks=[
                Table(
                    type="table",
                    caption="Data",
                    columns=["A", "B"],
                    rows=[["r1a", "r1b"], ["r2a", "r2b"], ["r3a", "r3b"]],
                )
            ],
        )
        split = DocumentSpec(
            doc_kind=DocKind.report,
            title="Doc",
            blocks=[
                Table(
                    type="table",
                    caption="Data",
                    columns=["A", "B"],
                    rows=[["r1a", "r1b"], ["r2a", "r2b"]],
                ),
                Table(
                    type="table",
                    caption="Data (cont.)",
                    columns=["A", "B"],
                    rows=[["r3a", "r3b"]],
                ),
            ],
        )
        ok, reasons = content_preserved(original, split)
        assert ok is True, f"Expected True, got reasons: {reasons}"
        assert reasons == []

    def test_split_prose_passes(self) -> None:
        """Legitimately split prose (two blocks with original text) should pass."""
        original = DocumentSpec(
            doc_kind=DocKind.report,
            title="Doc",
            blocks=[Prose(type="prose", text="First paragraph.\n\nSecond paragraph.")],
        )
        # After split_prose, the two blocks would have normalized text
        split = DocumentSpec(
            doc_kind=DocKind.report,
            title="Doc",
            blocks=[
                Prose(type="prose", text="First paragraph."),
                Prose(type="prose", text="Second paragraph."),
            ],
        )
        ok, reasons = content_preserved(original, split)
        assert ok is True, f"Expected True, got reasons: {reasons}"
        assert reasons == []


# ---------------------------------------------------------------------------
# map_pages_to_blocks / candidate_blocks_for_issue tests
# ---------------------------------------------------------------------------


def _make_fake_reader(page_texts: list[str]) -> Any:
    """Return a mock PdfReader whose pages yield the given texts."""
    pages = []
    for text in page_texts:
        page_mock = MagicMock()
        page_mock.extract_text.return_value = text
        pages.append(page_mock)
    reader_mock = MagicMock()
    reader_mock.pages = pages
    return reader_mock


class TestMapPagesToBlocks:
    def test_table_maps_to_page1_prose_to_page2(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        spec = DocumentSpec(
            doc_kind=DocKind.report,
            title="Test",
            blocks=[
                Table(
                    type="table",
                    caption="Sales Data",
                    columns=["Company", "Revenue"],
                    rows=[["Acme", "$10M"], ["Globex", "$20M"]],
                ),
                Prose(
                    type="prose",
                    text="This prose section contains information about growth.",
                ),
            ],
        )

        # Page 1 text matches table content; page 2 matches prose content
        fake_reader = _make_fake_reader(
            [
                "Sales Data Company Revenue Acme $10M Globex $20M",
                "This prose section contains information about growth.",
            ]
        )

        with patch("pypdf.PdfReader", return_value=fake_reader):
            page_map = map_pages_to_blocks(b"fake-pdf", spec)

        # Block 0 (Table) should map to page 1; block 1 (Prose) to page 2
        assert 0 in page_map.get(1, [])
        assert 1 in page_map.get(2, [])

    def test_overflow_issue_returns_table_block(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        spec = DocumentSpec(
            doc_kind=DocKind.report,
            title="Test",
            blocks=[
                Table(
                    type="table",
                    caption="Overflow Table",
                    columns=["A"],
                    rows=[["x"]],
                ),
                Prose(type="prose", text="Short prose."),
            ],
        )
        # page_map: page 1 → block 0 (the table)
        page_map = {1: [0]}
        issue = VisualIssue(
            page=1, category="overflow", severity="warning", message="overflow"
        )
        candidates = candidate_blocks_for_issue(issue, page_map, spec)
        assert 0 in candidates

    def test_labels_heuristic_returns_chart_blocks(self) -> None:
        spec = DocumentSpec(
            doc_kind=DocKind.report,
            title="Test",
            blocks=[
                Prose(type="prose", text="Some text."),
                Chart(
                    type="chart",
                    chart_type="bar",
                    title="Revenue",
                    series=[ChartSeries(name="S1", points=[ChartPoint(x="Q1", y=1.0)])],
                ),
            ],
        )
        page_map: dict[int, list[int]] = {}
        issue = VisualIssue(
            page=3, category="labels", severity="info", message="bad labels"
        )
        candidates = candidate_blocks_for_issue(issue, page_map, spec)
        assert 1 in candidates  # block index 1 is the Chart
        assert 0 not in candidates

    def test_hierarchy_heuristic(self) -> None:
        spec = DocumentSpec(
            doc_kind=DocKind.report,
            title="Test",
            blocks=[
                CoverHeader(type="cover_header", title="Big Title"),
                Heading(type="heading", text="Section"),
                SectionDivider(type="section_divider", title="Divider"),
                Prose(type="prose", text="Text."),
            ],
        )
        page_map: dict[int, list[int]] = {}
        issue = VisualIssue(
            page=5, category="hierarchy", severity="warning", message="hierarchy"
        )
        candidates = candidate_blocks_for_issue(issue, page_map, spec)
        # Should return CoverHeader (0), Heading (1), SectionDivider (2)
        assert 0 in candidates
        assert 1 in candidates
        assert 2 in candidates
        assert 3 not in candidates  # Prose not in hierarchy

    def test_no_match_returns_empty(self) -> None:
        spec = DocumentSpec(
            doc_kind=DocKind.report,
            title="Test",
            blocks=[Prose(type="prose", text="Short text.")],
        )
        page_map: dict[int, list[int]] = {}
        # "contrast" category with no matching blocks → []
        issue = VisualIssue(
            page=2, category="contrast", severity="info", message="low contrast"
        )
        candidates = candidate_blocks_for_issue(issue, page_map, spec)
        assert candidates == []

    def test_low_confidence_page_uses_overlap(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pages with very little text still record overlap-based block mapping."""
        spec = DocumentSpec(
            doc_kind=DocKind.report,
            title="Test",
            blocks=[
                Prose(type="prose", text="unique phrase alpha"),
            ],
        )
        # Only 3 tokens on page 1 — still some overlap expected
        fake_reader = _make_fake_reader(["unique phrase alpha"])
        with patch("pypdf.PdfReader", return_value=fake_reader):
            page_map = map_pages_to_blocks(b"fake-pdf", spec)
        assert 1 in page_map
        assert 0 in page_map[1]


# ---------------------------------------------------------------------------
# apply_patch tests
# ---------------------------------------------------------------------------


class TestApplyPatch:
    def _make_table_spec(
        self, rows: int, caption: str | None = "My Table"
    ) -> DocumentSpec:
        return DocumentSpec(
            doc_kind=DocKind.report,
            title="Doc",
            blocks=[
                Table(
                    type="table",
                    caption=caption,
                    columns=["Col1", "Col2"],
                    rows=[[f"r{i}a", f"r{i}b"] for i in range(rows)],
                )
            ],
        )

    def test_split_table_20_rows_max8(self) -> None:
        """20 rows with max=8 → 3 chunks: 8, 8, 4."""
        spec = self._make_table_spec(20)
        patch = VisualRevisionPatch(
            base_spec_hash=spec_hash(spec),
            operations=[SplitTable(block_index=0, max_rows_per_table=8)],
            rationale="Test",
        )
        result = apply_patch(spec, patch)
        assert isinstance(result, DocumentSpec)
        assert len(result.blocks) == 3
        tables = [b for b in result.blocks if isinstance(b, Table)]
        assert len(tables) == 3
        assert len(tables[0].rows) == 8
        assert len(tables[1].rows) == 8
        assert len(tables[2].rows) == 4
        # First chunk keeps original caption
        assert tables[0].caption == "My Table"
        # Continuation chunks get " (cont.)"
        assert tables[1].caption == "My Table (cont.)"
        assert tables[2].caption == "My Table (cont.)"

    def test_split_table_all_cells_preserved(self) -> None:
        spec = self._make_table_spec(20)
        patch = VisualRevisionPatch(
            base_spec_hash=spec_hash(spec),
            operations=[SplitTable(block_index=0, max_rows_per_table=8)],
            rationale="Test",
        )
        result = apply_patch(spec, patch)
        all_cells: list[str] = []
        for block in result.blocks:
            if isinstance(block, Table):
                for row in block.rows:
                    all_cells.extend(row)
        expected = [val for i in range(20) for val in [f"r{i}a", f"r{i}b"]]
        assert all_cells == expected

    def test_split_table_content_preserved(self) -> None:
        spec = self._make_table_spec(20)
        patch = VisualRevisionPatch(
            base_spec_hash=spec_hash(spec),
            operations=[SplitTable(block_index=0, max_rows_per_table=8)],
            rationale="Test",
        )
        result = apply_patch(spec, patch)
        ok, reasons = content_preserved(spec, result)
        assert ok is True, f"content_preserved failed: {reasons}"

    def test_split_table_no_caption(self) -> None:
        """Continuation chunks with no original caption get '(cont.)'."""
        spec = self._make_table_spec(15, caption=None)
        patch = VisualRevisionPatch(
            base_spec_hash=spec_hash(spec),
            operations=[SplitTable(block_index=0, max_rows_per_table=8)],
            rationale="Test",
        )
        result = apply_patch(spec, patch)
        tables = [b for b in result.blocks if isinstance(b, Table)]
        assert tables[0].caption is None
        assert tables[1].caption == "(cont.)"

    def test_split_prose_double_newline(self) -> None:
        """Prose with \\n\\n splits at paragraph boundary."""
        spec = DocumentSpec(
            doc_kind=DocKind.report,
            title="Doc",
            blocks=[Prose(type="prose", text="First paragraph.\n\nSecond paragraph.")],
        )
        patch = VisualRevisionPatch(
            base_spec_hash=spec_hash(spec),
            operations=[SplitProse(block_index=0)],
            rationale="Test",
        )
        result = apply_patch(spec, patch)
        prose_blocks = [b for b in result.blocks if isinstance(b, Prose)]
        assert len(prose_blocks) == 2
        texts = [b.text for b in prose_blocks]
        assert "First paragraph." in texts
        assert "Second paragraph." in texts

    def test_split_prose_content_preserved(self) -> None:
        spec = DocumentSpec(
            doc_kind=DocKind.report,
            title="Doc",
            blocks=[Prose(type="prose", text="First paragraph.\n\nSecond paragraph.")],
        )
        patch = VisualRevisionPatch(
            base_spec_hash=spec_hash(spec),
            operations=[SplitProse(block_index=0)],
            rationale="Test",
        )
        result = apply_patch(spec, patch)
        ok, reasons = content_preserved(spec, result)
        assert ok is True, f"content_preserved failed: {reasons}"

    def test_split_prose_sentence_boundary(self) -> None:
        """No \\n\\n → split at sentence boundary."""
        text = "First sentence. Second sentence. Third sentence."
        spec = DocumentSpec(
            doc_kind=DocKind.report,
            title="Doc",
            blocks=[Prose(type="prose", text=text)],
        )
        patch = VisualRevisionPatch(
            base_spec_hash=spec_hash(spec),
            operations=[SplitProse(block_index=0)],
            rationale="Test",
        )
        result = apply_patch(spec, patch)
        prose_blocks = [b for b in result.blocks if isinstance(b, Prose)]
        # Should have split at ". " boundaries before capital letters
        assert len(prose_blocks) > 1

    def test_wrong_hash_raises_value_error(self, simple_spec: DocumentSpec) -> None:
        patch = VisualRevisionPatch(
            base_spec_hash="wronghash",
            operations=[SplitTable(block_index=0, max_rows_per_table=5)],
            rationale="Test",
        )
        with pytest.raises(ValueError, match="mismatch"):
            apply_patch(simple_spec, patch)

    def test_result_is_document_spec(self, simple_spec: DocumentSpec) -> None:
        patch = VisualRevisionPatch(
            base_spec_hash=spec_hash(simple_spec),
            operations=[SplitTable(block_index=0, max_rows_per_table=2)],
            rationale="Test",
        )
        result = apply_patch(simple_spec, patch)
        assert isinstance(result, DocumentSpec)

    def test_original_spec_not_mutated(self) -> None:
        spec = DocumentSpec(
            doc_kind=DocKind.report,
            title="Doc",
            blocks=[
                Table(
                    type="table",
                    caption="T",
                    columns=["A"],
                    rows=[[str(i)] for i in range(20)],
                )
            ],
        )
        original_block_count = len(spec.blocks)
        patch = VisualRevisionPatch(
            base_spec_hash=spec_hash(spec),
            operations=[SplitTable(block_index=0, max_rows_per_table=5)],
            rationale="Test",
        )
        apply_patch(spec, patch)
        assert len(spec.blocks) == original_block_count

    def test_descending_order_for_multiple_ops(self) -> None:
        """Multiple ops applied in descending block_index order so indices stay valid."""
        spec = DocumentSpec(
            doc_kind=DocKind.report,
            title="Doc",
            blocks=[
                Table(
                    type="table",
                    caption="T1",
                    columns=["X"],
                    rows=[[str(i)] for i in range(15)],
                ),
                Table(
                    type="table",
                    caption="T2",
                    columns=["Y"],
                    rows=[[str(i)] for i in range(15)],
                ),
            ],
        )
        patch = VisualRevisionPatch(
            base_spec_hash=spec_hash(spec),
            operations=[
                SplitTable(block_index=0, max_rows_per_table=8),
                SplitTable(block_index=1, max_rows_per_table=8),
            ],
            rationale="Test",
        )
        result = apply_patch(spec, patch)
        tables = [b for b in result.blocks if isinstance(b, Table)]
        # Each 15-row table splits into 2 chunks → 4 total
        assert len(tables) == 4


# ---------------------------------------------------------------------------
# New op applier tests
# ---------------------------------------------------------------------------


class TestNewOpAppliers:
    def test_set_chart_labels_fills_missing(self) -> None:
        """SetChartLabels fills None title, x_axis_label, y_axis_label."""
        spec = DocumentSpec(
            doc_kind=DocKind.report,
            title="Doc",
            blocks=[
                Chart(
                    type="chart",
                    chart_type="bar",
                    title=None,
                    x_label=None,
                    y_label=None,
                    series=[
                        ChartSeries(name="Revenue", points=[ChartPoint(x="Q1", y=10.0)])
                    ],
                )
            ],
        )
        patch = VisualRevisionPatch(
            base_spec_hash=spec_hash(spec),
            operations=[
                SetChartLabels(
                    block_index=0,
                    title="Revenue by Quarter",
                    x_axis_label="Quarter",
                    y_axis_label="USD M",
                )
            ],
            rationale="Fill missing labels",
        )
        result = apply_patch(spec, patch)
        chart = result.blocks[0]
        assert isinstance(chart, Chart)
        assert chart.title == "Revenue by Quarter"
        assert chart.x_label == "Quarter"
        assert chart.y_label == "USD M"

    def test_set_chart_labels_does_not_overwrite_existing(self) -> None:
        """SetChartLabels must NOT overwrite existing non-empty labels."""
        spec = DocumentSpec(
            doc_kind=DocKind.report,
            title="Doc",
            blocks=[
                Chart(
                    type="chart",
                    chart_type="bar",
                    title="Existing Title",
                    x_label="Existing X",
                    y_label=None,
                    series=[
                        ChartSeries(name="S1", points=[ChartPoint(x="Q1", y=10.0)])
                    ],
                )
            ],
        )
        patch = VisualRevisionPatch(
            base_spec_hash=spec_hash(spec),
            operations=[
                SetChartLabels(
                    block_index=0,
                    title="New Title",
                    x_axis_label="New X",
                    y_axis_label="New Y",
                )
            ],
            rationale="Try to overwrite",
        )
        result = apply_patch(spec, patch)
        chart = result.blocks[0]
        assert isinstance(chart, Chart)
        assert chart.title == "Existing Title"  # NOT overwritten
        assert chart.x_label == "Existing X"  # NOT overwritten
        assert chart.y_label == "New Y"  # was None, so filled

    def test_set_chart_labels_fills_empty_series_name(self) -> None:
        """SetChartLabels fills empty series name positionally."""
        spec = DocumentSpec(
            doc_kind=DocKind.report,
            title="Doc",
            blocks=[
                Chart(
                    type="chart",
                    chart_type="bar",
                    series=[
                        ChartSeries(name="", points=[ChartPoint(x="Q1", y=1.0)]),
                        ChartSeries(
                            name="Existing", points=[ChartPoint(x="Q1", y=2.0)]
                        ),
                    ],
                )
            ],
        )
        patch = VisualRevisionPatch(
            base_spec_hash=spec_hash(spec),
            operations=[
                SetChartLabels(block_index=0, series_names=["Revenue", "Costs"])
            ],
            rationale="Fill series names",
        )
        result = apply_patch(spec, patch)
        chart = result.blocks[0]
        assert isinstance(chart, Chart)
        assert chart.series[0].name == "Revenue"  # was empty, filled
        assert chart.series[1].name == "Existing"  # was non-empty, NOT overwritten

    def test_change_chart_type_bar_to_line(self) -> None:
        """ChangeChartType: bar → line succeeds."""
        spec = DocumentSpec(
            doc_kind=DocKind.report,
            title="Doc",
            blocks=[
                Chart(
                    type="chart",
                    chart_type="bar",
                    series=[ChartSeries(name="S1", points=[ChartPoint(x="Q1", y=1.0)])],
                )
            ],
        )
        patch = VisualRevisionPatch(
            base_spec_hash=spec_hash(spec),
            operations=[ChangeChartType(block_index=0, chart_type="line")],
            rationale="Switch to line",
        )
        result = apply_patch(spec, patch)
        chart = result.blocks[0]
        assert isinstance(chart, Chart)
        assert chart.chart_type == "line"

    def test_change_chart_type_pie_to_line_raises(self) -> None:
        """ChangeChartType: pie → line raises ValueError (incompatible)."""
        spec = DocumentSpec(
            doc_kind=DocKind.report,
            title="Doc",
            blocks=[
                Chart(
                    type="chart",
                    chart_type="pie",
                    series=[ChartSeries(name="S1", points=[ChartPoint(x="A", y=1.0)])],
                )
            ],
        )
        patch = VisualRevisionPatch(
            base_spec_hash=spec_hash(spec),
            operations=[ChangeChartType(block_index=0, chart_type="line")],
            rationale="Incompatible change",
        )
        with pytest.raises(ValueError, match="incompatible"):
            apply_patch(spec, patch)

    def test_compact_stat_cards_moves_notes_to_prose(self) -> None:
        """CompactStatCards moves note text to a Prose block inserted after StatCards."""
        spec = DocumentSpec(
            doc_kind=DocKind.report,
            title="Doc",
            blocks=[
                StatCards(
                    type="stat_cards",
                    cards=[
                        StatCard(value="$1B", label="Revenue", note="YoY growth"),
                        StatCard(value="42%", label="Growth", note=None),
                    ],
                )
            ],
        )
        patch = VisualRevisionPatch(
            base_spec_hash=spec_hash(spec),
            operations=[CompactStatCards(block_index=0)],
            rationale="Compact notes",
        )
        result = apply_patch(spec, patch)
        assert len(result.blocks) == 2  # StatCards + Prose
        stat_cards = result.blocks[0]
        prose = result.blocks[1]
        assert isinstance(stat_cards, StatCards)
        assert isinstance(prose, Prose)
        # Notes cleared on cards
        for card in stat_cards.cards:
            assert card.note is None
        # Note text preserved verbatim in prose
        assert "YoY growth" in prose.text

    def test_set_theme_known_theme_succeeds(self) -> None:
        """SetTheme with a known theme name updates spec.theme."""
        spec = DocumentSpec(
            doc_kind=DocKind.report,
            title="Doc",
            blocks=[Prose(type="prose", text="Hello.")],
        )
        patch = VisualRevisionPatch(
            base_spec_hash=spec_hash(spec),
            operations=[SetTheme(theme="pitch")],
            rationale="Switch theme",
        )
        result = apply_patch(spec, patch)
        assert result.theme == "pitch"

    def test_set_theme_unknown_theme_raises(self) -> None:
        """SetTheme with unknown theme raises ValueError."""
        spec = DocumentSpec(
            doc_kind=DocKind.report,
            title="Doc",
            blocks=[Prose(type="prose", text="Hello.")],
        )
        patch = VisualRevisionPatch(
            base_spec_hash=spec_hash(spec),
            operations=[SetTheme(theme="nonexistent_theme_xyz")],
            rationale="Bad theme",
        )
        with pytest.raises(ValueError, match="unknown theme"):
            apply_patch(spec, patch)


# ---------------------------------------------------------------------------
# content_preserved extension tests
# ---------------------------------------------------------------------------


class TestContentPreservedExtensions:
    def test_set_chart_labels_fill_passes(self) -> None:
        """After SetChartLabels fills missing labels, content_preserved → True."""
        original = DocumentSpec(
            doc_kind=DocKind.report,
            title="Doc",
            blocks=[
                Chart(
                    type="chart",
                    chart_type="bar",
                    title=None,
                    x_label=None,
                    y_label=None,
                    series=[
                        ChartSeries(name="Revenue", points=[ChartPoint(x="Q1", y=10.0)])
                    ],
                )
            ],
        )
        patch = VisualRevisionPatch(
            base_spec_hash=spec_hash(original),
            operations=[
                SetChartLabels(
                    block_index=0,
                    title="Revenue Chart",
                    x_axis_label="Quarter",
                    y_axis_label="USD M",
                )
            ],
            rationale="Fill labels",
        )
        revised = apply_patch(original, patch)
        ok, reasons = content_preserved(original, revised)
        assert ok is True, f"Expected True, got reasons: {reasons}"

    def test_removing_existing_chart_label_fails(self) -> None:
        """Hand-built spec that removes an existing chart title → content_preserved False."""
        original = DocumentSpec(
            doc_kind=DocKind.report,
            title="Doc",
            blocks=[
                Chart(
                    type="chart",
                    chart_type="bar",
                    title="Important Title",
                    series=[ChartSeries(name="S1", points=[ChartPoint(x="Q1", y=1.0)])],
                )
            ],
        )
        # Revised removes the title (set to None)
        revised = DocumentSpec(
            doc_kind=DocKind.report,
            title="Doc",
            blocks=[
                Chart(
                    type="chart",
                    chart_type="bar",
                    title=None,
                    series=[ChartSeries(name="S1", points=[ChartPoint(x="Q1", y=1.0)])],
                )
            ],
        )
        ok, reasons = content_preserved(original, revised)
        assert ok is False
        assert any("missing atom" in r for r in reasons)

    def test_compact_stat_cards_passes(self) -> None:
        """After CompactStatCards, note text moved to Prose → content_preserved True."""
        original = DocumentSpec(
            doc_kind=DocKind.report,
            title="Doc",
            blocks=[
                StatCards(
                    type="stat_cards",
                    cards=[
                        StatCard(value="$1B", label="Revenue", note="YoY growth"),
                    ],
                )
            ],
        )
        patch = VisualRevisionPatch(
            base_spec_hash=spec_hash(original),
            operations=[CompactStatCards(block_index=0)],
            rationale="Compact",
        )
        revised = apply_patch(original, patch)
        ok, reasons = content_preserved(original, revised)
        assert ok is True, f"Expected True, got reasons: {reasons}"

    def test_dropping_stat_note_entirely_fails(self) -> None:
        """A spec that drops a stat note entirely (not moved to prose) → False."""
        original = DocumentSpec(
            doc_kind=DocKind.report,
            title="Doc",
            blocks=[
                StatCards(
                    type="stat_cards",
                    cards=[
                        StatCard(value="$1B", label="Revenue", note="YoY growth"),
                    ],
                )
            ],
        )
        # Revised drops the note entirely (not moved anywhere)
        revised = DocumentSpec(
            doc_kind=DocKind.report,
            title="Doc",
            blocks=[
                StatCards(
                    type="stat_cards",
                    cards=[
                        StatCard(value="$1B", label="Revenue", note=None),
                    ],
                )
            ],
        )
        ok, reasons = content_preserved(original, revised)
        assert ok is False
        assert any("missing atom" in r for r in reasons)


# ---------------------------------------------------------------------------
# propose_overflow_patch tests
# ---------------------------------------------------------------------------


class TestProposeOverflowPatch:
    def test_large_table_returns_patch(self) -> None:
        spec = DocumentSpec(
            doc_kind=DocKind.report,
            title="Doc",
            blocks=[
                Table(
                    type="table",
                    caption="Big Table",
                    columns=["A", "B"],
                    rows=[[f"r{i}a", f"r{i}b"] for i in range(20)],
                )
            ],
        )
        critique = VisualCritique(
            overall_score=6,
            summary="Overflow detected",
            issues=[
                VisualIssue(
                    page=1,
                    category="overflow",
                    severity="warning",
                    message="table overflow",
                )
            ],
        )
        page_map = {1: [0]}
        result = propose_overflow_patch(spec, critique, page_map)
        assert result is not None
        assert isinstance(result, VisualRevisionPatch)
        ops = result.operations
        assert len(ops) == 1
        assert isinstance(ops[0], SplitTable)
        assert ops[0].block_index == 0
        assert ops[0].max_rows_per_table == 12

    def test_no_overflow_issues_returns_none(self) -> None:
        spec = DocumentSpec(
            doc_kind=DocKind.report,
            title="Doc",
            blocks=[
                Table(
                    type="table",
                    caption="Table",
                    columns=["A"],
                    rows=[["x"]],
                )
            ],
        )
        critique = VisualCritique(
            overall_score=9,
            summary="Looks good",
            issues=[
                VisualIssue(
                    page=1, category="alignment", severity="info", message="minor"
                )
            ],
        )
        page_map: dict[int, list[int]] = {1: [0]}
        result = propose_overflow_patch(spec, critique, page_map)
        assert result is None

    def test_long_prose_overflow_returns_split_prose_patch(self) -> None:
        long_text = "A" * 600
        spec = DocumentSpec(
            doc_kind=DocKind.report,
            title="Doc",
            blocks=[Prose(type="prose", text=long_text)],
        )
        critique = VisualCritique(
            overall_score=5,
            summary="Overflow in prose",
            issues=[
                VisualIssue(
                    page=1,
                    category="overflow",
                    severity="error",
                    message="prose overflow",
                )
            ],
        )
        page_map = {1: [0]}
        result = propose_overflow_patch(spec, critique, page_map)
        assert result is not None
        ops = result.operations
        assert len(ops) == 1
        assert isinstance(ops[0], SplitProse)
        assert ops[0].block_index == 0

    def test_short_prose_not_included(self) -> None:
        """Prose ≤500 chars doesn't generate a SplitProse op even on overflow page."""
        short_text = "Short text."
        spec = DocumentSpec(
            doc_kind=DocKind.report,
            title="Doc",
            blocks=[Prose(type="prose", text=short_text)],
        )
        critique = VisualCritique(
            overall_score=5,
            summary="Overflow",
            issues=[
                VisualIssue(
                    page=1, category="overflow", severity="warning", message="overflow"
                )
            ],
        )
        page_map = {1: [0]}
        result = propose_overflow_patch(spec, critique, page_map)
        assert result is None

    def test_no_duplicate_ops_for_same_block(self) -> None:
        """Multiple overflow issues on the same block produce only one op."""
        spec = DocumentSpec(
            doc_kind=DocKind.report,
            title="Doc",
            blocks=[
                Table(
                    type="table",
                    caption="T",
                    columns=["A"],
                    rows=[[str(i)] for i in range(20)],
                )
            ],
        )
        critique = VisualCritique(
            overall_score=4,
            summary="Multiple overflow issues",
            issues=[
                VisualIssue(
                    page=1, category="overflow", severity="error", message="p1"
                ),
                VisualIssue(
                    page=2, category="overflow", severity="warning", message="p2"
                ),
            ],
        )
        page_map = {1: [0], 2: [0]}
        result = propose_overflow_patch(spec, critique, page_map)
        assert result is not None
        assert len(result.operations) == 1

    def test_patch_base_hash_matches_spec(self) -> None:
        spec = DocumentSpec(
            doc_kind=DocKind.report,
            title="Doc",
            blocks=[
                Table(
                    type="table",
                    caption="T",
                    columns=["A"],
                    rows=[[str(i)] for i in range(15)],
                )
            ],
        )
        critique = VisualCritique(
            overall_score=5,
            summary="Overflow",
            issues=[
                VisualIssue(page=1, category="overflow", severity="error", message="x")
            ],
        )
        page_map = {1: [0]}
        result = propose_overflow_patch(spec, critique, page_map)
        assert result is not None
        assert result.base_spec_hash == spec_hash(spec)


# ---------------------------------------------------------------------------
# propose_llm_patch tests
# ---------------------------------------------------------------------------


class TestProposeLlmPatch:
    def _labels_spec(self) -> DocumentSpec:
        return DocumentSpec(
            doc_kind=DocKind.report,
            title="Doc",
            blocks=[
                Chart(
                    type="chart",
                    chart_type="bar",
                    title=None,
                    series=[
                        ChartSeries(name="Revenue", points=[ChartPoint(x="Q1", y=1.0)])
                    ],
                )
            ],
        )

    def test_valid_propose_fn_returns_patch(self) -> None:
        """propose_fn returning a valid SetChartLabels patch → returned."""
        spec = self._labels_spec()
        critique = VisualCritique(
            overall_score=5,
            summary="Missing labels",
            issues=[
                VisualIssue(
                    page=1,
                    category="labels",
                    severity="warning",
                    message="no chart title",
                )
            ],
        )
        page_map: dict[int, list[int]] = {1: [0]}

        def fake_propose(ctx: LlmPatchContext) -> VisualRevisionPatch | None:
            return VisualRevisionPatch(
                base_spec_hash=ctx.base_spec_hash,
                operations=[SetChartLabels(block_index=0, title="Revenue Chart")],
                rationale="Add title",
            )

        result = propose_llm_patch(spec, critique, page_map, propose_fn=fake_propose)
        assert result is not None
        assert len(result.operations) == 1
        assert isinstance(result.operations[0], SetChartLabels)

    def test_bad_hash_patch_rejected(self) -> None:
        """propose_fn returning a patch with wrong base_spec_hash → None."""
        spec = self._labels_spec()
        critique = VisualCritique(
            overall_score=5,
            summary="Bad",
            issues=[
                VisualIssue(
                    page=1, category="labels", severity="warning", message="no title"
                )
            ],
        )
        page_map: dict[int, list[int]] = {}

        def fake_propose(ctx: LlmPatchContext) -> VisualRevisionPatch | None:
            return VisualRevisionPatch(
                base_spec_hash="badhashvalue",
                operations=[SetChartLabels(block_index=0, title="Title")],
                rationale="Bad hash patch",
            )

        result = propose_llm_patch(spec, critique, page_map, propose_fn=fake_propose)
        assert result is None

    def test_no_non_overflow_issues_returns_none(self) -> None:
        """Critique with only overflow issues → no candidates → None."""
        spec = self._labels_spec()
        critique = VisualCritique(
            overall_score=4,
            summary="Overflow only",
            issues=[
                VisualIssue(
                    page=1, category="overflow", severity="error", message="overflow"
                )
            ],
        )
        page_map: dict[int, list[int]] = {}
        called = [False]

        def fake_propose(ctx: LlmPatchContext) -> VisualRevisionPatch | None:
            called[0] = True
            return None

        result = propose_llm_patch(spec, critique, page_map, propose_fn=fake_propose)
        assert result is None
        assert not called[0]  # propose_fn not called when no non-overflow issues


# ---------------------------------------------------------------------------
# attempt_visual_revision tests
# ---------------------------------------------------------------------------


def _overflow_spec() -> DocumentSpec:
    """A spec with a large table that triggers an overflow patch."""
    return DocumentSpec(
        doc_kind=DocKind.report,
        title="Overflow Doc",
        blocks=[
            Table(
                type="table",
                caption="Big Table",
                columns=["A", "B"],
                rows=[[f"r{i}a", f"r{i}b"] for i in range(20)],
            )
        ],
    )


def _overflow_critique(score: int = 5) -> VisualCritique:
    """A VisualCritique with an overflow issue at score *score*."""
    return VisualCritique(
        overall_score=score,
        summary="Overflow detected",
        issues=[
            VisualIssue(
                page=1,
                category="overflow",
                severity="error",
                message="table overflow",
            )
        ],
    )


def _good_pdf() -> bytes:
    """Fake PDF bytes that pass validate_render (starts with %PDF)."""
    return b"%PDF-1.4 fake pdf bytes for testing purposes only"


def _better_pdf() -> bytes:
    """Second distinct fake PDF bytes (distinct from _good_pdf to be sure)."""
    return b"%PDF-1.4 revised candidate pdf for testing purposes"


class TestAttemptVisualRevision:
    def test_noop_score_acceptable(self) -> None:
        """Score >= trigger_below → noop immediately."""
        spec = _overflow_spec()
        critique = _overflow_critique(score=7)  # trigger_below=7, score==7 → noop

        outcome = attempt_visual_revision(
            spec,
            critique,
            render=lambda s: _good_pdf(),
            critique_fn=lambda pdf: _overflow_critique(score=8),
            trigger_below=7,
        )
        assert outcome.status == "noop"
        assert "acceptable" in outcome.reason

    def test_noop_no_overflow_patch(self) -> None:
        """No overflow issues → propose_overflow_patch returns None → noop."""
        spec = DocumentSpec(
            doc_kind=DocKind.report,
            title="Clean Doc",
            blocks=[Prose(type="prose", text="Short clean text.")],
        )
        critique = VisualCritique(
            overall_score=4,
            summary="Alignment issue only",
            issues=[
                VisualIssue(
                    page=1,
                    category="alignment",
                    severity="warning",
                    message="slight misalignment",
                )
            ],
        )
        outcome = attempt_visual_revision(
            spec,
            critique,
            render=lambda s: _good_pdf(),
            critique_fn=lambda pdf: VisualCritique(
                overall_score=8, summary="Good", issues=[]
            ),
        )
        assert outcome.status == "noop"
        assert "no actionable" in outcome.reason

    def test_happy_path_accepted(self) -> None:
        """Overflow table spec → patch applied → render → better critique → accepted."""
        spec = _overflow_spec()
        critique = _overflow_critique(score=5)

        call_count = [0]

        def fake_render(s: DocumentSpec) -> bytes:
            call_count[0] += 1
            return _better_pdf()

        def fake_critique(pdf: bytes) -> VisualCritique:
            return VisualCritique(overall_score=8, summary="Better now", issues=[])

        outcome = attempt_visual_revision(
            spec,
            critique,
            render=fake_render,
            critique_fn=fake_critique,
        )
        assert outcome.status == "accepted"
        assert outcome.revised_spec is not None
        # revised_spec should have more blocks (split table)
        assert len(outcome.revised_spec.blocks) > len(spec.blocks)
        assert outcome.revised_pdf == _better_pdf()
        assert outcome.new_critique is not None
        assert outcome.new_critique.overall_score == 8
        kinds = [e.kind for e in outcome.events]
        assert "visual_revision_accepted" in kinds

    def test_reject_no_improvement_same_score_fewer_errors(self) -> None:
        """Same score but fewer error-severity issues → accepted (tiebreak)."""
        spec = _overflow_spec()
        # original has 1 error-severity issue
        critique = VisualCritique(
            overall_score=5,
            summary="Overflow",
            issues=[
                VisualIssue(
                    page=1, category="overflow", severity="error", message="overflow"
                ),
            ],
        )

        def fake_critique(pdf: bytes) -> VisualCritique:
            # Same score but 0 error-severity issues → tiebreak accepts
            return VisualCritique(
                overall_score=5,
                summary="Better alignment but same score",
                issues=[
                    VisualIssue(
                        page=1,
                        category="alignment",
                        severity="warning",
                        message="minor",
                    )
                ],
            )

        outcome = attempt_visual_revision(
            spec,
            critique,
            render=lambda s: _better_pdf(),
            critique_fn=fake_critique,
        )
        assert outcome.status == "accepted"

    def test_reject_no_improvement(self) -> None:
        """Same score AND same number of error-severity issues → rejected."""
        spec = _overflow_spec()
        critique = VisualCritique(
            overall_score=5,
            summary="Overflow",
            issues=[
                VisualIssue(
                    page=1, category="overflow", severity="error", message="overflow"
                ),
            ],
        )

        def fake_critique(pdf: bytes) -> VisualCritique:
            # Same score, same number of errors → no improvement
            return VisualCritique(
                overall_score=5,
                summary="Still bad",
                issues=[
                    VisualIssue(
                        page=1,
                        category="overflow",
                        severity="error",
                        message="still overflow",
                    ),
                ],
            )

        outcome = attempt_visual_revision(
            spec,
            critique,
            render=lambda s: _better_pdf(),
            critique_fn=fake_critique,
        )
        assert outcome.status == "rejected"
        assert "no improvement" in outcome.reason

    def test_reject_content_not_preserved(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Monkeypatched content_preserved returns False → rejected."""
        import kortny.documents.revision as rev_mod

        spec = _overflow_spec()
        critique = _overflow_critique(score=5)

        monkeypatch.setattr(
            rev_mod,
            "content_preserved",
            lambda orig, revised: (False, ["missing atom: ('table_cell', 'r0a')"]),
        )

        outcome = attempt_visual_revision(
            spec,
            critique,
            render=lambda s: _better_pdf(),
            critique_fn=lambda pdf: VisualCritique(
                overall_score=9, summary="Good", issues=[]
            ),
        )
        assert outcome.status == "rejected"
        assert "content not preserved" in outcome.reason

    def test_reject_render_fail(self) -> None:
        """Render raises → rejected, no exception propagates out."""
        spec = _overflow_spec()
        critique = _overflow_critique(score=5)

        def exploding_render(s: DocumentSpec) -> bytes:
            raise RuntimeError("renderer exploded")

        outcome = attempt_visual_revision(
            spec,
            critique,
            render=exploding_render,
            critique_fn=lambda pdf: VisualCritique(
                overall_score=9, summary="Good", issues=[]
            ),
        )
        assert outcome.status == "rejected"
        assert "render failed" in outcome.reason

    def test_reject_validate_render_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """validate_render returns an error-severity issue → rejected."""
        import kortny.documents.revision as rev_mod
        from kortny.documents.critique import DocumentIssue

        spec = _overflow_spec()
        critique = _overflow_critique(score=5)

        monkeypatch.setattr(
            rev_mod,
            "validate_render",
            lambda data, fmt: [
                DocumentIssue(
                    code="invalid_pdf", severity="error", message="Not a PDF."
                )
            ],
        )

        outcome = attempt_visual_revision(
            spec,
            critique,
            render=lambda s: _better_pdf(),
            critique_fn=lambda pdf: VisualCritique(
                overall_score=9, summary="Good", issues=[]
            ),
        )
        assert outcome.status == "rejected"
        assert "render validation" in outcome.reason

    def test_events_noop_score(self) -> None:
        """Noop (high score) path: events contain started + noop."""
        spec = _overflow_spec()
        critique = _overflow_critique(score=9)  # well above trigger_below=7

        outcome = attempt_visual_revision(
            spec,
            critique,
            render=lambda s: _good_pdf(),
            critique_fn=lambda pdf: VisualCritique(
                overall_score=9, summary="Good", issues=[]
            ),
        )
        assert outcome.status == "noop"
        kinds = [e.kind for e in outcome.events]
        assert "visual_revision_started" in kinds
        assert "visual_revision_noop" in kinds
        # started event should have old_score set
        started = next(e for e in outcome.events if e.kind == "visual_revision_started")
        assert started.old_score == 9

    def test_events_accepted(self) -> None:
        """Accepted path: events contain started + accepted with old/new scores."""
        spec = _overflow_spec()
        critique = _overflow_critique(score=5)

        outcome = attempt_visual_revision(
            spec,
            critique,
            render=lambda s: _better_pdf(),
            critique_fn=lambda pdf: VisualCritique(
                overall_score=9, summary="Good", issues=[]
            ),
        )
        assert outcome.status == "accepted"
        kinds = [e.kind for e in outcome.events]
        assert "visual_revision_started" in kinds
        assert "visual_revision_accepted" in kinds
        accepted = next(
            e for e in outcome.events if e.kind == "visual_revision_accepted"
        )
        assert accepted.old_score == 5
        assert accepted.new_score == 9

    def test_deterministic_overflow_takes_priority_over_llm(self) -> None:
        """Overflow patch (deterministic) is proposed before calling llm_propose_fn."""
        spec = _overflow_spec()
        critique = _overflow_critique(score=5)

        llm_called = [False]

        def fake_llm_propose(ctx: LlmPatchContext) -> VisualRevisionPatch | None:
            llm_called[0] = True
            return None

        outcome = attempt_visual_revision(
            spec,
            critique,
            render=lambda s: _better_pdf(),
            critique_fn=lambda pdf: VisualCritique(
                overall_score=8, summary="Good", issues=[]
            ),
            llm_propose_fn=fake_llm_propose,
        )
        # Overflow patch should have been found and applied, so llm_propose_fn not called
        assert outcome.status == "accepted"
        assert not llm_called[0]

    def test_llm_patch_applied_when_no_overflow(self) -> None:
        """No overflow + llm_propose_fn proposing set_chart_labels + improved critique → accepted."""
        spec = DocumentSpec(
            doc_kind=DocKind.report,
            title="Labels Doc",
            blocks=[
                Chart(
                    type="chart",
                    chart_type="bar",
                    title=None,
                    series=[
                        ChartSeries(name="Revenue", points=[ChartPoint(x="Q1", y=1.0)])
                    ],
                )
            ],
        )
        critique = VisualCritique(
            overall_score=5,
            summary="Missing labels",
            issues=[
                VisualIssue(
                    page=1, category="labels", severity="warning", message="no title"
                )
            ],
        )

        def fake_llm_propose(ctx: LlmPatchContext) -> VisualRevisionPatch | None:
            return VisualRevisionPatch(
                base_spec_hash=ctx.base_spec_hash,
                operations=[SetChartLabels(block_index=0, title="Revenue Chart")],
                rationale="Add title",
            )

        outcome = attempt_visual_revision(
            spec,
            critique,
            render=lambda s: _better_pdf(),
            critique_fn=lambda pdf: VisualCritique(
                overall_score=8, summary="Better", issues=[]
            ),
            llm_propose_fn=fake_llm_propose,
        )
        assert outcome.status == "accepted"
        assert outcome.revised_spec is not None

    def test_llm_patch_content_loss_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """LLM patch that loses content (content_preserved forced False) → rejected."""
        import kortny.documents.revision as rev_mod

        spec = DocumentSpec(
            doc_kind=DocKind.report,
            title="Labels Doc",
            blocks=[
                Chart(
                    type="chart",
                    chart_type="bar",
                    title=None,
                    series=[
                        ChartSeries(name="Revenue", points=[ChartPoint(x="Q1", y=1.0)])
                    ],
                )
            ],
        )
        critique = VisualCritique(
            overall_score=5,
            summary="Missing labels",
            issues=[
                VisualIssue(
                    page=1, category="labels", severity="warning", message="no title"
                )
            ],
        )

        monkeypatch.setattr(
            rev_mod,
            "content_preserved",
            lambda orig, revised: (
                False,
                ["injected atom: ('chart_title', 'bad')"],
            ),
        )

        def fake_llm_propose(ctx: LlmPatchContext) -> VisualRevisionPatch | None:
            return VisualRevisionPatch(
                base_spec_hash=ctx.base_spec_hash,
                operations=[SetChartLabels(block_index=0, title="Revenue Chart")],
                rationale="Add title",
            )

        outcome = attempt_visual_revision(
            spec,
            critique,
            render=lambda s: _better_pdf(),
            critique_fn=lambda pdf: VisualCritique(
                overall_score=8, summary="Better", issues=[]
            ),
            llm_propose_fn=fake_llm_propose,
        )
        assert outcome.status == "rejected"
        assert "content not preserved" in outcome.reason

    def test_no_llm_propose_fn_no_overflow_is_noop(self) -> None:
        """llm_propose_fn=None + no overflow → noop (no LLM call possible)."""
        spec = DocumentSpec(
            doc_kind=DocKind.report,
            title="Labels Doc",
            blocks=[
                Chart(
                    type="chart",
                    chart_type="bar",
                    title=None,
                    series=[
                        ChartSeries(name="Revenue", points=[ChartPoint(x="Q1", y=1.0)])
                    ],
                )
            ],
        )
        critique = VisualCritique(
            overall_score=5,
            summary="Missing labels",
            issues=[
                VisualIssue(
                    page=1, category="labels", severity="warning", message="no title"
                )
            ],
        )

        outcome = attempt_visual_revision(
            spec,
            critique,
            render=lambda s: _better_pdf(),
            critique_fn=lambda pdf: VisualCritique(
                overall_score=8, summary="Better", issues=[]
            ),
            llm_propose_fn=None,
        )
        assert outcome.status == "noop"

    def test_two_iteration_improvement(self) -> None:
        """Two iterations: iter1 5→7, iter2 7→8 (different patches each round) → accepted, score 8."""
        # Spec with two independent large tables, each fixable in a separate iteration.
        spec = DocumentSpec(
            doc_kind=DocKind.report,
            title="Two Iteration Doc",
            blocks=[
                Table(
                    type="table",
                    caption="Table A",
                    columns=["A", "B"],
                    rows=[[f"r{i}a", f"r{i}b"] for i in range(20)],
                ),
                Table(
                    type="table",
                    caption="Table B",
                    columns=["X", "Y"],
                    rows=[[f"s{i}x", f"s{i}y"] for i in range(20)],
                ),
            ],
        )

        critique_calls: list[int] = []
        score_sequence = [7, 8]  # iter1 produces score 7, iter2 produces score 8

        def fake_critique(pdf: bytes) -> VisualCritique:
            score = score_sequence[min(len(critique_calls), len(score_sequence) - 1)]
            critique_calls.append(score)
            issues = []
            if score < 8:
                # Still has an overflow issue to fix in the next iteration
                issues = [
                    VisualIssue(
                        page=1,
                        category="overflow",
                        severity="error",
                        message="table overflow",
                    )
                ]
            return VisualCritique(
                overall_score=score, summary=f"score {score}", issues=issues
            )

        outcome = attempt_visual_revision(
            spec,
            VisualCritique(
                overall_score=5,
                summary="initial",
                issues=[
                    VisualIssue(
                        page=1,
                        category="overflow",
                        severity="error",
                        message="overflow",
                    )
                ],
            ),
            render=lambda s: _better_pdf(),
            critique_fn=fake_critique,
            max_iterations=3,
            early_stop_at=9,
        )
        assert outcome.status == "accepted"
        assert outcome.new_critique is not None
        assert outcome.new_critique.overall_score == 8
        # Events should have started + accepted (outer), and candidate rejection/noop per iteration
        kinds = [e.kind for e in outcome.events]
        assert "visual_revision_started" in kinds
        assert "visual_revision_accepted" in kinds
        accepted_ev = next(
            e for e in outcome.events if e.kind == "visual_revision_accepted"
        )
        assert accepted_ev.old_score == 5
        assert accepted_ev.new_score == 8
        # "2 iteration" in reason
        assert "2 iteration" in outcome.reason
        assert "5" in outcome.reason
        assert "8" in outcome.reason

    def test_early_stop_at_threshold(self) -> None:
        """Once score reaches early_stop_at=8, loop stops even if iterations remain."""
        spec = _overflow_spec()

        critique_calls: list[int] = []

        def fake_critique(pdf: bytes) -> VisualCritique:
            critique_calls.append(1)
            return VisualCritique(overall_score=8, summary="great", issues=[])

        outcome = attempt_visual_revision(
            spec,
            _overflow_critique(score=5),
            render=lambda s: _better_pdf(),
            critique_fn=fake_critique,
            max_iterations=5,
            early_stop_at=8,
        )
        assert outcome.status == "accepted"
        assert outcome.new_critique is not None
        assert outcome.new_critique.overall_score == 8
        # Should stop after first iteration reaching early_stop_at
        # Only one critique call (iter 1 produces 8 → loop stops before iter 2)
        assert len(critique_calls) == 1

    def test_max_iterations_cap(self) -> None:
        """Never more than max_iterations critique calls."""
        spec = _overflow_spec()
        critique_calls: list[VisualCritique] = []

        # Each call returns score 6, still has overflow → would loop forever without cap
        def fake_critique(pdf: bytes) -> VisualCritique:
            c = VisualCritique(
                overall_score=6,
                summary="still bad",
                issues=[
                    VisualIssue(
                        page=1,
                        category="overflow",
                        severity="error",
                        message="overflow",
                    )
                ],
            )
            critique_calls.append(c)
            return c

        attempt_visual_revision(
            spec,
            _overflow_critique(score=5),
            render=lambda s: _better_pdf(),
            critique_fn=fake_critique,
            max_iterations=2,
        )
        # At most 2 critique calls (one per iteration)
        assert len(critique_calls) <= 2

    def test_no_progress_stop(self) -> None:
        """Iter1 accepts (5→7); iter2 produces no improvement → returns iter1 result."""
        spec = _overflow_spec()

        call_count = [0]

        def fake_critique(pdf: bytes) -> VisualCritique:
            call_count[0] += 1
            if call_count[0] == 1:
                # Iter 1: improvement to 7
                return VisualCritique(
                    overall_score=7,
                    summary="better",
                    issues=[
                        VisualIssue(
                            page=1,
                            category="overflow",
                            severity="error",
                            message="overflow",
                        )
                    ],
                )
            else:
                # Iter 2: no improvement (score stays at 7)
                return VisualCritique(
                    overall_score=7,
                    summary="same",
                    issues=[
                        VisualIssue(
                            page=1,
                            category="overflow",
                            severity="error",
                            message="overflow",
                        )
                    ],
                )

        outcome = attempt_visual_revision(
            spec,
            _overflow_critique(score=5),
            render=lambda s: _better_pdf(),
            critique_fn=fake_critique,
            max_iterations=3,
            trigger_below=7,
        )
        # Should have accepted iter1's result
        assert outcome.status == "accepted"
        assert outcome.new_critique is not None
        assert outcome.new_critique.overall_score == 7

    def test_oscillation_guard(self) -> None:
        """A patch targeting the same (op, block_index) set as a previous accepted patch → loop stops."""
        # Use a spec with one large table (block 0) that keeps overflowing
        spec = _overflow_spec()

        call_count = [0]

        def fake_critique(pdf: bytes) -> VisualCritique:
            call_count[0] += 1
            # Always return improving score, always with overflow (to keep proposing)
            # but since the patch targets the same block, oscillation guard triggers
            score = min(5 + call_count[0], 8)
            return VisualCritique(
                overall_score=score,
                summary=f"score {score}",
                issues=[
                    VisualIssue(
                        page=1,
                        category="overflow",
                        severity="error",
                        message="overflow",
                    )
                ],
            )

        # We need the render to return something that still has the big table
        # so the overflow patch keeps targeting block 0.
        # But after iter1, the spec has been split. So overflow patch on iter2
        # would target a DIFFERENT block (the split chunks). To test oscillation,
        # we need to force the same patch signature.
        #
        # The cleanest way: monkeypatch propose_overflow_patch to always return
        # a patch for block_index=0, so the signature repeats.
        from unittest.mock import patch as mock_patch

        from kortny.documents.revision import SplitTable as _SplitTable
        from kortny.documents.revision import VisualRevisionPatch as _VRP
        from kortny.documents.revision import spec_hash as _spec_hash

        def always_same_patch(s: Any, c: Any, pm: Any) -> _VRP:
            return _VRP(
                base_spec_hash=_spec_hash(s),
                operations=[_SplitTable(block_index=0, max_rows_per_table=12)],
                rationale="overflow fix",
            )

        with mock_patch(
            "kortny.documents.revision.propose_overflow_patch",
            side_effect=always_same_patch,
        ):
            outcome = attempt_visual_revision(
                spec,
                _overflow_critique(score=5),
                render=lambda s: _better_pdf(),
                critique_fn=fake_critique,
                max_iterations=5,
                early_stop_at=10,
            )

        # Oscillation guard should have stopped after iter 2 (iter 1 accepted with sig {(split_table, 0)},
        # iter 2 would try the same sig → break before accepting)
        assert outcome.status == "accepted"
        # Only 1 critique call because iter1 accepts, iter2 detects oscillation before critique
        # Actually oscillation guard runs AFTER accept_if_improved, so critique may be called twice.
        # The guard: if result.status == "accepted" and patch_signature in seen → break (without appending to accepted).
        # So iter1: accepted, sig added. Iter2: accepted result comes back from _single_revision_attempt,
        # but outer loop sees sig already in seen → breaks.
        assert call_count[0] <= 2

    def test_max_iterations_1_reproduces_current_behavior(self) -> None:
        """max_iterations=1 must reproduce the EXACT current single-attempt behavior."""
        spec = _overflow_spec()
        critique = _overflow_critique(score=5)

        outcome = attempt_visual_revision(
            spec,
            critique,
            render=lambda s: _better_pdf(),
            critique_fn=lambda pdf: VisualCritique(
                overall_score=8, summary="Better now", issues=[]
            ),
            max_iterations=1,
        )
        # Same as test_happy_path_accepted
        assert outcome.status == "accepted"
        assert outcome.revised_spec is not None
        assert len(outcome.revised_spec.blocks) > len(spec.blocks)
        assert outcome.revised_pdf == _better_pdf()
        assert outcome.new_critique is not None
        assert outcome.new_critique.overall_score == 8
        kinds = [e.kind for e in outcome.events]
        assert "visual_revision_accepted" in kinds
        accepted = next(
            e for e in outcome.events if e.kind == "visual_revision_accepted"
        )
        assert accepted.old_score == 5
        assert accepted.new_score == 8

    def test_never_raises_render_failure_mid_loop(self) -> None:
        """A render failure on iteration 2 (after iter1 accepted) → returns iter1 best, no raise."""
        spec = _overflow_spec()

        call_count = [0]

        def fake_critique(pdf: bytes) -> VisualCritique:
            call_count[0] += 1
            return VisualCritique(
                overall_score=7,
                summary="ok",
                issues=[
                    VisualIssue(
                        page=1,
                        category="overflow",
                        severity="error",
                        message="overflow",
                    )
                ],
            )

        render_count = [0]

        def fake_render(s: DocumentSpec) -> bytes:
            render_count[0] += 1
            if render_count[0] >= 2:
                raise RuntimeError("render exploded on iter 2")
            return _better_pdf()

        outcome = attempt_visual_revision(
            spec,
            _overflow_critique(score=5),
            render=fake_render,
            critique_fn=fake_critique,
            max_iterations=3,
        )
        # Must not raise; should return whatever was accepted before the failure
        # Iter 1 accepted (render_count=1 succeeds), iter 2 render fails → returns iter1 best
        assert outcome.status == "accepted"
        assert outcome.new_critique is not None
        assert outcome.new_critique.overall_score == 7
