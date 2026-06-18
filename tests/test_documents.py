"""Tests for the Document Studio core (HIG-244 Phase 1).

Writer tests are pure string assertions (no DB, no binary). The render test
exercises the real ``typst`` binary and skips when it is unavailable.
"""

from __future__ import annotations

import io

import pytest
from pypdf import PdfReader

from kortny.documents import (
    PITCH_THEME,
    REPORT_THEME,
    Chart,
    DocKind,
    DocumentSpec,
    compile_chart_spec,
    render_chart_png,
    render_chart_svg,
    render_document,
    render_spec_pdf,
    resolve_theme,
    theme_names,
    typst_available,
)
from kortny.documents.typst_writer import esc

_FULL_SPEC = {
    "doc_kind": "report",
    "title": "Quarterly $15B Review",
    "blocks": [
        {
            "type": "cover_header",
            "eyebrow": "Market Brief",
            "title": "The SpaceX IPO",
            "subtitle": "What a public offering means",
            "accent_tail": "IPO",
            "meta": ["Kortny", "2026"],
        },
        {"type": "heading", "text": "Why now"},
        {"type": "prose", "text": "First paragraph.\n\nSecond paragraph."},
        {
            "type": "stat_cards",
            "cards": [
                {"value": "$350B", "label": "Valuation", "note": "midpoint"},
                {"value": "62%", "label": "Starlink"},
            ],
        },
        {
            "type": "table",
            "caption": "Comparables",
            "columns": ["Company", "Raise", "Sector"],
            "rows": [["Arm", "$4.9B", "Semis"], ["Reddit", "$748M"]],
        },
        {"type": "callout", "label": "Takeaway", "text": "It matters."},
        {"type": "pull_quote", "text": "A public currency.", "attribution": "Analyst"},
        {"type": "cta", "label": "View more →", "text": "All the data."},
        {
            "type": "section_divider",
            "index": "01",
            "label": "Section One",
            "title": "The Setup",
            "subtitle": "How we got here",
        },
    ],
}


def _spec(**overrides: object) -> DocumentSpec:
    return DocumentSpec.model_validate({**_FULL_SPEC, **overrides})


# --------------------------------------------------------------------------- #
# Escaping
# --------------------------------------------------------------------------- #


def test_esc_escapes_math_dollar() -> None:
    # $ opens math mode in Typst and would break compilation on currency.
    assert esc("$15B") == "\\$15B"


def test_esc_escapes_all_sigils() -> None:
    assert esc("a#b[c]*_`@<>") == "a\\#b\\[c\\]\\*\\_\\`\\@\\<\\>"


def test_esc_escapes_backslash_first() -> None:
    assert esc("a\\b") == "a\\\\b"


# --------------------------------------------------------------------------- #
# Theme resolution
# --------------------------------------------------------------------------- #


def test_theme_names_includes_builtins() -> None:
    names = theme_names()
    assert "report" in names
    assert "pitch" in names


def test_resolve_theme_explicit_name_wins() -> None:
    assert resolve_theme(doc_kind=DocKind.report, name="pitch") is PITCH_THEME


def test_resolve_theme_defaults_by_kind() -> None:
    assert resolve_theme(doc_kind=DocKind.pitch, name=None) is PITCH_THEME
    assert resolve_theme(doc_kind=DocKind.report, name=None) is REPORT_THEME
    # brief/memo fall back to the report theme.
    assert resolve_theme(doc_kind=DocKind.memo, name=None) is REPORT_THEME


def test_resolve_theme_unknown_name_falls_back_not_raises() -> None:
    assert resolve_theme(doc_kind=DocKind.report, name="nope") is REPORT_THEME


# --------------------------------------------------------------------------- #
# Writer output
# --------------------------------------------------------------------------- #


def test_render_document_is_deterministic() -> None:
    assert render_document(_spec()) == render_document(_spec())


def test_render_document_emits_theme_tokens() -> None:
    src = render_document(_spec(theme="report"))
    assert f'#let accent = rgb("{REPORT_THEME.colors.accent}")' in src
    assert f'#let display = "{REPORT_THEME.display_font}"' in src
    assert 'set document(title: "Quarterly \\$15B Review")' in src


def test_cover_uses_full_bleed_dark_page_and_accent_tail() -> None:
    src = render_document(_spec())
    # Cover is a dedicated dark page, not a background div.
    assert "#page(fill: dividerBg, header: none, footer: none" in src
    # accent_tail splits the title so "IPO" is accent-coloured.
    assert "The SpaceX #text(fill: accent)[IPO]" in src


def test_section_divider_is_its_own_page() -> None:
    src = render_document(_spec())
    assert src.count("#page(fill: dividerBg") == 2  # cover + one divider


def test_prose_splits_paragraphs() -> None:
    src = render_document(_spec())
    assert "First paragraph." in src
    assert "Second paragraph." in src


def test_stat_cards_grid_matches_card_count() -> None:
    src = render_document(_spec())
    # Two cards -> a two-column grid.
    assert "#grid(columns: (1fr, 1fr), gutter: 12pt," in src


def test_table_pads_ragged_rows() -> None:
    # The second row has 2 cells for a 3-column table; it must be padded so the
    # Typst grid stays valid rather than throwing.
    src = render_document(_spec())
    # 3 columns x (1 header-less) + body; ensure Reddit row rendered 3 cells.
    assert src.count("[#text(font: bodyFont, size: 10pt)[Reddit]]") == 1
    assert "[#text(font: bodyFont, size: 10pt)[Semis]]" in src


def test_callout_panel_is_content_sized_not_fixed_height() -> None:
    src = render_document(_spec())
    # A Typst block with no height -> shrinks to content (no empty-void bug).
    assert "#block(width: 100%, fill: panelTint, inset: 18pt" in src
    assert "height" not in _callout_fragment(src)


def _callout_fragment(src: str) -> str:
    start = src.index("fill: panelTint")
    return src[start : start + 200]


def test_currency_in_card_value_is_escaped() -> None:
    src = render_document(_spec())
    assert "[\\$350B]" in src


# --------------------------------------------------------------------------- #
# Render (real typst binary)
# --------------------------------------------------------------------------- #

_TYPST_MISSING = not typst_available()


@pytest.mark.skipif(_TYPST_MISSING, reason="typst binary not installed")
def test_render_spec_pdf_produces_valid_pdf() -> None:
    pdf = render_spec_pdf(_spec())
    assert pdf.startswith(b"%PDF")
    reader = PdfReader(io.BytesIO(pdf))
    assert len(reader.pages) >= 1


@pytest.mark.skipif(_TYPST_MISSING, reason="typst binary not installed")
def test_render_spec_pdf_both_themes() -> None:
    for theme in theme_names():
        pdf = render_spec_pdf(_spec(theme=theme))
        assert pdf.startswith(b"%PDF")


# --------------------------------------------------------------------------- #
# PPTX / DOCX writers (pure-Python, no binary)
# --------------------------------------------------------------------------- #


def test_render_pptx_produces_valid_deck() -> None:
    from pptx import Presentation

    from kortny.documents import render_pptx

    data = render_pptx(_spec())
    assert data[:2] == b"PK"  # OOXML zip
    prs = Presentation(io.BytesIO(data))
    # cover -> title slide; section_divider -> section slide; at least one
    # content slide for the heading group => 3+ slides.
    assert len(list(prs.slides)) >= 3


def test_render_pptx_both_themes() -> None:
    from kortny.documents import render_pptx

    for theme in theme_names():
        assert render_pptx(_spec(theme=theme))[:2] == b"PK"


def test_render_docx_produces_valid_document() -> None:
    from docx import Document

    from kortny.documents import render_docx

    data = render_docx(_spec())
    assert data[:2] == b"PK"
    doc = Document(io.BytesIO(data))
    # stat_cards + table + callout each render as a table.
    assert len(doc.tables) >= 3
    assert any("SpaceX" in p.text for p in doc.paragraphs)


def test_render_docx_both_themes() -> None:
    from kortny.documents import render_docx

    for theme in theme_names():
        assert render_docx(_spec(theme=theme))[:2] == b"PK"


# --------------------------------------------------------------------------- #
# Charts (vl-convert)
# --------------------------------------------------------------------------- #

_BAR = Chart.model_validate(
    {
        "chart_type": "bar",
        "title": "Sales",
        "x_label": "Q",
        "y_label": "USD",
        "series": [
            {"name": "Rev", "points": [{"x": "Q1", "y": 3}, {"x": "Q2", "y": 5}]}
        ],
    }
)
_MULTI_LINE = Chart.model_validate(
    {
        "chart_type": "line",
        "series": [
            {"name": "A", "points": [{"x": 2024, "y": 3}, {"x": 2025, "y": 7}]},
            {"name": "B", "points": [{"x": 2024, "y": 5}, {"x": 2025, "y": 4}]},
        ],
    }
)


def test_chart_spec_single_series_uses_accent_no_legend() -> None:
    spec = compile_chart_spec(_BAR, REPORT_THEME)
    # Single series -> mark painted with the brand accent, no color encoding.
    assert spec["mark"] == {"type": "bar", "color": REPORT_THEME.colors.accent}
    assert "color" not in spec["encoding"]
    # Author category order preserved (no alphabetical re-sort).
    assert spec["encoding"]["x"]["sort"] is None


def test_chart_spec_multi_series_uses_palette_and_color() -> None:
    spec = compile_chart_spec(_MULTI_LINE, REPORT_THEME)
    assert spec["mark"] == "line"
    assert spec["encoding"]["color"]["field"] == "series"
    # Numeric x detected as quantitative.
    assert spec["encoding"]["x"]["type"] == "quantitative"
    # Brand accent leads the categorical palette; theme font applied.
    assert spec["config"]["range"]["category"][0] == REPORT_THEME.colors.accent
    assert spec["config"]["font"] == REPORT_THEME.body_font


def test_chart_spec_pie_uses_arc_theta() -> None:
    pie = Chart.model_validate(
        {"chart_type": "pie", "series": [{"name": "s", "points": [{"x": "A", "y": 1}]}]}
    )
    spec = compile_chart_spec(pie, REPORT_THEME)
    assert spec["mark"]["type"] == "arc"
    assert "theta" in spec["encoding"]


def test_render_chart_svg_and_png() -> None:
    svg = render_chart_svg(_BAR, REPORT_THEME)
    assert svg.lstrip().startswith("<svg")
    png = render_chart_png(_BAR, REPORT_THEME)
    assert png[:4] == b"\x89PNG"


def test_typst_chart_block_emits_image_and_asset() -> None:
    from kortny.documents.typst_writer import build_typst

    spec = _spec(blocks=[*_FULL_SPEC["blocks"], _BAR.model_dump()])
    source, assets = build_typst(spec)
    # The chart is the 9th block (index 8) -> chart_8.svg.
    assert any(name.endswith(".svg") for name in assets)
    asset_name = next(n for n in assets if n.endswith(".svg"))
    assert f'#image("{asset_name}"' in source
    assert assets[asset_name].lstrip().startswith(b"<svg")


@pytest.mark.skipif(_TYPST_MISSING, reason="typst binary not installed")
def test_render_spec_pdf_with_chart() -> None:
    spec = _spec(blocks=[*_FULL_SPEC["blocks"], _BAR.model_dump()])
    pdf = render_spec_pdf(spec)
    assert pdf.startswith(b"%PDF")


def test_chart_embeds_in_pptx_and_docx() -> None:
    from docx import Document as DocxDocument
    from pptx import Presentation

    from kortny.documents import render_docx, render_pptx

    spec = _spec(blocks=[*_FULL_SPEC["blocks"], _BAR.model_dump()])
    prs = Presentation(io.BytesIO(render_pptx(spec)))
    # At least one picture shape across the deck.
    assert any(
        shape.shape_type is not None and "PICTURE" in str(shape.shape_type)
        for slide in prs.slides
        for shape in slide.shapes
    )
    doc = DocxDocument(io.BytesIO(render_docx(spec)))
    assert "graphicData" in doc.element.xml  # an embedded drawing
