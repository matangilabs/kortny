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
    DocKind,
    DocumentSpec,
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
