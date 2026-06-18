"""Document Studio — structured doc-spec IR + theme + Typst beauty engine.

See HIG-244. The agent emits a :class:`DocumentSpec` (JSON IR); a backend
writer renders it to a concrete format. Phase 1 ships the Typst PDF path.
"""

from __future__ import annotations

from kortny.documents.ir import (
    CTA,
    Block,
    Callout,
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
from kortny.documents.render import (
    DEFAULT_TYPST_BIN,
    DocumentRenderError,
    TypstNotAvailableError,
    render_spec_pdf,
    render_typst_pdf,
    typst_available,
)
from kortny.documents.themes import (
    PITCH_THEME,
    REPORT_THEME,
    Theme,
    ThemeColors,
    resolve_theme,
    theme_names,
)
from kortny.documents.typst_writer import render_document

__all__ = [
    "CTA",
    "Block",
    "Callout",
    "CoverHeader",
    "DEFAULT_TYPST_BIN",
    "DocKind",
    "DocumentRenderError",
    "DocumentSpec",
    "Heading",
    "PITCH_THEME",
    "Prose",
    "PullQuote",
    "REPORT_THEME",
    "SectionDivider",
    "StatCard",
    "StatCards",
    "Table",
    "Theme",
    "ThemeColors",
    "TypstNotAvailableError",
    "render_document",
    "render_spec_pdf",
    "render_typst_pdf",
    "resolve_theme",
    "theme_names",
    "typst_available",
]
