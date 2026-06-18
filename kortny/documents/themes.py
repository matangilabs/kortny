"""Theme objects for Document Studio (HIG-244).

A theme is the design system: a font trio, a colour token set, and a few layout
switches. It is authored once (here), not by the model per request — the agent
only ever emits the IR. Themes are resolved by name; ``doc_kind`` picks a
default when the spec does not name one.

Font families must be resolvable by the Typst render environment (system fonts
or a bundled ``--font-path`` dir). The families below are chosen to be
commonly available / bundlable OSS faces; substitution degrades gracefully but
should be avoided in production by shipping the fonts with the worker image.
"""

from __future__ import annotations

from dataclasses import dataclass

from kortny.documents.ir import DocKind


@dataclass(frozen=True, slots=True)
class ThemeColors:
    ink: str
    paper: str
    accent: str
    muted: str
    card: str
    rule: str
    # Foreground/muted tones used on dark divider pages.
    on_dark: str = "#FFFFFF"
    on_dark_muted: str = "#9A9A9C"
    divider_bg: str = "#0E0E0F"


@dataclass(frozen=True, slots=True)
class Theme:
    """A named design system applied by the writer."""

    name: str
    display_font: str
    body_font: str
    mono_font: str
    colors: ThemeColors
    # Big display headings: 700 reads as a confident editorial weight.
    display_weight: int = 700
    # Justify body prose (report-like) vs ragged-right (pitch-like).
    justify_body: bool = True
    # Render section dividers as full-bleed dark pages (the alternating rhythm).
    dark_dividers: bool = True
    page_margin_mm: int = 20
    # Accent tint used behind callouts / highlighted panels.
    panel_tint: str = "#F4EFE5"


# A confident editorial serif on a light page with orange accent — refines the
# look Kortny already ships, tuned for information-dense reports.
REPORT_THEME = Theme(
    name="report",
    display_font="Source Serif 4",
    body_font="Source Serif 4",
    mono_font="IBM Plex Mono",
    colors=ThemeColors(
        ink="#1A1714",
        paper="#FFFFFF",
        accent="#D2502A",
        muted="#6B6660",
        card="#FBF7F1",
        rule="#E4DDD3",
        on_dark="#F7F3ED",
        on_dark_muted="#A39C92",
        divider_bg="#1A1714",
    ),
    justify_body=True,
    dark_dividers=True,
    panel_tint="#FBF1EA",
)

# Big geometric sans, lots of negative space, single gold accent, full-bleed
# dark covers — the Viktor-class pitch look. Low density, high beauty.
PITCH_THEME = Theme(
    name="pitch",
    display_font="Archivo",
    body_font="Archivo",
    mono_font="Space Mono",
    colors=ThemeColors(
        ink="#0E0E0F",
        paper="#FFFFFF",
        accent="#B49A6B",
        muted="#6B6B6E",
        card="#F5F3EF",
        rule="#E2DED6",
        on_dark="#FFFFFF",
        on_dark_muted="#9A9A9C",
        divider_bg="#0E0E0F",
    ),
    display_weight=700,
    justify_body=False,
    dark_dividers=True,
    panel_tint="#F4EFE5",
)


_THEMES: dict[str, Theme] = {t.name: t for t in (REPORT_THEME, PITCH_THEME)}

_DEFAULT_THEME_BY_KIND: dict[DocKind, str] = {
    DocKind.pitch: "pitch",
    DocKind.report: "report",
    DocKind.brief: "report",
    DocKind.memo: "report",
}


def theme_names() -> tuple[str, ...]:
    """Names of every registered theme."""

    return tuple(_THEMES)


def resolve_theme(*, doc_kind: DocKind, name: str | None) -> Theme:
    """Resolve the theme for a spec: explicit ``name`` wins, else by ``doc_kind``.

    An unknown name falls back to the ``doc_kind`` default rather than raising —
    a bad theme name should never fail document generation.
    """

    if name is not None and name in _THEMES:
        return _THEMES[name]
    return _THEMES[_DEFAULT_THEME_BY_KIND[doc_kind]]
