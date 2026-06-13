#!/usr/bin/env python3
"""Render an HTML report to an editorial-grade PDF with WeasyPrint — sandbox-only.

The agent authors the report as HTML using component classes defined in
references/authoring.md. This script applies a full themed design system
(type scale, palette, cover, stat cards, ruled tables, pull-quotes, CSS bars,
page numbers via @page margin boxes) and renders to A4 PDF.

Three named styles are available; the agent selects based on document purpose:
  editorial-mono   — premium investment research / IPO / flagship PDFs (default)
  minimal          — internal docs, status updates, SaaS reports
  editorial-feature — data stories, trend reports, "State of X"

Two input modes:
  --html PATH       a full or fragment HTML file authored by the agent
  --html-stdin      read HTML from stdin

Fragment mode: HTML body only (no <html> root). The script wraps it in the
full document template, including a dark full-bleed cover when --title is
supplied. Full-document mode: a complete <!DOCTYPE html> document; the script
injects the theme CSS into <head> so page setup and design tokens always apply.

WeasyPrint compatibility notes
-------------------------------
All CSS in this script is WeasyPrint-compatible:
  - @page margin boxes (@bottom-center, @top-right, @bottom-left) for page
    numbers and running headers — WeasyPrint supports these fully.
  - named pages (page: cover) + @page cover { margin: 0 } for full-bleed
    cover and section-divider pages — WeasyPrint supports named pages.
  - string-set / content(string()) for running section name in @top-right.
  - break-before/after/inside: avoid for page-break hygiene.
  - CSS Grid (display: grid) — WeasyPrint supports this since v53.
  - print-color-adjust: exact — WeasyPrint always prints backgrounds.
  - NO color-mix() — replaced with literal rgba() fallbacks (WeasyPrint
    support is version-dependent; literals are safe everywhere).
  - NO JavaScript — WeasyPrint does not execute JS.
  - Variable fonts: fallback stack is used; bundle static TTFs for full
    fidelity (see TODO in FONTS section below).

Network is never used; remote <img src=...> is disabled. Reference local
images by absolute path under the workbench dir.
"""

from __future__ import annotations

import argparse
import html as html_lib
import sys
from datetime import date
from pathlib import Path

from weasyprint import HTML


# ---------------------------------------------------------------------------
# TODO — font bundling for full Editorial Mono fidelity
# ---------------------------------------------------------------------------
# To match the research spec exactly, bundle these OFL/SIL fonts as static
# TTF or woff2 files in scripts/fonts/ and update the @font-face src lines:
#
#   Fraunces-Regular.ttf / Fraunces-Bold.ttf (or variable font)
#     → SIL OFL 1.1 — https://fonts.google.com/specimen/Fraunces
#   SourceSerif4-Regular.ttf / SourceSerif4-Bold.ttf
#     → SIL OFL 1.1 — https://fonts.google.com/specimen/Source+Serif+4
#   IBMPlexMono-Regular.ttf / IBMPlexMono-SemiBold.ttf
#     → SIL OFL 1.1 — https://fonts.google.com/specimen/IBM+Plex+Mono
#   Newsreader-Regular.ttf / Newsreader-Bold.ttf (for editorial-feature)
#     → SIL OFL 1.1 — https://fonts.google.com/specimen/Newsreader
#   SpaceGrotesk-Regular.ttf / SpaceGrotesk-Medium.ttf (for editorial-feature)
#     → SIL OFL 1.1 — https://fonts.google.com/specimen/Space+Grotesk
#   Inter-Regular.ttf / Inter-SemiBold.ttf (for minimal)
#     → SIL OFL 1.1 — https://fonts.google.com/specimen/Inter
#   JetBrainsMono-Regular.ttf (for minimal)
#     → Apache 2.0 — https://www.jetbrains.com/legalforms/mono-type-license
#
# Until bundled, the strong system-font fallback stacks below produce excellent
# output: Georgia, serif covers Fraunces/Newsreader; Menlo/Courier covers
# IBM Plex Mono/JetBrains Mono. The design system (black cover, mono kickers,
# big display numbers, stat cards, ruled tables) delivers most of the premium
# look regardless of font.
#
# Pass base_url=scripts_dir to WeasyPrint so url("fonts/...") resolves:
#   HTML(string=doc, base_url=str(scripts_dir)).write_pdf(str(out))
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# THEME DEFINITIONS
# ---------------------------------------------------------------------------

# Each theme is a dict of CSS custom-property values + font declarations.
# The COMPONENT_CSS references only the semantic tokens so themes are
# interchangeable by swapping :root values.

_FONT_FACE_EDITORIAL = """
/* TODO: replace url() paths with bundled TTF/woff2 files for full fidelity */
/* Fallback stacks below are strong for WeasyPrint with system fonts */
"""

_FONT_FACE_MINIMAL = """
/* TODO: bundle Inter + JetBrains Mono for full Minimal SaaS fidelity */
"""

_FONT_FACE_FEATURE = """
/* TODO: bundle Fraunces + Newsreader + Space Grotesk for Editorial Feature fidelity */
"""


THEMES: dict[str, dict[str, str]] = {
    "editorial-mono": {
        "font_face": _FONT_FACE_EDITORIAL,
        # Font stack: Fraunces-like premium serif display, Newsreader body, IBM Plex Mono
        "--font-display": '"Fraunces", "Source Serif 4", Georgia, "Times New Roman", serif',
        "--font-body": '"Newsreader", "Source Serif 4", Georgia, serif',
        "--font-mono": '"IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, "Courier New", monospace',
        # Palette — Editorial Black (ink, B/W, one oxblood accent)
        "--ink": "#0A0A0A",
        "--ink-soft": "#6B6B6B",
        "--paper": "#FAFAF8",
        "--surface": "#FFFFFF",
        "--line": "#E4E4E4",
        "--line-strong": "#111111",
        "--accent": "#C8102E",
        "--accent-2": "#2B4A6F",
        "--cover-bg": "#0A0A0A",
        "--cover-ink": "#FAFAF8",
        "--zebra": "#F4F2EC",
        "--good": "#1F7A4D",
        "--warn": "#B9842B",
        "--bad": "#B3261E",
        "--callout-bg": "#F9EEEE",
        "--callout-border": "#E0B3B3",
    },
    "minimal": {
        "font_face": _FONT_FACE_MINIMAL,
        # Font stack: Inter / system-ui (clean grotesque)
        "--font-display": '"Inter Tight", "Inter", -apple-system, "Segoe UI", Arial, sans-serif',
        "--font-body": '"Inter", -apple-system, "Segoe UI", Arial, sans-serif',
        "--font-mono": '"JetBrains Mono", "IBM Plex Mono", ui-monospace, Menlo, monospace',
        # Palette — Minimal Light (clean white, indigo accent)
        "--ink": "#0C0E12",
        "--ink-soft": "#646B76",
        "--paper": "#FFFFFF",
        "--surface": "#F6F8FA",
        "--line": "#E6E9ED",
        "--line-strong": "#1A1D21",
        "--accent": "#635BFF",
        "--accent-2": "#3A7BD5",
        "--cover-bg": "#11161C",
        "--cover-ink": "#FFFFFF",
        "--zebra": "#F5F7F9",
        "--good": "#1A7F4B",
        "--warn": "#B9770A",
        "--bad": "#C2362F",
        "--callout-bg": "#EEEEFF",
        "--callout-border": "#C8C5FF",
    },
    "editorial-feature": {
        "font_face": _FONT_FACE_FEATURE,
        # Font stack: Fraunces display, Newsreader/Spectral body, Space Grotesk mono
        "--font-display": '"Fraunces", Georgia, "Times New Roman", serif',
        "--font-body": '"Newsreader", "Spectral", Georgia, serif',
        "--font-mono": '"Space Grotesk", "IBM Plex Mono", ui-monospace, Menlo, monospace',
        # Palette — Warm paper, editorial orange accent
        "--ink": "#1B1A17",
        "--ink-soft": "#6E675C",
        "--paper": "#FBF7F0",
        "--surface": "#FFFFFF",
        "--line": "#E3DCCF",
        "--line-strong": "#1B1A17",
        "--accent": "#E8552D",
        "--accent-2": "#2D6CDF",
        "--cover-bg": "#1B1A17",
        "--cover-ink": "#FBF7F0",
        "--zebra": "#F5F0E8",
        "--good": "#1E9E7A",
        "--warn": "#F2B705",
        "--bad": "#B3261E",
        "--callout-bg": "#FEF0EB",
        "--callout-border": "#F5C4B5",
    },
}


# ---------------------------------------------------------------------------
# BASE PAGE + TYPOGRAPHY + COMPONENT CSS (WeasyPrint-compatible)
# ---------------------------------------------------------------------------

BASE_CSS = """
%(font_face)s

/* ---- TOKENS ---- */
:root {
  --font-display: %(--font-display)s;
  --font-body:    %(--font-body)s;
  --font-mono:    %(--font-mono)s;
  --ink:          %(--ink)s;
  --ink-soft:     %(--ink-soft)s;
  --paper:        %(--paper)s;
  --surface:      %(--surface)s;
  --line:         %(--line)s;
  --line-strong:  %(--line-strong)s;
  --accent:       %(--accent-override)s;
  --accent-2:     %(--accent-2)s;
  --cover-bg:     %(--cover-bg)s;
  --cover-ink:    %(--cover-ink)s;
  --zebra:        %(--zebra)s;
  --good:         %(--good)s;
  --warn:         %(--warn)s;
  --bad:          %(--bad)s;
  --callout-bg:   %(--callout-bg)s;
  --callout-border: %(--callout-border)s;
}

/* ---- PAGE / PRINT ---- */
/* Real @page margins so EVERY page — including continuation pages — gets the
   content inset (the previous "margin:0 + .page-body padding" stranded
   continuation pages flush to the top edge). The margin boxes live in the
   page margin area and carry the running section name + page number. Cover and
   section dividers override to a full-bleed margin-0 named page below. */
@page {
  size: A4;
  margin: 24mm 22mm 22mm 22mm;
  @bottom-right {
    content: counter(page);
    font-family: var(--font-mono);
    font-size: 7pt;
    letter-spacing: 0.08em;
    color: %(--ink-soft)s;
  }
  @bottom-left {
    content: "KORTNY RESEARCH";
    font-family: var(--font-mono);
    font-size: 7pt;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: %(--ink-soft)s;
  }
  @top-right {
    content: string(section-title);
    font-family: var(--font-mono);
    font-size: 7pt;
    letter-spacing: 0.08em;
    color: %(--ink-soft)s;
  }
}
/* Named page for cover + divider — full bleed, no margin-box content */
@page cover {
  margin: 0;
  @bottom-right { content: none; }
  @bottom-left  { content: none; }
  @top-right    { content: none; }
}

html {
  print-color-adjust: exact;
  -webkit-print-color-adjust: exact;
  text-rendering: optimizeLegibility;
  font-kerning: normal;
  font-feature-settings: "kern" 1, "liga" 1;
}
body {
  margin: 0;
  background: var(--paper);
  color: var(--ink);
  font-family: var(--font-body);
  font-size: 10.5pt;
  line-height: 1.55;
}
a { color: var(--accent); text-decoration: none; }
p, li { orphans: 3; widows: 3; }

/* ---- CONTENT INSET ---- */
/* Content inset now comes from the @page margins (so continuation pages are
   inset too). .page-body is a plain flow wrapper — no padding. The cover lives
   outside it; section dividers full-bleed via their margin-0 named page. */
.page-body {
  padding: 0;
  box-sizing: border-box;
}

/* ---- TYPOGRAPHY ---- */
.kicker, .eyebrow {
  font-family: var(--font-mono);
  font-size: 8pt;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.16em;
  color: var(--accent);
  margin: 0 0 8px;
  break-after: avoid;
}
h1, .h1 {
  font-family: var(--font-display);
  font-weight: 700;
  font-size: 34pt;
  line-height: 1.05;
  letter-spacing: -0.02em;
  color: var(--ink);
  margin: 0 0 14px;
  break-after: avoid;
}
h2, .h2 {
  font-family: var(--font-display);
  font-weight: 600;
  font-size: 22pt;
  line-height: 1.1;
  letter-spacing: -0.01em;
  color: var(--ink);
  margin: 36px 0 12px;
  padding-bottom: 10px;
  border-bottom: 1.5px solid var(--line);
  break-after: avoid;
  /* WeasyPrint: registers the text for @top-right running header */
  string-set: section-title content();
}
h3, .h3 {
  font-family: var(--font-display);
  font-weight: 600;
  font-size: 15pt;
  line-height: 1.15;
  color: var(--ink);
  margin: 26px 0 8px;
  break-after: avoid;
}
h4, .h4 {
  font-family: var(--font-mono);
  font-weight: 600;
  font-size: 9pt;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  color: var(--ink-soft);
  margin: 18px 0 6px;
  break-after: avoid;
}
p { margin: 0 0 12px; max-width: 72ch; }
.lead {
  font-size: 13pt;
  line-height: 1.5;
  color: var(--ink);
  margin-bottom: 20px;
  max-width: 60ch;
}
ul, ol { margin: 0 0 12px 20px; }
li { margin-bottom: 4px; }
.source {
  font-family: var(--font-mono);
  font-size: 7.5pt;
  color: var(--ink-soft);
  margin: 4px 0 18px;
}
.muted { color: var(--ink-soft); }
.num {
  text-align: right;
  font-variant-numeric: tabular-nums lining-nums;
  font-feature-settings: "tnum" 1, "lnum" 1;
}

/* ---- COVER (full-bleed dark page) ---- */
/* Sits outside .page-body; width/min-height ensure A4 trim bleed.
   page: cover attaches to @page cover { margin: 0 } in WeasyPrint. */
.cover {
  page: cover;
  background: var(--cover-bg);
  color: var(--cover-ink);
  width: 210mm;
  min-height: 297mm;
  box-sizing: border-box;
  padding: 28mm 24mm;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  break-after: page;
  page-break-after: always;
}
.cover__top, .cover__bottom {
  display: flex;
  justify-content: space-between;
  font-family: var(--font-mono);
  font-size: 8pt;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  /* rgba literal instead of color-mix() for WeasyPrint compat */
  color: rgba(250, 250, 248, 0.70);
}
.cover__kicker {
  font-family: var(--font-mono);
  font-size: 9pt;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--accent);
  margin: 0 0 18px;
}
.cover__title {
  font-family: var(--font-display);
  font-weight: 700;
  font-size: 64pt;
  line-height: 0.98;
  letter-spacing: -0.025em;
  margin: 0 0 22px;
  max-width: 16ch;
  /* Override the global h1 color (dark body ink) — this sits on the dark
     cover, so it must use the light cover ink or it renders invisible. */
  color: var(--cover-ink);
}
.cover__deck {
  font-family: var(--font-body);
  font-size: 14pt;
  line-height: 1.45;
  max-width: 36ch;
  /* rgba literal for WeasyPrint compat */
  color: rgba(250, 250, 248, 0.88);
}
.cover__main { flex: 1; display: flex; flex-direction: column; justify-content: flex-end; padding: 32mm 0 22mm; }

/* ---- SECTION DIVIDER (full-bleed dark page) ---- */
/* Same named-page treatment as .cover for full bleed in WeasyPrint. */
.section-divider {
  page: cover;
  background: var(--cover-bg);
  color: var(--cover-ink);
  /* Full bleed: on its margin-0 named page inside the now-unpadded page-body,
     a full A4 width fills the trim on all four edges. */
  width: 210mm;
  min-height: 297mm;
  box-sizing: border-box;
  padding: 32mm 24mm;
  display: flex;
  flex-direction: column;
  justify-content: center;
  break-before: page;
  break-after: page;
  page-break-before: always;
  page-break-after: always;
}
.section-divider__num {
  font-family: var(--font-mono);
  font-size: 14pt;
  letter-spacing: 0.2em;
  color: var(--accent);
  margin-bottom: 10px;
}
.section-divider__kicker {
  font-family: var(--font-mono);
  font-size: 9pt;
  text-transform: uppercase;
  letter-spacing: 0.18em;
  color: rgba(250, 250, 248, 0.60);
  margin: 0 0 14px;
}
.section-divider__title {
  font-family: var(--font-display);
  font-weight: 600;
  font-size: 40pt;
  line-height: 1.02;
  letter-spacing: -0.02em;
  margin: 0 0 16px;
  max-width: 18ch;
  /* Override global heading color — dark divider page needs light ink. */
  color: var(--cover-ink);
}
.section-divider__deck {
  font-family: var(--font-body);
  font-size: 13pt;
  line-height: 1.5;
  max-width: 38ch;
  color: rgba(250, 250, 248, 0.85);
}

/* ---- STAT CARD GRID ---- */
/* CSS Grid; WeasyPrint has supported grid since v53. break-inside:avoid
   keeps each card on one page. */
.stat-grid {
  display: grid;
  gap: 14px;
  margin: 24px 0;
}
.stat-grid--4 { grid-template-columns: repeat(4, 1fr); }
.stat-grid--3 { grid-template-columns: repeat(3, 1fr); }
.stat-grid--2 { grid-template-columns: repeat(2, 1fr); }
.stat-card {
  background: var(--surface);
  border: 1px solid var(--line);
  border-top: 3px solid var(--accent);
  border-radius: 6px;
  padding: 18px 18px 14px;
  display: flex;
  flex-direction: column;
  gap: 6px;
  break-inside: avoid;
  page-break-inside: avoid;
}
.stat-card__label {
  font-family: var(--font-mono);
  font-size: 7.5pt;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  color: var(--ink-soft);
}
.stat-card__value {
  font-family: var(--font-display);
  font-weight: 700;
  font-size: 40pt;
  line-height: 1;
  letter-spacing: -0.02em;
  color: var(--ink);
  font-variant-numeric: tabular-nums lining-nums;
}
.stat-card__unit {
  font-size: 18pt;
  color: var(--ink-soft);
  margin-left: 2px;
}
.stat-card__sub {
  font-family: var(--font-mono);
  font-size: 8pt;
  color: var(--ink-soft);
}
.stat-card__sub--up   { color: var(--good); }
.stat-card__sub--down { color: var(--bad); }

/* ---- DATA TABLE ---- */
/* Hairline rules (not box border), bold first column, right-aligned numerics,
   mono CAPS headers. thead repeat on page break is standard behavior. */
.data-table {
  width: 100%%;
  border-collapse: collapse;
  margin: 14px 0;
  font-family: var(--font-body);
  font-size: 9.5pt;
  /* Keep short report tables whole so the totals row never strands on a
     separate page from its body. A table taller than a page still splits
     (rows stay intact via the tr rule); the .data-table--split modifier
     opts a long table back into mid-table breaking. */
  break-inside: avoid;
  page-break-inside: avoid;
}
.data-table--split { break-inside: auto; page-break-inside: auto; }
.data-table caption {
  caption-side: top;
  text-align: left;
  font-family: var(--font-mono);
  font-size: 8pt;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--ink-soft);
  margin-bottom: 8px;
  padding: 0;
}
.data-table thead th {
  font-family: var(--font-mono);
  font-size: 7.5pt;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--ink-soft);
  text-align: left;
  padding: 8px 10px;
  border-bottom: 2px solid var(--line-strong);
  background: transparent;
}
.data-table thead { display: table-header-group; }
.data-table tbody th,
.data-table tbody td,
.data-table tfoot th,
.data-table tfoot td {
  padding: 8px 10px;
  border-bottom: 1px solid var(--line);
}
.data-table tbody tr:nth-child(even) { background: var(--zebra); }
.data-table tfoot th,
.data-table tfoot td {
  font-weight: 600;
  border-top: 2px solid var(--line-strong);
  border-bottom: none;
}
.data-table th[scope="row"] { font-weight: 600; color: var(--ink); }
.data-table .num {
  text-align: right;
  font-variant-numeric: tabular-nums lining-nums;
  font-feature-settings: "tnum" 1, "lnum" 1;
}
.data-table tr { break-inside: avoid; page-break-inside: avoid; }

/* ---- PULL-QUOTE ---- */
.pull-quote {
  margin: 28px 0;
  padding: 0 0 0 22px;
  border-left: 3px solid var(--accent);
  break-inside: avoid;
  page-break-inside: avoid;
}
.pull-quote p {
  font-family: var(--font-display);
  font-weight: 500;
  font-style: italic;
  font-size: 17pt;
  line-height: 1.35;
  letter-spacing: -0.01em;
  color: var(--ink);
  margin: 0 0 8px;
  max-width: 32ch;
}
.pull-quote cite {
  font-family: var(--font-mono);
  font-style: normal;
  font-size: 8pt;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  color: var(--ink-soft);
}

/* ---- KEY TAKEAWAY / CALLOUT BOX ---- */
/* rgba literals instead of color-mix() for WeasyPrint compat. */
.key-takeaway {
  background: var(--callout-bg);
  border: 1px solid var(--callout-border);
  border-left: 3px solid var(--accent);
  border-radius: 6px;
  padding: 16px 18px;
  margin: 22px 0;
  break-inside: avoid;
  page-break-inside: avoid;
}
.key-takeaway__tag {
  display: block;
  font-family: var(--font-mono);
  font-size: 7.5pt;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  color: var(--accent);
  margin-bottom: 6px;
}
.key-takeaway p {
  font-family: var(--font-body);
  font-size: 10.5pt;
  line-height: 1.5;
  color: var(--ink);
  margin: 0;
  max-width: none;
}

/* ---- CSS CONCENTRATION BAR ---- */
/* A single horizontal stacked bar — use instead of a chart for simple splits.
   No font embedding risk; pure CSS; prints perfectly. */
.bar {
  display: flex;
  height: 34px;
  border-radius: 4px;
  overflow: hidden;
  margin: 14px 0 8px;
}
.bar-row {
  display: flex;
  align-items: center;
  justify-content: center;
  color: #ffffff;
  font-family: var(--font-mono);
  font-size: 8pt;
  font-weight: 600;
  letter-spacing: 0.04em;
}
.bar-legend {
  display: flex;
  gap: 18px;
  font-family: var(--font-mono);
  font-size: 8pt;
  color: var(--ink-soft);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-bottom: 14px;
  flex-wrap: wrap;
}
.bar-legend i {
  display: inline-block;
  width: 9px;
  height: 9px;
  border-radius: 2px;
  margin-right: 6px;
  vertical-align: middle;
}

/* ---- FIGURE / CHART WRAPPER ---- */
.figure {
  margin: 20px 0 24px;
  break-inside: avoid;
  page-break-inside: avoid;
}
.figure__cap {
  font-family: var(--font-mono);
  font-size: 8pt;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--ink-soft);
  margin: 0 0 10px;
}
.figure img, .figure svg { max-width: 100%%; height: auto; }

/* ---- SOURCES / FOOTNOTES BLOCK ---- */
.sources {
  margin-top: 28px;
  padding-top: 12px;
  border-top: 1px solid var(--line);
  break-inside: avoid;
  page-break-inside: avoid;
}
.sources__head {
  font-family: var(--font-mono);
  font-size: 8pt;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  color: var(--ink-soft);
  margin: 0 0 6px;
}
.sources ol, .sources ul {
  margin: 0;
  padding-left: 18px;
  font-family: var(--font-body);
  font-size: 8.5pt;
  line-height: 1.5;
  color: var(--ink-soft);
}
.sources li { margin-bottom: 3px; }
"""


# ---------------------------------------------------------------------------
# DOCUMENT TEMPLATES
# ---------------------------------------------------------------------------

COVER_TEMPLATE = """\
<section class="cover">
  <div class="cover__top">
    <span>KORTNY RESEARCH</span>
    <span>%(meta_right)s</span>
  </div>
  <div class="cover__main">
    <p class="cover__kicker">%(kicker)s</p>
    <h1 class="cover__title">%(title)s</h1>
    %(deck_html)s
  </div>
  <div class="cover__bottom">
    <span>%(bottom_left)s</span>
    <span></span>
  </div>
</section>"""

DOC_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>%(title_text)s</title>
<style>%(css)s</style>
</head>
<body>
%(cover)s
<main class="page-body">
%(body)s
</main>
</body>
</html>"""


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _build_css(theme_name: str, accent_override: str | None) -> str:
    """Interpolate the BASE_CSS template with theme tokens."""
    theme = THEMES[theme_name]
    vals: dict[str, str] = {k: v for k, v in theme.items() if k != "font_face"}
    vals["font_face"] = theme["font_face"]
    # Allow accent override (--accent flag)
    if accent_override:
        vals["--accent-override"] = accent_override
    else:
        vals["--accent-override"] = theme["--accent"]
    # Flatten remaining tokens for @page literal references (can't use CSS vars
    # inside @page margin box content in all WeasyPrint versions)
    return BASE_CSS % vals


def _build_cover(
    title: str,
    subtitle: str | None,
    kicker: str,
    doc_date: str,
) -> str:
    deck_html = (
        f'<p class="cover__deck">{html_lib.escape(subtitle)}</p>'
        if subtitle
        else ""
    )
    return COVER_TEMPLATE % {
        "meta_right": html_lib.escape(doc_date.upper()),
        "kicker": html_lib.escape(kicker),
        "title": html_lib.escape(title),
        "deck_html": deck_html,
        "bottom_left": "CONFIDENTIAL",
    }


def _looks_like_full_document(markup: str) -> bool:
    head = markup.lstrip()[:300].lower()
    return head.startswith("<!doctype") or "<html" in head


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------

def render(
    markup: str,
    out_path: Path,
    *,
    title: str | None = None,
    subtitle: str | None = None,
    kicker: str = "Research Report",
    accent: str | None = None,
    doc_date: str | None = None,
    style: str = "editorial-mono",
) -> Path:
    """Render *markup* (HTML fragment or full document) to a themed PDF.

    Args:
        markup:    HTML body fragment or full document string.
        out_path:  Destination .pdf path.
        title:     Cover title (fragment mode). In full-doc mode, used for
                   <title> only if not overridden in the document.
        subtitle:  Cover subtitle / deck text.
        kicker:    Mono CAPS eyebrow above the cover title.
        accent:    Override the theme accent color (hex string).
        doc_date:  Cover date string (default: today).
        style:     Theme name: editorial-mono | minimal | editorial-feature.
    """
    if style not in THEMES:
        raise ValueError(
            f"Unknown style {style!r}. Choose from: {', '.join(THEMES)}"
        )
    shown_date = doc_date or date.today().strftime("%d %b %Y").upper()
    css = _build_css(style, accent)

    if _looks_like_full_document(markup):
        # Full-document mode: inject theme CSS into existing <head>.
        document = markup
        if "</head>" in document:
            inject = f"<style>\n{css}\n</style>"
            document = document.replace("</head>", f"{inject}\n</head>", 1)
        else:
            document = f"<style>\n{css}\n</style>\n{document}"
    else:
        # Fragment mode: wrap with cover + page-body.
        cover_html = (
            _build_cover(title or "Report", subtitle, kicker, shown_date)
            if title
            else ""
        )
        document = DOC_TEMPLATE % {
            "title_text": html_lib.escape(title or "Report"),
            "css": css,
            "cover": cover_html,
            "body": markup,
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # base_url set to scripts/ dir so url("fonts/...") resolves when bundles exist.
    scripts_dir = Path(__file__).parent
    HTML(string=document, base_url=str(scripts_dir)).write_pdf(str(out_path))
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render an HTML report to an editorial-grade PDF (WeasyPrint)."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--html", help="Path to an HTML file (full doc or fragment).")
    src.add_argument(
        "--html-stdin", action="store_true", help="Read HTML from stdin."
    )
    parser.add_argument("--out", required=True, help="Output .pdf path.")
    parser.add_argument("--title", help="Cover title (fragment mode).")
    parser.add_argument("--subtitle", help="Cover subtitle / deck text.")
    parser.add_argument(
        "--kicker",
        default="Research Report",
        help="Mono CAPS eyebrow above cover title (default: 'Research Report').",
    )
    parser.add_argument(
        "--date", dest="doc_date", help="Cover date string (default: today)."
    )
    parser.add_argument(
        "--accent",
        default=None,
        help="Override theme accent color (hex, e.g. '#C8102E').",
    )
    parser.add_argument(
        "--style",
        default="editorial-mono",
        choices=list(THEMES),
        help=(
            "Named theme: editorial-mono (default, premium investment research), "
            "minimal (internal/SaaS docs), editorial-feature (data stories/trends)."
        ),
    )
    args = parser.parse_args(argv)

    markup = sys.stdin.read() if args.html_stdin else Path(args.html).read_text()
    out = render(
        markup,
        Path(args.out),
        title=args.title,
        subtitle=args.subtitle,
        kicker=args.kicker,
        accent=args.accent,
        doc_date=args.doc_date,
        style=args.style,
    )
    print(f"wrote pdf: {out} ({out.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
