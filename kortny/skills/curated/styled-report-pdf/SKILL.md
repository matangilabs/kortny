---
name: styled-report-pdf
description: Use when asked to produce a polished PDF report, formatted document, investment research note, one-pager, brief, or leave-behind to upload to Slack — when the deliverable is a styled, print-ready PDF with a cover, heading hierarchy, stat cards, tables, and page numbers, not plain text in a message.
metadata:
  version: 2.1.0
  display_name: Styled Report PDF
  tags: pdf, report, document, one-pager, brief, leave-behind, weasyprint, html, print, investment-research, editorial
---

## Goal

Produce a print-ready PDF that rivals top-tier investment-research reports —
full-bleed dark cover, monospace kicker labels, large display numbers in stat
cards, hairline-ruled data tables with right-aligned numerics, section-divider
pages, pull-quotes, CSS concentration bars, and automatic page numbers — then
upload the PDF to the Slack thread.

## Style selection (pick one, cite the reason)

Match on document purpose. The style controls the entire visual system
(palette, fonts, spacing, cover treatment). Use the **first match** below:

| Request signal | Style flag |
|---|---|
| IPO / investment research / equity note / flagship PDF / "research report" / finance | `--style editorial-mono` (default) |
| Internal update / status report / changelog / eng or product doc / "keep it simple" / SaaS | `--style minimal` |
| Data story / "State of X" / trend report / year-in-review / magazine / PR | `--style editorial-feature` |

When unsure, default to `editorial-mono`. It never looks wrong for a document
that needs to impress.

## Workflow

### 1. Decide the style

State which style you are using and why (one sentence). This anchors the
aesthetic for the whole document.

### 2. Author the HTML body

Write a **body fragment** (no `<html>` root) using the component class
vocabulary below. The script wraps it in the full document template,
applies the design system, and prepends the dark cover when `--title` is
supplied.

**Component class vocabulary — use these, do not invent new ones:**

| Component | Class(es) | When to use |
|---|---|---|
| Section kicker / eyebrow | `.kicker` or `.eyebrow` | Mono CAPS label above every H2; names the section type (e.g. "Executive Summary") |
| Section heading | `<h2>` | Major section title; drives the running header in the footer |
| Sub-heading | `<h3>` | Sub-section within a section |
| Lead paragraph | `<p class="lead">` | Standfirst / opening sentence under a heading; larger, airy |
| 4-up stat grid | `.stat-grid.stat-grid--4` > `.stat-card` | Four key metrics across the top of a section |
| 3-up stat grid | `.stat-grid.stat-grid--3` > `.stat-card` | Three metrics |
| 2-up stat grid | `.stat-grid.stat-grid--2` > `.stat-card` | Two metrics (or hero pair) |
| Stat card internals | `.stat-card__label` / `.stat-card__value` / `.stat-card__unit` / `.stat-card__sub` | Label above, big number, optional unit suffix, optional delta below |
| Delta up / down | `.stat-card__sub--up` / `.stat-card__sub--down` | Green / red delta on a stat card |
| Data table | `<table class="data-table">` with `<caption>`, `<thead>`, `<tbody>`, `<tfoot>` | Financial or comparison tables; always add `.num` to numeric cells |
| Numeric cell | class `.num` on `<td>` or `<th>` | Right-aligns and uses tabular figures — mandatory for every number column |
| Pull-quote | `.pull-quote` > `<p>` + `<cite>` | A memorable sentence that deserves isolation; use sparingly (1–2 per report) |
| Key takeaway box | `.key-takeaway` > `.key-takeaway__tag` + `<p>` | 1–2 sentence insight; accentuated left border + tinted background |
| CSS concentration bar | `.bar` > `.bar-row` (multiple) + `.bar-legend` > `<span>` | Part-to-whole split with ≤ 4 segments; replaces a pie chart |
| Figure / chart wrapper | `.figure` > `.figure__cap` + `<img>` or inline `<svg>` | Wraps any chart or image with a mono-CAPS caption |
| Source line | `<p class="source">` | Attribution under a table or figure |
| Sources block | `.sources` > `.sources__head` + `<ol>` | End-of-section or end-of-document references |
| Section divider page | `.section-divider` with `.section-divider__num` / `__kicker` / `__title` / `__deck` | Full-bleed dark page before a major section (optional but high-impact) |
| Muted text | class `.muted` | Secondary gray text anywhere |

**Authoring discipline:**
- Write prose paragraphs, not bullet walls. Reserve `<ul>` for genuine lists
  of ≥ 3 comparable items.
- Every H2 must be immediately preceded by a `.kicker` paragraph naming the
  section type.
- Stat grids go near the top of each section, before the prose.
- Numeric table cells always get class `.num` — this is the #1 quality signal
  in financial tables.
- CSS bars (`.bar`) replace pie charts for simple proportions; use a figure +
  chart image only for time-series or multi-variable data.
- Section dividers (`div.section-divider`) are optional but strongly
  recommended for reports with 3+ major sections — they give the premium
  "Viktor-style" investment-research feel.

### 3. Render

```bash
python scripts/render_pdf.py \
  --html body.html \
  --out report.pdf \
  --style editorial-mono \
  --title "SpaceX IPO Readiness Analysis" \
  --subtitle "A field analysis of capital formation and market timing." \
  --kicker "Investment Research — Confidential" \
  --date "13 Jun 2026" \
  --brand "Northwind Capital"
```

**Brand the report with the user's organization, never the assistant's own
name.** Pass `--brand` with the firm/team/company name — pull it from workspace
facts or the channel/workspace context. It appears on the cover top-left and the
page footer. If you genuinely don't know the org name, **omit `--brand`** (the
label is dropped cleanly) rather than guessing or branding it with the
assistant's name. In full-document mode, put the same name in your cover markup —
do not hardcode the assistant's name as the publisher.

Pass `--accent "#hex"` to override the theme accent for a specific brand color
(e.g. a client's primary color). Otherwise the theme default is used.

For a full `<!DOCTYPE html>` document (full-document mode), omit `--title` /
`--subtitle` / `--kicker` — they are already in the document's cover section.
The script injects the theme CSS into `<head>` automatically.

### 4. Read the PAGINATION REPORT (advisory — do not loop)

After each render the script prints a `PAGINATION REPORT` with per-page fill
levels. Treat it as **advisory polish, not a gate**. A slightly under-filled
last page is normal and completely fine — never re-render just to raise a fill
percentage. Render once and move on to export.

Re-render **at most once**, and only for a genuinely broken layout: a body page
nearly empty in the *middle* of the document, or a component split badly across
a page. If you do re-render, edit the body first, then render to a **new output
filename** (e.g. `report_v2.pdf`) — re-running the exact same command is
detected as a stuck loop and stopped by a safety circuit breaker before your PDF
is ever delivered. Cover/divider pages always read as full; that is expected.
An acceptable render that ships beats a perfect one that loops. (Pass
`--no-analyze` to skip the report for quick drafts.)

### 5. Export and post (required — the PDF stays in the sandbox until you do)

The render writes the PDF inside the sandbox workspace; it does not leave the
sandbox on its own. After the first acceptable render you MUST:

1. Export it with `sandbox_export_artifact` (pass the rendered path, e.g.
   `/workspace/report.pdf`) so it becomes a real downloadable artifact.
2. Post it to the Slack thread with a one-line summary of the key finding.

Do this exactly once — do not re-render after exporting. Optionally offer a
slide version (`deck-builder`) or the data as a workbook
(`spreadsheet-builder`).

## Layout discipline (non-negotiable)

- **One design system per document.** Do not mix component classes from
  different styles or invent new color/font overrides inline. Let the theme do
  its job.
- **Page-aware layout.** Table rows never break across a page (`break-inside:
  avoid` is baked in). Headings never strand at the bottom of a page. Every
  page shows a page number (bottom-right) and the report brand (bottom-left).
- **Cover block** for any report longer than one page: always pass `--title`.
- **Tables explained in prose.** A table supports the text; it does not
  replace it.
- **Local images only.** Reference images by absolute path under the workbench
  dir. There is no network at render time.

## Script reference

`scripts/render_pdf.py`

| Flag | Required | Description |
|---|---|---|
| `--html PATH` | one of these | Path to HTML file (full doc or fragment) |
| `--html-stdin` | one of these | Read HTML from stdin |
| `--out PATH` | yes | Output .pdf path |
| `--style NAME` | no | `editorial-mono` (default) / `minimal` / `editorial-feature` |
| `--title TEXT` | no | Cover title (fragment mode) |
| `--subtitle TEXT` | no | Cover deck/subtitle |
| `--kicker TEXT` | no | Mono CAPS eyebrow above cover title |
| `--date TEXT` | no | Cover date string (default: today) |
| `--brand TEXT` | no | Org/firm name for cover + footer (use the user's org, not the assistant's name; omit if unknown) |
| `--accent HEX` | no | Override theme accent color |

Deps: `weasyprint` (pango/cairo baked into sandbox image). No network. Display/
body/mono fonts are bundled in `scripts/fonts/` and embed automatically — no
font setup needed.
See `references/authoring.md` for the full component reference with worked examples.
