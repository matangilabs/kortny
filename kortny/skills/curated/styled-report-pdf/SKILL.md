---
name: styled-report-pdf
description: Use when asked to produce a polished PDF report, formatted document, investment research note, one-pager, brief, or leave-behind to upload to Slack — when the deliverable is a styled, print-ready PDF with a cover, heading hierarchy, stat cards, tables, charts, and page numbers, not plain text in a message.
metadata:
  version: 3.0.0
  display_name: Styled Report PDF
  tags: pdf, report, document, one-pager, brief, leave-behind, editorial, investment-research, document-studio
---

## Goal

Produce a print-ready, editorial-grade PDF — full-bleed dark cover, monospace
kicker labels, large display numbers in stat cards, hairline-ruled tables,
section-divider pages, pull-quotes, **charts where the data deserves a picture**,
and automatic page numbers — then post it to the Slack thread.

You do **not** write HTML or run a renderer. You author a structured document
spec and call the **`document_studio` tool**, which renders the themed PDF and
records the artifact for you.

## How to produce the document

Call `document_studio` with `format: "pdf"`, a `title`, and an ordered `blocks`
array. Pick `doc_kind` by purpose:

| Request signal | `doc_kind` |
|---|---|
| IPO / equity / investment research / flagship / "research report" / finance | `report` |
| Internal update / status / changelog / eng or product doc / "keep it simple" | `report` (lighter content) |
| Pitch / fundraise / persuasive one-pager / "make it beautiful" | `pitch` |

The block vocabulary (emit these as the `blocks` array — the tool schema has the
exact fields):

- `cover_header` — eyebrow (mono kicker), title, subtitle, meta (brand/date/confidentiality). One, first.
- `section_divider` — full-bleed dark break before a major section (use for reports with 3+ sections; high-impact).
- `heading` → `prose` — a section: prose paragraphs, not bullet walls.
- `stat_cards` — 2–4 headline metrics near the top of a section.
- `table` — comparison/financial data; supports the prose, never replaces it.
- `chart` — **see judgment below.**
- `callout` — a 1–2 sentence key takeaway.
- `pull_quote` — one memorable line, sparingly (1–2 per report).

## Data-display judgment (decide this yourself — the user won't ask)

When the content carries data, **show it the best way** rather than burying
numbers in prose. Reach for a `chart` block when a picture reads faster than the
numbers:

- **Compare categories** (regions, products, competitors) → `bar`.
- **Trend over time** (years, months, quarters) → `line` (or `area` for volume).
- **Composition / share of a whole**, ≤6 slices → `pie`.
- **Correlation** between two numeric variables → `scatter`.
- A handful of exact figures the reader will cite → `table` or `stat_cards`.

Prefer one clear chart with a takeaway title over a dense table when the point
is a comparison or a trend. Don't chart trivial one-or-two-number facts (use a
stat card) and don't add chart junk. A report that turns its data into a couple
of well-chosen visuals reads as genuinely analyzed, not transcribed.

## Discipline

- **Brand with the user's organization, not the assistant.** Put the firm/team
  name in the cover `meta` (pull from workspace facts / channel context). If you
  genuinely don't know it, omit it — never brand it "Kortny".
- **Prose over bullets.** Reserve lists for ≥3 genuinely comparable items.
- **One takeaway per chart**, stated in its `title`.
- **Tables explained in prose** — a table supports the text.
- Offer a slide version (`deck-builder`) or the data as a workbook
  (`spreadsheet-builder`) when useful — both also run through `document_studio`.

The tool renders and posts the artifact; you do not export or re-render. One
good document that ships beats a perfect one that loops.
