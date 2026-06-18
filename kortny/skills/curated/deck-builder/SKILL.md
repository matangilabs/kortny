---
name: deck-builder
description: Use when asked to build, generate, or produce a slide deck, PowerPoint, .pptx, presentation, pitch deck, or board/review deck to upload to Slack — when the deliverable is an actual slide file with a consistent theme and layout, not talking points in a message.
metadata:
  version: 2.0.0
  display_name: Deck Builder
  tags: deck, slides, presentation, powerpoint, pptx, pitch, board deck, review, document-studio
---

## Goal

Produce a `.pptx` deck that reads as one coherent document — one idea per slide,
a single theme throughout, generous margins — and post it to the Slack thread.

You do **not** run a build script. You author a structured document spec and
call the **`document_studio` tool** with `format: "pptx"`; it renders the themed
deck (cover → section dividers → content slides) and records the artifact.

## How to build the deck

Call `document_studio` with `format: "pptx"`, a `title`, `doc_kind: "pitch"`
(sparse, high-impact) or `"report"` (denser), and an ordered `blocks` array:

- `cover_header` — the title slide (eyebrow, title, subtitle, meta).
- `section_divider` — a section break slide.
- `heading` → following blocks flow onto that content slide.
- `prose` — keep it tight; one idea per slide, not a wall of text.
- `stat_cards` — headline metrics.
- `table` — comparison data.
- `chart` — **visualize data; see judgment below.**
- `pull_quote`, `cta` — a memorable line / a closing ask.

## Data-display judgment (decide this yourself — the user won't ask)

A deck is the format where a chart earns its keep most. When a slide is about
data, make it a `chart` block rather than a list of numbers:

- **Compare categories** → `bar`. **Trend over time** → `line`/`area`.
- **Share of a whole** (≤6 slices) → `pie`. **Correlation** → `scatter`.

One chart per data slide, with the takeaway in the chart `title`.

## Discipline

- **One idea per slide.** Two topics or a wall of bullets → two slides.
- **Titles are sentences** that carry the point ("Signups doubled after the
  referral launch"), not labels ("Signups").
- **Consistent theme** — pick `doc_kind`/theme once; the renderer applies one
  accent and font system to every slide.
- **Brand with the user's org**, not the assistant (cover `meta`).
- Offer a leave-behind PDF (`format: "pdf"`) of the same spec when useful.
