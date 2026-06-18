---
name: report-generator
description: Use when asked to generate a structured report from data, notes, or research — executive summary, detailed findings, audience-tiered versions — delivered as a Slack mrkdwn post plus a file upload for the full document.
metadata:
  version: 2.0.0
  display_name: Report Generator
  tags: report, executive summary, findings, analysis, audience-tiering, document, document-studio
---

## Goal

Produce a structured, audience-appropriate report and deliver it where the audience lives — a tight mrkdwn summary in Slack, with the full report as an uploaded file.

## Steps

1. **Clarify scope** — what is the report about? What data, notes, or research inputs are available? Who is the audience and what decision does this report support?
2. **Tier the audience** — use the audience-tiering table in `references/audience-tiers.md` to determine which version(s) to produce.
3. **Draft the report** — structure it per `references/report-structure.md`. Lead every section with the conclusion, not the methodology.
4. **Write the Slack summary** — a tight mrkdwn post with: TL;DR (1 sentence), 3-5 key findings (bullets), and the primary recommendation or next step. Post this to Slack inline.
5. **Produce the full report file** — call the **`document_studio` tool** (`format: "pdf"`, or `"docx"` if the user will keep editing it). Author the structured `blocks` (cover, headings + prose, stat_cards, table, callout) and **visualize data with `chart` blocks where it aids comprehension** — see judgment below. The tool renders the themed file and records the artifact.
6. **Offer tiered versions** — if multiple audiences need this (exec vs. team vs. technical), produce the exec version first and offer to adapt.

## Data-display judgment (decide this yourself — the user won't ask)

When findings carry data, show it the clearest way rather than burying numbers in prose: compare categories → `bar`; trend over time → `line`/`area`; share of a whole (≤6) → `pie`; correlation → `scatter`; a few exact figures → `stat_cards`/`table`. One chart, one takeaway (in its title).

## Output delivery

- **Slack post**: mrkdwn TL;DR + key findings bullets (always posted inline).
- **File upload**: full report via `document_studio` (`pdf` finished / `docx` editable). Reserve a plain `.md` upload only when a structured document is genuinely not wanted.

## Rules

- Conclusions before methodology. Executives read the first 3 bullets; put the finding there.
- Every quantitative claim cites its source or states "based on provided data".
- Do not pad reports with methodology descriptions when the audience is results-focused.
- Use workspace facts for brand voice, product names, and any known audience context.
- Reports > 1000 words belong in the file upload, not the Slack post.
