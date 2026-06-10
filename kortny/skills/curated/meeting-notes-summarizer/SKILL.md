---
name: meeting-notes-summarizer
description: Use when asked to turn raw meeting notes, call transcripts, or long discussion threads into a structured summary with decisions, action items, and owners.
metadata:
  version: 1.0.0
  display_name: Meeting Notes Summarizer
---

## Goal

Turn raw meeting notes or transcripts into a summary a teammate can act on without reading the source.

## Steps

1. Read the full source material before summarizing anything.
2. Identify the meeting's purpose and participants if stated; do not invent names.
3. Extract, in this order:
   - **Decisions made** — what was agreed, by whom.
   - **Action items** — task, owner, due date if mentioned. Flag items with no owner as `(unassigned)`.
   - **Open questions** — anything explicitly left unresolved.
   - **Key context** — 2-4 bullets of discussion that explain the decisions.
4. Keep each bullet to one line where possible. Preserve exact numbers, dates, and names from the source.
5. If the source is partial or cut off, say so at the top of the summary.

## Output shape

Lead with a one-line TL;DR. Then the four sections above, omitting any that are empty. Total length should be under a third of the source.
