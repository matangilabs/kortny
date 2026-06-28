---
name: spreadsheet-builder
description: Use when asked to build, generate, or produce an Excel spreadsheet, .xlsx workbook, financial model, budget, forecast, or formatted data table to upload to Slack — anything where the deliverable is a real working spreadsheet with formulas, number formats, and clean styling, not a table pasted into a message.
metadata:
  version: 1.0.0
  display_name: Spreadsheet Builder
  tags: spreadsheet, excel, xlsx, workbook, budget, forecast, financial model, formulas, openpyxl
---

## Goal

Produce a real, working `.xlsx` workbook that an analyst would accept without
reformatting — live formulas that compute, correct number formats, frozen
headers, and financial color-coding — then upload it to the Slack thread.

## When to use this vs. data-brief

`data-brief` turns numbers into a short prose summary for the channel. Reach
for `spreadsheet-builder` only when the deliverable is the file itself: a
budget, a forecast, a model, or a formatted table the requester will keep
working in.

## Steps

1. **Pin down the shape before building.** Confirm the sheets, the columns,
   which columns are money / percentages / counts, and which columns are
   computed from others. If the request is vague ("make me a budget"), state
   the structure you're assuming in one line before you build.
2. **Decide formats per column.** Money → `currency`; rates → `percent`;
   counts → `thousands`; dates → `date`; labels → `text`. Mark money and
   rate columns `"signed": true` so gains read green and losses read red.
3. **Express every computation as a real formula**, never a pre-computed
   number. A "Total" column is `={qty}{row}*{price}{row}`, not the product
   you worked out yourself — the workbook must recompute when the user edits
   an input. Use `"total_row": true` to sum numeric columns with `SUM`.
4. **Build with the script** (`scripts/build_workbook.py`) by passing a JSON
   spec. It applies frozen headers, the per-column number formats, the
   signed color-coding, the live formulas, and column auto-width.
5. **Upload the file to the thread** with a one-line description of what the
   workbook contains and which cells are inputs vs. computed. Offer a chart
   (`chart-maker`) or a PDF (`document_studio`) as the next artifact.

## Professional standards (non-negotiable)

- **Zero formula errors.** Reference columns by their letters; never leave a
  `#REF!` or a formula that points at an empty cell. If a formula can't be
  expressed cleanly, leave the cell as a plain value and say so.
- **Financial color-coding.** Negative money and negative rates are red,
  positive are green. The number format itself also reds the negatives so the
  convention survives a copy-paste into another sheet.
- **Frozen header row** on every sheet so the columns stay readable when the
  user scrolls.
- **Correct number formats** — never show raw `0.12` where `12.0%` belongs,
  never a bare `120000` where `120,000.00` belongs.
- **One topic per sheet.** Split a model into input / calc / output sheets
  rather than crowding one tab.

If known workspace facts describe the team's currency, fiscal calendar, or a
house format, honor them.

## Script

- `scripts/build_workbook.py` — inputs: `--spec spec.json` (or JSON on stdin),
  `--out workbook.xlsx`. Builds a multi-sheet styled workbook: frozen headers,
  per-column number formats (`currency` / `currency_whole` / `percent` /
  `thousands` / `number` / `date` / `text`), signed green/red coloring,
  `formula_columns` written as live Excel formulas, and an optional `total_row`.
  Deps: openpyxl. No network. See `references/spec-format.md` for the full
  spec reference.
