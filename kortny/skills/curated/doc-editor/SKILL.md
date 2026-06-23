---
name: doc-editor
description: Edit/revise an uploaded Word (.docx) document in place, preserving its formatting — use when the user asks to change, revise, rewrite, or clean up the wording of a document they've shared
metadata:
  version: 1.0.0
  display_name: Doc Editor
  tags: document_editing, docx, revision
---

## Goal

Revise a user-uploaded Word document (.docx) in place — changing the wording as requested while preserving the original formatting (fonts, bold/italic, heading styles, tables, headers, footers). Return the edited file to the user.

**V1 scope: .docx only.** Defer .xlsx, .pptx, .pdf, OCR, charts, tracked-changes, and structural restructuring to a later version.

## Steps

Follow these five steps exactly. Do not skip or reorder them.

**Step 1 — Stage the file into the sandbox**

```
sandbox_stage_file(file_id="<the Slack file ID>", dest_path="original.docx")
```

This downloads the binary .docx and places it at `/workspace/original.docx`.

**Step 2 — Extract editable text units**

```
run_skill_script("extract_docx_units.py")
```

This reads `/workspace/original.docx` and writes `/workspace/units.json`:
```json
{"units": [{"id": "p-0", "text": "...", "sha256": "..."}, ...]}
```

Units cover: body paragraphs (`p-<n>`), table cells (`tbl-<t>-<r>-<c>`), and header/footer paragraphs (`hdr-<n>`, `ftr-<n>`).

**Step 3 — You produce the patch**

Read `units.json` (use `sandbox_read_file("/workspace/units.json")`). Apply the user's requested edits — rewrite, remove, or replace wording as requested. Produce `/workspace/patch.json`:

```json
{"p-3": "Revised paragraph text here", "tbl-0-1-2": "Updated cell text"}
```

Include **only the units whose text changes** — unchanged units must not appear. Write it with `sandbox_write_file("/workspace/patch.json", content=...)`.

**Step 4 — Apply the patch to the .docx**

```
run_skill_script("apply_docx_patch.py")
```

This reads `original.docx` + `patch.json`, replaces the matched units' text while preserving run-level formatting (bold, italic, font, size), and saves `/workspace/revised.docx`.

**Step 5 — Export and post**

```
sandbox_export_artifact("/workspace/revised.docx")
```

This delivers `revised.docx` back to the user in Slack. Tell the user what you changed (a brief bulleted list of the main edits).

## Formatting preservation

The apply script preserves formatting best-effort at the run level: it sets the first run's text to the new value and clears the rest. This preserves bold, italic, font family, and font size on the opening run. Run-level variance within a paragraph (e.g. mixed bold/plain within one sentence) is flattened to the first run's style for patched units.

## Limitations

- .docx only (V1). Other formats are out of scope.
- Structural changes (add/delete paragraphs, reorder sections) are not supported — use `sandbox_bash` with a custom python-docx script for those.
- Tracked changes / revision marks are ignored.
- If `sandbox_stage_file` fails with `file_too_large_for_sandbox` (over 5 MB), tell the user the file is too large for in-sandbox editing and offer to extract and summarize the text instead (via `slack_file_read`).
- Charts and images inside the .docx are preserved (not modified) since only text units are patched.
