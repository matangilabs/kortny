# Provenance

**Concept source**: Anthropic proprietary internal document skills (pdf / report authoring)
**Script authorship**: Original (clean-room rewrite)
**Adaptation date**: 2026-06-12 (v1), 2026-06-13 (v2 editorial rebuild)
**Adapted by**: Agent D (HIG-239), v2 rebuild by Claude Sonnet (HIG-244 prep)

## What was adapted

The SKILL.md was written fresh. The script (`render_pdf.py`) and its embedded
design system are an original implementation using WeasyPrint — no code or
wording was copied from any proprietary upstream source. The upstream skill is
proprietary; only the *layout standards* (unprotectable ideas) informed this
skill.

**v2 changes (2026-06-13):** Complete redesign of the CSS design system based
on the research documents at `docs/research/doc-studio/`. The `BASE_CSS` was
replaced with a full themed system (3 named themes, 20+ CSS custom property
tokens, 10+ component classes). The script gained `--style` selection, a
`--kicker` flag, and named-page full-bleed cover support. SKILL.md and
authoring.md were fully rewritten.

## Script dependencies

- `render_pdf.py`: weasyprint only. Requires the pango / cairo system
  libraries (baked into the sandbox image). argparse, HTML in (file or stdin),
  `.pdf` file out, no network access (remote images fail closed).

## Bundled assets

**Fonts are bundled (HIG-244)** as variable TTFs in `scripts/fonts/` and wired
via the `_FONT_FACE_*` `@font-face` blocks in `render_pdf.py`. The render passes
`base_url=str(scripts_dir)` so relative `fonts/...` paths resolve, and WeasyPrint
(Pango/HarfBuzz) selects weights from the variable files. The system-font
fallback stacks in `THEMES` remain as a safety net.

| Font (file) | Used by | License |
|---|---|---|
| Fraunces.ttf / Fraunces-Italic.ttf | editorial-mono, editorial-feature (display) | SIL OFL 1.1 |
| Newsreader.ttf / Newsreader-Italic.ttf | editorial-mono, editorial-feature (body) | SIL OFL 1.1 |
| IBMPlexMono-Regular.ttf / IBMPlexMono-SemiBold.ttf | editorial-mono (mono) | SIL OFL 1.1 |
| SpaceGrotesk.ttf | editorial-feature (mono) | SIL OFL 1.1 |
| Inter.ttf / Inter-Italic.ttf | minimal (display + body) | SIL OFL 1.1 |
| JetBrainsMono.ttf | minimal (mono) | Apache 2.0 |

All sourced from the Google Fonts OFL repository (github.com/google/fonts).
Per-font notices are recorded in `LICENSE.txt`. Verified: fonts embed (subset)
in rendered PDFs across all three themes via WeasyPrint with pango/cairo.

## Chromium/Gotenberg upgrade (HIG-244)

The current WeasyPrint-only approach uses:
- Named pages (`page: cover`) + `@page cover { margin: 0 }` for full-bleed
  covers — WeasyPrint-native, not supported in Chromium.
- `@page` margin boxes for page numbers and running headers — WeasyPrint-native.
- `string-set` / `content(string())` for running section name — WeasyPrint-native.

When HIG-244 (Gotenberg/Chromium) lands, the Chromium path will need:
- Zero global `@page` margin + inset `.page-body` padding (already done).
- A `footer.html` template for page numbers (Chromium ignores margin boxes).
- Named-page full-bleed via the global-zero-margin pattern (already done —
  `.cover` and `.section-divider` use `width:210mm; min-height:297mm` which
  produces full bleed when global margin is 0).
- `printBackground=true` + `preferCssPageSize=true` Gotenberg form fields.
- Font woff2 files uploaded as multipart parts alongside the HTML.

## Slack-first adaptations

- Output targets a Slack file upload (the PDF itself).
- Aesthetic / accent color sourced from known workspace facts or `--accent` CLI.
- Images are referenced by absolute workbench path (no network at render time).

## License

Original concept source is proprietary — this skill and script are a
clean-room rewrite. This file and all files in this directory are subject to
the Kortny project license. If fonts are bundled in `scripts/fonts/`, each
font file is subject to its own OFL 1.1 or Apache 2.0 license respectively.
