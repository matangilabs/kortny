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

**No fonts are currently bundled.** The script uses strong system-font
fallback stacks:

- Display/headings: Fraunces → Source Serif 4 → Georgia → serif
- Body: Newsreader → Source Serif 4 → Georgia → serif
- Mono/kickers: IBM Plex Mono → ui-monospace → Menlo → monospace

For full Editorial Mono fidelity, the following OFL/Apache fonts should be
bundled as static TTF or woff2 files in `scripts/fonts/`:

| Font | License | Source |
|---|---|---|
| Fraunces (Regular + Bold) | SIL OFL 1.1 | https://fonts.google.com/specimen/Fraunces |
| Source Serif 4 (Regular + Bold) | SIL OFL 1.1 | https://fonts.google.com/specimen/Source+Serif+4 |
| IBM Plex Mono (Regular + SemiBold) | SIL OFL 1.1 | https://fonts.google.com/specimen/IBM+Plex+Mono |
| Newsreader (Regular + Bold) | SIL OFL 1.1 | https://fonts.google.com/specimen/Newsreader |
| Space Grotesk (Regular + Medium) | SIL OFL 1.1 | https://fonts.google.com/specimen/Space+Grotesk |
| Inter (Regular + SemiBold) | SIL OFL 1.1 | https://fonts.google.com/specimen/Inter |
| JetBrains Mono (Regular) | Apache 2.0 | https://www.jetbrains.com/legalforms/mono-type-license |

If `scripts/fonts/` is populated and `@font-face` src paths are uncommented
in `render_pdf.py`, pass `base_url=str(scripts_dir)` to WeasyPrint so
relative `fonts/...` paths resolve. The script already sets this.

If fonts are bundled, update `LICENSE.txt` to list each font with its OFL/
Apache notice. Until bundled, no additional license entries are needed.

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
