# Provenance

**Concept source**: HIG-288 (Kortny internal spec)
**Script authorship**: Original
**Adaptation date**: 2026-06-23
**Adapted by**: Agent

## Script dependencies

- `extract_docx_units.py`: python-docx only (no network).
- `apply_docx_patch.py`: python-docx only (no network).

Both scripts require python-docx>=1.2, which is baked into the sandbox image (`docker/sandbox-exec.Dockerfile`). The operator must rebuild the sandbox image (`docker compose build sandbox-exec-image`) after deploying this skill.

## License

Original implementation. Subject to the Kortny project license.
