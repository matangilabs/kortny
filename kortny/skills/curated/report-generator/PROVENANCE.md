# Provenance

Independently authored for Kortny (2026-06-12). Concept and capability slot informed by research into https://github.com/TerminalSkills/skills (Apache-2.0); no upstream text or files were copied.

## Design notes

- HTML output mode omitted per HIG-239: Kortny does not serve web pages. Output is: (1) Slack mrkdwn post (always, inline), (2) uploaded .md file, (3) PDF via the `document_studio` tool.
- Audience-tiering as a design dimension — multiple audience levels with different depth/framing — is a general practice; documented in `references/audience-tiers.md`.
- Structure guide provided as `references/structure-guide.md`.
- No scripts — pure text generation handled by the LLM.
