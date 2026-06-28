# Curated skill pack

Curated skills are the SKILL.md catalog Kortny ships with. They live as
directories under `kortny/skills/curated/` and are auto-ingested idempotently at
startup by `SkillRegistryService.ensure_curated_skills()` — adding a skill is
adding a directory; no registry code change is needed per skill.

Each skill is a directory with a `SKILL.md` (frontmatter: `name`, `description`,
`metadata.{version,display_name,tags}`) plus optional `references/*.md` loaded on
demand and, for builders, `scripts/`. Descriptions are written **selection-first**
— phrased around what a user actually asks in Slack — because the description is
what retrieval embeds.

## Tiers

- **default** — auto-enabled at workspace scope on seeding (the everyday pack;
  see `DEFAULT_PACK_SLUGS` alongside `PLAYBOOK_SKILL_SLUGS` in
  `kortny/skills/service.py`). Existing playbook skills remain auto-enabled
  unchanged.
- **catalog** — registered and discoverable but not auto-enabled; an admin
  enables them per workspace/channel/user from the dashboard.

## V-class

"V-class" (V1–V10) marks the Viktor-class competitive-moat abilities from the
HIG-239 ticket — the skills that anchor Kortny against competitors. The six
Agent-C originals are the moat core: they exist in no public registry and are
authored from scratch.

## Pack overview

| Slug | Tier | V-class | Source | License |
|------|------|---------|--------|---------|
| ambient-responder | default | — | original (HIG-229) | — |
| anticipatory-draft | default | — | original (HIG-229) | — |
| competitive-analysis | default | — | original (HIG-229) | — |
| data-brief | default | — | original (HIG-229) | — |
| decision-tracker | default | — | original (HIG-229) | — |
| meeting-notes-summarizer | default | — | original (HIG-229) | — |
| project-checkin | default | — | original (HIG-229) | — |
| weekly-status-report | default | — | original (HIG-229) | — |
| weekly-channel-digest | default | V1 | original (HIG-239) | — |
| thread-recap | default | V4 | original (HIG-239) | — |
| data-digest | default | V2 | original (HIG-239) | — |
| competitor-watch | catalog | V6 | original (HIG-239) | — |
| cited-research-brief | default | V10 | original (HIG-239) | — |
| lead-research | catalog | V5 | original (HIG-239) | — |
| internal-comms | default | — | Kortny + vendored refs (anthropics/skills) | Apache-2.0 (refs) |
| frontend-design | catalog | — | Kortny + vendored refs (anthropics/skills) | Apache-2.0 (refs) |
| theme-factory | catalog | — | Kortny + vendored refs (anthropics/skills) | Apache-2.0 (refs) |
| canvas-design | catalog | — | Kortny + vendored refs/fonts (anthropics/skills) | Apache-2.0 (refs) + OFL (fonts) |
| slack-gif-creator | catalog | — | original (concept: anthropics/skills) | Kortny |
| skill-creator | catalog | — | original (concept: anthropics/skills) | Kortny |
| article-extractor | catalog | — | original (concept: tapestry-skills, MIT) | Kortny |
| youtube-transcript | catalog | — | original (concept: tapestry-skills, MIT) | Kortny |
| financial-statements | catalog | — | original (concept: claude-cookbooks, MIT) | Kortny |
| brand-template | default | — | original (concept: claude-cookbooks, MIT) | Kortny |
| competitor-profiling | default | — | original (concept: coreyhaines31/marketingskills, MIT) | Kortny |
| comparison-pages | catalog | — | original (concept: coreyhaines31/marketingskills, MIT) | Kortny |
| prospecting | catalog | — | original (concept: coreyhaines31/marketingskills, MIT) | Kortny |
| revops | catalog | — | original (concept: coreyhaines31/marketingskills, MIT) | Kortny |
| marketing-analytics | catalog | — | original (concept: coreyhaines31/marketingskills, MIT) | Kortny |
| copywriting | catalog | — | original (concept: coreyhaines31/marketingskills, MIT) | Kortny |
| cold-email | catalog | — | original (concept: coreyhaines31/marketingskills, MIT) | Kortny |
| launch-strategy | catalog | — | original (concept: coreyhaines31/marketingskills, MIT) | Kortny |
| pricing-strategy | catalog | — | original (concept: coreyhaines31/marketingskills, MIT) | Kortny |
| summarize-meeting | default | — | original (concept: phuryn/pm-skills, MIT) | Kortny |
| release-notes | catalog | — | original (concept: phuryn/pm-skills, MIT) | Kortny |
| cohort-analysis | catalog | — | original (concept: phuryn/pm-skills, MIT) | Kortny |
| competitive-battlecard | catalog | — | original (concept: phuryn/pm-skills, MIT) | Kortny |
| lead-qualification | catalog | — | original (concept: TerminalSkills/skills, Apache-2.0) | Kortny |
| changelog-generator | catalog | — | original (concept: TerminalSkills/skills, Apache-2.0) | Kortny |
| report-generator | default | — | original (concept: TerminalSkills/skills, Apache-2.0) | Kortny |
| last30days | catalog | — | original (concept: mvanhorn/last30days-skill, MIT) | Kortny |
| spreadsheet-builder | default | V2/V3 | original (concept-rewrite) | — |
| deck-builder | catalog | V9 | original (concept-rewrite) | — |
| chart-maker | default | V3 | original (concept-rewrite) | — |

> Source/license columns for Agent A/B/D rows reflect the HIG-239 plan and are
> reconciled by the orchestrator at the gate against each dir's `PROVENANCE.md`
> and `LICENSE.txt`. Tiering follows `DEFAULT_PACK_SLUGS` (Agent E); any slot an
> agent honestly couldn't differentiate or license was skipped and noted in that
> agent's report rather than shipped.

## The six moat originals (Agent C)

Authored from scratch — they exist in no public registry. Each is selection-first
and built to use Kortny's own context (workspace facts, thread context, episodes,
knowledge graph) and tools (web search, Composio integrations when connected).

- **weekly-channel-digest** (V1) — recurring synthesis of a channel's week into
  themes, decisions, open questions, and notable links. Differs from
  `weekly-status-report` (team-accomplishment reporting): this synthesizes
  *channel activity*, answering "what happened in here while I was out."
- **thread-recap** (V4) — decisions, action items, owners, and open questions from
  the **live Slack thread**. Differs from `meeting-notes-summarizer`
  (pasted transcript/notes as input): the thread itself is the source.
- **data-digest** (V2) — source-agnostic **recurring** digest (CSV/sheet/SQL/
  Composio data) leading with deltas, trends, and anomalies vs the prior run.
  Differs from `data-brief` (one-shot story from a single posted file): this is
  longitudinal and comparison-driven.
- **competitor-watch** (V6) — scheduled fetch-diff-summarize of competitor sites,
  changelogs, and pricing; reports **only what changed**. Differs from
  `competitive-analysis` (one-time assessment): this is ongoing change monitoring.
- **cited-research-brief** (V10) — multi-source research with inline citations to
  a thread or canvas, a source-quality rubric, and explicit disagreement
  surfacing.
- **lead-research** (V5) — company/person research and qualification into a
  structured lead sheet; uses a connected CRM (Composio HubSpot/Salesforce/
  Apollo) when available, web research otherwise.

## Skipped

- **doc-coauthoring** (anthropics-side source, flagged by Agent A) — license
  ambiguous, so it was **not** harvested. Per the pack license rule, ambiguous /
  proprietary / non-permissive sources are concept-rewrite-only; this slot was
  dropped rather than copied. Revisit only if the upstream license is clarified
  to Apache-2.0/MIT, or rewrite the capability from scratch.
