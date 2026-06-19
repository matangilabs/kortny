"""Built-in procedural skills shipped with Kortny."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_CURATED_DIR = Path(__file__).parent / "curated"


def _curated_body(slug: str) -> str:
    """Return a curated skill's SKILL.md body (frontmatter stripped).

    Some built-ins inject a curated skill's body verbatim so the curated
    ``SKILL.md`` stays the single source of truth (it is also the open-sourceable
    artifact) while the built-in is the reliable always-seeded injection path.
    """

    text = (_CURATED_DIR / slug / "SKILL.md").read_text(encoding="utf-8")
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            return parts[2].strip()
    return text.strip()


@dataclass(frozen=True, slots=True)
class BuiltInSkillDefinition:
    """Immutable definition for a system-owned procedural skill."""

    slug: str
    name: str
    version: str
    description: str
    instructions_md: str
    intent_tags: tuple[str, ...]
    response_modes: tuple[str, ...]
    trigger_phrases: tuple[str, ...] = ()
    metadata: dict[str, str] | None = None


BUILTIN_SKILLS: tuple[BuiltInSkillDefinition, ...] = (
    BuiltInSkillDefinition(
        slug="slack-humanizer",
        name="Slack Humanizer",
        version="1.0.0",
        description=(
            "Use when turning a raw final answer into a concise, natural Slack "
            "reply while preserving factual meaning and execution evidence."
        ),
        intent_tags=("response_humanizer", "slack", "rendering", "style"),
        response_modes=(
            "quick_answer",
            "research_summary",
            "file_analysis",
            "artifact_delivery",
            "failure_recovery",
            "memory_recall",
            "multi_step_recap",
        ),
        trigger_phrases=("make this sound natural", "humanize", "slack reply"),
        metadata={"kind": "instruction_only", "source": "kortny"},
        instructions_md="""## Goal
Write the final answer like a capable teammate replying in Slack.

## Rules
- Preserve the substance, numbers, caveats, sources, artifacts, and tool outcomes from the response record.
- Do not introduce new claims or imply a tool succeeded if the record says it failed.
- Lead with the answer or recommendation.
- Prefer short paragraphs and tight bullets over report-style sections.
- Use Slack mrkdwn only: *bold*, simple bullets, and <https://url|label> links.
- Remove chatbot filler, generic sign-offs, and repetitive "If you want..." endings.
- Make uncertainty sound honest and useful, not apologetic.
- Suggest a next step only when it is specific and clearly useful.
""",
    ),
    BuiltInSkillDefinition(
        slug="research-synthesis",
        name="Research Synthesis",
        version="1.0.0",
        description=(
            "Use when synthesizing multiple search or integration results into "
            "a grounded, source-aware answer."
        ),
        intent_tags=("research", "synthesis", "sources"),
        response_modes=("research_summary",),
        trigger_phrases=("research", "compare", "audit", "find sources"),
        metadata={"kind": "instruction_only", "source": "kortny"},
        instructions_md="""## Goal
Turn multiple retrieved sources into a concise answer with a clear bottom line.

## Rules
- Separate observations from recommendations.
- Prefer the strongest recurring pattern over a long list of links.
- Mention source limitations when results are shallow, stale, or conflicting.
- Do not cite a source that was not present in the evidence pack.
""",
    ),
    BuiltInSkillDefinition(
        slug="analyst-grade-synthesis",
        name="Analyst-Grade Synthesis",
        version="1.0.0",
        description=(
            "Use when a response needs consulting-grade analysis: audits, "
            "reviews, comparisons, critiques, recommendations, or strategy memos."
        ),
        intent_tags=("analysis", "audit", "review", "critique", "decision"),
        response_modes=(
            "quick_answer",
            "research_summary",
            "file_analysis",
            "multi_step_recap",
        ),
        trigger_phrases=(
            "audit",
            "review",
            "critique",
            "compare",
            "recommend",
            "which one",
        ),
        metadata={"kind": "instruction_only", "source": "kortny"},
        instructions_md="""## Goal
Turn raw findings into an opinionated, evidence-aware coworker analysis.

## Rules
- Start with the recommendation or bottom line.
- State scope: what was inspected, compared, or assumed.
- Rank findings by importance instead of listing everything equally.
- Explain why each finding matters.
- Include concrete recommendations, reframes, or next moves where the evidence supports them.
- Call out the highest-leverage move when there is an obvious first step.
- Do not invent evidence, scores, sources, pages, or tool results.
- Keep the answer Slack-native and readable by a busy operator.
""",
    ),
    BuiltInSkillDefinition(
        slug="document-iteration",
        name="Document Iteration",
        version="1.0.0",
        description=(
            "Use when revising or extending an existing document while "
            "preserving document identity, title, and version lineage."
        ),
        intent_tags=("document", "iteration", "artifact"),
        response_modes=("artifact_delivery", "file_analysis"),
        trigger_phrases=("revise this", "extend this report", "make it longer"),
        metadata={"kind": "instruction_only", "source": "kortny"},
        instructions_md="""## Goal
Revise an existing document as a continuation of the same artifact.

## Rules
- Preserve the document title unless the user explicitly asks to change it.
- Preserve filename lineage and use version suffixes for revisions.
- Add materially new content when asked to elaborate; do not pad with repeated sections.
- Keep the visible Slack response focused on what changed.
""",
    ),
    BuiltInSkillDefinition(
        slug="slack-formatting",
        name="Slack Formatting",
        version="1.0.0",
        description=("Use when converting generic Markdown into Slack-native mrkdwn."),
        intent_tags=("slack", "formatting", "mrkdwn"),
        response_modes=(
            "quick_answer",
            "research_summary",
            "file_analysis",
            "memory_recall",
            "multi_step_recap",
        ),
        metadata={"kind": "instruction_only", "source": "kortny"},
        instructions_md="""## Goal
Make formatting render cleanly in Slack.

## Rules
- Convert Markdown bold to Slack `*bold*`.
- Convert Markdown links to Slack `<url|label>` links.
- Avoid Markdown headings and tables.
- Use short bullets when they improve scanning.
""",
    ),
    BuiltInSkillDefinition(
        slug="status-recap",
        name="Status Recap",
        version="1.0.0",
        description=(
            "Use when summarizing what happened, what changed, and what needs "
            "attention across a channel, thread, or recent task sequence."
        ),
        intent_tags=("summary", "recap", "status"),
        response_modes=("multi_step_recap", "research_summary"),
        trigger_phrases=("summarize", "what changed", "status"),
        metadata={"kind": "instruction_only", "source": "kortny"},
        instructions_md="""## Goal
Produce a compact status recap from recent context.

## Rules
- Lead with what matters now.
- Group related items instead of listing every message.
- Call out open questions or blockers only when the evidence supports them.
- Avoid sounding like an activity log.
""",
    ),
    BuiltInSkillDefinition(
        slug="slack-block-kit",
        name="Slack Block Kit Presentation",
        version="1.0.0",
        description=(
            "Use when a reply carries structured data — a list of entities with "
            "attributes, key-value facts/metrics, status, comparisons, or rows — "
            "to render it as native Slack Block Kit via a presentation hint."
        ),
        intent_tags=("slack", "block kit", "presentation", "data", "rendering"),
        response_modes=("all",),
        trigger_phrases=("show", "list", "status", "breakdown", "compare"),
        metadata={"kind": "instruction_only", "source": "kortny"},
        # Body is the curated SKILL.md (single source of truth); injected into the
        # humanizer so it reliably emits a presentation hint for structured data.
        instructions_md=_curated_body("slack-block-kit"),
    ),
)
