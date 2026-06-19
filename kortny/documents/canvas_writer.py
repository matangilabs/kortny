"""Render a Document Studio IR to Slack Canvas markdown (HIG-244 close-out).

Canvas is a *living, editable* delivery surface for prose-shaped docs (memos,
briefs, runbooks) — the team keeps it in the channel rather than downloading a
file. Canvas markdown has no chart support, so chart blocks are dropped and
reported in ``omitted_blocks`` (the caller tells the user to use pdf for charts).
"""

from __future__ import annotations

from kortny.documents.ir import (
    CTA,
    Callout,
    Chart,
    CoverHeader,
    DocumentSpec,
    Heading,
    Prose,
    PullQuote,
    SectionDivider,
    StatCards,
    Table,
)


def render_canvas_markdown(spec: DocumentSpec) -> tuple[str, list[str]]:
    """Return (canvas markdown, omitted block descriptions)."""

    lines: list[str] = []
    omitted: list[str] = []
    chart_index = 0

    for block in spec.blocks:
        if isinstance(block, CoverHeader):
            lines.append(f"# {block.title}")
            if block.subtitle:
                lines.append(f"_{block.subtitle}_")
            lines.extend(f"- {item}" for item in block.meta)
        elif isinstance(block, SectionDivider):
            prefix = f"{block.index} " if block.index else ""
            label = f"{block.label}: " if block.label else ""
            lines.append(f"## {prefix}{label}{block.title}")
        elif isinstance(block, Heading):
            lines.append(f"## {block.text}")
        elif isinstance(block, Prose):
            lines.extend(p.strip() for p in block.text.split("\n\n") if p.strip())
        elif isinstance(block, StatCards):
            lines.append("| Metric | Value | Note |")
            lines.append("| --- | --- | --- |")
            lines.extend(
                f"| {c.label} | {c.value} | {c.note or ''} |" for c in block.cards
            )
        elif isinstance(block, Table):
            if block.caption:
                lines.append(f"**{block.caption}**")
            lines.extend(_markdown_table(block))
        elif isinstance(block, Callout):
            label = f"**{block.label}:** " if block.label else ""
            lines.append(f"> {label}{block.text}")
        elif isinstance(block, PullQuote):
            lines.append(f"> {block.text}")
            if block.attribution:
                lines.append(f"> — {block.attribution}")
        elif isinstance(block, CTA):
            tail = f" — {block.text}" if block.text else ""
            lines.append(f"**{block.label}**{tail}")
        elif isinstance(block, Chart):
            chart_index += 1
            omitted.append(block.title or f"chart #{chart_index}")
        lines.append("")  # blank line between blocks

    if omitted:
        lines.append(
            "_Charts aren't supported in a canvas — re-render as PDF to include "
            f"them: {', '.join(omitted)}._"
        )
    markdown = "\n".join(lines).strip() + "\n"
    return markdown, omitted


def _markdown_table(block: Table) -> list[str]:
    rows = [f"| {' | '.join(block.columns)} |"]
    rows.append("| " + " | ".join("---" for _ in block.columns) + " |")
    ncol = len(block.columns)
    for row in block.rows:
        padded = (list(row) + [""] * ncol)[:ncol]
        rows.append(f"| {' | '.join(padded)} |")
    return rows


__all__ = ["render_canvas_markdown"]
