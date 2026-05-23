"""Slack mrkdwn normalization helpers."""

from __future__ import annotations

import re

CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
MARKDOWN_LINK_RE = re.compile(r"(?<!!)\[([^\]\n]+)\]\((https?://[^)\s]+)\)")
MARKDOWN_HEADING_RE = re.compile(r"(?m)^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$")
MARKDOWN_BOLD_RE = re.compile(r"(?<!\*)\*\*([^*\n][^\n]*?[^*\n]?)\*\*(?!\*)")
CODE_PLACEHOLDER = "<<KORTNY_CODE_{index}>>"


def normalize_slack_mrkdwn(text: str) -> str:
    """Normalize common Markdown shapes into Slack mrkdwn."""

    if not text:
        return text

    protected_text, protected_segments = _protect_code(text)
    normalized = MARKDOWN_LINK_RE.sub(_markdown_link_to_slack, protected_text)
    normalized = MARKDOWN_BOLD_RE.sub(r"*\1*", normalized)
    normalized = MARKDOWN_HEADING_RE.sub(_markdown_heading_to_slack, normalized)
    return _restore_code(normalized, protected_segments)


def _protect_code(text: str) -> tuple[str, list[str]]:
    protected_segments: list[str] = []

    def replace(match: re.Match[str]) -> str:
        protected_segments.append(match.group(0))
        return CODE_PLACEHOLDER.format(index=len(protected_segments) - 1)

    protected = CODE_BLOCK_RE.sub(replace, text)
    protected = INLINE_CODE_RE.sub(replace, protected)
    return protected, protected_segments


def _restore_code(text: str, protected_segments: list[str]) -> str:
    restored = text
    for index, segment in enumerate(protected_segments):
        restored = restored.replace(CODE_PLACEHOLDER.format(index=index), segment)
    return restored


def _markdown_link_to_slack(match: re.Match[str]) -> str:
    label = match.group(1).replace("|", "-").strip()
    url = match.group(2).strip()
    return f"<{url}|{label}>"


def _markdown_heading_to_slack(match: re.Match[str]) -> str:
    heading = match.group(1).strip()
    if heading.startswith("*") and heading.endswith("*"):
        return heading
    return f"*{heading}*"
