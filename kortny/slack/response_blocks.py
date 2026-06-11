"""Block Kit response register for agent replies (HIG-235).

Principle (Aneesh): voice stays prose, data goes Block Kit. The LLM never
authors Block Kit JSON. This module inspects the *humanized* response text and,
when it carries a GitHub-style markdown table, wraps it in a single ``markdown``
block so Slack renders the table natively. Prose without tables returns ``None``
so the posting path stays byte-identical (existing tests assert exact text).
"""

from __future__ import annotations

import re

from kortny.slack import blockkit

# A markdown table is a header row of pipe-delimited cells immediately followed
# by a separator row whose cells are runs of dashes (optionally wrapped in
# colons for alignment): ``|---|:--:|---|``. We only need to detect *presence*;
# rendering is delegated to Slack's markdown block.
_SEPARATOR_CELL = re.compile(r"^\s*:?-{1,}:?\s*$")


def _is_pipe_row(line: str) -> bool:
    """True when ``line`` looks like a table row (contains a pipe with content)."""

    return "|" in line and line.strip() != ""


def _is_separator_row(line: str) -> bool:
    """True when ``line`` is a markdown table separator row (``|---|---|``)."""

    stripped = line.strip()
    if "|" not in stripped or "-" not in stripped:
        return False
    # Trim a single leading/trailing pipe before splitting so ``|a|b|`` and
    # ``a|b`` both yield the same cell list.
    inner = stripped
    if inner.startswith("|"):
        inner = inner[1:]
    if inner.endswith("|"):
        inner = inner[:-1]
    cells = inner.split("|")
    if not cells:
        return False
    return all(_SEPARATOR_CELL.match(cell) for cell in cells)


def _contains_markdown_table(text: str) -> bool:
    """Detect ≥1 GitHub-style markdown table, ignoring fenced code blocks.

    A table is a pipe row directly followed by a separator row. Pipe characters
    inside fenced code blocks (``` or ~~~) do not count — code commonly contains
    pipes (shell, bitwise ops) that are not tables.
    """

    lines = text.splitlines()
    in_fence = False
    fence_marker = ""
    previous_was_pipe_row = False
    for raw_line in lines:
        stripped = raw_line.strip()
        if in_fence:
            # A closing fence uses the same marker character (``` or ~~~).
            if stripped.startswith(fence_marker):
                in_fence = False
                fence_marker = ""
            previous_was_pipe_row = False
            continue
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = True
            fence_marker = "```" if stripped.startswith("```") else "~~~"
            previous_was_pipe_row = False
            continue
        if previous_was_pipe_row and _is_separator_row(raw_line):
            return True
        previous_was_pipe_row = _is_pipe_row(raw_line) and not _is_separator_row(
            raw_line
        )
    return False


def render_response_blocks(text: str) -> list[dict] | None:
    """Return Block Kit blocks for ``text`` when it carries a markdown table.

    Returns ``[markdown_block(text)]`` when the text contains at least one
    GitHub-style markdown table AND fits within the markdown block character
    limit. Otherwise returns ``None`` so the caller posts plain prose unchanged
    (the ``text`` fallback on ``post_message`` always carries the full prose).
    """

    if len(text) > blockkit.MAX_MARKDOWN_BLOCK_CHARS:
        return None
    if not _contains_markdown_table(text):
        return None
    return [blockkit.markdown_block(text)]
