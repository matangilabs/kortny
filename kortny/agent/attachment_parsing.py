"""Pure leaf-level parser for Slack file attachment blocks (HIG-279 slice 2C).

This module has zero imports from ``kortny.llm.routing`` or ``kortny.agent``
packages so it can be safely imported by both ``kortny.llm.routing`` (for
deterministic vision-tier routing) and ``kortny.agent.context`` / ``kortny.agent.
image_attachments`` (for context assembly) without creating an import cycle.
"""

from __future__ import annotations

import re

_SLACK_FILES_BLOCK_RE = re.compile(r"<slack_files>\s*(.*?)\s*</slack_files>", re.S)
_SLACK_FILE_ENTRY_RE = re.compile(
    r"^\s*-\s+id:\s*(\S+)((?:\n(?!\s*-\s+id:).+)*)",
    re.M,
)
_SLACK_FILE_MIMETYPE_RE = re.compile(r"^\s+mimetype:\s*(\S+)\s*$", re.M)


def parse_image_attachment_pairs(input_text: str) -> list[tuple[str, str]]:
    """Return ``(file_id, mime)`` pairs for image entries in the ``<slack_files>`` block.

    Pure function with no I/O and no kortny imports beyond this module.  Returns
    an empty list when the input has no ``<slack_files>`` block or contains no
    image-MIME entries.  Safe to call from routing code without an import cycle.
    """

    block_match = _SLACK_FILES_BLOCK_RE.search(input_text)
    if block_match is None:
        return []
    block = block_match.group(1).strip()
    if not block:
        return []

    pairs: list[tuple[str, str]] = []
    for entry_match in _SLACK_FILE_ENTRY_RE.finditer(block):
        file_id = entry_match.group(1).strip()
        entry_body = entry_match.group(2)
        mime_match = _SLACK_FILE_MIMETYPE_RE.search(entry_body)
        if mime_match is None:
            continue
        mime = mime_match.group(1).strip()
        if mime.startswith("image/"):
            pairs.append((file_id, mime))
    return pairs
