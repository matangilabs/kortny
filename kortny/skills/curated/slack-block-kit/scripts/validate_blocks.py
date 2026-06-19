#!/usr/bin/env python3
"""Validate a Slack Block Kit payload against the platform's hard limits.

Standalone and dependency-free so it is usable outside Kortny. Returns a list of
human-readable violations (empty list = valid). Slack rejects over-limit
payloads server-side ("invalid_blocks" / "Blocks too long"), so validating
before posting — and degrading to plain text on failure — keeps a bad layout
from dropping the message entirely.

Usage:
    python validate_blocks.py blocks.json [--surface message|modal|home]
    echo '[...blocks...]' | python validate_blocks.py --surface message
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

# Slack-enforced limits (see references/elements.md).
MAX_MESSAGE_BLOCKS = 50
MAX_VIEW_BLOCKS = 100
MAX_MARKDOWN_CHARS = 12_000  # cumulative across all markdown blocks in a payload
MAX_SECTION_TEXT = 3_000
MAX_SECTION_FIELDS = 10
MAX_SECTION_FIELD_CHARS = 2_000
MAX_CONTEXT_ELEMENTS = 10
MAX_HEADER_CHARS = 150
MAX_TABLE_ROWS = 100
MAX_TABLE_COLUMNS = 20
MAX_TABLE_CHARS = 10_000
MAX_BUTTON_TEXT = 75
MAX_BUTTON_VALUE = 2_000

_MESSAGE_ONLY = {"table"}


def validate_blocks(
    blocks: list[dict[str, Any]], *, surface: str = "message"
) -> list[str]:
    """Return a list of limit/structure violations; empty means valid."""

    errors: list[str] = []
    if not isinstance(blocks, list):
        return ["payload must be a list of blocks"]

    cap = MAX_MESSAGE_BLOCKS if surface == "message" else MAX_VIEW_BLOCKS
    if len(blocks) > cap:
        errors.append(f"{len(blocks)} blocks exceeds the {surface} limit of {cap}")

    markdown_chars = 0
    seen_block_ids: set[str] = set()
    for i, block in enumerate(blocks):
        if not isinstance(block, dict):
            errors.append(f"block[{i}] is not an object")
            continue
        btype = block.get("type")
        loc = f"block[{i}] ({btype})"

        block_id = block.get("block_id")
        if isinstance(block_id, str) and block_id:
            if block_id in seen_block_ids:
                errors.append(f"{loc}: duplicate block_id {block_id!r}")
            seen_block_ids.add(block_id)

        if btype in _MESSAGE_ONLY and surface != "message":
            errors.append(f"{loc}: '{btype}' blocks render only on the message surface")

        if btype == "markdown":
            markdown_chars += len(str(block.get("text", "")))
        elif btype == "header":
            text = _plain_text(block.get("text"))
            if len(text) > MAX_HEADER_CHARS:
                errors.append(f"{loc}: header {len(text)} > {MAX_HEADER_CHARS} chars")
        elif btype == "section":
            errors.extend(_validate_section(block, loc))
        elif btype == "context":
            elements = block.get("elements") or []
            if len(elements) > MAX_CONTEXT_ELEMENTS:
                errors.append(
                    f"{loc}: {len(elements)} elements > {MAX_CONTEXT_ELEMENTS}"
                )
        elif btype == "table":
            errors.extend(_validate_table(block, loc))
        elif btype == "actions":
            errors.extend(_validate_actions(block, loc))

    if markdown_chars > MAX_MARKDOWN_CHARS:
        errors.append(
            f"markdown blocks total {markdown_chars} > {MAX_MARKDOWN_CHARS} chars"
        )
    return errors


def _plain_text(obj: Any) -> str:
    if isinstance(obj, dict):
        return str(obj.get("text", ""))
    return str(obj or "")


def _validate_section(block: dict[str, Any], loc: str) -> list[str]:
    errors: list[str] = []
    text = _plain_text(block.get("text"))
    if len(text) > MAX_SECTION_TEXT:
        errors.append(f"{loc}: section text {len(text)} > {MAX_SECTION_TEXT} chars")
    fields = block.get("fields") or []
    if len(fields) > MAX_SECTION_FIELDS:
        errors.append(f"{loc}: {len(fields)} fields > {MAX_SECTION_FIELDS}")
    for field in fields:
        if len(_plain_text(field)) > MAX_SECTION_FIELD_CHARS:
            errors.append(f"{loc}: a field exceeds {MAX_SECTION_FIELD_CHARS} chars")
            break
    return errors


def _validate_table(block: dict[str, Any], loc: str) -> list[str]:
    errors: list[str] = []
    rows = block.get("rows") or []
    if len(rows) > MAX_TABLE_ROWS:
        errors.append(f"{loc}: {len(rows)} rows > {MAX_TABLE_ROWS}")
    total = 0
    for row in rows:
        cells = row if isinstance(row, list) else []
        if len(cells) > MAX_TABLE_COLUMNS:
            errors.append(f"{loc}: {len(cells)} columns > {MAX_TABLE_COLUMNS}")
        for cell in cells:
            total += len(_plain_text(cell))
    if total > MAX_TABLE_CHARS:
        errors.append(f"{loc}: table {total} > {MAX_TABLE_CHARS} chars")
    return errors


def _validate_actions(block: dict[str, Any], loc: str) -> list[str]:
    errors: list[str] = []
    for element in block.get("elements") or []:
        if not isinstance(element, dict) or element.get("type") != "button":
            continue
        if len(_plain_text(element.get("text"))) > MAX_BUTTON_TEXT:
            errors.append(f"{loc}: button text > {MAX_BUTTON_TEXT} chars")
        if len(str(element.get("value", ""))) > MAX_BUTTON_VALUE:
            errors.append(f"{loc}: button value > {MAX_BUTTON_VALUE} chars")
    return errors


def _main() -> int:
    parser = argparse.ArgumentParser(description="Validate a Slack Block Kit payload.")
    parser.add_argument("path", nargs="?", help="JSON file of blocks (or stdin)")
    parser.add_argument(
        "--surface", default="message", choices=["message", "modal", "home"]
    )
    args = parser.parse_args()
    if args.path:
        with open(args.path, encoding="utf-8") as handle:
            raw = handle.read()
    else:
        raw = sys.stdin.read()
    payload = json.loads(raw)
    blocks = payload.get("blocks", payload) if isinstance(payload, dict) else payload
    errors = validate_blocks(blocks, surface=args.surface)
    if errors:
        print("INVALID:")
        for err in errors:
            print(f"  - {err}")
        return 1
    print("VALID")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
