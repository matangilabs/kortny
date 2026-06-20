"""Shared ChatMessage -> provider content serialization (HIG-279).

Text-only messages serialize to a plain string (byte-identical to before, so
prompt-cache prefixes are unchanged). Messages with images serialize to an
OpenAI/Anthropic-compatible content-blocks array, with base64 encoding done
HERE at the provider boundary (never stored on ChatMessage).
"""

from __future__ import annotations

import base64
from typing import Any

from kortny.llm.types import ChatMessage


def content_payload(message: ChatMessage) -> str | list[dict[str, Any]] | None:
    if not message.images:
        return message.content
    blocks: list[dict[str, Any]] = []
    if message.content:
        blocks.append({"type": "text", "text": message.content})
    for image in message.images:
        encoded = base64.b64encode(image.data).decode("ascii")
        blocks.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{image.mime};base64,{encoded}"},
            }
        )
    return blocks
