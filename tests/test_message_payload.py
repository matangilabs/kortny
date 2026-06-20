"""Tests for HIG-279 vision plumbing: ImagePart, content_payload, and provider integration."""

from __future__ import annotations

import base64
import json
from typing import Any

import httpx
import pytest

from kortny.llm.message_payload import content_payload
from kortny.llm.types import ChatMessage, ImagePart
from kortny.observability.content import render_chat_messages

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _png_bytes() -> bytes:
    """Minimal 1x1 PNG — valid but tiny; good enough for serialization tests."""
    return (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00"
        b"\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
    )


def _make_image(
    data: bytes | None = None,
    mime: str = "image/png",
    source: str = "slack_file:F123",
    alt: str | None = "a test image",
) -> ImagePart:
    return ImagePart(data=data or _png_bytes(), mime=mime, source=source, alt=alt)


# ---------------------------------------------------------------------------
# 1. content_payload — text-only messages
# ---------------------------------------------------------------------------


def test_content_payload_text_only_returns_plain_string() -> None:
    msg = ChatMessage(role="user", content="hello world")
    result = content_payload(msg)
    assert result == "hello world"
    assert result == msg.content


def test_content_payload_none_content_no_images_returns_none() -> None:
    msg = ChatMessage(role="assistant", content=None)
    result = content_payload(msg)
    assert result is None


def test_default_chat_message_has_empty_images() -> None:
    msg = ChatMessage(role="user", content="hi")
    assert msg.images == ()


# ---------------------------------------------------------------------------
# 2. content_payload — image messages
# ---------------------------------------------------------------------------


def test_content_payload_with_image_returns_blocks() -> None:
    img = _make_image()
    msg = ChatMessage(role="user", content="describe this", images=(img,))
    result = content_payload(msg)
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0] == {"type": "text", "text": "describe this"}
    image_block = result[1]
    assert image_block["type"] == "image_url"
    url = image_block["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    encoded_part = url[len("data:image/png;base64,") :]
    decoded = base64.b64decode(encoded_part)
    assert decoded == img.data


def test_content_payload_image_only_no_text_block() -> None:
    """When content is None/empty, no text block should appear."""
    img = _make_image()
    msg = ChatMessage(role="user", content=None, images=(img,))
    result = content_payload(msg)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["type"] == "image_url"


def test_content_payload_multiple_images() -> None:
    img1 = _make_image(source="slack_file:F001")
    img2 = _make_image(source="slack_file:F002")
    msg = ChatMessage(role="user", content="compare these", images=(img1, img2))
    result = content_payload(msg)
    assert isinstance(result, list)
    assert len(result) == 3
    assert result[0]["type"] == "text"
    assert result[1]["type"] == "image_url"
    assert result[2]["type"] == "image_url"


# ---------------------------------------------------------------------------
# 3. ImagePart properties
# ---------------------------------------------------------------------------


def test_image_part_byte_size() -> None:
    data = b"abc" * 10
    img = ImagePart(data=data, mime="image/jpeg", source="s3:bucket/key")
    assert img.byte_size == 30


def test_image_part_sha256_prefix_length_and_hex() -> None:
    img = _make_image()
    prefix = img.sha256_prefix
    assert len(prefix) == 12
    assert all(c in "0123456789abcdef" for c in prefix)


def test_image_part_sha256_prefix_deterministic() -> None:
    data = b"deterministic"
    img1 = ImagePart(data=data, mime="image/png", source="x")
    img2 = ImagePart(data=data, mime="image/png", source="x")
    assert img1.sha256_prefix == img2.sha256_prefix


# ---------------------------------------------------------------------------
# 4. OpenRouter provider parity tests
# ---------------------------------------------------------------------------


def test_openrouter_message_to_payload_text_only_parity() -> None:
    """_message_to_payload for text-only is byte-identical to the hand-written dict."""
    from kortny.llm.openrouter import _message_to_payload

    msg = ChatMessage(role="user", content="hello")
    result = _message_to_payload(msg)
    expected: dict[str, Any] = {"role": "user", "content": "hello"}
    assert result == expected


def test_openrouter_message_to_payload_image_message() -> None:
    from kortny.llm.openrouter import _message_to_payload

    img = _make_image()
    msg = ChatMessage(role="user", content="what is this?", images=(img,))
    result = _message_to_payload(msg)
    assert result["role"] == "user"
    content = result["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "what is this?"}
    url = content[1]["image_url"]["url"]
    decoded = base64.b64decode(url.split(",", 1)[1])
    assert decoded == img.data


def test_openrouter_provider_sends_image_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: an image message reaches the HTTP payload as content blocks."""
    captured_payloads: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_payloads.append(json.loads(request.read().decode()))
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "it is a test image"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )

    from kortny.llm import OpenRouterProvider

    provider = OpenRouterProvider(
        api_key="test-key",
        model="openai/gpt-4o",
        transport=httpx.MockTransport(handler),
    )
    img = _make_image()
    provider.complete([ChatMessage(role="user", content="describe", images=(img,))])

    messages = captured_payloads[0]["messages"]
    assert len(messages) == 1
    content = messages[0]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image_url"


# ---------------------------------------------------------------------------
# 5. LiteLLM provider parity tests
# ---------------------------------------------------------------------------


def test_litellm_message_to_payload_text_only_parity() -> None:
    from kortny.llm.litellm_provider import _message_to_payload

    msg = ChatMessage(role="system", content="be helpful")
    result = _message_to_payload(msg)
    expected: dict[str, Any] = {"role": "system", "content": "be helpful"}
    assert result == expected


def test_litellm_message_to_payload_image_message() -> None:
    from kortny.llm.litellm_provider import _message_to_payload

    img = _make_image()
    msg = ChatMessage(role="user", content="caption this", images=(img,))
    result = _message_to_payload(msg)
    assert result["role"] == "user"
    content = result["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "caption this"}
    url = content[1]["image_url"]["url"]
    decoded = base64.b64decode(url.split(",", 1)[1])
    assert decoded == img.data


# ---------------------------------------------------------------------------
# 6. Cache control — image_url blocks must NOT receive the cache marker
# ---------------------------------------------------------------------------


def test_cache_control_targets_text_block_not_image_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When content is [text, image_url], cache_control goes on the text block."""
    from kortny.llm.litellm_provider import _inject_cache_control

    # Build a pre-serialized message that looks like what content_payload produces
    # for a vision message (text + image_url).
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe this"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,abc123"},
                },
            ],
        }
    ]
    new_messages, _ = _inject_cache_control(messages, [])
    content = new_messages[0]["content"]
    assert isinstance(content, list)
    text_block = content[0]
    image_block = content[1]
    # Cache marker lands on the text block.
    assert text_block.get("cache_control") == {"type": "ephemeral"}
    # Image block must NOT get the cache marker.
    assert "cache_control" not in image_block


def test_cache_control_falls_back_to_last_block_when_no_text() -> None:
    """If a content list has only image_url blocks, fall back to last block."""
    from kortny.llm.litellm_provider import _inject_cache_control

    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,abc"},
                },
            ],
        }
    ]
    new_messages, _ = _inject_cache_control(messages, [])
    content = new_messages[0]["content"]
    # Fallback: marks the only (image_url) block.
    assert content[0].get("cache_control") == {"type": "ephemeral"}


# ---------------------------------------------------------------------------
# 7. Observability — images as metadata only, never bytes
# ---------------------------------------------------------------------------


def test_render_chat_messages_includes_image_metadata_in_full_mode() -> None:
    img = _make_image(alt="a cat")
    msg = ChatMessage(role="user", content="look at this", images=(img,))
    rendered = render_chat_messages([msg], "full")
    assert rendered is not None
    entry = rendered[0]
    assert "images" in entry
    image_meta = entry["images"][0]
    assert image_meta["mime"] == "image/png"
    assert image_meta["byte_size"] == img.byte_size
    assert image_meta["sha256_prefix"] == img.sha256_prefix
    assert image_meta["source"] == "slack_file:F123"
    assert image_meta["alt"] == "a cat"


def test_render_chat_messages_image_metadata_in_summaries_mode() -> None:
    img = _make_image(alt="a dog")
    msg = ChatMessage(role="user", content="look at this", images=(img,))
    rendered = render_chat_messages([msg], "summaries")
    assert rendered is not None
    entry = rendered[0]
    assert "images" in entry
    image_meta = entry["images"][0]
    assert "mime" in image_meta
    assert "byte_size" in image_meta
    assert "sha256_prefix" in image_meta


def test_render_chat_messages_never_contains_base64_or_raw_bytes() -> None:
    """Privacy invariant: no base64 or raw bytes in any capture mode."""
    img = _make_image()
    msg = ChatMessage(role="user", content="hi", images=(img,))
    b64_encoded = base64.b64encode(img.data).decode("ascii")

    for mode in ("full", "summaries"):
        rendered = render_chat_messages([msg], mode)
        assert rendered is not None
        serialized = json.dumps(rendered)
        assert b64_encoded not in serialized, f"base64 found in mode={mode!r}"
        # Raw bytes can't appear in JSON, but ensure the sha256 prefix is not
        # confused with the actual data blob.
        assert img.data.hex() not in serialized, f"hex bytes found in mode={mode!r}"


def test_render_chat_messages_no_images_key_when_no_images() -> None:
    msg = ChatMessage(role="user", content="plain text")
    for mode in ("full", "summaries"):
        rendered = render_chat_messages([msg], mode)
        assert rendered is not None
        assert "images" not in rendered[0]


def test_render_chat_messages_metadata_mode_returns_none() -> None:
    img = _make_image()
    msg = ChatMessage(role="user", content="hi", images=(img,))
    assert render_chat_messages([msg], "metadata") is None
