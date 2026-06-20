"""Tests for vision capability gating and image guards (HIG-279 Chunk B)."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from kortny.config.settings import Settings
from kortny.llm.litellm_catalog import LiteLLMModelCandidate, model_supports_vision
from kortny.llm.service import (
    ImageGuardError,
    VisionUnsupportedError,
    assert_vision_capable,
    enforce_image_guards,
)
from kortny.llm.types import ChatMessage, ImagePart

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SMALL_BYTES = b"\x89PNG\r\n"  # 6 bytes — well under any limit


def _img(
    *,
    mime: str = "image/png",
    size: int = 100,
    source: str = "test",
) -> ImagePart:
    return ImagePart(data=b"x" * size, mime=mime, source=source)


def _text_msg(content: str = "hello") -> ChatMessage:
    return ChatMessage(role="user", content=content)


def _image_msg(*images: ImagePart) -> ChatMessage:
    return ChatMessage(role="user", content="look at this", images=tuple(images))


def _base_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "SLACK_BOT_TOKEN": "xoxb-test",
        "SLACK_APP_TOKEN": "xapp-test",
        "SLACK_SIGNING_SECRET": "signing-secret",
        "LLM_PROVIDER": "openrouter",
        "LLM_API_KEY": "llm-key",
        "LLM_MODEL": "openai/gpt-4o",
        "COMPOSIO_API_KEY": "composio-key",
        "POSTGRES_URL": "postgresql://kortny:kortny@localhost/kortny",
    }
    values.update(overrides)
    return Settings.model_validate(values)


# ---------------------------------------------------------------------------
# model_supports_vision — catalog_row / capabilities_json branch
# ---------------------------------------------------------------------------


def test_model_supports_vision_returns_true_from_supports_vision_flag() -> None:
    catalog_row = {"supports_vision": True}
    assert model_supports_vision("openai", "gpt-4o", catalog_row=catalog_row) is True


def test_model_supports_vision_returns_true_from_input_modalities() -> None:
    catalog_row = {"input_modalities": ["text", "image"]}
    assert (
        model_supports_vision("openrouter", "openai/gpt-4o", catalog_row=catalog_row)
        is True
    )


def test_model_supports_vision_false_when_image_not_in_modalities() -> None:
    # supports_vision is absent and image is not in modalities
    catalog_row = {"input_modalities": ["text"]}

    # Patch out the live catalog lookups so the test is deterministic
    with (
        patch(
            "kortny.llm.litellm_catalog._local_model_candidate_for_identifier",
            return_value=None,
        ),
        patch(
            "kortny.llm.litellm_catalog._openrouter_model_candidate_for_identifier",
            return_value=None,
        ),
    ):
        result = model_supports_vision(
            "openrouter", "text-only/model", catalog_row=catalog_row
        )
    assert result is False


def test_model_supports_vision_false_when_catalog_row_is_none_and_model_unknown() -> (
    None
):
    with (
        patch(
            "kortny.llm.litellm_catalog._local_model_candidate_for_identifier",
            return_value=None,
        ),
        patch(
            "kortny.llm.litellm_catalog._openrouter_model_candidate_for_identifier",
            return_value=None,
        ),
    ):
        result = model_supports_vision("openai", "totally-unknown-model-xyz")
    assert result is False


# ---------------------------------------------------------------------------
# model_supports_vision — litellm local catalog branch
# ---------------------------------------------------------------------------


def test_model_supports_vision_returns_true_from_litellm_local_catalog() -> None:
    fake_candidate = LiteLLMModelCandidate(
        model_identifier="openai/gpt-4o",
        display_name="GPT-4o",
        provider_kind="openai",
        source="litellm_catalog",
        capabilities={"supports_vision": True},
        metadata={},
        input_price_per_mtok=None,
        output_price_per_mtok=None,
    )
    with patch(
        "kortny.llm.litellm_catalog._local_model_candidate_for_identifier",
        return_value=fake_candidate,
    ):
        result = model_supports_vision("openai", "openai/gpt-4o")
    assert result is True


def test_model_supports_vision_false_from_litellm_local_when_flag_missing() -> None:
    fake_candidate = LiteLLMModelCandidate(
        model_identifier="openai/gpt-3.5-turbo",
        display_name="GPT-3.5",
        provider_kind="openai",
        source="litellm_catalog",
        capabilities={},  # no supports_vision
        metadata={},
        input_price_per_mtok=None,
        output_price_per_mtok=None,
    )
    with (
        patch(
            "kortny.llm.litellm_catalog._local_model_candidate_for_identifier",
            return_value=fake_candidate,
        ),
        patch(
            "kortny.llm.litellm_catalog._openrouter_model_candidate_for_identifier",
            return_value=None,
        ),
    ):
        result = model_supports_vision("openai", "gpt-3.5-turbo")
    assert result is False


# ---------------------------------------------------------------------------
# model_supports_vision — OpenRouter architecture branch
# ---------------------------------------------------------------------------


def test_model_supports_vision_returns_true_from_openrouter_input_modalities() -> None:
    fake_or_candidate = LiteLLMModelCandidate(
        model_identifier="anthropic/claude-3-5-sonnet",
        display_name="Claude 3.5 Sonnet",
        provider_kind="openrouter",
        source="provider_api",
        capabilities={"input_modalities": ["text", "image"]},
        metadata={},
        input_price_per_mtok=None,
        output_price_per_mtok=None,
    )
    with (
        patch(
            "kortny.llm.litellm_catalog._local_model_candidate_for_identifier",
            return_value=None,
        ),
        patch(
            "kortny.llm.litellm_catalog._openrouter_model_candidate_for_identifier",
            return_value=fake_or_candidate,
        ),
    ):
        result = model_supports_vision("openrouter", "anthropic/claude-3-5-sonnet")
    assert result is True


def test_model_supports_vision_false_from_openrouter_when_image_absent() -> None:
    fake_or_candidate = LiteLLMModelCandidate(
        model_identifier="text-only/model",
        display_name="Text Only",
        provider_kind="openrouter",
        source="provider_api",
        capabilities={"input_modalities": ["text"]},
        metadata={},
        input_price_per_mtok=None,
        output_price_per_mtok=None,
    )
    with (
        patch(
            "kortny.llm.litellm_catalog._local_model_candidate_for_identifier",
            return_value=None,
        ),
        patch(
            "kortny.llm.litellm_catalog._openrouter_model_candidate_for_identifier",
            return_value=fake_or_candidate,
        ),
    ):
        result = model_supports_vision("openrouter", "text-only/model")
    assert result is False


def test_model_supports_vision_never_infers_from_model_name_string() -> None:
    # A model named "vision-model" must NOT be auto-approved from name alone.
    with (
        patch(
            "kortny.llm.litellm_catalog._local_model_candidate_for_identifier",
            return_value=None,
        ),
        patch(
            "kortny.llm.litellm_catalog._openrouter_model_candidate_for_identifier",
            return_value=None,
        ),
    ):
        result = model_supports_vision("openai", "vision-pro-ultra-max")
    assert result is False


# ---------------------------------------------------------------------------
# enforce_image_guards — pure helper tests (no DB, no provider)
# ---------------------------------------------------------------------------


def test_enforce_image_guards_passes_text_only_messages() -> None:
    settings = _base_settings(KORTNY_VISION_ENABLED="false")
    # Even with vision disabled, text-only requests must be unaffected.
    enforce_image_guards([_text_msg(), _text_msg("world")], settings)  # no raise


def test_enforce_image_guards_raises_when_vision_disabled() -> None:
    settings = _base_settings(KORTNY_VISION_ENABLED="false")
    with pytest.raises(ImageGuardError, match="disabled"):
        enforce_image_guards([_image_msg(_img())], settings)


def test_enforce_image_guards_raises_on_too_many_images() -> None:
    settings = _base_settings(KORTNY_VISION_MAX_IMAGES_PER_REQUEST="2")
    images = [_img(), _img(), _img()]  # 3 > max 2
    with pytest.raises(ImageGuardError, match="Too many images"):
        enforce_image_guards([_image_msg(*images)], settings)


def test_enforce_image_guards_passes_at_exact_limit() -> None:
    settings = _base_settings(KORTNY_VISION_MAX_IMAGES_PER_REQUEST="2")
    images = [_img(), _img()]  # exactly 2 — should pass
    enforce_image_guards([_image_msg(*images)], settings)  # no raise


def test_enforce_image_guards_raises_on_oversized_single_image() -> None:
    settings = _base_settings(KORTNY_VISION_MAX_IMAGE_BYTES="100")
    big_image = _img(size=101)
    with pytest.raises(ImageGuardError, match="too large"):
        enforce_image_guards([_image_msg(big_image)], settings)


def test_enforce_image_guards_raises_on_total_bytes_exceeded() -> None:
    settings = _base_settings(
        KORTNY_VISION_MAX_IMAGE_BYTES="200",
        KORTNY_VISION_MAX_TOTAL_IMAGE_BYTES="250",
    )
    # Each image is 150 bytes — under the per-image limit, but 300 > total 250.
    images = [_img(size=150), _img(size=150)]
    with pytest.raises(ImageGuardError, match="total size limit"):
        enforce_image_guards([_image_msg(*images)], settings)


def test_enforce_image_guards_raises_on_disallowed_mime() -> None:
    settings = _base_settings(KORTNY_VISION_ALLOWED_IMAGE_MIMES="image/png,image/jpeg")
    gif_image = _img(mime="image/gif")
    with pytest.raises(ImageGuardError, match="image/gif"):
        enforce_image_guards([_image_msg(gif_image)], settings)


def test_enforce_image_guards_passes_for_allowed_mimes() -> None:
    settings = _base_settings(
        KORTNY_VISION_ALLOWED_IMAGE_MIMES="image/png,image/jpeg,image/webp"
    )
    msgs = [
        _image_msg(_img(mime="image/png")),
        _image_msg(_img(mime="image/jpeg")),
        _image_msg(_img(mime="image/webp")),
    ]
    enforce_image_guards(msgs, settings)  # no raise


# ---------------------------------------------------------------------------
# assert_vision_capable — capability check helper
# ---------------------------------------------------------------------------


def test_assert_vision_capable_passes_for_vision_model() -> None:
    with patch(
        "kortny.llm.service.model_supports_vision",
        return_value=True,
    ):
        assert_vision_capable("openai", "gpt-4o")  # no raise


def test_assert_vision_capable_raises_for_text_only_model() -> None:
    with (
        patch(
            "kortny.llm.service.model_supports_vision",
            return_value=False,
        ),
        pytest.raises(VisionUnsupportedError, match="vision-capable"),
    ):
        assert_vision_capable("openai", "gpt-3.5-turbo")


def test_assert_vision_capable_passes_catalog_row_through() -> None:
    captured: dict[str, object] = {}

    def spy(
        provider_kind: str,
        model_identifier: str,
        *,
        catalog_row: object = None,
    ) -> bool:
        captured["catalog_row"] = catalog_row
        return True

    with patch("kortny.llm.service.model_supports_vision", side_effect=spy):
        stub_row = {"supports_vision": True}
        assert_vision_capable("openai", "gpt-4o", catalog_row=stub_row)

    assert captured["catalog_row"] is stub_row
