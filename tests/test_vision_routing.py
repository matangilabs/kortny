"""HIG-279 slice 2C: vision model tier and deterministic image-attachment routing.

Tests cover:
- ``_model_for_tier(ModelRouteTier.vision)`` resolves ``llm_vision_model`` when set,
  or falls back to ``llm_model`` when unset.
- ``route_for_task`` routes to the vision tier when the task input contains an
  image ``<slack_files>`` entry, regardless of any intent-decision events.
- ``route_for_task`` for a text-only task falls through to the existing routing
  (behaviour is byte-identical to before this slice was added).
- ``parse_image_attachment_pairs`` returns image pairs from a ``<slack_files>``
  block and an empty list for text-only input.
"""

from __future__ import annotations

from kortny.agent.attachment_parsing import parse_image_attachment_pairs
from kortny.config.settings import LLMProvider as SettingsLLMProvider
from kortny.config.settings import Settings
from kortny.db.models import Task, TaskEvent, TaskEventType
from kortny.llm import ModelRouter, ModelRouteTier

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SLACK_FILES_WITH_IMAGE = """Here is the image.

<slack_files>
- id: FIMG001
  name: screenshot.png
  mimetype: image/png
  size: 12345
</slack_files>
"""

_SLACK_FILES_TEXT_ONLY = """Here is a PDF.

<slack_files>
- id: FPDF001
  name: report.pdf
  mimetype: application/pdf
  size: 98765
</slack_files>
"""

_SLACK_FILES_MIXED = """Image and doc.

<slack_files>
- id: FIMG002
  name: chart.jpg
  mimetype: image/jpeg
  size: 54321
- id: FPDF002
  name: slides.pdf
  mimetype: application/pdf
  size: 11111
</slack_files>
"""

_TEXT_ONLY = "What is the capital of France?"


def _make_settings(*, vision_model: str | None = None) -> Settings:
    data: dict[str, object] = {
        "SLACK_BOT_TOKEN": "xoxb-test",
        "SLACK_APP_TOKEN": "xapp-test",
        "SLACK_SIGNING_SECRET": "signing-secret",
        "LLM_PROVIDER": SettingsLLMProvider.openrouter,
        "LLM_API_KEY": "openrouter-key",
        "LLM_MODEL": "openai/default",
        "LLM_CHEAP_MODEL": "anthropic/haiku",
        "LLM_STANDARD_MODEL": "openai/standard",
        "LLM_ANALYSIS_MODEL": "anthropic/sonnet",
        "LLM_DOCUMENT_MODEL": "openai/document",
        "LLM_HIGH_REASONING_MODEL": "openai/reasoning",
        "LLM_HUMANIZER_MODEL": "anthropic/humanizer",
        "POSTGRES_URL": "postgresql://kortny:kortny@localhost/kortny",
    }
    if vision_model is not None:
        data["LLM_VISION_MODEL"] = vision_model
    return Settings.model_validate(data)


# ---------------------------------------------------------------------------
# parse_image_attachment_pairs
# ---------------------------------------------------------------------------


def test_parse_image_attachment_pairs_returns_image_pair() -> None:
    pairs = parse_image_attachment_pairs(_SLACK_FILES_WITH_IMAGE)
    assert pairs == [("FIMG001", "image/png")]


def test_parse_image_attachment_pairs_skips_non_image_mimes() -> None:
    pairs = parse_image_attachment_pairs(_SLACK_FILES_TEXT_ONLY)
    assert pairs == []


def test_parse_image_attachment_pairs_returns_only_image_from_mixed_block() -> None:
    pairs = parse_image_attachment_pairs(_SLACK_FILES_MIXED)
    assert pairs == [("FIMG002", "image/jpeg")]


def test_parse_image_attachment_pairs_empty_for_text_only() -> None:
    pairs = parse_image_attachment_pairs(_TEXT_ONLY)
    assert pairs == []


def test_parse_image_attachment_pairs_empty_for_empty_string() -> None:
    pairs = parse_image_attachment_pairs("")
    assert pairs == []


# ---------------------------------------------------------------------------
# _model_for_tier(ModelRouteTier.vision)
# ---------------------------------------------------------------------------


def test_model_for_vision_tier_returns_vision_model_when_set() -> None:
    settings = _make_settings(vision_model="anthropic/claude-opus-4-5")
    router = ModelRouter(settings)
    model = router._model_for_tier(ModelRouteTier.vision)
    assert model == "anthropic/claude-opus-4-5"


def test_model_for_vision_tier_falls_back_to_llm_model_when_unset() -> None:
    settings = _make_settings(vision_model=None)
    router = ModelRouter(settings)
    model = router._model_for_tier(ModelRouteTier.vision)
    assert model == "openai/default"


def test_model_for_vision_tier_blank_string_treated_as_none() -> None:
    # Settings strips blank optional model strings to None, so a blank value
    # should fall back to llm_model just like an unset value.
    settings = Settings.model_validate(
        {
            "SLACK_BOT_TOKEN": "xoxb-test",
            "SLACK_APP_TOKEN": "xapp-test",
            "SLACK_SIGNING_SECRET": "signing-secret",
            "LLM_PROVIDER": SettingsLLMProvider.openrouter,
            "LLM_API_KEY": "openrouter-key",
            "LLM_MODEL": "openai/default",
            "LLM_VISION_MODEL": "   ",
            "POSTGRES_URL": "postgresql://kortny:kortny@localhost/kortny",
        }
    )
    router = ModelRouter(settings)
    assert router._model_for_tier(ModelRouteTier.vision) == "openai/default"


# ---------------------------------------------------------------------------
# route_for_task: vision branch
# ---------------------------------------------------------------------------


def test_route_for_task_image_task_routes_to_vision_tier() -> None:
    settings = _make_settings(vision_model="anthropic/claude-opus-4-5")
    router = ModelRouter(settings)
    task = Task(input=_SLACK_FILES_WITH_IMAGE)

    route = router.route_for_task(task)

    assert route.tier is ModelRouteTier.vision
    assert route.model == "anthropic/claude-opus-4-5"
    assert "vision" in route.reason
    assert "HIG-279" in route.reason


def test_route_for_task_image_task_vision_reason_string() -> None:
    """The reason must describe the image-routing decision clearly."""
    router = ModelRouter(_make_settings(vision_model="openai/gpt-4o"))
    task = Task(input=_SLACK_FILES_WITH_IMAGE)
    route = router.route_for_task(task)
    assert route.reason == "image attachment -> vision tier (HIG-279)"


def test_route_for_task_image_task_ignores_intent_events() -> None:
    """Vision routing fires BEFORE intent-decision events; any intent is overridden."""
    router = ModelRouter(_make_settings(vision_model="anthropic/claude-opus-4-5"))
    task = Task(input=_SLACK_FILES_WITH_IMAGE)
    # Even a "strong" intent event with no document hints should not prevent
    # the vision branch from winning.
    event = TaskEvent(
        seq=1,
        type=TaskEventType.log,
        payload={
            "message": "intent_classification_completed",
            "decision": {
                "classification": "task_request",
                "model_tier": "strong",
                "likely_tools": [],
            },
        },
    )
    route = router.route_for_task(task, events=[event])
    assert route.tier is ModelRouteTier.vision


def test_route_for_task_mixed_files_routes_to_vision() -> None:
    """A block with both image and non-image files still routes to vision."""
    router = ModelRouter(_make_settings(vision_model="anthropic/claude-opus-4-5"))
    task = Task(input=_SLACK_FILES_MIXED)
    route = router.route_for_task(task)
    assert route.tier is ModelRouteTier.vision


def test_route_for_task_image_fallback_to_llm_model_when_vision_unset() -> None:
    """When LLM_VISION_MODEL is unset, image tasks fall back to LLM_MODEL."""
    settings = _make_settings(vision_model=None)
    router = ModelRouter(settings)
    task = Task(input=_SLACK_FILES_WITH_IMAGE)
    route = router.route_for_task(task)
    assert route.tier is ModelRouteTier.vision
    assert route.model == "openai/default"


# ---------------------------------------------------------------------------
# route_for_task: text-only tasks are UNCHANGED
# ---------------------------------------------------------------------------


def test_route_for_task_text_only_standard_unchanged() -> None:
    """Short text-only tasks still route to standard (existing behaviour)."""
    settings = _make_settings(vision_model="anthropic/claude-opus-4-5")
    router = ModelRouter(settings)
    task = Task(input="what can you do?")
    route = router.route_for_task(task)
    assert route.tier is ModelRouteTier.standard


def test_route_for_task_text_only_analysis_unchanged() -> None:
    """Text tasks with analysis keywords still route to analysis tier."""
    settings = _make_settings(vision_model="anthropic/claude-opus-4-5")
    router = ModelRouter(settings)
    task = Task(input="research recent market trends")
    route = router.route_for_task(task)
    assert route.tier is ModelRouteTier.analysis


def test_route_for_task_text_only_document_unchanged() -> None:
    """Text tasks with document keywords still route to the document tier."""
    settings = _make_settings(vision_model="anthropic/claude-opus-4-5")
    router = ModelRouter(settings)
    task = Task(input="make a 3 page PDF report")
    route = router.route_for_task(task)
    assert route.tier is ModelRouteTier.document


def test_route_for_task_pdf_only_slack_file_not_vision() -> None:
    """A task with a PDF (non-image) Slack file does NOT route to vision."""
    settings = _make_settings(vision_model="anthropic/claude-opus-4-5")
    router = ModelRouter(settings)
    task = Task(input=_SLACK_FILES_TEXT_ONLY)
    route = router.route_for_task(task)
    # Must NOT be vision — the task_input_fallback for a PDF keyword-free short
    # input ends up at standard or analysis; the key invariant is not vision.
    assert route.tier is not ModelRouteTier.vision


def test_route_for_task_intent_classifier_still_wins_for_text_only() -> None:
    """For text-only tasks the intent classifier still wins over text-fallback."""
    settings = _make_settings(vision_model="anthropic/claude-opus-4-5")
    router = ModelRouter(settings)
    task = Task(input="put together a document")
    event = TaskEvent(
        seq=1,
        type=TaskEventType.log,
        payload={
            "message": "intent_classification_completed",
            "decision": {
                "classification": "task_request",
                "model_tier": "strong",
                "likely_tools": ["document_generation"],
            },
        },
    )
    route = router.route_for_task(task, events=[event])
    # Intent-driven document routing still wins for text-only tasks.
    assert route.tier is ModelRouteTier.document
    assert route.reason == "intent_classifier"
