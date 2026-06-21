"""Unit tests for _build_revision_patch_proposer (HIG-244 slice 3b).

Pure unit tests — no DB, no Postgres required.

Patch targets and timing:
- All lazy imports in the builder run when the builder is called, NOT when the
  returned closure is called.  Therefore all mocks must be active when
  ``_build_revision_patch_proposer(context)`` is invoked:
  - ``kortny.llm.runtime_config.select_runtime_model``
  - ``kortny.llm.runtime_config.create_provider_for_selection``
  - ``kortny.llm.LLMService`` (captured as a closure var at builder time)
- The closure is then called outside those patches, using the already-captured
  mock objects from the builder scope.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from kortny.config.settings import Settings
from kortny.documents.critique import VisualIssue
from kortny.documents.revision import (
    LlmPatchContext,
    SetChartLabels,
    VisualRevisionPatch,
)
from kortny.tools.native_runtime import (
    NativeToolBuildContext,
    _build_revision_patch_proposer,
)

# Patch targets — lazy imports live in source modules, not in native_runtime's namespace.
_SEL = "kortny.llm.runtime_config.select_runtime_model"
_PROV = "kortny.llm.runtime_config.create_provider_for_selection"
_LLM = "kortny.llm.LLMService"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "SLACK_BOT_TOKEN": "xoxb-test",
        "SLACK_APP_TOKEN": "xapp-test",
        "SLACK_SIGNING_SECRET": "signing-secret",
        "LLM_PROVIDER": "openrouter",
        "LLM_API_KEY": "openrouter-key",
        "LLM_MODEL": "openai/gpt-4o-mini",
        "AGENT_RUNTIME": "custom",
        "KORTNY_WORKFLOW_BACKEND": "inline",
        "POSTGRES_URL": "postgresql://kortny:kortny@localhost/kortny",
        "KORTNY_EMBEDDINGS_BACKEND": "disabled",
        "COMPOSIO_API_KEY": "composio-key",
    }
    base.update(overrides)
    return Settings.model_validate(base)


def _make_context() -> NativeToolBuildContext:
    task = MagicMock()
    task.id = "task-abc"
    task.installation_id = "inst-abc"
    return NativeToolBuildContext(
        settings=_make_settings(),
        session=MagicMock(),
        task=task,
        task_service=MagicMock(),
        working_dir=Path("/tmp"),
        web_search_tool=None,
        slack_history_client=None,
        slack_file_client=None,
        slack_identity_client=None,
        slack_action_client=None,
        memory_service=MagicMock(),
    )


def _make_mock_selection(model_name: str = "gpt-4o-mini") -> MagicMock:
    selection = MagicMock()
    selection.model.model = model_name
    selection.model.provider_kind = "openai"
    selection.provider_name = "openai"
    selection.model_route = MagicMock()
    return selection


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_proposer_valid_llm_response_returns_patch() -> None:
    """A valid LLM JSON response is parsed and returned as a VisualRevisionPatch.

    All patches must be active during the builder call so that imports captured
    as closure variables point to mock objects.
    """
    context = _make_context()
    mock_selection = _make_mock_selection()
    mock_provider = MagicMock()

    valid_patch_json = json.dumps(
        {
            "base_spec_hash": "abc123",
            "operations": [
                {
                    "op": "set_chart_labels",
                    "block_index": 0,
                    "title": "Revenue",
                    "x_axis_label": "Quarter",
                    "y_axis_label": "Amount",
                    "series_names": None,
                }
            ],
            "rationale": "Added missing chart labels",
        }
    )

    mock_completion = MagicMock()
    mock_completion.content = valid_patch_json

    mock_llm_instance = MagicMock()
    mock_llm_instance.complete.return_value = mock_completion

    # All patches must cover the builder call so captured closure vars are mocks.
    with (
        patch(_SEL, return_value=mock_selection),
        patch(_PROV, return_value=mock_provider),
        patch(_LLM, return_value=mock_llm_instance),
    ):
        proposer = _build_revision_patch_proposer(context)
        assert proposer is not None

        issue = VisualIssue(
            category="labels",
            severity="warning",
            message="Missing chart labels",
            page=1,
        )
        ctx = LlmPatchContext(
            base_spec_hash="abc123",
            issues=[issue],
            candidates={0: {"type": "chart", "chart_type": "bar"}},
        )
        result = proposer(ctx)

    assert result is not None
    assert isinstance(result, VisualRevisionPatch)
    assert result.base_spec_hash == "abc123"
    assert len(result.operations) == 1
    op = result.operations[0]
    assert isinstance(op, SetChartLabels)
    assert op.title == "Revenue"


def test_proposer_invalid_json_returns_none() -> None:
    """When the LLM returns invalid JSON, the proposer returns None without raising."""
    context = _make_context()
    mock_selection = _make_mock_selection()
    mock_provider = MagicMock()

    mock_completion = MagicMock()
    mock_completion.content = "this is not json at all"

    mock_llm_instance = MagicMock()
    mock_llm_instance.complete.return_value = mock_completion

    with (
        patch(_SEL, return_value=mock_selection),
        patch(_PROV, return_value=mock_provider),
        patch(_LLM, return_value=mock_llm_instance),
    ):
        proposer = _build_revision_patch_proposer(context)
        assert proposer is not None

        issue = VisualIssue(
            category="labels",
            severity="warning",
            message="Missing labels",
            page=1,
        )
        ctx = LlmPatchContext(
            base_spec_hash="abc123",
            issues=[issue],
            candidates={},
        )
        result = proposer(ctx)

    assert result is None


def test_proposer_no_model_configured_returns_none() -> None:
    """When select_runtime_model raises, the builder returns None (no model configured)."""
    context = _make_context()

    with patch(_SEL, side_effect=Exception("no model configured")):
        proposer = _build_revision_patch_proposer(context)

    assert proposer is None


def test_proposer_create_provider_failure_returns_none() -> None:
    """When create_provider_for_selection raises, the builder returns None."""
    context = _make_context()
    mock_selection = _make_mock_selection()

    with (
        patch(_SEL, return_value=mock_selection),
        patch(_PROV, side_effect=Exception("provider error")),
    ):
        proposer = _build_revision_patch_proposer(context)

    assert proposer is None
