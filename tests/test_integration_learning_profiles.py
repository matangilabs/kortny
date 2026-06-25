"""Tests for the capability profiler (HIG-295 Step A)."""

from __future__ import annotations

import json
import uuid
from unittest.mock import MagicMock

from kortny.integration_learning.profiles import (
    _parse_profile,
    build_capability_profile,
)

# --------------------------------------------------------------------------- #
# _parse_profile unit tests                                                    #
# --------------------------------------------------------------------------- #


def test_parse_profile_valid() -> None:
    content = json.dumps(
        {
            "summary": "Twelve Data provides real-time and historical market data.",
            "capability_buckets": ["historical OHLCV bars", "technical indicators"],
            "per_tool": [
                {
                    "tool_slug": "GET_ATR",
                    "enriched_description": "Fetch Average True Range (ATR) indicator values for a symbol.",
                },
            ],
            "cross_app_affinity_hints": ["pairs well with Alpaca"],
        }
    )
    profile = _parse_profile(content, toolkit_slug="twelve_data")
    assert profile is not None
    assert profile.summary.startswith("Twelve Data")
    assert "historical OHLCV bars" in profile.capability_buckets
    assert profile.per_tool[0]["tool_slug"] == "GET_ATR"
    assert "Average True Range" in profile.per_tool[0]["enriched_description"]


def test_parse_profile_bad_json() -> None:
    assert _parse_profile("not json", toolkit_slug="foo") is None


def test_parse_profile_empty_summary() -> None:
    content = json.dumps(
        {
            "summary": "",
            "capability_buckets": [],
            "per_tool": [],
            "cross_app_affinity_hints": [],
        }
    )
    assert _parse_profile(content, toolkit_slug="foo") is None


# --------------------------------------------------------------------------- #
# build_capability_profile with mocked LLM (no DB)                            #
# --------------------------------------------------------------------------- #


def _make_mock_llm(profile_json: str) -> MagicMock:
    completion = MagicMock()
    completion.content = profile_json
    llm = MagicMock()
    llm.complete.return_value = completion
    return llm


def test_build_capability_profile_no_cards_returns_none() -> None:
    """When there are no cards, build_capability_profile returns None without calling LLM."""
    session = MagicMock()
    session.execute.return_value.all.return_value = []
    llm = MagicMock()

    result = build_capability_profile(
        session,
        installation_id=uuid.uuid4(),
        toolkit_slug="twelve_data",
        llm=llm,
        task_id=uuid.uuid4(),
    )
    assert result is None
    llm.complete.assert_not_called()


def test_build_capability_profile_writes_enriched_description() -> None:
    """Profile pass writes enriched_description and returns parsed profile."""
    inst_id = uuid.uuid4()
    task_id = uuid.uuid4()

    profile_json = json.dumps(
        {
            "summary": "Twelve Data provides real-time financial market data.",
            "capability_buckets": ["technical indicators", "historical bars"],
            "per_tool": [
                {
                    "tool_slug": "GET_ATR",
                    "enriched_description": "Fetch Average True Range (ATR) values for a ticker symbol.",
                },
                {
                    "tool_slug": "GET_EOD",
                    "enriched_description": "Retrieve end-of-day OHLCV price data for a symbol.",
                },
            ],
            "cross_app_affinity_hints": [],
        }
    )

    # Build mock card rows
    card1 = MagicMock()
    card1.tool_slug = "GET_ATR"
    card1.name = "Get ATR"
    card1.description = "Get technical indicator"
    card1.side_effect = "read"
    card1.input_schema_json = {"properties": {"symbol": {}}, "required": ["symbol"]}

    card2 = MagicMock()
    card2.tool_slug = "GET_EOD"
    card2.name = "Get EOD"
    card2.description = "Get end of day data"
    card2.side_effect = "read"
    card2.input_schema_json = {}

    session = MagicMock()
    session.execute.return_value.all.return_value = [card1, card2]
    session.scalars.return_value.first.return_value = None  # no existing KG entity

    llm = _make_mock_llm(profile_json)

    result = build_capability_profile(
        session,
        installation_id=inst_id,
        toolkit_slug="twelve_data",
        llm=llm,
        task_id=task_id,
    )

    assert result is not None
    assert result.summary == "Twelve Data provides real-time financial market data."
    assert "technical indicators" in result.capability_buckets
    assert len(result.per_tool) == 2
    assert result.per_tool[0]["tool_slug"] == "GET_ATR"
    assert "Average True Range" in result.per_tool[0]["enriched_description"]

    # session.flush should have been called
    assert session.flush.called


# --------------------------------------------------------------------------- #
# tool_card_embedding_text prefers enriched_description                        #
# --------------------------------------------------------------------------- #


def test_tool_card_embedding_text_prefers_enriched() -> None:
    """tool_card_embedding_text uses enriched_description when present."""
    from kortny.tool_selection.budgeting import tool_card_embedding_text
    from kortny.tool_selection.models import ToolCard

    card = ToolCard(
        registry_name="composio__twelve_data__GET_ATR",
        provider="composio",
        toolkit_slug="twelve_data",
        display_name="Get ATR",
        description="Get technical indicator",
        capabilities=("read",),
        side_effect="read",
        enriched_description="Fetch Average True Range (ATR) indicator values for a symbol.",
    )
    text = tool_card_embedding_text(card)
    assert "Average True Range" in text
    assert "ATR" in text
    # The raw noisy description should NOT appear
    assert "Get technical indicator" not in text


def test_tool_card_embedding_text_falls_back_to_description() -> None:
    """tool_card_embedding_text falls back to raw description when no enriched."""
    from kortny.tool_selection.budgeting import tool_card_embedding_text
    from kortny.tool_selection.models import ToolCard

    card = ToolCard(
        registry_name="composio__twelve_data__GET_ATR",
        provider="composio",
        toolkit_slug="twelve_data",
        display_name="Get ATR",
        description="Get technical indicator data",
        capabilities=("read",),
        side_effect="read",
    )
    text = tool_card_embedding_text(card)
    assert "Get technical indicator data" in text
