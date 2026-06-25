"""Tests for the capability profiler (HIG-295 Step A)."""

from __future__ import annotations

import json
import uuid
from typing import Any
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


# --------------------------------------------------------------------------- #
# Chunked batching tests                                                        #
# --------------------------------------------------------------------------- #


def test_build_capability_profile_batches_all_tools() -> None:
    """A toolkit with > _PROFILER_BATCH_SIZE cards gets ALL tools enriched."""
    from kortny.integration_learning.profiles import _PROFILER_BATCH_SIZE

    inst_id = uuid.uuid4()
    task_id = uuid.uuid4()
    n_tools = _PROFILER_BATCH_SIZE * 2 + 10  # > 2 full batches

    # Build mock cards
    def _card(i: int) -> MagicMock:
        c = MagicMock()
        c.tool_slug = f"TOOL_{i}"
        c.name = f"Tool {i}"
        c.description = f"raw description {i}"
        c.side_effect = "read"
        c.input_schema_json = {}
        return c

    all_cards = [_card(i) for i in range(n_tools)]

    session = MagicMock()
    session.execute.return_value.all.return_value = all_cards
    session.scalars.return_value.first.return_value = None

    # LLM returns per_tool for whatever batch it receives
    call_count = 0

    def fake_complete(**kwargs: Any) -> MagicMock:
        nonlocal call_count
        # Parse the batch slugs out of the user message
        user_content = kwargs["messages"][1].content
        payload = json.loads(user_content)
        batch_tools = payload["tools"]
        per_tool = [
            {
                "tool_slug": t["tool_slug"],
                "enriched_description": f"Enriched: {t['tool_slug']}",
            }
            for t in batch_tools
        ]
        resp = MagicMock()
        resp.content = json.dumps(
            {
                "summary": "Test toolkit summary." if call_count == 0 else "ignored",
                "capability_buckets": ["bucket_a"] if call_count == 0 else [],
                "per_tool": per_tool,
                "cross_app_affinity_hints": [],
            }
        )
        call_count += 1
        return resp

    llm = MagicMock()
    llm.complete.side_effect = fake_complete

    result = build_capability_profile(
        session,
        installation_id=inst_id,
        toolkit_slug="bigapp",
        llm=llm,
        task_id=task_id,
    )

    assert result is not None
    # App-level fields come from the FIRST batch
    assert result.summary == "Test toolkit summary."
    assert "bucket_a" in result.capability_buckets

    # Multiple LLM calls happened (one per batch)
    expected_batches = -(-n_tools // _PROFILER_BATCH_SIZE)  # ceil division
    assert llm.complete.call_count == expected_batches

    # All n_tools got an enriched description written
    # session.execute was called once for the initial card query, then once per
    # tool for _write_enriched_descriptions — total > n_tools
    # We verify via session.flush being called (enriched_by_slug was non-empty)
    assert session.flush.called


def test_build_capability_profile_partial_batch_failure() -> None:
    """A batch whose LLM returns bad JSON is skipped; other batches still enrich."""
    from kortny.integration_learning.profiles import _PROFILER_BATCH_SIZE

    inst_id = uuid.uuid4()
    task_id = uuid.uuid4()
    n_tools = _PROFILER_BATCH_SIZE + 5  # two batches

    def _card(i: int) -> MagicMock:
        c = MagicMock()
        c.tool_slug = f"TOOL_{i}"
        c.name = f"Tool {i}"
        c.description = f"raw {i}"
        c.side_effect = "read"
        c.input_schema_json = {}
        return c

    all_cards = [_card(i) for i in range(n_tools)]
    session = MagicMock()
    session.execute.return_value.all.return_value = all_cards
    session.scalars.return_value.first.return_value = None

    call_count = 0

    def fake_complete(**kwargs: Any) -> MagicMock:
        nonlocal call_count
        resp = MagicMock()
        if call_count == 0:
            # First batch returns valid JSON
            user_content = kwargs["messages"][1].content
            payload = json.loads(user_content)
            per_tool = [
                {
                    "tool_slug": t["tool_slug"],
                    "enriched_description": f"Enriched: {t['tool_slug']}",
                }
                for t in payload["tools"]
            ]
            resp.content = json.dumps(
                {
                    "summary": "Good summary.",
                    "capability_buckets": ["b1"],
                    "per_tool": per_tool,
                    "cross_app_affinity_hints": [],
                }
            )
        else:
            # Second batch returns garbage
            resp.content = "not valid json at all"
        call_count += 1
        return resp

    llm = MagicMock()
    llm.complete.side_effect = fake_complete

    result = build_capability_profile(
        session,
        installation_id=inst_id,
        toolkit_slug="partialapp",
        llm=llm,
        task_id=task_id,
    )

    # Should succeed — first batch gave us app-level fields and some per_tool
    assert result is not None
    assert result.summary == "Good summary."
    # flush should have been called (at least one enriched description written)
    assert session.flush.called
    # Both LLM calls were attempted
    assert llm.complete.call_count == 2


def test_build_capability_profile_app_level_from_first_batch() -> None:
    """App-level summary comes from first batch, not later batches."""
    from kortny.integration_learning.profiles import _PROFILER_BATCH_SIZE

    inst_id = uuid.uuid4()
    task_id = uuid.uuid4()

    def _card(i: int) -> MagicMock:
        c = MagicMock()
        c.tool_slug = f"TOOL_{i}"
        c.name = f"Tool {i}"
        c.description = f"raw {i}"
        c.side_effect = "read"
        c.input_schema_json = {}
        return c

    all_cards = [_card(i) for i in range(_PROFILER_BATCH_SIZE + 1)]
    session = MagicMock()
    session.execute.return_value.all.return_value = all_cards
    session.scalars.return_value.first.return_value = None

    call_count = 0

    def fake_complete(**kwargs: Any) -> MagicMock:
        nonlocal call_count
        resp = MagicMock()
        user_content = kwargs["messages"][1].content
        payload = json.loads(user_content)
        per_tool = [
            {
                "tool_slug": t["tool_slug"],
                "enriched_description": f"Enriched: {t['tool_slug']}",
            }
            for t in payload["tools"]
        ]
        summary = "First batch summary." if call_count == 0 else "Second batch summary."
        call_count += 1
        resp.content = json.dumps(
            {
                "summary": summary,
                "capability_buckets": ["bucket"],
                "per_tool": per_tool,
                "cross_app_affinity_hints": [],
            }
        )
        return resp

    llm = MagicMock()
    llm.complete.side_effect = fake_complete

    result = build_capability_profile(
        session,
        installation_id=inst_id,
        toolkit_slug="firstbatchapp",
        llm=llm,
        task_id=task_id,
    )

    assert result is not None
    assert result.summary == "First batch summary."
    assert llm.complete.call_count == 2
