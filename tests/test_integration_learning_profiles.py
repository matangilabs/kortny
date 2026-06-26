"""Tests for the capability profiler (HIG-295 Step A + B)."""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any
from unittest.mock import MagicMock

import pytest

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


# --------------------------------------------------------------------------- #
# max_tools / only-unenriched behavior (HIG-295 Part 2)                       #
# --------------------------------------------------------------------------- #


def _make_card(slug: str, enriched: str | None = None) -> MagicMock:
    """Build a mock card row with optional enriched_description."""
    c = MagicMock()
    c.tool_slug = slug
    c.name = slug.replace("_", " ").title()
    c.description = f"raw description for {slug}"
    c.side_effect = "read"
    c.input_schema_json = {}
    c.card_sha = f"sha_{slug}"
    c.enriched_description = enriched
    return c


def _make_profile_llm(slug_prefix: str = "") -> MagicMock:
    """LLM that returns valid profile JSON, enriching whatever slugs it receives."""

    def fake_complete(**kwargs: Any) -> MagicMock:
        user_content = kwargs["messages"][1].content
        payload = json.loads(user_content)
        per_tool = [
            {
                "tool_slug": t["tool_slug"],
                "enriched_description": f"Enriched: {t['tool_slug']}",
            }
            for t in payload["tools"]
        ]
        resp = MagicMock()
        resp.content = json.dumps(
            {
                "summary": "Test toolkit summary.",
                "capability_buckets": ["bucket_a"],
                "per_tool": per_tool,
                "cross_app_affinity_hints": [],
            }
        )
        return resp

    llm = MagicMock()
    llm.complete.side_effect = fake_complete
    return llm


def test_max_tools_skips_already_enriched_cards() -> None:
    """max_tools mode: already-enriched cards are not re-processed."""
    inst_id = uuid.uuid4()
    task_id = uuid.uuid4()

    enriched_card = _make_card("TOOL_ENRICHED", enriched="Already enriched desc.")
    unenriched_card = _make_card("TOOL_UNENRICHED", enriched=None)

    session = MagicMock()
    session.execute.return_value.all.return_value = [enriched_card, unenriched_card]
    session.scalars.return_value.first.return_value = None

    llm = _make_profile_llm()

    result = build_capability_profile(
        session,
        installation_id=inst_id,
        toolkit_slug="myapp",
        llm=llm,
        task_id=task_id,
        max_tools=10,
    )

    assert result is not None
    # Only 1 LLM call (one unenriched card fits in one batch).
    assert llm.complete.call_count == 1
    # The LLM was passed only the unenriched card.
    call_args = llm.complete.call_args
    user_content = call_args.kwargs["messages"][1].content
    payload = json.loads(user_content)
    slugs_sent = [t["tool_slug"] for t in payload["tools"]]
    assert "TOOL_UNENRICHED" in slugs_sent
    assert "TOOL_ENRICHED" not in slugs_sent


def test_max_tools_caps_number_processed() -> None:
    """max_tools=2 processes at most 2 unenriched cards even if more exist."""
    inst_id = uuid.uuid4()
    task_id = uuid.uuid4()

    cards = [_make_card(f"TOOL_{i}", enriched=None) for i in range(5)]

    session = MagicMock()
    session.execute.return_value.all.return_value = cards
    session.scalars.return_value.first.return_value = None

    processed_slugs: list[str] = []

    def fake_complete(**kwargs: Any) -> MagicMock:
        user_content = kwargs["messages"][1].content
        payload = json.loads(user_content)
        for t in payload["tools"]:
            processed_slugs.append(t["tool_slug"])
        per_tool = [
            {
                "tool_slug": t["tool_slug"],
                "enriched_description": f"Enriched: {t['tool_slug']}",
            }
            for t in payload["tools"]
        ]
        resp = MagicMock()
        resp.content = json.dumps(
            {
                "summary": "Summary.",
                "capability_buckets": [],
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
        toolkit_slug="myapp",
        llm=llm,
        task_id=task_id,
        max_tools=2,
    )

    assert result is not None
    # Only the first 2 unenriched cards were processed.
    assert len(processed_slugs) == 2


def test_max_tools_stamps_digest_when_all_enriched() -> None:
    """When max_tools finishes the last unenriched cards, digest is stamped on KG entity."""
    inst_id = uuid.uuid4()
    task_id = uuid.uuid4()

    # Only 1 unenriched card; max_tools=10 covers it.
    unenriched = _make_card("TOOL_A", enriched=None)
    all_cards = [unenriched]

    existing_entity = MagicMock()
    existing_entity.attrs_json = {"kind": "capability_profile", "summary": "old"}

    session = MagicMock()
    session.execute.return_value.all.return_value = all_cards

    # scalars calls: first for KG entity check during profile, then for stamp.
    entity_result_1 = MagicMock()
    entity_result_1.first.return_value = existing_entity  # entity exists
    entity_result_2 = MagicMock()
    entity_result_2.first.return_value = existing_entity  # for stamp re-fetch

    session.scalars.side_effect = [entity_result_1, entity_result_2]

    llm = _make_profile_llm()

    result = build_capability_profile(
        session,
        installation_id=inst_id,
        toolkit_slug="myapp",
        llm=llm,
        task_id=task_id,
        max_tools=10,
    )

    assert result is not None
    # Digest should have been stamped: attrs_json was mutated on existing_entity.
    # session.flush should have been called (for the stamp).
    assert session.flush.called
    # The entity's attrs_json should now contain generated_from.
    final_attrs = existing_entity.attrs_json
    assert isinstance(final_attrs, dict)
    assert "generated_from" in final_attrs
    assert "card_sha_digest" in final_attrs["generated_from"]

    # The digest value should match SHA256 of sorted card_shas.
    expected_digest = hashlib.sha256(unenriched.card_sha.encode()).hexdigest()
    assert final_attrs["generated_from"]["card_sha_digest"] == expected_digest


def test_max_tools_no_digest_stamp_when_cards_remain() -> None:
    """Digest is NOT stamped when more unenriched cards remain after this call."""
    inst_id = uuid.uuid4()
    task_id = uuid.uuid4()

    # 3 unenriched cards, budget = 1 — 2 will remain.
    cards = [_make_card(f"TOOL_{i}", enriched=None) for i in range(3)]

    existing_entity = MagicMock()
    existing_entity.attrs_json = {}

    session = MagicMock()
    session.execute.return_value.all.return_value = cards

    entity_result = MagicMock()
    entity_result.first.return_value = existing_entity
    session.scalars.return_value = entity_result

    llm = _make_profile_llm()

    result = build_capability_profile(
        session,
        installation_id=inst_id,
        toolkit_slug="myapp",
        llm=llm,
        task_id=task_id,
        max_tools=1,
    )

    assert result is not None
    # attrs_json should NOT have had generated_from stamped.
    final_attrs = existing_entity.attrs_json
    assert "generated_from" not in final_attrs


def test_max_tools_kg_entity_written_only_on_first_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In max_tools mode, KG entity summary is written only when entity doesn't exist."""
    inst_id = uuid.uuid4()
    task_id = uuid.uuid4()

    unenriched = _make_card("TOOL_A", enriched=None)

    session = MagicMock()
    session.execute.return_value.all.return_value = [unenriched]

    # Entity already exists.
    existing_entity = MagicMock()
    existing_entity.attrs_json = {
        "kind": "capability_profile",
        "summary": "existing summary",
    }

    entity_result = MagicMock()
    entity_result.first.return_value = existing_entity
    session.scalars.return_value = entity_result

    llm = _make_profile_llm()

    # Patch _upsert_kg_profile_entity to track calls via MonkeyPatch.
    upsert_calls: list[bool] = []
    import kortny.integration_learning.profiles as profiles_module

    def tracking_upsert(*args: object, **kwargs: object) -> None:
        upsert_calls.append(True)

    monkeypatch.setattr(profiles_module, "_upsert_kg_profile_entity", tracking_upsert)

    build_capability_profile(
        session,
        installation_id=inst_id,
        toolkit_slug="myapp",
        llm=llm,
        task_id=task_id,
        max_tools=10,
    )

    # Entity exists → upsert should NOT have been called (skip on top-up run).
    assert not upsert_calls, "KG upsert should not run when entity already exists"


def test_no_max_tools_writes_kg_entity_always(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Legacy mode (max_tools=None) always writes the KG entity."""
    inst_id = uuid.uuid4()
    task_id = uuid.uuid4()

    card = _make_card("TOOL_A", enriched="Already enriched.")
    card2 = _make_card("TOOL_B", enriched=None)

    session = MagicMock()
    session.execute.return_value.all.return_value = [card, card2]
    session.scalars.return_value.first.return_value = None

    llm = _make_profile_llm()

    upsert_calls: list[bool] = []
    import kortny.integration_learning.profiles as profiles_module

    def tracking_upsert(*args: object, **kwargs: object) -> None:
        upsert_calls.append(True)
        # Don't call original to avoid GraphService mock complexity.

    monkeypatch.setattr(profiles_module, "_upsert_kg_profile_entity", tracking_upsert)

    result = build_capability_profile(
        session,
        installation_id=inst_id,
        toolkit_slug="myapp",
        llm=llm,
        task_id=task_id,
        # max_tools=None (legacy mode)
    )

    # Legacy mode (no max_tools) always calls upsert.
    assert upsert_calls, "KG upsert should always run in legacy mode"
    assert result is not None
