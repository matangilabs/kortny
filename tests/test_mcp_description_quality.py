"""Tests for MCP tool description quality scoring and enrichment (HIG-215).

Covers:
  - score_tool_description rubric (good ≥ 0.75; bare one-word < 0.5)
  - sha-gating: same description is not re-scored
  - enrichment stored and used by the provider card
  - discovery survives a raising enricher
  - DB-backed tests for the full upsert + quality pipeline
"""

from __future__ import annotations

import os
import sys
import uuid
from collections.abc import Iterator, Sequence
from decimal import Decimal
from pathlib import Path
from typing import cast

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, delete, select
from sqlalchemy.orm import Session

from kortny.dashboard.mcp_actions import upsert_discovered_tools
from kortny.dashboard.mcp_data import get_mcp_dashboard
from kortny.db.models import Installation, McpServer, McpServerTool, Task, TaskEvent
from kortny.db.session import make_engine, make_session_factory, normalize_database_url
from kortny.llm.types import ChatMessage, Completion, TokenUsage
from kortny.mcp.client import DiscoveredTool
from kortny.mcp.description_quality import (
    enrich_tool_description,
    score_tool_description,
    sha256_of_description,
)
from kortny.mcp.provider import McpExternalToolProvider
from kortny.tasks import TaskService
from kortny.tools.types import JsonObject, JsonSchema

ECHO_SERVER = str(Path(__file__).parent / "fixtures" / "mcp" / "echo_server.py")
ENCRYPTION_KEY = "mcp-test-encryption-key"

TEST_POSTGRES_URL = os.environ.get("KORTNY_TEST_POSTGRES_URL")

db_required = pytest.mark.skipif(
    TEST_POSTGRES_URL is None,
    reason="KORTNY_TEST_POSTGRES_URL is required for MCP DB tests",
)


# ---------------------------------------------------------------------------
# Fake LLM client for unit tests (no real LLM calls)
# ---------------------------------------------------------------------------


class FakeLLMClient:
    """Minimal fake LLMService for enrichment tests."""

    def __init__(
        self, response: str = "Enriched description.", raise_on_call: bool = False
    ) -> None:
        self.response = response
        self.raise_on_call = raise_on_call
        self.call_count = 0

    def complete(
        self,
        *,
        task_id: uuid.UUID,
        messages: Sequence[ChatMessage],
        tools: Sequence[JsonSchema] = (),
        response_format: JsonObject | None = None,
        prompt_name: str | None = None,
        prompt_source: str = "code",
    ) -> Completion:
        self.call_count += 1
        if self.raise_on_call:
            raise RuntimeError("Simulated LLM failure")
        return Completion(
            content=self.response,
            tool_calls=(),
            usage=TokenUsage(input_tokens=100, output_tokens=20),
        )


# ---------------------------------------------------------------------------
# Unit tests: scoring rubric
# ---------------------------------------------------------------------------


class TestScoreToolDescription:
    def test_good_description_scores_high(self) -> None:
        """A rich description with purpose, params, limitations, and usage gets >= 0.75."""
        name = "search_documents"
        desc = (
            "Searches the document index and returns matching results. "
            "Only returns up to 50 results per call. "
            "Use this when you need to find documents by keyword or topic. "
            "Requires the query parameter."
        )
        schema = {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query text"},
                "limit": {"type": "integer"},
            },
            "required": ["query"],
        }
        score = score_tool_description(name, desc, schema)
        assert score >= 0.75, f"Expected >= 0.75, got {score}"

    def test_bare_one_word_description_scores_below_threshold(self) -> None:
        """A tool named 'search' with a one-word description should score < 0.5."""
        score = score_tool_description("search", "search", {})
        assert score < 0.5, f"Expected < 0.5, got {score}"

    def test_empty_description_scores_below_threshold(self) -> None:
        # Empty description can still get the param-coverage credit (no required params),
        # but can never pass purpose clarity, limitations, or usage criteria.
        score = score_tool_description("my_tool", "", {})
        assert score < 0.5

    def test_short_description_no_purpose_credit(self) -> None:
        """Descriptions < 40 chars don't get purpose credit."""
        score = score_tool_description("my_tool", "Does stuff.", {})
        assert score < 0.5

    def test_full_score_all_criteria(self) -> None:
        """Verify each of the four 0.25 credits can be earned."""
        name = "create_ticket"
        desc = (
            "Creates a new support ticket in the issue tracker. "
            "Requires title and description fields. "
            "Cannot create tickets in closed projects. "
            "Use this when you need to file a new bug or request."
        )
        schema = {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Ticket title"},
                "description": {"type": "string", "description": "Ticket body"},
            },
            "required": ["title", "description"],
        }
        score = score_tool_description(name, desc, schema)
        assert score == 1.0, f"Expected 1.0, got {score}"

    def test_no_required_params_gives_full_param_coverage_credit(self) -> None:
        """Tools with no required params get the parameter coverage credit automatically."""
        desc = "Lists all available records. Only returns enabled items. Use this to see the full catalog."
        score = score_tool_description(
            "list_items", desc, {"type": "object", "properties": {}}
        )
        # Should get at least purpose + param_coverage + limitations + usage = 1.0
        assert score >= 0.75

    def test_parameter_coverage_by_description_in_schema(self) -> None:
        """Required param with 'description' field in schema earns coverage credit."""
        desc = (
            "Fetches the user profile for a given identifier. "
            "Does not return archived users. "
            "Use this when you need user metadata."
        )
        schema = {
            "type": "object",
            "properties": {
                "user_id": {"type": "string", "description": "The user's unique ID"},
            },
            "required": ["user_id"],
        }
        score = score_tool_description("get_user", desc, schema)
        # Should be at least 0.75 (purpose, param_coverage via schema, limitations, usage)
        assert score >= 0.75

    def test_parameter_coverage_by_mention_in_text(self) -> None:
        """Required param mentioned by name in description earns coverage credit."""
        desc = (
            "Sends an email message to the specified recipient. "
            "Cannot send to external domains. "
            "Use this when you need to notify a user via email. "
            "The recipient field must be a valid email address."
        )
        schema = {
            "type": "object",
            "properties": {
                "recipient": {"type": "string"},
                "body": {"type": "string"},
            },
            "required": ["recipient"],
        }
        score = score_tool_description("send_email", desc, schema)
        assert score >= 0.75

    def test_missing_required_param_no_coverage_credit(self) -> None:
        """A required param that has no schema description and is not mentioned scores lower."""
        desc = (
            "Creates a new workspace. Does not allow duplicate names. "
            "Use this when you need a new workspace."
        )
        schema = {
            "type": "object",
            "properties": {
                "workspace_name": {"type": "string"},  # no "description" key
                "secret_token": {
                    "type": "string"
                },  # not mentioned in desc, no description
            },
            "required": ["workspace_name", "secret_token"],
        }
        score = score_tool_description("create_workspace", desc, schema)
        # Missing param coverage credit: max 0.75
        assert score <= 0.75

    def test_score_is_deterministic(self) -> None:
        """Same inputs always produce the same score."""
        name = "do_thing"
        desc = "Does the thing. Cannot fail. Use this when you need the thing done."
        schema: dict = {}
        assert score_tool_description(name, desc, schema) == score_tool_description(
            name, desc, schema
        )


class TestSha256OfDescription:
    def test_same_content_same_hash(self) -> None:
        assert sha256_of_description("hello") == sha256_of_description("hello")

    def test_different_content_different_hash(self) -> None:
        assert sha256_of_description("hello") != sha256_of_description("world")

    def test_returns_hex_string(self) -> None:
        result = sha256_of_description("test")
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)


class TestEnrichToolDescription:
    def test_enricher_returns_llm_response(self) -> None:
        llm = FakeLLMClient(response="Better description of the tool.")
        result = enrich_tool_description(
            llm,
            name="my_tool",
            description="does stuff",
            input_schema={},
        )
        assert result == "Better description of the tool."
        assert llm.call_count == 1

    def test_enricher_truncates_to_600_chars(self) -> None:
        long_response = "A" * 700
        llm = FakeLLMClient(response=long_response)
        result = enrich_tool_description(
            llm,
            name="my_tool",
            description="does stuff",
            input_schema={},
        )
        assert result is not None
        assert len(result) == 600

    def test_enricher_returns_none_on_empty_response(self) -> None:
        llm = FakeLLMClient(response="")
        result = enrich_tool_description(
            llm,
            name="my_tool",
            description="does stuff",
            input_schema={},
        )
        assert result is None

    def test_enricher_returns_none_on_llm_exception(self) -> None:
        """A raising LLM client must not propagate the exception."""
        llm = FakeLLMClient(raise_on_call=True)
        result = enrich_tool_description(
            llm,
            name="my_tool",
            description="does stuff",
            input_schema={},
        )
        assert result is None

    def test_enricher_uses_synthetic_task_id_when_none(self) -> None:
        """enrich_tool_description must not raise when task_id=None."""
        llm = FakeLLMClient(response="A better description.")
        result = enrich_tool_description(
            llm,
            name="tool",
            description="stuff",
            input_schema={},
            task_id=None,
        )
        assert result is not None


# ---------------------------------------------------------------------------
# DB-backed fixtures (shared with test_mcp.py pattern)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def engine() -> Iterator[Engine]:
    assert TEST_POSTGRES_URL is not None
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", normalize_database_url(TEST_POSTGRES_URL))
    command.upgrade(config, "head")
    engine = make_engine(TEST_POSTGRES_URL)
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture
def db_session(engine: Engine) -> Iterator[Session]:
    session_factory = make_session_factory(engine=engine)
    with session_factory() as session:
        _cleanup(session)
        session.commit()
        yield session
        session.rollback()
        _cleanup(session)
        session.commit()


def _cleanup(session: Session) -> None:
    for model in (McpServerTool, McpServer, TaskEvent, Task, Installation):
        session.execute(delete(model))


def _installation(session: Session) -> Installation:
    installation = Installation(slack_team_id=f"T{uuid.uuid4().hex}")
    session.add(installation)
    session.flush()
    return installation


def _task(session: Session, installation: Installation) -> Task:
    thread_ts = f"{uuid.uuid4().int % 10**6}.{uuid.uuid4().int % 10**6}"
    return TaskService(session).create_task(
        installation_id=installation.id,
        slack_event_id=f"Ev{uuid.uuid4().hex}",
        slack_channel_id="C123",
        slack_thread_ts=thread_ts,
        slack_message_ts=thread_ts,
        slack_user_id="U123",
        input="do the thing",
    )


def _server(session: Session, installation: Installation) -> McpServer:
    server = McpServer(
        installation_id=installation.id,
        name="quality-test-server",
        transport="stdio",
        command=sys.executable,
        args=[ECHO_SERVER],
        status="enabled",
        created_by="test",
    )
    session.add(server)
    session.flush()
    return server


def _discovered(
    name: str,
    description: str,
    input_schema: dict | None = None,
) -> DiscoveredTool:
    return DiscoveredTool(
        name=name,
        description=description,
        input_schema=input_schema or {},
        read_only_hint=True,
        destructive_hint=None,
    )


# ---------------------------------------------------------------------------
# DB-backed tests: SHA gating, enrichment stored, provider card, resilience
# ---------------------------------------------------------------------------


@db_required
class TestUpsertDescriptionQuality:
    def test_upsert_scores_new_tool(self, db_session: Session) -> None:
        installation = _installation(db_session)
        server = _server(db_session, installation)
        discovered = [
            _discovered(
                "search_docs",
                "Searches the document store. Cannot return archived items. "
                "Use this when you need to find documents by keyword.",
                {"type": "object", "properties": {}, "required": []},
            )
        ]
        upsert_discovered_tools(
            db_session,
            server=server,
            discovered=cast(list[object], discovered),
            error=None,
            llm=None,
        )
        db_session.flush()

        row = db_session.scalar(
            select(McpServerTool).where(McpServerTool.server_id == server.id)
        )
        assert row is not None
        assert row.description_quality_score is not None
        assert row.description_sha256 is not None

    def test_sha_gating_no_rescore_on_same_content(self, db_session: Session) -> None:
        """When description has not changed, the quality score must not be recomputed."""
        installation = _installation(db_session)
        server = _server(db_session, installation)
        discovered = [
            _discovered(
                "stable_tool",
                "Retrieves stable items. Does not modify state. "
                "Use this when you need to list stable resources.",
            )
        ]
        # First upsert
        upsert_discovered_tools(
            db_session,
            server=server,
            discovered=cast(list[object], discovered),
            error=None,
            llm=None,
        )
        db_session.flush()

        row = db_session.scalar(
            select(McpServerTool).where(McpServerTool.server_id == server.id)
        )
        assert row is not None
        assert row.description_quality_score is not None
        assert row.description_sha256 is not None

        # Manually set score to a sentinel value to detect re-scoring
        row.description_quality_score = Decimal("0.111")
        db_session.flush()

        # Second upsert with same description — should NOT rescore (sha unchanged)
        upsert_discovered_tools(
            db_session,
            server=server,
            discovered=cast(list[object], discovered),
            error=None,
            llm=None,
        )
        db_session.flush()

        db_session.expire(row)
        updated_row = db_session.scalar(
            select(McpServerTool).where(McpServerTool.server_id == server.id)
        )
        assert updated_row is not None
        # Score must remain the sentinel value (not re-scored)
        assert float(updated_row.description_quality_score or 0) == pytest.approx(0.111)

    def test_enrichment_stored_when_score_below_threshold(
        self, db_session: Session
    ) -> None:
        """Poor-quality descriptions are enriched and the result is persisted."""
        installation = _installation(db_session)
        server = _server(db_session, installation)
        # "bad" — one word, will score < 0.5
        discovered = [_discovered("bad_tool", "bad")]

        llm = FakeLLMClient(
            response="Fetches bad items from the service. Use this when needed."
        )
        upsert_discovered_tools(
            db_session,
            server=server,
            discovered=cast(list[object], discovered),
            error=None,
            llm=llm,
        )
        db_session.flush()

        row = db_session.scalar(
            select(McpServerTool).where(McpServerTool.server_id == server.id)
        )
        assert row is not None
        assert row.description_quality_score is not None
        assert float(row.description_quality_score) < 0.5
        assert row.enriched_description is not None
        assert "Fetches bad items" in row.enriched_description
        assert llm.call_count == 1

    def test_enrichment_not_called_for_good_description(
        self, db_session: Session
    ) -> None:
        """Good descriptions (score >= 0.5) must not trigger enrichment."""
        installation = _installation(db_session)
        server = _server(db_session, installation)
        discovered = [
            _discovered(
                "good_tool",
                "Retrieves all active records from the data store. "
                "Does not return archived or deleted records. "
                "Use this when you need a full list of active entities.",
            )
        ]
        llm = FakeLLMClient(response="Should not be called.")
        upsert_discovered_tools(
            db_session,
            server=server,
            discovered=cast(list[object], discovered),
            error=None,
            llm=llm,
        )
        db_session.flush()

        row = db_session.scalar(
            select(McpServerTool).where(McpServerTool.server_id == server.id)
        )
        assert row is not None
        score = float(row.description_quality_score or 0)
        assert score >= 0.5
        assert row.enriched_description is None
        assert llm.call_count == 0

    def test_discovery_survives_raising_enricher(self, db_session: Session) -> None:
        """A raising LLM must never fail tool discovery."""
        installation = _installation(db_session)
        server = _server(db_session, installation)
        discovered = [_discovered("fragile_tool", "bad")]  # score < 0.5

        llm = FakeLLMClient(raise_on_call=True)
        # Must not raise
        count = upsert_discovered_tools(
            db_session,
            server=server,
            discovered=cast(list[object], discovered),
            error=None,
            llm=llm,
        )
        db_session.flush()

        assert count == 1
        row = db_session.scalar(
            select(McpServerTool).where(McpServerTool.server_id == server.id)
        )
        assert row is not None
        # Tool is persisted even though enrichment raised
        assert row.name == "fragile_tool"
        # enriched_description should remain None (enrichment failed silently)
        assert row.enriched_description is None


@db_required
class TestProviderCardUsesEnrichedDescription:
    def test_card_description_uses_enriched_when_present(
        self, db_session: Session
    ) -> None:
        """McpExternalToolProvider.tool_cards() uses enriched_description when set."""
        installation = _installation(db_session)
        server = McpServer(
            installation_id=installation.id,
            name="enriched-server",
            transport="stdio",
            command=sys.executable,
            args=[ECHO_SERVER],
            status="enabled",
            created_by="test",
        )
        db_session.add(server)
        db_session.flush()

        tool = McpServerTool(
            server_id=server.id,
            name="my_tool",
            description="raw original description",
            input_schema={},
            read_only_hint=True,
            enabled=True,
            enriched_description="LLM-improved description of the tool.",
        )
        db_session.add(tool)
        db_session.flush()

        task = _task(db_session, installation)
        db_session.commit()

        provider = McpExternalToolProvider(
            session=db_session,
            task=task,
            encryption_key=ENCRYPTION_KEY,
            tool_timeout_seconds=30,
        )
        (card,) = provider.tool_cards()
        assert "LLM-improved description" in card.description
        assert "raw original description" not in card.description

    def test_card_description_falls_back_to_raw_when_not_enriched(
        self, db_session: Session
    ) -> None:
        """Without enriched_description the provider falls back to the raw description."""
        installation = _installation(db_session)
        server = McpServer(
            installation_id=installation.id,
            name="raw-server",
            transport="stdio",
            command=sys.executable,
            args=[ECHO_SERVER],
            status="enabled",
            created_by="test",
        )
        db_session.add(server)
        db_session.flush()

        tool = McpServerTool(
            server_id=server.id,
            name="my_raw_tool",
            description="the original raw description",
            input_schema={},
            read_only_hint=None,
            enabled=True,
        )
        db_session.add(tool)
        db_session.flush()

        task = _task(db_session, installation)
        db_session.commit()

        provider = McpExternalToolProvider(
            session=db_session,
            task=task,
            encryption_key=ENCRYPTION_KEY,
            tool_timeout_seconds=30,
        )
        (card,) = provider.tool_cards()
        assert "the original raw description" in card.description


@db_required
class TestMcpToolRowQualityBadge:
    def test_badge_enriched_when_enriched_description_present(
        self, db_session: Session
    ) -> None:
        installation = _installation(db_session)
        server = _server(db_session, installation)
        discovered = [_discovered("bad", "bad")]
        llm = FakeLLMClient(response="Better description here for the tool.")
        upsert_discovered_tools(
            db_session,
            server=server,
            discovered=cast(list[object], discovered),
            error=None,
            llm=llm,
        )
        db_session.flush()

        dashboard = get_mcp_dashboard(db_session, installation.id)
        assert len(dashboard.servers) == 1
        (tool_row,) = dashboard.servers[0].tools
        assert tool_row.quality_badge == "accent"
        assert tool_row.quality_label == "enriched"

    def test_badge_warning_when_poor_no_enrichment(self, db_session: Session) -> None:
        installation = _installation(db_session)
        server = _server(db_session, installation)
        discovered = [_discovered("poor_no_llm", "bad")]
        # No LLM → no enrichment
        upsert_discovered_tools(
            db_session,
            server=server,
            discovered=cast(list[object], discovered),
            error=None,
            llm=None,
        )
        db_session.flush()

        dashboard = get_mcp_dashboard(db_session, installation.id)
        (tool_row,) = dashboard.servers[0].tools
        score = float(tool_row.description_quality_score or 0)
        assert score < 0.5
        assert tool_row.quality_badge == "warning"
        assert tool_row.quality_label == "poor"

    def test_badge_success_when_good_no_enrichment(self, db_session: Session) -> None:
        installation = _installation(db_session)
        server = _server(db_session, installation)
        discovered = [
            _discovered(
                "good_no_llm",
                "Retrieves all active items from the data store. "
                "Does not return archived records. "
                "Use this when you need a complete list of active items.",
            )
        ]
        upsert_discovered_tools(
            db_session,
            server=server,
            discovered=cast(list[object], discovered),
            error=None,
            llm=None,
        )
        db_session.flush()

        dashboard = get_mcp_dashboard(db_session, installation.id)
        (tool_row,) = dashboard.servers[0].tools
        score = float(tool_row.description_quality_score or 0)
        assert score >= 0.5
        assert tool_row.quality_badge == "success"
        assert tool_row.quality_label == "ok"
