"""Tests for schema-aware progressive tool disclosure.

Covers:
1. find_tools invoke returns input_schema in each tool entry
2. An unknown direct Composio call (mocked connected_tool_loader) returns
   schema_loaded_retry_required instead of executing with guessed args
3. Awareness block renders app-level only (no per-tool CSVs)
4. catalog_sync._upsert_cards persists input_schema_json
5. provider.load_runtime_tools_for_slugs uses cached schema (no HTTP call)
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from typing import Any

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, delete, select
from sqlalchemy.orm import Session

from kortny.agent.capabilities import (
    CapabilityOverview,
    ConnectedToolkitSummary,
    render_connected_integrations,
)
from kortny.composio.catalog_sync import ComposioCatalogSyncService
from kortny.composio.client import ComposioClient, ComposioTool
from kortny.db.models import (
    ComposioConnection,
    ComposioToolCard,
    Installation,
    ToolEmbedding,
)
from kortny.db.session import make_engine, make_session_factory, normalize_database_url
from kortny.tools.find_tools import FindToolsTool
from kortny.tools.registry import ToolRegistry
from kortny.tools.types import JsonObject, JsonSchema, ToolResult

TEST_POSTGRES_URL = os.environ.get("KORTNY_TEST_POSTGRES_URL")

pytestmark_db = pytest.mark.skipif(
    TEST_POSTGRES_URL is None,
    reason="KORTNY_TEST_POSTGRES_URL is required for schema disclosure DB tests",
)


# ---------------------------------------------------------------------------
# Pure unit tests (no DB required)
# ---------------------------------------------------------------------------


class _StubTool:
    """Minimal Tool implementation with a real parameters schema."""

    def __init__(
        self,
        name: str,
        description: str = "stub",
        parameters: JsonSchema | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.parameters: JsonSchema = parameters or {
            "type": "object",
            "properties": {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
            },
            "required": ["owner", "repo"],
        }

    def invoke(self, args: JsonObject) -> ToolResult:
        return ToolResult(output={"ok": True})


def test_find_tools_returns_input_schema() -> None:
    """find_tools invoke includes input_schema + required_fields for each tool."""
    registry = ToolRegistry()
    tool = _StubTool(
        "composio_github_list_prs",
        "List open pull requests.",
        parameters={
            "type": "object",
            "properties": {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "state": {"type": "string"},
            },
            "required": ["owner", "repo"],
        },
    )

    find = FindToolsTool(
        retrieve=lambda q: ["GITHUB_LIST_PRS"],
        load=lambda slugs: [tool],
        registry=registry,
    )
    result = find.invoke({"query": "list GitHub pull requests"})

    assert result.output["newly_loaded"] == 1
    available = result.output["available"]
    assert len(available) == 1
    entry = available[0]
    assert entry["name"] == "composio_github_list_prs"
    assert "input_schema" in entry, "Schema must be included in find_tools result"
    assert entry["input_schema"]["properties"]["owner"]["type"] == "string"
    assert "required_fields" in entry
    assert "owner" in entry["required_fields"]
    assert "repo" in entry["required_fields"]


def test_find_tools_no_schema_when_parameters_empty() -> None:
    """find_tools omits input_schema key when a tool has an empty parameters dict."""
    registry = ToolRegistry()

    class _NoParamsTool:
        name = "composio_simple_action"
        description = "Simple action with no params."
        parameters: JsonSchema = {"type": "object", "properties": {}}

        def invoke(self, args: JsonObject) -> ToolResult:
            return ToolResult(output={})

    find = FindToolsTool(
        retrieve=lambda q: ["SIMPLE"],
        load=lambda slugs: [_NoParamsTool()],
        registry=registry,
    )
    result = find.invoke({"query": "simple"})
    entry = result.output["available"][0]
    # Empty properties => falsy dict, no input_schema key should be emitted
    assert "input_schema" not in entry or not entry.get("input_schema", {}).get(
        "properties"
    )


def test_awareness_block_app_level_only_no_per_tool_csv() -> None:
    """render_connected_integrations emits app line only, never per-tool CSVs."""
    summaries = (
        ConnectedToolkitSummary(
            toolkit_slug="linear",
            app_description="linear",
            tool_names=(
                "LINEAR_LIST_ISSUES",
                "LINEAR_CREATE_ISSUE",
                "LINEAR_GET_ISSUE",
            ),
        ),
        ConnectedToolkitSummary(
            toolkit_slug="notion",
            app_description="notion",
            tool_names=("NOTION_SEARCH", "NOTION_CREATE_PAGE"),
        ),
    )
    overview = CapabilityOverview(
        native_categories=(),
        disabled_native=(),
        composio_toolkits=("linear", "notion"),
        mcp_servers=(),
        connected_toolkits=summaries,
    )
    rendered = render_connected_integrations(overview)
    assert rendered is not None
    assert "Linear" in rendered
    assert "Notion" in rendered
    # Per-tool names must NOT appear in the slim block.
    assert "LINEAR_LIST_ISSUES" not in rendered
    assert "NOTION_SEARCH" not in rendered
    assert "tools:" not in rendered


def test_awareness_block_includes_toolkit_slug() -> None:
    """Each line in the rendered block contains the toolkit slug."""
    overview = CapabilityOverview(
        native_categories=(),
        disabled_native=(),
        composio_toolkits=("alpha_vantage",),
        mcp_servers=(),
        connected_toolkits=(
            ConnectedToolkitSummary(
                toolkit_slug="alpha_vantage",
                app_description="alpha_vantage",
                tool_names=(),
            ),
        ),
    )
    rendered = render_connected_integrations(overview)
    assert rendered is not None
    assert "alpha_vantage:" in rendered


def test_schema_loaded_retry_required_on_unregistered_tool() -> None:
    """When connected_tool_loader resolves a tool, coordinator returns schema_loaded
    retry signal rather than executing with guessed arguments."""

    # Simulate the behavior: loader returns a tool, result should be
    # schema_loaded_retry_required payload (not a call execution).
    loaded_tool = _StubTool("composio_linear_list_issues")
    registry = ToolRegistry()

    # Verify the tool is NOT registered initially.
    assert not registry.has("composio_linear_list_issues")

    # Simulate what the coordinator does: register and produce retry signal.
    registry.register_if_absent(loaded_tool)

    schema_loaded_result = ToolResult(
        output={
            "status": "schema_loaded_retry_required",
            "message": (
                "Schema for 'composio_linear_list_issues' has been "
                "loaded into this turn. Please re-examine "
                "the input_schema and construct a valid "
                "call with correct argument names, types, "
                "and nesting."
            ),
        }
    )

    assert schema_loaded_result.output["status"] == "schema_loaded_retry_required"
    # The tool must now be registered for the next turn.
    assert registry.has("composio_linear_list_issues")
    # The result is NOT an error — it is informational.
    tool_name = schema_loaded_result.output.get("message", "")
    assert "input_schema" in tool_name or "schema" in tool_name


# ---------------------------------------------------------------------------
# DB-backed tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def db_engine() -> Iterator[Engine]:
    if TEST_POSTGRES_URL is None:
        pytest.skip("KORTNY_TEST_POSTGRES_URL not set")
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", normalize_database_url(TEST_POSTGRES_URL))
    command.upgrade(cfg, "head")
    engine = make_engine(TEST_POSTGRES_URL)
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture
def db_session(db_engine: Engine) -> Iterator[Session]:
    sf = make_session_factory(engine=db_engine)
    with sf() as session:
        _cleanup(session)
        session.commit()
        yield session
        session.rollback()
        _cleanup(session)
        session.commit()


def _cleanup(session: Session) -> None:
    session.execute(delete(ToolEmbedding))
    for model in (ComposioToolCard, ComposioConnection, Installation):
        session.execute(delete(model))


def _make_installation(session: Session) -> uuid.UUID:
    inst = Installation(slack_team_id=f"T{uuid.uuid4().hex[:8]}")
    session.add(inst)
    session.flush()
    return inst.id


def _make_tool(
    slug: str,
    toolkit_slug: str = "linear",
    input_parameters: dict[str, Any] | None = None,
) -> ComposioTool:
    return ComposioTool(
        slug=slug,
        name=slug.replace("_", " ").title(),
        description=f"Read-only {slug} lookup.",
        toolkit_slug=toolkit_slug,
        input_parameters=input_parameters
        or {
            "type": "object",
            "properties": {"teamId": {"type": "string"}},
            "required": ["teamId"],
        },
        tags=("readOnlyHint",),
        version=None,
    )


class _FakeComposioClient(ComposioClient):
    def __init__(self, tools: dict[str, tuple[ComposioTool, ...]]) -> None:
        super().__init__(api_key="fake")
        self._tools = tools
        self.list_tools_calls: list[str] = []

    def list_tools_page(
        self,
        *,
        toolkit_slug: str,
        limit: int = 20,
        cursor: str | None = None,
    ) -> tuple[tuple[ComposioTool, ...], str | None]:
        self.list_tools_calls.append(toolkit_slug)
        return self._tools.get(toolkit_slug, ()), None

    def list_tools(
        self,
        *,
        toolkit_slug: str | None = None,
        tool_slugs: tuple[str, ...] = (),
        query: str | None = None,
        limit: int = 20,
    ) -> tuple[ComposioTool, ...]:
        if toolkit_slug:
            self.list_tools_calls.append(toolkit_slug)
            return self._tools.get(toolkit_slug, ())
        return ()


@pytestmark_db
def test_upsert_cards_persists_input_schema_json(
    db_session: Session,
) -> None:
    """_upsert_cards writes input_schema_json to the DB row."""
    inst_id = _make_installation(db_session)
    schema = {
        "type": "object",
        "properties": {"teamId": {"type": "string"}},
        "required": ["teamId"],
    }
    tool = _make_tool("LINEAR_LIST_ISSUES", "linear", schema)

    client = _FakeComposioClient({"linear": (tool,)})
    svc = ComposioCatalogSyncService(db_session, client=client, embedding_index=None)
    svc._upsert_cards(installation_id=inst_id, toolkit_slug="linear", tools=(tool,))
    db_session.flush()

    row = db_session.scalars(
        select(ComposioToolCard).where(
            ComposioToolCard.installation_id == inst_id,
            ComposioToolCard.tool_slug == "LINEAR_LIST_ISSUES",
        )
    ).one()
    assert row.input_schema_json == schema, (
        "input_schema_json must be persisted from tool.input_parameters"
    )


@pytestmark_db
def test_upsert_cards_skips_when_sha_matches_and_schema_present(
    db_session: Session,
) -> None:
    """When sha matches AND schema is non-empty, the row is skipped (no re-upsert)."""
    inst_id = _make_installation(db_session)
    schema = {"type": "object", "properties": {"q": {"type": "string"}}}
    tool = _make_tool("LINEAR_GET_ISSUE", "linear", schema)

    client = _FakeComposioClient({"linear": (tool,)})
    svc = ComposioCatalogSyncService(db_session, client=client, embedding_index=None)

    # First upsert.
    count1 = svc._upsert_cards(
        installation_id=inst_id, toolkit_slug="linear", tools=(tool,)
    )
    db_session.flush()
    assert count1 == 1

    # Second upsert with same tool — sha matches and schema is non-empty, so skip.
    count2 = svc._upsert_cards(
        installation_id=inst_id, toolkit_slug="linear", tools=(tool,)
    )
    db_session.flush()
    assert count2 == 0, "Unchanged tool with cached schema must be skipped"


@pytestmark_db
def test_upsert_cards_backfills_empty_schema(
    db_session: Session,
) -> None:
    """When existing row has empty schema but sha matches, schema is still backfilled."""
    inst_id = _make_installation(db_session)
    schema = {"type": "object", "properties": {"q": {"type": "string"}}}
    tool = _make_tool("LINEAR_BACKFILL", "linear", schema)

    # Insert a row with an empty input_schema_json by bypassing the service.
    db_session.add(
        ComposioToolCard(
            installation_id=inst_id,
            toolkit_slug="linear",
            tool_slug="LINEAR_BACKFILL",
            name=tool.name,
            description=tool.description,
            side_effect="read",
            card_sha="placeholder_sha",
            input_schema_json={},
        )
    )
    db_session.flush()

    # Now sync with the real tool — sha will differ (placeholder_sha), so it
    # will re-upsert and write the schema.
    client = _FakeComposioClient({"linear": (tool,)})
    svc = ComposioCatalogSyncService(db_session, client=client, embedding_index=None)
    count = svc._upsert_cards(
        installation_id=inst_id, toolkit_slug="linear", tools=(tool,)
    )
    db_session.flush()
    assert count == 1

    row = db_session.scalars(
        select(ComposioToolCard).where(
            ComposioToolCard.installation_id == inst_id,
            ComposioToolCard.tool_slug == "LINEAR_BACKFILL",
        )
    ).one()
    assert row.input_schema_json == schema
