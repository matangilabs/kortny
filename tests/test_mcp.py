"""Tests for MCP server support: client bridge, execute tool, provider, models.

The stdio client tests spawn the real ``tests/fixtures/mcp/echo_server.py``
FastMCP server via ``sys.executable`` so coverage is genuinely end-to-end with
no network. DB-backed tests require ``KORTNY_TEST_POSTGRES_URL``.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from kortny.approvals import ApprovalScope, ToolApprovalPolicy
from kortny.autonomy import AutonomyLevel, AutonomyTier
from kortny.db.models import Installation, McpServer, McpServerTool, Task, TaskEvent
from kortny.db.session import make_engine, make_session_factory, normalize_database_url
from kortny.mcp.client import (
    McpClientError,
    call_server_tool,
    check_server,
    discover_server_tools,
)
from kortny.mcp.provider import McpExternalToolProvider
from kortny.mcp.sessions import McpSessionManager
from kortny.secrets import encrypt_secret_value
from kortny.tasks import TaskService
from kortny.tools.mcp_execute import McpExecuteTool, mcp_runtime_tool_name
from kortny.tools.types import RecoverableToolError

ECHO_SERVER = str(Path(__file__).parent / "fixtures" / "mcp" / "echo_server.py")
ENCRYPTION_KEY = "mcp-test-encryption-key"

TEST_POSTGRES_URL = os.environ.get("KORTNY_TEST_POSTGRES_URL")

db_required = pytest.mark.skipif(
    TEST_POSTGRES_URL is None,
    reason="KORTNY_TEST_POSTGRES_URL is required for MCP DB tests",
)


def _stdio_server(*, name: str = "echo", secret_env: bytes | None = None) -> McpServer:
    """Construct an unpersisted stdio McpServer pointing at the echo fixture."""

    return McpServer(
        id=uuid.uuid4(),
        installation_id=uuid.uuid4(),
        name=name,
        transport="stdio",
        command=sys.executable,
        args=[ECHO_SERVER],
        env_json={},
        headers_json={},
        secret_env=secret_env,
        status="enabled",
        created_by="test",
    )


# ---------------------------------------------------------------------------
# Client bridge (no DB)
# ---------------------------------------------------------------------------


class TestMcpClient:
    def test_discover_returns_both_tools_with_schemas(self) -> None:
        tools = discover_server_tools(_stdio_server(), encryption_key=ENCRYPTION_KEY)
        by_name = {tool.name: tool for tool in tools}

        assert set(by_name) == {"echo", "write_note", "server_pid", "big_echo"}
        assert by_name["echo"].read_only_hint is True
        assert by_name["echo"].input_schema.get("type") == "object"
        assert "text" in by_name["echo"].input_schema.get("properties", {})
        # write_note is unannotated -> no read-only hint.
        assert by_name["write_note"].read_only_hint is not True

    def test_check_server_returns_name_and_version(self) -> None:
        result = check_server(_stdio_server(), encryption_key=ENCRYPTION_KEY)
        assert "kortny-echo-fixture" in result

    def test_call_server_tool_round_trips_echo(self) -> None:
        result = call_server_tool(
            _stdio_server(),
            "echo",
            {"text": "hello"},
            encryption_key=ENCRYPTION_KEY,
            timeout_seconds=30,
        )
        assert result.is_error is False
        assert "echo: hello" in result.text

    def test_call_unknown_tool_returns_error_result(self) -> None:
        # FastMCP reports an unknown tool as an isError result, not a protocol
        # exception; the bridge surfaces that via is_error=True.
        result = call_server_tool(
            _stdio_server(),
            "does_not_exist",
            {},
            encryption_key=ENCRYPTION_KEY,
            timeout_seconds=30,
        )
        assert result.is_error is True
        assert "does_not_exist" in result.text

    def test_secret_env_merges_into_environment(self) -> None:
        secret = encrypt_secret_value(
            json.dumps({"SECRET_TOKEN": "abc"}), encryption_key=ENCRYPTION_KEY
        )
        # Server still launches and discovers tools with secrets present.
        tools = discover_server_tools(
            _stdio_server(secret_env=secret), encryption_key=ENCRYPTION_KEY
        )
        assert {tool.name for tool in tools} == {
            "echo",
            "write_note",
            "server_pid",
            "big_echo",
        }

    def test_unsupported_transport_raises(self) -> None:
        server = _stdio_server()
        server.transport = "carrier-pigeon"
        with pytest.raises(McpClientError):
            check_server(server, encryption_key=ENCRYPTION_KEY)


# ---------------------------------------------------------------------------
# Session manager: per-task session reuse (HIG-214) — no DB
# ---------------------------------------------------------------------------


def _pid_from_result(text: str) -> int:
    match = re.search(r"pid:\s*(\d+)", text)
    assert match is not None, f"no pid in result text: {text!r}"
    return int(match.group(1))


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class TestMcpSessionManager:
    def test_two_calls_reuse_one_session_and_subprocess(self) -> None:
        manager = McpSessionManager()
        server = _stdio_server()
        try:
            first = manager.call_tool(
                server,
                "server_pid",
                {},
                encryption_key=ENCRYPTION_KEY,
                timeout_seconds=30,
            )
            pid_one = _pid_from_result(first.text)

            started = time.perf_counter()
            second = manager.call_tool(
                server,
                "server_pid",
                {},
                encryption_key=ENCRYPTION_KEY,
                timeout_seconds=30,
            )
            second_latency = time.perf_counter() - started
            pid_two = _pid_from_result(second.text)

            # Same subprocess across calls -> one spawn, session reused.
            assert pid_one == pid_two
            # Reused session means no ~1s subprocess spawn on the 2nd call.
            assert second_latency < 0.5, f"second call too slow: {second_latency:.3f}s"
        finally:
            manager.close()

    def test_reconnect_after_session_invalidated(self) -> None:
        manager = McpSessionManager()
        server = _stdio_server()
        try:
            first = manager.call_tool(
                server,
                "server_pid",
                {},
                encryption_key=ENCRYPTION_KEY,
                timeout_seconds=30,
            )
            pid_one = _pid_from_result(first.text)

            # Invalidate the cached session out from under the manager, as a
            # transport failure would. The next call must reconnect once.
            key = str(server.id)
            assert key in manager._owners
            manager._drop_session(key)
            assert key not in manager._owners

            second = manager.call_tool(
                server,
                "server_pid",
                {},
                encryption_key=ENCRYPTION_KEY,
                timeout_seconds=30,
            )
            pid_two = _pid_from_result(second.text)
            # A fresh subprocess (new PID) proves a clean reconnect.
            assert pid_one != pid_two
            assert _process_alive(pid_two)
        finally:
            manager.close()

    def test_close_leaves_no_orphaned_subprocess(self) -> None:
        manager = McpSessionManager()
        result = manager.call_tool(
            _stdio_server(),
            "server_pid",
            {},
            encryption_key=ENCRYPTION_KEY,
            timeout_seconds=30,
        )
        pid = _pid_from_result(result.text)
        assert _process_alive(pid)

        manager.close()

        # Allow the OS a brief moment to reap the terminated child.
        deadline = time.perf_counter() + 5.0
        while _process_alive(pid) and time.perf_counter() < deadline:
            time.sleep(0.05)
        assert not _process_alive(pid), f"orphaned MCP subprocess pid={pid}"

    def test_close_is_idempotent(self) -> None:
        manager = McpSessionManager()
        manager.call_tool(
            _stdio_server(),
            "server_pid",
            {},
            encryption_key=ENCRYPTION_KEY,
            timeout_seconds=30,
        )
        manager.close()
        # A second close must not raise.
        manager.close()


# ---------------------------------------------------------------------------
# Tool naming + execute tool (no DB)
# ---------------------------------------------------------------------------


class TestMcpRuntimeToolName:
    def test_naming_pattern(self) -> None:
        assert mcp_runtime_tool_name("github", "create_issue") == (
            "mcp__github__create_issue"
        )

    def test_sanitizes_and_fits_64_chars(self) -> None:
        name = mcp_runtime_tool_name("My Server!", "Some Tool" * 20)
        assert len(name) <= 64
        assert name.startswith("mcp__my_server__")


class TestMcpExecuteTool:
    def _tool(
        self,
        *,
        tool_name: str = "echo",
        read_only: bool | None = True,
        schema: dict | None = None,
    ) -> McpExecuteTool:
        server = _stdio_server()
        server_tool = McpServerTool(
            id=uuid.uuid4(),
            server_id=server.id,
            name=tool_name,
            description="Echo text.",
            input_schema=schema
            or {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            read_only_hint=read_only,
            destructive_hint=None,
            enabled=True,
        )
        return McpExecuteTool(
            session=cast(Session, object()),
            task=None,
            server=server,
            tool=server_tool,
            encryption_key=ENCRYPTION_KEY,
            timeout_seconds=30,
        )

    def test_invoke_maps_text_result(self) -> None:
        tool = self._tool()
        result = tool.invoke({"text": "hi"})
        assert result.output["provider"] == "mcp"
        assert result.output["server"] == "echo"
        assert "echo: hi" in result.output["text"]

    def test_invoke_unknown_tool_raises_recoverable(self) -> None:
        tool = self._tool(tool_name="nope")
        with pytest.raises(RecoverableToolError) as excinfo:
            tool.invoke({"text": "x"})
        # Unknown tool comes back as an MCP isError result.
        assert excinfo.value.code == "mcp_tool_error"

    def test_parameters_and_name_derived(self) -> None:
        tool = self._tool()
        assert tool.name == "mcp__echo__echo"
        assert tool.parameters["type"] == "object"
        assert "text" in tool.parameters["properties"]

    def test_read_only_hint_routes_through_no_approval(self) -> None:
        policy = ToolApprovalPolicy()
        read_tool = self._tool(read_only=True)
        requirement = policy.requirement_for(read_tool, {})
        assert requirement.scope is ApprovalScope.none

    def test_write_tool_auto_audits_at_balanced(self) -> None:
        # HIG-223: a non-destructive MCP write maps to the implicit tier, so the
        # balanced default auto-approves it with an audit trail instead of
        # gating. Conservative still gates it (see classifier/approval tests).
        policy = ToolApprovalPolicy()
        write_tool = self._tool(
            tool_name="write_note",
            read_only=None,
            schema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        )
        requirement = policy.requirement_for(write_tool, {})
        assert requirement.required is False
        assert requirement.scope is ApprovalScope.none
        assert requirement.audit_autonomy is True
        assert requirement.autonomy_tier is AutonomyTier.implicit

    def test_write_tool_gated_at_conservative(self) -> None:
        policy = ToolApprovalPolicy()
        write_tool = self._tool(
            tool_name="write_note",
            read_only=None,
            schema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        )
        requirement = policy.requirement_for(
            write_tool, {}, autonomy_level=AutonomyLevel.conservative
        )
        assert requirement.required is True
        assert requirement.scope is ApprovalScope.user

    def test_destructive_hint_requires_approval_every_level(self) -> None:
        policy = ToolApprovalPolicy()
        delete_tool = self._tool(
            tool_name="delete_note",
            read_only=None,
            schema={
                "type": "object",
                "properties": {"id": {"type": "string"}},
                "required": ["id"],
            },
        )
        # destructive_hint=True on the server tool forces the explicit tier.
        delete_tool.server_tool.destructive_hint = True
        for level in AutonomyLevel:
            requirement = policy.requirement_for(
                delete_tool, {"id": "n1"}, autonomy_level=level
            )
            assert requirement.required is True
            assert requirement.scope is ApprovalScope.user

    def test_big_payload_is_bounded(self) -> None:
        server = _stdio_server()
        server_tool = McpServerTool(
            id=uuid.uuid4(),
            server_id=server.id,
            name="big_echo",
            description="Return a big payload.",
            input_schema={
                "type": "object",
                "properties": {"size": {"type": "integer"}},
                "required": ["size"],
            },
            read_only_hint=True,
            destructive_hint=None,
            enabled=True,
        )
        tool = McpExecuteTool(
            session=cast(Session, object()),
            task=None,
            server=server,
            tool=server_tool,
            encryption_key=ENCRYPTION_KEY,
            timeout_seconds=30,
            result_max_chars=2000,
        )
        result = tool.invoke({"size": 50000})
        assert result.output["truncated"] is True
        assert result.output["original_chars"] > 2000
        assert "truncation_hint" in result.output
        assert len(json.dumps(result.output, default=str)) <= 2000


# ---------------------------------------------------------------------------
# Models / migration constraints + provider (DB-backed)
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


def _persisted_server(
    session: Session,
    installation: Installation,
    *,
    name: str = "echo",
    status: str = "enabled",
) -> McpServer:
    server = McpServer(
        installation_id=installation.id,
        name=name,
        transport="stdio",
        command=sys.executable,
        args=[ECHO_SERVER],
        status=status,
        created_by="admin",
    )
    session.add(server)
    session.flush()
    return server


def _persisted_tool(
    session: Session,
    server: McpServer,
    *,
    name: str = "echo",
    read_only: bool | None = True,
    enabled: bool = True,
) -> McpServerTool:
    tool = McpServerTool(
        server_id=server.id,
        name=name,
        description=f"{name} tool",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        read_only_hint=read_only,
        enabled=enabled,
    )
    session.add(tool)
    session.flush()
    return tool


@db_required
class TestMcpModels:
    def test_round_trip(self, db_session: Session) -> None:
        installation = _installation(db_session)
        server = _persisted_server(db_session, installation)
        _persisted_tool(db_session, server)
        db_session.commit()

        loaded = db_session.scalar(select(McpServer).where(McpServer.id == server.id))
        assert loaded is not None
        assert loaded.transport == "stdio"
        assert loaded.status == "enabled"
        assert loaded.args == [ECHO_SERVER]

    def test_unique_name_per_installation(self, db_session: Session) -> None:
        installation = _installation(db_session)
        _persisted_server(db_session, installation, name="dup")
        db_session.flush()
        with pytest.raises(IntegrityError):
            # _persisted_server flushes internally, raising on the duplicate.
            _persisted_server(db_session, installation, name="dup")

    def test_transport_target_check_rejects_stdio_without_command(
        self, db_session: Session
    ) -> None:
        installation = _installation(db_session)
        server = McpServer(
            installation_id=installation.id,
            name="bad",
            transport="stdio",
            command=None,
            created_by="admin",
        )
        db_session.add(server)
        with pytest.raises(IntegrityError):
            db_session.flush()

    def test_transport_target_check_rejects_http_without_url(
        self, db_session: Session
    ) -> None:
        installation = _installation(db_session)
        server = McpServer(
            installation_id=installation.id,
            name="bad-http",
            transport="streamable_http",
            url=None,
            created_by="admin",
        )
        db_session.add(server)
        with pytest.raises(IntegrityError):
            db_session.flush()

    def test_invalid_transport_rejected(self, db_session: Session) -> None:
        installation = _installation(db_session)
        server = McpServer(
            installation_id=installation.id,
            name="bad-transport",
            transport="pigeon",
            url="http://x",
            created_by="admin",
        )
        db_session.add(server)
        with pytest.raises(IntegrityError):
            db_session.flush()


@db_required
class TestMcpProvider:
    def test_enabled_server_yields_cards_and_runtime_tools(
        self, db_session: Session
    ) -> None:
        installation = _installation(db_session)
        server = _persisted_server(db_session, installation)
        _persisted_tool(db_session, server, name="echo", read_only=True)
        _persisted_tool(db_session, server, name="write_note", read_only=None)
        task = _task(db_session, installation)
        db_session.commit()

        provider = McpExternalToolProvider(
            session=db_session,
            task=task,
            encryption_key=ENCRYPTION_KEY,
            tool_timeout_seconds=30,
        )
        cards = provider.tool_cards()
        runtime = provider.runtime_tools()

        names = {card.registry_name for card in cards}
        assert names == {"mcp__echo__echo", "mcp__echo__write_note"}
        side_effects = {card.registry_name: card.side_effect for card in cards}
        assert side_effects["mcp__echo__echo"] == "read"
        assert side_effects["mcp__echo__write_note"] == "write"
        assert {tool.name for tool in runtime} == names

    def test_disabled_server_excluded(self, db_session: Session) -> None:
        installation = _installation(db_session)
        server = _persisted_server(db_session, installation, status="disabled")
        _persisted_tool(db_session, server)
        task = _task(db_session, installation)
        db_session.commit()

        provider = McpExternalToolProvider(
            session=db_session,
            task=task,
            encryption_key=ENCRYPTION_KEY,
            tool_timeout_seconds=30,
        )
        assert provider.tool_cards() == ()
        assert provider.runtime_tools() == ()

    def test_disabled_tool_excluded(self, db_session: Session) -> None:
        installation = _installation(db_session)
        server = _persisted_server(db_session, installation)
        _persisted_tool(db_session, server, name="echo", enabled=True)
        _persisted_tool(db_session, server, name="write_note", enabled=False)
        task = _task(db_session, installation)
        db_session.commit()

        provider = McpExternalToolProvider(
            session=db_session,
            task=task,
            encryption_key=ENCRYPTION_KEY,
            tool_timeout_seconds=30,
        )
        names = {card.registry_name for card in provider.tool_cards()}
        assert names == {"mcp__echo__echo"}

    def test_load_runtime_tools_for_slugs_builds_only_requested(
        self, db_session: Session
    ) -> None:
        # The find_tools seam (HIG-269): load only the runtime names the
        # retriever surfaced, not the whole catalog.
        installation = _installation(db_session)
        server = _persisted_server(db_session, installation)
        _persisted_tool(db_session, server, name="echo", read_only=True)
        _persisted_tool(db_session, server, name="write_note", read_only=None)
        task = _task(db_session, installation)
        db_session.commit()

        provider = McpExternalToolProvider(
            session=db_session,
            task=task,
            encryption_key=ENCRYPTION_KEY,
            tool_timeout_seconds=30,
        )
        loaded = provider.load_runtime_tools_for_slugs(["mcp__echo__write_note"])
        assert {tool.name for tool in loaded} == {"mcp__echo__write_note"}

        # Unknown / empty slugs load nothing.
        assert provider.load_runtime_tools_for_slugs([]) == ()
        assert provider.load_runtime_tools_for_slugs(["mcp__echo__missing"]) == ()

    def test_cross_installation_isolation(self, db_session: Session) -> None:
        owner = _installation(db_session)
        other = _installation(db_session)
        owner_server = _persisted_server(db_session, owner, name="owned")
        _persisted_tool(db_session, owner_server)
        other_server = _persisted_server(db_session, other, name="foreign")
        _persisted_tool(db_session, other_server)
        task = _task(db_session, other)
        db_session.commit()

        provider = McpExternalToolProvider(
            session=db_session,
            task=task,
            encryption_key=ENCRYPTION_KEY,
            tool_timeout_seconds=30,
        )
        names = {card.toolkit_slug for card in provider.tool_cards()}
        assert names == {"foreign"}

    def test_runtime_tool_executes_end_to_end(self, db_session: Session) -> None:
        installation = _installation(db_session)
        server = _persisted_server(db_session, installation)
        _persisted_tool(db_session, server, name="echo", read_only=True)
        task = _task(db_session, installation)
        db_session.commit()

        provider = McpExternalToolProvider(
            session=db_session,
            task=task,
            encryption_key=ENCRYPTION_KEY,
            tool_timeout_seconds=30,
        )
        (tool,) = provider.runtime_tools()
        result = tool.invoke({"text": "live"})
        assert "echo: live" in result.output["text"]
