"""Write actions for the dashboard MCP servers admin page."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from kortny.db.models import McpServer, McpServerTool
from kortny.mcp.description_quality import (
    DescriptionEnricherLLMClient,
    enrich_tool_description,
    score_tool_description,
    sha256_of_description,
)
from kortny.secrets import SecretEncryptionError, encrypt_secret_value

logger = logging.getLogger(__name__)

VALID_TRANSPORTS = ("stdio", "streamable_http", "sse")


class McpServerError(ValueError):
    """Raised for validation errors on MCP server operations."""


def add_mcp_server(
    session: Session,
    *,
    installation_id: uuid.UUID,
    name: str,
    transport: str,
    command: str | None,
    args: list[str],
    url: str | None,
    env_pairs: list[tuple[str, str]],
    header_pairs: list[tuple[str, str]],
    secret_pairs: list[tuple[str, str]],
    created_by: str,
    encryption_key: str | None,
) -> McpServer:
    """Validate and persist a new MCP server registration."""

    name = name.strip()
    if not name:
        raise McpServerError("Server name is required.")
    if transport not in VALID_TRANSPORTS:
        raise McpServerError(
            f"Transport must be one of: {', '.join(VALID_TRANSPORTS)}."
        )
    if transport == "stdio":
        command = (command or "").strip()
        if not command:
            raise McpServerError("Command is required for stdio transport.")
        url = None
    else:
        url = (url or "").strip()
        if not url:
            raise McpServerError("URL is required for HTTP/SSE transport.")
        command = None
        args = []

    # Check uniqueness within the installation.
    existing = session.scalar(
        select(McpServer).where(
            McpServer.installation_id == installation_id,
            McpServer.name == name,
        )
    )
    if existing is not None:
        raise McpServerError(f"A server named '{name}' is already registered.")

    env_json: dict[str, str] = {}
    for k, v in env_pairs:
        if k.strip():
            env_json[k.strip()] = v

    headers_json: dict[str, str] = {}
    for k, v in header_pairs:
        if k.strip():
            headers_json[k.strip()] = v

    secret_env: bytes | None = None
    secret_dict: dict[str, str] = {k.strip(): v for k, v in secret_pairs if k.strip()}
    if secret_dict:
        if not encryption_key:
            raise McpServerError(
                "ENCRYPTION_KEY must be set to store secrets in the dashboard."
            )
        try:
            secret_env = encrypt_secret_value(
                json.dumps(secret_dict),
                encryption_key=encryption_key,
            )
        except SecretEncryptionError as exc:
            raise McpServerError(str(exc)) from exc

    server = McpServer(
        installation_id=installation_id,
        name=name,
        transport=transport,
        command=command if transport == "stdio" else None,
        args=args if transport == "stdio" else [],
        url=url if transport != "stdio" else None,
        env_json=env_json,
        headers_json=headers_json,
        secret_env=secret_env,
        status="enabled",
        created_by=created_by,
    )
    session.add(server)
    session.flush()
    return server


def remove_mcp_server(
    session: Session,
    *,
    installation_id: uuid.UUID,
    server_id: uuid.UUID,
) -> McpServer:
    """Hard-delete a registered MCP server (cascades to its tools)."""
    server = _get_server(session, installation_id, server_id)
    session.delete(server)
    session.flush()
    return server


def toggle_mcp_server(
    session: Session,
    *,
    installation_id: uuid.UUID,
    server_id: uuid.UUID,
) -> McpServer:
    """Flip enabled/disabled status of a registered MCP server."""
    server = _get_server(session, installation_id, server_id)
    server.status = "disabled" if server.status == "enabled" else "enabled"
    server.updated_at = datetime.now(UTC)
    session.flush()
    return server


def upsert_discovered_tools(
    session: Session,
    *,
    server: McpServer,
    discovered: list[object],
    error: str | None,
    llm: DescriptionEnricherLLMClient | None = None,
) -> int:
    """Persist tool discovery results; return number of tools upserted.

    After upserting each tool's core fields, runs the description quality
    pipeline when an LLM client is supplied:

    - Computes the SHA-256 of the raw description.
    - If the SHA changed (or the score has never been computed), scores the
      description using the deterministic rubric.
    - If the score is below the 0.5 threshold and the LLM is available,
      attempts one cheap-tier enrichment call and stores the result.

    Scoring/enrichment failures are logged and silently skipped so they never
    fail discovery.
    """
    now = datetime.now(UTC)
    server.last_discovery_at = now
    server.last_discovery_error = error
    server.updated_at = now

    if error or not discovered:
        session.flush()
        return 0

    seen_names: set[str] = set()
    upserted = 0
    for tool in discovered:
        tool_name = getattr(tool, "name", None)
        if not isinstance(tool_name, str) or not tool_name:
            continue
        seen_names.add(tool_name)
        existing = session.scalar(
            select(McpServerTool).where(
                McpServerTool.server_id == server.id,
                McpServerTool.name == tool_name,
            )
        )
        description = getattr(tool, "description", "") or ""
        input_schema = getattr(tool, "input_schema", {}) or {}
        read_only_hint = getattr(tool, "read_only_hint", None)
        destructive_hint = getattr(tool, "destructive_hint", None)
        if existing is None:
            row = McpServerTool(
                server_id=server.id,
                name=tool_name,
                description=description,
                input_schema=input_schema,
                read_only_hint=read_only_hint,
                destructive_hint=destructive_hint,
                enabled=True,
            )
            session.add(row)
            session.flush()  # populate row.id before quality pass
        else:
            row = existing
            row.description = description
            row.input_schema = input_schema
            row.read_only_hint = read_only_hint
            row.destructive_hint = destructive_hint
            row.updated_at = now
        upserted += 1

        # Quality scoring + optional enrichment
        _apply_description_quality(row, description, input_schema, llm=llm)

    session.flush()
    return upserted


# ---------------------------------------------------------------------------
# Description quality helpers
# ---------------------------------------------------------------------------

_QUALITY_THRESHOLD = 0.5


def _apply_description_quality(
    row: McpServerTool,
    description: str,
    input_schema: dict,
    *,
    llm: DescriptionEnricherLLMClient | None,
) -> None:
    """Score and optionally enrich one tool's description.  Never raises."""
    try:
        new_sha = sha256_of_description(description)
        sha_changed = row.description_sha256 != new_sha
        needs_score = sha_changed or row.description_quality_score is None

        if not needs_score:
            return

        # Always re-score when sha changed or score is missing
        score = score_tool_description(row.name, description, input_schema)
        row.description_quality_score = score  # type: ignore[assignment]
        row.description_sha256 = new_sha

        # Clear stale enriched description when the raw description changed
        if sha_changed:
            row.enriched_description = None

        # Enrich if below threshold and LLM is available
        if (
            score < _QUALITY_THRESHOLD
            and llm is not None
            and row.enriched_description is None
        ):
            enriched = enrich_tool_description(
                llm,
                name=row.name,
                description=description,
                input_schema=input_schema,
            )
            if enriched:
                row.enriched_description = enriched

    except Exception:
        logger.exception(
            "mcp_description_quality_failed",
            extra={"tool_name": row.name},
        )


def toggle_mcp_tool(
    session: Session,
    *,
    installation_id: uuid.UUID,
    server_id: uuid.UUID,
    tool_id: uuid.UUID,
) -> McpServerTool:
    """Flip enabled/disabled for a single tool."""
    # Verify server ownership first.
    _get_server(session, installation_id, server_id)
    tool = session.get(McpServerTool, tool_id)
    if tool is None or tool.server_id != server_id:
        raise McpServerError("Tool not found.")
    tool.enabled = not tool.enabled
    tool.updated_at = datetime.now(UTC)
    session.flush()
    return tool


def _get_server(
    session: Session,
    installation_id: uuid.UUID,
    server_id: uuid.UUID,
) -> McpServer:
    server = session.get(McpServer, server_id)
    if server is None or server.installation_id != installation_id:
        raise McpServerError("MCP server not found.")
    return server


def parse_kv_textarea(text: str) -> list[tuple[str, str]]:
    """Parse a key=value textarea (one pair per line) into a list of tuples."""
    pairs: list[tuple[str, str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            pairs.append((key.strip(), value.strip()))
    return pairs
