"""Kortny tool adapter that executes one cached MCP server tool."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass

from sqlalchemy.orm import Session

from kortny.db.models import McpServer, McpServerTool, Task
from kortny.mcp.client import McpClientError, McpToolCallResult, call_server_tool
from kortny.mcp.sessions import McpSessionManager
from kortny.observability.events import log_observation
from kortny.tools.result_budget import bound_tool_result
from kortny.tools.types import JsonObject, JsonSchema, RecoverableToolError, ToolResult

logger = logging.getLogger(__name__)
MAX_TOOL_NAME_LENGTH = 64
DEFAULT_RESULT_MAX_CHARS = 16000


@dataclass(frozen=True, slots=True)
class _McpToolDescriptor:
    """Read-only hint surface consumed by the approval policy.

    ``ToolApprovalPolicy._tool_is_explicitly_read_only`` inspects ``tool.tool``
    for a ``tags`` collection. Exposing the MCP ``readOnlyHint`` as a tag here
    routes read-only-annotated MCP tools through the same approval path as
    Composio read tools without inventing a new approval mechanism.
    """

    tags: tuple[str, ...]


class McpExecuteTool:
    """Execute one enabled MCP server tool through a fresh per-call session."""

    def __init__(
        self,
        *,
        session: Session,
        task: Task | None,
        server: McpServer,
        tool: McpServerTool,
        encryption_key: str,
        timeout_seconds: int,
        name: str | None = None,
        session_manager: McpSessionManager | None = None,
        result_max_chars: int = DEFAULT_RESULT_MAX_CHARS,
    ) -> None:
        self.session = session
        self.task = task
        self.server = server
        self.server_tool = tool
        self.encryption_key = encryption_key
        self.timeout_seconds = timeout_seconds
        self.session_manager = session_manager
        self.result_max_chars = result_max_chars
        self.name = name or mcp_runtime_tool_name(server.name, tool.name)
        self.description = _description(server, tool)
        self.parameters = _parameters(tool)
        # Surfaced for ToolApprovalPolicy._tool_is_explicitly_read_only.
        self.tool = _McpToolDescriptor(
            tags=("readonlyhint",) if tool.read_only_hint else ()
        )

    def invoke(self, args: JsonObject) -> ToolResult:
        arguments = dict(args)
        if not self.encryption_key:
            raise RecoverableToolError(
                code="mcp_encryption_key_missing",
                message=(
                    f"{self.name} cannot run because no ENCRYPTION_KEY is "
                    "configured for MCP secret decryption."
                ),
                details={"server": self.server.name, "tool": self.server_tool.name},
            )

        log_observation(
            logger,
            "mcp_tool_execution_started",
            task=self.task,
            provider="mcp",
            runtime_tool=self.name,
            server_name=self.server.name,
            mcp_tool=self.server_tool.name,
            transport=self.server.transport,
            argument_keys=sorted(arguments),
        )
        try:
            result: McpToolCallResult
            if self.session_manager is not None:
                result = self.session_manager.call_tool(
                    self.server,
                    self.server_tool.name,
                    arguments,
                    encryption_key=self.encryption_key,
                    timeout_seconds=float(self.timeout_seconds),
                )
            else:
                result = call_server_tool(
                    self.server,
                    self.server_tool.name,
                    arguments,
                    encryption_key=self.encryption_key,
                    timeout_seconds=int(self.timeout_seconds),
                )
        except McpClientError as exc:
            log_observation(
                logger,
                "mcp_tool_execution_failed",
                task=self.task,
                provider="mcp",
                runtime_tool=self.name,
                server_name=self.server.name,
                mcp_tool=self.server_tool.name,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise RecoverableToolError(
                code="mcp_call_failed",
                message=(
                    f"MCP tool {self.server_tool.name} on server "
                    f"{self.server.name} failed: {exc}"
                ),
                hint=(
                    "Use another available tool if the task can be completed "
                    "without this server, or report that the MCP server is "
                    "unreachable."
                ),
                details={"server": self.server.name, "tool": self.server_tool.name},
            ) from exc

        if result.is_error:
            log_observation(
                logger,
                "mcp_tool_execution_completed",
                task=self.task,
                provider="mcp",
                runtime_tool=self.name,
                server_name=self.server.name,
                mcp_tool=self.server_tool.name,
                successful=False,
            )
            raise RecoverableToolError(
                code="mcp_tool_error",
                message=(
                    f"MCP tool {self.server_tool.name} returned an error: "
                    f"{result.text or 'no error detail provided'}"
                ),
                hint=(
                    "Check the arguments against the tool's schema, then retry "
                    "or choose a different tool."
                ),
                details={
                    "server": self.server.name,
                    "tool": self.server_tool.name,
                    "text": result.text,
                },
            )

        log_observation(
            logger,
            "mcp_tool_execution_completed",
            task=self.task,
            provider="mcp",
            runtime_tool=self.name,
            server_name=self.server.name,
            mcp_tool=self.server_tool.name,
            successful=True,
        )
        output: JsonObject = {
            "provider": "mcp",
            "server": self.server.name,
            "tool": self.server_tool.name,
            "transport": self.server.transport,
            "text": result.text,
        }
        if result.structured is not None:
            output["structured"] = result.structured
        output = self._bound_output(output)
        return ToolResult(output=output)

    def _bound_output(self, output: JsonObject) -> JsonObject:
        bounded = bound_tool_result(
            output,
            max_chars=self.result_max_chars,
            hint=(
                f"MCP tool {self.server_tool.name} on server {self.server.name} "
                "returned a large payload; result truncated."
            ),
        )
        if bounded is not output:
            logger.info(
                "mcp tool result truncated runtime_tool=%s original_chars=%s "
                "final_chars=%s max_chars=%s",
                self.name,
                bounded.get("original_chars"),
                len(json.dumps(bounded, default=str)),
                self.result_max_chars,
            )
        return bounded


def mcp_runtime_tool_name(server_name: str, tool_name: str) -> str:
    """Return the runtime registry name ``mcp__<server>__<tool>`` (<= 64 chars)."""

    server = _safe_identifier(server_name)
    tool = _safe_identifier(tool_name)
    return _fit_tool_name(f"mcp__{server}__{tool}")


def _description(server: McpServer, tool: McpServerTool) -> str:
    required = ", ".join(_required_fields(tool.input_schema)) or "none"
    access = "read-only" if tool.read_only_hint else "write-capable"
    body = tool.description.strip() or tool.name
    return (
        f"MCP {access} tool {tool.name} from server '{server.name}'. {body} "
        f"Required fields: {required}. Omit unknown optional arguments."
    )


def _parameters(tool: McpServerTool) -> JsonSchema:
    schema = dict(tool.input_schema or {})
    if schema.get("type") != "object":
        schema["type"] = "object"
    if not isinstance(schema.get("properties"), dict):
        schema["properties"] = {}
    return schema


def _required_fields(parameters: object) -> tuple[str, ...]:
    if not isinstance(parameters, dict):
        return ()
    required = parameters.get("required")
    if not isinstance(required, list):
        return ()
    return tuple(str(item) for item in required if isinstance(item, str) and item)


def _safe_identifier(value: str) -> str:
    safe = re.sub(r"[^a-z0-9_]+", "_", value.casefold()).strip("_")
    return safe or "tool"


def _fit_tool_name(name: str) -> str:
    if len(name) <= MAX_TOOL_NAME_LENGTH:
        return name
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
    return f"{name[: MAX_TOOL_NAME_LENGTH - 9].rstrip('_')}_{digest}"


__all__ = ["McpExecuteTool", "mcp_runtime_tool_name"]
