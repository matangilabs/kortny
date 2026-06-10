"""Tiny stdio MCP server fixture for Kortny MCP integration tests.

Exposes two tools over stdio:
- ``echo(text)`` annotated ``readOnlyHint=True`` (maps to side_effect="read")
- ``write_note(text)`` unannotated (defaults to side_effect="write")

Run directly via ``python tests/fixtures/mcp/echo_server.py`` (the tests spawn
it with ``command=sys.executable, args=[<this file>]``).
"""

from __future__ import annotations

import os

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

server = FastMCP("kortny-echo-fixture")


@server.tool(
    description="Echo the provided text back to the caller.",
    annotations=ToolAnnotations(readOnlyHint=True),
)
def echo(text: str) -> str:
    """Return the input text unchanged."""

    return f"echo: {text}"


@server.tool(description="Record a note (simulated write).")
def write_note(text: str) -> str:
    """Pretend to persist a note and acknowledge."""

    return f"noted: {text}"


@server.tool(
    description="Return this server process's PID (for session-reuse tests).",
    annotations=ToolAnnotations(readOnlyHint=True),
)
def server_pid() -> str:
    """Report the OS process id of this stdio server subprocess."""

    return f"pid: {os.getpid()}"


@server.tool(
    description="Return a payload of approximately the requested character size.",
    annotations=ToolAnnotations(readOnlyHint=True),
)
def big_echo(size: int) -> str:
    """Return a string of length ``size`` (capped) for result-budget tests."""

    capped = max(0, min(int(size), 5_000_000))
    return "x" * capped


if __name__ == "__main__":
    server.run()
