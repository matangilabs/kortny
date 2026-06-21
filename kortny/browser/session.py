"""Persistent MCP browser session over Playwright-MCP (HIG-282).

Kortny's generic MCP client (kortny/mcp/client.py) is stateless per call --
it opens a fresh connection, runs one request, and closes. That model breaks
for a browser that must hold state across navigate->click->snapshot calls.

This module provides BrowserMcpSession: a dedicated per-task session that
holds ONE persistent MCP connection to a Playwright-MCP server, reused
across all browser tool calls, and closed at task end (chunk 3 wires the
lifecycle into the worker; here we just build and test the class).

Design: a background thread owns a private asyncio event loop. All async MCP
work runs on that loop. The public API is fully synchronous -- callers
dispatch via asyncio.run_coroutine_threadsafe(...).result(timeout=...).
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import threading
from dataclasses import dataclass
from typing import Any

from mcp import types as mcp_types
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from kortny.config import Settings


class BrowserSessionError(RuntimeError):
    """Raised on browser session transport or protocol failures."""


@dataclass(frozen=True, slots=True)
class BrowserToolResult:
    """Structured result from a Playwright-MCP tool call.

    Attributes:
        text: Concatenated text from all TextContent blocks.
        images: Decoded image bytes paired with MIME type, one per ImageContent.
        is_error: True when the MCP server flagged the call as an error.
    """

    text: str
    images: tuple[tuple[bytes, str], ...] = ()
    is_error: bool = False


class BrowserMcpSession:
    """A persistent MCP session to a Playwright-MCP server.

    Lifecycle:
        session = BrowserMcpSession(url, idle_timeout_seconds=120)
        session.open()   # starts background loop + MCP connection
        session.call_tool("browser_navigate", {"url": "https://example.com"})
        session.call_tool("browser_snapshot", {})
        session.close()  # tears down connection + loop + thread; idempotent

    Or use as a context manager:
        with BrowserMcpSession(url) as session:
            session.call_tool(...)
    """

    def __init__(
        self,
        url: str,
        *,
        idle_timeout_seconds: int = 120,
    ) -> None:
        if not url or not url.strip():
            raise BrowserSessionError("browser_url cannot be empty")
        self._url = url.strip()
        self._idle_timeout_seconds = idle_timeout_seconds
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._session: ClientSession | None = None
        self._open = False
        self._closed = False

    # ------------------------------------------------------------------
    # Public sync API
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Start the background loop and establish the MCP connection."""
        with self._lock:
            if self._closed:
                raise BrowserSessionError("Cannot reopen a closed BrowserMcpSession")
            if self._open:
                return
            self._start_background_loop()
            self._open = True

    def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        timeout_seconds: int | None = None,
    ) -> BrowserToolResult:
        """Call a Playwright-MCP tool and return a structured result.

        Returns a BrowserToolResult with:
          - text: concatenated TextContent blocks
          - images: base64-decoded ImageContent bytes + mime type
          - is_error: whether the MCP server flagged this as an error

        Raises BrowserSessionError on transport/protocol failure or if the
        session is not open / already closed.
        """
        self._require_open()
        timeout = (
            timeout_seconds
            if timeout_seconds is not None
            else self._idle_timeout_seconds
        )
        assert self._loop is not None  # guaranteed by _require_open
        future = asyncio.run_coroutine_threadsafe(
            self._async_call_tool(name, arguments),
            self._loop,
        )
        try:
            return future.result(timeout=timeout)
        except TimeoutError as exc:
            raise BrowserSessionError(
                f"Browser tool '{name}' timed out after {timeout}s"
            ) from exc
        except BrowserSessionError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise BrowserSessionError(
                f"Browser tool '{name}' failed: {type(exc).__name__}: {exc}"
            ) from exc

    def list_tools(self, *, timeout_seconds: int = 30) -> list[str]:
        """Return the names of tools the Playwright-MCP server exposes."""
        self._require_open()
        assert self._loop is not None
        future = asyncio.run_coroutine_threadsafe(
            self._async_list_tools(),
            self._loop,
        )
        try:
            return future.result(timeout=timeout_seconds)
        except BrowserSessionError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise BrowserSessionError(
                f"browser list_tools failed: {type(exc).__name__}: {exc}"
            ) from exc

    def close(self) -> None:
        """Tear down the MCP connection, stop the loop, join the thread.

        Idempotent -- safe to call multiple times or if open() was never called.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            loop = self._loop
            thread = self._thread

        if loop is not None:
            # Schedule async teardown then stop the loop
            asyncio.run_coroutine_threadsafe(self._async_close(), loop).result(
                timeout=10
            )
            loop.call_soon_threadsafe(loop.stop)

        if thread is not None:
            thread.join(timeout=15)

        with self._lock:
            self._session = None
            self._loop = None
            self._thread = None
            self._open = False

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> BrowserMcpSession:
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal async helpers (run on background loop)
    # ------------------------------------------------------------------

    async def _async_connect(self) -> None:
        """Open the streamable-HTTP MCP client and initialize the session.

        This coroutine is scheduled on the background loop via
        run_coroutine_threadsafe from _start_background_loop.

        IMPORTANT: The streamablehttp_client context manager must remain open
        for the lifetime of the session. We hold the context manager open by
        storing its __aexit__ in _cm_exit and the ClientSession in _session.
        """
        # We can't use `async with` here because we need to keep the
        # context managers open after this coroutine returns. Instead we
        # manually call __aenter__ and save __aexit__ for cleanup.
        cm = streamablehttp_client(self._url)
        (read, write, _get_session_id) = await cm.__aenter__()
        self._transport_cm = cm
        self._transport_read = read
        self._transport_write = write

        session_cm = ClientSession(read, write)
        session = await session_cm.__aenter__()
        self._session_cm = session_cm
        await session.initialize()
        self._session = session

    async def _async_close(self) -> None:
        """Cleanly close the session and transport."""
        session_cm = getattr(self, "_session_cm", None)
        if session_cm is not None:
            with contextlib.suppress(Exception):
                await session_cm.__aexit__(None, None, None)
        transport_cm = getattr(self, "_transport_cm", None)
        if transport_cm is not None:
            with contextlib.suppress(Exception):
                await transport_cm.__aexit__(None, None, None)
        self._session = None

    async def _async_call_tool(
        self, name: str, arguments: dict[str, Any]
    ) -> BrowserToolResult:
        session = self._session
        if session is None:
            raise BrowserSessionError("Browser session is not connected")
        result = await session.call_tool(name, arguments)
        return _parse_tool_result(result)

    async def _async_list_tools(self) -> list[str]:
        session = self._session
        if session is None:
            raise BrowserSessionError("Browser session is not connected")
        result = await session.list_tools()
        return [tool.name for tool in result.tools]

    # ------------------------------------------------------------------
    # Background loop management
    # ------------------------------------------------------------------

    def _start_background_loop(self) -> None:
        """Spin up a daemon thread with its own event loop, then connect."""
        loop = asyncio.new_event_loop()
        self._loop = loop

        # connect_done is set by the background thread once _async_connect
        # finishes (or fails); connect_error holds the exception if it failed.
        connect_done = threading.Event()
        connect_error: list[BaseException] = []

        def _run() -> None:
            asyncio.set_event_loop(loop)

            # Schedule the connect coroutine and wait for it
            async def _connect_and_signal() -> None:
                try:
                    await self._async_connect()
                except BaseException as exc:  # noqa: BLE001
                    connect_error.append(exc)
                finally:
                    connect_done.set()

            loop.run_until_complete(_connect_and_signal())
            # After connect, run the loop forever (processing tool calls etc.)
            loop.run_forever()
            loop.close()

        thread = threading.Thread(target=_run, daemon=True, name="browser-mcp-loop")
        self._thread = thread
        thread.start()

        # Wait for the connect phase (up to idle_timeout_seconds)
        if not connect_done.wait(timeout=self._idle_timeout_seconds):
            raise BrowserSessionError(
                f"Browser MCP connection timed out after {self._idle_timeout_seconds}s"
            )
        if connect_error:
            exc = connect_error[0]
            raise BrowserSessionError(
                f"Browser MCP connection failed: {type(exc).__name__}: {exc}"
            ) from exc

    def _require_open(self) -> None:
        if self._closed:
            raise BrowserSessionError("BrowserMcpSession is closed")
        if not self._open:
            raise BrowserSessionError(
                "BrowserMcpSession is not open -- call open() first"
            )


def _parse_tool_result(result: mcp_types.CallToolResult) -> BrowserToolResult:
    """Parse MCP CallToolResult into a structured BrowserToolResult.

    - TextContent blocks are joined into a single ``text`` string.
    - ImageContent blocks are base64-decoded to raw bytes and paired with
      their MIME type; each becomes one entry in ``images``.
    - ``is_error`` mirrors ``result.isError``.
    """
    text_parts: list[str] = []
    images: list[tuple[bytes, str]] = []

    for block in result.content:
        if isinstance(block, mcp_types.TextContent):
            text_parts.append(block.text)
        elif isinstance(block, mcp_types.ImageContent):
            raw = base64.b64decode(block.data)
            images.append((raw, block.mimeType))

    return BrowserToolResult(
        text="\n".join(p for p in text_parts if p),
        images=tuple(images),
        is_error=bool(result.isError),
    )


def open_browser_session(settings: Settings) -> BrowserMcpSession | None:
    """Return an open BrowserMcpSession if browser is enabled, else None.

    When KORTNY_BROWSER_URL is unset, this returns None and callers
    should no-op gracefully (browser tools unavailable).
    """
    url = settings.browser_url
    if not url:
        return None
    session = BrowserMcpSession(
        url,
        idle_timeout_seconds=settings.browser_session_idle_timeout_seconds,
    )
    session.open()
    return session


__all__ = [
    "BrowserMcpSession",
    "BrowserSessionError",
    "BrowserToolResult",
    "open_browser_session",
]
