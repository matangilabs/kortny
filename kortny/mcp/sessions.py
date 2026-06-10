"""Per-task MCP session reuse over a single dedicated event loop.

The stateless :mod:`kortny.mcp.client` helpers open a fresh transport +
``ClientSession`` per call; for stdio servers that spawns a subprocess every
time (~1s of latency). ``McpSessionManager`` keeps one initialized session per
server alive for the lifetime of a task and reuses it across tool calls.

MCP transports and ``ClientSession`` are anyio-based async context managers
whose cancel scopes MUST be entered AND exited in the *same* asyncio task. A
naive ``AsyncExitStack`` entered on one ``run_coroutine_threadsafe`` coroutine
and ``aclose()``d on another later coroutine runs the enter/exit in different
tasks and raises ``Attempted to exit cancel scope in a different task``.

To respect that, each server gets a dedicated long-lived *owner task* on our
background loop. That task opens the transport + session, then services tool
calls from an asyncio queue, and finally exits the context managers itself when
asked to close — so enter and exit always share one task. The synchronous
public API submits work to the owner task and blocks on the result.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Coroutine
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, TypeVar

from mcp.client.session import ClientSession

from kortny.db.models import McpServer
from kortny.mcp.client import (
    McpClientError,
    McpToolCallResult,
    _decrypt_secret_env,
    _open_transport,
    _tool_call_result,
)

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

# Margin added to the per-call async timeout for the blocking ``.result()``
# wait, so the inner ``asyncio.wait_for`` is what actually fires on timeout.
_RESULT_MARGIN_SECONDS = 5.0
# Bounded join when stopping the loop thread / owner tasks on ``close()``.
_THREAD_JOIN_SECONDS = 5.0


@dataclass(slots=True)
class _CallRequest:
    """One queued tool call awaiting its owner task."""

    tool_name: str
    arguments: dict
    timeout_seconds: float
    future: asyncio.Future[McpToolCallResult]


@dataclass(slots=True)
class _SessionOwner:
    """Handle to a per-server owner task and its work queue."""

    server: McpServer
    encryption_key: str
    queue: asyncio.Queue[_CallRequest | None] = field(default_factory=asyncio.Queue)
    ready: asyncio.Future[None] | None = None
    task: asyncio.Task[None] | None = None


class McpSessionManager:
    """Reuse one initialized MCP session per server across tool calls.

    Lifecycle: created per task (one per :class:`McpExternalToolProvider`),
    used for any number of ``call_tool`` invocations, then ``close()``d when the
    task finishes. ``close()`` is idempotent and never raises.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._owners: dict[str, _SessionOwner] = {}
        self._closed = False

    # -- public sync API ----------------------------------------------------

    def call_tool(
        self,
        server: McpServer,
        tool_name: str,
        arguments: dict,
        *,
        encryption_key: str,
        timeout_seconds: float,
    ) -> McpToolCallResult:
        """Call one tool, reusing (or lazily opening) this server's session.

        On a transport/protocol error the cached session is dropped and one
        reconnect is attempted before surfacing :class:`McpClientError`.
        """

        if self._closed:
            raise McpClientError(
                f"MCP session manager is closed; cannot call '{server.name}'"
            )
        key = self._server_key(server)

        try:
            return self._submit(
                self._call_via_owner(
                    key,
                    server,
                    tool_name,
                    arguments,
                    encryption_key=encryption_key,
                    timeout_seconds=timeout_seconds,
                ),
                timeout=timeout_seconds,
            )
        except TimeoutError as exc:
            self._drop_session(key)
            raise McpClientError(
                f"MCP server '{server.name}' timed out after {timeout_seconds}s"
            ) from exc
        except McpClientError:
            self._drop_session(key)
            raise
        except Exception as exc:  # noqa: BLE001 - transport/proto error -> reconnect once
            logger.info(
                "mcp session call failed, attempting one reconnect server=%s tool=%s error=%s",
                server.name,
                tool_name,
                exc,
            )
            self._drop_session(key)
            try:
                return self._submit(
                    self._call_via_owner(
                        key,
                        server,
                        tool_name,
                        arguments,
                        encryption_key=encryption_key,
                        timeout_seconds=timeout_seconds,
                    ),
                    timeout=timeout_seconds,
                )
            except Exception as retry_exc:  # noqa: BLE001 - normalize after retry
                self._drop_session(key)
                raise McpClientError(
                    f"MCP server '{server.name}' ({server.transport}) failed after "
                    f"reconnect: {retry_exc}"
                ) from retry_exc

    def close(self) -> None:
        """Close every cached session, stop the loop, and join the thread.

        Idempotent and never raises.
        """

        with self._lock:
            if self._closed:
                return
            self._closed = True
            loop = self._loop
            thread = self._thread
            owners = self._owners
            self._owners = {}

        if loop is not None and not loop.is_closed():
            try:
                future = asyncio.run_coroutine_threadsafe(
                    self._aclose_all(owners), loop
                )
                future.result(timeout=_THREAD_JOIN_SECONDS)
            except Exception as exc:  # noqa: BLE001 - best-effort teardown
                logger.warning("mcp session manager teardown error: %s", exc)
            finally:
                loop.call_soon_threadsafe(loop.stop)

        if thread is not None:
            thread.join(timeout=_THREAD_JOIN_SECONDS)
            if thread.is_alive():
                logger.warning("mcp session manager loop thread did not stop in time")

    # -- loop ownership -----------------------------------------------------

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._closed:
                raise McpClientError("MCP session manager is closed")
            if self._loop is not None:
                return self._loop
            loop = asyncio.new_event_loop()
            thread = threading.Thread(
                target=self._run_loop,
                args=(loop,),
                name="mcp-session-loop",
                daemon=True,
            )
            thread.start()
            self._loop = loop
            self._thread = thread
            return loop

    @staticmethod
    def _run_loop(loop: asyncio.AbstractEventLoop) -> None:
        asyncio.set_event_loop(loop)
        try:
            loop.run_forever()
        finally:
            try:
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
            except Exception:  # noqa: BLE001 - shutdown best effort
                pending = set()
            loop.close()

    def _submit(self, coro: Coroutine[Any, Any, _T], *, timeout: float) -> _T:
        loop = self._get_loop()
        future: Future[_T] = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result(timeout=timeout + _RESULT_MARGIN_SECONDS)

    # -- owner orchestration (runs on the loop thread) ----------------------

    async def _call_via_owner(
        self,
        key: str,
        server: McpServer,
        tool_name: str,
        arguments: dict,
        *,
        encryption_key: str,
        timeout_seconds: float,
    ) -> McpToolCallResult:
        owner = await self._ensure_owner(key, server, encryption_key=encryption_key)
        loop = asyncio.get_running_loop()
        request = _CallRequest(
            tool_name=tool_name,
            arguments=arguments,
            timeout_seconds=timeout_seconds,
            future=loop.create_future(),
        )
        await owner.queue.put(request)
        return await request.future

    async def _ensure_owner(
        self,
        key: str,
        server: McpServer,
        *,
        encryption_key: str,
    ) -> _SessionOwner:
        owner = self._owners.get(key)
        if owner is not None and owner.task is not None and not owner.task.done():
            return owner

        loop = asyncio.get_running_loop()
        owner = _SessionOwner(server=server, encryption_key=encryption_key)
        owner.ready = loop.create_future()
        owner.task = loop.create_task(self._run_owner(owner))
        self._owners[key] = owner
        # Surface a failed startup (e.g. bad transport) to the caller.
        assert owner.ready is not None
        await owner.ready
        return owner

    async def _run_owner(self, owner: _SessionOwner) -> None:
        """Own one server's session for its whole lifetime in a single task."""

        assert owner.ready is not None
        ready = owner.ready
        secrets = _decrypt_secret_env(owner.server, encryption_key=owner.encryption_key)
        try:
            async with (
                _open_transport(owner.server, secrets=secrets) as (read, write),
                ClientSession(read, write) as session,
            ):
                await session.initialize()
                if not ready.done():
                    ready.set_result(None)
                await self._service_calls(owner, session)
        except BaseException as exc:  # noqa: BLE001 - propagate startup failures
            if not ready.done():
                ready.set_exception(
                    exc if isinstance(exc, Exception) else McpClientError(str(exc))
                )
            else:
                # Already serving: drain queued requests with the failure.
                self._fail_pending(owner, exc)
            # Re-raise non-Exception (cancellation) so the loop unwinds cleanly.
            if not isinstance(exc, Exception):
                raise

    async def _service_calls(
        self,
        owner: _SessionOwner,
        session: ClientSession,
    ) -> None:
        while True:
            request = await owner.queue.get()
            if request is None:  # close sentinel
                return
            if request.future.done():
                continue
            try:
                result = await asyncio.wait_for(
                    session.call_tool(request.tool_name, request.arguments),
                    timeout=request.timeout_seconds,
                )
                request.future.set_result(_tool_call_result(result))
            except BaseException as exc:  # noqa: BLE001 - surface to caller
                if not request.future.done():
                    request.future.set_exception(
                        exc if isinstance(exc, Exception) else McpClientError(str(exc))
                    )
                # A failed call may have corrupted the session; stop serving so
                # the owner exits its context cleanly and the caller reconnects.
                if not isinstance(exc, TimeoutError):
                    return

    def _fail_pending(self, owner: _SessionOwner, exc: BaseException) -> None:
        while not owner.queue.empty():
            try:
                request = owner.queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            if request is not None and not request.future.done():
                request.future.set_exception(
                    exc if isinstance(exc, Exception) else McpClientError(str(exc))
                )

    def _drop_session(self, key: str) -> None:
        """Tear down a single cached session on the loop thread (best effort)."""

        loop = self._loop
        owner = self._owners.pop(key, None)
        if owner is None:
            return
        if loop is None or loop.is_closed():
            return
        try:
            future = asyncio.run_coroutine_threadsafe(self._aclose_owner(owner), loop)
            future.result(timeout=_THREAD_JOIN_SECONDS)
        except Exception as exc:  # noqa: BLE001 - best-effort drop
            logger.warning("mcp session drop error server_key=%s: %s", key, exc)

    async def _aclose_owner(self, owner: _SessionOwner) -> None:
        task = owner.task
        if task is None or task.done():
            return
        # Ask the owner task to exit its context manager in its own task.
        await owner.queue.put(None)
        try:
            await asyncio.wait_for(task, timeout=_THREAD_JOIN_SECONDS)
        except (TimeoutError, asyncio.CancelledError):
            task.cancel()
            try:
                await task
            except BaseException as exc:  # noqa: BLE001 - cancellation teardown
                logger.debug("mcp owner cancel teardown: %s", exc)

    async def _aclose_all(self, owners: dict[str, _SessionOwner]) -> None:
        for owner in owners.values():
            try:
                await self._aclose_owner(owner)
            except Exception as exc:  # noqa: BLE001 - teardown best effort
                logger.warning("mcp session aclose error: %s", exc)

    @staticmethod
    def _server_key(server: McpServer) -> str:
        return str(server.id) if server.id is not None else server.name


__all__ = ["McpSessionManager"]
