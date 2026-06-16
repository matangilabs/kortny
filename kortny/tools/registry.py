"""Tool registry used by the agent loop and provider adapters."""

from __future__ import annotations

import concurrent.futures
import logging
from collections.abc import Iterable

from kortny.config import Settings
from kortny.tools.catalog import ToolDescriptor, tool_descriptor, tool_timeout_seconds
from kortny.tools.types import (
    JsonObject,
    JsonSchema,
    RecoverableToolError,
    Tool,
    ToolResult,
)

logger = logging.getLogger(__name__)

TOOL_TIMEOUT_ERROR_CODE = "tool_execution_timeout"


def invoke_tool_with_timeout(tool: Tool, args: JsonObject) -> ToolResult:
    """Invoke a tool, enforcing its catalog ``timeout_seconds`` deadline.

    HIG-195: native in-process tools have no inherent deadline, so an unbounded
    network call (or a wedged third-party client) could otherwise hang the worker
    until its queue lease expires, causing a duplicate execution on requeue. We
    run the synchronous ``invoke`` in a single-use thread and wait at most
    ``timeout_seconds``. On timeout we raise a recoverable timeout error so the
    coordinator's error policy can route recovery.

    KNOWN PYTHON LIMITATION: a timed-out thread cannot be forcibly killed, so the
    worker moves on while the underlying ``invoke`` thread may linger until it
    returns or its own inner (network/sandbox) timeout fires. That is acceptable
    because (a) the lingering thread's eventual result is *discarded* — we never
    read the future again and never record it, so it cannot corrupt task state —
    and (b) the queue lease + heartbeat (HIG-195 part B) remain the real defense.
    Sandbox/code tools set a deadline above their inner sandbox limit so the
    inner limit (which can clean up its container) fires first.
    """

    timeout_seconds = tool_timeout_seconds(tool.name)
    if timeout_seconds <= 0:
        return tool.invoke(args)

    # Single-use executor so each call's thread is isolated; we deliberately do
    # NOT block on shutdown (wait=False) when the call times out, letting the
    # daemon-like worker thread linger harmlessly with its discarded result.
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix=f"kortny-tool-{tool.name}",
    )
    future = executor.submit(tool.invoke, args)
    try:
        result = future.result(timeout=timeout_seconds)
    except concurrent.futures.TimeoutError as exc:
        # Discard the lingering computation: do not wait, do not read the future.
        future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        logger.warning(
            "tool invocation timed out tool=%s timeout_seconds=%s",
            tool.name,
            timeout_seconds,
        )
        raise RecoverableToolError(
            code=TOOL_TIMEOUT_ERROR_CODE,
            message=(
                f"{tool.name} did not finish within {timeout_seconds}s and was stopped."
            ),
            hint=(
                "Treat this tool call as unavailable for this run. Try a "
                "narrower request or a different tool; do not retry the same "
                "call unchanged."
            ),
            details={
                "tool": tool.name,
                "timeout_seconds": timeout_seconds,
            },
        ) from exc
    else:
        executor.shutdown(wait=False)
        return result


class ToolRegistryError(RuntimeError):
    """Base class for tool registry failures."""


class DuplicateToolError(ToolRegistryError):
    """Raised when a tool name is registered more than once."""


class ToolNotFoundError(ToolRegistryError):
    """Raised when a requested tool is not registered."""


class ToolRegistry:
    """In-memory registry for Kortny tools."""

    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        """Register a tool by name."""

        if not tool.name:
            raise ValueError("Tool name is required")
        if tool.name in self._tools:
            raise DuplicateToolError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        """Return a registered tool by name."""

        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolNotFoundError(f"Tool not registered: {name}") from exc

    def has(self, name: str) -> bool:
        """Return whether a tool is registered under ``name``."""

        return name in self._tools

    def invoke(self, name: str, args: JsonObject) -> ToolResult:
        """Invoke a registered tool under its catalog timeout deadline."""

        return invoke_tool_with_timeout(self.get(name), args)

    def schemas(self) -> tuple[JsonSchema, ...]:
        """Return provider-neutral tool declarations for LLM adapters."""

        return tuple(
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
            for tool in self._tools.values()
        )

    def descriptors(
        self,
        *,
        settings: Settings | None = None,
    ) -> tuple[ToolDescriptor, ...]:
        """Return metadata-rich descriptors for registered tools."""

        return tuple(
            tool_descriptor(tool, settings=settings) for tool in self._tools.values()
        )

    def names(self) -> tuple[str, ...]:
        """Return registered tool names in registration order."""

        return tuple(self._tools)
