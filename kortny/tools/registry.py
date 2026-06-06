"""Tool registry used by the agent loop and provider adapters."""

from __future__ import annotations

from collections.abc import Iterable

from kortny.config import Settings
from kortny.tools.catalog import ToolDescriptor, tool_descriptor
from kortny.tools.types import JsonObject, JsonSchema, Tool, ToolResult


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

    def invoke(self, name: str, args: JsonObject) -> ToolResult:
        """Invoke a registered tool."""

        return self.get(name).invoke(args)

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
