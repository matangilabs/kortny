"""Provider-neutral tool contract."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Protocol, TypeAlias

JsonObject: TypeAlias = dict[str, Any]
JsonSchema: TypeAlias = dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolArtifact:
    """A file-like artifact produced by a tool invocation."""

    filename: str
    path: str | None = None
    mime_type: str | None = None
    size_bytes: int | None = None


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Result returned by every Kortny tool."""

    output: JsonObject
    cost_usd: Decimal = Decimal("0")
    artifacts: tuple[ToolArtifact, ...] = ()


class RecoverableToolError(RuntimeError):
    """A tool failure the coordinator should feed back to the model."""

    def __init__(
        self,
        *,
        code: str,
        message: str,
        hint: str | None = None,
        details: JsonObject | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint
        self.details = details or {}

    def to_payload(self) -> JsonObject:
        """Return the structured error payload expected by the agent loop."""

        payload: JsonObject = {
            "code": self.code,
            "message": self.message,
            "recoverable": True,
        }
        if self.hint:
            payload["hint"] = self.hint
        if self.details:
            payload["details"] = self.details
        return payload


class Tool(Protocol):
    """The interface every native or external tool adapter implements."""

    name: str
    description: str
    parameters: JsonSchema

    def invoke(self, args: JsonObject) -> ToolResult:
        """Run the tool with JSON arguments and return a structured result."""
        ...
