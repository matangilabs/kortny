"""Provider-neutral tool selection models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ToolCard:
    """Compact tool description used before full schema registration."""

    registry_name: str
    provider: str
    display_name: str
    description: str
    capabilities: tuple[str, ...]
    side_effect: str
    toolkit_slug: str | None = None
    tool_slugs: tuple[str, ...] = ()
    visibility_scope_type: str | None = None
    visibility_scope_id: str | None = None
    can_replace_native_tools: tuple[str, ...] = ()

    def prompt_payload(self) -> dict[str, object]:
        """Return a compact JSON-safe payload for selector prompts."""

        return {
            "registry_name": self.registry_name,
            "provider": self.provider,
            "display_name": self.display_name,
            "description": self.description,
            "capabilities": list(self.capabilities),
            "side_effect": self.side_effect,
            "toolkit_slug": self.toolkit_slug,
            "tool_slugs": list(self.tool_slugs),
            "visibility_scope_type": self.visibility_scope_type,
            "can_replace_native_tools": list(self.can_replace_native_tools),
        }


@dataclass(frozen=True, slots=True)
class ToolSelection:
    """One selected tool with confidence and reason."""

    registry_name: str
    confidence: float
    reason: str


@dataclass(frozen=True, slots=True)
class ToolSelectionResult:
    """Ranked tool selection output for one task."""

    selected_tools: tuple[ToolSelection, ...] = ()
    suppressed_native_tools: tuple[str, ...] = ()
    rejected_tools: tuple[ToolSelection, ...] = ()
    route_reason: str = "no_external_tool_needed"
    fallback_used: bool = False

    @property
    def selected_names(self) -> tuple[str, ...]:
        return tuple(selection.registry_name for selection in self.selected_tools)
