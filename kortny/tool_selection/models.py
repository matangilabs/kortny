"""Provider-neutral tool selection models."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_PROMPT_DESCRIPTION_CHARS = 280


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
    tool_count: int | None = None
    required_fields: tuple[str, ...] = ()
    visibility_scope_type: str | None = None
    visibility_scope_id: str | None = None
    can_replace_native_tools: tuple[str, ...] = ()
    enriched_description: str | None = None

    def prompt_payload(
        self,
        *,
        max_description_chars: int | None = None,
    ) -> dict[str, object]:
        """Return a compact JSON-safe payload for selector prompts."""

        description = self.description
        if max_description_chars is not None:
            description = _shorten(description, max_chars=max_description_chars)
        return {
            "registry_name": self.registry_name,
            "provider": self.provider,
            "display_name": self.display_name,
            "description": description,
            "capabilities": list(self.capabilities),
            "side_effect": self.side_effect,
            "toolkit_slug": self.toolkit_slug,
            "tool_slugs": list(self.tool_slugs),
            "tool_count": self.tool_count,
            "required_fields": list(self.required_fields),
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
    prompt_chars: int | None = None
    prompt_char_budget: int | None = None
    budget_omitted_candidate_names: tuple[str, ...] = ()

    @property
    def selected_names(self) -> tuple[str, ...]:
        return tuple(selection.registry_name for selection in self.selected_tools)


def _shorten(value: str, *, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    if max_chars <= 3:
        return value[:max_chars]
    return value[: max_chars - 3].rstrip() + "..."
