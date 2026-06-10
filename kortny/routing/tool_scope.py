"""Tool exposure funnel helpers."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from kortny.tools.types import JsonObject

NATIVE_TOOL_SCOPE_APPLIED_MESSAGE = "native_tool_scope_applied"
SCHEDULE_MUTATION_TOOL_NAMES = frozenset(
    {
        "create_schedule",
        "update_schedule",
        "pause_schedule",
        "resume_schedule",
        "cancel_schedule",
    }
)
SCHEDULE_TOOL_NAMES = SCHEDULE_MUTATION_TOOL_NAMES | {
    "list_schedules",
    "get_schedule",
}


class ScopedTool(Protocol):
    """Minimal tool shape needed by the native scoping policy."""

    name: str


@dataclass(frozen=True, slots=True)
class NativeToolScopeDecision:
    """Native tool exposure decision for one task."""

    selected_tools: tuple[ScopedTool, ...]
    original_tool_names: tuple[str, ...]
    selected_tool_names: tuple[str, ...]
    suppressed_tool_names: tuple[str, ...]
    reason_codes: tuple[str, ...]
    schedule_mutation_allowed: bool
    intent_classification: str | None
    likely_tools: tuple[str, ...]

    def to_payload(self) -> JsonObject:
        return {
            "message": NATIVE_TOOL_SCOPE_APPLIED_MESSAGE,
            "original_tool_count": len(self.original_tool_names),
            "selected_tool_count": len(self.selected_tool_names),
            "suppressed_tool_count": len(self.suppressed_tool_names),
            "original_tool_names": list(self.original_tool_names),
            "selected_tool_names": list(self.selected_tool_names),
            "suppressed_tool_names": list(self.suppressed_tool_names),
            "reason_codes": list(self.reason_codes),
            "schedule_mutation_allowed": self.schedule_mutation_allowed,
            "intent_classification": self.intent_classification,
            "likely_tools": list(self.likely_tools),
        }


class NativeToolScopePolicy:
    """Apply hard native-tool exposure policy before semantic selection."""

    def apply(
        self,
        *,
        tools: Sequence[ScopedTool],
        task_input: str,
        intent_decision: Mapping[str, object] | None = None,
    ) -> NativeToolScopeDecision:
        original_tools = tuple(tools)
        original_names = tuple(tool.name for tool in original_tools)
        likely_tools = _likely_tools(intent_decision)
        schedule_mutation_allowed = _schedule_mutation_allowed(
            task_input=task_input,
            likely_tools=likely_tools,
        )
        suppressed_names: tuple[str, ...] = ()
        reason_codes: tuple[str, ...]
        if schedule_mutation_allowed:
            reason_codes = ("schedule_mutation_tools_allowed",)
        else:
            suppressed_names = tuple(
                name for name in original_names if name in SCHEDULE_MUTATION_TOOL_NAMES
            )
            reason_codes = (
                ("schedule_mutation_tools_hidden",)
                if suppressed_names
                else ("no_native_scope_changes",)
            )

        suppressed = set(suppressed_names)
        selected_tools = tuple(
            tool for tool in original_tools if tool.name not in suppressed
        )
        selected_names = tuple(tool.name for tool in selected_tools)
        return NativeToolScopeDecision(
            selected_tools=selected_tools,
            original_tool_names=original_names,
            selected_tool_names=selected_names,
            suppressed_tool_names=suppressed_names,
            reason_codes=reason_codes,
            schedule_mutation_allowed=schedule_mutation_allowed,
            intent_classification=_optional_string(
                intent_decision.get("classification")
                if intent_decision is not None
                else None
            ),
            likely_tools=likely_tools,
        )


def _schedule_mutation_allowed(
    *,
    task_input: str,
    likely_tools: tuple[str, ...],
) -> bool:
    likely_tool_names = set(likely_tools)
    if likely_tool_names & SCHEDULE_TOOL_NAMES:
        return True
    if any(
        tool.startswith("schedule") or tool.startswith("scheduler")
        for tool in likely_tool_names
    ):
        return True
    return bool(_SCHEDULE_MANAGEMENT_RE.search(task_input.casefold()))


def _likely_tools(intent_decision: Mapping[str, object] | None) -> tuple[str, ...]:
    if intent_decision is None:
        return ()
    return tuple(sorted(_string_set(intent_decision.get("likely_tools"))))


def _string_set(value: object) -> set[str]:
    if not isinstance(value, list | tuple | set):
        return set()
    return {item for item in value if isinstance(item, str) and item}


def _optional_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


_SCHEDULE_MANAGEMENT_RE = re.compile(
    r"\b(every|daily|weekly|monthly|schedule|scheduled|schedules|recurring|"
    r"cron|run every|each\s+"
    r"(?:day|week|month|monday|tuesday|wednesday|thursday|friday|"
    r"saturday|sunday))\b"
)
