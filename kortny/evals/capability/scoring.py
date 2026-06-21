"""Pure scoring for the capability-grounding eval (HIG-274).

No DB/LLM — given cases + classify/floor callables, run structured assertions
and return a typed report. CI exercises the scorer with fake callables; the
live runner (runner.py) wires real classifiers and an offline floor stub.

Assertions per case:

1. **Likely tools constraint**: for each toolkit in ``expected_in_likely_tools``,
   it must appear in ``likely_tools`` OR ``toolkit_affinity``.
2. **Needs-connection constraint**: for each category in
   ``expected_in_needs_connection``, it must appear in ``needs_connection``; and
   none of those categories should appear in ``likely_tools`` (they cannot be
   simultaneously "needs connection" and "already surfaced as likely").
3. **Floor constraint**: for each toolkit in ``expected_floor_toolkits``,
   ``floor_fn`` must return ≥1 tool name.
4. **Scope constraint**: for each toolkit in ``forbidden_scope_toolkits``,
   ``floor_fn`` must return 0 tools whose name contains the forbidden slug.
5. **Global cap**: total distinct tools loaded by ``floor_fn`` must be ≤ 24.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from kortny.evals.capability.cases import CapabilityCase

# Given a CapabilityCase, return (likely_tools, toolkit_affinity, needs_connection).
ClassifyFn = Callable[
    [CapabilityCase],
    tuple[list[str], tuple[str, ...], tuple[str, ...]],
]

# Given a CapabilityCase and the implied toolkits (union of
# expected_floor_toolkits), return a list of loaded tool names.
FloorFn = Callable[[CapabilityCase, tuple[str, ...]], list[str]]

_GLOBAL_TOOL_CAP = 24


@dataclass(frozen=True, slots=True)
class CapabilityAssertionResult:
    case_id: int
    request: str
    passed: bool
    failures: tuple[str, ...]  # what failed and why


@dataclass(frozen=True, slots=True)
class CapabilityReport:
    case_count: int
    passed: int
    failed: int
    results: tuple[CapabilityAssertionResult, ...]

    def summary_line(self) -> str:
        return f"passed={self.passed}/{self.case_count} failed={self.failed}"

    @property
    def failures(self) -> tuple[CapabilityAssertionResult, ...]:
        return tuple(r for r in self.results if not r.passed)


def score_capability(
    cases: Sequence[CapabilityCase],
    classify_fn: ClassifyFn,
    floor_fn: FloorFn,
) -> CapabilityReport:
    """Run all assertions across all cases and return a typed report."""

    results: list[CapabilityAssertionResult] = []

    for idx, case in enumerate(cases):
        failures: list[str] = []

        # --- invoke callables ---
        likely_tools, toolkit_affinity, needs_connection = classify_fn(case)
        loaded_tools = floor_fn(case, case.expected_floor_toolkits)

        # Normalize to lower-case sets for membership checks.
        likely_lower = {t.casefold() for t in likely_tools}
        affinity_lower = {t.casefold() for t in toolkit_affinity}
        needs_lower = {c.casefold() for c in needs_connection}
        loaded_lower = [t.casefold() for t in loaded_tools]

        # 1. Likely-tools constraint
        for toolkit in case.expected_in_likely_tools:
            tk = toolkit.casefold()
            if tk not in likely_lower and tk not in affinity_lower:
                failures.append(
                    f"expected '{toolkit}' in likely_tools or toolkit_affinity, "
                    f"got likely={sorted(likely_lower)} affinity={sorted(affinity_lower)}"
                )

        # 2. Needs-connection constraint
        for category in case.expected_in_needs_connection:
            cat = category.casefold()
            if cat not in needs_lower:
                failures.append(
                    f"expected '{category}' in needs_connection, got {sorted(needs_lower)}"
                )
            # The category must NOT also appear in likely_tools (would be contradictory).
            if cat in likely_lower:
                failures.append(
                    f"'{category}' in needs_connection appeared in likely_tools (must not)"
                )

        # 3. Floor constraint — each expected floor toolkit must have ≥1 tool.
        for toolkit in case.expected_floor_toolkits:
            tk = toolkit.casefold()
            has_tool = any(tk in tool_name for tool_name in loaded_lower)
            if not has_tool:
                failures.append(
                    f"floor loaded 0 tools for connected toolkit '{toolkit}'"
                )

        # 4. Scope constraint — forbidden toolkits must contribute 0 tools.
        for forbidden in case.forbidden_scope_toolkits:
            fb = forbidden.casefold()
            leaked = [t for t in loaded_lower if fb in t]
            if leaked:
                failures.append(f"floor loaded out-of-scope toolkit '{forbidden}'")

        # 5. Global cap
        total_tools = len(loaded_lower)
        if total_tools > _GLOBAL_TOOL_CAP:
            failures.append(
                f"floor loaded {total_tools} tools, exceeds global cap of {_GLOBAL_TOOL_CAP}"
            )

        results.append(
            CapabilityAssertionResult(
                case_id=idx + 1,
                request=case.request,
                passed=len(failures) == 0,
                failures=tuple(failures),
            )
        )

    passed = sum(1 for r in results if r.passed)
    return CapabilityReport(
        case_count=len(results),
        passed=passed,
        failed=len(results) - passed,
        results=tuple(results),
    )
