"""Pure scoring for the cross-app orchestration eval.

No DB/LLM — given cases + a run callable, run three structured assertions per
case and return a typed report. CI exercises the scorer with a stub run_fn; the
live runner (runner.py) wires a real agent execution.

Assertions per case:

1. **apps_pass**: every toolkit slug in ``expected_apps`` must appear in the
   set of apps actually called. Failure here means the agent skipped a required
   integration.
2. **tools_pass**: when ``must_use_tools`` is True, the called-apps set must be
   non-empty (at least one tool was invoked). Failure here is the context-leak
   guard — the agent answered from injected episodic/KG context without ever
   calling a real tool.
3. **scope_pass**: the intersection of called_apps and ``forbidden_apps`` must
   be empty. Failure means the agent called an out-of-scope integration.

A case passes iff all three assertions pass. The report gives per-case
breakdowns so a glance shows *which* assertion failed and *what* was called vs
expected.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from kortny.evals.orchestration.cases import OrchestrationCase

# Given an OrchestrationCase, return the set of Composio toolkit slugs whose
# tools were actually called during execution, plus a flag for whether ANY tool
# at all was invoked (for the must_use_tools guard) and the final answer text
# (optional, for debugging).
RunResult = tuple[frozenset[str], bool, str]  # (called_apps, any_tool_called, answer)

RunFn = Callable[[OrchestrationCase], RunResult]


@dataclass(frozen=True, slots=True)
class OrchestrationAssertionResult:
    case_id: int
    request: str
    passed: bool
    apps_pass: bool
    tools_pass: bool
    scope_pass: bool
    expected_apps: tuple[str, ...]
    called_apps: frozenset[str]
    any_tool_called: bool
    failures: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class OrchestrationReport:
    case_count: int
    passed: int
    failed: int
    pass_rate: float
    results: tuple[OrchestrationAssertionResult, ...]

    def summary_line(self) -> str:
        pct = f"{self.pass_rate * 100:.0f}%"
        return f"passed={self.passed}/{self.case_count} failed={self.failed} ({pct})"

    @property
    def failures(self) -> tuple[OrchestrationAssertionResult, ...]:
        return tuple(r for r in self.results if not r.passed)


def score_orchestration(
    cases: Sequence[OrchestrationCase],
    run_fn: RunFn,
) -> OrchestrationReport:
    """Run all assertions across all cases and return a typed report."""

    results: list[OrchestrationAssertionResult] = []

    for idx, case in enumerate(cases):
        called_apps, any_tool_called, _answer = run_fn(case)

        # Normalize to lower-case sets for membership checks.
        called_lower = frozenset(a.casefold() for a in called_apps)
        expected_lower = frozenset(a.casefold() for a in case.expected_apps)
        forbidden_lower = frozenset(a.casefold() for a in case.forbidden_apps)

        # 1. apps_pass: every expected app must appear in called apps.
        missing = expected_lower - called_lower
        apps_pass = len(missing) == 0

        # 2. tools_pass: when must_use_tools, at least one tool must be called.
        tools_pass = (not case.must_use_tools) or any_tool_called

        # 3. scope_pass: no forbidden app must appear in called apps.
        leaked = called_lower & forbidden_lower
        scope_pass = len(leaked) == 0

        failures: list[str] = []
        if not apps_pass:
            failures.append(
                f"expected apps not called: {sorted(missing)!r}; "
                f"got called={sorted(called_lower)!r}"
            )
        if not tools_pass:
            failures.append(
                "must_use_tools=True but no tool was called "
                "(answer likely came from injected context)"
            )
        if not scope_pass:
            failures.append(f"forbidden apps were called: {sorted(leaked)!r}")

        case_passed = apps_pass and tools_pass and scope_pass
        results.append(
            OrchestrationAssertionResult(
                case_id=idx + 1,
                request=case.request,
                passed=case_passed,
                apps_pass=apps_pass,
                tools_pass=tools_pass,
                scope_pass=scope_pass,
                expected_apps=case.expected_apps,
                called_apps=called_lower,
                any_tool_called=any_tool_called,
                failures=tuple(failures),
            )
        )

    passed = sum(1 for r in results if r.passed)
    total = len(results)
    return OrchestrationReport(
        case_count=total,
        passed=passed,
        failed=total - passed,
        pass_rate=passed / total if total > 0 else 0.0,
        results=tuple(results),
    )
