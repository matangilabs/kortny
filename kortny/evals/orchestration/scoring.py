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
from typing import NamedTuple

from kortny.evals.orchestration.cases import OrchestrationCase


class RunResult(NamedTuple):
    """Result of running a single orchestration case.

    The live runner returns a ``RunResult`` for every case, including skipped
    ones. The scorer handles both new-style ``RunResult`` NamedTuple values and
    legacy 3-tuple returns for backward compatibility.
    """

    called_apps: frozenset[str]
    any_tool_called: bool
    answer: str
    skipped: bool = False
    skip_reason: str = ""


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
    skipped: bool = False
    skip_reason: str = ""


@dataclass(frozen=True, slots=True)
class OrchestrationReport:
    case_count: int
    passed: int
    failed: int
    skipped: int
    pass_rate: float
    results: tuple[OrchestrationAssertionResult, ...]

    def summary_line(self) -> str:
        pct = f"{self.pass_rate * 100:.0f}%"
        skip_part = f" {self.skipped} skipped" if self.skipped else ""
        return (
            f"passed={self.passed}/{self.case_count - self.skipped} "
            f"failed={self.failed} ({pct}){skip_part}"
        )

    @property
    def failures(self) -> tuple[OrchestrationAssertionResult, ...]:
        return tuple(r for r in self.results if not r.passed and not r.skipped)


def score_orchestration(
    cases: Sequence[OrchestrationCase],
    run_fn: RunFn,
) -> OrchestrationReport:
    """Run all assertions across all cases and return a typed report."""

    results: list[OrchestrationAssertionResult] = []

    for idx, case in enumerate(cases):
        run_result = run_fn(case)

        called_apps = run_result.called_apps
        any_tool_called = run_result.any_tool_called
        _answer = run_result.answer
        is_skipped = run_result.skipped
        skip_reason = run_result.skip_reason

        if is_skipped:
            results.append(
                OrchestrationAssertionResult(
                    case_id=idx + 1,
                    request=case.request,
                    passed=True,
                    apps_pass=True,
                    tools_pass=True,
                    scope_pass=True,
                    expected_apps=case.expected_apps,
                    called_apps=frozenset(),
                    any_tool_called=False,
                    failures=(),
                    skipped=True,
                    skip_reason=skip_reason,
                )
            )
            continue

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
            called_set = sorted(called_lower)
            missing_set = sorted(missing)
            # premature_final: for multi-app cases where SOME but not ALL apps
            # were reached and the agent produced an answer anyway, emit a more
            # descriptive failure string so "dropped a leg" is distinguishable
            # from "reached nothing".
            reached_any_expected = bool(called_lower & expected_lower)
            is_multi_app = len(case.expected_apps) >= 2
            if is_multi_app and reached_any_expected and any_tool_called and _answer:
                reached = sorted(called_lower & expected_lower)
                failures.append(
                    f"premature_final: reached {reached!r} but not "
                    f"{missing_set!r} before answering"
                )
            else:
                failures.append(
                    f"expected apps not called: {missing_set!r}; "
                    f"got called={called_set!r}"
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
                skipped=False,
                skip_reason="",
            )
        )

    skipped_count = sum(1 for r in results if r.skipped)
    non_skipped = [r for r in results if not r.skipped]
    passed = sum(1 for r in non_skipped if r.passed)
    total = len(results)
    denominator = total - skipped_count
    return OrchestrationReport(
        case_count=total,
        passed=passed,
        failed=denominator - passed,
        skipped=skipped_count,
        pass_rate=passed / denominator if denominator > 0 else 0.0,
        results=tuple(results),
    )
