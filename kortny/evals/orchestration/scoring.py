"""Pure scoring for the cross-app orchestration eval.

No DB/LLM — given cases + a run callable, run structured assertions per case
and return a typed report. CI exercises the scorer with a stub run_fn; the
live runner (runner.py) wires a real agent execution.

Core assertions per case (always evaluated):

1. **apps_pass**: every toolkit slug in ``expected_apps`` must appear in the
   set of apps actually called. Failure here means the agent skipped a required
   integration.
2. **tools_pass**: when ``must_use_tools`` is True, the called-apps set must be
   non-empty (at least one tool was invoked). Failure here is the context-leak
   guard — the agent answered from injected episodic/KG context without ever
   calling a real tool.
3. **scope_pass**: the intersection of called_apps and ``forbidden_apps`` must
   be empty. Failure means the agent called an out-of-scope integration.

Granularity assertions (only evaluated when the case declares them):

4. **tools_slug_pass**: every slug in ``expected_tool_slugs`` must appear in
   ``called_tool_slugs``. Checks specific tool-level coverage.
5. **required_arg_pass**: every key in ``required_arg_keys`` must appear in the
   union of argument keys across all tool calls.
6. **approval_pass**: when ``approval_expected`` is set on the case,
   ``approval_paused`` must match exactly (either the task paused for approval
   or it did not, as declared).
7. **budget_pass**: when ``max_turns`` or ``max_cost_usd`` is set, the run must
   stay within those bounds.

A case passes iff all applicable assertions pass. The report gives per-case
breakdowns so a glance shows *which* assertion failed and *what* was called vs
expected.

Generalization measurement
--------------------------
``generalization_report(report)`` partitions results by ``tuning_split``
(``"train"`` vs ``"holdout"``) and computes:
- ``train_score``: pass rate among train cases.
- ``holdout_score``: pass rate among holdout cases.
- ``generalization_delta``: ``train_score - holdout_score``.
- ``overfit_warning``: True when delta > 0.12.

A large delta signals that tuning improved train cases but hurt generalization
to unseen apps — the canonical overfitting signal.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import NamedTuple

from kortny.evals.orchestration.cases import OrchestrationCase

# Generalization-delta threshold above which we flag potential overfitting.
_OVERFIT_DELTA_THRESHOLD = 0.12


class RunResult(NamedTuple):
    """Result of running a single orchestration case.

    The live runner returns a ``RunResult`` for every case, including skipped
    ones. The scorer handles both new-style ``RunResult`` NamedTuple values and
    legacy 3-tuple returns for backward compatibility.

    New fields (S2) carry granularity data extracted from TaskEvents:
    - ``called_tool_slugs``: the full set of individual tool slugs invoked
      (e.g. ``"GITHUB_LIST_PULL_REQUESTS"``), not just the app-level slugs.
    - ``called_arg_keys``: union of all argument key names across every tool
      call (e.g. ``{"owner", "repo", "state"}``).
    - ``approval_paused``: True if the task ended ``waiting_approval`` or a
      ``tool_approval_required`` event was recorded.
    - ``turn_count``: number of LLM call events (coordinator turns).
    - ``cost_usd``: total task cost in USD.

    All new fields default so existing call-sites (tests, fixtures) remain
    valid without changes.
    """

    called_apps: frozenset[str]
    any_tool_called: bool
    answer: str
    skipped: bool = False
    skip_reason: str = ""
    # S2 granularity fields — default so existing constructions are back-compat.
    called_tool_slugs: frozenset[str] = frozenset()
    called_arg_keys: frozenset[str] = frozenset()
    approval_paused: bool = False
    turn_count: int = 0
    cost_usd: float = 0.0


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
    # S2 granularity assertion results — None when the case did not declare
    # the corresponding assertion (field absent from OrchestrationCase).
    tools_slug_pass: bool | None = None
    required_arg_pass: bool | None = None
    approval_pass: bool | None = None
    budget_pass: bool | None = None
    # S2 split for generalization measurement.
    tuning_split: str = "train"


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


def generalization_report(report: OrchestrationReport) -> dict[str, object]:
    """Partition results by tuning_split and compute the generalization delta.

    Returns a dict with:
    - ``train_score``: pass rate among non-skipped train cases (float).
    - ``holdout_score``: pass rate among non-skipped holdout cases (float).
    - ``generalization_delta``: ``train_score - holdout_score`` (float).
    - ``overfit_warning``: True when delta > 0.12 (bool).
    - ``train_n``: number of non-skipped train cases scored (int).
    - ``holdout_n``: number of non-skipped holdout cases scored (int).

    When there are no holdout cases (e.g. early in development) all scores are
    0.0 and ``overfit_warning`` is False.
    """
    train_results = [
        r for r in report.results if not r.skipped and r.tuning_split == "train"
    ]
    holdout_results = [
        r for r in report.results if not r.skipped and r.tuning_split == "holdout"
    ]

    def _pass_rate(rs: list[OrchestrationAssertionResult]) -> float:
        if not rs:
            return 0.0
        return sum(1 for r in rs if r.passed) / len(rs)

    train_score = _pass_rate(train_results)
    holdout_score = _pass_rate(holdout_results)
    delta = train_score - holdout_score
    # Only flag overfitting when there are BOTH train AND holdout results to
    # compare — a delta computed against zero holdout cases is meaningless.
    overfit_warning = (
        bool(train_results)
        and bool(holdout_results)
        and delta > _OVERFIT_DELTA_THRESHOLD
    )
    return {
        "train_score": train_score,
        "holdout_score": holdout_score,
        "generalization_delta": delta,
        "overfit_warning": overfit_warning,
        "train_n": len(train_results),
        "holdout_n": len(holdout_results),
    }


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
                    tuning_split=case.tuning_split,
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

        # --- S2 granularity assertions (only when declared on the case) -------

        # 4. tools_slug_pass: individual tool slugs declared as expected.
        tools_slug_pass: bool | None = None
        if case.expected_tool_slugs:
            called_slugs_lower = frozenset(
                s.casefold() for s in run_result.called_tool_slugs
            )
            missing_slugs = (
                frozenset(s.casefold() for s in case.expected_tool_slugs)
                - called_slugs_lower
            )
            tools_slug_pass = len(missing_slugs) == 0
            if not tools_slug_pass:
                failures.append(
                    f"expected tool slugs not called: {sorted(missing_slugs)!r}; "
                    f"got called_tool_slugs={sorted(called_slugs_lower)!r}"
                )

        # 5. required_arg_pass: argument keys that must appear in any call.
        required_arg_pass: bool | None = None
        if case.required_arg_keys:
            called_arg_lower = frozenset(
                k.casefold() for k in run_result.called_arg_keys
            )
            missing_args = (
                frozenset(k.casefold() for k in case.required_arg_keys)
                - called_arg_lower
            )
            required_arg_pass = len(missing_args) == 0
            if not required_arg_pass:
                failures.append(
                    f"required arg keys missing: {sorted(missing_args)!r}; "
                    f"got called_arg_keys={sorted(called_arg_lower)!r}"
                )

        # 6. approval_pass: approval-gate behavior matches declaration.
        approval_pass: bool | None = None
        if case.approval_expected is not None:
            approval_pass = run_result.approval_paused == case.approval_expected
            if not approval_pass:
                expected_str = (
                    "approval pause expected"
                    if case.approval_expected
                    else "no approval expected"
                )
                got_str = (
                    "task paused"
                    if run_result.approval_paused
                    else "task did not pause"
                )
                failures.append(f"approval_pass: {expected_str} but {got_str}")

        # 7. budget_pass: turn count and cost within declared limits.
        budget_pass: bool | None = None
        if case.max_turns is not None or case.max_cost_usd is not None:
            turn_ok = case.max_turns is None or run_result.turn_count <= case.max_turns
            cost_ok = (
                case.max_cost_usd is None or run_result.cost_usd <= case.max_cost_usd
            )
            budget_pass = turn_ok and cost_ok
            if not turn_ok:
                failures.append(
                    f"turn_count={run_result.turn_count} exceeds max_turns={case.max_turns}"
                )
            if not cost_ok:
                failures.append(
                    f"cost_usd={run_result.cost_usd:.4f} exceeds "
                    f"max_cost_usd={case.max_cost_usd}"
                )

        # A case passes iff all applicable assertions pass.
        granularity_passed = all(
            v is not False
            for v in (tools_slug_pass, required_arg_pass, approval_pass, budget_pass)
        )
        case_passed = apps_pass and tools_pass and scope_pass and granularity_passed

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
                tools_slug_pass=tools_slug_pass,
                required_arg_pass=required_arg_pass,
                approval_pass=approval_pass,
                budget_pass=budget_pass,
                tuning_split=case.tuning_split,
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
