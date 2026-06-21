"""Tests for the capability-grounding eval harness (HIG-274).

Pure: exercises the scorer and seed-dataset integrity. No live LLM, no DB.
The offline runner (kortny.evals.capability.runner) needs an API key + DB and
runs on demand only.
"""

from __future__ import annotations

from kortny.evals.capability.cases import (
    SEED_CAPABILITY_CASES,
    CapabilityCase,
)
from kortny.evals.capability.scoring import (
    CapabilityAssertionResult,
    CapabilityReport,
    score_capability,
)
from kortny.intent.models import IntentSurface

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DM = IntentSurface.dm
_APP = IntentSurface.app_mention


def _perfect_classify(
    case: CapabilityCase,
) -> tuple[list[str], tuple[str, ...], tuple[str, ...]]:
    """Return classify output that satisfies every case's assertions.

    - ``likely_tools`` contains all ``expected_in_likely_tools`` entries.
    - ``needs_connection`` contains all ``expected_in_needs_connection`` entries
      and none of those appear in ``likely_tools``.
    """
    # Build likely_tools from expected_in_likely_tools, minus any
    # expected_in_needs_connection entries (to avoid the contradiction check).
    needs_set = {c.casefold() for c in case.expected_in_needs_connection}
    likely = [t for t in case.expected_in_likely_tools if t.casefold() not in needs_set]
    needs_connection = tuple(case.expected_in_needs_connection)
    return likely, (), needs_connection


def _perfect_floor(
    case: CapabilityCase, implied_toolkits: tuple[str, ...]
) -> list[str]:
    """Return one synthetic tool per expected floor toolkit, nothing more."""
    return [f"{tk}_tool_1" for tk in case.expected_floor_toolkits]


# ---------------------------------------------------------------------------
# 1. Perfect classify_fn satisfies all assertions
# ---------------------------------------------------------------------------


def test_classify_fn_satisfies_all_assertions() -> None:
    """A classify_fn that mirrors expected values must pass every case."""
    # Use a small representative subset to keep the test focused.
    cases = (SEED_CAPABILITY_CASES[0],)  # "summarize my open Linear tasks"
    report = score_capability(cases, _perfect_classify, _perfect_floor)
    assert report.passed == 1
    assert report.failed == 0
    assert report.results[0].passed


# ---------------------------------------------------------------------------
# 2. Missing likely_toolkit causes failure
# ---------------------------------------------------------------------------


def test_classify_fn_missing_likely_toolkit_fails() -> None:
    """classify_fn returning empty likely_tools for case 1 must fail."""
    case = SEED_CAPABILITY_CASES[0]  # expects "linear" in likely_tools

    def bad_classify(
        c: CapabilityCase,
    ) -> tuple[list[str], tuple[str, ...], tuple[str, ...]]:
        return [], (), ()

    report = score_capability((case,), bad_classify, _perfect_floor)
    assert report.failed == 1
    result = report.results[0]
    assert not result.passed
    assert any("linear" in f and "likely_tools" in f for f in result.failures)


# ---------------------------------------------------------------------------
# 3. Unconnected capability caught by needs_connection
# ---------------------------------------------------------------------------


def test_unconnected_capability_caught_by_needs_connection() -> None:
    """Case 4 ("check my calendar", linear connected): correct path passes."""
    case = SEED_CAPABILITY_CASES[3]  # expects needs_connection=("calendar",)

    # Correct: needs_connection includes "calendar", not in likely_tools
    def good_classify(
        c: CapabilityCase,
    ) -> tuple[list[str], tuple[str, ...], tuple[str, ...]]:
        return [], (), ("calendar",)

    report = score_capability((case,), good_classify, lambda c, t: [])
    assert report.passed == 1, report.results[0].failures

    # Wrong: needs_connection empty but "calendar" in likely_tools — contradicts
    def bad_classify(
        c: CapabilityCase,
    ) -> tuple[list[str], tuple[str, ...], tuple[str, ...]]:
        return ["calendar"], (), ()

    report2 = score_capability((case,), bad_classify, lambda c, t: [])
    assert report2.failed == 1
    result = report2.results[0]
    assert any("needs_connection" in f for f in result.failures)


# ---------------------------------------------------------------------------
# 4. Floor fn returning a tool satisfies the floor assertion
# ---------------------------------------------------------------------------


def test_floor_fn_loads_expected_tool_satisfies_assertion() -> None:
    """floor_fn returning a linear tool must pass the floor constraint."""
    case = SEED_CAPABILITY_CASES[0]  # floor expects ("linear",)

    def floor_with_tool(c: CapabilityCase, toolkits: tuple[str, ...]) -> list[str]:
        return ["linear_tool_1"]

    report = score_capability((case,), _perfect_classify, floor_with_tool)
    assert report.passed == 1


# ---------------------------------------------------------------------------
# 5. Floor fn returning zero tools fails the floor assertion
# ---------------------------------------------------------------------------


def test_floor_fn_zero_tools_fails_assertion() -> None:
    """floor_fn returning [] must fail the floor constraint for linear."""
    case = SEED_CAPABILITY_CASES[0]  # floor expects ("linear",)

    report = score_capability((case,), _perfect_classify, lambda c, t: [])
    assert report.failed == 1
    result = report.results[0]
    assert any("floor loaded 0 tools" in f and "linear" in f for f in result.failures)


# ---------------------------------------------------------------------------
# 6. Scope violation fails the forbidden-toolkit assertion
# ---------------------------------------------------------------------------


def test_scope_violation_fails_forbidden_toolkit() -> None:
    """floor_fn loading a forbidden toolkit must fail the scope assertion."""
    case = SEED_CAPABILITY_CASES[11]  # forbidden_scope_toolkits=("linear_personal",)

    def leaky_floor(c: CapabilityCase, toolkits: tuple[str, ...]) -> list[str]:
        # Load the expected github tool AND the forbidden personal linear tool.
        return ["github_tool_1", "linear_personal_list_issues"]

    report = score_capability((case,), _perfect_classify, leaky_floor)
    assert report.failed == 1
    result = report.results[0]
    assert any("linear_personal" in f for f in result.failures)


# ---------------------------------------------------------------------------
# 7. Global cap exceeded fails
# ---------------------------------------------------------------------------


def test_global_cap_exceeded_fails() -> None:
    """floor_fn returning 25 tools must fail the global cap assertion."""
    # Use a minimal case with no floor toolkits so only the cap assertion fires.
    case = CapabilityCase(
        request="do a thing",
        connected_toolkits=(),
        surface=_DM,
    )

    def bloated_floor(c: CapabilityCase, toolkits: tuple[str, ...]) -> list[str]:
        return [f"tool_{i}" for i in range(25)]

    report = score_capability(
        (case,),
        lambda c: ([], (), ()),
        bloated_floor,
    )
    assert report.failed == 1
    result = report.results[0]
    assert any("exceeds global cap of 24" in f for f in result.failures)


# ---------------------------------------------------------------------------
# 8. Report summary_line format
# ---------------------------------------------------------------------------


def test_report_summary_line_format() -> None:
    """summary_line must match the expected template."""
    report = CapabilityReport(
        case_count=15,
        passed=13,
        failed=2,
        results=(),
    )
    line = report.summary_line()
    assert "13/15" in line
    assert "2" in line


# ---------------------------------------------------------------------------
# 9. All 15 seed cases pass with the perfect classify+floor fn
# ---------------------------------------------------------------------------


def test_all_seed_cases_pass_with_perfect_fn() -> None:
    """The perfect classify+floor fn must score all 15 seed cases as passed."""
    report = score_capability(SEED_CAPABILITY_CASES, _perfect_classify, _perfect_floor)
    assert report.passed == 15, [
        (r.case_id, r.request, r.failures) for r in report.failures
    ]
    assert report.failed == 0
    assert report.case_count == 15


# ---------------------------------------------------------------------------
# 10. Connected-but-stale is treated as connected (no health gate)
# ---------------------------------------------------------------------------


def test_connected_but_stale_treated_as_connected() -> None:
    """Case 13 (notion connected, health may be stale): floor must load tools."""
    case = SEED_CAPABILITY_CASES[
        12
    ]  # "what's stale in Notion?" + connected=("notion",)
    assert "notion" in case.connected_toolkits
    assert "notion" in case.expected_floor_toolkits

    def floor_with_notion(c: CapabilityCase, toolkits: tuple[str, ...]) -> list[str]:
        return ["notion_search_pages"]

    report = score_capability((case,), _perfect_classify, floor_with_notion)
    assert report.passed == 1, report.results[0].failures


# ---------------------------------------------------------------------------
# Dataset integrity
# ---------------------------------------------------------------------------


def test_seed_dataset_has_15_cases() -> None:
    assert len(SEED_CAPABILITY_CASES) == 15


def test_seed_dataset_requests_are_unique() -> None:
    requests = [c.request for c in SEED_CAPABILITY_CASES]
    assert len(requests) == len(set(requests)), "duplicate request text in seed cases"


def test_seed_dataset_surfaces_are_valid() -> None:
    valid = set(IntentSurface)
    for case in SEED_CAPABILITY_CASES:
        assert case.surface in valid


def test_report_failures_property_matches_failed_count() -> None:
    """failures property must return exactly the cases where passed=False."""
    r1 = CapabilityAssertionResult(case_id=1, request="a", passed=True, failures=())
    r2 = CapabilityAssertionResult(
        case_id=2, request="b", passed=False, failures=("oops",)
    )
    report = CapabilityReport(case_count=2, passed=1, failed=1, results=(r1, r2))
    assert len(report.failures) == 1
    assert report.failures[0].case_id == 2
