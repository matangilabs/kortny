"""Tests for the cross-app orchestration eval harness.

Pure: exercises the scorer and seed-dataset integrity with stub run functions.
No live LLM, no DB, no Composio. The offline runner
(kortny.evals.orchestration.runner) needs a live install and runs on demand.
"""

from __future__ import annotations

from kortny.evals.orchestration.cases import (
    SEED_ORCHESTRATION_CASES,
    OrchestrationCase,
)
from kortny.evals.orchestration.runner import _toolkit_slug_from_tool_name
from kortny.evals.orchestration.scoring import (
    OrchestrationAssertionResult,
    OrchestrationReport,
    RunResult,
    score_orchestration,
)
from kortny.intent.models import IntentSurface

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DM = IntentSurface.dm
_APP = IntentSurface.app_mention


def _make_stub_run_fn(
    case_map: dict[str, tuple[frozenset[str], bool]],
) -> object:
    """Build a RunFn that returns pre-recorded called_apps for each request.

    The map key is the case request string. Any case not in the map gets
    (frozenset(), False) — no tools called.
    """

    def run(case: OrchestrationCase) -> RunResult:
        called_apps, any_tool_called = case_map.get(case.request, (frozenset(), False))
        return called_apps, any_tool_called, ""

    return run


# ---------------------------------------------------------------------------
# 1. Passing case: all expected apps called
# ---------------------------------------------------------------------------


def test_expected_apps_called_passes() -> None:
    """A case where all expected apps are called must pass all assertions."""
    case = OrchestrationCase(
        request="open my github prs",
        connected_toolkits=("github",),
        surface=_DM,
        expected_apps=("github",),
    )
    run_fn = _make_stub_run_fn({"open my github prs": (frozenset({"github"}), True)})
    report = score_orchestration((case,), run_fn)  # type: ignore[arg-type]
    assert report.passed == 1
    assert report.failed == 0
    result = report.results[0]
    assert result.passed
    assert result.apps_pass
    assert result.tools_pass
    assert result.scope_pass


# ---------------------------------------------------------------------------
# 2. Missing expected app fails apps_pass
# ---------------------------------------------------------------------------


def test_missing_expected_app_fails() -> None:
    """A case where expected app was not called must fail on apps_pass."""
    case = OrchestrationCase(
        request="summarize my PRs and create a Linear ticket",
        connected_toolkits=("github", "linear"),
        surface=_DM,
        expected_apps=("github", "linear"),
    )
    # Only github called — linear missing.
    run_fn = _make_stub_run_fn({case.request: (frozenset({"github"}), True)})
    report = score_orchestration((case,), run_fn)  # type: ignore[arg-type]
    assert report.failed == 1
    result = report.results[0]
    assert not result.passed
    assert not result.apps_pass
    assert result.tools_pass
    assert result.scope_pass
    assert any("linear" in f for f in result.failures)


# ---------------------------------------------------------------------------
# 3. must_use_tools=True with no tool called fails tools_pass
# ---------------------------------------------------------------------------


def test_must_use_tools_no_tool_called_fails() -> None:
    """When must_use_tools=True and no tool was invoked, tools_pass must fail.

    This is the context-leak guard: the agent answered from injected episodic
    or KG context without making a live API call.
    """
    case = OrchestrationCase(
        request="what did I ship this week?",
        connected_toolkits=("github", "linear"),
        surface=_DM,
        expected_apps=("github", "linear"),
        must_use_tools=True,
    )
    # No tools called at all — context leak scenario.
    run_fn = _make_stub_run_fn({case.request: (frozenset(), False)})
    report = score_orchestration((case,), run_fn)  # type: ignore[arg-type]
    assert report.failed == 1
    result = report.results[0]
    assert not result.passed
    assert not result.tools_pass
    assert not result.apps_pass
    assert any("must_use_tools" in f for f in result.failures)


# ---------------------------------------------------------------------------
# 4. must_use_tools=True with correct tools called passes
# ---------------------------------------------------------------------------


def test_must_use_tools_with_tool_called_passes() -> None:
    """When must_use_tools=True and the expected tool was called, it passes."""
    case = OrchestrationCase(
        request="has my latest PR been merged yet?",
        connected_toolkits=("github",),
        surface=_DM,
        expected_apps=("github",),
        must_use_tools=True,
    )
    run_fn = _make_stub_run_fn({case.request: (frozenset({"github"}), True)})
    report = score_orchestration((case,), run_fn)  # type: ignore[arg-type]
    assert report.passed == 1
    result = report.results[0]
    assert result.passed
    assert result.tools_pass


# ---------------------------------------------------------------------------
# 5. Forbidden app called fails scope_pass
# ---------------------------------------------------------------------------


def test_forbidden_app_called_fails_scope() -> None:
    """When a forbidden app is called, scope_pass must fail."""
    case = OrchestrationCase(
        request="what's AAPL trading at?",
        connected_toolkits=("twelve_data", "github"),
        surface=_DM,
        expected_apps=("twelve_data",),
        must_use_tools=True,
        forbidden_apps=("github", "linear"),
    )
    # twelve_data called (correct), but github also called (forbidden noise).
    run_fn = _make_stub_run_fn(
        {case.request: (frozenset({"twelve_data", "github"}), True)}
    )
    report = score_orchestration((case,), run_fn)  # type: ignore[arg-type]
    assert report.failed == 1
    result = report.results[0]
    assert not result.passed
    assert not result.scope_pass
    assert result.apps_pass
    assert any("github" in f for f in result.failures)


# ---------------------------------------------------------------------------
# 6. No forbidden apps called passes scope_pass
# ---------------------------------------------------------------------------


def test_no_forbidden_apps_passes_scope() -> None:
    """When no forbidden apps are called, scope_pass must be True."""
    case = OrchestrationCase(
        request="what's AAPL trading at?",
        connected_toolkits=("twelve_data",),
        surface=_DM,
        expected_apps=("twelve_data",),
        must_use_tools=True,
        forbidden_apps=("github", "linear", "gmail"),
    )
    run_fn = _make_stub_run_fn({case.request: (frozenset({"twelve_data"}), True)})
    report = score_orchestration((case,), run_fn)  # type: ignore[arg-type]
    assert report.passed == 1
    result = report.results[0]
    assert result.scope_pass


# ---------------------------------------------------------------------------
# 7. Aggregate pass_rate computed correctly
# ---------------------------------------------------------------------------


def test_aggregate_pass_rate() -> None:
    """pass_rate must be passed / case_count."""
    case_a = OrchestrationCase(
        request="req-a",
        connected_toolkits=("github",),
        surface=_DM,
        expected_apps=("github",),
    )
    case_b = OrchestrationCase(
        request="req-b",
        connected_toolkits=("linear",),
        surface=_DM,
        expected_apps=("linear",),
    )
    case_c = OrchestrationCase(
        request="req-c",
        connected_toolkits=("gmail",),
        surface=_DM,
        expected_apps=("gmail",),
    )
    run_fn = _make_stub_run_fn(
        {
            "req-a": (frozenset({"github"}), True),  # pass
            "req-b": (frozenset(), False),  # fail — missing linear
            "req-c": (frozenset({"gmail"}), True),  # pass
        }
    )
    report = score_orchestration((case_a, case_b, case_c), run_fn)  # type: ignore[arg-type]
    assert report.case_count == 3
    assert report.passed == 2
    assert report.failed == 1
    assert abs(report.pass_rate - 2 / 3) < 1e-9


# ---------------------------------------------------------------------------
# 8. Summary line format
# ---------------------------------------------------------------------------


def test_summary_line_format() -> None:
    """summary_line must include passed/total, failed count, and a percentage."""
    report = OrchestrationReport(
        case_count=10,
        passed=7,
        failed=3,
        pass_rate=0.7,
        results=(),
    )
    line = report.summary_line()
    assert "7/10" in line
    assert "3" in line
    assert "70%" in line


# ---------------------------------------------------------------------------
# 9. failures property returns only failed cases
# ---------------------------------------------------------------------------


def test_failures_property_only_returns_failed() -> None:
    """failures must return exactly the cases where passed=False."""
    r1 = OrchestrationAssertionResult(
        case_id=1,
        request="a",
        passed=True,
        apps_pass=True,
        tools_pass=True,
        scope_pass=True,
        expected_apps=(),
        called_apps=frozenset(),
        any_tool_called=False,
        failures=(),
    )
    r2 = OrchestrationAssertionResult(
        case_id=2,
        request="b",
        passed=False,
        apps_pass=False,
        tools_pass=True,
        scope_pass=True,
        expected_apps=("linear",),
        called_apps=frozenset(),
        any_tool_called=False,
        failures=("expected apps not called: ['linear']",),
    )
    report = OrchestrationReport(
        case_count=2, passed=1, failed=1, pass_rate=0.5, results=(r1, r2)
    )
    assert len(report.failures) == 1
    assert report.failures[0].case_id == 2


# ---------------------------------------------------------------------------
# 10. Toolkit slug extraction from composio tool names
# ---------------------------------------------------------------------------


def test_toolkit_slug_extraction_composio() -> None:
    """composio_ prefix tool names yield the correct toolkit slug."""
    assert (
        _toolkit_slug_from_tool_name("composio_github_list_pull_requests") == "github"
    )
    assert _toolkit_slug_from_tool_name("composio_linear_list_issues") == "linear"
    assert _toolkit_slug_from_tool_name("composio_twelve_data_get_price") == "twelve"
    assert (
        _toolkit_slug_from_tool_name("composio_googlecalendar_list_events")
        == "googlecalendar"
    )


def test_toolkit_slug_extraction_non_composio() -> None:
    """Non-Composio tool names return None."""
    assert _toolkit_slug_from_tool_name("mcp__github__list_prs") is None
    assert _toolkit_slug_from_tool_name("slack_post_message") is None
    assert _toolkit_slug_from_tool_name("search_web") is None
    assert _toolkit_slug_from_tool_name("") is None


# ---------------------------------------------------------------------------
# Dataset integrity
# ---------------------------------------------------------------------------


def test_seed_dataset_non_empty() -> None:
    """SEED_ORCHESTRATION_CASES must be non-empty."""
    assert len(SEED_ORCHESTRATION_CASES) > 0


def test_seed_dataset_requests_are_unique() -> None:
    """No two seed cases may have the same request text."""
    requests = [c.request for c in SEED_ORCHESTRATION_CASES]
    assert len(requests) == len(set(requests)), "duplicate request text in seed cases"


def test_seed_dataset_surfaces_are_valid() -> None:
    """All seed case surfaces must be valid IntentSurface values."""
    valid = set(IntentSurface)
    for case in SEED_ORCHESTRATION_CASES:
        assert case.surface in valid, f"invalid surface {case.surface!r}"


def test_seed_dataset_expected_apps_subset_of_connected() -> None:
    """Every case's expected_apps must be a subset of connected_toolkits."""
    for case in SEED_ORCHESTRATION_CASES:
        connected_set = set(case.connected_toolkits)
        for app in case.expected_apps:
            assert app in connected_set, (
                f"case {case.request!r}: expected_app {app!r} not in "
                f"connected_toolkits {sorted(connected_set)!r}"
            )
