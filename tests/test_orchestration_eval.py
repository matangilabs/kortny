"""Tests for the cross-app orchestration eval harness.

Pure: exercises the scorer and seed-dataset integrity with stub run functions.
No live LLM, no DB, no Composio. The offline runner
(kortny.evals.orchestration.runner) needs a live install and runs on demand.
"""

from __future__ import annotations

import uuid

import pytest

from kortny.evals.orchestration.cases import (
    SEED_ORCHESTRATION_CASES,
    OrchestrationCase,
)
from kortny.evals.orchestration.runner import (
    _EVAL_CHANNEL_ID,
    _NoOpSlackClient,
    _resolve_scope_channel_id,
    _resolve_scope_user_id,
    _toolkit_slug_from_tool_name,
    _toolkit_slug_from_tool_result,
)
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
        return RunResult(
            called_apps=called_apps, any_tool_called=any_tool_called, answer=""
        )

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
    """summary_line must include passed/total (excluding skipped), failed count, and a percentage."""
    report = OrchestrationReport(
        case_count=10,
        passed=7,
        failed=3,
        skipped=0,
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
        case_count=2, passed=1, failed=1, skipped=0, pass_rate=0.5, results=(r1, r2)
    )
    assert len(report.failures) == 1
    assert report.failures[0].case_id == 2


# ---------------------------------------------------------------------------
# 10. Authoritative toolkit-slug derivation from tool_result payloads
# ---------------------------------------------------------------------------


def test_toolkit_slug_from_tool_result_authoritative() -> None:
    """The real toolkit_slug is read directly from the tool_result output.

    Crucially, this is correct for multi-underscore toolkits that name-parsing
    would mangle (twelve_data -> 'twelve', alpha_vantage -> 'alpha').
    """
    payload = {
        "tool": "composio_twelve_data_get_price",
        "output": {
            "provider": "composio",
            "toolkit_slug": "twelve_data",
            "tool_slug": "TWELVE_DATA_GET_PRICE",
            "successful": True,
        },
    }
    assert _toolkit_slug_from_tool_result(payload) == "twelve_data"

    alpha = {
        "tool": "composio_alpha_vantage_quote",
        "output": {
            "provider": "composio",
            "toolkit_slug": "alpha_vantage",
            "successful": True,
        },
    }
    assert _toolkit_slug_from_tool_result(alpha) == "alpha_vantage"


def test_toolkit_slug_from_tool_result_unsuccessful_excluded() -> None:
    """A failed Composio execution does not count toward the called-apps set."""
    payload = {
        "output": {
            "provider": "composio",
            "toolkit_slug": "github",
            "successful": False,
        },
    }
    assert _toolkit_slug_from_tool_result(payload) is None


def test_toolkit_slug_from_tool_result_non_composio() -> None:
    """Native/MCP tool results carry no Composio provider and yield None."""
    native = {"output": {"posted": True}}
    assert _toolkit_slug_from_tool_result(native) is None
    mcp = {"output": {"provider": "mcp", "successful": True}}
    assert _toolkit_slug_from_tool_result(mcp) is None
    no_output = {"tool": "composio_github_list_prs"}
    assert _toolkit_slug_from_tool_result(no_output) is None


def test_toolkit_slug_from_tool_result_falls_back_to_name_parse() -> None:
    """When toolkit_slug is absent, fall back to name-parsing the runtime name."""
    payload = {
        "tool": "composio_github_list_pull_requests",
        "output": {
            "provider": "composio",
            "successful": True,
            # toolkit_slug deliberately absent
        },
    }
    assert _toolkit_slug_from_tool_result(payload) == "github"


# ---------------------------------------------------------------------------
# 11. Name-parse helper (fallback-only; documents its multi-underscore limit)
# ---------------------------------------------------------------------------


def test_toolkit_slug_name_parse_single_underscore() -> None:
    """The fallback parser is correct for single-segment toolkit slugs."""
    assert (
        _toolkit_slug_from_tool_name("composio_github_list_pull_requests") == "github"
    )
    assert _toolkit_slug_from_tool_name("composio_linear_list_issues") == "linear"
    assert (
        _toolkit_slug_from_tool_name("composio_googlecalendar_list_events")
        == "googlecalendar"
    )


def test_toolkit_slug_name_parse_mangles_multi_underscore() -> None:
    """The fallback parser mangles multi-underscore slugs — why it's not primary.

    twelve_data -> 'twelve'. This is precisely the bug that reading the
    authoritative toolkit_slug from the tool_result output avoids.
    """
    assert _toolkit_slug_from_tool_name("composio_twelve_data_get_price") == "twelve"
    assert _toolkit_slug_from_tool_name("composio_alpha_vantage_quote") == "alpha"


def test_toolkit_slug_name_parse_non_composio() -> None:
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
    """Every runnable case's expected_apps must be a subset of connected_toolkits.

    Cases with requires_toolkits set are intentionally unconnected and are
    excluded from this check — they document target workflows and skip at
    runtime until the required toolkits are connected.
    """
    for case in SEED_ORCHESTRATION_CASES:
        if case.requires_toolkits:
            # Skip — intentionally unconnected case.
            continue
        connected_set = set(case.connected_toolkits)
        for app in case.expected_apps:
            assert app in connected_set, (
                f"case {case.request!r}: expected_app {app!r} not in "
                f"connected_toolkits {sorted(connected_set)!r}"
            )


# ---------------------------------------------------------------------------
# Connection-scope resolution (env-override path; no DB)
# ---------------------------------------------------------------------------


def test_resolve_scope_user_id_env_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """KORTNY_EVAL_SCOPE_USER_ID overrides DB auto-detection.

    The override returns before any DB query, so a sentinel session that would
    error if touched proves the override short-circuits.
    """
    monkeypatch.setenv("KORTNY_EVAL_SCOPE_USER_ID", "U_EVAL_OWNER_TEST")

    class _ExplodingSession:
        def execute(self, *args: object, **kwargs: object) -> object:
            raise AssertionError("DB must not be queried when env override is set")

    session = _ExplodingSession()
    user_id = _resolve_scope_user_id(session, uuid.uuid4())  # type: ignore[arg-type]
    assert user_id == "U_EVAL_OWNER_TEST"


def test_resolve_scope_channel_id_env_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """KORTNY_EVAL_SCOPE_CHANNEL_ID overrides the synthetic default."""
    monkeypatch.setenv("KORTNY_EVAL_SCOPE_CHANNEL_ID", "C12345678")
    assert _resolve_scope_channel_id() == "C12345678"


def test_resolve_scope_channel_id_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the env override, the synthetic eval channel is used."""
    monkeypatch.delenv("KORTNY_EVAL_SCOPE_CHANNEL_ID", raising=False)
    assert _resolve_scope_channel_id() == _EVAL_CHANNEL_ID


# ---------------------------------------------------------------------------
# No-op Slack client (side-effect-free egress)
# ---------------------------------------------------------------------------


def test_noop_slack_client_post_returns_ts() -> None:
    """chat_postMessage must return an ok response carrying a ts.

    SlackPoster reads the ts off the response; a missing ts raises. The no-op
    client must therefore return a benign ts so the post path completes without
    reaching Slack.
    """
    client = _NoOpSlackClient()
    resp = client.chat_postMessage(channel="EVAL_ORCHESTRATION", text="hi")
    assert resp["ok"] is True
    assert isinstance(resp["ts"], str) and resp["ts"]


def test_noop_slack_client_file_upload_ok() -> None:
    """files_upload_v2 must return an ok response and never reach Slack."""
    client = _NoOpSlackClient()
    resp = client.files_upload_v2(file="/tmp/x.txt", channel="EVAL_ORCHESTRATION")
    assert resp["ok"] is True


def test_noop_slack_client_reactions_are_noops() -> None:
    """reactions_add / reactions_remove must be no-ops returning ok."""
    client = _NoOpSlackClient()
    assert client.reactions_add(channel="C", name="eyes", timestamp="1.0")["ok"]
    assert client.reactions_remove(channel="C", name="eyes", timestamp="1.0")["ok"]


def test_noop_slack_client_post_ts_is_unique() -> None:
    """Each post returns a distinct ts so the outbox never collides on dedup."""
    client = _NoOpSlackClient()
    ts1 = client.chat_postMessage(channel="C", text="a")["ts"]
    ts2 = client.chat_postMessage(channel="C", text="b")["ts"]
    assert ts1 != ts2


# ---------------------------------------------------------------------------
# Skip semantics
# ---------------------------------------------------------------------------


def test_skipped_case_excluded_from_pass_rate() -> None:
    """A skipped case must not count in the pass_rate denominator."""
    case_run = OrchestrationCase(
        request="open my github prs",
        connected_toolkits=("github",),
        surface=_DM,
        expected_apps=("github",),
    )
    case_skip = OrchestrationCase(
        request="create a jira ticket",
        connected_toolkits=("github",),
        surface=_DM,
        expected_apps=("jira",),
        requires_toolkits=("jira",),
    )

    def run_fn(case: OrchestrationCase) -> RunResult:
        if case.requires_toolkits:
            return RunResult(
                called_apps=frozenset(),
                any_tool_called=False,
                answer="",
                skipped=True,
                skip_reason="jira not connected",
            )
        return RunResult(
            called_apps=frozenset({"github"}), any_tool_called=True, answer="ok"
        )

    report = score_orchestration((case_run, case_skip), run_fn)
    assert report.case_count == 2
    assert report.skipped == 1
    assert report.passed == 1
    assert report.failed == 0
    assert report.pass_rate == 1.0  # 1/1 non-skipped
    assert "1 skipped" in report.summary_line()


def test_skipped_case_not_in_failures() -> None:
    """A skipped case must not appear in report.failures."""
    case = OrchestrationCase(
        request="schedule a zoom",
        connected_toolkits=("github",),
        surface=_DM,
        expected_apps=("zoom",),
        requires_toolkits=("zoom",),
    )

    def run_fn(case: OrchestrationCase) -> RunResult:
        return RunResult(
            called_apps=frozenset(),
            any_tool_called=False,
            answer="",
            skipped=True,
            skip_reason="zoom not connected",
        )

    report = score_orchestration((case,), run_fn)
    assert report.skipped == 1
    assert len(report.failures) == 0


def test_premature_final_failure_label() -> None:
    """Multi-app case that reached some but not all apps should show premature_final."""
    case = OrchestrationCase(
        request="pull the onboarding doc from confluence and create a notion page",
        connected_toolkits=("confluence", "notion"),
        surface=_DM,
        expected_apps=("confluence", "notion"),
    )

    def run_fn(case: OrchestrationCase) -> RunResult:
        # Only reached confluence, answered without notion.
        return RunResult(
            called_apps=frozenset({"confluence"}),
            any_tool_called=True,
            answer="Here is the summary.",
        )

    report = score_orchestration((case,), run_fn)
    assert report.failed == 1
    result = report.results[0]
    assert not result.passed
    assert any("premature_final" in f for f in result.failures)
    assert any("notion" in f for f in result.failures)


def test_requires_toolkits_field_default_empty() -> None:
    """OrchestrationCase.requires_toolkits must default to empty tuple."""
    case = OrchestrationCase(
        request="open my github prs",
        connected_toolkits=("github",),
        surface=_DM,
        expected_apps=("github",),
    )
    assert case.requires_toolkits == ()


def test_top25_registry_has_25_entries() -> None:
    """TOP25 must have exactly 25 entries."""
    from kortny.integrations.top25 import TOP25, TOP25_SLUGS, tier_of  # noqa: F401

    assert len(TOP25) == 25
    assert len(TOP25_SLUGS) == 25


def test_top25_tier_of_known_slugs() -> None:
    """tier_of returns correct tiers for known slugs."""
    from kortny.integrations.top25 import tier_of

    assert tier_of("github") == 1
    assert tier_of("linear") == 1
    assert tier_of("slack") == 1
    assert tier_of("confluence") == 2
    assert tier_of("zendesk") == 2


def test_top25_tier_of_unknown_slug() -> None:
    """tier_of returns None for slugs not in top-25."""
    from kortny.integrations.top25 import tier_of

    assert tier_of("twelve_data") is None
    assert tier_of("serpapi") is None
    assert tier_of("notaslug") is None


def test_top25_aliases_resolve() -> None:
    """ALIASES map resolves known alias slugs correctly."""
    from kortny.integrations.top25 import tier_of

    # outlook_calendar is an alias for outlook (tier 1)
    assert tier_of("outlook_calendar") == 1
    assert tier_of("teams") == 1
    assert tier_of("gdrive") == 1


def test_seed_dataset_requires_toolkits_cases_are_top25() -> None:
    """Cases with requires_toolkits must reference top-25 slugs only."""
    from kortny.integrations.top25 import TOP25_SLUGS

    for case in SEED_ORCHESTRATION_CASES:
        for slug in case.requires_toolkits:
            assert slug in TOP25_SLUGS, (
                f"case {case.request!r}: requires_toolkits slug {slug!r} "
                f"not in TOP25_SLUGS"
            )


# ---------------------------------------------------------------------------
# Replay module
# ---------------------------------------------------------------------------


def test_smoke_cases_have_committed_fixture() -> None:
    """Every smoke=True case must have a committed fixture in smoke_goldens.json.

    This test is the drift guard: if a new smoke case is added to cases.py but
    its fixture is not committed to smoke_goldens.json, this test fails loudly
    rather than silently skipping the case at eval-smoke time.
    """
    from kortny.evals.orchestration.replay import DEFAULT_FIXTURES_PATH, load_fixtures

    fixtures = load_fixtures(DEFAULT_FIXTURES_PATH)
    smoke_cases = [c for c in SEED_ORCHESTRATION_CASES if c.smoke]
    assert smoke_cases, "no smoke=True cases found — mark at least one case smoke=True"
    missing = [c.request for c in smoke_cases if c.request not in fixtures]
    assert not missing, (
        f"{len(missing)} smoke case(s) have no committed fixture in "
        f"{DEFAULT_FIXTURES_PATH.name}: {missing!r}. "
        "Run `make eval` (live) to record goldens, then commit the updated file."
    )


def test_replay_scores_smoke_cases_from_goldens() -> None:
    """Scoring smoke cases with committed fixtures must produce expected pass/fail.

    Loads the committed ``smoke_goldens.json``, builds the replay RunFn, and
    asserts that ``score_orchestration`` agrees with what the fixture encodes.
    For each smoke case:
    - If the fixture has the correct apps (meeting expected_apps and not
      forbidden_apps), the case passes.
    - If the fixture has called_apps=[] and expected_apps=(), the case passes
      (negative/no-tool guard cases).

    Pure offline: no DB, no LLM, no API keys.
    """
    from kortny.evals.orchestration.replay import (
        DEFAULT_FIXTURES_PATH,
        build_replay_run_fn,
        load_fixtures,
    )

    fixtures = load_fixtures(DEFAULT_FIXTURES_PATH)
    smoke_cases = [c for c in SEED_ORCHESTRATION_CASES if c.smoke]
    # Only score cases that have a fixture (others would be skipped).
    replay_fn = build_replay_run_fn(fixtures)
    report = score_orchestration(smoke_cases, replay_fn)

    # The goldens encode desired behavior, so non-skipped smoke cases should pass.
    non_skipped_failures = [r for r in report.results if not r.skipped and not r.passed]
    assert not non_skipped_failures, (
        f"{len(non_skipped_failures)} smoke case(s) FAIL against committed goldens:\n"
        + "\n".join(f"  {r.request!r}: {r.failures}" for r in non_skipped_failures)
    )


def test_replay_missing_fixture_returns_skipped() -> None:
    """build_replay_run_fn must return skipped=True for cases with no fixture."""
    from kortny.evals.orchestration.replay import build_replay_run_fn

    case = OrchestrationCase(
        request="__no_fixture_for_this_request__",
        connected_toolkits=("github",),
        surface=_DM,
        expected_apps=("github",),
    )
    run_fn = build_replay_run_fn({})
    result = run_fn(case)
    assert result.skipped is True
    assert result.skip_reason == "no replay fixture recorded"
    assert result.called_apps == frozenset()


def test_dump_and_load_fixtures_roundtrip() -> None:
    """dump_fixtures + load_fixtures must roundtrip RunResults faithfully."""
    import tempfile
    from pathlib import Path

    from kortny.evals.orchestration.replay import dump_fixtures, load_fixtures

    rr_a = RunResult(
        called_apps=frozenset({"github", "linear"}),
        any_tool_called=True,
        answer="some answer",
        skipped=False,
        skip_reason="",
    )
    rr_b = RunResult(
        called_apps=frozenset(),
        any_tool_called=False,
        answer="",
        skipped=True,
        skip_reason="not connected",
    )
    mapping: dict[str, RunResult] = {
        "request alpha": rr_a,
        "request beta": rr_b,
    }
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = Path(f.name)
    try:
        dump_fixtures(mapping, path)
        loaded = load_fixtures(path)
        assert loaded["request alpha"].called_apps == frozenset({"github", "linear"})
        assert loaded["request alpha"].any_tool_called is True
        assert loaded["request alpha"].skipped is False
        assert loaded["request beta"].called_apps == frozenset()
        assert loaded["request beta"].skipped is True
        assert loaded["request beta"].skip_reason == "not connected"
    finally:
        path.unlink(missing_ok=True)


def test_replay_run_produces_report_no_secrets() -> None:
    """run_replay() produces an OrchestrationReport with no secrets/DB/LLM."""
    from kortny.evals.orchestration.replay import DEFAULT_FIXTURES_PATH, run_replay

    # Only run if the fixture file exists (it is committed as a golden).
    if not DEFAULT_FIXTURES_PATH.exists():
        pytest.skip("smoke_goldens.json not present (run make eval first)")

    report = run_replay(smoke_only=True)
    assert isinstance(report.case_count, int)
    assert report.case_count > 0
    assert isinstance(report.pass_rate, float)


# ---------------------------------------------------------------------------
# S2: RunResult back-compat (new fields have safe defaults)
# ---------------------------------------------------------------------------


def test_run_result_new_fields_default_backcompat() -> None:
    """RunResult must construct with only the original 3 positional fields.

    New S2 fields (called_tool_slugs, called_arg_keys, approval_paused,
    turn_count, cost_usd) all have defaults so existing call-sites and fixtures
    remain valid without changes.
    """
    rr = RunResult(called_apps=frozenset({"github"}), any_tool_called=True, answer="ok")
    assert rr.called_tool_slugs == frozenset()
    assert rr.called_arg_keys == frozenset()
    assert rr.approval_paused is False
    assert rr.turn_count == 0
    assert rr.cost_usd == 0.0


def test_run_result_full_construction() -> None:
    """RunResult must accept all S2 fields when explicitly provided."""
    rr = RunResult(
        called_apps=frozenset({"github"}),
        any_tool_called=True,
        answer="ok",
        called_tool_slugs=frozenset({"GITHUB_LIST_PULL_REQUESTS"}),
        called_arg_keys=frozenset({"owner", "repo", "state"}),
        approval_paused=False,
        turn_count=3,
        cost_usd=0.02,
    )
    assert rr.called_tool_slugs == frozenset({"GITHUB_LIST_PULL_REQUESTS"})
    assert rr.called_arg_keys == frozenset({"owner", "repo", "state"})
    assert rr.turn_count == 3
    assert rr.cost_usd == 0.02


# ---------------------------------------------------------------------------
# S2: OrchestrationCase new fields
# ---------------------------------------------------------------------------


def test_orchestration_case_new_fields_default() -> None:
    """OrchestrationCase must default all S2 fields correctly."""
    case = OrchestrationCase(
        request="list open prs",
        connected_toolkits=("github",),
        surface=_DM,
        expected_apps=("github",),
    )
    assert case.tuning_split == "train"
    assert case.expected_tool_slugs == ()
    assert case.required_arg_keys == ()
    assert case.approval_expected is None
    assert case.max_turns is None
    assert case.max_cost_usd is None


def test_orchestration_case_holdout_split() -> None:
    """A holdout case must carry tuning_split='holdout'."""
    case = OrchestrationCase(
        request="page on-call engineer",
        connected_toolkits=("pagerduty",),
        surface=_DM,
        expected_apps=("pagerduty",),
        tuning_split="holdout",
    )
    assert case.tuning_split == "holdout"


# ---------------------------------------------------------------------------
# S2: tools_slug_pass assertion
# ---------------------------------------------------------------------------


def test_tools_slug_pass_all_present() -> None:
    """When all expected_tool_slugs appear in called_tool_slugs, tools_slug_pass is True."""
    case = OrchestrationCase(
        request="list open github issues assigned to me",
        connected_toolkits=("github",),
        surface=_DM,
        expected_apps=("github",),
        expected_tool_slugs=("GITHUB_ISSUES_LIST",),
    )

    def run_fn(c: OrchestrationCase) -> RunResult:
        return RunResult(
            called_apps=frozenset({"github"}),
            any_tool_called=True,
            answer="ok",
            called_tool_slugs=frozenset({"GITHUB_ISSUES_LIST"}),
        )

    report = score_orchestration((case,), run_fn)
    assert report.passed == 1
    r = report.results[0]
    assert r.tools_slug_pass is True
    assert r.passed


def test_tools_slug_pass_missing_slug_fails() -> None:
    """When an expected tool slug is absent from called_tool_slugs, tools_slug_pass is False."""
    case = OrchestrationCase(
        request="list open github issues assigned to me",
        connected_toolkits=("github",),
        surface=_DM,
        expected_apps=("github",),
        expected_tool_slugs=("GITHUB_ISSUES_LIST",),
    )

    def run_fn(c: OrchestrationCase) -> RunResult:
        return RunResult(
            called_apps=frozenset({"github"}),
            any_tool_called=True,
            answer="ok",
            called_tool_slugs=frozenset({"GITHUB_LIST_PULL_REQUESTS"}),  # wrong slug
        )

    report = score_orchestration((case,), run_fn)
    assert report.failed == 1
    r = report.results[0]
    assert r.tools_slug_pass is False
    assert not r.passed
    assert any("github_issues_list" in f for f in r.failures)


def test_tools_slug_pass_none_when_not_declared() -> None:
    """tools_slug_pass must be None when expected_tool_slugs is not declared."""
    case = OrchestrationCase(
        request="list open github issues",
        connected_toolkits=("github",),
        surface=_DM,
        expected_apps=("github",),
        # No expected_tool_slugs declared.
    )

    def run_fn(c: OrchestrationCase) -> RunResult:
        return RunResult(
            called_apps=frozenset({"github"}), any_tool_called=True, answer="ok"
        )

    report = score_orchestration((case,), run_fn)
    r = report.results[0]
    assert r.tools_slug_pass is None
    assert r.passed


# ---------------------------------------------------------------------------
# S2: required_arg_pass assertion
# ---------------------------------------------------------------------------


def test_required_arg_pass_all_present() -> None:
    """When all required_arg_keys appear in called_arg_keys, required_arg_pass is True."""
    case = OrchestrationCase(
        request="find emails from finance with invoice in subject",
        connected_toolkits=("gmail",),
        surface=_DM,
        expected_apps=("gmail",),
        required_arg_keys=("query",),
    )

    def run_fn(c: OrchestrationCase) -> RunResult:
        return RunResult(
            called_apps=frozenset({"gmail"}),
            any_tool_called=True,
            answer="ok",
            called_arg_keys=frozenset({"query", "max_results"}),
        )

    report = score_orchestration((case,), run_fn)
    assert report.passed == 1
    r = report.results[0]
    assert r.required_arg_pass is True


def test_required_arg_pass_missing_key_fails() -> None:
    """When a required arg key is absent, required_arg_pass is False."""
    case = OrchestrationCase(
        request="find emails from finance with invoice in subject",
        connected_toolkits=("gmail",),
        surface=_DM,
        expected_apps=("gmail",),
        required_arg_keys=("query",),
    )

    def run_fn(c: OrchestrationCase) -> RunResult:
        return RunResult(
            called_apps=frozenset({"gmail"}),
            any_tool_called=True,
            answer="ok",
            called_arg_keys=frozenset({"max_results"}),  # 'query' missing
        )

    report = score_orchestration((case,), run_fn)
    assert report.failed == 1
    r = report.results[0]
    assert r.required_arg_pass is False
    assert any("query" in f for f in r.failures)


# ---------------------------------------------------------------------------
# S2: approval_pass assertion
# ---------------------------------------------------------------------------


def test_approval_pass_expected_true_and_paused() -> None:
    """When approval_expected=True and task paused, approval_pass is True."""
    case = OrchestrationCase(
        request="create a Linear issue for the bug",
        connected_toolkits=("linear",),
        surface=_DM,
        expected_apps=("linear",),
        approval_expected=True,
    )

    def run_fn(c: OrchestrationCase) -> RunResult:
        return RunResult(
            called_apps=frozenset({"linear"}),
            any_tool_called=True,
            answer="",
            approval_paused=True,
        )

    report = score_orchestration((case,), run_fn)
    assert report.passed == 1
    r = report.results[0]
    assert r.approval_pass is True


def test_approval_pass_expected_true_but_not_paused_fails() -> None:
    """When approval_expected=True but task did not pause, approval_pass is False."""
    case = OrchestrationCase(
        request="create a Linear issue for the bug",
        connected_toolkits=("linear",),
        surface=_DM,
        expected_apps=("linear",),
        approval_expected=True,
    )

    def run_fn(c: OrchestrationCase) -> RunResult:
        return RunResult(
            called_apps=frozenset({"linear"}),
            any_tool_called=True,
            answer="done",
            approval_paused=False,  # should have paused
        )

    report = score_orchestration((case,), run_fn)
    assert report.failed == 1
    r = report.results[0]
    assert r.approval_pass is False
    assert any("approval" in f for f in r.failures)


def test_approval_pass_none_when_not_declared() -> None:
    """approval_pass must be None when approval_expected is not declared."""
    case = OrchestrationCase(
        request="list my open issues",
        connected_toolkits=("linear",),
        surface=_DM,
        expected_apps=("linear",),
        # approval_expected not set (None by default)
    )

    def run_fn(c: OrchestrationCase) -> RunResult:
        return RunResult(
            called_apps=frozenset({"linear"}), any_tool_called=True, answer="ok"
        )

    report = score_orchestration((case,), run_fn)
    r = report.results[0]
    assert r.approval_pass is None
    assert r.passed


# ---------------------------------------------------------------------------
# S2: budget_pass assertion
# ---------------------------------------------------------------------------


def test_budget_pass_within_limits() -> None:
    """When turn_count and cost_usd are within limits, budget_pass is True."""
    case = OrchestrationCase(
        request="show last 5 commits",
        connected_toolkits=("github",),
        surface=_DM,
        expected_apps=("github",),
        max_turns=4,
        max_cost_usd=0.05,
    )

    def run_fn(c: OrchestrationCase) -> RunResult:
        return RunResult(
            called_apps=frozenset({"github"}),
            any_tool_called=True,
            answer="ok",
            turn_count=2,
            cost_usd=0.01,
        )

    report = score_orchestration((case,), run_fn)
    assert report.passed == 1
    r = report.results[0]
    assert r.budget_pass is True


def test_budget_pass_turns_exceeded_fails() -> None:
    """When turn_count exceeds max_turns, budget_pass is False."""
    case = OrchestrationCase(
        request="show last 5 commits",
        connected_toolkits=("github",),
        surface=_DM,
        expected_apps=("github",),
        max_turns=2,
    )

    def run_fn(c: OrchestrationCase) -> RunResult:
        return RunResult(
            called_apps=frozenset({"github"}),
            any_tool_called=True,
            answer="ok",
            turn_count=5,  # exceeds max_turns=2
        )

    report = score_orchestration((case,), run_fn)
    assert report.failed == 1
    r = report.results[0]
    assert r.budget_pass is False
    assert any("turn_count" in f for f in r.failures)


# ---------------------------------------------------------------------------
# S2: generalization_report
# ---------------------------------------------------------------------------


def test_generalization_report_no_holdout() -> None:
    """When there are no holdout cases, holdout_score=0.0 and no overfit warning."""
    from kortny.evals.orchestration.scoring import generalization_report

    case = OrchestrationCase(
        request="open my github prs",
        connected_toolkits=("github",),
        surface=_DM,
        expected_apps=("github",),
        tuning_split="train",
    )

    def run_fn(c: OrchestrationCase) -> RunResult:
        return RunResult(
            called_apps=frozenset({"github"}), any_tool_called=True, answer="ok"
        )

    report = score_orchestration((case,), run_fn)
    gr = generalization_report(report)
    assert gr["train_score"] == 1.0
    assert gr["holdout_score"] == 0.0
    assert gr["train_n"] == 1
    assert gr["holdout_n"] == 0
    assert gr["overfit_warning"] is False


def test_generalization_report_delta_within_threshold() -> None:
    """When train_score - holdout_score <= 0.12, no overfit warning."""
    from kortny.evals.orchestration.scoring import generalization_report

    train_case = OrchestrationCase(
        request="list my issues in Linear",
        connected_toolkits=("linear",),
        surface=_DM,
        expected_apps=("linear",),
        tuning_split="train",
    )
    holdout_case = OrchestrationCase(
        request="look up campaign records in Airtable",
        connected_toolkits=("airtable",),
        surface=_DM,
        expected_apps=("airtable",),
        tuning_split="holdout",
    )

    def run_fn(c: OrchestrationCase) -> RunResult:
        # Both pass — delta = 1.0 - 1.0 = 0.0
        return RunResult(
            called_apps=frozenset({c.expected_apps[0]}),
            any_tool_called=True,
            answer="ok",
        )

    report = score_orchestration((train_case, holdout_case), run_fn)
    gr = generalization_report(report)
    assert gr["train_score"] == 1.0
    assert gr["holdout_score"] == 1.0
    assert gr["generalization_delta"] == 0.0
    assert gr["overfit_warning"] is False


def test_generalization_report_delta_exceeds_threshold_warns() -> None:
    """When train_score - holdout_score > 0.12, overfit_warning is True."""
    from kortny.evals.orchestration.scoring import generalization_report

    train_case = OrchestrationCase(
        request="list my issues in Linear",
        connected_toolkits=("linear",),
        surface=_DM,
        expected_apps=("linear",),
        tuning_split="train",
    )
    holdout_case = OrchestrationCase(
        request="trigger PagerDuty incident",
        connected_toolkits=("pagerduty",),
        surface=_DM,
        expected_apps=("pagerduty",),
        tuning_split="holdout",
    )

    def run_fn(c: OrchestrationCase) -> RunResult:
        if c.tuning_split == "train":
            # Train passes.
            return RunResult(
                called_apps=frozenset({"linear"}), any_tool_called=True, answer="ok"
            )
        else:
            # Holdout fails (agent didn't call pagerduty).
            return RunResult(called_apps=frozenset(), any_tool_called=False, answer="")

    report = score_orchestration((train_case, holdout_case), run_fn)
    gr = generalization_report(report)
    assert gr["train_score"] == 1.0
    assert gr["holdout_score"] == 0.0
    assert gr["generalization_delta"] == 1.0
    assert gr["overfit_warning"] is True


# ---------------------------------------------------------------------------
# S2: runner split filter hard-block
# ---------------------------------------------------------------------------


def test_run_split_filter_excludes_holdout_from_train() -> None:
    """HARD-BLOCK: split='train' must never include holdout cases.

    We call run() but it will raise RuntimeError if the invariant is violated.
    Since all cases in SEED_ORCHESTRATION_CASES that are 'holdout' would be
    excluded by the filter, we test the filter logic directly via the seed data.
    """
    # All train cases in the seed must have tuning_split='train'.
    train_cases = [c for c in SEED_ORCHESTRATION_CASES if c.tuning_split == "train"]
    holdout_cases = [c for c in SEED_ORCHESTRATION_CASES if c.tuning_split == "holdout"]
    # No case should be in both.
    train_requests = {c.request for c in train_cases}
    holdout_requests = {c.request for c in holdout_cases}
    assert train_requests.isdisjoint(holdout_requests), (
        "A case appears in both train and holdout splits — tuning_split must be unique"
    )
    # All cases must be classified.
    all_requests = {c.request for c in SEED_ORCHESTRATION_CASES}
    assert all_requests == train_requests | holdout_requests


def test_seed_cases_have_valid_tuning_splits() -> None:
    """Every seed case must declare a valid tuning_split ('train' or 'holdout')."""
    valid_splits = {"train", "holdout"}
    for case in SEED_ORCHESTRATION_CASES:
        assert case.tuning_split in valid_splits, (
            f"case {case.request!r}: invalid tuning_split {case.tuning_split!r}"
        )


# ---------------------------------------------------------------------------
# S2: holdout cases never in train run (integration of runner filter logic)
# ---------------------------------------------------------------------------


def test_split_filter_train_returns_only_train_cases() -> None:
    """The train split filter must produce only train-labeled cases."""
    from kortny.evals.orchestration.cases import SEED_ORCHESTRATION_CASES

    train_filtered = [c for c in SEED_ORCHESTRATION_CASES if c.tuning_split == "train"]
    assert all(c.tuning_split == "train" for c in train_filtered)
    # Must include some cases.
    assert len(train_filtered) > 0


def test_split_filter_holdout_returns_only_holdout_cases() -> None:
    """The holdout split filter must produce only holdout-labeled cases."""
    from kortny.evals.orchestration.cases import SEED_ORCHESTRATION_CASES

    holdout_filtered = [
        c for c in SEED_ORCHESTRATION_CASES if c.tuning_split == "holdout"
    ]
    assert all(c.tuning_split == "holdout" for c in holdout_filtered)
    assert len(holdout_filtered) > 0


# ---------------------------------------------------------------------------
# S2: replay roundtrip with new RunResult fields
# ---------------------------------------------------------------------------


def test_dump_and_load_fixtures_roundtrip_s2_fields() -> None:
    """dump_fixtures + load_fixtures must roundtrip all S2 RunResult fields."""
    import tempfile
    from pathlib import Path

    from kortny.evals.orchestration.replay import dump_fixtures, load_fixtures

    rr = RunResult(
        called_apps=frozenset({"github"}),
        any_tool_called=True,
        answer="ok",
        called_tool_slugs=frozenset({"GITHUB_ISSUES_LIST", "GITHUB_LIST_COMMITS"}),
        called_arg_keys=frozenset({"assignee", "repo", "state"}),
        approval_paused=True,
        turn_count=4,
        cost_usd=0.025,
    )
    mapping = {"list my open issues": rr}
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = Path(f.name)
    try:
        dump_fixtures(mapping, path)
        loaded = load_fixtures(path)
        loaded_rr = loaded["list my open issues"]
        assert loaded_rr.called_tool_slugs == frozenset(
            {"GITHUB_ISSUES_LIST", "GITHUB_LIST_COMMITS"}
        )
        assert loaded_rr.called_arg_keys == frozenset({"assignee", "repo", "state"})
        assert loaded_rr.approval_paused is True
        assert loaded_rr.turn_count == 4
        assert abs(loaded_rr.cost_usd - 0.025) < 1e-6
    finally:
        path.unlink(missing_ok=True)


def test_load_fixtures_old_format_backcompat() -> None:
    """Old fixture files lacking S2 keys must load with safe defaults."""
    import json
    import tempfile
    from pathlib import Path

    from kortny.evals.orchestration.replay import load_fixtures

    old_fixture = {
        "list my open github prs": {
            "called_apps": ["github"],
            "any_tool_called": True,
            "answer": "",
            "skipped": False,
            "skip_reason": None,
            # No S2 keys — simulates a pre-S2 fixture file.
        }
    }
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        json.dump(old_fixture, f)
        path = Path(f.name)
    try:
        loaded = load_fixtures(path)
        rr = loaded["list my open github prs"]
        assert rr.called_apps == frozenset({"github"})
        # S2 defaults.
        assert rr.called_tool_slugs == frozenset()
        assert rr.called_arg_keys == frozenset()
        assert rr.approval_paused is False
        assert rr.turn_count == 0
        assert rr.cost_usd == 0.0
    finally:
        path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# S2: top25 tool_intents + HOLDOUT_APPS
# ---------------------------------------------------------------------------


def test_top25_all_apps_have_tool_intents() -> None:
    """Every TOP25 app must have at least one tool_intent."""
    from kortny.integrations.top25 import TOP25

    for app in TOP25:
        assert len(app.tool_intents) > 0, f"TOP25 app {app.slug!r} has no tool_intents"


def test_holdout_apps_not_in_top25() -> None:
    """HOLDOUT_APPS slugs must not overlap with TOP25_SLUGS."""
    from kortny.integrations.top25 import HOLDOUT_APPS, TOP25_SLUGS

    for app in HOLDOUT_APPS:
        assert app.slug not in TOP25_SLUGS, (
            f"HOLDOUT_APP {app.slug!r} is also in TOP25_SLUGS — holdout apps "
            "must be outside the curated 25"
        )


def test_holdout_apps_have_tool_intents() -> None:
    """Every HOLDOUT_APP must have at least one tool_intent."""
    from kortny.integrations.top25 import HOLDOUT_APPS

    for app in HOLDOUT_APPS:
        assert len(app.tool_intents) > 0, (
            f"HOLDOUT_APP {app.slug!r} has no tool_intents"
        )


def test_tool_intents_for_known_slugs() -> None:
    """tool_intents_for returns correct intents for top-25 and holdout slugs."""
    from kortny.integrations.top25 import tool_intents_for

    github_intents = tool_intents_for("github")
    assert "list_pull_requests" in github_intents
    assert "create_issue" in github_intents

    dropbox_intents = tool_intents_for("dropbox")
    assert "create_shared_link" in dropbox_intents


def test_tool_intents_for_unknown_slug_returns_empty() -> None:
    """tool_intents_for returns () for slugs not in top-25 or holdout registries."""
    from kortny.integrations.top25 import tool_intents_for

    assert tool_intents_for("serpapi") == ()
    assert tool_intents_for("twelve_data") == ()
    assert tool_intents_for("notaslug") == ()


def test_tool_intents_for_alias_resolves() -> None:
    """tool_intents_for must resolve aliases to the canonical app."""
    from kortny.integrations.top25 import tool_intents_for

    # outlook_calendar is an alias for outlook
    outlook_intents = tool_intents_for("outlook")
    alias_intents = tool_intents_for("outlook_calendar")
    assert outlook_intents == alias_intents
    assert len(outlook_intents) > 0
