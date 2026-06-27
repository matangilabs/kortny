"""Replay module for the cross-app orchestration eval.

Provides fixture serialization/deserialization and a replay RunFn so the eval
can be scored offline — no live agent, no DB, no API keys.

Workflow
--------
1. Run ``make eval`` (live, needs real infra) to execute the orchestration cases
   against a live install. The runner records each ``RunResult`` to the fixtures
   file automatically.
2. Commit the updated ``fixtures/smoke_goldens.json``.
3. Run ``make eval-smoke`` (offline) at any time: loads committed fixtures,
   builds the replay RunFn, scores the smoke subset — $0, no secrets required.

Fixture format (``fixtures/smoke_goldens.json``)
------------------------------------------------
A JSON object keyed by ``case.request`` (the stable case identifier).  Each
value is a ``RunResult`` encoded as:

.. code-block:: json

    {
      "<request string>": {
        "called_apps": ["github", "linear"],
        "any_tool_called": true,
        "answer": "",
        "skipped": false,
        "skip_reason": null
      }
    }

``called_apps`` is a list (sorted for determinism); ``frozenset`` is
reconstructed on load.

CLI
---
Run the replay directly::

    uv run python -m kortny.evals.orchestration.replay
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from kortny.evals.orchestration.cases import SEED_ORCHESTRATION_CASES, OrchestrationCase
from kortny.evals.orchestration.scoring import (
    OrchestrationAssertionResult,
    OrchestrationReport,
    RunFn,
    RunResult,
    score_orchestration,
)

FIXTURES_DIR: Path = Path(__file__).parent / "fixtures"
DEFAULT_FIXTURES_PATH: Path = FIXTURES_DIR / "smoke_goldens.json"


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _run_result_to_dict(result: RunResult) -> dict[str, Any]:
    """Serialize a RunResult to a JSON-compatible dict."""
    return {
        "called_apps": sorted(result.called_apps),
        "any_tool_called": result.any_tool_called,
        "answer": result.answer,
        "skipped": result.skipped,
        "skip_reason": result.skip_reason if result.skip_reason else None,
    }


def _dict_to_run_result(data: dict[str, Any]) -> RunResult:
    """Deserialize a dict (from JSON) back to a RunResult."""
    return RunResult(
        called_apps=frozenset(data.get("called_apps") or []),
        any_tool_called=bool(data.get("any_tool_called", False)),
        answer=str(data.get("answer") or ""),
        skipped=bool(data.get("skipped", False)),
        skip_reason=str(data.get("skip_reason") or ""),
    )


# ---------------------------------------------------------------------------
# Public API: dump / load
# ---------------------------------------------------------------------------


def dump_fixtures(
    results: Sequence[OrchestrationAssertionResult] | Mapping[str, RunResult],
    path: Path,
) -> None:
    """Write RunResults to a fixtures JSON file keyed by case request.

    Accepts either:
    - A ``Sequence[OrchestrationAssertionResult]`` as returned by the scorer
      (carries both the request string and the raw RunResult fields).
    - A ``Mapping[str, RunResult]`` keyed by request string directly.

    Merges into any existing fixture file so a partial re-run does not wipe
    cases that were not re-executed.
    """
    existing: dict[str, Any] = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            existing = {}

    if isinstance(results, Mapping):
        new_entries: dict[str, Any] = {
            req: _run_result_to_dict(rr) for req, rr in results.items()
        }
    else:
        new_entries = {}
        for ar in results:
            rr = RunResult(
                called_apps=ar.called_apps,
                any_tool_called=ar.any_tool_called,
                answer="",
                skipped=ar.skipped,
                skip_reason=ar.skip_reason,
            )
            new_entries[ar.request] = _run_result_to_dict(rr)

    merged = {**existing, **new_entries}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")


def load_fixtures(path: Path) -> dict[str, RunResult]:
    """Load a fixtures JSON file and return a dict keyed by case request.

    Returns an empty dict if the file does not exist.
    """
    if not path.exists():
        return {}
    raw: dict[str, Any] = json.loads(path.read_text())
    return {req: _dict_to_run_result(data) for req, data in raw.items()}


# ---------------------------------------------------------------------------
# Replay RunFn
# ---------------------------------------------------------------------------


def build_replay_run_fn(fixtures: Mapping[str, RunResult]) -> RunFn:
    """Return a RunFn that looks up pre-recorded results from ``fixtures``.

    For each case, the request string is used as the lookup key.  If no fixture
    is found, the case is returned as skipped so the scorer excludes it from the
    pass rate — the operator knows which cases need a live re-run to produce
    goldens.

    This RunFn is $0 and needs no secrets, DB, or network.
    """

    def run_fn(case: OrchestrationCase) -> RunResult:
        recorded = fixtures.get(case.request)
        if recorded is None:
            return RunResult(
                called_apps=frozenset(),
                any_tool_called=False,
                answer="",
                skipped=True,
                skip_reason="no replay fixture recorded",
            )
        return recorded

    return run_fn


# ---------------------------------------------------------------------------
# Smoke replay: score only smoke=True cases
# ---------------------------------------------------------------------------


def run_replay(
    *,
    fixtures_path: Path = DEFAULT_FIXTURES_PATH,
    smoke_only: bool = True,
) -> OrchestrationReport:
    """Load committed fixtures and score the orchestration cases offline.

    Args:
        fixtures_path: Path to the fixtures JSON file.
        smoke_only: When True (default), only score cases with ``smoke=True``.
            Set to False to replay all seed cases (non-smoke ones without a
            fixture will appear as skipped).

    Returns:
        An ``OrchestrationReport`` scored purely from fixture data.
    """
    fixtures = load_fixtures(fixtures_path)
    replay_fn = build_replay_run_fn(fixtures)
    cases = (
        [c for c in SEED_ORCHESTRATION_CASES if c.smoke]
        if smoke_only
        else list(SEED_ORCHESTRATION_CASES)
    )
    return score_orchestration(cases, replay_fn)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _main_replay() -> None:
    """CLI: run offline smoke replay and print results."""
    report = run_replay()
    print(f"\nORCHESTRATION SMOKE REPLAY: {report.summary_line()}\n")
    for result in report.results:
        if result.skipped:
            print(f"  [SKIP] {result.request!r}")
            print(f"         reason: {result.skip_reason}")
            continue
        status = "PASS" if result.passed else "FAIL"
        apps_label = (
            f"called={sorted(result.called_apps)!r} "
            f"expected={sorted(result.expected_apps)!r}"
        )
        print(f"  [{status}] {result.request!r}")
        print(f"         {apps_label}")
        for failure in result.failures:
            print(f"         ! {failure}")
    print()
    if not report.failures:
        print("All smoke cases passed (replay).")
    else:
        print(f"{report.failed} smoke case(s) failed — see above for details.")
    sys.exit(0 if not report.failures else 1)


if __name__ == "__main__":
    _main_replay()
