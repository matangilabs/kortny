"""Offline CLI runner for the response-pipeline substance-drop eval (HIG-287).

Runs all seed cases through the guard + scorer and prints a discrimination /
pass-rate summary.  No DB, LLM, or rendering required — runs on demand.

    uv run python -m kortny.evals.response_pipeline.runner
"""

from __future__ import annotations

import logging

from kortny.evals.response_pipeline.cases import SEED_RESPONSE_PIPELINE_CASES
from kortny.evals.response_pipeline.scoring import (
    ResponsePipelineReport,
    score_response_pipeline,
)

logger = logging.getLogger(__name__)


def run() -> ResponsePipelineReport:
    return score_response_pipeline(SEED_RESPONSE_PIPELINE_CASES)


def _main() -> None:
    logging.basicConfig(level=logging.WARNING)
    report = run()
    print(f"\nRESPONSE PIPELINE EVAL: {report.summary_line}\n")
    for result in report.results:
        status = "PASS" if result.passed else "FAIL"
        guard_label = "prerender=FIRED" if result.guard_fired else "prerender=silent"
        post_guard_label = (
            "postrender=FIRED"
            if result.post_render_guard_fired
            else "postrender=silent"
        )
        expected = (
            "expected=fire" if result.expects_guard_trigger else "expected=silent"
        )
        token_note = (
            f"missing_tokens={result.missing_tokens}" if result.missing_tokens else ""
        )
        parts = [
            f"  [{status}]",
            f"{result.case_name:<40}",
            guard_label,
            post_guard_label,
            expected,
        ]
        if token_note:
            parts.append(token_note)
        print(" ".join(parts))

    if report.discrimination < 1.0:
        mismatches = [
            r for r in report.results if r.guard_fired != r.expects_guard_trigger
        ]
        print(f"\nPre-render guard mismatches ({len(mismatches)}):")
        for m in mismatches:
            print(
                f"  {m.case_name}: guard_fired={m.guard_fired}, "
                f"expects={m.expects_guard_trigger}"
            )
    if report.post_render_discrimination < 1.0:
        mismatches_post = [
            r
            for r in report.results
            if r.post_render_guard_fired != r.expects_post_render_guard_trigger
        ]
        print(f"\nPost-render guard mismatches ({len(mismatches_post)}):")
        for m in mismatches_post:
            print(
                f"  {m.case_name}: post_render_guard_fired={m.post_render_guard_fired}, "
                f"expects={m.expects_post_render_guard_trigger}"
            )


if __name__ == "__main__":
    _main()
