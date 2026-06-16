"""Eval harness CLI.

Run the cheap_fast bake-off over the bundled intent cases:

    uv run python -m kortny.evals run \
        --model openai/gpt-4o-mini --model google/gemini-2.0-flash --repeats 3

With no ``--model`` it evaluates the configured cheap tier (the incumbent), so
it runs out of the box. THIS COMMAND CALLS THE LIVE PROVIDER and costs money;
it needs LLM_API_KEY. Everything else in kortny.evals is pure functions.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from kortny.config import load_settings
from kortny.evals.cases import load_cases
from kortny.evals.providers import ModelCandidate
from kortny.evals.runner import run_matrix
from kortny.evals.scoring import aggregate, render_table, summaries_to_json


def build_parser() -> argparse.ArgumentParser:
    """Build the eval harness argument parser."""

    parser = argparse.ArgumentParser(
        prog="python -m kortny.evals",
        description="Offline agent eval harness (cheap_fast bake-off).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Run the case matrix against one or more candidate models.",
    )
    run_parser.add_argument(
        "--cases",
        type=Path,
        default=None,
        help="Path to a case file (defaults to the bundled cheap_fast cases).",
    )
    run_parser.add_argument(
        "--model",
        action="append",
        dest="models",
        default=None,
        help="Candidate model id (repeatable). Omit to use the configured cheap tier.",
    )
    run_parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Runs per case (default 1); raise for steadier latency percentiles.",
    )
    run_parser.add_argument(
        "--format",
        choices=("table", "json"),
        default="table",
        help="Output format (default table).",
    )
    run_parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to write the JSON summary.",
    )
    return parser


def _candidates(
    models: Sequence[str] | None, settings_cheap: str
) -> list[ModelCandidate]:
    if models:
        return [ModelCandidate(name=model, model=model) for model in models]
    return [ModelCandidate(name=f"cheap:{settings_cheap}", model=settings_cheap)]


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint; returns a process exit code."""

    args = build_parser().parse_args(argv)
    settings = load_settings()
    incumbent = settings.llm_cheap_model or settings.llm_model
    candidates = _candidates(args.models, incumbent)
    case_file = load_cases(args.cases)

    results = run_matrix(
        settings,
        candidates,
        case_file.intent_cases,
        repeats=max(1, args.repeats),
    )
    summaries = aggregate(results)

    payload = {"summaries": summaries_to_json(summaries)}
    if args.out is not None:
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)

    if args.format == "json":
        print(json.dumps(payload, indent=2))
    else:
        print(render_table(summaries))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
