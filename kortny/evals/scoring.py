"""Aggregate per-call results into a per-candidate comparison table."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from kortny.evals.runner import CaseResult


def _percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile (q in [0, 1]); 0.0 for empty input."""

    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


@dataclass(frozen=True, slots=True)
class CandidateSummary:
    """Rolled-up metrics for one candidate across all its case runs."""

    name: str
    n_calls: int
    n_errors: int
    json_validity_rate: float
    intent_label_accuracy: float
    latency_p50_ms: float
    latency_p95_ms: float
    cost_per_call_usd: float


def aggregate(results: Sequence[CaseResult]) -> list[CandidateSummary]:
    """Group results by candidate and compute the comparison metrics."""

    grouped: dict[str, list[CaseResult]] = {}
    for result in results:
        grouped.setdefault(result.candidate_name, []).append(result)

    summaries: list[CandidateSummary] = []
    for name, runs in grouped.items():
        # JSON-validity is scored over calls that produced content (provider
        # transport failures are counted separately as errors, not as invalid
        # JSON, so a flaky network doesn't read as a model defect).
        errors = [r for r in runs if _is_transport_error(r)]
        completed = [r for r in runs if not _is_transport_error(r)]
        valid = [r for r in completed if r.json_valid]
        labeled = [r for r in completed if r.label_correct is not None]
        correct = [r for r in labeled if r.label_correct]
        latencies = [float(r.latency_ms) for r in completed]
        costs = [float(r.cost_usd) for r in runs if r.cost_usd is not None]
        summaries.append(
            CandidateSummary(
                name=name,
                n_calls=len(runs),
                n_errors=len(errors),
                json_validity_rate=_safe_ratio(len(valid), len(completed)),
                intent_label_accuracy=_safe_ratio(len(correct), len(labeled)),
                latency_p50_ms=_percentile(latencies, 0.5),
                latency_p95_ms=_percentile(latencies, 0.95),
                cost_per_call_usd=(sum(costs) / len(costs)) if costs else 0.0,
            )
        )
    summaries.sort(key=lambda summary: summary.name)
    return summaries


def _is_transport_error(result: CaseResult) -> bool:
    """A provider/transport failure: errored before producing parseable content."""

    return (
        result.error is not None
        and result.label_correct is None
        and (result.input_tokens == 0 and result.output_tokens == 0)
    )


def _safe_ratio(numerator: int, denominator: int) -> float:
    return (numerator / denominator) if denominator else 0.0


def render_table(summaries: Sequence[CandidateSummary]) -> str:
    """Render a fixed-width comparison table."""

    header = (
        f"{'model':<28} {'calls':>5} {'err':>4} {'json%':>6} "
        f"{'intent%':>8} {'p50ms':>7} {'p95ms':>7} {'$/call':>10}"
    )
    rows = [header, "-" * len(header)]
    for summary in summaries:
        rows.append(
            f"{summary.name[:28]:<28} {summary.n_calls:>5} {summary.n_errors:>4} "
            f"{summary.json_validity_rate * 100:>5.0f}% "
            f"{summary.intent_label_accuracy * 100:>7.0f}% "
            f"{summary.latency_p50_ms:>7.0f} {summary.latency_p95_ms:>7.0f} "
            f"{summary.cost_per_call_usd:>10.5f}"
        )
    return "\n".join(rows)


def summaries_to_json(summaries: Sequence[CandidateSummary]) -> list[dict[str, object]]:
    """Machine-readable form of the comparison table."""

    return [
        {
            "name": s.name,
            "n_calls": s.n_calls,
            "n_errors": s.n_errors,
            "json_validity_rate": s.json_validity_rate,
            "intent_label_accuracy": s.intent_label_accuracy,
            "latency_p50_ms": s.latency_p50_ms,
            "latency_p95_ms": s.latency_p95_ms,
            "cost_per_call_usd": s.cost_per_call_usd,
        }
        for s in summaries
    ]
