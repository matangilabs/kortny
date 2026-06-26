"""Live runner for provider contract-test probes.

Run manually with a configured provider. Never run in CI.

Requires real provider credentials and will make live API calls that cost
money. Each probe sends one or two LLM requests. Use a cheap model per
provider to minimize cost.

Usage::

    uv run python -m kortny.evals.providers.runner \\
        --provider groq \\
        --model groq/llama-3.1-70b-versatile \\
        --api-key $GROQ_API_KEY

Or configure via env and import:

    from kortny.evals.providers.runner import run_probes_for_target
    from kortny.evals.providers.cases import ProviderProbeTarget, PROVIDER_PROBES
    target = ProviderProbeTarget(
        provider_kind="groq",
        model_identifier="groq/llama-3.1-70b-versatile",
        label="Groq Llama 70B",
    )
    results = run_probes_for_target(target, api_key="sk-...")
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

from kortny.evals.providers.cases import (
    ProviderProbe,
    ProviderProbeTarget,
)
from kortny.evals.providers.scoring import SCORING_FUNCTIONS


@dataclasses.dataclass(frozen=True, slots=True)
class ProbeResult:
    """Result of running a single provider probe."""

    probe_name: str
    passed: bool
    skipped: bool
    skip_reason: str
    check_results: dict[str, bool]
    error: str | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class TargetRunResult:
    """Result of running all probes for one provider target."""

    target: ProviderProbeTarget
    probe_results: tuple[ProbeResult, ...]

    @property
    def passed(self) -> bool:
        return all(r.passed or r.skipped for r in self.probe_results)

    @property
    def pass_count(self) -> int:
        return sum(1 for r in self.probe_results if r.passed)

    @property
    def skip_count(self) -> int:
        return sum(1 for r in self.probe_results if r.skipped)

    @property
    def fail_count(self) -> int:
        return sum(1 for r in self.probe_results if not r.passed and not r.skipped)


def run_probes_for_target(
    target: ProviderProbeTarget,
    *,
    api_key: str | None = None,
    api_base: str | None = None,
    capabilities: Mapping[str, object] | None = None,
) -> TargetRunResult:
    """Run all probes for one provider target. Makes real API calls.

    Parameters
    ----------
    target:
        The provider/model pair to test.
    api_key:
        API key for the provider (pass None for instance-role providers).
    api_base:
        Optional base URL (required for openai_compatible).
    capabilities:
        Pre-fetched capability flags. If omitted, fetched from
        ``normalize_model_capabilities``.
    """
    from kortny.llm.capabilities import normalize_model_capabilities

    resolved_caps: Mapping[str, object] = capabilities or normalize_model_capabilities(
        target.model_identifier
    )

    results: list[ProbeResult] = []
    for probe in target.probes:
        result = _run_probe(
            probe=probe,
            target=target,
            api_key=api_key,
            api_base=api_base,
            capabilities=resolved_caps,
        )
        results.append(result)

    return TargetRunResult(target=target, probe_results=tuple(results))


def _run_probe(
    *,
    probe: ProviderProbe,
    target: ProviderProbeTarget,
    api_key: str | None,
    api_base: str | None,
    capabilities: Mapping[str, object],
) -> ProbeResult:
    # Skip if required capability is not claimed
    if probe.required_capability and not capabilities.get(probe.required_capability):
        return ProbeResult(
            probe_name=probe.name,
            passed=False,
            skipped=True,
            skip_reason=f"capability '{probe.required_capability}' not claimed",
            check_results={},
        )

    try:
        import litellm

        messages = _build_messages(probe)
        kwargs: dict[str, Any] = {
            "model": target.model_identifier,
            "messages": messages,
        }
        if api_key:
            kwargs["api_key"] = api_key
        if api_base:
            kwargs["api_base"] = api_base

        raw = litellm.completion(**kwargs)
        response: dict[str, Any] = dict(raw) if hasattr(raw, "__iter__") else {}
        # Normalize to dict
        if hasattr(raw, "model_dump"):
            response = raw.model_dump()
        elif hasattr(raw, "__dict__"):
            response = vars(raw)
    except Exception as exc:
        return ProbeResult(
            probe_name=probe.name,
            passed=False,
            skipped=False,
            skip_reason="",
            check_results={},
            error=str(exc),
        )

    check_results: dict[str, bool] = {}
    for check_name in probe.expected_checks:
        fn = SCORING_FUNCTIONS.get(check_name)
        if fn is None:
            check_results[check_name] = False
            continue
        try:
            check_results[check_name] = bool(fn(response))
        except Exception:
            check_results[check_name] = False

    passed = all(check_results.values())
    return ProbeResult(
        probe_name=probe.name,
        passed=passed,
        skipped=False,
        skip_reason="",
        check_results=check_results,
    )


def _build_messages(probe: ProviderProbe) -> list[dict[str, Any]]:
    """Build a minimal message list for the probe hint."""
    hint = probe.build_messages_hint
    if not hint:
        return [{"role": "user", "content": "Hello."}]
    kind, _, rest = hint.partition(":")
    if kind == "single_user_message":
        return [{"role": "user", "content": rest}]
    if kind == "tool_call_request":
        # rest = "tool_name:user_message"
        _tool_name, _, user_msg = rest.partition(":")
        return [{"role": "user", "content": user_msg or "Call the tool."}]
    if kind == "json_schema_request":
        return [{"role": "user", "content": rest}]
    if kind == "image_message":
        # Minimal 1x1 white PNG as base64
        tiny_png = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8"
            "z8BQDwADhQGAWjR9awAAAABJRU5ErkJggg=="
        )
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": rest or "Describe this image."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{tiny_png}"},
                    },
                ],
            }
        ]
    return [{"role": "user", "content": hint}]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run provider contract-test probes.")
    parser.add_argument("--provider", required=True, help="Provider kind (e.g. groq)")
    parser.add_argument(
        "--model",
        required=True,
        help="Model identifier (e.g. groq/llama-3.1-70b-versatile)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key (optional for instance-role providers)",
    )
    parser.add_argument(
        "--api-base", default=None, help="Base URL (for openai_compatible)"
    )
    parser.add_argument("--label", default=None, help="Human-readable label")
    args = parser.parse_args()

    probe_target = ProviderProbeTarget(
        provider_kind=args.provider,
        model_identifier=args.model,
        label=args.label or f"{args.provider} / {args.model}",
    )
    run_result = run_probes_for_target(
        probe_target,
        api_key=args.api_key,
        api_base=args.api_base,
    )
    print(f"\nProvider probe results for: {run_result.target.label}")
    print(
        f"  Passed: {run_result.pass_count} / Skip: {run_result.skip_count} / Fail: {run_result.fail_count}"
    )
    for pr in run_result.probe_results:
        status = "SKIP" if pr.skipped else ("PASS" if pr.passed else "FAIL")
        print(f"  [{status}] {pr.probe_name}")
        if pr.error:
            print(f"         error: {pr.error}")
        for check_name, check_passed in pr.check_results.items():
            mark = "✓" if check_passed else "✗"
            print(f"         {mark} {check_name}")
