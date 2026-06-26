"""CI-safe structural tests for the provider eval probe definitions.

No live API calls. No LLM calls. Validates that probe and scoring definitions
are well-formed and self-consistent.
"""

from __future__ import annotations

from kortny.evals.providers.cases import (
    PROVIDER_PROBES,
    TEXT_COMPLETION_PROBE,
    TOOL_CALL_PROBE,
    VISION_PROBE,
    ProviderProbeTarget,
)
from kortny.evals.providers.scoring import SCORING_FUNCTIONS


class TestProbeDefinitions:
    def test_all_probes_have_names(self) -> None:
        for probe in PROVIDER_PROBES:
            assert probe.name, f"Probe must have a non-empty name: {probe}"

    def test_all_probes_have_descriptions(self) -> None:
        for probe in PROVIDER_PROBES:
            assert probe.description, f"Probe must have a description: {probe.name}"

    def test_all_expected_checks_exist_in_scoring(self) -> None:
        """Every check name referenced in a probe must exist in SCORING_FUNCTIONS."""
        for probe in PROVIDER_PROBES:
            for check_name in probe.expected_checks:
                assert check_name in SCORING_FUNCTIONS, (
                    f"Probe '{probe.name}' references unknown check '{check_name}'. "
                    f"Available: {sorted(SCORING_FUNCTIONS)}"
                )

    def test_probe_names_unique(self) -> None:
        names = [probe.name for probe in PROVIDER_PROBES]
        assert len(names) == len(set(names)), "Probe names must be unique"

    def test_required_capability_is_string(self) -> None:
        for probe in PROVIDER_PROBES:
            assert isinstance(probe.required_capability, str)

    def test_canonical_probes_present(self) -> None:
        names = {p.name for p in PROVIDER_PROBES}
        assert "text_completion" in names
        assert "tool_call" in names
        assert "json_structured_output" in names
        assert "vision_if_claimed" in names
        assert "cost_reporting_present" in names

    def test_text_completion_has_no_required_capability(self) -> None:
        assert TEXT_COMPLETION_PROBE.required_capability == ""

    def test_tool_call_requires_tools_capability(self) -> None:
        assert TOOL_CALL_PROBE.required_capability == "tools"

    def test_vision_requires_vision_capability(self) -> None:
        assert VISION_PROBE.required_capability == "vision"


class TestScoringFunctions:
    def test_scoring_functions_are_callable(self) -> None:
        for name, fn in SCORING_FUNCTIONS.items():
            assert callable(fn), f"Scoring function '{name}' must be callable"

    def test_response_has_content_pass(self) -> None:
        fn = SCORING_FUNCTIONS["response_has_content"]
        assert callable(fn)
        response = {"choices": [{"message": {"content": "Hello world"}}]}
        assert fn(response) is True

    def test_response_has_content_fail_empty(self) -> None:
        fn = SCORING_FUNCTIONS["response_has_content"]
        assert callable(fn)
        response: dict[str, object] = {"choices": []}
        assert fn(response) is False

    def test_no_error_response_pass(self) -> None:
        fn = SCORING_FUNCTIONS["no_error_response"]
        assert callable(fn)
        assert fn({"choices": []}) is True

    def test_no_error_response_fail(self) -> None:
        fn = SCORING_FUNCTIONS["no_error_response"]
        assert callable(fn)
        assert fn({"error": "rate_limited"}) is False

    def test_tool_call_present_pass(self) -> None:
        fn = SCORING_FUNCTIONS["tool_call_present"]
        assert callable(fn)
        response = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "1",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": "{}",
                                },
                            }
                        ],
                    }
                }
            ]
        }
        assert fn(response) is True

    def test_tool_call_present_fail_no_calls(self) -> None:
        fn = SCORING_FUNCTIONS["tool_call_present"]
        assert callable(fn)
        response = {"choices": [{"message": {"content": "The weather is sunny."}}]}
        assert fn(response) is False

    def test_response_is_valid_json_pass(self) -> None:
        fn = SCORING_FUNCTIONS["response_is_valid_json"]
        assert callable(fn)
        response = {"choices": [{"message": {"content": '{"result": 42}'}}]}
        assert fn(response) is True

    def test_response_is_valid_json_fail(self) -> None:
        fn = SCORING_FUNCTIONS["response_is_valid_json"]
        assert callable(fn)
        response = {"choices": [{"message": {"content": "not json"}}]}
        assert fn(response) is False

    def test_usage_tokens_present_pass(self) -> None:
        fn = SCORING_FUNCTIONS["usage_tokens_present"]
        assert callable(fn)
        response = {"usage": {"prompt_tokens": 10, "completion_tokens": 20}}
        assert fn(response) is True

    def test_usage_tokens_present_fail(self) -> None:
        fn = SCORING_FUNCTIONS["usage_tokens_present"]
        assert callable(fn)
        assert fn({}) is False


class TestProviderProbeTarget:
    def test_default_probes_are_all_probes(self) -> None:
        target = ProviderProbeTarget(
            provider_kind="groq",
            model_identifier="groq/llama-3.1-70b-versatile",
            label="Groq test",
        )
        assert target.probes == PROVIDER_PROBES

    def test_custom_probes_subset(self) -> None:
        target = ProviderProbeTarget(
            provider_kind="groq",
            model_identifier="groq/llama-3.1-70b-versatile",
            label="Groq text only",
            probes=(TEXT_COMPLETION_PROBE,),
        )
        assert len(target.probes) == 1
        assert target.probes[0].name == "text_completion"
