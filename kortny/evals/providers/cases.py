"""Provider contract-test probe definitions.

Each probe targets one semantic capability: text completion, tool calling,
JSON-schema structured output, vision (if the model claims it), and whether
cost/usage metadata is present in the response.

These are pure data definitions. The live runner (runner.py) executes them
against a real provider. This file has no external imports and is CI-safe.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ProviderProbe:
    """A single capability probe for a provider/model pair.

    Attributes:
        name: Short identifier (e.g. ``"text_completion"``).
        required_capability: Capability key from ``normalize_model_capabilities``
            that must be True for this probe to run. Empty string means always run.
        description: Human-readable description of what is being tested.
        expected_checks: Names of scoring functions that must pass.
    """

    name: str
    required_capability: str
    description: str
    expected_checks: tuple[str, ...]
    build_messages_hint: str = ""
    """Hint for the runner about how to build the messages list for this probe."""


# ---------------------------------------------------------------------------
# Canonical probe set
# ---------------------------------------------------------------------------

TEXT_COMPLETION_PROBE = ProviderProbe(
    name="text_completion",
    required_capability="",
    description="Basic text generation: the model must return a non-empty string response.",
    expected_checks=("response_has_content", "no_error_response"),
    build_messages_hint="single_user_message:Say hello in one sentence.",
)

TOOL_CALL_PROBE = ProviderProbe(
    name="tool_call",
    required_capability="tools",
    description=(
        "Tool/function calling: the model must invoke the provided dummy tool "
        "rather than answering inline."
    ),
    expected_checks=("tool_call_present", "no_error_response"),
    build_messages_hint="tool_call_request:get_current_weather:What is the weather in Paris?",
)

JSON_STRUCTURED_OUTPUT_PROBE = ProviderProbe(
    name="json_structured_output",
    required_capability="structured_output",
    description=(
        "JSON schema enforcement: the model must return a response that is valid "
        "JSON conforming to a minimal schema."
    ),
    expected_checks=("response_is_valid_json", "no_error_response"),
    build_messages_hint="json_schema_request:Return JSON with key 'result' set to 42.",
)

VISION_PROBE = ProviderProbe(
    name="vision_if_claimed",
    required_capability="vision",
    description=(
        "Vision: the model must describe a provided image when vision is claimed "
        "in capabilities."
    ),
    expected_checks=("response_has_content", "no_error_response"),
    build_messages_hint="image_message:Describe this image briefly.",
)

COST_REPORTING_PROBE = ProviderProbe(
    name="cost_reporting_present",
    required_capability="",
    description=(
        "Cost/usage reporting: the response must carry token usage metadata "
        "(prompt_tokens and completion_tokens) so cost attribution works."
    ),
    expected_checks=("usage_tokens_present", "no_error_response"),
    build_messages_hint="single_user_message:Hello.",
)


PROVIDER_PROBES: tuple[ProviderProbe, ...] = (
    TEXT_COMPLETION_PROBE,
    TOOL_CALL_PROBE,
    JSON_STRUCTURED_OUTPUT_PROBE,
    VISION_PROBE,
    COST_REPORTING_PROBE,
)


@dataclass(frozen=True, slots=True)
class ProviderProbeTarget:
    """Binds a provider account to the probe set.

    Attributes:
        provider_kind: LiteLLM provider kind (e.g. ``"groq"``).
        model_identifier: Model identifier to test (e.g. ``"groq/llama-3.1-70b-versatile"``).
        label: Human-readable label for reports.
        probes: Subset of ``PROVIDER_PROBES`` to run (defaults to all).
    """

    provider_kind: str
    model_identifier: str
    label: str
    probes: tuple[ProviderProbe, ...] = field(default=PROVIDER_PROBES)
