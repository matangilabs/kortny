import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from kortny.config import Settings
from kortny.db.models import Task
from kortny.intent.depth_overrides import classify_depth_override
from kortny.workflow.handoff import evaluate_runtime_handoff

BENCHMARK_PATH = Path("tests/fixtures/kortny_product_benchmark.json")

ALLOWED_GUARDRAILS = {
    "slack_response_quality",
    "latency_cost",
    "correct_context_tool_selection",
    "operator_trust_debuggability",
}
ALLOWED_ROUTES = {"inline", "planned_candidate"}
ALLOWED_RUNTIME_CLASSES = {
    "quick_response",
    "inline_tool_task",
    "durable_workflow_task",
    "scheduled_workflow_task",
}
ALLOWED_EXECUTION_PATHS = {
    "inline",
    "durable_workflow",
    "scheduled_workflow",
}
ALLOWED_BACKEND_BASELINES = {"inline", "temporal_candidate"}
ALLOWED_ROUTING_RISKS = {
    "none",
    "regression_guard",
    "over_routing",
    "under_routing",
    "classifier_over_routing",
    "classifier_under_routing",
    "multi_intent_collapse",
    "tool_inventory_accuracy",
}
EXPECTED_SURFACES = {"dm", "channel_mention"}
EXPECTED_SECTIONS = {"slack", "logs", "db", "dashboard"}
EXPECTED_ROUTING_KEYS = {
    "expected_runtime_class",
    "expected_intent",
    "expected_execution_path",
    "expected_tool_families",
    "current_runtime_class_baseline",
    "current_backend_baseline",
    "routing_risk",
}


def test_product_benchmark_fixture_is_well_formed() -> None:
    benchmark = _load_benchmark()
    scenarios = benchmark["scenarios"]
    ids = [scenario["id"] for scenario in scenarios]
    routing_taxonomy = benchmark["routing_taxonomy"]

    assert benchmark["version"] == 2
    assert routing_taxonomy["version"] == 1
    assert set(routing_taxonomy["runtime_classes"]) == ALLOWED_RUNTIME_CLASSES
    assert set(routing_taxonomy["execution_paths"]) == ALLOWED_EXECUTION_PATHS
    assert set(routing_taxonomy["risk_types"]) == ALLOWED_ROUTING_RISKS
    assert len(scenarios) == 12
    assert len(ids) == len(set(ids))
    assert {
        scenario["primary_guardrail"] for scenario in scenarios
    } == ALLOWED_GUARDRAILS
    assert {scenario["surface"] for scenario in scenarios} == EXPECTED_SURFACES

    for scenario in scenarios:
        assert scenario["desired_runtime_route"] in ALLOWED_ROUTES
        assert scenario["baseline_classifier_route"] in ALLOWED_ROUTES
        assert scenario["primary_guardrail"] in ALLOWED_GUARDRAILS
        assert scenario["prompt"].strip()
        routing = scenario["routing_expectation"]
        assert set(routing) == EXPECTED_ROUTING_KEYS
        assert routing["expected_runtime_class"] in ALLOWED_RUNTIME_CLASSES
        assert routing["expected_intent"].strip()
        assert routing["expected_execution_path"] in ALLOWED_EXECUTION_PATHS
        assert isinstance(routing["expected_tool_families"], list)
        assert all(
            isinstance(tool_family, str) and tool_family.strip()
            for tool_family in routing["expected_tool_families"]
        )
        assert routing["current_runtime_class_baseline"] in ALLOWED_RUNTIME_CLASSES
        assert routing["current_backend_baseline"] in ALLOWED_BACKEND_BASELINES
        assert routing["routing_risk"] in ALLOWED_ROUTING_RISKS
        assert set(scenario["expected"]) == EXPECTED_SECTIONS
        for checks in scenario["expected"].values():
            assert isinstance(checks, list)
            assert checks
            assert all(isinstance(check, str) and check.strip() for check in checks)


def test_product_benchmark_records_current_depth_override_baseline() -> None:
    benchmark = _load_benchmark()

    for scenario in benchmark["scenarios"]:
        override = classify_depth_override(text=scenario["prompt"])
        # Map the unified-router deterministic depth override onto the legacy
        # planned/inline route taxonomy used by the benchmark fixture: only
        # deep_workflow overrides are planned candidates; everything else (no
        # forced depth, quick, standard) is inline.
        route = (
            "planned_candidate"
            if override is not None and override.depth == "deep_workflow"
            else "inline"
        )

        assert route == scenario["baseline_classifier_route"], (
            f"{scenario['id']} depth-override route changed from benchmark "
            "baseline. If this is an intended product improvement, update the "
            "benchmark baseline and known_gap note."
        )


def test_product_benchmark_records_current_handoff_baseline() -> None:
    benchmark = _load_benchmark()

    for scenario in benchmark["scenarios"]:
        decision = evaluate_runtime_handoff(
            settings=_settings(),
            task=_task(scenario["prompt"]),
        )
        expected = scenario["routing_expectation"]
        backend_baseline = (
            "temporal_candidate"
            if decision.recommended_backend == "temporal"
            else "inline"
        )

        assert (
            decision.runtime_class.value == expected["current_runtime_class_baseline"]
        ), (
            f"{scenario['id']} handoff runtime changed from benchmark baseline. "
            "If this is an intended routing improvement, update the benchmark "
            "baseline and routing_risk note."
        )
        assert backend_baseline == expected["current_backend_baseline"], (
            f"{scenario['id']} handoff backend recommendation changed from "
            "benchmark baseline. If intended, update the benchmark."
        )


def test_product_benchmark_tracks_known_route_gaps() -> None:
    benchmark = _load_benchmark()
    known_gap_ids = {
        scenario["id"]
        for scenario in benchmark["scenarios"]
        if scenario["desired_runtime_route"] != scenario["baseline_classifier_route"]
    }

    assert known_gap_ids == {
        "linear_project_tasks",
        "james_bond_ranked_research",
        "website_cpt_audit",
        "pypl_report_iteration",
    }
    for scenario in benchmark["scenarios"]:
        if scenario["id"] in known_gap_ids:
            assert scenario["known_gap"]


def test_product_benchmark_tracks_current_runtime_risks() -> None:
    benchmark = _load_benchmark()
    runtime_risk_ids = {
        scenario["id"]
        for scenario in benchmark["scenarios"]
        if scenario["routing_expectation"]["routing_risk"] != "none"
    }

    assert runtime_risk_ids == {
        "schedule_state_query",
        "memory_preference_followup",
        "tool_inventory_question",
        "linear_project_tasks",
        "james_bond_ranked_research",
        "website_cpt_audit",
        "memory_forget_no_match",
        "pypl_report_iteration",
    }
    for scenario in benchmark["scenarios"]:
        expected = scenario["routing_expectation"]
        if expected["routing_risk"] == "none":
            assert scenario["known_gap"] is None
        else:
            assert scenario["known_gap"] or expected["routing_risk"] in {
                "regression_guard",
                "tool_inventory_accuracy",
            }


def _load_benchmark() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(BENCHMARK_PATH.read_text()))


def _task(input_text: str) -> Task:
    return cast(
        Task,
        SimpleNamespace(
            input=input_text,
            identity_kind=None,
        ),
    )


def _settings() -> Settings:
    return Settings.model_validate(
        {
            "SLACK_BOT_TOKEN": "xoxb-test",
            "SLACK_APP_TOKEN": "xapp-test",
            "SLACK_SIGNING_SECRET": "secret",
            "LLM_PROVIDER": "openrouter",
            "LLM_API_KEY": "llm-key",
            "LLM_MODEL": "openai/gpt-4o",
            "COMPOSIO_API_KEY": "composio-key",
            "POSTGRES_URL": "postgresql://kortny:kortny@localhost/kortny",
            "KORTNY_WORKFLOW_BACKEND": "inline",
        }
    )
