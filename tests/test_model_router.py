from kortny.config.settings import LLMProvider as SettingsLLMProvider
from kortny.config.settings import Settings
from kortny.db.models import Task, TaskEvent, TaskEventType
from kortny.llm import ModelRouter, ModelRouteTier


def make_settings() -> Settings:
    return Settings.model_validate(
        {
            "SLACK_BOT_TOKEN": "xoxb-test",
            "SLACK_APP_TOKEN": "xapp-test",
            "SLACK_SIGNING_SECRET": "signing-secret",
            "LLM_PROVIDER": SettingsLLMProvider.openrouter,
            "LLM_API_KEY": "openrouter-key",
            "LLM_MODEL": "openai/default",
            "LLM_CHEAP_MODEL": "anthropic/haiku",
            "LLM_STANDARD_MODEL": "openai/standard",
            "LLM_ANALYSIS_MODEL": "anthropic/sonnet",
            "LLM_DOCUMENT_MODEL": "openai/document",
            "LLM_HIGH_REASONING_MODEL": "openai/reasoning",
            "POSTGRES_URL": "postgresql://kortny:kortny@localhost/kortny",
        }
    )


def test_model_router_resolves_configured_tiers() -> None:
    router = ModelRouter(make_settings())

    assert (
        router.route_for_tier(
            ModelRouteTier.cheap_fast,
            reason="test",
        ).model
        == "anthropic/haiku"
    )
    assert (
        router.route_for_tier(
            ModelRouteTier.document,
            reason="test",
        ).model
        == "openai/document"
    )


def test_model_router_uses_intent_decision_before_text_fallback() -> None:
    router = ModelRouter(make_settings())
    task = Task(input="Can you answer this quickly?")
    event = TaskEvent(
        seq=1,
        type=TaskEventType.log,
        payload={
            "message": "intent_classification_completed",
            "decision": {
                "classification": "task_request",
                "model_tier": "cheap",
                "likely_tools": ["pdf_generator"],
            },
        },
    )

    route = router.route_for_task(task, events=[event])

    assert route.tier is ModelRouteTier.document
    assert route.model == "openai/document"
    assert route.reason == "intent_classifier"


def test_model_router_routes_file_review_to_analysis() -> None:
    router = ModelRouter(make_settings())
    task = Task(input="Please review the attached file")
    event = TaskEvent(
        seq=1,
        type=TaskEventType.log,
        payload={
            "message": "intent_classification_completed",
            "decision": {
                "classification": "task_request",
                "model_tier": "standard",
                "likely_tools": ["slack_file_read"],
            },
        },
    )

    route = router.route_for_task(task, events=[event])

    assert route.tier is ModelRouteTier.analysis
    assert route.model == "anthropic/sonnet"


def test_model_router_uses_task_text_fallbacks() -> None:
    router = ModelRouter(make_settings())

    document_route = router.route_for_task(Task(input="make a 3 page PDF report"))
    analysis_route = router.route_for_task(Task(input="research recent market trends"))
    standard_route = router.route_for_task(Task(input="what can you do?"))

    assert document_route.tier is ModelRouteTier.document
    assert analysis_route.tier is ModelRouteTier.analysis
    assert standard_route.tier is ModelRouteTier.standard
