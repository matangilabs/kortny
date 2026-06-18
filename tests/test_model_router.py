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
            "LLM_HUMANIZER_MODEL": "anthropic/humanizer",
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
    assert (
        router.route_for_tier(
            ModelRouteTier.humanizer,
            reason="test",
        ).model
        == "anthropic/humanizer"
    )


def test_model_router_humanizer_falls_back_to_cheap_before_standard() -> None:
    # HIG-268: with LLM_HUMANIZER_MODEL unset, the humanizer (a stylistic
    # rewrite on the response critical path) must resolve to the cheap/fast
    # tier, not the slower standard tier that previously sat on the path.
    settings = Settings.model_validate(
        {
            "SLACK_BOT_TOKEN": "xoxb-test",
            "SLACK_APP_TOKEN": "xapp-test",
            "SLACK_SIGNING_SECRET": "signing-secret",
            "LLM_PROVIDER": SettingsLLMProvider.openrouter,
            "LLM_API_KEY": "openrouter-key",
            "LLM_MODEL": "openai/default",
            "LLM_CHEAP_MODEL": "anthropic/haiku",
            "LLM_STANDARD_MODEL": "openai/standard",
            # LLM_HUMANIZER_MODEL intentionally unset.
            "POSTGRES_URL": "postgresql://kortny:kortny@localhost/kortny",
        }
    )
    router = ModelRouter(settings)

    route = router.route_for_tier(ModelRouteTier.humanizer, reason="test")

    assert route.model == "anthropic/haiku"


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


def test_model_router_routes_document_generation_intent_to_document() -> None:
    # HIG-265: a report/document deliverable (signalled via document_generation
    # in likely_tools/toolkit_affinity) must route to the document tier even
    # when the classifier rated it "strong" — which otherwise goes to
    # high_reasoning (Opus). Synthesis belongs on the document model, not Opus.
    router = ModelRouter(make_settings())
    task = Task(input="put together an investment research PDF on Nvidia earnings")
    event = TaskEvent(
        seq=1,
        type=TaskEventType.log,
        payload={
            "message": "intent_classification_completed",
            "decision": {
                "classification": "task_request",
                "model_tier": "strong",
                "likely_tools": ["search", "financial_data", "document_generation"],
                "toolkit_affinity": ["search", "document_generation"],
            },
        },
    )

    route = router.route_for_task(task, events=[event])

    assert route.tier is ModelRouteTier.document


def test_model_router_routes_document_from_objective_when_tools_hallucinated() -> None:
    # Real-world regression: the classifier rated a report "strong" and emitted
    # hallucinated likely_tools (firecrawl/serpapi/exa — not Kortny tools), so the
    # tool-hint carve-out missed and it fell through to high_reasoning (Opus). The
    # objective text it wrote still clearly describes a document, so we route to
    # the document tier from that.
    router = ModelRouter(make_settings())
    task = Task(input="make a polished PDF report on open-source AI coding agents")
    event = TaskEvent(
        seq=1,
        type=TaskEventType.log,
        payload={
            "message": "intent_classification_completed",
            "decision": {
                "classification": "task_request",
                "model_tier": "strong",
                "likely_tools": ["firecrawl", "serpapi", "exa"],
                "reason": "Produce a polished PDF report on AI coding agents.",
            },
        },
    )

    route = router.route_for_task(task, events=[event])

    assert route.tier is ModelRouteTier.document


def test_model_router_strong_nondocument_intent_still_high_reasoning() -> None:
    # Guard: non-document "strong" tasks keep routing to high_reasoning so the
    # document carve-out above doesn't accidentally demote genuine hard reasoning.
    router = ModelRouter(make_settings())
    task = Task(input="reason carefully about this hard multi-step strategy call")
    event = TaskEvent(
        seq=1,
        type=TaskEventType.log,
        payload={
            "message": "intent_classification_completed",
            "decision": {
                "classification": "task_request",
                "model_tier": "strong",
                "likely_tools": [],
            },
        },
    )

    route = router.route_for_task(task, events=[event])

    assert route.tier is ModelRouteTier.high_reasoning


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
