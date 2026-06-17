import json
import uuid
from collections.abc import Sequence

from kortny.llm import ChatMessage, Completion, TokenUsage
from kortny.tool_selection import (
    HeuristicToolSelector,
    LLMToolSelector,
    ToolCard,
    ToolCatalogService,
    compact_tool_cards,
)
from kortny.tool_selection.selector import MIN_EXTERNAL_PROMPT_CANDIDATES
from kortny.tools import EchoTool
from kortny.tools.types import JsonObject, JsonSchema


def test_heuristic_selector_selects_firecrawl_for_current_research() -> None:
    native_cards = (native_web_search_card(),)
    external_cards = (firecrawl_card(),)

    result = HeuristicToolSelector().select(
        task_id=uuid.uuid4(),
        task_input="find recent AI observability tooling and summarize the top options",
        native_cards=native_cards,
        external_cards=external_cards,
    )

    assert [selection.registry_name for selection in result.selected_tools] == [
        "composio_firecrawl_search"
    ]
    assert result.suppressed_native_tools == ("web_search",)
    assert result.fallback_used is True


def test_heuristic_selector_rejects_irrelevant_external_tool() -> None:
    result = HeuristicToolSelector().select(
        task_id=uuid.uuid4(),
        task_input="remember that I prefer concise Slack replies",
        native_cards=(native_web_search_card(),),
        external_cards=(firecrawl_card(),),
    )

    assert result.selected_tools == ()
    assert result.suppressed_native_tools == ()
    assert result.rejected_tools[0].registry_name == "composio_firecrawl_search"


def test_heuristic_selector_keeps_channel_summary_slack_native() -> None:
    result = HeuristicToolSelector().select(
        task_id=uuid.uuid4(),
        task_input=(
            "summarize the last few decisions in this channel and call out "
            "anything unresolved"
        ),
        native_cards=(native_slack_history_card(),),
        external_cards=(firecrawl_card(),),
    )

    assert result.selected_tools == ()
    assert result.suppressed_native_tools == ()
    assert result.route_reason == "heuristic_no_external_match"


def test_heuristic_selector_does_not_use_web_for_linear_summary() -> None:
    result = HeuristicToolSelector().select(
        task_id=uuid.uuid4(),
        task_input="summarize my open tasks in the Linear kortny project",
        native_cards=(native_slack_history_card(),),
        external_cards=(firecrawl_card(),),
    )

    assert result.selected_tools == ()
    assert result.suppressed_native_tools == ()


def test_llm_selector_filters_to_allowed_tools_and_native_suppressions() -> None:
    provider = FakeSelectorLLM(
        content="""
        {
          "selected_tools": [
            {"registry_name": "composio_firecrawl_search", "confidence": 0.91, "reason": "Needs current web search"},
            {"registry_name": "unknown_external", "confidence": 0.99, "reason": "Not allowed"}
          ],
          "suppressed_native_tools": ["web_search", "unknown_native"],
          "rejected_tools": [],
          "route_reason": "needs_current_web_research"
        }
        """
    )

    result = LLMToolSelector(provider).select(
        task_id=uuid.uuid4(),
        task_input="search recent AI observability tools",
        native_cards=(native_web_search_card(),),
        external_cards=(firecrawl_card(),),
    )

    assert [selection.registry_name for selection in result.selected_tools] == [
        "composio_firecrawl_search"
    ]
    assert result.suppressed_native_tools == ("web_search",)
    assert result.route_reason == "needs_current_web_research"
    assert provider.prompt_names == ["kortny.tool_selector"]
    assert result.prompt_chars is not None
    assert result.prompt_char_budget == 12000


def test_llm_selector_coerces_singleton_boolean_confidence_values() -> None:
    provider = FakeSelectorLLM(
        content="""
        {
          "selected_tools": [],
          "suppressed_native_tools": [],
          "rejected_tools": [
            {"registry_name": "composio_firecrawl_search", "confidence": [true], "reason": "Not needed"}
          ],
          "route_reason": "no_external_needed"
        }
        """
    )

    result = LLMToolSelector(provider).select(
        task_id=uuid.uuid4(),
        task_input="what tools do you have access to?",
        native_cards=(native_web_search_card(),),
        external_cards=(firecrawl_card(),),
    )

    assert result.selected_tools == ()
    assert result.rejected_tools[0].registry_name == "composio_firecrawl_search"
    assert result.rejected_tools[0].confidence == 1.0
    assert result.route_reason == "no_external_needed"


def test_llm_selector_trims_prompt_payload_to_budget() -> None:
    cards = tuple(
        ToolCard(
            registry_name=f"composio_tool_{index}",
            provider="composio",
            display_name=f"Tool {index}",
            description="Search large enterprise systems. " * 40,
            capabilities=("workspace_search", "document_search"),
            side_effect="read",
            toolkit_slug="other",
            tool_slugs=(f"OTHER_TOOL_{index}",),
            required_fields=("query",),
        )
        for index in range(12)
    )
    provider = FakeSelectorLLM(
        content="""
        {
          "selected_tools": [],
          "suppressed_native_tools": [],
          "rejected_tools": [],
          "route_reason": "no_external_needed"
        }
        """
    )

    result = LLMToolSelector(provider, max_prompt_chars=2200).select(
        task_id=uuid.uuid4(),
        task_input="summarize recent channel decisions",
        native_cards=(native_slack_history_card(),),
        external_cards=cards,
    )

    assert result.prompt_chars is not None
    assert result.prompt_char_budget == 2200
    assert result.budget_omitted_candidate_names
    assert result.route_reason == "no_external_needed+prompt_budget_trimmed"
    user_content = provider.messages[0][1].content
    assert user_content is not None
    user_payload = json.loads(user_content)
    assert len(user_payload["external_candidates"]) < len(cards)
    # Trimming never strips externals below the floor, even if the prompt
    # stays over budget (the native section alone can exceed small budgets).
    assert len(user_payload["external_candidates"]) == MIN_EXTERNAL_PROMPT_CANDIDATES


def test_llm_selector_never_trims_externals_to_zero() -> None:
    cards = tuple(
        ToolCard(
            registry_name=f"composio_tool_{index}",
            provider="composio",
            display_name=f"Tool {index}",
            description="Search large enterprise systems. " * 40,
            capabilities=("workspace_search", "document_search"),
            side_effect="read",
            toolkit_slug="other",
            tool_slugs=(f"OTHER_TOOL_{index}",),
            required_fields=("query",),
        )
        for index in range(12)
    )
    provider = FakeSelectorLLM(
        content="""
        {
          "selected_tools": [],
          "suppressed_native_tools": [],
          "rejected_tools": [],
          "route_reason": "no_external_needed"
        }
        """
    )

    # Budget smaller than the system prompt itself: old behavior emptied the
    # candidate list and the selector answered "no external tools available".
    result = LLMToolSelector(provider, max_prompt_chars=1000).select(
        task_id=uuid.uuid4(),
        task_input="summarize recent channel decisions",
        native_cards=(native_slack_history_card(),),
        external_cards=cards,
    )

    user_content = provider.messages[0][1].content
    assert user_content is not None
    user_payload = json.loads(user_content)
    assert len(user_payload["external_candidates"]) == MIN_EXTERNAL_PROMPT_CANDIDATES
    assert result.prompt_chars is not None
    assert result.prompt_chars > 1000


def test_compact_tool_cards_keeps_relevant_candidates_under_budget() -> None:
    cards = tuple(
        ToolCard(
            registry_name=f"composio_other_{index}",
            provider="composio",
            display_name=f"Other {index}",
            description="Generic integration tool.",
            capabilities=("external_tool",),
            side_effect="read",
            toolkit_slug="other",
        )
        for index in range(10)
    ) + (firecrawl_card(),)

    selected, compaction = compact_tool_cards(
        task_input="Use Firecrawl to search recent AI observability tooling",
        cards=cards,
        max_candidates=3,
    )

    assert compaction.compacted is True
    assert compaction.original_candidate_count == 11
    assert compaction.selected_candidate_count == 3
    assert "composio_firecrawl_search" in {card.registry_name for card in selected}
    # Most relevant first: the selector prompt fitter trims from the tail,
    # so compaction output must be relevance-ordered, not catalog-ordered.
    assert selected[0].registry_name == "composio_firecrawl_search"


def test_compact_tool_cards_floor_keeps_connected_named_toolkit() -> None:
    # HIG-274: "my open Linear issues" scores low lexically against a generic
    # Linear card, so relevance trimming would drop it and the selector would
    # return []. The reachability floor protects the intent-named connected
    # toolkit so it survives to the selector.
    cards = tuple(
        ToolCard(
            registry_name=f"composio_other_{index}",
            provider="composio",
            display_name=f"Other {index}",
            description="Generic integration tool.",
            capabilities=("external_tool",),
            side_effect="read",
            toolkit_slug="other",
        )
        for index in range(6)
    ) + (
        ToolCard(
            registry_name="composio_linear_list_issues",
            provider="composio",
            display_name="Linear list issues",
            description="List issues assigned to a user.",
            capabilities=("external_tool",),
            side_effect="read",
            toolkit_slug="linear",
        ),
    )

    unprotected, _ = compact_tool_cards(
        task_input="what's on my plate today",
        cards=cards,
        max_candidates=3,
    )
    assert "composio_linear_list_issues" not in {
        card.registry_name for card in unprotected
    }

    protected, compaction = compact_tool_cards(
        task_input="what's on my plate today",
        cards=cards,
        max_candidates=3,
        protected_toolkits=frozenset({"linear"}),
    )
    assert "composio_linear_list_issues" in {card.registry_name for card in protected}
    assert compaction.reason == "relevance_cap_floor"


def test_llm_selector_forces_tools_for_explicitly_named_toolkit() -> None:
    # Reproduces the production incident: the cheap selector LLM declined the
    # context7 MCP tools ("native web_search can check the docs") even though
    # the user named the server verbatim. Explicit naming must bypass LLM
    # judgment.
    mcp_cards = tuple(
        ToolCard(
            registry_name=f"mcp__context7__{slug}",
            provider="mcp",
            display_name=f"{slug} via context7 (MCP)",
            description="Query library documentation from context7.",
            capabilities=("external_tool",),
            side_effect="read",
            toolkit_slug="context7",
            tool_slugs=(slug,),
        )
        for slug in ("query_docs", "resolve_library_id")
    )
    provider = FakeSelectorLLM(
        content="""
        {
          "selected_tools": [],
          "suppressed_native_tools": [],
          "rejected_tools": [],
          "route_reason": "native web_search can check the docs"
        }
        """
    )

    result = LLMToolSelector(provider).select(
        task_id=uuid.uuid4(),
        task_input="Can you access the context7 mcp server and check Electron docs?",
        native_cards=(native_slack_history_card(),),
        external_cards=mcp_cards,
    )

    assert set(result.selected_names) == {
        "mcp__context7__query_docs",
        "mcp__context7__resolve_library_id",
    }
    assert "explicit_toolkit_forced" in result.route_reason


def test_llm_selector_floors_connected_toolkit_from_grounded_intent() -> None:
    # HIG-274 / task c65e7b2f: "what's on my plate" never names Linear, but the
    # capability-grounded classifier infers toolkit_affinity=["linear"] because
    # Linear is connected. The selector LLM still returned [] and routed native,
    # so the agent fabricated "Linear isn't wired in". The intent-grounded floor
    # must force the connected, intent-named toolkit in.
    linear_cards = tuple(
        ToolCard(
            registry_name=f"composio_linear_{slug}",
            provider="composio",
            display_name=f"Linear {slug}",
            description="Work with Linear issues.",
            capabilities=("external_tool",),
            side_effect="read",
            toolkit_slug="linear",
            tool_slugs=(slug,),
        )
        for slug in ("list_issues",)
    )
    provider = FakeSelectorLLM(
        content="""
        {
          "selected_tools": [],
          "suppressed_native_tools": [],
          "rejected_tools": [],
          "route_reason": "no specific tool mentioned, use native"
        }
        """
    )

    result = LLMToolSelector(provider).select(
        task_id=uuid.uuid4(),
        task_input="what's on my plate today?",
        native_cards=(native_slack_history_card(),),
        external_cards=linear_cards,
        toolkit_affinity=("linear",),
        likely_tools=("linear",),
    )

    assert "composio_linear_list_issues" in set(result.selected_names)
    assert "intent_grounded_floor" in result.route_reason


def test_compact_tool_cards_ranks_mcp_tools_above_generic_catalog_cards() -> None:
    composio_cards = tuple(
        ToolCard(
            registry_name=f"composio_other_{index}",
            provider="composio",
            display_name=f"Other {index}",
            description="Generic integration tool.",
            capabilities=("external_tool",),
            side_effect="read",
            toolkit_slug="other",
        )
        for index in range(30)
    )
    mcp_card = ToolCard(
        registry_name="mcp__context7__query-docs",
        provider="mcp",
        display_name="query-docs via context7 (MCP)",
        description="Query library documentation from the context7 MCP server.",
        capabilities=("external_tool",),
        side_effect="read",
        toolkit_slug="context7",
    )
    # MCP provider registers after Composio, so the MCP card sits last.
    cards = composio_cards + (mcp_card,)

    selected, compaction = compact_tool_cards(
        task_input="can you access the context7 mcp server?",
        cards=cards,
        max_candidates=5,
    )

    assert compaction.compacted is True
    # The admin-registered MCP tool must survive the cap and lead the
    # ranking so later prompt-budget tail-trimming cannot drop it.
    assert selected[0].registry_name == "mcp__context7__query-docs"


def test_native_tool_cards_use_catalog_metadata() -> None:
    cards = ToolCatalogService().native_cards((EchoTool(),))

    assert cards[0].registry_name == "echo"
    assert cards[0].display_name == "Echo"
    assert cards[0].capabilities == ("diagnostic",)
    assert cards[0].side_effect == "read"
    assert cards[0].required_fields == ("message",)


def test_llm_selector_includes_intent_fields_in_payload_when_provided() -> None:
    provider = FakeSelectorLLM(
        content="""
        {
          "selected_tools": [
            {"registry_name": "composio_firecrawl_search", "confidence": 0.88, "reason": "Matches web research intent"}
          ],
          "suppressed_native_tools": ["web_search"],
          "rejected_tools": [],
          "route_reason": "intent_web_research"
        }
        """
    )

    result = LLMToolSelector(provider).select(
        task_id=uuid.uuid4(),
        task_input="find recent AI research papers",
        native_cards=(native_web_search_card(),),
        external_cards=(firecrawl_card(),),
        intent_classification="task_request",
        likely_tools=["web_search", "current_research"],
    )

    assert [selection.registry_name for selection in result.selected_tools] == [
        "composio_firecrawl_search"
    ]
    # Verify intent fields were sent to the LLM
    user_content = provider.messages[0][1].content
    assert user_content is not None
    user_payload = json.loads(user_content)
    assert "intent" in user_payload
    assert user_payload["intent"]["classification"] == "task_request"
    assert "web_search" in user_payload["intent"]["likely_tools"]
    assert "current_research" in user_payload["intent"]["likely_tools"]
    assert "note" in user_payload


def test_llm_selector_omits_intent_fields_when_not_provided() -> None:
    provider = FakeSelectorLLM(
        content="""
        {
          "selected_tools": [],
          "suppressed_native_tools": [],
          "rejected_tools": [],
          "route_reason": "no_external_needed"
        }
        """
    )

    LLMToolSelector(provider).select(
        task_id=uuid.uuid4(),
        task_input="summarize this channel",
        native_cards=(native_web_search_card(),),
        external_cards=(firecrawl_card(),),
    )

    user_content = provider.messages[0][1].content
    assert user_content is not None
    user_payload = json.loads(user_content)
    assert "intent" not in user_payload
    assert "note" in user_payload


def test_heuristic_selector_accepts_intent_params_without_error() -> None:
    result = HeuristicToolSelector().select(
        task_id=uuid.uuid4(),
        task_input="find recent AI observability tooling",
        native_cards=(native_web_search_card(),),
        external_cards=(firecrawl_card(),),
        intent_classification="task_request",
        likely_tools=["web_search"],
    )

    # Heuristic selector ignores intent params but must not raise
    assert (
        result.selected_tools == ()
        or result.selected_tools[0].registry_name == "composio_firecrawl_search"
    )


def native_web_search_card() -> ToolCard:
    return ToolCard(
        registry_name="web_search",
        provider="native",
        display_name="web_search",
        description="Searches the public web.",
        capabilities=("web_search", "current_research"),
        side_effect="read",
    )


def native_slack_history_card() -> ToolCard:
    return ToolCard(
        registry_name="slack_channel_history",
        provider="native",
        display_name="slack_channel_history",
        description="Reads recent Slack channel messages and thread context.",
        capabilities=("slack_context", "channel_summary", "decision_recall"),
        side_effect="read",
    )


def firecrawl_card() -> ToolCard:
    return ToolCard(
        registry_name="composio_firecrawl_search",
        provider="composio",
        display_name="Firecrawl web research",
        description="Search or scrape current web content.",
        capabilities=("web_search", "web_scrape", "current_research"),
        side_effect="read",
        toolkit_slug="firecrawl",
        tool_slugs=("FIRECRAWL_SEARCH",),
        required_fields=("q",),
        visibility_scope_type="user",
        visibility_scope_id="U123",
        can_replace_native_tools=("web_search",),
    )


class FakeSelectorLLM:
    def __init__(self, *, content: str) -> None:
        self.content = content
        self.prompt_names: list[str | None] = []
        self.messages: list[tuple[ChatMessage, ...]] = []

    def complete(
        self,
        *,
        task_id: uuid.UUID,
        messages: Sequence[ChatMessage],
        tools: Sequence[JsonSchema] = (),
        response_format: JsonObject | None = None,
        prompt_name: str | None = None,
        prompt_source: str = "code",
    ) -> Completion:
        del task_id, tools, response_format, prompt_source
        self.prompt_names.append(prompt_name)
        self.messages.append(tuple(messages))
        return Completion(
            content=self.content,
            tool_calls=(),
            usage=TokenUsage(input_tokens=10, output_tokens=10),
            model="test-selector",
        )


def test_compact_tool_cards_semantic_paraphrase_retrieves_linear_card() -> None:
    # Acceptance (HIG-219): "check our issue tracker" must surface the Linear
    # card even though the input never says "linear" — pure word overlap fails
    # here, seeded fake embeddings succeed.
    from kortny.tool_selection import tool_card_embedding_text
    from tests.fake_embeddings import FakeEmbeddingBackend

    linear = ToolCard(
        registry_name="composio_linear_search_issues",
        provider="composio",
        display_name="Linear issue search",
        description="Find and list Linear issues, tickets, and bugs.",
        capabilities=("issue_tracking", "external_tool"),
        side_effect="read",
        toolkit_slug="linear",
    )
    unrelated = tuple(
        ToolCard(
            registry_name=f"composio_other_{index}",
            provider="composio",
            display_name=f"Other {index}",
            description="Generic integration tool.",
            capabilities=("external_tool",),
            side_effect="read",
            toolkit_slug="other",
        )
        for index in range(10)
    )
    cards = unrelated + (linear,)

    backend = FakeEmbeddingBackend()
    query = "can you check our issue tracker for anything urgent?"
    query_vector = backend.embed_query(query)
    semantic_scores = {
        card.registry_name: sum(
            a * b
            for a, b in zip(
                query_vector,
                backend.embed_passages([tool_card_embedding_text(card)])[0],
                strict=True,
            )
        )
        for card in cards
    }

    selected, compaction = compact_tool_cards(
        task_input=query,
        cards=cards,
        max_candidates=5,
        semantic_scores=semantic_scores,
    )

    assert compaction.compacted is True
    assert selected[0].registry_name == "composio_linear_search_issues"


def test_compact_tool_cards_hybrid_blends_semantic_and_lexical() -> None:
    semantic_favorite = ToolCard(
        registry_name="composio_semantic_pick",
        provider="composio",
        display_name="Semantic pick",
        description="Generic integration tool.",
        capabilities=("external_tool",),
        side_effect="read",
        toolkit_slug="zzz",
    )
    lexical_favorite = ToolCard(
        registry_name="composio_acme_tool",
        provider="composio",
        display_name="Acme tool",
        description="Generic integration tool.",
        capabilities=("external_tool",),
        side_effect="read",
        toolkit_slug="acme",
    )
    cards = (semantic_favorite, lexical_favorite)
    task_input = "use acme for this request"

    # Without semantic scores, the lexical toolkit match wins.
    lexical_only, _ = compact_tool_cards(
        task_input=task_input,
        cards=cards,
        max_candidates=1,
    )
    assert lexical_only[0].registry_name == "composio_acme_tool"

    # A strong semantic score (0.6 weight) overrides the lexical favorite
    # (0.4 weight): 0.6*0.9 + 0.4*0.05 > 0.6*0.0 + 0.4*0.54.
    hybrid, _ = compact_tool_cards(
        task_input=task_input,
        cards=cards,
        max_candidates=1,
        semantic_scores={"composio_semantic_pick": 0.9},
    )
    assert hybrid[0].registry_name == "composio_semantic_pick"


def test_compact_tool_cards_none_semantic_scores_matches_legacy_exactly() -> None:
    cards = tuple(
        ToolCard(
            registry_name=f"composio_other_{index}",
            provider="composio",
            display_name=f"Other {index}",
            description="Generic integration tool.",
            capabilities=("external_tool",),
            side_effect="read",
            toolkit_slug="other",
        )
        for index in range(10)
    ) + (firecrawl_card(),)

    legacy_selected, legacy_compaction = compact_tool_cards(
        task_input="Use Firecrawl to search recent AI observability tooling",
        cards=cards,
        max_candidates=3,
    )
    explicit_selected, explicit_compaction = compact_tool_cards(
        task_input="Use Firecrawl to search recent AI observability tooling",
        cards=cards,
        max_candidates=3,
        semantic_scores=None,
    )

    assert explicit_selected == legacy_selected
    assert explicit_compaction == legacy_compaction

    # Within budget + no semantic scores: cards pass through untouched in
    # registration order, exactly as before HIG-219.
    within, within_compaction = compact_tool_cards(
        task_input="anything",
        cards=cards[:2],
        max_candidates=10,
        semantic_scores=None,
    )
    assert within == cards[:2]
    assert within_compaction.reason == "within_budget"


def test_compact_tool_cards_semantic_within_budget_reorders_by_relevance() -> None:
    cards = (
        ToolCard(
            registry_name="composio_low",
            provider="composio",
            display_name="Low",
            description="Generic integration tool.",
            capabilities=("external_tool",),
            side_effect="read",
            toolkit_slug="other",
        ),
        ToolCard(
            registry_name="composio_high",
            provider="composio",
            display_name="High",
            description="Generic integration tool.",
            capabilities=("external_tool",),
            side_effect="read",
            toolkit_slug="other",
        ),
    )

    selected, compaction = compact_tool_cards(
        task_input="anything",
        cards=cards,
        max_candidates=10,
        semantic_scores={"composio_high": 0.9, "composio_low": 0.1},
    )

    assert [card.registry_name for card in selected] == [
        "composio_high",
        "composio_low",
    ]
    assert compaction.omitted_candidate_count == 0
    assert compaction.reason == "within_budget"
