import json
import uuid
from collections.abc import Sequence

from kortny.llm import ChatMessage, Completion, TokenUsage
from kortny.tool_selection import (
    HeuristicToolSelector,
    LLMToolSelector,
    ToolCard,
    compact_tool_cards,
)
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
    assert result.prompt_chars <= 2200
    assert result.prompt_char_budget == 2200
    assert result.budget_omitted_candidate_names
    assert result.route_reason == "no_external_needed+prompt_budget_trimmed"
    user_content = provider.messages[0][1].content
    assert user_content is not None
    user_payload = json.loads(user_content)
    assert len(user_payload["external_candidates"]) < len(cards)


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
