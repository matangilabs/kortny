import uuid

from kortny.llm import ChatMessage, Completion, TokenUsage
from kortny.tool_selection import (
    HeuristicToolSelector,
    LLMToolSelector,
    ToolCard,
)


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
        "composio_execute"
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
    assert result.rejected_tools[0].registry_name == "composio_execute"


def test_llm_selector_filters_to_allowed_tools_and_native_suppressions() -> None:
    provider = FakeSelectorLLM(
        content="""
        {
          "selected_tools": [
            {"registry_name": "composio_execute", "confidence": 0.91, "reason": "Needs current web search"},
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
        "composio_execute"
    ]
    assert result.suppressed_native_tools == ("web_search",)
    assert result.route_reason == "needs_current_web_research"
    assert provider.prompt_names == ["kortny.tool_selector"]


def native_web_search_card() -> ToolCard:
    return ToolCard(
        registry_name="web_search",
        provider="native",
        display_name="web_search",
        description="Searches the public web.",
        capabilities=("web_search", "current_research"),
        side_effect="read",
    )


def firecrawl_card() -> ToolCard:
    return ToolCard(
        registry_name="composio_execute",
        provider="composio",
        display_name="Firecrawl web research",
        description="Search or scrape current web content.",
        capabilities=("web_search", "web_scrape", "current_research"),
        side_effect="read",
        toolkit_slug="firecrawl",
        tool_slugs=("FIRECRAWL_SEARCH", "FIRECRAWL_SCRAPE"),
        visibility_scope_type="user",
        visibility_scope_id="U123",
        can_replace_native_tools=("web_search",),
    )


class FakeSelectorLLM:
    def __init__(self, *, content: str) -> None:
        self.content = content
        self.prompt_names: list[str | None] = []

    def complete(
        self,
        *,
        task_id: uuid.UUID,
        messages: tuple[ChatMessage, ...],
        tools: tuple[dict, ...] = (),
        response_format: dict | None = None,
        prompt_name: str | None = None,
        prompt_source: str = "code",
    ) -> Completion:
        del task_id, messages, tools, response_format, prompt_source
        self.prompt_names.append(prompt_name)
        return Completion(
            content=self.content,
            tool_calls=(),
            usage=TokenUsage(input_tokens=10, output_tokens=10),
            model="test-selector",
        )
