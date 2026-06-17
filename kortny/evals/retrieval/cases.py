"""Seed dataset for tool-retrieval evaluation (Linear HIG-271).

Each case maps a realistic Slack query to the tool(s) that should be retrieved
to answer it. ``expected_tool_slugs`` is the *acceptable* set: for most cases one
specific tool is correct; for genuinely-equivalent capabilities (web search,
stock price) any one in the set suffices, so ``hit_rate`` is the headline there
while Recall@k/nDCG@k still reward ranking a correct tool highly.

Ground-truth slugs are verified against the live Composio catalog. This is a
SEED — it must grow from real Slack tasks (especially failures) so the eval
tracks what users actually ask. The first two cases are the live regressions
that motivated the Orchestration Spine (tasks c65e7b2f / 02910a27).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class RetrievalCase:
    """One query and the tool(s) a good retriever should surface for it."""

    query: str
    expected_tool_slugs: tuple[str, ...]
    note: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)
    # The connected toolkit(s) the grounded intent (HIG-274 toolkit_affinity)
    # would supply for this query; used to measure the retrieval grounding boost.
    implies_toolkits: tuple[str, ...] = field(default_factory=tuple)


SEED_RETRIEVAL_CASES: tuple[RetrievalCase, ...] = (
    RetrievalCase(
        query="what's on my plate today?",
        expected_tool_slugs=("LINEAR_LIST_LINEAR_ISSUES", "LINEAR_SEARCH_ISSUES"),
        note="Live regression (task c65e7b2f): retriever ranked finance tools, "
        "dropped Linear entirely. 'my plate' must reach the work tracker.",
        tags=("regression", "linear", "persona"),
        implies_toolkits=("linear",),
    ),
    RetrievalCase(
        query="what notes can you see on Notion?",
        expected_tool_slugs=("NOTION_SEARCH_NOTION_PAGE",),
        note="Live regression (task 02910a27): selector picked ID-lookup tools "
        "and missed the search tool, so the agent claimed it could not browse.",
        tags=("regression", "notion", "search-vs-id"),
        implies_toolkits=("notion",),
    ),
    RetrievalCase(
        query="search my Notion for the launch planning doc",
        expected_tool_slugs=("NOTION_SEARCH_NOTION_PAGE",),
        tags=("notion", "search"),
        implies_toolkits=("notion",),
    ),
    RetrievalCase(
        query="pull the rows from my Notion roadmap database",
        expected_tool_slugs=("NOTION_QUERY_DATABASE",),
        note="Disambiguation: database query, not page search.",
        tags=("notion", "database"),
        implies_toolkits=("notion",),
    ),
    RetrievalCase(
        query="list the open Linear issues assigned to me",
        expected_tool_slugs=("LINEAR_LIST_LINEAR_ISSUES", "LINEAR_SEARCH_ISSUES"),
        tags=("linear",),
        implies_toolkits=("linear",),
    ),
    RetrievalCase(
        query="scrape this webpage and give me the text",
        expected_tool_slugs=("FIRECRAWL_SCRAPE",),
        tags=("firecrawl", "scrape"),
        implies_toolkits=("firecrawl",),
    ),
    RetrievalCase(
        query="find recent articles about agent tool retrieval",
        expected_tool_slugs=("EXA_SEARCH", "SERPAPI_SEARCH", "FIRECRAWL_SEARCH"),
        note="Any web-search tool suffices; hit_rate is the headline metric.",
        tags=("web-search", "any-of"),
        implies_toolkits=("exa", "serpapi", "firecrawl"),
    ),
    RetrievalCase(
        query="what's the latest price of AAPL?",
        expected_tool_slugs=(
            "ALPHA_VANTAGE_GLOBAL_QUOTE",
            "TWELVE_DATA_QUOTE",
            "TWELVE_DATA_GET_PRICE",
            "ALPACA_GET_LATEST_QUOTE_FOR_STOCK_SYMBOL",
        ),
        note="Any market-data quote tool suffices; hit_rate is the headline.",
        tags=("market-data", "any-of"),
        implies_toolkits=("alpha_vantage", "twelve_data", "alpaca"),
    ),
    RetrievalCase(
        query="what are my open positions?",
        expected_tool_slugs=("ALPACA_GET_ALL_OPEN_POSITIONS",),
        tags=("alpaca", "positions"),
        implies_toolkits=("alpaca",),
    ),
    RetrievalCase(
        query="search Confluence for the onboarding page",
        expected_tool_slugs=("CONFLUENCE_CQL_SEARCH",),
        tags=("confluence", "search"),
        implies_toolkits=("confluence",),
    ),
    RetrievalCase(
        query="crawl our documentation site",
        expected_tool_slugs=("FIRECRAWL_CRAWL",),
        note="Disambiguation: crawl whole site, not single-page scrape.",
        tags=("firecrawl", "crawl"),
        implies_toolkits=("firecrawl",),
    ),
)
