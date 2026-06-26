"""Cross-app tool-orchestration eval cases.

Each case captures a realistic user request together with the set of toolkits
that are connected for this task. The assertions cover three independent
orchestration behaviors:

- ``expected_apps``: Composio toolkit slugs whose tools MUST be called to
  answer correctly. The agent must actually invoke at least one tool from each
  of these toolkits, not answer from cached/injected context.
- ``must_use_tools``: when True, the answer MUST come from a live tool call,
  not from episodic/observe/KG context that happens to mention the same
  information. This is the context-leak guard — the key regression for cases
  where Kortny answered from stale channel context instead of calling the API.
- ``forbidden_apps``: toolkits that must NOT be called. Used for scope/noise
  isolation (e.g. a finance query must not trigger GitHub/Linear calls).

``CONNECTED_LIVE`` is the realistic connected set for a typical Kortny install;
reuse it as the ``connected_toolkits`` default wherever the specific toolkit
mix does not matter for the assertion.

Expand from anonymized real history over time; this seed is the floor the
cross-app orchestration path is held to.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from kortny.intent.models import IntentSurface

# Realistic connected set for a typical Kortny install covering dev, comms,
# productivity, finance, and search verticals.
CONNECTED_LIVE: tuple[str, ...] = (
    "github",
    "gmail",
    "googlecalendar",
    "linear",
    "notion",
    "confluence",
    "twelve_data",
    "alpha_vantage",
    "alpaca",
    "vercel",
    "supabase",
    "serpapi",
    "firecrawl",
    "exa",
)


@dataclass(frozen=True, slots=True)
class OrchestrationCase:
    request: str
    connected_toolkits: tuple[str, ...]
    surface: IntentSurface
    expected_apps: tuple[str, ...]
    must_use_tools: bool = False
    forbidden_apps: tuple[str, ...] = ()
    note: str = ""
    tags: tuple[str, ...] = field(default=())


_DM = IntentSurface.dm
_APP = IntentSurface.app_mention
_CHAN = IntentSurface.channel_message

SEED_ORCHESTRATION_CASES: tuple[OrchestrationCase, ...] = (
    # 1 — cross-app explicit: GitHub PRs → Linear ticket; clear two-app intent
    OrchestrationCase(
        request="summarize my open GitHub PRs and file a ticket for anything that needs follow-up",
        connected_toolkits=CONNECTED_LIVE,
        surface=_DM,
        expected_apps=("github", "linear"),
        tags=("cross_app", "explicit"),
        note="live PASS case 2026-06-25",
    ),
    # 2 — cross-app implicit with context-leak guard; THE regression target
    # (answered from stale #rag channel context instead of GitHub+Linear)
    OrchestrationCase(
        request="what did I ship this week, and what's still open?",
        connected_toolkits=CONNECTED_LIVE,
        surface=_DM,
        expected_apps=("github", "linear"),
        must_use_tools=True,
        tags=("cross_app", "implicit", "context_leak_guard"),
        note="live FAIL 2026-06-25 — answered from stale #rag channel context instead of GitHub+Linear; THE target case",
    ),
    # 3 — cross-app explicit: email + calendar write
    OrchestrationCase(
        request="find my most recent email from Acme and put a 30-min follow-up on my calendar tomorrow afternoon",
        connected_toolkits=CONNECTED_LIVE,
        surface=_DM,
        expected_apps=("gmail", "googlecalendar"),
        tags=("cross_app", "explicit"),
    ),
    # 4 — single-app status with context-leak guard
    OrchestrationCase(
        request="what are my open issues in Linear?",
        connected_toolkits=CONNECTED_LIVE,
        surface=_DM,
        expected_apps=("linear",),
        must_use_tools=True,
        tags=("single_app", "status"),
    ),
    # 5 — single-app implicit with context-leak guard
    OrchestrationCase(
        request="any important unread emails this morning?",
        connected_toolkits=CONNECTED_LIVE,
        surface=_DM,
        expected_apps=("gmail",),
        must_use_tools=True,
        tags=("single_app", "implicit", "context_leak_guard"),
    ),
    # 6 — single-app calendar read with context-leak guard
    OrchestrationCase(
        request="what's on my calendar tomorrow?",
        connected_toolkits=CONNECTED_LIVE,
        surface=_DM,
        expected_apps=("googlecalendar",),
        must_use_tools=True,
    ),
    # 7 — single-app implicit with context-leak guard
    OrchestrationCase(
        request="has my latest PR been merged yet?",
        connected_toolkits=CONNECTED_LIVE,
        surface=_DM,
        expected_apps=("github",),
        must_use_tools=True,
        tags=("implicit", "context_leak_guard"),
    ),
    # 8 — single knowledge-base pull
    OrchestrationCase(
        request="pull the latest spec from Confluence and give me the gist",
        connected_toolkits=CONNECTED_LIVE,
        surface=_DM,
        expected_apps=("confluence",),
        tags=("knowledge",),
    ),
    # 9 — cross-app doc-to-ticket: Notion → Linear
    OrchestrationCase(
        request="turn the open questions in our Notion doc into Linear issues",
        connected_toolkits=CONNECTED_LIVE,
        surface=_DM,
        expected_apps=("notion", "linear"),
        tags=("cross_app",),
    ),
    # 10 — finance ticker; scope isolation (must NOT drag in dev/work toolkits)
    OrchestrationCase(
        request="what's AAPL trading at right now?",
        connected_toolkits=CONNECTED_LIVE,
        surface=_DM,
        expected_apps=("twelve_data",),
        must_use_tools=True,
        forbidden_apps=("github", "linear", "gmail"),
        tags=("single_app", "finance", "scope"),
        note="ensures finance still routes correctly and no cross-domain noise",
    ),
    # 11 — source-priority guard: Kortny-managed state (schedules) is priority 1,
    # NOT a connected app. An internal-state question must not reach for a
    # connected integration. (Guards the precedence-table fix.)
    OrchestrationCase(
        request="what do I have scheduled right now?",
        connected_toolkits=CONNECTED_LIVE,
        surface=_DM,
        expected_apps=(),
        forbidden_apps=("github", "linear", "gmail", "googlecalendar", "notion"),
        tags=("source_priority", "internal_state", "guard"),
        note="schedule state is internal (Tier-0/schedule tools), never a "
        "connected app",
    ),
    # 12 — over-reach guard: a pure-knowledge question must answer directly and
    # NOT fire a connected integration. (Guards against the precedence rule
    # over-steering cheap models into reaching for tools on stable facts.)
    OrchestrationCase(
        request="explain how OAuth 2.0 authorization code flow works",
        connected_toolkits=CONNECTED_LIVE,
        surface=_DM,
        expected_apps=(),
        forbidden_apps=("github", "linear", "gmail", "googlecalendar", "notion"),
        tags=("source_priority", "knowledge", "guard"),
        note="stable general knowledge → answer directly, no connected tool",
    ),
)
