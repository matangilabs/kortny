"""Cross-app tool-orchestration eval cases.

Each case captures a realistic user request together with the set of toolkits
that are connected for this task. The assertions cover orchestration behaviors:

Core assertions (always evaluated):

- ``expected_apps``: Composio toolkit slugs whose tools MUST be called to
  answer correctly. The agent must actually invoke at least one tool from each
  of these toolkits, not answer from cached/injected context.
- ``must_use_tools``: when True, the answer MUST come from a live tool call,
  not from episodic/observe/KG context that happens to mention the same
  information. This is the context-leak guard.
- ``forbidden_apps``: toolkits that must NOT be called. Used for scope/noise
  isolation (e.g. a finance query must not trigger GitHub/Linear calls).

Granularity assertions (only evaluated when declared; default to no assertion):

- ``expected_tool_slugs``: individual tool-level slugs (e.g.
  ``"GITHUB_LIST_PULL_REQUESTS"``) that must appear in the executed tool calls.
  Checked against ``RunResult.called_tool_slugs``.
- ``required_arg_keys``: argument key names (e.g. ``"owner"``, ``"repo"``)
  that must appear in the union of all tool call argument dictionaries.
  Checked against ``RunResult.called_arg_keys``.
- ``approval_expected``: when set (True/False), the task's approval-pause
  behavior must match exactly. Checked against ``RunResult.approval_paused``.
- ``max_turns``: upper bound on the number of LLM turns (``RunResult.turn_count``).
- ``max_cost_usd``: upper bound on total task cost in USD (``RunResult.cost_usd``).

Tuning split axis (ORTHOGONAL to ``Top25App.tier``):

- ``tuning_split``: ``"train"`` (default) or ``"holdout"``. Train cases may be
  tuned against; holdout cases use apps NOT in the curated Top-25 and are NEVER
  tuned against — they measure generalization. The scorer computes
  ``generalization_delta = train_score - holdout_score``; a delta > 0.12 flags
  potential overfitting.
  HARD RULE: the runner's ``split`` filter prevents holdout cases from being
  included in a train-only run and vice versa.

``CONNECTED_LIVE`` is the realistic connected set for a typical Kortny install;
reuse it as the ``connected_toolkits`` default wherever the specific toolkit
mix does not matter for the assertion.

``smoke=True`` marks the subset of cases that run offline via ``make eval-smoke``
(replay mode, no live agent, no API keys). Add ``smoke=True`` to any case that
has a committed fixture in ``fixtures/smoke_goldens.json``.

Expand from anonymized real history over time; this seed is the floor the
cross-app orchestration path is held to.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

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
    requires_toolkits: tuple[str, ...] = ()
    smoke: bool = False
    """Mark True for cases included in the offline smoke subset.

    Smoke cases must have a committed fixture in ``fixtures/smoke_goldens.json``
    so ``make eval-smoke`` can score them without a live agent or API keys.
    """

    # --- S2 tuning-split axis ------------------------------------------------
    tuning_split: Literal["train", "holdout"] = "train"
    """Whether this case is tuned against (``"train"``) or held out for
    generalization measurement (``"holdout"``).

    Train cases may be improved by prompt tuning. Holdout cases MUST NOT be
    tuned against — they use apps outside the curated Top-25 so the
    generalization delta measures how well orchestration transfers to new apps.
    """

    # --- S2 granularity assertions (all default to no assertion) -------------
    expected_tool_slugs: tuple[str, ...] = ()
    """Individual tool-level slugs that must appear in called_tool_slugs.

    Empty (default) means no tool-slug assertion is made for this case.
    """
    required_arg_keys: tuple[str, ...] = ()
    """Argument key names that must appear in the union of all tool call args.

    Empty (default) means no argument-key assertion is made for this case.
    """
    approval_expected: bool | None = None
    """When set, the task's approval-pause behavior must match this value.

    None (default) means no approval assertion is made for this case.
    """
    max_turns: int | None = None
    """Upper bound on coordinator LLM turns. None means no turn-budget check."""
    max_cost_usd: float | None = None
    """Upper bound on total task cost in USD. None means no cost-budget check."""


_DM = IntentSurface.dm
_APP = IntentSurface.app_mention
_CHAN = IntentSurface.channel_message

SEED_ORCHESTRATION_CASES: tuple[OrchestrationCase, ...] = (
    # 1 — cross-app explicit: GitHub PRs + Linear ticket; clear two-app intent
    OrchestrationCase(
        request="summarize my open GitHub PRs and file a ticket for anything that needs follow-up",
        connected_toolkits=CONNECTED_LIVE,
        surface=_DM,
        expected_apps=("github", "linear"),
        tags=("cross_app", "explicit"),
        note="cross-app explicit: both apps named in the request; expects both to be called",
        smoke=True,
    ),
    # 2 — cross-app implicit with context-leak guard; THE regression target
    # (agent must call live APIs, not answer from stale cached context)
    OrchestrationCase(
        request="what did I ship this week, and what's still open?",
        connected_toolkits=CONNECTED_LIVE,
        surface=_DM,
        expected_apps=("github", "linear"),
        must_use_tools=True,
        tags=("cross_app", "implicit", "context_leak_guard"),
        note="context-leak regression: agent must call live GitHub+Linear, not answer from cached context",
        smoke=True,
    ),
    # 3 — cross-app explicit: email + calendar write
    OrchestrationCase(
        request="find my most recent email from Acme and put a 30-min follow-up on my calendar tomorrow afternoon",
        connected_toolkits=CONNECTED_LIVE,
        surface=_DM,
        expected_apps=("gmail", "googlecalendar"),
        tags=("cross_app", "explicit"),
        note="cross-app write: read email then write calendar event",
        smoke=True,
    ),
    # 4 — single-app status with context-leak guard
    OrchestrationCase(
        request="what are my open issues in Linear?",
        connected_toolkits=CONNECTED_LIVE,
        surface=_DM,
        expected_apps=("linear",),
        must_use_tools=True,
        tags=("single_app", "status"),
        note="single-app read with must_use_tools: must call Linear API not answer from memory",
        smoke=True,
    ),
    # 5 — single-app implicit with context-leak guard
    OrchestrationCase(
        request="any important unread emails this morning?",
        connected_toolkits=CONNECTED_LIVE,
        surface=_DM,
        expected_apps=("gmail",),
        must_use_tools=True,
        tags=("single_app", "implicit", "context_leak_guard"),
        note="possessive-routing: must call Gmail not answer from context",
        smoke=True,
    ),
    # 6 — single-app calendar read with context-leak guard
    OrchestrationCase(
        request="what's on my calendar tomorrow?",
        connected_toolkits=CONNECTED_LIVE,
        surface=_DM,
        expected_apps=("googlecalendar",),
        must_use_tools=True,
        note="possessive-routing: 'my calendar' must call Google Calendar API",
        smoke=True,
    ),
    # 7 — single-app implicit with context-leak guard
    OrchestrationCase(
        request="has my latest PR been merged yet?",
        connected_toolkits=CONNECTED_LIVE,
        surface=_DM,
        expected_apps=("github",),
        must_use_tools=True,
        tags=("implicit", "context_leak_guard"),
        note="possessive-routing: 'my PR' must call GitHub not answer from memory",
        smoke=True,
    ),
    # 8 — single knowledge-base pull
    OrchestrationCase(
        request="pull the latest spec from Confluence and give me the gist",
        connected_toolkits=CONNECTED_LIVE,
        surface=_DM,
        expected_apps=("confluence",),
        tags=("knowledge",),
        note="single-app knowledge retrieval from Confluence",
    ),
    # 9 — cross-app doc-to-ticket: Notion + Linear
    OrchestrationCase(
        request="turn the open questions in our Notion doc into Linear issues",
        connected_toolkits=CONNECTED_LIVE,
        surface=_DM,
        expected_apps=("notion", "linear"),
        tags=("cross_app",),
        note="cross-app write: read Notion then create Linear issues",
        smoke=True,
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
        note="finance scope isolation: must call twelve_data, must not call dev/work toolkits",
        smoke=True,
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
        note="schedule state is internal (Tier-0/schedule tools), must not call connected integrations",
        smoke=True,
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
        note="stable general knowledge: answer directly, no connected tool should be called",
        smoke=True,
    ),
    # 13 — cross-app: GitHub PRs + Linear issue (flaky CI angle)
    OrchestrationCase(
        request="summarize my open GitHub PRs and open a Linear issue for the one with flaky CI",
        connected_toolkits=CONNECTED_LIVE,
        surface=_DM,
        expected_apps=("github", "linear"),
        tags=("cross_app", "dev_tickets", "top25"),
        note="dev-to-tickets workflow variation: read GitHub then create Linear issue",
    ),
    # 14 — cross-app: Confluence + Notion knowledge transfer
    OrchestrationCase(
        request="pull the onboarding doc from Confluence and draft a summary page in Notion",
        connected_toolkits=CONNECTED_LIVE,
        surface=_DM,
        expected_apps=("confluence", "notion"),
        tags=("cross_app", "knowledge_deliverable", "top25"),
        note="knowledge-to-deliverable: read Confluence then write Notion",
    ),
    # 15 — cross-app: Gmail + Google Calendar (email thread + calendar write)
    OrchestrationCase(
        request="find the latest email from the project sync thread and add a 30-min review to my calendar Thursday",
        connected_toolkits=CONNECTED_LIVE,
        surface=_DM,
        expected_apps=("gmail", "googlecalendar"),
        tags=("cross_app", "email_calendar", "top25"),
        note="email-to-calendar workflow: read Gmail then write Google Calendar",
    ),
    # 16 — cross-app: Notion + Linear (open questions to tickets)
    OrchestrationCase(
        request="look at our product spec in Notion and file Linear issues for any open questions",
        connected_toolkits=CONNECTED_LIVE,
        surface=_DM,
        expected_apps=("notion", "linear"),
        tags=("cross_app", "tracker_comms", "top25"),
        note="Notion-to-Linear workflow variation; both apps connected",
    ),
    # 17 — SKIP (jira not connected): dev tickets via Jira
    OrchestrationCase(
        request="create a Jira ticket for the authentication bug I just found in the login flow",
        connected_toolkits=CONNECTED_LIVE,
        surface=_DM,
        expected_apps=("jira",),
        requires_toolkits=("jira",),
        tags=("single_app", "dev_tickets", "top25", "unconnected"),
        note="tier-1 top-25; skip until jira connected in eval workspace",
    ),
    # 18 — SKIP (hubspot not connected): meeting CRM logging
    OrchestrationCase(
        request="log this call outcome in HubSpot: we agreed on a pilot with the client",
        connected_toolkits=CONNECTED_LIVE,
        surface=_DM,
        expected_apps=("hubspot",),
        requires_toolkits=("hubspot",),
        tags=("single_app", "meeting_crm", "top25", "unconnected"),
        note="tier-1 top-25; skip until hubspot connected in eval workspace",
    ),
    # 19 — SKIP (zoom not connected): schedule a Zoom meeting
    OrchestrationCase(
        request="schedule a Zoom for the launch sync — invite the product team for next Friday at 2 PM",
        connected_toolkits=CONNECTED_LIVE,
        surface=_DM,
        expected_apps=("zoom",),
        requires_toolkits=("zoom",),
        tags=("single_app", "calendar", "top25", "unconnected"),
        note="tier-1 top-25; skip until zoom connected in eval workspace",
    ),
    # --- Train cases upgraded with S2 granularity assertions -----------------
    # 20 — train, slug-level: GitHub list-issues; must call the right tool slug
    # and pass the expected argument key names.
    OrchestrationCase(
        request="list my open GitHub issues assigned to me",
        connected_toolkits=CONNECTED_LIVE,
        surface=_DM,
        expected_apps=("github",),
        must_use_tools=True,
        expected_tool_slugs=("GITHUB_ISSUES_LIST",),
        required_arg_keys=("assignee",),
        tags=("single_app", "dev", "slug_assertion", "arg_assertion"),
        note="train slug+arg assertion: must call GITHUB_ISSUES_LIST with assignee key",
        smoke=True,
    ),
    # 21 — train, slug-level + approval: Linear create issue requires approval
    OrchestrationCase(
        request="create a Linear issue titled 'Fix login timeout bug' in the Backend project",
        connected_toolkits=CONNECTED_LIVE,
        surface=_DM,
        expected_apps=("linear",),
        must_use_tools=True,
        expected_tool_slugs=("LINEAR_CREATE_LINEAR_ISSUE",),
        required_arg_keys=("title",),
        approval_expected=True,
        tags=("single_app", "pm", "slug_assertion", "arg_assertion", "approval"),
        note="train slug+arg+approval assertion: create Linear issue must pause for approval",
        smoke=True,
    ),
    # 22 — train, budget assertion: single-app read should be cheap and fast
    OrchestrationCase(
        request="show me the last 5 commits on the main branch of my primary repo",
        connected_toolkits=CONNECTED_LIVE,
        surface=_DM,
        expected_apps=("github",),
        must_use_tools=True,
        expected_tool_slugs=("GITHUB_LIST_COMMITS",),
        max_turns=4,
        max_cost_usd=0.05,
        tags=("single_app", "dev", "slug_assertion", "budget"),
        note="train budget assertion: simple GitHub read should not exceed 4 turns or $0.05",
        smoke=True,
    ),
    # 23 — train, slug+arg: Gmail search with subject filter
    OrchestrationCase(
        request="find emails from the finance team with 'invoice' in the subject",
        connected_toolkits=CONNECTED_LIVE,
        surface=_DM,
        expected_apps=("gmail",),
        must_use_tools=True,
        required_arg_keys=("query",),
        tags=("single_app", "comms", "arg_assertion"),
        note="train arg assertion: Gmail search must pass a query key with filter params",
        smoke=True,
    ),
    # --- Holdout cases (generalization tier) ---------------------------------
    # These use apps from HOLDOUT_APPS (NOT in the curated Top-25).
    # tuning_split="holdout" — NEVER tune prompts against these cases.
    # 24 — holdout: Dropbox shared link creation
    OrchestrationCase(
        request="create a shared link for the Q3 report in Dropbox so anyone with the link can view it",
        connected_toolkits=("dropbox",),
        surface=_DM,
        expected_apps=("dropbox",),
        must_use_tools=True,
        tuning_split="holdout",
        tags=("single_app", "docs", "holdout"),
        note="holdout: Dropbox shared-link creation — generalization to storage apps not in Top-25",
        smoke=True,
    ),
    # 25 — holdout: PagerDuty on-call alert
    OrchestrationCase(
        request="page the on-call engineer — there's a critical outage in the payments service",
        connected_toolkits=("pagerduty",),
        surface=_DM,
        expected_apps=("pagerduty",),
        must_use_tools=True,
        tuning_split="holdout",
        tags=("single_app", "support", "holdout"),
        note="holdout: PagerDuty incident trigger — generalization to ops tooling",
        smoke=True,
    ),
    # 26 — holdout: Airtable record lookup
    OrchestrationCase(
        request="look up the campaign records in Airtable where status is 'active'",
        connected_toolkits=("airtable",),
        surface=_DM,
        expected_apps=("airtable",),
        must_use_tools=True,
        tuning_split="holdout",
        tags=("single_app", "pm", "holdout"),
        note="holdout: Airtable record filter — generalization to low-code DB apps",
        smoke=True,
    ),
    # 27 — holdout: Twilio SMS notification (write; expects approval gate)
    OrchestrationCase(
        request="send an SMS to the on-call number saying the deploy completed successfully",
        connected_toolkits=("twilio",),
        surface=_DM,
        expected_apps=("twilio",),
        must_use_tools=True,
        approval_expected=True,
        tuning_split="holdout",
        tags=("single_app", "comms", "holdout", "approval"),
        note="holdout: Twilio SMS send — generalization to messaging APIs; write should gate",
        smoke=True,
    ),
)
