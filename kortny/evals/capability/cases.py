"""Labeled capability-grounding eval cases (HIG-274).

Each case captures a realistic user request paired with the set of toolkits
that are connected for this task. The assertions cover three independent
grounding behaviors:

- ``expected_in_likely_tools``: toolkits the classifier MUST surface in
  ``likely_tools`` or ``toolkit_affinity`` when those toolkits are connected.
- ``expected_in_needs_connection``: capability categories that MUST appear in
  ``needs_connection`` when the user asks for them but they are NOT connected.
- ``expected_floor_toolkits``: toolkits the reachability floor MUST load at
  least one tool from (when connected).
- ``forbidden_scope_toolkits``: toolkits that must NOT be loaded — used to
  test scope isolation (e.g. personal-scoped connectors must not leak into
  channel-scoped tasks).

Expand from anonymized real history over time; this seed is the floor the
capability-grounding path is held to.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from kortny.intent.models import IntentSurface


@dataclass(frozen=True, slots=True)
class CapabilityCase:
    request: str
    connected_toolkits: tuple[str, ...]
    surface: IntentSurface
    expected_in_likely_tools: tuple[str, ...] = ()
    expected_in_needs_connection: tuple[str, ...] = ()
    expected_floor_toolkits: tuple[str, ...] = ()
    forbidden_scope_toolkits: tuple[str, ...] = ()
    note: str = ""
    tags: tuple[str, ...] = field(default=())


_DM = IntentSurface.dm
_APP = IntentSurface.app_mention
_CHAN = IntentSurface.channel_message

SEED_CAPABILITY_CASES: tuple[CapabilityCase, ...] = (
    # 1 — single work-tracker connected; basic grounding pass
    CapabilityCase(
        request="summarize my open Linear tasks",
        connected_toolkits=("linear",),
        surface=_DM,
        expected_in_likely_tools=("linear",),
        expected_floor_toolkits=("linear",),
        note="Linear connected; classifier must surface it and floor must load tools",
        tags=("grounding", "work_tracker"),
    ),
    # 2 — two work trackers connected; both must surface
    CapabilityCase(
        request="what's on my plate today?",
        connected_toolkits=("linear", "notion"),
        surface=_DM,
        expected_in_likely_tools=("linear", "notion"),
        expected_floor_toolkits=("linear", "notion"),
        note="Multi-toolkit grounding; both connected trackers must reach the floor",
        tags=("grounding", "multi_toolkit"),
    ),
    # 3 — Notion-specific query
    CapabilityCase(
        request="find action items in Notion",
        connected_toolkits=("notion",),
        surface=_DM,
        expected_in_likely_tools=("notion",),
        expected_floor_toolkits=("notion",),
        note="Explicit Notion reference must map to connected Notion toolkit",
        tags=("grounding", "explicit_toolkit"),
    ),
    # 4 — calendar not connected; must appear in needs_connection
    CapabilityCase(
        request="check my calendar",
        connected_toolkits=("linear",),
        surface=_DM,
        expected_in_needs_connection=("calendar",),
        note="Calendar asked for but not connected; must surface in needs_connection",
        tags=("needs_connection", "calendar"),
    ),
    # 5 — assignment query with a single work tracker
    CapabilityCase(
        request="what issues are assigned to me?",
        connected_toolkits=("linear",),
        surface=_DM,
        expected_in_likely_tools=("linear",),
        expected_floor_toolkits=("linear",),
        note="Issue-assignment intent maps to connected Linear",
        tags=("grounding", "work_tracker"),
    ),
    # 6 — GitHub PR creation; GitHub connected
    CapabilityCase(
        request="create a PR on our feature branch",
        connected_toolkits=("github",),
        surface=_APP,
        expected_in_likely_tools=("github",),
        expected_floor_toolkits=("github",),
        note="Explicit GitHub action with GitHub connected",
        tags=("grounding", "vcs"),
    ),
    # 7 — multi-toolkit request; both connected
    CapabilityCase(
        request="check my calendar and open Linear issues",
        connected_toolkits=("linear", "googlecalendar"),
        surface=_DM,
        expected_in_likely_tools=("linear", "googlecalendar"),
        expected_floor_toolkits=("linear", "googlecalendar"),
        note="Hybrid request; both toolkits connected and must be grounded",
        tags=("grounding", "multi_toolkit"),
    ),
    # 8 — Slack message send; native Slack tool exists; no external connection required
    CapabilityCase(
        request="send a Slack message to the team",
        connected_toolkits=(),
        surface=_DM,
        expected_in_likely_tools=("slack",),
        note=(
            "Native Slack tool covers this; classifier should surface 'slack' "
            "even with no external toolkits connected"
        ),
        tags=("native_tool", "slack"),
    ),
    # 9 — explicit Notion search
    CapabilityCase(
        request="search Notion for the onboarding docs",
        connected_toolkits=("notion",),
        surface=_DM,
        expected_in_likely_tools=("notion",),
        expected_floor_toolkits=("notion",),
        note="Notion search maps to connected toolkit; floor must load Notion tools",
        tags=("grounding", "search"),
    ),
    # 10 — Jira instead of Linear; different work tracker
    CapabilityCase(
        request="what's on my plate?",
        connected_toolkits=("jira",),
        surface=_DM,
        expected_in_likely_tools=("jira",),
        expected_floor_toolkits=("jira",),
        note="'My plate' resolves to connected Jira, not Linear (not connected)",
        tags=("grounding", "work_tracker", "jira"),
    ),
    # 11 — calendar IS connected; needs_connection must be empty
    CapabilityCase(
        request="what's on my schedule today?",
        connected_toolkits=("googlecalendar",),
        surface=_DM,
        expected_in_likely_tools=("googlecalendar",),
        expected_floor_toolkits=("googlecalendar",),
        expected_in_needs_connection=(),
        note="Calendar IS connected; needs_connection must be empty for this request",
        tags=("grounding", "calendar", "connected"),
    ),
    # 12 — scope isolation: personal Linear connector must not load for channel task
    CapabilityCase(
        request="search GitHub for open PRs",
        connected_toolkits=("github",),
        surface=_CHAN,
        expected_in_likely_tools=("github",),
        expected_floor_toolkits=("github",),
        forbidden_scope_toolkits=("linear_personal",),
        note=(
            "Channel-scoped task; personal-scope Linear connector must not leak "
            "into this channel task"
        ),
        tags=("scope_isolation", "vcs"),
    ),
    # 13 — connected-but-health-unknown is treated as connected (no health gate)
    CapabilityCase(
        request="what's stale in Notion?",
        connected_toolkits=("notion",),
        surface=_DM,
        expected_in_likely_tools=("notion",),
        expected_floor_toolkits=("notion",),
        note=(
            "Notion connected but health could be stale; reachability floor must "
            "still load tools — no health gate blocks the floor"
        ),
        tags=("grounding", "health_agnostic"),
    ),
    # 14 — email not connected; must appear in needs_connection
    CapabilityCase(
        request="summarize my emails",
        connected_toolkits=(),
        surface=_DM,
        expected_in_needs_connection=("email",),
        note="Email asked for but no email toolkit connected; must surface in needs_connection",
        tags=("needs_connection", "email"),
    ),
    # 15 — Asana; different work tracker for focus queries
    CapabilityCase(
        request="what should I focus on today?",
        connected_toolkits=("asana",),
        surface=_DM,
        expected_in_likely_tools=("asana",),
        expected_floor_toolkits=("asana",),
        note="'Focus today' resolves to connected Asana work tracker",
        tags=("grounding", "work_tracker", "asana"),
    ),
)
