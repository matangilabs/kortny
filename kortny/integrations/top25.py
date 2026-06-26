"""Top-25 de-facto team integration registry (HIG-294).

A frozen, pure-data catalogue of the 25 integration apps that every team uses.
Referenced by the orchestration eval cases (requires_toolkits field) and will
later drive the retrieval prior (tier weighting in tool-selection).

Tier-1 = ranks 1-15 (dev+tickets, email+calendar+meetings, both CRMs, docs/knowledge).
Tier-2 = ranks 16-25 (specialist apps still in the curated set).

Outlook Calendar actions live inside the ``outlook`` toolkit — there is no
standalone slug; use ``outlook`` everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Top25App:
    slug: str
    tier: int  # 1 (ranks 1-15) or 2 (ranks 16-25)
    display_name: str
    workflow_family: str  # one of: comms, dev, calendar, docs, crm, pm, support


TOP25: tuple[Top25App, ...] = (
    # ---- Tier 1 (ranks 1-15) -----------------------------------------------
    Top25App("slack", 1, "Slack", "comms"),
    Top25App("gmail", 1, "Gmail", "comms"),
    Top25App("googlecalendar", 1, "Google Calendar", "calendar"),
    Top25App("github", 1, "GitHub", "dev"),
    Top25App("outlook", 1, "Outlook", "comms"),  # covers Outlook Calendar too
    Top25App("googledrive", 1, "Google Drive", "docs"),
    Top25App("notion", 1, "Notion", "docs"),
    Top25App("jira", 1, "Jira", "pm"),
    Top25App("salesforce", 1, "Salesforce", "crm"),
    Top25App("hubspot", 1, "HubSpot", "crm"),
    Top25App("googledocs", 1, "Google Docs", "docs"),
    Top25App("zoom", 1, "Zoom", "calendar"),
    Top25App("microsoft_teams", 1, "MS Teams", "comms"),
    Top25App("linear", 1, "Linear", "pm"),
    Top25App("asana", 1, "Asana", "pm"),
    # ---- Tier 2 (ranks 16-25) -----------------------------------------------
    Top25App("confluence", 2, "Confluence", "docs"),
    Top25App("zendesk", 2, "Zendesk", "support"),
    Top25App("monday", 2, "Monday", "pm"),
    Top25App("trello", 2, "Trello", "pm"),
    Top25App("clickup", 2, "ClickUp", "pm"),
    Top25App("stripe", 2, "Stripe", "crm"),
    Top25App("gitlab", 2, "GitLab", "dev"),
    Top25App("intercom", 2, "Intercom", "support"),
    Top25App("calendly", 2, "Calendly", "calendar"),
    Top25App("googlemeet", 2, "Google Meet", "calendar"),
)

# Convenience: frozenset of all slugs for O(1) membership checks.
TOP25_SLUGS: frozenset[str] = frozenset(app.slug for app in TOP25)

# Alias map: canonical slug → same slug (identity), plus aliases where one
# slug covers multiple products (e.g. outlook calendar lives inside outlook).
ALIASES: dict[str, str] = {
    "outlook_calendar": "outlook",
    "office365": "outlook",
    "ms_calendar": "outlook",
    "teams": "microsoft_teams",
    "google_meet": "googlemeet",
    "gdrive": "googledrive",
    "gdocs": "googledocs",
}

# Tier lookup dict keyed by slug.
_TIER_MAP: dict[str, int] = {app.slug: app.tier for app in TOP25}


def tier_of(slug: str) -> int | None:
    """Return the tier (1 or 2) for a given toolkit slug, or None if not in top-25.

    Resolves aliases transparently.
    """
    canonical = ALIASES.get(slug, slug)
    return _TIER_MAP.get(canonical)
