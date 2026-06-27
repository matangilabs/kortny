"""Top-25 de-facto team integration registry (HIG-294).

A frozen, pure-data catalogue of the 25 integration apps that every team uses.
Referenced by the orchestration eval cases (requires_toolkits field) and drives
the capability profiler's intent-targeting (``tool_intents`` hints).

Tier-1 = ranks 1-15 (dev+tickets, email+calendar+meetings, both CRMs, docs/knowledge).
Tier-2 = ranks 16-25 (specialist apps still in the curated set).

Outlook Calendar actions live inside the ``outlook`` toolkit — there is no
standalone slug; use ``outlook`` everywhere.

``tool_intents`` are semantic workflow-intent labels for each app — a handful of
the most common operations expressed as intent strings (e.g.
``"search_issues"``, ``"create_event"``). These are injected into the capability
profiler's system prompt so enriched_description text is tuned toward the bounded
workflow set the curated stack actually needs, rather than generic descriptions.

``HOLDOUT_APPS`` is a separate tuple of apps used ONLY for generalization-tier
eval holdout cases. These are NOT in ``TOP25`` and NEVER in ``TOP25_SLUGS``.
The tuning loop must never touch holdout cases — they exist solely to measure
how well orchestration transfers to new apps outside the curated set.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Top25App:
    slug: str
    tier: int  # 1 (ranks 1-15) or 2 (ranks 16-25); holdout apps use tier=3
    display_name: str
    workflow_family: str  # one of: comms, dev, calendar, docs, crm, pm, support
    tool_intents: tuple[str, ...] = field(default=())
    """Semantic workflow intent labels for this app.

    Used by the capability profiler to optimize enriched_description text toward
    the most common operations. A handful (3-8) per app is ideal.
    """


TOP25: tuple[Top25App, ...] = (
    # ---- Tier 1 (ranks 1-15) -----------------------------------------------
    Top25App(
        "slack",
        1,
        "Slack",
        "comms",
        ("send_message", "search_messages", "list_channels", "post_to_channel"),
    ),
    Top25App(
        "gmail",
        1,
        "Gmail",
        "comms",
        ("search_emails", "read_email", "send_email", "list_inbox", "reply_to_email"),
    ),
    Top25App(
        "googlecalendar",
        1,
        "Google Calendar",
        "calendar",
        (
            "list_events",
            "create_event",
            "update_event",
            "cancel_event",
            "check_availability",
        ),
    ),
    Top25App(
        "github",
        1,
        "GitHub",
        "dev",
        (
            "list_pull_requests",
            "get_pull_request",
            "list_issues",
            "create_issue",
            "comment_on_issue",
            "list_commits",
            "search_repositories",
        ),
    ),
    Top25App(
        "outlook",
        1,
        "Outlook",
        "comms",  # covers Outlook Calendar too
        (
            "send_email",
            "list_inbox",
            "search_emails",
            "create_calendar_event",
            "list_calendar_events",
        ),
    ),
    Top25App(
        "googledrive",
        1,
        "Google Drive",
        "docs",
        ("search_files", "get_file", "list_files", "create_folder", "share_file"),
    ),
    Top25App(
        "notion",
        1,
        "Notion",
        "docs",
        (
            "search_pages",
            "get_page",
            "create_page",
            "update_page",
            "list_databases",
            "query_database",
        ),
    ),
    Top25App(
        "jira",
        1,
        "Jira",
        "pm",
        (
            "list_issues",
            "get_issue",
            "create_issue",
            "update_issue",
            "add_comment",
            "search_issues",
        ),
    ),
    Top25App(
        "salesforce",
        1,
        "Salesforce",
        "crm",
        (
            "search_records",
            "get_record",
            "create_record",
            "update_record",
            "list_opportunities",
        ),
    ),
    Top25App(
        "hubspot",
        1,
        "HubSpot",
        "crm",
        (
            "search_contacts",
            "get_contact",
            "create_contact",
            "log_activity",
            "list_deals",
        ),
    ),
    Top25App(
        "googledocs",
        1,
        "Google Docs",
        "docs",
        ("get_document", "create_document", "append_text", "search_documents"),
    ),
    Top25App(
        "zoom",
        1,
        "Zoom",
        "calendar",
        ("create_meeting", "list_meetings", "get_meeting", "delete_meeting"),
    ),
    Top25App(
        "microsoft_teams",
        1,
        "MS Teams",
        "comms",
        ("send_message", "list_channels", "search_messages", "post_to_channel"),
    ),
    Top25App(
        "linear",
        1,
        "Linear",
        "pm",
        (
            "list_issues",
            "get_issue",
            "create_issue",
            "update_issue",
            "search_issues",
            "list_projects",
            "add_comment",
        ),
    ),
    Top25App(
        "asana",
        1,
        "Asana",
        "pm",
        (
            "list_tasks",
            "get_task",
            "create_task",
            "update_task",
            "list_projects",
            "search_tasks",
        ),
    ),
    # ---- Tier 2 (ranks 16-25) -----------------------------------------------
    Top25App(
        "confluence",
        2,
        "Confluence",
        "docs",
        ("search_pages", "get_page", "create_page", "update_page", "list_spaces"),
    ),
    Top25App(
        "zendesk",
        2,
        "Zendesk",
        "support",
        (
            "list_tickets",
            "get_ticket",
            "create_ticket",
            "update_ticket",
            "search_tickets",
        ),
    ),
    Top25App(
        "monday",
        2,
        "Monday",
        "pm",
        ("list_items", "get_item", "create_item", "update_item", "list_boards"),
    ),
    Top25App(
        "trello",
        2,
        "Trello",
        "pm",
        ("list_cards", "create_card", "update_card", "list_boards", "move_card"),
    ),
    Top25App(
        "clickup",
        2,
        "ClickUp",
        "pm",
        ("list_tasks", "create_task", "update_task", "get_task", "list_spaces"),
    ),
    Top25App(
        "stripe",
        2,
        "Stripe",
        "crm",
        (
            "list_customers",
            "get_customer",
            "list_charges",
            "get_charge",
            "list_invoices",
        ),
    ),
    Top25App(
        "gitlab",
        2,
        "GitLab",
        "dev",
        (
            "list_merge_requests",
            "get_merge_request",
            "list_issues",
            "create_issue",
            "list_commits",
        ),
    ),
    Top25App(
        "intercom",
        2,
        "Intercom",
        "support",
        (
            "list_conversations",
            "get_conversation",
            "reply_to_conversation",
            "search_contacts",
        ),
    ),
    Top25App(
        "calendly",
        2,
        "Calendly",
        "calendar",
        ("list_event_types", "list_scheduled_events", "get_event", "cancel_event"),
    ),
    Top25App(
        "googlemeet",
        2,
        "Google Meet",
        "calendar",
        ("create_meeting", "get_meeting", "list_meetings"),
    ),
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


# ---------------------------------------------------------------------------
# Holdout apps — generalization tier
# ---------------------------------------------------------------------------
# These apps are NOT in TOP25 and NOT in TOP25_SLUGS.  They exist only for
# eval holdout cases (tuning_split="holdout") that measure whether orchestration
# generalizes to new apps outside the curated set.  The tuning loop MUST NEVER
# train on cases that reference these apps.

HOLDOUT_APPS: tuple[Top25App, ...] = (
    Top25App(
        "dropbox",
        3,
        "Dropbox",
        "docs",
        (
            "upload_file",
            "download_file",
            "create_shared_link",
            "list_folder",
            "move_file",
            "copy_file",
            "search_files",
        ),
    ),
    Top25App(
        "airtable",
        3,
        "Airtable",
        "pm",
        (
            "list_records",
            "get_record",
            "create_record",
            "update_record",
            "search_records",
            "list_bases",
        ),
    ),
    Top25App(
        "pagerduty",
        3,
        "PagerDuty",
        "support",
        (
            "trigger_incident",
            "list_incidents",
            "get_incident",
            "acknowledge_incident",
            "resolve_incident",
            "list_on_call",
        ),
    ),
    Top25App(
        "twilio",
        3,
        "Twilio",
        "comms",
        ("send_sms", "make_call", "list_messages", "get_message"),
    ),
    Top25App(
        "loom",
        3,
        "Loom",
        "comms",
        ("list_videos", "get_video", "share_video"),
    ),
    Top25App(
        "pipedrive",
        3,
        "Pipedrive",
        "crm",
        (
            "list_deals",
            "get_deal",
            "create_deal",
            "update_deal",
            "list_persons",
            "search_persons",
        ),
    ),
    Top25App(
        "shopify",
        3,
        "Shopify",
        "crm",
        (
            "list_orders",
            "get_order",
            "list_products",
            "get_product",
            "update_product",
            "list_customers",
        ),
    ),
)

# Convenience: frozenset of holdout slugs for O(1) membership checks.
HOLDOUT_SLUGS: frozenset[str] = frozenset(app.slug for app in HOLDOUT_APPS)

# Lookup dict for holdout apps by slug (takes last entry per slug).
_HOLDOUT_MAP: dict[str, Top25App] = {app.slug: app for app in HOLDOUT_APPS}


def tool_intents_for(slug: str) -> tuple[str, ...]:
    """Return the tool_intents for a given toolkit slug.

    Checks TOP25 first, then HOLDOUT_APPS. Returns an empty tuple if the slug
    is not found in either registry (e.g. long-tail Composio tools).
    """
    canonical = ALIASES.get(slug, slug)
    for app in TOP25:
        if app.slug == canonical:
            return app.tool_intents
    holdout = _HOLDOUT_MAP.get(canonical)
    if holdout is not None:
        return holdout.tool_intents
    return ()
