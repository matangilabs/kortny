"""Registry of the 3 launch triggers with pure deterministic scorers.

No LLM calls, no I/O. Each scorer receives the trigger event ``data`` dict
(the ``data`` field from the Composio V3 webhook envelope) and returns a
``TriggerDecision`` with an importance score and a routing decision.

Launch trigger slugs are the UPPERCASE Composio trigger names (matching the
pattern used by Composio's REST catalog, e.g. GITHUB_PULL_REQUEST_EVENT).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class TriggerDecision:
    """Result of a deterministic trigger scorer."""

    importance: float
    decision: Literal["ask", "silent", "digest"]
    reason: str


@dataclass(frozen=True)
class LaunchTrigger:
    """Descriptor for one of the 3 launch triggers."""

    slug: str
    toolkit_slug: str
    display_name: str
    scorer: Callable[[dict[str, Any]], TriggerDecision]


# ---------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------


def _score_github_review_request(data: dict[str, Any]) -> TriggerDecision:
    """GitHub: direct review request on a non-draft pull request → ask.

    Signals we look for:
    - ``action == "review_requested"``
    - ``pull_request.draft == False`` (or absent, defaulting to non-draft)
    - ``requested_reviewer`` is present (direct personal request, not a team)
    """
    action = str(data.get("action") or "")
    pr = data.get("pull_request") or data.get("pullRequest") or {}
    is_draft = bool(pr.get("draft", False))
    requested_reviewer = data.get("requested_reviewer") or data.get("requestedReviewer")

    if action == "review_requested" and not is_draft and requested_reviewer:
        return TriggerDecision(
            importance=0.85,
            decision="ask",
            reason="direct review request on a non-draft pull request",
        )
    if action == "review_requested" and is_draft:
        return TriggerDecision(
            importance=0.2,
            decision="silent",
            reason="review requested on a draft pull request — not actionable yet",
        )
    return TriggerDecision(
        importance=0.1,
        decision="silent",
        reason=f"GitHub PR event action={action!r} is not a direct review request",
    )


def _score_important_email(data: dict[str, Any]) -> TriggerDecision:
    """Gmail: email with IMPORTANT/CATEGORY_PERSONAL label or non-bulk sender → ask.

    Bulk/marketing signals (any one → silent):
    - List-Unsubscribe header present in raw headers
    - Sender address contains ``noreply``, ``no-reply``, ``notifications@``
    - Subject starts with ``[``
    - Label ``CATEGORY_PROMOTIONS``, ``CATEGORY_SOCIAL``, or ``CATEGORY_UPDATES``

    Ask signals:
    - Label ``IMPORTANT`` or ``CATEGORY_PERSONAL``
    - No bulk signals detected
    """
    labels: list[str] = []
    raw_labels = data.get("labelIds") or data.get("labels") or []
    if isinstance(raw_labels, list):
        labels = [str(label).upper() for label in raw_labels]

    payload = data.get("payload") or data.get("message") or data
    headers_list: list[dict[str, Any]] = []
    raw_headers = payload.get("headers") or []
    if isinstance(raw_headers, list):
        headers_list = [h for h in raw_headers if isinstance(h, dict)]

    def _header(name: str) -> str:
        name_lower = name.lower()
        for h in headers_list:
            if str(h.get("name") or "").lower() == name_lower:
                return str(h.get("value") or "")
        return ""

    from_addr = _header("from").lower()
    subject = _header("subject")
    list_unsub = _header("list-unsubscribe")

    # Bulk signals
    bulk_sender = any(
        marker in from_addr
        for marker in ("noreply", "no-reply", "notifications@", "donotreply")
    )
    bulk_labels = bool(
        set(labels) & {"CATEGORY_PROMOTIONS", "CATEGORY_SOCIAL", "CATEGORY_UPDATES"}
    )
    bulk_subject = subject.startswith("[")
    has_list_unsub = bool(list_unsub.strip())

    if any([bulk_sender, bulk_labels, bulk_subject, has_list_unsub]):
        return TriggerDecision(
            importance=0.3,
            decision="silent",
            reason="email shows bulk/marketing signals (sender, labels, or headers)",
        )

    # Ask signals
    important_labels = bool(set(labels) & {"IMPORTANT", "CATEGORY_PERSONAL"})
    if important_labels:
        return TriggerDecision(
            importance=0.75,
            decision="ask",
            reason="email has IMPORTANT or CATEGORY_PERSONAL label",
        )

    # No clear signal either way — treat as silent to err on the side of quiet
    return TriggerDecision(
        importance=0.4,
        decision="silent",
        reason="email has no clear importance signal",
    )


def _score_calendar_event_upcoming(data: dict[str, Any]) -> TriggerDecision:
    """Google Calendar: near-term accepted event with attendees → ask.

    Ask signals (all must be true):
    - Start time is within the next 24 hours (uses start.dateTime; all-day events
      with only start.date are treated as not near-term)
    - The attendee status for the owner is ``accepted``
    - Event has multiple attendees (not a solo block)
    - Duration < 4 hours (prep makes sense for normal meetings)
    """
    import datetime as dt

    event = data.get("event") or data

    # All-day events have no dateTime → treat as not near-term
    start_dict = event.get("start") or {}
    start_datetime_str: str | None = start_dict.get("dateTime")
    if not start_datetime_str:
        return TriggerDecision(
            importance=0.2,
            decision="silent",
            reason="all-day event or missing start dateTime — no prep needed",
        )

    try:
        # Parse ISO 8601 with timezone offset
        start_dt = dt.datetime.fromisoformat(start_datetime_str)
    except ValueError:
        return TriggerDecision(
            importance=0.2,
            decision="silent",
            reason="could not parse event start dateTime",
        )

    now_utc = dt.datetime.now(tz=dt.UTC)
    if start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=dt.UTC)

    hours_until = (start_dt - now_utc).total_seconds() / 3600

    if hours_until < 0 or hours_until > 24:
        return TriggerDecision(
            importance=0.2,
            decision="silent",
            reason=f"event is {hours_until:.1f}h away — outside 24h prep window",
        )

    # Duration check
    end_dict = event.get("end") or {}
    end_datetime_str: str | None = end_dict.get("dateTime")
    if end_datetime_str:
        try:
            end_dt = dt.datetime.fromisoformat(end_datetime_str)
            if end_dt.tzinfo is None:
                end_dt = end_dt.replace(tzinfo=dt.UTC)
            duration_hours = (end_dt - start_dt).total_seconds() / 3600
            if duration_hours >= 4:
                return TriggerDecision(
                    importance=0.3,
                    decision="silent",
                    reason=f"event duration {duration_hours:.1f}h — too long for prep suggestion",
                )
        except ValueError:
            pass

    # Attendee check: must have multiple attendees and owner must have accepted
    attendees: list[dict[str, Any]] = []
    raw_attendees = event.get("attendees") or []
    if isinstance(raw_attendees, list):
        attendees = [a for a in raw_attendees if isinstance(a, dict)]

    if len(attendees) < 2:
        return TriggerDecision(
            importance=0.25,
            decision="silent",
            reason="solo event or no attendees — no prep suggestion needed",
        )

    # Check if any attendee (the owner/self) has accepted
    has_accepted = any(
        str(a.get("responseStatus") or "").lower() == "accepted" for a in attendees
    )
    if not has_accepted:
        return TriggerDecision(
            importance=0.3,
            decision="silent",
            reason="no accepted attendee found — may be unconfirmed",
        )

    return TriggerDecision(
        importance=0.7,
        decision="ask",
        reason=f"meeting in {hours_until:.1f}h with accepted attendees — prep suggestion",
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

LAUNCH_TRIGGERS: dict[str, LaunchTrigger] = {
    "GITHUB_PULL_REQUEST_EVENT": LaunchTrigger(
        slug="GITHUB_PULL_REQUEST_EVENT",
        toolkit_slug="github",
        display_name="GitHub Pull Request Review Request",
        scorer=_score_github_review_request,
    ),
    "GMAIL_NEW_GMAIL_MESSAGE": LaunchTrigger(
        slug="GMAIL_NEW_GMAIL_MESSAGE",
        toolkit_slug="gmail",
        display_name="Gmail New Message",
        scorer=_score_important_email,
    ),
    "GOOGLECALENDAR_EVENT_TRIGGERED": LaunchTrigger(
        slug="GOOGLECALENDAR_EVENT_TRIGGERED",
        toolkit_slug="googlecalendar",
        display_name="Google Calendar Event Upcoming",
        scorer=_score_calendar_event_upcoming,
    ),
}
