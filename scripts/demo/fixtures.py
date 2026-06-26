"""Deterministic "Northwind" fixture story for the demo workspace seeder.

The story spans a configurable backdated window and is generated from the
current time at seed time — no LLM, no network, no randomness. Five personas
talk across multiple channels and exhibit the patterns the ambient stack
(observe -> witness -> automation) must discover:

1. A recurring standup post Mon/Wed/Thu in #engineering (automation bait).
2. A weekly status report compiled by hand every Friday in #product.
3. A one-shot Stripe webhook verification in #ops.
4. A Redshift vs BigQuery decision thread in #engineering that trails off.
5. A product roadmap file share in #product.
6. A magic moment: Dana asks Kortny for a v2 launch status in #launch.
7. Greetings and ops chatter as noise the extractor must discriminate against.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, time, timedelta
from datetime import date as date_cls

SIM_MARKER_KEY = "sim"
SIM_SOURCE = "workspace_simulator"
SIM_TASK_IDENTITY_SOURCE = "sim"
SIM_TASK_IDENTITY_PREFIX = "synthetic:sim:"
SIM_EVENT_ID_PREFIX = "sim:"

DEFAULT_SIM_DAYS = 21


@dataclass(frozen=True, slots=True)
class SimPersona:
    """A fake team member; the user id is intentionally non-real."""

    user_id: str
    display_name: str
    role: str
    icon_emoji: str = ""


@dataclass(frozen=True, slots=True)
class SimMessage:
    """One deterministic channel message in the fixture story."""

    slug: str
    persona: SimPersona
    sent_at: datetime
    text: str
    pattern: str
    channel_name: str = "general"
    thread_slug: str | None = None
    # Compact Slack-style files summary (id, name, filetype, size). Entries
    # carry the SIM_MARKER_KEY flag so the ambient-file gate never tries to
    # download a file id that does not exist in Slack.
    files: tuple[Mapping[str, object], ...] = field(default=())

    @property
    def message_ts(self) -> str:
        """Slack-style message timestamp derived from the send time."""
        return f"{self.sent_at.timestamp():.6f}"


DANA = SimPersona("USIM01", "Dana Okonkwo", "Founder / CEO", icon_emoji=":star:")
PRIYA = SimPersona("USIM02", "Priya Raman", "Product Manager", icon_emoji=":clipboard:")
MARCO = SimPersona("USIM03", "Marco Diaz", "Senior Engineer", icon_emoji=":computer:")
LENA = SimPersona("USIM04", "Lena Foss", "Engineer", icon_emoji=":wrench:")
THEO = SimPersona(
    "USIM05", "Theo Brandt", "Ops / RevOps", icon_emoji=":chart_with_upwards_trend:"
)

PERSONAS: tuple[SimPersona, ...] = (DANA, PRIYA, MARCO, LENA, THEO)

_STANDUP_POSTERS: tuple[SimPersona, ...] = (LENA, MARCO)

_STANDUP_LINES: tuple[str, ...] = (
    "{name}: shipped {done}, today working on {next}. No blockers.",
    "{name}: wrapped up {done} yesterday, focusing on {next} today.",
    "{name}: {done} is done and merged, picking up {next} now.",
)

_DONE_ITEMS: tuple[str, ...] = (
    "the auth refactor",
    "the billing API integration",
    "the webhook retry logic",
    "the dashboard pagination fix",
    "the Stripe event handler",
    "the user invite flow",
    "the onboarding email templates",
)

_NEXT_ITEMS: tuple[str, ...] = (
    "rate-limit middleware",
    "the metrics pipeline",
    "error boundary cleanup",
    "the CSV export endpoint",
    "a Datadog alert for payment failures",
    "the seat management UI",
    "load testing the new endpoints",
)

_STATUS_REPLIES: tuple[tuple[SimPersona, str], ...] = (
    (
        MARCO,
        "API hardening is done; starting on the metrics export endpoint this week.",
    ),
    (LENA, "Finished the billing flow, picking up the onboarding polish next."),
)

_OPS_CHATTER: tuple[str, ...] = (
    "heads up: rotating the Stripe webhook signing secret this afternoon",
    "Heroku dyno upgraded to standard-2x, memory headroom looks good",
    "staging deploy went out clean, no alerts so far",
    "reminder: freeze period starts Monday before the v2 launch",
)


def build_story(*, now: datetime, days: int) -> tuple[SimMessage, ...]:
    """Build the deterministic message history for a backdated window.

    Messages are ordered by send time and every send time is strictly within
    [now - days, now]. The schedule is calendar-aware so the standup pattern
    on Mon/Wed/Thu is real for any window.
    """
    if days < 1:
        raise ValueError("days must be >= 1")
    now = now.astimezone(UTC)
    window_start = now - timedelta(days=days)

    messages: list[SimMessage] = []
    day_count = days + 1
    for day_index in range(day_count):
        date = (window_start + timedelta(days=day_index)).date()
        weekday = date.weekday()
        day_messages: list[SimMessage] = []

        # Noise: Mon/Wed Lena says good morning
        if weekday in (0, 2):
            day_messages.append(
                SimMessage(
                    slug=f"noise-greeting-{date.isoformat()}",
                    persona=LENA,
                    sent_at=_at(date, 9, 5),
                    text="morning all :coffee: ready to ship",
                    pattern="noise",
                    channel_name="general",
                )
            )

        # Noise: Tue/Thu Theo posts ops chatter
        if weekday in (1, 3):
            day_messages.append(
                SimMessage(
                    slug=f"noise-ops-{date.isoformat()}",
                    persona=THEO,
                    sent_at=_at(date, 14, 20),
                    text=_OPS_CHATTER[day_index % len(_OPS_CHATTER)],
                    pattern="noise",
                    channel_name="general",
                )
            )

        # Standup: Mon(0)/Wed(2)/Thu(3) alternating Lena and Marco
        if weekday in (0, 2, 3):
            poster = _STANDUP_POSTERS[day_index % len(_STANDUP_POSTERS)]
            done = _DONE_ITEMS[day_index % len(_DONE_ITEMS)]
            nxt = _NEXT_ITEMS[day_index % len(_NEXT_ITEMS)]
            line = _STANDUP_LINES[day_index % len(_STANDUP_LINES)]
            day_messages.append(
                SimMessage(
                    slug=f"standup-{date.isoformat()}",
                    persona=poster,
                    sent_at=_at(date, 9, 30),
                    text=line.format(name=poster.display_name, done=done, next=nxt),
                    pattern="standup",
                    channel_name="engineering",
                )
            )

        # Weekly status: Priya asks Friday, Marco/Lena reply, Priya posts summary
        if weekday == 4:
            day_messages.extend(_friday_status_messages(date))

        messages.extend(
            message
            for message in day_messages
            if window_start <= message.sent_at <= now
        )

    # One-shot: Stripe webhook issue from Theo (middle of window)
    messages.extend(
        message
        for message in _one_shot_messages(window_start, days)
        if window_start <= message.sent_at <= now
    )
    # Unresolved decision: Redshift vs BigQuery in #engineering
    messages.extend(
        message
        for message in _vendor_decision_messages(window_start, days)
        if window_start <= message.sent_at <= now
    )
    # File share: Priya drops product roadmap PDF in #product
    messages.extend(
        message
        for message in _file_share_messages(window_start, days)
        if window_start <= message.sent_at <= now
    )
    # Magic moment: Dana mentions @Kortny near end of window in #launch
    messages.extend(
        message
        for message in _magic_moment_messages(window_start, days)
        if window_start <= message.sent_at <= now
    )

    messages.sort(key=lambda message: (message.sent_at, message.slug))
    return tuple(messages)


def _friday_status_messages(date: date_cls) -> list[SimMessage]:
    ask_slug = f"weekly-status-ask-{date.isoformat()}"
    msgs: list[SimMessage] = [
        SimMessage(
            slug=ask_slug,
            persona=PRIYA,
            sent_at=_at(date, 10, 0),
            text=(
                "It's Friday — please drop your status in this thread by noon "
                "so I can compile the weekly update for stakeholders."
            ),
            pattern="weekly_status",
            channel_name="product",
        )
    ]
    for offset_minutes, (persona, update) in zip(
        (30, 55), _STATUS_REPLIES, strict=True
    ):
        msgs.append(
            SimMessage(
                slug=f"weekly-status-reply-{persona.user_id.lower()}-{date.isoformat()}",
                persona=persona,
                sent_at=_at(date, 10, offset_minutes),
                text=update,
                pattern="weekly_status",
                channel_name="product",
                thread_slug=ask_slug,
            )
        )
    msgs.append(
        SimMessage(
            slug=f"weekly-status-summary-{date.isoformat()}",
            persona=PRIYA,
            sent_at=_at(date, 12, 0),
            text=(
                f"Weekly status, week of {date.isoformat()} (compiled by hand again): "
                "API hardening done, billing flow shipped, onboarding polish in progress. "
                "Pasting into the stakeholder email now."
            ),
            pattern="weekly_status",
            channel_name="product",
        )
    )
    return msgs


def _one_shot_messages(
    window_start: datetime,
    days: int,
) -> tuple[SimMessage, ...]:
    """A Stripe webhook verification ask that nobody resolves."""
    date = (window_start + timedelta(days=max(days // 2, 1))).date()
    return (
        SimMessage(
            slug=f"oneshot-stripe-webhook-{date.isoformat()}",
            persona=THEO,
            sent_at=_at(date, 11, 15),
            text=(
                "We're seeing intermittent 401s on the Stripe webhook endpoint. "
                "Someone needs to verify the signing secret is rotated correctly "
                "before the v2 launch — I haven't had time."
            ),
            pattern="one_shot",
            channel_name="ops",
        ),
    )


def _vendor_decision_messages(
    window_start: datetime,
    days: int,
) -> tuple[SimMessage, ...]:
    """A two-day Redshift vs BigQuery debate that trails off unresolved."""
    first = (window_start + timedelta(days=max(days // 3, 1))).date()
    second = first + timedelta(days=1)
    root_slug = f"metrics-pipeline-{first.isoformat()}"
    return (
        SimMessage(
            slug=root_slug,
            persona=MARCO,
            sent_at=_at(first, 13, 0),
            text=(
                "We need to pick a data warehouse for the metrics pipeline: "
                "Redshift is cheaper but BigQuery has better serverless scaling. "
                "Thoughts before I start the PoC?"
            ),
            pattern="unresolved_decision",
            channel_name="engineering",
        ),
        SimMessage(
            slug=f"metrics-pipeline-reply1-{first.isoformat()}",
            persona=LENA,
            sent_at=_at(first, 13, 25),
            text="BigQuery's pricing model is unpredictable at our scale; leaning Redshift.",
            pattern="unresolved_decision",
            channel_name="engineering",
            thread_slug=root_slug,
        ),
        SimMessage(
            slug=f"metrics-pipeline-reply2-{second.isoformat()}",
            persona=THEO,
            sent_at=_at(second, 9, 40),
            text="Either works for now budget-wise. Do we have a decision owner?",
            pattern="unresolved_decision",
            channel_name="engineering",
            thread_slug=root_slug,
        ),
        SimMessage(
            slug=f"metrics-pipeline-reply3-{second.isoformat()}",
            persona=MARCO,
            sent_at=_at(second, 16, 10),
            text="Let's revisit next week, swamped with the API hardening sprint.",
            pattern="unresolved_decision",
            channel_name="engineering",
            thread_slug=root_slug,
        ),
    )


def _file_share_messages(
    window_start: datetime,
    days: int,
) -> tuple[SimMessage, ...]:
    """Priya drops the product roadmap PDF near the end of the window."""
    date = (window_start + timedelta(days=max(days - 2, 1))).date()
    return (
        SimMessage(
            slug=f"file-share-roadmap-{date.isoformat()}",
            persona=PRIYA,
            sent_at=_at(date, 15, 10),
            text=(
                "Dropping the v2 product roadmap here before the stakeholder sync — "
                "let me know if anything looks off."
            ),
            pattern="file_share",
            channel_name="product",
            files=(
                {
                    "id": "FSIMNWRM01",
                    "name": "northwind-v2-roadmap.pdf",
                    "filetype": "pdf",
                    "size": 182_400,
                    SIM_MARKER_KEY: True,
                },
            ),
        ),
    )


def _magic_moment_messages(
    window_start: datetime,
    days: int,
) -> tuple[SimMessage, ...]:
    """Dana asks @Kortny for a v2 launch status near the end of the window."""
    date = (window_start + timedelta(days=max(days - 1, 1))).date()
    return (
        SimMessage(
            slug=f"magic-moment-launch-{date.isoformat()}",
            persona=DANA,
            sent_at=_at(date, 10, 0),
            text=(
                "@Kortny we're launching v2 Thursday — where are we? "
                "What shipped this week and what's still open?"
            ),
            pattern="magic_moment",
            channel_name="launch",
        ),
    )


def _at(date: date_cls, hour: int, minute: int) -> datetime:
    return datetime.combine(date, time(hour=hour, minute=minute), tzinfo=UTC)
