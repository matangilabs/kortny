"""Channel access gate for cross-channel tool reads.

Gates reads against the asker's membership of the target channel to prevent
the bot's broader channel access from leaking information to users who should
not see that channel.

Exemption: synthetic, scheduled, and other system-initiated tasks have no
human asker and run with the bot's full access.  The gate keys on
``task.identity_kind`` — only ``slack_message``, ``slack_event``, and
``manual`` tasks (i.e. those originated by a real Slack user interaction)
are subject to the membership check.

The current task's own channel is always allowed without a membership lookup
(the asker is participating there by definition).

Membership is checked via the Slack ``conversations.members`` API (paginated)
and the result is cached per gate instance (one instance per task invocation)
to avoid N+1 API calls within a single tool chain.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from kortny.db.models import Task
from kortny.tools.types import RecoverableToolError

logger = logging.getLogger(__name__)

# identity_kind values that represent a real Slack user initiating a task.
# Synthetic and scheduled tasks bypass the gate entirely.
_USER_INITIATED_KINDS: frozenset[str] = frozenset(
    {"slack_message", "slack_event", "manual"}
)

# Pagination ceiling for conversations.members — we stop after this many
# pages.  200 pages x 200 per page = 40 000 members, which is safely above
# any realistic Slack workspace.
_MEMBERS_PAGE_LIMIT = 200
_MEMBERS_PER_PAGE = 200


class MembershipCheckClient(Protocol):
    """Subset of Slack WebClient needed by ChannelAccessGate."""

    def conversations_members(
        self,
        *,
        channel: str,
        cursor: str | None = None,
        limit: int | None = None,
    ) -> Any:
        """Return one page of channel member IDs."""


class ChannelAccessGate:
    """Enforce asker membership before allowing cross-channel reads.

    One instance is created per task execution (inside the tool factory) so
    the per-task membership cache stays scoped to a single agent turn cycle.
    """

    def __init__(
        self,
        task: Task,
        client: MembershipCheckClient | None,
    ) -> None:
        self._task = task
        self._client = client
        # Cache: channel_id -> bool (True = asker is a member)
        self._membership_cache: dict[str, bool] = {}

    def can_read(self, target_channel_id: str) -> bool:
        """Return True if the asker is allowed to read target_channel_id.

        Returns True unconditionally for non-user-initiated tasks and for the
        task's own channel.  Does NOT raise — use this when you need to test
        multiple channels and deny only if none are accessible.
        """
        task = self._task
        if task.identity_kind not in _USER_INITIATED_KINDS:
            return True
        if target_channel_id == task.slack_channel_id:
            return True
        return self._is_member(task.slack_user_id, target_channel_id)

    def check(self, target_channel_id: str) -> None:
        """Raise RecoverableToolError if the asker cannot read target_channel_id.

        This is a no-op when:
        - The task is not user-initiated (synthetic/scheduled/etc.)
        - The target channel is the task's own channel.

        Raises RecoverableToolError(code="channel_access_denied") on denial.
        """
        task = self._task

        # Synthetic, scheduled, or any other non-user-initiated task bypasses
        # the gate so ambient pipelines (observe, witness, assessment, intent
        # classifier, knowledge graph refresh) continue to work.
        if task.identity_kind not in _USER_INITIATED_KINDS:
            return

        # Current task channel is always allowed — the asker is there.
        if target_channel_id == task.slack_channel_id:
            return

        asker_id = task.slack_user_id
        is_member = self._is_member(asker_id, target_channel_id)
        if not is_member:
            logger.info(
                "channel_access_denied asker=%s target=%s task_channel=%s "
                "identity_kind=%s task_id=%s",
                asker_id,
                target_channel_id,
                task.slack_channel_id,
                task.identity_kind,
                task.id,
            )
            raise RecoverableToolError(
                code="channel_access_denied",
                message=(
                    f"You do not have access to channel {target_channel_id}. "
                    "Kortny can only read channels you are a member of."
                ),
                hint=(
                    "If you meant the current channel, retry without specifying "
                    "a channel_id. Otherwise, join the channel first or ask the "
                    "channel owner to share the content with you."
                ),
            )

    def _is_member(self, slack_user_id: str, channel_id: str) -> bool:
        """Return True if slack_user_id is a member of channel_id.

        Uses the per-instance cache; falls back to the Slack API if not cached.
        If no client is available, denies access (fail-closed).
        """
        if channel_id in self._membership_cache:
            return self._membership_cache[channel_id]

        if self._client is None:
            # No live client — we cannot verify membership, so deny.
            logger.warning(
                "channel_access_gate: no client available for membership check "
                "channel=%s asker=%s; denying",
                channel_id,
                slack_user_id,
            )
            self._membership_cache[channel_id] = False
            return False

        result = self._fetch_membership(slack_user_id, channel_id)
        self._membership_cache[channel_id] = result
        return result

    def _fetch_membership(self, slack_user_id: str, channel_id: str) -> bool:
        """Call conversations.members (paginated) to check membership."""
        assert self._client is not None
        cursor: str | None = None
        pages_fetched = 0
        try:
            while pages_fetched < _MEMBERS_PAGE_LIMIT:
                response = self._client.conversations_members(
                    channel=channel_id,
                    cursor=cursor,
                    limit=_MEMBERS_PER_PAGE,
                )
                pages_fetched += 1
                members, next_cursor = _parse_members_response(response)
                if slack_user_id in members:
                    return True
                if not next_cursor:
                    break
                cursor = next_cursor
        except Exception as exc:
            # Treat Slack API errors as "cannot verify -> deny" (fail-closed).
            # Common case: channel_not_found, not_in_channel, etc.
            logger.info(
                "channel_access_gate: membership check failed "
                "channel=%s asker=%s error=%r; denying",
                channel_id,
                slack_user_id,
                exc,
            )
            return False
        return False


def _parse_members_response(
    response: Any,
) -> tuple[frozenset[str], str | None]:
    """Extract member IDs and next_cursor from a conversations.members response."""
    if isinstance(response, dict):
        payload = response
    else:
        payload = getattr(response, "data", None) or {}

    raw_members = payload.get("members") or []
    members = frozenset(m for m in raw_members if isinstance(m, str))

    metadata = payload.get("response_metadata") or {}
    next_cursor: str | None = None
    if isinstance(metadata, dict):
        c = metadata.get("next_cursor")
        if isinstance(c, str) and c:
            next_cursor = c

    return members, next_cursor
