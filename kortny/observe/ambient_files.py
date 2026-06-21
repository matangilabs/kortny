"""HIG-231: ambient file analysis — gated in-thread briefs for observed file posts.

Observed channel messages that carry whitelisted data files (xlsx / csv / pdf)
can spawn a synthetic analysis task that replies in the file's thread with a
short, evidence-first brief. The surface is strictly opt-in: only channels
whose ObservePolicy has ``proactivity_status == "full"`` ever get a post.

Budget accounting is shared with HIG-198 channel posts through
``witness_delivery_log`` rows: this module writes ``decision='ambient_file_brief'``
rows (keyed ``channel:{channel_id}`` in ``slack_user_id``) at task-creation time
and counts both ``ambient_file_brief`` and ``channel_sent`` rows for the weekly
channel window. Do not modify ``kortny/witness/`` from here — only the
``WitnessDeliveryLog`` model is imported.

Everything in this module that runs at Slack ingress is cheap: a few indexed
DB reads plus at most one task insert. No LLM calls happen here.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from kortny.config import Settings, load_settings
from kortny.db.models import (
    Installation,
    ObservationEvent,
    ObservePolicy,
    Task,
    WitnessDeliveryLog,
)
from kortny.tasks import TaskIdentity, TaskService

logger = logging.getLogger(__name__)

AMBIENT_FILE_TYPES = frozenset({"xlsx", "csv", "pdf"})
AMBIENT_FILE_BRIEF_DECISION = "ambient_file_brief"
# HIG-198 channel posts write this decision into the same log; both kinds
# count against the shared weekly per-channel proactivity window
# (KORTNY_WITNESS_CHANNEL_POSTS_PER_WEEK).
CHANNEL_POST_DECISION = "channel_sent"
# Channel-scoped delivery-log rows key slack_user_id as "channel:{channel_id}"
# (same convention as the HIG-198 channel rows in kortny/witness/runner.py,
# WITNESS_CHANNEL_LOG_USER_PREFIX) so both features see each other's rows.
CHANNEL_LOG_USER_PREFIX = "channel:"
WEEKLY_WINDOW_DAYS = 7
AMBIENT_TASK_SOURCE = "ambient-file"
AMBIENT_SOURCE_SURFACE = "ambient_file"
# Simulator fixtures mark fake file entries so the gate never tries to
# analyze a file id that does not exist in Slack.
SIM_FILE_MARKER = "sim"
FILES_SUMMARY_MAX_ENTRIES = 10
_SUMMARY_KEYS = ("id", "name", "filetype", "size")


@dataclass(frozen=True, slots=True)
class AmbientFileCandidate:
    """One whitelisted file attached to an observed channel message."""

    file_id: str
    name: str | None
    filetype: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class AmbientFileDecision:
    """Outcome of the ambient-file gate for one observed message."""

    created: bool
    reason: str
    task: Task | None = None
    candidates: tuple[AmbientFileCandidate, ...] = ()


def ambient_file_identity_key(channel_id: str, file_id: str) -> str:
    """Deterministic task identity key: dedup is forever per channel+file id."""

    return f"synthetic:{AMBIENT_TASK_SOURCE}:{channel_id}:{file_id}"


def summarize_event_files(
    files: Sequence[Mapping[str, Any]],
    *,
    limit: int = FILES_SUMMARY_MAX_ENTRIES,
) -> list[dict[str, Any]]:
    """Compact (id, name, filetype, size) summary for visibility_metadata."""

    summary: list[dict[str, Any]] = []
    for file in files[:limit]:
        entry: dict[str, Any] = {}
        for key in _SUMMARY_KEYS:
            value = file.get(key)
            if value is not None:
                entry[key] = value
        if file.get(SIM_FILE_MARKER):
            entry[SIM_FILE_MARKER] = True
        if entry:
            summary.append(entry)
    return summary


def detect_file_candidates(
    event: Mapping[str, Any],
    *,
    max_mb: int,
) -> tuple[AmbientFileCandidate, ...]:
    """Whitelisted, size-bounded, non-simulated files on a message event."""

    raw_files = event.get("files")
    if not isinstance(raw_files, list):
        return ()
    max_bytes = max_mb * 1024 * 1024
    candidates: list[AmbientFileCandidate] = []
    for file in raw_files:
        if not isinstance(file, Mapping):
            continue
        if file.get(SIM_FILE_MARKER):
            continue
        file_id = file.get("id")
        if not isinstance(file_id, str) or not file_id.strip():
            continue
        filetype = file.get("filetype")
        if not isinstance(filetype, str) or filetype.lower() not in AMBIENT_FILE_TYPES:
            continue
        size = file.get("size")
        if not isinstance(size, int) or isinstance(size, bool):
            continue
        if size < 0 or size > max_bytes:
            continue
        raw_name = file.get("name")
        name = raw_name if isinstance(raw_name, str) and raw_name.strip() else None
        candidates.append(
            AmbientFileCandidate(
                file_id=file_id.strip(),
                name=name,
                filetype=filetype.lower(),
                size_bytes=size,
            )
        )
    return tuple(candidates)


def build_ambient_file_brief_input(
    *,
    channel_id: str,
    candidates: Sequence[AmbientFileCandidate],
) -> str:
    """Instructing input for the synthetic analysis task."""

    lines = []
    for candidate in candidates:
        size_mb = candidate.size_bytes / (1024 * 1024)
        label = candidate.name or candidate.file_id
        lines.append(
            f"- {label} ({candidate.filetype}, {size_mb:.1f} MB, "
            f"Slack file id {candidate.file_id})"
        )
    file_lines = "\n".join(lines)
    return (
        f"A teammate shared a data file in <#{channel_id}> without asking for "
        "anything. Read the shared file(s) with the slack_file_read tool; for "
        "xlsx or csv contents, compute over the data in the sandbox instead of "
        "eyeballing it. Then reply in this thread with a short, evidence-first "
        "brief: (1) the story the data tells, (2) two or three standouts worth "
        "attention, citing the rows or figures you used, and (3) one next "
        "artifact you could put together if it would help — offer it, do not "
        "build it. This is analysis only: do not modify anything and do not "
        "use external tools beyond reading the file(s) and sandbox compute. "
        "Keep the register low-key and conversational.\n"
        f"Files:\n{file_lines}"
    )


class AmbientFileAnalysisService:
    """Gate observed file posts into at most one budgeted analysis task."""

    def __init__(
        self,
        *,
        session: Session,
        settings: Settings,
        task_service: TaskService | None = None,
    ) -> None:
        self.session = session
        self.settings = settings
        self.task_service = task_service or TaskService(session)

    def maybe_create_analysis_task(
        self,
        *,
        installation: Installation,
        policy: ObservePolicy,
        observation: ObservationEvent,
        event: Mapping[str, Any],
    ) -> AmbientFileDecision:
        """Run all gates (cheap DB reads only) and create at most one task."""

        if not self.settings.ambient_files_enabled:
            return AmbientFileDecision(created=False, reason="disabled")

        candidates = detect_file_candidates(
            event,
            max_mb=self.settings.ambient_file_max_mb,
        )
        if not candidates:
            return AmbientFileDecision(created=False, reason="no_candidates")

        if policy.proactivity_status != "full":
            return AmbientFileDecision(
                created=False,
                reason="policy_not_full",
                candidates=candidates,
            )

        channel_id = observation.channel_id
        identity_keys = [
            ambient_file_identity_key(channel_id, candidate.file_id)
            for candidate in candidates
        ]
        already_analyzed = self.session.scalars(
            select(Task.identity_key).where(
                Task.installation_id == installation.id,
                Task.identity_key.in_(identity_keys),
            )
        ).first()
        if already_analyzed is not None:
            return AmbientFileDecision(
                created=False,
                reason="duplicate",
                candidates=candidates,
            )

        now = datetime.now(UTC)
        day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        daily_count = self._count_log_rows(
            installation_id=installation.id,
            channel_id=channel_id,
            decisions=(AMBIENT_FILE_BRIEF_DECISION,),
            since=day_start,
        )
        if daily_count >= self.settings.ambient_file_briefs_per_day:
            logger.info(
                "ambient file brief deferred reason=daily_budget channel=%s "
                "file_id=%s count=%s limit=%s",
                channel_id,
                candidates[0].file_id,
                daily_count,
                self.settings.ambient_file_briefs_per_day,
            )
            return AmbientFileDecision(
                created=False,
                reason="daily_budget_exhausted",
                candidates=candidates,
            )

        weekly_budget = self.settings.witness_channel_posts_per_week
        weekly_count = self._count_log_rows(
            installation_id=installation.id,
            channel_id=channel_id,
            decisions=(AMBIENT_FILE_BRIEF_DECISION, CHANNEL_POST_DECISION),
            since=now - timedelta(days=WEEKLY_WINDOW_DAYS),
        )
        if weekly_count >= weekly_budget:
            logger.info(
                "ambient file brief deferred reason=weekly_budget channel=%s "
                "file_id=%s count=%s limit=%s",
                channel_id,
                candidates[0].file_id,
                weekly_count,
                weekly_budget,
            )
            return AmbientFileDecision(
                created=False,
                reason="weekly_budget_exhausted",
                candidates=candidates,
            )

        primary = candidates[0]
        input_text = build_ambient_file_brief_input(
            channel_id=channel_id,
            candidates=candidates,
        )
        thread_ts = observation.thread_ts or observation.message_ts
        task = self.task_service.create_task(
            installation_id=installation.id,
            slack_channel_id=channel_id,
            slack_user_id=(observation.user_id or installation.bot_user_id or "system"),
            slack_thread_ts=thread_ts,
            slack_message_ts=observation.message_ts,
            input=input_text,
            identity=TaskIdentity.synthetic(
                source=AMBIENT_TASK_SOURCE,
                source_id=f"{channel_id}:{primary.file_id}",
                input_text=input_text,
                payload={
                    "channel_id": channel_id,
                    "file_id": primary.file_id,
                    "file_ids": [candidate.file_id for candidate in candidates],
                    "message_ts": observation.message_ts,
                    "observation_id": str(observation.id),
                    "runtime_cost_ceiling_usd": str(
                        self.settings.ambient_task_cost_ceiling_usd
                    ),
                },
            ),
            source_surface=AMBIENT_SOURCE_SURFACE,
        )
        # Budget counts from creation time, not completion time.
        self.session.add(
            WitnessDeliveryLog(
                installation_id=installation.id,
                slack_user_id=f"{CHANNEL_LOG_USER_PREFIX}{channel_id}",
                candidate_id=None,
                decision=AMBIENT_FILE_BRIEF_DECISION,
                reason=(
                    f"ambient file brief file_id={primary.file_id} "
                    f"message_ts={observation.message_ts}"
                ),
            )
        )
        self.session.flush()
        logger.info(
            "ambient file brief task created task_id=%s channel=%s file_id=%s",
            task.id,
            channel_id,
            primary.file_id,
        )
        return AmbientFileDecision(
            created=True,
            reason="created",
            task=task,
            candidates=candidates,
        )

    def _count_log_rows(
        self,
        *,
        installation_id: uuid.UUID,
        channel_id: str,
        decisions: tuple[str, ...],
        since: datetime,
    ) -> int:
        count = self.session.scalar(
            select(func.count())
            .select_from(WitnessDeliveryLog)
            .where(
                WitnessDeliveryLog.installation_id == installation_id,
                WitnessDeliveryLog.slack_user_id
                == f"{CHANNEL_LOG_USER_PREFIX}{channel_id}",
                WitnessDeliveryLog.decision.in_(decisions),
                WitnessDeliveryLog.created_at >= since,
            )
        )
        return int(count or 0)


@lru_cache(maxsize=1)
def _default_settings() -> Settings:
    return load_settings()


def maybe_create_ambient_file_brief(
    *,
    session: Session,
    installation: Installation,
    policy: ObservePolicy,
    observation: ObservationEvent,
    event: Mapping[str, Any],
    task_service: TaskService | None = None,
    settings: Settings | None = None,
) -> AmbientFileDecision:
    """Ingress seam: best-effort gate that never breaks observation capture.

    The no-files fast path returns before settings are even resolved, so the
    common case adds zero queries and zero settings work to message ingress.
    """

    raw_files = event.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        return AmbientFileDecision(created=False, reason="no_files")

    try:
        resolved_settings = settings or _default_settings()
    except Exception:
        logger.exception("ambient file gate could not load settings; skipping")
        return AmbientFileDecision(created=False, reason="settings_unavailable")

    service = AmbientFileAnalysisService(
        session=session,
        settings=resolved_settings,
        task_service=task_service,
    )
    try:
        return service.maybe_create_analysis_task(
            installation=installation,
            policy=policy,
            observation=observation,
            event=event,
        )
    except Exception:
        logger.exception(
            "ambient file gate failed channel=%s message_ts=%s",
            observation.channel_id,
            observation.message_ts,
        )
        return AmbientFileDecision(created=False, reason="error")
