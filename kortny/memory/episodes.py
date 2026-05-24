"""Episodic task memory.

workspace_state stores confirmed durable facts. Episodes store compact,
deterministic summaries of work Kortny already did, with provenance back to the
task/events/artifacts that produced them.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from kortny.db.models import Artifact, Episode, Task, TaskEvent, TaskEventType
from kortny.db.models import TaskStatus as DbTaskStatus

EPISODE_OUTCOMES = {
    DbTaskStatus.succeeded,
    DbTaskStatus.failed,
    DbTaskStatus.cancelled,
}
MAX_EPISODE_SUMMARY_CHARS = 2_000
MAX_EPISODE_SOURCE_REFS = 12


@dataclass(frozen=True, slots=True)
class TaskEpisode:
    """Public service view of an episode row."""

    id: uuid.UUID
    installation_id: uuid.UUID
    task_id: uuid.UUID
    channel_id: str
    user_id: str
    thread_ts: str | None
    summary: str
    tools_used: tuple[str, ...]
    artifacts_created: tuple[dict[str, Any], ...]
    source_refs: tuple[dict[str, Any], ...]
    outcome: str
    error: dict[str, Any] | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class RelevantEpisode:
    """An episode selected for a task with the retrieval reason."""

    episode: TaskEpisode
    relation: str


class EpisodeService:
    """Records and retrieves bounded task episodes."""

    def __init__(self, session: Session, *, commit_after_write: bool = False) -> None:
        self.session = session
        self.commit_after_write = commit_after_write

    def record_task(self, task: Task | uuid.UUID) -> TaskEpisode | None:
        """Create or refresh the compact episode for a terminal task.

        The summarizer is deterministic: it uses the task result/error plus
        structured tool, source, and artifact events. It does not call an LLM.
        """

        task_obj = self._resolve_task(task)
        outcome = DbTaskStatus(task_obj.status)
        if outcome not in EPISODE_OUTCOMES:
            return None

        events = _task_events(self.session, task_obj)
        artifacts = _task_artifacts(self.session, task_obj)
        episode = self.session.scalar(
            select(Episode).where(Episode.task_id == task_obj.id).with_for_update()
        )
        if episode is None:
            episode = Episode(
                installation_id=task_obj.installation_id,
                task_id=task_obj.id,
                channel_id=task_obj.slack_channel_id,
                user_id=task_obj.slack_user_id,
                thread_ts=task_obj.slack_thread_ts,
                summary="",
                tools_used=[],
                artifacts_created=[],
                source_refs=[],
                outcome=outcome.value,
            )
            self.session.add(episode)

        episode.installation_id = task_obj.installation_id
        episode.channel_id = task_obj.slack_channel_id
        episode.user_id = task_obj.slack_user_id
        episode.thread_ts = task_obj.slack_thread_ts
        episode.summary = _episode_summary(task_obj, artifacts)
        episode.tools_used = list(_tools_used(events))
        episode.artifacts_created = list(_artifact_refs(artifacts))
        episode.source_refs = list(_source_refs(events))
        episode.outcome = outcome.value
        episode.error_json = _error_json(task_obj, events)

        self.session.flush()
        self._commit_if_requested()
        return _episode_from_row(episode)

    def relevant_for_task(
        self,
        task: Task | uuid.UUID,
        *,
        limit: int = 5,
    ) -> tuple[RelevantEpisode, ...]:
        """Return bounded episodes relevant to a task.

        Retrieval is explicit and scoped: same Slack thread first, then same
        channel, then same user. There is intentionally no global workspace
        injection in this MVP.
        """

        if limit < 1:
            return ()

        task_obj = self._resolve_task(task)
        selected: dict[uuid.UUID, RelevantEpisode] = {}
        base = select(Episode).where(
            Episode.installation_id == task_obj.installation_id,
            Episode.task_id != task_obj.id,
        )

        if task_obj.slack_thread_ts:
            self._collect(
                selected,
                relation="same_thread",
                limit=limit,
                statement=base.where(
                    Episode.channel_id == task_obj.slack_channel_id,
                    Episode.thread_ts == task_obj.slack_thread_ts,
                ),
            )

        self._collect(
            selected,
            relation="same_channel",
            limit=limit,
            statement=base.where(Episode.channel_id == task_obj.slack_channel_id),
        )
        self._collect(
            selected,
            relation="same_user",
            limit=limit,
            statement=base.where(Episode.user_id == task_obj.slack_user_id),
        )
        return tuple(selected.values())

    def _collect(
        self,
        selected: dict[uuid.UUID, RelevantEpisode],
        *,
        relation: str,
        limit: int,
        statement: Any,
    ) -> None:
        if len(selected) >= limit:
            return

        rows = self.session.scalars(
            statement.order_by(Episode.created_at.desc(), Episode.id.desc()).limit(
                limit
            )
        )
        for episode in rows:
            if len(selected) >= limit:
                break
            if episode.id in selected:
                continue
            selected[episode.id] = RelevantEpisode(
                episode=_episode_from_row(episode),
                relation=relation,
            )

    def _resolve_task(self, task: Task | uuid.UUID) -> Task:
        if isinstance(task, Task):
            if task.id is None:
                self.session.flush()
            return task

        task_obj = self.session.scalar(select(Task).where(Task.id == task))
        if task_obj is None:
            raise LookupError(f"Task not found: {task}")
        return task_obj

    def _commit_if_requested(self) -> None:
        if self.commit_after_write:
            self.session.commit()


def _task_events(session: Session, task: Task) -> tuple[TaskEvent, ...]:
    return tuple(
        session.scalars(
            select(TaskEvent)
            .where(TaskEvent.task_id == task.id)
            .order_by(TaskEvent.seq)
        )
    )


def _task_artifacts(session: Session, task: Task) -> tuple[Artifact, ...]:
    return tuple(
        session.scalars(
            select(Artifact)
            .where(Artifact.task_id == task.id)
            .order_by(Artifact.created_at, Artifact.id)
        )
    )


def _episode_summary(task: Task, artifacts: Sequence[Artifact]) -> str:
    if task.result_summary and task.result_summary.strip():
        return _shorten(
            task.result_summary.strip(), max_chars=MAX_EPISODE_SUMMARY_CHARS
        )

    status = DbTaskStatus(task.status)
    if status is DbTaskStatus.failed:
        error = _error_summary(task.error)
        if error:
            return _shorten(
                f"Task failed: {error}", max_chars=MAX_EPISODE_SUMMARY_CHARS
            )
        return "Task failed before producing a final summary."
    if status is DbTaskStatus.cancelled:
        return "Task was cancelled before completion."
    if artifacts:
        filenames = ", ".join(artifact.filename for artifact in artifacts[:5])
        return _shorten(
            f"Generated {len(artifacts)} artifact(s): {filenames}",
            max_chars=MAX_EPISODE_SUMMARY_CHARS,
        )
    return "Task completed without a final summary."


def _tools_used(events: Sequence[TaskEvent]) -> tuple[str, ...]:
    tools: list[str] = []
    seen: set[str] = set()
    for event in events:
        if event.type != TaskEventType.tool_call:
            continue
        tool = _payload_str(event.payload, "tool")
        if tool is None or tool in seen:
            continue
        tools.append(tool)
        seen.add(tool)
    return tuple(tools)


def _artifact_refs(artifacts: Sequence[Artifact]) -> tuple[dict[str, Any], ...]:
    refs: list[dict[str, Any]] = []
    for artifact in artifacts:
        ref: dict[str, Any] = {
            "artifact_id": str(artifact.id),
            "filename": artifact.filename,
        }
        if artifact.slack_file_id:
            ref["slack_file_id"] = artifact.slack_file_id
        if artifact.mime_type:
            ref["mime_type"] = artifact.mime_type
        if artifact.size_bytes is not None:
            ref["size_bytes"] = artifact.size_bytes
        refs.append(ref)
    return tuple(refs)


def _source_refs(events: Sequence[TaskEvent]) -> tuple[dict[str, Any], ...]:
    refs: list[dict[str, Any]] = []
    seen: set[tuple[str | None, str | None, str | None]] = set()
    for event in events:
        if event.type != TaskEventType.tool_result:
            continue
        payload = event.payload
        if _payload_str(payload, "tool") != "web_search":
            continue

        output = payload.get("output")
        output_payload = output if isinstance(output, Mapping) else {}
        query = _payload_str(payload, "query") or _payload_str(output_payload, "query")
        results = _payload_list(payload, "results") or _payload_list(
            output_payload, "results"
        )
        for result in results:
            if not isinstance(result, Mapping):
                continue
            title = _payload_str(result, "title")
            url = _payload_str(result, "url")
            snippet = _payload_str(result, "snippet")
            if title is None and url is None:
                continue
            key = (query, title, url)
            if key in seen:
                continue
            seen.add(key)
            ref: dict[str, Any] = {"tool": "web_search"}
            if query:
                ref["query"] = query
            if title:
                ref["title"] = title
            if url:
                ref["url"] = url
            if snippet:
                ref["snippet"] = _shorten(snippet, max_chars=280)
            refs.append(ref)
            if len(refs) >= MAX_EPISODE_SOURCE_REFS:
                return tuple(refs)
    return tuple(refs)


def _error_json(task: Task, events: Sequence[TaskEvent]) -> dict[str, Any] | None:
    if isinstance(task.error, dict):
        return dict(task.error)
    for event in reversed(events):
        if event.type != TaskEventType.error:
            continue
        return dict(event.payload)
    return None


def _episode_from_row(row: Episode) -> TaskEpisode:
    return TaskEpisode(
        id=row.id,
        installation_id=row.installation_id,
        task_id=row.task_id,
        channel_id=row.channel_id,
        user_id=row.user_id,
        thread_ts=row.thread_ts,
        summary=row.summary,
        tools_used=tuple(item for item in row.tools_used if isinstance(item, str)),
        artifacts_created=tuple(
            dict(item) for item in row.artifacts_created if isinstance(item, Mapping)
        ),
        source_refs=tuple(
            dict(item) for item in row.source_refs if isinstance(item, Mapping)
        ),
        outcome=row.outcome,
        error=dict(row.error_json) if isinstance(row.error_json, Mapping) else None,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _payload_str(payload: Mapping[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _payload_list(payload: Mapping[str, Any], key: str) -> list[Any]:
    value = payload.get(key)
    if isinstance(value, list):
        return value
    return []


def _error_summary(error: dict | None) -> str | None:
    if not error:
        return None
    error_type = error.get("type")
    message = error.get("message")
    if isinstance(error_type, str) and isinstance(message, str):
        return f"{error_type}: {message}"
    return str(error)


def _shorten(value: str, *, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[: max(0, max_chars - 3)].rstrip() + "..."
