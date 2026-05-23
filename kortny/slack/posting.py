"""Slack message and file posting for task results."""

from __future__ import annotations

import mimetypes
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session

from kortny.db.models import Artifact, Task, TaskEventType
from kortny.tasks import TaskService


class SlackPostingError(RuntimeError):
    """Raised when Slack returns an unexpected posting response."""


class SlackPostingClient(Protocol):
    """Subset of the Slack WebClient used by result posting."""

    def chat_postMessage(
        self,
        *,
        channel: str,
        text: str,
        thread_ts: str | None = None,
    ) -> Mapping[str, Any]:
        """Post a Slack message."""

    def files_upload_v2(
        self,
        *,
        file: str,
        filename: str | None = None,
        title: str | None = None,
        channel: str | None = None,
        initial_comment: str | None = None,
        thread_ts: str | None = None,
    ) -> Mapping[str, Any]:
        """Upload and share a Slack file."""


@dataclass(frozen=True, slots=True)
class SlackThread:
    """Slack channel/thread target, optionally tied to a task."""

    channel_id: str
    thread_ts: str
    task_id: uuid.UUID | None = None

    @classmethod
    def from_task(cls, task: Task) -> SlackThread:
        thread_ts = task.slack_thread_ts or task.slack_message_ts
        if not thread_ts:
            raise ValueError("Task has no Slack thread timestamp")
        return cls(
            channel_id=task.slack_channel_id,
            thread_ts=thread_ts,
            task_id=task.id,
        )


class SlackPoster:
    """Posts task results and artifacts back to Slack."""

    def __init__(
        self,
        *,
        session: Session,
        client: SlackPostingClient,
        task_service: TaskService | None = None,
    ) -> None:
        self.session = session
        self.client = client
        self.task_service = task_service or TaskService(session)

    def post_message(
        self,
        thread: SlackThread,
        text: str,
        *,
        purpose: str = "result",
    ) -> str:
        """Post text into a Slack thread and return the Slack message ts."""

        post_thread_ts = _post_thread_ts(thread)
        response = self.client.chat_postMessage(
            channel=thread.channel_id,
            text=text,
            thread_ts=post_thread_ts,
        )
        message_ts = _response_ts(response)
        if message_ts is None:
            raise SlackPostingError("Slack chat_postMessage response is missing ts")

        if thread.task_id is not None:
            self.task_service.append_event(
                thread.task_id,
                TaskEventType.message_posted,
                {
                    "channel": thread.channel_id,
                    "thread_ts": post_thread_ts,
                    "message_ts": message_ts,
                    "text": text,
                    "purpose": purpose,
                },
            )
        return message_ts

    def upload_file(
        self,
        thread: SlackThread,
        path: str | Path,
        *,
        artifact: Artifact | uuid.UUID | None = None,
        initial_comment: str | None = None,
        title: str | None = None,
        now: datetime | None = None,
    ) -> str:
        """Upload a file into a Slack thread and mark the artifact as posted."""

        file_path = Path(path)
        if not file_path.is_file():
            raise FileNotFoundError(file_path)
        task_id = thread.task_id
        if task_id is None:
            raise ValueError("SlackThread.task_id is required for file uploads")

        artifact_obj = self._resolve_or_create_artifact(thread, file_path, artifact)
        if artifact_obj.posted_at is not None:
            if artifact_obj.slack_file_id:
                return artifact_obj.slack_file_id
            raise SlackPostingError("Artifact is posted but missing slack_file_id")

        post_thread_ts = _post_thread_ts(thread)
        response = self.client.files_upload_v2(
            file=str(file_path),
            filename=file_path.name,
            title=title or file_path.name,
            channel=thread.channel_id,
            initial_comment=initial_comment,
            thread_ts=post_thread_ts,
        )
        slack_file_id = _response_file_id(response)
        if slack_file_id is None:
            raise SlackPostingError("Slack files_upload_v2 response is missing file id")

        artifact_obj.slack_file_id = slack_file_id
        artifact_obj.posted_at = now or datetime.now(UTC)
        self.session.flush()

        self.task_service.append_event(
            task_id,
            TaskEventType.message_posted,
            {
                "channel": thread.channel_id,
                "thread_ts": post_thread_ts,
                "slack_file_id": slack_file_id,
                "artifact_id": str(artifact_obj.id),
                "filename": artifact_obj.filename,
                "purpose": "file_upload",
            },
        )
        return slack_file_id

    def _resolve_or_create_artifact(
        self,
        thread: SlackThread,
        path: Path,
        artifact: Artifact | uuid.UUID | None,
    ) -> Artifact:
        if thread.task_id is None:
            raise ValueError("SlackThread.task_id is required for file uploads")

        if isinstance(artifact, Artifact):
            return artifact
        if isinstance(artifact, uuid.UUID):
            artifact_obj = self.session.scalar(
                select(Artifact).where(Artifact.id == artifact)
            )
            if artifact_obj is None:
                raise LookupError(f"Artifact not found: {artifact}")
            return artifact_obj

        artifact_obj = self._find_artifact_by_path(thread.task_id, path)
        if artifact_obj is not None:
            return artifact_obj

        mime_type, _ = mimetypes.guess_type(path.name)
        artifact_obj = Artifact(
            task_id=thread.task_id,
            filename=path.name,
            mime_type=mime_type,
            size_bytes=path.stat().st_size,
            storage_path=str(path),
        )
        self.session.add(artifact_obj)
        self.session.flush()
        self.task_service.append_event(
            thread.task_id,
            TaskEventType.artifact_created,
            {
                "artifact_id": str(artifact_obj.id),
                "filename": artifact_obj.filename,
                "mime_type": artifact_obj.mime_type,
                "size_bytes": artifact_obj.size_bytes,
                "storage_path": artifact_obj.storage_path,
            },
        )
        return artifact_obj

    def _find_artifact_by_path(
        self,
        task_id: uuid.UUID,
        path: Path,
    ) -> Artifact | None:
        storage_path = str(path)
        artifact = self.session.scalar(
            select(Artifact)
            .where(
                Artifact.task_id == task_id,
                Artifact.storage_path == storage_path,
            )
            .order_by(Artifact.created_at.desc())
            .limit(1)
        )
        if artifact is not None:
            return artifact

        return self.session.scalar(
            select(Artifact)
            .where(
                Artifact.task_id == task_id,
                Artifact.filename == path.name,
            )
            .order_by(Artifact.created_at.desc())
            .limit(1)
        )


def _post_thread_ts(thread: SlackThread) -> str | None:
    # Slack direct-message channels are linear by default. Posting with thread_ts
    # creates a hidden DM thread, which makes Kortny feel like it replied twice.
    if thread.channel_id.startswith("D"):
        return None
    return thread.thread_ts


def _response_ts(response: Mapping[str, Any]) -> str | None:
    ts = response.get("ts")
    if isinstance(ts, str) and ts:
        return ts
    message = response.get("message")
    if isinstance(message, Mapping):
        message_ts = message.get("ts")
        if isinstance(message_ts, str) and message_ts:
            return message_ts
    return None


def _response_file_id(response: Mapping[str, Any]) -> str | None:
    file_obj = response.get("file")
    if isinstance(file_obj, Mapping):
        file_id = file_obj.get("id")
        if isinstance(file_id, str) and file_id:
            return file_id

    files = response.get("files")
    if isinstance(files, list) and files:
        first_file = files[0]
        if isinstance(first_file, Mapping):
            file_id = first_file.get("id")
            if isinstance(file_id, str) and file_id:
                return file_id

    return None
