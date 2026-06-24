"""Slack message and file posting for task results."""

from __future__ import annotations

import logging
import mimetypes
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session

from kortny.db.models import Artifact, Task, TaskEvent, TaskEventType
from kortny.observability import set_span_attributes, start_span
from kortny.slack.egress import scan_outbound_urls
from kortny.slack.formatting import normalize_user_facing_text
from kortny.slack.outbox import (
    SlackSideEffectOutbox,
    slack_file_upload_key,
    slack_message_key,
)
from kortny.tasks import TaskService

logger = logging.getLogger(__name__)

# HIG-169 P0.1: outbound posts never trigger Slack's unfurl preview fetch, the
# exfiltration vector behind the Slack AI / Anthropic Slack MCP incidents.
EGRESS_URL_FLAGGED_MESSAGE = "egress_url_flagged"


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
        blocks: list[dict[str, Any]] | None = None,
        unfurl_links: bool = True,
        unfurl_media: bool = True,
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
    thread_ts: str | None
    task_id: uuid.UUID | None = None
    is_assistant: bool = False

    @classmethod
    def from_task(cls, task: Task) -> SlackThread:
        thread_ts = task.slack_thread_ts or task.slack_message_ts
        if not thread_ts and not _allows_root_delivery(task):
            raise ValueError("Task has no Slack thread timestamp")
        return cls(
            channel_id=task.slack_channel_id,
            thread_ts=thread_ts,
            task_id=task.id,
            is_assistant=_is_assistant_task(task),
        )


class SlackPoster:
    """Posts task results and artifacts back to Slack."""

    def __init__(
        self,
        *,
        session: Session,
        client: SlackPostingClient,
        task_service: TaskService | None = None,
        egress_url_allowlist: frozenset[str] = frozenset(),
    ) -> None:
        self.session = session
        self.client = client
        self.task_service = task_service or TaskService(session)
        self.egress_url_allowlist = egress_url_allowlist

    def post_message(
        self,
        thread: SlackThread,
        text: str,
        *,
        purpose: str = "result",
        blocks: list[dict[str, Any]] | None = None,
        idempotency_purpose: str | None = None,
    ) -> str:
        """Post text into a Slack thread and return the Slack message ts.

        ``idempotency_purpose`` overrides the outbox idempotency key without
        changing the recorded ``purpose``. This lets a single task post several
        distinct messages of the same purpose (e.g. one approval prompt per
        approval) without the outbox deduping the later ones against the first,
        while reaction/purpose-based lookups still match on ``purpose`` (HIG-248).
        """

        post_thread_ts = _post_thread_ts(thread)
        slack_text = normalize_user_facing_text(text)
        slack_blocks = _normalize_blocks(blocks)
        self._flag_egress_urls(thread, slack_text, purpose=purpose)
        if thread.channel_id == "playground":
            import uuid

            message_ts = f"1710000000.{uuid.uuid4().hex[:6]}"
            if thread.task_id is not None:
                self.task_service.append_event(
                    thread.task_id,
                    TaskEventType.message_posted,
                    {
                        "channel": thread.channel_id,
                        "thread_ts": post_thread_ts,
                        "message_ts": message_ts,
                        "text": slack_text,
                        "purpose": purpose,
                        "slack_side_effect_id": "dummy",
                        "idempotency_key": "dummy",
                    },
                )
            return message_ts
        with start_span(
            "slack.post_message",
            attributes={
                "kortny.task.id": thread.task_id,
                "slack.channel_id": thread.channel_id,
                "slack.thread_ts": post_thread_ts,
                "slack.message_purpose": purpose,
                "slack.text_chars": len(slack_text),
            },
        ):
            side_effect_id = None
            idempotency_key = None
            deduped = False
            if thread.task_id is None:
                response = self.client.chat_postMessage(
                    channel=thread.channel_id,
                    text=slack_text,
                    thread_ts=post_thread_ts,
                    blocks=slack_blocks,
                    unfurl_links=False,
                    unfurl_media=False,
                )
            else:
                task = self._resolve_task(thread.task_id)
                idempotency_key = slack_message_key(
                    task.id, idempotency_purpose or purpose
                )
                request: dict[str, Any] = {
                    "channel": thread.channel_id,
                    "text": slack_text,
                    "thread_ts": post_thread_ts,
                }
                if slack_blocks is not None:
                    request["blocks"] = slack_blocks
                result = SlackSideEffectOutbox(self.session).deliver(
                    installation_id=task.installation_id,
                    task_id=task.id,
                    idempotency_key=idempotency_key,
                    operation="chat_postMessage",
                    purpose=purpose,
                    target_channel_id=thread.channel_id,
                    target_thread_ts=post_thread_ts,
                    request=request,
                    call=lambda: self.client.chat_postMessage(
                        channel=thread.channel_id,
                        text=slack_text,
                        thread_ts=post_thread_ts,
                        blocks=slack_blocks,
                        unfurl_links=False,
                        unfurl_media=False,
                    ),
                )
                response = result.response
                side_effect_id = str(result.side_effect.id)
                deduped = result.deduped
            posted_message_ts = _response_ts(response)
            if posted_message_ts is None:
                raise SlackPostingError("Slack chat_postMessage response is missing ts")
            set_span_attributes(
                {
                    "slack.posted_message_ts": posted_message_ts,
                    "slack.side_effect_id": side_effect_id,
                    "slack.side_effect_deduped": deduped,
                }
            )

        if thread.task_id is not None and not self._message_event_exists(
            task_id=thread.task_id,
            side_effect_id=side_effect_id,
        ):
            event_payload: dict[str, Any] = {
                "channel": thread.channel_id,
                "thread_ts": post_thread_ts,
                "message_ts": posted_message_ts,
                "text": slack_text,
                "purpose": purpose,
                "slack_side_effect_id": side_effect_id,
                "idempotency_key": idempotency_key,
            }
            if slack_blocks is not None:
                event_payload["blocks"] = slack_blocks
            self.task_service.append_event(
                thread.task_id,
                TaskEventType.message_posted,
                event_payload,
            )
        return posted_message_ts

    def clear_assistant_status(self, thread: SlackThread) -> None:
        """Clear the assistant loading status (``setStatus`` with empty string).

        Belt-and-suspenders for the assistant pane: ``stopStream`` already ends
        the live status, but an explicit empty-string ``setStatus`` guarantees
        the indicator clears even on the fallback post path or after the worker's
        out-of-band reply. Never raises — a status-clear failure must not fail
        the task.
        """

        if not thread.is_assistant or thread.thread_ts is None:
            return
        setter = getattr(self.client, "assistant_threads_setStatus", None)
        if not callable(setter):
            return
        try:
            # Clear the status with an empty status string. Do NOT pass
            # loading_messages=[] — Slack rejects an empty list
            # (invalid_arguments); the empty status clears the whole indicator,
            # loading messages included.
            setter(
                channel_id=thread.channel_id,
                thread_ts=thread.thread_ts,
                status="",
            )
        except Exception:
            logger.warning(
                "failed to clear assistant status channel=%s thread_ts=%s",
                thread.channel_id,
                thread.thread_ts,
                exc_info=True,
            )

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
        task = self._resolve_task(task_id)

        artifact_obj = self._resolve_or_create_artifact(thread, file_path, artifact)
        if thread.channel_id == "playground":
            import uuid

            slack_file_id = f"F_dummy_{uuid.uuid4().hex[:8]}"
            artifact_obj.slack_file_id = slack_file_id
            artifact_obj.posted_at = now or datetime.now(UTC)
            self.session.flush()
            self.task_service.append_event(
                task_id,
                TaskEventType.message_posted,
                {
                    "channel": thread.channel_id,
                    "thread_ts": _post_thread_ts(thread),
                    "slack_file_id": slack_file_id,
                    "artifact_id": str(artifact_obj.id),
                    "filename": artifact_obj.filename,
                    "purpose": "file_upload",
                    "slack_side_effect_id": "dummy",
                    "idempotency_key": "dummy",
                },
            )
            return slack_file_id

        if artifact_obj.posted_at is not None:
            if artifact_obj.slack_file_id:
                return artifact_obj.slack_file_id
            raise SlackPostingError("Artifact is posted but missing slack_file_id")

        post_thread_ts = _post_thread_ts(thread)
        slack_initial_comment = (
            normalize_user_facing_text(initial_comment)
            if initial_comment is not None
            else None
        )
        with start_span(
            "slack.upload_file",
            attributes={
                "kortny.task.id": task_id,
                "kortny.artifact.id": artifact_obj.id,
                "slack.channel_id": thread.channel_id,
                "slack.thread_ts": post_thread_ts,
                "file.name": file_path.name,
                "file.size_bytes": file_path.stat().st_size,
                "file.initial_comment_chars": len(slack_initial_comment or ""),
            },
        ):
            idempotency_key = slack_file_upload_key(artifact_obj.id)
            result = SlackSideEffectOutbox(self.session).deliver(
                installation_id=task.installation_id,
                task_id=task_id,
                idempotency_key=idempotency_key,
                operation="files_upload_v2",
                purpose="file_upload",
                target_channel_id=thread.channel_id,
                target_thread_ts=post_thread_ts,
                request={
                    "file": str(file_path),
                    "filename": file_path.name,
                    "title": title or file_path.name,
                    "channel": thread.channel_id,
                    "initial_comment": slack_initial_comment,
                    "thread_ts": post_thread_ts,
                    "artifact_id": str(artifact_obj.id),
                },
                call=lambda: self.client.files_upload_v2(
                    file=str(file_path),
                    filename=file_path.name,
                    title=title or file_path.name,
                    channel=thread.channel_id,
                    initial_comment=slack_initial_comment,
                    thread_ts=post_thread_ts,
                ),
            )
            response = result.response
            uploaded_file_id = _response_file_id(response)
            if uploaded_file_id is None:
                raise SlackPostingError(
                    "Slack files_upload_v2 response is missing file id"
                )
            set_span_attributes(
                {
                    "slack.file_id": uploaded_file_id,
                    "slack.side_effect_id": str(result.side_effect.id),
                    "slack.side_effect_deduped": result.deduped,
                }
            )

        artifact_obj.slack_file_id = uploaded_file_id
        artifact_obj.posted_at = now or datetime.now(UTC)
        self.session.flush()

        if not self._message_event_exists(
            task_id=task_id,
            side_effect_id=str(result.side_effect.id),
        ):
            self.task_service.append_event(
                task_id,
                TaskEventType.message_posted,
                {
                    "channel": thread.channel_id,
                    "thread_ts": post_thread_ts,
                    "slack_file_id": uploaded_file_id,
                    "artifact_id": str(artifact_obj.id),
                    "filename": artifact_obj.filename,
                    "purpose": "file_upload",
                    "slack_side_effect_id": str(result.side_effect.id),
                    "idempotency_key": idempotency_key,
                },
            )
        return uploaded_file_id

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

    def _flag_egress_urls(
        self,
        thread: SlackThread,
        slack_text: str,
        *,
        purpose: str,
    ) -> None:
        """Flag + log outbound URLs that look like exfiltration payloads.

        Observability only — never blocks the post. The unfurl-off flags on the
        actual ``chat_postMessage`` call are the real exfil fix; this records
        which non-allowlisted hosts an outbound URL with a suspicious query
        string is pointing at, so an operator can audit attempted exfiltration.
        """

        flagged = scan_outbound_urls(
            slack_text,
            allowlist=self.egress_url_allowlist,
        )
        if not flagged:
            return
        for entry in flagged:
            logger.debug(
                "egress url flagged host=%s url_chars=%s longest_value=%s "
                "channel=%s purpose=%s",
                entry.host,
                len(entry.url),
                entry.longest_value_len,
                thread.channel_id,
                purpose,
            )
        if thread.task_id is None:
            return
        self.task_service.append_event(
            thread.task_id,
            TaskEventType.log,
            {
                "message": EGRESS_URL_FLAGGED_MESSAGE,
                "purpose": purpose,
                "channel": thread.channel_id,
                "flagged": [
                    {
                        "host": entry.host,
                        "url_chars": len(entry.url),
                        "longest_query_value_len": entry.longest_value_len,
                    }
                    for entry in flagged
                ],
            },
        )

    def _resolve_task(self, task_id: uuid.UUID) -> Task:
        task = self.session.scalar(select(Task).where(Task.id == task_id))
        if task is None:
            raise LookupError(f"Task not found: {task_id}")
        return task

    def _message_event_exists(
        self,
        *,
        task_id: uuid.UUID,
        side_effect_id: str | None,
    ) -> bool:
        if side_effect_id is None:
            return False
        return (
            self.session.scalar(
                select(TaskEvent.id)
                .where(
                    TaskEvent.task_id == task_id,
                    TaskEvent.type == TaskEventType.message_posted,
                    TaskEvent.payload["slack_side_effect_id"].as_string()
                    == side_effect_id,
                )
                .limit(1)
            )
            is not None
        )


def _post_thread_ts(thread: SlackThread) -> str | None:
    # Assistant ("Agents & AI Apps") threads are DM ("D") channels but they are
    # genuinely threaded: the live agent pane is bound to the assistant thread
    # root, and Slack only renders the reply (and auto-clears the loading status)
    # when the app posts WITH that thread_ts. So assistant replies must stay
    # threaded even though the channel starts with "D" (HIG-247).
    if thread.is_assistant:
        return thread.thread_ts
    # Non-assistant direct-message channels are linear by default. Posting with
    # thread_ts creates a hidden DM thread, which makes Kortny feel like it
    # replied twice.
    if thread.channel_id.startswith("D"):
        return None
    return thread.thread_ts


def _is_assistant_task(task: Task) -> bool:
    payload = task.identity_payload if isinstance(task.identity_payload, dict) else {}
    return payload.get("source_surface") == "assistant"


def _normalize_blocks(
    blocks: list[dict[str, Any]] | None,
) -> list[dict[str, Any]] | None:
    if blocks is None:
        return None
    normalized = _normalize_block_value(blocks)
    if not isinstance(normalized, list):
        return None
    return [item for item in normalized if isinstance(item, dict)]


def _normalize_block_value(value: Any, *, key: str | None = None) -> Any:
    if isinstance(value, str) and key in {"text", "alt_text"}:
        return normalize_user_facing_text(value)
    if isinstance(value, list):
        return [_normalize_block_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(item_key): _normalize_block_value(item_value, key=str(item_key))
            for item_key, item_value in value.items()
        }
    return value


def _allows_root_delivery(task: Task) -> bool:
    if task.identity_kind == "scheduled":
        return True
    payload = task.identity_payload if isinstance(task.identity_payload, dict) else {}
    return payload.get("source") == "witness_autopilot"


def _response_mapping(response: Any) -> Mapping[str, Any]:
    if isinstance(response, Mapping):
        return response
    data = getattr(response, "data", None)
    if isinstance(data, Mapping):
        return data
    to_dict = getattr(response, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, Mapping):
            return payload
    return {}


def _response_ts(response: Any) -> str | None:
    payload = _response_mapping(response)
    ts = payload.get("ts")
    if isinstance(ts, str) and ts:
        return ts
    message = payload.get("message")
    if isinstance(message, Mapping):
        message_ts = message.get("ts")
        if isinstance(message_ts, str) and message_ts:
            return message_ts
    return None


def _response_file_id(response: Any) -> str | None:
    payload = _response_mapping(response)
    file_obj = payload.get("file")
    if isinstance(file_obj, Mapping):
        file_id = file_obj.get("id")
        if isinstance(file_id, str) and file_id:
            return file_id

    files = payload.get("files")
    if isinstance(files, list) and files:
        first_file = files[0]
        if isinstance(first_file, Mapping):
            file_id = first_file.get("id")
            if isinstance(file_id, str) and file_id:
                return file_id

    return None
