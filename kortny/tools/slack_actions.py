"""Slack-native low-risk action tools."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from typing import Any, Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session

from kortny.db.models import Task, TaskEvent, TaskEventType
from kortny.slack.outbox import (
    SlackSideEffectOutbox,
    slack_bookmark_key,
    slack_canvas_edit_key,
    slack_channel_canvas_key,
    slack_pin_key,
    slack_reaction_key,
)
from kortny.slack_mrkdwn import normalize_user_facing_text
from kortny.tasks import TaskService
from kortny.tools.types import JsonObject, JsonSchema, ToolResult

MAX_TOOL_REPLY_CHARS = 4_000
MAX_BOOKMARK_TITLE_CHARS = 150
MAX_CANVAS_TITLE_CHARS = 150
MAX_CANVAS_MARKDOWN_CHARS = 12_000
MAX_CANVAS_LOOKUP_TEXT_CHARS = 200
SLACK_TS_RE = re.compile(r"^\d{10,}\.\d+$")
REACTION_NAME_RE = re.compile(r"^[A-Za-z0-9_+-]+(?:::skin-tone-[2-6])?$")
HTTP_LINK_RE = re.compile(r"^https?://[^\s]+$", re.IGNORECASE)
CANVAS_ID_RE = re.compile(r"^[A-Za-z0-9]{4,}$")
CANVAS_SECTION_ID_RE = re.compile(r"^[A-Za-z0-9:_-]{6,}$")
CANVAS_SECTION_TYPES = frozenset({"h1", "h2", "h3", "any_header"})
CANVAS_CONTENT_OPERATIONS = frozenset(
    {"insert_at_start", "insert_at_end", "insert_before", "insert_after", "replace"}
)


class SlackActionClient(Protocol):
    """Subset of Slack WebClient used by native Slack action tools."""

    def chat_postMessage(
        self,
        *,
        channel: str,
        text: str,
        thread_ts: str | None = None,
    ) -> Any:
        """Post a Slack message."""

    def reactions_add(
        self,
        *,
        channel: str,
        timestamp: str,
        name: str,
    ) -> Any:
        """Add a Slack reaction."""

    def pins_add(
        self,
        *,
        channel: str,
        timestamp: str,
    ) -> Any:
        """Pin a Slack message."""

    def bookmarks_add(
        self,
        *,
        channel_id: str,
        title: str,
        type: str,
        link: str,
        emoji: str | None = None,
    ) -> Any:
        """Add a Slack bookmark."""

    def conversations_canvases_create(
        self,
        *,
        channel_id: str,
        document_content: dict[str, str],
        title: str | None = None,
    ) -> Any:
        """Create a Slack channel canvas."""

    def conversations_info(
        self,
        *,
        channel: str,
    ) -> Any:
        """Fetch Slack channel info (includes the channel canvas file id)."""

    def canvases_edit(
        self,
        *,
        canvas_id: str,
        changes: list[dict[str, Any]],
    ) -> Any:
        """Edit a Slack canvas."""

    def canvases_sections_lookup(
        self,
        *,
        canvas_id: str,
        criteria: dict[str, Any],
    ) -> Any:
        """Look up Slack canvas sections."""


class SlackReplyThreadTool:
    """Post a Slack reply in the current task thread."""

    name = "slack_reply_thread"
    description = (
        "Posts a Slack message into the current task's DM or active thread. "
        "Use this only when the user explicitly asks Kortny to post, reply, "
        "send, or leave a visible Slack message before the final answer. Do "
        "not use it for ordinary final answers because Kortny's final response "
        "is posted automatically. The tool is scoped to the current Slack "
        "channel and thread."
    )
    parameters: JsonSchema = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Slack mrkdwn text to post into the current thread or DM.",
            },
            "channel_id": {
                "type": "string",
                "description": (
                    "Optional current Slack channel ID. Omit unless copying the "
                    "current task channel exactly."
                ),
            },
            "thread_ts": {
                "type": "string",
                "description": (
                    "Optional current parent thread timestamp. Omit to use the "
                    "task thread. This cannot target another thread."
                ),
            },
            "purpose": {
                "type": "string",
                "description": (
                    "Optional short purpose for trace labels, such as follow_up "
                    "or status_update."
                ),
            },
        },
        "required": ["text"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        *,
        client: SlackActionClient,
        session: Session,
        task: Task,
        task_service: TaskService | None = None,
    ) -> None:
        self.client = client
        self.session = session
        self.task = task
        self.task_service = task_service or TaskService(session)

    def invoke(self, args: JsonObject) -> ToolResult:
        text = _coerce_reply_text(args.get("text"))
        channel_id = _current_channel_arg(
            value=args.get("channel_id"),
            task=self.task,
            tool_name=self.name,
        )
        thread_ts = _current_thread_arg(args.get("thread_ts"), task=self.task)
        purpose = _coerce_purpose(args.get("purpose"), default="tool_reply_thread")
        post_thread_ts = _post_thread_ts(channel_id=channel_id, thread_ts=thread_ts)
        idempotency_key = _reply_idempotency_key(
            task=self.task,
            purpose=purpose,
            channel_id=channel_id,
            thread_ts=post_thread_ts,
            text=text,
        )
        request: JsonObject = {
            "channel": channel_id,
            "text": text,
            "thread_ts": post_thread_ts,
        }
        result = SlackSideEffectOutbox(self.session).deliver(
            installation_id=self.task.installation_id,
            task_id=self.task.id,
            idempotency_key=idempotency_key,
            operation="chat_postMessage",
            purpose=purpose,
            target_channel_id=channel_id,
            target_thread_ts=post_thread_ts,
            request=request,
            call=lambda: self.client.chat_postMessage(
                channel=channel_id,
                text=text,
                thread_ts=post_thread_ts,
            ),
        )
        response = _require_ok(result.response, "chat.postMessage")
        message_ts = _response_ts(response)
        if message_ts is None:
            raise RuntimeError("Slack chat.postMessage response is missing ts")
        side_effect_id = str(result.side_effect.id)
        if not _task_event_exists(
            self.session,
            task=self.task,
            event_type=TaskEventType.message_posted,
            side_effect_id=side_effect_id,
        ):
            self.task_service.append_event(
                self.task,
                TaskEventType.message_posted,
                {
                    "channel": channel_id,
                    "thread_ts": post_thread_ts,
                    "message_ts": message_ts,
                    "text": text,
                    "purpose": purpose,
                    "slack_side_effect_id": side_effect_id,
                    "idempotency_key": idempotency_key,
                    "tool": self.name,
                    "deduped": result.deduped,
                },
            )
        return ToolResult(
            output={
                "successful": True,
                "channel": channel_id,
                "thread_ts": post_thread_ts,
                "message_ts": message_ts,
                "text_chars": len(text),
                "purpose": purpose,
                "deduped": result.deduped,
                "slack_side_effect_id": side_effect_id,
            }
        )


class SlackAddReactionTool:
    """Add a reaction to the current triggering Slack message."""

    name = "slack_add_reaction"
    description = (
        "Adds an emoji reaction to the current triggering Slack message. Use "
        "this for lightweight acknowledgements, status markers, or explicit "
        "user requests to react. The tool is scoped to the current Slack "
        "channel and triggering message by default."
    )
    parameters: JsonSchema = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": (
                    "Emoji reaction name without surrounding colons, for "
                    "example eyes, white_check_mark, or thumbsup."
                ),
            },
            "channel_id": {
                "type": "string",
                "description": (
                    "Optional current Slack channel ID. Omit unless copying the "
                    "current task channel exactly."
                ),
            },
            "message_ts": {
                "type": "string",
                "description": (
                    "Optional current triggering message timestamp. Omit to use "
                    "the task message timestamp. This cannot target another "
                    "message."
                ),
            },
        },
        "required": ["name"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        *,
        client: SlackActionClient,
        session: Session,
        task: Task,
        task_service: TaskService | None = None,
    ) -> None:
        self.client = client
        self.session = session
        self.task = task
        self.task_service = task_service or TaskService(session)

    def invoke(self, args: JsonObject) -> ToolResult:
        reaction = _coerce_reaction_name(args.get("name"))
        channel_id = _current_channel_arg(
            value=args.get("channel_id"),
            task=self.task,
            tool_name=self.name,
        )
        message_ts = _current_message_ts_arg(
            args.get("message_ts"),
            task=self.task,
            tool_name=self.name,
        )
        idempotency_key = slack_reaction_key(
            task_id=self.task.id,
            operation="reactions_add",
            channel_id=channel_id,
            message_ts=message_ts,
            reaction=reaction,
        )
        request: JsonObject = {
            "channel": channel_id,
            "timestamp": message_ts,
            "name": reaction,
        }
        result = SlackSideEffectOutbox(self.session).deliver(
            installation_id=self.task.installation_id,
            task_id=self.task.id,
            idempotency_key=idempotency_key,
            operation="reactions_add",
            purpose="tool_reaction",
            target_channel_id=channel_id,
            target_message_ts=message_ts,
            request=request,
            call=lambda: self.client.reactions_add(
                channel=channel_id,
                timestamp=message_ts,
                name=reaction,
            ),
        )
        response = _require_ok(result.response, "reactions.add")
        side_effect_id = str(result.side_effect.id)
        if not _task_event_exists(
            self.session,
            task=self.task,
            event_type=TaskEventType.log,
            side_effect_id=side_effect_id,
        ):
            self.task_service.append_event(
                self.task,
                TaskEventType.log,
                {
                    "message": "slack_reaction_added",
                    "channel": channel_id,
                    "message_ts": message_ts,
                    "reaction": reaction,
                    "purpose": "tool_reaction",
                    "slack_side_effect_id": side_effect_id,
                    "idempotency_key": idempotency_key,
                    "tool": self.name,
                    "deduped": result.deduped,
                    "deduped_by_slack": response.get("deduped_by_slack") is True,
                },
            )
        return ToolResult(
            output={
                "successful": True,
                "channel": channel_id,
                "message_ts": message_ts,
                "reaction": reaction,
                "deduped": result.deduped,
                "deduped_by_slack": response.get("deduped_by_slack") is True,
                "slack_side_effect_id": side_effect_id,
            }
        )


class SlackPinMessageTool:
    """Pin the current triggering Slack message."""

    name = "slack_pin_message"
    description = (
        "Pins the current triggering Slack message in its channel. Use this "
        "only when the user explicitly asks to pin this message or when Kortny "
        "has just produced a message that should be kept visible and the user "
        "asked for it. The tool is scoped to the current Slack channel and "
        "current message."
    )
    parameters: JsonSchema = {
        "type": "object",
        "properties": {
            "channel_id": {
                "type": "string",
                "description": (
                    "Optional current Slack channel ID. Omit unless copying the "
                    "current task channel exactly."
                ),
            },
            "message_ts": {
                "type": "string",
                "description": (
                    "Optional current triggering message timestamp. Omit to use "
                    "the task message timestamp. This cannot target another "
                    "message."
                ),
            },
        },
        "additionalProperties": False,
    }

    def __init__(
        self,
        *,
        client: SlackActionClient,
        session: Session,
        task: Task,
        task_service: TaskService | None = None,
    ) -> None:
        self.client = client
        self.session = session
        self.task = task
        self.task_service = task_service or TaskService(session)

    def invoke(self, args: JsonObject) -> ToolResult:
        channel_id = _current_channel_arg(
            value=args.get("channel_id"),
            task=self.task,
            tool_name=self.name,
        )
        message_ts = _current_message_ts_arg(
            args.get("message_ts"),
            task=self.task,
            tool_name=self.name,
        )
        idempotency_key = slack_pin_key(
            task_id=self.task.id,
            channel_id=channel_id,
            message_ts=message_ts,
        )
        request: JsonObject = {
            "channel": channel_id,
            "timestamp": message_ts,
        }
        result = SlackSideEffectOutbox(self.session).deliver(
            installation_id=self.task.installation_id,
            task_id=self.task.id,
            idempotency_key=idempotency_key,
            operation="pins_add",
            purpose="tool_pin_message",
            target_channel_id=channel_id,
            target_message_ts=message_ts,
            request=request,
            call=lambda: self.client.pins_add(
                channel=channel_id,
                timestamp=message_ts,
            ),
        )
        response = _require_ok(result.response, "pins.add")
        side_effect_id = str(result.side_effect.id)
        if not _task_event_exists(
            self.session,
            task=self.task,
            event_type=TaskEventType.log,
            side_effect_id=side_effect_id,
        ):
            self.task_service.append_event(
                self.task,
                TaskEventType.log,
                {
                    "message": "slack_message_pinned",
                    "channel": channel_id,
                    "message_ts": message_ts,
                    "purpose": "tool_pin_message",
                    "slack_side_effect_id": side_effect_id,
                    "idempotency_key": idempotency_key,
                    "tool": self.name,
                    "deduped": result.deduped,
                    "deduped_by_slack": response.get("deduped_by_slack") is True,
                },
            )
        return ToolResult(
            output={
                "successful": True,
                "channel": channel_id,
                "message_ts": message_ts,
                "deduped": result.deduped,
                "deduped_by_slack": response.get("deduped_by_slack") is True,
                "slack_side_effect_id": side_effect_id,
            }
        )


class SlackAddBookmarkTool:
    """Add a link bookmark to the current Slack channel."""

    name = "slack_add_bookmark"
    description = (
        "Adds a link bookmark to the current Slack channel header. Use this "
        "only when the user explicitly asks to bookmark a link or save a link "
        "for the channel. This slice supports link bookmarks only and is scoped "
        "to the current Slack channel."
    )
    parameters: JsonSchema = {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Short human-readable bookmark title.",
            },
            "link": {
                "type": "string",
                "description": "HTTP or HTTPS link to bookmark.",
            },
            "emoji": {
                "type": "string",
                "description": (
                    "Optional emoji tag without surrounding colons, for example "
                    "bookmark, link, or white_check_mark."
                ),
            },
            "channel_id": {
                "type": "string",
                "description": (
                    "Optional current Slack channel ID. Omit unless copying the "
                    "current task channel exactly."
                ),
            },
        },
        "required": ["title", "link"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        *,
        client: SlackActionClient,
        session: Session,
        task: Task,
        task_service: TaskService | None = None,
    ) -> None:
        self.client = client
        self.session = session
        self.task = task
        self.task_service = task_service or TaskService(session)

    def invoke(self, args: JsonObject) -> ToolResult:
        channel_id = _current_channel_arg(
            value=args.get("channel_id"),
            task=self.task,
            tool_name=self.name,
        )
        if channel_id.startswith("D"):
            raise ValueError("slack_add_bookmark is only available in Slack channels")
        title = _coerce_bookmark_title(args.get("title"))
        link = _coerce_bookmark_link(args.get("link"))
        emoji = _coerce_optional_emoji(args.get("emoji"))
        digest = _bookmark_digest(title=title, link=link)
        idempotency_key = slack_bookmark_key(
            task_id=self.task.id,
            channel_id=channel_id,
            digest=digest,
        )
        request: JsonObject = {
            "channel_id": channel_id,
            "title": title,
            "type": "link",
            "link": link,
        }
        if emoji is not None:
            request["emoji"] = emoji
        result = SlackSideEffectOutbox(self.session).deliver(
            installation_id=self.task.installation_id,
            task_id=self.task.id,
            idempotency_key=idempotency_key,
            operation="bookmarks_add",
            purpose="tool_add_bookmark",
            target_channel_id=channel_id,
            request=request,
            call=lambda: self.client.bookmarks_add(
                channel_id=channel_id,
                title=title,
                type="link",
                link=link,
                emoji=emoji,
            ),
        )
        response = _require_ok(result.response, "bookmarks.add")
        bookmark_id = _response_bookmark_id(response)
        side_effect_id = str(result.side_effect.id)
        if not _task_event_exists(
            self.session,
            task=self.task,
            event_type=TaskEventType.log,
            side_effect_id=side_effect_id,
        ):
            self.task_service.append_event(
                self.task,
                TaskEventType.log,
                {
                    "message": "slack_bookmark_added",
                    "channel": channel_id,
                    "bookmark_id": bookmark_id,
                    "title": title,
                    "link": link,
                    "purpose": "tool_add_bookmark",
                    "slack_side_effect_id": side_effect_id,
                    "idempotency_key": idempotency_key,
                    "tool": self.name,
                    "deduped": result.deduped,
                },
            )
        return ToolResult(
            output={
                "successful": True,
                "channel": channel_id,
                "bookmark_id": bookmark_id,
                "title": title,
                "link": link,
                "deduped": result.deduped,
                "slack_side_effect_id": side_effect_id,
            }
        )


class SlackCreateChannelCanvasTool:
    """Create the current Slack channel's canvas."""

    name = "slack_create_channel_canvas"
    description = (
        "Creates the current Slack channel canvas with Markdown content. Use "
        "this when the user explicitly asks Kortny to create or set up a "
        "channel canvas, project canvas, team notes canvas, or channel hub. "
        "This is scoped to the current Slack channel and is not available in DMs."
    )
    parameters: JsonSchema = {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Short title for the channel canvas.",
            },
            "markdown": {
                "type": "string",
                "description": (
                    "Markdown content for the canvas. Canvas markdown supports "
                    "headings, lists, checkboxes, links, tables, and mentions."
                ),
            },
            "channel_id": {
                "type": "string",
                "description": (
                    "Optional current Slack channel ID. Omit unless copying the "
                    "current task channel exactly."
                ),
            },
        },
        "required": ["title", "markdown"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        *,
        client: SlackActionClient,
        session: Session,
        task: Task,
        task_service: TaskService | None = None,
    ) -> None:
        self.client = client
        self.session = session
        self.task = task
        self.task_service = task_service or TaskService(session)

    def invoke(self, args: JsonObject) -> ToolResult:
        channel_id = _current_channel_arg(
            value=args.get("channel_id"),
            task=self.task,
            tool_name=self.name,
        )
        if channel_id.startswith("D"):
            raise ValueError(
                "slack_create_channel_canvas is only available in Slack channels"
            )
        title = _coerce_canvas_title(args.get("title"), tool_name=self.name)
        markdown = _coerce_canvas_markdown(args.get("markdown"), tool_name=self.name)
        document_content = _canvas_document_content(markdown)
        digest = _canvas_digest(title=title, markdown=markdown)
        idempotency_key = slack_channel_canvas_key(
            task_id=self.task.id,
            channel_id=channel_id,
            digest=digest,
        )
        request: JsonObject = {
            "channel_id": channel_id,
            "title": title,
            "document_content": document_content,
        }
        result = SlackSideEffectOutbox(self.session).deliver(
            installation_id=self.task.installation_id,
            task_id=self.task.id,
            idempotency_key=idempotency_key,
            operation="conversations_canvases_create",
            purpose="tool_create_channel_canvas",
            target_channel_id=channel_id,
            request=request,
            call=lambda: self.client.conversations_canvases_create(
                channel_id=channel_id,
                title=title,
                document_content=document_content,
            ),
        )
        response = _require_ok(result.response, "conversations.canvases.create")
        canvas_id = _response_canvas_id(response)
        if canvas_id is None:
            raise RuntimeError(
                "Slack conversations.canvases.create response is missing canvas_id"
            )
        side_effect_id = str(result.side_effect.id)
        if not _task_event_exists(
            self.session,
            task=self.task,
            event_type=TaskEventType.log,
            side_effect_id=side_effect_id,
        ):
            self.task_service.append_event(
                self.task,
                TaskEventType.log,
                {
                    "message": "slack_channel_canvas_created",
                    "channel": channel_id,
                    "canvas_id": canvas_id,
                    "title": title,
                    "markdown_chars": len(markdown),
                    "purpose": "tool_create_channel_canvas",
                    "slack_side_effect_id": side_effect_id,
                    "idempotency_key": idempotency_key,
                    "tool": self.name,
                    "deduped": result.deduped,
                },
            )
        return ToolResult(
            output={
                "successful": True,
                "channel": channel_id,
                "canvas_id": canvas_id,
                "title": title,
                "markdown_chars": len(markdown),
                "deduped": result.deduped,
                "slack_side_effect_id": side_effect_id,
            }
        )


class SlackLookupCanvasSectionsTool:
    """Find sections in a known Slack canvas."""

    name = "slack_lookup_canvas_sections"
    description = (
        "Finds section IDs in a Slack canvas by heading type and/or text. "
        "Use this before slack_edit_canvas when the user asks to update a "
        "specific section, insert near a section, or replace a section but only "
        "the section heading or text is known. This is read-only. When "
        "canvas_id is omitted it targets the current channel's canvas "
        "automatically."
    )
    parameters: JsonSchema = {
        "type": "object",
        "properties": {
            "canvas_id": {
                "type": "string",
                "description": (
                    "Slack canvas ID, such as F1234ABCD. Omit to target the "
                    "current channel's canvas."
                ),
            },
            "contains_text": {
                "type": "string",
                "description": (
                    "Text that the target section contains, usually the heading "
                    "or a distinctive phrase."
                ),
            },
            "section_types": {
                "type": "array",
                "description": (
                    "Heading section types to search. Use any_header when the "
                    "heading level is unknown."
                ),
                "items": {
                    "type": "string",
                    "enum": ["h1", "h2", "h3", "any_header"],
                },
            },
        },
        "required": [],
        "additionalProperties": False,
    }

    def __init__(self, *, client: SlackActionClient, task: Task) -> None:
        self.client = client
        self.task = task

    def invoke(self, args: JsonObject) -> ToolResult:
        canvas_id = _resolve_canvas_id(
            args.get("canvas_id"),
            client=self.client,
            task=self.task,
            tool_name=self.name,
        )
        criteria = _coerce_canvas_lookup_criteria(args, tool_name=self.name)
        response = _require_ok(
            self.client.canvases_sections_lookup(
                canvas_id=canvas_id,
                criteria=criteria,
            ),
            "canvases.sections.lookup",
        )
        sections = _response_canvas_sections(response)
        section_ids = [
            section["section_id"]
            for section in sections
            if isinstance(section.get("section_id"), str)
        ]
        return ToolResult(
            output={
                "successful": True,
                "canvas_id": canvas_id,
                "criteria": criteria,
                "section_count": len(sections),
                "section_ids": section_ids,
                "sections": sections,
            }
        )


class SlackEditCanvasTool:
    """Edit an existing Slack canvas."""

    name = "slack_edit_canvas"
    description = (
        "Edits a Slack canvas by appending, inserting, replacing, or "
        "renaming content. Use this when the user asks Kortny to update a "
        "canvas. When canvas_id is omitted it targets the current channel's "
        "canvas automatically — the usual case for 'the canvas' in a channel. "
        "This tool performs one canvas edit per call."
    )
    parameters: JsonSchema = {
        "type": "object",
        "properties": {
            "canvas_id": {
                "type": "string",
                "description": (
                    "Slack canvas ID, such as F1234ABCD. Omit to target the "
                    "current channel's canvas."
                ),
            },
            "operation": {
                "type": "string",
                "description": (
                    "Canvas edit operation: insert_at_end, insert_at_start, "
                    "insert_before, insert_after, replace, or rename."
                ),
                "enum": [
                    "insert_at_end",
                    "insert_at_start",
                    "insert_before",
                    "insert_after",
                    "replace",
                    "rename",
                ],
            },
            "markdown": {
                "type": "string",
                "description": (
                    "Markdown content for content operations. Required unless "
                    "operation is rename."
                ),
            },
            "title": {
                "type": "string",
                "description": "New canvas title. Required for rename.",
            },
            "section_id": {
                "type": "string",
                "description": (
                    "Section ID required for insert_before, insert_after, and "
                    "section-level replace."
                ),
            },
        },
        "required": ["operation"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        *,
        client: SlackActionClient,
        session: Session,
        task: Task,
        task_service: TaskService | None = None,
    ) -> None:
        self.client = client
        self.session = session
        self.task = task
        self.task_service = task_service or TaskService(session)

    def invoke(self, args: JsonObject) -> ToolResult:
        canvas_id = _resolve_canvas_id(
            args.get("canvas_id"),
            client=self.client,
            task=self.task,
            tool_name=self.name,
        )
        operation = _coerce_canvas_operation(args.get("operation"), tool_name=self.name)
        change = _canvas_change_from_args(
            args, operation=operation, tool_name=self.name
        )
        digest = _canvas_edit_digest(canvas_id=canvas_id, change=change)
        idempotency_key = slack_canvas_edit_key(
            task_id=self.task.id,
            canvas_id=canvas_id,
            digest=digest,
        )
        request: JsonObject = {
            "canvas_id": canvas_id,
            "changes": [change],
        }
        result = SlackSideEffectOutbox(self.session).deliver(
            installation_id=self.task.installation_id,
            task_id=self.task.id,
            idempotency_key=idempotency_key,
            operation="canvases_edit",
            purpose="tool_edit_canvas",
            request=request,
            call=lambda: self.client.canvases_edit(
                canvas_id=canvas_id,
                changes=[change],
            ),
        )
        response = _require_ok(result.response, "canvases.edit")
        side_effect_id = str(result.side_effect.id)
        if not _task_event_exists(
            self.session,
            task=self.task,
            event_type=TaskEventType.log,
            side_effect_id=side_effect_id,
        ):
            self.task_service.append_event(
                self.task,
                TaskEventType.log,
                {
                    "message": "slack_canvas_edited",
                    "canvas_id": canvas_id,
                    "operation": operation,
                    "section_id": change.get("section_id"),
                    "purpose": "tool_edit_canvas",
                    "slack_side_effect_id": side_effect_id,
                    "idempotency_key": idempotency_key,
                    "tool": self.name,
                    "deduped": result.deduped,
                    "ok": response.get("ok") is not False,
                },
            )
        return ToolResult(
            output={
                "successful": True,
                "canvas_id": canvas_id,
                "operation": operation,
                "section_id": change.get("section_id"),
                "deduped": result.deduped,
                "slack_side_effect_id": side_effect_id,
            }
        )


def _coerce_reply_text(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("slack_reply_thread 'text' is required")
    text = normalize_user_facing_text(value)
    if not text:
        raise ValueError("slack_reply_thread 'text' cannot be empty")
    if len(text) > MAX_TOOL_REPLY_CHARS:
        raise ValueError(
            f"slack_reply_thread 'text' must be {MAX_TOOL_REPLY_CHARS} characters or fewer"
        )
    return text


def _current_channel_arg(*, value: object, task: Task, tool_name: str) -> str:
    if value is None:
        return task.slack_channel_id
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{tool_name} 'channel_id' must be a Slack channel ID")
    channel_id = value.strip()
    if channel_id != task.slack_channel_id:
        raise ValueError(f"{tool_name} can only target the current Slack channel")
    return channel_id


def _current_thread_arg(value: object, *, task: Task) -> str | None:
    current_thread_ts = task.slack_thread_ts or task.slack_message_ts
    if value is None:
        return current_thread_ts
    if not isinstance(value, str) or not value.strip():
        raise ValueError("slack_reply_thread 'thread_ts' must be a Slack timestamp")
    thread_ts = value.strip()
    allowed = {item for item in (task.slack_thread_ts, task.slack_message_ts) if item}
    if thread_ts not in allowed:
        raise ValueError("slack_reply_thread can only target the current Slack thread")
    return thread_ts


def _current_message_ts_arg(
    value: object,
    *,
    task: Task,
    tool_name: str,
) -> str:
    current_message_ts = task.slack_message_ts
    if current_message_ts is None and _is_slack_timestamp(task.slack_thread_ts):
        current_message_ts = task.slack_thread_ts
    if value is None:
        if current_message_ts is None:
            raise ValueError(f"{tool_name} requires a current message timestamp")
        return current_message_ts
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{tool_name} 'message_ts' must be a Slack timestamp")
    message_ts = value.strip()
    if not _is_slack_timestamp(message_ts):
        raise ValueError(f"{tool_name} 'message_ts' must be a Slack timestamp")
    if message_ts != current_message_ts:
        raise ValueError(f"{tool_name} can only target the current Slack message")
    return message_ts


def _coerce_reaction_name(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("slack_add_reaction 'name' is required")
    reaction = value.strip().strip(":")
    if not reaction or not REACTION_NAME_RE.match(reaction):
        raise ValueError("slack_add_reaction 'name' must be a valid Slack emoji name")
    return reaction


def _coerce_bookmark_title(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("slack_add_bookmark 'title' is required")
    title = " ".join(value.strip().split())
    if not title:
        raise ValueError("slack_add_bookmark 'title' cannot be empty")
    if len(title) > MAX_BOOKMARK_TITLE_CHARS:
        raise ValueError(
            f"slack_add_bookmark 'title' must be {MAX_BOOKMARK_TITLE_CHARS} characters or fewer"
        )
    return title


def _coerce_bookmark_link(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("slack_add_bookmark 'link' is required")
    link = value.strip()
    if not HTTP_LINK_RE.match(link):
        raise ValueError(
            "slack_add_bookmark 'link' must start with http:// or https://"
        )
    return link


def _coerce_optional_emoji(value: object) -> str | None:
    if value is None:
        return None
    return _coerce_reaction_name(value)


def _coerce_canvas_title(value: object, *, tool_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{tool_name} 'title' is required")
    title = " ".join(value.strip().split())
    if not title:
        raise ValueError(f"{tool_name} 'title' cannot be empty")
    if len(title) > MAX_CANVAS_TITLE_CHARS:
        raise ValueError(
            f"{tool_name} 'title' must be {MAX_CANVAS_TITLE_CHARS} characters or fewer"
        )
    return title


def _coerce_canvas_markdown(value: object, *, tool_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{tool_name} 'markdown' is required")
    markdown = value.strip()
    if not markdown:
        raise ValueError(f"{tool_name} 'markdown' cannot be empty")
    if len(markdown) > MAX_CANVAS_MARKDOWN_CHARS:
        raise ValueError(
            f"{tool_name} 'markdown' must be {MAX_CANVAS_MARKDOWN_CHARS} characters or fewer"
        )
    return markdown


def _resolve_canvas_id(
    value: object,
    *,
    client: SlackActionClient,
    task: Task,
    tool_name: str,
) -> str:
    """Return an explicit canvas_id, or the current channel's canvas.

    The channel canvas is discoverable via ``conversations.info`` ->
    ``channel.properties.canvas.file_id``, so "edit the canvas" in a channel
    must not require the model to dig the id out of prior context (it
    routinely can't — the id lives only in an old task's tool result).
    """

    if value is not None and (not isinstance(value, str) or value.strip()):
        return _coerce_canvas_id(value, tool_name=tool_name)
    response = _require_ok(
        client.conversations_info(channel=task.slack_channel_id),
        "conversations.info",
    )
    channel = response.get("channel")
    properties = channel.get("properties") if isinstance(channel, Mapping) else None
    canvas = properties.get("canvas") if isinstance(properties, Mapping) else None
    file_id = canvas.get("file_id") if isinstance(canvas, Mapping) else None
    if isinstance(file_id, str) and file_id:
        return file_id
    raise ValueError(
        f"{tool_name}: no 'canvas_id' was given and the current channel has "
        "no channel canvas. Pass the canvas_id explicitly (for a standalone "
        "canvas) or create the channel canvas first."
    )


def _coerce_canvas_id(value: object, *, tool_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{tool_name} 'canvas_id' is required")
    canvas_id = value.strip()
    if not CANVAS_ID_RE.match(canvas_id):
        raise ValueError(f"{tool_name} 'canvas_id' must be a Slack canvas ID")
    return canvas_id


def _coerce_canvas_section_id(value: object, *, tool_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{tool_name} 'section_id' is required for this operation")
    section_id = value.strip()
    if not CANVAS_SECTION_ID_RE.match(section_id):
        raise ValueError(f"{tool_name} 'section_id' must be a Slack canvas section ID")
    return section_id


def _coerce_canvas_lookup_text(value: object, *, tool_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{tool_name} 'contains_text' must be a string")
    text = " ".join(value.strip().split())
    if not text:
        raise ValueError(f"{tool_name} 'contains_text' cannot be empty")
    if len(text) > MAX_CANVAS_LOOKUP_TEXT_CHARS:
        raise ValueError(
            f"{tool_name} 'contains_text' must be {MAX_CANVAS_LOOKUP_TEXT_CHARS} characters or fewer"
        )
    return text


def _coerce_canvas_section_types(value: object, *, tool_name: str) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list | tuple):
        raise ValueError(f"{tool_name} 'section_types' must be a list")
    if not value:
        raise ValueError(f"{tool_name} 'section_types' cannot be empty")
    section_types: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{tool_name} 'section_types' entries must be strings")
        section_type = item.strip()
        if section_type not in CANVAS_SECTION_TYPES:
            allowed = ", ".join(sorted(CANVAS_SECTION_TYPES))
            raise ValueError(
                f"{tool_name} 'section_types' entries must be one of: {allowed}"
            )
        if section_type not in section_types:
            section_types.append(section_type)
    return section_types


def _coerce_canvas_lookup_criteria(args: JsonObject, *, tool_name: str) -> JsonObject:
    criteria: JsonObject = {}
    contains_text = _coerce_canvas_lookup_text(
        args.get("contains_text"),
        tool_name=tool_name,
    )
    section_types = _coerce_canvas_section_types(
        args.get("section_types"),
        tool_name=tool_name,
    )
    if contains_text is not None:
        criteria["contains_text"] = contains_text
    if section_types is not None:
        criteria["section_types"] = section_types
    if not criteria:
        raise ValueError(
            f"{tool_name} requires 'contains_text' or 'section_types' criteria"
        )
    return criteria


def _coerce_canvas_operation(value: object, *, tool_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{tool_name} 'operation' is required")
    operation = value.strip()
    allowed = CANVAS_CONTENT_OPERATIONS | {"rename"}
    if operation not in allowed:
        allowed_text = ", ".join(sorted(allowed))
        raise ValueError(f"{tool_name} 'operation' must be one of: {allowed_text}")
    return operation


def _canvas_document_content(markdown: str) -> dict[str, str]:
    return {"type": "markdown", "markdown": markdown}


def _canvas_change_from_args(
    args: JsonObject,
    *,
    operation: str,
    tool_name: str,
) -> JsonObject:
    if operation == "rename":
        return {
            "operation": "rename",
            "title_content": _canvas_document_content(
                _coerce_canvas_title(args.get("title"), tool_name=tool_name)
            ),
        }

    change: JsonObject = {
        "operation": operation,
        "document_content": _canvas_document_content(
            _coerce_canvas_markdown(args.get("markdown"), tool_name=tool_name)
        ),
    }
    section_id = args.get("section_id")
    if operation in {"insert_before", "insert_after"} or section_id is not None:
        change["section_id"] = _coerce_canvas_section_id(
            section_id,
            tool_name=tool_name,
        )
    return change


def _coerce_purpose(value: object, *, default: str) -> str:
    if value is None:
        return default
    if not isinstance(value, str):
        raise ValueError("slack_reply_thread 'purpose' must be a string")
    purpose = value.strip().lower().replace(" ", "_")
    if not purpose:
        return default
    if not re.match(r"^[a-z0-9_-]{1,40}$", purpose):
        raise ValueError("slack_reply_thread 'purpose' must be a short label")
    return f"tool_{purpose}" if not purpose.startswith("tool_") else purpose


def _post_thread_ts(*, channel_id: str, thread_ts: str | None) -> str | None:
    if channel_id.startswith("D"):
        return None
    return thread_ts


def _reply_idempotency_key(
    *,
    task: Task,
    purpose: str,
    channel_id: str,
    thread_ts: str | None,
    text: str,
) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"slack:tool_reply:{task.id}:{purpose}:{channel_id}:{thread_ts or 'root'}:{digest}"


def _bookmark_digest(*, title: str, link: str) -> str:
    return hashlib.sha256(f"{title}\n{link}".encode()).hexdigest()[:16]


def _canvas_digest(*, title: str, markdown: str) -> str:
    return hashlib.sha256(f"{title}\n{markdown}".encode()).hexdigest()[:16]


def _canvas_edit_digest(*, canvas_id: str, change: JsonObject) -> str:
    payload = repr(sorted((key, repr(value)) for key, value in change.items()))
    return hashlib.sha256(f"{canvas_id}\n{payload}".encode()).hexdigest()[:16]


def _require_ok(response: Mapping[str, Any], method: str) -> Mapping[str, Any]:
    ok = response.get("ok")
    if ok is False:
        error = response.get("error")
        raise RuntimeError(f"Slack {method} failed: {error or 'unknown_error'}")
    return response


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


def _response_bookmark_id(response: Mapping[str, Any]) -> str | None:
    bookmark = response.get("bookmark")
    if isinstance(bookmark, Mapping):
        bookmark_id = bookmark.get("id")
        if isinstance(bookmark_id, str) and bookmark_id:
            return bookmark_id
    return None


def _response_canvas_id(response: Mapping[str, Any]) -> str | None:
    canvas_id = response.get("canvas_id")
    if isinstance(canvas_id, str) and canvas_id:
        return canvas_id
    canvas = response.get("canvas")
    if isinstance(canvas, Mapping):
        nested_canvas_id = canvas.get("id") or canvas.get("canvas_id")
        if isinstance(nested_canvas_id, str) and nested_canvas_id:
            return nested_canvas_id
    return None


def _response_canvas_sections(response: Mapping[str, Any]) -> list[JsonObject]:
    sections = response.get("sections")
    if not isinstance(sections, list):
        return []
    normalized: list[JsonObject] = []
    for section in sections:
        payload = _normalize_canvas_section(section)
        if payload is not None:
            normalized.append(payload)
    return normalized


def _normalize_canvas_section(section: object) -> JsonObject | None:
    if isinstance(section, str):
        return {"section_id": section}
    if not isinstance(section, Mapping):
        return None
    section_id = section.get("id") or section.get("section_id")
    if not isinstance(section_id, str) or not section_id:
        return None
    payload: JsonObject = {"section_id": section_id}
    section_type = section.get("type") or section.get("section_type")
    if isinstance(section_type, str) and section_type:
        payload["section_type"] = section_type
    text = section.get("text") or section.get("plain_text")
    if isinstance(text, str) and text:
        payload["text"] = text
    return payload


def _is_slack_timestamp(value: str | None) -> bool:
    return isinstance(value, str) and SLACK_TS_RE.match(value) is not None


def _task_event_exists(
    session: Session,
    *,
    task: Task,
    event_type: TaskEventType,
    side_effect_id: str,
) -> bool:
    return (
        session.scalar(
            select(TaskEvent.id)
            .where(
                TaskEvent.task_id == task.id,
                TaskEvent.type == event_type,
                TaskEvent.payload["slack_side_effect_id"].as_string() == side_effect_id,
            )
            .limit(1)
        )
        is not None
    )
