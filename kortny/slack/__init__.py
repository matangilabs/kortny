"""Slack ingress and posting helpers."""

from kortny.slack.acknowledgement import (
    ROOT_ACK_FALLBACK_TEXT,
    LLMAcknowledgementGenerator,
    StaticAcknowledgementGenerator,
)
from kortny.slack.app import acknowledge_then_handle, create_bolt_app, run_socket_mode
from kortny.slack.comments import (
    ARTIFACT_COMMENT_FALLBACK_TEXT,
    LLMArtifactCommentGenerator,
    StaticArtifactCommentGenerator,
)
from kortny.slack.formatting import normalize_slack_mrkdwn
from kortny.slack.ingress import AppMentionResult, SlackIngress
from kortny.slack.posting import (
    SlackPoster,
    SlackPostingError,
    SlackThread,
)

__all__ = [
    "AppMentionResult",
    "ARTIFACT_COMMENT_FALLBACK_TEXT",
    "LLMArtifactCommentGenerator",
    "LLMAcknowledgementGenerator",
    "ROOT_ACK_FALLBACK_TEXT",
    "SlackPoster",
    "SlackPostingError",
    "SlackThread",
    "SlackIngress",
    "StaticArtifactCommentGenerator",
    "StaticAcknowledgementGenerator",
    "acknowledge_then_handle",
    "create_bolt_app",
    "normalize_slack_mrkdwn",
    "run_socket_mode",
]
