"""Slack ingress and posting helpers."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from kortny.slack_mrkdwn import normalize_slack_mrkdwn as _normalize_slack_mrkdwn

normalize_slack_mrkdwn = _normalize_slack_mrkdwn

_EXPORTS = {
    "AppMentionResult": ("kortny.slack.ingress", "AppMentionResult"),
    "ARTIFACT_COMMENT_FALLBACK_TEXT": (
        "kortny.slack.comments",
        "ARTIFACT_COMMENT_FALLBACK_TEXT",
    ),
    "LLMArtifactCommentGenerator": (
        "kortny.slack.comments",
        "LLMArtifactCommentGenerator",
    ),
    "LLMAcknowledgementGenerator": (
        "kortny.slack.acknowledgement",
        "LLMAcknowledgementGenerator",
    ),
    "LLMResponseSynthesizer": ("kortny.slack.humanizer", "LLMResponseSynthesizer"),
    "ResponseSynthesisResult": ("kortny.slack.humanizer", "ResponseSynthesisResult"),
    "ROOT_ACK_FALLBACK_TEXT": (
        "kortny.slack.acknowledgement",
        "ROOT_ACK_FALLBACK_TEXT",
    ),
    "SlackIngress": ("kortny.slack.ingress", "SlackIngress"),
    "SlackPoster": ("kortny.slack.posting", "SlackPoster"),
    "SlackPostingError": ("kortny.slack.posting", "SlackPostingError"),
    "SlackSideEffectOutbox": ("kortny.slack.outbox", "SlackSideEffectOutbox"),
    "SlackSideEffectRecoveryResult": (
        "kortny.slack.outbox",
        "SlackSideEffectRecoveryResult",
    ),
    "SlackSideEffectResult": ("kortny.slack.outbox", "SlackSideEffectResult"),
    "SlackThread": ("kortny.slack.posting", "SlackThread"),
    "StaticAcknowledgementGenerator": (
        "kortny.slack.acknowledgement",
        "StaticAcknowledgementGenerator",
    ),
    "StaticArtifactCommentGenerator": (
        "kortny.slack.comments",
        "StaticArtifactCommentGenerator",
    ),
    "StaticResponseSynthesizer": (
        "kortny.slack.humanizer",
        "StaticResponseSynthesizer",
    ),
    "SynthesisContext": ("kortny.slack.synthesis", "SynthesisContext"),
    "SynthesisOutcome": ("kortny.slack.synthesis", "SynthesisOutcome"),
    "acknowledge_then_handle": ("kortny.slack.app", "acknowledge_then_handle"),
    "create_bolt_app": ("kortny.slack.app", "create_bolt_app"),
    "run_socket_mode": ("kortny.slack.app", "run_socket_mode"),
    "synthesize_response": ("kortny.slack.humanizer", "synthesize_response"),
}

__all__ = sorted((*_EXPORTS, "normalize_slack_mrkdwn"))


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
