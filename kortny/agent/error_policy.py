"""Centralized tool-error taxonomy and recovery policy."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

from kortny.tools.types import JsonObject, RecoverableToolError


class ExecutionErrorCategory(StrEnum):
    """Controller-owned categories for tool/runtime failures."""

    schema_argument_validation = "schema_argument_validation"
    reference_resolution = "reference_resolution"
    auth_connection = "auth_connection"
    rate_limited = "rate_limited"
    transient_transport = "transient_transport"
    empty_result_or_bad_tool_choice = "empty_result_or_bad_tool_choice"
    permission_policy = "permission_policy"
    destructive_action_risk = "destructive_action_risk"
    budget_exhausted = "budget_exhausted"
    unknown = "unknown"


class RecoveryAction(StrEnum):
    """Recommended controller/model action after a classified failure."""

    patch_arguments = "patch_arguments"
    resolve_reference = "resolve_reference"
    wait_auth = "wait_auth"
    retry_with_backoff = "retry_with_backoff"
    switch_tool_or_broaden_query = "switch_tool_or_broaden_query"
    ask_user = "ask_user"
    stop_safely = "stop_safely"
    continue_ = "continue"


@dataclass(frozen=True, slots=True)
class ClassifiedToolError:
    """A provider-neutral failure classification."""

    code: str
    message: str
    category: ExecutionErrorCategory
    recovery_action: RecoveryAction
    recoverable: bool
    retryable: bool
    user_action_required: bool
    hint: str | None = None
    retry_after: str | None = None
    details: JsonObject = field(default_factory=dict)

    def to_payload(self) -> JsonObject:
        """Return a JSON-safe payload."""

        payload = asdict(self)
        payload["category"] = self.category.value
        payload["recovery_action"] = self.recovery_action.value
        return {key: value for key, value in payload.items() if value is not None}


def classify_recoverable_tool_error(
    error: RecoverableToolError,
) -> ClassifiedToolError:
    """Classify a recoverable exception raised by a tool."""

    payload = error.to_payload()
    return classify_tool_error_payload(payload)


def classify_tool_error_payload(payload: Mapping[str, Any]) -> ClassifiedToolError:
    """Classify a tool-result error payload."""

    code = _string(payload.get("code")) or "unknown_error"
    message = _string(payload.get("message")) or code
    hint = _string(payload.get("hint"))
    details = _json_object(payload.get("details"))
    retry_after = _string(payload.get("retry_after"))
    recoverable = payload.get("recoverable") is True
    category, action = _category_and_action(code)
    retryable = action is RecoveryAction.retry_with_backoff
    user_action_required = action in {
        RecoveryAction.ask_user,
        RecoveryAction.wait_auth,
        RecoveryAction.stop_safely,
    }

    return ClassifiedToolError(
        code=code,
        message=message,
        category=category,
        recovery_action=action,
        recoverable=recoverable,
        retryable=retryable,
        user_action_required=user_action_required,
        hint=hint,
        retry_after=retry_after,
        details=details,
    )


def classify_exception(error: Exception) -> ClassifiedToolError:
    """Classify a terminal exception for trace payloads."""

    code = type(error).__name__
    message = str(error) or code
    category, action = _category_and_action(f"{code}_{message}")
    if category is ExecutionErrorCategory.unknown:
        action = RecoveryAction.stop_safely

    # Derive the flags from the chosen action (mirrors the payload classifier)
    # rather than hardcoding False — a transient transport exception (e.g. a
    # read timeout) is genuinely retryable and recoverable, so the coordinator
    # can retry / route around it instead of hard-failing the task.
    retryable = action is RecoveryAction.retry_with_backoff
    recoverable = action in {
        RecoveryAction.retry_with_backoff,
        RecoveryAction.switch_tool_or_broaden_query,
        RecoveryAction.patch_arguments,
        RecoveryAction.resolve_reference,
    }
    user_action_required = action in {
        RecoveryAction.ask_user,
        RecoveryAction.wait_auth,
        RecoveryAction.stop_safely,
    }

    return ClassifiedToolError(
        code=code,
        message=message,
        category=category,
        recovery_action=action,
        recoverable=recoverable,
        retryable=retryable,
        user_action_required=user_action_required,
    )


def enrich_error_payload(
    payload: Mapping[str, Any],
    classification: ClassifiedToolError | None = None,
) -> JsonObject:
    """Return a tool error payload with taxonomy fields attached."""

    classified = classification or classify_tool_error_payload(payload)
    enriched = dict(payload)
    enriched["category"] = classified.category.value
    enriched["recovery_action"] = classified.recovery_action.value
    enriched["retryable"] = classified.retryable
    enriched["user_action_required"] = classified.user_action_required
    return enriched


def _category_and_action(
    code: str,
) -> tuple[ExecutionErrorCategory, RecoveryAction]:
    normalized = code.casefold().replace("-", "_").replace(" ", "_")

    if any(token in normalized for token in ("missing_required", "invalid_argument")):
        return (
            ExecutionErrorCategory.schema_argument_validation,
            RecoveryAction.patch_arguments,
        )
    if normalized in {"min_pages_not_met", "output_constraint_not_met"}:
        return (
            ExecutionErrorCategory.schema_argument_validation,
            RecoveryAction.patch_arguments,
        )
    if any(
        token in normalized
        for token in (
            "file_not_found",
            "channel_not_found",
            "invalid_file_id",
            "invalid_channel",
            "not_found",
            "unknown_channel",
            "unknown_file",
            "reference",
            "missing_identifier",
        )
    ):
        return (
            ExecutionErrorCategory.reference_resolution,
            RecoveryAction.resolve_reference,
        )
    if any(
        token in normalized
        for token in (
            "missing_connection",
            "invalid_auth",
            "not_authed",
            "auth",
            "unauthorized",
            "expired",
            "account_not_connected",
        )
    ):
        return ExecutionErrorCategory.auth_connection, RecoveryAction.wait_auth
    if "rate_limited" in normalized or "too_many_requests" in normalized:
        return ExecutionErrorCategory.rate_limited, RecoveryAction.retry_with_backoff
    if any(
        token in normalized
        for token in (
            "request_failed",
            "upstream_unavailable",
            "timeout",
            # "The read operation timed out" normalizes to "..._timed_out", which
            # does NOT contain "timeout" — a transient read timeout used to slip
            # through to the unknown -> stop_safely fallback and hard-fail the
            # task. Match the "timed out" / connection-error wording too.
            "timed_out",
            "read_timeout",
            "connect_timeout",
            "connectionerror",
            "connection_error",
            "connection_reset",
            "temporarily_unavailable",
            "service_unavailable",
            "bad_gateway",
            "gateway_timeout",
            "http_5",
        )
    ):
        return (
            ExecutionErrorCategory.transient_transport,
            RecoveryAction.retry_with_backoff,
        )
    if any(
        token in normalized
        for token in ("empty", "no_results", "low_content", "bad_tool_choice")
    ):
        return (
            ExecutionErrorCategory.empty_result_or_bad_tool_choice,
            RecoveryAction.switch_tool_or_broaden_query,
        )
    if any(
        token in normalized
        for token in (
            "secret_not_stored",
            "permission",
            "forbidden",
            "access_denied",
            "policy",
            "scope_missing",
        )
    ):
        return ExecutionErrorCategory.permission_policy, RecoveryAction.stop_safely
    if "destructive" in normalized or "approval_required" in normalized:
        return (
            ExecutionErrorCategory.destructive_action_risk,
            RecoveryAction.ask_user,
        )
    if "budget" in normalized or "max_" in normalized:
        return ExecutionErrorCategory.budget_exhausted, RecoveryAction.stop_safely

    return ExecutionErrorCategory.unknown, RecoveryAction.ask_user


def _string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _json_object(value: object) -> JsonObject:
    return dict(value) if isinstance(value, dict) else {}
