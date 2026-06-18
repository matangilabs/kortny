from kortny.agent import ExecutionErrorCategory, RecoveryAction
from kortny.agent.error_policy import classify_exception, classify_tool_error_payload
from kortny.composio import ComposioConnectionError


def test_error_policy_classifies_missing_required_arguments() -> None:
    classification = classify_tool_error_payload(
        {
            "code": "missing_required_arguments",
            "message": "database_id is required",
            "recoverable": True,
        }
    )

    assert classification.category is ExecutionErrorCategory.schema_argument_validation
    assert classification.recovery_action is RecoveryAction.patch_arguments
    assert classification.retryable is False
    assert classification.user_action_required is False


def test_error_policy_classifies_reference_resolution_errors() -> None:
    classification = classify_tool_error_payload(
        {
            "code": "channel_not_found",
            "message": "Slack channel not found",
            "recoverable": True,
        }
    )

    assert classification.category is ExecutionErrorCategory.reference_resolution
    assert classification.recovery_action is RecoveryAction.resolve_reference


def test_error_policy_classifies_rate_limits() -> None:
    classification = classify_tool_error_payload(
        {
            "code": "rate_limited",
            "message": "Search quota reached",
            "recoverable": True,
            "retry_after": "2",
        }
    )

    assert classification.category is ExecutionErrorCategory.rate_limited
    assert classification.recovery_action is RecoveryAction.retry_with_backoff
    assert classification.retryable is True
    assert classification.retry_after == "2"


def test_error_policy_classifies_read_timeout_as_recoverable_transient() -> None:
    # Regression: "The read operation timed out" normalizes to "..._timed_out",
    # which does not contain "timeout" — it used to fall through to the unknown
    # -> stop_safely path and hard-fail the task. A transient read timeout must
    # be transient transport: retryable AND recoverable so the loop recovers.
    classification = classify_exception(
        ComposioConnectionError("The read operation timed out")
    )

    assert classification.category is ExecutionErrorCategory.transient_transport
    assert classification.recovery_action is RecoveryAction.retry_with_backoff
    assert classification.retryable is True
    assert classification.recoverable is True
    assert classification.user_action_required is False


def test_error_policy_timed_out_payload_is_transient() -> None:
    classification = classify_tool_error_payload(
        {"code": "ComposioConnectionError", "message": "The read operation timed out"}
    )
    assert classification.category is ExecutionErrorCategory.transient_transport
    assert classification.retryable is True


def test_error_policy_unknown_exception_still_stops_safely() -> None:
    # The carve-out must not make every exception recoverable: a genuinely
    # unknown error still stops safely and is not retried.
    classification = classify_exception(ValueError("something inexplicable"))

    assert classification.category is ExecutionErrorCategory.unknown
    assert classification.recovery_action is RecoveryAction.stop_safely
    assert classification.retryable is False
    assert classification.recoverable is False


def test_error_policy_classifies_permission_policy_blocks() -> None:
    classification = classify_tool_error_payload(
        {
            "code": "secret_not_stored",
            "message": "Secrets are not saved in memory",
            "recoverable": True,
        }
    )

    assert classification.category is ExecutionErrorCategory.permission_policy
    assert classification.recovery_action is RecoveryAction.stop_safely
    assert classification.user_action_required is True
