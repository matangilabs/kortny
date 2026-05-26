from kortny.agent import ExecutionErrorCategory, RecoveryAction
from kortny.agent.error_policy import classify_tool_error_payload


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
