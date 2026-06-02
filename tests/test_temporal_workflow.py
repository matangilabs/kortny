from kortny.workflow.temporal import (
    KORTNY_TEMPORAL_TASK_QUEUE,
    KortnyTaskWorkflow,
    KortnyWorkflowInput,
    KortnyWorkflowResult,
)


def test_temporal_workflow_payloads_are_json_safe() -> None:
    workflow_input = KortnyWorkflowInput(
        task_id="task-1",
        installation_id="installation-1",
        slack_channel_id="C123",
        slack_thread_ts="123.456",
        slack_user_id="U123",
        input="Research this and summarize it.",
    )
    workflow_result = KortnyWorkflowResult(
        task_id="task-1",
        status="accepted",
        summary="Workflow accepted.",
    )

    assert workflow_input.to_payload() == {
        "task_id": "task-1",
        "installation_id": "installation-1",
        "slack_channel_id": "C123",
        "slack_thread_ts": "123.456",
        "slack_user_id": "U123",
        "input": "Research this and summarize it.",
    }
    assert workflow_result.to_payload() == {
        "task_id": "task-1",
        "status": "accepted",
        "summary": "Workflow accepted.",
    }


def test_temporal_workflow_exposes_progress_query_shape() -> None:
    workflow = KortnyTaskWorkflow()

    assert KORTNY_TEMPORAL_TASK_QUEUE == "kortny-workflows"
    assert workflow.progress() == {"status": "created", "summary": ""}
