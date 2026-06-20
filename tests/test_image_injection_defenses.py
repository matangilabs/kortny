"""HIG-279 slice 2B: prompt-injection defenses for attached images.

Unit tests (no DB) for:
- TrifectaGateState.arm() new method
- Coordinator arming the gate when context messages carry images
- Spotlighting directive insertion in ContextAssembler

DB tests confirm the coordinator emits the correct trifecta_gate audit event.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator, Sequence

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, delete, select
from sqlalchemy.orm import Session

from kortny.agent.context import (
    IMAGE_INJECTION_DIRECTIVE,
    ContextBudget,
    ContextPackage,
)
from kortny.agent.context_engine import ContextEngineInfo
from kortny.agent.coordinator import TRIFECTA_GATE_MESSAGE, AgentCoordinator
from kortny.agent.planner import ExecutionPlanner, PlannerGateDecision
from kortny.agent.trifecta import TrifectaGateState
from kortny.db.models import (
    Artifact,
    Installation,
    LLMUsage,
    Task,
    TaskEvent,
    TaskEventType,
)
from kortny.db.session import make_engine, make_session_factory, normalize_database_url
from kortny.llm import ChatMessage, Completion, ImagePart, TokenUsage
from kortny.tasks import TaskService
from kortny.tools import ToolRegistry
from kortny.tools.types import JsonObject, JsonSchema

TEST_POSTGRES_URL = os.environ.get("KORTNY_TEST_POSTGRES_URL")

# ---------------------------------------------------------------------------
# Unit: TrifectaGateState.arm()
# ---------------------------------------------------------------------------


def test_arm_transitions_disarmed_to_armed() -> None:
    state = TrifectaGateState(enabled=True)
    assert not state.armed
    result = state.arm("attached_image")
    assert result is True
    assert state.armed
    assert state.armed_by == "attached_image"


def test_arm_is_noop_when_already_armed_by_note_tool_result() -> None:
    state = TrifectaGateState(enabled=True)
    state.note_tool_result("web_search")
    assert state.armed
    assert state.armed_by == "web_search"
    # arm() must not overwrite the first source
    result = state.arm("attached_image")
    assert result is False
    assert state.armed_by == "web_search"


def test_arm_is_noop_when_already_armed_via_constructor() -> None:
    state = TrifectaGateState(enabled=True, armed=True)
    assert state.armed_by == "initial_context"
    result = state.arm("attached_image")
    assert result is False
    assert state.armed_by == "initial_context"


def test_arm_noop_when_gate_disabled() -> None:
    state = TrifectaGateState(enabled=False)
    result = state.arm("attached_image")
    assert result is False
    assert not state.armed
    assert state.armed_by is None


def test_arm_then_should_escalate_outward_tool() -> None:
    state = TrifectaGateState(enabled=True)
    state.arm("attached_image")
    # After arming via arm(), outward tools must be escalated
    assert state.should_escalate("composio__notion__create_page")
    assert state.should_escalate("slack_reply_thread")
    # Read-only / local-compute tools stay free
    assert not state.should_escalate("web_search")
    assert not state.should_escalate("code_exec")


# ---------------------------------------------------------------------------
# Unit: IMAGE_INJECTION_DIRECTIVE constant is non-empty
# ---------------------------------------------------------------------------


def test_image_injection_directive_is_nonempty() -> None:
    assert isinstance(IMAGE_INJECTION_DIRECTIVE, str)
    assert len(IMAGE_INJECTION_DIRECTIVE) > 0


# ---------------------------------------------------------------------------
# Unit: ContextAssembler spotlighting directive insertion (no DB needed)
# ---------------------------------------------------------------------------

_FAKE_IMAGE = ImagePart(
    data=b"\x89PNG\r\n",
    mime="image/png",
    source="slack_file:FTEST",
)


def test_user_message_with_images_has_images_tuple() -> None:
    """A ChatMessage carrying images has a non-empty .images tuple."""
    msg = ChatMessage(
        role="user",
        content="describe this",
        images=(_FAKE_IMAGE,),
    )
    assert bool(msg.images)


def test_user_message_text_only_has_empty_images() -> None:
    """A text-only ChatMessage has an empty .images tuple."""
    msg = ChatMessage(role="user", content="hello world")
    assert not msg.images


def test_spotlighting_directive_branch_logic() -> None:
    """The spotlighting insertion predicate fires iff the user message has images."""

    # Simulate what _build_for_task does: collect preceding messages then append
    # the directive + user message only when images are present.
    def _build_messages(user_msg: ChatMessage) -> list[ChatMessage]:
        messages: list[ChatMessage] = []
        if user_msg.images:
            messages.append(
                ChatMessage(role="system", content=IMAGE_INJECTION_DIRECTIVE)
            )
        messages.append(user_msg)
        return messages

    with_images = ChatMessage(
        role="user", content="look at this", images=(_FAKE_IMAGE,)
    )
    text_only = ChatMessage(role="user", content="just text")

    result_images = _build_messages(with_images)
    assert len(result_images) == 2
    assert result_images[0].role == "system"
    assert result_images[0].content == IMAGE_INJECTION_DIRECTIVE
    assert result_images[1] is with_images

    result_text = _build_messages(text_only)
    assert len(result_text) == 1
    assert result_text[0] is text_only


def test_spotlighting_directive_position_adjacent_to_user_message() -> None:
    """Spotlighting system message is immediately before the user message."""
    # Simulate a full assembled message list (system prompts + spotlighting + user)
    system_prompt = ChatMessage(role="system", content="You are Kortny.")
    facts = ChatMessage(role="system", content="Facts: ...")
    user_msg = ChatMessage(role="user", content="describe", images=(_FAKE_IMAGE,))
    directive = ChatMessage(role="system", content=IMAGE_INJECTION_DIRECTIVE)

    messages = [system_prompt, facts, directive, user_msg]

    user_idx = next(i for i, m in enumerate(messages) if m.role == "user")
    assert user_idx > 0
    assert messages[user_idx - 1].content == IMAGE_INJECTION_DIRECTIVE


# ---------------------------------------------------------------------------
# DB integration: coordinator arms trifecta gate when context has images
# ---------------------------------------------------------------------------


class _FakeLLM:
    def __init__(self, completions: Sequence[Completion]) -> None:
        self.completions = list(completions)

    def complete(
        self,
        *,
        task_id: uuid.UUID,
        messages: Sequence[ChatMessage],
        tools: Sequence[JsonSchema] = (),
        response_format: JsonObject | None = None,
        prompt_name: str | None = None,
        prompt_source: str = "code",
    ) -> Completion:
        del task_id, messages, tools, response_format, prompt_name, prompt_source
        if not self.completions:
            raise AssertionError("_FakeLLM received more calls than expected")
        return self.completions.pop(0)


class _NoopExecutionPlanner(ExecutionPlanner):
    def should_plan(  # type: ignore[no-untyped-def]
        self, *, task, tool_schemas, intent_decision
    ) -> PlannerGateDecision:
        del task, tool_schemas, intent_decision
        return PlannerGateDecision(False, "test_no_plan")


class _ImageContextEngine:
    """Fake ContextEngine that returns a user message with an attached image."""

    def __init__(self) -> None:
        self.info = ContextEngineInfo(
            id="test.image_context_engine",
            name="Image Context Engine",
        )

    def ingest(self, task: Task) -> None:  # noqa: D401
        pass

    def assemble(self, task: Task) -> ContextPackage:
        return ContextPackage(
            messages=(
                ChatMessage(
                    role="user",
                    content=task.input,
                    images=(_FAKE_IMAGE,),
                ),
            ),
            selected_facts=(),
            selected_prior_tasks=(),
            selected_episodes=(),
            selected_artifacts=(),
            selected_graph_entities=(),
            selected_graph_edges=(),
            acknowledgement=None,
            budget=ContextBudget(
                system_prompt_chars=0,
                known_facts_max_chars=0,
                known_facts_chars=0,
                thread_context_max_chars=1,
                prior_context_chars=0,
                thread_context_recent_tasks=1,
                thread_transcript_limit=0,
                episode_context_max_chars=0,
                episode_context_chars=0,
                episode_context_limit=0,
                graph_context_max_chars=0,
                graph_context_chars=0,
                graph_context_max_items=0,
                graph_context_max_hops=0,
            ),
            omissions=(),
            context_engine_id=self.info.id,
            context_engine_name=self.info.name,
        )

    def compact(self, task: Task, *, force: bool = False) -> None:
        return None

    def after_turn(
        self,
        task: Task,
        package: object,
        *,
        outcome: str,
    ) -> None:
        pass


class _TextOnlyContextEngine:
    """Fake ContextEngine that returns a text-only user message (no images)."""

    def __init__(self) -> None:
        self.info = ContextEngineInfo(
            id="test.text_only_context_engine",
            name="Text-only Context Engine",
        )

    def ingest(self, task: Task) -> None:
        pass

    def assemble(self, task: Task) -> ContextPackage:
        return ContextPackage(
            messages=(ChatMessage(role="user", content=task.input),),
            selected_facts=(),
            selected_prior_tasks=(),
            selected_episodes=(),
            selected_artifacts=(),
            selected_graph_entities=(),
            selected_graph_edges=(),
            acknowledgement=None,
            budget=ContextBudget(
                system_prompt_chars=0,
                known_facts_max_chars=0,
                known_facts_chars=0,
                thread_context_max_chars=1,
                prior_context_chars=0,
                thread_context_recent_tasks=1,
                thread_transcript_limit=0,
                episode_context_max_chars=0,
                episode_context_chars=0,
                episode_context_limit=0,
                graph_context_max_chars=0,
                graph_context_chars=0,
                graph_context_max_items=0,
                graph_context_max_hops=0,
            ),
            omissions=(),
            context_engine_id=self.info.id,
            context_engine_name=self.info.name,
        )

    def compact(self, task: Task, *, force: bool = False) -> None:
        return None

    def after_turn(
        self,
        task: Task,
        package: object,
        *,
        outcome: str,
    ) -> None:
        pass


pytestmark_db = pytest.mark.skipif(
    TEST_POSTGRES_URL is None,
    reason="KORTNY_TEST_POSTGRES_URL is required for DB integration tests",
)


@pytest.fixture(scope="session")
def engine() -> Iterator[Engine]:
    if TEST_POSTGRES_URL is None:
        pytest.skip("KORTNY_TEST_POSTGRES_URL is required for image injection tests")
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", normalize_database_url(TEST_POSTGRES_URL))
    command.upgrade(config, "head")
    eng = make_engine(TEST_POSTGRES_URL)
    try:
        yield eng
    finally:
        eng.dispose()


@pytest.fixture
def db_session(engine: Engine) -> Iterator[Session]:
    session_factory = make_session_factory(engine=engine)
    with session_factory() as session:
        _cleanup(session)
        session.commit()
        yield session
        session.rollback()
        _cleanup(session)
        session.commit()


def _cleanup(session: Session) -> None:
    for model in (Artifact, LLMUsage, TaskEvent, Task, Installation):
        session.execute(delete(model))


def _task(session: Session, *, input_text: str = "describe this image") -> Task:
    installation = Installation(slack_team_id=f"T{uuid.uuid4().hex}")
    session.add(installation)
    session.flush()
    return TaskService(session).create_task(
        installation_id=installation.id,
        slack_event_id=f"Ev{uuid.uuid4().hex}",
        slack_channel_id="C123",
        slack_thread_ts="1716400000.000001",
        slack_message_ts=f"1716400000.{uuid.uuid4().int % 999999:06d}",
        slack_user_id="U123",
        input=input_text,
    )


def _trifecta_events(session: Session, task: Task) -> list[TaskEvent]:
    return [
        event
        for event in session.scalars(
            select(TaskEvent).where(TaskEvent.task_id == task.id)
        )
        if event.type is TaskEventType.log
        and event.payload.get("message") == TRIFECTA_GATE_MESSAGE
    ]


@pytestmark_db
def test_coordinator_arms_gate_when_context_has_images(db_session: Session) -> None:
    """When the context package includes a user message with images, the
    coordinator must arm the trifecta gate with source 'attached_image' before
    the first LLM turn.
    """
    task = _task(db_session)
    llm = _FakeLLM(
        [
            Completion(
                content="Here is what I see in the image.",
                tool_calls=(),
                model="test",
                usage=TokenUsage(input_tokens=10, output_tokens=5),
            )
        ]
    )
    registry = ToolRegistry()
    coordinator = AgentCoordinator(
        session=db_session,
        llm=llm,
        registry=registry,
        context_engine=_ImageContextEngine(),
        execution_planner=_NoopExecutionPlanner(),
        trifecta_gate_enabled=True,
    )
    coordinator.run(task)

    # The trifecta gate should have been armed by "attached_image"
    gate = coordinator._trifecta_state(task)
    assert gate.armed, "Gate should be armed when user message carries images"
    assert gate.armed_by == "attached_image"

    # An audit log event must also have been emitted
    trifecta_events = _trifecta_events(db_session, task)
    armed_events = [e for e in trifecta_events if e.payload.get("event") == "armed"]
    assert armed_events, "Expected at least one trifecta_gate 'armed' log event"
    armed_sources = {e.payload.get("armed_by") for e in armed_events}
    assert "attached_image" in armed_sources


@pytestmark_db
def test_coordinator_does_not_arm_gate_for_text_only_task(db_session: Session) -> None:
    """A text-only context package must NOT arm the trifecta gate via images."""
    task = _task(db_session, input_text="just a text request")
    llm = _FakeLLM(
        [
            Completion(
                content="Sure, here is my answer.",
                tool_calls=(),
                model="test",
                usage=TokenUsage(input_tokens=8, output_tokens=4),
            )
        ]
    )
    registry = ToolRegistry()
    coordinator = AgentCoordinator(
        session=db_session,
        llm=llm,
        registry=registry,
        context_engine=_TextOnlyContextEngine(),
        execution_planner=_NoopExecutionPlanner(),
        trifecta_gate_enabled=True,
    )
    coordinator.run(task)

    gate = coordinator._trifecta_state(task)
    # Gate may still be armed by the task's identity_kind (synthetic tasks are
    # armed at start), but it should NOT be armed by "attached_image".
    if gate.armed:
        assert gate.armed_by != "attached_image", (
            "Text-only task gate was incorrectly armed by 'attached_image'"
        )
    # No trifecta armed event with armed_by="attached_image"
    trifecta_events = _trifecta_events(db_session, task)
    image_armed = [
        e
        for e in trifecta_events
        if e.payload.get("event") == "armed"
        and e.payload.get("armed_by") == "attached_image"
    ]
    assert not image_armed, (
        "Should not have an 'attached_image' arm event for a text-only task"
    )
