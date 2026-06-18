"""HIG-226 channel style cards: derivation pass, humanizer wiring, ack line."""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, delete, select
from sqlalchemy.orm import Session

from kortny.config.settings import Settings
from kortny.consolidator import ConsolidationService
from kortny.consolidator.style_cards import (
    STYLE_CARD_BATCH_SIZE,
    StyleCardPass,
)
from kortny.db.models import (
    ConsolidationRun,
    Installation,
    LLMUsage,
    ObservationEvent,
    ObserveChannelProfile,
    ObservePolicy,
    Task,
    TaskEvent,
)
from kortny.db.session import make_engine, make_session_factory, normalize_database_url
from kortny.llm import (
    ChatMessage,
    Completion,
    LLMService,
    ModelRoute,
    ModelRouteTier,
    TokenUsage,
)
from kortny.observe.style_cards import (
    PINNED_STYLE_KEY,
    STYLE_CARD_INPUT_SHA_KEY,
    STYLE_CARD_KEY,
    STYLE_CARD_UPDATED_AT_KEY,
    ChannelStyleCard,
    load_channel_style,
    parse_style_card,
    reset_style_card,
    set_pinned_style,
)
from kortny.slack.acknowledgement import (
    ACK_SYSTEM_PROMPT,
    LLMAcknowledgementGenerator,
)
from kortny.slack.humanizer import (
    RESPONSE_HUMANIZER_SYSTEM_PROMPT,
    ChannelStyleCardResolver,
    ResponseStyleProfile,
    SlackSurface,
    build_response_record,
)
from kortny.tasks import TaskService
from kortny.tools.types import JsonObject, JsonSchema

TEST_POSTGRES_URL = os.environ.get("KORTNY_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    TEST_POSTGRES_URL is None,
    reason="KORTNY_TEST_POSTGRES_URL is required for style card tests",
)

NOW = datetime(2026, 6, 11, 3, 0, 0, tzinfo=UTC)

CARD_NOTES = "Quick, informal replies; emoji reactions are common."
CARD_PAYLOAD: dict[str, object] = {
    "formality": "casual",
    "brevity": "terse",
    "emoji_culture": "heavy",
    "punctuation": "relaxed",
    "common_phrases": ["ship it", "lgtm"],
    "threading_norm": "threads_heavy",
    "notes": CARD_NOTES,
}


@pytest.fixture(scope="session")
def engine() -> Iterator[Engine]:
    assert TEST_POSTGRES_URL is not None

    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", normalize_database_url(TEST_POSTGRES_URL))
    command.upgrade(config, "heads")

    engine = make_engine(TEST_POSTGRES_URL)
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture
def db_session(engine: Engine) -> Iterator[Session]:
    session_factory = make_session_factory(engine=engine)
    with session_factory() as session:
        cleanup_database(session)
        session.commit()
        yield session
        session.rollback()
        cleanup_database(session)
        session.commit()


def cleanup_database(session: Session) -> None:
    for model in (
        ConsolidationRun,
        ObservationEvent,
        ObserveChannelProfile,
        ObservePolicy,
        LLMUsage,
        TaskEvent,
        Task,
        Installation,
    ):
        session.execute(delete(model))


def create_installation(session: Session) -> Installation:
    installation = Installation(slack_team_id=f"T{uuid.uuid4().hex}")
    session.add(installation)
    session.flush()
    return installation


def create_task(
    session: Session,
    installation: Installation,
    *,
    channel_id: str = "C_MAIN",
    input_text: str = "summarize the launch thread",
) -> Task:
    return TaskService(session).create_task(
        installation_id=installation.id,
        slack_event_id=f"Ev{uuid.uuid4().hex}",
        slack_channel_id=channel_id,
        slack_thread_ts="1780000000.000100",
        slack_message_ts=f"1780000000.{uuid.uuid4().hex[:6]}",
        slack_user_id="U_USER",
        input=input_text,
    )


def create_profile(
    session: Session,
    installation: Installation,
    *,
    channel_id: str = "C_MAIN",
    summary: str = "Engineering channel for launch coordination.",
    profile_json: dict[str, object] | None = None,
    last_profiled_at: datetime | None = None,
    profile_status: str = "active",
) -> ObserveChannelProfile:
    profile = ObserveChannelProfile(
        installation_id=installation.id,
        channel_id=channel_id,
        profile_status=profile_status,
        profile_version=1,
        summary=summary,
        profile_json=profile_json or {},
        last_profiled_at=last_profiled_at or (NOW - timedelta(days=1)),
    )
    session.add(profile)
    session.flush()
    return profile


def create_observation_messages(
    session: Session,
    installation: Installation,
    *,
    channel_id: str = "C_MAIN",
    count: int = 30,
    observed_at: datetime | None = None,
    text: str = "ship it :rocket:",
) -> None:
    base = observed_at or (NOW - timedelta(days=2))
    for index in range(count):
        session.add(
            ObservationEvent(
                installation_id=installation.id,
                slack_team_id=installation.slack_team_id,
                channel_id=channel_id,
                user_id="U_USER",
                event_type="message",
                slack_event_id=f"Ob{uuid.uuid4().hex}",
                message_ts=f"178000{index:04d}.000100",
                raw_payload_checksum=uuid.uuid4().hex,
                text_preview=f"{text} #{index}",
                observed_at=base + timedelta(minutes=index),
            )
        )
    session.flush()


class FakeStyleLLMProvider:
    model = "openai/gpt-4o-mini"

    def __init__(self, completions: list[Completion]) -> None:
        self.completions = completions
        self.calls: list[tuple[tuple[ChatMessage, ...], JsonObject | None]] = []

    def complete(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[JsonSchema] = (),
        *,
        response_format: JsonObject | None = None,
        max_output_tokens: int | None = None,
    ) -> Completion:
        del tools
        self.calls.append((tuple(messages), response_format))
        if not self.completions:
            raise AssertionError("FakeStyleLLMProvider got too many calls")
        return self.completions.pop(0)


def make_completion(payload: dict[str, object]) -> Completion:
    return Completion(
        content=json.dumps(payload),
        tool_calls=(),
        usage=TokenUsage(input_tokens=120, output_tokens=40),
        cost_usd=Decimal("0.000100"),
        model="openai/gpt-4o-mini",
    )


def card_completion(*channel_ids: str) -> Completion:
    return make_completion(
        {
            "cards": [
                {"channel_id": channel_id, **CARD_PAYLOAD} for channel_id in channel_ids
            ]
        }
    )


def make_llm(
    session: Session,
    provider: FakeStyleLLMProvider,
) -> LLMService:
    return LLMService(
        session=session,
        provider=provider,
        provider_name="openrouter",
        task_service=TaskService(session),
        model_route=ModelRoute(
            tier=ModelRouteTier.cheap_fast,
            model=provider.model,
            reason="test",
        ),
    )


def sample_card() -> ChannelStyleCard:
    card = parse_style_card(CARD_PAYLOAD)
    assert card is not None
    return card


def make_settings(**overrides: object) -> Settings:
    payload: dict[str, object] = {
        "SLACK_BOT_TOKEN": "xoxb-test",
        "SLACK_APP_TOKEN": "xapp-test",
        "SLACK_SIGNING_SECRET": "signing-secret",
        "LLM_PROVIDER": "openrouter",
        "LLM_API_KEY": "openrouter-key",
        "LLM_MODEL": "openai/gpt-4o-mini",
        "POSTGRES_URL": "postgresql://kortny:kortny@localhost/kortny",
        "COMPOSIO_API_KEY": "composio-test",
        "KORTNY_STYLE_CARDS_ENABLED": True,
        "KORTNY_STYLE_CARD_MIN_MESSAGES": 30,
    }
    payload.update(overrides)
    return Settings.model_validate(payload)


# --- ResponseStyleProfile mapping (pure) -------------------------------------


def test_parse_style_card_validates_dimensions() -> None:
    card = parse_style_card(CARD_PAYLOAD)
    assert card is not None
    assert card.formality == "casual"
    assert card.brevity == "terse"
    assert card.common_phrases == ("ship it", "lgtm")

    assert parse_style_card(None) is None
    assert parse_style_card({**CARD_PAYLOAD, "formality": "shouty"}) is None
    assert parse_style_card({**CARD_PAYLOAD, "threading_norm": "nope"}) is None


def test_from_style_card_maps_register_onto_profile() -> None:
    surface = SlackSurface(kind="channel", threaded=True)

    profile = ResponseStyleProfile.from_style_card(sample_card(), surface)

    assert profile.tone == "casual, friendly, direct"
    assert profile.brevity == "very concise"
    assert profile.polish == "relaxed"
    assert profile.humor == "off_by_default"
    assert "casual formality" in profile.channel_voice
    assert "terse replies" in profile.channel_voice
    assert "emoji are welcome" in profile.channel_voice
    assert "relaxed punctuation" in profile.channel_voice
    assert CARD_NOTES in profile.channel_voice


def test_from_style_card_channel_voice_is_bounded() -> None:
    card = parse_style_card({**CARD_PAYLOAD, "notes": "registers " * 60})
    assert card is not None

    profile = ResponseStyleProfile.from_style_card(
        card, SlackSurface(kind="channel", threaded=False)
    )

    assert len(profile.channel_voice) <= 240


def test_from_style_card_dm_surface_returns_static_default() -> None:
    profile = ResponseStyleProfile.from_style_card(
        sample_card(), SlackSurface(kind="dm", threaded=False)
    )

    assert profile == ResponseStyleProfile()
    assert profile.channel_voice == ""


def test_default_style_profile_payload_is_byte_identical_to_today() -> None:
    profile = ResponseStyleProfile()

    assert profile.to_payload() == {
        "tone": "approachable, steady, direct",
        "brevity": "concise",
        "polish": "professional",
        "humor": "off_by_default",
        "proactive_suggestions": "only_when_clearly_useful",
    }
    assert "channel_voice" not in profile.to_payload()


def test_humanizer_prompt_contains_channel_voice_rule() -> None:
    assert "channel_voice" in RESPONSE_HUMANIZER_SYSTEM_PROMPT
    assert "Never imitate a specific person." in RESPONSE_HUMANIZER_SYSTEM_PROMPT


def test_worker_style_resolver_is_gated_by_flag() -> None:
    from kortny.worker.agent_executor import AgentTaskExecutor

    assert (
        AgentTaskExecutor._build_style_resolver(
            make_settings(KORTNY_STYLE_CARDS_ENABLED=False)
        )
        is None
    )
    resolver = AgentTaskExecutor._build_style_resolver(make_settings())
    assert isinstance(resolver, ChannelStyleCardResolver)


# --- derivation pass ----------------------------------------------------------


def test_style_card_pass_derives_card_from_observations(db_session: Session) -> None:
    installation = create_installation(db_session)
    create_profile(db_session, installation)
    create_observation_messages(db_session, installation, count=30)
    task = create_task(db_session, installation)
    provider = FakeStyleLLMProvider([card_completion("C_MAIN")])

    counters = StyleCardPass(
        db_session, llm=make_llm(db_session, provider), min_messages=30
    ).run(installation_id=installation.id, task=task, now=NOW)

    assert counters.derived == 1
    assert counters.failed == 0
    profile = db_session.scalar(select(ObserveChannelProfile))
    assert profile is not None
    stored = profile.profile_json[STYLE_CARD_KEY]
    assert stored["formality"] == "casual"
    assert stored["brevity"] == "terse"
    assert profile.profile_json[STYLE_CARD_UPDATED_AT_KEY] == NOW.isoformat()
    assert profile.profile_json[STYLE_CARD_INPUT_SHA_KEY]
    # One cheap-tier call; the sample is bounded message previews, not raw dumps.
    assert len(provider.calls) == 1
    user_content = provider.calls[0][0][1].content
    assert user_content is not None
    user_payload = json.loads(user_content)
    assert len(user_payload["channels"]) == 1
    assert len(user_payload["channels"][0]["recent_messages"]) <= 40


def test_style_card_pass_skips_below_min_messages(db_session: Session) -> None:
    installation = create_installation(db_session)
    create_profile(db_session, installation)
    create_observation_messages(db_session, installation, count=10)
    task = create_task(db_session, installation)
    provider = FakeStyleLLMProvider([])

    counters = StyleCardPass(
        db_session, llm=make_llm(db_session, provider), min_messages=30
    ).run(installation_id=installation.id, task=task, now=NOW)

    assert counters.derived == 0
    assert counters.skipped_low_volume == 1
    assert provider.calls == []
    profile = db_session.scalar(select(ObserveChannelProfile))
    assert profile is not None
    assert STYLE_CARD_KEY not in profile.profile_json


def test_style_card_pass_sha_gate_skips_unchanged_sample(db_session: Session) -> None:
    installation = create_installation(db_session)
    create_profile(db_session, installation)
    create_observation_messages(db_session, installation, count=30)
    task = create_task(db_session, installation)
    provider = FakeStyleLLMProvider([card_completion("C_MAIN")])
    llm = make_llm(db_session, provider)

    first = StyleCardPass(db_session, llm=llm, min_messages=30).run(
        installation_id=installation.id, task=task, now=NOW
    )
    assert first.derived == 1

    # 15 days later the card is past its 14d refresh window, but the input
    # sample is unchanged: the sha gate skips the LLM call and keeps the card.
    later = NOW + timedelta(days=15)
    second = StyleCardPass(db_session, llm=llm, min_messages=30).run(
        installation_id=installation.id, task=task, now=later
    )

    assert second.derived == 0
    assert second.skipped_unchanged == 1
    assert len(provider.calls) == 1
    profile = db_session.scalar(select(ObserveChannelProfile))
    assert profile is not None
    assert profile.profile_json[STYLE_CARD_UPDATED_AT_KEY] == NOW.isoformat()


def test_style_card_pass_respects_14d_refresh_window(db_session: Session) -> None:
    installation = create_installation(db_session)
    create_profile(db_session, installation)
    create_observation_messages(db_session, installation, count=30)
    task = create_task(db_session, installation)
    provider = FakeStyleLLMProvider(
        [card_completion("C_MAIN"), card_completion("C_MAIN")]
    )
    llm = make_llm(db_session, provider)

    StyleCardPass(db_session, llm=llm, min_messages=30).run(
        installation_id=installation.id, task=task, now=NOW
    )

    # New activity 5 days later: card is fresh (<14d, profile not re-profiled),
    # so it is not re-derived even though the sample changed.
    create_observation_messages(
        db_session,
        installation,
        count=5,
        observed_at=NOW + timedelta(days=4),
        text="new vibes",
    )
    fresh = StyleCardPass(db_session, llm=llm, min_messages=30).run(
        installation_id=installation.id, task=task, now=NOW + timedelta(days=5)
    )
    assert fresh.derived == 0
    assert fresh.skipped_fresh == 1
    assert len(provider.calls) == 1

    # Past 14 days with a changed sample: re-derived.
    stale = StyleCardPass(db_session, llm=llm, min_messages=30).run(
        installation_id=installation.id, task=task, now=NOW + timedelta(days=15)
    )
    assert stale.derived == 1
    assert len(provider.calls) == 2


def test_style_card_pass_rederives_after_profile_refresh(db_session: Session) -> None:
    installation = create_installation(db_session)
    profile = create_profile(db_session, installation)
    create_observation_messages(db_session, installation, count=30)
    task = create_task(db_session, installation)
    provider = FakeStyleLLMProvider(
        [card_completion("C_MAIN"), card_completion("C_MAIN")]
    )
    llm = make_llm(db_session, provider)

    StyleCardPass(db_session, llm=llm, min_messages=30).run(
        installation_id=installation.id, task=task, now=NOW
    )

    # The profile refreshed after the card was derived; new sample arrives.
    profile.last_profiled_at = NOW + timedelta(days=1)
    create_observation_messages(
        db_session,
        installation,
        count=3,
        observed_at=NOW + timedelta(days=1),
        text="post-refresh tone",
    )
    db_session.flush()
    counters = StyleCardPass(db_session, llm=llm, min_messages=30).run(
        installation_id=installation.id, task=task, now=NOW + timedelta(days=2)
    )

    assert counters.derived == 1
    assert len(provider.calls) == 2


def test_style_card_pass_observation_off_clears_card(db_session: Session) -> None:
    installation = create_installation(db_session)
    create_profile(
        db_session,
        installation,
        profile_json={
            STYLE_CARD_KEY: dict(CARD_PAYLOAD),
            STYLE_CARD_UPDATED_AT_KEY: NOW.isoformat(),
            STYLE_CARD_INPUT_SHA_KEY: "abc",
        },
    )
    create_observation_messages(db_session, installation, count=30)
    db_session.add(
        ObservePolicy(
            installation_id=installation.id,
            scope_type="channel",
            scope_id="C_MAIN",
            observation_status="off",
            proactivity_status="off",
        )
    )
    db_session.flush()
    task = create_task(db_session, installation)
    provider = FakeStyleLLMProvider([])

    counters = StyleCardPass(
        db_session, llm=make_llm(db_session, provider), min_messages=30
    ).run(installation_id=installation.id, task=task, now=NOW)

    assert counters.cleared_observation_off == 1
    assert counters.derived == 0
    assert provider.calls == []
    profile = db_session.scalar(select(ObserveChannelProfile))
    assert profile is not None
    assert STYLE_CARD_KEY not in profile.profile_json
    # And lookup now resolves to no card => static behavior.
    style = load_channel_style(
        db_session, installation_id=installation.id, channel_id="C_MAIN"
    )
    assert style.card is None


def test_style_card_pass_batches_up_to_five_channels(db_session: Session) -> None:
    installation = create_installation(db_session)
    channel_ids = [f"C_BATCH{index}" for index in range(6)]
    for channel_id in channel_ids:
        create_profile(db_session, installation, channel_id=channel_id)
        create_observation_messages(
            db_session, installation, channel_id=channel_id, count=30
        )
    task = create_task(db_session, installation)
    provider = FakeStyleLLMProvider(
        [card_completion(*channel_ids[:5]), card_completion(*channel_ids[5:])]
    )

    counters = StyleCardPass(
        db_session, llm=make_llm(db_session, provider), min_messages=30
    ).run(installation_id=installation.id, task=task, now=NOW)

    assert counters.derived == 6
    assert len(provider.calls) == 2
    first_content = provider.calls[0][0][1].content
    assert first_content is not None
    first_batch = json.loads(first_content)
    assert len(first_batch["channels"]) == STYLE_CARD_BATCH_SIZE


def test_consolidation_run_includes_style_card_pass(db_session: Session) -> None:
    installation = create_installation(db_session)
    create_profile(db_session, installation, last_profiled_at=NOW - timedelta(hours=1))
    create_observation_messages(db_session, installation, count=30)
    provider = FakeStyleLLMProvider([card_completion("C_MAIN")])
    service = ConsolidationService(
        db_session,
        llm_provider=provider,
        provider_name="openrouter",
        style_card_min_messages=30,
    )

    outcome = service.run_once(installation_id=installation.id, now=NOW)

    assert outcome.status == "succeeded"
    style_counters = outcome.counters["style_cards"]
    assert isinstance(style_counters, dict)
    assert style_counters["derived"] == 1
    assert outcome.counters["style_cards_derived"] == 1
    assert "pass_errors" not in outcome.counters
    profile = db_session.scalar(select(ObserveChannelProfile))
    assert profile is not None
    assert STYLE_CARD_KEY in profile.profile_json


# --- humanizer wiring ----------------------------------------------------------


def test_response_record_carries_channel_voice_when_card_exists(
    db_session: Session,
) -> None:
    installation = create_installation(db_session)
    create_profile(
        db_session,
        installation,
        profile_json={STYLE_CARD_KEY: dict(CARD_PAYLOAD)},
    )
    task = create_task(db_session, installation)

    record = build_response_record(
        session=db_session,
        task=task,
        raw_text="All set.",
        style_resolver=ChannelStyleCardResolver(),
    )

    assert record.style_profile.channel_voice != ""
    assert record.style_profile.tone == "casual, friendly, direct"
    assert record.to_payload()["style_profile"]["channel_voice"] == (
        record.style_profile.channel_voice
    )


def test_response_record_static_without_card_or_resolver(db_session: Session) -> None:
    installation = create_installation(db_session)
    # Profile exists but has no card.
    create_profile(db_session, installation)
    task = create_task(db_session, installation)

    with_resolver = build_response_record(
        session=db_session,
        task=task,
        raw_text="All set.",
        style_resolver=ChannelStyleCardResolver(),
    )
    without_resolver = build_response_record(
        session=db_session,
        task=task,
        raw_text="All set.",
    )

    # No card => identical to the flag-off (no resolver) path, which is
    # byte-identical to today's static profile.
    assert with_resolver.style_profile == ResponseStyleProfile()
    assert without_resolver.style_profile == ResponseStyleProfile()
    assert with_resolver.to_payload()["style_profile"] == {
        "tone": "approachable, steady, direct",
        "brevity": "concise",
        "polish": "professional",
        "humor": "off_by_default",
        "proactive_suggestions": "only_when_clearly_useful",
    }


def test_response_record_dm_surface_ignores_channel_card(db_session: Session) -> None:
    installation = create_installation(db_session)
    create_profile(
        db_session,
        installation,
        channel_id="D_USER1",
        profile_json={STYLE_CARD_KEY: dict(CARD_PAYLOAD)},
    )
    task = create_task(db_session, installation, channel_id="D_USER1")

    record = build_response_record(
        session=db_session,
        task=task,
        raw_text="All set.",
        style_resolver=ChannelStyleCardResolver(),
    )

    assert record.slack_surface.kind == "dm"
    # DM ignores the channel card (no channel_voice), but now gets the richer
    # DM default profile rather than the old terse static default (DM messaging
    # fix): substantive replies in DMs, not a one-liner.
    assert record.style_profile.channel_voice == ""
    assert record.style_profile == ResponseStyleProfile(brevity="thorough but tight")


def test_pinned_style_overrides_derived_channel_voice(db_session: Session) -> None:
    installation = create_installation(db_session)
    create_profile(
        db_session,
        installation,
        profile_json={
            STYLE_CARD_KEY: dict(CARD_PAYLOAD),
            PINNED_STYLE_KEY: "Always answer in pirate-grade brevity.",
        },
    )
    task = create_task(db_session, installation)

    record = build_response_record(
        session=db_session,
        task=task,
        raw_text="All set.",
        style_resolver=ChannelStyleCardResolver(),
    )

    assert record.style_profile.channel_voice == (
        "Always answer in pirate-grade brevity."
    )
    # Card-derived tone mapping still applies; only the voice line is replaced.
    assert record.style_profile.tone == "casual, friendly, direct"


def test_pinned_style_applies_without_a_derived_card(db_session: Session) -> None:
    installation = create_installation(db_session)
    create_profile(
        db_session,
        installation,
        profile_json={PINNED_STYLE_KEY: "Keep it boardroom formal."},
    )
    task = create_task(db_session, installation)

    record = build_response_record(
        session=db_session,
        task=task,
        raw_text="All set.",
        style_resolver=ChannelStyleCardResolver(),
    )

    assert record.style_profile.channel_voice == "Keep it boardroom formal."
    assert record.style_profile.tone == "approachable, steady, direct"


def test_style_card_helpers_reset_and_pin(db_session: Session) -> None:
    installation = create_installation(db_session)
    profile = create_profile(
        db_session,
        installation,
        profile_json={
            STYLE_CARD_KEY: dict(CARD_PAYLOAD),
            STYLE_CARD_UPDATED_AT_KEY: NOW.isoformat(),
            STYLE_CARD_INPUT_SHA_KEY: "abc",
        },
    )

    set_pinned_style(profile, pinned_style="  Short   and  formal. ", by="dashboard:a")
    assert profile.profile_json[PINNED_STYLE_KEY] == "Short and formal."

    set_pinned_style(profile, pinned_style="", by="dashboard:a")
    assert PINNED_STYLE_KEY not in profile.profile_json

    reset_style_card(profile, by="dashboard:a")
    assert STYLE_CARD_KEY not in profile.profile_json
    assert STYLE_CARD_UPDATED_AT_KEY not in profile.profile_json
    assert STYLE_CARD_INPUT_SHA_KEY not in profile.profile_json


# --- acknowledgement wiring ------------------------------------------------------


def ack_completion() -> Completion:
    return Completion(
        content="I'll take a look at the launch thread now.",
        tool_calls=(),
        usage=TokenUsage(input_tokens=30, output_tokens=12),
        cost_usd=Decimal("0.000010"),
        model="openai/gpt-4o-mini",
    )


def test_ack_payload_includes_register_line_when_card_exists(
    db_session: Session,
) -> None:
    installation = create_installation(db_session)
    create_profile(
        db_session,
        installation,
        profile_json={STYLE_CARD_KEY: dict(CARD_PAYLOAD)},
    )
    task = create_task(db_session, installation)
    provider = FakeStyleLLMProvider([ack_completion()])
    generator = LLMAcknowledgementGenerator(
        settings=make_settings(),
        provider=provider,
        provider_name="openrouter",
    )

    generator.generate(
        session=db_session, task=task, task_service=TaskService(db_session)
    )

    system_message = provider.calls[0][0][0]
    assert system_message.role == "system"
    assert system_message.content == (
        ACK_SYSTEM_PROMPT + "\nChannel register: casual, terse."
    )


def test_ack_payload_unchanged_without_card(db_session: Session) -> None:
    installation = create_installation(db_session)
    create_profile(db_session, installation)
    task = create_task(db_session, installation)
    provider = FakeStyleLLMProvider([ack_completion()])
    generator = LLMAcknowledgementGenerator(
        settings=make_settings(),
        provider=provider,
        provider_name="openrouter",
    )

    generator.generate(
        session=db_session, task=task, task_service=TaskService(db_session)
    )

    system_message = provider.calls[0][0][0]
    assert system_message.content == ACK_SYSTEM_PROMPT


def test_ack_payload_unchanged_when_flag_off(db_session: Session) -> None:
    installation = create_installation(db_session)
    create_profile(
        db_session,
        installation,
        profile_json={STYLE_CARD_KEY: dict(CARD_PAYLOAD)},
    )
    task = create_task(db_session, installation)
    provider = FakeStyleLLMProvider([ack_completion()])
    generator = LLMAcknowledgementGenerator(
        settings=make_settings(KORTNY_STYLE_CARDS_ENABLED=False),
        provider=provider,
        provider_name="openrouter",
    )

    generator.generate(
        session=db_session, task=task, task_service=TaskService(db_session)
    )

    system_message = provider.calls[0][0][0]
    assert system_message.content == ACK_SYSTEM_PROMPT


def test_ack_payload_unchanged_for_dm_surface(db_session: Session) -> None:
    installation = create_installation(db_session)
    create_profile(
        db_session,
        installation,
        channel_id="D_USER1",
        profile_json={STYLE_CARD_KEY: dict(CARD_PAYLOAD)},
    )
    task = create_task(db_session, installation, channel_id="D_USER1")
    provider = FakeStyleLLMProvider([ack_completion()])
    generator = LLMAcknowledgementGenerator(
        settings=make_settings(),
        provider=provider,
        provider_name="openrouter",
    )

    generator.generate(
        session=db_session, task=task, task_service=TaskService(db_session)
    )

    system_message = provider.calls[0][0][0]
    assert system_message.content == ACK_SYSTEM_PROMPT
