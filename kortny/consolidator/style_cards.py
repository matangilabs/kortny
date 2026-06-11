"""Style-card derivation pass (HIG-226, Slice C).

Runs after the hygiene/profile-refresh pass: for channels with an active
profile and enough recent observed messages, one cheap-tier LLM call (batched
up to 5 channels per call) produces a structured register card stored in
``ObserveChannelProfile.profile_json["style_card"]``. The card captures the
channel's collective register — formality, brevity, emoji/punctuation norms,
threading — never an individual's voice. A sha gate over the input sample
skips channels whose observable register input has not changed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from kortny.db.models import (
    ObservationEvent,
    ObserveChannelProfile,
    ObservePolicy,
    Task,
)
from kortny.llm import ChatMessage, LLMService
from kortny.observe.style_cards import (
    STYLE_CARD_INPUT_SHA_KEY,
    STYLE_CARD_KEY,
    STYLE_CARD_UPDATED_AT_KEY,
    ChannelStyleCard,
    parse_style_card,
    reset_style_card,
    style_card_from_profile,
)
from kortny.tools.types import JsonObject

logger = logging.getLogger(__name__)

STYLE_CARD_PROMPT_NAME = "kortny.style_card_extractor"
STYLE_CARD_RESPONSE_FORMAT: JsonObject = {"type": "json_object"}
STYLE_CARD_SYSTEM_PROMPT = (
    "You analyze the communication register of Slack channels for Kortny, a "
    "Slack-native AI coworker. For each channel you receive a profile summary "
    "and a bounded sample of recent messages. Describe the channel's "
    "collective register only — never the style of any individual person. "
    "common_phrases must be channel idioms shared by the group, not personal "
    "catchphrases. Return JSON only: "
    '{"cards":[{"channel_id":"...","formality":"casual|neutral|formal",'
    '"brevity":"terse|moderate|expansive","emoji_culture":"none|light|heavy",'
    '"punctuation":"relaxed|standard","common_phrases":["up to 5 channel idioms"],'
    '"threading_norm":"threads_heavy|mixed|top_level",'
    '"notes":"one sentence of register guidance"}]} '
    "— include exactly one card per input channel."
)

DEFAULT_STYLE_CARD_MIN_MESSAGES = 30
STYLE_CARD_WINDOW = timedelta(days=30)
STYLE_CARD_MAX_AGE = timedelta(days=14)
STYLE_CARD_BATCH_SIZE = 5
STYLE_CARD_SAMPLE_LIMIT = 40
STYLE_CARD_SAMPLE_TEXT_MAX_CHARS = 280


@dataclass(frozen=True, slots=True)
class StyleCardCounters:
    """Per-run counters for the style-card derivation pass."""

    derived: int = 0
    skipped_fresh: int = 0
    skipped_unchanged: int = 0
    skipped_low_volume: int = 0
    cleared_observation_off: int = 0
    failed: int = 0

    def to_payload(self) -> dict[str, int]:
        return {
            "derived": self.derived,
            "skipped_fresh": self.skipped_fresh,
            "skipped_unchanged": self.skipped_unchanged,
            "skipped_low_volume": self.skipped_low_volume,
            "cleared_observation_off": self.cleared_observation_off,
            "failed": self.failed,
        }


@dataclass(frozen=True, slots=True)
class _PendingChannel:
    profile: ObserveChannelProfile
    sample_sha: str
    payload: JsonObject


class StyleCardPass:
    """Derive channel style cards from retained observation data."""

    def __init__(
        self,
        session: Session,
        *,
        llm: LLMService | None,
        min_messages: int = DEFAULT_STYLE_CARD_MIN_MESSAGES,
    ) -> None:
        self.session = session
        self.llm = llm
        self.min_messages = min_messages

    def run(
        self,
        *,
        installation_id: uuid.UUID,
        task: Task,
        now: datetime | None = None,
    ) -> StyleCardCounters:
        effective_now = now or datetime.now(UTC)
        profiles = list(
            self.session.scalars(
                select(ObserveChannelProfile)
                .where(
                    ObserveChannelProfile.installation_id == installation_id,
                    ObserveChannelProfile.profile_status == "active",
                )
                .order_by(ObserveChannelProfile.channel_id)
            )
        )
        derived = 0
        skipped_fresh = 0
        skipped_unchanged = 0
        skipped_low_volume = 0
        cleared_observation_off = 0
        failed = 0
        pending: list[_PendingChannel] = []
        for profile in profiles:
            if not self._observation_allowed(installation_id, profile.channel_id):
                if style_card_from_profile(profile.profile_json) is not None:
                    reset_style_card(profile)
                    profile.updated_at = effective_now
                    cleared_observation_off += 1
                continue
            card = style_card_from_profile(profile.profile_json)
            if card is not None and not self._needs_refresh(profile, effective_now):
                skipped_fresh += 1
                continue
            sample = self._sample_texts(
                installation_id, profile.channel_id, effective_now
            )
            message_count = self._recent_message_count(
                installation_id, profile.channel_id, effective_now
            )
            if message_count < self.min_messages:
                skipped_low_volume += 1
                continue
            sample_sha = _sample_sha(profile.summary, sample)
            stored_sha = (
                profile.profile_json.get(STYLE_CARD_INPUT_SHA_KEY)
                if isinstance(profile.profile_json, dict)
                else None
            )
            if card is not None and stored_sha == sample_sha:
                skipped_unchanged += 1
                continue
            pending.append(
                _PendingChannel(
                    profile=profile,
                    sample_sha=sample_sha,
                    payload={
                        "channel_id": profile.channel_id,
                        "profile_summary": (profile.summary or "")[:1200],
                        "recent_messages": list(sample),
                    },
                )
            )

        if pending and self.llm is not None:
            for start in range(0, len(pending), STYLE_CARD_BATCH_SIZE):
                batch = pending[start : start + STYLE_CARD_BATCH_SIZE]
                cards = self._extract_cards(task, batch)
                for item in batch:
                    card = cards.get(item.profile.channel_id)
                    if card is None:
                        failed += 1
                        continue
                    _store_card(
                        item.profile,
                        card=card,
                        sample_sha=item.sample_sha,
                        now=effective_now,
                    )
                    derived += 1
        self.session.flush()
        return StyleCardCounters(
            derived=derived,
            skipped_fresh=skipped_fresh,
            skipped_unchanged=skipped_unchanged,
            skipped_low_volume=skipped_low_volume,
            cleared_observation_off=cleared_observation_off,
            failed=failed,
        )

    def _observation_allowed(self, installation_id: uuid.UUID, channel_id: str) -> bool:
        policy = self.session.scalar(
            select(ObservePolicy).where(
                ObservePolicy.installation_id == installation_id,
                ObservePolicy.scope_type == "channel",
                ObservePolicy.scope_id == channel_id,
            )
        )
        if policy is None:
            return True
        return policy.observation_status != "off" and policy.paused_at is None

    def _needs_refresh(self, profile: ObserveChannelProfile, now: datetime) -> bool:
        payload = profile.profile_json if isinstance(profile.profile_json, dict) else {}
        updated_at = _parse_timestamp(payload.get(STYLE_CARD_UPDATED_AT_KEY))
        if updated_at is None:
            return True
        if updated_at < now - STYLE_CARD_MAX_AGE:
            return True
        last_profiled = profile.last_profiled_at
        if last_profiled is not None:
            if last_profiled.tzinfo is None:
                last_profiled = last_profiled.replace(tzinfo=UTC)
            if updated_at < last_profiled:
                return True
        return False

    def _recent_message_count(
        self,
        installation_id: uuid.UUID,
        channel_id: str,
        now: datetime,
    ) -> int:
        return int(
            self.session.scalar(
                select(func.count())
                .select_from(ObservationEvent)
                .where(
                    ObservationEvent.installation_id == installation_id,
                    ObservationEvent.channel_id == channel_id,
                    ObservationEvent.event_type == "message",
                    ObservationEvent.observed_at > now - STYLE_CARD_WINDOW,
                )
            )
            or 0
        )

    def _sample_texts(
        self,
        installation_id: uuid.UUID,
        channel_id: str,
        now: datetime,
    ) -> tuple[str, ...]:
        rows = list(
            self.session.scalars(
                select(ObservationEvent.text_preview)
                .where(
                    ObservationEvent.installation_id == installation_id,
                    ObservationEvent.channel_id == channel_id,
                    ObservationEvent.event_type == "message",
                    ObservationEvent.observed_at > now - STYLE_CARD_WINDOW,
                    ObservationEvent.text_preview.is_not(None),
                )
                .order_by(ObservationEvent.observed_at.desc())
                .limit(STYLE_CARD_SAMPLE_LIMIT)
            )
        )
        texts = [
            text.strip()[:STYLE_CARD_SAMPLE_TEXT_MAX_CHARS]
            for text in rows
            if isinstance(text, str) and text.strip()
        ]
        texts.reverse()
        return tuple(texts)

    def _extract_cards(
        self,
        task: Task,
        batch: list[_PendingChannel],
    ) -> dict[str, ChannelStyleCard]:
        assert self.llm is not None
        try:
            completion = self.llm.complete(
                task_id=task.id,
                messages=(
                    ChatMessage(role="system", content=STYLE_CARD_SYSTEM_PROMPT),
                    ChatMessage(
                        role="user",
                        content=json.dumps(
                            {"channels": [item.payload for item in batch]},
                            separators=(",", ":"),
                            default=str,
                        ),
                    ),
                ),
                response_format=STYLE_CARD_RESPONSE_FORMAT,
                prompt_name=STYLE_CARD_PROMPT_NAME,
            )
        except Exception:
            logger.exception("style card extraction call failed task_id=%s", task.id)
            return {}
        try:
            parsed = json.loads(completion.content or "{}")
        except json.JSONDecodeError:
            return {}
        raw_cards = parsed.get("cards") if isinstance(parsed, dict) else None
        if not isinstance(raw_cards, list):
            return {}
        cards: dict[str, ChannelStyleCard] = {}
        for raw in raw_cards:
            if not isinstance(raw, dict):
                continue
            channel_id = raw.get("channel_id")
            card = parse_style_card(raw)
            if isinstance(channel_id, str) and channel_id and card is not None:
                cards[channel_id] = card
        return cards


def _store_card(
    profile: ObserveChannelProfile,
    *,
    card: ChannelStyleCard,
    sample_sha: str,
    now: datetime,
) -> None:
    payload = (
        dict(profile.profile_json) if isinstance(profile.profile_json, dict) else {}
    )
    payload[STYLE_CARD_KEY] = card.to_payload()
    payload[STYLE_CARD_UPDATED_AT_KEY] = now.isoformat()
    payload[STYLE_CARD_INPUT_SHA_KEY] = sample_sha
    profile.profile_json = payload
    profile.updated_at = now


def _parse_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


def _sample_sha(summary: str | None, sample: tuple[str, ...]) -> str:
    digest_input = json.dumps(
        {"summary": summary or "", "messages": list(sample)},
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(digest_input.encode("utf-8")).hexdigest()
