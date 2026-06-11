"""Channel style cards: learned per-channel register descriptors (HIG-226).

A style card is a small set of explicit register dimensions (formality,
brevity, emoji/punctuation norms, threading) the consolidator derives from
channel-scoped observation data Kortny already retains under ObservePolicy.
Cards describe the channel's collective register — never individual people —
and live inside ``ObserveChannelProfile.profile_json``; raw message samples
never leave the derivation pass.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from kortny.db.models import ObserveChannelProfile

STYLE_CARD_KEY = "style_card"
STYLE_CARD_UPDATED_AT_KEY = "style_card_updated_at"
STYLE_CARD_INPUT_SHA_KEY = "style_card_input_sha"
PINNED_STYLE_KEY = "pinned_style"

STYLE_CARD_FORMALITY_VALUES = frozenset({"casual", "neutral", "formal"})
STYLE_CARD_BREVITY_VALUES = frozenset({"terse", "moderate", "expansive"})
STYLE_CARD_EMOJI_VALUES = frozenset({"none", "light", "heavy"})
STYLE_CARD_PUNCTUATION_VALUES = frozenset({"relaxed", "standard"})
STYLE_CARD_THREADING_VALUES = frozenset({"threads_heavy", "mixed", "top_level"})
STYLE_CARD_MAX_PHRASES = 5
STYLE_CARD_NOTES_MAX_CHARS = 240
PINNED_STYLE_MAX_CHARS = 240


@dataclass(frozen=True, slots=True)
class ChannelStyleCard:
    """Structured register descriptors for one channel."""

    formality: str
    brevity: str
    emoji_culture: str
    punctuation: str
    common_phrases: tuple[str, ...]
    threading_norm: str
    notes: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "formality": self.formality,
            "brevity": self.brevity,
            "emoji_culture": self.emoji_culture,
            "punctuation": self.punctuation,
            "common_phrases": list(self.common_phrases),
            "threading_norm": self.threading_norm,
            "notes": self.notes,
        }


@dataclass(frozen=True, slots=True)
class ChannelStyle:
    """Style card plus operator override for one channel."""

    card: ChannelStyleCard | None
    pinned_style: str | None


def parse_style_card(value: object) -> ChannelStyleCard | None:
    """Validate a stored or model-produced style card; None when unusable."""

    if not isinstance(value, dict):
        return None
    formality = value.get("formality")
    brevity = value.get("brevity")
    emoji_culture = value.get("emoji_culture")
    punctuation = value.get("punctuation")
    threading_norm = value.get("threading_norm")
    if formality not in STYLE_CARD_FORMALITY_VALUES:
        return None
    if brevity not in STYLE_CARD_BREVITY_VALUES:
        return None
    if emoji_culture not in STYLE_CARD_EMOJI_VALUES:
        return None
    if punctuation not in STYLE_CARD_PUNCTUATION_VALUES:
        return None
    if threading_norm not in STYLE_CARD_THREADING_VALUES:
        return None
    raw_phrases = value.get("common_phrases")
    phrases: list[str] = []
    if isinstance(raw_phrases, list):
        for item in raw_phrases:
            if isinstance(item, str) and item.strip():
                phrases.append(item.strip()[:80])
            if len(phrases) >= STYLE_CARD_MAX_PHRASES:
                break
    raw_notes = value.get("notes")
    notes = (
        raw_notes.strip()[:STYLE_CARD_NOTES_MAX_CHARS]
        if isinstance(raw_notes, str)
        else ""
    )
    return ChannelStyleCard(
        formality=formality,
        brevity=brevity,
        emoji_culture=emoji_culture,
        punctuation=punctuation,
        common_phrases=tuple(phrases),
        threading_norm=threading_norm,
        notes=notes,
    )


def style_card_from_profile(profile_json: object) -> ChannelStyleCard | None:
    """Read the style card out of a profile_json payload."""

    if not isinstance(profile_json, dict):
        return None
    return parse_style_card(profile_json.get(STYLE_CARD_KEY))


def pinned_style_from_profile(profile_json: object) -> str | None:
    """Read the operator pinned-style override out of a profile_json payload."""

    if not isinstance(profile_json, dict):
        return None
    pinned = profile_json.get(PINNED_STYLE_KEY)
    if isinstance(pinned, str) and pinned.strip():
        return pinned.strip()[:PINNED_STYLE_MAX_CHARS]
    return None


def load_channel_style(
    session: Session,
    *,
    installation_id: uuid.UUID,
    channel_id: str,
) -> ChannelStyle:
    """Load the active channel's style card + pinned override, if any."""

    profile = session.scalar(
        select(ObserveChannelProfile).where(
            ObserveChannelProfile.installation_id == installation_id,
            ObserveChannelProfile.channel_id == channel_id,
            ObserveChannelProfile.profile_status == "active",
        )
    )
    if profile is None:
        return ChannelStyle(card=None, pinned_style=None)
    return ChannelStyle(
        card=style_card_from_profile(profile.profile_json),
        pinned_style=pinned_style_from_profile(profile.profile_json),
    )


def reset_style_card(profile: ObserveChannelProfile, *, by: str | None = None) -> None:
    """Clear the derived style card; the consolidator re-derives it later."""

    payload = (
        dict(profile.profile_json) if isinstance(profile.profile_json, dict) else {}
    )
    payload.pop(STYLE_CARD_KEY, None)
    payload.pop(STYLE_CARD_UPDATED_AT_KEY, None)
    payload.pop(STYLE_CARD_INPUT_SHA_KEY, None)
    if by:
        payload["style_card_reset_by"] = by
    profile.profile_json = payload


def set_pinned_style(
    profile: ObserveChannelProfile,
    *,
    pinned_style: str,
    by: str | None = None,
) -> None:
    """Set or clear the operator pinned-style override."""

    payload = (
        dict(profile.profile_json) if isinstance(profile.profile_json, dict) else {}
    )
    normalized = " ".join(pinned_style.split())[:PINNED_STYLE_MAX_CHARS]
    if normalized:
        payload[PINNED_STYLE_KEY] = normalized
        if by:
            payload["pinned_style_set_by"] = by
    else:
        payload.pop(PINNED_STYLE_KEY, None)
        payload.pop("pinned_style_set_by", None)
    profile.profile_json = payload
