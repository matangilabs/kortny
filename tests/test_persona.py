"""Agent persona naming: display-name derivation and prompt personalization.

Self-hosters brand their install via SLACK_APP_NAME, so nothing the agent says
about itself may be a hardcoded "Kortny". These tests pin the two pieces that
guarantee that: the derived display name and the placeholder substitution. They
also pin that the default config stays byte-identical to the pre-feature "Kortny"
output, which is what keeps the rest of the suite's exact-text assertions valid.
"""

from __future__ import annotations

import pytest

from kortny.agent.coordinator import DEFAULT_SYSTEM_PROMPT
from kortny.config import Settings
from kortny.config.settings import LLMProvider as SettingsLLMProvider
from kortny.persona import AGENT_NAME_TOKEN, personalize
from kortny.slack.humanizer import RESPONSE_HUMANIZER_SYSTEM_PROMPT


def _settings(app_name: str | None = None) -> Settings:
    payload = {
        "SLACK_BOT_TOKEN": "xoxb-test",
        "SLACK_APP_TOKEN": "xapp-test",
        "SLACK_SIGNING_SECRET": "signing-secret",
        "LLM_PROVIDER": SettingsLLMProvider.openrouter,
        "LLM_API_KEY": "test-key",
        "LLM_MODEL": "openai/gpt-test",
        "COMPOSIO_API_KEY": "composio-key",
        "POSTGRES_URL": "postgresql://kortny:kortny@localhost:5432/kortny_test",
    }
    if app_name is not None:
        payload["SLACK_APP_NAME"] = app_name
    return Settings.model_validate(payload)


@pytest.mark.parametrize(
    ("app_name", "expected"),
    [
        (None, "Kortny"),  # default "kortny" reads as a name
        ("kortny", "Kortny"),
        ("acme", "Acme"),
        ("Acme Bot", "Acme Bot"),  # authored casing is preserved
        ("ACME", "ACME"),  # already has casing — left alone
        ("data-genie", "Data-Genie"),
    ],
)
def test_agent_display_name_casing(app_name: str | None, expected: str) -> None:
    assert _settings(app_name).agent_display_name == expected


def test_personalize_substitutes_token() -> None:
    template = f"You are {AGENT_NAME_TOKEN}, a coworker. Ask {AGENT_NAME_TOKEN}."
    assert personalize(template, "Acme") == "You are Acme, a coworker. Ask Acme."


def test_personalize_leaves_tokenless_text_untouched() -> None:
    text = 'A prompt with literal braces {"message": "hi"} and no placeholder.'
    assert personalize(text, "Acme") is text


def test_personalize_default_name_is_byte_identical_to_legacy() -> None:
    # Resolving the token with the default display name must reproduce the exact
    # legacy wording, so existing exact-text assertions elsewhere keep holding.
    default_name = _settings().agent_display_name
    assert AGENT_NAME_TOKEN in DEFAULT_SYSTEM_PROMPT
    assert "Kortny" not in DEFAULT_SYSTEM_PROMPT
    rendered = personalize(DEFAULT_SYSTEM_PROMPT, default_name)
    assert "You are Kortny, a Slack-native AI coworker." in rendered
    assert AGENT_NAME_TOKEN not in rendered


def test_humanizer_prompt_personalizes_to_configured_name() -> None:
    assert AGENT_NAME_TOKEN in RESPONSE_HUMANIZER_SYSTEM_PROMPT
    assert "Kortny" not in RESPONSE_HUMANIZER_SYSTEM_PROMPT
    rendered = personalize(RESPONSE_HUMANIZER_SYSTEM_PROMPT, "Acme")
    assert "Write as Acme, one Slack-native coworker." in rendered
    assert AGENT_NAME_TOKEN not in rendered
