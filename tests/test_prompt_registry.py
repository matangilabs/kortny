"""Tests for the prompt registry (HIG-203)."""

from __future__ import annotations

from kortny.prompts import PROMPT_REGISTRY, prompt_version, register_prompt

_INTENT = "kortny.intent_classifier"


def test_intent_classifier_is_registered_with_version() -> None:
    assert prompt_version(_INTENT) == "2"
    definition = PROMPT_REGISTRY[_INTENT]
    assert definition.subsystem == "intent"
    assert definition.description


def test_unknown_prompt_has_no_version() -> None:
    assert prompt_version("kortny.not_a_real_prompt") is None
    assert prompt_version(None) is None


def test_registered_names_are_namespaced() -> None:
    assert PROMPT_REGISTRY, "registry should not be empty"
    for name, definition in PROMPT_REGISTRY.items():
        assert name.startswith("kortny."), name
        assert definition.name == name
        assert definition.version


def test_register_prompt_is_idempotent_by_name() -> None:
    register_prompt(
        name="kortny.test_tmp", subsystem="test", version="1", description="x"
    )
    register_prompt(
        name="kortny.test_tmp", subsystem="test", version="2", description="y"
    )
    assert prompt_version("kortny.test_tmp") == "2"
    del PROMPT_REGISTRY["kortny.test_tmp"]
