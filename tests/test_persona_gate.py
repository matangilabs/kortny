"""Tests for the deterministic persona-relevance gate (HIG-277)."""

from __future__ import annotations

import pytest

from kortny.intent.persona_gate import persona_relevant_for_text


@pytest.mark.parametrize(
    "text",
    [
        "what's on my plate today?",
        "what should I focus on this week",
        "show me my open PRs",
        "what are my issues in Linear",
        "anything assigned to me?",
        "check my inbox",
        "what's on my calendar tomorrow",
        "my pipeline status",
    ],
)
def test_role_relative_asks_are_persona_relevant(text: str) -> None:
    assert persona_relevant_for_text(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "what is the capital of France?",
        "summarize this channel",
        "create a Linear issue for the launch",
        "draft a report on the EV market",
        "who won the world cup in 2022",
        "explain how RAG works",
        "",
    ],
)
def test_factual_or_neutral_asks_are_not_persona_relevant(text: str) -> None:
    assert persona_relevant_for_text(text) is False


def test_case_insensitive() -> None:
    assert persona_relevant_for_text("WHAT'S ON MY PLATE") is True
