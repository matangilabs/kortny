"""Tests for the presentation hint parser + Block Kit renderer (HIG-255).

Pure (no DB, no LLM): exercises parse_presentation and render_blocks directly.
"""

from __future__ import annotations

from dataclasses import dataclass

from kortny.slack import blockkit
from kortny.slack.presentation import (
    CardItem,
    CardsElement,
    ContextElement,
    DividerElement,
    FieldItem,
    FieldsElement,
    HeaderElement,
    ItemsElement,
    ListItem,
    PresentationHint,
    SourceCardItem,
    SourcesElement,
    TableElement,
    parse_presentation,
)
from kortny.slack.response_render import render_blocks
from kortny.slack.source_index import build_source_index


def _types(blocks: list[dict]) -> list[str]:
    return [b["type"] for b in blocks]


@dataclass(frozen=True)
class _FakeEvidence:
    urls: list[str] | None
    preview: str | None


# --------------------------------------------------------------------------- #
# parse_presentation — lenient
# --------------------------------------------------------------------------- #


def test_parse_keeps_known_elements() -> None:
    hint = parse_presentation(
        {
            "version": 1,
            "elements": [
                {"type": "fields", "items": [{"label": "Status", "value": "Active"}]},
                {"type": "context", "items": ["Source: Linear"]},
            ],
        }
    )
    assert hint is not None
    assert [type(e).__name__ for e in hint.elements] == [
        "FieldsElement",
        "ContextElement",
    ]


def test_parse_drops_unknown_element_but_keeps_good_ones() -> None:
    hint = parse_presentation(
        {
            "elements": [
                {"type": "chart", "data": [1, 2, 3]},  # unknown → dropped
                {"type": "fields", "items": [{"label": "A", "value": "1"}]},
            ]
        }
    )
    assert hint is not None
    assert len(hint.elements) == 1
    assert isinstance(hint.elements[0], FieldsElement)


def test_parse_drops_invalid_element_not_whole_hint() -> None:
    hint = parse_presentation(
        {
            "elements": [
                {"type": "fields", "items": []},  # invalid: min_length 1
                {"type": "context", "items": ["ok"]},
            ]
        }
    )
    assert hint is not None
    assert len(hint.elements) == 1
    assert isinstance(hint.elements[0], ContextElement)


def test_parse_returns_none_for_garbage() -> None:
    assert parse_presentation(None) is None
    assert parse_presentation({"elements": "nope"}) is None
    assert parse_presentation({"elements": []}) is None
    assert parse_presentation({"elements": [{"type": "chart"}]}) is None


def test_parse_caps_element_count() -> None:
    many = {"elements": [{"type": "context", "items": [str(i)]} for i in range(20)]}
    hint = parse_presentation(many)
    assert hint is not None
    assert len(hint.elements) == 8  # MAX_PRESENTATION_ELEMENTS


# --------------------------------------------------------------------------- #
# render_blocks — no hint (legacy behavior preserved)
# --------------------------------------------------------------------------- #


def test_no_hint_plain_prose_returns_none() -> None:
    assert render_blocks("Just a normal answer.") is None


def test_no_hint_markdown_table_wraps_in_markdown_block() -> None:
    text = "Here:\n\n| A | B |\n| --- | --- |\n| 1 | 2 |"
    blocks = render_blocks(text)
    assert blocks is not None
    assert _types(blocks) == ["markdown"]


def test_empty_message_returns_none() -> None:
    assert render_blocks("   ", PresentationHint()) is None


# --------------------------------------------------------------------------- #
# render_blocks — with hint
# --------------------------------------------------------------------------- #


def test_fields_hint_renders_voice_plus_section() -> None:
    hint = PresentationHint(
        elements=[
            FieldsElement(
                title="Schedule",
                items=[
                    FieldItem(label="Cadence", value="Daily"),
                    FieldItem(label="Next run", value="09:00"),
                ],
            )
        ]
    )
    blocks = render_blocks("Your schedule is set.", hint)
    assert blocks is not None
    assert _types(blocks) == ["markdown", "section"]
    section = blocks[1]
    assert section["text"]["text"] == "*Schedule*"
    assert len(section["fields"]) == 2


def test_table_hint_renders_native_table_on_message_surface() -> None:
    hint = PresentationHint(
        elements=[
            TableElement(
                columns=["User", "Cost"],
                rows=[["alice", "$3"], ["bob", "$5"]],
            )
        ]
    )
    blocks = render_blocks("Usage by user:", hint)
    assert blocks is not None
    assert _types(blocks) == ["markdown", "table"]
    table = blocks[1]
    # Header row + 2 data rows = 3 rows.
    assert len(table["rows"]) == 3
    assert table["rows"][0][0]["text"] == "User"


def test_table_hint_degrades_to_markdown_off_message_surface() -> None:
    hint = PresentationHint(elements=[TableElement(columns=["A"], rows=[["1"]])])
    blocks = render_blocks("x", hint, surface="modal")
    assert blocks is not None
    # No native table block off the message surface.
    assert "table" not in _types(blocks)
    assert _types(blocks) == ["markdown", "markdown"]


def test_cards_hint_renders_card_blocks() -> None:
    hint = PresentationHint(
        elements=[
            CardsElement(
                items=[
                    CardItem(title="HIG-255", body="Block Kit humanizer"),
                    CardItem(title="HIG-273", body="Governance"),
                ]
            )
        ]
    )
    blocks = render_blocks("Top issues:", hint)
    assert blocks is not None
    assert _types(blocks) == ["markdown", "card", "card"]


def test_card_with_long_body_falls_back_to_section() -> None:
    long_body = "x" * (blockkit.MAX_CARD_BODY_CHARS + 50)
    hint = PresentationHint(
        elements=[CardsElement(items=[CardItem(title="Big", body=long_body)])]
    )
    blocks = render_blocks("entity:", hint)
    assert blocks is not None
    # Card block can't hold the body → section fallback, no card block.
    assert "card" not in _types(blocks)
    assert "section" in _types(blocks)


def test_context_hint_renders_context_block() -> None:
    hint = PresentationHint(
        elements=[ContextElement(items=["Source: Linear, 8 issues", "Fresh: 2m ago"])]
    )
    blocks = render_blocks("Here you go.", hint)
    assert blocks is not None
    assert _types(blocks) == ["markdown", "context"]
    assert len(blocks[1]["elements"]) == 2


def test_oversized_prose_falls_back_to_prose_only() -> None:
    huge = "x" * (blockkit.MAX_MARKDOWN_BLOCK_CHARS + 1)
    hint = PresentationHint(elements=[ContextElement(items=["src"])])
    assert render_blocks(huge, hint) is None


def test_hint_with_all_elements_dropped_returns_none_for_plain_prose() -> None:
    # An element that renders nothing usable + plain prose → None (plain text),
    # never a lone markdown block.
    hint = parse_presentation({"elements": [{"type": "chart"}]})  # all dropped
    assert hint is None
    assert render_blocks("plain answer", hint) is None


# --------------------------------------------------------------------------- #
# items element — section + context + divider lists
# --------------------------------------------------------------------------- #


def test_items_render_section_context_divider() -> None:
    hint = PresentationHint(
        elements=[
            ItemsElement(
                items=[
                    ListItem(
                        title="HIG-276",
                        facts=[FieldItem(label="Status", value="In progress")],
                        context=["Source: Linear"],
                    ),
                    ListItem(
                        title="HIG-255", facts=[FieldItem(label="Owner", value="A")]
                    ),
                ]
            )
        ]
    )
    blocks = render_blocks("You've got 2 open items.", hint)
    assert blocks is not None
    # voice + [section, context] + divider + [section]
    assert _types(blocks) == [
        "markdown",
        "section",
        "context",
        "divider",
        "section",
    ]
    assert blocks[1]["text"]["text"].startswith("*HIG-276*")


def test_items_can_disable_dividers() -> None:
    hint = PresentationHint(
        elements=[
            ItemsElement(
                dividers=False,
                items=[ListItem(title="A"), ListItem(title="B")],
            )
        ]
    )
    blocks = render_blocks("two items", hint)
    assert blocks is not None
    assert "divider" not in _types(blocks)


# --------------------------------------------------------------------------- #
# sources element — server-bound URLs only
# --------------------------------------------------------------------------- #


def test_sources_resolve_refs_to_evidence_urls() -> None:
    index = build_source_index(
        [
            _FakeEvidence(urls=["https://nasa.gov/artemis"], preview="Mission page"),
            _FakeEvidence(urls=["https://news.example/splashdown"], preview="Report"),
        ]
    )
    hint = PresentationHint(
        elements=[
            SourcesElement(
                items=[
                    SourceCardItem(source_ref="source:0", body="Official page"),
                    SourceCardItem(source_ref="source:1"),
                ]
            )
        ]
    )
    blocks = render_blocks("Here's what I drew on:", hint, source_index=index)
    assert blocks is not None
    # carousel of 2 source cards after the voice block
    assert _types(blocks) == ["markdown", "carousel"]
    cards = blocks[1]["elements"]
    assert len(cards) == 2
    # The real URL is bound from evidence, never the LLM.
    assert "https://nasa.gov/artemis" in cards[0]["body"]["text"]


def test_sources_drop_unresolved_ref() -> None:
    index = build_source_index([_FakeEvidence(urls=["https://a.test/x"], preview=None)])
    hint = PresentationHint(
        elements=[
            SourcesElement(
                items=[
                    SourceCardItem(source_ref="source:0"),
                    SourceCardItem(source_ref="source:99"),  # unresolved → dropped
                ]
            )
        ]
    )
    blocks = render_blocks("sources:", hint, source_index=index)
    assert blocks is not None
    assert _types(blocks) == ["markdown", "carousel"]
    assert len(blocks[1]["elements"]) == 1


def test_sources_without_index_renders_nothing() -> None:
    hint = PresentationHint(
        elements=[SourcesElement(items=[SourceCardItem(source_ref="source:0")])]
    )
    assert render_blocks("plain answer", hint, source_index=None) is None


def test_header_and_divider_compose_a_multi_section_answer() -> None:
    hint = PresentationHint(
        elements=[
            HeaderElement(text="Morning digest"),
            FieldsElement(items=[FieldItem(label="MRR", value="$182k")]),
            DividerElement(),
            ItemsElement(items=[ListItem(title="HIG-255")]),
        ]
    )
    blocks = render_blocks("Here's your morning digest.", hint)
    assert blocks is not None
    assert _types(blocks) == ["markdown", "header", "section", "divider", "section"]
    assert blocks[1]["text"]["text"] == "Morning digest"


def test_build_source_index_skips_non_http_and_caps() -> None:
    index = build_source_index(
        [
            _FakeEvidence(urls=["ftp://nope", "https://ok.test/1"], preview="p"),
            _FakeEvidence(urls=None, preview="none"),
        ]
    )
    assert index.resolve("source:0") is not None
    assert index.resolve("source:0").url == "https://ok.test/1"  # type: ignore[union-attr]
    assert index.resolve("source:1") is None
    assert index.available()[0]["domain"] == "ok.test"
