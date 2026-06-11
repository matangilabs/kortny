"""Unit tests for Slack Block Kit builders. No DB required."""

from __future__ import annotations

import pytest

from kortny.slack import blockkit


def test_constants_exported() -> None:
    assert blockkit.HOME_ACTION_PREFIX == "kortny_home_"
    assert blockkit.WITNESS_ACTION_PREFIX == "kortny_witness_"
    assert blockkit.MAX_MESSAGE_BLOCKS == 50
    assert blockkit.MAX_VIEW_BLOCKS == 100
    assert blockkit.MAX_MARKDOWN_BLOCK_CHARS == 12_000


def test_header_happy_path() -> None:
    assert blockkit.header("Hello") == {
        "type": "header",
        "text": {"type": "plain_text", "text": "Hello"},
    }


def test_header_limit() -> None:
    with pytest.raises(ValueError):
        blockkit.header("x" * 151)


def test_section_text_only() -> None:
    assert blockkit.section("*bold*") == {
        "type": "section",
        "text": {"type": "mrkdwn", "text": "*bold*"},
    }


def test_section_with_fields_and_accessory() -> None:
    accessory = blockkit.button("Go", "kortny_home_go")
    block = blockkit.section("hi", fields=["a", "b"], accessory=accessory)
    assert block["text"] == {"type": "mrkdwn", "text": "hi"}
    assert block["fields"] == [
        {"type": "mrkdwn", "text": "a"},
        {"type": "mrkdwn", "text": "b"},
    ]
    assert block["accessory"] == accessory


def test_section_requires_text_or_fields() -> None:
    with pytest.raises(ValueError):
        blockkit.section()


def test_section_text_limit() -> None:
    with pytest.raises(ValueError):
        blockkit.section("x" * 3001)


def test_section_too_many_fields() -> None:
    with pytest.raises(ValueError):
        blockkit.section(fields=["x"] * 11)


def test_section_field_too_long() -> None:
    with pytest.raises(ValueError):
        blockkit.section(fields=["x" * 2001])


def test_context_happy_path() -> None:
    assert blockkit.context("a", "b") == {
        "type": "context",
        "elements": [
            {"type": "mrkdwn", "text": "a"},
            {"type": "mrkdwn", "text": "b"},
        ],
    }


def test_context_requires_element() -> None:
    with pytest.raises(ValueError):
        blockkit.context()


def test_context_limit() -> None:
    with pytest.raises(ValueError):
        blockkit.context(*["x"] * 11)


def test_divider() -> None:
    assert blockkit.divider() == {"type": "divider"}


def test_markdown_block_happy_path() -> None:
    assert blockkit.markdown_block("| a | b |") == {
        "type": "markdown",
        "text": "| a | b |",
    }


def test_markdown_block_limit() -> None:
    with pytest.raises(ValueError):
        blockkit.markdown_block("x" * 12_001)


def test_button_minimal() -> None:
    assert blockkit.button("Click", "kortny_home_click") == {
        "type": "button",
        "text": {"type": "plain_text", "text": "Click"},
        "action_id": "kortny_home_click",
    }


def test_button_full() -> None:
    block = blockkit.button(
        "Accept",
        "kortny_witness_accept",
        value="abc",
        style="primary",
        url="https://example.com",
        confirm_title="Sure?",
        confirm_text="Really?",
    )
    assert block["value"] == "abc"
    assert block["style"] == "primary"
    assert block["url"] == "https://example.com"
    assert block["confirm"]["title"] == {"type": "plain_text", "text": "Sure?"}
    assert block["confirm"]["text"] == {"type": "mrkdwn", "text": "Really?"}


def test_button_omits_empty_value() -> None:
    assert "value" not in blockkit.button("X", "kortny_home_x")


def test_button_text_limit() -> None:
    with pytest.raises(ValueError):
        blockkit.button("x" * 76, "kortny_home_x")


def test_button_value_limit() -> None:
    with pytest.raises(ValueError):
        blockkit.button("X", "kortny_home_x", value="x" * 2001)


def test_actions_happy_path() -> None:
    btn = blockkit.button("X", "kortny_home_x")
    block = blockkit.actions(btn, block_id="row1")
    assert block == {"type": "actions", "elements": [btn], "block_id": "row1"}


def test_actions_requires_element() -> None:
    with pytest.raises(ValueError):
        blockkit.actions()


def test_actions_limit() -> None:
    btn = blockkit.button("X", "kortny_home_x")
    with pytest.raises(ValueError):
        blockkit.actions(*[btn] * 26)


def test_table_happy_path() -> None:
    block = blockkit.table(
        [["h1", "h2"], ["a", "b"]],
        column_alignments=["left", "right"],
    )
    assert block["type"] == "table"
    assert block["rows"] == [
        [
            {"type": "raw_text", "text": "h1"},
            {"type": "raw_text", "text": "h2"},
        ],
        [
            {"type": "raw_text", "text": "a"},
            {"type": "raw_text", "text": "b"},
        ],
    ]
    assert block["column_settings"] == [{"align": "left"}, {"align": "right"}]


def test_table_no_alignments() -> None:
    block = blockkit.table([["a"]])
    assert "column_settings" not in block


def test_table_requires_rows() -> None:
    with pytest.raises(ValueError):
        blockkit.table([])


def test_table_row_limit() -> None:
    with pytest.raises(ValueError):
        blockkit.table([["x"]] * 101)


def test_table_column_limit() -> None:
    with pytest.raises(ValueError):
        blockkit.table([["x"] * 21])


def test_table_alignment_count_mismatch() -> None:
    with pytest.raises(ValueError):
        blockkit.table([["a", "b"]], column_alignments=["left"])


def test_static_select_happy_path() -> None:
    block = blockkit.static_select(
        "kortny_home_pick",
        [("Label A", "a"), ("Label B", "b")],
        placeholder="Pick one",
        initial="b",
    )
    assert block["type"] == "static_select"
    assert block["action_id"] == "kortny_home_pick"
    assert block["options"][0] == {
        "text": {"type": "plain_text", "text": "Label A"},
        "value": "a",
    }
    assert block["placeholder"] == {"type": "plain_text", "text": "Pick one"}
    assert block["initial_option"]["value"] == "b"


def test_static_select_requires_options() -> None:
    with pytest.raises(ValueError):
        blockkit.static_select("kortny_home_pick", [])


def test_plain_text_input_minimal() -> None:
    assert blockkit.plain_text_input("kortny_home_in") == {
        "type": "plain_text_input",
        "action_id": "kortny_home_in",
    }


def test_plain_text_input_full() -> None:
    block = blockkit.plain_text_input(
        "kortny_home_in",
        multiline=True,
        placeholder="type here",
        initial="seed",
    )
    assert block["multiline"] is True
    assert block["placeholder"] == {"type": "plain_text", "text": "type here"}
    assert block["initial_value"] == "seed"


def test_input_block_happy_path() -> None:
    element = blockkit.plain_text_input("kortny_home_in")
    block = blockkit.input_block(
        "Name",
        element,
        block_id="name_block",
        optional=True,
        hint="your name",
    )
    assert block == {
        "type": "input",
        "block_id": "name_block",
        "label": {"type": "plain_text", "text": "Name"},
        "element": element,
        "optional": True,
        "hint": {"type": "plain_text", "text": "your name"},
    }


def test_input_block_no_hint() -> None:
    element = blockkit.plain_text_input("kortny_home_in")
    block = blockkit.input_block("Name", element, block_id="name_block")
    assert "hint" not in block
    assert block["optional"] is False


def test_home_view_happy_path() -> None:
    blocks = [blockkit.header("Home"), blockkit.divider()]
    assert blockkit.home_view(blocks) == {"type": "home", "blocks": blocks}


def test_home_view_block_limit() -> None:
    with pytest.raises(ValueError):
        blockkit.home_view([blockkit.divider()] * 101)


def test_modal_happy_path() -> None:
    blocks = [blockkit.section("body")]
    view = blockkit.modal(
        "Add skill",
        blocks,
        callback_id="kortny_home_add_skill",
        submit="Add",
        close="Nope",
        private_metadata="meta",
    )
    assert view == {
        "type": "modal",
        "callback_id": "kortny_home_add_skill",
        "title": {"type": "plain_text", "text": "Add skill"},
        "submit": {"type": "plain_text", "text": "Add"},
        "close": {"type": "plain_text", "text": "Nope"},
        "private_metadata": "meta",
        "blocks": blocks,
    }


def test_modal_defaults() -> None:
    view = blockkit.modal("Title", [], callback_id="kortny_home_x")
    assert view["submit"] == {"type": "plain_text", "text": "Save"}
    assert view["close"] == {"type": "plain_text", "text": "Cancel"}
    assert view["private_metadata"] == ""


def test_modal_title_limit() -> None:
    with pytest.raises(ValueError):
        blockkit.modal("x" * 25, [], callback_id="kortny_home_x")


def test_modal_submit_limit() -> None:
    with pytest.raises(ValueError):
        blockkit.modal("Title", [], callback_id="kortny_home_x", submit="x" * 25)


def test_modal_close_limit() -> None:
    with pytest.raises(ValueError):
        blockkit.modal("Title", [], callback_id="kortny_home_x", close="x" * 25)


def test_modal_block_limit() -> None:
    with pytest.raises(ValueError):
        blockkit.modal("Title", [blockkit.divider()] * 101, callback_id="kortny_home_x")
