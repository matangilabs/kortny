"""HIG-235: Block Kit response register (markdown-table detection)."""

from __future__ import annotations

from kortny.slack import blockkit
from kortny.slack.response_blocks import render_response_blocks

_TABLE = (
    "Here's the breakdown:\n"
    "\n"
    "| Region | Revenue |\n"
    "| --- | --- |\n"
    "| West | $120k |\n"
    "| East | $98k |\n"
)


def test_table_returns_single_markdown_block() -> None:
    blocks = render_response_blocks(_TABLE)
    assert blocks == [blockkit.markdown_block(_TABLE)]
    assert len(blocks) == 1
    assert blocks[0]["type"] == "markdown"
    assert blocks[0]["text"] == _TABLE


def test_aligned_separator_row_detected() -> None:
    text = "| Name | Score |\n|:---|---:|\n| Ana | 9 |\n"
    blocks = render_response_blocks(text)
    assert blocks is not None
    assert blocks[0]["type"] == "markdown"


def test_separator_without_outer_pipes_detected() -> None:
    text = "Col A | Col B\n--- | ---\nfoo | bar\n"
    blocks = render_response_blocks(text)
    assert blocks is not None


def test_multiple_tables_still_single_block() -> None:
    text = (
        "First:\n"
        "| A | B |\n"
        "| --- | --- |\n"
        "| 1 | 2 |\n"
        "\n"
        "Second:\n"
        "| C | D |\n"
        "| --- | --- |\n"
        "| 3 | 4 |\n"
    )
    blocks = render_response_blocks(text)
    assert blocks is not None
    assert len(blocks) == 1
    assert blocks[0]["text"] == text


def test_prose_without_table_returns_none() -> None:
    text = (
        "I dug through the thread and the gist is that the deploy failed "
        "because the migration lock was held. I've cleared it; you can retry."
    )
    assert render_response_blocks(text) is None


def test_prose_with_inline_pipe_but_no_separator_returns_none() -> None:
    # A single pipe-ish line with no following separator row is not a table.
    text = "Run `cat foo | grep bar` and tell me what you see."
    assert render_response_blocks(text) is None


def test_pipe_followed_by_text_not_separator_returns_none() -> None:
    text = "| Some | Header |\nnot a separator row at all\n| a | b |\n"
    assert render_response_blocks(text) is None


def test_over_limit_with_table_returns_none() -> None:
    header = "| A | B |\n| --- | --- |\n"
    # Pad past the 12k markdown block limit while still containing a table.
    filler = "| x | y |\n" * 2000
    text = header + filler
    assert len(text) > blockkit.MAX_MARKDOWN_BLOCK_CHARS
    assert render_response_blocks(text) is None


def test_just_under_limit_with_table_returns_block() -> None:
    header = "| A | B |\n| --- | --- |\n"
    body = "| x | y |\n"
    text = header + body
    while len(text) + len(body) <= blockkit.MAX_MARKDOWN_BLOCK_CHARS:
        text += body
    assert len(text) <= blockkit.MAX_MARKDOWN_BLOCK_CHARS
    blocks = render_response_blocks(text)
    assert blocks is not None
    assert blocks[0]["text"] == text


def test_pipes_inside_code_fence_not_counted_as_table() -> None:
    text = (
        "Here's a snippet:\n"
        "```\n"
        "| not | a | table |\n"
        "| --- | --- | --- |\n"
        "echo done\n"
        "```\n"
        "That's the whole thing."
    )
    assert render_response_blocks(text) is None


def test_tilde_code_fence_excluded() -> None:
    text = "~~~\n| a | b |\n| --- | --- |\n~~~\ndone"
    assert render_response_blocks(text) is None


def test_table_after_closed_code_fence_still_detected() -> None:
    text = "```\nsome code\n```\n| A | B |\n| --- | --- |\n| 1 | 2 |\n"
    blocks = render_response_blocks(text)
    assert blocks is not None


def test_empty_text_returns_none() -> None:
    assert render_response_blocks("") is None
