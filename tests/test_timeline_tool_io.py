"""Timeline surfaces tool call input/output as first-class sections."""

from kortny.dashboard.data import _TOOL_IO_MAX_CHARS, _tool_io_json


def test_tool_io_json_renders_arguments() -> None:
    out = _tool_io_json({"query": "Hacker News", "num": 10})
    assert out is not None
    assert '"query": "Hacker News"' in out


def test_tool_io_json_empty_is_none() -> None:
    # Empty/absent input or output -> no section (None), not "{}".
    assert _tool_io_json(None) is None
    assert _tool_io_json({}) is None
    assert _tool_io_json("") is None


def test_tool_io_json_truncates_large_output() -> None:
    big = {"results": ["x" * 1000 for _ in range(100)]}
    out = _tool_io_json(big)
    assert out is not None
    assert out.endswith("(truncated — see Raw payload)")
    assert len(out) <= _TOOL_IO_MAX_CHARS + 40
