"""Regression tests for the consolidator's Slack response helpers (HIG-277).

slack_sdk's ``SlackResponse`` delegates ``.get``/``__getitem__`` to its ``.data``
dict but is NOT itself a ``Mapping``. An ``isinstance(response, Mapping)`` guard
therefore failed silently, so DM open + title resolution always returned None
and the org/user-profile DM proposals never fired. These tests pin the fix:
the helpers must read ``response.data``.
"""

from __future__ import annotations

from collections.abc import ItemsView, Iterator, KeysView, ValuesView
from typing import Any

from kortny.consolidator.service import (
    _open_dm_channel,
    _slack_payload,
    _slack_user_title,
)


class FakeSlackResponse:
    """Mimics slack_sdk.SlackResponse: holds a dict in .data, not a Mapping."""

    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data

    # SlackResponse delegates these to .data but does not register as a Mapping.
    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def keys(self) -> KeysView[str]:
        return self.data.keys()

    def values(self) -> ValuesView[Any]:
        return self.data.values()

    def items(self) -> ItemsView[str, Any]:
        return self.data.items()

    def __iter__(self) -> Iterator[str]:
        return iter(self.data)


class FakeClient:
    def __init__(self, *, dm: dict[str, Any], info: dict[str, Any]) -> None:
        self._dm = dm
        self._info = info

    def conversations_open(self, *, users: str) -> FakeSlackResponse:
        del users
        return FakeSlackResponse(self._dm)

    def users_info(self, *, user: str) -> FakeSlackResponse:
        del user
        return FakeSlackResponse(self._info)


def test_slack_payload_reads_data_of_non_mapping_response() -> None:
    response = FakeSlackResponse({"ok": True, "channel": {"id": "D1"}})
    payload = _slack_payload(response)
    assert payload is not None
    assert payload["channel"] == {"id": "D1"}


def test_open_dm_channel_extracts_id_from_slack_response() -> None:
    client = FakeClient(dm={"ok": True, "channel": {"id": "D0DEV"}}, info={})
    assert _open_dm_channel(client, "U_DEV") == "D0DEV"  # type: ignore[arg-type]


def test_slack_user_title_extracts_title_from_slack_response() -> None:
    client = FakeClient(
        dm={},
        info={"ok": True, "user": {"profile": {"title": "Software Engineer"}}},
    )
    assert _slack_user_title(client, "U_DEV") == "Software Engineer"  # type: ignore[arg-type]


def test_helpers_return_none_on_empty_payload() -> None:
    client = FakeClient(dm={"ok": True}, info={"ok": True})
    assert _open_dm_channel(client, "U_DEV") is None  # type: ignore[arg-type]
    assert _slack_user_title(client, "U_DEV") is None  # type: ignore[arg-type]
