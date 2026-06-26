"""Pure unit tests for verify_and_parse_trigger_webhook.

No database, no I/O. Tests construct valid HMAC signatures the same way
Composio does, then verify them round-trip.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time

import pytest

from kortny.composio.client import (
    ParsedTriggerEvent,
    TriggerParseError,
    TriggerSignatureError,
    TriggerTimestampError,
    verify_and_parse_trigger_webhook,
)

SECRET = "test-webhook-secret-key"


def _make_envelope(
    *,
    trigger_slug: str = "GITHUB_PULL_REQUEST_EVENT",
    trigger_id: str | None = "ti_abc123",
    connected_account_id: str | None = "ca_xyz",
    user_id: str | None = "user_1",
    event_id: str = "evt_001",
    data: dict | None = None,
) -> dict:
    return {
        "id": event_id,
        "type": "composio.trigger.message",
        "metadata": {
            "trigger_slug": trigger_slug,
            "trigger_id": trigger_id,
            "connected_account_id": connected_account_id,
            "user_id": user_id,
        },
        "data": data or {"action": "review_requested"},
        "timestamp": "2026-06-26T00:00:00Z",
    }


def _sign(
    *,
    webhook_id: str,
    webhook_timestamp: str,
    body: bytes,
    secret: str = SECRET,
) -> str:
    """Build the v1,<base64> signature the same way Composio does."""
    signed_string = f"{webhook_id}.{webhook_timestamp}.{body.decode('utf-8')}"
    key = secret.encode("utf-8")
    mac = hmac.new(key, signed_string.encode("utf-8"), hashlib.sha256).digest()
    return "v1," + base64.b64encode(mac).decode("ascii")


def _make_headers(
    *,
    webhook_id: str = "wh_test",
    webhook_timestamp: str | None = None,
    sig: str | None = None,
    body: bytes = b"",
    secret: str = SECRET,
) -> dict[str, str]:
    ts = webhook_timestamp or str(int(time.time()))
    signature = (
        sig
        if sig is not None
        else _sign(
            webhook_id=webhook_id, webhook_timestamp=ts, body=body, secret=secret
        )
    )
    return {
        "webhook-id": webhook_id,
        "webhook-timestamp": ts,
        "webhook-signature": signature,
    }


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_valid_signature_parses_correctly() -> None:
    envelope = _make_envelope()
    body = json.dumps(envelope).encode()
    headers = _make_headers(body=body)

    result = verify_and_parse_trigger_webhook(
        raw_body=body, headers=headers, secret=SECRET
    )

    assert isinstance(result, ParsedTriggerEvent)
    assert result.id == "evt_001"
    assert result.type == "composio.trigger.message"
    assert result.trigger_slug == "GITHUB_PULL_REQUEST_EVENT"
    assert result.trigger_id == "ti_abc123"
    assert result.connected_account_id == "ca_xyz"
    assert result.user_id == "user_1"
    assert result.data == {"action": "review_requested"}
    assert result.timestamp == "2026-06-26T00:00:00Z"


def test_none_fields_parse_as_none() -> None:
    envelope = _make_envelope(trigger_id=None, connected_account_id=None, user_id=None)
    body = json.dumps(envelope).encode()
    headers = _make_headers(body=body)

    result = verify_and_parse_trigger_webhook(
        raw_body=body, headers=headers, secret=SECRET
    )

    assert result.trigger_id is None
    assert result.connected_account_id is None
    assert result.user_id is None


# ---------------------------------------------------------------------------
# Signature errors
# ---------------------------------------------------------------------------


def test_bad_signature_raises_signature_error() -> None:
    envelope = _make_envelope()
    body = json.dumps(envelope).encode()
    ts = str(int(time.time()))
    headers = {
        "webhook-id": "wh_test",
        "webhook-timestamp": ts,
        "webhook-signature": "v1,invalidsignatureXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX==",
    }

    with pytest.raises(TriggerSignatureError):
        verify_and_parse_trigger_webhook(raw_body=body, headers=headers, secret=SECRET)


def test_wrong_secret_raises_signature_error() -> None:
    envelope = _make_envelope()
    body = json.dumps(envelope).encode()
    headers = _make_headers(body=body, secret="wrong-secret")

    with pytest.raises(TriggerSignatureError):
        verify_and_parse_trigger_webhook(raw_body=body, headers=headers, secret=SECRET)


# ---------------------------------------------------------------------------
# Timestamp errors
# ---------------------------------------------------------------------------


def test_old_timestamp_raises_timestamp_error() -> None:
    envelope = _make_envelope()
    body = json.dumps(envelope).encode()
    old_ts = str(int(time.time()) - 600)  # 10 minutes ago
    headers = _make_headers(body=body, webhook_timestamp=old_ts)

    with pytest.raises(TriggerTimestampError):
        verify_and_parse_trigger_webhook(raw_body=body, headers=headers, secret=SECRET)


def test_future_timestamp_raises_timestamp_error() -> None:
    envelope = _make_envelope()
    body = json.dumps(envelope).encode()
    future_ts = str(int(time.time()) + 600)  # 10 minutes in the future
    headers = _make_headers(body=body, webhook_timestamp=future_ts)

    with pytest.raises(TriggerTimestampError):
        verify_and_parse_trigger_webhook(raw_body=body, headers=headers, secret=SECRET)


def test_invalid_timestamp_string_raises_timestamp_error() -> None:
    envelope = _make_envelope()
    body = json.dumps(envelope).encode()
    ts = str(int(time.time()))
    sig = _sign(webhook_id="wh_test", webhook_timestamp=ts, body=body)
    headers = {
        "webhook-id": "wh_test",
        "webhook-timestamp": "not-a-number",
        "webhook-signature": sig,
    }

    with pytest.raises(TriggerTimestampError):
        verify_and_parse_trigger_webhook(raw_body=body, headers=headers, secret=SECRET)


# ---------------------------------------------------------------------------
# Parse errors
# ---------------------------------------------------------------------------


def test_malformed_json_raises_parse_error() -> None:
    body = b"{not valid json"
    headers = _make_headers(body=body)

    with pytest.raises(TriggerParseError):
        verify_and_parse_trigger_webhook(raw_body=body, headers=headers, secret=SECRET)


def test_missing_trigger_slug_raises_parse_error() -> None:
    envelope = {
        "id": "evt_001",
        "type": "composio.trigger.message",
        "metadata": {},  # no trigger_slug
        "data": {},
        "timestamp": "2026-06-26T00:00:00Z",
    }
    body = json.dumps(envelope).encode()
    headers = _make_headers(body=body)

    with pytest.raises(TriggerParseError):
        verify_and_parse_trigger_webhook(raw_body=body, headers=headers, secret=SECRET)


def test_non_dict_envelope_raises_parse_error() -> None:
    body = json.dumps(["not", "a", "dict"]).encode()
    headers = _make_headers(body=body)

    with pytest.raises(TriggerParseError):
        verify_and_parse_trigger_webhook(raw_body=body, headers=headers, secret=SECRET)


# ---------------------------------------------------------------------------
# Multiple signatures
# ---------------------------------------------------------------------------


def test_multiple_signatures_valid_one_passes() -> None:
    """webhook-signature with two entries — one invalid, one valid — should pass."""
    envelope = _make_envelope()
    body = json.dumps(envelope).encode()
    ts = str(int(time.time()))
    valid_sig = _sign(webhook_id="wh_test", webhook_timestamp=ts, body=body)
    # Combine an invalid sig with the valid one, space-separated
    combined = f"v1,AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA= {valid_sig}"
    headers = {
        "webhook-id": "wh_test",
        "webhook-timestamp": ts,
        "webhook-signature": combined,
    }

    result = verify_and_parse_trigger_webhook(
        raw_body=body, headers=headers, secret=SECRET
    )
    assert result.trigger_slug == "GITHUB_PULL_REQUEST_EVENT"
