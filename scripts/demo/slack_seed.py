"""Northwind demo workspace Slack seeder.

WARNING: This script posts to a real Slack workspace. It must NEVER be run
against an unintended workspace, and must NEVER be run against the connected
enterprise workspace. Always use a dedicated demo workspace.

Usage (dry-run, safe by default):
    uv run python -m scripts.demo.slack_seed \\
        --token xoxb-... \\
        --channels general=C0123,engineering=C0456,product=C0789,ops=C0012,launch=C0345

Usage (live post — requires explicit confirmation flag):
    uv run python -m scripts.demo.slack_seed \\
        --token xoxb-... \\
        --channels general=C0123,engineering=C0456,product=C0789,ops=C0012,launch=C0345 \\
        --i-understand-this-posts-to-real-slack
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime

from slack_sdk import WebClient

from scripts.demo.fixtures import DEFAULT_SIM_DAYS, SimMessage, build_story


def _check_safety(args: argparse.Namespace) -> None:
    """Refuse to post unless every explicit safety gate is cleared."""
    missing: list[str] = []
    if not args.token:
        missing.append("--token / SLACK_DEMO_TOKEN")
    if not args.channels:
        missing.append("--channels / SLACK_DEMO_CHANNELS")
    if missing:
        print(
            "error: refusing to run — missing required parameters:\n"
            + "\n".join(f"  {m}" for m in missing),
            file=sys.stderr,
        )
        raise SystemExit(1)
    if not args.confirm:
        print(
            "Dry-run mode (default). Pass --i-understand-this-posts-to-real-slack "
            "to actually post to Slack.",
            file=sys.stderr,
        )


def _parse_channels(raw: str) -> dict[str, str]:
    """Parse 'name=CID,name2=CID2' into {name: CID}."""
    result: dict[str, str] = {}
    for part in raw.split(","):
        part = part.strip()
        if "=" in part:
            name, cid = part.split("=", 1)
            result[name.strip()] = cid.strip()
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.demo.slack_seed",
        description=(
            "Post the Northwind demo fixture story to a real Slack workspace. "
            "DRY-RUN by default — pass --i-understand-this-posts-to-real-slack to post."
        ),
    )
    parser.add_argument(
        "--token",
        default=None,
        metavar="TOKEN",
        help="Slack bot token (or set SLACK_DEMO_TOKEN). Never defaults to SLACK_BOT_TOKEN.",
    )
    parser.add_argument(
        "--channels",
        default=None,
        metavar="CHANNEL_MAP",
        help=(
            "Comma-separated name=CID pairs, e.g. "
            "general=C0123,engineering=C0456,product=C0789,ops=C0012,launch=C0345 "
            "(or set SLACK_DEMO_CHANNELS)."
        ),
    )
    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_SIM_DAYS,
        help=f"Story window in days (default {DEFAULT_SIM_DAYS}).",
    )
    parser.add_argument(
        "--i-understand-this-posts-to-real-slack",
        dest="confirm",
        action="store_true",
        default=False,
        help="Required to actually post; omit for dry-run.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    import os

    parser = build_parser()
    args = parser.parse_args(argv)

    # Fill from env — NEVER fall back to SLACK_BOT_TOKEN.
    if not args.token:
        args.token = os.environ.get("SLACK_DEMO_TOKEN")
    if not args.channels:
        args.channels = os.environ.get("SLACK_DEMO_CHANNELS")

    _check_safety(args)

    channel_map = _parse_channels(args.channels or "")
    now = datetime.now(UTC)
    messages = build_story(now=now, days=args.days)

    if not args.confirm:
        _dry_run(messages, channel_map)
        return 0

    return _live_post(messages, channel_map, token=args.token or "")


def _dry_run(messages: tuple[SimMessage, ...], channel_map: dict[str, str]) -> None:
    print(f"[DRY-RUN] Would post {len(messages)} messages:")
    for msg in messages:
        cid = channel_map.get(msg.channel_name, f"<unmapped:{msg.channel_name}>")
        thread = f" (thread:{msg.thread_slug})" if msg.thread_slug else ""
        print(
            f"  [{msg.channel_name}/{cid}]{thread} "
            f"{msg.persona.display_name}: {msg.text[:80]!r}"
        )


def _live_post(
    messages: tuple[SimMessage, ...],
    channel_map: dict[str, str],
    *,
    token: str,
) -> int:
    client = WebClient(token=token)
    posted: dict[str, str] = {}  # slug -> ts
    errors = 0

    for msg in messages:
        cid = channel_map.get(msg.channel_name)
        if cid is None:
            print(
                f"  skip: no channel mapping for {msg.channel_name!r}",
                file=sys.stderr,
            )
            continue

        parent_ts = posted.get(msg.thread_slug) if msg.thread_slug else None

        kwargs: dict[str, object] = {
            "channel": cid,
            "text": msg.text,
            "username": msg.persona.display_name,
            "icon_emoji": msg.persona.icon_emoji or ":robot_face:",
        }
        if parent_ts is not None:
            kwargs["thread_ts"] = parent_ts

        try:
            resp = client.chat_postMessage(**kwargs)  # type: ignore[arg-type]
            ts = resp.get("ts")
            if isinstance(ts, str):
                posted[msg.slug] = ts
            print(
                f"  posted [{msg.channel_name}] {msg.persona.display_name}: {msg.text[:60]!r}"
            )
        except Exception as exc:
            print(f"  error posting {msg.slug!r}: {exc}", file=sys.stderr)
            errors += 1

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
