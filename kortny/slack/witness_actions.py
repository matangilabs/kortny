"""Witness Block Kit action handler registration (HIG-235)."""

from __future__ import annotations

from slack_bolt import App
from sqlalchemy.orm import Session, sessionmaker

from kortny.config import Settings


def register_witness_actions(
    app: App,
    *,
    settings: Settings,
    session_factory: sessionmaker[Session] | None,
) -> None:
    """Implemented by work package C (HIG-235)."""
