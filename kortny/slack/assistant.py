"""Assistant thread surface registration (HIG-236)."""

from __future__ import annotations

from slack_bolt import App
from sqlalchemy.orm import Session, sessionmaker

from kortny.config import Settings


def register_assistant(
    app: App,
    *,
    settings: Settings,
    session_factory: sessionmaker[Session] | None,
) -> None:
    """Implemented by work package B (HIG-236)."""
