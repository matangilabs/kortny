"""App Home console surface registration (HIG-232)."""

from __future__ import annotations

from slack_bolt import App
from sqlalchemy.orm import Session, sessionmaker

from kortny.config import Settings


def register_app_home(
    app: App,
    *,
    settings: Settings,
    session_factory: sessionmaker[Session] | None,
) -> None:
    """Implemented by work package A (HIG-232)."""
