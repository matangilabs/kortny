"""Agent persona naming for prompts and user-facing copy.

Self-hosters brand their install via SLACK_APP_NAME. So the agent's name must
not be hardcoded as "Kortny" in any string a user or the model can see. Instead,
LLM prompts and Slack copy carry the ``AGENT_NAME_TOKEN`` placeholder and are
rendered through :func:`personalize` with ``settings.agent_display_name`` at the
point of use (prompt build site or Slack post site).

Using a plain ``str.replace`` of a unique sentinel token (rather than
``str.format``) keeps this safe for prompts that contain literal ``{`` / ``}``
characters, e.g. JSON examples in system prompts.
"""

from __future__ import annotations

AGENT_NAME_TOKEN = "__AGENT_NAME__"


def personalize(text: str, agent_name: str) -> str:
    """Substitute the agent-name placeholder with the configured display name.

    Returns ``text`` unchanged when it carries no placeholder, so call sites can
    apply it unconditionally without worrying about extra work or surprises.
    """

    if AGENT_NAME_TOKEN not in text:
        return text
    return text.replace(AGENT_NAME_TOKEN, agent_name)
