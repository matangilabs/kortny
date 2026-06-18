"""Central prompt registry (HIG-203).

~40 prompt constants are scattered across a dozen modules with no versioning and
no single inventory, so prompt edits ship blind. This registry is the one place
that knows every LLM prompt's name, owning subsystem, and version — so prompt
versions surface in LLM-usage rows (correlate quality changes with prompt
changes) and there is a single list to evaluate against.

Increment 1 registers metadata + version for the prompts that flow through
``LLMService`` (keyed by their existing ``prompt_name``); ``LLMService`` auto-
fills the version from here. Moving every prompt's TEXT into the registry (so
call sites import the body from one place) is a follow-on refactor; this is the
inventory + version source it builds on.
"""

from kortny.prompts.registry import (
    PROMPT_REGISTRY,
    PromptDefinition,
    prompt_version,
    register_prompt,
)

__all__ = [
    "PROMPT_REGISTRY",
    "PromptDefinition",
    "prompt_version",
    "register_prompt",
]
