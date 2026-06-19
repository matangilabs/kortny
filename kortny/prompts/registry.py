"""The prompt registry: name -> (subsystem, version, description) (HIG-203).

Every LLM prompt that flows through ``LLMService`` is registered here by its
``prompt_name``. The registry is the source of truth for prompt versions, so
they appear in usage rows and traces. Bump a prompt's ``version`` here when you
change its text, so quality changes can be correlated with prompt changes after
the fact.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PromptDefinition:
    """Metadata for one registered prompt."""

    name: str
    subsystem: str
    version: str
    description: str


PROMPT_REGISTRY: dict[str, PromptDefinition] = {}


def register_prompt(
    *, name: str, subsystem: str, version: str, description: str
) -> PromptDefinition:
    """Register (or replace) a prompt definition. Idempotent by name."""

    definition = PromptDefinition(
        name=name, subsystem=subsystem, version=version, description=description
    )
    PROMPT_REGISTRY[name] = definition
    return definition


def prompt_version(name: str | None) -> str | None:
    """Return the registered version for a prompt name, or None if unregistered."""

    if not name:
        return None
    definition = PROMPT_REGISTRY.get(name)
    return definition.version if definition is not None else None


def _seed() -> None:
    """Register the known LLM prompts.

    Versions start at "1" and should be bumped when the prompt text changes.
    This is the inventory; the prompt bodies still live next to their subsystems
    (moving the text in is a follow-on refactor).
    """

    seed: tuple[tuple[str, ...], ...] = (
        # name, subsystem, one-line description, [version (defaults to "1")]
        (
            "kortny.intent_classifier",
            "intent",
            "Classify an inbound Slack message into a routing decision.",
        ),
        (
            "kortny.agent_coordinator.system",
            "agent",
            "Coordinator system persona + tool-use rules.",
        ),
        (
            "kortny.execution_planner",
            "agent",
            "Author an explicit execution plan for a task.",
        ),
        ("kortny.recovery_planner", "agent", "Re-plan after a failed tool call."),
        (
            "kortny.response_humanizer",
            "slack",
            "Rewrite the agent answer in Kortny's Slack voice.",
            "2",  # v2 (HIG-255): may emit an optional presentation hint
        ),
        ("kortny.ack_generator", "slack", "Generate a short acknowledgement line."),
        (
            "kortny.artifact_comment",
            "slack",
            "Comment posted alongside a generated artifact.",
        ),
        (
            "kortny.schedule_parser",
            "scheduler",
            "Parse a natural-language schedule request.",
        ),
        (
            "kortny.semantic_router.shadow",
            "routing",
            "Shadow semantic runtime-class router.",
        ),
        (
            "kortny.org_profile_extractor",
            "consolidator",
            "Infer the workspace org profile.",
        ),
        (
            "kortny.user_profile_extractor",
            "consolidator",
            "Infer a user's persona (role + work surfaces).",
        ),
        (
            "kortny.style_card_extractor",
            "consolidator",
            "Infer per-surface response style cards.",
        ),
        (
            "kortny.project_inference_namer",
            "consolidator",
            "Name an inferred project cluster.",
        ),
        (
            "kortny.consolidator_merge",
            "consolidator",
            "Merge near-duplicate graph entities.",
        ),
        (
            "kortny.consolidator_promotion",
            "consolidator",
            "Adjudicate episode→graph promotion.",
        ),
        (
            "kortny.witness_task_response_extractor",
            "witness",
            "Extract a witnessable answer from a task.",
        ),
        (
            "kortny.witness_channel_profile_extractor",
            "witness",
            "Extract opportunities from a channel profile.",
        ),
        (
            "kortny.witness_autopilot_reviewer",
            "witness",
            "Review an autopilot opportunity before delivery.",
        ),
        (
            "kortny.knowledge_graph.channel_semantic_extractor",
            "knowledge_graph",
            "Extract channel semantic facts.",
        ),
        (
            "kortny.mcp_description_enricher",
            "mcp",
            "Enrich a sparse MCP tool description.",
        ),
        (
            "kortny.tool_approval_prompt",
            "agent",
            "Synthesize a tool-approval request prompt.",
        ),
    )
    for entry in seed:
        name, subsystem, description = entry[0], entry[1], entry[2]
        version = entry[3] if len(entry) > 3 else "1"
        register_prompt(
            name=name, subsystem=subsystem, version=version, description=description
        )


_seed()
