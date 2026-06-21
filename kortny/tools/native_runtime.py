"""Native tool class and factory registration."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kortny.documents.critique import DocumentVisualCritic
    from kortny.documents.revision import LlmPatchContext, VisualRevisionPatch

from sqlalchemy.orm import Session

from kortny.config import Settings
from kortny.db.models import Task
from kortny.execution import (
    create_sandbox_runner_from_settings,
    create_sandbox_session_client_from_settings,
)
from kortny.memory import WorkspaceStateService
from kortny.tasks import TaskService
from kortny.tools.catalog import dashboard_native_tool_names, runtime_native_tool_names
from kortny.tools.code_exec import CodeExecTool
from kortny.tools.deploy_site import DeploySiteTool
from kortny.tools.document_studio import DocumentStudioTool
from kortny.tools.list_integrations import DescribeToolsTool, ListIntegrationsTool
from kortny.tools.pdf_generator import PdfGeneratorTool
from kortny.tools.resolve_slack_identity import ResolveSlackIdentityTool
from kortny.tools.sandbox_workbench import (
    SandboxBashTool,
    SandboxExportArtifactTool,
    SandboxPublishPreviewTool,
    SandboxReadFileTool,
    SandboxWriteFileTool,
    WorkbenchSession,
)
from kortny.tools.schedules import (
    CancelScheduleTool,
    CreateScheduleTool,
    GetScheduleTool,
    ListSchedulesTool,
    PauseScheduleTool,
    ResumeScheduleTool,
    UpdateScheduleTool,
)
from kortny.tools.search_observed_slack_history import SearchObservedSlackHistoryTool
from kortny.tools.skills import (
    LoadSkillResourceTool,
    LoadSkillTool,
    RunSkillScriptTool,
)
from kortny.tools.slack_actions import (
    SlackAddBookmarkTool,
    SlackAddReactionTool,
    SlackCreateChannelCanvasTool,
    SlackEditCanvasTool,
    SlackLookupCanvasSectionsTool,
    SlackPinMessageTool,
    SlackReplyThreadTool,
)
from kortny.tools.slack_channel_history import (
    ObservationChannelHistoryCache,
    SlackChannelHistoryTool,
)
from kortny.tools.slack_file_read import PdfPageOcr, SlackFileReadTool
from kortny.tools.slack_identity_info import SlackChannelInfoTool, SlackUserInfoTool
from kortny.tools.types import Tool
from kortny.tools.web_search import WebSearchTool
from kortny.tools.workspace_graph import DeclareProjectTool, QueryWorkspaceGraphTool
from kortny.tools.workspace_memory import (
    ForgetFactTool,
    InspectMemoryTool,
    RecallFactTool,
    RememberFactTool,
)


@dataclass(frozen=True, slots=True)
class NativeToolBuildContext:
    """Runtime dependencies needed to instantiate native tools for one task."""

    settings: Settings
    session: Session
    task: Task
    task_service: TaskService
    working_dir: Path
    web_search_tool: Tool | None
    slack_history_client: Any
    slack_file_client: Any
    slack_identity_client: Any
    slack_action_client: Any
    memory_service: WorkspaceStateService


NativeToolFactory = Callable[[NativeToolBuildContext], Tool | None]
NativeInventoryToolFactory = Callable[[NativeToolBuildContext, Sequence[Tool]], Tool]


@dataclass(frozen=True, slots=True)
class NativeToolRegistration:
    """One native tool class plus its runtime factory."""

    name: str
    tool_class: type[Any]
    factory: NativeToolFactory


@dataclass(frozen=True, slots=True)
class NativeInventoryToolRegistration:
    """Native inventory tools whose output needs the native tool list."""

    name: str
    tool_class: type[Any]
    factory: NativeInventoryToolFactory


def build_native_tools(context: NativeToolBuildContext) -> tuple[Tool, ...]:
    """Instantiate task-scoped native tools in registration order."""

    tools: list[Tool] = []
    for registration in NATIVE_TOOL_REGISTRATIONS:
        tool = registration.factory(context)
        if tool is not None:
            tools.append(tool)
    return tuple(tools)


def build_native_inventory_tools(
    context: NativeToolBuildContext,
    native_tools: Sequence[Tool],
) -> tuple[Tool, ...]:
    """Instantiate task-scoped native inventory tools."""

    return tuple(
        registration.factory(context, native_tools)
        for registration in NATIVE_INVENTORY_TOOL_REGISTRATIONS
    )


def native_tool_classes_by_name() -> dict[str, type[Any]]:
    """Return registered native tool classes keyed by catalog name."""

    classes = {
        registration.name: registration.tool_class
        for registration in NATIVE_TOOL_REGISTRATIONS
    }
    classes.update(
        {
            registration.name: registration.tool_class
            for registration in NATIVE_INVENTORY_TOOL_REGISTRATIONS
        }
    )
    return classes


def native_dashboard_tool_classes() -> tuple[type[Any], ...]:
    """Return dashboard-exposed tool classes in metadata catalog order."""

    classes_by_name = native_tool_classes_by_name()
    return tuple(classes_by_name[name] for name in dashboard_native_tool_names())


def _build_web_search_tool(context: NativeToolBuildContext) -> Tool | None:
    return context.web_search_tool


def _build_pdf_generator_tool(context: NativeToolBuildContext) -> Tool:
    return PdfGeneratorTool(
        working_dir=context.working_dir,
        session=context.session,
        task_id=context.task.id,
        task_service=context.task_service,
    )


def _build_document_visual_critic(
    context: NativeToolBuildContext,
) -> DocumentVisualCritic | None:
    """Build the document visual-critic callable (HIG-244 VLM critic).

    Returns None when:
    - No vision-capable model is configured, or
    - ``doc_visual_critic_enabled`` is False.

    The returned callable takes ``Sequence[bytes]`` (page PNGs) and returns a
    :class:`~kortny.documents.critique.VisualCritique`.  Pages are batched
    within ``vision_max_images_per_request``.  Multi-batch results are merged:
    issues from all batches are combined; overall_score is the minimum across
    batches (most conservative).
    """
    from kortny.documents.critique import (  # noqa: PLC0415
        VisualCritique,
        VisualIssue,
    )
    from kortny.llm import ChatMessage, LLMService  # noqa: PLC0415
    from kortny.llm.litellm_catalog import model_supports_vision  # noqa: PLC0415
    from kortny.llm.routing import ModelRouter as _ModelRouter  # noqa: PLC0415
    from kortny.llm.routing import ModelRouteTier  # noqa: PLC0415
    from kortny.llm.runtime_config import (  # noqa: PLC0415
        create_provider_for_selection,
        select_runtime_model,
    )
    from kortny.llm.types import ImagePart  # noqa: PLC0415

    settings = context.settings
    if not settings.doc_visual_critic_enabled:
        return None

    session = context.session
    task = context.task
    task_service = context.task_service

    vision_route = _ModelRouter(settings).route_for_tier(
        ModelRouteTier.vision,
        reason="document_visual_critic (HIG-244)",
    )
    runtime_selection = select_runtime_model(
        session=session,
        settings=settings,
        installation_id=task.installation_id,
        model_route=vision_route,
    )
    resolved_model = runtime_selection.model.model
    provider_kind = runtime_selection.model.provider_kind

    if not model_supports_vision(provider_kind, resolved_model):
        return None

    provider = create_provider_for_selection(
        settings=settings, selection=runtime_selection
    )
    provider_name = runtime_selection.provider_name
    batch_size = settings.vision_max_images_per_request

    _CRITIQUE_SYSTEM = (
        "You are a meticulous document-design reviewer. "
        "You will be shown rendered page images of a document. "
        "Score the overall visual quality 0-10 (10 = flawless, publication-ready; "
        "0 = completely broken layout). "
        "List concrete visual defects: overflowing text/elements, misalignment, "
        "unlabelled charts or axes, cramped or excessive whitespace, low contrast, "
        "weak visual hierarchy, typographic issues (wrong size, poor readability). "
        "Judge the RENDERED appearance, not the correctness of the content. "
        "Respond ONLY with valid JSON matching the provided schema."
    )

    def visual_critic(page_pngs: Sequence[bytes]) -> VisualCritique:
        llm = LLMService(
            session=session,
            provider=provider,
            provider_name=provider_name,
            task_service=task_service,
            model_route=runtime_selection.model_route,
            settings=settings,
        )

        page_list = list(page_pngs)
        all_issues: list[VisualIssue] = []
        min_score: int = 10
        last_summary = ""

        for batch_start in range(0, len(page_list), batch_size):
            batch = page_list[batch_start : batch_start + batch_size]
            images = tuple(
                ImagePart(
                    data=png,
                    mime="image/png",
                    source="doc_page",
                    alt=f"Page {batch_start + idx + 1}",
                )
                for idx, png in enumerate(batch)
            )
            messages = [
                ChatMessage(role="system", content=_CRITIQUE_SYSTEM),
                ChatMessage(
                    role="user",
                    content=(
                        "Review the document page(s) shown. "
                        "Provide a visual-quality critique as JSON."
                    ),
                    images=images,
                ),
            ]
            completion = llm.complete(
                task_id=task.id,
                messages=messages,
                response_format={"type": "json_object"},
                prompt_name="kortny.document_visual_critic",
                prompt_source="code",
            )
            if completion.content:
                batch_critique = VisualCritique.model_validate_json(completion.content)
                all_issues.extend(batch_critique.issues)
                min_score = min(min_score, batch_critique.overall_score)
                last_summary = batch_critique.summary

        return VisualCritique(
            overall_score=min_score,
            summary=last_summary,
            issues=all_issues,
        )

    critic: DocumentVisualCritic = visual_critic
    return critic


def _build_revision_patch_proposer(
    context: NativeToolBuildContext,
) -> Callable[[LlmPatchContext], VisualRevisionPatch | None] | None:
    """Build the LLM-driven revision patch proposer callable (HIG-244 slice 3b).

    Returns None when no analysis-tier model is configured or on any setup
    error.  The returned callable accepts an ``LlmPatchContext`` and returns a
    ``VisualRevisionPatch`` or ``None`` on any failure (never raises).
    """
    import json  # noqa: PLC0415

    from kortny.documents.revision import (  # noqa: PLC0415
        VisualRevisionPatch,
    )
    from kortny.llm import ChatMessage, LLMService  # noqa: PLC0415
    from kortny.llm.routing import ModelRouter as _ModelRouter  # noqa: PLC0415
    from kortny.llm.routing import ModelRouteTier  # noqa: PLC0415
    from kortny.llm.runtime_config import (  # noqa: PLC0415
        create_provider_for_selection,
        select_runtime_model,
    )

    settings = context.settings
    session = context.session
    task = context.task
    task_service = context.task_service

    try:
        text_route = _ModelRouter(settings).route_for_tier(
            ModelRouteTier.analysis,
            reason="revision_patch_proposer (HIG-244)",
        )
        runtime_selection = select_runtime_model(
            session=session,
            settings=settings,
            installation_id=task.installation_id,
            model_route=text_route,
        )
    except Exception:
        return None

    try:
        provider = create_provider_for_selection(
            settings=settings, selection=runtime_selection
        )
    except Exception:
        return None

    provider_name = runtime_selection.provider_name

    _PROPOSER_SYSTEM = (
        "You are a document-layout fixer. "
        "Choose ONLY from these operations to fix the presentation defects listed:\n"
        "- set_chart_labels: fill MISSING chart title/axis labels/series names "
        "(never overwrite existing)\n"
        "- change_chart_type: switch between bar, line, area only (never to/from pie)\n"
        "- compact_stat_cards: move stat card notes to a prose block below\n"
        "- set_theme: change the document theme (known themes only)\n"
        "- split_table: split an overflowing table into smaller chunks\n"
        "- split_prose: split an overflowing prose block at natural boundaries\n\n"
        "Rules:\n"
        "1. Do NOT invent or change any text content — only fix presentation/layout.\n"
        "2. Output a single VisualRevisionPatch JSON with base_spec_hash set to "
        "the provided value.\n"
        "3. Only reference block_index values that appear in the candidate blocks.\n"
        "4. If no fix is possible, output an empty operations list."
    )

    def revision_proposer(ctx: LlmPatchContext) -> VisualRevisionPatch | None:
        try:
            issue_lines = "\n".join(
                f"- [{issue.category}] {issue.severity}: {issue.message} "
                f"(page {issue.page})"
                for issue in ctx.issues
            )
            user_content = (
                f"{issue_lines}\n\n"
                f"Candidate blocks:\n{json.dumps(ctx.candidates, indent=2)}\n\n"
                f"base_spec_hash: {ctx.base_spec_hash}"
            )
            messages = [
                ChatMessage(role="system", content=_PROPOSER_SYSTEM),
                ChatMessage(role="user", content=user_content),
            ]
            llm = LLMService(
                session=session,
                provider=provider,
                provider_name=provider_name,
                task_service=task_service,
                model_route=runtime_selection.model_route,
                settings=settings,
            )
            completion = llm.complete(
                task_id=task.id,
                messages=messages,
                response_format={"type": "json_object"},
                prompt_name="kortny.revision_patch_proposer",
                prompt_source="code",
            )
            if not completion.content:
                return None
            return VisualRevisionPatch.model_validate_json(completion.content)
        except Exception:
            return None

    return revision_proposer


def _build_document_studio_tool(context: NativeToolBuildContext) -> Tool:
    raw_paths = context.settings.document_font_paths
    font_paths = tuple(p for p in raw_paths.split(":") if p)
    return DocumentStudioTool(
        working_dir=context.working_dir,
        font_paths=font_paths,
        session=context.session,
        task_id=context.task.id,
        task_service=context.task_service,
        slack_client=context.slack_action_client,
        visual_critic=_build_document_visual_critic(context),
        visual_critic_max_pages=context.settings.doc_visual_critic_max_pages,
    )


def _build_slack_channel_history_tool(context: NativeToolBuildContext) -> Tool:
    return SlackChannelHistoryTool(
        context.slack_history_client,
        default_channel_id=context.task.slack_channel_id,
        cache=ObservationChannelHistoryCache(
            context.session,
            installation_id=context.task.installation_id,
        ),
    )


def _build_slack_file_read_tool(context: NativeToolBuildContext) -> Tool:
    return SlackFileReadTool(
        client=context.slack_file_client,
        bot_token=context.settings.slack_bot_token,
        working_dir=context.working_dir,
        max_file_size_bytes=context.settings.slack_file_read_max_bytes,
        session=context.session,
        pdf_ocr=_build_pdf_ocr_callable(context),
        pdf_ocr_max_pages=context.settings.pdf_ocr_max_pages,
    )


def _build_pdf_ocr_callable(
    context: NativeToolBuildContext,
) -> PdfPageOcr | None:
    """Build the PDF OCR callable if a vision-capable model is configured.

    Returns None when no vision model is available so the tool gracefully
    falls back to the scanned_pdf_needs_vision_model warning path.

    The returned callable batches page PNG bytes within
    ``vision_max_images_per_request`` (default 5) so it never exceeds the
    per-request image-count guard enforced inside ``LLMService.complete``.
    """
    from kortny.llm import ChatMessage, LLMService
    from kortny.llm.litellm_catalog import model_supports_vision
    from kortny.llm.routing import ModelRouter as _ModelRouter
    from kortny.llm.routing import ModelRouteTier
    from kortny.llm.runtime_config import (
        create_provider_for_selection,
        select_runtime_model,
    )
    from kortny.llm.types import ImagePart

    settings = context.settings
    session = context.session
    task = context.task
    task_service = context.task_service

    # Resolve the vision-tier model the same way the coordinator does.
    vision_route = _ModelRouter(settings).route_for_tier(
        ModelRouteTier.vision,
        reason="pdf_ocr (HIG-279 slice 3b-2)",
    )

    # Resolve through DB config so DB-overridden provider/model is respected.
    runtime_selection = select_runtime_model(
        session=session,
        settings=settings,
        installation_id=task.installation_id,
        model_route=vision_route,
    )
    # After select_runtime_model the model may differ (DB override); use the
    # resolved model and provider_kind for the vision-capability check.
    resolved_model = runtime_selection.model.model
    provider_kind = runtime_selection.model.provider_kind

    if not model_supports_vision(provider_kind, resolved_model):
        return None

    # Build the provider now so the callable doesn't re-resolve on every call.
    provider = create_provider_for_selection(
        settings=settings, selection=runtime_selection
    )
    provider_name = runtime_selection.provider_name

    batch_size = settings.vision_max_images_per_request

    def pdf_ocr(page_pngs: Sequence[bytes]) -> str:
        llm = LLMService(
            session=session,
            provider=provider,
            provider_name=provider_name,
            task_service=task_service,
            model_route=runtime_selection.model_route,
            settings=settings,
        )

        system_prompt = (
            "You are a faithful document transcriber. "
            "Convert the provided scanned page image(s) to Markdown, "
            "preserving all headings, tables, lists, and reading order exactly. "
            "Output only the transcription — no commentary, preamble, or meta-text."
        )

        parts: list[str] = []
        page_list = list(page_pngs)
        # Batch pages within the per-request image-count guard.
        for batch_start in range(0, len(page_list), batch_size):
            batch = page_list[batch_start : batch_start + batch_size]
            images = tuple(
                ImagePart(
                    data=png,
                    mime="image/png",
                    source="pdf_page",
                    alt=f"Page {batch_start + idx + 1}",
                )
                for idx, png in enumerate(batch)
            )
            messages = [
                ChatMessage(role="system", content=system_prompt),
                ChatMessage(
                    role="user",
                    content=(
                        "Transcribe the scanned document page(s) shown "
                        "in the attached image(s) to Markdown."
                    ),
                    images=images,
                ),
            ]
            completion = llm.complete(
                task_id=task.id,
                messages=messages,
                prompt_name="kortny.pdf_ocr",
                prompt_source="code",
            )
            if completion.content:
                # Label each batch so the agent can orient within long docs.
                batch_end = batch_start + len(batch)
                header = (
                    f"## Pages {batch_start + 1}–{batch_end}\n\n"
                    if len(page_list) > batch_size
                    else ""
                )
                parts.append(header + completion.content)

        return "\n\n".join(parts)

    ocr_callable: PdfPageOcr = pdf_ocr
    return ocr_callable


def _build_describe_tools_tool(
    context: NativeToolBuildContext,
    native_tools: Sequence[Tool],
) -> Tool:
    return DescribeToolsTool(
        session=context.session,
        task=context.task,
        native_tools=native_tools,
    )


def _build_list_integrations_tool(
    context: NativeToolBuildContext,
    native_tools: Sequence[Tool],
) -> Tool:
    return ListIntegrationsTool(
        session=context.session,
        task=context.task,
        native_tools=native_tools,
    )


def _build_code_exec_tool(context: NativeToolBuildContext) -> Tool | None:
    runner = create_sandbox_runner_from_settings(context.settings)
    if runner is None:
        context.task_service.append_event(
            context.task,
            "log",
            {
                "message": "native_tool_unavailable",
                "tool": "code_exec",
                "reason": "missing_sandbox_runner_url",
                "env_var": "KORTNY_SANDBOX_RUNNER_URL",
            },
        )
        return None
    return CodeExecTool(
        runner=runner,
        image=context.settings.sandbox_default_image,
        task=context.task,
        task_service=context.task_service,
    )


def _workbench_session(context: NativeToolBuildContext) -> WorkbenchSession | None:
    client = create_sandbox_session_client_from_settings(context.settings)
    if client is None:
        return None
    return WorkbenchSession(
        client=client,
        task=context.task,
        task_service=context.task_service,
    )


def _log_workbench_unavailable(
    context: NativeToolBuildContext, tool_name: str, reason: str
) -> None:
    context.task_service.append_event(
        context.task,
        "log",
        {
            "message": "native_tool_unavailable",
            "tool": tool_name,
            "reason": reason,
            "env_var": "KORTNY_SANDBOX_RUNNER_URL",
        },
    )


def _build_sandbox_bash_tool(context: NativeToolBuildContext) -> Tool | None:
    workbench = _workbench_session(context)
    if workbench is None:
        _log_workbench_unavailable(
            context, "sandbox_bash", "missing_sandbox_runner_url"
        )
        return None
    return SandboxBashTool(workbench=workbench)


def _build_sandbox_write_file_tool(context: NativeToolBuildContext) -> Tool | None:
    workbench = _workbench_session(context)
    if workbench is None:
        return None
    return SandboxWriteFileTool(workbench=workbench)


def _build_sandbox_read_file_tool(context: NativeToolBuildContext) -> Tool | None:
    workbench = _workbench_session(context)
    if workbench is None:
        return None
    return SandboxReadFileTool(workbench=workbench)


def _build_sandbox_export_artifact_tool(
    context: NativeToolBuildContext,
) -> Tool | None:
    workbench = _workbench_session(context)
    if workbench is None:
        return None
    return SandboxExportArtifactTool(
        workbench=workbench,
        working_dir=context.working_dir,
        session=context.session,
        task_id=context.task.id,
        task_service=context.task_service,
    )


def _build_sandbox_publish_preview_tool(
    context: NativeToolBuildContext,
) -> Tool | None:
    workbench = _workbench_session(context)
    if workbench is None:
        return None
    settings = context.settings
    if (
        not settings.artifacts_dir
        or not settings.public_base_url
        or not settings.preview_signing_secret
    ):
        _log_workbench_unavailable(
            context, "sandbox_publish_preview", "missing_preview_configuration"
        )
        return None
    return SandboxPublishPreviewTool(
        workbench=workbench,
        artifacts_dir=Path(settings.artifacts_dir),
        public_base_url=settings.public_base_url,
        signing_secret=settings.preview_signing_secret,
    )


def _build_deploy_site_tool(context: NativeToolBuildContext) -> Tool | None:
    settings = context.settings
    if not settings.netlify_auth_token and not settings.vercel_token:
        return None
    workbench = _workbench_session(context)
    if workbench is None:
        return None
    return DeploySiteTool(
        workbench=workbench,
        netlify_token=settings.netlify_auth_token,
        vercel_token=settings.vercel_token,
        vercel_team_id=settings.vercel_team_id,
    )


def _has_enabled_skills(context: NativeToolBuildContext) -> bool:
    from kortny.skills import SkillRegistryService

    return bool(
        SkillRegistryService(
            context.session, task_service=context.task_service
        ).enabled_skills_for_task(context.task)
    )


def _build_load_skill_tool(context: NativeToolBuildContext) -> Tool | None:
    if not _has_enabled_skills(context):
        return None
    return LoadSkillTool(
        session=context.session,
        task=context.task,
        task_service=context.task_service,
    )


def _build_load_skill_resource_tool(context: NativeToolBuildContext) -> Tool | None:
    if not _has_enabled_skills(context):
        return None
    return LoadSkillResourceTool(
        session=context.session,
        task=context.task,
        task_service=context.task_service,
    )


def _has_enabled_skill_scripts(context: NativeToolBuildContext) -> bool:
    from kortny.skills import SkillRegistryService

    enabled = SkillRegistryService(
        context.session, task_service=context.task_service
    ).enabled_skills_for_task(context.task)
    return any(skill.has_scripts for skill in enabled)


def _build_run_skill_script_tool(context: NativeToolBuildContext) -> Tool | None:
    if not _has_enabled_skill_scripts(context):
        return None
    workbench = _workbench_session(context)
    if workbench is None:
        _log_workbench_unavailable(
            context, "run_skill_script", "missing_sandbox_runner_url"
        )
        return None
    return RunSkillScriptTool(
        session=context.session,
        task=context.task,
        task_service=context.task_service,
        workbench=workbench,
    )


NATIVE_TOOL_REGISTRATIONS: tuple[NativeToolRegistration, ...] = (
    NativeToolRegistration("web_search", WebSearchTool, _build_web_search_tool),
    NativeToolRegistration(
        "pdf_generator", PdfGeneratorTool, _build_pdf_generator_tool
    ),
    NativeToolRegistration(
        "document_studio", DocumentStudioTool, _build_document_studio_tool
    ),
    NativeToolRegistration("code_exec", CodeExecTool, _build_code_exec_tool),
    NativeToolRegistration("sandbox_bash", SandboxBashTool, _build_sandbox_bash_tool),
    NativeToolRegistration(
        "sandbox_write_file", SandboxWriteFileTool, _build_sandbox_write_file_tool
    ),
    NativeToolRegistration(
        "sandbox_read_file", SandboxReadFileTool, _build_sandbox_read_file_tool
    ),
    NativeToolRegistration(
        "sandbox_export_artifact",
        SandboxExportArtifactTool,
        _build_sandbox_export_artifact_tool,
    ),
    NativeToolRegistration(
        "sandbox_publish_preview",
        SandboxPublishPreviewTool,
        _build_sandbox_publish_preview_tool,
    ),
    NativeToolRegistration("deploy_site", DeploySiteTool, _build_deploy_site_tool),
    NativeToolRegistration(
        "slack_channel_history",
        SlackChannelHistoryTool,
        _build_slack_channel_history_tool,
    ),
    NativeToolRegistration(
        "search_observed_slack_history",
        SearchObservedSlackHistoryTool,
        lambda context: SearchObservedSlackHistoryTool(
            session=context.session,
            task=context.task,
        ),
    ),
    NativeToolRegistration(
        "resolve_slack_identity",
        ResolveSlackIdentityTool,
        lambda context: ResolveSlackIdentityTool(
            session=context.session,
            task=context.task,
        ),
    ),
    NativeToolRegistration(
        "slack_user_info",
        SlackUserInfoTool,
        lambda context: SlackUserInfoTool(
            client=context.slack_identity_client,
            session=context.session,
            task=context.task,
        ),
    ),
    NativeToolRegistration(
        "slack_channel_info",
        SlackChannelInfoTool,
        lambda context: SlackChannelInfoTool(
            client=context.slack_identity_client,
            session=context.session,
            task=context.task,
        ),
    ),
    NativeToolRegistration(
        "slack_reply_thread",
        SlackReplyThreadTool,
        lambda context: SlackReplyThreadTool(
            client=context.slack_action_client,
            session=context.session,
            task=context.task,
            task_service=context.task_service,
        ),
    ),
    NativeToolRegistration(
        "slack_add_reaction",
        SlackAddReactionTool,
        lambda context: SlackAddReactionTool(
            client=context.slack_action_client,
            session=context.session,
            task=context.task,
            task_service=context.task_service,
        ),
    ),
    NativeToolRegistration(
        "slack_pin_message",
        SlackPinMessageTool,
        lambda context: SlackPinMessageTool(
            client=context.slack_action_client,
            session=context.session,
            task=context.task,
            task_service=context.task_service,
        ),
    ),
    NativeToolRegistration(
        "slack_add_bookmark",
        SlackAddBookmarkTool,
        lambda context: SlackAddBookmarkTool(
            client=context.slack_action_client,
            session=context.session,
            task=context.task,
            task_service=context.task_service,
        ),
    ),
    NativeToolRegistration(
        "slack_create_channel_canvas",
        SlackCreateChannelCanvasTool,
        lambda context: SlackCreateChannelCanvasTool(
            client=context.slack_action_client,
            session=context.session,
            task=context.task,
            task_service=context.task_service,
        ),
    ),
    NativeToolRegistration(
        "slack_lookup_canvas_sections",
        SlackLookupCanvasSectionsTool,
        lambda context: SlackLookupCanvasSectionsTool(
            client=context.slack_action_client,
            task=context.task,
        ),
    ),
    NativeToolRegistration(
        "slack_edit_canvas",
        SlackEditCanvasTool,
        lambda context: SlackEditCanvasTool(
            client=context.slack_action_client,
            session=context.session,
            task=context.task,
            task_service=context.task_service,
        ),
    ),
    NativeToolRegistration(
        "slack_file_read", SlackFileReadTool, _build_slack_file_read_tool
    ),
    NativeToolRegistration(
        "remember_fact",
        RememberFactTool,
        lambda context: RememberFactTool(
            service=context.memory_service,
            task=context.task,
        ),
    ),
    NativeToolRegistration(
        "recall_fact",
        RecallFactTool,
        lambda context: RecallFactTool(
            service=context.memory_service,
            task=context.task,
        ),
    ),
    NativeToolRegistration(
        "inspect_memory",
        InspectMemoryTool,
        lambda context: InspectMemoryTool(
            service=context.memory_service,
            task=context.task,
        ),
    ),
    NativeToolRegistration(
        "forget_fact",
        ForgetFactTool,
        lambda context: ForgetFactTool(
            service=context.memory_service,
            task=context.task,
        ),
    ),
    NativeToolRegistration(
        "query_workspace_graph",
        QueryWorkspaceGraphTool,
        lambda context: QueryWorkspaceGraphTool(
            session=context.session,
            task=context.task,
        ),
    ),
    NativeToolRegistration(
        "declare_project",
        DeclareProjectTool,
        lambda context: DeclareProjectTool(
            session=context.session,
            task=context.task,
        ),
    ),
    NativeToolRegistration("load_skill", LoadSkillTool, _build_load_skill_tool),
    NativeToolRegistration(
        "load_skill_resource",
        LoadSkillResourceTool,
        _build_load_skill_resource_tool,
    ),
    NativeToolRegistration(
        "run_skill_script",
        RunSkillScriptTool,
        _build_run_skill_script_tool,
    ),
    NativeToolRegistration(
        "list_schedules",
        ListSchedulesTool,
        lambda context: ListSchedulesTool(
            session=context.session,
            task=context.task,
        ),
    ),
    NativeToolRegistration(
        "get_schedule",
        GetScheduleTool,
        lambda context: GetScheduleTool(
            session=context.session,
            task=context.task,
        ),
    ),
    NativeToolRegistration(
        "create_schedule",
        CreateScheduleTool,
        lambda context: CreateScheduleTool(
            session=context.session,
            task=context.task,
            task_service=context.task_service,
        ),
    ),
    NativeToolRegistration(
        "update_schedule",
        UpdateScheduleTool,
        lambda context: UpdateScheduleTool(
            session=context.session,
            task=context.task,
            task_service=context.task_service,
        ),
    ),
    NativeToolRegistration(
        "pause_schedule",
        PauseScheduleTool,
        lambda context: PauseScheduleTool(
            session=context.session,
            task=context.task,
            task_service=context.task_service,
        ),
    ),
    NativeToolRegistration(
        "resume_schedule",
        ResumeScheduleTool,
        lambda context: ResumeScheduleTool(
            session=context.session,
            task=context.task,
            task_service=context.task_service,
        ),
    ),
    NativeToolRegistration(
        "cancel_schedule",
        CancelScheduleTool,
        lambda context: CancelScheduleTool(
            session=context.session,
            task=context.task,
            task_service=context.task_service,
        ),
    ),
)

NATIVE_INVENTORY_TOOL_REGISTRATIONS: tuple[NativeInventoryToolRegistration, ...] = (
    NativeInventoryToolRegistration(
        "describe_tools",
        DescribeToolsTool,
        _build_describe_tools_tool,
    ),
    NativeInventoryToolRegistration(
        "list_integrations",
        ListIntegrationsTool,
        _build_list_integrations_tool,
    ),
)

_registered_names = set(native_tool_classes_by_name())
_runtime_metadata_names = set(runtime_native_tool_names())
if _registered_names != _runtime_metadata_names:
    missing_metadata = ", ".join(sorted(_registered_names - _runtime_metadata_names))
    missing_registration = ", ".join(
        sorted(_runtime_metadata_names - _registered_names)
    )
    details = []
    if missing_metadata:
        details.append(f"missing metadata: {missing_metadata}")
    if missing_registration:
        details.append(f"missing registration: {missing_registration}")
    raise RuntimeError(f"Native tool catalog mismatch ({'; '.join(details)})")
