"""Tools for loading and running enabled procedural skills during task execution.

The L1 name+description block in the task context advertises enabled skills;
these tools are the L2/L3 progressive-disclosure path: ``load_skill`` returns
the full SKILL.md instructions, ``load_skill_resource`` returns one bundled
reference/asset/script file, and ``run_skill_script`` executes a trusted
skill's bundled script inside the per-task sandbox. Script execution is gated on
the skill's trust level (``trusted`` only); untrusted/community skills keep
their scripts viewable but never executable.
"""

from __future__ import annotations

import shlex

from sqlalchemy import select
from sqlalchemy.orm import Session

from kortny.db.models import (
    ProceduralSkillVersion,
    SkillFile,
    Task,
    TaskEventType,
)
from kortny.execution.sandbox import SandboxUnavailableError
from kortny.execution.sandbox_sessions import SandboxSessionError
from kortny.skills import (
    EXECUTION_INVOCATION,
    SCRIPT_EXECUTION_INVOCATION,
    EnabledSkill,
    SkillActivation,
    SkillRegistryService,
)
from kortny.tasks import TaskService
from kortny.tools.sandbox_workbench import (
    MAX_BASH_TIMEOUT_SECONDS,
    WorkbenchSession,
    _exec_output,
    _sandbox_error_result,
)
from kortny.tools.types import (
    JsonObject,
    JsonSchema,
    RecoverableToolError,
    ToolResult,
)

MAX_RESOURCE_CHARS = 60_000
DEFAULT_SCRIPT_TIMEOUT_SECONDS = 300
SCRIPT_INTERPRETERS: dict[str, str] = {
    ".py": "python",
    ".sh": "bash",
    ".bash": "bash",
}
MATERIALIZED_SKILLS_ROOT = "/workspace/skills"


class LoadSkillTool:
    """Load the full instructions for a skill enabled in this task's scope."""

    name = "load_skill"
    description = (
        "Loads the full instructions for one of the available skills listed "
        "in <available_skills>. Call this BEFORE doing the work whenever a "
        "skill's description matches the task, then follow the returned "
        "instructions."
    )
    parameters: JsonSchema = {
        "type": "object",
        "properties": {
            "slug": {
                "type": "string",
                "description": "The skill slug from the available skills list.",
            },
        },
        "required": ["slug"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        *,
        session: Session,
        task: Task,
        task_service: TaskService,
    ) -> None:
        self.session = session
        self.task = task
        self.registry = SkillRegistryService(session, task_service=task_service)

    def invoke(self, args: JsonObject) -> ToolResult:
        slug = str(args.get("slug") or "").strip()
        enabled = _enabled_skill_or_raise(self.registry, self.task, slug)
        version = self.session.get(ProceduralSkillVersion, enabled.version_id)
        if version is None:  # pragma: no cover - enablement implies a version
            raise RecoverableToolError(
                code="skill_version_missing",
                message=f"Skill '{slug}' has no active version.",
            )
        resource_paths = list(
            self.session.scalars(
                select(SkillFile.path)
                .where(SkillFile.skill_version_id == enabled.version_id)
                .order_by(SkillFile.path)
            )
        )
        self.registry.record_invocation(
            self.task,
            activation=SkillActivation(
                skill_id=enabled.skill_id,
                skill_version_id=enabled.version_id,
                slug=enabled.slug,
                name=enabled.name,
                version=enabled.version,
                owner_type=enabled.owner_type,
                trust_level=enabled.trust_level,
                instructions_md=version.instructions_md,
                selected_reason="model-triggered via load_skill",
            ),
            invocation_kind=EXECUTION_INVOCATION,
            response_mode="execution",
        )
        output: JsonObject = {
            "slug": enabled.slug,
            "name": enabled.name,
            "version": enabled.version,
            "trust_level": enabled.trust_level,
            "instructions_md": version.instructions_md,
            "resources": resource_paths,
        }
        if any(path.startswith("scripts/") for path in resource_paths):
            if enabled.trust_level == "trusted":
                output["scripts_note"] = (
                    "Bundled scripts are runnable in this task's sandbox via "
                    "run_skill_script(slug, path); they are also viewable with "
                    "load_skill_resource."
                )
            else:
                output["scripts_note"] = (
                    "Bundled scripts are viewable with load_skill_resource but "
                    "are not executable at this skill's trust level."
                )
        return ToolResult(output=output)


class LoadSkillResourceTool:
    """Load one bundled file (reference/asset/script) from an enabled skill."""

    name = "load_skill_resource"
    description = (
        "Loads a bundled resource file from an enabled skill, e.g. "
        "'references/guide.md'. Use the resource paths returned by load_skill."
    )
    parameters: JsonSchema = {
        "type": "object",
        "properties": {
            "slug": {
                "type": "string",
                "description": "The skill slug from the available skills list.",
            },
            "path": {
                "type": "string",
                "description": (
                    "Relative resource path, e.g. 'references/guide.md', "
                    "'assets/template.txt', or 'scripts/run.py'."
                ),
            },
        },
        "required": ["slug", "path"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        *,
        session: Session,
        task: Task,
        task_service: TaskService,
    ) -> None:
        self.session = session
        self.task = task
        self.registry = SkillRegistryService(session, task_service=task_service)

    def invoke(self, args: JsonObject) -> ToolResult:
        slug = str(args.get("slug") or "").strip()
        path = str(args.get("path") or "").strip()
        enabled = _enabled_skill_or_raise(self.registry, self.task, slug)
        resource = self.session.scalar(
            select(SkillFile).where(
                SkillFile.skill_version_id == enabled.version_id,
                SkillFile.path == path,
            )
        )
        if resource is None:
            available = list(
                self.session.scalars(
                    select(SkillFile.path)
                    .where(SkillFile.skill_version_id == enabled.version_id)
                    .order_by(SkillFile.path)
                )
            )
            raise RecoverableToolError(
                code="skill_resource_not_found",
                message=f"Resource '{path}' not found in skill '{slug}'.",
                hint=f"Available resources: {', '.join(available) or 'none'}",
            )
        if resource.content_text is None:
            raise RecoverableToolError(
                code="skill_resource_binary",
                message=(
                    f"Resource '{path}' is binary and cannot be returned as text."
                ),
            )
        content = resource.content_text
        truncated = False
        if len(content) > MAX_RESOURCE_CHARS:
            content = content[:MAX_RESOURCE_CHARS]
            truncated = True
        return ToolResult(
            output={
                "slug": enabled.slug,
                "path": path,
                "kind": resource.kind,
                "content": content,
                "truncated": truncated,
            }
        )


class RunSkillScriptTool:
    """Run a trusted skill's bundled script inside the per-task sandbox."""

    name = "run_skill_script"
    description = (
        "Runs one of a trusted skill's bundled scripts (scripts/*.py, *.sh, "
        "*.bash) inside this task's isolated sandbox workspace. Only skills at "
        "the 'trusted' trust level can run scripts. Call load_skill first to "
        "see a skill's instructions and available scripts."
    )
    parameters: JsonSchema = {
        "type": "object",
        "properties": {
            "slug": {
                "type": "string",
                "description": "The skill slug from the available skills list.",
            },
            "path": {
                "type": "string",
                "description": (
                    "Script path, e.g. 'scripts/run.py' or bare 'run.py' "
                    "(normalized under scripts/)."
                ),
            },
            "args": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional command-line arguments for the script.",
            },
            "timeout_seconds": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_BASH_TIMEOUT_SECONDS,
                "default": DEFAULT_SCRIPT_TIMEOUT_SECONDS,
                "description": "Wall-clock timeout for the script.",
            },
        },
        "required": ["slug", "path"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        *,
        session: Session,
        task: Task,
        task_service: TaskService,
        workbench: WorkbenchSession,
    ) -> None:
        self.session = session
        self.task = task
        self.task_service = task_service
        self.workbench = workbench
        self.registry = SkillRegistryService(session, task_service=task_service)

    def invoke(self, args: JsonObject) -> ToolResult:
        slug = str(args.get("slug") or "").strip()
        enabled = _enabled_skill_or_raise(self.registry, self.task, slug)
        version = self.session.get(ProceduralSkillVersion, enabled.version_id)
        if version is None:  # pragma: no cover - enablement implies a version
            raise RecoverableToolError(
                code="skill_version_missing",
                message=f"Skill '{slug}' has no active version.",
            )

        if enabled.trust_level != "trusted":
            self._record_blocked(enabled, self._normalize_path(args))
            raise RecoverableToolError(
                code="skill_scripts_blocked_by_trust",
                message=(
                    f"Skill '{slug}' is '{enabled.trust_level}', so its scripts "
                    "cannot be executed."
                ),
                hint=(
                    "An admin can promote this skill to trusted on the "
                    "dashboard Skills page."
                ),
            )

        script_path = self._normalize_path(args)
        files = list(
            self.session.scalars(
                select(SkillFile)
                .where(SkillFile.skill_version_id == enabled.version_id)
                .order_by(SkillFile.path)
            )
        )
        self._resolve_script(files, slug, script_path)
        interpreter = self._interpreter_for(slug, script_path)

        self._materialize(files, slug, version)

        command = self._build_command(interpreter, script_path, args)
        workdir = f"{MATERIALIZED_SKILLS_ROOT}/{slug}"
        try:
            session_info = self.workbench.ensure()
            result = self.workbench.client.exec(
                session_info.session_id,
                command,
                workdir=workdir,
                timeout_seconds=self._timeout_seconds(args),
            )
        except (SandboxUnavailableError, SandboxSessionError) as exc:
            return _sandbox_error_result(exc)

        self._record_execution(
            enabled,
            version,
            path=script_path,
            command=command,
            exit_code=result.exit_code,
            stdout_len=len(result.stdout),
            stderr_len=len(result.stderr),
            timed_out=result.timed_out,
        )

        output: JsonObject = {
            "slug": enabled.slug,
            "path": script_path,
            **_exec_output(result),
        }
        return ToolResult(output=output)

    def _normalize_path(self, args: JsonObject) -> str:
        raw = str(args.get("path") or "").strip()
        if not raw:
            raise RecoverableToolError(
                code="skill_script_path_required",
                message="Argument 'path' is required.",
            )
        if raw.startswith("scripts/"):
            return raw
        if "/" in raw:
            raise RecoverableToolError(
                code="skill_script_unsupported",
                message=(
                    f"Script path '{raw}' must be under scripts/ for skill execution."
                ),
            )
        return f"scripts/{raw}"

    def _resolve_script(
        self, files: list[SkillFile], slug: str, script_path: str
    ) -> SkillFile:
        scripts = [f for f in files if f.kind == "script"]
        match = next((f for f in scripts if f.path == script_path), None)
        if match is None:
            available = ", ".join(f.path for f in scripts) or "none"
            raise RecoverableToolError(
                code="skill_script_not_found",
                message=f"Script '{script_path}' not found in skill '{slug}'.",
                hint=f"Available scripts: {available}",
            )
        return match

    def _interpreter_for(self, slug: str, script_path: str) -> str:
        suffix = "." + script_path.rsplit(".", 1)[-1] if "." in script_path else ""
        interpreter = SCRIPT_INTERPRETERS.get(suffix)
        if interpreter is None:
            supported = ", ".join(sorted(SCRIPT_INTERPRETERS))
            raise RecoverableToolError(
                code="skill_script_unsupported",
                message=(
                    f"Script '{script_path}' has an unsupported extension; "
                    f"only {supported} are runnable."
                ),
            )
        return interpreter

    def _materialize(
        self,
        files: list[SkillFile],
        slug: str,
        version: ProceduralSkillVersion,
    ) -> None:
        session_info = self.workbench.ensure()
        base = f"{MATERIALIZED_SKILLS_ROOT}/{slug}"
        for file in files:
            if file.content_text is not None:
                content = file.content_text.encode("utf-8")
            elif file.content_bytes is not None:
                content = file.content_bytes
            else:  # pragma: no cover - defensive
                continue
            self.workbench.client.write_file(
                session_info.session_id, f"{base}/{file.path}", content
            )
        self.workbench.client.write_file(
            session_info.session_id,
            f"{base}/SKILL.md",
            version.instructions_md.encode("utf-8"),
        )

    def _build_command(
        self, interpreter: str, script_path: str, args: JsonObject
    ) -> str:
        parts = [interpreter, script_path]
        raw_args = args.get("args")
        if raw_args is not None:
            if not isinstance(raw_args, list) or not all(
                isinstance(item, str) for item in raw_args
            ):
                raise RecoverableToolError(
                    code="skill_script_invalid_args",
                    message="Argument 'args' must be an array of strings.",
                )
            parts.extend(raw_args)
        return shlex.join(parts)

    def _timeout_seconds(self, args: JsonObject) -> int:
        value = args.get("timeout_seconds", DEFAULT_SCRIPT_TIMEOUT_SECONDS)
        if not isinstance(value, int):
            raise RecoverableToolError(
                code="skill_script_invalid_timeout",
                message="Argument 'timeout_seconds' must be an integer.",
            )
        if value < 1 or value > MAX_BASH_TIMEOUT_SECONDS:
            raise RecoverableToolError(
                code="skill_script_invalid_timeout",
                message=(
                    f"Argument 'timeout_seconds' must be between 1 and "
                    f"{MAX_BASH_TIMEOUT_SECONDS}."
                ),
            )
        return value

    def _record_blocked(self, enabled: EnabledSkill, path: str) -> None:
        self.task_service.append_event(
            self.task,
            TaskEventType.log,
            {
                "message": "skill_script_blocked",
                "slug": enabled.slug,
                "version": enabled.version,
                "trust_level": enabled.trust_level,
                "path": path,
            },
        )

    def _record_execution(
        self,
        enabled: EnabledSkill,
        version: ProceduralSkillVersion,
        *,
        path: str,
        command: str,
        exit_code: int,
        stdout_len: int,
        stderr_len: int,
        timed_out: bool,
    ) -> None:
        self.registry.record_invocation(
            self.task,
            activation=SkillActivation(
                skill_id=enabled.skill_id,
                skill_version_id=enabled.version_id,
                slug=enabled.slug,
                name=enabled.name,
                version=enabled.version,
                owner_type=enabled.owner_type,
                trust_level=enabled.trust_level,
                instructions_md=version.instructions_md,
                selected_reason=f"model-triggered via run_skill_script ({path})",
            ),
            invocation_kind=SCRIPT_EXECUTION_INVOCATION,
            response_mode="execution",
        )
        self.task_service.append_event(
            self.task,
            TaskEventType.log,
            {
                "message": "skill_script_executed",
                "slug": enabled.slug,
                "version": enabled.version,
                "path": path,
                "command": command,
                "exit_code": exit_code,
                "stdout_len": stdout_len,
                "stderr_len": stderr_len,
                "timed_out": timed_out,
            },
        )


def _enabled_skill_or_raise(
    registry: SkillRegistryService,
    task: Task,
    slug: str,
) -> EnabledSkill:
    if not slug:
        raise RecoverableToolError(
            code="skill_slug_required",
            message="Argument 'slug' is required.",
        )
    enabled = {skill.slug: skill for skill in registry.enabled_skills_for_task(task)}
    skill = enabled.get(slug)
    if skill is None:
        raise RecoverableToolError(
            code="skill_not_enabled",
            message=f"Skill '{slug}' is not enabled for this task.",
            hint=f"Enabled skills: {', '.join(sorted(enabled)) or 'none'}",
        )
    return skill
