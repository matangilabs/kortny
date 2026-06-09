"""Trusted-host static site deployment to Netlify or Vercel.

The sandbox produces files; this tool runs on the worker, extracts them,
and calls the provider API with tokens from settings. Integration tokens
never enter the sandbox, so prompt-injected code cannot exfiltrate them.
"""

from __future__ import annotations

import base64
import io
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any

import httpx

from kortny.execution.preview import (
    UnsafeArchiveError,
    extract_tar_to_dir,
    safe_slug,
)
from kortny.execution.sandbox import SandboxUnavailableError
from kortny.execution.sandbox_sessions import SandboxSessionError
from kortny.tools.sandbox_workbench import WorkbenchSession
from kortny.tools.types import JsonObject, JsonSchema, ToolResult

NETLIFY_API_BASE = "https://api.netlify.com/api/v1"
VERCEL_API_BASE = "https://api.vercel.com"
MAX_DEPLOY_BYTES = 25 * 1024 * 1024
MAX_DEPLOY_FILES = 500
DEPLOY_POLL_SECONDS = 3.0
DEPLOY_POLL_ATTEMPTS = 40


class DeploySiteTool:
    """Deploy a sandbox-built static site through a connected provider."""

    name = "deploy_site"
    description = (
        "Deploys a built static site directory from the sandbox workspace to "
        "Netlify or Vercel and returns the live URL. Only use when the user "
        "explicitly asks to deploy or publish. Build and verify the site with "
        "sandbox_bash first; pass the directory that contains index.html "
        "(for example /workspace/app/dist)."
    )
    parameters: JsonSchema = {
        "type": "object",
        "properties": {
            "provider": {
                "type": "string",
                "enum": ["netlify", "vercel"],
                "description": "Deployment provider to use.",
            },
            "source_path": {
                "type": "string",
                "description": (
                    "Sandbox directory under /workspace containing the built "
                    "static site."
                ),
            },
            "site_name": {
                "type": "string",
                "description": "Project/site name on the provider.",
            },
            "production": {
                "type": "boolean",
                "default": False,
                "description": "Deploy to production instead of a preview.",
            },
        },
        "required": ["provider", "source_path", "site_name"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        *,
        workbench: WorkbenchSession,
        netlify_token: str | None = None,
        vercel_token: str | None = None,
        vercel_team_id: str | None = None,
        http_client: httpx.Client | None = None,
        poll_seconds: float = DEPLOY_POLL_SECONDS,
    ) -> None:
        self.workbench = workbench
        self.netlify_token = netlify_token
        self.vercel_token = vercel_token
        self.vercel_team_id = vercel_team_id
        self.http_client = http_client
        self.poll_seconds = poll_seconds

    def invoke(self, args: JsonObject) -> ToolResult:
        provider = args.get("provider")
        if provider not in ("netlify", "vercel"):
            raise ValueError("deploy_site 'provider' must be netlify or vercel")
        source_path = args.get("source_path")
        if not isinstance(source_path, str) or not source_path.strip():
            raise ValueError("deploy_site requires a non-empty 'source_path'")
        site_name = safe_slug(str(args.get("site_name") or ""), default="kortny-site")
        production = bool(args.get("production", False))

        token = self.netlify_token if provider == "netlify" else self.vercel_token
        if not token:
            return _error_result(
                code="deploy_provider_not_configured",
                message=(
                    f"No {provider} token is configured. Set "
                    f"{'NETLIFY_AUTH_TOKEN' if provider == 'netlify' else 'VERCEL_TOKEN'} "
                    "or connect the integration."
                ),
            )

        files = self._collect_files(source_path.strip())
        if isinstance(files, ToolResult):
            return files
        if not files:
            return _error_result(
                code="deploy_source_empty",
                message=f"No files found at sandbox path {source_path}.",
            )

        if provider == "netlify":
            return self._deploy_netlify(
                files, site_name=site_name, token=token, production=production
            )
        return self._deploy_vercel(
            files, site_name=site_name, token=token, production=production
        )

    def _collect_files(
        self, source_path: str
    ) -> dict[str, bytes] | ToolResult:
        try:
            session = self.workbench.ensure()
            tar_bytes = self.workbench.client.export_archive(
                session.session_id, source_path
            )
        except (SandboxUnavailableError, SandboxSessionError) as exc:
            return _error_result(code="sandbox_session_error", message=str(exc))

        staging = Path(f"/tmp/kortny-deploy-{uuid.uuid4().hex[:8]}")
        staging.mkdir(parents=True, exist_ok=True)
        staging = staging.resolve()
        try:
            extracted = extract_tar_to_dir(
                tar_bytes, staging, max_bytes=MAX_DEPLOY_BYTES
            )
        except UnsafeArchiveError as exc:
            return _error_result(
                code="deploy_source_invalid",
                message=f"Unsafe sandbox archive: {exc}",
            )
        if len(extracted) > MAX_DEPLOY_FILES:
            return _error_result(
                code="deploy_source_too_large",
                message=f"Deploy exceeds {MAX_DEPLOY_FILES} files.",
            )
        return {
            str(path.relative_to(staging)): path.read_bytes()
            for path in extracted
        }

    def _deploy_netlify(
        self,
        files: dict[str, bytes],
        *,
        site_name: str,
        token: str,
        production: bool,
    ) -> ToolResult:
        del production  # Netlify zip deploys to the site's main context.
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, content in sorted(files.items()):
                archive.writestr(name, content)
        zip_bytes = zip_buffer.getvalue()

        client = self._client()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/zip",
        }
        try:
            site_id = self._netlify_site_id(client, site_name, token)
            if site_id is None:
                response = client.post(
                    f"{NETLIFY_API_BASE}/sites",
                    params={"name": site_name},
                    content=zip_bytes,
                    headers=headers,
                    timeout=120.0,
                )
                if not response.is_success:
                    return _provider_error("netlify", response)
                payload = response.json()
                deploy_id = _str_field(payload, "deploy_id") or _str_field(
                    payload, "id"
                )
                site_url = _str_field(payload, "ssl_url") or _str_field(
                    payload, "url"
                )
            else:
                response = client.post(
                    f"{NETLIFY_API_BASE}/sites/{site_id}/deploys",
                    content=zip_bytes,
                    headers=headers,
                    timeout=120.0,
                )
                if not response.is_success:
                    return _provider_error("netlify", response)
                payload = response.json()
                deploy_id = _str_field(payload, "id")
                site_url = _str_field(payload, "ssl_url") or _str_field(
                    payload, "deploy_ssl_url"
                )
            state, deploy_url = self._poll_netlify_deploy(client, deploy_id, token)
        except httpx.HTTPError as exc:
            return _error_result(
                code="deploy_failed",
                message=f"Netlify request failed: {type(exc).__name__}: {exc}",
            )
        return ToolResult(
            output={
                "successful": state == "ready",
                "provider": "netlify",
                "site_name": site_name,
                "deploy_id": deploy_id,
                "state": state,
                "url": deploy_url or site_url,
                "file_count": len(files),
            }
        )

    def _netlify_site_id(
        self, client: httpx.Client, site_name: str, token: str
    ) -> str | None:
        response = client.get(
            f"{NETLIFY_API_BASE}/sites",
            params={"name": site_name},
            headers={"Authorization": f"Bearer {token}"},
            timeout=30.0,
        )
        if not response.is_success:
            return None
        payload = response.json()
        if not isinstance(payload, list):
            return None
        for site in payload:
            if isinstance(site, dict) and site.get("name") == site_name:
                site_id = site.get("id")
                return site_id if isinstance(site_id, str) else None
        return None

    def _poll_netlify_deploy(
        self, client: httpx.Client, deploy_id: str | None, token: str
    ) -> tuple[str, str | None]:
        if not deploy_id:
            return "unknown", None
        state = "unknown"
        deploy_url: str | None = None
        for _ in range(DEPLOY_POLL_ATTEMPTS):
            response = client.get(
                f"{NETLIFY_API_BASE}/deploys/{deploy_id}",
                headers={"Authorization": f"Bearer {token}"},
                timeout=30.0,
            )
            if not response.is_success:
                break
            payload = response.json()
            state = _str_field(payload, "state") or "unknown"
            deploy_url = _str_field(payload, "ssl_url") or _str_field(
                payload, "deploy_ssl_url"
            )
            if state in ("ready", "error"):
                break
            time.sleep(self.poll_seconds)
        return state, deploy_url

    def _deploy_vercel(
        self,
        files: dict[str, bytes],
        *,
        site_name: str,
        token: str,
        production: bool,
    ) -> ToolResult:
        client = self._client()
        params: dict[str, str] = {}
        if self.vercel_team_id:
            params["teamId"] = self.vercel_team_id
        body: JsonObject = {
            "name": site_name,
            "files": [
                {
                    "file": name,
                    "data": base64.b64encode(content).decode("ascii"),
                    "encoding": "base64",
                }
                for name, content in sorted(files.items())
            ],
            "projectSettings": {"framework": None},
        }
        if production:
            body["target"] = "production"
        try:
            response = client.post(
                f"{VERCEL_API_BASE}/v13/deployments",
                params=params,
                json=body,
                headers={"Authorization": f"Bearer {token}"},
                timeout=120.0,
            )
        except httpx.HTTPError as exc:
            return _error_result(
                code="deploy_failed",
                message=f"Vercel request failed: {type(exc).__name__}: {exc}",
            )
        if not response.is_success:
            return _provider_error("vercel", response)
        payload = response.json()
        url = _str_field(payload, "url")
        return ToolResult(
            output={
                "successful": True,
                "provider": "vercel",
                "site_name": site_name,
                "deploy_id": _str_field(payload, "id"),
                "state": _str_field(payload, "readyState") or "QUEUED",
                "url": f"https://{url}" if url and "://" not in url else url,
                "file_count": len(files),
            }
        )

    def _client(self) -> httpx.Client:
        return self.http_client or httpx.Client(timeout=120.0)


def _provider_error(provider: str, response: httpx.Response) -> ToolResult:
    return _error_result(
        code="deploy_failed",
        message=(
            f"{provider} API returned {response.status_code}: "
            f"{response.text[:300]}"
        ),
    )


def _error_result(*, code: str, message: str) -> ToolResult:
    return ToolResult(
        output={
            "successful": False,
            "error": {"code": code, "message": message, "recoverable": True},
        }
    )


def _str_field(payload: Any, key: str) -> str | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get(key)
    return value if isinstance(value, str) and value else None
