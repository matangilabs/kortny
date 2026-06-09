import base64
import io
import json
import tarfile
import zipfile

import httpx
import pytest

from kortny.tools.deploy_site import DeploySiteTool
from kortny.tools.sandbox_workbench import WorkbenchSession
from tests.test_sandbox_workbench_tools import FakeSessionClient, FakeTask


def _tar_with(entries: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for name, content in entries.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


def _tool(
    handler,
    *,
    netlify_token: str | None = None,
    vercel_token: str | None = None,
) -> DeploySiteTool:
    client = FakeSessionClient()
    client.archive = _tar_with(
        {"dist/index.html": b"<html></html>", "dist/app.js": b"console.log(1)"}
    )
    workbench = WorkbenchSession(client=client, task=FakeTask())
    return DeploySiteTool(
        workbench=workbench,
        netlify_token=netlify_token,
        vercel_token=vercel_token,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        poll_seconds=0.0,
    )


def test_deploy_requires_configured_provider_token() -> None:
    tool = _tool(lambda request: httpx.Response(500))

    result = tool.invoke(
        {
            "provider": "netlify",
            "source_path": "/workspace/dist",
            "site_name": "my-dash",
        }
    )

    assert result.output["successful"] is False
    assert result.output["error"]["code"] == "deploy_provider_not_configured"


def test_deploy_netlify_creates_site_with_zip_and_polls_ready() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/api/v1/sites":
            return httpx.Response(200, json=[])
        if request.method == "POST" and request.url.path == "/api/v1/sites":
            assert request.headers["content-type"] == "application/zip"
            with zipfile.ZipFile(io.BytesIO(request.content)) as archive:
                seen["zip_names"] = sorted(archive.namelist())
            return httpx.Response(
                200,
                json={"id": "site-1", "deploy_id": "deploy-1", "url": "http://x"},
            )
        if request.url.path == "/api/v1/deploys/deploy-1":
            return httpx.Response(
                200,
                json={"state": "ready", "ssl_url": "https://my-dash.netlify.app"},
            )
        return httpx.Response(404)

    tool = _tool(handler, netlify_token="netlify-token")

    result = tool.invoke(
        {
            "provider": "netlify",
            "source_path": "/workspace/dist",
            "site_name": "My Dash",
        }
    )

    assert result.output["successful"] is True
    assert result.output["url"] == "https://my-dash.netlify.app"
    assert result.output["site_name"] == "my-dash"
    assert seen["zip_names"] == ["app.js", "index.html"]


def test_deploy_netlify_reuses_existing_site() -> None:
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(f"{request.method} {request.url.path}")
        if request.method == "GET" and request.url.path == "/api/v1/sites":
            return httpx.Response(200, json=[{"id": "site-9", "name": "my-dash"}])
        if request.url.path == "/api/v1/sites/site-9/deploys":
            return httpx.Response(200, json={"id": "deploy-9"})
        if request.url.path == "/api/v1/deploys/deploy-9":
            return httpx.Response(
                200, json={"state": "ready", "ssl_url": "https://d.netlify.app"}
            )
        return httpx.Response(404)

    tool = _tool(handler, netlify_token="netlify-token")

    result = tool.invoke(
        {
            "provider": "netlify",
            "source_path": "/workspace/dist",
            "site_name": "my-dash",
        }
    )

    assert result.output["successful"] is True
    assert "POST /api/v1/sites/site-9/deploys" in paths


def test_deploy_vercel_inlines_files_as_base64() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v13/deployments"
        body = json.loads(request.content)
        seen["files"] = {entry["file"]: entry for entry in body["files"]}
        seen["name"] = body["name"]
        seen["target"] = body.get("target")
        return httpx.Response(
            200,
            json={"id": "dpl_1", "url": "my-dash.vercel.app", "readyState": "QUEUED"},
        )

    tool = _tool(handler, vercel_token="vercel-token")

    result = tool.invoke(
        {
            "provider": "vercel",
            "source_path": "/workspace/dist",
            "site_name": "my-dash",
            "production": True,
        }
    )

    assert result.output["successful"] is True
    assert result.output["url"] == "https://my-dash.vercel.app"
    assert seen["name"] == "my-dash"
    assert seen["target"] == "production"
    index_entry = seen["files"]["index.html"]
    assert base64.b64decode(index_entry["data"]) == b"<html></html>"
    assert index_entry["encoding"] == "base64"


def test_deploy_maps_provider_error_to_recoverable_result() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/api/v1/sites":
            return httpx.Response(200, json=[])
        return httpx.Response(401, text="unauthorized")

    tool = _tool(handler, netlify_token="bad-token")

    result = tool.invoke(
        {
            "provider": "netlify",
            "source_path": "/workspace/dist",
            "site_name": "my-dash",
        }
    )

    assert result.output["successful"] is False
    assert result.output["error"]["code"] == "deploy_failed"
    assert "401" in result.output["error"]["message"]


def test_deploy_validates_arguments() -> None:
    tool = _tool(lambda request: httpx.Response(500), netlify_token="t")

    with pytest.raises(ValueError):
        tool.invoke({"provider": "render", "source_path": "/x", "site_name": "a"})
    with pytest.raises(ValueError):
        tool.invoke({"provider": "netlify", "source_path": "", "site_name": "a"})
