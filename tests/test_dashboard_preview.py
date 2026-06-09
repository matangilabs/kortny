from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from kortny.dashboard.app import create_app
from kortny.dashboard.settings import DashboardAuthMode, DashboardSettings
from kortny.execution.preview import preview_token


def _client(tmp_path: Path, *, secret: str | None = "preview-secret") -> TestClient:
    settings = DashboardSettings(
        postgres_url="postgresql://kortny:kortny@localhost:5432/kortny",
        auth_mode=DashboardAuthMode.bootstrap,
        artifacts_dir=str(tmp_path),
        preview_signing_secret=secret,
    )
    app = create_app(settings=settings, session_factory=sessionmaker())
    return TestClient(app)


def _publish(tmp_path: Path, task_id: str, slug: str) -> None:
    site = tmp_path / task_id / slug
    site.mkdir(parents=True)
    (site / "index.html").write_text("<html>dash</html>")
    (site / "data.json").write_text("{}")


def test_preview_serves_published_file_without_login(tmp_path: Path) -> None:
    _publish(tmp_path, "task-1", "dash")
    token = preview_token("preview-secret", "task-1", "dash")
    client = _client(tmp_path)

    response = client.get(f"/preview/{token}/task-1/dash/index.html")

    assert response.status_code == 200
    assert response.text == "<html>dash</html>"
    assert "text/html" in response.headers["content-type"]


def test_preview_rejects_bad_token(tmp_path: Path) -> None:
    _publish(tmp_path, "task-1", "dash")
    client = _client(tmp_path)

    response = client.get("/preview/0000000000000000/task-1/dash/index.html")

    assert response.status_code == 404


def test_preview_token_is_scoped_to_task_and_slug(tmp_path: Path) -> None:
    _publish(tmp_path, "task-1", "dash")
    _publish(tmp_path, "task-2", "dash")
    token_for_task_1 = preview_token("preview-secret", "task-1", "dash")
    client = _client(tmp_path)

    response = client.get(f"/preview/{token_for_task_1}/task-2/dash/index.html")

    assert response.status_code == 404


def test_preview_blocks_path_traversal(tmp_path: Path) -> None:
    _publish(tmp_path, "task-1", "dash")
    (tmp_path / "secret.txt").write_text("nope")
    token = preview_token("preview-secret", "task-1", "dash")
    client = _client(tmp_path)

    response = client.get(f"/preview/{token}/task-1/dash/..%2F..%2Fsecret.txt")

    assert response.status_code == 404


def test_preview_404_when_not_configured(tmp_path: Path) -> None:
    _publish(tmp_path, "task-1", "dash")
    token = preview_token("preview-secret", "task-1", "dash")
    client = _client(tmp_path, secret=None)

    response = client.get(f"/preview/{token}/task-1/dash/index.html")

    assert response.status_code == 404
