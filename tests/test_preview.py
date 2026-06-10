import io
import tarfile
from pathlib import Path

import pytest

from kortny.execution.preview import (
    UnsafeArchiveError,
    extract_tar_to_dir,
    preview_token,
    preview_url,
    safe_slug,
    verify_preview_token,
)


def _tar_with(entries: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for name, content in entries.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


def test_preview_token_is_deterministic_and_verifiable() -> None:
    token = preview_token("secret", "task-1", "dash")

    assert token == preview_token("secret", "task-1", "dash")
    assert len(token) == 16
    assert verify_preview_token("secret", "task-1", "dash", token)
    assert not verify_preview_token("secret", "task-1", "other", token)
    assert not verify_preview_token("other", "task-1", "dash", token)


def test_preview_url_embeds_token_and_paths() -> None:
    url = preview_url("https://kortny.example.com/", "secret", "task-1", "dash")

    token = preview_token("secret", "task-1", "dash")
    assert url == (f"https://kortny.example.com/preview/{token}/task-1/dash/index.html")


def test_safe_slug_normalizes_input() -> None:
    assert safe_slug("Sales Dashboard Q2!") == "sales-dashboard-q2"
    assert safe_slug("   ") == "preview"
    assert safe_slug("a" * 200) == "a" * 64


def test_extract_strips_common_root_directory(tmp_path: Path) -> None:
    tar_bytes = _tar_with({"dist/index.html": b"<html></html>", "dist/js/app.js": b"x"})

    extracted = extract_tar_to_dir(tar_bytes, tmp_path)

    assert (tmp_path / "index.html").is_file()
    assert (tmp_path / "js" / "app.js").is_file()
    assert len(extracted) == 2


def test_extract_rejects_traversal_and_absolute_paths(tmp_path: Path) -> None:
    with pytest.raises(UnsafeArchiveError):
        extract_tar_to_dir(_tar_with({"../../evil": b"x"}), tmp_path)


def test_extract_rejects_links(tmp_path: Path) -> None:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        link = tarfile.TarInfo(name="link")
        link.type = tarfile.SYMTYPE
        link.linkname = "/etc/passwd"
        tar.addfile(link)

    with pytest.raises(UnsafeArchiveError):
        extract_tar_to_dir(buffer.getvalue(), tmp_path)


def test_extract_enforces_size_budget(tmp_path: Path) -> None:
    tar_bytes = _tar_with({"big.bin": b"x" * 1024})

    with pytest.raises(UnsafeArchiveError):
        extract_tar_to_dir(tar_bytes, tmp_path, max_bytes=512)
