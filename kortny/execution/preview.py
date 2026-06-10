"""Signed preview URLs and safe tar extraction for sandbox artifacts.

Exported sandbox files land under the shared artifacts directory and are
served by the dashboard at capability URLs: the token is an HMAC over the
task id and slug, so links work from Slack without dashboard login while
remaining unguessable.
"""

from __future__ import annotations

import hashlib
import hmac
import io
import re
import tarfile
from pathlib import Path

PREVIEW_TOKEN_CHARS = 16
SAFE_SLUG_RE = re.compile(r"[^a-z0-9-]+")
MAX_EXTRACT_BYTES = 50 * 1024 * 1024
MAX_EXTRACT_MEMBERS = 2_000


class UnsafeArchiveError(ValueError):
    """Raised when a sandbox archive tries to escape its target directory."""


def preview_token(secret: str, task_id: str, slug: str) -> str:
    """Return the capability token for one preview path."""

    digest = hmac.new(
        secret.encode("utf-8"),
        f"{task_id}/{slug}".encode(),
        hashlib.sha256,
    ).hexdigest()
    return digest[:PREVIEW_TOKEN_CHARS]


def verify_preview_token(secret: str, task_id: str, slug: str, token: str) -> bool:
    """Constant-time check of one preview token."""

    expected = preview_token(secret, task_id, slug)
    return hmac.compare_digest(expected, token)


def preview_url(base_url: str, secret: str, task_id: str, slug: str) -> str:
    """Return the public preview URL for one published directory."""

    token = preview_token(secret, task_id, slug)
    base = base_url.rstrip("/")
    return f"{base}/preview/{token}/{task_id}/{slug}/index.html"


def safe_slug(value: str, *, default: str = "preview") -> str:
    """Normalize a user/model-supplied slug for filesystem and URL use."""

    slug = SAFE_SLUG_RE.sub("-", value.strip().casefold()).strip("-")
    return slug[:64] or default


def extract_tar_to_dir(
    tar_bytes: bytes,
    destination: Path,
    *,
    strip_root: bool = True,
    max_bytes: int = MAX_EXTRACT_BYTES,
) -> list[Path]:
    """Safely extract a sandbox tar archive into a destination directory.

    Rejects absolute paths, parent traversal, links, and oversized content.
    When ``strip_root`` is set and every member shares one top-level
    directory, that directory is stripped so files land directly in
    ``destination``.
    """

    destination.mkdir(parents=True, exist_ok=True)
    resolved_destination = destination.resolve()
    extracted: list[Path] = []
    total_bytes = 0

    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r") as tar:
        members = tar.getmembers()
        if len(members) > MAX_EXTRACT_MEMBERS:
            raise UnsafeArchiveError("Archive has too many entries")
        root_prefix = _common_root(members) if strip_root else None
        for member in members:
            if member.issym() or member.islnk():
                raise UnsafeArchiveError("Archive must not contain links")
            if not (member.isfile() or member.isdir()):
                continue
            name = member.name
            if root_prefix and name.startswith(root_prefix):
                name = name[len(root_prefix) :].lstrip("/")
            if not name:
                continue
            target = (resolved_destination / name).resolve()
            if not target.is_relative_to(resolved_destination):
                raise UnsafeArchiveError("Archive entry escapes destination")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            total_bytes += member.size
            if total_bytes > max_bytes:
                raise UnsafeArchiveError(f"Archive content exceeds {max_bytes} bytes")
            source = tar.extractfile(member)
            if source is None:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read())
            extracted.append(target)
    return extracted


def _common_root(members: list[tarfile.TarInfo]) -> str | None:
    roots = {member.name.split("/", 1)[0] for member in members if member.name}
    if len(roots) != 1:
        return None
    root = next(iter(roots))
    if any(member.name == root and member.isfile() for member in members):
        return None
    return root
