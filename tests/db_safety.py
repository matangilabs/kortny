"""Safety checks for destructive Postgres-backed tests."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import SplitResult, urlsplit


class UnsafeTestDatabaseError(RuntimeError):
    """Raised when tests are pointed at a non-test database."""


PRODUCTION_ENV_VALUES = frozenset({"prod", "production"})


@dataclass(frozen=True, slots=True)
class DatabaseTarget:
    """Comparable database endpoint without credentials."""

    scheme: str
    hostname: str
    port: int | None
    username: str
    database_name: str


def assert_safe_test_database(
    database_url: str,
    *,
    runtime_database_url: str | None = None,
    environment: str | None = None,
) -> None:
    """Refuse destructive tests unless the database target is explicitly safe."""

    normalized_environment = (environment or "").strip().casefold()
    if normalized_environment in PRODUCTION_ENV_VALUES:
        raise UnsafeTestDatabaseError(
            "Refusing to run destructive DB-backed tests while the environment "
            f"is marked {environment!r}."
        )

    parsed = urlsplit(database_url)
    database_name = parsed.path.lstrip("/")
    if not _is_test_database_name(database_name):
        raise UnsafeTestDatabaseError(
            "Refusing to run destructive DB-backed tests against database "
            f"{database_name!r}. KORTNY_TEST_POSTGRES_URL must point to an explicit "
            "test database, for example "
            "postgresql://kortny:kortny@localhost:5432/kortny_test."
        )

    if runtime_database_url and _database_target(parsed) == _database_target(
        urlsplit(runtime_database_url)
    ):
        raise UnsafeTestDatabaseError(
            "Refusing to run destructive DB-backed tests because "
            "KORTNY_TEST_POSTGRES_URL points at the same database target as "
            "POSTGRES_URL."
        )


def _is_test_database_name(database_name: str) -> bool:
    # "_test_" covers per-xdist-worker clones such as "kortny_test_gw0".
    return (
        database_name.startswith("test_")
        or database_name.endswith("_test")
        or "_test_" in database_name
    )


def _database_target(parsed: SplitResult) -> DatabaseTarget:
    return DatabaseTarget(
        scheme=parsed.scheme,
        hostname=parsed.hostname or "",
        port=parsed.port,
        username=parsed.username or "",
        database_name=parsed.path.lstrip("/"),
    )
