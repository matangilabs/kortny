"""Global pytest safeguards and xdist per-worker database management.

Serial runs (no ``-n``) behave as before: every DB-backed test file migrates
and uses the single ``KORTNY_TEST_POSTGRES_URL`` database.

Under pytest-xdist the controller migrates that database once (it becomes the
template) and clones one copy per worker (``<name>_gw0``, ``<name>_gw1``, ...)
with ``CREATE DATABASE ... TEMPLATE`` — milliseconds per clone. Each worker
process then rewrites ``KORTNY_TEST_POSTGRES_URL`` to its private clone before
test modules import, so the DELETE-based cleanup fixtures never contend.
"""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.config import Config as AlembicConfig

from kortny.db.session import normalize_database_url
from tests.db_safety import UnsafeTestDatabaseError, assert_safe_test_database


def pytest_configure(config: pytest.Config) -> None:
    """Prevent any DB-backed test from wiping the local development database."""

    database_url = os.environ.get("KORTNY_TEST_POSTGRES_URL")
    if not database_url:
        return

    workerinput: dict[str, Any] | None = getattr(config, "workerinput", None)
    if workerinput is not None:
        # xdist worker: point this process at its private database clone.
        # This runs before test modules import, so module-level
        # KORTNY_TEST_POSTGRES_URL reads pick up the clone URL.
        database_url = _worker_database_url(database_url, workerinput["workerid"])
        os.environ["KORTNY_TEST_POSTGRES_URL"] = database_url

    try:
        assert_safe_test_database(
            database_url,
            runtime_database_url=os.environ.get("POSTGRES_URL"),
            environment=_environment_marker(),
        )
    except UnsafeTestDatabaseError as exc:
        pytest.exit(str(exc), returncode=2)


def pytest_xdist_setupnodes(config: pytest.Config, specs: list[Any]) -> None:
    """Controller-only xdist hook: prepare one database clone per worker."""

    del config
    base_url = os.environ.get("KORTNY_TEST_POSTGRES_URL")
    if not base_url:
        return
    try:
        assert_safe_test_database(
            base_url,
            runtime_database_url=os.environ.get("POSTGRES_URL"),
            environment=_environment_marker(),
        )
    except UnsafeTestDatabaseError as exc:
        pytest.exit(str(exc), returncode=2)

    _migrate_template(base_url)
    _clone_worker_databases(base_url, [spec.id for spec in specs])


def _migrate_template(base_url: str) -> None:
    alembic_config = AlembicConfig("alembic.ini")
    alembic_config.set_main_option("sqlalchemy.url", normalize_database_url(base_url))
    command.upgrade(alembic_config, "head")


def _clone_worker_databases(base_url: str, worker_ids: list[str]) -> None:
    parsed = urlsplit(normalize_database_url(base_url))
    base_name = parsed.path.lstrip("/")
    admin_url = urlunsplit(parsed._replace(path="/postgres"))
    engine = sa.create_engine(
        admin_url, isolation_level="AUTOCOMMIT", poolclass=sa.pool.NullPool
    )
    try:
        with engine.connect() as conn:
            # CREATE DATABASE ... TEMPLATE fails while the template has open
            # connections; drop any stragglers (e.g. from the migration run).
            conn.execute(
                sa.text(
                    "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                    "WHERE datname = :db AND pid <> pg_backend_pid()"
                ),
                {"db": base_name},
            )
            for worker_id in worker_ids:
                clone_name = f"{base_name}_{worker_id}"
                conn.exec_driver_sql(
                    f'DROP DATABASE IF EXISTS "{clone_name}" WITH (FORCE)'
                )
                conn.exec_driver_sql(
                    f'CREATE DATABASE "{clone_name}" TEMPLATE "{base_name}"'
                )
    finally:
        engine.dispose()


def _worker_database_url(base_url: str, worker_id: str) -> str:
    parsed = urlsplit(base_url)
    return urlunsplit(parsed._replace(path=f"{parsed.path}_{worker_id}"))


def _environment_marker() -> str | None:
    return (
        os.environ.get("KORTNY_ENV")
        or os.environ.get("APP_ENV")
        or os.environ.get("ENVIRONMENT")
    )
