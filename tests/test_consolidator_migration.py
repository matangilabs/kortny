"""HIG-225: migration 0031 roundtrip including the valid_at backfill.

Runs against a scratch database (``<test_db>_mig``) so upgrading/downgrading
never disturbs the shared test database other files use.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from urllib.parse import urlsplit, urlunsplit

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.config import Config

from kortny.db.session import normalize_database_url

TEST_POSTGRES_URL = os.environ.get("KORTNY_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    TEST_POSTGRES_URL is None,
    reason="KORTNY_TEST_POSTGRES_URL is required for migration tests",
)


@pytest.fixture
def scratch_url() -> Iterator[str]:
    assert TEST_POSTGRES_URL is not None
    parsed = urlsplit(normalize_database_url(TEST_POSTGRES_URL))
    base_name = parsed.path.lstrip("/")
    scratch_name = f"{base_name}_mig"
    admin_url = urlunsplit(parsed._replace(path="/postgres"))
    admin_engine = sa.create_engine(
        admin_url, isolation_level="AUTOCOMMIT", poolclass=sa.pool.NullPool
    )
    try:
        with admin_engine.connect() as conn:
            conn.exec_driver_sql(
                f'DROP DATABASE IF EXISTS "{scratch_name}" WITH (FORCE)'
            )
            conn.exec_driver_sql(f'CREATE DATABASE "{scratch_name}"')
        yield urlunsplit(parsed._replace(path=f"/{scratch_name}"))
        with admin_engine.connect() as conn:
            conn.exec_driver_sql(
                f'DROP DATABASE IF EXISTS "{scratch_name}" WITH (FORCE)'
            )
    finally:
        admin_engine.dispose()


def _alembic_config(url: str) -> Config:
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", url)
    return config


def test_migration_0031_backfills_valid_at_and_roundtrips(scratch_url: str) -> None:
    config = _alembic_config(scratch_url)
    command.upgrade(config, "0030")

    engine = sa.create_engine(scratch_url, poolclass=sa.pool.NullPool)
    installation_id = uuid.uuid4()
    entity_id = uuid.uuid4()
    try:
        with engine.begin() as conn:
            conn.execute(
                sa.text(
                    "INSERT INTO installations (id, slack_team_id) VALUES (:id, :team)"
                ),
                {"id": str(installation_id), "team": f"T{uuid.uuid4().hex[:8]}"},
            )
            conn.execute(
                sa.text(
                    "INSERT INTO kg_entities "
                    "(id, installation_id, entity_type, canonical_key, "
                    "visibility_scope_type, source_type, created_at) "
                    "VALUES (:id, :installation_id, 'project', 'legacy_row', "
                    "'workspace', 'task_summary', "
                    "'2026-05-01T12:00:00+00:00')"
                ),
                {"id": str(entity_id), "installation_id": str(installation_id)},
            )

        command.upgrade(config, "0031")

        with engine.connect() as conn:
            row = conn.execute(
                sa.text(
                    "SELECT created_at, valid_at, invalid_at, system_expired_at "
                    "FROM kg_entities WHERE id = :id"
                ),
                {"id": str(entity_id)},
            ).one()
            assert row.valid_at == row.created_at
            assert row.invalid_at is None
            assert row.system_expired_at is None

            # consolidation_runs exists and accepts a row.
            conn.execute(
                sa.text(
                    "INSERT INTO consolidation_runs (installation_id, status) "
                    "VALUES (:installation_id, 'succeeded')"
                ),
                {"installation_id": str(installation_id)},
            )
            # tool_embeddings now accepts the memory kinds.
            conn.execute(
                sa.text(
                    "INSERT INTO tool_embeddings "
                    "(kind, ref_key, model, dim, content_sha256, embedding) "
                    "VALUES ('fact', :ref, 'test-model', 3, :sha, '[1,0,0]')"
                ),
                {"ref": str(uuid.uuid4()), "sha": "a" * 64},
            )
            # user_confirmed is a valid graph source type.
            conn.execute(
                sa.text(
                    "INSERT INTO kg_entities "
                    "(installation_id, entity_type, canonical_key, "
                    "visibility_scope_type, source_type) "
                    "VALUES (:installation_id, 'firm_fact', 'confirmed_row', "
                    "'workspace', 'user_confirmed')"
                ),
                {"installation_id": str(installation_id)},
            )
            conn.commit()

        # Downgrade removes the new schema cleanly.
        command.downgrade(config, "0030")
        with engine.connect() as conn:
            columns = {
                row.column_name
                for row in conn.execute(
                    sa.text(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_name = 'kg_entities'"
                    )
                )
            }
            assert "valid_at" not in columns
            assert "invalid_at" not in columns
            assert "system_expired_at" not in columns
            tables = {
                row.table_name
                for row in conn.execute(
                    sa.text(
                        "SELECT table_name FROM information_schema.tables "
                        "WHERE table_schema = 'public'"
                    )
                )
            }
            assert "consolidation_runs" not in tables
            kinds = set(
                conn.scalars(sa.text("SELECT DISTINCT kind FROM tool_embeddings"))
            )
            assert "fact" not in kinds
            source_types = set(
                conn.scalars(sa.text("SELECT DISTINCT source_type FROM kg_entities"))
            )
            assert "user_confirmed" not in source_types

        # And the upgrade applies again (roundtrip).
        command.upgrade(config, "0031")
    finally:
        engine.dispose()
