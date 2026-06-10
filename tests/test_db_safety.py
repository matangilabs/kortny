import pytest

from tests.db_safety import UnsafeTestDatabaseError, assert_safe_test_database


def test_db_safety_allows_explicit_test_database_names() -> None:
    assert_safe_test_database("postgresql://kortny:kortny@localhost:5432/kortny_test")
    assert_safe_test_database("postgresql://kortny:kortny@localhost:5432/test_kortny")
    # Per-xdist-worker clone names
    assert_safe_test_database(
        "postgresql://kortny:kortny@localhost:5432/kortny_test_gw0"
    )


def test_db_safety_rejects_default_development_database() -> None:
    with pytest.raises(UnsafeTestDatabaseError):
        assert_safe_test_database("postgresql://kortny:kortny@localhost:5432/kortny")


def test_db_safety_rejects_production_environment_marker() -> None:
    with pytest.raises(UnsafeTestDatabaseError):
        assert_safe_test_database(
            "postgresql://kortny:kortny@localhost:5432/kortny_test",
            environment="production",
        )


def test_db_safety_rejects_runtime_database_target() -> None:
    with pytest.raises(UnsafeTestDatabaseError):
        assert_safe_test_database(
            "postgresql://kortny:secret@localhost:5432/kortny_test",
            runtime_database_url="postgresql://kortny:other@localhost:5432/kortny_test",
        )
