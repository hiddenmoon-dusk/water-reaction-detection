import sqlite3
from pathlib import Path

import pytest

from water_server import create_app
from water_server.db import get_db
from water_server.security import verify_password


def _insert_release_batch(db, batch_id, model_generation, status="reserved"):
    db.execute(
        """
        INSERT INTO release_batches (
            batch_id, model_generation, dataset_generation, status,
            reserved_at, expires_at, published_at
        ) VALUES (?, ?, 1, ?, '2026-07-16T00:00:00Z',
                  '2026-07-17T00:00:00Z', NULL)
        """,
        (batch_id, model_generation, status),
    )


def _insert_platform_release(
    db, release_id, batch_id, platform="desktop", is_current=0
):
    db.execute(
        """
        INSERT INTO platform_releases (
            release_id, batch_id, platform, version_code, version_name,
            original_filename, stored_path, sha256, size_bytes, uploaded_at,
            is_current
        ) VALUES (?, ?, ?, 1, '1.0', 'release.zip', 'releases/release.zip',
                  'abc123', 100, '2026-07-16T00:00:00Z', ?)
        """,
        (release_id, batch_id, platform, is_current),
    )


def test_app_initializes_generation_and_admin(db):
    state = db.execute("SELECT * FROM app_state WHERE id = 1").fetchone()

    assert state["dataset_generation"] == 1
    assert state["model_generation"] == 1
    assert state["current_release_id"] == "initial"
    assert state["admin_password_hash"] != "test-admin-password-2026"
    assert verify_password("test-admin-password-2026", state["admin_password_hash"])


def test_database_enables_wal_and_foreign_keys(db):
    assert db.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    assert db.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_fresh_database_has_platform_release_schema(db):
    installation_columns = {
        row["name"] for row in db.execute("PRAGMA table_info(installations)")
    }
    table_names = {
        row["name"]
        for row in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    missing_schema = {
        name
        for name, present in {
            "installations.client_platform": "client_platform"
            in installation_columns,
            "release_batches": "release_batches" in table_names,
            "platform_releases": "platform_releases" in table_names,
        }.items()
        if not present
    }

    assert missing_schema == set()
    assert db.execute("SELECT COUNT(*) FROM release_batches").fetchone()[0] == 0
    assert db.execute("SELECT COUNT(*) FROM platform_releases").fetchone()[0] == 0


def test_fresh_installations_default_to_desktop_and_reject_null(db):
    db.execute(
        """
        INSERT INTO installations (
            installation_id, token_hash, app_release_id, model_generation,
            created_at
        ) VALUES ('fresh-default', 'hash', 'initial', 1,
                  '2026-07-16T00:00:00Z')
        """
    )

    row = db.execute(
        "SELECT client_platform FROM installations WHERE installation_id = ?",
        ("fresh-default",),
    ).fetchone()
    assert row["client_platform"] == "desktop"

    with pytest.raises(sqlite3.IntegrityError):
        db.execute(
            """
            INSERT INTO installations (
                installation_id, token_hash, app_release_id, model_generation,
                client_platform, created_at
            ) VALUES ('fresh-null', 'hash', 'initial', 1, NULL,
                      '2026-07-16T00:00:00Z')
            """
        )


def test_installations_platform_index_has_expected_columns(db):
    indexes = {
        row["name"] for row in db.execute("PRAGMA index_list(installations)")
    }
    columns = [
        row["name"]
        for row in db.execute("PRAGMA index_info(idx_installations_platform)")
    ]

    assert "idx_installations_platform" in indexes
    assert columns == ["client_platform", "active"]


def test_legacy_installations_table_is_migrated(tmp_path):
    database_path = tmp_path / "legacy.db"
    with sqlite3.connect(database_path) as legacy_db:
        legacy_db.execute(
            """
            CREATE TABLE installations (
                installation_id TEXT PRIMARY KEY,
                token_hash TEXT NOT NULL,
                app_release_id TEXT NOT NULL,
                model_generation INTEGER NOT NULL,
                active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                last_seen_at TEXT
            )
            """
        )
        legacy_db.execute(
            """
            INSERT INTO installations (
                installation_id, token_hash, app_release_id, model_generation,
                active, created_at, last_seen_at
            ) VALUES ('legacy-1', 'legacy-hash', 'legacy-release', 7, 0,
                      '2026-07-15T00:00:00Z', '2026-07-15T01:00:00Z')
            """
        )

    config = {
        "TESTING": True,
        "DATABASE": str(database_path),
        "STORAGE_ROOT": str(tmp_path / "storage"),
        "SESSION_COOKIE_SECURE": False,
    }
    app = create_app(config)

    with app.app_context():
        db = get_db()
        legacy_row = db.execute(
            "SELECT * FROM installations WHERE installation_id = 'legacy-1'"
        ).fetchone()
        assert dict(legacy_row) == {
            "installation_id": "legacy-1",
            "token_hash": "legacy-hash",
            "app_release_id": "legacy-release",
            "model_generation": 7,
            "active": 0,
            "created_at": "2026-07-15T00:00:00Z",
            "last_seen_at": "2026-07-15T01:00:00Z",
            "client_platform": "desktop",
        }

        db.execute(
            """
            INSERT INTO installations (
                installation_id, token_hash, app_release_id, model_generation,
                created_at
            ) VALUES ('legacy-default', 'hash', 'initial', 1,
                      '2026-07-16T00:00:00Z')
            """
        )
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(
                """
                INSERT INTO installations (
                    installation_id, token_hash, app_release_id,
                    model_generation, client_platform, created_at
                ) VALUES ('legacy-null', 'hash', 'initial', 1, NULL,
                          '2026-07-16T00:00:00Z')
                """
            )
        db.commit()
        first_snapshot = [
            tuple(row)
            for row in db.execute(
                "SELECT * FROM installations ORDER BY installation_id"
            )
        ]

    second_app = create_app(config)
    with second_app.app_context():
        db = get_db()
        second_snapshot = [
            tuple(row)
            for row in db.execute(
                "SELECT * FROM installations ORDER BY installation_id"
            )
        ]
        default_platform = db.execute(
            """
            SELECT client_platform FROM installations
            WHERE installation_id = 'legacy-default'
            """
        ).fetchone()[0]

    assert second_snapshot == first_snapshot
    assert default_platform == "desktop"


def test_release_batch_rejects_invalid_status(db):
    with pytest.raises(sqlite3.IntegrityError):
        _insert_release_batch(db, "invalid-status", 10, status="staged")


def test_platform_release_rejects_invalid_platform(db):
    _insert_release_batch(db, "invalid-platform-batch", 11)

    with pytest.raises(sqlite3.IntegrityError):
        _insert_platform_release(
            db, "invalid-platform", "invalid-platform-batch", platform="ios"
        )


def test_platform_release_rejects_missing_batch(db):
    assert db.execute("PRAGMA foreign_keys").fetchone()[0] == 1

    with pytest.raises(sqlite3.IntegrityError):
        _insert_platform_release(db, "orphan-release", "missing-batch")


def test_only_one_current_release_is_allowed_per_platform(db):
    _insert_release_batch(db, "current-batch-1", 12)
    _insert_release_batch(db, "current-batch-2", 13)
    _insert_platform_release(
        db, "current-desktop-1", "current-batch-1", is_current=1
    )

    with pytest.raises(sqlite3.IntegrityError):
        _insert_platform_release(
            db, "current-desktop-2", "current-batch-2", is_current=1
        )


def test_platform_release_rejects_non_boolean_current_flag(db):
    _insert_release_batch(db, "invalid-current-batch", 14)

    with pytest.raises(sqlite3.IntegrityError):
        _insert_platform_release(
            db, "invalid-current", "invalid-current-batch", is_current=2
        )


def test_initialization_rolls_back_schema_and_migration_on_app_state_failure(
    tmp_path,
):
    database_path = tmp_path / "rollback.db"
    with sqlite3.connect(database_path) as legacy_db:
        legacy_db.executescript(
            """
            CREATE TABLE installations (
                installation_id TEXT PRIMARY KEY,
                token_hash TEXT NOT NULL,
                app_release_id TEXT NOT NULL,
                model_generation INTEGER NOT NULL,
                active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                last_seen_at TEXT
            );
            INSERT INTO installations (
                installation_id, token_hash, app_release_id, model_generation,
                active, created_at, last_seen_at
            ) VALUES ('legacy-rollback', 'hash', 'release', 3, 1,
                      '2026-07-15T00:00:00Z', NULL);
            CREATE TABLE app_state (
                id INTEGER PRIMARY KEY,
                dataset_generation INTEGER NOT NULL,
                model_generation INTEGER NOT NULL,
                current_release_id TEXT NOT NULL,
                admin_password_hash TEXT NOT NULL CHECK (0),
                updated_at TEXT NOT NULL
            );
            """
        )

    with pytest.raises(sqlite3.IntegrityError):
        create_app(
            {
                "TESTING": True,
                "DATABASE": str(database_path),
                "STORAGE_ROOT": str(tmp_path / "storage"),
                "SESSION_COOKIE_SECURE": False,
            }
        )

    with sqlite3.connect(database_path) as db:
        installation_columns = {
            row[1] for row in db.execute("PRAGMA table_info(installations)")
        }
        installation_indexes = {
            row[1] for row in db.execute("PRAGMA index_list(installations)")
        }
        table_names = {
            row[0]
            for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        legacy_row = db.execute(
            "SELECT * FROM installations WHERE installation_id = 'legacy-rollback'"
        ).fetchone()

    assert installation_columns == {
        "installation_id",
        "token_hash",
        "app_release_id",
        "model_generation",
        "active",
        "created_at",
        "last_seen_at",
    }
    assert "idx_installations_platform" not in installation_indexes
    assert table_names == {"app_state", "installations"}
    assert legacy_row == (
        "legacy-rollback",
        "hash",
        "release",
        3,
        1,
        "2026-07-15T00:00:00Z",
        None,
    )


def test_app_does_not_create_default_instance_when_database_is_configured(
    tmp_path, monkeypatch
):
    from flask import Flask

    default_instance = Path(
        Flask("water_server", instance_relative_config=True).instance_path
    )
    original_mkdir = Path.mkdir

    def guarded_mkdir(path, *args, **kwargs):
        if path == default_instance:
            raise OSError("default instance path is read-only")
        return original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", guarded_mkdir)

    app = create_app(
        {
            "TESTING": True,
            "DATABASE": str(tmp_path / "instance" / "app.db"),
            "STORAGE_ROOT": str(tmp_path / "storage"),
            "SESSION_COOKIE_SECURE": False,
        }
    )

    assert Path(app.config["DATABASE"]).is_file()
