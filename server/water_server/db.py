from __future__ import annotations

import sqlite3
from pathlib import Path

from flask import current_app, g

from .security import hash_password


def get_db() -> sqlite3.Connection:
    if "db" not in g:
        connection = sqlite3.connect(
            current_app.config["DATABASE"],
            detect_types=sqlite3.PARSE_DECLTYPES,
            timeout=30,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        g.db = connection
    return g.db


def close_db(_error=None) -> None:
    connection = g.pop("db", None)
    if connection is not None:
        connection.close()


def _column_names(db: sqlite3.Connection, table: str) -> set[str]:
    return {row["name"] for row in db.execute(f"PRAGMA table_info({table})")}


def migrate_database(db: sqlite3.Connection) -> None:
    if "client_platform" not in _column_names(db, "installations"):
        db.execute(
            """
            ALTER TABLE installations
            ADD COLUMN client_platform TEXT NOT NULL DEFAULT 'desktop'
            """
        )
    db.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_installations_platform
        ON installations(client_platform, active)
        """
    )


def _execute_schema(db: sqlite3.Connection, schema: str) -> None:
    pending: list[str] = []
    for character in schema:
        pending.append(character)
        if character == ";" and sqlite3.complete_statement("".join(pending)):
            db.execute("".join(pending).strip())
            pending.clear()

    if "".join(pending).strip():
        raise sqlite3.OperationalError("incomplete schema statement")


def initialize_database(app) -> None:
    schema_path = Path(__file__).with_name("schema.sql")
    with app.app_context():
        db = get_db()
        try:
            db.execute("BEGIN IMMEDIATE")
            _execute_schema(db, schema_path.read_text(encoding="utf-8"))
            migrate_database(db)
            state = db.execute("SELECT id FROM app_state WHERE id = 1").fetchone()
            if state is None:
                db.execute(
                    """
                    INSERT INTO app_state (
                        id, dataset_generation, model_generation,
                        current_release_id, admin_password_hash, updated_at
                    ) VALUES (1, 1, 1, 'initial', ?, CURRENT_TIMESTAMP)
                    """,
                    (hash_password(app.config["ADMIN_INITIAL_PASSWORD"]),),
                )
            db.commit()
        except BaseException:
            db.rollback()
            raise
