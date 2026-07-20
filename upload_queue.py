from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class UploadTask:
    upload_id: str
    archive_path: Path
    dataset_generation: int
    app_release_id: str
    model_generation: int
    status: str = "pending"
    attempts: int = 0
    next_attempt_at: float = 0
    last_error: str | None = None


class UploadQueue:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            self.path,
            check_same_thread=False,
            timeout=15,
        )
        self._connection.row_factory = sqlite3.Row
        with self._lock:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=NORMAL")
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS upload_tasks (
                    upload_id TEXT PRIMARY KEY,
                    archive_path TEXT NOT NULL,
                    dataset_generation INTEGER NOT NULL,
                    app_release_id TEXT NOT NULL,
                    model_generation INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    next_attempt_at REAL NOT NULL DEFAULT 0,
                    last_error TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            self._connection.execute(
                """
                UPDATE upload_tasks
                SET status = 'pending', updated_at = ?
                WHERE status = 'uploading'
                """,
                (time.time(),),
            )
            self._connection.commit()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def enqueue(self, task: UploadTask) -> None:
        now = time.time()
        with self._lock:
            self._connection.execute(
                """
                INSERT OR IGNORE INTO upload_tasks (
                    upload_id, archive_path, dataset_generation,
                    app_release_id, model_generation, status, attempts,
                    next_attempt_at, last_error, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 'pending', 0, 0, NULL, ?, ?)
                """,
                (
                    task.upload_id,
                    str(task.archive_path),
                    task.dataset_generation,
                    task.app_release_id,
                    task.model_generation,
                    now,
                    now,
                ),
            )
            self._connection.commit()

    def _task_from_row(self, row: sqlite3.Row | None) -> UploadTask | None:
        if row is None:
            return None
        return UploadTask(
            upload_id=row["upload_id"],
            archive_path=Path(row["archive_path"]),
            dataset_generation=row["dataset_generation"],
            app_release_id=row["app_release_id"],
            model_generation=row["model_generation"],
            status=row["status"],
            attempts=row["attempts"],
            next_attempt_at=row["next_attempt_at"],
            last_error=row["last_error"],
        )

    def get(self, upload_id: str) -> UploadTask | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM upload_tasks WHERE upload_id = ?",
                (upload_id,),
            ).fetchone()
        return self._task_from_row(row)

    def claim_next(self, now: float | None = None) -> UploadTask | None:
        current = time.time() if now is None else now
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            row = self._connection.execute(
                """
                SELECT * FROM upload_tasks
                WHERE status = 'pending'
                   OR (status = 'retry_wait' AND next_attempt_at <= ?)
                ORDER BY created_at, upload_id
                LIMIT 1
                """,
                (current,),
            ).fetchone()
            if row is None:
                self._connection.commit()
                return None
            self._connection.execute(
                """
                UPDATE upload_tasks
                SET status = 'uploading',
                    attempts = attempts + 1,
                    updated_at = ?,
                    last_error = NULL
                WHERE upload_id = ?
                """,
                (current, row["upload_id"]),
            )
            self._connection.commit()
        return self.get(row["upload_id"])

    def schedule_retry(
        self,
        upload_id: str,
        delay: float,
        error: str,
        now: float | None = None,
    ) -> None:
        current = time.time() if now is None else now
        with self._lock:
            self._connection.execute(
                """
                UPDATE upload_tasks
                SET status = 'retry_wait',
                    next_attempt_at = ?,
                    last_error = ?,
                    updated_at = ?
                WHERE upload_id = ?
                """,
                (current + delay, error[:500], current, upload_id),
            )
            self._connection.commit()

    def mark_uploaded(self, upload_id: str) -> None:
        task = self.get(upload_id)
        now = time.time()
        with self._lock:
            self._connection.execute(
                """
                UPDATE upload_tasks
                SET status = 'uploaded',
                    next_attempt_at = 0,
                    last_error = NULL,
                    updated_at = ?
                WHERE upload_id = ?
                """,
                (now, upload_id),
            )
            self._connection.commit()
        if task is not None:
            task.archive_path.unlink(missing_ok=True)

    def mark_rejected(self, upload_id: str, error: str) -> None:
        now = time.time()
        with self._lock:
            self._connection.execute(
                """
                UPDATE upload_tasks
                SET status = 'rejected',
                    next_attempt_at = 0,
                    last_error = ?,
                    updated_at = ?
                WHERE upload_id = ?
                """,
                (error[:500], now, upload_id),
            )
            self._connection.commit()

    def pending_count(self) -> int:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM upload_tasks
                WHERE status IN ('pending', 'uploading', 'retry_wait')
                """
            ).fetchone()
        return int(row["count"])

    def total_count(self) -> int:
        with self._lock:
            row = self._connection.execute(
                "SELECT COUNT(*) AS count FROM upload_tasks"
            ).fetchone()
        return int(row["count"])
