from __future__ import annotations

import shutil
import tempfile
import zipfile
from pathlib import Path

from .validation import WATER_TYPES


def build_results_export(db, storage_root: Path) -> Path:
    temp_root = Path(storage_root) / "temp"
    temp_root.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix="results-",
        suffix=".zip",
        dir=temp_root,
        delete=False,
    )
    path = Path(handle.name)
    handle.close()

    rows = db.execute(
        "SELECT upload_id, water_type, storage_path FROM uploads ORDER BY water_type, received_at"
    ).fetchall()
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for water_type in WATER_TYPES:
            archive.writestr(f"{water_type}/", b"")
        for row in rows:
            sample_dir = Path(row["storage_path"])
            for name in ("original.jpg", "annotated.png", "result.json"):
                source = sample_dir / name
                if source.is_file():
                    archive.write(
                        source,
                        arcname=f"{row['water_type']}/{row['upload_id']}/{name}",
                    )
    return path


def clear_results(db, storage_root: Path) -> int:
    results_dir = Path(storage_root) / "results"
    trash_dir = Path(storage_root) / "temp" / "results-clear-trash"
    if trash_dir.exists():
        shutil.rmtree(trash_dir)
    if results_dir.exists():
        results_dir.replace(trash_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    for water_type in WATER_TYPES:
        (results_dir / water_type).mkdir()

    try:
        db.execute("BEGIN")
        db.execute("DELETE FROM uploads")
        db.execute(
            """
            UPDATE app_state
            SET dataset_generation = dataset_generation + 1,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = 1
            """
        )
        generation = db.execute(
            "SELECT dataset_generation FROM app_state WHERE id = 1"
        ).fetchone()[0]
        db.commit()
    except Exception:
        db.rollback()
        shutil.rmtree(results_dir, ignore_errors=True)
        if trash_dir.exists():
            trash_dir.replace(results_dir)
        raise
    shutil.rmtree(trash_dir, ignore_errors=True)
    return int(generation)
