from __future__ import annotations

import json
import shutil
import stat
import uuid
import zipfile
from pathlib import Path, PurePosixPath

from flask import current_app

from .release_batches import (
    InvalidReleaseBatch,
    _run_orphan_cleanup_best_effort,
    _safe_original_filename,
    cleanup_failed_legacy_release_file,
    publish_bundle,
    reserve_batch,
)


class InvalidRelease(ValueError):
    pass


REQUIRED_MODEL_NAMES = {"reaction_classifier.h5", "yolov8n.pt"}


def _safe_release_infos(
    archive: zipfile.ZipFile,
    max_uncompressed: int,
) -> list[zipfile.ZipInfo]:
    infos = archive.infolist()
    if not infos or len(infos) > 20000:
        raise InvalidRelease("发布包文件数量无效")
    total = 0
    for info in infos:
        path = PurePosixPath(info.filename)
        mode = info.external_attr >> 16
        if path.is_absolute() or ".." in path.parts or stat.S_ISLNK(mode) or info.flag_bits & 0x1:
            raise InvalidRelease("发布包包含不安全路径")
        total += info.file_size
        if total > max_uncompressed:
            raise InvalidRelease("发布包解压后尺寸过大")
    return infos


def _expire_legacy_reservation(db, batch_id: str) -> None:
    try:
        db.execute("BEGIN IMMEDIATE")
        db.execute(
            """
            UPDATE release_batches
            SET status = 'expired'
            WHERE batch_id = ? AND status = 'reserved'
            """,
            (batch_id,),
        )
        db.commit()
    except BaseException:
        db.rollback()
        raise


def _expire_and_cleanup_failed_legacy(db, batch_id: str) -> None:
    _expire_legacy_reservation(db, batch_id)
    cleanup_failed_legacy_release_file(db, batch_id)


def publish_desktop_release(
    source_path: Path,
    db,
    *,
    original_filename: str | None = None,
    before_commit=None,
) -> dict:
    if db.in_transaction:
        raise RuntimeError(
            "publish_desktop_release must own transaction; "
            "connection is already in a transaction"
        )

    temp_dir = Path(current_app.config["STORAGE_ROOT"]) / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = temp_dir / f"legacy-desktop-{uuid.uuid4().hex}.zip"
    reservation = None
    _run_orphan_cleanup_best_effort(db)

    try:
        reservation = reserve_batch(db)
        with zipfile.ZipFile(source_path) as source:
            infos = _safe_release_infos(
                source,
                current_app.config["MAX_RELEASE_UNCOMPRESSED"],
            )
            basenames = {
                PurePosixPath(info.filename).name
                for info in infos
                if not info.is_dir()
            }
            missing = REQUIRED_MODEL_NAMES - basenames
            if missing:
                raise InvalidRelease(f"发布包缺少文件: {', '.join(sorted(missing))}")
            if not any(name.lower().endswith(".exe") for name in basenames):
                raise InvalidRelease("发布包缺少 EXE")

            release_id = reservation.desktop_release_id
            release_payload = {
                "schema_version": 1,
                "release_batch_id": reservation.batch_id,
                "app_release_id": release_id,
                "model_generation": reservation.model_generation,
                "dataset_generation": reservation.dataset_generation,
                "app_version_code": reservation.model_generation,
                "app_version_name": f"legacy-{reservation.model_generation}",
                "api_base_url": current_app.config.get(
                    "PUBLIC_BASE_URL",
                    "https://example.invalid",
                ),
                "bootstrap_token": current_app.config["BOOTSTRAP_TOKEN"],
            }
            with zipfile.ZipFile(
                candidate_path,
                "w",
                compression=zipfile.ZIP_DEFLATED,
                allowZip64=True,
            ) as target:
                for info in infos:
                    if (
                        info.is_dir()
                        or PurePosixPath(info.filename).name == "release.json"
                    ):
                        continue
                    with source.open(info, "r") as source_member:
                        with target.open(
                            info,
                            "w",
                            force_zip64=True,
                        ) as target_member:
                            shutil.copyfileobj(
                                source_member,
                                target_member,
                                length=1024 * 1024,
                            )
                target.writestr(
                    "release.json",
                    json.dumps(release_payload, ensure_ascii=False, indent=2),
                )

        bundle = publish_bundle(
            db,
            reservation.batch_id,
            desktop_path=candidate_path,
            original_filenames={
                "desktop": _safe_original_filename(
                    original_filename,
                    source_path.name,
                )
            },
            before_commit=before_commit,
        )
        platform = bundle["platforms"][0]
        path = Path(
            db.execute(
                "SELECT stored_path FROM platform_releases WHERE release_id = ?",
                (platform["release_id"],),
            ).fetchone()[0]
        )
        return {
            "release_id": platform["release_id"],
            "model_generation": reservation.model_generation,
            "sha256": platform["sha256"],
            "path": path,
        }
    except zipfile.BadZipFile as exc:
        if reservation is not None:
            _expire_and_cleanup_failed_legacy(db, reservation.batch_id)
        raise InvalidRelease("文件不是有效 ZIP") from exc
    except InvalidReleaseBatch as exc:
        if reservation is not None:
            _expire_and_cleanup_failed_legacy(db, reservation.batch_id)
        raise InvalidRelease(str(exc)) from exc
    except BaseException as publication_error:
        if reservation is not None:
            try:
                _expire_and_cleanup_failed_legacy(db, reservation.batch_id)
            except BaseException as expiration_error:
                raise RuntimeError(
                    "legacy release failed and its reservation could not be "
                    "expired and cleaned: "
                    f"{expiration_error}"
                ) from publication_error
        raise
    finally:
        try:
            candidate_path.unlink(missing_ok=True)
        finally:
            _run_orphan_cleanup_best_effort(db)
