from __future__ import annotations

import hashlib
import json
import lzma
import os
import re
import shutil
import stat
import subprocess
import unicodedata
import uuid
import zipfile
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath

from flask import current_app


class InvalidReleaseBatch(ValueError):
    pass


@dataclass(frozen=True)
class ReservedBatch:
    batch_id: str
    model_generation: int
    dataset_generation: int
    reserved_at: datetime
    expires_at: datetime

    @property
    def android_release_id(self) -> str:
        return f"{self.batch_id}-android"

    @property
    def desktop_release_id(self) -> str:
        return f"{self.batch_id}-desktop"


@dataclass(frozen=True)
class DesktopArtifact:
    path: Path
    batch_id: str
    release_id: str
    model_generation: int
    dataset_generation: int
    version_code: int
    version_name: str
    size_bytes: int


@dataclass(frozen=True)
class AndroidArtifact:
    path: Path
    batch_id: str
    release_id: str
    model_generation: int
    dataset_generation: int
    version_code: int
    version_name: str
    size_bytes: int
    signing_fingerprint: str | None = None


@dataclass(frozen=True)
class _ReleaseDeletionCandidate:
    path: Path
    resolved_path: Path
    batch_id: str
    release_id: str
    st_dev: int
    st_ino: int
    st_mtime_ns: int


_MANIFEST_FIELDS = {
    "schema_version",
    "release_batch_id",
    "app_release_id",
    "model_generation",
    "dataset_generation",
    "app_version_code",
    "app_version_name",
}

_WINDOWS_DEVICE_BASENAME = re.compile(
    r"^(?:con|prn|aux|nul|conin\$|conout\$|com[1-9¹²³]|lpt[1-9¹²³])$"
)
_WINDOWS_INVALID_FILENAME_CHARACTER = re.compile(r'[<>"|?*\x00-\x1f]')
_IMMUTABLE_RELEASE_FILENAME = re.compile(
    r"^(?P<batch_id>[0-9a-f]{32})-(?:desktop\.zip|android\.apk)$"
)
_MAX_ORPHAN_SCAN_ENTRIES = 1000
_MAX_ORPHAN_DELETIONS = 1000
_POSIX_RELEASE_PERMISSIONS = os.name == "posix"


def _windows_archive_path_key(filename: str, *, is_directory: bool) -> str:
    parts = filename.split("/")
    if is_directory:
        # ZIP directory entries conventionally have exactly one trailing slash.
        parts = parts[:-1]
    if not parts or any(part in {"", "."} for part in parts):
        raise InvalidReleaseBatch(
            "desktop artifact contains an invalid Windows path segment"
        )

    normalized_parts: list[str] = []
    for part in parts:
        if ":" in part:
            raise InvalidReleaseBatch(
                "desktop artifact contains a Windows alternate data stream path"
            )
        if _WINDOWS_INVALID_FILENAME_CHARACTER.search(part):
            raise InvalidReleaseBatch(
                "desktop artifact contains an invalid Windows member name"
            )
        normalized = unicodedata.normalize("NFC", part).casefold().rstrip(" .")
        if not normalized:
            raise InvalidReleaseBatch(
                "desktop artifact contains an invalid Windows path segment"
            )
        basename = normalized.split(".", 1)[0].rstrip(" .")
        if _WINDOWS_DEVICE_BASENAME.fullmatch(basename):
            raise InvalidReleaseBatch(
                "desktop artifact contains a Windows reserved device name"
            )
        normalized_parts.append(normalized)
    return "/".join(normalized_parts)


def _safe_artifact_infos(
    archive: zipfile.ZipFile,
    *,
    desktop_paths: bool = False,
) -> list[zipfile.ZipInfo]:
    infos = archive.infolist()
    if not infos or len(infos) > 20_000:
        raise InvalidReleaseBatch("artifact member count is invalid")
    total = 0
    raw_names: set[str] = set()
    normalized_names: set[str] = set()
    for info in infos:
        filename = info.filename
        path = PurePosixPath(filename)
        mode = info.external_attr >> 16
        if (
            not filename
            or "\\" in filename
            or path.is_absolute()
            or ".." in path.parts
            or re.match(r"^[A-Za-z]:", filename)
            or stat.S_ISLNK(mode)
            or info.flag_bits & 0x1
        ):
            raise InvalidReleaseBatch("artifact contains an unsafe ZIP member")
        if filename in raw_names:
            raise InvalidReleaseBatch("artifact contains a duplicate member path")
        raw_names.add(filename)
        if desktop_paths:
            normalized = _windows_archive_path_key(
                info.orig_filename,
                is_directory=info.orig_filename.endswith("/"),
            )
            if normalized in normalized_names:
                raise InvalidReleaseBatch(
                    "desktop artifact contains a duplicate normalized path"
                )
            normalized_names.add(normalized)
        total += info.file_size
        if total > current_app.config["MAX_RELEASE_UNCOMPRESSED"]:
            raise InvalidReleaseBatch("artifact uncompressed size is too large")
    return infos


def _stream_artifact_members(
    archive: zipfile.ZipFile,
    infos: list[zipfile.ZipInfo],
    manifest_path: str,
) -> dict:
    manifests = [info for info in infos if info.filename == manifest_path]
    if len(manifests) != 1 or manifests[0].is_dir():
        raise InvalidReleaseBatch(f"artifact must contain exactly one {manifest_path}")
    if manifests[0].file_size > 1024 * 1024:
        raise InvalidReleaseBatch("artifact manifest is too large")

    manifest_info = manifests[0]
    manifest_chunks: list[bytes] = []
    actual_total = 0
    max_uncompressed = current_app.config["MAX_RELEASE_UNCOMPRESSED"]
    try:
        for info in infos:
            if info.is_dir():
                continue
            member_total = 0
            with archive.open(info, "r") as stream:
                while True:
                    chunk = stream.read(1024 * 1024)
                    if not chunk:
                        break
                    member_total += len(chunk)
                    actual_total += len(chunk)
                    if actual_total > max_uncompressed:
                        raise InvalidReleaseBatch(
                            "artifact uncompressed size is too large"
                        )
                    if info is manifest_info:
                        if member_total > 1024 * 1024:
                            raise InvalidReleaseBatch(
                                "artifact manifest is too large"
                            )
                        manifest_chunks.append(chunk)
            if member_total != info.file_size:
                raise InvalidReleaseBatch("artifact contains invalid ZIP data")
        raw = b"".join(manifest_chunks)
        def unique_object(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise InvalidReleaseBatch(f"duplicate JSON key: {key}")
                result[key] = value
            return result

        payload = json.loads(
            raw.decode("utf-8"), object_pairs_hook=unique_object
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InvalidReleaseBatch("artifact manifest is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise InvalidReleaseBatch("artifact manifest must be a JSON object")
    missing = _MANIFEST_FIELDS - payload.keys()
    if missing:
        raise InvalidReleaseBatch(
            "artifact manifest is missing fields: " + ", ".join(sorted(missing))
        )
    return payload


def _validate_manifest(payload: dict, reservation, platform: str) -> tuple:
    expected_batch = reservation["batch_id"]
    expected_release = f"{expected_batch}-{platform}"
    integer_fields = (
        "schema_version",
        "model_generation",
        "dataset_generation",
        "app_version_code",
    )
    if any(
        isinstance(payload[field], bool) or not isinstance(payload[field], int)
        for field in integer_fields
    ):
        raise InvalidReleaseBatch("artifact manifest integer fields are invalid")
    if payload["schema_version"] != 1:
        raise InvalidReleaseBatch("unsupported artifact schema_version")
    if payload["release_batch_id"] != expected_batch:
        raise InvalidReleaseBatch("artifact release batch does not match reservation")
    if payload["app_release_id"] != expected_release:
        raise InvalidReleaseBatch(f"artifact app_release_id must be {expected_release}")
    if payload["model_generation"] != reservation["model_generation"]:
        raise InvalidReleaseBatch("artifact model generation does not match reservation")
    if payload["dataset_generation"] != reservation["dataset_generation"]:
        raise InvalidReleaseBatch("artifact dataset generation does not match reservation")
    if payload["app_version_code"] <= 0:
        raise InvalidReleaseBatch("app_version_code must be a positive integer")
    version_name = payload["app_version_name"]
    if (
        not isinstance(version_name, str)
        or not version_name.strip()
        or len(version_name) > 128
    ):
        raise InvalidReleaseBatch("app_version_name must be a non-empty string")
    return (
        expected_batch,
        expected_release,
        payload["model_generation"],
        payload["dataset_generation"],
        payload["app_version_code"],
        version_name.strip(),
    )


def read_desktop_artifact(source_path: Path, reservation) -> DesktopArtifact:
    path = Path(source_path)
    try:
        size_bytes = path.stat().st_size
        max_bytes = current_app.config["MAX_CONTENT_LENGTH"]
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
            raise RuntimeError("MAX_CONTENT_LENGTH must be a positive integer")
        if size_bytes > max_bytes:
            raise InvalidReleaseBatch("desktop artifact is too large")
        with zipfile.ZipFile(path) as archive:
            infos = _safe_artifact_infos(archive, desktop_paths=True)
            payload = _stream_artifact_members(archive, infos, "release.json")
            files = [info.filename for info in infos if not info.is_dir()]
            basenames = {PurePosixPath(name).name for name in files}
            required = {"reaction_classifier.h5", "yolov8n.pt"}
            missing = required - basenames
            if missing:
                raise InvalidReleaseBatch(
                    "desktop artifact is missing files: "
                    + ", ".join(sorted(missing))
                )
            if not any(name.lower().endswith(".exe") for name in basenames):
                raise InvalidReleaseBatch("desktop artifact is missing an EXE")
            values = _validate_manifest(payload, reservation, "desktop")
    except InvalidReleaseBatch:
        raise
    except (
        zipfile.BadZipFile,
        lzma.LZMAError,
        zlib.error,
        RuntimeError,
        EOFError,
        OSError,
        NotImplementedError,
    ) as exc:
        raise InvalidReleaseBatch(
            "desktop artifact contains invalid ZIP data"
        ) from exc
    return DesktopArtifact(path, *values, size_bytes)


def read_android_artifact(source_path: Path, reservation) -> AndroidArtifact:
    path = Path(source_path)
    try:
        size_bytes = path.stat().st_size
    except OSError as exc:
        raise InvalidReleaseBatch("Android artifact cannot be read") from exc
    max_bytes = current_app.config["MAX_ANDROID_APK_BYTES"]
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise RuntimeError("MAX_ANDROID_APK_BYTES must be a positive integer")
    if size_bytes > max_bytes:
        raise InvalidReleaseBatch("Android artifact is too large")
    try:
        with zipfile.ZipFile(path) as archive:
            infos = _safe_artifact_infos(archive)
            payload = _stream_artifact_members(
                archive, infos, "assets/model-manifest.json"
            )
            names = {info.filename for info in infos if not info.is_dir()}
            required = {
                "assets/detector.tflite",
                "assets/classifier.tflite",
            }
            missing = required - names
            if missing:
                raise InvalidReleaseBatch(
                    "Android artifact is missing files: "
                    + ", ".join(sorted(missing))
                )
            values = _validate_manifest(payload, reservation, "android")
    except InvalidReleaseBatch:
        raise
    except (
        zipfile.BadZipFile,
        lzma.LZMAError,
        zlib.error,
        RuntimeError,
        EOFError,
        OSError,
        NotImplementedError,
    ) as exc:
        raise InvalidReleaseBatch(
            "Android artifact contains invalid ZIP data"
        ) from exc
    return AndroidArtifact(path, *values, size_bytes)


def _parse_fingerprint(value: str, description: str) -> str:
    if re.fullmatch(r"[0-9A-Fa-f]{64}", value):
        return value.lower()
    if re.fullmatch(r"(?:[0-9A-Fa-f]{2}:){31}[0-9A-Fa-f]{2}", value):
        return value.replace(":", "").lower()
    raise InvalidReleaseBatch(f"{description} fingerprint is invalid")


def verify_apk_signature(path: Path) -> str:
    command = [
        current_app.config["APKSIGNER_PATH"],
        "verify",
        "--verbose",
        "--print-certs",
        str(path),
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise InvalidReleaseBatch("APK signature verification timed out") from exc
    except OSError as exc:
        raise InvalidReleaseBatch("APK signature verifier could not be started") from exc
    if completed.returncode != 0:
        raise InvalidReleaseBatch("APK signature verification failed")
    output = f"{completed.stdout}\n{completed.stderr}"
    matches = re.findall(
        r"Signer #\d+ certificate SHA-256 digest:\s*([^\r\n]+)",
        output,
        re.IGNORECASE,
    )
    if len(matches) != 1:
        raise InvalidReleaseBatch("APK must contain exactly one signer certificate")
    actual = _parse_fingerprint(matches[0].strip(), "APK signer")
    configured = current_app.config.get("ANDROID_SIGNING_CERT_SHA256", "")
    if not configured:
        if not current_app.config.get("TESTING", False):
            raise InvalidReleaseBatch("configured APK signing fingerprint is required")
        return actual
    expected = _parse_fingerprint(configured, "configured APK signing")
    if actual != expected:
        raise InvalidReleaseBatch("APK signer certificate does not match configuration")
    return actual


def _stream_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_utc(value: str, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise InvalidReleaseBatch(f"release reservation {field} is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise InvalidReleaseBatch(f"release reservation {field} is not timezone-aware")
    return parsed.astimezone(timezone.utc)


def _load_publishable_batch(db, batch_id: str, now: datetime):
    if not isinstance(batch_id, str) or not batch_id:
        raise InvalidReleaseBatch("release batch id is required")
    row = db.execute(
        "SELECT * FROM release_batches WHERE batch_id = ?", (batch_id,)
    ).fetchone()
    if row is None:
        raise InvalidReleaseBatch("unknown release batch")
    if row["status"] not in ("reserved", "partial", "published"):
        raise InvalidReleaseBatch("release batch is not publishable")
    if row["status"] != "published" and _parse_utc(row["expires_at"], "expiry") <= now:
        raise InvalidReleaseBatch("release reservation has expired")
    return row


def _flush_release_directory(storage: Path) -> bool:
    if os.name == "nt":
        current_app.logger.warning(
            "directory fsync is unsupported on Windows; "
            "release directory durability is best effort"
        )
        return False
    directory_flag = getattr(os, "O_DIRECTORY", None)
    if directory_flag is None:
        raise OSError("directory fsync is unsupported on this platform")
    descriptor = os.open(storage, os.O_RDONLY | directory_flag)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return True


def _positive_integer_config(name: str) -> int:
    configured = current_app.config[name]
    if isinstance(configured, bool):
        raise ValueError(f"{name} must be a positive integer")
    if isinstance(configured, int):
        value = configured
    elif (
        isinstance(configured, str)
        and configured.isascii()
        and configured.isdecimal()
    ):
        value = int(configured)
    else:
        raise ValueError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _orphan_batch_is_eligible(row, current: datetime) -> bool:
    if row["status"] == "expired":
        return True
    return row["status"] in {"reserved", "partial"} and (
        _parse_utc(row["expires_at"], "expiry") <= current
    )


def _release_references(db) -> tuple[set[str], set[Path]]:
    release_ids: set[str] = set()
    stored_paths: set[Path] = set()
    for row in db.execute(
        "SELECT release_id, stored_path FROM platform_releases"
    ):
        release_ids.add(row["release_id"])
        stored_paths.add(Path(row["stored_path"]).resolve(strict=False))
    return release_ids, stored_paths


def _release_deletion_candidate(
    path: Path,
    resolved_storage: Path,
    batch_id: str,
    release_id: str,
    *,
    cutoff_ns: int | None,
) -> _ReleaseDeletionCandidate | None:
    try:
        candidate_stat = path.lstat()
        resolved_path = path.resolve(strict=True)
    except (FileNotFoundError, OSError, RuntimeError):
        return None
    if (
        not stat.S_ISREG(candidate_stat.st_mode)
        or resolved_path.parent != resolved_storage
        or (
            cutoff_ns is not None
            and candidate_stat.st_mtime_ns >= cutoff_ns
        )
    ):
        return None
    return _ReleaseDeletionCandidate(
        path=path,
        resolved_path=resolved_path,
        batch_id=batch_id,
        release_id=release_id,
        st_dev=candidate_stat.st_dev,
        st_ino=candidate_stat.st_ino,
        st_mtime_ns=candidate_stat.st_mtime_ns,
    )


def _coordinated_delete_release_candidates(
    db,
    storage: Path,
    candidates: list[_ReleaseDeletionCandidate],
    *,
    current: datetime,
    cutoff_ns: int | None,
    legacy_only: bool = False,
) -> int:
    if db.in_transaction:
        raise RuntimeError("release deletion requires no active transaction")
    if not candidates:
        return 0

    deleted = 0
    transaction_started = False
    try:
        db.execute("BEGIN IMMEDIATE")
        transaction_started = True
        referenced_release_ids, referenced_paths = _release_references(db)
        for candidate in candidates:
            batch = db.execute(
                "SELECT status, expires_at FROM release_batches "
                "WHERE batch_id = ?",
                (candidate.batch_id,),
            ).fetchone()
            if batch is None:
                continue
            if legacy_only:
                eligible = batch["status"] == "expired"
            else:
                eligible = _orphan_batch_is_eligible(batch, current)
            if (
                not eligible
                or candidate.release_id in referenced_release_ids
                or candidate.resolved_path in referenced_paths
            ):
                continue
            try:
                final_stat = candidate.path.lstat()
                final_resolved = candidate.path.resolve(strict=True)
            except (FileNotFoundError, OSError, RuntimeError):
                continue
            if (
                not stat.S_ISREG(final_stat.st_mode)
                or final_stat.st_dev != candidate.st_dev
                or final_stat.st_ino != candidate.st_ino
                or final_resolved != candidate.resolved_path
                or final_resolved.parent != storage.resolve(strict=True)
                or (
                    cutoff_ns is not None
                    and final_stat.st_mtime_ns >= cutoff_ns
                )
            ):
                continue
            candidate.path.unlink()
            deleted += 1
            if deleted >= _MAX_ORPHAN_DELETIONS:
                break
        if deleted:
            _flush_release_directory(storage)
        db.commit()
        transaction_started = False
    except BaseException:
        if transaction_started or db.in_transaction:
            db.rollback()
        raise
    return deleted


def cleanup_orphan_release_files(db, now: datetime | None = None) -> int:
    if db.in_transaction:
        raise RuntimeError("orphan cleanup requires no active transaction")
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None or current.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    current = current.astimezone(timezone.utc)
    grace_hours = _positive_integer_config("RELEASE_ORPHAN_GRACE_HOURS")
    cutoff_ns = int(
        (current - timedelta(hours=grace_hours)).timestamp() * 1_000_000_000
    )
    storage = Path(current_app.config["STORAGE_ROOT"]) / "releases"
    if not storage.is_dir():
        return 0
    resolved_storage = storage.resolve(strict=True)
    referenced_release_ids, referenced_paths = _release_references(db)
    candidates: list[_ReleaseDeletionCandidate] = []
    with os.scandir(storage) as entries:
        for scanned, entry in enumerate(entries):
            if scanned >= _MAX_ORPHAN_SCAN_ENTRIES:
                break
            match = _IMMUTABLE_RELEASE_FILENAME.fullmatch(entry.name)
            if match is None or entry.is_symlink():
                continue
            batch_id = match.group("batch_id")
            release_id = entry.name.rsplit(".", 1)[0]
            candidate = _release_deletion_candidate(
                Path(entry.path),
                resolved_storage,
                batch_id,
                release_id,
                cutoff_ns=cutoff_ns,
            )
            if candidate is None:
                continue
            batch = db.execute(
                "SELECT status, expires_at FROM release_batches WHERE batch_id = ?",
                (batch_id,),
            ).fetchone()
            if (
                batch is None
                or not _orphan_batch_is_eligible(batch, current)
                or release_id in referenced_release_ids
                or candidate.resolved_path in referenced_paths
            ):
                continue
            candidates.append(candidate)
    return _coordinated_delete_release_candidates(
        db,
        storage,
        candidates,
        current=current,
        cutoff_ns=cutoff_ns,
    )


def cleanup_failed_legacy_release_file(db, batch_id: str) -> bool:
    if db.in_transaction:
        raise RuntimeError("failed legacy cleanup requires no active transaction")
    if re.fullmatch(r"[0-9a-f]{32}", batch_id) is None:
        raise InvalidReleaseBatch("legacy release batch id is unsafe for cleanup")
    storage = Path(current_app.config["STORAGE_ROOT"]) / "releases"
    if not storage.is_dir():
        return False
    resolved_storage = storage.resolve(strict=True)
    release_id = f"{batch_id}-desktop"
    candidate_path = storage / f"{release_id}.zip"
    batch = db.execute(
        "SELECT status FROM release_batches WHERE batch_id = ?", (batch_id,)
    ).fetchone()
    if batch is None or batch["status"] != "expired":
        return False
    referenced_release_ids, referenced_paths = _release_references(db)
    candidate = _release_deletion_candidate(
        candidate_path,
        resolved_storage,
        batch_id,
        release_id,
        cutoff_ns=None,
    )
    if (
        candidate is None
        or release_id in referenced_release_ids
        or candidate.resolved_path in referenced_paths
    ):
        return False
    return bool(
        _coordinated_delete_release_candidates(
            db,
            storage,
            [candidate],
            current=datetime.now(timezone.utc),
            cutoff_ns=None,
            legacy_only=True,
        )
    )


def _run_orphan_cleanup_best_effort(db) -> None:
    try:
        cleanup_orphan_release_files(db)
    except Exception:
        current_app.logger.exception("release orphan cleanup failed")


def _safe_original_filename(value: str | None, fallback: str) -> str:
    candidate = value if isinstance(value, str) else ""
    candidate = candidate.replace("\\", "/").rsplit("/", 1)[-1]
    candidate = "".join(
        character
        for character in candidate
        if ord(character) >= 32 and ord(character) != 127
    ).strip()
    if candidate in {"", ".", ".."}:
        candidate = fallback
    if len(candidate) > 255:
        suffix = PurePosixPath(candidate).suffix
        if 0 < len(suffix) <= 16:
            candidate = candidate[: 255 - len(suffix)] + suffix
        else:
            candidate = candidate[:255]
    return candidate


def _create_immutable_artifact(
    storage: Path,
    platform: str,
    artifact: DesktopArtifact | AndroidArtifact,
    expected_sha256: str,
) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,199}", artifact.release_id):
        raise InvalidReleaseBatch("artifact release id is unsafe for storage")
    suffix = ".zip" if platform == "desktop" else ".apk"
    immutable_path = storage / f"{artifact.release_id}{suffix}"
    staging = storage / f".{artifact.release_id}-{uuid.uuid4().hex}.staging"
    staging_created = False
    try:
        with artifact.path.open("rb") as source, staging.open("xb") as target:
            staging_created = True
            shutil.copyfileobj(source, target, length=1024 * 1024)
            target.flush()
            os.fsync(target.fileno())
        if _stream_sha256(staging) != expected_sha256:
            raise InvalidReleaseBatch("artifact changed while being staged")
        try:
            os.link(staging, immutable_path)
        except FileExistsError:
            if not immutable_path.is_file():
                raise InvalidReleaseBatch("immutable release path is not a file")
            if _stream_sha256(immutable_path) != expected_sha256:
                raise InvalidReleaseBatch(
                    "immutable release path contains a different artifact"
                )
        if _POSIX_RELEASE_PERMISSIONS:
            os.chmod(immutable_path, 0o444)
    finally:
        if staging_created:
            staging.unlink(missing_ok=True)
            _flush_release_directory(storage)
    return immutable_path


def publish_bundle(
    db,
    batch_id: str,
    desktop_path: Path | None = None,
    android_path: Path | None = None,
    *,
    original_filenames: dict[str, str] | None = None,
    before_commit: Callable[[list[dict]], None] | None = None,
) -> dict:
    if db.in_transaction:
        raise RuntimeError(
            "publish_bundle must own transaction; connection is already in a transaction"
        )
    _run_orphan_cleanup_best_effort(db)
    now = datetime.now(timezone.utc)
    reservation = _load_publishable_batch(db, batch_id, now)
    if desktop_path is None and android_path is None:
        raise InvalidReleaseBatch("at least one release artifact is required")
    artifacts: dict[str, DesktopArtifact | AndroidArtifact] = {}
    if desktop_path is not None:
        artifacts["desktop"] = read_desktop_artifact(desktop_path, reservation)
    if android_path is not None:
        android = read_android_artifact(android_path, reservation)
        fingerprint = verify_apk_signature(android.path)
        artifacts["android"] = AndroidArtifact(
            **{**android.__dict__, "signing_fingerprint": fingerprint}
        )

    hashes = {platform: _stream_sha256(item.path) for platform, item in artifacts.items()}
    existing = {
        row["platform"]: row
        for row in db.execute(
            "SELECT * FROM platform_releases WHERE batch_id = ?", (batch_id,)
        ).fetchall()
    }
    for platform in artifacts:
        row = existing.get(platform)
        if row is not None and row["sha256"] != hashes[platform]:
            raise InvalidReleaseBatch(
                f"release batch already has a different artifact for {platform}"
            )

    storage = Path(current_app.config["STORAGE_ROOT"]) / "releases"
    storage.mkdir(parents=True, exist_ok=True)
    stored_paths = {
        platform: _create_immutable_artifact(
            storage, platform, artifact, hashes[platform]
        )
        for platform, artifact in artifacts.items()
    }

    activated: list[dict] = []
    results: list[dict] = []
    transaction_started = False
    try:
        db.execute("BEGIN IMMEDIATE")
        transaction_started = True
        reservation = _load_publishable_batch(
            db, batch_id, datetime.now(timezone.utc)
        )
        state = db.execute("SELECT * FROM app_state WHERE id = 1").fetchone()
        if state is None:
            raise RuntimeError("app_state id=1 is missing")
        if state["dataset_generation"] != reservation["dataset_generation"]:
            raise InvalidReleaseBatch(
                "dataset generation changed; reserve and build the release again"
            )

        for platform, artifact in artifacts.items():
            row = db.execute(
                "SELECT * FROM platform_releases WHERE batch_id = ? AND platform = ?",
                (batch_id, platform),
            ).fetchone()
            if row is not None:
                if row["sha256"] != hashes[platform]:
                    raise InvalidReleaseBatch(
                        f"release batch already has a different artifact for {platform}"
                    )
                if Path(row["stored_path"]) != stored_paths[platform]:
                    db.execute(
                        "UPDATE platform_releases SET stored_path = ? "
                        "WHERE release_id = ?",
                        (str(stored_paths[platform]), row["release_id"]),
                    )
                results.append(
                    {
                        "platform": platform,
                        "release_id": row["release_id"],
                        "sha256": row["sha256"],
                        "idempotent": True,
                    }
                )
                continue
            if reservation["status"] == "published":
                raise InvalidReleaseBatch("published release batch cannot accept new artifacts")

            current = db.execute(
                """
                SELECT pr.release_id, rb.model_generation
                FROM platform_releases AS pr
                JOIN release_batches AS rb ON rb.batch_id = pr.batch_id
                WHERE pr.platform = ? AND pr.is_current = 1
                """,
                (platform,),
            ).fetchone()
            if (
                current is not None
                and reservation["model_generation"] <= current["model_generation"]
            ):
                raise InvalidReleaseBatch(
                    f"{platform} already has a newer generation"
                )
            if (
                platform == "desktop"
                and reservation["model_generation"] <= state["model_generation"]
            ):
                raise InvalidReleaseBatch(
                    "desktop legacy state already has a newer generation"
                )
            db.execute(
                "UPDATE platform_releases SET is_current = 0 WHERE platform = ?",
                (platform,),
            )
            uploaded_at = datetime.now(timezone.utc).isoformat()
            db.execute(
                """
                INSERT INTO platform_releases (
                    release_id, batch_id, platform, version_code, version_name,
                    original_filename, stored_path, sha256, size_bytes,
                    uploaded_at, is_current
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                """,
                (
                    artifact.release_id,
                    batch_id,
                    platform,
                    artifact.version_code,
                    artifact.version_name,
                    _safe_original_filename(
                        (original_filenames or {}).get(platform),
                        artifact.path.name,
                    ),
                    str(stored_paths[platform]),
                    hashes[platform],
                    artifact.size_bytes,
                    uploaded_at,
                ),
            )
            if platform == "desktop":
                db.execute(
                    """
                    UPDATE app_state
                    SET current_release_id = ?, model_generation = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = 1
                    """,
                    (artifact.release_id, artifact.model_generation),
                )
            db.execute(
                "UPDATE installations SET active = 0 WHERE client_platform = ?",
                (platform,),
            )
            item = {
                "platform": platform,
                "release_id": artifact.release_id,
                "sha256": hashes[platform],
                "idempotent": False,
            }
            activated.append(item)
            results.append(item)

        platform_count = db.execute(
            "SELECT COUNT(*) FROM platform_releases WHERE batch_id = ?", (batch_id,)
        ).fetchone()[0]
        status = "published" if platform_count == 2 else "partial"
        db.execute(
            """
            UPDATE release_batches
            SET status = ?, published_at = COALESCE(published_at, ?)
            WHERE batch_id = ?
            """,
            (status, datetime.now(timezone.utc).isoformat(), batch_id),
        )
        if before_commit is not None and activated:
            before_commit(activated)
        db.commit()
        transaction_started = False
    except BaseException:
        if transaction_started or db.in_transaction:
            db.rollback()
        raise

    _run_orphan_cleanup_best_effort(db)
    return {"batch_id": batch_id, "status": status, "platforms": results}


def _next_model_generation(db):
    if not db.in_transaction:
        raise RuntimeError(
            "generation allocation requires an active write transaction"
        )

    state = db.execute("SELECT * FROM app_state WHERE id = 1").fetchone()
    if state is None:
        raise RuntimeError("app_state id=1 is missing")

    release_batch_generation = db.execute(
        "SELECT MAX(model_generation) FROM release_batches"
    ).fetchone()[0]
    desktop_release_generation = db.execute(
        "SELECT MAX(model_generation) FROM desktop_releases"
    ).fetchone()[0]
    generation_floor = max(
        state["model_generation"],
        release_batch_generation or state["model_generation"],
        desktop_release_generation or state["model_generation"],
    )
    return state, generation_floor + 1


def _reservation_parameters(
    now: datetime | None,
) -> tuple[datetime, int]:
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None or current.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    current = current.astimezone(timezone.utc)

    reservation_hours = _positive_integer_config("RELEASE_RESERVATION_HOURS")
    return current, reservation_hours


def _reserve_batch_in_transaction(
    db,
    current: datetime,
    reservation_hours: int,
) -> ReservedBatch:
    state, model_generation = _next_model_generation(db)
    batch_id = uuid.uuid4().hex
    expires_at = current + timedelta(hours=reservation_hours)
    db.execute(
        """
        INSERT INTO release_batches (
            batch_id, model_generation, dataset_generation, status,
            reserved_at, expires_at, published_at
        ) VALUES (?, ?, ?, 'reserved', ?, ?, NULL)
        """,
        (
            batch_id,
            model_generation,
            state["dataset_generation"],
            current.isoformat(),
            expires_at.isoformat(),
        ),
    )
    return ReservedBatch(
        batch_id=batch_id,
        model_generation=model_generation,
        dataset_generation=state["dataset_generation"],
        reserved_at=current,
        expires_at=expires_at,
    )


def _reserve_batch(
    db,
    now: datetime | None,
    before_commit: Callable[[ReservedBatch], None] | None,
) -> ReservedBatch:
    if db.in_transaction:
        raise RuntimeError(
            "reserve_batch must own transaction; "
            "connection is already in a transaction"
        )

    current, reservation_hours = _reservation_parameters(now)

    try:
        db.execute("BEGIN IMMEDIATE")
        reserved = _reserve_batch_in_transaction(
            db,
            current,
            reservation_hours,
        )
        if before_commit is not None:
            before_commit(reserved)
        db.commit()
    except BaseException:
        db.rollback()
        raise

    return reserved


def reserve_batch(db, now: datetime | None = None) -> ReservedBatch:
    return _reserve_batch(db, now, None)


def reserve_batch_with_callback(
    db,
    before_commit: Callable[[ReservedBatch], None],
    now: datetime | None = None,
) -> ReservedBatch:
    return _reserve_batch(db, now, before_commit)
