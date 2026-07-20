from __future__ import annotations

import hashlib
import os
import re
import stat
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from flask import Blueprint, current_app, jsonify, render_template, send_file

from .db import get_db
from .validation import WATER_TYPES


bp = Blueprint("public", __name__)

_APK_MIMETYPE = "application/vnd.android.package-archive"
_ANDROID_RELEASE_ID = re.compile(r"[0-9a-f]{32}-android")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_APK_DOWNLOAD_NAME = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._ +()-]{0,246}\.apk", re.IGNORECASE
)
_NO_CACHE = "no-store, no-cache, must-revalidate"
_IMMUTABLE_CACHE = "public, max-age=31536000, immutable"
_HASH_CHUNK_SIZE = 1024 * 1024
_MAX_INTEGRITY_CACHE_ENTRIES = 64
_INTEGRITY_STATE_INIT_LOCK = threading.Lock()


@dataclass
class _IntegrityFlight:
    event: threading.Event = field(default_factory=threading.Event)
    result: bool | None = None


@dataclass
class _MobileIntegrityState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    verified: OrderedDict = field(default_factory=OrderedDict)
    in_flight: dict = field(default_factory=dict)


def statistics_payload() -> dict:
    db = get_db()
    state = db.execute("SELECT * FROM app_state WHERE id = 1").fetchone()
    rows = db.execute(
        """
        SELECT water_type,
               COALESCE(SUM(positive_count), 0) AS positive_count,
               COALESCE(SUM(negative_count), 0) AS negative_count
        FROM uploads
        WHERE dataset_generation = ?
        GROUP BY water_type
        """,
        (state["dataset_generation"],),
    ).fetchall()
    by_type = {row["water_type"]: row for row in rows}
    result = []
    for water_type in WATER_TYPES:
        row = by_type.get(water_type)
        positive = int(row["positive_count"]) if row else 0
        negative = int(row["negative_count"]) if row else 0
        total = positive + negative
        result.append(
            {
                "water_type": water_type,
                "positive_count": positive,
                "negative_count": negative,
                "total_count": total,
                "positive_ratio": positive / total if total else None,
                "negative_ratio": negative / total if total else None,
            }
        )
    return {
        "water_types": result,
        "updated_at": state["updated_at"],
        "dataset_generation": state["dataset_generation"],
    }


@bp.get("/")
def index():
    return render_template("index.html", statistics=statistics_payload())


@bp.get("/api/v1/public/statistics")
def public_statistics():
    return jsonify(statistics_payload())


@bp.get("/downloads/desktop")
def desktop_download():
    db = get_db()
    row = db.execute(
        "SELECT stored_path FROM platform_releases "
        "WHERE platform = 'desktop' AND is_current = 1"
    ).fetchone()
    legacy_fallback = row is None
    if legacy_fallback:
        row = db.execute(
            "SELECT stored_path FROM desktop_releases WHERE is_current = 1"
        ).fetchone()
    if row is None:
        return render_template(
            "message.html",
            title="电脑端程序尚未发布",
            message="管理员尚未上传可下载的电脑端检测程序。",
        ), 404
    path = Path(row["stored_path"])
    if not path.is_file():
        return render_template(
            "message.html",
            title="下载文件暂不可用",
            message="服务器中的当前发布文件缺失，请联系管理员。",
        ), 503
    if legacy_fallback and current_app.config["USE_X_ACCEL_REDIRECT"]:
        response = current_app.response_class()
        response.headers["X-Accel-Redirect"] = (
            "/_desktop_release/desktop-latest.zip"
        )
        response.headers["Content-Type"] = "application/zip"
        response.headers["Content-Disposition"] = (
            'attachment; filename="water-detection-desktop.zip"'
        )
        return response
    return send_file(
        path,
        as_attachment=True,
        download_name="水体反应管检测系统.zip",
        mimetype="application/zip",
    )


@bp.get("/downloads/mobile")
def mobile_download():
    row = _current_mobile_release()
    if row is None:
        return _mobile_release_not_found()
    return _serve_mobile_release(row, _NO_CACHE)


@bp.get("/downloads/mobile/<release_id>.apk")
def versioned_mobile_download(release_id):
    if _ANDROID_RELEASE_ID.fullmatch(release_id) is None:
        return _mobile_release_not_found()
    row = _mobile_release_by_id(release_id)
    if row is None:
        return _mobile_release_not_found()
    return _serve_mobile_release(row, _IMMUTABLE_CACHE)


@bp.get("/api/v1/mobile/releases/current")
def current_mobile_release():
    row = _current_mobile_release()
    if row is None:
        return _mobile_release_not_found()
    verified = _open_verified_mobile_file(row)
    if verified is None:
        return _mobile_release_unavailable()
    verified_file, _verified_stat, _resolved_path = verified
    verified_file.close()
    try:
        download_url = _mobile_download_url(row["release_id"])
    except (TypeError, ValueError):
        return _invalid_public_base_url()
    response = jsonify(
        release_id=row["release_id"],
        model_generation=row["model_generation"],
        dataset_generation=row["dataset_generation"],
        version_code=row["version_code"],
        version_name=row["version_name"],
        size_bytes=row["size_bytes"],
        sha256=row["sha256"],
        download_url=download_url,
        mandatory=False,
        release_notes="",
    )
    response.headers["Cache-Control"] = _NO_CACHE
    return response


def _current_mobile_release():
    return get_db().execute(
        """
        SELECT p.release_id, p.version_code, p.version_name,
               p.original_filename, p.stored_path, p.sha256, p.size_bytes,
               b.model_generation, b.dataset_generation
        FROM platform_releases AS p
        JOIN release_batches AS b ON b.batch_id = p.batch_id
        WHERE p.platform = 'android' AND p.is_current = 1
        """
    ).fetchone()


def _mobile_release_by_id(release_id):
    return get_db().execute(
        """
        SELECT p.release_id, p.version_code, p.version_name,
               p.original_filename, p.stored_path, p.sha256, p.size_bytes,
               b.model_generation, b.dataset_generation
        FROM platform_releases AS p
        JOIN release_batches AS b ON b.batch_id = p.batch_id
        WHERE p.platform = 'android' AND p.release_id = ?
        """,
        (release_id,),
    ).fetchone()


def _validated_mobile_release_path(row):
    release_id = row["release_id"]
    sha256 = row["sha256"]
    size_bytes = row["size_bytes"]
    if (
        not isinstance(release_id, str)
        or _ANDROID_RELEASE_ID.fullmatch(release_id) is None
        or not isinstance(sha256, str)
        or _SHA256.fullmatch(sha256) is None
        or not isinstance(size_bytes, int)
        or isinstance(size_bytes, bool)
        or size_bytes < 0
    ):
        return None
    try:
        releases = (
            Path(current_app.config["STORAGE_ROOT"]) / "releases"
        ).resolve(strict=True)
        stored_path = Path(row["stored_path"])
        stored_stat = stored_path.lstat()
        resolved_path = stored_path.resolve(strict=True)
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError):
        return None
    if (
        not stat.S_ISREG(stored_stat.st_mode)
        or resolved_path.parent != releases
        or resolved_path.name != f"{release_id}.apk"
        or stored_stat.st_size != size_bytes
    ):
        return None
    return resolved_path, stored_stat


def _open_verified_mobile_file(row):
    validated = _validated_mobile_release_path(row)
    if validated is None:
        return None
    resolved_path, path_stat = validated
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    file_descriptor = None
    verified_file = None
    ownership_transferred = False
    try:
        file_descriptor = os.open(resolved_path, flags)
        opened_stat = os.fstat(file_descriptor)
        if (
            not stat.S_ISREG(opened_stat.st_mode)
            or opened_stat.st_dev != path_stat.st_dev
            or opened_stat.st_ino != path_stat.st_ino
            or opened_stat.st_size != path_stat.st_size
            or opened_stat.st_size != row["size_bytes"]
        ):
            return None
        verified_file = os.fdopen(file_descriptor, "rb")
        file_descriptor = None
        cache_key = _mobile_integrity_cache_key(
            resolved_path, opened_stat, row["sha256"]
        )
        if not _verify_open_mobile_file(
            verified_file, cache_key, row["sha256"]
        ):
            return None
        after_hash_stat = os.fstat(verified_file.fileno())
        if (
            after_hash_stat.st_dev != opened_stat.st_dev
            or after_hash_stat.st_ino != opened_stat.st_ino
            or after_hash_stat.st_size != opened_stat.st_size
            or after_hash_stat.st_mtime_ns != opened_stat.st_mtime_ns
            or after_hash_stat.st_ctime_ns != opened_stat.st_ctime_ns
        ):
            return None
        verified_file.seek(0)
        ownership_transferred = True
        return verified_file, opened_stat, resolved_path
    except (OSError, TypeError, ValueError):
        return None
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        if verified_file is not None and not ownership_transferred:
            verified_file.close()


def _is_posix_platform():
    return os.name == "posix"


def _mobile_integrity_cache_key(path, opened_stat, expected_sha):
    return (
        str(path),
        opened_stat.st_dev,
        opened_stat.st_ino,
        opened_stat.st_size,
        opened_stat.st_mtime_ns,
        opened_stat.st_ctime_ns,
        expected_sha,
    )


def _hash_open_mobile_file(opened_file):
    digest = hashlib.sha256()
    while chunk := opened_file.read(_HASH_CHUNK_SIZE):
        digest.update(chunk)
    return digest.hexdigest()


def _mobile_integrity_state():
    state = current_app.extensions.get("mobile_release_integrity")
    if state is not None:
        return state
    with _INTEGRITY_STATE_INIT_LOCK:
        return current_app.extensions.setdefault(
            "mobile_release_integrity", _MobileIntegrityState()
        )


def _verify_open_mobile_file(opened_file, cache_key, expected_sha):
    if not _is_posix_platform():
        try:
            return _hash_open_mobile_file(opened_file) == expected_sha
        except Exception:
            return False
    state = _mobile_integrity_state()
    leader = False
    with state.lock:
        if cache_key in state.verified:
            state.verified.move_to_end(cache_key)
            return True
        flight = state.in_flight.get(cache_key)
        if flight is None:
            flight = _IntegrityFlight()
            state.in_flight[cache_key] = flight
            leader = True
    if not leader:
        flight.event.wait()
        return flight.result is True
    result = False
    try:
        try:
            result = _hash_open_mobile_file(opened_file) == expected_sha
        except Exception:
            result = False
    finally:
        with state.lock:
            try:
                if result:
                    state.verified[cache_key] = None
                    state.verified.move_to_end(cache_key)
                    while len(state.verified) > _MAX_INTEGRITY_CACHE_ENTRIES:
                        state.verified.popitem(last=False)
                flight.result = result
            finally:
                state.in_flight.pop(cache_key, None)
                flight.event.set()
    return result


def _serve_mobile_release(row, cache_control):
    verified = _open_verified_mobile_file(row)
    if verified is None:
        return _mobile_release_unavailable()
    verified_file, verified_stat, resolved_path = verified
    download_name = _safe_apk_download_name(
        row["original_filename"], row["release_id"]
    )
    if _should_use_x_accel(resolved_path, verified_stat):
        verified_file.close()
        response = current_app.response_class()
        response.headers["X-Accel-Redirect"] = (
            f"/_mobile_release/{row['release_id']}.apk"
        )
        response.headers["Content-Type"] = _APK_MIMETYPE
        response.headers.set(
            "Content-Disposition", "attachment", filename=download_name
        )
    else:
        try:
            response = send_file(
                verified_file,
                as_attachment=True,
                download_name=download_name,
                mimetype=_APK_MIMETYPE,
            )
        except BaseException:
            verified_file.close()
            raise
        response.call_on_close(verified_file.close)
    response.headers["Cache-Control"] = cache_control
    return response


def _should_use_x_accel(resolved_path, file_stat):
    return (
        current_app.config["USE_X_ACCEL_REDIRECT"]
        and current_app.config.get("TRUSTED_IMMUTABLE_RELEASES", False)
        and _is_posix_platform()
        and _release_can_be_offloaded(resolved_path, file_stat)
    )


def _release_can_be_offloaded(resolved_path, file_stat):
    configured_releases = (
        Path(current_app.config["STORAGE_ROOT"]) / "releases"
    )
    directory_chain = _immutable_directory_stat_chain(
        configured_releases, resolved_path.parent
    )
    if directory_chain is None:
        return False
    try:
        effective_uid = os.geteuid()
        effective_groups = set(os.getgroups())
        effective_groups.add(os.getegid())
    except (AttributeError, OSError):
        return False
    return _trusted_x_accel_ancestry_permissions(
        directory_chain,
        file_stat,
        effective_uid,
        effective_groups,
    )


def _path_lstat(path):
    return path.lstat()


def _path_resolve(path):
    return path.resolve(strict=True)


def _immutable_directory_stat_chain(
    configured_releases, expected_resolved_directory
):
    try:
        current = Path(configured_releases).absolute()
        expected = Path(expected_resolved_directory)
        if _path_resolve(current) != expected:
            return None
        chain = []
        while True:
            before = _path_lstat(current)
            if (
                not stat.S_ISDIR(before.st_mode)
                or stat.S_ISLNK(before.st_mode)
                or _path_resolve(current) != current
            ):
                return None
            after = _path_lstat(current)
            if (
                before.st_dev != after.st_dev
                or before.st_ino != after.st_ino
                or not stat.S_ISDIR(after.st_mode)
                or stat.S_ISLNK(after.st_mode)
            ):
                return None
            chain.append(after)
            parent = current.parent
            if parent == current:
                return chain
            current = parent
    except (OSError, RuntimeError, TypeError, ValueError):
        return None


def _trusted_x_accel_permissions(
    directory_stat, file_stat, effective_uid, effective_groups
):
    return _trusted_x_accel_ancestry_permissions(
        [directory_stat], file_stat, effective_uid, effective_groups
    )


def _trusted_x_accel_ancestry_permissions(
    directory_chain, file_stat, effective_uid, effective_groups
):
    if (
        effective_uid == 0
        or not directory_chain
        or not stat.S_ISREG(file_stat.st_mode)
        or file_stat.st_uid == effective_uid
        or _stat_is_writable_by_process(
            file_stat, effective_uid, effective_groups
        )
    ):
        return False
    for directory_stat in directory_chain:
        if (
            not stat.S_ISDIR(directory_stat.st_mode)
            or stat.S_ISLNK(directory_stat.st_mode)
            or directory_stat.st_uid == effective_uid
            or _stat_is_writable_by_process(
                directory_stat, effective_uid, effective_groups
            )
        ):
            return False
    return True


def _stat_is_writable_by_process(file_stat, effective_uid, effective_groups):
    if effective_uid == 0:
        return True
    if file_stat.st_uid == effective_uid:
        return bool(file_stat.st_mode & stat.S_IWUSR)
    if file_stat.st_gid in effective_groups:
        return bool(file_stat.st_mode & stat.S_IWGRP)
    return bool(file_stat.st_mode & stat.S_IWOTH)


def _mobile_download_url(release_id):
    download_path = f"/downloads/mobile/{release_id}.apk"
    base_url = current_app.config.get("PUBLIC_BASE_URL")
    if not base_url:
        return download_path
    if (
        not isinstance(base_url, str)
        or any(character.isspace() for character in base_url)
        or "\\" in base_url
        or "?" in base_url
        or "#" in base_url
    ):
        raise ValueError("invalid public base URL")
    parsed = urlsplit(base_url)
    try:
        hostname = parsed.hostname
        parsed.port
    except ValueError as exc:
        raise ValueError("invalid public base URL") from exc
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not parsed.netloc
        or hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("invalid public base URL")
    joined_path = f"{parsed.path.rstrip('/')}{download_path}"
    return urlunsplit(
        (parsed.scheme.lower(), parsed.netloc, joined_path, "", "")
    )


def _safe_apk_download_name(original_filename, release_id: str) -> str:
    fallback = f"{release_id}.apk"
    if not isinstance(original_filename, str):
        return fallback
    if (
        _SAFE_APK_DOWNLOAD_NAME.fullmatch(original_filename) is None
        or len(original_filename) > 255
    ):
        return fallback
    return original_filename


def _mobile_release_not_found():
    return jsonify(
        code="not_found", message="mobile release not found"
    ), 404


def _mobile_release_unavailable():
    return jsonify(
        code="mobile_release_unavailable",
        message="mobile release is unavailable",
    ), 503


def _invalid_public_base_url():
    return jsonify(
        code="invalid_public_base_url",
        message="mobile release configuration is invalid",
    ), 503
