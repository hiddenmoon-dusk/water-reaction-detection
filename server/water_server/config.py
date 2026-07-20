from __future__ import annotations

import os
from pathlib import Path


def _strict_environment_boolean(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"{name} must be true, false, 1, or 0")


def default_config(instance_path: str) -> dict:
    root = Path(os.environ.get("WATER_APP_ROOT", Path(instance_path).parent))
    return {
        "SECRET_KEY": os.environ.get("WATER_SECRET_KEY", "development-change-me"),
        "DATABASE": os.environ.get(
            "WATER_DATABASE",
            str(Path(instance_path) / "app.db"),
        ),
        "STORAGE_ROOT": os.environ.get(
            "WATER_STORAGE_ROOT",
            str(root / "storage"),
        ),
        "PUBLIC_BASE_URL": os.environ.get(
            "WATER_PUBLIC_BASE_URL",
            "https://hiddenmoon.duckdns.org",
        ),
        "ADMIN_INITIAL_PASSWORD": os.environ.get(
            "WATER_ADMIN_INITIAL_PASSWORD", "change-me-before-use"
        ),
        "BOOTSTRAP_TOKEN": os.environ.get(
            "WATER_BOOTSTRAP_TOKEN",
            "water-reaction-bootstrap-v1",
        ),
        "MAX_CONTENT_LENGTH": 2 * 1024 * 1024 * 1024,
        "MAX_RESULT_UPLOAD_BYTES": 64 * 1024 * 1024,
        "MAX_ARCHIVE_UNCOMPRESSED": 96 * 1024 * 1024,
        "MAX_RELEASE_UNCOMPRESSED": 4 * 1024 * 1024 * 1024,
        "MAX_ANDROID_APK_BYTES": int(
            os.environ.get("WATER_MAX_ANDROID_APK_BYTES", 512 * 1024 * 1024)
        ),
        "APKSIGNER_PATH": os.environ.get("WATER_APKSIGNER_PATH", "apksigner"),
        "ANDROID_SIGNING_CERT_SHA256": os.environ.get(
            "WATER_ANDROID_SIGNING_CERT_SHA256", ""
        ),
        "RELEASE_RESERVATION_HOURS": 24,
        "RELEASE_ORPHAN_GRACE_HOURS": 24,
        "USE_X_ACCEL_REDIRECT": True,
        "TRUSTED_IMMUTABLE_RELEASES": _strict_environment_boolean(
            "WATER_TRUSTED_IMMUTABLE_RELEASES", False
        ),
        "SESSION_COOKIE_HTTPONLY": True,
        "SESSION_COOKIE_SAMESITE": "Lax",
        "SESSION_COOKIE_SECURE": True,
        "PERMANENT_SESSION_LIFETIME": 3600,
    }
