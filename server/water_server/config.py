from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlsplit


_PRODUCTION_PLACEHOLDERS = {
    "",
    "change-me-before-use",
    "development-change-me",
    "water-reaction-bootstrap-v1",
}


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


def validate_production_config(config: dict) -> None:
    """Reject unsafe defaults before the production service opens its database."""
    if not config.get("PRODUCTION", False):
        return

    for name in ("SECRET_KEY", "ADMIN_INITIAL_PASSWORD", "BOOTSTRAP_TOKEN"):
        value = str(config.get(name, "")).strip()
        if value in _PRODUCTION_PLACEHOLDERS:
            environment_name = {
                "SECRET_KEY": "WATER_SECRET_KEY",
                "ADMIN_INITIAL_PASSWORD": "WATER_ADMIN_INITIAL_PASSWORD",
                "BOOTSTRAP_TOKEN": "WATER_BOOTSTRAP_TOKEN",
            }[name]
            raise RuntimeError(
                f"生产模式必须设置 {environment_name}，不能使用默认占位值"
            )

    if len(str(config["SECRET_KEY"])) < 32:
        raise RuntimeError("生产模式的 WATER_SECRET_KEY 至少需要 32 个字符")

    if len(str(config["ADMIN_INITIAL_PASSWORD"])) < 12:
        raise RuntimeError(
            "生产模式的 WATER_ADMIN_INITIAL_PASSWORD 至少需要 12 个字符"
        )
    if len(str(config["BOOTSTRAP_TOKEN"])) < 16:
        raise RuntimeError(
            "生产模式的 WATER_BOOTSTRAP_TOKEN 至少需要 16 个字符"
        )

    public_url = urlsplit(str(config.get("PUBLIC_BASE_URL", "")).strip())
    if public_url.scheme != "https" or not public_url.netloc:
        raise RuntimeError("生产模式的 WATER_PUBLIC_BASE_URL 必须是 HTTPS 地址")
    if public_url.username or public_url.password:
        raise RuntimeError("WATER_PUBLIC_BASE_URL 不能包含账号或密码")


def default_config(instance_path: str) -> dict:
    root = Path(os.environ.get("WATER_APP_ROOT", Path(instance_path).parent))
    return {
        "PRODUCTION": _strict_environment_boolean("WATER_PRODUCTION", False),
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
            "https://example.invalid",
        ),
        "ADMIN_INITIAL_PASSWORD": os.environ.get(
            "WATER_ADMIN_INITIAL_PASSWORD",
            "change-me-before-use",
        ),
        "BOOTSTRAP_TOKEN": os.environ.get(
            "WATER_BOOTSTRAP_TOKEN",
            "change-me-before-use",
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
