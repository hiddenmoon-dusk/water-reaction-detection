from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path


class ReleaseConfigError(ValueError):
    pass


@dataclass(frozen=True)
class ModelPaths:
    detector: Path
    classifier: Path


@dataclass(frozen=True)
class ReleaseConfig:
    schema_version: int
    app_release_id: str
    model_generation: int
    dataset_generation: int
    api_base_url: str
    bootstrap_token: str


def runtime_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def model_paths(root: Path | None = None) -> ModelPaths:
    base = Path(root) if root is not None else runtime_root()
    return ModelPaths(
        detector=base / "yolov8n.pt",
        classifier=base / "reaction_classifier.h5",
    )


def load_release(path: Path | None = None) -> ReleaseConfig:
    release_path = Path(path) if path is not None else runtime_root() / "release.json"
    try:
        payload = json.loads(release_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleaseConfigError(f"无法读取发布配置: {release_path}") from exc

    required = (
        "schema_version",
        "app_release_id",
        "model_generation",
        "dataset_generation",
        "api_base_url",
        "bootstrap_token",
    )
    for field in required:
        if field not in payload:
            raise ReleaseConfigError(f"发布配置缺少字段: {field}")

    try:
        schema_version = int(payload["schema_version"])
        model_generation = int(payload["model_generation"])
        dataset_generation = int(payload["dataset_generation"])
    except (TypeError, ValueError) as exc:
        raise ReleaseConfigError("发布配置中的代次字段必须是整数") from exc

    app_release_id = str(payload["app_release_id"]).strip()
    api_base_url = str(payload["api_base_url"]).strip().rstrip("/")
    bootstrap_token = str(payload["bootstrap_token"]).strip()
    if schema_version != 1:
        raise ReleaseConfigError(f"不支持的发布配置版本: {schema_version}")
    if not app_release_id:
        raise ReleaseConfigError("app_release_id 不能为空")
    if model_generation < 1 or dataset_generation < 1:
        raise ReleaseConfigError("model_generation 和 dataset_generation 必须大于 0")
    if not api_base_url.startswith(("http://", "https://")):
        raise ReleaseConfigError("api_base_url 必须是 HTTP(S) 地址")
    if not bootstrap_token:
        raise ReleaseConfigError("bootstrap_token 不能为空")

    return ReleaseConfig(
        schema_version=schema_version,
        app_release_id=app_release_id,
        model_generation=model_generation,
        dataset_generation=dataset_generation,
        api_base_url=api_base_url,
        bootstrap_token=bootstrap_token,
    )


def _is_writable_directory(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / f".write-test-{os.getpid()}"
        probe.write_bytes(b"ok")
        probe.unlink()
        return True
    except OSError:
        return False


def result_root(
    root: Path | None = None,
    documents: Path | None = None,
) -> Path:
    base = Path(root) if root is not None else runtime_root()
    preferred = base / "结果"
    if _is_writable_directory(preferred):
        return preferred

    docs = (
        Path(documents)
        if documents is not None
        else Path.home() / "Documents"
    )
    fallback = docs / "水体反应管检测结果"
    if not _is_writable_directory(fallback):
        raise OSError(f"无法创建结果目录: {preferred} 或 {fallback}")
    return fallback


def app_data_root() -> Path:
    local_app_data = os.environ.get("LOCALAPPDATA")
    base = Path(local_app_data) if local_app_data else Path.home() / ".water-reaction-lab"
    path = base / "WaterReactionLab"
    path.mkdir(parents=True, exist_ok=True)
    return path
