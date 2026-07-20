from __future__ import annotations

import json
import shutil
import uuid
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

from client_config import ReleaseConfig


WATER_TYPES = {"污水", "生活用水", "养殖水体"}
LABELS = {"已反应", "未反应"}
MODES = {"normal", "scan", "manual"}


@dataclass(frozen=True)
class SavedResult:
    upload_id: str
    directory: Path
    archive_path: Path


def next_numbered_directory(water_dir: Path) -> Path:
    water_dir.mkdir(parents=True, exist_ok=True)
    existing = {
        int(path.name)
        for path in water_dir.iterdir()
        if path.is_dir() and path.name.isdigit()
    }
    number = 1
    while number in existing:
        number += 1
    return water_dir / f"{number:03d}"


def _write_encoded(path: Path, extension: str, image_bgr: np.ndarray) -> None:
    ok, encoded = cv2.imencode(extension, image_bgr)
    if not ok:
        raise OSError(f"无法编码图片: {path.name}")
    path.write_bytes(encoded.tobytes())


def _annotation_values(annotation: Sequence) -> tuple[int, int, int, int, str, float]:
    if len(annotation) < 6:
        raise ValueError("检测标注字段不足")
    x1, y1, x2, y2, label, confidence = annotation[:6]
    label = str(label)
    if label not in LABELS:
        raise ValueError(f"不支持的检测标签: {label}")
    coordinates = tuple(int(value) for value in (x1, y1, x2, y2))
    if min(coordinates) < 0 or coordinates[2] <= coordinates[0] or coordinates[3] <= coordinates[1]:
        raise ValueError("检测框坐标无效")
    confidence = float(confidence)
    if not 0 <= confidence <= 1:
        raise ValueError("置信度必须在 0 到 1 之间")
    return (*coordinates, label, confidence)


def _annotated_image(image_rgb: np.ndarray, annotations: Iterable[Sequence]) -> np.ndarray:
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    for annotation in annotations:
        x1, y1, x2, y2, label, confidence = _annotation_values(annotation)
        color = (80, 180, 35) if label == "已反应" else (40, 80, 225)
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, 3)
        cv2.putText(
            image_bgr,
            f"{label} {confidence:.1%}",
            (x1, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
    return image_bgr


def _result_payload(
    upload_id: str,
    water_type: str,
    mode: str,
    annotations: list[Sequence],
    release: ReleaseConfig,
) -> dict:
    results = []
    for index, annotation in enumerate(annotations, start=1):
        x1, y1, x2, y2, label, confidence = _annotation_values(annotation)
        results.append(
            {
                "id": index,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "label": label,
                "confidence": round(confidence, 4),
            }
        )
    return {
        "schema_version": 1,
        "upload_id": upload_id,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "water_type": water_type,
        "mode": mode,
        "app_release_id": release.app_release_id,
        "model_generation": release.model_generation,
        "dataset_generation": release.dataset_generation,
        "results": results,
    }


def save_result(
    result_root: Path,
    archive_root: Path,
    water_type: str,
    mode: str,
    image_rgb: np.ndarray,
    annotations: Iterable[Sequence],
    release: ReleaseConfig,
) -> SavedResult:
    if water_type not in WATER_TYPES:
        raise ValueError(f"不支持的水体类型: {water_type}")
    if mode not in MODES:
        raise ValueError(f"不支持的检测模式: {mode}")
    if not isinstance(image_rgb, np.ndarray) or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("原图必须是 RGB 三通道图片")

    annotation_list = list(annotations)
    if not annotation_list:
        raise ValueError("没有可保存的检测结果")
    for annotation in annotation_list:
        _annotation_values(annotation)

    upload_id = str(uuid.uuid4())
    final_dir = next_numbered_directory(Path(result_root) / water_type)
    temp_dir = final_dir.with_name(final_dir.name + f".{upload_id}.tmp")
    archive_dir = Path(archive_root)
    archive_path = archive_dir / f"{upload_id}.zip"
    temp_archive = archive_path.with_suffix(".zip.tmp")

    temp_dir.mkdir(parents=True, exist_ok=False)
    archive_dir.mkdir(parents=True, exist_ok=True)
    try:
        original_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        _write_encoded(temp_dir / "original.jpg", ".jpg", original_bgr)
        _write_encoded(
            temp_dir / "annotated.png",
            ".png",
            _annotated_image(image_rgb, annotation_list),
        )
        payload = _result_payload(upload_id, water_type, mode, annotation_list, release)
        (temp_dir / "result.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        temp_dir.replace(final_dir)
        with zipfile.ZipFile(temp_archive, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in ("original.jpg", "annotated.png", "result.json"):
                archive.write(final_dir / name, arcname=name)
        temp_archive.replace(archive_path)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        if final_dir.exists() and not archive_path.exists():
            shutil.rmtree(final_dir, ignore_errors=True)
        temp_archive.unlink(missing_ok=True)
        raise

    return SavedResult(upload_id, final_dir, archive_path)
