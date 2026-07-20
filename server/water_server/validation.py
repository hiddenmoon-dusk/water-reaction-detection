from __future__ import annotations

import json
from datetime import datetime


WATER_TYPES = ("污水", "生活用水", "养殖水体")
MODES = {"normal", "scan", "manual"}
LABELS = {"已反应", "未反应"}


class InvalidPayload(ValueError):
    pass


def validate_result_payload(data: bytes) -> dict:
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InvalidPayload("result.json 不是有效 UTF-8 JSON") from exc

    required = (
        "schema_version",
        "upload_id",
        "captured_at",
        "water_type",
        "mode",
        "app_release_id",
        "model_generation",
        "dataset_generation",
        "results",
    )
    for field in required:
        if field not in payload:
            raise InvalidPayload(f"缺少字段: {field}")

    if payload["schema_version"] != 1:
        raise InvalidPayload("不支持的数据格式版本")
    if not isinstance(payload["upload_id"], str) or not 1 <= len(payload["upload_id"]) <= 80:
        raise InvalidPayload("upload_id 无效")
    captured_at = payload["captured_at"]
    if isinstance(captured_at, str) and captured_at.endswith("Z"):
        # Python 3.10 does not parse the ISO-8601 UTC designator directly,
        # while Java/Instant and older Android builds commonly emit it.
        captured_at = captured_at[:-1] + "+00:00"
        payload["captured_at"] = captured_at
    try:
        datetime.fromisoformat(captured_at)
    except (TypeError, ValueError) as exc:
        raise InvalidPayload("captured_at 无效") from exc
    if payload["water_type"] not in WATER_TYPES:
        raise InvalidPayload("水体类型无效")
    if payload["mode"] not in MODES:
        raise InvalidPayload("检测模式无效")
    if not isinstance(payload["app_release_id"], str) or not payload["app_release_id"]:
        raise InvalidPayload("app_release_id 无效")
    if not isinstance(payload["model_generation"], int):
        raise InvalidPayload("model_generation 无效")
    if not isinstance(payload["dataset_generation"], int):
        raise InvalidPayload("dataset_generation 无效")
    if payload.get("client_platform", "desktop") not in {"desktop", "android"}:
        raise InvalidPayload("client_platform 无效")
    if "app_version_code" in payload and (
        not isinstance(payload["app_version_code"], int)
        or isinstance(payload["app_version_code"], bool)
        or payload["app_version_code"] < 1
    ):
        raise InvalidPayload("app_version_code 无效")
    if "device_model" in payload and (
        not isinstance(payload["device_model"], str)
        or len(payload["device_model"]) > 200
    ):
        raise InvalidPayload("device_model 无效")

    results = payload["results"]
    if not isinstance(results, list) or not 1 <= len(results) <= 5000:
        raise InvalidPayload("results 必须包含 1 到 5000 个反应管")
    seen_ids = set()
    for item in results:
        if not isinstance(item, dict):
            raise InvalidPayload("反应管记录必须是对象")
        for field in ("id", "x1", "y1", "x2", "y2", "label", "confidence"):
            if field not in item:
                raise InvalidPayload(f"反应管记录缺少字段: {field}")
        if not isinstance(item["id"], int) or item["id"] < 1 or item["id"] in seen_ids:
            raise InvalidPayload("反应管 id 无效或重复")
        seen_ids.add(item["id"])
        coordinates = (item["x1"], item["y1"], item["x2"], item["y2"])
        if not all(isinstance(value, int) for value in coordinates):
            raise InvalidPayload("坐标必须是整数")
        if min(coordinates) < 0 or item["x2"] <= item["x1"] or item["y2"] <= item["y1"]:
            raise InvalidPayload("检测框坐标无效")
        if item["label"] not in LABELS:
            raise InvalidPayload("标签无效")
        confidence = item["confidence"]
        if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            raise InvalidPayload("置信度无效")
    return payload
