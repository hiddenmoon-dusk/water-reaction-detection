import json

from water_server.validation import validate_result_payload


def test_validate_result_payload_accepts_utc_z_suffix():
    payload = {
        "schema_version": 1,
        "upload_id": "z-time-upload",
        "captured_at": "2026-07-18T12:01:49Z",
        "water_type": "污水",
        "mode": "normal",
        "app_release_id": "release-android",
        "model_generation": 6,
        "dataset_generation": 3,
        "client_platform": "android",
        "results": [{
            "id": 1,
            "x1": 1,
            "y1": 1,
            "x2": 20,
            "y2": 20,
            "label": "已反应",
            "confidence": 0.9,
        }],
    }

    validated = validate_result_payload(json.dumps(payload).encode("utf-8"))

    assert validated["captured_at"].endswith("+00:00")
