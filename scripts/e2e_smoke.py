from __future__ import annotations

import argparse
import io
import json
import re
import tempfile
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import requests
from PIL import Image


WATER_TYPES = ("污水", "生活用水", "养殖水体")


def extract_csrf_token(html: str) -> str:
    match = re.search(r'name="csrf_token"\s+value="([^"]+)"', html)
    if not match:
        raise ValueError("CSRF token not found")
    return match.group(1)


def build_sample_archive(
    path: Path,
    release: dict,
    upload_id: str | None = None,
) -> Path:
    upload_id = upload_id or f"smoke-{uuid.uuid4().hex}"
    original = io.BytesIO()
    annotated = io.BytesIO()
    Image.new("RGB", (64, 64), "white").save(original, format="JPEG")
    Image.new("RGB", (64, 64), "white").save(annotated, format="PNG")
    payload = {
        "schema_version": 1,
        "upload_id": upload_id,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "water_type": "污水",
        "mode": "normal",
        "app_release_id": release["app_release_id"],
        "model_generation": release["model_generation"],
        "dataset_generation": release["dataset_generation"],
        "results": [
            {
                "id": 1,
                "x1": 4,
                "y1": 4,
                "x2": 28,
                "y2": 56,
                "label": "已反应",
                "confidence": 0.98,
            },
            {
                "id": 2,
                "x1": 34,
                "y1": 4,
                "x2": 58,
                "y2": 56,
                "label": "未反应",
                "confidence": 0.97,
            },
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("original.jpg", original.getvalue())
        archive.writestr("annotated.png", annotated.getvalue())
        archive.writestr(
            "result.json",
            json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        )
    return path


def _check_response(response: requests.Response, expected: set[int]) -> dict:
    if response.status_code not in expected:
        raise RuntimeError(
            f"{response.request.method} {response.url} returned "
            f"{response.status_code}: {response.text[:500]}"
        )
    try:
        return response.json()
    except ValueError:
        return {}


def run_smoke(
    base_url: str,
    release: dict,
    bootstrap_token: str,
    admin_password: str | None = None,
    clear: bool = False,
    export_path: Path | None = None,
) -> dict:
    base_url = base_url.rstrip("/")
    session = requests.Session()
    registration = session.post(
        f"{base_url}/api/v1/client/register",
        json={
            "bootstrap_token": bootstrap_token,
            "app_release_id": release["app_release_id"],
            "model_generation": release["model_generation"],
        },
        timeout=(10, 30),
    )
    credentials = _check_response(registration, {201})
    headers = {
        "Authorization": f"Bearer {credentials['token']}",
        "X-Installation-ID": credentials["installation_id"],
    }

    with tempfile.TemporaryDirectory(prefix="water-e2e-") as temp_dir:
        archive_path = build_sample_archive(
            Path(temp_dir) / "sample.zip",
            release,
        )
        with archive_path.open("rb") as stream:
            created = session.post(
                f"{base_url}/api/v1/results",
                headers=headers,
                files={"file": ("sample.zip", stream, "application/zip")},
                timeout=(10, 90),
            )
        created_payload = _check_response(created, {201})
        with archive_path.open("rb") as stream:
            duplicate = session.post(
                f"{base_url}/api/v1/results",
                headers=headers,
                files={"file": ("sample.zip", stream, "application/zip")},
                timeout=(10, 90),
            )
        duplicate_payload = _check_response(duplicate, {208})

    statistics = _check_response(
        session.get(
            f"{base_url}/api/v1/public/statistics",
            timeout=(10, 30),
        ),
        {200},
    )
    wastewater = next(
        row for row in statistics["water_types"] if row["water_type"] == "污水"
    )
    if wastewater["positive_count"] < 1 or wastewater["negative_count"] < 1:
        raise RuntimeError("public statistics did not include the smoke sample")

    result = {
        "created": created_payload,
        "duplicate": duplicate_payload,
        "statistics": statistics,
    }
    if admin_password is None:
        return result

    login = session.post(
        f"{base_url}/admin/login",
        data={"password": admin_password},
        allow_redirects=False,
        timeout=(10, 30),
    )
    _check_response(login, {302})
    dashboard = session.get(f"{base_url}/admin", timeout=(10, 30))
    _check_response(dashboard, {200})
    csrf_token = extract_csrf_token(dashboard.text)

    export = session.get(
        f"{base_url}/admin/results.zip",
        timeout=(10, 120),
    )
    _check_response(export, {200})
    export_path = export_path or Path("output") / "e2e-results-export.zip"
    export_path.parent.mkdir(parents=True, exist_ok=True)
    export_path.write_bytes(export.content)
    with zipfile.ZipFile(export_path) as archive:
        names = set(archive.namelist())
    for water_type in WATER_TYPES:
        if f"{water_type}/" not in names:
            raise RuntimeError(f"export is missing {water_type}/")
    result["export_path"] = str(export_path.resolve())

    if clear:
        cleared = session.post(
            f"{base_url}/admin/results/clear",
            headers={"X-CSRF-Token": csrf_token},
            data={"password": admin_password},
            timeout=(10, 120),
        )
        result["clear"] = _check_response(cleared, {200})
        after_clear = _check_response(
            session.get(
                f"{base_url}/api/v1/public/statistics",
                timeout=(10, 30),
            ),
            {200},
        )
        if any(row["total_count"] for row in after_clear["water_types"]):
            raise RuntimeError("statistics were not empty after clear")
        result["after_clear"] = after_clear
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url",
        default="https://hiddenmoon.duckdns.org",
    )
    parser.add_argument(
        "--release-json",
        type=Path,
        default=Path("dist/水体反应管检测系统/release.json"),
    )
    parser.add_argument("--admin-password")
    parser.add_argument("--clear", action="store_true")
    parser.add_argument(
        "--export-path",
        type=Path,
        default=Path("output/e2e-results-export.zip"),
    )
    args = parser.parse_args()
    release = json.loads(args.release_json.read_text(encoding="utf-8"))
    result = run_smoke(
        args.base_url,
        release,
        release["bootstrap_token"],
        admin_password=args.admin_password,
        clear=args.clear,
        export_path=args.export_path,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
