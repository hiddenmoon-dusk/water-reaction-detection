import io
import json
import sys
import zipfile
from pathlib import Path

import pytest


SERVER_ROOT = Path(__file__).resolve().parents[1]
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))

from water_server import create_app
from water_server.db import get_db


@pytest.fixture()
def app(tmp_path):
    app = create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "test-secret",
            "DATABASE": str(tmp_path / "instance" / "app.db"),
            "STORAGE_ROOT": str(tmp_path / "storage"),
            "ADMIN_INITIAL_PASSWORD": "test-admin-password-2026",
            "BOOTSTRAP_TOKEN": "bootstrap-test",
            "SESSION_COOKIE_SECURE": False,
            "USE_X_ACCEL_REDIRECT": False,
        }
    )
    yield app


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def db(app):
    with app.app_context():
        yield get_db()


@pytest.fixture()
def auth_client(client):
    registration = client.post(
        "/api/v1/client/register",
        json={
            "bootstrap_token": "bootstrap-test",
            "app_release_id": "initial",
            "model_generation": 1,
        },
    )
    payload = registration.get_json()
    client.environ_base["HTTP_AUTHORIZATION"] = f"Bearer {payload['token']}"
    client.environ_base["HTTP_X_INSTALLATION_ID"] = payload["installation_id"]
    return client


@pytest.fixture()
def admin_client(client):
    response = client.post(
        "/admin/login",
        data={"password": "test-admin-password-2026"},
        follow_redirects=False,
    )
    assert response.status_code in (200, 302)
    with client.session_transaction() as session:
        client.environ_base["HTTP_X_CSRF_TOKEN"] = session["csrf_token"]
    return client


def build_result_zip(
    tmp_path,
    *,
    upload_id="upload-1",
    water_type="污水",
    labels=("已反应", "未反应"),
    dataset_generation=1,
    model_generation=1,
    app_release_id="initial",
    client_platform=None,
    app_version_code=None,
    device_model=None,
):
    from PIL import Image

    result_path = tmp_path / f"{upload_id}.zip"
    image_buffer = io.BytesIO()
    Image.new("RGB", (32, 32), "white").save(image_buffer, format="JPEG")
    image_bytes = image_buffer.getvalue()
    annotated_buffer = io.BytesIO()
    Image.new("RGB", (32, 32), "white").save(annotated_buffer, format="PNG")
    payload = {
        "schema_version": 1,
        "upload_id": upload_id,
        "captured_at": "2026-06-13T12:00:00+00:00",
        "water_type": water_type,
        "mode": "normal",
        "app_release_id": app_release_id,
        "model_generation": model_generation,
        "dataset_generation": dataset_generation,
        "results": [
            {
                "id": index,
                "x1": 1,
                "y1": 1,
                "x2": 20,
                "y2": 20,
                "label": label,
                "confidence": 0.9,
            }
            for index, label in enumerate(labels, start=1)
        ],
    }
    if client_platform is not None:
        payload["client_platform"] = client_platform
    if app_version_code is not None:
        payload["app_version_code"] = app_version_code
    if device_model is not None:
        payload["device_model"] = device_model
    with zipfile.ZipFile(result_path, "w") as archive:
        archive.writestr("original.jpg", image_bytes)
        archive.writestr("annotated.png", annotated_buffer.getvalue())
        archive.writestr("result.json", json.dumps(payload, ensure_ascii=False))
    return result_path


@pytest.fixture()
def result_zip(tmp_path):
    return build_result_zip(
        tmp_path,
        labels=("已反应", "已反应", "未反应"),
    )
