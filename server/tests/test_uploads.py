from pathlib import Path

from conftest import build_result_zip


def publish_android(db, tmp_path, *, batch_id="a" * 32, generation=2):
    release_id = f"{batch_id}-android"
    artifact = tmp_path / f"{release_id}.apk"
    artifact.write_bytes(b"apk")
    db.execute(
        """
        INSERT INTO release_batches (
            batch_id, model_generation, dataset_generation, status,
            reserved_at, expires_at, published_at
        ) VALUES (?, ?, 1, 'partial', CURRENT_TIMESTAMP,
                  '2099-01-01T00:00:00+00:00', CURRENT_TIMESTAMP)
        """,
        (batch_id, generation),
    )
    db.execute(
        """
        INSERT INTO platform_releases (
            release_id, batch_id, platform, version_code, version_name,
            original_filename, stored_path, sha256, size_bytes,
            uploaded_at, is_current
        ) VALUES (?, ?, 'android', 2, '1.0.1', 'app.apk', ?, ?, 3,
                  CURRENT_TIMESTAMP, 1)
        """,
        (release_id, batch_id, str(artifact), "0" * 64),
    )
    db.commit()
    return release_id


def post_zip(client, path: Path):
    with path.open("rb") as stream:
        return client.post(
            "/api/v1/results",
            data={"file": (stream, path.name)},
            content_type="multipart/form-data",
        )


def test_registration_rejects_wrong_bootstrap_token(client):
    response = client.post(
        "/api/v1/client/register",
        json={
            "bootstrap_token": "wrong",
            "app_release_id": "initial",
            "model_generation": 1,
        },
    )

    assert response.status_code == 403


def test_android_registration_uses_android_current_release(client, db, tmp_path):
    release_id = publish_android(db, tmp_path)

    response = client.post(
        "/api/v1/client/register",
        json={
            "bootstrap_token": "bootstrap-test",
            "client_platform": "android",
            "app_release_id": release_id,
            "model_generation": 2,
        },
    )

    assert response.status_code == 201
    installation = db.execute(
        "SELECT client_platform FROM installations WHERE installation_id = ?",
        (response.get_json()["installation_id"],),
    ).fetchone()
    assert installation["client_platform"] == "android"


def test_new_desktop_release_does_not_expire_android_upload(
    client, db, tmp_path
):
    release_id = publish_android(db, tmp_path)
    registration = client.post(
        "/api/v1/client/register",
        json={
            "bootstrap_token": "bootstrap-test",
            "client_platform": "android",
            "app_release_id": release_id,
            "model_generation": 2,
        },
    )
    credentials = registration.get_json()
    client.environ_base["HTTP_AUTHORIZATION"] = (
        f"Bearer {credentials['token']}"
    )
    client.environ_base["HTTP_X_INSTALLATION_ID"] = credentials[
        "installation_id"
    ]
    db.execute(
        """
        UPDATE app_state
        SET current_release_id = 'new-desktop', model_generation = 3
        WHERE id = 1
        """
    )
    db.commit()
    archive = build_result_zip(
        tmp_path,
        upload_id="android-upload",
        app_release_id=release_id,
        model_generation=2,
        client_platform="android",
        app_version_code=2,
        device_model="Pixel",
    )

    response = post_zip(client, archive)

    assert response.status_code == 201


def test_config_returns_current_generations(client):
    response = client.get("/api/v1/client/config")

    assert response.status_code == 200
    assert response.get_json()["dataset_generation"] == 1
    assert response.get_json()["model_generation"] == 1
    assert response.get_json()["current_release_id"] == "initial"
    assert response.get_json()["max_upload_bytes"] == 64 * 1024 * 1024


def test_server_allows_large_admin_packages_but_limits_result_archives(app):
    assert app.config["MAX_CONTENT_LENGTH"] == 2 * 1024 * 1024 * 1024
    assert app.config["MAX_RESULT_UPLOAD_BYTES"] == 64 * 1024 * 1024
    assert app.config["MAX_RELEASE_UNCOMPRESSED"] == 4 * 1024 * 1024 * 1024


def test_multiple_tubes_are_counted_individually(auth_client, result_zip, db):
    response = post_zip(auth_client, result_zip)

    assert response.status_code == 201
    assert response.get_json()["positive_count"] == 2
    assert response.get_json()["negative_count"] == 1
    rows = db.execute(
        "SELECT label FROM tube_results ORDER BY tube_index"
    ).fetchall()
    assert [row["label"] for row in rows] == ["已反应", "已反应", "未反应"]


def test_duplicate_upload_is_idempotent(auth_client, result_zip, db):
    assert post_zip(auth_client, result_zip).status_code == 201

    duplicate = post_zip(auth_client, result_zip)

    assert duplicate.status_code == 208
    count = db.execute("SELECT COUNT(*) FROM uploads").fetchone()[0]
    assert count == 1


def test_expired_dataset_generation_is_rejected(auth_client, tmp_path):
    archive = build_result_zip(
        tmp_path,
        upload_id="old-generation",
        dataset_generation=0,
    )

    response = post_zip(auth_client, archive)

    assert response.status_code == 409
    assert response.get_json()["code"] == "generation_expired"


def test_upload_status_does_not_expose_storage_path(auth_client, result_zip):
    post_zip(auth_client, result_zip)

    response = auth_client.get("/api/v1/uploads/upload-1")

    assert response.status_code == 200
    assert "storage_path" not in response.get_json()
