import io
import zipfile

from conftest import build_result_zip
from test_uploads import post_zip


def test_export_contains_three_water_folders_and_uploaded_files(
    auth_client, admin_client, result_zip
):
    assert post_zip(auth_client, result_zip).status_code == 201

    response = admin_client.get("/admin/results.zip")

    assert response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(response.data)) as archive:
        names = set(archive.namelist())
        assert "污水/upload-1/original.jpg" in names
        assert "污水/upload-1/annotated.png" in names
        assert "污水/upload-1/result.json" in names
        assert any(name.startswith("生活用水/") for name in names)
        assert any(name.startswith("养殖水体/") for name in names)


def test_clear_increments_generation_and_deletes_results(
    auth_client, admin_client, result_zip, db, app
):
    assert post_zip(auth_client, result_zip).status_code == 201

    response = admin_client.post(
        "/admin/results/clear",
        data={"password": "test-admin-password-2026"},
    )

    assert response.status_code == 200
    state = db.execute("SELECT dataset_generation FROM app_state WHERE id = 1").fetchone()
    assert state["dataset_generation"] == 2
    assert db.execute("SELECT COUNT(*) FROM uploads").fetchone()[0] == 0
    storage = app.config["STORAGE_ROOT"]
    assert list(__import__("pathlib").Path(storage).glob("results/*/*/result.json")) == []


def test_old_generation_upload_is_rejected_after_clear(
    auth_client, admin_client, tmp_path
):
    admin_client.post(
        "/admin/results/clear",
        data={"password": "test-admin-password-2026"},
    )
    archive = build_result_zip(
        tmp_path,
        upload_id="old-after-clear",
        dataset_generation=1,
    )

    response = post_zip(auth_client, archive)

    assert response.status_code == 409
    assert response.get_json()["code"] == "generation_expired"
