import io
import json
import zipfile

from conftest import build_result_zip
from test_uploads import post_zip


def test_zip_traversal_is_rejected(auth_client, tmp_path):
    archive_path = tmp_path / "traversal.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../result.json", "{}")

    response = post_zip(auth_client, archive_path)

    assert response.status_code == 400
    assert response.get_json()["code"] == "invalid_archive"


def test_unexpected_file_is_rejected(auth_client, tmp_path):
    valid = build_result_zip(tmp_path, upload_id="extra-file")
    rewritten = tmp_path / "extra.zip"
    with zipfile.ZipFile(valid) as source, zipfile.ZipFile(rewritten, "w") as target:
        for name in source.namelist():
            target.writestr(name, source.read(name))
        target.writestr("secret.txt", "not allowed")

    response = post_zip(auth_client, rewritten)

    assert response.status_code == 400
    assert response.get_json()["code"] == "invalid_archive"


def test_invalid_label_is_rejected(auth_client, tmp_path):
    archive = build_result_zip(
        tmp_path,
        upload_id="invalid-label",
        labels=("不确定",),
    )

    response = post_zip(auth_client, archive)

    assert response.status_code == 400
    assert response.get_json()["code"] == "invalid_payload"
