import json
import zipfile

from scripts.e2e_smoke import build_sample_archive, extract_csrf_token


def test_build_sample_archive_contains_two_tube_results(tmp_path):
    release = {
        "app_release_id": "release-1",
        "model_generation": 2,
        "dataset_generation": 1,
    }

    archive_path = build_sample_archive(tmp_path / "sample.zip", release)

    with zipfile.ZipFile(archive_path) as archive:
        assert set(archive.namelist()) == {
            "original.jpg",
            "annotated.png",
            "result.json",
        }
        payload = json.loads(archive.read("result.json"))
    assert [item["label"] for item in payload["results"]] == [
        "已反应",
        "未反应",
    ]


def test_extract_csrf_token_reads_dashboard_field():
    html = '<input type="hidden" name="csrf_token" value="token-123">'

    assert extract_csrf_token(html) == "token-123"
