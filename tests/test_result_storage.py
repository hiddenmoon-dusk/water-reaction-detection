import json
import zipfile

import numpy as np

from client_config import ReleaseConfig
from result_storage import next_numbered_directory, save_result


def sample_release():
    return ReleaseConfig(
        schema_version=1,
        app_release_id="release-1",
        model_generation=1,
        dataset_generation=1,
        api_base_url="https://water.example.test",
        bootstrap_token="test",
    )


def test_next_numbered_directory_reuses_first_gap(tmp_path):
    water_dir = tmp_path / "污水"
    (water_dir / "001").mkdir(parents=True)
    (water_dir / "003").mkdir()

    assert next_numbered_directory(water_dir).name == "002"


def test_save_result_writes_three_files_metadata_and_archive(tmp_path):
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    annotations = [
        (5, 6, 40, 50, "已反应", 0.9234),
        (50, 10, 100, 70, "未反应", 0.8123),
    ]

    saved = save_result(
        result_root=tmp_path / "结果",
        archive_root=tmp_path / "queue",
        water_type="污水",
        mode="normal",
        image_rgb=image,
        annotations=annotations,
        release=sample_release(),
    )

    assert {path.name for path in saved.directory.iterdir()} == {
        "original.jpg",
        "annotated.png",
        "result.json",
    }
    payload = json.loads((saved.directory / "result.json").read_text("utf-8"))
    assert payload["upload_id"] == saved.upload_id
    assert payload["water_type"] == "污水"
    assert payload["app_release_id"] == "release-1"
    assert payload["model_generation"] == 1
    assert payload["dataset_generation"] == 1
    assert [item["label"] for item in payload["results"]] == ["已反应", "未反应"]

    with zipfile.ZipFile(saved.archive_path) as archive:
        assert set(archive.namelist()) == {
            "original.jpg",
            "annotated.png",
            "result.json",
        }


def test_save_result_rejects_unknown_label(tmp_path):
    image = np.zeros((20, 20, 3), dtype=np.uint8)

    try:
        save_result(
            result_root=tmp_path / "结果",
            archive_root=tmp_path / "queue",
            water_type="污水",
            mode="normal",
            image_rgb=image,
            annotations=[(0, 0, 10, 10, "不确定", 0.5)],
            release=sample_release(),
        )
    except ValueError as exc:
        assert "标签" in str(exc)
    else:
        raise AssertionError("unknown labels must be rejected")
