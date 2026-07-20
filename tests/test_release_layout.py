import json
import os
from pathlib import Path


def release_dir():
    return Path(
        os.environ.get(
            "WATER_RELEASE_DIR",
            Path("dist") / "水体反应管检测系统",
        )
    )


def test_release_has_launcher_models_and_metadata():
    root = release_dir()

    assert (root / "水体反应管检测系统.exe").is_file()
    assert (root / "reaction_classifier.h5").is_file()
    assert (root / "yolov8n.pt").is_file()
    release = json.loads((root / "release.json").read_text(encoding="utf-8"))
    assert release["schema_version"] == 1
    assert release["app_release_id"]
    assert release["model_generation"] >= 1
