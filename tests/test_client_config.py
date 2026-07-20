import json
import sys
from pathlib import Path

import pytest

from client_config import (
    ReleaseConfigError,
    load_release,
    model_paths,
    result_root,
    runtime_root,
)


def test_runtime_root_uses_executable_when_frozen(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "app.exe"))

    assert runtime_root() == tmp_path


def test_required_models_are_external(tmp_path):
    paths = model_paths(tmp_path)

    assert paths.detector == tmp_path / "yolov8n.pt"
    assert paths.classifier == tmp_path / "reaction_classifier.h5"


def test_load_release_requires_all_fields(tmp_path):
    path = tmp_path / "release.json"
    path.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")

    with pytest.raises(ReleaseConfigError, match="app_release_id"):
        load_release(path)


def test_load_release_returns_validated_values(tmp_path):
    path = tmp_path / "release.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "app_release_id": "initial",
                "model_generation": 1,
                "dataset_generation": 1,
                "api_base_url": "https://hiddenmoon.duckdns.org",
                "bootstrap_token": "bootstrap-v1",
            }
        ),
        encoding="utf-8",
    )

    release = load_release(path)

    assert release.app_release_id == "initial"
    assert release.model_generation == 1
    assert release.api_base_url == "https://hiddenmoon.duckdns.org"


def test_result_root_falls_back_to_documents_when_runtime_is_not_writable(
    monkeypatch, tmp_path
):
    runtime = tmp_path / "runtime"
    documents = tmp_path / "Documents"
    runtime.mkdir()

    def fake_probe(path: Path) -> bool:
        return path == documents / "水体反应管检测结果"

    monkeypatch.setattr("client_config._is_writable_directory", fake_probe)

    assert result_root(runtime, documents) == documents / "水体反应管检测结果"
