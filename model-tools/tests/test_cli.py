from __future__ import annotations

import importlib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib
from types import SimpleNamespace

import numpy as np
import pytest

from water_models.contracts import ModelManifest, TensorSpec
from water_models.inference import Detection


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BATCH_ID = "a" * 32


def _conversion_specs() -> dict[str, object]:
    return {
        "detector_input": TensorSpec("images", (1, 640, 640, 3), "float32"),
        "detector_output": TensorSpec("output0", (1, 5, 8400), "float32"),
        "classifier_input": TensorSpec(
            "serving_default_input", (1, 128, 128, 3), "float32"
        ),
        "classifier_output": TensorSpec("StatefulPartitionedCall", (1, 1), "float32"),
        "class_names": ["lib"],
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(
    detector_source: Path,
    classifier_source: Path,
    detector_tflite: Path,
    classifier_tflite: Path,
    *,
    detector_confidence: float = 0.3,
    classifier_threshold: float = 0.5,
    nms_iou: float = 0.45,
) -> ModelManifest:
    specs = _conversion_specs()
    return ModelManifest(
        schema_version=1,
        release_batch_id=BATCH_ID,
        app_release_id=f"{BATCH_ID}-android",
        app_version_code=7,
        app_version_name="7.0.0",
        model_generation=5,
        dataset_generation=4,
        detector_sha256=_sha256(detector_tflite),
        classifier_sha256=_sha256(classifier_tflite),
        detector_source_sha256=_sha256(detector_source),
        classifier_source_sha256=_sha256(classifier_source),
        detector_input=specs["detector_input"],
        detector_output=specs["detector_output"],
        classifier_input=specs["classifier_input"],
        classifier_output=specs["classifier_output"],
        detector_confidence=detector_confidence,
        classifier_threshold=classifier_threshold,
        nms_iou=nms_iou,
        class_names=("lib",),
        conversion={
            "precision": "fp16",
            "ultralytics": "8.4.25",
            "tensorflow": "2.21.0",
        },
    )


def _inspection(
    *,
    detector: bool,
    input_signature: list[int] | None = None,
    output_signature: list[int] | None = None,
) -> object:
    specs = _conversion_specs()
    input_spec = specs["detector_input" if detector else "classifier_input"]
    output_spec = specs["detector_output" if detector else "classifier_output"]

    def tensor(spec: TensorSpec, signature: list[int] | None) -> object:
        return SimpleNamespace(
            name=spec.name,
            shape=list(spec.shape),
            shape_signature=list(spec.shape) if signature is None else signature,
            dtype=spec.dtype,
        )

    return SimpleNamespace(
        input=tensor(input_spec, input_signature),
        output=tensor(output_spec, output_signature),
    )


@pytest.fixture
def compatible_inspector(monkeypatch: pytest.MonkeyPatch) -> None:
    cli = importlib.import_module("water_models.cli")

    def inspect(path: Path) -> object:
        detector = path.name == "detector.tflite"
        return _inspection(
            detector=detector,
            input_signature=None if detector else [-1, 128, 128, 3],
            output_signature=None if detector else [-1, 1],
        )

    monkeypatch.setattr(cli, "inspect_tflite", inspect, raising=False)


@pytest.fixture(scope="session")
def real_classifier_artifact(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, object]:
    import tensorflow as tf

    from water_models.conversion import convert_classifier, inspect_tflite

    directory = tmp_path_factory.mktemp("cli-real-classifier")
    source = directory / "classifier.h5"
    target = directory / "classifier.tflite"
    model = tf.keras.Sequential(
        [
            tf.keras.Input((128, 128, 3), name="pixels"),
            tf.keras.layers.Rescaling(1 / 255.0),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.save(source)
    convert_classifier(source, target, precision="fp16")
    inspection = inspect_tflite(target)
    yield target, inspection
    tf.keras.backend.clear_session()


def _source_files(tmp_path: Path) -> tuple[Path, Path]:
    detector = tmp_path / "detector.pt"
    classifier = tmp_path / "classifier.h5"
    detector.write_bytes(b"source detector")
    classifier.write_bytes(b"source classifier")
    return detector, classifier


def _convert_arguments(
    detector: Path, classifier: Path, output: Path, **overrides: object
) -> list[str]:
    values: dict[str, object] = {
        "detector": detector,
        "classifier": classifier,
        "output": output,
        "batch_id": BATCH_ID,
        "app_release_id": f"{BATCH_ID}-android",
        "model_generation": 2,
        "dataset_generation": 1,
        "version_code": 1,
        "version_name": "1.0.0",
    }
    values.update(overrides)
    arguments = ["convert"]
    for name, value in values.items():
        arguments.extend((f"--{name.replace('_', '-')}", str(value)))
    return arguments


def _fake_conversion(
    detector: Path,
    classifier: Path,
    output_dir: Path,
    *,
    precision: str,
) -> dict[str, object]:
    assert detector.name == "detector.pt"
    assert classifier.name == "classifier.h5"
    assert precision == "fp16"
    (output_dir / "detector.tflite").write_bytes(b"converted detector")
    (output_dir / "classifier.tflite").write_bytes(b"converted classifier")
    return _conversion_specs()


def _compare_paths(tmp_path: Path) -> dict[str, Path]:
    paths = {
        "old_detector": tmp_path / "old detector.pt",
        "old_classifier": tmp_path / "old classifier.h5",
        "new_detector": tmp_path / "new detector.pt",
        "new_classifier": tmp_path / "new classifier.h5",
        "mobile_dir": tmp_path / "mobile bundle",
        "images": tmp_path / "sample images",
        "report": tmp_path / "reports" / "parity report.json",
    }
    for name in (
        "old_detector",
        "old_classifier",
        "new_detector",
        "new_classifier",
    ):
        paths[name].write_bytes(name.encode("ascii"))
    paths["mobile_dir"].mkdir()
    (paths["mobile_dir"] / "detector.tflite").write_bytes(b"detector")
    (paths["mobile_dir"] / "classifier.tflite").write_bytes(b"classifier")
    manifest = _manifest(
        paths["new_detector"],
        paths["new_classifier"],
        paths["mobile_dir"] / "detector.tflite",
        paths["mobile_dir"] / "classifier.tflite",
    )
    (paths["mobile_dir"] / "model-manifest.json").write_text(
        manifest.to_json(), encoding="utf-8"
    )
    paths["images"].mkdir()
    (paths["images"] / "image 01.JPG").write_bytes(b"not decoded by seam")
    return paths


def _compare_arguments(paths: dict[str, Path]) -> list[str]:
    arguments = ["compare"]
    for name in (
        "old_detector",
        "old_classifier",
        "new_detector",
        "new_classifier",
        "mobile_dir",
        "images",
        "report",
    ):
        arguments.extend((f"--{name.replace('_', '-')}", str(paths[name])))
    return arguments


def _comparison_payload(
    *, hard_fail: bool = False, confirm: bool = False
) -> dict[str, object]:
    return {
        "report": {"conversion_passed": not hard_fail},
        "decision": {
            "hard_fail": hard_fail,
            "can_override": confirm,
            "requires_confirmation": confirm,
            "behavior_warning": confirm,
            "overridden": False,
            "operator_reason": None,
            "reasons": ["reason"] if hard_fail or confirm else [],
        },
    }


def test_cli_help_returns_success_without_leaking_system_exit(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = importlib.import_module("water_models.cli")

    assert cli.main(["--help"]) == 0
    assert "usage:" in capsys.readouterr().out


@pytest.mark.parametrize("arguments", ([], ["not-a-command"]))
def test_cli_usage_errors_return_two_without_leaking_system_exit(
    arguments: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    cli = importlib.import_module("water_models.cli")

    assert cli.main(arguments) == 2
    assert "usage:" in capsys.readouterr().err


def test_python_module_cli_help_smoke() -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT_ROOT / "src")

    result = subprocess.run(
        [sys.executable, "-m", "water_models.cli", "--help"],
        capture_output=True,
        check=False,
        env=environment,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_inspect_prints_strict_json_and_writes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = importlib.import_module("water_models.cli")
    detector, classifier = _source_files(tmp_path)
    expected = {
        "detector": {"task": "detect", "names": {"0": "lib"}},
        "classifier": {
            "input": {
                "name": "input",
                "shape": [None, 128, 128, 3],
                "dtype": "float32",
            },
            "output": {"name": "output", "shape": [None, 1], "dtype": "float32"},
        },
    }
    monkeypatch.setattr(cli, "inspect_source_models", lambda *_: expected)
    before = set(tmp_path.iterdir())

    code = cli.main(
        ["inspect", "--detector", str(detector), "--classifier", str(classifier)]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert json.loads(captured.out) == expected
    assert captured.err == ""
    assert set(tmp_path.iterdir()) == before


def test_convert_command_writes_complete_atomic_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    compatible_inspector: None,
) -> None:
    cli = importlib.import_module("water_models.cli")
    detector, classifier = _source_files(tmp_path)
    output = tmp_path / "generated output" / "bundle"
    monkeypatch.setattr(cli, "convert_models", _fake_conversion)

    code = cli.main(_convert_arguments(detector, classifier, output))

    assert code == 0
    assert {path.name for path in output.iterdir()} == {
        "detector.tflite",
        "classifier.tflite",
        "model-manifest.json",
    }
    manifest = json.loads((output / "model-manifest.json").read_text("utf-8"))
    assert manifest["release_batch_id"] == BATCH_ID
    assert manifest["app_release_id"] == f"{BATCH_ID}-android"
    assert manifest["conversion"]["precision"] == "fp16"
    assert not list(output.parent.glob(f".{output.name}.*.tmp"))
    assert json.loads(capsys.readouterr().out)["output"] == str(output)


def test_convert_validates_release_id_before_conversion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = importlib.import_module("water_models.cli")
    detector, classifier = _source_files(tmp_path)
    called = False

    def fake(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal called
        called = True
        return _conversion_specs()

    monkeypatch.setattr(cli, "convert_models", fake)

    code = cli.main(
        _convert_arguments(
            detector,
            classifier,
            tmp_path / "bundle",
            app_release_id="wrong-android",
        )
    )

    assert code == 1
    assert not called
    assert "app_release_id" in capsys.readouterr().err


@pytest.mark.parametrize("existing_kind", ("directory", "file", "symlink"))
def test_convert_rejects_any_existing_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    existing_kind: str,
) -> None:
    cli = importlib.import_module("water_models.cli")
    detector, classifier = _source_files(tmp_path)
    output = tmp_path / "bundle"
    if existing_kind == "directory":
        output.mkdir()
    elif existing_kind == "file":
        output.write_text("existing", encoding="utf-8")
    else:
        target = tmp_path / "elsewhere"
        target.mkdir()
        try:
            output.symlink_to(target, target_is_directory=True)
        except OSError as error:
            pytest.skip(f"symlinks unavailable: {error}")
    monkeypatch.setattr(cli, "convert_models", _fake_conversion)

    assert cli.main(_convert_arguments(detector, classifier, output)) == 1


def test_convert_rejects_output_below_a_linked_ancestor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = importlib.import_module("water_models.cli")
    detector, classifier = _source_files(tmp_path)
    real_parent = tmp_path / "real parent"
    (real_parent / "child").mkdir(parents=True)
    linked_parent = tmp_path / "linked parent"
    try:
        linked_parent.symlink_to(real_parent, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"symlinks unavailable: {error}")
    called = False

    def fake(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal called
        called = True
        return _conversion_specs()

    monkeypatch.setattr(cli, "convert_models", fake)

    assert (
        cli.main(
            _convert_arguments(detector, classifier, linked_parent / "child" / "bundle")
        )
        == 1
    )
    assert not called
    assert not (real_parent / "child" / "bundle").exists()


@pytest.mark.parametrize("mode", ("conversion", "manifest", "source-change"))
def test_convert_failure_cleans_owned_staging_and_never_publishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    cli = importlib.import_module("water_models.cli")
    detector, classifier = _source_files(tmp_path)
    output = tmp_path / "bundle"

    def fake(
        detector_path: Path,
        classifier_path: Path,
        output_dir: Path,
        *,
        precision: str,
    ) -> dict[str, object]:
        (output_dir / "detector.tflite").write_bytes(b"detector")
        if mode == "conversion":
            raise RuntimeError("conversion exploded")
        (output_dir / "classifier.tflite").write_bytes(b"classifier")
        if mode == "source-change":
            detector_path.write_bytes(b"changed")
        specs = _conversion_specs()
        if mode == "manifest":
            specs["class_names"] = ["wrong"]
        return specs

    monkeypatch.setattr(cli, "convert_models", fake)

    assert cli.main(_convert_arguments(detector, classifier, output)) == 1
    assert not output.exists()
    assert not list(tmp_path.glob(f".{output.name}.*.tmp"))


def test_convert_failure_cleans_absolute_staging_for_relative_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = importlib.import_module("water_models.cli")
    detector, classifier = _source_files(tmp_path)
    original_mkdtemp = cli.tempfile.mkdtemp

    def absolute_mkdtemp(*args: object, **kwargs: object) -> str:
        return str(Path(original_mkdtemp(*args, **kwargs)).resolve())

    def fail_conversion(*args: object, **kwargs: object) -> dict[str, object]:
        del args, kwargs
        raise OSError("primary conversion failure")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli.tempfile, "mkdtemp", absolute_mkdtemp)
    monkeypatch.setattr(cli, "convert_models", fail_conversion)

    assert cli.main(_convert_arguments(detector, classifier, Path("bundle"))) == 1
    assert "primary conversion failure" in capsys.readouterr().err
    assert not Path("bundle").exists()
    assert not list(Path.cwd().glob(".bundle.*.tmp"))


def test_convert_rename_race_preserves_winner_and_cleans_only_own_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    compatible_inspector: None,
) -> None:
    cli = importlib.import_module("water_models.cli")
    detector, classifier = _source_files(tmp_path)
    output = tmp_path / "bundle"

    def fake(
        detector_path: Path,
        classifier_path: Path,
        output_dir: Path,
        *,
        precision: str,
    ) -> dict[str, object]:
        del detector_path, classifier_path, precision
        (output_dir / "detector.tflite").write_bytes(b"loser detector")
        (output_dir / "classifier.tflite").write_bytes(b"loser classifier")
        output.mkdir()
        (output / "winner.txt").write_text("winner", encoding="utf-8")
        return _conversion_specs()

    monkeypatch.setattr(cli, "convert_models", fake)

    assert cli.main(_convert_arguments(detector, classifier, output)) == 1
    assert (output / "winner.txt").read_text("utf-8") == "winner"
    assert not list(tmp_path.glob(f".{output.name}.*.tmp"))


def test_convert_cleanup_error_does_not_replace_primary_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = importlib.import_module("water_models.cli")
    detector, classifier = _source_files(tmp_path)
    output = tmp_path / "bundle"
    original_cleanup = cli._remove_owned_staging

    def fail_conversion(*args: object, **kwargs: object) -> dict[str, object]:
        del args, kwargs
        raise RuntimeError("primary conversion failure")

    def fail_cleanup(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise cli.CliError("secondary cleanup failure")

    monkeypatch.setattr(cli, "convert_models", fail_conversion)
    monkeypatch.setattr(cli, "_remove_owned_staging", fail_cleanup)

    assert cli.main(_convert_arguments(detector, classifier, output)) == 1
    error = capsys.readouterr().err
    assert "primary conversion failure" in error
    assert "note: staging cleanup also failed:" in error
    assert "secondary cleanup failure" in error
    for staging in tmp_path.glob(f".{output.name}.*.tmp"):
        original_cleanup(staging, tmp_path, f".{output.name}.")


def test_convert_uses_unique_staging_directories_for_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = importlib.import_module("water_models.cli")
    detector, classifier = _source_files(tmp_path)
    seen: list[Path] = []

    def fail(
        detector_path: Path,
        classifier_path: Path,
        output_dir: Path,
        *,
        precision: str,
    ) -> dict[str, object]:
        del detector_path, classifier_path, precision
        seen.append(output_dir)
        raise RuntimeError("stop after observing staging")

    monkeypatch.setattr(cli, "convert_models", fail)

    assert cli.main(_convert_arguments(detector, classifier, tmp_path / "one")) == 1
    assert cli.main(_convert_arguments(detector, classifier, tmp_path / "two")) == 1
    assert len(seen) == 2
    assert seen[0] != seen[1]
    assert all(not path.exists() for path in seen)


@pytest.mark.parametrize(
    ("hard_fail", "confirm", "expected_code"),
    ((False, False, 0), (False, True, 2), (True, False, 3)),
)
def test_compare_writes_atomic_report_and_returns_policy_exit_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hard_fail: bool,
    confirm: bool,
    expected_code: int,
    compatible_inspector: None,
) -> None:
    cli = importlib.import_module("water_models.cli")
    paths = _compare_paths(tmp_path)
    payload = _comparison_payload(hard_fail=hard_fail, confirm=confirm)
    monkeypatch.setattr(cli, "compare_models", lambda **_: payload)

    code = cli.main(_compare_arguments(paths))

    assert code == expected_code
    written = json.loads(paths["report"].read_text("utf-8"))
    assert written["report"] == payload["report"]
    assert written["decision"] == payload["decision"]
    assert written["bundle"]["app_release_id"] == f"{BATCH_ID}-android"
    assert written["bundle"]["detector_sha256"] == _sha256(
        paths["mobile_dir"] / "detector.tflite"
    )
    assert not list(paths["report"].parent.glob(f".{paths['report'].name}.*.tmp"))


def test_compare_runtime_error_returns_one_without_partial_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    compatible_inspector: None,
) -> None:
    cli = importlib.import_module("water_models.cli")
    paths = _compare_paths(tmp_path)
    paths["report"].parent.mkdir()
    paths["report"].write_text("old report", encoding="utf-8")

    def fail(**kwargs: object) -> dict[str, object]:
        del kwargs
        raise RuntimeError("inference failed")

    monkeypatch.setattr(cli, "compare_models", fail)

    assert cli.main(_compare_arguments(paths)) == 1
    assert paths["report"].read_text("utf-8") == "old report"


def test_compare_rejects_non_json_number_without_replacing_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    compatible_inspector: None,
) -> None:
    cli = importlib.import_module("water_models.cli")
    paths = _compare_paths(tmp_path)
    paths["report"].parent.mkdir()
    paths["report"].write_text("old report", encoding="utf-8")
    payload = _comparison_payload()
    payload["report"] = {"bad": float("nan")}
    monkeypatch.setattr(cli, "compare_models", lambda **_: payload)

    assert cli.main(_compare_arguments(paths)) == 1
    assert paths["report"].read_text("utf-8") == "old report"


def test_compare_rejects_empty_images_before_loading_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = importlib.import_module("water_models.cli")
    paths = _compare_paths(tmp_path)
    (paths["images"] / "image 01.JPG").unlink()
    called = False

    def fake(**kwargs: object) -> dict[str, object]:
        del kwargs
        nonlocal called
        called = True
        return _comparison_payload()

    monkeypatch.setattr(cli, "compare_models", fake)

    assert cli.main(_compare_arguments(paths)) == 1
    assert not called
    assert not paths["report"].exists()


def test_compare_rejects_report_aliasing_an_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    compatible_inspector: None,
) -> None:
    cli = importlib.import_module("water_models.cli")
    paths = _compare_paths(tmp_path)
    paths["report"] = paths["new_detector"]
    called = False

    def fake(**kwargs: object) -> dict[str, object]:
        del kwargs
        nonlocal called
        called = True
        return _comparison_payload()

    monkeypatch.setattr(cli, "compare_models", fake)

    assert cli.main(_compare_arguments(paths)) == 1
    assert not called


@pytest.mark.parametrize("interrupt", (KeyboardInterrupt, SystemExit))
def test_compare_does_not_swallow_process_interrupts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupt: type[BaseException],
    compatible_inspector: None,
) -> None:
    cli = importlib.import_module("water_models.cli")
    paths = _compare_paths(tmp_path)

    def stop(**kwargs: object) -> dict[str, object]:
        del kwargs
        raise interrupt()

    monkeypatch.setattr(cli, "compare_models", stop)

    with pytest.raises(interrupt):
        cli.main(_compare_arguments(paths))


def test_commands_preserve_windows_compatible_unicode_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    compatible_inspector: None,
) -> None:
    cli = importlib.import_module("water_models.cli")
    directory = tmp_path / "模型 文件"
    directory.mkdir()
    detector, classifier = _source_files(directory)
    output = directory / "输出 bundle"
    seen: list[Path] = []

    def fake(
        detector_path: Path,
        classifier_path: Path,
        output_dir: Path,
        *,
        precision: str,
    ) -> dict[str, object]:
        del classifier_path, precision
        seen.append(detector_path)
        (output_dir / "detector.tflite").write_bytes(b"detector")
        (output_dir / "classifier.tflite").write_bytes(b"classifier")
        return _conversion_specs()

    monkeypatch.setattr(cli, "convert_models", fake)

    assert cli.main(_convert_arguments(detector, classifier, output)) == 0
    assert seen == [detector]


def test_convert_rejects_uninspectable_fake_models_before_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = importlib.import_module("water_models.cli")
    detector, classifier = _source_files(tmp_path)
    output = tmp_path / "bundle"
    monkeypatch.setattr(cli, "convert_models", _fake_conversion)

    def reject(path: Path) -> object:
        raise RuntimeError(f"invalid TFLite: {path.name}")

    monkeypatch.setattr(cli, "inspect_tflite", reject, raising=False)

    assert cli.main(_convert_arguments(detector, classifier, output)) == 1
    assert "TFLite inspection failed" in capsys.readouterr().err
    assert not output.exists()
    assert not list(tmp_path.glob(f".{output.name}.*.tmp"))


def test_convert_rejects_model_changed_by_final_inspection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = importlib.import_module("water_models.cli")
    detector, classifier = _source_files(tmp_path)
    output = tmp_path / "bundle"
    monkeypatch.setattr(cli, "convert_models", _fake_conversion)

    def mutate(path: Path) -> object:
        if path.name == "detector.tflite":
            path.write_bytes(path.read_bytes() + b"changed")
        return _inspection(detector=path.name == "detector.tflite")

    monkeypatch.setattr(cli, "inspect_tflite", mutate, raising=False)

    assert cli.main(_convert_arguments(detector, classifier, output)) == 1
    assert not output.exists()
    assert not list(tmp_path.glob(f".{output.name}.*.tmp"))


def test_cli_verifier_accepts_real_classifier_dynamic_batch_signature(
    real_classifier_artifact: tuple[Path, object],
) -> None:
    cli = importlib.import_module("water_models.cli")
    path, inspection = real_classifier_artifact
    assert inspection.input.shape == [1, 128, 128, 3]
    assert inspection.input.shape_signature == [-1, 128, 128, 3]
    assert inspection.output.shape == [1, 1]
    assert inspection.output.shape_signature == [-1, 1]
    input_spec = TensorSpec(
        inspection.input.name,
        tuple(inspection.input.shape),
        inspection.input.dtype,
    )
    output_spec = TensorSpec(
        inspection.output.name,
        tuple(inspection.output.shape),
        inspection.output.dtype,
    )

    cli._inspect_stable_tflite(
        path,
        input_spec,
        output_spec,
        _sha256(path),
        context="real classifier",
        model_kind="classifier",
    )


def test_compare_requires_manifest_before_calling_seam(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = importlib.import_module("water_models.cli")
    paths = _compare_paths(tmp_path)
    (paths["mobile_dir"] / "model-manifest.json").unlink()
    called = False

    def fake(**kwargs: object) -> dict[str, object]:
        del kwargs
        nonlocal called
        called = True
        return _comparison_payload()

    monkeypatch.setattr(cli, "compare_models", fake)

    assert cli.main(_compare_arguments(paths)) == 1
    assert not called
    assert not paths["report"].exists()


@pytest.mark.parametrize(
    "changed",
    ("mobile_detector", "mobile_classifier", "new_detector", "new_classifier"),
)
def test_compare_rejects_bundle_or_new_source_hash_mismatch_before_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed: str,
    compatible_inspector: None,
) -> None:
    cli = importlib.import_module("water_models.cli")
    paths = _compare_paths(tmp_path)
    target = {
        "mobile_detector": paths["mobile_dir"] / "detector.tflite",
        "mobile_classifier": paths["mobile_dir"] / "classifier.tflite",
        "new_detector": paths["new_detector"],
        "new_classifier": paths["new_classifier"],
    }[changed]
    target.write_bytes(target.read_bytes() + b"tampered")
    called = False

    def fake(**kwargs: object) -> dict[str, object]:
        del kwargs
        nonlocal called
        called = True
        return _comparison_payload()

    monkeypatch.setattr(cli, "compare_models", fake)

    assert cli.main(_compare_arguments(paths)) == 1
    assert not called
    assert not paths["report"].exists()


@pytest.mark.parametrize(
    "problem",
    (
        "name",
        "shape",
        "dtype",
        "detector_dynamic_batch",
        "classifier_spatial_dynamic",
        "classifier_output_dynamic",
        "classifier_rank",
    ),
)
def test_compare_rejects_mobile_tensor_contract_mismatch_before_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    problem: str,
) -> None:
    cli = importlib.import_module("water_models.cli")
    paths = _compare_paths(tmp_path)

    def inspect(path: Path) -> object:
        detector = path.name == "detector.tflite"
        result = _inspection(detector=detector)
        if problem == "name":
            result.input.name = "wrong"
        elif problem == "shape":
            result.output.shape[-1] += 1
            result.output.shape_signature[-1] += 1
        elif problem == "dtype":
            result.input.dtype = "float16"
        elif problem == "detector_dynamic_batch" and detector:
            result.input.shape_signature[0] = -1
        elif problem == "classifier_spatial_dynamic" and not detector:
            result.input.shape_signature = [-1, 128, -1, 3]
        elif problem == "classifier_output_dynamic" and not detector:
            result.output.shape_signature = [-1, -1]
        elif problem == "classifier_rank" and not detector:
            result.input.shape_signature = [-1, 128, 128]
        return result

    monkeypatch.setattr(cli, "inspect_tflite", inspect, raising=False)
    called = False

    def fake(**kwargs: object) -> dict[str, object]:
        del kwargs
        nonlocal called
        called = True
        return _comparison_payload()

    monkeypatch.setattr(cli, "compare_models", fake)

    assert cli.main(_compare_arguments(paths)) == 1
    assert not called
    assert not paths["report"].exists()


def test_compare_passes_verified_manifest_to_seam_and_reports_bundle_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    compatible_inspector: None,
) -> None:
    cli = importlib.import_module("water_models.cli")
    paths = _compare_paths(tmp_path)
    manifest = _manifest(
        paths["new_detector"],
        paths["new_classifier"],
        paths["mobile_dir"] / "detector.tflite",
        paths["mobile_dir"] / "classifier.tflite",
        detector_confidence=0.61,
        classifier_threshold=0.72,
        nms_iou=0.83,
    )
    (paths["mobile_dir"] / "model-manifest.json").write_text(
        manifest.to_json(), encoding="utf-8"
    )
    seen: list[ModelManifest] = []

    def fake(**kwargs: object) -> dict[str, object]:
        seen.append(kwargs["manifest"])
        return _comparison_payload()

    monkeypatch.setattr(cli, "compare_models", fake)

    assert cli.main(_compare_arguments(paths)) == 0
    assert seen == [manifest]
    written = json.loads(paths["report"].read_text("utf-8"))
    assert written["bundle"] == {
        "app_release_id": manifest.app_release_id,
        "batch_id": manifest.release_batch_id,
        "version_code": manifest.app_version_code,
        "version_name": manifest.app_version_name,
        "model_generation": manifest.model_generation,
        "dataset_generation": manifest.dataset_generation,
        "detector_sha256": manifest.detector_sha256,
        "classifier_sha256": manifest.classifier_sha256,
        "detector_source_sha256": manifest.detector_source_sha256,
        "classifier_source_sha256": manifest.classifier_source_sha256,
        "detector_confidence": 0.61,
        "classifier_threshold": 0.72,
        "nms_iou": 0.83,
    }


def test_crop_uses_legacy_desktop_integer_truncation() -> None:
    cli = importlib.import_module("water_models.cli")
    image = np.arange(4 * 4 * 3, dtype=np.uint8).reshape((4, 4, 3))
    detection = Detection(0.2, 0.2, 1.2, 1.2, 0.9)

    crop = cli._crop_for_detection(image, detection)

    assert crop.shape == (1, 1, 3)
    assert np.array_equal(crop, image[0:1, 0:1])


def test_compare_models_drives_source_and_mobile_with_manifest_thresholds() -> None:
    cli = importlib.import_module("water_models.cli")
    detector_source = Path("new.pt")
    classifier_source = Path("new.h5")
    manifest = ModelManifest(
        schema_version=1,
        release_batch_id=BATCH_ID,
        app_release_id=f"{BATCH_ID}-android",
        app_version_code=1,
        app_version_name="1.0.0",
        model_generation=1,
        dataset_generation=1,
        detector_sha256="1" * 64,
        classifier_sha256="2" * 64,
        detector_source_sha256="3" * 64,
        classifier_source_sha256="4" * 64,
        detector_input=TensorSpec("images", (1, 640, 640, 3), "float32"),
        detector_output=TensorSpec("output0", (1, 5, 8400), "float32"),
        classifier_input=TensorSpec(
            "serving_default_input", (1, 128, 128, 3), "float32"
        ),
        classifier_output=TensorSpec("StatefulPartitionedCall", (1, 1), "float32"),
        detector_confidence=0.61,
        classifier_threshold=0.72,
        nms_iou=0.83,
        class_names=("lib",),
        conversion={
            "precision": "fp16",
            "ultralytics": "8.4.25",
            "tensorflow": "2.21.0",
        },
    )
    detector_calls: list[tuple[str, dict[str, object]]] = []
    allocated: list[str] = []

    class Interpreter:
        def __init__(self, path: Path) -> None:
            self.path = path

        def allocate_tensors(self) -> None:
            allocated.append(self.path.name)

    def detector_loader(path: Path) -> str:
        return path.name

    def classifier_loader(path: Path) -> str:
        return path.name

    def detect_source(model: str, image: np.ndarray, **kwargs: object) -> list[object]:
        del image
        detector_calls.append((f"source:{model}", kwargs))
        return []

    def detect_mobile(
        interpreter: Interpreter, image: np.ndarray, **kwargs: object
    ) -> list[object]:
        del image
        detector_calls.append((f"mobile:{interpreter.path.name}", kwargs))
        return []

    payload = cli.compare_models(
        old_detector=Path("old.pt"),
        old_classifier=Path("old.h5"),
        new_detector=detector_source,
        new_classifier=classifier_source,
        mobile_dir=Path("bundle"),
        images=(Path("sample.jpg"),),
        manifest=manifest,
        detector_loader=detector_loader,
        classifier_loader=classifier_loader,
        interpreter_loader=Interpreter,
        image_loader=lambda path: np.zeros((2, 2, 3), dtype=np.uint8),
        color_converter=lambda image: image,
        detect_source_fn=detect_source,
        detect_tflite_fn=detect_mobile,
        classify_source_fn=lambda model, crop: 0.0,
        classify_tflite_fn=lambda model, crop: 0.0,
        clear_session=lambda: None,
    )

    common_options = {
        "conf": 0.61,
        "nms_iou": 0.83,
        "max_candidates": 3_000,
        "max_detections": 300,
    }
    source_options = {**common_options, "configure_model_nms": True}
    assert detector_calls == [
        ("source:new.pt", source_options),
        ("mobile:detector.tflite", common_options),
        ("source:old.pt", source_options),
    ]
    assert allocated == ["detector.tflite", "classifier.tflite"]
    assert payload["report"]["conversion_passed"] is True


def test_project_declares_test_dependencies() -> None:
    metadata = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )

    assert metadata["project"]["optional-dependencies"]["test"] == [
        "pytest>=8.4,<9",
        "jsonschema>=4.25,<5",
    ]


def test_generated_model_outputs_are_ignored() -> None:
    lines = (PROJECT_ROOT.parent / ".gitignore").read_text("utf-8").splitlines()

    for expected in (
        "model-tools/.work/",
        "model-tools/.pytest_cache/",
        "model-contract/generated/",
        "*.tflite",
    ):
        assert expected in lines
    assert "model-contract/golden/" not in lines
