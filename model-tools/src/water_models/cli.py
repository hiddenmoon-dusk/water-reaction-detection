"""Command-line workflows for inspecting, converting, and comparing models."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
from typing import Any

from .contracts import ModelManifest, TensorSpec


_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp"})
_BUNDLE_FILES = frozenset(
    {"detector.tflite", "classifier.tflite", "model-manifest.json"}
)
_CONVERTED_FILES = frozenset({"detector.tflite", "classifier.tflite"})


class CliError(RuntimeError):
    """A command could not safely complete."""


@dataclass(frozen=True)
class _FileSnapshot:
    sha256: str
    device: int
    inode: int
    size: int
    modified_ns: int
    changed_ns: int


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="water-models",
        description="Validate and convert water detection models for Android releases.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    inspect_parser = commands.add_parser(
        "inspect", help="Inspect source detector and classifier contracts."
    )
    inspect_parser.add_argument("--detector", required=True, type=Path)
    inspect_parser.add_argument("--classifier", required=True, type=Path)

    convert_parser = commands.add_parser(
        "convert", help="Create a new, atomically published Android model bundle."
    )
    convert_parser.add_argument("--detector", required=True, type=Path)
    convert_parser.add_argument("--classifier", required=True, type=Path)
    convert_parser.add_argument("--output", required=True, type=Path)
    convert_parser.add_argument("--batch-id", required=True)
    convert_parser.add_argument("--app-release-id", required=True)
    convert_parser.add_argument("--model-generation", required=True, type=int)
    convert_parser.add_argument("--dataset-generation", required=True, type=int)
    convert_parser.add_argument("--version-code", required=True, type=int)
    convert_parser.add_argument("--version-name", required=True)
    convert_parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    convert_parser.add_argument("--detector-confidence", type=float, default=0.3)
    convert_parser.add_argument("--classifier-threshold", type=float, default=0.5)
    convert_parser.add_argument("--nms-iou", type=float, default=0.45)

    compare_parser = commands.add_parser(
        "compare", help="Compare source models with an Android model bundle."
    )
    compare_parser.add_argument("--old-detector", required=True, type=Path)
    compare_parser.add_argument("--old-classifier", required=True, type=Path)
    compare_parser.add_argument("--new-detector", required=True, type=Path)
    compare_parser.add_argument("--new-classifier", required=True, type=Path)
    compare_parser.add_argument("--mobile-dir", required=True, type=Path)
    compare_parser.add_argument("--images", required=True, type=Path)
    compare_parser.add_argument("--report", required=True, type=Path)
    return parser


def _path_exists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _is_link_or_reparse(path: Path) -> bool:
    details = path.lstat()
    attributes = int(getattr(details, "st_file_attributes", 0))
    reparse = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    return stat.S_ISLNK(details.st_mode) or bool(attributes & reparse)


def _normalized_path(path: Path) -> str:
    return os.path.normcase(os.fspath(path.resolve(strict=False)))


def _paths_alias(left: Path, right: Path) -> bool:
    if _normalized_path(left) == _normalized_path(right):
        return True
    if not _path_exists(left) or not _path_exists(right):
        return False
    try:
        return os.path.samefile(left, right)
    except OSError:
        return False


def _require_regular_file(path: Path, label: str) -> None:
    try:
        details = path.stat()
    except OSError as error:
        raise CliError(f"{label} is not a readable file: {path}") from error
    if not stat.S_ISREG(details.st_mode):
        raise CliError(f"{label} must be a regular file: {path}")


def _require_real_directory(path: Path, label: str) -> None:
    try:
        details = path.lstat()
    except OSError as error:
        raise CliError(f"{label} is not a readable directory: {path}") from error
    if _is_link_or_reparse(path) or not stat.S_ISDIR(details.st_mode):
        raise CliError(f"{label} must be a real directory: {path}")


def _require_unlinked_directory_chain(path: Path, label: str) -> None:
    absolute = Path(os.path.abspath(path))
    for directory in reversed((absolute, *absolute.parents)):
        try:
            if _is_link_or_reparse(directory):
                raise CliError(
                    f"{label} must not traverse a link or reparse point: {directory}"
                )
        except OSError as error:
            raise CliError(f"failed to inspect {label}: {directory}") from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise CliError(f"failed to hash file: {path}") from error
    return digest.hexdigest()


def _file_identity(path: Path, label: str) -> tuple[int, int, int, int, int]:
    _require_regular_file(path, label)
    try:
        details = path.stat()
    except OSError as error:
        raise CliError(f"failed to stat {label}: {path}") from error
    return (
        details.st_dev,
        details.st_ino,
        details.st_size,
        details.st_mtime_ns,
        details.st_ctime_ns,
    )


def _stable_file_snapshot(path: Path, label: str) -> _FileSnapshot:
    before = _file_identity(path, label)
    digest = _sha256(path)
    after = _file_identity(path, label)
    if before != after:
        raise CliError(f"{label} changed while it was being hashed: {path}")
    return _FileSnapshot(digest, *after)


def _snapshot(path: Path) -> _FileSnapshot:
    return _stable_file_snapshot(path, "model source")


def _assert_source_unchanged(path: Path, original: _FileSnapshot) -> None:
    if _snapshot(path) != original:
        raise CliError(f"model source changed during conversion: {path}")


def _installed_version(distribution: str) -> str:
    try:
        value = metadata.version(distribution)
    except metadata.PackageNotFoundError as error:
        raise CliError(
            f"required package metadata is unavailable: {distribution}"
        ) from error
    if not value.strip():
        raise CliError(f"installed package version is empty: {distribution}")
    return value


def _tensor_json(tensor: object, fallback_name: str) -> dict[str, object]:
    name = str(getattr(tensor, "name", fallback_name))
    dtype_value = getattr(tensor, "dtype", "float32")
    dtype = str(getattr(dtype_value, "name", dtype_value))
    raw_shape = getattr(tensor, "shape", None)
    if raw_shape is None:
        raise CliError(f"{fallback_name} tensor has no shape")
    try:
        shape = [None if item is None else int(item) for item in raw_shape]
    except (TypeError, ValueError) as error:
        raise CliError(f"{fallback_name} tensor shape is invalid") from error
    return {"name": name, "shape": shape, "dtype": dtype}


def inspect_source_models(detector: Path, classifier: Path) -> dict[str, object]:
    """Load and strictly describe source contracts without writing files.

    This is the injection seam used by CLI tests; production uses Ultralytics
    ``YOLO(str(path))`` and Keras ``load_model(..., compile=False)``.
    """

    from ultralytics import YOLO
    import tensorflow as tf

    detector_model: object | None = None
    classifier_model: object | None = None
    try:
        detector_model = YOLO(str(detector))
        classifier_model = tf.keras.models.load_model(classifier, compile=False)
        task = getattr(detector_model, "task", None)
        names = getattr(detector_model, "names", None)
        if (
            task != "detect"
            or not isinstance(names, Mapping)
            or dict(names) != {0: "lib"}
        ):
            raise CliError(
                "detector contract must be task 'detect' with names {0: 'lib'}"
            )
        if tuple(classifier_model.input_shape) != (None, 128, 128, 3):
            raise CliError("classifier input shape must be [null, 128, 128, 3]")
        if tuple(classifier_model.output_shape) != (None, 1):
            raise CliError("classifier output shape must be [null, 1]")
        inputs = list(classifier_model.inputs)
        outputs = list(classifier_model.outputs)
        if len(inputs) != 1 or len(outputs) != 1:
            raise CliError("classifier must expose exactly one input and output")
        return {
            "detector": {"task": task, "names": {"0": "lib"}},
            "classifier": {
                "input": _tensor_json(inputs[0], "classifier input"),
                "output": _tensor_json(outputs[0], "classifier output"),
            },
        }
    finally:
        detector_model = None
        classifier_model = None
        tf.keras.backend.clear_session()


def _to_tensor_spec(tensor: object) -> TensorSpec:
    return TensorSpec(
        name=str(getattr(tensor, "name")),
        shape=tuple(int(item) for item in getattr(tensor, "shape")),
        dtype=str(getattr(tensor, "dtype")),
    )


def convert_models(
    detector: Path,
    classifier: Path,
    output_dir: Path,
    *,
    precision: str,
) -> dict[str, object]:
    """Convert both models into ``output_dir`` and return manifest tensor specs.

    Callers own ``output_dir``. The returned mapping has five exact keys:
    four ``TensorSpec`` values and ``class_names`` equal to ``["lib"]``.
    """

    from .conversion import (
        convert_classifier,
        convert_detector,
        inspect_tflite as inspect_converted_tflite,
    )

    detector_spec = convert_detector(
        detector, output_dir / "detector.tflite", precision
    )
    convert_classifier(classifier, output_dir / "classifier.tflite", precision)
    classifier_spec = inspect_converted_tflite(output_dir / "classifier.tflite")
    return {
        "detector_input": _to_tensor_spec(detector_spec.input),
        "detector_output": _to_tensor_spec(detector_spec.output),
        "classifier_input": _to_tensor_spec(classifier_spec.input),
        "classifier_output": _to_tensor_spec(classifier_spec.output),
        "class_names": detector_spec.class_names,
    }


def inspect_tflite(model_path: Path) -> object:
    """Independently inspect a published/staged TFLite file.

    Kept as a CLI-level seam so transaction tests can supply complete tensor
    metadata without loading real models.
    """

    from .conversion import inspect_tflite as inspect_model

    return inspect_model(model_path)


def _prevalidate_manifest(arguments: argparse.Namespace) -> dict[str, str]:
    versions = {
        "precision": arguments.precision,
        "ultralytics": _installed_version("ultralytics"),
        "tensorflow": _installed_version("tensorflow"),
    }
    placeholder = "0" * 64
    ModelManifest(
        schema_version=1,
        release_batch_id=arguments.batch_id,
        app_release_id=arguments.app_release_id,
        app_version_code=arguments.version_code,
        app_version_name=arguments.version_name,
        model_generation=arguments.model_generation,
        dataset_generation=arguments.dataset_generation,
        detector_sha256=placeholder,
        classifier_sha256=placeholder,
        detector_source_sha256=placeholder,
        classifier_source_sha256=placeholder,
        detector_input=TensorSpec("input", (1, 640, 640, 3), "float32"),
        detector_output=TensorSpec("output", (1, 5, 1), "float32"),
        classifier_input=TensorSpec("input", (1, 128, 128, 3), "float32"),
        classifier_output=TensorSpec("output", (1, 1), "float32"),
        detector_confidence=arguments.detector_confidence,
        classifier_threshold=arguments.classifier_threshold,
        nms_iou=arguments.nms_iou,
        class_names=("lib",),
        conversion=versions,
    )
    return versions


def _validate_conversion_specs(value: object) -> dict[str, object]:
    if type(value) is not dict:
        raise CliError("convert_models must return a specification mapping")
    expected = {
        "detector_input",
        "detector_output",
        "classifier_input",
        "classifier_output",
        "class_names",
    }
    if set(value) != expected:
        raise CliError("convert_models returned incomplete or extra specifications")
    for key in expected - {"class_names"}:
        if type(value[key]) is not TensorSpec:
            raise CliError(f"convert_models {key} must be a TensorSpec")
    return value


def _directory_entries(directory: Path) -> frozenset[str]:
    try:
        return frozenset(path.name for path in directory.iterdir())
    except OSError as error:
        raise CliError(f"failed to inspect staging directory: {directory}") from error


def _validate_staged_models(staging: Path) -> None:
    if _directory_entries(staging) != _CONVERTED_FILES:
        raise CliError("conversion must create exactly both TFLite model files")
    for name in _CONVERTED_FILES:
        path = staging / name
        try:
            if _is_link_or_reparse(path):
                raise CliError(f"converted model must not be a link: {path}")
            details = path.lstat()
        except OSError as error:
            raise CliError(f"failed to inspect converted model: {path}") from error
        if not stat.S_ISREG(details.st_mode) or details.st_size <= 0:
            raise CliError(f"converted model must be a non-empty regular file: {path}")


def _sync_file(path: Path) -> None:
    try:
        with path.open("rb+") as stream:
            os.fsync(stream.fileno())
    except OSError as error:
        raise CliError(f"failed to sync file: {path}") from error


def _sync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_new_file(path: Path, data: bytes) -> None:
    try:
        with path.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as error:
        raise CliError(f"failed to write new file: {path}") from error


def _remove_owned_staging(staging: Path, parent: Path, prefix: str) -> None:
    if not _path_exists(staging):
        return
    if not staging.name.startswith(prefix):
        raise CliError(f"refusing to clean unowned staging path: {staging}")
    try:
        if _is_link_or_reparse(staging):
            raise CliError(f"refusing to follow changed staging path: {staging}")
        resolved_staging = staging.resolve(strict=True)
        resolved_parent = parent.resolve(strict=True)
        if resolved_staging.parent != resolved_parent:
            raise CliError(f"staging path escaped its parent: {staging}")
        shutil.rmtree(staging)
    except CliError:
        raise
    except OSError as error:
        raise CliError(f"failed to clean staging directory: {staging}") from error


def _prepare_output(output: Path, sources: Sequence[Path]) -> None:
    if _path_exists(output):
        raise CliError(f"output must not already exist: {output}")
    if any(_paths_alias(output, source) for source in sources):
        raise CliError("output must not alias a model source")
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise CliError(f"failed to create output parent: {output.parent}") from error
    _require_real_directory(output.parent, "output parent")
    _require_unlinked_directory_chain(output.parent, "output parent")


def _create_staging(output: Path) -> tuple[Path, str]:
    prefix = f".{output.name}."
    try:
        name = tempfile.mkdtemp(prefix=prefix, suffix=".tmp", dir=output.parent)
    except OSError as error:
        raise CliError(
            f"failed to create output staging directory: {output.parent}"
        ) from error
    staging = Path(name)
    _require_real_directory(staging, "output staging")
    return staging, prefix


def _build_manifest(
    arguments: argparse.Namespace,
    specs: dict[str, object],
    versions: dict[str, str],
    detector_source: _FileSnapshot,
    classifier_source: _FileSnapshot,
    detector_sha256: str,
    classifier_sha256: str,
) -> ModelManifest:
    return ModelManifest(
        schema_version=1,
        release_batch_id=arguments.batch_id,
        app_release_id=arguments.app_release_id,
        app_version_code=arguments.version_code,
        app_version_name=arguments.version_name,
        model_generation=arguments.model_generation,
        dataset_generation=arguments.dataset_generation,
        detector_sha256=detector_sha256,
        classifier_sha256=classifier_sha256,
        detector_source_sha256=detector_source.sha256,
        classifier_source_sha256=classifier_source.sha256,
        detector_input=specs["detector_input"],
        detector_output=specs["detector_output"],
        classifier_input=specs["classifier_input"],
        classifier_output=specs["classifier_output"],
        detector_confidence=arguments.detector_confidence,
        classifier_threshold=arguments.classifier_threshold,
        nms_iou=arguments.nms_iou,
        class_names=specs["class_names"],
        conversion=versions,
    )


def _validate_inspected_tensor(
    inspected: object,
    expected: TensorSpec,
    *,
    model_kind: str,
    tensor_role: str,
    context: str,
) -> None:
    if model_kind not in {"detector", "classifier"}:
        raise ValueError("model_kind must be 'detector' or 'classifier'")
    if tensor_role not in {"input", "output"}:
        raise ValueError("tensor_role must be 'input' or 'output'")
    label = f"{context} {tensor_role}"
    try:
        name = inspected.name
        shape = tuple(int(item) for item in inspected.shape)
        signature = tuple(int(item) for item in inspected.shape_signature)
        dtype = inspected.dtype
    except Exception as error:
        raise CliError(f"{label} inspection metadata is incomplete") from error
    if name != expected.name:
        raise CliError(f"{label} tensor name does not match manifest")
    if shape != expected.shape:
        raise CliError(f"{label} tensor shape does not match manifest")
    allowed_signatures = {expected.shape}
    if model_kind == "classifier":
        allowed_signatures.add((-1, *expected.shape[1:]))
    if signature not in allowed_signatures:
        if model_kind == "detector":
            raise CliError(f"{label} tensor shape signature must be static")
        raise CliError(f"{label} tensor shape signature may only have a dynamic batch")
    if dtype != expected.dtype:
        raise CliError(f"{label} tensor dtype does not match manifest")


def _inspect_stable_tflite(
    path: Path,
    input_spec: TensorSpec,
    output_spec: TensorSpec,
    expected_sha256: str,
    *,
    context: str,
    model_kind: str,
) -> None:
    before = _stable_file_snapshot(path, f"{context} TFLite model")
    if before.sha256 != expected_sha256:
        raise CliError(f"{context} TFLite hash does not match manifest: {path}")
    try:
        inspection = inspect_tflite(path)
    except Exception as error:
        raise CliError(f"{context} TFLite inspection failed: {path}") from error
    try:
        input_tensor = inspection.input
        output_tensor = inspection.output
    except Exception as error:
        raise CliError(f"{context} TFLite inspection is incomplete: {path}") from error
    _validate_inspected_tensor(
        input_tensor,
        input_spec,
        model_kind=model_kind,
        tensor_role="input",
        context=context,
    )
    _validate_inspected_tensor(
        output_tensor,
        output_spec,
        model_kind=model_kind,
        tensor_role="output",
        context=context,
    )
    after = _stable_file_snapshot(path, f"{context} TFLite model")
    if after != before:
        raise CliError(f"{context} TFLite changed during final inspection: {path}")


def _verify_staged_bundle(
    staging: Path,
    manifest: ModelManifest,
    detector_source: Path,
    classifier_source: Path,
    detector_snapshot: _FileSnapshot,
    classifier_snapshot: _FileSnapshot,
) -> None:
    if _directory_entries(staging) != _BUNDLE_FILES:
        raise CliError("bundle must contain exactly two models and one manifest")
    payload = (staging / "model-manifest.json").read_bytes()
    parsed = ModelManifest.from_json(payload)
    if parsed != manifest:
        raise CliError("written manifest did not round-trip exactly")
    _inspect_stable_tflite(
        staging / "detector.tflite",
        manifest.detector_input,
        manifest.detector_output,
        manifest.detector_sha256,
        context="final detector",
        model_kind="detector",
    )
    _inspect_stable_tflite(
        staging / "classifier.tflite",
        manifest.classifier_input,
        manifest.classifier_output,
        manifest.classifier_sha256,
        context="final classifier",
        model_kind="classifier",
    )
    _assert_source_unchanged(detector_source, detector_snapshot)
    _assert_source_unchanged(classifier_source, classifier_snapshot)


def _handle_inspect(arguments: argparse.Namespace) -> int:
    _require_regular_file(arguments.detector, "detector source")
    _require_regular_file(arguments.classifier, "classifier source")
    payload = inspect_source_models(arguments.detector, arguments.classifier)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


def _handle_convert(arguments: argparse.Namespace) -> int:
    versions = _prevalidate_manifest(arguments)
    detector = arguments.detector
    classifier = arguments.classifier
    output = arguments.output
    _require_regular_file(detector, "detector source")
    _require_regular_file(classifier, "classifier source")
    if _paths_alias(detector, classifier):
        raise CliError("detector and classifier sources must be distinct")
    _prepare_output(output, (detector, classifier))
    detector_snapshot = _snapshot(detector)
    classifier_snapshot = _snapshot(classifier)
    staging, prefix = _create_staging(output)
    published = False
    try:
        specs = _validate_conversion_specs(
            convert_models(
                detector,
                classifier,
                staging,
                precision=arguments.precision,
            )
        )
        _validate_staged_models(staging)
        _assert_source_unchanged(detector, detector_snapshot)
        _assert_source_unchanged(classifier, classifier_snapshot)
        detector_output = staging / "detector.tflite"
        classifier_output = staging / "classifier.tflite"
        _sync_file(detector_output)
        _sync_file(classifier_output)
        detector_hash = _stable_file_snapshot(
            detector_output, "converted detector"
        ).sha256
        classifier_hash = _stable_file_snapshot(
            classifier_output, "converted classifier"
        ).sha256
        manifest = _build_manifest(
            arguments,
            specs,
            versions,
            detector_snapshot,
            classifier_snapshot,
            detector_hash,
            classifier_hash,
        )
        _write_new_file(
            staging / "model-manifest.json", manifest.to_json().encode("utf-8")
        )
        _verify_staged_bundle(
            staging,
            manifest,
            detector,
            classifier,
            detector_snapshot,
            classifier_snapshot,
        )
        _sync_directory(staging)
        if _path_exists(output):
            raise CliError(f"output appeared during conversion: {output}")
        try:
            staging.replace(output)
        except OSError as error:
            raise CliError(f"failed to atomically publish bundle: {output}") from error
        published = True
        _sync_directory(output.parent)
    except BaseException as primary_error:
        try:
            _remove_owned_staging(staging, output.parent, prefix)
        except BaseException as cleanup_error:
            primary_error.add_note(f"staging cleanup also failed: {cleanup_error!r}")
        raise
    if not published:
        raise CliError("bundle was not published")
    print(
        json.dumps(
            {
                "output": str(output),
                "files": sorted(_BUNDLE_FILES),
                "manifest": manifest.to_dict(),
            },
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


def _list_images(directory: Path) -> tuple[Path, ...]:
    _require_real_directory(directory, "images directory")
    try:
        images = tuple(
            sorted(
                (
                    path
                    for path in directory.iterdir()
                    if path.suffix.lower() in _IMAGE_SUFFIXES
                ),
                key=lambda path: (path.name.casefold(), path.name),
            )
        )
    except OSError as error:
        raise CliError(f"failed to list images directory: {directory}") from error
    if not images:
        raise CliError("images directory contains no supported image files")
    for image in images:
        if _is_link_or_reparse(image):
            raise CliError(f"sample image must not be a link: {image}")
        _require_regular_file(image, "sample image")
    return images


def _path_is_within(path: Path, directory: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(directory.resolve(strict=True))
    except (OSError, ValueError):
        return False
    return True


def _validate_compare_paths(arguments: argparse.Namespace) -> tuple[Path, ...]:
    model_paths = (
        arguments.old_detector,
        arguments.old_classifier,
        arguments.new_detector,
        arguments.new_classifier,
    )
    for path in model_paths:
        _require_regular_file(path, "source model")
    _require_real_directory(arguments.mobile_dir, "mobile bundle")
    mobile_models = (
        arguments.mobile_dir / "detector.tflite",
        arguments.mobile_dir / "classifier.tflite",
        arguments.mobile_dir / "model-manifest.json",
    )
    for path in mobile_models:
        if _is_link_or_reparse(path):
            raise CliError(f"mobile model must not be a link: {path}")
        _require_regular_file(path, "mobile model")
    images = _list_images(arguments.images)
    report = arguments.report
    aliases = (
        *model_paths,
        *mobile_models,
        arguments.images,
        arguments.mobile_dir,
        *images,
    )
    if any(_paths_alias(report, path) for path in aliases):
        raise CliError("report must not alias any comparison input")
    if _path_is_within(report, arguments.images) or _path_is_within(
        report, arguments.mobile_dir
    ):
        raise CliError("report must not be written inside an input directory")
    if _path_exists(report) and _is_link_or_reparse(report):
        raise CliError("report must not be a link or reparse point")
    return images


def _read_verified_manifest(arguments: argparse.Namespace) -> ModelManifest:
    manifest_path = arguments.mobile_dir / "model-manifest.json"
    try:
        payload = manifest_path.read_bytes()
    except OSError as error:
        raise CliError(f"failed to read model manifest: {manifest_path}") from error
    try:
        manifest = ModelManifest.from_json(payload)
    except ValueError as error:
        raise CliError(f"invalid model manifest: {manifest_path}") from error

    detector_path = arguments.mobile_dir / "detector.tflite"
    classifier_path = arguments.mobile_dir / "classifier.tflite"
    _inspect_stable_tflite(
        detector_path,
        manifest.detector_input,
        manifest.detector_output,
        manifest.detector_sha256,
        context="mobile detector",
        model_kind="detector",
    )
    _inspect_stable_tflite(
        classifier_path,
        manifest.classifier_input,
        manifest.classifier_output,
        manifest.classifier_sha256,
        context="mobile classifier",
        model_kind="classifier",
    )
    if (
        _stable_file_snapshot(arguments.new_detector, "new detector source").sha256
        != manifest.detector_source_sha256
    ):
        raise CliError("new detector source hash does not match model manifest")
    if (
        _stable_file_snapshot(arguments.new_classifier, "new classifier source").sha256
        != manifest.classifier_source_sha256
    ):
        raise CliError("new classifier source hash does not match model manifest")
    return manifest


def _bundle_evidence(manifest: ModelManifest) -> dict[str, object]:
    return {
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
        "detector_confidence": manifest.detector_confidence,
        "classifier_threshold": manifest.classifier_threshold,
        "nms_iou": manifest.nms_iou,
    }


def _crop_for_detection(image: Any, detection: object) -> Any:
    """Crop with legacy desktop ``int`` truncation for cross-platform parity."""

    height, width = image.shape[:2]
    x1 = max(0, min(width, int(getattr(detection, "x1"))))
    y1 = max(0, min(height, int(getattr(detection, "y1"))))
    x2 = max(0, min(width, int(getattr(detection, "x2"))))
    y2 = max(0, min(height, int(getattr(detection, "y2"))))
    if x2 <= x1 or y2 <= y1:
        raise CliError("detector produced a degenerate classifier crop")
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        raise CliError("detector produced an empty classifier crop")
    return crop


def _classify_predictions(
    detections: Sequence[object],
    image: Any,
    classifier: Callable[[Any], float],
    threshold: float,
) -> tuple[list[object], list[float]]:
    from .parity import Prediction

    predictions: list[object] = []
    probabilities: list[float] = []
    for detection in detections:
        probability = classifier(_crop_for_detection(image, detection))
        label = "lib" if probability >= threshold else "not-lib"
        predictions.append(Prediction(detection, label))
        probabilities.append(probability)
    return predictions, probabilities


def _behavior_changes(
    old_predictions: Sequence[object], new_predictions: Sequence[object]
) -> tuple[int, int]:
    from .parity import match_detections

    denominator = max(len(old_predictions), len(new_predictions))
    equal_matches = sum(
        match.label_equal
        for match in match_detections(old_predictions, new_predictions)
    )
    return denominator - equal_matches, denominator


def compare_models(
    *,
    old_detector: Path,
    old_classifier: Path,
    new_detector: Path,
    new_classifier: Path,
    mobile_dir: Path,
    images: Sequence[Path],
    manifest: ModelManifest,
    detector_loader: Callable[[Path], object] | None = None,
    classifier_loader: Callable[[Path], object] | None = None,
    interpreter_loader: Callable[[Path], object] | None = None,
    image_loader: Callable[[Path], Any] | None = None,
    color_converter: Callable[[Any], Any] | None = None,
    detect_source_fn: Callable[..., Sequence[object]] | None = None,
    detect_tflite_fn: Callable[..., Sequence[object]] | None = None,
    classify_source_fn: Callable[[object, Any], float] | None = None,
    classify_tflite_fn: Callable[[object, Any], float] | None = None,
    clear_session: Callable[[], None] | None = None,
) -> dict[str, object]:
    """Run source/mobile parity and old/new behavior comparison.

    Old/new label-change denominator is the per-image maximum detection count.
    Every item not in an equal-label IoU-greedy match counts as one change.
    Detection and classification thresholds come only from ``manifest``.
    """

    from .inference import (
        DEFAULT_MAX_CANDIDATES,
        DEFAULT_MAX_DETECTIONS,
        classify_source as default_classify_source,
        classify_tflite as default_classify_tflite,
        detect_source as default_detect_source,
        detect_tflite as default_detect_tflite,
    )
    from .parity import compare_conversion, evaluate

    if type(manifest) is not ModelManifest:
        raise TypeError("manifest must be a ModelManifest")

    tensorflow_module: object | None = None
    if classifier_loader is None or interpreter_loader is None or clear_session is None:
        import tensorflow as tensorflow_module

    if detector_loader is None:
        from ultralytics import YOLO

        def default_detector_loader(path: Path) -> object:
            return YOLO(str(path))

        effective_detector_loader = default_detector_loader
    else:
        effective_detector_loader = detector_loader

    if classifier_loader is None:

        def default_classifier_loader(path: Path) -> object:
            return tensorflow_module.keras.models.load_model(path, compile=False)

        effective_classifier_loader = default_classifier_loader
    else:
        effective_classifier_loader = classifier_loader

    if interpreter_loader is None:

        def default_interpreter_loader(path: Path) -> object:
            return tensorflow_module.lite.Interpreter(model_path=str(path))

        effective_interpreter_loader = default_interpreter_loader
    else:
        effective_interpreter_loader = interpreter_loader

    if image_loader is None or color_converter is None:
        import cv2

    if image_loader is None:

        def default_image_loader(path: Path) -> Any:
            return cv2.imread(str(path), cv2.IMREAD_COLOR)

        effective_image_loader = default_image_loader
    else:
        effective_image_loader = image_loader

    if color_converter is None:

        def default_color_converter(image: Any) -> Any:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        effective_color_converter = default_color_converter
    else:
        effective_color_converter = color_converter

    effective_detect_source = detect_source_fn or default_detect_source
    effective_detect_tflite = detect_tflite_fn or default_detect_tflite
    effective_classify_source = classify_source_fn or default_classify_source
    effective_classify_tflite = classify_tflite_fn or default_classify_tflite
    if clear_session is None:

        def default_clear_session() -> None:
            tensorflow_module.keras.backend.clear_session()

        effective_clear_session = default_clear_session
    else:
        effective_clear_session = clear_session

    old_detector_model: object | None = None
    old_classifier_model: object | None = None
    new_detector_model: object | None = None
    new_classifier_model: object | None = None
    detector_interpreter: object | None = None
    classifier_interpreter: object | None = None
    try:
        old_detector_model = effective_detector_loader(old_detector)
        old_classifier_model = effective_classifier_loader(old_classifier)
        new_detector_model = effective_detector_loader(new_detector)
        new_classifier_model = effective_classifier_loader(new_classifier)
        detector_interpreter = effective_interpreter_loader(
            mobile_dir / "detector.tflite"
        )
        classifier_interpreter = effective_interpreter_loader(
            mobile_dir / "classifier.tflite"
        )
        detector_interpreter.allocate_tensors()
        classifier_interpreter.allocate_tensors()

        reference_detections: list[list[object]] = []
        mobile_detections: list[list[object]] = []
        reference_probabilities: list[list[float]] = []
        mobile_probabilities: list[list[float]] = []
        sample_ids: list[str] = []
        old_count = 0
        new_count = 0
        label_changes = 0
        label_denominator = 0

        for image_path in images:
            bgr = effective_image_loader(image_path)
            if bgr is None:
                raise CliError(f"failed to read sample image: {image_path}")
            try:
                rgb = effective_color_converter(bgr)
            except Exception as error:
                raise CliError(
                    f"failed to convert sample image to RGB: {image_path}"
                ) from error

            detector_options = {
                "conf": manifest.detector_confidence,
                "nms_iou": manifest.nms_iou,
                "max_candidates": DEFAULT_MAX_CANDIDATES,
                "max_detections": DEFAULT_MAX_DETECTIONS,
            }
            source_detector_options = {
                **detector_options,
                "configure_model_nms": True,
            }
            new_raw = effective_detect_source(
                new_detector_model, rgb, **source_detector_options
            )
            mobile_raw = effective_detect_tflite(
                detector_interpreter,
                rgb,
                **detector_options,
            )
            old_raw = effective_detect_source(
                old_detector_model, rgb, **source_detector_options
            )
            new_predictions, new_probabilities = _classify_predictions(
                new_raw,
                rgb,
                lambda crop: effective_classify_source(new_classifier_model, crop),
                manifest.classifier_threshold,
            )
            mobile_predictions, mobile_probs = _classify_predictions(
                mobile_raw,
                rgb,
                lambda crop: effective_classify_tflite(classifier_interpreter, crop),
                manifest.classifier_threshold,
            )
            old_predictions, _ = _classify_predictions(
                old_raw,
                rgb,
                lambda crop: effective_classify_source(old_classifier_model, crop),
                manifest.classifier_threshold,
            )
            changes, denominator = _behavior_changes(old_predictions, new_predictions)
            old_count += len(old_predictions)
            new_count += len(new_predictions)
            label_changes += changes
            label_denominator += denominator
            reference_detections.append(new_predictions)
            mobile_detections.append(mobile_predictions)
            reference_probabilities.append(new_probabilities)
            mobile_probabilities.append(mobile_probs)
            sample_ids.append(image_path.name)

        label_change_ratio = (
            label_changes / label_denominator if label_denominator else 0.0
        )
        report = compare_conversion(
            reference_detections,
            mobile_detections,
            reference_probabilities,
            mobile_probabilities,
            old_count=old_count,
            new_count=new_count,
            label_change_ratio=label_change_ratio,
            sample_ids=sample_ids,
        )
        decision = evaluate(report)
        return {"report": report.to_dict(), "decision": decision.to_dict()}
    finally:
        old_detector_model = None
        old_classifier_model = None
        new_detector_model = None
        new_classifier_model = None
        detector_interpreter = None
        classifier_interpreter = None
        effective_clear_session()


def _validate_comparison_payload(payload: object) -> dict[str, object]:
    if type(payload) is not dict or set(payload) != {"report", "decision"}:
        raise CliError("compare_models must return report and decision objects")
    if type(payload["report"]) is not dict or type(payload["decision"]) is not dict:
        raise CliError("comparison report and decision must be objects")
    decision = payload["decision"]
    for key in ("hard_fail", "requires_confirmation"):
        if type(decision.get(key)) is not bool:
            raise CliError(f"comparison decision {key} must be a bool")
    if decision["hard_fail"] and decision["requires_confirmation"]:
        raise CliError("hard failure cannot require confirmation")
    return payload


def _atomic_write_json(target: Path, payload: object) -> None:
    try:
        encoded = (
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise CliError("report contains a non-JSON value") from error
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise CliError(f"failed to create report directory: {target.parent}") from error
    _require_real_directory(target.parent, "report parent")
    _require_unlinked_directory_chain(target.parent, "report parent")
    temporary: Path | None = None
    descriptor: int | None = None
    try:
        descriptor, name = tempfile.mkstemp(
            dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
        )
        temporary = Path(name)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = None
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(target)
        temporary = None
        _sync_directory(target.parent)
    except BaseException as primary_error:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError as cleanup_error:
                primary_error.add_note(f"descriptor cleanup failed: {cleanup_error!r}")
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError as cleanup_error:
                primary_error.add_note(f"report cleanup failed: {cleanup_error!r}")
        raise


def _handle_compare(arguments: argparse.Namespace) -> int:
    image_paths = _validate_compare_paths(arguments)
    manifest = _read_verified_manifest(arguments)
    payload = _validate_comparison_payload(
        compare_models(
            old_detector=arguments.old_detector,
            old_classifier=arguments.old_classifier,
            new_detector=arguments.new_detector,
            new_classifier=arguments.new_classifier,
            mobile_dir=arguments.mobile_dir,
            images=image_paths,
            manifest=manifest,
        )
    )
    report_payload = {**payload, "bundle": _bundle_evidence(manifest)}
    _atomic_write_json(arguments.report, report_payload)
    decision = payload["decision"]
    if decision["hard_fail"]:
        code = 3
    elif decision["requires_confirmation"]:
        code = 2
    else:
        code = 0
    print(
        json.dumps(
            {"report": str(arguments.report), "exit_code": code, "decision": decision},
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return code


def main(argv: Sequence[str] | None = None) -> int:
    """Run the model tools CLI without leaking argparse ``SystemExit``."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()
    if not arguments:
        parser.print_usage(sys.stderr)
        return 2
    try:
        parsed = parser.parse_args(arguments)
    except SystemExit as error:
        return int(error.code)

    handlers: dict[str, Callable[[argparse.Namespace], int]] = {
        "inspect": _handle_inspect,
        "convert": _handle_convert,
        "compare": _handle_compare,
    }
    try:
        return handlers[parsed.command](parsed)
    except Exception as error:
        print(f"water-models: error: {error}", file=sys.stderr)
        for note in getattr(error, "__notes__", ()):
            print(f"water-models: note: {note}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
