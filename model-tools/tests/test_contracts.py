from __future__ import annotations

import json
from dataclasses import fields
import operator
from pathlib import Path
from types import MappingProxyType
from typing import Callable

import jsonschema
import pytest

from water_models import contracts
from water_models.contracts import ModelManifest, TensorSpec


SCHEMA_PATH = (
    Path(__file__).resolve().parents[2] / "model-contract" / "model-manifest.schema.json"
)


def valid_manifest_dict() -> dict[str, object]:
    batch_id = "0123456789abcdef0123456789abcdef"
    return {
        "schema_version": 1,
        "release_batch_id": batch_id,
        "app_release_id": f"{batch_id}-android",
        "app_version_code": 42,
        "app_version_name": "1.2.3",
        "model_generation": 7,
        "dataset_generation": 9,
        "detector_sha256": "a" * 64,
        "classifier_sha256": "b" * 64,
        "detector_source_sha256": "c" * 64,
        "classifier_source_sha256": "d" * 64,
        "detector_input": {
            "name": "images",
            "shape": [1, 640, 640, 3],
            "dtype": "float32",
        },
        "detector_output": {
            "name": "output0",
            "shape": [1, 5, 8400],
            "dtype": "float32",
        },
        "classifier_input": {
            "name": "image",
            "shape": [1, 128, 128, 3],
            "dtype": "float32",
        },
        "classifier_output": {
            "name": "score",
            "shape": [1, 1],
            "dtype": "float32",
        },
        "detector_confidence": 0.25,
        "classifier_threshold": 0.5,
        "nms_iou": 0.45,
        "class_names": ["lib"],
        "conversion": {
            "precision": "float32",
            "tensorflow": "2.21.0",
            "ultralytics": "8.4.25",
        },
    }


def test_manifest_round_trip_is_stable() -> None:
    source = valid_manifest_dict()

    manifest = ModelManifest.from_dict(source)
    encoded = manifest.to_json()

    assert manifest.to_dict() == source
    assert encoded == json.dumps(
        source,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    assert ModelManifest.from_json(encoded) == manifest
    assert ModelManifest.from_json(encoded.encode("utf-8")) == manifest


def test_manifest_uses_immutable_container_types() -> None:
    manifest = ModelManifest.from_dict(valid_manifest_dict())

    assert type(manifest.detector_input.shape) is tuple
    assert type(manifest.class_names) is tuple
    assert isinstance(manifest.conversion, MappingProxyType)


@pytest.mark.parametrize(
    ("mutate", "error_type"),
    (
        (lambda manifest: manifest.detector_input.shape.append(99), AttributeError),
        (lambda manifest: operator.setitem(manifest.detector_input.shape, 0, 99), TypeError),
        (lambda manifest: manifest.class_names.append("mutated"), AttributeError),
        (lambda manifest: operator.setitem(manifest.class_names, 0, "mutated"), TypeError),
        (
            lambda manifest: manifest.conversion.update({"precision": "mutated"}),
            AttributeError,
        ),
        (
            lambda manifest: operator.setitem(
                manifest.conversion, "precision", "mutated"
            ),
            TypeError,
        ),
    ),
)
def test_manifest_mutation_fails_without_changing_serialized_output(
    mutate: Callable[[ModelManifest], None], error_type: type[Exception]
) -> None:
    manifest = ModelManifest.from_dict(valid_manifest_dict())
    encoded = manifest.to_json()
    caught: Exception | None = None

    try:
        mutate(manifest)
    except (AttributeError, TypeError) as error:
        caught = error

    assert manifest.to_json() == encoded
    assert isinstance(caught, error_type)


def test_manifest_defensively_copies_input_containers() -> None:
    source = valid_manifest_dict()
    manifest = ModelManifest.from_dict(source)
    encoded = manifest.to_json()

    source["detector_input"]["shape"].append(99)  # type: ignore[index,union-attr]
    source["class_names"].append("mutated")  # type: ignore[union-attr]
    source["conversion"]["precision"] = "mutated"  # type: ignore[index]

    assert manifest.to_json() == encoded


@pytest.mark.parametrize(
    "shape",
    ([1, 5, 0], [1, 5], [1, 6, 8400], [1, 5, True]),
)
def test_manifest_rejects_invalid_detector_output(shape: list[object]) -> None:
    data = valid_manifest_dict()
    data["detector_output"] = {
        "name": "output0",
        "shape": shape,
        "dtype": "float32",
    }

    with pytest.raises(ValueError):
        ModelManifest.from_dict(data)


@pytest.mark.parametrize("change", ["missing", "extra"])
def test_manifest_rejects_missing_and_extra_fields(change: str) -> None:
    data = valid_manifest_dict()
    if change == "missing":
        del data["schema_version"]
    else:
        data["unexpected"] = "nope"

    with pytest.raises(ValueError, match=change):
        ModelManifest.from_dict(data)


@pytest.mark.parametrize(
    "field",
    ("schema_version", "app_version_code", "model_generation", "dataset_generation"),
)
def test_manifest_rejects_bool_integer_fields(field: str) -> None:
    data = valid_manifest_dict()
    data[field] = True

    with pytest.raises(ValueError):
        ModelManifest.from_dict(data)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("detector_sha256", "A" * 64),
        ("classifier_sha256", "a" * 63),
        ("detector_source_sha256", "g" * 64),
        ("classifier_source_sha256", 64),
    ),
)
def test_manifest_rejects_invalid_hashes(field: str, value: object) -> None:
    data = valid_manifest_dict()
    data[field] = value

    with pytest.raises(ValueError):
        ModelManifest.from_dict(data)


@pytest.mark.parametrize(
    "release_id",
    (
        "0123456789abcdef0123456789abcdef-desktop",
        "f" * 32 + "-android",
        "0123456789ABCDEF0123456789ABCDEF-android",
    ),
)
def test_manifest_rejects_invalid_or_mismatched_release_id(release_id: str) -> None:
    data = valid_manifest_dict()
    data["app_release_id"] = release_id

    with pytest.raises(ValueError):
        ModelManifest.from_dict(data)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), True])
@pytest.mark.parametrize(
    "field", ("detector_confidence", "classifier_threshold", "nms_iou")
)
def test_manifest_rejects_non_finite_and_bool_thresholds(
    field: str, value: object
) -> None:
    data = valid_manifest_dict()
    data[field] = value

    with pytest.raises(ValueError):
        ModelManifest.from_dict(data)


@pytest.mark.parametrize(
    "field", ("detector_confidence", "classifier_threshold", "nms_iou")
)
def test_manifest_rejects_huge_integer_threshold_as_value_error(field: str) -> None:
    data = valid_manifest_dict()
    data[field] = 10**400

    with pytest.raises(ValueError, match=field):
        ModelManifest.from_dict(data)


@pytest.mark.parametrize(
    "field",
    ("detector_input", "detector_output", "classifier_input", "classifier_output"),
)
def test_nested_tensor_errors_include_manifest_field_context(field: str) -> None:
    data = valid_manifest_dict()
    data[field] = {"name": "tensor", "shape": [0], "dtype": "float32"}

    with pytest.raises(ValueError, match=field):
        ModelManifest.from_dict(data)


def test_manifest_from_json_rejects_duplicate_keys() -> None:
    encoded = ModelManifest.from_dict(valid_manifest_dict()).to_json()
    duplicate = encoded.replace('"schema_version": 1', '"schema_version": 1,\n  "schema_version": 1')

    with pytest.raises(ValueError, match="duplicate"):
        ModelManifest.from_json(duplicate)


@pytest.mark.parametrize(
    "payload",
    (
        "[]",
        '{"detector_confidence": NaN}',
        '{"detector_confidence": Infinity}',
        b"\xff",
    ),
)
def test_manifest_from_json_rejects_invalid_documents(payload: str | bytes) -> None:
    with pytest.raises(ValueError):
        ModelManifest.from_json(payload)


def test_tensor_spec_rejects_extra_fields() -> None:
    with pytest.raises(ValueError, match="extra"):
        TensorSpec.from_dict(
            {
                "name": "images",
                "shape": [1, 640, 640, 3],
                "dtype": "float32",
                "layout": "NHWC",
            }
        )


@pytest.mark.parametrize(
    "data",
    (
        {"name": " ", "shape": [1], "dtype": "float32"},
        {"name": "x", "shape": [], "dtype": "float32"},
        {"name": "x", "shape": [True], "dtype": "float32"},
        {"name": "x", "shape": [0], "dtype": "float32"},
        {"name": "x", "shape": [1], "dtype": "float64"},
    ),
)
def test_tensor_spec_rejects_invalid_values(data: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        TensorSpec.from_dict(data)


def test_schema_and_dataclass_fields_are_synchronized() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    manifest_fields = {field.name for field in fields(ModelManifest)}
    tensor_fields = {field.name for field in fields(TensorSpec)}

    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == manifest_fields
    assert set(schema["properties"]) == manifest_fields
    assert schema["$defs"]["tensorSpec"]["additionalProperties"] is False
    assert set(schema["$defs"]["tensorSpec"]["required"]) == tensor_fields
    assert set(schema["$defs"]["tensorSpec"]["properties"]) == tensor_fields
    assert schema["properties"]["conversion"]["additionalProperties"] is False


def test_schema_accepts_valid_manifest_and_rejects_contract_violations() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    valid = valid_manifest_dict()

    jsonschema.validate(valid, schema)

    invalid = valid_manifest_dict()
    invalid["detector_output"] = {
        "name": "output0",
        "shape": [1, 5, 0],
        "dtype": "float32",
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(invalid, schema)

    invalid = valid_manifest_dict()
    invalid["conversion"] = {
        **invalid["conversion"],  # type: ignore[arg-type]
        "extra": "not allowed",
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(invalid, schema)


def test_schema_documents_cross_field_rule_and_runtime_enforces_it() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    constraint = "app_release_id == release_batch_id + '-android'"
    mismatched = valid_manifest_dict()
    mismatched["app_release_id"] = f"{'f' * 32}-android"

    assert schema["x-cross-field-constraints"] == [constraint]
    assert constraint in schema["$comment"]
    jsonschema.Draft202012Validator(schema).validate(mismatched)
    with pytest.raises(ValueError, match="app_release_id"):
        contracts.validate_manifest_payload(mismatched)
