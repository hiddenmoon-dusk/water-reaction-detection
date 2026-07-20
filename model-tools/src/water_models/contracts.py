"""Strict data contracts for Android model release manifests."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
from types import MappingProxyType
from typing import Any, ClassVar, Mapping


_BATCH_ID_PATTERN = re.compile(r"[0-9a-f]{32}")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_DTYPES = frozenset({"float32", "float16", "uint8", "int8"})
_CONVERSION_KEYS = frozenset({"precision", "ultralytics", "tensorflow"})


def _require_exact_keys(
    data: object, expected: frozenset[str], label: str
) -> dict[str, Any]:
    if type(data) is not dict:
        raise ValueError(f"{label} must be an object")
    typed_data = data
    actual = frozenset(typed_data)
    missing = expected - actual
    extra = actual - expected
    if missing:
        raise ValueError(f"{label} has missing fields: {sorted(missing)!r}")
    if extra:
        rendered = sorted((repr(key) for key in extra))
        raise ValueError(f"{label} has extra fields: {rendered!r}")
    return typed_data


def _require_positive_int(value: object, label: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")


def _require_nonblank_string(
    value: object, label: str, *, max_length: int | None = None
) -> None:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    if max_length is not None and len(value) > max_length:
        raise ValueError(f"{label} must contain at most {max_length} characters")


def _require_threshold(value: object, label: str) -> None:
    if type(value) not in (int, float):
        raise ValueError(f"{label} must be a finite number between 0 and 1")
    if not 0 <= value <= 1 or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number between 0 and 1")


@dataclass(frozen=True)
class TensorSpec:
    """The name, dimensions, and scalar type of a model tensor."""

    name: str
    shape: tuple[int, ...]
    dtype: str

    _FIELDS: ClassVar[frozenset[str]] = frozenset({"name", "shape", "dtype"})

    def __post_init__(self) -> None:
        _require_nonblank_string(self.name, "tensor name")
        if type(self.shape) not in (list, tuple) or not self.shape:
            raise ValueError("tensor shape must be a non-empty sequence")
        for dimension in self.shape:
            _require_positive_int(dimension, "tensor shape dimension")
        if type(self.dtype) is not str or self.dtype not in _DTYPES:
            raise ValueError(f"tensor dtype must be one of {sorted(_DTYPES)!r}")
        object.__setattr__(self, "shape", tuple(self.shape))

    @classmethod
    def from_dict(cls, data: object) -> TensorSpec:
        values = _require_exact_keys(data, cls._FIELDS, "TensorSpec")
        if type(values["shape"]) is not list:
            raise ValueError("TensorSpec.shape must be a list")
        return cls(name=values["name"], shape=values["shape"], dtype=values["dtype"])

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "shape": list(self.shape), "dtype": self.dtype}


@dataclass(frozen=True)
class ModelManifest:
    """Validated metadata tying converted models to an Android app release."""

    schema_version: int
    release_batch_id: str
    app_release_id: str
    app_version_code: int
    app_version_name: str
    model_generation: int
    dataset_generation: int
    detector_sha256: str
    classifier_sha256: str
    detector_source_sha256: str
    classifier_source_sha256: str
    detector_input: TensorSpec
    detector_output: TensorSpec
    classifier_input: TensorSpec
    classifier_output: TensorSpec
    detector_confidence: float
    classifier_threshold: float
    nms_iou: float
    class_names: tuple[str, ...]
    conversion: Mapping[str, str]

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema_version",
            "release_batch_id",
            "app_release_id",
            "app_version_code",
            "app_version_name",
            "model_generation",
            "dataset_generation",
            "detector_sha256",
            "classifier_sha256",
            "detector_source_sha256",
            "classifier_source_sha256",
            "detector_input",
            "detector_output",
            "classifier_input",
            "classifier_output",
            "detector_confidence",
            "classifier_threshold",
            "nms_iou",
            "class_names",
            "conversion",
        }
    )

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("schema_version must be integer 1")
        if type(self.release_batch_id) is not str or not _BATCH_ID_PATTERN.fullmatch(
            self.release_batch_id
        ):
            raise ValueError("release_batch_id must be 32 lowercase hexadecimal characters")
        expected_release_id = f"{self.release_batch_id}-android"
        if type(self.app_release_id) is not str or self.app_release_id != expected_release_id:
            raise ValueError("app_release_id must equal '<release_batch_id>-android'")

        _require_positive_int(self.app_version_code, "app_version_code")
        _require_nonblank_string(
            self.app_version_name, "app_version_name", max_length=128
        )
        _require_positive_int(self.model_generation, "model_generation")
        _require_positive_int(self.dataset_generation, "dataset_generation")

        for label, value in (
            ("detector_sha256", self.detector_sha256),
            ("classifier_sha256", self.classifier_sha256),
            ("detector_source_sha256", self.detector_source_sha256),
            ("classifier_source_sha256", self.classifier_source_sha256),
        ):
            if type(value) is not str or not _SHA256_PATTERN.fullmatch(value):
                raise ValueError(f"{label} must be 64 lowercase hexadecimal characters")

        for label, value in (
            ("detector_input", self.detector_input),
            ("detector_output", self.detector_output),
            ("classifier_input", self.classifier_input),
            ("classifier_output", self.classifier_output),
        ):
            if type(value) is not TensorSpec:
                raise ValueError(f"{label} must be a TensorSpec")

        if self.detector_input.shape != (1, 640, 640, 3):
            raise ValueError("detector_input shape must be [1, 640, 640, 3]")
        if (
            len(self.detector_output.shape) != 3
            or self.detector_output.shape[:2] != (1, 5)
        ):
            raise ValueError("detector_output shape must be [1, 5, N] with N positive")
        if self.classifier_input.shape != (1, 128, 128, 3):
            raise ValueError("classifier_input shape must be [1, 128, 128, 3]")
        if self.classifier_output.shape != (1, 1):
            raise ValueError("classifier_output shape must be [1, 1]")

        _require_threshold(self.detector_confidence, "detector_confidence")
        _require_threshold(self.classifier_threshold, "classifier_threshold")
        _require_threshold(self.nms_iou, "nms_iou")

        if type(self.class_names) not in (list, tuple) or tuple(self.class_names) != (
            "lib",
        ):
            raise ValueError("class_names must be exactly ['lib']")
        object.__setattr__(self, "class_names", tuple(self.class_names))

        conversion = _require_exact_keys(self.conversion, _CONVERSION_KEYS, "conversion")
        for key, value in conversion.items():
            _require_nonblank_string(value, f"conversion.{key}")
        object.__setattr__(
            self, "conversion", MappingProxyType(dict(conversion))
        )

    @classmethod
    def from_dict(cls, data: object) -> ModelManifest:
        values = _require_exact_keys(data, cls._FIELDS, "ModelManifest")
        tensor_fields = (
            "detector_input",
            "detector_output",
            "classifier_input",
            "classifier_output",
        )
        converted = dict(values)
        for field_name in tensor_fields:
            try:
                converted[field_name] = TensorSpec.from_dict(values[field_name])
            except ValueError as error:
                raise ValueError(f"{field_name}: {error}") from error
        if type(values["class_names"]) is not list:
            raise ValueError("class_names must be a list")
        return cls(**converted)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "release_batch_id": self.release_batch_id,
            "app_release_id": self.app_release_id,
            "app_version_code": self.app_version_code,
            "app_version_name": self.app_version_name,
            "model_generation": self.model_generation,
            "dataset_generation": self.dataset_generation,
            "detector_sha256": self.detector_sha256,
            "classifier_sha256": self.classifier_sha256,
            "detector_source_sha256": self.detector_source_sha256,
            "classifier_source_sha256": self.classifier_source_sha256,
            "detector_input": self.detector_input.to_dict(),
            "detector_output": self.detector_output.to_dict(),
            "classifier_input": self.classifier_input.to_dict(),
            "classifier_output": self.classifier_output.to_dict(),
            "detector_confidence": self.detector_confidence,
            "classifier_threshold": self.classifier_threshold,
            "nms_iou": self.nms_iou,
            "class_names": list(self.class_names),
            "conversion": dict(self.conversion),
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ) + "\n"

    @classmethod
    def from_json(cls, payload: str | bytes) -> ModelManifest:
        if type(payload) is bytes:
            try:
                payload = payload.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ValueError("manifest must use valid UTF-8") from error
        elif type(payload) is not str:
            raise ValueError("manifest JSON must be str or bytes")

        def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate JSON key: {key}")
                result[key] = value
            return result

        def reject_constant(value: str) -> None:
            raise ValueError(f"invalid JSON numeric constant: {value}")

        try:
            decoded = json.loads(
                payload,
                object_pairs_hook=reject_duplicates,
                parse_constant=reject_constant,
            )
        except json.JSONDecodeError as error:
            raise ValueError("invalid manifest JSON") from error
        if type(decoded) is not dict:
            raise ValueError("manifest JSON must contain an object")
        return cls.from_dict(decoded)


def validate_manifest_payload(payload: object) -> ModelManifest:
    """Use the runtime manifest contract as the cross-field validation authority."""

    if type(payload) is dict:
        return ModelManifest.from_dict(payload)
    if type(payload) in (str, bytes):
        return ModelManifest.from_json(payload)
    raise ValueError("manifest payload must be an object, JSON string, or UTF-8 bytes")
