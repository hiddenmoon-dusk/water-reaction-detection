"""Immutable conversion-parity reports and release policy decisions.

The conversion comparison is deliberately image-scoped: detections are never
matched across samples.  Behavior-change metrics compare old and new source
models independently from the reference/mobile conversion contract.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal
from numbers import Integral, Real
from typing import Sequence

import numpy as np

from .geometry import box_iou
from .inference import Detection


CONFIDENCE_THRESHOLD = 0.3
MINIMUM_IOU = 0.9
MAX_CONFIDENCE_DELTA = 0.05
MAX_PROBABILITY_DELTA = 0.05
MAX_COUNT_CHANGE_RATIO = 0.3
MAX_LABEL_CHANGE_RATIO = 0.2


def _require_bool(name: str, value: object) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a bool")
    return value


def _require_non_negative_int(name: str, value: object) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _require_finite_real(
    name: str,
    value: object,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    try:
        converted = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be finite") from error
    if not np.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and converted < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and converted > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return converted


def _require_optional_finite_real(
    name: str,
    value: object,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    if value is None:
        return None
    return _require_finite_real(
        name, value, minimum=minimum, maximum=maximum
    )


def _decimal_delta(left: float, right: float) -> float:
    """Preserve the explicit decimal meaning of finite float inputs."""

    return float(abs(Decimal(str(float(left))) - Decimal(str(float(right)))))


def _tuple_of_indexes(name: str, value: object) -> tuple[int, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be a tuple")
    indexes = tuple(
        _require_non_negative_int(f"{name}[{index}]", item)
        for index, item in enumerate(value)
    )
    if len(indexes) != len(set(indexes)):
        raise ValueError(f"{name} must not contain duplicates")
    return indexes


def _behavior_metrics(
    old_count: int, new_count: int, label_change_ratio: float
) -> tuple[float | None, bool, bool, bool]:
    if old_count == 0:
        count_ratio = 0.0 if new_count == 0 else None
        count_unbounded = new_count > 0
    else:
        ratio = Decimal(abs(new_count - old_count)) / Decimal(old_count)
        try:
            converted = float(ratio)
        except OverflowError:
            converted = float("inf")
        count_unbounded = not np.isfinite(converted)
        count_ratio = None if count_unbounded else converted
    warning = (
        count_unbounded
        or count_ratio is not None
        and count_ratio > MAX_COUNT_CHANGE_RATIO
        or label_change_ratio > MAX_LABEL_CHANGE_RATIO
    )
    disappeared = old_count > 0 and new_count == 0
    return count_ratio, count_unbounded, warning, disappeared


@dataclass(frozen=True)
class Prediction:
    """A detector prediction paired with its exact classifier label."""

    detection: Detection
    label: str

    def __post_init__(self) -> None:
        if type(self.detection) is not Detection:
            raise TypeError("detection must be a Detection")
        if not isinstance(self.label, str) or not self.label:
            raise ValueError("label must be a non-empty string")

    def to_dict(self) -> dict[str, object]:
        return {
            "box": [
                self.detection.x1,
                self.detection.y1,
                self.detection.x2,
                self.detection.y2,
            ],
            "score": self.detection.score,
            "label": self.label,
        }


@dataclass(frozen=True)
class Match:
    """One deterministic reference/mobile assignment within one image."""

    reference_index: int
    mobile_index: int
    iou: float
    confidence_delta: float
    label_equal: bool
    probability_delta: float | None = None

    def __post_init__(self) -> None:
        _require_non_negative_int("reference_index", self.reference_index)
        _require_non_negative_int("mobile_index", self.mobile_index)
        iou = _require_finite_real(
            "iou", self.iou, minimum=MINIMUM_IOU, maximum=1.0
        )
        confidence_delta = _require_finite_real(
            "confidence_delta", self.confidence_delta, minimum=0.0, maximum=1.0
        )
        _require_bool("label_equal", self.label_equal)
        probability_delta = _require_optional_finite_real(
            "probability_delta",
            self.probability_delta,
            minimum=0.0,
            maximum=1.0,
        )
        object.__setattr__(self, "iou", iou)
        object.__setattr__(self, "confidence_delta", confidence_delta)
        object.__setattr__(self, "probability_delta", probability_delta)

    def to_dict(self) -> dict[str, object]:
        return {
            "reference_index": self.reference_index,
            "mobile_index": self.mobile_index,
            "iou": self.iou,
            "confidence_delta": self.confidence_delta,
            "label_equal": self.label_equal,
            "probability_delta": self.probability_delta,
        }


@dataclass(frozen=True)
class ImageParity:
    """Strong conversion-parity evidence for one image."""

    sample_id: str
    reference_count: int
    mobile_count: int
    matches: tuple[Match, ...]
    unmatched_reference_indexes: tuple[int, ...]
    unmatched_mobile_indexes: tuple[int, ...]
    min_match_iou: float | None
    max_confidence_delta: float | None
    max_probability_delta: float | None
    label_mismatch_count: int
    conversion_passed: bool

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id:
            raise ValueError("sample_id must be a non-empty string")
        reference_count = _require_non_negative_int(
            "reference_count", self.reference_count
        )
        mobile_count = _require_non_negative_int(
            "mobile_count", self.mobile_count
        )
        if type(self.matches) is not tuple or any(
            type(match) is not Match for match in self.matches
        ):
            raise TypeError("matches must be a tuple of Match values")
        unmatched_reference = _tuple_of_indexes(
            "unmatched_reference_indexes", self.unmatched_reference_indexes
        )
        unmatched_mobile = _tuple_of_indexes(
            "unmatched_mobile_indexes", self.unmatched_mobile_indexes
        )
        reference_indexes = tuple(
            match.reference_index for match in self.matches
        )
        mobile_indexes = tuple(match.mobile_index for match in self.matches)
        if len(reference_indexes) != len(set(reference_indexes)):
            raise ValueError("matches must use each reference index at most once")
        if len(mobile_indexes) != len(set(mobile_indexes)):
            raise ValueError("matches must use each mobile index at most once")
        if set(reference_indexes) & set(unmatched_reference):
            raise ValueError("matched and unmatched reference indexes overlap")
        if set(mobile_indexes) & set(unmatched_mobile):
            raise ValueError("matched and unmatched mobile indexes overlap")
        if any(index >= reference_count for index in reference_indexes):
            raise ValueError(
                "match reference_index must be less than reference_count"
            )
        if any(index >= mobile_count for index in mobile_indexes):
            raise ValueError("match mobile_index must be less than mobile_count")
        if any(index >= reference_count for index in unmatched_reference):
            raise ValueError(
                "unmatched reference index must be less than reference_count"
            )
        if any(index >= mobile_count for index in unmatched_mobile):
            raise ValueError(
                "unmatched mobile index must be less than mobile_count"
            )
        if reference_count != len(self.matches) + len(unmatched_reference):
            raise ValueError("reference_count does not match image assignments")
        if mobile_count != len(self.matches) + len(unmatched_mobile):
            raise ValueError("mobile_count does not match image assignments")
        if set(reference_indexes) | set(unmatched_reference) != set(
            range(reference_count)
        ):
            raise ValueError(
                "reference assignments must exactly partition range(reference_count)"
            )
        if set(mobile_indexes) | set(unmatched_mobile) != set(
            range(mobile_count)
        ):
            raise ValueError(
                "mobile assignments must exactly partition range(mobile_count)"
            )

        expected_min_iou = min(
            (match.iou for match in self.matches), default=None
        )
        expected_max_confidence = max(
            (match.confidence_delta for match in self.matches), default=None
        )
        expected_max_probability = max(
            (
                match.probability_delta
                for match in self.matches
                if match.probability_delta is not None
            ),
            default=None,
        )
        min_iou = _require_optional_finite_real(
            "min_match_iou", self.min_match_iou, minimum=MINIMUM_IOU, maximum=1.0
        )
        max_confidence = _require_optional_finite_real(
            "max_confidence_delta",
            self.max_confidence_delta,
            minimum=0.0,
            maximum=1.0,
        )
        max_probability = _require_optional_finite_real(
            "max_probability_delta",
            self.max_probability_delta,
            minimum=0.0,
            maximum=1.0,
        )
        if min_iou != expected_min_iou:
            raise ValueError("min_match_iou does not match matches")
        if max_confidence != expected_max_confidence:
            raise ValueError("max_confidence_delta does not match matches")
        if max_probability != expected_max_probability:
            raise ValueError("max_probability_delta does not match matches")
        expected_label_mismatches = sum(
            not match.label_equal for match in self.matches
        )
        label_mismatches = _require_non_negative_int(
            "label_mismatch_count", self.label_mismatch_count
        )
        if label_mismatches != expected_label_mismatches:
            raise ValueError("label_mismatch_count does not match matches")
        conversion_passed = _require_bool(
            "conversion_passed", self.conversion_passed
        )
        expected_passed = (
            reference_count == mobile_count
            and len(self.matches) == reference_count
            and not unmatched_reference
            and not unmatched_mobile
            and all(match.iou >= MINIMUM_IOU for match in self.matches)
            and all(
                match.confidence_delta <= MAX_CONFIDENCE_DELTA
                for match in self.matches
            )
            and all(match.label_equal for match in self.matches)
            and all(
                match.probability_delta is not None
                and match.probability_delta <= MAX_PROBABILITY_DELTA
                for match in self.matches
            )
        )
        if conversion_passed != expected_passed:
            raise ValueError("conversion_passed does not match image evidence")
        object.__setattr__(self, "min_match_iou", min_iou)
        object.__setattr__(self, "max_confidence_delta", max_confidence)
        object.__setattr__(self, "max_probability_delta", max_probability)

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "reference_count": self.reference_count,
            "mobile_count": self.mobile_count,
            "matches": [match.to_dict() for match in self.matches],
            "unmatched_reference_indexes": list(
                self.unmatched_reference_indexes
            ),
            "unmatched_mobile_indexes": list(self.unmatched_mobile_indexes),
            "min_match_iou": self.min_match_iou,
            "max_confidence_delta": self.max_confidence_delta,
            "max_probability_delta": self.max_probability_delta,
            "label_mismatch_count": self.label_mismatch_count,
            "conversion_passed": self.conversion_passed,
        }


@dataclass(frozen=True)
class ParityReport:
    """Deeply immutable aggregate conversion and behavior report."""

    images: tuple[ImageParity, ...]
    reference_count: int
    mobile_count: int
    min_match_iou: float | None
    max_confidence_delta: float | None
    max_probability_delta: float | None
    label_mismatch_count: int
    conversion_passed: bool
    all_expected_detections_disappeared: bool
    invalid_reasons: tuple[str, ...]
    old_count: int
    new_count: int
    count_change_ratio: float | None
    count_change_unbounded: bool
    label_change_ratio: float
    behavior_warning: bool
    behavior_all_detections_disappeared: bool

    def __post_init__(self) -> None:
        if type(self.images) is not tuple or any(
            type(image) is not ImageParity for image in self.images
        ):
            raise TypeError("images must be a tuple of ImageParity values")
        if type(self.invalid_reasons) is not tuple:
            raise TypeError("invalid_reasons must be a tuple")
        if any(
            not isinstance(reason, str) or not reason
            for reason in self.invalid_reasons
        ):
            raise ValueError("invalid_reasons must contain non-empty strings")
        if len(self.invalid_reasons) != len(set(self.invalid_reasons)):
            raise ValueError("invalid_reasons must not contain duplicates")
        if not self.images and not self.invalid_reasons:
            raise ValueError("at least one sample is required")

        expected_reference_count = sum(
            image.reference_count for image in self.images
        )
        expected_mobile_count = sum(image.mobile_count for image in self.images)
        reference_count = _require_non_negative_int(
            "reference_count", self.reference_count
        )
        mobile_count = _require_non_negative_int(
            "mobile_count", self.mobile_count
        )
        if reference_count != expected_reference_count:
            raise ValueError("reference_count does not match images")
        if mobile_count != expected_mobile_count:
            raise ValueError("mobile_count does not match images")

        image_ious = tuple(
            image.min_match_iou
            for image in self.images
            if image.min_match_iou is not None
        )
        image_confidence_deltas = tuple(
            image.max_confidence_delta
            for image in self.images
            if image.max_confidence_delta is not None
        )
        image_probability_deltas = tuple(
            image.max_probability_delta
            for image in self.images
            if image.max_probability_delta is not None
        )
        expected_min_iou = min(image_ious, default=None)
        expected_max_confidence = max(image_confidence_deltas, default=None)
        expected_max_probability = max(image_probability_deltas, default=None)
        min_iou = _require_optional_finite_real(
            "min_match_iou", self.min_match_iou, minimum=MINIMUM_IOU, maximum=1.0
        )
        max_confidence = _require_optional_finite_real(
            "max_confidence_delta",
            self.max_confidence_delta,
            minimum=0.0,
            maximum=1.0,
        )
        max_probability = _require_optional_finite_real(
            "max_probability_delta",
            self.max_probability_delta,
            minimum=0.0,
            maximum=1.0,
        )
        if min_iou != expected_min_iou:
            raise ValueError("min_match_iou does not match images")
        if max_confidence != expected_max_confidence:
            raise ValueError("max_confidence_delta does not match images")
        if max_probability != expected_max_probability:
            raise ValueError("max_probability_delta does not match images")
        expected_label_mismatches = sum(
            image.label_mismatch_count for image in self.images
        )
        label_mismatches = _require_non_negative_int(
            "label_mismatch_count", self.label_mismatch_count
        )
        if label_mismatches != expected_label_mismatches:
            raise ValueError("label_mismatch_count does not match images")

        conversion_passed = _require_bool(
            "conversion_passed", self.conversion_passed
        )
        expected_conversion_passed = (
            len(self.images) > 0
            and not self.invalid_reasons
            and all(image.conversion_passed for image in self.images)
        )
        if conversion_passed != expected_conversion_passed:
            raise ValueError("conversion_passed does not match report evidence")
        all_expected_disappeared = _require_bool(
            "all_expected_detections_disappeared",
            self.all_expected_detections_disappeared,
        )
        if all_expected_disappeared != (
            reference_count > 0 and mobile_count == 0
        ):
            raise ValueError(
                "all_expected_detections_disappeared does not match counts"
            )

        old_count = _require_non_negative_int("old_count", self.old_count)
        new_count = _require_non_negative_int("new_count", self.new_count)
        label_ratio = _require_finite_real(
            "label_change_ratio",
            self.label_change_ratio,
            minimum=0.0,
            maximum=1.0,
        )
        expected_ratio, expected_unbounded, expected_warning, expected_disappeared = (
            _behavior_metrics(old_count, new_count, label_ratio)
        )
        count_ratio = _require_optional_finite_real(
            "count_change_ratio", self.count_change_ratio, minimum=0.0
        )
        if count_ratio != expected_ratio:
            raise ValueError("count_change_ratio does not match behavior counts")
        count_unbounded = _require_bool(
            "count_change_unbounded", self.count_change_unbounded
        )
        behavior_warning = _require_bool(
            "behavior_warning", self.behavior_warning
        )
        behavior_disappeared = _require_bool(
            "behavior_all_detections_disappeared",
            self.behavior_all_detections_disappeared,
        )
        if count_unbounded != expected_unbounded:
            raise ValueError(
                "count_change_unbounded does not match behavior counts"
            )
        if behavior_warning != expected_warning:
            raise ValueError("behavior_warning does not match behavior metrics")
        if behavior_disappeared != expected_disappeared:
            raise ValueError(
                "behavior_all_detections_disappeared does not match counts"
            )
        object.__setattr__(self, "min_match_iou", min_iou)
        object.__setattr__(self, "max_confidence_delta", max_confidence)
        object.__setattr__(self, "max_probability_delta", max_probability)
        object.__setattr__(self, "label_change_ratio", label_ratio)
        object.__setattr__(self, "count_change_ratio", count_ratio)

    def to_dict(self) -> dict[str, object]:
        return {
            "images": [image.to_dict() for image in self.images],
            "reference_count": self.reference_count,
            "mobile_count": self.mobile_count,
            "min_match_iou": self.min_match_iou,
            "max_confidence_delta": self.max_confidence_delta,
            "max_probability_delta": self.max_probability_delta,
            "label_mismatch_count": self.label_mismatch_count,
            "conversion_passed": self.conversion_passed,
            "all_expected_detections_disappeared": (
                self.all_expected_detections_disappeared
            ),
            "invalid_reasons": list(self.invalid_reasons),
            "behavior": {
                "old_count": self.old_count,
                "new_count": self.new_count,
                "count_change_ratio": self.count_change_ratio,
                "count_change_unbounded": self.count_change_unbounded,
                "label_change_ratio": self.label_change_ratio,
                "warning": self.behavior_warning,
                "all_detections_disappeared": (
                    self.behavior_all_detections_disappeared
                ),
            },
        }


@dataclass(frozen=True)
class ParityDecision:
    """Immutable release decision derived from a :class:`ParityReport`."""

    hard_fail: bool
    can_override: bool
    requires_confirmation: bool
    behavior_warning: bool
    overridden: bool
    operator_reason: str | None
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        hard_fail = _require_bool("hard_fail", self.hard_fail)
        can_override = _require_bool("can_override", self.can_override)
        requires_confirmation = _require_bool(
            "requires_confirmation", self.requires_confirmation
        )
        behavior_warning = _require_bool(
            "behavior_warning", self.behavior_warning
        )
        overridden = _require_bool("overridden", self.overridden)
        if self.operator_reason is not None:
            if (
                not isinstance(self.operator_reason, str)
                or not self.operator_reason
                or self.operator_reason != self.operator_reason.strip()
            ):
                raise ValueError(
                    "operator_reason must be a non-empty trimmed string or None"
                )
        if type(self.reasons) is not tuple:
            raise TypeError("reasons must be a tuple")
        if any(
            not isinstance(reason, str) or not reason for reason in self.reasons
        ):
            raise ValueError("reasons must contain non-empty strings")
        if len(self.reasons) != len(set(self.reasons)):
            raise ValueError("reasons must not contain duplicates")

        if hard_fail:
            if (
                can_override
                or requires_confirmation
                or overridden
                or self.operator_reason is not None
                or not self.reasons
            ):
                raise ValueError("hard failure decision state is inconsistent")
            return
        if behavior_warning:
            if not can_override or not self.reasons:
                raise ValueError("behavior warning decision must be overridable")
            if overridden:
                if requires_confirmation or self.operator_reason is None:
                    raise ValueError(
                        "overridden decision requires a recorded operator reason"
                    )
            elif not requires_confirmation or self.operator_reason is not None:
                raise ValueError(
                    "unconfirmed behavior warning decision state is inconsistent"
                )
            return
        if (
            can_override
            or requires_confirmation
            or overridden
            or self.operator_reason is not None
            or self.reasons
        ):
            raise ValueError("passing decision state is inconsistent")

    def to_dict(self) -> dict[str, object]:
        return {
            "hard_fail": self.hard_fail,
            "can_override": self.can_override,
            "requires_confirmation": self.requires_confirmation,
            "behavior_warning": self.behavior_warning,
            "overridden": self.overridden,
            "operator_reason": self.operator_reason,
            "reasons": list(self.reasons),
        }


def _strict_sequence(name: str, value: object) -> tuple[object, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list or tuple")
    return tuple(value)


def _prediction_sequence(name: str, value: object) -> tuple[Prediction, ...]:
    items = _strict_sequence(name, value)
    if any(type(item) is not Prediction for item in items):
        raise ValueError(f"{name} must contain only Prediction values")
    return items  # type: ignore[return-value]


def _probability_sequence(
    name: str, value: object, expected_length: int
) -> tuple[float, ...]:
    items = _strict_sequence(name, value)
    if len(items) != expected_length:
        raise ValueError(
            f"{name} length must match its unfiltered detection count"
        )
    probabilities: list[float] = []
    for index, item in enumerate(items):
        if isinstance(item, bool) or not isinstance(item, Real):
            raise ValueError(f"{name}[{index}] must be a real number")
        try:
            probability = float(item)
        except OverflowError as error:
            raise ValueError(f"{name}[{index}] must be finite") from error
        if not np.isfinite(probability):
            raise ValueError(f"{name}[{index}] must be finite")
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"{name}[{index}] must be between 0 and 1")
        probabilities.append(probability)
    return tuple(probabilities)


def _filtered(
    predictions: Sequence[Prediction],
) -> tuple[tuple[int, Prediction], ...]:
    return tuple(
        (index, item)
        for index, item in enumerate(predictions)
        if item.detection.score >= CONFIDENCE_THRESHOLD
    )


def match_detections(
    reference: Sequence[Prediction],
    mobile: Sequence[Prediction],
) -> tuple[Match, ...]:
    """Globally rank all eligible IoUs, then greedily assign one-to-one.

    Both sides are filtered at confidence ``>= 0.3``. Candidate pairs are
    sorted by descending IoU, then reference index, then mobile index. Only
    IoUs at least ``0.90`` are eligible.
    """

    reference_items = _prediction_sequence("reference", reference)
    mobile_items = _prediction_sequence("mobile", mobile)
    filtered_reference = _filtered(reference_items)
    filtered_mobile = _filtered(mobile_items)
    if not filtered_reference or not filtered_mobile:
        return ()

    reference_boxes = np.asarray(
        [
            (
                item.detection.x1,
                item.detection.y1,
                item.detection.x2,
                item.detection.y2,
            )
            for _, item in filtered_reference
        ],
        dtype=np.float64,
    )
    mobile_boxes = np.asarray(
        [
            (
                item.detection.x1,
                item.detection.y1,
                item.detection.x2,
                item.detection.y2,
            )
            for _, item in filtered_mobile
        ],
        dtype=np.float64,
    )
    overlaps = box_iou(reference_boxes, mobile_boxes)
    if not np.all(np.isfinite(overlaps)):
        raise ValueError("IoU matrix must contain only finite values")

    candidates: list[tuple[float, int, int, Prediction, Prediction]] = []
    for reference_position, (_, reference_item) in enumerate(
        filtered_reference
    ):
        for mobile_position, (_, mobile_item) in enumerate(
            filtered_mobile
        ):
            overlap = float(overlaps[reference_position, mobile_position])
            if overlap >= MINIMUM_IOU:
                candidates.append(
                    (
                        overlap,
                        reference_position,
                        mobile_position,
                        reference_item,
                        mobile_item,
                    )
                )
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

    used_reference: set[int] = set()
    used_mobile: set[int] = set()
    matches: list[Match] = []
    for overlap, reference_index, mobile_index, reference_item, mobile_item in candidates:
        if reference_index in used_reference or mobile_index in used_mobile:
            continue
        used_reference.add(reference_index)
        used_mobile.add(mobile_index)
        matches.append(
            Match(
                reference_index=reference_index,
                mobile_index=mobile_index,
                iou=overlap,
                confidence_delta=_decimal_delta(
                    reference_item.detection.score,
                    mobile_item.detection.score,
                ),
                label_equal=reference_item.label == mobile_item.label,
            )
        )
    return tuple(matches)


def _image_report(
    sample_id: str,
    reference: tuple[Prediction, ...],
    mobile: tuple[Prediction, ...],
    reference_probabilities: tuple[float, ...],
    mobile_probabilities: tuple[float, ...],
) -> ImageParity:
    reference_filtered = _filtered(reference)
    mobile_filtered = _filtered(mobile)
    matches = tuple(
        replace(
            match,
            probability_delta=_decimal_delta(
                reference_probabilities[
                    reference_filtered[match.reference_index][0]
                ],
                mobile_probabilities[mobile_filtered[match.mobile_index][0]],
            ),
        )
        for match in match_detections(reference, mobile)
    )
    matched_reference = {match.reference_index for match in matches}
    matched_mobile = {match.mobile_index for match in matches}
    unmatched_reference = tuple(
        position
        for position in range(len(reference_filtered))
        if position not in matched_reference
    )
    unmatched_mobile = tuple(
        position
        for position in range(len(mobile_filtered))
        if position not in matched_mobile
    )
    min_iou = min((match.iou for match in matches), default=None)
    max_confidence = max(
        (match.confidence_delta for match in matches), default=None
    )
    max_probability = max(
        (
            match.probability_delta
            for match in matches
            if match.probability_delta is not None
        ),
        default=None,
    )
    label_mismatches = sum(not match.label_equal for match in matches)
    reference_count = len(reference_filtered)
    mobile_count = len(mobile_filtered)
    passed = (
        reference_count == mobile_count
        and len(matches) == reference_count
        and not unmatched_reference
        and not unmatched_mobile
        and all(match.iou >= MINIMUM_IOU for match in matches)
        and all(
            match.confidence_delta <= MAX_CONFIDENCE_DELTA for match in matches
        )
        and all(match.label_equal for match in matches)
        and all(
            match.probability_delta is not None
            and match.probability_delta <= MAX_PROBABILITY_DELTA
            for match in matches
        )
    )
    return ImageParity(
        sample_id=sample_id,
        reference_count=reference_count,
        mobile_count=mobile_count,
        matches=matches,
        unmatched_reference_indexes=unmatched_reference,
        unmatched_mobile_indexes=unmatched_mobile,
        min_match_iou=min_iou,
        max_confidence_delta=max_confidence,
        max_probability_delta=max_probability,
        label_mismatch_count=label_mismatches,
        conversion_passed=passed,
    )


def _behavior_count(
    name: str, value: object, invalid_reasons: list[str]
) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        invalid_reasons.append(f"{name} must be a non-negative integer")
        return 0
    converted = int(value)
    if converted < 0:
        invalid_reasons.append(f"{name} must be a non-negative integer")
        return 0
    return converted


def _behavior_ratio(value: object, invalid_reasons: list[str]) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        invalid_reasons.append("label_change_ratio must be between 0 and 1")
        return 0.0
    try:
        converted = float(value)
    except OverflowError:
        invalid_reasons.append("label_change_ratio must be finite")
        return 0.0
    if not np.isfinite(converted) or not 0.0 <= converted <= 1.0:
        invalid_reasons.append("label_change_ratio must be between 0 and 1")
        return 0.0
    return converted


def compare_conversion(
    reference_detections: object,
    mobile_detections: object,
    reference_probabilities: object,
    mobile_probabilities: object,
    *,
    old_count: object,
    new_count: object,
    label_change_ratio: object,
    sample_ids: object = None,
) -> ParityReport:
    """Build a multi-image conversion and behavior report from raw results.

    The first four arguments are equally-sized nested lists/tuples, one inner
    collection per image. Probabilities align with unfiltered detections, so a
    low-confidence detection still requires one valid probability value.
    Invalid inputs are captured in ``invalid_reasons`` and become a hard policy
    failure instead of leaking NaN/Inf into a report.
    """

    invalid_reasons: list[str] = []
    outer_values: list[tuple[object, ...]] = []
    for name, value in (
        ("reference_detections", reference_detections),
        ("mobile_detections", mobile_detections),
        ("reference_probabilities", reference_probabilities),
        ("mobile_probabilities", mobile_probabilities),
    ):
        try:
            outer_values.append(_strict_sequence(name, value))
        except ValueError as error:
            invalid_reasons.append(str(error))
            outer_values.append(())

    reference_outer, mobile_outer, reference_probability_outer, mobile_probability_outer = (
        outer_values
    )
    lengths = tuple(len(value) for value in outer_values)
    image_count = lengths[0] if len(set(lengths)) == 1 else 0
    if len(set(lengths)) != 1:
        invalid_reasons.append("per-image input collections must have equal length")
    elif image_count == 0:
        invalid_reasons.append("at least one sample is required")

    if sample_ids is None:
        ids = tuple(f"sample_{index}" for index in range(image_count))
    else:
        try:
            raw_ids = _strict_sequence("sample_ids", sample_ids)
        except ValueError as error:
            invalid_reasons.append(str(error))
            raw_ids = ()
        if len(raw_ids) != image_count:
            invalid_reasons.append("sample_ids length must match image count")
            ids = tuple(f"sample_{index}" for index in range(image_count))
        else:
            validated_ids: list[str] = []
            for index, raw_id in enumerate(raw_ids):
                if not isinstance(raw_id, str) or not raw_id:
                    invalid_reasons.append(
                        f"sample_ids[{index}] must be a non-empty string"
                    )
                    validated_ids.append(f"sample_{index}")
                else:
                    validated_ids.append(raw_id)
            ids = tuple(validated_ids)

    images: list[ImageParity] = []
    for index in range(image_count):
        inputs_valid = True
        try:
            reference = _prediction_sequence(
                f"reference_detections[{index}]", reference_outer[index]
            )
        except ValueError as error:
            invalid_reasons.append(str(error))
            reference = ()
            inputs_valid = False
        try:
            mobile = _prediction_sequence(
                f"mobile_detections[{index}]", mobile_outer[index]
            )
        except ValueError as error:
            invalid_reasons.append(str(error))
            mobile = ()
            inputs_valid = False
        try:
            reference_probs = _probability_sequence(
                f"reference_probabilities[{index}]",
                reference_probability_outer[index],
                len(reference),
            )
        except ValueError as error:
            invalid_reasons.append(str(error))
            reference_probs = ()
            inputs_valid = False
        try:
            mobile_probs = _probability_sequence(
                f"mobile_probabilities[{index}]",
                mobile_probability_outer[index],
                len(mobile),
            )
        except ValueError as error:
            invalid_reasons.append(str(error))
            mobile_probs = ()
            inputs_valid = False
        if inputs_valid:
            images.append(
                _image_report(
                    ids[index],
                    reference,
                    mobile,
                    reference_probs,
                    mobile_probs,
                )
            )

    old = _behavior_count("old_count", old_count, invalid_reasons)
    new = _behavior_count("new_count", new_count, invalid_reasons)
    label_ratio = _behavior_ratio(label_change_ratio, invalid_reasons)
    (
        count_ratio,
        count_unbounded,
        behavior_warning,
        behavior_disappeared,
    ) = _behavior_metrics(old, new, label_ratio)

    reference_count = sum(image.reference_count for image in images)
    mobile_count = sum(image.mobile_count for image in images)
    ious = tuple(
        image.min_match_iou
        for image in images
        if image.min_match_iou is not None
    )
    confidence_deltas = tuple(
        image.max_confidence_delta
        for image in images
        if image.max_confidence_delta is not None
    )
    probability_deltas = tuple(
        image.max_probability_delta
        for image in images
        if image.max_probability_delta is not None
    )
    invalid = tuple(dict.fromkeys(invalid_reasons))
    conversion_passed = (
        len(images) > 0
        and not invalid
        and all(image.conversion_passed for image in images)
    )
    return ParityReport(
        images=tuple(images),
        reference_count=reference_count,
        mobile_count=mobile_count,
        min_match_iou=min(ious, default=None),
        max_confidence_delta=max(confidence_deltas, default=None),
        max_probability_delta=max(probability_deltas, default=None),
        label_mismatch_count=sum(
            image.label_mismatch_count for image in images
        ),
        conversion_passed=conversion_passed,
        all_expected_detections_disappeared=(
            reference_count > 0 and mobile_count == 0
        ),
        invalid_reasons=invalid,
        old_count=old,
        new_count=new,
        count_change_ratio=count_ratio,
        count_change_unbounded=count_unbounded,
        label_change_ratio=label_ratio,
        behavior_warning=behavior_warning,
        behavior_all_detections_disappeared=behavior_disappeared,
    )


def evaluate(
    report: ParityReport, operator_reason: str | None = None
) -> ParityDecision:
    """Evaluate hard conversion failures separately from behavior warnings."""

    if type(report) is not ParityReport:
        raise TypeError("report must be a ParityReport")
    if operator_reason is not None and not isinstance(operator_reason, str):
        raise TypeError("operator_reason must be a string or None")

    hard_reasons: list[str] = []
    if report.invalid_reasons:
        hard_reasons.append("report contains invalid input")
    if not report.conversion_passed:
        hard_reasons.append("strong conversion parity failed")
    if report.all_expected_detections_disappeared:
        hard_reasons.append("all expected conversion detections disappeared")
    if report.behavior_all_detections_disappeared:
        hard_reasons.append("all old-model behavior detections disappeared")
    if hard_reasons:
        return ParityDecision(
            hard_fail=True,
            can_override=False,
            requires_confirmation=False,
            behavior_warning=report.behavior_warning,
            overridden=False,
            operator_reason=None,
            reasons=tuple(hard_reasons),
        )

    if report.behavior_warning:
        reason = operator_reason.strip() if operator_reason is not None else ""
        confirmed = bool(reason)
        return ParityDecision(
            hard_fail=False,
            can_override=True,
            requires_confirmation=not confirmed,
            behavior_warning=True,
            overridden=confirmed,
            operator_reason=reason if confirmed else None,
            reasons=("old/new model behavior exceeds warning threshold",),
        )

    return ParityDecision(
        hard_fail=False,
        can_override=False,
        requires_confirmation=False,
        behavior_warning=False,
        overridden=False,
        operator_reason=None,
        reasons=(),
    )
