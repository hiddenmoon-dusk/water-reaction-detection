"""Shared image geometry and non-maximum suppression utilities."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Sequence

import cv2
import numpy as np


def _finite_real(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    try:
        converted = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be finite") from error
    if not np.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    converted = int(value)
    if converted <= 0:
        raise ValueError(f"{name} must be positive")
    return converted


def _resized_dimension(original: int, scale: float) -> int:
    scaled = original * scale
    if not np.isfinite(scaled):
        raise ValueError("scaled image dimension must be finite")
    return max(1, int(round(scaled)))


@dataclass(frozen=True)
class LetterboxTransform:
    """Mapping from square letterbox coordinates back to an original image."""

    scale: float
    pad_x: float
    pad_y: float
    original_width: int
    original_height: int

    def __post_init__(self) -> None:
        scale = _finite_real("scale", self.scale)
        pad_x = _finite_real("pad_x", self.pad_x)
        pad_y = _finite_real("pad_y", self.pad_y)
        _positive_int("original_width", self.original_width)
        _positive_int("original_height", self.original_height)
        if scale <= 0:
            raise ValueError("scale must be positive")
        if pad_x < 0 or pad_y < 0:
            raise ValueError("padding must be non-negative")

    def to_original(
        self, box: Sequence[Real]
    ) -> tuple[float, float, float, float]:
        """Undo and clip an ``xyxy`` box to the image.

        Clipping can produce a zero-area box. Callers must filter such results
        before passing boxes to :func:`nms`.
        """

        try:
            coordinates = tuple(box)
        except TypeError as error:
            raise TypeError("box must be a sequence of four coordinates") from error
        if len(coordinates) != 4:
            raise ValueError("box must contain four coordinates")
        x1, y1, x2, y2 = (
            _finite_real(f"box[{index}]", value)
            for index, value in enumerate(coordinates)
        )
        if x2 < x1 or y2 < y1:
            raise ValueError("box coordinates must be ordered as x1, y1, x2, y2")

        resized_width = _resized_dimension(self.original_width, float(self.scale))
        resized_height = _resized_dimension(self.original_height, float(self.scale))
        scale_x = resized_width / self.original_width
        scale_y = resized_height / self.original_height
        original_x1 = (x1 - float(self.pad_x)) / scale_x
        original_y1 = (y1 - float(self.pad_y)) / scale_y
        original_x2 = (x2 - float(self.pad_x)) / scale_x
        original_y2 = (y2 - float(self.pad_y)) / scale_y
        width = float(self.original_width)
        height = float(self.original_height)
        return (
            min(max(original_x1, 0.0), width),
            min(max(original_y1, 0.0), height),
            min(max(original_x2, 0.0), width),
            min(max(original_y2, 0.0), height),
        )


def letterbox(
    image: np.ndarray, size: int = 640
) -> tuple[np.ndarray, LetterboxTransform]:
    """Resize an RGB-like uint8 image into a square canvas without distortion."""

    target_size = _positive_int("size", size)
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy array")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape HxWx3")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("image dimensions must be non-empty")
    if image.dtype != np.uint8:
        raise TypeError("image dtype must be uint8")

    height, width = image.shape[:2]
    scale = min(target_size / width, target_size / height)
    resized_width = _resized_dimension(width, scale)
    resized_height = _resized_dimension(height, scale)
    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )

    horizontal_padding = target_size - resized_width
    vertical_padding = target_size - resized_height
    pad_x = horizontal_padding // 2
    pad_y = vertical_padding // 2
    output = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    output[
        pad_y : pad_y + resized_height,
        pad_x : pad_x + resized_width,
    ] = resized

    transform = LetterboxTransform(scale, pad_x, pad_y, width, height)
    return np.ascontiguousarray(output), transform


def _boxes_array(name: str, boxes: object) -> np.ndarray:
    array = np.asarray(boxes)
    if array.ndim != 2 or array.shape[1:] != (4,):
        raise ValueError(f"{name} must have shape Nx4")
    if array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numbers")
    converted = array.astype(np.float64, copy=False)
    if not np.all(np.isfinite(converted)):
        raise ValueError(f"{name} must contain only finite coordinates")
    if converted.size and np.any(converted[:, 2:] <= converted[:, :2]):
        raise ValueError(f"{name} boxes must have positive width and height")
    return converted


def _pairwise_iou(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.shape[0] == 0 or right.shape[0] == 0:
        return np.zeros((left.shape[0], right.shape[0]), dtype=np.float64)

    def normalized_axis(
        start_index: int, end_index: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        left_scale = np.maximum(
            np.abs(left[:, start_index]), np.abs(left[:, end_index])
        )
        right_scale = np.maximum(
            np.abs(right[:, start_index]), np.abs(right[:, end_index])
        )
        pair_scale = np.maximum(left_scale[:, None], right_scale[None, :])
        pair_scale[pair_scale == 0.0] = 1.0
        with np.errstate(under="ignore"):
            left_start = left[:, None, start_index] / pair_scale
            left_end = left[:, None, end_index] / pair_scale
            right_start = right[None, :, start_index] / pair_scale
            right_end = right[None, :, end_index] / pair_scale
        intersection = np.maximum(
            np.minimum(left_end, right_end) - np.maximum(left_start, right_start),
            0.0,
        )
        return intersection, left_end - left_start, right_end - right_start

    intersection_width, left_width, right_width = normalized_axis(0, 2)
    intersection_height, left_height, right_height = normalized_axis(1, 3)
    with np.errstate(under="ignore"):
        intersection = intersection_width * intersection_height
        left_area = left_width * left_height
        right_area = right_width * right_height
    union = left_area + right_area - intersection
    if not all(
        np.all(np.isfinite(values))
        for values in (intersection, left_area, right_area, union)
    ):
        raise ValueError("IoU calculation produced non-finite values")
    result = np.zeros_like(union)
    np.divide(intersection, union, out=result, where=union > 0)
    return result


def box_iou(left: object, right: object) -> np.ndarray:
    """Return pairwise continuous-coordinate IoU for two ``Nx4`` box arrays."""

    left_array = _boxes_array("left", left)
    right_array = _boxes_array("right", right)
    return _pairwise_iou(left_array, right_array)


def nms(
    boxes: object,
    scores: object,
    iou_threshold: Real,
    max_candidates: int | None = None,
    max_detections: int | None = None,
) -> list[int]:
    """Select boxes by stable descending score using continuous-coordinate IoU.

    Degenerate inputs are rejected strictly. Callers must filter zero-area boxes
    produced by clipping before calling this function.
    """

    threshold = _finite_real("iou_threshold", iou_threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("iou_threshold must be between 0 and 1")
    candidate_limit = (
        None
        if max_candidates is None
        else _positive_int("max_candidates", max_candidates)
    )
    detection_limit = (
        None
        if max_detections is None
        else _positive_int("max_detections", max_detections)
    )
    boxes_array = _boxes_array("boxes", boxes)
    scores_array = np.asarray(scores)
    if scores_array.ndim != 1:
        raise ValueError("scores must have shape N")
    if scores_array.dtype.kind not in "iuf":
        raise TypeError("scores must contain real numbers")
    scores_float = scores_array.astype(np.float64, copy=False)
    if scores_float.shape[0] != boxes_array.shape[0]:
        raise ValueError("scores length must match boxes")
    if not np.all(np.isfinite(scores_float)):
        raise ValueError("scores must contain only finite values")
    if boxes_array.shape[0] == 0:
        return []

    original_indexes = np.arange(boxes_array.shape[0])
    order = np.lexsort((original_indexes, -scores_float))
    if candidate_limit is not None:
        order = order[:candidate_limit]
    kept: list[int] = []
    while order.size and (
        detection_limit is None or len(kept) < detection_limit
    ):
        current = int(order[0])
        kept.append(current)
        remaining = order[1:]
        if not remaining.size:
            break
        overlaps = _pairwise_iou(
            boxes_array[current : current + 1], boxes_array[remaining]
        )[0]
        order = remaining[overlaps <= threshold]
    return kept


def tile_origins(
    width: int,
    height: int,
    tile: int = 640,
    overlap: Real = 0.2,
    max_tiles: int | None = None,
) -> list[tuple[int, int]]:
    """Return row-major tile origins that always include both far image edges."""

    image_width = _positive_int("width", width)
    image_height = _positive_int("height", height)
    tile_size = _positive_int("tile", tile)
    tile_limit = (
        None if max_tiles is None else _positive_int("max_tiles", max_tiles)
    )
    overlap_value = _finite_real("overlap", overlap)
    if not 0.0 <= overlap_value < 1.0:
        raise ValueError("overlap must be at least 0 and less than 1")
    if overlap_value == 0.0:
        stride = tile_size
    else:
        raw_stride = int(tile_size * (1.0 - overlap_value))
        stride = min(tile_size, max(1, raw_stride))

    def axis_count(length: int) -> int:
        final_origin = max(length - tile_size, 0)
        count = final_origin // stride + 1
        if (count - 1) * stride != final_origin:
            count += 1
        return count

    x_count = axis_count(image_width)
    y_count = axis_count(image_height)
    if tile_limit is not None and x_count * y_count > tile_limit:
        raise ValueError(f"tile grid exceeds max_tiles={tile_limit}")

    def axis_origins(length: int) -> list[int]:
        final_origin = max(length - tile_size, 0)
        values = list(range(0, final_origin + 1, stride))
        if values[-1] != final_origin:
            values.append(final_origin)
        return values

    x_origins = axis_origins(image_width)
    y_origins = axis_origins(image_height)
    return [(x, y) for y in y_origins for x in x_origins]
