from dataclasses import FrozenInstanceError, fields
from inspect import signature

import numpy as np
import pytest

from water_models.geometry import LetterboxTransform, box_iou, letterbox, nms, tile_origins


def test_letterbox_transform_is_frozen_and_maps_clipped_coordinates() -> None:
    transform = LetterboxTransform(
        scale=640 / 600,
        pad_x=0,
        pad_y=160,
        original_width=600,
        original_height=300,
    )

    assert transform.to_original((64, 192, 576, 448)) == pytest.approx(
        (60.0, 30.0, 540.0, 270.0)
    )
    assert transform.to_original((-20, 100, 700, 600)) == (
        0.0,
        0.0,
        600.0,
        300.0,
    )
    with pytest.raises(FrozenInstanceError):
        transform.scale = 2.0  # type: ignore[misc]


def test_letterbox_transform_preserves_exact_five_field_contract() -> None:
    assert [field.name for field in fields(LetterboxTransform)] == [
        "scale",
        "pad_x",
        "pad_y",
        "original_width",
        "original_height",
    ]


@pytest.mark.parametrize(
    "kwargs",
    (
        {"scale": 0},
        {"scale": -1},
        {"scale": np.inf},
        {"scale": np.nan},
        {"scale": True},
        {"pad_x": -1},
        {"pad_x": np.inf},
        {"pad_y": -1},
        {"pad_y": np.nan},
        {"original_width": 0},
        {"original_width": True},
        {"original_height": -1},
    ),
)
def test_letterbox_transform_rejects_invalid_fields(kwargs: dict[str, object]) -> None:
    values: dict[str, object] = {
        "scale": 1.0,
        "pad_x": 0,
        "pad_y": 0,
        "original_width": 20,
        "original_height": 10,
    }
    values.update(kwargs)

    with pytest.raises((TypeError, ValueError)):
        LetterboxTransform(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "box",
    (
        (0, 0, 1),
        (0, 0, 1, 1, 2),
        (np.nan, 0, 1, 1),
        (0, 0, np.inf, 1),
        (2, 0, 1, 1),
        (0, 2, 1, 1),
        (False, 0, 1, 1),
    ),
)
def test_to_original_rejects_invalid_boxes(box: tuple[object, ...]) -> None:
    transform = LetterboxTransform(1.0, 0, 0, 10, 10)

    with pytest.raises((TypeError, ValueError)):
        transform.to_original(box)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("shape", "expected_scale", "expected_pad"),
    (
        ((300, 600, 3), 640 / 600, (0, 160)),
        ((600, 300, 3), 640 / 600, (160, 0)),
        ((640, 640, 3), 1.0, (0, 0)),
        ((1, 1, 3), 640.0, (0, 0)),
    ),
)
def test_letterbox_shapes_scale_and_padding(
    shape: tuple[int, int, int],
    expected_scale: float,
    expected_pad: tuple[int, int],
) -> None:
    image = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)

    output, transform = letterbox(image)

    assert output.shape == (640, 640, 3)
    assert output.dtype == np.uint8
    assert output.flags.c_contiguous
    assert transform.scale == pytest.approx(expected_scale)
    assert (transform.pad_x, transform.pad_y) == expected_pad
    assert (transform.original_width, transform.original_height) == (shape[1], shape[0])


def test_letterbox_uses_114_padding_and_deterministic_odd_split() -> None:
    image = np.zeros((2, 4, 3), dtype=np.uint8)

    output, transform = letterbox(image, size=7)

    assert transform == LetterboxTransform(7 / 4, 0, 1, 4, 2)
    assert output.shape == (7, 7, 3)
    assert np.all(output[0] == 114)
    assert np.all(output[-2:] == 114)
    assert np.all(output[1:5] == 0)


def test_letterbox_known_box_round_trip() -> None:
    _, transform = letterbox(np.zeros((300, 600, 3), dtype=np.uint8))

    assert transform.to_original((64, 192, 576, 448)) == pytest.approx(
        (60.0, 30.0, 540.0, 270.0)
    )


def test_letterbox_uses_actual_resized_axes_for_inverse_transform() -> None:
    output, transform = letterbox(np.zeros((2, 3, 3), dtype=np.uint8), size=5)

    assert output.shape == (5, 5, 3)
    assert transform.scale == pytest.approx(5 / 3)
    assert transform.to_original((0, 1, 5, 4)) == pytest.approx((0, 0, 3, 2))


@pytest.mark.parametrize(
    ("shape", "size"),
    (((3, 7, 3), 11), ((7, 3, 3), 11), ((5, 13, 3), 17)),
)
def test_letterbox_content_boundary_round_trips_after_resize_rounding(
    shape: tuple[int, int, int], size: int
) -> None:
    _, transform = letterbox(np.zeros(shape, dtype=np.uint8), size=size)

    resized_width = max(1, int(round(shape[1] * transform.scale)))
    resized_height = max(1, int(round(shape[0] * transform.scale)))
    content_box = (
        transform.pad_x,
        transform.pad_y,
        transform.pad_x + resized_width,
        transform.pad_y + resized_height,
    )
    assert transform.to_original(content_box) == pytest.approx(
        (0, 0, shape[1], shape[0])
    )


def test_direct_transform_reconstructs_integer_resize_dimensions() -> None:
    transform = LetterboxTransform(2.0, 3, 5, 10, 20)

    assert transform.to_original((3, 5, 23, 45)) == (0.0, 0.0, 10.0, 20.0)


@pytest.mark.parametrize(
    ("image", "size"),
    (
        (np.empty((0, 5, 3), dtype=np.uint8), 640),
        (np.empty((5, 0, 3), dtype=np.uint8), 640),
        (np.zeros((5, 5), dtype=np.uint8), 640),
        (np.zeros((5, 5, 1), dtype=np.uint8), 640),
        (np.zeros((5, 5, 4), dtype=np.uint8), 640),
        (np.zeros((5, 5, 3), dtype=np.float32), 640),
        (np.zeros((5, 5, 3), dtype=np.uint8), 0),
        (np.zeros((5, 5, 3), dtype=np.uint8), -1),
        (np.zeros((5, 5, 3), dtype=np.uint8), True),
        (np.zeros((5, 5, 3), dtype=np.uint8), 1.5),
    ),
)
def test_letterbox_rejects_invalid_input(image: np.ndarray, size: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        letterbox(image, size=size)  # type: ignore[arg-type]


def test_letterbox_rejects_non_array() -> None:
    with pytest.raises(TypeError):
        letterbox([[[0, 0, 0]]])  # type: ignore[arg-type]


def test_box_iou_uses_continuous_coordinates_and_touching_is_zero() -> None:
    left = np.array([[0, 0, 10, 10], [0, 0, 100, 100]], dtype=np.float64)
    right = np.array([[10, 0, 20, 10], [5, 5, 100, 100]], dtype=np.float64)

    result = box_iou(left, right)

    assert result.shape == (2, 2)
    assert result[0, 0] == 0.0
    assert result[1, 1] == pytest.approx(9025 / 10000)


@pytest.mark.parametrize(
    "degenerate",
    (
        np.array([[0, 0, 0, 5]], dtype=np.float64),
        np.array([[0, 0, 5, 0]], dtype=np.float64),
    ),
)
def test_iou_and_nms_strictly_reject_zero_area_boxes(
    degenerate: np.ndarray,
) -> None:
    normal = np.array([[0, 0, 10, 10]], dtype=np.float64)

    with pytest.raises(ValueError):
        box_iou(degenerate, normal)
    with pytest.raises(ValueError):
        nms(degenerate, np.array([1.0]), 0.5)


def test_box_iou_still_rejects_reversed_coordinates() -> None:
    with pytest.raises(ValueError):
        box_iou(
            np.array([[1, 0, 0, 1]], dtype=np.float64),
            np.array([[0, 0, 1, 1]], dtype=np.float64),
        )


def test_box_iou_normalizes_huge_finite_coordinates_before_area_math() -> None:
    huge = np.array([[0, 0, 1e200, 1e200]], dtype=np.float64)

    with np.errstate(over="raise", invalid="raise"):
        result = box_iou(huge, huge)

    assert result[0, 0] == pytest.approx(1.0)


def test_box_iou_normalizes_each_axis_for_disparate_finite_scales() -> None:
    wide_and_short = np.array([[0, 0, 1e200, 1e-200]], dtype=np.float64)

    with np.errstate(over="raise", under="raise", invalid="raise"):
        result = box_iou(wide_and_short, wide_and_short)

    assert result[0, 0] == pytest.approx(1.0)


def test_box_iou_pair_scales_are_not_polluted_by_unrelated_boxes() -> None:
    boxes = np.array(
        [
            [0, 0, 1e200, 1e200],
            [0, 0, 1e-200, 1e-200],
            [0, 0, 1e-200, 1e-200],
        ],
        dtype=np.float64,
    )

    with np.errstate(over="raise", under="raise", invalid="raise"):
        overlaps = box_iou(boxes, boxes)

    assert overlaps[1, 2] == pytest.approx(1.0)


def test_nms_suppresses_tiny_duplicate_in_mixed_scale_batch() -> None:
    boxes = np.array(
        [
            [0, 0, 1e200, 1e200],
            [0, 0, 1e-200, 1e-200],
            [0, 0, 1e-200, 1e-200],
        ],
        dtype=np.float64,
    )

    assert nms(boxes, np.array([0.1, 0.9, 0.8]), 0.5) == [1, 0]


def test_nms_keeps_low_overlap_boxes_with_huge_finite_coordinates() -> None:
    boxes = np.array(
        [[0, 0, 1e200, 1e200], [9e199, 9e199, 1.9e200, 1.9e200]],
        dtype=np.float64,
    )

    overlap = box_iou(boxes[:1], boxes[1:])[0, 0]

    assert overlap == pytest.approx(0.01 / 1.99)
    assert overlap < 0.1
    assert nms(boxes, np.array([0.9, 0.8]), 0.1) == [0, 1]


@pytest.mark.parametrize(
    ("boxes", "scores", "threshold"),
    (
        (np.array([0, 0, 1, 1]), np.array([1.0]), 0.5),
        (np.zeros((1, 5)), np.array([1.0]), 0.5),
        (np.array([[0, 0, np.nan, 1]]), np.array([1.0]), 0.5),
        (np.array([[0, 0, np.inf, 1]]), np.array([1.0]), 0.5),
        (np.array([[1, 0, 0, 1]]), np.array([1.0]), 0.5),
        (np.array([[0, 0, 1, 1]]), np.array([[1.0]]), 0.5),
        (np.array([[0, 0, 1, 1]]), np.array([]), 0.5),
        (np.array([[0, 0, 1, 1]]), np.array([np.nan]), 0.5),
        (np.array([[0, 0, 1, 1]]), np.array([1.0]), True),
        (np.array([[0, 0, 1, 1]]), np.array([1.0]), -0.1),
        (np.array([[0, 0, 1, 1]]), np.array([1.0]), 1.1),
        (np.array([[0, 0, 1, 1]]), np.array([1.0]), np.inf),
        (np.array([[0, 0, 1, 1]]), np.array([1.0]), 10**1000),
    ),
)
def test_nms_rejects_invalid_inputs(
    boxes: np.ndarray,
    scores: np.ndarray,
    threshold: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        nms(boxes, scores, threshold)  # type: ignore[arg-type]


def test_nms_suppresses_high_overlap_and_keeps_stable_score_order() -> None:
    boxes = np.array(
        [[0, 0, 100, 100], [5, 5, 100, 100], [200, 200, 250, 250]],
        dtype=np.float32,
    )

    assert nms(boxes, np.array([0.9, 0.8, 0.7]), 0.5) == [0, 2]
    assert nms(boxes, np.array([0.9, 0.9, 0.9]), 0.5) == [0, 2]
    assert all(type(index) is int for index in nms(boxes, np.ones(3), 0.5))


def test_nms_keeps_boxes_that_only_touch() -> None:
    boxes = np.array([[0, 0, 10, 10], [10, 0, 20, 10]], dtype=np.float64)

    assert nms(boxes, np.array([0.5, 0.4]), 0.0) == [0, 1]


def test_nms_strictly_rejects_candidate_degenerate_after_clipping() -> None:
    transform = LetterboxTransform(1.0, 0, 0, 10, 10)
    clipped_outside = transform.to_original((20, 20, 30, 30))
    boxes = np.array([clipped_outside, (1, 1, 9, 9)], dtype=np.float64)

    assert clipped_outside == (10.0, 10.0, 10.0, 10.0)
    with pytest.raises(ValueError):
        nms(boxes, np.array([1.0, 0.5]), 0.5)


def test_nms_empty_input_returns_empty_list() -> None:
    assert nms(np.empty((0, 4)), np.empty((0,)), 0.5) == []


def test_nms_defaults_are_unlimited_and_keep_every_valid_detection() -> None:
    x1 = np.arange(301, dtype=np.float64) * 2
    boxes = np.column_stack((x1, np.zeros(301), x1 + 1, np.ones(301)))
    parameters = signature(nms).parameters

    assert parameters["max_candidates"].default is None
    assert parameters["max_detections"].default is None
    assert nms(boxes, np.ones(301), 0.5) == list(range(301))
    assert nms(
        boxes,
        np.ones(301),
        0.5,
        max_candidates=None,
        max_detections=None,
    ) == list(range(301))


def test_nms_limits_sorted_candidates_and_output_count() -> None:
    boxes = np.array(
        [[0, 0, 1, 1], [2, 0, 3, 1], [4, 0, 5, 1], [6, 0, 7, 1]],
        dtype=np.float64,
    )
    scores = np.array([0.1, 0.9, 0.8, 0.7])

    assert nms(
        boxes,
        scores,
        0.5,
        max_candidates=2,
        max_detections=1,
    ) == [1]
    assert nms(
        boxes,
        np.ones(4),
        0.5,
        max_candidates=3,
        max_detections=3,
    ) == [0, 1, 2]


@pytest.mark.parametrize(
    ("max_candidates", "max_detections"),
    ((0, 1), (True, 1), (1, 0), (1, False), (1.5, 1), (1, 1.5)),
)
def test_nms_rejects_invalid_processing_limits(
    max_candidates: object, max_detections: object
) -> None:
    with pytest.raises((TypeError, ValueError)):
        nms(
            np.empty((0, 4)),
            np.empty((0,)),
            0.5,
            max_candidates=max_candidates,  # type: ignore[arg-type]
            max_detections=max_detections,  # type: ignore[arg-type]
        )


def test_tile_origins_covers_edges_in_row_major_order() -> None:
    origins = tile_origins(2032, 2427)

    assert origins[0] == (0, 0)
    assert origins[-1] == (1392, 1787)
    assert origins[:4] == [(0, 0), (512, 0), (1024, 0), (1392, 0)]
    assert origins[4] == (0, 512)
    assert len(origins) == len(set(origins))


@pytest.mark.parametrize(
    ("width", "height", "tile", "overlap", "expected"),
    (
        (100, 200, 640, 0.2, [(0, 0)]),
        (640, 640, 640, 0.2, [(0, 0)]),
        (700, 640, 640, 0.0, [(0, 0), (60, 0)]),
        (1280, 640, 640, 0.0, [(0, 0), (640, 0)]),
    ),
)
def test_tile_origins_handles_small_exact_and_duplicate_edges(
    width: int,
    height: int,
    tile: int,
    overlap: float,
    expected: list[tuple[int, int]],
) -> None:
    assert tile_origins(width, height, tile, overlap) == expected


def test_tile_origins_enforces_max_tiles_before_materializing() -> None:
    assert len(tile_origins(100, 100, tile=10, overlap=0, max_tiles=100)) == 100

    with pytest.raises(ValueError):
        tile_origins(100_000, 100_000, tile=1, overlap=0, max_tiles=10_000)


def test_tile_origins_default_is_unlimited_and_covers_complete_grid() -> None:
    assert signature(tile_origins).parameters["max_tiles"].default is None
    assert len(tile_origins(101, 101, tile=1, overlap=0)) == 10_201


def test_tile_origins_overlap_zero_handles_large_integer_dimensions_exactly() -> None:
    assert tile_origins(
        2**53 + 7,
        1,
        tile=2**53 + 3,
        overlap=0,
    ) == [(0, 0), (4, 0)]


def test_tile_origins_clamps_rounded_stride_to_prevent_large_integer_gaps() -> None:
    tile = 2**53 + 3

    assert tile_origins(tile * 3, 1, tile=tile, overlap=1e-20) == [
        (0, 0),
        (tile, 0),
        (tile * 2, 0),
    ]


def test_tile_origins_clamps_positive_overlap_stride_to_at_least_one() -> None:
    origins = tile_origins(10, 10, tile=5, overlap=0.9999999999999999)

    assert len(origins) == 36
    assert origins[-1] == (5, 5)


@pytest.mark.parametrize(
    ("kwargs", "error_type"),
    (
        ({"max_tiles": 0}, ValueError),
        ({"max_tiles": True}, TypeError),
    ),
)
def test_tile_origins_rejects_invalid_explicit_limit(
    kwargs: dict[str, object], error_type: type[Exception]
) -> None:
    values: dict[str, object] = {
        "width": 100,
        "height": 100,
        "tile": 10,
        "overlap": 0.0,
        "max_tiles": 10_000,
    }
    values.update(kwargs)

    with pytest.raises(error_type):
        tile_origins(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("width", "height", "tile", "overlap"),
    (
        (0, 10, 5, 0.2),
        (-1, 10, 5, 0.2),
        (True, 10, 5, 0.2),
        (10, 0, 5, 0.2),
        (10, False, 5, 0.2),
        (10, 10, 0, 0.2),
        (10, 10, True, 0.2),
        (10, 10, 5, True),
        (10, 10, 5, -0.1),
        (10, 10, 5, 1.0),
        (10, 10, 5, np.nan),
        (10, 10, 5, np.inf),
    ),
)
def test_tile_origins_rejects_invalid_parameters(
    width: object,
    height: object,
    tile: object,
    overlap: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        tile_origins(width, height, tile, overlap)  # type: ignore[arg-type]
