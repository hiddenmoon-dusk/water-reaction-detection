from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest

import water_models.inference as inference_module
from water_models.inference import (
    Detection,
    InferenceError,
    classify_source,
    classify_tflite,
    decode_yolo,
    detect_default,
    detect_source,
    detect_tflite,
    detect_tiled,
    suppress_detections,
)


def _detail(index: int, shape: tuple[int, ...], dtype: object) -> dict[str, object]:
    return {
        "index": index,
        "shape": np.array(shape, dtype=np.int32),
        "shape_signature": np.array(shape, dtype=np.int32),
        "dtype": dtype,
    }


class FakeInterpreter:
    def __init__(
        self,
        *,
        input_shape: tuple[int, ...],
        output: np.ndarray,
        input_dtype: object = np.float32,
        output_dtype: object = np.float32,
        input_signature: tuple[int, ...] | None = None,
        output_signature: tuple[int, ...] | None = None,
        fail_at: str | None = None,
    ) -> None:
        self.inputs = [_detail(7, input_shape, input_dtype)]
        self.outputs = [_detail(11, tuple(output.shape), output_dtype)]
        if input_signature is not None:
            self.inputs[0]["shape_signature"] = np.array(input_signature)
        if output_signature is not None:
            self.outputs[0]["shape_signature"] = np.array(output_signature)
        self.output = output
        self.fail_at = fail_at
        self.last_input: np.ndarray | None = None
        self.invoked = False

    def get_input_details(self) -> list[dict[str, object]]:
        if self.fail_at == "details":
            raise RuntimeError("details broke")
        return self.inputs

    def get_output_details(self) -> list[dict[str, object]]:
        return self.outputs

    def set_tensor(self, index: int, value: np.ndarray) -> None:
        assert index == 7
        if self.fail_at == "set":
            raise RuntimeError("set broke")
        self.last_input = value.copy()

    def invoke(self) -> None:
        if self.fail_at == "invoke":
            raise RuntimeError("invoke broke")
        self.invoked = True

    def get_tensor(self, index: int) -> np.ndarray:
        assert index == 11
        if self.fail_at == "get":
            raise RuntimeError("get broke")
        return self.output.copy()


def _detector_output(*columns: tuple[float, float, float, float, float]) -> np.ndarray:
    return np.asarray(columns, dtype=np.float32).T[None, :, :]


def test_detection_is_frozen_and_strictly_valid() -> None:
    detection = Detection(1, 2, 3, 4, 0.5)
    assert detection == Detection(1.0, 2.0, 3.0, 4.0, 0.5)
    with pytest.raises(FrozenInstanceError):
        detection.score = 0.9  # type: ignore[misc]


@pytest.mark.parametrize(
    "values, error",
    [
        ((0, 0, 0, 1, 0.5), ValueError),
        ((0, 0, 1, 0, 0.5), ValueError),
        ((2, 0, 1, 1, 0.5), ValueError),
        ((0, 0, 1, 1, -0.1), ValueError),
        ((0, 0, 1, 1, 1.1), ValueError),
        ((0, 0, np.nan, 1, 0.5), ValueError),
        ((0, 0, 1, 1, np.inf), ValueError),
        ((False, 0, 1, 1, 0.5), TypeError),
    ],
)
def test_detection_rejects_invalid_values(
    values: tuple[object, ...], error: type[Exception]
) -> None:
    with pytest.raises(error):
        Detection(*values)  # type: ignore[arg-type]


def test_decode_yolo_maps_actual_integer_letterbox_and_stably_suppresses() -> None:
    output = _detector_output(
        (320, 320, 320, 160, 0.9),
        (322, 320, 320, 160, 0.7),
    )
    detections = decode_yolo(
        output, original_size=(600, 300), conf=0.3, nms_iou=0.45
    )
    assert len(detections) == 1
    assert detections[0].x1 == 150
    assert detections[0].y1 == 75
    assert detections[0].x2 == 450
    assert detections[0].y2 == 225
    assert detections[0].score == pytest.approx(0.9)

    odd = decode_yolo(
        _detector_output((320, 319.5, 640, 427, 0.8)),
        original_size=(3, 2),
        conf=0,
    )
    assert [(item.x1, item.y1, item.x2, item.y2) for item in odd] == [
        (0, 0, 3, 2)
    ]
    assert odd[0].score == pytest.approx(0.8)


def test_decode_yolo_filters_confidence_and_degenerate_clips_before_nms() -> None:
    output = _detector_output(
        (-20, 320, 10, 20, 0.99),
        (320, 320, 100, 100, 0.29),
        (320, 320, 100, 100, 0.30),
    )
    detections = decode_yolo(output, original_size=(640, 640), conf=0.3)
    assert [(item.x1, item.y1, item.x2, item.y2) for item in detections] == [
        (270, 270, 370, 370)
    ]
    assert detections[0].score == pytest.approx(0.3)


@pytest.mark.parametrize(
    "output, error",
    [
        (np.zeros((1, 5, 0), dtype=np.float32), ValueError),
        (np.zeros((5, 2), dtype=np.float32), ValueError),
        (np.zeros((2, 5, 2), dtype=np.float32), ValueError),
        (np.zeros((1, 6, 2), dtype=np.float32), ValueError),
        (np.zeros((1, 5, 2), dtype=np.float64), TypeError),
        (np.full((1, 5, 2), np.nan, dtype=np.float32), ValueError),
    ],
)
def test_decode_yolo_rejects_bad_raw_tensor(
    output: np.ndarray, error: type[Exception]
) -> None:
    with pytest.raises(error):
        decode_yolo(output, original_size=(640, 640))


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"original_size": (0, 1)}, ValueError),
        ({"original_size": (1, 1, 1)}, ValueError),
        ({"original_size": (True, 1)}, TypeError),
        ({"conf": np.nan}, ValueError),
        ({"conf": -0.1}, ValueError),
        ({"nms_iou": 1.1}, ValueError),
        ({"max_candidates": 0}, ValueError),
        ({"max_detections": False}, TypeError),
    ],
)
def test_decode_yolo_rejects_invalid_options(
    kwargs: dict[str, object], error: type[Exception]
) -> None:
    kwargs = {"original_size": (640, 640), **kwargs}
    with pytest.raises(error):
        decode_yolo(
            np.zeros((1, 5, 0), dtype=np.float32), **kwargs  # type: ignore[arg-type]
        )


def test_decode_yolo_uses_explicit_stable_candidate_and_detection_caps() -> None:
    columns = [(10.0 + index * 2, 10.0, 1.0, 1.0, 0.5) for index in range(8)]
    detections = decode_yolo(
        _detector_output(*columns),
        original_size=(640, 640),
        conf=0,
        max_candidates=5,
        max_detections=3,
    )
    assert [detection.x1 for detection in detections] == [9.5, 11.5, 13.5]


def test_decode_yolo_bounds_candidates_before_nms_and_keeps_stable_top_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[int, list[float]]] = []

    def observe_nms(
        boxes: np.ndarray,
        scores: np.ndarray,
        iou_threshold: float,
        *,
        max_candidates: int,
        max_detections: int,
    ) -> list[int]:
        seen.append((len(boxes), scores.tolist()))
        assert len(boxes) <= 1
        return list(range(len(boxes)))

    monkeypatch.setattr(inference_module, "nms", observe_nms)
    scores = np.linspace(0.001, 1.0, 1000, dtype=np.float32)
    output = np.zeros((1, 5, 1000), dtype=np.float32)
    output[0, 0:4, :] = np.array([[320], [320], [10], [10]], dtype=np.float32)
    output[0, 4, :] = scores
    detections = decode_yolo(
        output,
        original_size=(640, 640),
        conf=0,
        max_candidates=1,
    )
    assert [item.score for item in detections] == [pytest.approx(1.0)]

    tied = decode_yolo(
        _detector_output(
            (10, 10, 2, 2, 0.8),
            (20, 10, 2, 2, 0.8),
            (30, 10, 2, 2, 0.8),
        ),
        original_size=(640, 640),
        conf=0,
        max_candidates=1,
    )
    assert tied[0].x1 == 9
    assert seen == [(1, [1.0]), (1, [pytest.approx(0.8)])]


def test_decode_yolo_degenerate_high_score_does_not_starve_valid_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        inference_module,
        "nms",
        lambda boxes, scores, *args, **kwargs: list(range(len(boxes))),
    )
    detections = decode_yolo(
        _detector_output(
            (-100, 10, 2, 2, 0.99),
            (20, 10, 2, 2, 0.8),
        ),
        original_size=(640, 640),
        conf=0,
        max_candidates=1,
    )
    assert len(detections) == 1
    assert detections[0].score == pytest.approx(0.8)


def test_decode_yolo_returns_empty_when_no_confidence_meets_threshold() -> None:
    assert decode_yolo(
        _detector_output((10, 10, 1, 1, 0.29)),
        original_size=(640, 640),
        conf=0.3,
    ) == []


def test_decode_yolo_rejects_confidence_outside_probability_range() -> None:
    with pytest.raises(ValueError, match="confidence"):
        decode_yolo(
            _detector_output((10, 10, 1, 1, -0.01)),
            original_size=(640, 640),
            conf=0,
        )


class TorchLike:
    def __init__(self, value: object) -> None:
        self.value = value
        self.calls: list[str] = []

    def detach(self) -> TorchLike:
        self.calls.append("detach")
        return self

    def cpu(self) -> TorchLike:
        self.calls.append("cpu")
        return self

    def numpy(self) -> np.ndarray:
        self.calls.append("numpy")
        return np.asarray(self.value)


class TensorFlowLike:
    def __init__(self, value: object, failure: Exception | None = None) -> None:
        self.value = value
        self.failure = failure
        self.calls: list[str] = []

    def cpu(self) -> TensorFlowLike:
        self.calls.append("cpu")
        raise AssertionError("TensorFlow-like cpu() must not be called")

    def numpy(self) -> np.ndarray:
        self.calls.append("numpy")
        if self.failure is not None:
            raise self.failure
        return np.asarray(self.value)


def test_source_detector_calls_exact_api_and_parses_torch_like_tensors() -> None:
    xyxy = TorchLike([[1, 2, 10, 20]])
    confidence = TorchLike([0.75])

    class Model:
        def __call__(self, image: np.ndarray, **kwargs: object) -> object:
            assert image is rgb
            assert kwargs == {"conf": 0.3, "verbose": False}
            return [SimpleNamespace(boxes=SimpleNamespace(xyxy=xyxy, conf=confidence))]

    rgb = np.zeros((20, 10, 3), dtype=np.uint8)
    assert detect_source(Model(), rgb) == [Detection(1, 2, 10, 20, 0.75)]
    assert xyxy.calls == ["detach", "cpu", "numpy"]
    assert confidence.calls == ["detach", "cpu", "numpy"]


def test_tensorflow_like_value_uses_numpy_without_deprecated_cpu() -> None:
    output = TensorFlowLike([[0.75]])
    assert classify_source(lambda *args, **kwargs: output, _rgb()) == pytest.approx(
        0.75
    )
    assert output.calls == ["numpy"]


def test_tensor_conversion_failure_preserves_external_cause() -> None:
    failure = RuntimeError("numpy conversion broke")
    output = TensorFlowLike([[0.75]], failure=failure)
    with pytest.raises(InferenceError) as caught:
        classify_source(lambda *args, **kwargs: output, _rgb())
    assert caught.value.__cause__ is failure
    assert output.calls == ["numpy"]


def test_source_detector_accepts_empty_boxes() -> None:
    result = SimpleNamespace(
        boxes=SimpleNamespace(
            xyxy=np.empty((0, 4), dtype=np.float32),
            conf=np.empty((0,), dtype=np.float32),
        )
    )
    assert detect_source(lambda *args, **kwargs: [result], _rgb()) == []


def test_source_detector_defensively_filters_results_below_confidence() -> None:
    result = SimpleNamespace(
        boxes=SimpleNamespace(
            xyxy=np.array([[0, 0, 1, 1], [1, 1, 2, 2]], dtype=np.float32),
            conf=np.array([0.29, 0.3], dtype=np.float32),
        )
    )
    detections = detect_source(lambda *args, **kwargs: [result], _rgb(), conf=0.3)
    assert len(detections) == 1
    assert detections[0].score == pytest.approx(0.3)


def test_source_detector_applies_local_nms_without_changing_model_kwargs() -> None:
    result = SimpleNamespace(
        boxes=SimpleNamespace(
            xyxy=np.array(
                [[0, 0, 10, 10], [1, 1, 11, 11]], dtype=np.float32
            ),
            conf=np.array([0.9, 0.8], dtype=np.float32),
        )
    )
    calls: list[dict[str, object]] = []

    def model(image: np.ndarray, **kwargs: object) -> object:
        calls.append(kwargs)
        return [result]

    strict = detect_source(model, _rgb(20, 20), nms_iou=0.45)
    wide = detect_source(model, _rgb(20, 20), nms_iou=0.7)

    assert [item.score for item in strict] == [pytest.approx(0.9)]
    assert [item.score for item in wide] == [pytest.approx(0.9), pytest.approx(0.8)]
    assert calls == [
        {"conf": 0.3, "verbose": False},
        {"conf": 0.3, "verbose": False},
    ]


def test_source_detector_can_opt_in_to_matching_upstream_nms_kwargs() -> None:
    result = SimpleNamespace(
        boxes=SimpleNamespace(
            xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
            conf=np.array([0.9], dtype=np.float32),
        )
    )
    calls: list[dict[str, object]] = []

    def model(image: np.ndarray, **kwargs: object) -> object:
        calls.append(kwargs)
        return [result]

    detect_source(
        model,
        _rgb(20, 20),
        nms_iou=0.83,
        max_candidates=1_000,
        configure_model_nms=True,
    )
    assert calls == [
        {
            "conf": 0.3,
            "verbose": False,
            "iou": 0.83,
            "max_det": 1_000,
        }
    ]


def test_source_detector_opt_in_prevents_upstream_default_nms_overfiltering() -> None:
    all_boxes = np.array(
        [[0, 0, 10, 10], [0.5, 0.5, 10.5, 10.5]], dtype=np.float32
    )
    all_scores = np.array([0.9, 0.8], dtype=np.float32)

    def model(image: np.ndarray, **kwargs: object) -> object:
        count = 2 if kwargs.get("iou") == 0.83 else 1
        return [
            SimpleNamespace(
                boxes=SimpleNamespace(
                    xyxy=all_boxes[:count], conf=all_scores[:count]
                )
            )
        ]

    default = detect_source(model, _rgb(20, 20), nms_iou=0.83)
    configured = detect_source(
        model,
        _rgb(20, 20),
        nms_iou=0.83,
        configure_model_nms=True,
    )
    assert [item.score for item in default] == [pytest.approx(0.9)]
    assert [item.score for item in configured] == [
        pytest.approx(0.9),
        pytest.approx(0.8),
    ]


def test_source_detector_opt_in_forwards_candidate_cap_past_upstream_300() -> None:
    boxes = np.array(
        [[index * 2, 0, index * 2 + 1, 1] for index in range(350)],
        dtype=np.float32,
    )
    scores = np.linspace(0.5, 0.9, 350, dtype=np.float32)

    def model(image: np.ndarray, **kwargs: object) -> object:
        count = int(kwargs.get("max_det", 300))
        return [
            SimpleNamespace(
                boxes=SimpleNamespace(xyxy=boxes[:count], conf=scores[:count])
            )
        ]

    default = detect_source(
        model,
        _rgb(2, 700),
        max_candidates=1_000,
        max_detections=1_000,
    )
    configured = detect_source(
        model,
        _rgb(2, 700),
        max_candidates=1_000,
        max_detections=1_000,
        configure_model_nms=True,
    )
    assert len(default) == 300
    assert len(configured) == 350


def test_source_detector_default_supports_legacy_model_without_nms_kwargs() -> None:
    result = SimpleNamespace(
        boxes=SimpleNamespace(
            xyxy=np.empty((0, 4), dtype=np.float32),
            conf=np.empty((0,), dtype=np.float32),
        )
    )

    def legacy_model(
        image: np.ndarray, *, conf: float, verbose: bool
    ) -> object:
        return [result]

    assert detect_source(legacy_model, _rgb()) == []
    with pytest.raises(InferenceError) as caught:
        detect_source(legacy_model, _rgb(), configure_model_nms=True)
    assert isinstance(caught.value.__cause__, TypeError)


def test_source_detector_nms_is_stable_for_ties_and_honors_caps() -> None:
    tied = SimpleNamespace(
        boxes=SimpleNamespace(
            xyxy=np.array([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=np.float32),
            conf=np.array([0.8, 0.8], dtype=np.float32),
        )
    )
    detections = detect_source(
        lambda *args, **kwargs: [tied],
        _rgb(20, 20),
        nms_iou=0.45,
        max_candidates=1,
        max_detections=1,
    )
    assert len(detections) == 1
    assert detections[0].x1 == 0

    capped = SimpleNamespace(
        boxes=SimpleNamespace(
            xyxy=np.array(
                [[0, 0, 5, 5], [10, 0, 15, 5], [20, 0, 25, 5]],
                dtype=np.float32,
            ),
            conf=np.array([0.7, 0.8, 0.9], dtype=np.float32),
        )
    )
    detections = detect_source(
        lambda *args, **kwargs: [capped],
        _rgb(10, 30),
        max_candidates=2,
        max_detections=1,
    )
    assert len(detections) == 1
    assert detections[0].score == pytest.approx(0.9)


def test_source_detector_empty_result_stays_empty_after_local_nms() -> None:
    result = SimpleNamespace(
        boxes=SimpleNamespace(
            xyxy=np.empty((0, 4), dtype=np.float32),
            conf=np.empty((0,), dtype=np.float32),
        )
    )
    assert detect_source(
        lambda *args, **kwargs: [result],
        _rgb(),
        nms_iou=0,
        max_candidates=1,
        max_detections=1,
    ) == []


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"conf": np.nan}, ValueError),
        ({"nms_iou": -0.1}, ValueError),
        ({"nms_iou": 1.1}, ValueError),
        ({"max_candidates": 0}, ValueError),
        ({"max_candidates": False}, TypeError),
        ({"max_detections": 0}, ValueError),
        ({"max_detections": 1.5}, TypeError),
        ({"configure_model_nms": 1}, TypeError),
        ({"configure_model_nms": None}, TypeError),
    ],
)
def test_source_detector_validates_nms_options_before_model_call(
    kwargs: dict[str, object], error: type[Exception]
) -> None:
    calls = 0

    def model(*args: object, **model_kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return []

    with pytest.raises(error):
        detect_source(model, _rgb(), **kwargs)  # type: ignore[arg-type]
    assert calls == 0


@pytest.mark.parametrize(
    "results",
    [
        [],
        [SimpleNamespace(boxes=None), SimpleNamespace(boxes=None)],
        [SimpleNamespace(boxes=SimpleNamespace(xyxy=np.zeros((1, 5)), conf=[0.5]))],
        [SimpleNamespace(boxes=SimpleNamespace(xyxy=[[0, 0, 1, 1]], conf=[[0.5]]))],
        [SimpleNamespace(boxes=SimpleNamespace(xyxy=[[0, 0, 1, 1]], conf=[np.nan]))],
    ],
)
def test_source_detector_rejects_bad_result_contract(results: object) -> None:
    with pytest.raises(InferenceError):
        detect_source(lambda *args, **kwargs: results, _rgb())


def test_source_detector_wraps_model_failure_with_cause() -> None:
    failure = RuntimeError("model broke")

    def model(*args: object, **kwargs: object) -> object:
        raise failure

    with pytest.raises(InferenceError) as caught:
        detect_source(model, _rgb())
    assert caught.value.__cause__ is failure


def _rgb(height: int = 8, width: int = 10, value: int = 0) -> np.ndarray:
    return np.full((height, width, 3), value, dtype=np.uint8)


@pytest.mark.parametrize(
    "image, error",
    [
        (np.zeros((2, 2), dtype=np.uint8), ValueError),
        (np.zeros((0, 2, 3), dtype=np.uint8), ValueError),
        (np.zeros((2, 2, 3), dtype=np.float32), TypeError),
        ([[[0, 0, 0]]], TypeError),
    ],
)
def test_public_image_adapters_reject_invalid_rgb_input(
    image: object, error: type[Exception]
) -> None:
    with pytest.raises(error):
        detect_source(lambda *args, **kwargs: [], image)  # type: ignore[arg-type]
    with pytest.raises(error):
        classify_source(lambda batch: [[0.5]], image)  # type: ignore[arg-type]


def test_tflite_detector_preprocesses_normalized_rgb_and_decodes() -> None:
    interpreter = FakeInterpreter(
        input_shape=(1, 640, 640, 3),
        output=_detector_output((0.5, 0.5, 0.5, 0.25, 0.9)),
    )
    image = _rgb(300, 600, 255)
    detections = detect_tflite(interpreter, image)
    assert [(item.x1, item.y1, item.x2, item.y2) for item in detections] == [
        (150, 75, 450, 225)
    ]
    assert detections[0].score == pytest.approx(0.9)
    assert interpreter.invoked
    assert interpreter.last_input is not None
    assert interpreter.last_input.shape == (1, 640, 640, 3)
    assert interpreter.last_input.dtype == np.float32
    assert interpreter.last_input.max() == 1.0
    assert interpreter.last_input.min() == pytest.approx(114 / 255)


def test_tflite_detector_explicit_raw_float_contract_does_not_normalize() -> None:
    interpreter = FakeInterpreter(
        input_shape=(1, 640, 640, 3),
        output=np.zeros((1, 5, 1), dtype=np.float32),
    )
    detect_tflite(interpreter, _rgb(value=255), normalize=False)
    assert interpreter.last_input is not None
    assert interpreter.last_input.max() == 255.0


@pytest.mark.parametrize(
    "mutate",
    [
        lambda item: item.inputs.append(_detail(8, (1, 640, 640, 3), np.float32)),
        lambda item: item.outputs.append(_detail(12, (1, 5, 1), np.float32)),
        lambda item: item.inputs.__setitem__(
            0, _detail(7, (1, 320, 320, 3), np.float32)
        ),
        lambda item: item.inputs.__setitem__(0, _detail(7, (1, 640, 640, 3), np.uint8)),
        lambda item: item.outputs.__setitem__(0, _detail(11, (1, 6, 1), np.float32)),
        lambda item: item.outputs.__setitem__(0, _detail(11, (1, 5, 1), np.float64)),
        lambda item: item.inputs[0].__setitem__(
            "shape_signature", np.array((1, -1, 640, 3))
        ),
        lambda item: item.outputs[0].__setitem__(
            "shape_signature", np.array((1, 5, -1))
        ),
    ],
)
def test_tflite_detector_rejects_bad_tensor_contract(mutate: object) -> None:
    interpreter = FakeInterpreter(
        input_shape=(1, 640, 640, 3),
        output=np.zeros((1, 5, 1), dtype=np.float32),
    )
    mutate(interpreter)  # type: ignore[operator]
    with pytest.raises(InferenceError):
        detect_tflite(interpreter, _rgb())


def test_tflite_detector_rejects_malformed_tensor_details_as_inference_error() -> None:
    interpreter = FakeInterpreter(
        input_shape=(1, 640, 640, 3),
        output=np.zeros((1, 5, 1), dtype=np.float32),
    )
    interpreter.inputs = None  # type: ignore[assignment]
    with pytest.raises(InferenceError):
        detect_tflite(interpreter, _rgb())


@pytest.mark.parametrize("dynamic_tensor", ["input", "output"])
def test_tflite_detector_rejects_dynamic_batch_before_inference(
    dynamic_tensor: str,
) -> None:
    interpreter = FakeInterpreter(
        input_shape=(1, 640, 640, 3),
        input_signature=(
            (-1, 640, 640, 3)
            if dynamic_tensor == "input"
            else (1, 640, 640, 3)
        ),
        output=np.zeros((1, 5, 1), dtype=np.float32),
        output_signature=(
            (-1, 5, 1) if dynamic_tensor == "output" else (1, 5, 1)
        ),
    )
    with pytest.raises(InferenceError):
        detect_tflite(interpreter, _rgb())
    assert interpreter.last_input is None
    assert interpreter.invoked is False


@pytest.mark.parametrize("failure_point", ["details", "set", "invoke", "get"])
def test_tflite_detector_wraps_interpreter_failures(failure_point: str) -> None:
    interpreter = FakeInterpreter(
        input_shape=(1, 640, 640, 3),
        output=np.zeros((1, 5, 1), dtype=np.float32),
        fail_at=failure_point,
    )
    with pytest.raises(InferenceError) as caught:
        detect_tflite(interpreter, _rgb())
    assert isinstance(caught.value.__cause__, RuntimeError)


def test_source_classifier_passes_resized_raw_float_pixels_and_training_false() -> None:
    seen: dict[str, object] = {}

    def model(batch: np.ndarray, *, training: bool) -> np.ndarray:
        seen["batch"] = batch
        seen["training"] = training
        return np.array([[0.625]], dtype=np.float32)

    assert classify_source(model, _rgb(40, 20, 255)) == pytest.approx(0.625)
    batch = seen["batch"]
    assert isinstance(batch, np.ndarray)
    assert batch.shape == (1, 128, 128, 3)
    assert batch.dtype == np.float32
    assert batch.max() == 255.0
    assert seen["training"] is False


def test_source_classifier_falls_back_for_callable_without_training_keyword() -> None:
    class Model:
        def __call__(self, batch: np.ndarray) -> np.ndarray:
            return np.array(0.25, dtype=np.float32)

    assert classify_source(Model(), _rgb()) == pytest.approx(0.25)


@pytest.mark.parametrize(
    "output",
    [
        np.array([0.5], dtype=np.float32),
        np.array([[0.5, 0.4]], dtype=np.float32),
        np.array([[np.nan]], dtype=np.float32),
        np.array([[1.1]], dtype=np.float32),
        "not-a-number",
    ],
)
def test_source_classifier_rejects_bad_output(output: np.ndarray) -> None:
    with pytest.raises(InferenceError):
        classify_source(lambda *args, **kwargs: output, _rgb())


def test_source_classifier_wraps_call_failure_and_preserves_cause() -> None:
    failure = RuntimeError("classifier broke")

    def model(*args: object, **kwargs: object) -> object:
        raise failure

    with pytest.raises(InferenceError) as caught:
        classify_source(model, _rgb())
    assert caught.value.__cause__ is failure


def test_tflite_classifier_uses_raw_float_pixels_and_exact_indexes() -> None:
    interpreter = FakeInterpreter(
        input_shape=(1, 128, 128, 3),
        output=np.array([[0.8]], dtype=np.float32),
    )
    probability = classify_tflite(interpreter, _rgb(40, 20, 255))
    assert interpreter.last_input is not None
    assert interpreter.last_input.shape == (1, 128, 128, 3)
    assert interpreter.last_input.dtype == np.float32
    assert interpreter.last_input.max() == 255.0
    assert interpreter.invoked
    assert probability == pytest.approx(0.8)


@pytest.mark.parametrize(
    "input_shape, output, input_dtype, output_dtype",
    [
        ((1, 64, 64, 3), np.array([[0.8]], dtype=np.float32), np.float32, np.float32),
        ((1, 128, 128, 3), np.array([0.8], dtype=np.float32), np.float32, np.float32),
        ((1, 128, 128, 3), np.array([[0.8]], dtype=np.float32), np.uint8, np.float32),
        ((1, 128, 128, 3), np.array([[0.8]], dtype=np.float64), np.float32, np.float64),
    ],
)
def test_tflite_classifier_rejects_bad_contract(
    input_shape: tuple[int, ...],
    output: np.ndarray,
    input_dtype: object,
    output_dtype: object,
) -> None:
    interpreter = FakeInterpreter(
        input_shape=input_shape,
        output=output,
        input_dtype=input_dtype,
        output_dtype=output_dtype,
    )
    with pytest.raises(InferenceError):
        classify_tflite(interpreter, _rgb())


@pytest.mark.parametrize("target", ["input", "output"])
@pytest.mark.parametrize("bad_index", [7.0, "7", -1, True])
def test_tflite_rejects_invalid_tensor_index_before_inference(
    target: str, bad_index: object
) -> None:
    interpreter = FakeInterpreter(
        input_shape=(1, 128, 128, 3),
        output=np.array([[0.8]], dtype=np.float32),
    )
    details = interpreter.inputs if target == "input" else interpreter.outputs
    details[0]["index"] = bad_index
    with pytest.raises(InferenceError, match="index"):
        classify_tflite(interpreter, _rgb())
    assert interpreter.last_input is None
    assert interpreter.invoked is False


def test_tflite_accepts_nonnegative_numpy_integer_tensor_indexes() -> None:
    interpreter = FakeInterpreter(
        input_shape=(1, 128, 128, 3),
        output=np.array([[0.8]], dtype=np.float32),
    )
    interpreter.inputs[0]["index"] = np.int64(7)
    interpreter.outputs[0]["index"] = np.int32(11)
    assert classify_tflite(interpreter, _rgb()) == pytest.approx(0.8)


@pytest.mark.parametrize("target", ["input", "output"])
@pytest.mark.parametrize(
    "field, bad_value",
    [
        ("quantization", (0.5, 0)),
        ("quantization", (0.0, 1)),
        ("quantization", (0.0,)),
        ("quantization", 0),
        ("quantization", (False, 0)),
        ("quantization_parameters", {}),
        (
            "quantization_parameters",
            {"scales": [1.0], "zero_points": []},
        ),
        (
            "quantization_parameters",
            {"scales": [], "zero_points": [0]},
        ),
        ("quantization_parameters", None),
    ],
)
def test_float32_tflite_rejects_quantized_or_malformed_metadata_before_inference(
    target: str, field: str, bad_value: object
) -> None:
    interpreter = FakeInterpreter(
        input_shape=(1, 128, 128, 3),
        output=np.array([[0.8]], dtype=np.float32),
    )
    details = interpreter.inputs if target == "input" else interpreter.outputs
    details[0][field] = bad_value
    with pytest.raises(InferenceError, match="quantization"):
        classify_tflite(interpreter, _rgb())
    assert interpreter.last_input is None
    assert interpreter.invoked is False


def test_float32_tflite_accepts_explicit_unquantized_metadata() -> None:
    interpreter = FakeInterpreter(
        input_shape=(1, 128, 128, 3),
        output=np.array([[0.8]], dtype=np.float32),
    )
    for detail in (*interpreter.inputs, *interpreter.outputs):
        detail["quantization"] = (0.0, 0)
        detail["quantization_parameters"] = {
            "scales": np.array([], dtype=np.float32),
            "zero_points": np.array([], dtype=np.int32),
            "quantized_dimension": 0,
        }
    assert classify_tflite(interpreter, _rgb()) == pytest.approx(0.8)


def test_tflite_classifier_rejects_dynamic_and_invalid_probability() -> None:
    dynamic = FakeInterpreter(
        input_shape=(1, 128, 128, 3),
        input_signature=(1, -1, 128, 3),
        output=np.array([[0.8]], dtype=np.float32),
    )
    with pytest.raises(InferenceError):
        classify_tflite(dynamic, _rgb())

    invalid = FakeInterpreter(
        input_shape=(1, 128, 128, 3),
        output=np.array([[np.nan]], dtype=np.float32),
    )
    with pytest.raises(InferenceError):
        classify_tflite(invalid, _rgb())


def test_tflite_classifier_accepts_dynamic_batch_signatures() -> None:
    interpreter = FakeInterpreter(
        input_shape=(1, 128, 128, 3),
        input_signature=(-1, 128, 128, 3),
        output=np.array([[0.8]], dtype=np.float32),
        output_signature=(-1, 1),
    )
    assert classify_tflite(interpreter, _rgb()) == pytest.approx(0.8)
    assert interpreter.invoked is True


@pytest.mark.parametrize(
    "input_signature, output_signature",
    [
        ((1, -1, 128, 3), (1, 1)),
        ((1, 128, 128, 3), (1, -1)),
        ((-1, 128, 128), (-1, 1)),
        ((-1, 128, 128, 3), (-1, 1, 1)),
    ],
)
def test_tflite_classifier_rejects_non_batch_dynamic_or_wrong_rank_signatures(
    input_signature: tuple[int, ...], output_signature: tuple[int, ...]
) -> None:
    interpreter = FakeInterpreter(
        input_shape=(1, 128, 128, 3),
        input_signature=input_signature,
        output=np.array([[0.8]], dtype=np.float32),
        output_signature=output_signature,
    )
    with pytest.raises(InferenceError):
        classify_tflite(interpreter, _rgb())
    assert interpreter.last_input is None
    assert interpreter.invoked is False


@pytest.mark.parametrize(
    "actual_output",
    [np.array(0.8, dtype=np.float32), np.array([[0.8]], dtype=np.float64)],
)
def test_tflite_classifier_rejects_actual_output_contract_mismatch(
    actual_output: np.ndarray,
) -> None:
    interpreter = FakeInterpreter(
        input_shape=(1, 128, 128, 3),
        output=np.array([[0.8]], dtype=np.float32),
    )
    interpreter.output = actual_output
    with pytest.raises(InferenceError):
        classify_tflite(interpreter, _rgb())


def test_tflite_classifier_wraps_invoke_failure_with_cause() -> None:
    interpreter = FakeInterpreter(
        input_shape=(1, 128, 128, 3),
        output=np.array([[0.8]], dtype=np.float32),
        fail_at="invoke",
    )
    with pytest.raises(InferenceError) as caught:
        classify_tflite(interpreter, _rgb())
    assert isinstance(caught.value.__cause__, RuntimeError)


def test_suppress_detections_is_stable_and_explicitly_bounded() -> None:
    detections = [
        Detection(0, 0, 10, 10, 0.8),
        Detection(1, 1, 10, 10, 0.8),
        Detection(20, 0, 30, 10, 0.8),
    ]
    assert suppress_detections(detections, max_detections=1) == [detections[0]]


def test_default_pipeline_validates_output_and_runs_final_nms() -> None:
    image = _rgb(40, 40)
    seen: list[np.ndarray] = []

    def detector(tile: np.ndarray) -> list[Detection]:
        seen.append(tile)
        return [Detection(0, 0, 20, 20, 0.9), Detection(1, 1, 20, 20, 0.8)]

    assert detect_default(detector, image) == [Detection(0, 0, 20, 20, 0.9)]
    assert seen == [image]

    with pytest.raises(InferenceError):
        detect_default(lambda _: [object()], image)  # type: ignore[list-item]


def test_default_pipeline_bounds_clipped_candidates_before_nms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nms_sizes: list[int] = []

    def observe_nms(
        boxes: np.ndarray,
        scores: np.ndarray,
        *args: object,
        **kwargs: object,
    ) -> list[int]:
        nms_sizes.append(len(boxes))
        return list(range(len(boxes)))

    monkeypatch.setattr(inference_module, "nms", observe_nms)

    def detector(image: np.ndarray) -> list[Detection]:
        return [
            Detection(0, 0, 10, 10, (index + 1) / 100)
            for index in range(100)
        ]

    detections = detect_default(detector, _rgb(20, 20), max_candidates=1)
    assert nms_sizes == [1]
    assert len(detections) == 1
    assert detections[0].score == pytest.approx(1.0)


def test_default_pipeline_streams_large_callback_through_clip_and_top_k(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_clip = inference_module._clip_box
    state = {"yielded": 0, "clipped": 0}
    nms_sizes: list[int] = []

    def observe_clip(
        box: object, width: int, height: int
    ) -> tuple[float, float, float, float] | None:
        state["clipped"] += 1
        return original_clip(box, width, height)  # type: ignore[arg-type]

    def observe_nms(
        boxes: np.ndarray, scores: np.ndarray, *args: object, **kwargs: object
    ) -> list[int]:
        nms_sizes.append(len(boxes))
        return list(range(len(boxes)))

    def detector(image: np.ndarray) -> object:
        for index in range(5_000):
            assert state["clipped"] == index
            state["yielded"] += 1
            yield Detection(0, 0, 10, 10, 0.5)

    monkeypatch.setattr(inference_module, "_clip_box", observe_clip)
    monkeypatch.setattr(inference_module, "nms", observe_nms)
    detections = detect_default(detector, _rgb(20, 20), max_candidates=1)
    assert len(detections) == 1
    assert state == {"yielded": 5_000, "clipped": 5_000}
    assert nms_sizes == [1]


@pytest.mark.parametrize(
    "case",
    [
        "non_iterable",
        "call_failure",
        "iter_failure",
        "first_bad",
        "late_bad",
        "late_raise",
    ],
)
def test_default_pipeline_rejects_callback_stream_errors_before_nms(
    monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    nms_calls = 0
    failure = RuntimeError(f"{case} broke")

    def forbidden_nms(*args: object, **kwargs: object) -> list[int]:
        nonlocal nms_calls
        nms_calls += 1
        return []

    if case == "non_iterable":
        def detector(image: np.ndarray) -> object:
            return 123
    elif case == "call_failure":
        def detector(image: np.ndarray) -> object:
            raise failure
    elif case == "iter_failure":
        class BrokenIterable:
            def __iter__(self) -> object:
                raise failure

        def detector(image: np.ndarray) -> object:
            return BrokenIterable()
    elif case == "first_bad":
        def detector(image: np.ndarray) -> object:
            return iter([object()])
    elif case == "late_bad":
        def detector(image: np.ndarray) -> object:
            return iter([Detection(0, 0, 1, 1, 0.5), object()])
    else:
        def late_failure() -> object:
            yield Detection(0, 0, 1, 1, 0.5)
            raise failure

        def detector(image: np.ndarray) -> object:
            return late_failure()

    monkeypatch.setattr(inference_module, "nms", forbidden_nms)
    with pytest.raises(InferenceError) as caught:
        detect_default(detector, _rgb())  # type: ignore[arg-type]
    assert nms_calls == 0
    if case in {"call_failure", "iter_failure", "late_raise"}:
        assert caught.value.__cause__ is failure
    else:
        assert isinstance(caught.value.__cause__, TypeError)


def test_default_pipeline_wraps_detector_failure_with_cause() -> None:
    failure = RuntimeError("detector broke")

    def detector(image: np.ndarray) -> list[Detection]:
        raise failure

    with pytest.raises(InferenceError) as caught:
        detect_default(detector, _rgb())
    assert caught.value.__cause__ is failure


@pytest.mark.parametrize(
    "pipeline, kwargs",
    [
        (detect_default, {"nms_iou": np.nan}),
        (detect_default, {"max_candidates": 0}),
        (detect_tiled, {"nms_iou": -0.1}),
        (detect_tiled, {"max_detections": False}),
    ],
)
def test_pipelines_validate_options_before_calling_detector(
    pipeline: object, kwargs: dict[str, object]
) -> None:
    calls = 0

    def detector(image: np.ndarray) -> list[Detection]:
        nonlocal calls
        calls += 1
        return []

    with pytest.raises((TypeError, ValueError)):
        pipeline(detector, _rgb(), **kwargs)  # type: ignore[operator]
    assert calls == 0


def test_tiled_pipeline_offsets_clips_and_globally_suppresses_overlap() -> None:
    image = _rgb(100, 150)
    calls: list[tuple[int, int]] = []

    def detector(tile: np.ndarray) -> list[Detection]:
        calls.append(tile.shape[:2])
        if len(calls) == 1:
            return [Detection(75, 20, 100, 50, 0.8)]
        return [
            Detection(25, 20, 50, 50, 0.9),
            Detection(100, 0, 120, 10, 0.7),
        ]

    detections = detect_tiled(
        detector, image, tile=100, overlap=0.5, max_tiles=2
    )
    assert calls == [(100, 100), (100, 100)]
    assert detections == [Detection(75, 20, 100, 50, 0.9)]


def test_tiled_pipeline_handles_small_image_and_edge_box() -> None:
    image = _rgb(20, 30)
    detections = detect_tiled(
        lambda tile: [Detection(0, 0, 35, 25, 0.6)],
        image,
        tile=640,
        max_tiles=1,
    )
    assert detections == [Detection(0, 0, 30, 20, 0.6)]


def test_tiled_pipeline_clips_to_each_tile_before_applying_offset() -> None:
    calls = 0

    def detector(tile_image: np.ndarray) -> list[Detection]:
        nonlocal calls
        calls += 1
        return [Detection(90, 0, 120, 10, 0.7)] if calls == 1 else []

    detections = detect_tiled(
        detector,
        _rgb(100, 150),
        tile=100,
        overlap=0.5,
        max_tiles=2,
    )
    assert detections == [Detection(90, 0, 100, 10, 0.7)]


def test_tiled_pipeline_filters_box_degenerate_after_global_clip() -> None:
    assert detect_tiled(
        lambda tile: [Detection(1000, 0, 1001, 1, 0.9)],
        _rgb(),
        max_tiles=1,
    ) == []


def test_tiled_pipeline_enforces_tile_and_global_detection_limits() -> None:
    with pytest.raises(ValueError, match="max_tiles"):
        detect_tiled(lambda tile: [], _rgb(100, 250), tile=100, max_tiles=2)

    detections = detect_tiled(
        lambda tile: [Detection(0, 0, 1, 1, 0.5)],
        _rgb(100, 250),
        tile=100,
        overlap=0,
        max_tiles=3,
        max_detections=2,
    )
    assert len(detections) == 2


def test_tiled_pipeline_incrementally_bounds_global_candidates_without_tile_nms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_merge = inference_module._merge_bounded_detections
    merge_sizes: list[tuple[int, int, int]] = []
    nms_sizes: list[int] = []

    def observe_merge(
        current: list[Detection], incoming: list[Detection], max_candidates: int
    ) -> list[Detection]:
        result = original_merge(current, incoming, max_candidates)
        merge_sizes.append((len(current), len(incoming), len(result)))
        return result

    def observe_nms(
        boxes: np.ndarray, scores: np.ndarray, *args: object, **kwargs: object
    ) -> list[int]:
        nms_sizes.append(len(boxes))
        return list(range(len(boxes)))

    monkeypatch.setattr(inference_module, "_merge_bounded_detections", observe_merge)
    monkeypatch.setattr(inference_module, "nms", observe_nms)

    def run(tile_scores: list[float]) -> list[Detection]:
        calls = 0

        def detector(tile_image: np.ndarray) -> list[Detection]:
            nonlocal calls
            score = tile_scores[calls]
            calls += 1
            return [Detection(1, 1, 2, 2, score) for _ in range(100)]

        return detect_tiled(
            detector,
            _rgb(100, 300),
            tile=100,
            overlap=0,
            max_tiles=3,
            max_candidates=1,
        )

    highest = run([0.7, 0.8, 0.9])
    tied = run([0.8, 0.8, 0.8])

    assert [(item.x1, item.score) for item in highest] == [
        (201, pytest.approx(0.9))
    ]
    assert [(item.x1, item.score) for item in tied] == [
        (1, pytest.approx(0.8))
    ]
    assert nms_sizes == [1, 1]
    assert merge_sizes == [
        (0, 1, 1),
        (1, 1, 1),
        (1, 1, 1),
        (0, 1, 1),
        (1, 1, 1),
        (1, 1, 1),
    ]


def test_tiled_pipeline_streams_each_callback_before_requesting_next_item(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_clip = inference_module._clip_box
    state = {"yielded": 0, "clipped": 0}

    def observe_clip(
        box: object, width: int, height: int
    ) -> tuple[float, float, float, float] | None:
        state["clipped"] += 1
        return original_clip(box, width, height)  # type: ignore[arg-type]

    def detector(image: np.ndarray) -> object:
        for index in range(1_000):
            assert state["clipped"] == index
            state["yielded"] += 1
            yield Detection(0, 0, 10, 10, 0.5)

    monkeypatch.setattr(inference_module, "_clip_box", observe_clip)
    detections = detect_tiled(
        detector,
        _rgb(100, 100),
        tile=100,
        max_tiles=1,
        max_candidates=1,
    )
    assert len(detections) == 1
    assert state == {"yielded": 1_000, "clipped": 1_001}
