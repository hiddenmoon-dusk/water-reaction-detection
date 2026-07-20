"""Strict RGB inference adapters shared by source models and TFLite models.

All image arguments in this module are RGB ``uint8`` arrays.  The adapters do
not infer or silently convert BGR input.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import inspect
from itertools import chain
from numbers import Integral, Real
from typing import Callable, Iterable, Sequence

import cv2
import numpy as np

from .geometry import LetterboxTransform, letterbox, nms, tile_origins


DEFAULT_MAX_CANDIDATES = 3_000
DEFAULT_MAX_DETECTIONS = 300
DEFAULT_MAX_TILES = 1_024


class InferenceError(RuntimeError):
    """An external model/interpreter failed or violated its output contract."""


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


def _threshold(name: str, value: object) -> float:
    converted = _finite_real(name, value)
    if not 0.0 <= converted <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return converted


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    converted = int(value)
    if converted <= 0:
        raise ValueError(f"{name} must be positive")
    return converted


def _rgb_image(name: str, image: object) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise TypeError(f"{name} must be a numpy array")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"{name} must have shape HxWx3")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError(f"{name} dimensions must be non-empty")
    if image.dtype != np.uint8:
        raise TypeError(f"{name} dtype must be uint8")
    return image


@dataclass(frozen=True)
class Detection:
    """One finite, positive-area ``xyxy`` detection in image coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float
    score: float

    def __post_init__(self) -> None:
        values = (
            _finite_real("x1", self.x1),
            _finite_real("y1", self.y1),
            _finite_real("x2", self.x2),
            _finite_real("y2", self.y2),
            _threshold("score", self.score),
        )
        if values[2] <= values[0] or values[3] <= values[1]:
            raise ValueError("detection must have positive width and height")
        for field_name, value in zip(
            ("x1", "y1", "x2", "y2", "score"), values, strict=True
        ):
            object.__setattr__(self, field_name, value)


def _bounded_top_detections(
    detections: Iterable[Detection], max_candidates: int
) -> list[Detection]:
    """Keep stable score-descending top-K detections in O(K) memory."""

    limit = _positive_int("max_candidates", max_candidates)
    try:
        iterator = iter(detections)
    except TypeError as error:
        raise TypeError("detections must be iterable") from error

    heap: list[tuple[float, int, Detection]] = []
    for order, item in enumerate(iterator):
        if type(item) is not Detection:
            raise InferenceError(
                "detector output must contain only Detection values"
            )
        entry = (item.score, -order, item)
        if len(heap) < limit:
            heapq.heappush(heap, entry)
        elif entry[:2] > heap[0][:2]:
            heapq.heapreplace(heap, entry)
    heap.sort(key=lambda entry: (-entry[0], -entry[1]))
    return [entry[2] for entry in heap]


def _merge_bounded_detections(
    current: list[Detection],
    incoming: list[Detection],
    max_candidates: int,
) -> list[Detection]:
    """Incrementally merge two already bounded, globally ordered batches."""

    return _bounded_top_detections(chain(current, incoming), max_candidates)


def _original_size(value: object) -> tuple[int, int]:
    if not isinstance(value, (tuple, list)):
        raise TypeError("original_size must be a width, height sequence")
    if len(value) != 2:
        raise ValueError("original_size must contain width and height")
    return (
        _positive_int("original width", value[0]),
        _positive_int("original height", value[1]),
    )


def _letterbox_transform(
    width: int, height: int, size: int = 640
) -> LetterboxTransform:
    scale = min(size / width, size / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    return LetterboxTransform(
        scale=scale,
        pad_x=(size - resized_width) // 2,
        pad_y=(size - resized_height) // 2,
        original_width=width,
        original_height=height,
    )


def _clip_box(
    box: Sequence[Real], width: int, height: int
) -> tuple[float, float, float, float] | None:
    if len(box) != 4:
        raise ValueError("box must contain four coordinates")
    x1, y1, x2, y2 = (
        _finite_real(f"box[{index}]", value) for index, value in enumerate(box)
    )
    if x2 < x1 or y2 < y1:
        raise ValueError("box coordinates must be ordered")
    clipped = (
        min(max(x1, 0.0), float(width)),
        min(max(y1, 0.0), float(height)),
        min(max(x2, 0.0), float(width)),
        min(max(y2, 0.0), float(height)),
    )
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        return None
    return clipped


def suppress_detections(
    detections: Iterable[Detection],
    *,
    nms_iou: Real = 0.45,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    max_detections: int = DEFAULT_MAX_DETECTIONS,
) -> list[Detection]:
    """Run stable global NMS with explicit inference-layer resource limits."""

    threshold = _threshold("nms_iou", nms_iou)
    candidate_limit = _positive_int("max_candidates", max_candidates)
    detection_limit = _positive_int("max_detections", max_detections)
    items = _bounded_top_detections(detections, candidate_limit)
    if not items:
        return []
    boxes = np.asarray(
        [(item.x1, item.y1, item.x2, item.y2) for item in items],
        dtype=np.float64,
    )
    scores = np.asarray([item.score for item in items], dtype=np.float64)
    kept = nms(
        boxes,
        scores,
        threshold,
        max_candidates=candidate_limit,
        max_detections=detection_limit,
    )
    return [items[index] for index in kept]


def decode_yolo(
    output: np.ndarray,
    original_size: tuple[int, int],
    conf: Real = 0.3,
    nms_iou: Real = 0.45,
    *,
    coordinates_normalized: bool = False,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    max_detections: int = DEFAULT_MAX_DETECTIONS,
) -> list[Detection]:
    """Decode one exported YOLO ``[1, 5, N]`` float32 tensor."""

    width, height = _original_size(original_size)
    confidence = _threshold("conf", conf)
    iou = _threshold("nms_iou", nms_iou)
    candidate_limit = _positive_int("max_candidates", max_candidates)
    detection_limit = _positive_int("max_detections", max_detections)
    if type(coordinates_normalized) is not bool:
        raise TypeError("coordinates_normalized must be a bool")
    if not isinstance(output, np.ndarray):
        raise TypeError("output must be a numpy array")
    if output.ndim != 3 or output.shape[:2] != (1, 5) or output.shape[2] <= 0:
        raise ValueError("YOLO output must have exact shape [1, 5, N]")
    if output.dtype != np.float32:
        raise TypeError("YOLO output dtype must be float32")
    if not np.all(np.isfinite(output)):
        raise ValueError("YOLO output must contain only finite values")
    scores = output[0, 4]
    if np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("YOLO confidence must be between 0 and 1")

    transform = _letterbox_transform(width, height)
    coordinate_scale = 640.0 if coordinates_normalized else 1.0

    def valid_detections() -> Iterable[Detection]:
        for cx, cy, box_width, box_height, score_value in output[0].T:
            score = float(score_value)
            if score < confidence:
                continue
            if box_width <= 0 or box_height <= 0:
                continue
            center_x = float(cx) * coordinate_scale
            center_y = float(cy) * coordinate_scale
            half_width = float(box_width) * coordinate_scale / 2.0
            half_height = float(box_height) * coordinate_scale / 2.0
            restored = transform.to_original(
                (
                    center_x - half_width,
                    center_y - half_height,
                    center_x + half_width,
                    center_y + half_height,
                )
            )
            if restored[2] <= restored[0] or restored[3] <= restored[1]:
                continue
            yield Detection(*restored, score)

    decoded = _bounded_top_detections(valid_detections(), candidate_limit)
    return suppress_detections(
        decoded,
        nms_iou=iou,
        max_candidates=candidate_limit,
        max_detections=detection_limit,
    )


def _as_numpy(value: object, name: str) -> np.ndarray:
    try:
        converted = value
        detach = getattr(converted, "detach", None)
        if callable(detach):
            converted = detach()
            cpu = getattr(converted, "cpu", None)
            if callable(cpu):
                converted = cpu()
        numpy_method = getattr(converted, "numpy", None)
        if callable(numpy_method):
            converted = numpy_method()
        return np.asarray(converted)
    except Exception as error:
        raise InferenceError(f"failed to read {name}") from error


def detect_source(
    model: object,
    image_rgb: np.ndarray,
    *,
    conf: Real = 0.3,
    nms_iou: Real = 0.45,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    max_detections: int = DEFAULT_MAX_DETECTIONS,
    configure_model_nms: bool = False,
) -> list[Detection]:
    """Run Ultralytics, then align its results with the mobile NMS contract."""

    image = _rgb_image("image_rgb", image_rgb)
    confidence = _threshold("conf", conf)
    iou = _threshold("nms_iou", nms_iou)
    candidate_limit = _positive_int("max_candidates", max_candidates)
    detection_limit = _positive_int("max_detections", max_detections)
    if type(configure_model_nms) is not bool:
        raise TypeError("configure_model_nms must be a bool")
    try:
        if configure_model_nms:
            results_value = model(  # type: ignore[operator]
                image,
                conf=confidence,
                verbose=False,
                iou=iou,
                max_det=candidate_limit,
            )
        else:
            results_value = model(  # type: ignore[operator]
                image, conf=confidence, verbose=False
            )
    except Exception as error:
        raise InferenceError("source detector call failed") from error
    try:
        results = list(results_value)
    except Exception as error:
        raise InferenceError("source detector must return one result") from error
    if len(results) != 1:
        raise InferenceError("source detector must return exactly one result")
    boxes = getattr(results[0], "boxes", None)
    if boxes is None:
        raise InferenceError("source detector result is missing boxes")
    xyxy = _as_numpy(getattr(boxes, "xyxy", None), "source detector boxes")
    scores = _as_numpy(getattr(boxes, "conf", None), "source detector scores")
    if xyxy.ndim != 2 or xyxy.shape[1:] != (4,):
        raise InferenceError("source detector boxes must have shape Nx4")
    if scores.ndim != 1 or scores.shape[0] != xyxy.shape[0]:
        raise InferenceError("source detector scores must have shape N")
    if xyxy.dtype.kind not in "iuf" or scores.dtype.kind not in "iuf":
        raise InferenceError("source detector outputs must contain real numbers")
    if not np.all(np.isfinite(xyxy)) or not np.all(np.isfinite(scores)):
        raise InferenceError("source detector outputs must be finite")

    height, width = image.shape[:2]
    detections: list[Detection] = []
    for box, score_value in zip(xyxy, scores, strict=True):
        score = float(score_value)
        if not 0.0 <= score <= 1.0:
            raise InferenceError("source detector confidence must be between 0 and 1")
        if score < confidence:
            continue
        try:
            clipped = _clip_box(box, width, height)
        except (TypeError, ValueError) as error:
            raise InferenceError(
                "source detector returned invalid coordinates"
            ) from error
        if clipped is not None:
            detections.append(Detection(*clipped, score))
    return suppress_detections(
        detections,
        nms_iou=iou,
        max_candidates=candidate_limit,
        max_detections=detection_limit,
    )


def _shape(detail: dict[str, object], field: str = "shape") -> tuple[int, ...]:
    try:
        value = np.asarray(detail[field])
        if value.ndim != 1 or value.dtype.kind not in "iu":
            raise ValueError
        return tuple(int(item) for item in value)
    except Exception as error:
        raise InferenceError(f"invalid TFLite {field}") from error


def _tensor_index(detail: dict[str, object], label: str) -> int:
    value = detail.get("index")
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise InferenceError(f"TFLite {label} index must be an integer")
    index = int(value)
    if index < 0:
        raise InferenceError(f"TFLite {label} index must be non-negative")
    return index


def _validate_unquantized_metadata(
    detail: dict[str, object], label: str
) -> None:
    try:
        if "quantization" in detail:
            quantization = tuple(detail["quantization"])  # type: ignore[arg-type]
            if len(quantization) != 2:
                raise ValueError("quantization must contain scale and zero point")
            scale, zero_point = quantization
            if (
                isinstance(scale, bool)
                or not isinstance(scale, Real)
                or not np.isfinite(float(scale))
                or float(scale) != 0.0
                or isinstance(zero_point, bool)
                or not isinstance(zero_point, Integral)
                or int(zero_point) != 0
            ):
                raise ValueError("float32 tensor must be unquantized")

        if "quantization_parameters" in detail:
            parameters = detail["quantization_parameters"]
            if not isinstance(parameters, dict):
                raise TypeError("quantization_parameters must be a dictionary")
            for field in ("scales", "zero_points"):
                if field not in parameters:
                    raise ValueError(f"missing {field}")
                raw = parameters[field]
                if not isinstance(raw, (list, tuple, np.ndarray)):
                    raise TypeError(f"{field} must be an array")
                values = np.asarray(raw)
                if values.ndim != 1 or values.size != 0:
                    raise ValueError(f"{field} must be empty")
    except Exception as error:
        raise InferenceError(
            f"TFLite {label} quantization metadata must be unquantized"
        ) from error


def _validate_tflite_contract(
    interpreter: object,
    *,
    input_shape: tuple[int, ...],
    output_prefix: tuple[int, ...],
    output_exact: bool,
    allow_dynamic_batch: bool = False,
) -> tuple[int, int]:
    try:
        inputs = interpreter.get_input_details()  # type: ignore[attr-defined]
        outputs = interpreter.get_output_details()  # type: ignore[attr-defined]
    except Exception as error:
        raise InferenceError("failed to read TFLite tensor details") from error
    try:
        if len(inputs) != 1 or len(outputs) != 1:
            raise InferenceError(
                "TFLite model must expose one input and one output"
            )
        input_detail, output_detail = inputs[0], outputs[0]
        if not isinstance(input_detail, dict) or not isinstance(output_detail, dict):
            raise TypeError("tensor details must be dictionaries")
    except InferenceError:
        raise
    except Exception as error:
        raise InferenceError("invalid TFLite tensor details") from error
    actual_input = _shape(input_detail)
    actual_output = _shape(output_detail)
    input_signature = (
        _shape(input_detail, "shape_signature")
        if "shape_signature" in input_detail
        else actual_input
    )
    output_signature = (
        _shape(output_detail, "shape_signature")
        if "shape_signature" in output_detail
        else actual_output
    )
    def valid_signature(
        signature: tuple[int, ...], expected: tuple[int, ...]
    ) -> bool:
        return signature == expected or (
            allow_dynamic_batch
            and len(signature) == len(expected)
            and signature[0] == -1
            and signature[1:] == expected[1:]
        )

    if actual_input != input_shape or not valid_signature(
        input_signature, input_shape
    ):
        raise InferenceError(f"TFLite input contract mismatch: {input_shape}")
    output_matches = (
        actual_output == output_prefix
        if output_exact
        else len(actual_output) == len(output_prefix) + 1
        and actual_output[: len(output_prefix)] == output_prefix
        and actual_output[-1] > 0
    )
    if not output_matches or not valid_signature(output_signature, actual_output):
        raise InferenceError("TFLite output shape contract mismatch")
    try:
        input_dtype = np.dtype(input_detail["dtype"])
        output_dtype = np.dtype(output_detail["dtype"])
    except Exception as error:
        raise InferenceError("invalid TFLite tensor dtype") from error
    if input_dtype != np.dtype(np.float32) or output_dtype != np.dtype(np.float32):
        raise InferenceError("TFLite input and output dtypes must be float32")
    input_index = _tensor_index(input_detail, "input")
    output_index = _tensor_index(output_detail, "output")
    _validate_unquantized_metadata(input_detail, "input")
    _validate_unquantized_metadata(output_detail, "output")
    return input_index, output_index


def _invoke_tflite(
    interpreter: object,
    input_index: int,
    output_index: int,
    batch: np.ndarray,
) -> np.ndarray:
    try:
        interpreter.set_tensor(input_index, batch)  # type: ignore[attr-defined]
        interpreter.invoke()  # type: ignore[attr-defined]
        output = interpreter.get_tensor(output_index)  # type: ignore[attr-defined]
    except Exception as error:
        raise InferenceError("TFLite inference failed") from error
    if not isinstance(output, np.ndarray):
        raise InferenceError("TFLite output must be a numpy array")
    return output


def detect_tflite(
    interpreter: object,
    image_rgb: np.ndarray,
    *,
    conf: Real = 0.3,
    nms_iou: Real = 0.45,
    normalize: bool = True,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    max_detections: int = DEFAULT_MAX_DETECTIONS,
) -> list[Detection]:
    """Run a raw float32 YOLO TFLite detector on RGB input.

    ``normalize=True`` is the exported YOLO contract and divides letterboxed
    pixels by 255.  ``False`` is an explicit raw-float contract; dtype is never
    guessed from interpreter metadata.
    """

    image = _rgb_image("image_rgb", image_rgb)
    confidence = _threshold("conf", conf)
    iou = _threshold("nms_iou", nms_iou)
    candidate_limit = _positive_int("max_candidates", max_candidates)
    detection_limit = _positive_int("max_detections", max_detections)
    if type(normalize) is not bool:
        raise TypeError("normalize must be a bool")
    input_index, output_index = _validate_tflite_contract(
        interpreter,
        input_shape=(1, 640, 640, 3),
        output_prefix=(1, 5),
        output_exact=False,
        allow_dynamic_batch=False,
    )
    transformed, _ = letterbox(image, 640)
    batch = transformed.astype(np.float32)[None, ...]
    if normalize:
        batch /= np.float32(255.0)
    output = _invoke_tflite(interpreter, input_index, output_index, batch)
    try:
        return decode_yolo(
            output,
            original_size=(image.shape[1], image.shape[0]),
            conf=confidence,
            nms_iou=iou,
            coordinates_normalized=True,
            max_candidates=candidate_limit,
            max_detections=detection_limit,
        )
    except (TypeError, ValueError) as error:
        raise InferenceError("TFLite detector returned invalid output") from error


def _resize_classifier_input(crop: np.ndarray) -> np.ndarray:
    resized = cv2.resize(crop, (128, 128))
    return resized.astype(np.float32)[None, ...]


def _classifier_probability(output: object, source: str) -> float:
    array = _as_numpy(output, f"{source} classifier output")
    if array.dtype.kind not in "iuf":
        raise InferenceError(f"{source} classifier output must be numeric")
    if array.shape == ():
        value = float(array)
    elif array.shape == (1, 1):
        value = float(array[0, 0])
    else:
        raise InferenceError(f"{source} classifier output must be scalar or [1, 1]")
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise InferenceError(
            f"{source} classifier probability must be finite and between 0 and 1"
        )
    return value


def _accepts_training(model: object) -> bool:
    try:
        signature = inspect.signature(model)
    except (TypeError, ValueError):
        return True
    return "training" in signature.parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def classify_source(model: object, crop: np.ndarray) -> float:
    """Run the source Keras classifier on raw-valued RGB float32 pixels."""

    image = _rgb_image("crop", crop)
    batch = _resize_classifier_input(image)
    try:
        if _accepts_training(model):
            output = model(batch, training=False)  # type: ignore[operator]
        else:
            output = model(batch)  # type: ignore[operator]
    except Exception as error:
        raise InferenceError("source classifier call failed") from error
    return _classifier_probability(output, "source")


def classify_tflite(interpreter: object, crop: np.ndarray) -> float:
    """Run the float32 TFLite classifier on raw-valued RGB pixels."""

    image = _rgb_image("crop", crop)
    input_index, output_index = _validate_tflite_contract(
        interpreter,
        input_shape=(1, 128, 128, 3),
        output_prefix=(1, 1),
        output_exact=True,
        allow_dynamic_batch=True,
    )
    output = _invoke_tflite(
        interpreter, input_index, output_index, _resize_classifier_input(image)
    )
    if output.shape != (1, 1) or output.dtype != np.float32:
        raise InferenceError(
            "TFLite classifier output must have exact shape [1, 1] and dtype float32"
        )
    return _classifier_probability(output, "TFLite")


DetectorFunction = Callable[[np.ndarray], Iterable[Detection]]


def _call_detector(
    detector_fn: DetectorFunction, image: np.ndarray
) -> Iterable[Detection]:
    """Lazily validate one callback stream before bounded downstream use."""

    try:
        detections = detector_fn(image)
    except Exception as error:
        raise InferenceError("detector function call failed") from error
    try:
        iterator = iter(detections)
    except Exception as error:
        raise InferenceError("failed to iterate detector output") from error

    while True:
        try:
            item = next(iterator)
        except StopIteration:
            return
        except Exception as error:
            raise InferenceError("detector output iteration failed") from error
        if type(item) is not Detection:
            error = TypeError("detector output item must be a Detection")
            raise InferenceError(
                "detector output must contain only Detection values"
            ) from error
        yield item


def _offset_and_clip(
    detections: Iterable[Detection],
    *,
    offset_x: int,
    offset_y: int,
    width: int,
    height: int,
) -> Iterable[Detection]:
    for detection in detections:
        clipped = _clip_box(
            (
                detection.x1 + offset_x,
                detection.y1 + offset_y,
                detection.x2 + offset_x,
                detection.y2 + offset_y,
            ),
            width,
            height,
        )
        if clipped is not None:
            yield Detection(*clipped, detection.score)


def detect_default(
    detector_fn: DetectorFunction,
    image_rgb: np.ndarray,
    *,
    nms_iou: Real = 0.45,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    max_detections: int = DEFAULT_MAX_DETECTIONS,
) -> list[Detection]:
    """Run one RGB image through a detector and apply final global NMS."""

    image = _rgb_image("image_rgb", image_rgb)
    iou = _threshold("nms_iou", nms_iou)
    candidate_limit = _positive_int("max_candidates", max_candidates)
    detection_limit = _positive_int("max_detections", max_detections)
    detections = _call_detector(detector_fn, image)
    clipped = _offset_and_clip(
        detections,
        offset_x=0,
        offset_y=0,
        width=image.shape[1],
        height=image.shape[0],
    )
    return suppress_detections(
        clipped,
        nms_iou=iou,
        max_candidates=candidate_limit,
        max_detections=detection_limit,
    )


def detect_tiled(
    detector_fn: DetectorFunction,
    image_rgb: np.ndarray,
    *,
    tile: int = 640,
    overlap: Real = 0.2,
    nms_iou: Real = 0.45,
    max_tiles: int = DEFAULT_MAX_TILES,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    max_detections: int = DEFAULT_MAX_DETECTIONS,
) -> list[Detection]:
    """Detect overlapping RGB tiles, offset to global coordinates, then NMS."""

    image = _rgb_image("image_rgb", image_rgb)
    tile_size = _positive_int("tile", tile)
    tile_limit = _positive_int("max_tiles", max_tiles)
    iou = _threshold("nms_iou", nms_iou)
    candidate_limit = _positive_int("max_candidates", max_candidates)
    detection_limit = _positive_int("max_detections", max_detections)
    origins = tile_origins(
        width=image.shape[1],
        height=image.shape[0],
        tile=tile_size,
        overlap=overlap,
        max_tiles=tile_limit,
    )
    accumulated: list[Detection] = []
    for origin_x, origin_y in origins:
        tile_image = image[
            origin_y : min(origin_y + tile_size, image.shape[0]),
            origin_x : min(origin_x + tile_size, image.shape[1]),
        ]
        local = _call_detector(detector_fn, tile_image)
        local_bounded = _bounded_top_detections(
            _offset_and_clip(
                local,
                offset_x=0,
                offset_y=0,
                width=tile_image.shape[1],
                height=tile_image.shape[0],
            ),
            candidate_limit,
        )
        global_bounded = _bounded_top_detections(
            _offset_and_clip(
                local_bounded,
                offset_x=origin_x,
                offset_y=origin_y,
                width=image.shape[1],
                height=image.shape[0],
            ),
            candidate_limit,
        )
        accumulated = _merge_bounded_detections(
            accumulated,
            global_bounded,
            candidate_limit,
        )
    return suppress_detections(
        accumulated,
        nms_iou=iou,
        max_candidates=candidate_limit,
        max_detections=detection_limit,
    )
