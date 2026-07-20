from dataclasses import FrozenInstanceError, replace
import json

import numpy as np
import pytest

from water_models.inference import Detection
from water_models import parity as parity_module
from water_models.parity import (
    Match,
    ParityDecision,
    ParityReport,
    Prediction,
    compare_conversion,
    evaluate,
    match_detections,
)


def prediction(
    box: tuple[float, float, float, float] = (0, 0, 19, 10),
    score: float = 0.9,
    label: str = "lib",
) -> Prediction:
    return Prediction(Detection(*box, score), label)


def report_for(
    reference: list[list[Prediction]] | None = None,
    mobile: list[list[Prediction]] | None = None,
    reference_probabilities: list[list[float]] | None = None,
    mobile_probabilities: list[list[float]] | None = None,
    *,
    old_count: object = 10,
    new_count: object = 10,
    label_change_ratio: object = 0.0,
    sample_ids: object = None,
) -> ParityReport:
    reference = [[prediction()]] if reference is None else reference
    mobile = [[prediction()]] if mobile is None else mobile
    reference_probabilities = (
        [[0.5]] if reference_probabilities is None else reference_probabilities
    )
    mobile_probabilities = (
        [[0.5]] if mobile_probabilities is None else mobile_probabilities
    )
    return compare_conversion(
        reference,
        mobile,
        reference_probabilities,
        mobile_probabilities,
        old_count=old_count,
        new_count=new_count,
        label_change_ratio=label_change_ratio,
        sample_ids=sample_ids,
    )


def test_predictions_and_matches_are_frozen() -> None:
    item = prediction()
    match = match_detections([item], [item])[0]

    with pytest.raises(FrozenInstanceError):
        item.label = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        match.iou = 0.0  # type: ignore[misc]
    assert isinstance(match, Match)


def test_prediction_reuses_strict_detection_validation() -> None:
    with pytest.raises(ValueError, match="finite"):
        prediction((0, 0, np.nan, 1))
    with pytest.raises(ValueError, match="between 0 and 1"):
        prediction(score=1.1)
    with pytest.raises(ValueError, match="positive"):
        prediction((1, 0, 1, 1))
    with pytest.raises(ValueError, match="label"):
        Prediction(Detection(0, 0, 1, 1, 0.5), "")


def test_global_greedy_matching_considers_all_pairs_before_assignment() -> None:
    reference = [
        prediction((0, 0, 100, 100)),
        prediction((2, 0, 102, 100)),
    ]
    mobile = [
        prediction((2, 0, 102, 100)),
        prediction((-2, 0, 98, 100)),
    ]

    matches = match_detections(reference, mobile)

    assert [(match.reference_index, match.mobile_index) for match in matches] == [
        (1, 0),
        (0, 1),
    ]
    assert matches[0].iou == 1.0


def test_global_greedy_ties_use_reference_then_mobile_index() -> None:
    same = prediction()

    matches = match_detections([same, same], [same, same])

    assert [(match.reference_index, match.mobile_index) for match in matches] == [
        (0, 0),
        (1, 1),
    ]


def test_matching_empty_collections_is_explicitly_empty() -> None:
    assert match_detections([], []) == ()
    report = report_for([], [], [], [], old_count=0, new_count=0)
    decision = evaluate(report)
    assert report.images == ()
    assert report.conversion_passed is False
    assert "at least one sample is required" in report.invalid_reasons
    assert decision.hard_fail is True
    assert decision.can_override is False
    json.dumps(report.to_dict(), allow_nan=False)


def test_one_empty_image_is_a_valid_conversion_sample() -> None:
    report = report_for([[]], [[]], [[]], [[]], old_count=0, new_count=0)

    assert len(report.images) == 1
    assert report.images[0].reference_count == 0
    assert report.images[0].mobile_count == 0
    assert report.invalid_reasons == ()
    assert report.conversion_passed is True
    assert evaluate(report).hard_fail is False


def test_multiple_images_never_match_detections_across_images() -> None:
    report = report_for(
        [[prediction()], []],
        [[], [prediction()]],
        [[0.5], []],
        [[], [0.5]],
        sample_ids=["left", "right"],
    )

    assert [image.sample_id for image in report.images] == ["left", "right"]
    assert [image.matches for image in report.images] == [(), ()]
    assert evaluate(report).hard_fail is True


def test_iou_confidence_probability_and_filter_boundaries_are_inclusive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        parity_module, "box_iou", lambda left, right: np.array([[0.9]])
    )
    reference = prediction((0, 0, 19, 10), score=0.3)
    mobile = prediction((0, 0, 19, 10), score=0.35)

    report = report_for(
        [[reference]],
        [[mobile]],
        [[0.4]],
        [[0.45]],
    )

    assert report.images[0].matches[0].iou == pytest.approx(0.9)
    assert report.images[0].matches[0].confidence_delta == 0.05
    assert report.images[0].matches[0].probability_delta == 0.05
    assert report.conversion_passed is True


def test_decimal_delta_boundaries_hold_across_multiple_images() -> None:
    report = report_for(
        [[prediction(score=0.35)], [prediction(score=0.65)]],
        [[prediction(score=0.40)], [prediction(score=0.60)]],
        [[0.10], [0.90]],
        [[0.15], [0.85]],
        sample_ids=["first", "second"],
    )

    assert [match.confidence_delta for image in report.images for match in image.matches] == [
        0.05,
        0.05,
    ]
    assert [match.probability_delta for image in report.images for match in image.matches] == [
        0.05,
        0.05,
    ]
    assert report.max_confidence_delta == 0.05
    assert report.max_probability_delta == 0.05
    assert report.conversion_passed is True


def test_iou_one_ulp_below_minimum_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        parity_module,
        "box_iou",
        lambda left, right: np.array([[np.nextafter(0.9, 0.0)]]),
    )
    report = report_for(
        [[prediction((0, 0, 19, 10))]],
        [[prediction((0, 0, 19, 10))]],
        [[0.5]],
        [[0.5]],
    )

    assert report.images[0].matches == ()
    assert evaluate(report).hard_fail is True


def test_point_eight_nine_conversion_mismatch_is_not_overridable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        parity_module, "box_iou", lambda left, right: np.array([[0.89]])
    )

    decision = evaluate(report_for(), "accept mismatch")

    assert decision.hard_fail is True
    assert decision.can_override is False
    assert decision.overridden is False


def test_confidence_delta_one_ulp_over_limit_fails() -> None:
    report = report_for(
        [[prediction(score=0.3)]],
        [[prediction(score=float(np.nextafter(0.35, np.inf)))]],
        [[0.5]],
        [[0.5]],
    )

    assert report.max_confidence_delta > 0.05
    assert evaluate(report).hard_fail is True


def test_probability_delta_one_ulp_over_limit_fails() -> None:
    report = report_for(
        reference_probabilities=[[0.4]],
        mobile_probabilities=[[float(np.nextafter(0.45, np.inf))]],
    )

    assert report.max_probability_delta > 0.05
    assert evaluate(report).hard_fail is True


def test_confidence_filter_includes_point_three_and_excludes_lower_values() -> None:
    low = float(np.nextafter(0.3, 0.0))
    report = report_for(
        [[prediction(score=low), prediction((30, 0, 40, 10), score=0.3)]],
        [[prediction(score=low), prediction((30, 0, 40, 10), score=0.3)]],
        [[0.1, 0.2]],
        [[0.1, 0.2]],
    )

    assert report.reference_count == report.mobile_count == 1
    assert report.images[0].matches[0].reference_index == 0
    assert report.images[0].matches[0].mobile_index == 0
    assert report.images[0].matches[0].probability_delta == 0.0
    assert report.conversion_passed is True


def test_count_mismatch_is_a_non_overridable_conversion_failure() -> None:
    report = report_for(
        [[prediction(), prediction((30, 0, 40, 10))]],
        [[prediction()]],
        [[0.5, 0.5]],
        [[0.5]],
    )

    decision = evaluate(report, "accept it")
    assert report.reference_count == 2
    assert report.mobile_count == 1
    assert decision.hard_fail is True
    assert decision.can_override is False
    assert decision.overridden is False


def test_label_mismatch_is_a_non_overridable_conversion_failure() -> None:
    report = report_for(mobile=[[prediction(label="other")]])

    assert report.label_mismatch_count == 1
    assert evaluate(report).hard_fail is True


@pytest.mark.parametrize(
    ("reference_probabilities", "mobile_probabilities"),
    (
        ([], [[0.5]]),
        ([[0.5]], []),
        ([[np.nan]], [[0.5]]),
        ([[0.5]], [[np.inf]]),
        ([[-0.1]], [[0.5]]),
        ([[0.5]], [[1.1]]),
        ([[[0.5]]], [[0.5]]),
    ),
)
def test_probability_shape_length_finiteness_and_range_are_hard_failures(
    reference_probabilities: object,
    mobile_probabilities: object,
) -> None:
    report = compare_conversion(
        [[prediction()]],
        [[prediction()]],
        reference_probabilities,
        mobile_probabilities,
        old_count=1,
        new_count=1,
        label_change_ratio=0,
    )

    assert report.invalid_reasons
    assert evaluate(report).hard_fail is True


@pytest.mark.parametrize(
    ("reference", "mobile", "sample_ids"),
    (
        ([[prediction()]], [], None),
        ([prediction()], [[prediction()]], None),
        ([[object()]], [[prediction()]], None),
        ([[prediction()]], [[prediction()]], []),
    ),
)
def test_detection_and_outer_shape_errors_are_hard_failures(
    reference: object, mobile: object, sample_ids: object
) -> None:
    report = compare_conversion(
        reference,
        mobile,
        [[0.5]],
        [[0.5]],
        old_count=1,
        new_count=1,
        label_change_ratio=0,
        sample_ids=sample_ids,
    )

    assert report.invalid_reasons
    assert evaluate(report).hard_fail is True


def test_all_expected_conversion_detections_disappearing_is_hard_failure() -> None:
    report = report_for(
        [[prediction()]],
        [[prediction(score=float(np.nextafter(0.3, 0.0)))]],
        [[0.5]],
        [[0.5]],
    )

    decision = evaluate(report, "mobile detector intentionally changed")
    assert report.all_expected_detections_disappeared is True
    assert decision.hard_fail is True
    assert decision.can_override is False


def test_behavior_count_change_over_thirty_percent_requires_confirmation() -> None:
    report = report_for(old_count=10, new_count=14, label_change_ratio=0.1)
    decision = evaluate(report)

    assert report.count_change_ratio == pytest.approx(0.4)
    assert report.behavior_warning is True
    assert decision.hard_fail is False
    assert decision.can_override is True
    assert decision.requires_confirmation is True


def test_behavior_warning_boundaries_are_strictly_greater_than_policy() -> None:
    at_boundary = report_for(old_count=10, new_count=13, label_change_ratio=0.2)
    count_over = report_for(old_count=10, new_count=14, label_change_ratio=0.2)
    label_over = report_for(
        old_count=10,
        new_count=13,
        label_change_ratio=float(np.nextafter(0.2, np.inf)),
    )

    assert at_boundary.behavior_warning is False
    assert count_over.behavior_warning is True
    assert label_over.behavior_warning is True


def test_zero_behavior_baseline_has_explicit_json_safe_semantics() -> None:
    unchanged = report_for(old_count=0, new_count=0)
    unbounded = report_for(old_count=0, new_count=1)

    assert unchanged.count_change_ratio == 0.0
    assert unchanged.count_change_unbounded is False
    assert unchanged.behavior_warning is False
    assert unbounded.count_change_ratio is None
    assert unbounded.count_change_unbounded is True
    assert unbounded.behavior_warning is True
    assert evaluate(unbounded).requires_confirmation is True


def test_old_behavior_detections_all_disappearing_is_never_overridable() -> None:
    decision = evaluate(report_for(old_count=10, new_count=0), "expected change")

    assert decision.hard_fail is True
    assert decision.can_override is False
    assert decision.overridden is False


@pytest.mark.parametrize(
    ("old_count", "new_count", "label_change_ratio"),
    ((-1, 0, 0), (True, 0, 0), (0, -1, 0), (0, 0, np.nan), (0, 0, 1.1)),
)
def test_invalid_behavior_metrics_are_hard_failures(
    old_count: object, new_count: object, label_change_ratio: object
) -> None:
    report = report_for(
        old_count=old_count,
        new_count=new_count,
        label_change_ratio=label_change_ratio,
    )

    assert report.invalid_reasons
    assert evaluate(report).hard_fail is True


def test_trimmed_operator_reason_confirms_behavior_warning_and_is_recorded() -> None:
    decision = evaluate(
        report_for(old_count=10, new_count=14),
        "  model was deliberately retrained  ",
    )

    assert decision.hard_fail is False
    assert decision.behavior_warning is True
    assert decision.can_override is True
    assert decision.requires_confirmation is False
    assert decision.overridden is True
    assert decision.operator_reason == "model was deliberately retrained"


def test_blank_operator_reason_does_not_confirm_behavior_warning() -> None:
    decision = evaluate(report_for(old_count=10, new_count=14), " \t\n ")

    assert decision.requires_confirmation is True
    assert decision.overridden is False
    assert decision.operator_reason is None


def test_passing_report_is_not_overridable_and_ignores_operator_reason() -> None:
    decision = evaluate(report_for(), "unneeded reason")

    assert decision.hard_fail is False
    assert decision.behavior_warning is False
    assert decision.can_override is False
    assert decision.requires_confirmation is False
    assert decision.overridden is False
    assert decision.operator_reason is None


def test_report_and_decision_are_frozen_and_input_mutation_cannot_change_report() -> None:
    reference = [[prediction()]]
    mobile = [[prediction()]]
    probabilities = [[0.5]]
    report = report_for(reference, mobile, probabilities, probabilities)
    decision = evaluate(report)

    reference[0].clear()
    mobile.clear()
    probabilities[0][0] = 1.0

    assert report.reference_count == report.mobile_count == 1
    assert report.images[0].matches
    with pytest.raises(FrozenInstanceError):
        report.reference_count = 0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        decision.hard_fail = True  # type: ignore[misc]
    assert isinstance(decision, ParityDecision)


@pytest.mark.parametrize(
    "changes",
    (
        {"reference_index": -1},
        {"reference_index": True},
        {"mobile_index": -1},
        {"iou": np.nan},
        {"iou": np.inf},
        {"iou": 0.89},
        {"confidence_delta": -0.01},
        {"confidence_delta": np.nan},
        {"label_equal": 1},
        {"probability_delta": np.inf},
        {"probability_delta": -0.01},
    ),
)
def test_match_rejects_non_json_safe_or_impossible_state(
    changes: dict[str, object],
) -> None:
    item = prediction()
    match = match_detections([item], [item])[0]

    with pytest.raises((TypeError, ValueError)):
        replace(match, **changes)


@pytest.mark.parametrize(
    "changes",
    (
        {"sample_id": ""},
        {"reference_count": True},
        {"reference_count": 2},
        {"matches": []},
        {"unmatched_reference_indexes": [0]},
        {"unmatched_reference_indexes": (0,)},
        {"min_match_iou": np.nan},
        {"max_confidence_delta": np.inf},
        {"max_probability_delta": -0.01},
        {"label_mismatch_count": 1},
        {"conversion_passed": False},
    ),
)
def test_image_parity_rejects_non_json_safe_or_inconsistent_state(
    changes: dict[str, object],
) -> None:
    image = report_for().images[0]

    with pytest.raises((TypeError, ValueError)):
        replace(image, **changes)


@pytest.mark.parametrize(
    "changes",
    (
        {"reference_index": 99, "mobile_index": 88},
        {"reference_index": 1},
        {"mobile_index": 1},
    ),
)
def test_image_parity_rejects_match_indexes_outside_filtered_counts(
    changes: dict[str, int],
) -> None:
    image = report_for().images[0]
    invalid_match = replace(image.matches[0], **changes)

    with pytest.raises(ValueError, match="index"):
        replace(image, matches=(invalid_match,))


@pytest.mark.parametrize(
    ("field", "index"),
    (
        ("unmatched_reference_indexes", 1),
        ("unmatched_reference_indexes", 2),
        ("unmatched_mobile_indexes", 1),
        ("unmatched_mobile_indexes", 2),
    ),
)
def test_image_parity_rejects_unmatched_indexes_outside_filtered_counts(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    index: int,
) -> None:
    monkeypatch.setattr(
        parity_module, "box_iou", lambda left, right: np.array([[0.89]])
    )
    image = report_for().images[0]

    with pytest.raises(ValueError, match="index"):
        replace(image, **{field: (index,)})


def test_image_parity_rejects_finite_but_incomplete_assignment_partition() -> None:
    report = report_for(
        [[prediction(), prediction((30, 0, 40, 10))]],
        [[prediction(), prediction((30, 0, 40, 10))]],
        [[0.5, 0.5]],
        [[0.5, 0.5]],
    )
    image = report.images[0]
    invalid_matches = (
        image.matches[0],
        replace(image.matches[1], reference_index=2),
    )

    with pytest.raises(ValueError, match="index|partition"):
        replace(image, matches=invalid_matches)


@pytest.mark.parametrize(
    "changes",
    (
        {"images": []},
        {"reference_count": True},
        {"reference_count": 2},
        {"min_match_iou": np.nan},
        {"max_confidence_delta": np.inf},
        {"max_probability_delta": -0.01},
        {"label_mismatch_count": 1},
        {"conversion_passed": False},
        {"invalid_reasons": ["bad"]},
        {"invalid_reasons": ("bad",)},
        {"all_expected_detections_disappeared": True},
        {"count_change_ratio": 0.1},
        {"count_change_unbounded": True},
        {"behavior_warning": True},
        {"behavior_all_detections_disappeared": True},
        {"label_change_ratio": np.nan},
    ),
)
def test_parity_report_rejects_non_json_safe_or_inconsistent_state(
    changes: dict[str, object],
) -> None:
    report = report_for()

    with pytest.raises((TypeError, ValueError)):
        replace(report, **changes)


def test_parity_report_rejects_empty_valid_corpus_on_direct_replace() -> None:
    report = report_for()

    with pytest.raises(ValueError, match="at least one sample"):
        replace(
            report,
            images=(),
            reference_count=0,
            mobile_count=0,
            min_match_iou=None,
            max_confidence_delta=None,
            max_probability_delta=None,
            label_mismatch_count=0,
        )


def test_invalid_builder_report_is_still_internally_valid_and_json_safe() -> None:
    report = compare_conversion(
        [[prediction()]],
        [[prediction()]],
        [[np.nan]],
        [[0.5]],
        old_count=1,
        new_count=1,
        label_change_ratio=0,
    )

    assert report.images == ()
    assert report.invalid_reasons
    assert report.conversion_passed is False
    json.dumps(report.to_dict(), allow_nan=False)
    with pytest.raises(ValueError):
        replace(report, conversion_passed=True)


def test_parity_decision_rejects_inconsistent_state_and_mutable_reasons() -> None:
    passed = evaluate(report_for())
    warning = evaluate(report_for(old_count=10, new_count=14))
    hard = evaluate(report_for(old_count=10, new_count=0))

    invalid = (
        (passed, {"can_override": True}),
        (passed, {"requires_confirmation": True}),
        (passed, {"reasons": []}),
        (warning, {"overridden": True}),
        (warning, {"operator_reason": "reason"}),
        (hard, {"can_override": True}),
        (hard, {"operator_reason": "reason"}),
        (hard, {"hard_fail": 1}),
    )
    for decision, changes in invalid:
        with pytest.raises((TypeError, ValueError)):
            replace(decision, **changes)


def test_to_dict_is_complete_json_safe_and_detached() -> None:
    report = report_for(sample_ids=["图像-1"])
    decision = evaluate(report)

    payload = report.to_dict()
    encoded = json.dumps(payload, allow_nan=False, ensure_ascii=False, sort_keys=True)
    decision_encoded = json.dumps(
        decision.to_dict(), allow_nan=False, ensure_ascii=False, sort_keys=True
    )
    payload["images"][0]["matches"].clear()

    assert "图像-1" in encoded
    assert '"hard_fail": false' in decision_encoded
    assert report.images[0].matches
