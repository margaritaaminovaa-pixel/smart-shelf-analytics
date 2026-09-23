"""Bounding-box geometry and detection aggregation."""

from __future__ import annotations

from pydantic import ValidationError
import pytest

from smart_shelf.cv.models import BoundingBox, Detection, DetectionResult, ShelfGeometry

pytestmark = pytest.mark.unit


def box(x1: float, y1: float, x2: float, y2: float) -> BoundingBox:
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)


def test_geometry_helpers() -> None:
    unit = box(10, 20, 30, 60)
    assert unit.width == 20
    assert unit.height == 40
    assert unit.area == 800
    assert unit.center == (20.0, 40.0)


def test_degenerate_box_is_rejected() -> None:
    with pytest.raises(ValidationError, match="degenerate"):
        box(30, 0, 10, 10)


def test_iou_of_identical_boxes_is_one() -> None:
    assert box(0, 0, 10, 10).iou(box(0, 0, 10, 10)) == pytest.approx(1.0)


def test_iou_of_disjoint_boxes_is_zero() -> None:
    assert box(0, 0, 10, 10).iou(box(50, 50, 60, 60)) == 0.0


def test_iou_of_half_overlap() -> None:
    # Two 10x10 boxes sharing a 5x10 strip: 50 / (100 + 100 - 50).
    assert box(0, 0, 10, 10).iou(box(5, 0, 15, 10)) == pytest.approx(50 / 150)


def test_normalized_requires_positive_dimensions() -> None:
    with pytest.raises(ValueError, match="positive"):
        box(0, 0, 4, 4).normalized(0, 10)


def test_normalized_scales_into_the_unit_square() -> None:
    assert box(0, 0, 50, 25).normalized(100, 50) == (0.0, 0.0, 0.5, 0.5)


def _result(detections: list[Detection]) -> DetectionResult:
    return DetectionResult(
        detections=detections,
        image_width=100,
        image_height=100,
        backend="test",
        duration_ms=1.0,
        geometry=ShelfGeometry(row_count=2),
    )


def test_grouping_orders_rows_top_down_and_left_to_right() -> None:
    right = Detection(box=box(60, 0, 70, 10), label="p", confidence=0.5, shelf_level=0)
    left = Detection(box=box(10, 0, 20, 10), label="p", confidence=0.9, shelf_level=0)
    lower = Detection(box=box(10, 50, 20, 60), label="p", confidence=0.7, shelf_level=1)

    grouped = _result([right, left, lower]).by_shelf_level()

    assert list(grouped) == [0, 1]
    assert [d.box.x1 for d in grouped[0]] == [10.0, 60.0]


def test_mean_confidence_of_an_empty_result_is_zero() -> None:
    assert _result([]).mean_confidence == 0.0
    assert _result([]).occupancy_ratio() == 0.0


def test_occupancy_never_exceeds_one() -> None:
    overlapping = [
        Detection(box=box(0, 0, 100, 100), label="p", confidence=0.5),
        Detection(box=box(0, 0, 100, 100), label="p", confidence=0.5),
    ]
    assert _result(overlapping).occupancy_ratio() == 1.0
