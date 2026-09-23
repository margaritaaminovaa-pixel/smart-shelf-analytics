"""Detection overlay rendering."""

from __future__ import annotations

import numpy as np
import pytest

from smart_shelf.cv.detector import HeuristicShelfDetector
from smart_shelf.cv.models import (
    BoundingBox,
    Detection,
    DetectionResult,
    ShelfGeometry,
)
from smart_shelf.cv.preprocessing import PreparedImage
from smart_shelf.ui.annotate import (
    ROW_PALETTE,
    annotate_detections,
    row_color,
    row_color_hex,
    to_rgb,
)

pytestmark = pytest.mark.unit


def blank(height: int = 120, width: int = 200, value: int = 200) -> np.ndarray:
    return np.full((height, width, 3), value, dtype=np.uint8)


def result(detections: list[Detection], **kwargs: object) -> DetectionResult:
    return DetectionResult(
        detections=detections,
        image_width=200,
        image_height=120,
        backend="test",
        duration_ms=1.0,
        **kwargs,  # type: ignore[arg-type]
    )


def detection(level: int, x1: float = 10, y1: float = 10) -> Detection:
    return Detection(
        box=BoundingBox(x1=x1, y1=y1, x2=x1 + 40, y2=y1 + 40),
        label="product",
        confidence=0.9,
        shelf_level=level,
        facing_index=0,
    )


def test_palette_wraps_for_deep_shelves() -> None:
    assert row_color(0) == ROW_PALETTE[0]
    assert row_color(len(ROW_PALETTE)) == ROW_PALETTE[0]
    assert row_color(len(ROW_PALETTE) + 2) == ROW_PALETTE[2]


def test_hex_conversion_reverses_bgr() -> None:
    # ROW_PALETTE is BGR; the hex form is RGB for HTML.
    blue, green, red = row_color(0)
    assert row_color_hex(0) == f"#{red:02x}{green:02x}{blue:02x}"


def test_hex_is_always_seven_characters() -> None:
    assert all(len(row_color_hex(level)) == 7 for level in range(len(ROW_PALETTE) + 3))


def test_to_rgb_swaps_channels() -> None:
    bgr = np.zeros((2, 2, 3), dtype=np.uint8)
    bgr[:, :, 0] = 255  # blue in BGR
    assert to_rgb(bgr)[0, 0].tolist() == [0, 0, 255]


def test_annotation_does_not_mutate_the_input() -> None:
    source = blank()
    before = source.copy()
    annotate_detections(source, result([detection(0)]))
    assert np.array_equal(source, before)


def test_annotation_preserves_shape_and_dtype() -> None:
    source = blank()
    out = annotate_detections(source, result([detection(0)]))
    assert out.shape == source.shape
    assert out.dtype == np.uint8


def test_annotation_actually_draws_something() -> None:
    source = blank()
    out = annotate_detections(source, result([detection(0)]), show_bands=False)
    assert not np.array_equal(out, to_rgb(source))


def test_empty_result_with_bands_off_is_a_passthrough() -> None:
    source = blank()
    out = annotate_detections(source, result([]), show_bands=False)
    assert np.array_equal(out, to_rgb(source))


def test_rows_are_drawn_in_distinct_colours() -> None:
    both = annotate_detections(
        blank(),
        result([detection(0, x1=10), detection(1, x1=100)]),
        show_bands=False,
        show_labels=False,
    )
    # Sample the top-left corner of each box's border.
    first = tuple(both[10, 10].tolist())
    second = tuple(both[10, 100].tolist())
    assert first != second


def test_bands_tint_the_full_width() -> None:
    geometry = ShelfGeometry(row_count=1, row_bands=[(0.0, 60.0)])
    tinted = annotate_detections(
        blank(), result([detection(0)], geometry=geometry), show_labels=False
    )
    # A column far from any box is still washed by the band.
    assert tuple(tinted[30, 190].tolist()) != (200, 200, 200)
    # Below the band nothing is tinted.
    assert tuple(tinted[110, 190].tolist()) == (200, 200, 200)


def test_labels_can_be_switched_off() -> None:
    args = {"show_bands": False}
    with_labels = annotate_detections(blank(), result([detection(0)]), show_labels=True, **args)
    without = annotate_detections(blank(), result([detection(0)]), show_labels=False, **args)
    assert not np.array_equal(with_labels, without)


def test_a_box_at_the_top_edge_still_renders() -> None:
    # The label chip has no room above y=0 and must fall back inside the box.
    out = annotate_detections(blank(), result([detection(0, y1=0)]))
    assert out.shape == (120, 200, 3)


def test_renders_a_real_detection_result(prepared_image: PreparedImage) -> None:
    detections = HeuristicShelfDetector().detect(prepared_image)
    out = annotate_detections(prepared_image.pixels, detections)
    assert out.shape == prepared_image.pixels.shape
    assert detections.geometry.row_count == 3
