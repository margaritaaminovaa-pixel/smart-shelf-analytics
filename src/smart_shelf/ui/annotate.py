"""Detection overlay rendering.

Boxes are coloured by shelf level rather than by confidence, so it is obvious
at a glance whether rows were grouped correctly. The translucent band behind
each row shows the vertical extent the clustering assigned, which makes the
effect of the merging tolerance visible.

Nothing here imports Streamlit, so it is testable without the ``ui`` extra.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from smart_shelf.cv.models import DetectionResult

BGRImage = NDArray[np.uint8]

ROW_PALETTE: tuple[tuple[int, int, int], ...] = (
    (86, 180, 233),
    (0, 158, 115),
    (213, 94, 0),
    (204, 121, 167),
    (240, 228, 66),
    (0, 114, 178),
    (230, 159, 0),
)
"""BGR row colours, chosen to stay distinguishable for common colour deficiencies."""

BAND_ALPHA = 0.16
"""Opacity of the shelf-row band wash."""


def row_color(shelf_level: int) -> tuple[int, int, int]:
    """Return the BGR colour assigned to ``shelf_level``."""
    return ROW_PALETTE[shelf_level % len(ROW_PALETTE)]


def row_color_hex(shelf_level: int) -> str:
    """Return the row colour as ``#rrggbb``, for HTML legends."""
    blue, green, red = row_color(shelf_level)
    return f"#{red:02x}{green:02x}{blue:02x}"


def _scaled(value: float, width: int, *, minimum: int = 1) -> int:
    """Scale a nominal 1000px-wide measurement to this image, with a floor."""
    return max(minimum, round(value * width / 1000.0))


def _draw_bands(canvas: BGRImage, result: DetectionResult) -> None:
    """Wash each clustered shelf row with its colour."""
    if not result.geometry.row_bands:
        return
    overlay = canvas.copy()
    height, width = canvas.shape[:2]
    for level, (top, bottom) in enumerate(result.geometry.row_bands):
        y1 = int(max(0, min(height - 1, top)))
        y2 = int(max(0, min(height, bottom)))
        if y2 <= y1:
            continue
        cv2.rectangle(overlay, (0, y1), (width, y2), row_color(level), -1)
    cv2.addWeighted(overlay, BAND_ALPHA, canvas, 1 - BAND_ALPHA, 0, dst=canvas)


def _draw_label(
    canvas: BGRImage, text: str, x: int, y: int, colour: tuple[int, int, int], width: int
) -> None:
    """Draw a filled chip with ``text`` sitting above ``(x, y)``."""
    scale = max(0.32, 0.42 * width / 1000.0)
    thickness = _scaled(1.4, width)
    (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    pad = _scaled(4, width, minimum=2)
    chip_top = y - text_h - baseline - 2 * pad
    if chip_top < 0:  # No room above the box: drop the chip inside it instead.
        chip_top = y
    cv2.rectangle(
        canvas,
        (x, chip_top),
        (x + text_w + 2 * pad, chip_top + text_h + baseline + 2 * pad),
        colour,
        -1,
    )
    cv2.putText(
        canvas,
        text,
        (x + pad, chip_top + text_h + pad),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (20, 20, 20),
        thickness,
        cv2.LINE_AA,
    )


def annotate_detections(
    pixels: BGRImage,
    result: DetectionResult,
    *,
    show_bands: bool = True,
    show_labels: bool = True,
) -> NDArray[np.uint8]:
    """Draw ``result`` over ``pixels`` and return an **RGB** image.

    Args:
        pixels: the BGR image the detections were computed on.
        result: detections to draw, in the same pixel coordinates.
        show_bands: wash each clustered shelf row with its colour.
        show_labels: draw a ``R<row>·F<facing> <confidence>`` chip per box.

    Returns:
        A new RGB array, ready to hand to an image widget. The input is not
        modified.
    """
    canvas: BGRImage = np.ascontiguousarray(pixels.copy(), dtype=np.uint8)
    width = canvas.shape[1]

    if show_bands:
        _draw_bands(canvas, result)

    box_thickness = _scaled(2.4, width, minimum=1)
    for detection in result.detections:
        colour = row_color(detection.shelf_level)
        box = detection.box
        x1, y1 = int(box.x1), int(box.y1)
        x2, y2 = int(box.x2), int(box.y2)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, box_thickness)
        if show_labels:
            _draw_label(
                canvas,
                f"R{detection.shelf_level}-F{detection.facing_index} {detection.confidence:.2f}",
                x1,
                y1,
                colour,
                width,
            )

    rgb: NDArray[np.uint8] = np.ascontiguousarray(
        cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), dtype=np.uint8
    )
    return rgb


def to_rgb(pixels: BGRImage) -> NDArray[np.uint8]:
    """Convert a BGR image to RGB without drawing anything."""
    rgb: NDArray[np.uint8] = np.ascontiguousarray(
        cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB), dtype=np.uint8
    )
    return rgb
