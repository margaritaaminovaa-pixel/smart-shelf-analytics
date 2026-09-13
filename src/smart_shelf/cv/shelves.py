"""Shelf-row inference.

Planogram compliance is expressed per shelf row ("level 2, third facing from the
left"), so raw boxes have to be bucketed into rows before anything downstream
can reason about position. Rows are recovered by 1-D agglomerative clustering of
box centres on the y axis, which is robust to the slight perspective tilt of
hand-held store photos without needing a calibrated camera.
"""

from __future__ import annotations

from smart_shelf.cv.models import BoundingBox, ShelfGeometry

DEFAULT_ROW_TOLERANCE = 0.6
"""Row separation as a multiple of the median box height, when unspecified."""


def cluster_rows(
    boxes: list[BoundingBox],
    *,
    image_height: int,
    tolerance: float = DEFAULT_ROW_TOLERANCE,
) -> tuple[list[int], ShelfGeometry]:
    """Assign each box a shelf level and describe the resulting row bands.

    Args:
        boxes: candidate product boxes in pixel coordinates.
        image_height: height of the image the boxes were found in.
        tolerance: row separation as a multiple of the median box height. Two
            boxes join the same row when their centres are closer than this.

    Returns:
        A list of shelf levels aligned with ``boxes`` (0 = topmost row), and the
        :class:`ShelfGeometry` describing each row's vertical band.
    """
    if not boxes:
        return [], ShelfGeometry(row_count=0)

    heights = sorted(box.height for box in boxes)
    median_height = heights[len(heights) // 2] or (image_height / 10.0)
    threshold = max(1.0, median_height * tolerance)

    order = sorted(range(len(boxes)), key=lambda i: boxes[i].center[1])
    levels = [0] * len(boxes)
    bands: list[list[float]] = []

    current_level = 0
    current_band = [boxes[order[0]].y1, boxes[order[0]].y2]
    previous_center = boxes[order[0]].center[1]
    levels[order[0]] = 0

    for index in order[1:]:
        center_y = boxes[index].center[1]
        if center_y - previous_center > threshold:
            bands.append(current_band)
            current_level += 1
            current_band = [boxes[index].y1, boxes[index].y2]
        else:
            current_band[0] = min(current_band[0], boxes[index].y1)
            current_band[1] = max(current_band[1], boxes[index].y2)
        levels[index] = current_level
        previous_center = center_y

    bands.append(current_band)
    geometry = ShelfGeometry(
        row_count=current_level + 1,
        row_bands=[(float(top), float(bottom)) for top, bottom in bands],
    )
    return levels, geometry


def facing_indices(boxes: list[BoundingBox], levels: list[int]) -> list[int]:
    """Number boxes left to right within each shelf row."""
    indices = [0] * len(boxes)
    for level in set(levels):
        members = [i for i, lvl in enumerate(levels) if lvl == level]
        members.sort(key=lambda i: boxes[i].x1)
        for slot, i in enumerate(members):
            indices[i] = slot
    return indices
