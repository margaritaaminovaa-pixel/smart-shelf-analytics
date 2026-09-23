"""Shelf-row clustering and facing ordering."""

from __future__ import annotations

import pytest

from smart_shelf.cv.models import BoundingBox
from smart_shelf.cv.shelves import cluster_rows, facing_indices

pytestmark = pytest.mark.unit


def grid(rows: int, columns: int, *, row_pitch: int = 100, height: int = 60) -> list[BoundingBox]:
    return [
        BoundingBox(
            x1=float(column * 50),
            y1=float(row * row_pitch),
            x2=float(column * 50 + 40),
            y2=float(row * row_pitch + height),
        )
        for row in range(rows)
        for column in range(columns)
    ]


def test_empty_input_yields_no_rows() -> None:
    levels, geometry = cluster_rows([], image_height=100)
    assert levels == []
    assert geometry.row_count == 0


def test_a_clean_grid_recovers_every_row() -> None:
    boxes = grid(rows=3, columns=4)
    levels, geometry = cluster_rows(boxes, image_height=400)
    assert geometry.row_count == 3
    assert sorted(set(levels)) == [0, 1, 2]
    assert levels.count(0) == 4


def test_rows_are_numbered_from_the_top() -> None:
    top = BoundingBox(x1=0, y1=0, x2=10, y2=20)
    bottom = BoundingBox(x1=0, y1=200, x2=10, y2=220)
    levels, _ = cluster_rows([bottom, top], image_height=300)
    assert levels == [1, 0]


def test_slight_tilt_does_not_split_a_row() -> None:
    # Hand-held photos tilt a few pixels across the bay; that must stay one row.
    tilted = [
        BoundingBox(x1=float(i * 50), y1=float(i * 4), x2=float(i * 50 + 40), y2=float(60 + i * 4))
        for i in range(6)
    ]
    _, geometry = cluster_rows(tilted, image_height=300)
    assert geometry.row_count == 1


def test_row_bands_cover_their_members() -> None:
    boxes = grid(rows=2, columns=3)
    levels, geometry = cluster_rows(boxes, image_height=300)
    for index, box in enumerate(boxes):
        top, bottom = geometry.row_bands[levels[index]]
        assert top <= box.y1
        assert bottom >= box.y2


def test_facings_are_numbered_left_to_right_within_each_row() -> None:
    boxes = [
        BoundingBox(x1=200, y1=0, x2=240, y2=60),
        BoundingBox(x1=0, y1=0, x2=40, y2=60),
        BoundingBox(x1=100, y1=0, x2=140, y2=60),
    ]
    assert facing_indices(boxes, [0, 0, 0]) == [2, 0, 1]


def test_facings_restart_per_row() -> None:
    boxes = [
        BoundingBox(x1=0, y1=0, x2=40, y2=60),
        BoundingBox(x1=100, y1=0, x2=140, y2=60),
        BoundingBox(x1=0, y1=200, x2=40, y2=260),
    ]
    assert facing_indices(boxes, [0, 0, 1]) == [0, 1, 0]
