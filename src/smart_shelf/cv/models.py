"""Value objects exchanged between the CV layer and everything downstream.

These are deliberately backend-agnostic: a YOLO run and the OpenCV fallback
produce the same shapes, so the VLM prompt builder and the API never learn which
detector was used.
"""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

Confidence = Annotated[float, Field(ge=0.0, le=1.0)]


class BoundingBox(BaseModel):
    """Axis-aligned box in absolute pixel coordinates, ``x1 <= x2``/``y1 <= y2``."""

    model_config = ConfigDict(frozen=True)

    x1: float = Field(ge=0)
    y1: float = Field(ge=0)
    x2: float = Field(ge=0)
    y2: float = Field(ge=0)

    @model_validator(mode="after")
    def _validate_ordering(self) -> Self:
        if self.x2 < self.x1 or self.y2 < self.y1:
            msg = f"degenerate bounding box: {self.x1, self.y1, self.x2, self.y2}"
            raise ValueError(msg)
        return self

    @property
    def width(self) -> float:
        """Box width in pixels."""
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """Box height in pixels."""
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        """Box area in square pixels."""
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        """Box centre as ``(x, y)``."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def iou(self, other: BoundingBox) -> float:
        """Intersection over union with ``other``; ``0.0`` when disjoint."""
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def normalized(self, width: int, height: int) -> tuple[float, float, float, float]:
        """Return ``(x1, y1, x2, y2)`` scaled into the unit square."""
        if width <= 0 or height <= 0:
            msg = "image dimensions must be positive"
            raise ValueError(msg)
        return (self.x1 / width, self.y1 / height, self.x2 / width, self.y2 / height)


class Detection(BaseModel):
    """One detected object on a shelf."""

    model_config = ConfigDict(frozen=True)

    box: BoundingBox
    label: str = Field(min_length=1, max_length=128)
    confidence: Confidence
    shelf_level: int = Field(default=0, ge=0, description="0 = topmost shelf row.")
    facing_index: int = Field(default=0, ge=0, description="Left-to-right slot in the row.")


class ShelfGeometry(BaseModel):
    """Inferred shelf-row layout, used to reason about planogram positions."""

    model_config = ConfigDict(frozen=True)

    row_count: int = Field(ge=0)
    row_bands: list[tuple[float, float]] = Field(
        default_factory=list, description="(y_top, y_bottom) pixel band per shelf row."
    )


class DetectionResult(BaseModel):
    """Full output of a detection pass over a single shelf image."""

    model_config = ConfigDict(frozen=True)

    detections: list[Detection] = Field(default_factory=list)
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    backend: str
    duration_ms: float = Field(ge=0)
    geometry: ShelfGeometry = Field(default_factory=lambda: ShelfGeometry(row_count=0))

    @property
    def count(self) -> int:
        """Number of detected objects."""
        return len(self.detections)

    @property
    def mean_confidence(self) -> float:
        """Average detection confidence, or ``0.0`` when nothing was found."""
        if not self.detections:
            return 0.0
        return sum(d.confidence for d in self.detections) / len(self.detections)

    def by_shelf_level(self) -> dict[int, list[Detection]]:
        """Group detections by shelf row, each row ordered left to right."""
        grouped: dict[int, list[Detection]] = {}
        for det in self.detections:
            grouped.setdefault(det.shelf_level, []).append(det)
        for row in grouped.values():
            row.sort(key=lambda d: d.box.x1)
        return dict(sorted(grouped.items()))

    def occupancy_ratio(self) -> float:
        """Fraction of the image covered by detected products, clamped to ``[0, 1]``."""
        image_area = float(self.image_width * self.image_height)
        if image_area <= 0:
            return 0.0
        covered = sum(d.box.area for d in self.detections)
        return min(1.0, covered / image_area)
