"""Computer-vision layer: image preprocessing and shelf object detection."""

from __future__ import annotations

from smart_shelf.cv.detector import (
    HeuristicShelfDetector,
    HybridDetector,
    ObjectDetector,
    YoloDetector,
    build_detector,
    ensure_weights,
)
from smart_shelf.cv.models import BoundingBox, Detection, DetectionResult, ShelfGeometry
from smart_shelf.cv.preprocessing import (
    PreparedImage,
    decode_image,
    encode_jpeg,
    prepare_image,
)

__all__ = [
    "BoundingBox",
    "Detection",
    "DetectionResult",
    "HeuristicShelfDetector",
    "HybridDetector",
    "ObjectDetector",
    "PreparedImage",
    "ShelfGeometry",
    "YoloDetector",
    "build_detector",
    "decode_image",
    "encode_jpeg",
    "ensure_weights",
    "prepare_image",
]
