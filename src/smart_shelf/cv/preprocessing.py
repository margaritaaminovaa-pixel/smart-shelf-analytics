"""Image decoding and normalisation.

Uploads arrive as raw bytes of unknown size, orientation and colour balance.
:func:`prepare_image` turns them into a bounded, contrast-normalised BGR array
plus a JPEG re-encoding suitable for a multimodal prompt, and records the scale
factor so detections can be mapped back to original coordinates if needed.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib

import cv2
import numpy as np
from numpy.typing import NDArray

from smart_shelf.core.exceptions import InvalidImageError
from smart_shelf.core.logging import get_logger

logger = get_logger(__name__)

BGRImage = NDArray[np.uint8]


@dataclass(slots=True, frozen=True)
class PreparedImage:
    """A decoded shelf image together with everything the pipeline needs about it."""

    pixels: BGRImage
    width: int
    height: int
    original_width: int
    original_height: int
    scale: float
    sha256: str

    @property
    def shape(self) -> tuple[int, int]:
        """Prepared size as ``(height, width)``."""
        return self.height, self.width


def decode_image(payload: bytes) -> BGRImage:
    """Decode image bytes into a 3-channel BGR array.

    Raises:
        InvalidImageError: the payload is empty or not a decodable image.
    """
    if not payload:
        raise InvalidImageError("uploaded file is empty")

    buffer = np.frombuffer(payload, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise InvalidImageError("payload is not a decodable image (expected JPEG/PNG/WebP)")
    return np.ascontiguousarray(image, dtype=np.uint8)


def resize_to_max_edge(image: BGRImage, max_edge: int) -> tuple[BGRImage, float]:
    """Downscale so the longest edge is at most ``max_edge``. Never upscales."""
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_edge:
        return image, 1.0
    scale = max_edge / float(longest)
    resized = cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return np.ascontiguousarray(resized, dtype=np.uint8), scale


def normalize_illumination(image: BGRImage) -> BGRImage:
    """Apply CLAHE to the luminance channel.

    Retail shelves are lit unevenly - top rows blow out under spotlights while
    bottom rows fall into shadow. Equalising luminance before detection keeps
    recall roughly constant across shelf levels.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    merged = cv2.merge((clahe.apply(lightness), a_channel, b_channel))
    result = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return np.ascontiguousarray(result, dtype=np.uint8)


def prepare_image(
    payload: bytes,
    *,
    max_edge: int = 1280,
    equalize: bool = True,
) -> PreparedImage:
    """Decode, bound and colour-normalise an uploaded shelf image."""
    original = decode_image(payload)
    original_height, original_width = original.shape[:2]

    resized, scale = resize_to_max_edge(original, max_edge)
    pixels = normalize_illumination(resized) if equalize else resized
    height, width = pixels.shape[:2]

    prepared = PreparedImage(
        pixels=pixels,
        width=width,
        height=height,
        original_width=original_width,
        original_height=original_height,
        scale=scale,
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    logger.debug(
        "image.prepared",
        width=width,
        height=height,
        scale=round(scale, 3),
        sha256=prepared.sha256[:12],
    )
    return prepared


def encode_jpeg(image: BGRImage, *, quality: int = 88) -> bytes:
    """Encode a BGR array as JPEG bytes."""
    success, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:  # pragma: no cover - OpenCV only fails on malformed arrays
        raise InvalidImageError("failed to JPEG-encode the prepared image")
    return bytes(buffer.tobytes())


def to_data_url(image: BGRImage, *, quality: int = 88) -> str:
    """Encode a BGR array as a ``data:`` URL for multimodal prompts."""
    encoded = base64.b64encode(encode_jpeg(image, quality=quality)).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"
