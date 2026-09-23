"""Image decoding, resizing and illumination normalisation."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from smart_shelf.core.exceptions import InvalidImageError
from smart_shelf.cv.preprocessing import (
    decode_image,
    encode_jpeg,
    normalize_illumination,
    prepare_image,
    resize_to_max_edge,
    to_data_url,
)

pytestmark = pytest.mark.unit


def test_decode_rejects_empty_payload() -> None:
    with pytest.raises(InvalidImageError, match="empty"):
        decode_image(b"")


def test_decode_rejects_non_image_payload() -> None:
    with pytest.raises(InvalidImageError, match="not a decodable image"):
        decode_image(b"this is a text file, not a JPEG")


def test_decode_returns_three_channels(synthetic_image_bytes: bytes) -> None:
    image = decode_image(synthetic_image_bytes)
    assert image.ndim == 3
    assert image.shape[2] == 3
    assert image.dtype == np.uint8


def test_resize_never_upscales() -> None:
    small = np.zeros((50, 80, 3), dtype=np.uint8)
    resized, scale = resize_to_max_edge(small, 1280)
    assert scale == 1.0
    assert resized.shape == small.shape


def test_resize_bounds_the_longest_edge() -> None:
    wide = np.zeros((600, 2400, 3), dtype=np.uint8)
    resized, scale = resize_to_max_edge(wide, 1200)
    assert max(resized.shape[:2]) == 1200
    assert scale == pytest.approx(0.5)


def test_normalize_illumination_preserves_shape_and_dtype() -> None:
    noisy = np.full((60, 60, 3), 40, dtype=np.uint8)
    noisy[:30] = 200
    equalized = normalize_illumination(noisy)
    assert equalized.shape == noisy.shape
    assert equalized.dtype == np.uint8


def test_prepare_image_records_scale_and_digest(shelf_image_bytes: bytes) -> None:
    prepared = prepare_image(shelf_image_bytes, max_edge=512)
    assert max(prepared.width, prepared.height) == 512
    assert prepared.original_width > prepared.width
    assert prepared.scale < 1.0
    assert len(prepared.sha256) == 64
    assert prepared.shape == (prepared.height, prepared.width)


def test_prepare_image_is_deterministic(shelf_image_bytes: bytes) -> None:
    first = prepare_image(shelf_image_bytes)
    second = prepare_image(shelf_image_bytes)
    assert first.sha256 == second.sha256
    assert np.array_equal(first.pixels, second.pixels)


def test_encode_jpeg_roundtrips(synthetic_image_bytes: bytes) -> None:
    original = decode_image(synthetic_image_bytes)
    encoded = encode_jpeg(original)
    assert encoded[:2] == b"\xff\xd8"
    assert cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR) is not None


def test_data_url_has_the_expected_prefix(synthetic_image_bytes: bytes) -> None:
    url = to_data_url(decode_image(synthetic_image_bytes))
    assert url.startswith("data:image/jpeg;base64,")
