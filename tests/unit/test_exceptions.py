"""Domain exception hierarchy."""

from __future__ import annotations

import pytest

from smart_shelf.core.exceptions import (
    AuditNotFoundError,
    ConfigurationError,
    DetectionError,
    InvalidImageError,
    NotificationError,
    PersistenceError,
    SmartShelfError,
    VisionModelError,
)

pytestmark = pytest.mark.unit

EXPECTED: list[tuple[type[SmartShelfError], int, str]] = [
    (ConfigurationError, 500, "configuration_error"),
    (InvalidImageError, 422, "invalid_image"),
    (DetectionError, 502, "detection_failed"),
    (VisionModelError, 502, "vlm_failed"),
    (PersistenceError, 500, "persistence_error"),
    (NotificationError, 502, "notification_failed"),
    (AuditNotFoundError, 404, "audit_not_found"),
]


@pytest.mark.parametrize(("error_type", "status_code", "code"), EXPECTED)
def test_each_error_declares_its_transport_mapping(
    error_type: type[SmartShelfError], status_code: int, code: str
) -> None:
    error = error_type("something went wrong")
    assert isinstance(error, SmartShelfError)
    assert error.status_code == status_code
    assert error.code == code


def test_codes_are_unique() -> None:
    codes = [code for _, _, code in EXPECTED]
    assert len(set(codes)) == len(codes)


def test_details_default_to_an_empty_mapping() -> None:
    assert DetectionError("boom").details == {}


def test_serialisation_carries_message_and_details() -> None:
    error = InvalidImageError("too big", details={"size_bytes": 99})
    assert error.to_dict() == {
        "code": "invalid_image",
        "message": "too big",
        "details": {"size_bytes": 99},
    }


def test_errors_remain_ordinary_exceptions() -> None:
    with pytest.raises(SmartShelfError, match="boom"):
        raise DetectionError("boom")
