"""Domain exception hierarchy.

Every failure the application raises on purpose derives from
:class:`SmartShelfError`, which carries an HTTP status code and a stable
machine-readable ``code``. The API error handler translates any of them into a
consistent JSON envelope, so transport concerns never leak into the domain.
"""

from __future__ import annotations

from typing import Any


class SmartShelfError(Exception):
    """Base class for all expected application failures."""

    status_code: int = 500
    code: str = "internal_error"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = details or {}

    def to_dict(self) -> dict[str, Any]:
        """Render as the API error envelope body."""
        return {"code": self.code, "message": self.message, "details": self.details}


class ConfigurationError(SmartShelfError):
    """A required setting is missing, malformed or mutually inconsistent."""

    status_code = 500
    code = "configuration_error"


class InvalidRequestError(SmartShelfError):
    """A request is syntactically valid but semantically inconsistent."""

    status_code = 422
    code = "invalid_request"


class InvalidImageError(SmartShelfError):
    """The uploaded payload is not a usable image."""

    status_code = 422
    code = "invalid_image"


class DetectionError(SmartShelfError):
    """The object detection backend failed to produce detections."""

    status_code = 502
    code = "detection_failed"


class VisionModelError(SmartShelfError):
    """The vision-language model failed or returned unusable structured output."""

    status_code = 502
    code = "vlm_failed"


class PersistenceError(SmartShelfError):
    """The audit store could not be read from or written to."""

    status_code = 500
    code = "persistence_error"


class NotificationError(SmartShelfError):
    """An outbound alert channel rejected the notification."""

    status_code = 502
    code = "notification_failed"


class AuditNotFoundError(SmartShelfError):
    """The requested audit record does not exist."""

    status_code = 404
    code = "audit_not_found"
