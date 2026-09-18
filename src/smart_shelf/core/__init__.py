"""Cross-cutting concerns: configuration, logging, errors and observability."""

from __future__ import annotations

from smart_shelf.core.config import Settings, get_settings
from smart_shelf.core.exceptions import (
    ConfigurationError,
    DetectionError,
    InvalidImageError,
    InvalidRequestError,
    NotificationError,
    PersistenceError,
    SmartShelfError,
    VisionModelError,
)
from smart_shelf.core.logging import configure_logging, get_logger
from smart_shelf.core.observability import Tracer, get_tracer

__all__ = [
    "ConfigurationError",
    "DetectionError",
    "InvalidImageError",
    "InvalidRequestError",
    "NotificationError",
    "PersistenceError",
    "Settings",
    "SmartShelfError",
    "Tracer",
    "VisionModelError",
    "configure_logging",
    "get_logger",
    "get_settings",
    "get_tracer",
]
