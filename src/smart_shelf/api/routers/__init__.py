"""API routers grouped by resource."""

from __future__ import annotations

from smart_shelf.api.routers.audit import router as audit_router
from smart_shelf.api.routers.health import router as health_router

__all__ = ["audit_router", "health_router"]
