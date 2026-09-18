"""Liveness and readiness endpoints."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Response, status

from smart_shelf import __version__
from smart_shelf.api.dependencies import ContainerDep
from smart_shelf.api.schemas import ComponentHealth, HealthResponse

router = APIRouter(tags=["system"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="System health check",
    description=(
        "Reports the status of every runtime dependency. Returns 200 when the "
        "service can accept audits and 503 when a required component is down."
    ),
)
async def health(container: ContainerDep, response: Response) -> HealthResponse:
    """Probe each component and roll the results into one status."""
    database_ok = await container.repository.health_check()
    components = [
        ComponentHealth(
            name="database",
            status="ok" if database_ok else "down",
            detail=container.settings.sqlite_path,
        ),
        ComponentHealth(
            name="detector",
            status="ok",
            detail=getattr(container.detector, "name", "unknown"),
        ),
        ComponentHealth(
            name="vlm",
            status="ok",
            detail=getattr(container.engine, "name", "unknown"),
        ),
        ComponentHealth(
            name="notifications",
            status="degraded" if container.settings.notifications_dry_run else "ok",
            detail="dry-run" if container.settings.notifications_dry_run else "live",
        ),
    ]

    overall: Literal["ok", "degraded"] = "ok"
    if not database_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        overall = "degraded"
    elif any(component.status == "degraded" for component in components):
        overall = "degraded"

    return HealthResponse(
        status=overall,
        version=__version__,
        environment=container.settings.environment.value,
        components=components,
    )
