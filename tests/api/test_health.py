"""GET /health."""

from __future__ import annotations

from httpx import AsyncClient
import pytest

from smart_shelf import __version__
from smart_shelf.api.dependencies import ApplicationContainer

pytestmark = pytest.mark.api


async def test_health_reports_every_component(client: AsyncClient) -> None:
    response = await client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["version"] == __version__
    assert body["environment"] == "test"
    assert {component["name"] for component in body["components"]} == {
        "database",
        "detector",
        "vlm",
        "notifications",
    }


async def test_dry_run_notifications_mark_the_service_degraded(client: AsyncClient) -> None:
    body = (await client.get("/health")).json()
    notifications = next(c for c in body["components"] if c["name"] == "notifications")
    assert notifications["status"] == "degraded"
    assert body["status"] == "degraded"


async def test_health_names_the_bound_backends(
    client: AsyncClient, container: ApplicationContainer
) -> None:
    body = (await client.get("/health")).json()
    detail = {component["name"]: component["detail"] for component in body["components"]}
    assert detail["detector"] == container.detector.name
    assert detail["vlm"] == container.engine.name


async def test_health_returns_503_when_the_database_is_gone(
    client: AsyncClient, container: ApplicationContainer
) -> None:
    await container.repository.close()

    response = await client.get("/health")

    assert response.status_code == 503
    assert response.json()["status"] == "degraded"
    database = next(c for c in response.json()["components"] if c["name"] == "database")
    assert database["status"] == "down"


async def test_every_response_carries_a_request_id(client: AsyncClient) -> None:
    response = await client.get("/health")
    assert response.headers["X-Request-ID"]
    assert float(response.headers["X-Response-Time-ms"]) >= 0


async def test_a_supplied_request_id_is_echoed_back(client: AsyncClient) -> None:
    response = await client.get("/health", headers={"X-Request-ID": "trace-me"})
    assert response.headers["X-Request-ID"] == "trace-me"
