"""The shared error envelope and the OpenAPI contract."""

from __future__ import annotations

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
import pytest

from smart_shelf.api.dependencies import ApplicationContainer
from smart_shelf.core.exceptions import VisionModelError

pytestmark = pytest.mark.api


async def test_unknown_routes_use_the_shared_envelope(client: AsyncClient) -> None:
    response = await client.get("/api/v1/does-not-exist")

    assert response.status_code == 404
    body = response.json()
    assert body["error"]["code"] == "http_404"
    assert body["error"]["details"] == {}
    assert body["request_id"]


async def test_wrong_method_is_wrapped_too(client: AsyncClient) -> None:
    response = await client.delete("/health")

    assert response.status_code == 405
    assert response.json()["error"]["code"] == "http_405"


async def test_domain_errors_map_to_their_declared_status(
    app: FastAPI, container: ApplicationContainer
) -> None:
    @app.get("/__boom_domain")
    async def boom() -> None:
        raise VisionModelError("provider timed out", details={"backend": "anthropic"})

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as http:
        response = await http.get("/__boom_domain")

    assert response.status_code == 502
    body = response.json()
    assert body["error"]["code"] == "vlm_failed"
    assert body["error"]["message"] == "provider timed out"
    assert body["error"]["details"] == {"backend": "anthropic"}


async def test_unexpected_errors_never_leak_internals(
    app: FastAPI, container: ApplicationContainer
) -> None:
    @app.get("/__boom_unexpected")
    async def boom() -> None:
        raise RuntimeError("database password is hunter2")

    # Starlette re-raises after the 500 handler runs, so the transport is told
    # not to propagate it - exactly what a real ASGI server does.
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://testserver") as http:
        response = await http.get("/__boom_unexpected")

    assert response.status_code == 500
    body = response.json()
    assert body["error"]["code"] == "internal_error"
    assert "hunter2" not in response.text


async def test_validation_errors_omit_raw_input_and_context(client: AsyncClient) -> None:
    response = await client.get("/api/v1/audit/history", params={"limit": 9999})

    assert response.status_code == 422
    errors = response.json()["error"]["details"]["errors"]
    assert errors
    assert all("input" not in error and "ctx" not in error for error in errors)
